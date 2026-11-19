"""RAPO-2709 theo cơ chế trong tài liệu ``RAPO_2709_guithay_ver3.docx``.

Mô-đun này được tách riêng để không thay đổi RAPO, TD-CR-RAPO và
TD-CR-RAPO-V3 hiện có. Thuật toán làm việc trên giant tour, giải mã bằng
Split động, và đếm đúng một NFE cho mỗi hoán vị được đánh giá đầy đủ.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterable

from mamut_routing_lib.td import apply_ready_time, restrict_domain

from core.data.evaluator import (
    HCMCEvaluationResult,
    HCMCTDVRPEvaluator,
    RouteEvaluation,
    RouteLegTrace,
    RouteStopTrace,
)
from core.data.models import RoadEdge, RoadNode, TDVRPInstance, TrafficOverlay
from core.models import Solution


ELITE_OPERATORS = ("relocate", "swap", "two_opt", "two_opt_star")


@dataclass(frozen=True)
class RAPO2709Config:
    """Các tham số cố định được mô tả trong bài báo."""

    population_size: int = 10
    max_function_evaluations: int = 100
    destroy_fraction: float = 0.10
    criticality_floor: float = 0.10
    elite_budget_fraction: float = 0.20
    aos_epsilon: float = 0.10
    aos_alpha: float = 0.20
    # Giữ 0 để không thay đổi các benchmark cũ; web app truyền 08:00.
    earliest_departure: float = 0.0
    latest_departure: float | None = None
    # Benchmark công bằng dùng exact Split giống TDVRPProblem. App cũ có thể
    # giữ False để dùng proxy time-zero nhanh hơn.
    exact_split: bool = False
    seed: int | None = None

    def validate(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size phải từ 2 trở lên")
        if self.max_function_evaluations < self.population_size:
            raise ValueError("Ngân sách NFE phải đủ để đánh giá toàn bộ quần thể ban đầu")
        if not 0 < self.destroy_fraction <= 1:
            raise ValueError("destroy_fraction phải thuộc (0, 1]")
        if self.criticality_floor < 0:
            raise ValueError("criticality_floor không được âm")
        if not 0 <= self.elite_budget_fraction <= 1:
            raise ValueError("elite_budget_fraction phải thuộc [0, 1]")
        if self.latest_departure is not None and self.latest_departure < self.earliest_departure:
            raise ValueError("latest_departure phải lớn hơn hoặc bằng earliest_departure")
        if not 0 <= self.aos_epsilon <= 1 / len(ELITE_OPERATORS):
            raise ValueError("aos_epsilon quá lớn so với số toán tử Elite")
        if not 0 < self.aos_alpha <= 1:
            raise ValueError("aos_alpha phải thuộc (0, 1]")
        if type(self.exact_split) is not bool:
            raise ValueError("exact_split phải là bool")


@dataclass(frozen=True)
class _LegState:
    source: int
    target: int
    departure: float
    arrival: float
    travel_time: float


@dataclass(frozen=True)
class _StopState:
    customer: int
    arrival: float
    service_start: float
    departure: float
    waiting: float
    service: float
    load: float


@dataclass(frozen=True)
class _RouteState:
    customers: tuple[int, ...]
    departure: float
    end_time: float
    duration: float
    travel_time: float
    service_time: float
    waiting_time: float
    load: float
    legs: tuple[_LegState, ...]
    stops: tuple[_StopState, ...]


@dataclass(frozen=True)
class RAPO2709Candidate:
    permutation: tuple[int, ...]
    routes: tuple[tuple[int, ...], ...]
    route_states: tuple[_RouteState, ...]
    fitness: float


@dataclass(frozen=True)
class RAPO2709Result:
    best_solution: Solution
    best_evaluation: HCMCEvaluationResult
    best_permutation: tuple[int, ...]
    convergence: list[float]
    runtime_seconds: float
    evaluations: int
    nfe_by_branch: dict[str, int]
    operator_scores: dict[str, float]
    operator_uses: dict[str, int]
    operator_successes: dict[str, int]
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def nfe(self) -> int:
        """Tên thay thế để tương thích với ExperimentRunner."""
        return self.evaluations


class _ExactSplitEvaluator:
    """Split tối ưu và định giá thời gian xuất phát chính xác bằng NDCPWLF."""

    def __init__(
        self,
        instance: TDVRPInstance,
        graph: tuple[dict[int, RoadNode], dict[str, RoadEdge]],
        traffic: TrafficOverlay,
        earliest_departure: float = 0.0,
        latest_departure: float | None = None,
        exact_split: bool = False,
    ) -> None:
        if instance.arrival_time_functions is None:
            raise ValueError(
                "RAPO-2709 cần arrival_time_functions chính thức; "
                "hãy nạp Poryos với materialize_atfs=True"
            )
        self.instance = instance
        self.graph = graph
        self.traffic = traffic
        self.earliest_departure = float(earliest_departure)
        self.latest_departure = float(latest_departure) if latest_departure is not None else None
        self.exact_split = bool(exact_split)
        self.atfs = instance.arrival_time_functions
        self.route_cache: dict[tuple[int, ...], _RouteState | None] = {}
        self.free_flow_cache: dict[tuple[int, int], float] = {}
        self.time_zero_cache: dict[tuple[int, int], float] = {}

    def evaluate(self, permutation: Iterable[int]) -> RAPO2709Candidate:
        order = tuple(permutation)
        self._validate_permutation(order)
        routes, states, fitness = self._split(order)
        return RAPO2709Candidate(order, routes, states, fitness)

    def _validate_permutation(self, order: tuple[int, ...]) -> None:
        expected = set(range(1, self.instance.num_customers + 1))
        if len(order) != self.instance.num_customers or set(order) != expected:
            raise ValueError("Ứng viên RAPO-2709 không phải hoán vị đầy đủ của khách hàng")

    def _split(
        self,
        order: tuple[int, ...],
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[_RouteState, ...], float]:
        """Order-first, split-second bằng quy hoạch động trên các đoạn liên tiếp.

        ``exact_split=True`` dùng duration NDCPWLF chính xác cho từng đoạn và
        cùng quy tắc Split với TDVRPProblem. Chế độ False giữ decoder proxy cũ
        cho app, dùng chi phí cung tại thời điểm 0 để giảm thời gian xử lý.
        """
        if self.exact_split:
            return self._split_exact(order)
        n = len(order)
        best_proxy_cost = [math.inf] * (n + 1)
        best_route_count = [n + 1] * (n + 1)
        predecessor = [-1] * (n + 1)
        best_proxy_cost[0] = 0.0
        best_route_count[0] = 0

        for end in range(1, n + 1):
            load = 0.0
            proxy_route_cost = 0.0
            for start in range(end - 1, -1, -1):
                customer = order[start]
                load += self.instance.demands[customer]
                if load > self.instance.vehicle_capacity + 1e-9:
                    break
                if start == end - 1:
                    proxy_route_cost = (
                        self.instance.service_times[customer]
                        + self.travel_at_zero(self.instance.depot, customer)
                        + self.travel_at_zero(customer, self.instance.depot)
                    )
                else:
                    next_customer = order[start + 1]
                    proxy_route_cost += (
                        self.instance.service_times[customer]
                        + self.travel_at_zero(self.instance.depot, customer)
                        + self.travel_at_zero(customer, next_customer)
                        - self.travel_at_zero(self.instance.depot, next_customer)
                    )
                if not math.isfinite(best_proxy_cost[start]):
                    continue
                candidate_cost = best_proxy_cost[start] + proxy_route_cost
                candidate_routes = best_route_count[start] + 1
                key = (candidate_cost, candidate_routes, start)
                current = (best_proxy_cost[end], best_route_count[end], predecessor[end])
                if key < current:
                    best_proxy_cost[end] = candidate_cost
                    best_route_count[end] = candidate_routes
                    predecessor[end] = start

        if predecessor[n] < 0:
            raise ValueError("Split Decoder không tìm được cách phân tuyến khả thi")

        routes_reversed: list[tuple[int, ...]] = []
        cursor = n
        while cursor > 0:
            start = predecessor[cursor]
            if start < 0:
                raise RuntimeError("Split Decoder mất trạng thái truy vết")
            routes_reversed.append(order[start:cursor])
            cursor = start
        routes = tuple(reversed(routes_reversed))
        states_list = [self.route_state(route) for route in routes]
        if any(state is None for state in states_list):
            raise ValueError("Split Decoder tạo ra tuyến không khả thi theo thời gian")
        states = tuple(state for state in states_list if state is not None)
        return routes, states, sum(state.duration for state in states)

    def _split_exact(
        self,
        order: tuple[int, ...],
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[_RouteState, ...], float]:
        """Minimum-duration capacity-feasible Split shared by all baselines."""

        n = len(order)
        costs = [0.0] + [math.inf] * n
        partitions: list[tuple[tuple[int, ...], ...] | None] = [()] + [None] * n
        for end in range(1, n + 1):
            load = 0.0
            for start in range(end - 1, -1, -1):
                load += self.instance.demands[order[start]]
                if load > self.instance.vehicle_capacity + 1e-9:
                    break
                state = self.route_state(order[start:end])
                if state is None or partitions[start] is None:
                    continue
                candidate_cost = costs[start] + state.duration
                if candidate_cost < costs[end]:
                    costs[end] = candidate_cost
                    partitions[end] = (*partitions[start], order[start:end])
        if partitions[n] is None:
            raise ValueError("Exact Split không tìm được cách phân tuyến khả thi")
        routes = tuple(sorted(partitions[n], key=lambda route: route[0]))
        states_list = [self.route_state(route) for route in routes]
        if any(state is None for state in states_list):
            raise ValueError("Exact Split tạo ra tuyến không khả thi theo thời gian")
        states = tuple(state for state in states_list if state is not None)
        return routes, states, sum(state.duration for state in states)

    def route_state(self, customers: tuple[int, ...]) -> _RouteState | None:
        cached = self.route_cache.get(customers)
        if customers in self.route_cache:
            return cached
        state = self._compose_and_schedule(customers)
        self.route_cache[customers] = state
        return state

    def _compose_and_schedule(self, customers: tuple[int, ...]) -> _RouteState | None:
        if not customers:
            return None
        start, end = self.instance.horizon
        start = max(float(start), self.earliest_departure)
        if self.latest_departure is not None:
            end = min(float(end), self.latest_departure)
        end = float(end)
        if start > end:
            raise ValueError("earliest_departure phải nhỏ hơn hoặc bằng latest_departure (và cuối horizon)")
        depot = self.instance.depot
        accumulator = None
        previous = depot

        for customer in customers:
            arc = self.atfs[(previous, customer)]
            accumulator = (
                restrict_domain(arc, start, end)
                if accumulator is None
                else arc.compose(accumulator, slope_one_exact=True)
            )
            if accumulator.is_empty():
                return None
            window = self._time_window(customer)
            accumulator = apply_ready_time(
                accumulator,
                earliest=window[0] if window is not None else None,
                latest=window[1] if window is not None else None,
                service_time=float(self.instance.service_times[customer]),
            )
            if accumulator.is_empty():
                return None
            previous = customer

        accumulator = self.atfs[(previous, depot)].compose(accumulator, slope_one_exact=True)
        if accumulator.is_empty():
            return None
        depot_window = self._time_window(depot)
        if depot_window is not None:
            accumulator = apply_ready_time(
                accumulator,
                earliest=None,
                latest=depot_window[1],
                service_time=0.0,
            )
        if accumulator.is_empty():
            return None

        duration, departure = accumulator.min_shifted_image()
        return self._schedule(customers, float(departure), float(duration))

    def _schedule(
        self,
        customers: tuple[int, ...],
        departure: float,
        exact_duration: float,
    ) -> _RouteState:
        depot = self.instance.depot
        clock = departure
        load = travel = service = waiting = 0.0
        legs: list[_LegState] = []
        stops: list[_StopState] = [
            _StopState(depot, departure, departure, departure, 0.0, 0.0, 0.0)
        ]
        previous = depot

        for target in (*customers, depot):
            arrival = self.arrival_at(previous, target, clock)
            leg_travel = arrival - clock
            legs.append(_LegState(previous, target, clock, arrival, leg_travel))
            travel += leg_travel
            service_start = arrival
            stop_waiting = stop_service = 0.0
            if target != depot:
                window = self._time_window(target)
                if window is not None:
                    service_start = max(arrival, window[0])
                stop_waiting = service_start - arrival
                stop_service = float(self.instance.service_times[target])
                load += self.instance.demands[target]
                waiting += stop_waiting
                service += stop_service
            clock = service_start + stop_service
            stops.append(
                _StopState(
                    target,
                    arrival,
                    service_start,
                    clock,
                    stop_waiting,
                    stop_service,
                    load,
                )
            )
            previous = target

        # Giá trị từ phép hợp thành là chuẩn; lịch tuần tự được giữ để tính criticality.
        return _RouteState(
            customers,
            departure,
            departure + exact_duration,
            exact_duration,
            travel,
            service,
            waiting,
            load,
            tuple(legs),
            tuple(stops),
        )

    def _time_window(self, customer: int) -> tuple[float, float] | None:
        if self.instance.time_windows is None:
            return None
        return self.instance.time_windows[customer]

    def travel_at_zero(self, source: int, target: int) -> float:
        if source == target:
            return 0.0
        key = (source, target)
        if key not in self.time_zero_cache:
            try:
                departure = 0.0
                arrival = self.arrival_at(source, target, departure)
            except (ValueError, IndexError):
                departure = float(self.instance.horizon[0])
                arrival = self.arrival_at(source, target, departure)
            self.time_zero_cache[key] = arrival - departure
        return self.time_zero_cache[key]

    def arrival_at(self, source: int, target: int, departure: float) -> float:
        """Đánh giá ATF và khử sai số ulp rất nhỏ tại biên miền xác định."""
        if source == target:
            return departure
        atf = self.atfs[(source, target)]
        value = float(departure)
        low, high = float(atf.xs[0]), float(atf.xs[-1])
        tolerance = 1e-9 * max(1.0, abs(low), abs(high))
        if low - tolerance <= value < low:
            value = low
        elif high < value <= high + tolerance:
            value = high
        return float(atf.evaluate(value))

    def free_flow(self, source: int, target: int) -> float:
        if source == target:
            return 0.0
        key = (source, target)
        if key not in self.free_flow_cache:
            self.free_flow_cache[key] = float(self.atfs[key].min_shifted_image()[0])
        return self.free_flow_cache[key]

    def materialize(self, candidate: RAPO2709Candidate) -> tuple[Solution, HCMCEvaluationResult]:
        """Tạo nghiệm và route trace đầy đủ cho nghiệm tốt nhất cuối cùng."""
        depot = self.instance.depot
        solution = Solution([[depot, *route, depot] for route in candidate.routes])
        path_evaluator = HCMCTDVRPEvaluator(self.instance, self.graph, self.traffic, start_time=0.0)
        route_evaluations: list[RouteEvaluation] = []

        for vehicle_index, state in enumerate(candidate.route_states):
            node_path: list[int] = [self.instance.road_node_ids[depot]]
            edge_path: list[str] = []
            leg_traces: list[RouteLegTrace] = []
            total_distance = 0.0
            for leg in state.legs:
                source_node = self.instance.road_node_ids[leg.source]
                target_node = self.instance.road_node_ids[leg.target]
                try:
                    path = path_evaluator._find_path(source_node, target_node, leg.departure)
                    leg_nodes = list(path.node_path)
                    leg_edges = list(path.edge_path)
                    distance = sum(self.graph[1][edge_id].distance_m for edge_id in leg_edges)
                except (KeyError, ValueError):
                    # ATF vẫn là nguồn objective; fallback này chỉ giữ trace không bị mất.
                    leg_nodes = [source_node, target_node]
                    leg_edges = []
                    x1, y1 = self.instance.coordinates[leg.source]
                    x2, y2 = self.instance.coordinates[leg.target]
                    distance = math.hypot(x1 - x2, y1 - y2)
                node_path.extend(leg_nodes[1:])
                edge_path.extend(leg_edges)
                total_distance += distance
                leg_traces.append(
                    RouteLegTrace(
                        leg.source,
                        leg.target,
                        leg.departure,
                        leg.arrival,
                        leg.travel_time,
                        distance,
                        leg_nodes,
                        leg_edges,
                    )
                )

            stop_traces = [
                RouteStopTrace(
                    stop.customer,
                    self.instance.road_node_ids[stop.customer],
                    stop.arrival,
                    stop.service_start,
                    stop.departure,
                    stop.waiting,
                    stop.service,
                    self.instance.demands[stop.customer],
                    stop.load,
                )
                for stop in state.stops
            ]
            route_evaluations.append(
                RouteEvaluation(
                    vehicle_index,
                    list(state.customers),
                    node_path,
                    edge_path,
                    state.departure,
                    state.end_time,
                    state.travel_time,
                    state.service_time,
                    total_distance,
                    state.load,
                    state.waiting_time,
                    True,
                    [],
                    leg_traces,
                    stop_traces,
                )
            )

        total_travel = sum(route.travel_time for route in route_evaluations)
        total_service = sum(route.service_time for route in route_evaluations)
        total_waiting = sum(route.waiting_time for route in route_evaluations)
        total_distance = sum(route.distance for route in route_evaluations)
        evaluation = HCMCEvaluationResult(
            total_travel,
            total_service,
            total_distance,
            candidate.fitness,
            candidate.fitness,
            True,
            [],
            route_evaluations,
            total_waiting,
            "exact_ndcpwlf",
        )
        return solution, evaluation


class _NFELedger:
    """Mọi lần gọi evaluate đều tiêu thụ đúng một NFE, kể cả ứng viên lỗi."""

    def __init__(self, evaluator: _ExactSplitEvaluator, budget: int) -> None:
        self.evaluator = evaluator
        self.budget = budget
        self.used = 0
        self.by_branch = {"initialization": 0, "ARO": 0, "POA": 0, "Elite": 0}

    def evaluate(self, permutation: Iterable[int], branch: str) -> RAPO2709Candidate | None:
        if self.used >= self.budget:
            return None
        self.used += 1
        self.by_branch[branch] += 1
        try:
            return self.evaluator.evaluate(permutation)
        except (KeyError, ValueError):
            return None


def order_crossover(
    first_parent: tuple[int, ...],
    second_parent: tuple[int, ...],
    rng: random.Random,
) -> tuple[int, ...]:
    """Order Crossover: giữ một đoạn của cha thứ nhất, điền theo cha thứ hai."""
    n = len(first_parent)
    if n < 2:
        return first_parent
    left, right = sorted(rng.sample(range(n), 2))
    child: list[int | None] = [None] * n
    child[left : right + 1] = first_parent[left : right + 1]
    preserved = set(first_parent[left : right + 1])
    rotated_second = second_parent[right + 1 :] + second_parent[: right + 1]
    remaining = [value for value in rotated_second if value not in preserved]
    positions = list(range(right + 1, n)) + list(range(0, left))
    for position, value in zip(positions, remaining):
        child[position] = value
    result = tuple(int(value) for value in child if value is not None)
    if result == first_parent:
        mutable = list(result)
        i, j = rng.sample(range(n), 2)
        mutable[i], mutable[j] = mutable[j], mutable[i]
        result = tuple(mutable)
    return result


def compute_criticality(
    candidate: RAPO2709Candidate,
    evaluator: _ExactSplitEvaluator,
) -> dict[int, float]:
    """Tính D, K, W đúng trên lịch chạy của nghiệm đã Split."""
    detour: dict[int, float] = {}
    congestion: dict[int, float] = {}
    route_weight: dict[int, float] = {}

    for route in candidate.route_states:
        for index, customer in enumerate(route.customers):
            incoming = route.legs[index]
            outgoing = route.legs[index + 1]
            direct = 0.0
            if incoming.source != outgoing.target:
                key = (incoming.source, outgoing.target)
                if key in evaluator.atfs:
                    direct_arrival = evaluator.arrival_at(
                        incoming.source,
                        outgoing.target,
                        incoming.departure,
                    )
                    direct = direct_arrival - incoming.departure
            detour[customer] = max(0.0, incoming.travel_time + outgoing.travel_time - direct)

            free_in = evaluator.free_flow(incoming.source, customer)
            free_out = evaluator.free_flow(customer, outgoing.target)
            eta_in = max(0.0, (incoming.travel_time - free_in) / max(1e-6, free_in))
            eta_out = max(0.0, (outgoing.travel_time - free_out) / max(1e-6, free_out))
            congestion[customer] = 0.5 * (eta_in + eta_out)
            route_weight[customer] = (
                evaluator.instance.service_times[customer]
                + incoming.travel_time
                + outgoing.travel_time
            ) / max(1e-6, route.duration)

    normalized = [_minmax(values) for values in (detour, congestion, route_weight)]
    return {
        customer: sum(values.get(customer, 0.5) for values in normalized) / 3.0
        for customer in candidate.permutation
    }


def _minmax(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if high - low <= 1e-12:
        return {key: 0.5 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def _weighted_sample_without_replacement(
    population: tuple[int, ...],
    criticality: dict[int, float],
    count: int,
    floor: float,
    rng: random.Random,
) -> list[int]:
    remaining = list(population)
    selected: list[int] = []
    for _ in range(min(count, len(remaining))):
        weights = [criticality.get(customer, 0.0) + floor for customer in remaining]
        threshold = rng.random() * sum(weights)
        cumulative = 0.0
        chosen_index = len(remaining) - 1
        for index, weight in enumerate(weights):
            cumulative += weight
            if threshold <= cumulative:
                chosen_index = index
                break
        selected.append(remaining.pop(chosen_index))
    return selected


def regret2_reconstruct(
    partial: list[int],
    removed: list[int],
    evaluator: _ExactSplitEvaluator,
) -> tuple[int, ...]:
    """Chèn lại toàn cục theo regret-2 và chi phí cung tại thời điểm 0."""
    pending = removed[:]
    while pending:
        choice: tuple[float, float, int, int] | None = None
        for customer in pending:
            options: list[tuple[float, int]] = []
            for position in range(len(partial) + 1):
                before = partial[position - 1] if position else evaluator.instance.depot
                after = partial[position] if position < len(partial) else evaluator.instance.depot
                delta = (
                    evaluator.instance.service_times[customer]
                    + evaluator.travel_at_zero(before, customer)
                    + evaluator.travel_at_zero(customer, after)
                    - evaluator.travel_at_zero(before, after)
                )
                options.append((delta, position))
            options.sort(key=lambda item: (item[0], item[1]))
            regret = options[1][0] - options[0][0] if len(options) > 1 else 0.0
            candidate_choice = (regret, -options[0][0], -customer, options[0][1])
            if choice is None or candidate_choice > choice:
                choice = candidate_choice
        if choice is None:
            raise RuntimeError("Regret-2 không chọn được khách hàng để chèn")
        customer = -choice[2]
        partial.insert(choice[3], customer)
        pending.remove(customer)
    return tuple(partial)


def poa_restructure(
    parent: RAPO2709Candidate,
    evaluator: _ExactSplitEvaluator,
    config: RAPO2709Config,
    rng: random.Random,
) -> tuple[int, ...]:
    criticality = compute_criticality(parent, evaluator)
    n = len(parent.permutation)
    remove_count = max(1, min(n - 1, math.ceil(config.destroy_fraction * n))) if n > 1 else 1
    removed = _weighted_sample_without_replacement(
        parent.permutation,
        criticality,
        remove_count,
        config.criticality_floor,
        rng,
    )
    removed_set = set(removed)
    partial = [customer for customer in parent.permutation if customer not in removed_set]
    return regret2_reconstruct(partial, removed, evaluator)


def _elite_probabilities(scores: dict[str, float], epsilon: float) -> list[float]:
    nonnegative = [max(0.0, scores[name]) for name in ELITE_OPERATORS]
    total = sum(nonnegative)
    base = [value / total for value in nonnegative] if total > 0 else [0.25] * 4
    multiplier = 1.0 - len(ELITE_OPERATORS) * epsilon
    return [multiplier * probability + epsilon for probability in base]


def _elite_move(
    permutation: tuple[int, ...],
    operator: str,
    rng: random.Random,
) -> tuple[int, ...]:
    result = list(permutation)
    if len(result) < 2:
        return tuple(result)
    if operator == "swap":
        left, right = rng.sample(range(len(result)), 2)
        result[left], result[right] = result[right], result[left]
    elif operator == "relocate":
        source, target = rng.sample(range(len(result)), 2)
        value = result.pop(source)
        result.insert(target, value)
    else:
        # Theo cấu hình bài báo, two_opt_star dùng nhánh dự phòng trên giant tour.
        left, right = sorted(rng.sample(range(len(result)), 2))
        result[left : right + 1] = reversed(result[left : right + 1])
    return tuple(result)


def _best_index(population: list[RAPO2709Candidate]) -> int:
    return min(range(len(population)), key=lambda index: population[index].fitness)


def optimize(
    instance: TDVRPInstance,
    graph: tuple[dict[int, RoadNode], dict[str, RoadEdge]],
    traffic: TrafficOverlay,
    config: RAPO2709Config | None = None,
    seed: int | None = None,
    initial_population: Iterable[Iterable[int]] | None = None,
    **_: object,
) -> RAPO2709Result:
    """Chạy RAPO-2709 với bộ điều phối ARO/POA/Elite lấy mẫu đều."""
    config = config or RAPO2709Config()
    config.validate()
    rng = random.Random(config.seed if seed is None else seed)
    evaluator = _ExactSplitEvaluator(
        instance,
        graph,
        traffic,
        earliest_departure=config.earliest_departure,
        latest_departure=config.latest_departure,
        exact_split=config.exact_split,
    )
    ledger = _NFELedger(evaluator, config.max_function_evaluations)
    started = perf_counter()

    population: list[RAPO2709Candidate] = []
    archive: RAPO2709Candidate | None = None
    convergence: list[float] = []
    base = list(range(1, instance.num_customers + 1))
    if initial_population is None:
        initial_orders = [tuple(rng.sample(base, len(base))) for _ in range(config.population_size)]
        initialization_policy = "independent_random"
    else:
        initial_orders = [tuple(int(customer) for customer in order) for order in initial_population]
        if len(initial_orders) != config.population_size:
            raise ValueError("Quần thể P0 phải có đúng population_size hoán vị")
        for order in initial_orders:
            evaluator._validate_permutation(order)
        initialization_policy = "provided_common_population"

    for permutation in initial_orders:
        candidate = ledger.evaluate(permutation, "initialization")
        if candidate is None:
            raise RuntimeError("Không thể khởi tạo quần thể RAPO-2709 khả thi")
        population.append(candidate)
        if archive is None or candidate.fitness < archive.fitness:
            archive = candidate
        convergence.append(archive.fitness)

    assert archive is not None
    scores = {name: 1.0 for name in ELITE_OPERATORS}
    uses = {name: 0 for name in ELITE_OPERATORS}
    successes = {name: 0 for name in ELITE_OPERATORS}
    elite_quota = math.floor(config.elite_budget_fraction * config.max_function_evaluations)

    while ledger.used < ledger.budget:
        available = ["ARO", "POA"]
        if ledger.by_branch["Elite"] < elite_quota:
            available.append("Elite")
        branch = rng.choice(available)

        if branch == "ARO":
            parent_index = rng.randrange(len(population))
            partner_index = rng.choice([index for index in range(len(population)) if index != parent_index])
            parent = population[parent_index]
            child_order = order_crossover(parent.permutation, population[partner_index].permutation, rng)
            child = ledger.evaluate(child_order, "ARO")
            if child is not None and child.fitness < parent.fitness:
                population[parent_index] = child
                if child.fitness < archive.fitness:
                    archive = child

        elif branch == "POA":
            parent_index = rng.randrange(len(population))
            parent = population[parent_index]
            child_order = poa_restructure(parent, evaluator, config, rng)
            child = ledger.evaluate(child_order, "POA")
            if child is not None and child.fitness < parent.fitness:
                population[parent_index] = child
                if child.fitness < archive.fitness:
                    archive = child

        else:
            parent_index = _best_index(population)
            parent = population[parent_index]
            probabilities = _elite_probabilities(scores, config.aos_epsilon)
            operator = rng.choices(ELITE_OPERATORS, weights=probabilities, k=1)[0]
            uses[operator] += 1
            child_order = _elite_move(parent.permutation, operator, rng)
            child = ledger.evaluate(child_order, "Elite")
            reward = 0.0
            if child is not None:
                relative = max(
                    0.0,
                    (parent.fitness - child.fitness) / max(parent.fitness, 1e-6),
                )
                if child.fitness < archive.fitness:
                    reward = 2.0 * relative
                elif child.fitness < parent.fitness:
                    reward = relative
                if child.fitness < parent.fitness:
                    population[parent_index] = child
                    successes[operator] += 1
                if child.fitness < archive.fitness:
                    archive = child
            scores[operator] = (
                (1.0 - config.aos_alpha) * scores[operator]
                + config.aos_alpha * reward
            )

        convergence.append(archive.fitness)

    solution, evaluation = evaluator.materialize(archive)
    return RAPO2709Result(
        solution,
        evaluation,
        archive.permutation,
        convergence,
        perf_counter() - started,
        ledger.used,
        dict(ledger.by_branch),
        scores,
        uses,
        successes,
        {
            "variant": "RAPO-2709",
            "objective": "sum of exact optimal route durations",
            "split_decoder": (
                "exact minimum-duration order-first split-second"
                if config.exact_split
                else "time-zero proxy order-first split-second"
            ),
            "departure_time_mode": "exact_ndcpwlf",
            "fleet_policy": "unlimited_nonempty_routes",
            "initialization_policy": initialization_policy,
            "elite_quota": elite_quota,
            "route_count": len(archive.routes),
            "strict_nfe": ledger.used == ledger.budget,
        },
    )


class RAPO2709:
    """Adapter hướng đối tượng đồng nhất với các thuật toán còn lại."""

    name = "RAPO-2709"

    def optimize(self, instance, graph, traffic, config=None, seed=None, **kwargs):
        return optimize(instance, graph, traffic, config=config, seed=seed, **kwargs)
