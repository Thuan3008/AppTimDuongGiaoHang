"""Evaluator TDVRP có route trace và tối ưu thời điểm xuất phát."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import inf

from core.models import Solution
from core.routing.fastest_path import FastestPathResult, find_fastest_path, find_pinned_fastest_path

from .models import RoadEdge, RoadNode, TDVRPInstance, TrafficOverlay
from .solution import validate_solution


OBJECTIVE_NAME = "total_route_duration"


@dataclass(frozen=True)
class RouteLegTrace:
    """Chi tiết một chặng giữa hai điểm của route."""

    from_customer_id: int
    to_customer_id: int
    departure_time: float
    arrival_time: float
    travel_time: float
    distance: float
    node_path: list[int]
    edge_path: list[str]


@dataclass(frozen=True)
class RouteStopTrace:
    """Mốc thời gian tại depot hoặc khách hàng."""

    customer_id: int
    road_node_id: int
    arrival_time: float
    service_start_time: float
    departure_time: float
    waiting_time: float
    service_time: float
    demand: float
    cumulative_load: float


@dataclass(frozen=True)
class RouteEvaluation:
    vehicle_index: int
    customer_ids: list[int]
    node_path: list[int]
    edge_path: list[str]
    start_time: float
    end_time: float
    travel_time: float
    service_time: float
    distance: float
    load: float
    waiting_time: float = 0.0
    feasible: bool = True
    errors: list[str] = field(default_factory=list)
    legs: list[RouteLegTrace] = field(default_factory=list)
    stops: list[RouteStopTrace] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass(frozen=True)
class HCMCEvaluationResult:
    total_travel_time: float
    total_service_time: float
    total_distance: float
    total_route_duration: float
    fitness: float
    feasible: bool
    errors: list[str] = field(default_factory=list)
    routes: list[RouteEvaluation] = field(default_factory=list)
    total_waiting_time: float = 0.0
    departure_time_mode: str = "fixed"


class HCMCTDVRPEvaluator:
    """Đánh giá tuần tự theo thời gian trên road graph.

    Objective ``total_route_duration`` được giữ cố định là
    ``travel_time + service_time``.
    Waiting time được ghi riêng trong trace và không cộng vào objective.
    """

    def __init__(
        self,
        instance: TDVRPInstance,
        graph: tuple[dict[int, RoadNode], dict[str, RoadEdge]],
        traffic: TrafficOverlay,
        start_time: float | datetime = 8 * 3600,
        sample_step: float | None = None,
        fitness_fn=None,
        departure_time_mode: str = "fixed",
        departure_time_step: float | None = None,
        departure_time_window: tuple[float, float] | None = None,
    ) -> None:
        if departure_time_mode not in {"fixed", "optimize"}:
            raise ValueError("departure_time_mode must be 'fixed' or 'optimize'")
        self.instance = instance
        self.nodes, self.edges = graph
        self.traffic = traffic
        self.start_time = _seconds_since_midnight(start_time)
        self.sample_step = sample_step or instance.sample_step
        self.fitness_fn = fitness_fn or (lambda travel_time, service_time: travel_time + service_time)
        self.departure_time_mode = departure_time_mode
        self.departure_time_step = float(departure_time_step or instance.sample_step)
        if self.departure_time_step <= 0:
            raise ValueError("departure_time_step must be positive")
        horizon_start, horizon_end = map(float, instance.horizon)
        window = departure_time_window or (horizon_start, horizon_end)
        window_start, window_end = map(float, window)
        if not horizon_start <= window_start < window_end <= horizon_end:
            raise ValueError("departure_time_window must be inside instance.horizon")
        self.departure_time_window = (window_start, window_end)
        self.customer_by_id = {customer.id: customer for customer in instance.customers}
        self.depot_road_node = instance.road_node_ids[instance.depot]
        self.routing_cache: dict[tuple[int, int, int], FastestPathResult] = {}
        self.pinned_path_cache: dict[tuple[int, int], tuple[list[int], list[str]]] = {}
        self.departure_cache: dict[tuple[int, ...], RouteEvaluation] = {}
        self.routing_policy = str(instance.metadata.get("routing_policy", "dynamic_fastest_path"))

    def evaluate(self, solution: Solution) -> HCMCEvaluationResult:
        validation = validate_solution(solution, self.instance)
        if not validation.feasible:
            return HCMCEvaluationResult(
                0.0, 0.0, 0.0, inf, inf, False, validation.errors,
                departure_time_mode=self.departure_time_mode,
            )

        total_travel = total_service = total_distance = total_waiting = 0.0
        route_results: list[RouteEvaluation] = []
        errors: list[str] = []
        for vehicle_index, route in enumerate(solution.routes):
            if self.departure_time_mode == "optimize":
                route_result = self._optimize_route_departure(vehicle_index, route)
            else:
                route_result = self._evaluate_route(vehicle_index, route, self.start_time)
            route_results.append(route_result)
            total_travel += route_result.travel_time
            total_service += route_result.service_time
            total_distance += route_result.distance
            total_waiting += route_result.waiting_time
            errors.extend(route_result.errors)
        total_route_duration = total_travel + total_service
        feasible = not errors
        fitness = float(self.fitness_fn(total_travel, total_service)) if feasible else inf
        return HCMCEvaluationResult(
            total_travel,
            total_service,
            total_distance,
            total_route_duration if feasible else inf,
            fitness,
            feasible,
            errors,
            route_results,
            total_waiting,
            self.departure_time_mode,
        )

    def _optimize_route_departure(self, vehicle_index: int, route: list[int]) -> RouteEvaluation:
        customer_ids = tuple(customer_id for customer_id in route if customer_id != self.instance.depot)
        cached = self.departure_cache.get(customer_ids)
        if cached is not None:
            return _replace_vehicle_index(cached, vehicle_index)
        if not customer_ids:
            result = self._evaluate_route(vehicle_index, route, self.departure_time_window[0])
            self.departure_cache[customer_ids] = result
            return result

        candidates = self._departure_candidates()
        start, end = self.departure_time_window
        if self.start_time not in candidates and start <= self.start_time <= end:
            candidates.append(self.start_time)
        evaluations = [
            self._evaluate_route(vehicle_index, route, departure, include_path_trace=False)
            for departure in candidates
        ]
        feasible = [item for item in evaluations if item.feasible]
        pool = feasible or evaluations
        best = min(
            pool,
            key=lambda item: (item.travel_time + item.service_time, item.waiting_time, item.start_time),
        )
        detailed_best = self._evaluate_route(vehicle_index, route, best.start_time, include_path_trace=True)
        self.departure_cache[customer_ids] = detailed_best
        return detailed_best

    def _departure_candidates(self) -> list[float]:
        """Các mốc khởi hành; cuối cửa sổ riêng được tính, cuối horizon thì không."""
        start, end = self.departure_time_window
        horizon_end = float(self.instance.horizon[1])
        count = int((end - start) // self.departure_time_step)
        candidates = [start + index * self.departure_time_step for index in range(count + 1)]
        if end >= horizon_end:
            candidates = [value for value in candidates if value < horizon_end]
        elif candidates[-1] < end:
            candidates.append(end)
        return candidates or [start]

    def _evaluate_route(
        self,
        vehicle_index: int,
        route: list[int],
        start_time: float,
        include_path_trace: bool = True,
    ) -> RouteEvaluation:
        clock = float(start_time)
        current_node = self.depot_road_node
        current_customer_id = self.instance.depot
        total_travel = total_service = total_distance = total_waiting = load = 0.0
        node_path: list[int] = [current_node]
        edge_path: list[str] = []
        legs: list[RouteLegTrace] = []
        stops: list[RouteStopTrace] = [
            RouteStopTrace(self.instance.depot, current_node, clock, clock, clock, 0.0, 0.0, 0.0, 0.0)
        ]
        errors: list[str] = []
        customer_ids = [customer_id for customer_id in route if customer_id != self.instance.depot]

        for customer_id in customer_ids + [self.instance.depot]:
            target_node = self.depot_road_node if customer_id == self.instance.depot else self.customer_by_id[customer_id].road_node_id
            leg_departure = clock
            if leg_departure > self.instance.horizon[1] + 1e-9:
                errors.append(
                    f"Xe {vehicle_index + 1}: không thể xuất phát chặng tiếp theo sau horizon {self.instance.horizon[1]:.3f}"
                )
                break
            if self.instance.arrival_time_functions is not None and not include_path_trace and current_customer_id != customer_id:
                # Khi quét departure time, chỉ cần ATF; road trace được dựng lại cho ứng viên tốt nhất.
                atf = self.instance.arrival_time_functions[(current_customer_id, customer_id)]
                official_arrival = float(atf.evaluate(leg_departure))
                path = FastestPathResult([current_node, target_node], [], official_arrival - leg_departure)
            else:
                path = self._find_path(current_node, target_node, leg_departure)
                if self.instance.arrival_time_functions is not None and current_customer_id != customer_id:
                    # ATF chính thức là nguồn objective; road path vẫn được giữ để vẽ trace.
                    atf = self.instance.arrival_time_functions[(current_customer_id, customer_id)]
                    official_arrival = float(atf.evaluate(leg_departure))
                    path = FastestPathResult(path.node_path, path.edge_path, official_arrival - leg_departure)
            distance = sum(self.edges[edge_id].distance_m for edge_id in path.edge_path)
            arrival = leg_departure + path.travel_time_seconds
            total_travel += path.travel_time_seconds
            total_distance += distance
            node_path.extend(path.node_path[1:])
            edge_path.extend(path.edge_path)
            legs.append(
                RouteLegTrace(
                    current_customer_id,
                    customer_id,
                    leg_departure,
                    arrival,
                    path.travel_time_seconds,
                    distance,
                    list(path.node_path),
                    list(path.edge_path),
                )
            )

            waiting = service = demand = 0.0
            service_start = arrival
            if customer_id != self.instance.depot:
                customer = self.customer_by_id[customer_id]
                demand = customer.demand
                service = customer.service_time
                if self.instance.time_windows is not None:
                    earliest, latest = self.instance.time_windows[customer_id]
                    service_start = max(arrival, earliest)
                    waiting = service_start - arrival
                    if service_start > latest + 1e-9:
                        errors.append(
                            f"Xe {vehicle_index + 1}, khách {customer_id}: bắt đầu phục vụ {service_start:.3f} sau deadline {latest:.3f}"
                        )
                load += demand
                total_service += service
                total_waiting += waiting
            clock = service_start + service
            stops.append(
                RouteStopTrace(
                    customer_id,
                    target_node,
                    arrival,
                    service_start,
                    clock,
                    waiting,
                    service,
                    demand,
                    load,
                )
            )
            current_node = target_node
            current_customer_id = customer_id

        if clock > self.instance.horizon[1] + 1e-9:
            errors.append(
                f"Xe {vehicle_index + 1}: kết thúc {clock:.3f} vượt horizon {self.instance.horizon[1]:.3f}"
            )
        return RouteEvaluation(
            vehicle_index=vehicle_index,
            customer_ids=customer_ids,
            node_path=node_path,
            edge_path=edge_path,
            start_time=float(start_time),
            end_time=clock,
            travel_time=total_travel,
            service_time=total_service,
            distance=total_distance,
            load=load,
            waiting_time=total_waiting,
            feasible=not errors,
            errors=errors,
            legs=legs,
            stops=stops,
        )

    def _find_path(self, source_node: int, target_node: int, departure_time: float) -> FastestPathResult:
        if self.routing_policy == "pinned_free_flow_paths":
            return find_pinned_fastest_path(
                (self.nodes, self.edges),
                self.traffic,
                source_node,
                target_node,
                departure_time,
                path_cache=self.pinned_path_cache,
            )
        return find_fastest_path(
            (self.nodes, self.edges),
            self.traffic,
            source_node,
            target_node,
            departure_time=departure_time,
            sample_step=self.sample_step,
            cache=self.routing_cache,
        )


def _replace_vehicle_index(route: RouteEvaluation, vehicle_index: int) -> RouteEvaluation:
    """Tái sử dụng cache route nhưng giữ đúng chỉ số xe của nghiệm hiện tại."""
    return RouteEvaluation(
        vehicle_index,
        route.customer_ids,
        route.node_path,
        route.edge_path,
        route.start_time,
        route.end_time,
        route.travel_time,
        route.service_time,
        route.distance,
        route.load,
        route.waiting_time,
        route.feasible,
        route.errors,
        route.legs,
        route.stops,
    )


def _seconds_since_midnight(value: float | datetime) -> float:
    if isinstance(value, datetime):
        return float(value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1_000_000)
    return float(value) % 86400
