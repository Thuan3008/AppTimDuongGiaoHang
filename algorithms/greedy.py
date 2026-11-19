"""Capacity-aware nearest-feasible greedy baseline for HCMC TDVRP."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from time import perf_counter

from core.data.evaluator import HCMCEvaluationResult, HCMCTDVRPEvaluator
from core.data.models import RoadEdge, RoadNode, TDVRPInstance, TrafficOverlay
from core.models import Solution
from core.routing.fastest_path import FastestPathResult, find_fastest_paths_from_source, find_pinned_fastest_paths_from_source


@dataclass(frozen=True)
class GreedyResult:
    solution: Solution
    evaluation: HCMCEvaluationResult
    runtime_seconds: float


def build_greedy_solution(
    instance: TDVRPInstance,
    graph: tuple[dict[int, RoadNode], dict[str, RoadEdge]],
    traffic: TrafficOverlay,
    start_time: float | datetime = 8 * 3600,
) -> Solution:
    """Build routes by repeatedly choosing the fastest capacity-feasible next customer."""
    nodes, edges = graph
    clock_start = _seconds_since_midnight(start_time)
    unserved = set(range(1, instance.num_customers + 1))
    customer_by_id = {customer.id: customer for customer in instance.customers}
    routes: list[list[int]] = []
    pinned_cache: dict[tuple[int, int], tuple[list[int], list[str]]] = {}

    for _vehicle_index in range(instance.num_vehicles):
        route = [instance.depot]
        current_node = instance.road_node_ids[instance.depot]
        clock = clock_start
        load = 0.0

        while unserved:
            best: tuple[float, int, FastestPathResult] | None = None
            # Poryos ghim đường free-flow; HCMC cũ vẫn dùng fastest path động.
            if instance.metadata.get("routing_policy") == "pinned_free_flow_paths":
                paths = find_pinned_fastest_paths_from_source(
                    (nodes, edges),
                    traffic,
                    current_node,
                    clock,
                    path_cache=pinned_cache,
                    target_nodes={customer_by_id[customer_id].road_node_id for customer_id in unserved},
                )
            else:
                paths = find_fastest_paths_from_source(
                    (nodes, edges),
                    traffic,
                    current_node,
                    clock,
                    target_nodes={customer_by_id[customer_id].road_node_id for customer_id in unserved},
                )
            for customer_id in sorted(unserved):
                customer = customer_by_id[customer_id]
                if load + customer.demand > instance.vehicle_capacity + 1e-9:
                    continue
                path = paths[customer.road_node_id]
                candidate = (path.travel_time_seconds, customer_id, path)
                if best is None or candidate[:2] < best[:2]:
                    best = candidate

            if best is None:
                break
            _, customer_id, path = best
            customer = customer_by_id[customer_id]
            route.append(customer_id)
            unserved.remove(customer_id)
            load += customer.demand
            clock += path.travel_time_seconds + customer.service_time
            current_node = customer.road_node_id

        route.append(instance.depot)
        routes.append(route)
        if not unserved:
            break

    if unserved:
        raise ValueError(f"Greedy could not assign customers: {sorted(unserved)}")
    routes.extend([[instance.depot, instance.depot] for _ in range(instance.num_vehicles - len(routes))])
    return Solution(routes=routes)


def run_greedy(
    instance: TDVRPInstance,
    graph: tuple[dict[int, RoadNode], dict[str, RoadEdge]],
    traffic: TrafficOverlay,
    start_time: float | datetime = 8 * 3600,
) -> GreedyResult:
    started = perf_counter()
    solution = build_greedy_solution(instance, graph, traffic, start_time)
    evaluation = HCMCTDVRPEvaluator(instance, graph, traffic, start_time).evaluate(solution)
    return GreedyResult(solution, evaluation, perf_counter() - started)


def _seconds_since_midnight(value: float | datetime) -> float:
    if isinstance(value, datetime):
        return float(value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1_000_000)
    return float(value) % 86400
