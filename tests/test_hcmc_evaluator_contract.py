"""Contract tests for the road-graph TDVRP evaluator."""

from dataclasses import replace

from core.data.evaluator import OBJECTIVE_NAME, HCMCTDVRPEvaluator
from core.data.models import (
    Customer,
    RoadEdge,
    RoadNode,
    TDVRPInstance,
    TrafficEdgeProfile,
    TrafficOverlay,
)
from core.data.solution import validate_solution
from core.models import Solution


def _speeds(default: float, *, peak: float | None = None) -> dict[str, float]:
    values = {f"{hour:02d}": default for hour in range(24)}
    if peak is not None:
        values["08"] = peak
    return values


def _bundle():
    nodes = {
        0: RoadNode(0, 10.0, 106.0),
        1: RoadNode(1, 10.1, 106.1),
        2: RoadNode(2, 10.05, 106.05),
        3: RoadNode(3, 10.2, 106.2),
    }
    edges = {
        "direct": RoadEdge("direct", 0, 1, "Direct", 1000, 60, True),
        "detour-a": RoadEdge("detour-a", 0, 2, "Detour A", 1200, 60, True),
        "detour-b": RoadEdge("detour-b", 2, 1, "Detour B", 1200, 60, True),
        "customer-2": RoadEdge("customer-2", 0, 3, "Customer 2", 1000, 60, True),
    }
    traffic = TrafficOverlay(
        "test-hourly",
        {
            "direct": TrafficEdgeProfile("direct", "Direct", _speeds(60, peak=5)),
            "detour-a": TrafficEdgeProfile("detour-a", "Detour A", _speeds(60)),
            "detour-b": TrafficEdgeProfile("detour-b", "Detour B", _speeds(60)),
            "customer-2": TrafficEdgeProfile("customer-2", "Customer 2", _speeds(60)),
        },
    )
    customers = [
        Customer(1, 1, 4, 30, (10.1, 106.1)),
        Customer(2, 3, 4, 40, (10.2, 106.2)),
    ]
    instance = TDVRPInstance(
        instance_name="synthetic-contract",
        benchmark_name="HCMC-TDVRP",
        num_customers=2,
        num_vehicles=2,
        vehicle_capacity=5,
        coordinates=[(10.0, 106.0), (10.1, 106.1), (10.2, 106.2)],
        demands=[0, 4, 4],
        service_times=[0, 30, 40],
        road_node_ids=[0, 1, 3],
        depot=0,
        reference_lla={"lat": 10.0, "lon": 106.0, "alt": 0.0},
        horizon=(0, 86400),
        sample_step=300,
        graph_path="synthetic",
        traffic_path="synthetic",
        metadata={"routing_policy": "dynamic_fastest_path"},
        customers=customers,
    )
    solution = Solution([[0, 1, 0], [0, 2, 0]])
    return instance, (nodes, edges), traffic, solution


def test_objective_is_total_route_duration():
    instance, graph, traffic, solution = _bundle()
    result = HCMCTDVRPEvaluator(instance, graph, traffic, start_time=3600).evaluate(solution)
    assert OBJECTIVE_NAME == "total_route_duration"
    assert result.fitness == result.total_route_duration
    assert result.total_route_duration == result.total_travel_time + result.total_service_time


def test_departure_time_changes_travel_time_and_selected_edges():
    instance, graph, traffic, solution = _bundle()
    off_peak = HCMCTDVRPEvaluator(instance, graph, traffic, start_time=3600).evaluate(solution)
    peak = HCMCTDVRPEvaluator(instance, graph, traffic, start_time=8 * 3600).evaluate(solution)
    assert off_peak.total_travel_time != peak.total_travel_time
    assert off_peak.routes[0].legs[0].edge_path == ["direct"]
    assert peak.routes[0].legs[0].edge_path == ["detour-a", "detour-b"]
    assert off_peak.routes[0].edge_path != peak.routes[0].edge_path


def test_capacity_violation_is_rejected_before_evaluation():
    instance, graph, traffic, _ = _bundle()
    invalid = Solution([[0, 1, 2, 0], [0, 0]])
    validation = validate_solution(invalid, instance)
    result = HCMCTDVRPEvaluator(instance, graph, traffic).evaluate(invalid)
    assert not validation.feasible
    assert any("exceeds capacity" in error for error in validation.errors)
    assert not result.feasible
    assert result.fitness == float("inf")


def test_time_window_violation_is_reported():
    instance, graph, traffic, solution = _bundle()
    constrained = replace(instance, time_windows=[(0, 86400), (0, 10), (0, 86400)])
    result = HCMCTDVRPEvaluator(constrained, graph, traffic, start_time=0).evaluate(solution)
    assert not result.feasible
    assert any("deadline" in error for error in result.errors)


def test_vehicles_are_simulated_independently_from_same_departure():
    instance, graph, traffic, solution = _bundle()
    result = HCMCTDVRPEvaluator(instance, graph, traffic, start_time=3600).evaluate(solution)
    assert result.feasible
    assert len(result.routes) == 2
    assert [route.start_time for route in result.routes] == [3600, 3600]
    assert [route.customer_ids for route in result.routes] == [[1], [2]]
    assert all(route.load <= instance.vehicle_capacity for route in result.routes)


def test_every_route_returns_to_depot_with_a_final_leg():
    instance, graph, traffic, solution = _bundle()
    result = HCMCTDVRPEvaluator(instance, graph, traffic, start_time=3600).evaluate(solution)
    for route in result.routes:
        assert route.stops[0].customer_id == instance.depot
        assert route.stops[-1].customer_id == instance.depot
        assert route.legs[-1].to_customer_id == instance.depot
        assert route.node_path[-1] == instance.road_node_ids[instance.depot]
