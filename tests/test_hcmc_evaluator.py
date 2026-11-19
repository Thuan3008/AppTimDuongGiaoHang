from pathlib import Path

from core.data.evaluator import HCMCTDVRPEvaluator
from core.data.load_instance import load_instance
from core.data.load_road_graph import load_road_graph
from core.data.load_traffic import load_traffic
from core.data.solution import repair_solution
from core.models import Solution


ROOT = Path(__file__).parents[1]


def hcmc_bundle():
    instance = load_instance(ROOT / "data/hcmc/instances/hcmc-n100-heavy.vrp.json")
    graph = load_road_graph(instance.graph_path)
    traffic = load_traffic(instance.traffic_path, graph[1])
    return instance, graph, traffic


def test_evaluator_simulates_all_hcmc_customers():
    instance, graph, traffic = hcmc_bundle()
    solution = repair_solution(Solution([[0, 1, 2, 3], [0, 4, 5]]), instance)
    result = HCMCTDVRPEvaluator(instance, graph, traffic, start_time=8 * 3600).evaluate(solution)
    assert result.feasible, result.errors
    assert result.total_route_duration == result.total_travel_time + result.total_service_time
    assert result.fitness == result.total_route_duration
    assert result.total_travel_time > 0
    assert result.total_service_time > 0
    assert result.total_distance > 0
    assert len(result.routes) == instance.num_vehicles
    assert sum(len(route.customer_ids) for route in result.routes) == instance.num_customers
    assert all(route.load <= instance.vehicle_capacity for route in result.routes)


def test_evaluator_rejects_invalid_solution_without_mutating_it():
    instance, graph, traffic = hcmc_bundle()
    solution = Solution([[0, 1, 1, 0]])
    original = [route[:] for route in solution.routes]
    result = HCMCTDVRPEvaluator(instance, graph, traffic).evaluate(solution)
    assert not result.feasible
    assert result.fitness == float("inf")
    assert solution.routes == original


def test_service_time_changes_next_departure_and_total_route_end():
    instance, graph, traffic = hcmc_bundle()
    first = repair_solution(Solution([[0, 1]]), instance)
    result = HCMCTDVRPEvaluator(instance, graph, traffic, start_time=8 * 3600).evaluate(first)
    active = next(route for route in result.routes if route.customer_ids)
    assert active.end_time > active.start_time + active.travel_time
