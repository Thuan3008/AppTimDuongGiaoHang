from pathlib import Path

from algorithms.greedy import run_greedy
from core.data.load_instance import load_instance
from core.data.load_road_graph import load_road_graph
from core.data.load_traffic import load_traffic
from core.data.solution import validate_solution


ROOT = Path(__file__).parents[1]


def test_greedy_hcmc_baseline_is_feasible_and_uses_evaluator():
    instance = load_instance(ROOT / "data/hcmc/instances/hcmc-n100-heavy.vrp.json")
    graph = load_road_graph(instance.graph_path)
    traffic = load_traffic(instance.traffic_path, graph[1])
    result = run_greedy(instance, graph, traffic, start_time=8 * 3600)
    validation = validate_solution(result.solution, instance)
    assert validation.feasible, validation.errors
    assert result.evaluation.feasible
    assert result.evaluation.fitness == result.evaluation.total_route_duration
    assert result.evaluation.total_travel_time > 0
    assert result.evaluation.total_distance > 0
    assert result.runtime_seconds >= 0


def test_greedy_is_deterministic_for_fixed_hcmc_data():
    instance = load_instance(ROOT / "data/hcmc/instances/hcmc-n100-heavy.vrp.json")
    graph = load_road_graph(instance.graph_path)
    traffic = load_traffic(instance.traffic_path, graph[1])
    first = run_greedy(instance, graph, traffic, start_time=17 * 3600)
    second = run_greedy(instance, graph, traffic, start_time=17 * 3600)
    assert first.solution.routes == second.solution.routes
    assert first.evaluation.fitness == second.evaluation.fitness
