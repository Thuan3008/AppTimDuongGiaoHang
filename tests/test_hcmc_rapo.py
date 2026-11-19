from pathlib import Path

from algorithms.hcmc_rapo import RAPOConfig, run_rapo
from core.data.load_instance import load_instance
from core.data.load_road_graph import load_road_graph
from core.data.load_traffic import load_traffic
from core.data.solution import validate_solution

ROOT = Path(__file__).parents[1]

def test_rapo_runs_on_hcmc():
    instance = load_instance(ROOT / "data/hcmc/instances/hcmc-n10-heavy.vrp.json")
    graph = load_road_graph(instance.graph_path)
    traffic = load_traffic(instance.traffic_path, graph[1])
    result = run_rapo(instance, graph, traffic, RAPOConfig(population_size=4, max_function_evaluations=24, max_iterations=3, stagnation_limit=2), seed=42)
    assert result.best_evaluation.feasible
    assert validate_solution(result.best_solution, instance).feasible
    assert result.evaluations <= 24
    assert result.convergence[-1] <= result.convergence[0]
