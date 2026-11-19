from pathlib import Path

from algorithms.td_cr_rapo_v3 import TDCRRAPOV3Config, optimize
from core.data.load_instance import load_instance
from core.data.load_road_graph import load_road_graph
from core.data.load_traffic import load_traffic
from core.data.solution import validate_solution

ROOT = Path(__file__).parents[1]


def test_td_cr_rapo_v3_hcmc_run_and_accounting():
    instance = load_instance(ROOT / "data/hcmc/instances/hcmc-n10-heavy.vrp.json")
    graph = load_road_graph(instance.graph_path)
    traffic = load_traffic(instance.traffic_path, graph[1])
    result = optimize(instance, graph, traffic, TDCRRAPOV3Config(population_size=4, max_function_evaluations=30, max_iterations=40), seed=42)
    assert result.best_evaluation.feasible
    assert validate_solution(result.best_solution, instance).feasible
    assert result.nfe <= 30
    assert sum(result.spending.values()) == result.nfe
    assert result.metadata["encoding"] == "permutation + split decoder"
    assert len(result.best_permutation) == instance.num_customers

