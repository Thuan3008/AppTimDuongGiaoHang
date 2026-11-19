from pathlib import Path

from algorithms.td_cr_rapo import TDCRRAPOConfig, optimize, split_decoder
from core.data.load_instance import load_instance
from core.data.load_road_graph import load_road_graph
from core.data.load_traffic import load_traffic
from core.data.solution import validate_solution

ROOT = Path(__file__).parents[1]


def test_split_decoder_preserves_route_contract():
    instance = load_instance(ROOT / "data/hcmc/instances/hcmc-n10-heavy.vrp.json")
    solution = split_decoder(range(1, instance.num_customers + 1), instance)
    assert validate_solution(solution, instance).feasible


def test_td_cr_rapo_runs_with_strict_nfe():
    instance = load_instance(ROOT / "data/hcmc/instances/hcmc-n10-heavy.vrp.json")
    graph = load_road_graph(instance.graph_path)
    traffic = load_traffic(instance.traffic_path, graph[1])
    result = optimize(instance, graph, traffic, TDCRRAPOConfig(population_size=4, max_function_evaluations=24, max_iterations=40), seed=42)
    assert result.valid_architecture
    assert result.best_evaluation.feasible
    assert validate_solution(result.best_solution, instance).feasible
    assert sum(result.nfe_by_branch.values()) <= 24
    assert result.nfe_by_branch["initialization"] == 4
    assert len(result.best_permutation) == instance.num_customers

