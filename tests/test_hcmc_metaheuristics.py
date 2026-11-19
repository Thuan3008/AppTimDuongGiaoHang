from pathlib import Path

from algorithms.hcmc_aro_goc import AROConfig, run_aro
from algorithms.hcmc_poa_goc import POAConfig, run_poa
from core.data.load_instance import load_instance
from core.data.load_road_graph import load_road_graph
from core.data.load_traffic import load_traffic
from core.data.solution import validate_solution


ROOT = Path(__file__).parents[1]


def hcmc_bundle():
    instance = load_instance(ROOT / "data/hcmc/instances/hcmc-n10-heavy.vrp.json")
    graph = load_road_graph(instance.graph_path)
    traffic = load_traffic(instance.traffic_path, graph[1])
    return instance, graph, traffic


def test_discrete_aro_runs_independently_on_hcmc():
    instance, graph, traffic = hcmc_bundle()
    result = run_aro(instance, graph, traffic, config=AROConfig(population_size=4, iterations=3, seed=42))
    assert result.best_evaluation.feasible
    assert validate_solution(result.best_solution, instance).feasible
    assert len(result.convergence) == 4
    assert result.convergence[-1] <= result.convergence[0]


def test_discrete_poa_runs_independently_on_hcmc():
    instance, graph, traffic = hcmc_bundle()
    result = run_poa(instance, graph, traffic, config=POAConfig(population_size=4, iterations=3, seed=42))
    assert result.best_evaluation.feasible
    assert validate_solution(result.best_solution, instance).feasible
    assert len(result.convergence) == 4
    assert result.convergence[-1] <= result.convergence[0]
