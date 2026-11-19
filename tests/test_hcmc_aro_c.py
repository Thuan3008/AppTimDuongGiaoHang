from pathlib import Path

import numpy as np

from algorithms.hcmc_aro_c import AROContinuousConfig, AROContinuous, run_aro_c
from core.data.load_instance import load_instance
from core.data.load_road_graph import load_road_graph
from core.data.load_traffic import load_traffic
from core.data.solution import validate_solution


ROOT = Path(__file__).parents[1]


def test_aro_c_propose_keeps_random_key_dimension():
    rng = np.random.default_rng(42)
    population = rng.random((4, 10))
    fitness = np.array([4.0, 3.0, 2.0, 1.0])
    candidate = next(AROContinuous.propose(population, fitness, 0, 0, 5, rng))
    assert candidate.shape == (10,)
    assert np.isfinite(candidate).all()


def test_aro_c_runs_on_hcmc_and_decodes_feasible_routes():
    instance = load_instance(ROOT / "data/hcmc/instances/hcmc-n10-heavy.vrp.json")
    graph = load_road_graph(instance.graph_path)
    traffic = load_traffic(instance.traffic_path, graph[1])
    result = run_aro_c(instance, graph, traffic, AROContinuousConfig(population_size=4, max_function_evaluations=16, max_iterations=3), seed=42)
    assert result.best_evaluation.feasible
    assert validate_solution(result.best_solution, instance).feasible
    assert len(result.convergence) == 4
    assert result.convergence[-1] <= result.convergence[0]
