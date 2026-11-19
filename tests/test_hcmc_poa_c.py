from pathlib import Path

import numpy as np

from algorithms.hcmc_poa_c import PufferfishConfig, PufferfishOptimization, calculate_defense_radius, run_poa_c
from core.data.load_instance import load_instance
from core.data.load_road_graph import load_road_graph
from core.data.load_traffic import load_traffic
from core.data.solution import validate_solution


ROOT = Path(__file__).parents[1]


def test_poa_permutation_operators_and_radius():
    rng = np.random.default_rng(42)
    parent = tuple(range(1, 11))
    target = tuple(reversed(parent))
    attack = PufferfishOptimization.predator_attack(parent, target, rng)
    defense = PufferfishOptimization.pufferfish_defense(parent, 1, rng)
    assert sorted(attack) == list(parent)
    assert sorted(defense) == list(parent)
    assert calculate_defense_radius(10, 2) >= calculate_defense_radius(10, 3)


def test_poa_c_uses_two_nfe_per_cycle_and_returns_feasible_routes():
    instance = load_instance(ROOT / "data/hcmc/instances/hcmc-n10-heavy.vrp.json")
    graph = load_road_graph(instance.graph_path)
    traffic = load_traffic(instance.traffic_path, graph[1])
    result = run_poa_c(instance, graph, traffic, PufferfishConfig(population_size=4, max_function_evaluations=12, max_iterations=3), seed=42)
    assert result.best_evaluation.feasible
    assert validate_solution(result.best_solution, instance).feasible
    assert result.nfe <= 12
    assert result.metadata["attack_nfe"] == result.metadata["defense_nfe"]
    assert len(result.convergence) >= 1
