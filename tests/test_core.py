from datetime import datetime
from pathlib import Path

import numpy as np

from algorithms.ms_irapo import MSIRAPOConfig, MSIRAPOOptimizer
from algorithms.operators.repair import repair_solution
from algorithms.population import initialize_population, split_population
from core.distance import build_distance_matrix
from core.models import Customer, Depot, Solution, TDVRPProblem, Vehicle
from core.travel_time import SimulatedTravelTimeProvider
from core.validation import validate_solution
from utils.data_loader import load_problem


def problem():
    p = TDVRPProblem(Depot(), [Customer(1, 0, 0, 2, 5), Customer(2, 3, 4, 2, 5), Customer(3, 6, 0, 2, 5)], [Vehicle(1, 5), Vehicle(2, 5)], base_speed=1)
    p.distance_matrix = build_distance_matrix(p.nodes); return p


def test_distance_and_traffic():
    p = problem(); assert np.isclose(p.distance_matrix[1, 2], 5)
    provider = SimulatedTravelTimeProvider(p)
    assert provider.get_travel_time(0, 1, datetime(2026, 1, 1, 8)) != provider.get_travel_time(0, 1, datetime(2026, 1, 1, 11))


def test_validation_repair_duplicate_missing():
    p = problem(); bad = Solution([[0, 1, 1, 0], [0, 0]])
    assert not validate_solution(bad, p).feasible
    fixed = repair_solution(bad, p); assert validate_solution(fixed, p).feasible


def test_population_split_and_seed():
    p = load_problem(Path(__file__).parents[1] / "data")
    a = initialize_population(p, 12, np.random.default_rng(42)); b = initialize_population(p, 12, np.random.default_rng(42))
    assert [s.routes for s in a] == [s.routes for s in b]
    assert [len(x) for x in split_population(a, 5)] == [3, 3, 2, 2, 2]


def test_optimizer_is_feasible_and_supports_one_swarm():
    p = load_problem(Path(__file__).parents[1] / "data")
    config = MSIRAPOConfig(population_size=12, num_swarms=1, max_iterations=8, migration_interval=2, seed=7)
    result = MSIRAPOOptimizer(datetime(2026, 1, 1, 8)).solve(p, config)
    assert result.global_best.is_feasible and len(result.convergence) == 9 and not result.migration_events


def test_hong_kong_benchmark_is_loaded():
    p = load_problem(Path(__file__).parents[1] / "data")
    assert p.instance_name == "poryos-hong_kong-n10-poi-bpr-light"
    assert len(p.customers) == 10 and len(p.vehicles) == 4
    assert max(c.service_time for c in p.customers) < 20
