from pathlib import Path
import random

import numpy as np

from algorithms.aro_original import (
    AROOriginal,
    AROOriginalConfig,
    detour_foraging,
    energy_shrink_factor,
    random_hiding,
)
from algorithms.poa_original import (
    POAOriginal,
    POAOriginalConfig,
    defense_mechanism,
    predator_attack,
)
from core.tdvrp_common import TDVRPProblem


ROOT = Path(__file__).parents[1]
INSTANCE = (
    ROOT
    / "data/TDVRP/hong_kong/n=25/poryos-hong_kong-n25-hyb/bpr-heavy"
    / "poryos-hong_kong-n25-hyb-bpr-heavy.vrp.json"
)


def common_population(dimension: int, size: int, seed: int = 42):
    rng = random.Random(seed)
    customers = list(range(1, dimension + 1))
    return [tuple(rng.sample(customers, dimension)) for _ in range(size)]


def test_aro_original_operators_keep_random_key_domain():
    rng = np.random.default_rng(42)
    current = rng.random(25)
    peer = rng.random(25)
    assert energy_shrink_factor(1, 10, rng) >= 0.0
    for candidate in (
        detour_foraging(current, peer, 1, 10, rng),
        random_hiding(current, 8, 10, rng),
    ):
        assert candidate.shape == current.shape
        assert np.isfinite(candidate).all()
        assert np.all((0.0 <= candidate) & (candidate <= 1.0))


def test_poa_original_operators_keep_random_key_domain_and_shrink_defense():
    current = np.linspace(0.1, 0.9, 25)
    target = current[::-1]
    attack = predator_attack(current, target, np.random.default_rng(7))
    early = defense_mechanism(current, 1, np.random.default_rng(9))
    late = defense_mechanism(current, 10, np.random.default_rng(9))
    assert np.all((0.0 <= attack) & (attack <= 1.0))
    assert np.linalg.norm(late - current) <= np.linalg.norm(early - current)


def test_original_aro_and_poa_use_same_p0_and_exact_nfe_on_hong_kong():
    problem = TDVRPProblem(
        INSTANCE,
        collection_root=ROOT / "data/TDVRP",
        vehicle_limit=problem_vehicle_limit(INSTANCE),
    )
    population_size = 4
    nfe = 12
    initial = common_population(problem.dimension, population_size)
    aro = AROOriginal().optimize(
        problem,
        AROOriginalConfig(population_size, nfe),
        seed=42,
        initial_population=initial,
    )
    poa = POAOriginal().optimize(
        problem,
        POAOriginalConfig(population_size, nfe),
        seed=42,
        initial_population=initial,
    )
    assert aro.nfe == poa.nfe == nfe
    assert aro.metadata["initialization"] == "paired canonical rank keys"
    assert poa.metadata["initialization"] == "paired canonical rank keys"
    assert sorted(customer for route in aro.routes for customer in route if customer) == list(
        range(1, problem.dimension + 1)
    )
    assert sorted(customer for route in poa.routes for customer in route if customer) == list(
        range(1, problem.dimension + 1)
    )


def problem_vehicle_limit(_instance_path: Path) -> int:
    # The benchmark protocol used by RAPO-2709 permits up to n non-empty routes.
    return 25
