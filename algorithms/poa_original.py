"""Original Pufferfish Optimization Algorithm for Poryos/Hong Kong TDVRP.

POA's two continuous phases are preserved.  Continuous positions in [0, 1]^n
are decoded by random-key sorting, then evaluated by the shared exact
time-dependent Split evaluator.  This keeps Predator Attack and Defense
Mechanism separate from the route operators used by RAPO.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
import time
from typing import Iterable

import numpy as np

from algorithms.rapo import Candidate, EvaluationLedger
from algorithms.rapo_operators import from_keys, rank_keys
from core.tdvrp_common import OptimizationResult, TDVRPProblem


@dataclass(frozen=True)
class POAOriginalConfig:
    population_size: int = 30
    max_function_evaluations: int = 1000
    max_iterations: int | None = None

    @classmethod
    def from_mapping(cls, config=None) -> "POAOriginalConfig":
        if isinstance(config, cls):
            return config
        values = dict(config or {})
        if not values.pop("minimization", True):
            raise ValueError("TDVRP requires minimization")
        values.pop("use_evaluation_budget", None)
        values.pop("run_id", None)
        for source, target in (
            ("pop_size", "population_size"),
            ("max_nfe", "max_function_evaluations"),
            ("target_nfe", "max_function_evaluations"),
        ):
            if source in values:
                values.setdefault(target, values.pop(source))
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown POA-original configuration: {sorted(unknown)}")
        result = cls(**values)
        if type(result.population_size) is not int or result.population_size < 2:
            raise ValueError("population_size must be an integer >= 2")
        if (
            type(result.max_function_evaluations) is not int
            or result.max_function_evaluations < result.population_size
        ):
            raise ValueError("NFE budget must cover the initial population")
        if result.max_iterations is not None and (
            type(result.max_iterations) is not int or result.max_iterations < 0
        ):
            raise ValueError("max_iterations must be None or a non-negative integer")
        return result


@dataclass(frozen=True)
class Pufferfish:
    position: np.ndarray
    candidate: Candidate

    @property
    def cost(self) -> float:
        return float(self.candidate.cost)


def repair_bounds(position: np.ndarray) -> np.ndarray:
    """Project a POA position to the random-key domain [0, 1]^n."""

    return np.clip(np.asarray(position, dtype=float), 0.0, 1.0)


def predator_attack(
    current: np.ndarray,
    target: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """POA Eq. (5): x(P1) = x + r*(SP - I*x), I in {1, 2}."""

    x_i = np.asarray(current, dtype=float)
    sp_i = np.asarray(target, dtype=float)
    if x_i.shape != sp_i.shape or x_i.ndim != 1:
        raise ValueError("current and target must be one-dimensional vectors of equal size")
    random_scale = rng.random(len(x_i))
    integer_switch = rng.integers(1, 3, size=len(x_i))
    return repair_bounds(x_i + random_scale * (sp_i - integer_switch * x_i))


def defense_mechanism(
    current: np.ndarray,
    iteration: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """POA Eq. (7) on [0,1]^n: x(P2) = x + (1-2r)/t."""

    if iteration < 1:
        raise ValueError("POA iteration must be >= 1")
    x_i = np.asarray(current, dtype=float)
    if x_i.ndim != 1 or not len(x_i):
        raise ValueError("current must be a non-empty one-dimensional vector")
    random_scale = rng.random(len(x_i))
    return repair_bounds(x_i + (1.0 - 2.0 * random_scale) / iteration)


def _prepare_initial_positions(
    problem: TDVRPProblem,
    config: POAOriginalConfig,
    rng: np.random.Generator,
    initial_population: Iterable[Iterable[int]] | None,
) -> list[np.ndarray]:
    random_positions = rng.random((config.population_size, problem.dimension))
    if initial_population is None:
        return [row.copy() for row in random_positions]
    permutations = [tuple(int(customer) for customer in row) for row in initial_population]
    if len(permutations) != config.population_size:
        raise ValueError("initial_population must contain exactly population_size permutations")
    expected = list(range(1, problem.dimension + 1))
    if any(sorted(permutation) != expected for permutation in permutations):
        raise ValueError("Every initial candidate must be a permutation of customers 1..n")
    return [rank_keys(permutation) for permutation in permutations]


class POAOriginal:
    """Two-phase continuous POA with a random-key TDVRP decoder."""

    name = "POA-Original"
    hybrid_type = "continuous-original-random-keys"

    def optimize(
        self,
        problem: TDVRPProblem,
        config=None,
        seed: int | None = None,
        observer=None,
        initial_population: Iterable[Iterable[int]] | None = None,
    ) -> OptimizationResult:
        if not isinstance(problem, TDVRPProblem):
            raise TypeError("Expected the shared TDVRPProblem")
        run_id = int(config.get("run_id", 1)) if isinstance(config, dict) else 1
        settings = POAOriginalConfig.from_mapping(config)
        rng = np.random.default_rng(seed)
        ledger = EvaluationLedger(problem, settings.max_function_evaluations)
        ledger.spending.update(poa_attack=0, poa_defense=0)
        started = time.perf_counter()

        positions = _prepare_initial_positions(problem, settings, rng, initial_population)
        population = [
            Pufferfish(position, ledger.evaluate(from_keys(position), "initialization"))
            for position in positions
        ]
        best = min(population, key=lambda fish: fish.cost)
        convergence = [best.cost]
        generation = 0

        if observer is not None:
            observer.on_generation(
                [fish.candidate for fish in population],
                best.cost,
                generation,
                ledger.nfe,
                time.perf_counter() - started,
            )

        while ledger.remaining and (
            settings.max_iterations is None or generation < settings.max_iterations
        ):
            generation += 1
            for index in range(len(population)):
                if not ledger.remaining:
                    break
                parent = population[index]
                better_indices = [
                    candidate_index
                    for candidate_index, candidate in enumerate(population)
                    if candidate_index != index and candidate.cost < parent.cost
                ]

                # The source paper does not define CP_i = empty explicitly.
                # Following the supplied report, skip Attack and continue with Defense.
                if better_indices:
                    target_index = int(rng.choice(better_indices))
                    attack_position = predator_attack(
                        parent.position, population[target_index].position, rng
                    )
                    attack = ledger.evaluate(from_keys(attack_position), "poa_attack")
                    if attack.cost < parent.cost:
                        population[index] = Pufferfish(attack_position, attack)
                        parent = population[index]
                        if attack.cost < best.cost:
                            best = parent

                if not ledger.remaining:
                    break
                defense_position = defense_mechanism(parent.position, generation, rng)
                defense = ledger.evaluate(from_keys(defense_position), "poa_defense")
                if defense.cost < parent.cost:
                    population[index] = Pufferfish(defense_position, defense)
                    if defense.cost < best.cost:
                        best = population[index]

            generation_best = min(population, key=lambda fish: fish.cost)
            if generation_best.cost < best.cost:
                best = generation_best
            convergence.append(best.cost)
            if observer is not None:
                observer.on_generation(
                    [fish.candidate for fish in population],
                    best.cost,
                    generation,
                    ledger.nfe,
                    time.perf_counter() - started,
                )

        if ledger.nfe != settings.max_function_evaluations and settings.max_iterations is None:
            raise AssertionError("POA-original did not consume the strict NFE budget")
        if sorted(customer for route in best.candidate.routes for customer in route) != list(
            range(1, problem.dimension + 1)
        ):
            raise AssertionError("POA-original archive is not a complete customer solution")

        return OptimizationResult(
            algorithm=self.name,
            benchmark=problem.name,
            seed=seed,
            best_fitness=best.cost,
            best_solution=best.position.tolist(),
            best_permutation=list(best.candidate.permutation),
            routes=[[0, *route, 0] for route in best.candidate.routes],
            departure_times=[item[1] for item in best.candidate.evaluations],
            return_times=[item[2] for item in best.candidate.evaluations],
            route_durations=[item[0] for item in best.candidate.evaluations],
            route_loads=[
                sum(problem.demands[customer] for customer in route)
                for route in best.candidate.routes
            ],
            convergence_curve=convergence,
            runtime_seconds=time.perf_counter() - started,
            population_size=settings.population_size,
            max_iterations=generation,
            nfe=ledger.nfe,
            dimension=problem.dimension,
            hybrid_type=self.hybrid_type,
            run_id=run_id,
            metadata={
                "objective": "Duration",
                "units": "seconds",
                "feasible": True,
                "implementation": "poa-original-continuous-random-keys-v1",
                "equations": "POA Eq. (4)-(7)",
                "encoding": "continuous random keys + shared minimum-duration Split",
                "adaptation_scope": "representation/decoder only",
                "population_update": "sequential",
                "empty_target_policy": "skip attack, execute defense",
                "initialization": (
                    "paired canonical rank keys"
                    if initial_population is not None
                    else "uniform random keys"
                ),
                "config": asdict(settings),
                "completed_iterations": generation,
                "evaluations": ledger.nfe,
                "nfe_by_phase": dict(ledger.spending),
                "stop_reason": (
                    "nfe_budget" if not ledger.remaining else "iteration_limit"
                ),
            },
        )


POA = POAOriginal
POAConfig = POAOriginalConfig

