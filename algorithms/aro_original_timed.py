# Timed experiment copy; original source remains unchanged.
"""Original Artificial Rabbits Optimization for Poryos/Hong Kong TDVRP.

The ARO equations are kept in their continuous form.  A position in [0, 1]^n
is converted to a customer permutation with random keys; the shared TDVRP
problem then applies its capacity-feasible minimum-duration Split decoder and
the exact time-dependent evaluator.  Random keys are the representation
adapter only: they are not a new ARO search operator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
import time
from typing import Iterable

import numpy as np

from algorithms.rapo import Candidate
from algorithms.timed_runtime import TimedLedger as EvaluationLedger
from algorithms.rapo_operators import from_keys, rank_keys
from core.tdvrp_common import OptimizationResult, TDVRPProblem


@dataclass(frozen=True)
class AROOriginalConfig:
    population_size: int = 30
    max_function_evaluations: int = 1000
    max_iterations: int | None = None

    @classmethod
    def from_mapping(cls, config=None) -> "AROOriginalConfig":
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
            raise ValueError(f"Unknown ARO-original configuration: {sorted(unknown)}")
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
class Rabbit:
    position: np.ndarray
    candidate: Candidate

    @property
    def cost(self) -> float:
        return float(self.candidate.cost)


def repair_bounds(position: np.ndarray) -> np.ndarray:
    """Project a continuous ARO position to the random-key domain [0, 1]^n."""

    return np.clip(np.asarray(position, dtype=float), 0.0, 1.0)


def energy_shrink_factor(t: int, max_iterations: int, rng: np.random.Generator) -> float:
    """ARO Eq. (6)-(7): A(t) = 4(1-t/T) ln(1/r)."""

    if not 1 <= t <= max_iterations:
        raise ValueError("ARO iteration t must satisfy 1 <= t <= T")
    r = max(float(rng.random()), np.finfo(float).tiny)
    return float(4.0 * (1.0 - t / max_iterations) * math.log(1.0 / r))


def running_operator(
    dimension: int,
    t: int,
    max_iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """ARO Eq. (3)-(5): R = L*c with a random mapping vector c."""

    if dimension < 1:
        raise ValueError("dimension must be positive")
    if not 1 <= t <= max_iterations:
        raise ValueError("ARO iteration t must satisfy 1 <= t <= T")
    r2 = float(rng.random())
    length = (
        math.e - math.exp(((t - 1) / max_iterations) ** 2)
    ) * math.sin(2.0 * math.pi * r2)
    selected_count = int(math.ceil(float(rng.random()) * dimension))
    mapping = np.zeros(dimension, dtype=float)
    if selected_count:
        selected = rng.choice(dimension, size=selected_count, replace=False)
        mapping[selected] = 1.0
    return length * mapping


def detour_foraging(
    current: np.ndarray,
    peer: np.ndarray,
    t: int,
    max_iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """ARO Eq. (2): generate one exploration candidate around another rabbit."""

    x_i = np.asarray(current, dtype=float)
    x_j = np.asarray(peer, dtype=float)
    if x_i.shape != x_j.shape or x_i.ndim != 1:
        raise ValueError("current and peer must be one-dimensional vectors of equal size")
    running = running_operator(len(x_i), t, max_iterations, rng)
    r1 = float(rng.random())
    noise_switch = 1.0 if 0.5 * (0.05 + r1) >= 0.5 else 0.0
    normal_noise = rng.normal(0.0, 1.0, size=len(x_i))
    trial = x_j + running * (x_i - x_j) + noise_switch * normal_noise
    return repair_bounds(trial)


def random_hiding(
    current: np.ndarray,
    t: int,
    max_iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """ARO Eq. (10)-(13): choose one random burrow and exploit around it."""

    x_i = np.asarray(current, dtype=float)
    if x_i.ndim != 1 or not len(x_i):
        raise ValueError("current must be a non-empty one-dimensional vector")
    dimension = len(x_i)
    r4 = float(rng.random())
    hiding_factor = ((max_iterations - t + 1) / max_iterations) * r4
    direction = np.zeros(dimension, dtype=float)
    direction[int(rng.integers(0, dimension))] = 1.0
    burrow = x_i + hiding_factor * direction * x_i
    r5 = float(rng.random())
    running = running_operator(dimension, t, max_iterations, rng)
    return repair_bounds(x_i + running * (r5 * burrow - x_i))


def _other_rabbit(index: int, population_size: int, rng: np.random.Generator) -> int:
    if population_size < 2:
        raise ValueError("Detour foraging requires at least two rabbits")
    peer = int(rng.integers(0, population_size - 1))
    return peer + int(peer >= index)


def _prepare_initial_positions(
    problem: TDVRPProblem,
    config: AROOriginalConfig,
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
    # The common P0 controls the decoded solutions. Canonical rank keys provide
    # one deterministic continuous representative for each shared permutation.
    return [rank_keys(permutation) for permutation in permutations]


class AROOriginal:
    """Continuous ARO equations with a random-key TDVRP decoder."""

    name = "ARO-Original"
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
        settings = AROOriginalConfig.from_mapping(config)
        rng = np.random.default_rng(seed)
        ledger = EvaluationLedger(problem, settings.max_function_evaluations)
        ledger.spending.update(aro_detour=0, aro_hiding=0)
        started = time.perf_counter()

        positions = _prepare_initial_positions(problem, settings, rng, initial_population)
        population = [
            Rabbit(position, ledger.evaluate(from_keys(position), "initialization"))
            for position in positions
        ]
        best = min(population, key=lambda rabbit: rabbit.cost)
        convergence = [best.cost]
        generation = 0
        search_budget = settings.max_function_evaluations - settings.population_size
        time_horizon = settings.max_iterations or max(
            1, math.ceil(search_budget / settings.population_size)
        )

        time_horizon = 1000000
        if observer is not None:
            observer.on_generation(
                [rabbit.candidate for rabbit in population],
                best.cost,
                generation,
                ledger.nfe,
                time.perf_counter() - started,
            )

        while ledger.remaining and (
            settings.max_iterations is None or generation < settings.max_iterations
        ):
            generation += 1
            snapshot = [rabbit.position.copy() for rabbit in population]
            for index, parent in enumerate(population):
                if not ledger.remaining:
                    break
                energy = energy_shrink_factor(
                    max(1, int(problem.time_budget.fraction * 1000000)), time_horizon, rng
                )
                if energy > 1.0:
                    peer_index = _other_rabbit(index, len(population), rng)
                    trial_position = detour_foraging(
                        snapshot[index],
                        snapshot[peer_index],
                        max(1, int(problem.time_budget.fraction * 1000000)),
                        time_horizon,
                        rng,
                    )
                    phase = "aro_detour"
                else:
                    trial_position = random_hiding(
                        snapshot[index],
                        max(1, int(problem.time_budget.fraction * 1000000)),
                        time_horizon,
                        rng,
                    )
                    phase = "aro_hiding"
                trial = ledger.evaluate(from_keys(trial_position), phase)
                if trial.cost < parent.cost:
                    population[index] = Rabbit(trial_position, trial)
                    if trial.cost < best.cost:
                        best = population[index]
            generation_best = min(population, key=lambda rabbit: rabbit.cost)
            if generation_best.cost < best.cost:
                best = generation_best
            convergence.append(best.cost)
            if observer is not None:
                observer.on_generation(
                    [rabbit.candidate for rabbit in population],
                    best.cost,
                    generation,
                    ledger.nfe,
                    time.perf_counter() - started,
                )

        if ledger.nfe != settings.max_function_evaluations and settings.max_iterations is None:
            raise AssertionError("ARO-original did not consume the strict NFE budget")
        if sorted(customer for route in best.candidate.routes for customer in route) != list(
            range(1, problem.dimension + 1)
        ):
            raise AssertionError("ARO-original archive is not a complete customer solution")

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
                "implementation": "aro-original-continuous-random-keys-v1",
                "equations": "ARO Eq. (2)-(7), (10)-(13)",
                "encoding": "continuous random keys + shared minimum-duration Split",
                "adaptation_scope": "representation/decoder only",
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


ARO = AROOriginal
AROConfig = AROOriginalConfig

