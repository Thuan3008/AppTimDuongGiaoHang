"""POA-C: Pufferfish Optimization Algorithm adapted to HCMC TDVRP.

The route representation is permutation-native: attack uses path relinking
toward a strictly better pufferfish, while defense uses a shrinking inversion.
Each POA cycle consumes exactly two objective evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
import math
from pathlib import Path
import sys
import time
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.data.evaluator import HCMCEvaluationResult, HCMCTDVRPEvaluator
from core.data.models import RoadEdge, RoadNode, TDVRPInstance, TrafficOverlay
from core.data.solution import repair_solution
from core.models import Solution

from .hcmc_aro_c import HCMCCandidate, AROContinuousResult, from_keys


@runtime_checkable
class HasCost(Protocol):
    cost: float


def validate_customer_permutation(permutation: Sequence[int] | tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(permutation, (tuple, list)):
        raise TypeError(f"Expected tuple or list of integers, got {type(permutation).__name__}")
    if not permutation:
        raise ValueError("Permutation cannot be empty")
    try:
        values = tuple(int(value) for value in permutation)
    except (TypeError, ValueError) as error:
        raise TypeError(f"All elements in permutation must be integers: {error}") from error
    if 0 in values:
        raise ValueError("Permutation must not contain depot ID 0")
    if len(set(values)) != len(values) or set(values) != set(range(1, len(values) + 1)):
        raise ValueError(f"Permutation must contain exactly the customer set 1..{len(values)}")
    return values


def select_pufferfish_target(parent_idx: int, population: Sequence[HasCost], rng: np.random.Generator) -> HasCost:
    if not population:
        raise ValueError("Population cannot be empty")
    if parent_idx < 0 or parent_idx >= len(population):
        raise IndexError(f"parent_idx {parent_idx} out of range")
    if len(population) == 1:
        return population[parent_idx]
    parent = population[parent_idx]
    better_indices = [index for index, member in enumerate(population) if index != parent_idx and member.cost < parent.cost]
    choices = better_indices or [index for index in range(len(population)) if index != parent_idx]
    return population[int(rng.choice(choices))]


def pufferfish_attack(parent: Sequence[int] | tuple[int, ...], target: Sequence[int] | tuple[int, ...], rng: np.random.Generator) -> tuple[int, ...]:
    parent_values = validate_customer_permutation(parent)
    target_values = validate_customer_permutation(target)
    if len(parent_values) != len(target_values):
        raise ValueError("Permutations must have identical lengths")
    mismatches = [index for index in range(len(parent_values)) if parent_values[index] != target_values[index]]
    if not mismatches:
        return parent_values
    intensity = int(rng.choice([1, 2]))
    step = max(1, min(len(mismatches), int(math.floor(intensity * float(rng.random()) * len(mismatches)))))
    candidate = list(parent_values)
    for _ in range(step):
        differences = [index for index in range(len(candidate)) if candidate[index] != target_values[index]]
        if not differences:
            break
        position = int(rng.choice(differences))
        desired = target_values[position]
        swap_position = candidate.index(desired)
        candidate[position], candidate[swap_position] = candidate[swap_position], candidate[position]
    return tuple(candidate)


def calculate_defense_radius(n: int, t: int) -> int:
    if n < 1 or t < 1:
        raise ValueError("n and t must be positive")
    width_max = max(2, n // 2)
    return max(2, min(n, width_max // t + 1))


def pufferfish_defense(permutation: Sequence[int] | tuple[int, ...], t: int, rng: np.random.Generator) -> tuple[int, ...]:
    values = validate_customer_permutation(permutation)
    if len(values) == 1:
        return values
    radius = calculate_defense_radius(len(values), t)
    length = int(rng.integers(2, radius + 1))
    start = int(rng.integers(0, len(values) - length + 1))
    candidate = list(values)
    candidate[start:start + length] = reversed(candidate[start:start + length])
    return tuple(candidate)


@dataclass
class PufferfishConfig:
    population_size: int = 30
    max_function_evaluations: int = 1000
    max_iterations: int | None = None

    @classmethod
    def from_mapping(cls, config: dict[str, Any] | None = None) -> tuple["PufferfishConfig", int]:
        values = dict(config or {})
        values.pop("minimization", None)
        values.pop("use_evaluation_budget", None)
        run_id_value = values.pop("run_id", 1)
        try:
            run_id = int(run_id_value)
        except (TypeError, ValueError):
            run_id = 1
        for source, target in (("pop_size", "population_size"), ("max_nfe", "max_function_evaluations"), ("target_nfe", "max_function_evaluations")):
            if source in values:
                values.setdefault(target, values.pop(source))
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown POA-C config keys: {sorted(unknown)}")
        cfg = cls(**values)
        if not isinstance(cfg.population_size, (int, np.integer)) or isinstance(cfg.population_size, bool) or cfg.population_size < 1:
            raise ValueError(f"population_size must be a positive integer, got {cfg.population_size!r}")
        if not isinstance(cfg.max_function_evaluations, (int, np.integer)) or isinstance(cfg.max_function_evaluations, bool) or cfg.max_function_evaluations < 1:
            raise ValueError(f"max_function_evaluations must be a positive integer, got {cfg.max_function_evaluations!r}")
        if cfg.max_iterations is not None and (not isinstance(cfg.max_iterations, (int, np.integer)) or isinstance(cfg.max_iterations, bool) or cfg.max_iterations < 0):
            raise ValueError(f"max_iterations must be None or a non-negative integer, got {cfg.max_iterations!r}")
        return cfg, run_id


@dataclass
class POAResult:
    best_solution: Solution
    best_evaluation: HCMCEvaluationResult
    convergence: list[float]
    runtime_seconds: float
    nfe: int
    metadata: dict[str, Any]


class POAEvaluationLedger:
    def __init__(self, instance, evaluator, budget):
        self.instance = instance
        self.evaluator = evaluator
        self.budget = budget
        self.nfe = 0
        self.spending: dict[str, int] = {}

    @property
    def remaining(self):
        return self.budget - self.nfe

    def evaluate(self, permutation, phase):
        if self.remaining < 1:
            raise RuntimeError("POA-C evaluation budget exhausted")
        values = validate_customer_permutation(permutation)
        solution = repair_solution(Solution([[self.instance.depot, *values, self.instance.depot]]), self.instance)
        evaluation = self.evaluator.evaluate(solution)
        self.nfe += 1
        self.spending[phase] = self.spending.get(phase, 0) + 1
        return HCMCCandidate(solution, evaluation, list(values))


class PufferfishOptimization:
    name = "POA-C"
    select_target = staticmethod(select_pufferfish_target)
    predator_attack = staticmethod(pufferfish_attack)
    pufferfish_defense = staticmethod(pufferfish_defense)
    calculate_defense_radius = staticmethod(calculate_defense_radius)
    validate_permutation = staticmethod(validate_customer_permutation)

    def __init__(self, **kwargs: Any):
        self.default_config = dict(kwargs)

    def initialize(self, instance, population_size, rng, ledger):
        effective_population = min(population_size, ledger.remaining)
        positions = rng.random((effective_population, instance.num_customers))
        return [ledger.evaluate(from_keys(position), "initialization") for position in positions]

    def optimize(self, instance: TDVRPInstance, graph: tuple[dict[int, RoadNode], dict[str, RoadEdge]], traffic: TrafficOverlay, config: dict[str, Any] | PufferfishConfig | None = None, seed: int | None = None, start_time: float | datetime = 8 * 3600, **kwargs: Any) -> POAResult:
        merged = dict(self.default_config)
        if isinstance(config, PufferfishConfig):
            merged.update(config.__dict__)
        elif config:
            merged.update(config)
        merged.update(kwargs)
        cfg, run_id = PufferfishConfig.from_mapping(merged)
        rng = np.random.default_rng(seed)
        started = time.perf_counter()
        evaluator = HCMCTDVRPEvaluator(instance, graph, traffic, start_time=start_time)
        ledger = POAEvaluationLedger(instance, evaluator, cfg.max_function_evaluations)
        effective_population = min(cfg.population_size, cfg.max_function_evaluations)
        population = self.initialize(instance, effective_population, rng, ledger)
        if not population:
            raise RuntimeError("POA-C failed to initialize population")
        if cfg.max_function_evaluations <= effective_population or cfg.max_iterations == 0:
            best = min(population, key=lambda member: member.cost)
            return POAResult(best.solution, best.evaluation, [best.cost], time.perf_counter() - started, ledger.nfe, {"config": cfg.__dict__, "run_id": run_id, "effective_population_size": effective_population, "search_cycles": 0, "unused_budget": ledger.remaining})

        best = min(population, key=lambda member: member.cost)
        convergence = [best.cost]
        generation = 0
        while ledger.remaining >= 2 and (cfg.max_iterations is None or generation < cfg.max_iterations):
            generation += 1
            for index in range(len(population)):
                if ledger.remaining < 2:
                    break
                parent = population[index]
                target = select_pufferfish_target(index, population, rng)
                attack = ledger.evaluate(pufferfish_attack(parent.permutation, target.permutation, rng), "pf_attack")
                state = attack if attack.cost < parent.cost else parent
                defense = ledger.evaluate(pufferfish_defense(state.permutation, generation, rng), "pf_defense")
                population[index] = defense if defense.cost < state.cost else state
            best = min([best, *population], key=lambda member: member.cost)
            convergence.append(best.cost)

        init_nfe = ledger.spending.get("initialization", 0)
        attack_nfe = ledger.spending.get("pf_attack", 0)
        defense_nfe = ledger.spending.get("pf_defense", 0)
        assert ledger.nfe <= cfg.max_function_evaluations
        assert attack_nfe == defense_nfe
        assert ledger.nfe == init_nfe + attack_nfe + defense_nfe
        metadata = {"config": cfg.__dict__, "run_id": run_id, "effective_population_size": effective_population, "search_cycles": attack_nfe, "attack_nfe": attack_nfe, "defense_nfe": defense_nfe, "unused_budget": ledger.remaining}
        return POAResult(best.solution, best.evaluation, convergence, time.perf_counter() - started, ledger.nfe, metadata)

    @staticmethod
    def propose(population, fitness, i, t, max_iter, rng):
        better = np.flatnonzero(fitness < fitness[i])
        peer = int(rng.choice(better)) if len(better) else int(np.argmin(fitness))
        yield population[i] + rng.random() * (population[peer] - rng.integers(1, 3) * population[i])
        yield population[i] + (1 - 2 * rng.random(population.shape[1])) / (t + 1)


def run_poa_c(instance, graph, traffic, config: PufferfishConfig | None = None, seed: int | None = None, start_time: float | datetime = 8 * 3600) -> POAResult:
    return PufferfishOptimization().optimize(instance, graph, traffic, config=config, seed=seed, start_time=start_time)


POA = PufferfishOptimization
PFD = PufferfishOptimization
PFDConfig = PufferfishConfig

