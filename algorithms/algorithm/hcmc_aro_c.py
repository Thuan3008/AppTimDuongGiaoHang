"""ARO-C: continuous random-key Artificial Rabbits Optimization for HCMC TDVRP."""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.data.evaluator import HCMCEvaluationResult, HCMCTDVRPEvaluator
from core.data.models import RoadEdge, RoadNode, TDVRPInstance, TrafficOverlay
from core.data.solution import repair_solution
from core.models import Solution


@dataclass
class AROContinuousConfig:
    population_size: int = 30
    max_function_evaluations: int = 1000
    max_iterations: int | None = None

    @classmethod
    def from_mapping(cls, config: dict[str, Any] | None = None) -> tuple["AROContinuousConfig", int]:
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
            raise ValueError(f"Unknown ARO-C config keys: {sorted(unknown)}")
        cfg = cls(**values)
        if not isinstance(cfg.population_size, (int, np.integer)) or isinstance(cfg.population_size, bool) or cfg.population_size < 1:
            raise ValueError(f"population_size must be a positive integer, got {cfg.population_size!r}")
        if not isinstance(cfg.max_function_evaluations, (int, np.integer)) or isinstance(cfg.max_function_evaluations, bool) or cfg.max_function_evaluations < 1:
            raise ValueError(f"max_function_evaluations must be a positive integer, got {cfg.max_function_evaluations!r}")
        if cfg.max_iterations is not None and (not isinstance(cfg.max_iterations, (int, np.integer)) or isinstance(cfg.max_iterations, bool) or cfg.max_iterations < 0):
            raise ValueError(f"max_iterations must be None or a non-negative integer, got {cfg.max_iterations!r}")
        return cfg, run_id


@dataclass
class HCMCCandidate:
    solution: Solution
    evaluation: HCMCEvaluationResult
    permutation: list[int]

    @property
    def cost(self) -> float:
        return float(self.evaluation.fitness)

    @property
    def routes(self):
        return self.solution.routes


@dataclass
class RabbitState:
    position: np.ndarray
    candidate: HCMCCandidate

    @property
    def cost(self) -> float:
        return self.candidate.cost

    @property
    def permutation(self):
        return self.candidate.permutation

    @property
    def routes(self):
        return self.candidate.routes


@dataclass
class EvaluationLedger:
    instance: TDVRPInstance
    evaluator: HCMCTDVRPEvaluator
    budget: int
    nfe: int = 0
    spending: dict[str, int] | None = None

    def __post_init__(self):
        self.spending = {}

    @property
    def remaining(self) -> int:
        return self.budget - self.nfe

    def evaluate(self, position: np.ndarray, phase: str) -> HCMCCandidate:
        if self.remaining < 1:
            raise RuntimeError("ARO-C evaluation budget exhausted")
        position = repair_bounds(position)
        permutation = from_keys(position)
        solution = _decode_permutation(permutation, self.instance)
        evaluation = self.evaluator.evaluate(solution)
        self.nfe += 1
        self.spending[phase] = self.spending.get(phase, 0) + 1
        return HCMCCandidate(solution, evaluation, permutation)


@dataclass
class AROContinuousResult:
    best_solution: Solution
    best_evaluation: HCMCEvaluationResult
    convergence: list[float]
    runtime_seconds: float
    nfe: int
    metadata: dict[str, Any]


def from_keys(position: np.ndarray) -> list[int]:
    return [int(index) + 1 for index in np.argsort(position, kind="stable")]


def validate_position(position: np.ndarray, dimension: int) -> np.ndarray:
    x = np.asarray(position, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"ARO position must be one-dimensional, got shape {x.shape}")
    if len(x) != dimension:
        raise ValueError(f"Expected position dimension {dimension}, got {len(x)}")
    if not np.all(np.isfinite(x)):
        raise ValueError("ARO position contains NaN or infinite values")
    return x


def repair_bounds(position: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(position, dtype=float), 0.0, 1.0)


def calculate_energy(t: int, max_iter: int, rng: np.random.Generator) -> float:
    if t < 1:
        raise ValueError(f"Iteration t must be >= 1, got {t}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    r = max(float(rng.random()), np.finfo(float).tiny)
    return float(4.0 * max(0.0, 1.0 - t / max_iter) * math.log(1.0 / r))


def running_operator(dimension: int, t: int, max_iter: int, rng: np.random.Generator) -> np.ndarray:
    if dimension < 1:
        raise ValueError(f"dimension must be >= 1, got {dimension}")
    progress = (t - 1) / max_iter
    length = (math.e - math.exp(progress**2)) * math.sin(2.0 * math.pi * float(rng.random()))
    number_selected = max(1, min(dimension, int(math.ceil(float(rng.random()) * dimension))))
    selected = rng.choice(dimension, size=number_selected, replace=False)
    mask = np.zeros(dimension, dtype=float)
    mask[selected] = 1.0
    return length * mask


def select_detour_peer(rabbit_index: int, population_size: int, rng: np.random.Generator) -> int:
    if population_size < 1:
        raise ValueError("Population size must be >= 1")
    if rabbit_index < 0 or rabbit_index >= population_size:
        raise IndexError(f"rabbit_index {rabbit_index} out of range")
    if population_size == 1:
        return rabbit_index
    peer = int(rng.integers(0, population_size - 1))
    return peer + (peer >= rabbit_index)


def detour_foraging(current: np.ndarray, peer: np.ndarray, t: int, max_iter: int, rng: np.random.Generator) -> np.ndarray:
    x_i = np.asarray(current, dtype=float)
    x_j = np.asarray(peer, dtype=float)
    if x_i.shape != x_j.shape:
        raise ValueError("Current and peer positions must have identical shape")
    running = running_operator(len(x_i), t, max_iter, rng)
    noise_switch = 1.0 if 0.5 * (0.05 + float(rng.random())) >= 0.5 else 0.0
    noise = rng.normal(loc=0.0, scale=1.0, size=len(x_i))
    return repair_bounds(x_j + running * (x_i - x_j) + noise_switch * noise)


def random_hiding(current: np.ndarray, t: int, max_iter: int, rng: np.random.Generator) -> np.ndarray:
    x_i = np.asarray(current, dtype=float)
    dimension = len(x_i)
    if dimension < 1:
        raise ValueError("Random hiding requires dimension >= 1")
    r4 = float(rng.random())
    direction = np.zeros(dimension, dtype=float)
    direction[int(rng.integers(0, dimension))] = 1.0
    hiding = ((max_iter - t + 1) / max_iter) * r4
    burrow = x_i + hiding * direction * x_i
    running = running_operator(dimension, t, max_iter, rng)
    return repair_bounds(x_i + running * (r4 * burrow - x_i))


class AROContinuous:
    name = "ARO-C"
    hybrid_type = "continuous-random-keys"
    calculate_energy = staticmethod(calculate_energy)
    running_operator = staticmethod(running_operator)
    detour_foraging = staticmethod(detour_foraging)
    random_hiding = staticmethod(random_hiding)
    select_peer = staticmethod(select_detour_peer)
    repair_bounds = staticmethod(repair_bounds)

    def __init__(self, **kwargs: Any):
        self.default_config = dict(kwargs)

    @staticmethod
    def evaluate_position(position: np.ndarray, ledger: EvaluationLedger, phase: str) -> HCMCCandidate:
        return ledger.evaluate(position, phase)

    def initialize(self, instance, population_size, rng, ledger) -> list[RabbitState]:
        effective_population = min(population_size, ledger.remaining)
        positions = rng.random((effective_population, instance.num_customers))
        return [RabbitState(position.copy(), self.evaluate_position(position, ledger, "initialization")) for position in positions]

    def optimize(self, instance: TDVRPInstance, graph: tuple[dict[int, RoadNode], dict[str, RoadEdge]], traffic: TrafficOverlay, config: dict[str, Any] | AROContinuousConfig | None = None, seed: int | None = None, start_time: float | datetime = 8 * 3600, **kwargs: Any) -> AROContinuousResult:
        merged = dict(self.default_config)
        if isinstance(config, AROContinuousConfig):
            merged.update(config.__dict__)
        elif config:
            merged.update(config)
        merged.update(kwargs)
        cfg, run_id = AROContinuousConfig.from_mapping(merged)
        rng = np.random.default_rng(seed)
        started = time.perf_counter()
        evaluator = HCMCTDVRPEvaluator(instance, graph, traffic, start_time=start_time)
        ledger = EvaluationLedger(instance, evaluator, cfg.max_function_evaluations)
        effective_population = min(cfg.population_size, cfg.max_function_evaluations)
        population = self.initialize(instance, effective_population, rng, ledger)
        if not population:
            raise RuntimeError("ARO-C failed to initialize population")
        if cfg.max_iterations is not None:
            time_horizon = max(1, cfg.max_iterations)
        else:
            time_horizon = max(1, math.ceil(max(0, cfg.max_function_evaluations - effective_population) / effective_population))
        best = min(population, key=lambda rabbit: rabbit.cost)
        convergence = [best.cost]
        generation = 0
        while ledger.remaining >= 1 and (cfg.max_iterations is None or generation < cfg.max_iterations):
            generation += 1
            snapshot = [rabbit.position.copy() for rabbit in population]
            for index, parent in enumerate(population):
                if ledger.remaining < 1:
                    break
                energy = calculate_energy(generation, time_horizon, rng)
                if energy > 1.0:
                    peer_index = select_detour_peer(index, len(population), rng)
                    trial_position = detour_foraging(snapshot[index], snapshot[peer_index], generation, time_horizon, rng)
                    phase = "aro_detour"
                else:
                    trial_position = random_hiding(snapshot[index], generation, time_horizon, rng)
                    phase = "aro_hiding"
                trial = self.evaluate_position(trial_position, ledger, phase)
                if trial.cost < parent.cost:
                    population[index] = RabbitState(trial_position.copy(), trial)
            best = min([best, *population], key=lambda rabbit: rabbit.cost)
            convergence.append(best.cost)
        initialization_nfe = ledger.spending.get("initialization", 0)
        detour_nfe = ledger.spending.get("aro_detour", 0)
        hiding_nfe = ledger.spending.get("aro_hiding", 0)
        assert ledger.nfe <= cfg.max_function_evaluations
        assert ledger.nfe == initialization_nfe + detour_nfe + hiding_nfe
        search_nfe = detour_nfe + hiding_nfe
        metadata = {"config": cfg.__dict__, "run_id": run_id, "representation": "continuous_random_keys", "effective_population_size": effective_population, "time_horizon": time_horizon, "search_updates": search_nfe, "detour_updates": detour_nfe, "hiding_updates": hiding_nfe, "detour_ratio": detour_nfe / max(1, search_nfe), "hiding_ratio": hiding_nfe / max(1, search_nfe), "unused_budget": ledger.remaining}
        return AROContinuousResult(best.candidate.solution, best.candidate.evaluation, convergence, time.perf_counter() - started, ledger.nfe, metadata)

    @staticmethod
    def propose(population, fitness, i, t, max_iter, rng):
        population = np.asarray(population, dtype=float)
        energy = calculate_energy(t + 1, max_iter, rng)
        if energy > 1.0:
            peer = select_detour_peer(i, len(population), rng)
            yield detour_foraging(population[i], population[peer], t + 1, max_iter, rng)
        else:
            yield random_hiding(population[i], t + 1, max_iter, rng)


def _decode_permutation(permutation: list[int], instance: TDVRPInstance) -> Solution:
    return repair_solution(Solution([[instance.depot, *permutation, instance.depot]]), instance)


def run_aro_c(instance, graph, traffic, config: AROContinuousConfig | None = None, seed: int | None = None, start_time: float | datetime = 8 * 3600) -> AROContinuousResult:
    return AROContinuous().optimize(instance, graph, traffic, config=config, seed=seed, start_time=start_time)


ARO_C = AROContinuous
ARO = AROContinuous
AROConfig = AROContinuousConfig
