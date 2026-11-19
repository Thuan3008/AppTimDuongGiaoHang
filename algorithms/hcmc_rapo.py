"""RAPO for the HCMC time-dependent VRP.

This is a standalone, HCMC-native RAPO implementation.  It coordinates
adaptive ARO-style exploration (route perturbations) with POA-style elite
refinement, while every candidate follows repair -> validate -> evaluate.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

from core.data.evaluator import HCMCEvaluationResult, HCMCTDVRPEvaluator
from core.data.models import RoadEdge, RoadNode, TDVRPInstance, TrafficOverlay
from core.data.solution import repair_solution
from core.models import Solution

from .greedy import build_greedy_solution
from .hcmc_search import make_evaluator
from .operators.discrete import cross_route_swap, insertion, relocate, reverse_two_opt, swap
from .operators.hcmc_pipeline import repair_validate_evaluate


OPERATOR_FUNCTIONS: dict[str, Callable[[Solution, random.Random], Solution]] = {
    "swap": swap,
    "insertion": insertion,
    "relocate": relocate,
    "reverse_two_opt": reverse_two_opt,
    "cross_route_swap": cross_route_swap,
}
FAMILIES = {
    "ARO": ("swap", "insertion", "relocate"),
    "POA": ("reverse_two_opt", "cross_route_swap", "insertion"),
}


@dataclass
class RAPOConfig:
    population_size: int = 30
    max_function_evaluations: int = 1000
    max_iterations: int | None = None
    variant: str = "RAPO-V3"
    elite_ratio: float = 0.2
    initial_aro_probability: float = 0.5
    reward_alpha: float = 0.25
    stagnation_limit: int = 12
    restart_ratio: float = 0.25
    elite_budget_fraction: float = 0.2
    diversity_threshold: float = 0.10
    adaptive_family_selection: bool = True
    enable_elite_refinement: bool = True
    enable_restarts: bool = True
    seed: int | None = None

    def validate(self) -> None:
        if self.population_size < 2 or self.max_function_evaluations < self.population_size:
            raise ValueError("population_size >= 2 and evaluation budget >= population_size are required")
        if not 0 < self.elite_ratio <= 1 or not 0 <= self.initial_aro_probability <= 1:
            raise ValueError("elite_ratio and initial_aro_probability are out of range")
        if not 0 <= self.restart_ratio < 1 or not 0 <= self.elite_budget_fraction < 1:
            raise ValueError("restart_ratio and elite_budget_fraction are out of range")


@dataclass(frozen=True)
class RAPOResult:
    best_solution: Solution
    best_evaluation: HCMCEvaluationResult
    convergence: list[float]
    runtime_seconds: float
    evaluations: int
    metadata: dict[str, object] = field(default_factory=dict)


class _Ledger:
    def __init__(self, evaluator: HCMCTDVRPEvaluator, instance: TDVRPInstance, budget: int):
        self.evaluator, self.instance, self.budget = evaluator, instance, budget
        self.evaluations = 0

    def evaluate(self, candidate: Solution) -> tuple[Solution, HCMCEvaluationResult] | None:
        if self.evaluations >= self.budget:
            return None
        self.evaluations += 1
        return repair_validate_evaluate(candidate, self.instance, self.evaluator)


class _AdaptiveController:
    def __init__(self, config: RAPOConfig):
        self.config = config
        self.family_uses = {name: 0 for name in FAMILIES}
        self.family_rewards = {name: 0.0 for name in FAMILIES}
        self.operator_uses = {name: 0 for names in FAMILIES.values() for name in names}

    def family(self, rng: random.Random, force_explore: bool = False) -> str:
        if force_explore or any(value == 0 for value in self.family_uses.values()):
            return min(self.family_uses, key=self.family_uses.get)
        p_aro = self.config.initial_aro_probability
        total = sum(self.family_rewards.values())
        if self.config.adaptive_family_selection and total > 0:
            p_aro = (1 - self.config.reward_alpha) * p_aro + self.config.reward_alpha * self.family_rewards["ARO"] / total
        return "ARO" if rng.random() < p_aro else "POA"

    def operator(self, family: str, rng: random.Random) -> str:
        names = FAMILIES[family]
        unexplored = [name for name in names if self.operator_uses[name] == 0]
        name = rng.choice(unexplored or list(names))
        self.operator_uses[name] += 1
        self.family_uses[family] += 1
        return name

    def update(self, family: str, improvement: float) -> None:
        self.family_rewards[family] = max(0.0, self.family_rewards[family] * (1 - self.config.reward_alpha) + max(0.0, improvement))


def _diversity(population: list[Solution]) -> float:
    if len(population) < 2:
        return 1.0
    permutations = [[customer for route in s.routes for customer in route[1:-1]] for s in population]
    total = count = 0
    for i in range(len(permutations)):
        for j in range(i + 1, len(permutations)):
            total += sum(a != b for a, b in zip(permutations[i], permutations[j])) / max(1, len(permutations[i]))
            count += 1
    return total / max(1, count)


def run_rapo(
    instance: TDVRPInstance,
    graph: tuple[dict[int, RoadNode], dict[tuple[int, int], RoadEdge]],
    traffic: TrafficOverlay,
    config: RAPOConfig | None = None,
    seed: int | None = None,
    start_time: float = 0.0,
    departure_time_mode: str = "fixed",
    departure_time_step: float | None = None,
    departure_time_window: tuple[float, float] | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> RAPOResult:
    config = config or RAPOConfig()
    config.validate()
    rng = random.Random(config.seed if seed is None else seed)
    evaluator = make_evaluator(
        instance, graph, traffic, start_time,
        departure_time_mode, departure_time_step, departure_time_window,
    )
    ledger = _Ledger(evaluator, instance, config.max_function_evaluations)
    started = perf_counter()

    initial = build_greedy_solution(instance, graph, traffic, start_time)
    scored = ledger.evaluate(initial)
    if scored is None:
        raise RuntimeError("RAPO budget is too small to initialize")
    population: list[tuple[Solution, HCMCEvaluationResult]] = [scored]
    ids = list(range(1, instance.num_customers + 1))
    while len(population) < config.population_size and ledger.evaluations < ledger.budget:
        rng.shuffle(ids)
        candidate = Solution([[instance.depot, *ids, instance.depot]])
        scored = ledger.evaluate(candidate)
        if scored is not None:
            population.append(scored)
    if len(population) < 2:
        raise RuntimeError("RAPO could not create an initial population")

    controller = _AdaptiveController(config)
    best = min(population, key=lambda item: item[1].fitness)
    convergence = [best[1].fitness]
    stagnant = 0
    max_iterations = config.max_iterations or max(1, config.max_function_evaluations // max(1, config.population_size))

    for iteration in range(max_iterations):
        if ledger.evaluations >= ledger.budget:
            break
        generation_best = best
        for index, (parent, parent_eval) in enumerate(population):
            if ledger.evaluations >= ledger.budget:
                break
            family = controller.family(rng, force_explore=_diversity([item[0] for item in population]) < config.diversity_threshold)
            operator_name = controller.operator(family, rng)
            candidate = OPERATOR_FUNCTIONS[operator_name](parent, rng)
            evaluated = ledger.evaluate(candidate)
            if evaluated is None:
                break
            child, child_eval = evaluated
            if child_eval.fitness < parent_eval.fitness:
                improvement = parent_eval.fitness - child_eval.fitness
                population[index] = (child, child_eval)
                controller.update(family, improvement / max(1.0, abs(parent_eval.fitness)))
                if child_eval.fitness < generation_best[1].fitness:
                    generation_best = (child, child_eval)

        # POA-style exploitation of the elite set.
        elite_count = max(1, int(len(population) * config.elite_ratio))
        if config.enable_elite_refinement and config.elite_budget_fraction > 0:
            elite_budget = max(1, int(config.max_function_evaluations * config.elite_budget_fraction))
            elite = sorted(population, key=lambda item: item[1].fitness)[:elite_count]
            for parent, parent_eval in elite:
                if ledger.evaluations >= ledger.budget or ledger.evaluations >= config.max_function_evaluations - max(0, elite_budget - 1):
                    break
                operator_name = rng.choice(FAMILIES["POA"])
                evaluated = ledger.evaluate(OPERATOR_FUNCTIONS[operator_name](parent, rng))
                if evaluated and evaluated[1].fitness < parent_eval.fitness:
                    population[population.index((parent, parent_eval))] = evaluated
                    generation_best = min(generation_best, evaluated, key=lambda item: item[1].fitness)

        improved = generation_best[1].fitness < best[1].fitness
        if improved:
            best, stagnant = generation_best, 0
        else:
            stagnant += 1

        # Partial restart preserves elites and refreshes the worst solutions.
        if config.enable_restarts and stagnant >= config.stagnation_limit and ledger.evaluations < ledger.budget:
            elite = sorted(population, key=lambda item: item[1].fitness)[:elite_count]
            elite_ids = {id(item) for item in elite}
            restart_count = max(1, int(len(population) * config.restart_ratio))
            for index in sorted(range(len(population)), key=lambda i: population[i][1].fitness, reverse=True):
                if restart_count <= 0 or ledger.evaluations >= ledger.budget:
                    break
                if id(population[index]) in elite_ids:
                    continue
                rng.shuffle(ids)
                evaluated = ledger.evaluate(Solution([[instance.depot, *ids, instance.depot]]))
                if evaluated:
                    population[index] = evaluated
                restart_count -= 1
            stagnant = 0
        convergence.append(best[1].fitness)
        if progress_callback is not None:
            progress_callback(ledger.evaluations, ledger.budget, best[1].fitness)

    return RAPOResult(
        best_solution=best[0], best_evaluation=best[1], convergence=convergence,
        runtime_seconds=perf_counter() - started, evaluations=ledger.evaluations,
        metadata={
            "variant": config.variant,
            "objective": "total_route_duration",
            "population_size": len(population),
            "diversity": _diversity([item[0] for item in population]),
            "adaptive_family_selection": config.adaptive_family_selection,
            "elite_refinement": config.enable_elite_refinement,
            "restarts": config.enable_restarts,
        },
    )


class RAPO:
    name = "RAPO"

    def optimize(self, instance, graph, traffic, config=None, seed=None, start_time=0.0, **kwargs):
        return run_rapo(instance, graph, traffic, config=config, seed=seed, start_time=start_time, **kwargs)


RAPOConfigHCMC = RAPOConfig
