"""Original discrete ARO adaptation for HCMC TDVRP."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter

from core.data.models import TDVRPInstance, RoadEdge, RoadNode, TrafficOverlay

from .hcmc_search import EvaluationBudget, EvaluationBudgetExhausted, SearchResult, ScoredSolution, apply_random_operators, better, initial_population, make_evaluator, score


@dataclass(frozen=True)
class AROConfig:
    population_size: int = 12
    iterations: int = 25
    seed: int = 42
    max_function_evaluations: int | None = None


def run_aro(
    instance: TDVRPInstance,
    graph: tuple[dict[int, RoadNode], dict[str, RoadEdge]],
    traffic: TrafficOverlay,
    start_time: float | datetime = 8 * 3600,
    config: AROConfig = AROConfig(),
    departure_time_mode: str = "fixed",
    departure_time_step: float | None = None,
) -> SearchResult:
    started = perf_counter()
    rng = random.Random(config.seed)
    evaluator = make_evaluator(instance, graph, traffic, start_time, departure_time_mode, departure_time_step)
    budget = EvaluationBudget(evaluator, config.max_function_evaluations) if config.max_function_evaluations is not None else None
    population = initial_population(instance, graph, traffic, start_time, config.population_size, rng, evaluator, budget)
    global_best = min(population, key=lambda item: item.evaluation.fitness)
    convergence = [global_best.evaluation.fitness]

    for iteration in range(1, config.iterations + 1):
        next_population: list[ScoredSolution] = []
        for index, current in enumerate(population):
            if budget is not None and budget.remaining <= 0:
                population = next_population or population
                return SearchResult(global_best.solution, global_best.evaluation, convergence, perf_counter() - started, budget.used)
            energy = 4.0 * (1.0 - iteration / config.iterations) * math.log(1.0 / max(rng.random(), 1e-12))
            if energy > 1.0 and len(population) > 1:
                better_members = [item for j, item in enumerate(population) if j != index and item.evaluation.fitness < current.evaluation.fitness]
                guide = rng.choice(better_members) if better_members else rng.choice([item for item in population if item is not current])
                candidate = _detour_toward_guide(current.solution, guide.solution, rng)
                candidate = apply_random_operators(candidate, rng, count=2)
            else:
                candidate = apply_random_operators(current.solution, rng, count=1)
            trial = score(candidate, instance, evaluator, budget)
            next_population.append(better(current, trial))
        population = next_population
        global_best = better(global_best, min(population, key=lambda item: item.evaluation.fitness))
        convergence.append(global_best.evaluation.fitness)
    return SearchResult(global_best.solution, global_best.evaluation, convergence, perf_counter() - started, budget.used if budget else 0)


def _detour_toward_guide(current, guide, rng):
    """Move one customer toward a guide solution for exploration."""
    candidate = current.copy()
    guide_customers = [customer_id for route in guide.routes for customer_id in route if customer_id != 0]
    if guide_customers:
        customer_id = rng.choice(guide_customers)
        for route in candidate.routes:
            if customer_id in route:
                route.remove(customer_id)
                break
        route_index = rng.randrange(len(candidate.routes))
        position = rng.randint(1, len(candidate.routes[route_index]) - 1)
        candidate.routes[route_index].insert(position, customer_id)
    return candidate
