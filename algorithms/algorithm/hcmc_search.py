from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter

from core.data.evaluator import HCMCEvaluationResult, HCMCTDVRPEvaluator
from core.data.models import RoadEdge, RoadNode, TDVRPInstance, TrafficOverlay
from core.data.solution import repair_solution
from core.models import Solution

from .greedy import build_greedy_solution
from .operators.discrete import OPERATORS
from .operators.hcmc_pipeline import repair_validate_evaluate


class EvaluationBudgetExhausted(RuntimeError):
    """Raised internally when a legacy algorithm reaches its NFE budget."""


class EvaluationBudget:
    def __init__(self, evaluator, limit: int):
        self.evaluator = evaluator
        self.limit = limit
        self.used = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def score(self, candidate: Solution, instance: TDVRPInstance) -> ScoredSolution:
        if self.used >= self.limit:
            raise EvaluationBudgetExhausted
        self.used += 1
        evaluated = repair_validate_evaluate(candidate, instance, self.evaluator)
        if evaluated is None:
            raise ValueError("Candidate could not be repaired into a feasible HCMC solution")
        solution, evaluation = evaluated
        return ScoredSolution(solution, evaluation)


@dataclass(frozen=True)
class ScoredSolution:
    solution: Solution
    evaluation: HCMCEvaluationResult


@dataclass(frozen=True)
class SearchResult:
    best_solution: Solution
    best_evaluation: HCMCEvaluationResult
    convergence: list[float]
    runtime_seconds: float
    evaluations: int = 0


def make_evaluator(
    instance,
    graph,
    traffic,
    start_time,
    departure_time_mode: str = "fixed",
    departure_time_step: float | None = None,
    departure_time_window: tuple[float, float] | None = None,
):
    """Tạo evaluator với cùng chính sách departure-time cho mọi thuật toán."""
    return HCMCTDVRPEvaluator(
        instance,
        graph,
        traffic,
        start_time=start_time,
        departure_time_mode=departure_time_mode,
        departure_time_step=departure_time_step,
        departure_time_window=departure_time_window,
    )


def score(candidate: Solution, instance, evaluator, budget: EvaluationBudget | None = None) -> ScoredSolution:
    if budget is not None:
        return budget.score(candidate, instance)
    evaluated = repair_validate_evaluate(candidate, instance, evaluator)
    if evaluated is None:
        raise ValueError("Candidate could not be repaired into a feasible HCMC solution")
    solution, evaluation = evaluated
    return ScoredSolution(solution, evaluation)


def initial_population(instance, graph, traffic, start_time, size: int, rng: random.Random, evaluator=None, budget: EvaluationBudget | None = None) -> list[ScoredSolution]:
    evaluator = evaluator or make_evaluator(instance, graph, traffic, start_time)
    population = [score(build_greedy_solution(instance, graph, traffic, start_time), instance, evaluator, budget)]
    customer_ids = list(range(1, instance.num_customers + 1))
    while len(population) < size:
        rng.shuffle(customer_ids)
        population.append(score(repair_solution(Solution([[instance.depot, *customer_ids, instance.depot]]), instance), instance, evaluator, budget))
    return population


def apply_random_operators(solution: Solution, rng: random.Random, count: int = 1) -> Solution:
    candidate = solution.copy()
    for _ in range(max(1, count)):
        candidate = rng.choice(OPERATORS)(candidate, rng)
    return candidate


def better(left: ScoredSolution, right: ScoredSolution) -> ScoredSolution:
    return left if left.evaluation.fitness <= right.evaluation.fitness else right


def _seconds_since_midnight(value: float | datetime) -> float:
    if isinstance(value, datetime):
        return float(value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1_000_000)
    return float(value) % 86400
