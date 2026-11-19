"""Deterministic capacity-aware Greedy using the benchmark's exact evaluator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

from core.tdvrp_common import TDVRPProblem


@dataclass(frozen=True)
class ExactGreedyResult:
    algorithm: str
    fitness: float
    routes: list[list[int]]
    departure_times: list[float]
    return_times: list[float]
    route_durations: list[float]
    route_loads: list[float]
    runtime_seconds: float
    construction_evaluations: int
    feasible: bool

    def to_dict(self) -> dict:
        return asdict(self)


def run_exact_greedy(problem: TDVRPProblem) -> ExactGreedyResult:
    """Build routes by minimum exact marginal route duration.

    Greedy is deterministic and does not have a population or an NFE search
    budget.  Every partial route is scored by the same route_cost function used
    by the stochastic baselines' Split decoder.
    """

    if not isinstance(problem, TDVRPProblem):
        raise TypeError("Expected the shared TDVRPProblem")
    started = perf_counter()
    unserved = set(range(1, problem.dimension + 1))
    routes: list[tuple[int, ...]] = []
    current: list[int] = []
    current_load = 0.0
    current_duration = 0.0
    construction_evaluations = 0

    while unserved:
        choices: list[tuple[float, float, int, float]] = []
        for customer in sorted(unserved):
            demand = float(problem.demands[customer])
            if current_load + demand > problem.capacity:
                continue
            duration = float(problem.route_cost(tuple([*current, customer]))[0])
            construction_evaluations += 1
            marginal = duration - current_duration
            choices.append((marginal, duration, customer, demand))

        if not choices:
            if not current:
                raise ValueError("Greedy could not place any remaining customer")
            routes.append(tuple(current))
            current = []
            current_load = 0.0
            current_duration = 0.0
            continue

        _, current_duration, selected, demand = min(choices)
        current.append(selected)
        current_load += demand
        unserved.remove(selected)

    if current:
        routes.append(tuple(current))
    evaluation = problem.evaluate_candidate(tuple(routes))
    if not evaluation.feasible:
        raise AssertionError("Exact Greedy produced an infeasible solution")
    details = evaluation.route_evaluations
    return ExactGreedyResult(
        algorithm="Greedy",
        fitness=float(evaluation.fitness),
        routes=[[0, *route, 0] for route in routes],
        departure_times=[float(item[1]) for item in details],
        return_times=[float(item[2]) for item in details],
        route_durations=[float(item[0]) for item in details],
        route_loads=[
            float(sum(problem.demands[customer] for customer in route))
            for route in routes
        ],
        runtime_seconds=perf_counter() - started,
        construction_evaluations=construction_evaluations,
        feasible=True,
    )

