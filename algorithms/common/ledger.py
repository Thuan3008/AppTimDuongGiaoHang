"""Standard EvaluationLedger, Candidate representation, and OptimizationResult builder.

Ensures every algorithm strictly obeys the Canonical V4 evaluation pipeline,
monotonic convergence reporting, and NFE fairness.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any
import numpy as np

from core.tdvrp_common import TDVRPProblem, OptimizationResult
from algorithms.rapo_operators import rank_keys


@dataclass(frozen=True)
class Candidate:
    permutation: tuple[int, ...]
    cost: float
    routes: tuple[tuple[int, ...], ...]
    evaluations: tuple[tuple[float, float, float], ...]


class EvaluationLedger:
    """Strict accounting ledger for TDVRP function evaluations.
    
    1 call to problem.evaluate_permutation = 1 NFE.
    nfe_used <= limit is strictly guaranteed.
    """
    def __init__(self, problem: TDVRPProblem, limit: int):
        if not isinstance(problem, TDVRPProblem):
            raise TypeError("Expected TDVRPProblem instance")
        if limit < 1:
            raise ValueError(f"Limit must be positive, got {limit}")
        self.problem = problem
        self.limit = limit
        self.nfe = 0
        self.spending: dict[str, int] = {}
        self.best: Candidate | None = None
        self.best_nfe: int = 0
        self.convergence: list[dict[str, Any]] = []
        self.search_log: list[dict[str, Any]] = []

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.nfe)

    def evaluate(self, permutation: tuple[int, ...] | list[int], phase: str = "main") -> Candidate:
        if self.nfe >= self.limit:
            raise RuntimeError(f"NFE budget exhausted: {self.nfe}/{self.limit}")
        perm_tuple = tuple(int(c) for c in permutation)
        self.nfe += 1
        self.spending[phase] = self.spending.get(phase, 0) + 1

        cost, routes = self.problem.evaluate_permutation(perm_tuple)
        details = tuple(self.problem.route_cost(r) for r in routes)
        candidate = Candidate(perm_tuple, cost, routes, details)

        improved = False
        if self.best is None or candidate.cost < self.best.cost:
            self.best = candidate
            self.best_nfe = self.nfe
            improved = True

        self.convergence.append({
            "nfe": self.nfe,
            "best_objective": self.best.cost,
        })
        self.search_log.append({
            "nfe": self.nfe,
            "phase": phase,
            "candidate_cost": candidate.cost,
            "global_best": self.best.cost,
            "improved": improved,
        })
        return candidate


def build_optimization_result(
    algorithm: str,
    problem: TDVRPProblem,
    ledger: EvaluationLedger,
    seed: int | None,
    runtime_seconds: float,
    population_size: int = 1,
    max_iterations: int = 0,
    run_id: int = 1,
    extra_metadata: dict[str, Any] | None = None,
) -> OptimizationResult:
    best = ledger.best
    if best is None:
        raise RuntimeError("No candidate evaluated in ledger")

    # Invariant verification
    flat = sorted(c for r in best.routes for c in r)
    if flat != list(range(1, problem.dimension + 1)):
        raise AssertionError(f"Archive violates customer coverage: found {len(flat)}, expected {problem.dimension}")
    if any(not r or sum(problem.demands[c] for c in r) > problem.capacity for r in best.routes):
        raise AssertionError("Archive violates capacity constraint")

    meta: dict[str, Any] = {
        "objective": "Duration",
        "units": "seconds",
        "feasible": True,
        "encoding": "permutation + shared minimum-duration Split",
        "departure_domain": [0, problem.horizon],
        "travel_time_domain": "route departure restricted to nonempty T(R); no extrapolation",
        "model_version": 4,
        "evaluator_version": "TDVRP-CANONICAL-V1",
        "split_decoder_version": "split-canonical-v1",
        "vehicle_limit_K": problem.vehicle_limit,
        "penalty_weights": {
            "miss": problem.alpha_miss,
            "dup": problem.alpha_dup,
            "cap": problem.alpha_cap,
            "fleet": problem.alpha_fleet,
            "M0": problem.penalty_fixed,
            "M1": problem.penalty_multiplier,
        },
        "evaluations": ledger.nfe,
        "nfe_by_phase": dict(ledger.spending),
        "best_nfe": ledger.best_nfe,
        "convergence": ledger.convergence,
        "search_log": ledger.search_log,
    }
    if extra_metadata:
        meta.update(extra_metadata)

    res = OptimizationResult(
        algorithm=algorithm,
        benchmark=problem.name,
        seed=seed,
        best_fitness=best.cost,
        best_solution=rank_keys(best.permutation).tolist(),
        routes=[[0, *r, 0] for r in best.routes],
        departure_times=[e[1] for e in best.evaluations],
        return_times=[e[2] for e in best.evaluations],
        route_durations=[e[0] for e in best.evaluations],
        route_loads=[sum(problem.demands[c] for c in r) for r in best.routes],
        convergence_curve=[pt["best_objective"] for pt in ledger.convergence],
        runtime_seconds=runtime_seconds,
        population_size=population_size,
        max_iterations=max_iterations,
        nfe=ledger.nfe,
        dimension=problem.dimension,
        metadata=meta,
    )
    res.best_permutation = list(best.permutation)
    return res
