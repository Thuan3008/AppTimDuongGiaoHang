from __future__ import annotations

from algorithms.population import capacity_routes
from core.models import Solution, TDVRPProblem


def repair_solution(solution: Solution, problem: TDVRPProblem) -> Solution:
    flat = [node for route in solution.routes for node in route if node != 0 and node in problem.customer_ids]
    valid = []
    seen = set()
    for node in flat:
        if node not in seen:
            valid.append(node); seen.add(node)
    missing = [node for node in problem.customer_ids if node not in seen]
    return Solution(capacity_routes(valid + missing, problem))
