from __future__ import annotations

from collections import Counter

from .models import Solution, TDVRPProblem, ValidationResult


def validate_solution(solution: Solution, problem: TDVRPProblem) -> ValidationResult:
    errors: list[str] = []
    expected = set(problem.customer_ids)
    seen = [node for route in solution.routes for node in route if node != problem.depot.id]
    counts = Counter(seen)
    if len(solution.routes) != len(problem.vehicles):
        errors.append(f"Expected {len(problem.vehicles)} routes, got {len(solution.routes)}")
    if set(seen) != expected:
        errors.append("Customer set is missing or contains unknown customers")
    if any(count > 1 for count in counts.values()):
        errors.append("A customer is served more than once")
    for idx, route in enumerate(solution.routes):
        if not route or route[0] != 0 or route[-1] != 0:
            errors.append(f"Route {idx + 1} must start and end at depot")
        if sum(next((c.demand for c in problem.customers if c.id == n), 0) for n in route) > problem.vehicles[idx].capacity + 1e-9:
            errors.append(f"Route {idx + 1} exceeds vehicle capacity")
    return ValidationResult(not errors, errors)
