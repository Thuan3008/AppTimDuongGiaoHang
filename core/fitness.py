from __future__ import annotations

from .evaluator import RouteEvaluator
from .models import Solution, TDVRPProblem
from .validation import validate_solution


class FitnessEvaluator:
    """Minimized objective: total time in hours, with a guarded infeasibility penalty."""
    def __init__(self, problem: TDVRPProblem, route_evaluator: RouteEvaluator):
        self.problem = problem
        self.route_evaluator = route_evaluator

    def evaluate(self, solution: Solution) -> float:
        validation = validate_solution(solution, self.problem)
        result = self.route_evaluator.evaluate(solution)
        solution.objective = result.total_travel_time + result.total_service_time / 60
        solution.total_distance = result.total_distance
        solution.total_travel_time = result.total_travel_time
        solution.is_feasible = validation.feasible
        solution.fitness = solution.objective if validation.feasible else solution.objective + 1e6 + len(validation.errors) * 1e4
        return solution.fitness
