from __future__ import annotations

from core.data.evaluator import HCMCEvaluationResult, HCMCTDVRPEvaluator
from core.data.models import TDVRPInstance
from core.data.solution import repair_solution, validate_solution
from core.models import Solution


def repair_validate_evaluate(
    candidate: Solution,
    instance: TDVRPInstance,
    evaluator: HCMCTDVRPEvaluator,
) -> tuple[Solution, HCMCEvaluationResult] | None:
    """Apply the common candidate -> repair -> validate -> evaluate pipeline."""
    repaired = repair_solution(candidate, instance)
    validation = validate_solution(repaired, instance)
    if not validation.feasible:
        return None
    result = evaluator.evaluate(repaired)
    if not result.feasible:
        return None
    return repaired, result

