"""Common infrastructure for TDVRP algorithms."""
from .ledger import Candidate, EvaluationLedger, build_optimization_result

__all__ = ["Candidate", "EvaluationLedger", "build_optimization_result"]

