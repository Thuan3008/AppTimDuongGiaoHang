from __future__ import annotations

from abc import ABC, abstractmethod

from core.models import OptimizationResult, TDVRPProblem


class BaseOptimizer(ABC):
    @abstractmethod
    def solve(self, problem: TDVRPProblem, config) -> OptimizationResult:
        raise NotImplementedError
