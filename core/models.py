from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Customer:
    id: int
    x: float
    y: float
    demand: float
    service_time: float


@dataclass(frozen=True)
class Depot:
    id: int = 0
    x: float = 50.0
    y: float = 50.0


@dataclass(frozen=True)
class Vehicle:
    id: int
    capacity: float


@dataclass
class TDVRPProblem:
    depot: Depot
    customers: list[Customer]
    vehicles: list[Vehicle]
    base_speed: float = 1.0
    traffic_multipliers: dict[str, float] = field(default_factory=lambda: {"normal": 1.0, "morning_peak": 1.5, "evening_peak": 1.7})
    distance_matrix: Any = None
    instance_name: str = "demo"

    @property
    def customer_ids(self) -> list[int]:
        return [c.id for c in self.customers]

    @property
    def nodes(self) -> list[Customer | Depot]:
        return [self.depot, *self.customers]


@dataclass
class Solution:
    routes: list[list[int]]
    fitness: float = float("inf")
    objective: float = float("inf")
    total_distance: float = 0.0
    total_travel_time: float = 0.0
    is_feasible: bool = False

    def copy(self) -> "Solution":
        return Solution([route[:] for route in self.routes], self.fitness, self.objective, self.total_distance, self.total_travel_time, self.is_feasible)


@dataclass
class ValidationResult:
    feasible: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class RouteTimeline:
    vehicle_id: int
    rows: list[dict[str, Any]]
    distance: float
    travel_time: float
    service_time: float


@dataclass
class EvaluationResult:
    total_distance: float
    total_travel_time: float
    total_service_time: float
    total_route_time: float
    routes: list[RouteTimeline]


@dataclass
class SwarmState:
    index: int
    role: str
    p_aro: float
    p_poa: float
    individuals: list[Solution] = field(default_factory=list)
    local_best: Solution | None = None
    probability_history: list[dict[str, float]] = field(default_factory=list)


@dataclass
class MigrationEvent:
    iteration: int
    source_swarm: int
    target_swarm: int
    fitness: float


@dataclass
class OptimizationResult:
    global_best: Solution
    evaluation: EvaluationResult
    convergence: list[float]
    local_best_history: list[list[float]]
    swarms: list[SwarmState]
    migration_events: list[MigrationEvent]
    runtime_seconds: float
