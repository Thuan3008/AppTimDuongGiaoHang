"""Canonical solution representation and repair rules for HCMC TDVRP."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.models import Solution

from .models import TDVRPInstance


@dataclass(frozen=True)
class SolutionValidation:
    feasible: bool
    errors: list[str] = field(default_factory=list)


def empty_solution(instance: TDVRPInstance) -> Solution:
    """Create one empty depot-to-depot route for every vehicle."""
    return Solution(routes=[[instance.depot, instance.depot] for _ in range(instance.num_vehicles)])


def validate_solution(solution: Solution, instance: TDVRPInstance) -> SolutionValidation:
    """Validate the route contract without changing the candidate."""
    errors: list[str] = []
    expected = set(range(1, instance.num_customers + 1))
    if len(solution.routes) != instance.num_vehicles:
        errors.append(f"Expected {instance.num_vehicles} routes, got {len(solution.routes)}")

    seen: list[int] = []
    for route_index, route in enumerate(solution.routes):
        if len(route) < 2 or route[0] != instance.depot or route[-1] != instance.depot:
            errors.append(f"Route {route_index + 1} must start and end at depot {instance.depot}")
        for customer_id in route:
            if customer_id == instance.depot:
                continue
            if customer_id not in expected:
                errors.append(f"Route {route_index + 1} contains unknown customer {customer_id}")
            else:
                seen.append(customer_id)
        demand = sum(instance.demands[customer_id] for customer_id in route if customer_id in expected)
        if route_index < instance.num_vehicles and demand > instance.vehicle_capacity + 1e-9:
            errors.append(f"Route {route_index + 1} exceeds capacity: {demand:g} > {instance.vehicle_capacity:g}")

    duplicates = sorted(customer_id for customer_id in set(seen) if seen.count(customer_id) > 1)
    missing = sorted(expected - set(seen))
    if duplicates:
        errors.append(f"Duplicate customers: {duplicates}")
    if missing:
        errors.append(f"Missing customers: {missing}")
    return SolutionValidation(not errors, errors)


def repair_solution(solution: Solution, instance: TDVRPInstance) -> Solution:
    """Repair a candidate deterministically, then return exact vehicle routes.

    The first occurrence of every valid customer is kept. Unknown and duplicate
    IDs are discarded; missing customers are appended in ascending ID order.
    Customers are then packed in first-fit order while preserving their order.
    """
    expected = set(range(1, instance.num_customers + 1))
    ordered: list[int] = []
    seen: set[int] = set()
    for route in solution.routes:
        for customer_id in route:
            if customer_id in expected and customer_id not in seen:
                ordered.append(customer_id)
                seen.add(customer_id)
    ordered.extend(customer_id for customer_id in sorted(expected) if customer_id not in seen)

    if any(instance.demands[customer_id] > instance.vehicle_capacity for customer_id in ordered):
        raise ValueError("A customer demand exceeds vehicle capacity")
    if sum(instance.demands[customer_id] for customer_id in ordered) > instance.num_vehicles * instance.vehicle_capacity + 1e-9:
        raise ValueError("Total demand exceeds the available vehicle capacity")

    routes: list[list[int]] = [[instance.depot, instance.depot] for _ in range(instance.num_vehicles)]
    loads = [0.0] * instance.num_vehicles
    for customer_id in ordered:
        demand = instance.demands[customer_id]
        vehicle_index = next((index for index, load in enumerate(loads) if load + demand <= instance.vehicle_capacity + 1e-9), None)
        if vehicle_index is None:
            raise ValueError(f"Could not place customer {customer_id} after repair")
        routes[vehicle_index].insert(-1, customer_id)
        loads[vehicle_index] += demand
    return Solution(routes=routes)

