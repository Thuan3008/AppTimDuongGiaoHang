from __future__ import annotations

from core.models import Solution, TDVRPProblem


def capacity_routes(customer_ids: list[int], problem: TDVRPProblem) -> list[list[int]]:
    routes = [[0] for _ in problem.vehicles]
    loads = [0.0] * len(problem.vehicles)
    demand = {c.id: c.demand for c in problem.customers}
    for cid in customer_ids:
        placed = False
        for idx, vehicle in enumerate(problem.vehicles):
            if loads[idx] + demand[cid] <= vehicle.capacity:
                routes[idx].append(cid); loads[idx] += demand[cid]; placed = True; break
        if not placed:
            # Instance generation should be feasible; keep the customer in the lightest route for a diagnosable validation failure.
            idx = min(range(len(routes)), key=lambda i: loads[i]); routes[idx].append(cid); loads[idx] += demand[cid]
    return [route + [0] for route in routes]


def initialize_population(problem: TDVRPProblem, size: int, rng) -> list[Solution]:
    ids = problem.customer_ids
    population = []
    for _ in range(size):
        shuffled = list(rng.permutation(ids))
        population.append(Solution(capacity_routes(shuffled, problem)))
    return population


def split_population(population: list[Solution], num_swarms: int) -> list[list[Solution]]:
    if num_swarms < 1 or num_swarms > len(population):
        raise ValueError("num_swarms must be between 1 and population_size")
    result = [[] for _ in range(num_swarms)]
    for idx, individual in enumerate(population):
        result[idx % num_swarms].append(individual)
    return result
