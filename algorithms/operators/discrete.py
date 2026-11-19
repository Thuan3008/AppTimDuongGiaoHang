"""Discrete neighbourhood operators for depot-delimited HCMC routes."""

from __future__ import annotations

import random

from core.models import Solution


def _customer_positions(solution: Solution) -> list[tuple[int, int]]:
    return [(route_index, position) for route_index, route in enumerate(solution.routes) for position in range(1, len(route) - 1)]


def swap(solution: Solution, rng: random.Random) -> Solution:
    candidate = solution.copy()
    positions = _customer_positions(candidate)
    if len(positions) >= 2:
        left, right = rng.sample(positions, 2)
        candidate.routes[left[0]][left[1]], candidate.routes[right[0]][right[1]] = candidate.routes[right[0]][right[1]], candidate.routes[left[0]][left[1]]
    return candidate


def insertion(solution: Solution, rng: random.Random) -> Solution:
    candidate = solution.copy()
    positions = _customer_positions(candidate)
    if not positions:
        return candidate
    source_route, source_position = rng.choice(positions)
    customer_id = candidate.routes[source_route].pop(source_position)
    target_route = rng.randrange(len(candidate.routes))
    target_position = rng.randint(1, len(candidate.routes[target_route]) - 1)
    candidate.routes[target_route].insert(target_position, customer_id)
    return candidate


def relocate(solution: Solution, rng: random.Random) -> Solution:
    candidate = solution.copy()
    non_empty = [index for index, route in enumerate(candidate.routes) if len(route) > 2]
    if not non_empty:
        return candidate
    source_route = rng.choice(non_empty)
    source_position = rng.randrange(1, len(candidate.routes[source_route]) - 1)
    customer_id = candidate.routes[source_route].pop(source_position)
    target_route = rng.randrange(len(candidate.routes))
    target_position = rng.randint(1, len(candidate.routes[target_route]) - 1)
    candidate.routes[target_route].insert(target_position, customer_id)
    return candidate


def reverse_two_opt(solution: Solution, rng: random.Random) -> Solution:
    candidate = solution.copy()
    eligible = [index for index, route in enumerate(candidate.routes) if len(route) > 3]
    if not eligible:
        return candidate
    route_index = rng.choice(eligible)
    route = candidate.routes[route_index]
    left, right = sorted(rng.sample(range(1, len(route) - 1), 2))
    route[left:right + 1] = reversed(route[left:right + 1])
    return candidate


def cross_route_swap(solution: Solution, rng: random.Random) -> Solution:
    candidate = solution.copy()
    eligible = [index for index, route in enumerate(candidate.routes) if len(route) > 2]
    if len(eligible) < 2:
        return candidate
    left_route, right_route = rng.sample(eligible, 2)
    left_position = rng.randrange(1, len(candidate.routes[left_route]) - 1)
    right_position = rng.randrange(1, len(candidate.routes[right_route]) - 1)
    candidate.routes[left_route][left_position], candidate.routes[right_route][right_position] = candidate.routes[right_route][right_position], candidate.routes[left_route][left_position]
    return candidate


OPERATORS = (swap, insertion, relocate, reverse_two_opt, cross_route_swap)

