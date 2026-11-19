"""Destroy and repair operators for ALNS and large neighborhood search.

All heuristic calculations use static arc times and demands from TDVRPProblem,
consuming 0 canonical NFE during candidate construction.
"""
from __future__ import annotations
import math
from typing import Sequence
import numpy as np

from core.tdvrp_common import TDVRPProblem


def get_static_travel_time(problem: TDVRPProblem, i: int, j: int) -> float:
    """Return static/free-flow travel time between node i and j."""
    if i == j:
        return 0.0
    arc = problem.arcs.get((i, j))
    if arc is not None and len(arc.xs) > 0:
        return float(arc.ys[0] - arc.xs[0])
    return 100.0


def random_removal(
    p: tuple[int, ...] | list[int],
    routes: Sequence[Sequence[int]] | None,
    q: int,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    """Remove q customers uniformly at random.
    
    Returns (partial_permutation, removed_customers).
    """
    arr = list(p)
    n = len(arr)
    q = max(1, min(n - 1, q))
    removed_indices = set(int(i) for i in rng.choice(n, q, replace=False))
    partial = [c for i, c in enumerate(arr) if i not in removed_indices]
    removed = [arr[i] for i in sorted(removed_indices)]
    return partial, removed


def worst_removal(
    p: tuple[int, ...] | list[int],
    routes: Sequence[Sequence[int]] | None,
    q: int,
    rng: np.random.Generator,
    problem: TDVRPProblem,
    randomization_p: float = 3.0,
) -> tuple[list[int], list[int]]:
    """Remove q customers that contribute most to detour cost."""
    arr = list(p)
    n = len(arr)
    q = max(1, min(n - 1, q))
    if n <= 2:
        return random_removal(arr, routes, q, rng)

    costs: list[tuple[float, int]] = []
    for idx, c in enumerate(arr):
        pred = arr[idx - 1] if idx > 0 else 0
        succ = arr[idx + 1] if idx < n - 1 else 0
        d_with = get_static_travel_time(problem, pred, c) + get_static_travel_time(problem, c, succ)
        d_without = get_static_travel_time(problem, pred, succ)
        cost_contrib = d_with - d_without
        costs.append((cost_contrib, c))

    # Sort descending by cost contribution
    costs.sort(key=lambda x: x[0], reverse=True)

    removed_set: set[int] = set()
    while len(removed_set) < q and costs:
        # Randomized choice with power parameter
        r = rng.random()
        idx = int(len(costs) * (r ** randomization_p))
        idx = min(idx, len(costs) - 1)
        removed_set.add(costs.pop(idx)[1])

    partial = [c for c in arr if c not in removed_set]
    removed = [c for c in arr if c in removed_set]
    return partial, removed


def related_removal(
    p: tuple[int, ...] | list[int],
    routes: Sequence[Sequence[int]] | None,
    q: int,
    rng: np.random.Generator,
    problem: TDVRPProblem,
    phi_dist: float = 1.0,
    phi_dem: float = 1.0,
    randomization_p: float = 3.0,
) -> tuple[list[int], list[int]]:
    """Shaw related removal: remove customers related by distance and demand."""
    arr = list(p)
    n = len(arr)
    q = max(1, min(n - 1, q))
    if n <= 2:
        return random_removal(arr, routes, q, rng)

    # Start with a random seed customer
    seed = int(rng.choice(arr))
    removed_set: set[int] = {seed}

    while len(removed_set) < q:
        # Pick one reference from removed set
        ref = int(rng.choice(list(removed_set)))
        remaining = [c for c in arr if c not in removed_set]
        if not remaining:
            break

        # Calculate relatedness: lower R means more related
        scored: list[tuple[float, int]] = []
        for c in remaining:
            d = get_static_travel_time(problem, ref, c)
            dem_diff = abs(problem.demands[ref] - problem.demands[c])
            r_val = phi_dist * d + phi_dem * dem_diff
            scored.append((r_val, c))

        scored.sort(key=lambda x: x[0])
        r = rng.random()
        idx = int(len(scored) * (r ** randomization_p))
        idx = min(idx, len(scored) - 1)
        removed_set.add(scored[idx][1])

    partial = [c for c in arr if c not in removed_set]
    removed = [c for c in arr if c in removed_set]
    return partial, removed


def route_removal(
    p: tuple[int, ...] | list[int],
    routes: Sequence[Sequence[int]] | None,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    """Remove all customers belonging to one decoded route."""
    arr = list(p)
    n = len(arr)
    valid_routes = [[c for c in r if c != 0] for r in (routes or []) if len([c for c in r if c != 0]) >= 1]
    if len(valid_routes) < 2 or n <= 2:
        # Fallback to removing max(1, n // 3) customers randomly
        return random_removal(arr, routes, max(1, n // 3), rng)

    chosen_route = valid_routes[int(rng.integers(len(valid_routes)))]
    removed_set = set(chosen_route)
    if len(removed_set) >= n:
        # Avoid removing all customers
        removed_set = set(chosen_route[:max(1, len(chosen_route) - 1)])

    partial = [c for c in arr if c not in removed_set]
    removed = [c for c in arr if c in removed_set]
    return partial, removed


def greedy_insertion(
    p_partial: list[int],
    removed: list[int],
    rng: np.random.Generator,
    problem: TDVRPProblem,
) -> tuple[int, ...]:
    """Greedy cheapest insertion based on static arc distances."""
    arr = list(p_partial)
    to_insert = list(removed)
    rng.shuffle(to_insert)

    for c in to_insert:
        if not arr:
            arr.append(c)
            continue
        best_pos = 0
        best_delta = float("inf")
        # Try inserting at each position
        for pos in range(len(arr) + 1):
            pred = arr[pos - 1] if pos > 0 else 0
            succ = arr[pos] if pos < len(arr) else 0
            delta = (get_static_travel_time(problem, pred, c)
                     + get_static_travel_time(problem, c, succ)
                     - get_static_travel_time(problem, pred, succ))
            if delta < best_delta:
                best_delta = delta
                best_pos = pos
        arr.insert(best_pos, c)
    return tuple(arr)


def regret2_insertion(
    p_partial: list[int],
    removed: list[int],
    rng: np.random.Generator,
    problem: TDVRPProblem,
) -> tuple[int, ...]:
    """Regret-2 insertion based on difference between 2nd best and best insertion positions."""
    arr = list(p_partial)
    unassigned = list(removed)

    while unassigned:
        if not arr:
            c = unassigned.pop(int(rng.integers(len(unassigned))))
            arr.append(c)
            continue

        best_cust_idx = 0
        max_regret = -float("inf")
        best_target_pos = 0

        for u_idx, c in enumerate(unassigned):
            pos_deltas: list[tuple[float, int]] = []
            for pos in range(len(arr) + 1):
                pred = arr[pos - 1] if pos > 0 else 0
                succ = arr[pos] if pos < len(arr) else 0
                delta = (get_static_travel_time(problem, pred, c)
                         + get_static_travel_time(problem, c, succ)
                         - get_static_travel_time(problem, pred, succ))
                pos_deltas.append((delta, pos))

            pos_deltas.sort(key=lambda x: x[0])
            c1_delta, c1_pos = pos_deltas[0]
            c2_delta = pos_deltas[1][0] if len(pos_deltas) > 1 else c1_delta

            regret = c2_delta - c1_delta
            if regret > max_regret:
                max_regret = regret
                best_cust_idx = u_idx
                best_target_pos = c1_pos

        c_to_insert = unassigned.pop(best_cust_idx)
        arr.insert(best_target_pos, c_to_insert)

    return tuple(arr)

