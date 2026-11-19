"""Route-Aware Neighborhood Operators for TDVRP.

Operators read decoded canonical routes to guide moves in permutation space:
- route_aware_relocate: inter-route / intra-route customer relocation
- route_aware_swap: inter-route / intra-route customer swap
- route_aware_two_opt: 2-opt reversal within a single decoded route
- route_aware_two_opt_star: cross-route tail exchange (2-opt*)

All operators output a valid permutation of (1..n) and preserve permutation invariants.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np

from algorithms.operators.permutation import (
    swap_indices,
    relocate_indices,
    swap_random,
    relocate_random,
    inversion_random,
    two_opt,
    two_opt_star,
)


def _get_clean_routes(routes: Sequence[Sequence[int]] | None) -> list[list[int]]:
    """Extract non-empty customer sequences from routes, stripping depot 0 if present."""
    if not routes:
        return []
    clean = []
    for r in routes:
        custs = [int(c) for c in r if c != 0]
        if custs:
            clean.append(custs)
    return clean


def route_aware_relocate(
    permutation: Sequence[int],
    routes: Sequence[Sequence[int]] | None,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Relocate a customer to another position guided by route boundaries."""
    arr = [int(c) for c in permutation]
    n = len(arr)
    if n < 2:
        return tuple(arr)

    clean = _get_clean_routes(routes)
    if len(clean) >= 2:
        # Inter-route relocate: move customer ca from route ra to route rb
        ra_idx, rb_idx = rng.choice(len(clean), 2, replace=False)
        ra = clean[int(ra_idx)]
        rb = clean[int(rb_idx)]

        ca = int(rng.choice(ra))
        cb = int(rng.choice(rb))

        idx_a = arr.index(ca)
        arr.pop(idx_a)
        idx_b = arr.index(cb)

        # Insert before or after cb
        insert_pos = idx_b + int(rng.choice([0, 1]))
        insert_pos = max(0, min(len(arr), insert_pos))
        arr.insert(insert_pos, ca)
    elif len(clean) == 1 and len(clean[0]) >= 2:
        # Intra-route relocate
        r = clean[0]
        ca, cb = rng.choice(r, 2, replace=False)
        idx_a = arr.index(int(ca))
        arr.pop(idx_a)
        idx_b = arr.index(int(cb))
        insert_pos = idx_b + int(rng.choice([0, 1]))
        insert_pos = max(0, min(len(arr), insert_pos))
        arr.insert(insert_pos, int(ca))
    else:
        return relocate_random(arr, rng)

    # Invariant check
    if sorted(arr) != list(range(1, n + 1)):
        return relocate_random(permutation, rng)

    # If move was a no-op, enforce a distinct move
    if tuple(arr) == tuple(permutation) and n >= 2:
        return relocate_random(permutation, rng)

    return tuple(arr)


def route_aware_swap(
    permutation: Sequence[int],
    routes: Sequence[Sequence[int]] | None,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Swap two customers, preferring inter-route swaps when multiple routes exist."""
    arr = [int(c) for c in permutation]
    n = len(arr)
    if n < 2:
        return tuple(arr)

    clean = _get_clean_routes(routes)
    if len(clean) >= 2:
        ra_idx, rb_idx = rng.choice(len(clean), 2, replace=False)
        ra = clean[int(ra_idx)]
        rb = clean[int(rb_idx)]
        ca = int(rng.choice(ra))
        cb = int(rng.choice(rb))
        idx_a = arr.index(ca)
        idx_b = arr.index(cb)
        swap_indices(arr, idx_a, idx_b)
    elif len(clean) == 1 and len(clean[0]) >= 2:
        ca, cb = rng.choice(clean[0], 2, replace=False)
        idx_a = arr.index(int(ca))
        idx_b = arr.index(int(cb))
        swap_indices(arr, idx_a, idx_b)
    else:
        return swap_random(arr, rng)

    if sorted(arr) != list(range(1, n + 1)):
        return swap_random(permutation, rng)

    if tuple(arr) == tuple(permutation) and n >= 2:
        return swap_random(permutation, rng)

    return tuple(arr)


def route_aware_two_opt(
    permutation: Sequence[int],
    routes: Sequence[Sequence[int]] | None,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """2-opt reversal within a decoded route."""
    res = two_opt(permutation, routes, rng)
    if sorted(res) != list(range(1, len(permutation) + 1)):
        return inversion_random(permutation, rng)
    return res


def route_aware_two_opt_star(
    permutation: Sequence[int],
    routes: Sequence[Sequence[int]] | None,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """2-opt* cross-route tail exchange."""
    res = two_opt_star(permutation, routes, rng)
    if sorted(res) != list(range(1, len(permutation) + 1)):
        return swap_random(permutation, rng)
    return res

