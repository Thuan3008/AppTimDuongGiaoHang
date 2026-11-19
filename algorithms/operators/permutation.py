"""Shared discrete permutation operators for TDVRP.

Provides basic and compound neighborhood moves:
- swap, relocate, inversion
- two_opt (within decoded route)
- two_opt_star (cross-route tail swap)
- double_relocate, block_swap, segment_exchange
- order_crossover (OX)
- align_towards (discrete path relinking)
"""
from __future__ import annotations
import numpy as np


def swap_indices(p: list[int], i: int, j: int) -> None:
    p[i], p[j] = p[j], p[i]


def relocate_indices(p: list[int], a: int, b: int) -> None:
    cust = p.pop(a)
    p.insert(b, cust)


def swap_random(p: tuple[int, ...] | list[int], rng: np.random.Generator) -> tuple[int, ...]:
    arr = list(p)
    n = len(arr)
    if n < 2:
        return tuple(arr)
    i, j = rng.choice(n, 2, replace=False)
    swap_indices(arr, int(i), int(j))
    return tuple(arr)


def relocate_random(p: tuple[int, ...] | list[int], rng: np.random.Generator) -> tuple[int, ...]:
    arr = list(p)
    n = len(arr)
    if n < 2:
        return tuple(arr)
    a, b = rng.choice(n, 2, replace=False)
    relocate_indices(arr, int(a), int(b))
    return tuple(arr)


def inversion_random(p: tuple[int, ...] | list[int], rng: np.random.Generator) -> tuple[int, ...]:
    arr = list(p)
    n = len(arr)
    if n < 2:
        return tuple(arr)
    a, b = sorted(int(x) for x in rng.choice(n, 2, replace=False))
    arr[a:b + 1] = reversed(arr[a:b + 1])
    return tuple(arr)


def two_opt(
    p: tuple[int, ...] | list[int],
    routes: tuple[tuple[int, ...], ...] | list[list[int]] | None,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """2-opt reversal within a single decoded route."""
    arr = list(p)
    n = len(arr)
    if n < 2:
        return tuple(arr)
    eligible = [r for r in (routes or []) if len(r) >= 2]
    if not eligible:
        return inversion_random(arr, rng)

    route = eligible[int(rng.integers(len(eligible)))]
    # strip depot 0 if present
    route_custs = [c for c in route if c != 0]
    if len(route_custs) < 2:
        return inversion_random(arr, rng)

    left, right = sorted(int(x) for x in rng.choice(len(route_custs), 2, replace=False))
    sub = route_custs[left:right + 1]
    positions = [arr.index(c) for c in sub]
    a, b = min(positions), max(positions)
    arr[a:b + 1] = reversed(arr[a:b + 1])
    return tuple(arr)


def two_opt_star(
    p: tuple[int, ...] | list[int],
    routes: tuple[tuple[int, ...], ...] | list[list[int]] | None,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """2-opt* cross-route tail exchange."""
    arr = list(p)
    n = len(arr)
    if n < 2:
        return tuple(arr)
    valid_routes = [[c for c in r if c != 0] for r in (routes or []) if len([c for c in r if c != 0]) >= 1]
    if len(valid_routes) < 2:
        return swap_random(arr, rng)

    r1_idx, r2_idx = rng.choice(len(valid_routes), 2, replace=False)
    r1 = valid_routes[int(r1_idx)]
    r2 = valid_routes[int(r2_idx)]

    cut1 = int(rng.integers(len(r1)))
    cut2 = int(rng.integers(len(r2)))

    tail1 = set(r1[cut1:])
    tail2 = set(r2[cut2:])

    if not tail1 and not tail2:
        return swap_random(arr, rng)

    # In permutation, swap the relative order of tail1 and tail2 elements
    new_arr = []
    t1_list = [c for c in arr if c in tail1]
    t2_list = [c for c in arr if c in tail2]
    
    # Simple tail exchange in permutation: replace tail1 members with tail2, and vice versa
    t1_iter = iter(t1_list)
    t2_iter = iter(t2_list)
    for c in arr:
        if c in tail1:
            try:
                new_arr.append(next(t2_iter))
            except StopIteration:
                new_arr.append(c)
        elif c in tail2:
            try:
                new_arr.append(next(t1_iter))
            except StopIteration:
                new_arr.append(c)
        else:
            new_arr.append(c)

    # Append any remaining tail items if sets had different sizes
    rem_t1 = list(t1_iter)
    rem_t2 = list(t2_iter)
    if rem_t1 or rem_t2:
        # Fallback to double relocate for clean invariants
        return double_relocate(arr, rng)

    if sorted(new_arr) != list(range(1, n + 1)):
        return double_relocate(arr, rng)

    return tuple(new_arr)


def double_relocate(p: tuple[int, ...] | list[int], rng: np.random.Generator) -> tuple[int, ...]:
    """Two successive random relocate moves."""
    arr = list(p)
    n = len(arr)
    if n < 2:
        return tuple(arr)
    for _ in range(2):
        a, b = rng.choice(n, 2, replace=False)
        relocate_indices(arr, int(a), int(b))
    return tuple(arr)


def block_swap(p: tuple[int, ...] | list[int], rng: np.random.Generator) -> tuple[int, ...]:
    """Swap two disjoint non-empty blocks."""
    arr = list(p)
    n = len(arr)
    if n < 4:
        return swap_random(arr, rng)
    width = int(rng.integers(1, max(2, n // 3)))
    a = int(rng.integers(0, n - 2 * width + 1))
    b = int(rng.integers(a + width, n - width + 1))
    arr[a:a + width], arr[b:b + width] = arr[b:b + width], arr[a:a + width]
    return tuple(arr)


def segment_exchange(p: tuple[int, ...] | list[int], rng: np.random.Generator) -> tuple[int, ...]:
    """Exchange two segments of possibly different lengths."""
    arr = list(p)
    n = len(arr)
    if n < 4:
        return swap_random(arr, rng)
    w1 = int(rng.integers(1, max(2, n // 3)))
    w2 = int(rng.integers(1, max(2, n // 3)))
    if w1 + w2 >= n:
        w1, w2 = 1, 1
    a = int(rng.integers(0, n - (w1 + w2) + 1))
    b = int(rng.integers(a + w1, n - w2 + 1))
    seg1 = arr[a:a + w1]
    seg2 = arr[b:b + w2]
    mid = arr[a + w1:b]
    arr = arr[:a] + seg2 + mid + seg1 + arr[b + w2:]
    return tuple(arr)


def order_crossover(
    first: tuple[int, ...] | list[int],
    second: tuple[int, ...] | list[int],
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """One OX child: retain [a:b] from first parent, cyclically fill from second parent."""
    n = len(first)
    if n < 2:
        return tuple(first)
    a, b = sorted(int(i) for i in rng.choice(n + 1, 2, replace=False))
    child: list[int | None] = [None] * n
    child[a:b] = first[a:b]
    retained = set(first[a:b])
    donor = [second[(b + i) % n] for i in range(n) if second[(b + i) % n] not in retained]
    slots = [(b + i) % n for i in range(n) if child[(b + i) % n] is None]
    for slot, customer in zip(slots, donor):
        child[slot] = customer
    return tuple(int(x) for x in child)


def align_towards(
    p: tuple[int, ...] | list[int],
    target: tuple[int, ...] | list[int],
    steps: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Discrete path-relinking move: perform up to `steps` swaps towards target permutation."""
    arr = list(p)
    n = len(arr)
    target_list = list(target)
    for _ in range(steps):
        mismatches = [i for i in range(n) if arr[i] != target_list[i]]
        if not mismatches:
            break
        idx = int(rng.choice(mismatches))
        target_val = target_list[idx]
        cur_pos = arr.index(target_val)
        swap_indices(arr, idx, cur_pos)
    return tuple(arr)

