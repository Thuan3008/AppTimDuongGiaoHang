"""Time-Dependent Criticality Analysis for TDVRP Candidates.

Computes customer and arc criticality scores using only canonical route trace,
arrival/departure timestamps, and arc travel-time information.

Strictly follows:
- Zero BKS or future information usage.
- Zero extra candidate evaluations or Split calls (0 NFE overhead).
- Normalized components: TravelContribution, TemporalContribution, RouteContribution.
- Fixed weights: alpha = 1/3, beta = 1/3, gamma = 1/3.
- Output normalized to [0, 1].
"""
from __future__ import annotations
from typing import Any, Sequence
import numpy as np


def compute_customer_criticality(
    problem: Any,
    routes: Sequence[Sequence[int]],
    route_evaluations: Sequence[tuple[float, float, float]] | None = None,
    alpha: float = 1.0 / 3.0,
    beta: float = 1.0 / 3.0,
    gamma: float = 1.0 / 3.0,
) -> dict[int, float]:
    """Compute normalized criticality score C_i in [0, 1] for each customer 1..n.

    C_i = alpha * TravelContribution_i + beta * TemporalContribution_i + gamma * RouteContribution_i

    Args:
        problem: TDVRPProblem instance.
        routes: Decoded canonical routes (e.g., ((1, 2), (3, 4))).
        route_evaluations: Sequence of (duration, departure, return_time) tuples per route.
        alpha: Weight for TravelContribution (default 1/3).
        beta: Weight for TemporalContribution (default 1/3).
        gamma: Weight for RouteContribution (default 1/3).

    Returns:
        dict mapping customer id (1..n) to criticality score in [0.0, 1.0].
    """
    n = getattr(problem, "dimension", 10)
    demands = getattr(problem, "demands", (0,) * (n + 1))
    services = getattr(problem, "services", (0,) * (n + 1))
    arcs = getattr(problem, "arcs", {})

    # If route_evaluations not provided or mismatched, obtain from problem.route_cost
    if route_evaluations is None or len(route_evaluations) != len(routes):
        evals = []
        for r in routes:
            if hasattr(problem, "route_cost"):
                try:
                    evals.append(problem.route_cost(r))
                except Exception:
                    evals.append((float(len(r)), 0.0, float(len(r))))
            else:
                evals.append((float(len(r)), 0.0, float(len(r))))
        route_evaluations = evals

    travel_raw: dict[int, float] = {}
    temporal_raw: dict[int, float] = {}
    route_raw: dict[int, float] = {}

    for route, (dur, departure, ret_time) in zip(routes, route_evaluations):
        dur = max(1e-6, float(dur))
        m = len(route)
        if m == 0:
            continue

        # Simulate route timestamps if arcs are available
        clock = float(departure)
        arrival_times: list[float] = []
        departure_times: list[float] = []
        prev = 0

        has_arcs = bool(arcs)
        if has_arcs:
            for c in route:
                arc = arcs.get((prev, c))
                if arc is not None:
                    try:
                        clock = float(arc(clock))
                    except Exception:
                        clock += 10.0
                else:
                    clock += 10.0
                arrival_times.append(clock)
                clock += float(services[c])
                departure_times.append(clock)
                prev = c
        else:
            # Synthetic / mock fallback
            step = dur / max(1, 2 * m)
            cur = departure
            for c in route:
                cur += step
                arrival_times.append(cur)
                cur += float(services[c]) + step
                departure_times.append(cur)

        # Compute per-customer contributions within this route
        for k, c in enumerate(route):
            u = 0 if k == 0 else route[k - 1]
            v = 0 if k == m - 1 else route[k + 1]

            t_dep_u = departure if k == 0 else departure_times[k - 1]
            t_arr_c = arrival_times[k]
            t_dep_c = departure_times[k]
            t_arr_v = ret_time if k == m - 1 else arrival_times[k + 1]

            # 1. TravelContribution: Marginal detour around customer c
            tt_in = max(0.0, t_arr_c - t_dep_u)
            tt_out = max(0.0, t_arr_v - t_dep_c)
            tt_direct = 0.0
            if has_arcs and (u, v) in arcs:
                try:
                    arc_uv = arcs[(u, v)]
                    tt_direct = max(0.0, float(arc_uv(t_dep_u)) - t_dep_u)
                except Exception:
                    tt_direct = 0.0
            detour = max(0.0, (tt_in + tt_out) - tt_direct)
            travel_raw[c] = detour

            # 2. TemporalContribution: Traffic delay / congestion severity
            cong_in = 0.0
            cong_out = 0.0
            if has_arcs:
                arc_in = arcs.get((u, c))
                if arc_in is not None and hasattr(arc_in, "xs") and hasattr(arc_in, "ys"):
                    try:
                        min_tt = min(y - x for x, y in zip(arc_in.xs, arc_in.ys))
                        cong_in = max(0.0, (tt_in - min_tt) / max(1e-6, min_tt))
                    except Exception:
                        cong_in = 0.0

                arc_out = arcs.get((c, v))
                if arc_out is not None and hasattr(arc_out, "xs") and hasattr(arc_out, "ys"):
                    try:
                        min_tt = min(y - x for x, y in zip(arc_out.xs, arc_out.ys))
                        cong_out = max(0.0, (tt_out - min_tt) / max(1e-6, min_tt))
                    except Exception:
                        cong_out = 0.0
            else:
                cong_in = 0.0
                cong_out = 0.0

            temporal_raw[c] = 0.5 * (cong_in + cong_out)

            # 3. RouteContribution: Fraction of route duration consumed by c
            c_share = (float(services[c]) + tt_in + tt_out) / dur
            route_raw[c] = c_share

    # Ensure all customers 1..n have raw entries
    all_customers = list(range(1, n + 1))
    for c in all_customers:
        if c not in travel_raw:
            travel_raw[c] = 0.0
        if c not in temporal_raw:
            temporal_raw[c] = 0.0
        if c not in route_raw:
            route_raw[c] = 0.0

    # Normalize each component across all customers to [0, 1]
    def _normalize(vals: dict[int, float]) -> dict[int, float]:
        arr = [vals[c] for c in all_customers]
        mn, mx = min(arr), max(arr)
        if mx > mn + 1e-12:
            return {c: (vals[c] - mn) / (mx - mn) for c in all_customers}
        return {c: 0.5 for c in all_customers}

    norm_travel = _normalize(travel_raw)
    norm_temporal = _normalize(temporal_raw)
    norm_route = _normalize(route_raw)

    criticality: dict[int, float] = {}
    for c in all_customers:
        score = (
            alpha * norm_travel[c]
            + beta * norm_temporal[c]
            + gamma * norm_route[c]
        )
        criticality[c] = float(np.clip(score, 0.0, 1.0))

    return criticality


def compute_removal_probabilities(
    criticality: dict[int, float],
    customers: Sequence[int],
    epsilon: float = 0.10,
) -> np.ndarray:
    """Compute removal probability distribution p_i proportional to epsilon + C_i.

    Guarantees epsilon > 0 so every candidate has non-zero removal probability,
    while critical customers have significantly higher probability.
    """
    scores = np.array([criticality.get(int(c), 0.5) + epsilon for c in customers], dtype=float)
    total = np.sum(scores)
    if total <= 0 or not np.isfinite(total):
        return np.ones(len(customers), dtype=float) / len(customers)
    return scores / total

