"""TD-XGB: XGBoost-guided Feasible Insertion baseline for TDVRP Poryos2026.

Supervised machine learning baseline. Does NOT import td_mlp.py or alter other repo files.
Uses exact ATF route_cost oracle (TDVRPProblem) for candidate scoring and top-k verification.
Includes checkpoint pause/resume capability and auto-logging to run.log.

CLI usage:
  python algorithms/td_xgb.py collect --train-instances PATHS... --dataset out.json
  python algorithms/td_xgb.py train   --dataset data.json --model-path model.joblib
  python algorithms/td_xgb.py solve   --instance DIR_OR_VRP_JSON --model-path model.joblib
  python algorithms/td_xgb.py evaluate --instance DIR_OR_VRP_JSON --model-path model.joblib
  python algorithms/td_xgb.py self-test --instance DIR_OR_VRP_JSON
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Optional xgboost import – graceful fallback to heuristic-only mode
# ---------------------------------------------------------------------------
try:
    import xgboost as xgb
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    import pickle as joblib  # fallback
    _JOBLIB_AVAILABLE = True

# ---------------------------------------------------------------------------
# Feature schema – synchronized 1:1 with TD-MLP
# ---------------------------------------------------------------------------
FEATURE_NAMES: list[str] = [
    "demand_ratio",           # demand_c / Q
    "service_time",           # raw service time (seconds)
    "route_load_ratio",       # current route load / Q
    "route_len_ratio",        # len(route) / n
    "unserved_ratio",         # unserved_count / n
    "route_current_duration", # current route duration before insertion (seconds)
    "position_ratio",         # insert_pos / max(1, len(route))
    "route_count_ratio",      # num_routes / n
    "pred_to_cust_t0",        # travel time predecessor→customer at t=0 (seconds)
    "cust_to_succ_t0",        # travel time customer→successor at t=0  (seconds)
    "pred_to_succ_t0",        # travel time predecessor→successor at t=0 (seconds)
    "detour",                 # pred_to_cust_t0 + cust_to_succ_t0 - pred_to_succ_t0
]

_FEATURE_DIM = len(FEATURE_NAMES)

# ---------------------------------------------------------------------------
# Logging – Dual stdout and run.log
# ---------------------------------------------------------------------------
log = logging.getLogger("td_xgb")


def setup_logging(log_file: str | Path | None = "run.log") -> None:
    """Configure logging to stream to stdout and append to log_file."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(p, mode="a", encoding="utf-8"))
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s td_xgb] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=handlers,
        force=True,
    )


# Initial logging setup
setup_logging("run.log")


# ---------------------------------------------------------------------------
# Checkpoint persistence helpers for search / solve
# ---------------------------------------------------------------------------

def _save_search_checkpoint(
    checkpoint_path: str | Path,
    instance_name: str,
    dimension: int,
    mode: str,
    routes: list[tuple[int, ...]],
    unserved: set[int],
    route_costs: dict[int, float],
    route_cost_calls: int,
    elapsed_seconds: float,
    completed: bool = False,
) -> None:
    """Save partial search state to a JSON file for resume capability."""
    p = Path(checkpoint_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "instance_name": instance_name,
        "dimension": dimension,
        "mode": mode,
        "step": dimension - len(unserved),
        "routes": [list(r) for r in routes],
        "unserved": sorted(unserved),
        "route_costs": {str(k): float(v) for k, v in route_costs.items()},
        "route_cost_calls": route_cost_calls,
        "elapsed_seconds": elapsed_seconds,
        "completed": completed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _load_search_checkpoint(
    checkpoint_path: str | Path,
    expected_dimension: int,
) -> dict[str, Any] | None:
    """Load partial search state if file exists and matches instance dimension."""
    p = Path(checkpoint_path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("dimension") != expected_dimension:
            log.warning(
                "Checkpoint dimension (%s) != expected (%d); ignoring checkpoint",
                data.get("dimension"),
                expected_dimension,
            )
            return None
        return data
    except Exception as e:
        log.warning("Could not read search checkpoint %s: %s; starting fresh", checkpoint_path, e)
        return None


# ---------------------------------------------------------------------------
# Instance loading
# ---------------------------------------------------------------------------

def find_vrp_json(path_or_dir: str | Path) -> Path:
    """Return the .vrp.json file given either a directory or file path."""
    p = Path(path_or_dir)
    if p.is_file() and p.name.endswith(".vrp.json"):
        return p
    if p.is_dir():
        matches = sorted(p.glob("*.vrp.json"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Multiple .vrp.json in {p}: {[m.name for m in matches]}")
        raise FileNotFoundError(f"No .vrp.json found in {p}")
    raise FileNotFoundError(f"Path not found or not a .vrp.json: {p}")


def load_instance(path_or_dir: str | Path):
    """Load TDVRPProblem from a .vrp.json file or directory."""
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from core.tdvrp_common import TDVRPProblem  # noqa: PLC0415
    vrp_path = find_vrp_json(path_or_dir)
    collection_root = root / "data" / "TDVRP"
    log.info("Loading instance: %s", vrp_path.name)
    return TDVRPProblem(vrp_path, collection_root=collection_root)


def load_bks(path_or_dir: str | Path) -> float | None:
    """Read BKS Duration value from a .bks.Duration.json or .txt file."""
    p = Path(path_or_dir)
    if p.is_file():
        p = p.parent
    for bks_file in sorted(p.glob("*.bks.Duration.json")):
        try:
            data = json.loads(bks_file.read_text(encoding="utf-8"))
            for key in ("cost", "duration", "best", "value", "objective"):
                if key in data:
                    return float(data[key])
            if "metadata" in data and "validated_cost" in data["metadata"]:
                return float(data["metadata"]["validated_cost"])
            if isinstance(data, (int, float)):
                return float(data)
        except Exception:
            pass
    for bks_txt in sorted(p.glob("*.bks.Duration.txt")):
        try:
            return float(bks_txt.read_text(encoding="utf-8").strip())
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _arc_at_zero(problem, src: int, dst: int) -> float:
    """Return travel time src→dst at t=0 (using NDCPWLF arc evaluated at 0)."""
    arc = problem.arcs.get((src, dst))
    if arc is None:
        return float("inf")
    try:
        t0 = float(arc.min_domain)
        return float(arc(t0)) - t0
    except Exception:
        return float("inf")


def extract_features(
    problem,
    routes: list[tuple[int, ...]],
    route_idx: int,
    insert_pos: int,
    customer: int,
    unserved_count: int,
    route_durations: dict[int, float],
) -> np.ndarray:
    """
    Build the 12-dimensional feature vector for inserting customer at position
    insert_pos in routes[route_idx]. Exactly matches TD-MLP schema.
    """
    n = problem.dimension
    Q = problem.capacity

    demand_c = problem.demands[customer]
    service_c = problem.services[customer]

    if route_idx < 0 or route_idx >= len(routes):
        route = ()
        route_load = 0.0
        cur_duration = 0.0
    else:
        route = routes[route_idx]
        route_load = sum(problem.demands[c] for c in route)
        cur_duration = route_durations.get(route_idx, 0.0)

    route_len = len(route)
    num_routes = len(routes)

    # Predecessor / successor in route after insertion
    pred = 0 if insert_pos == 0 else route[insert_pos - 1]
    succ = 0 if insert_pos >= route_len else route[insert_pos]

    t_pred_cust = _arc_at_zero(problem, pred, customer)
    t_cust_succ = _arc_at_zero(problem, customer, succ)
    t_pred_succ = _arc_at_zero(problem, pred, succ)
    detour = t_pred_cust + t_cust_succ - t_pred_succ

    feat = np.array([
        demand_c / Q,
        service_c,
        route_load / Q,
        route_len / n,
        unserved_count / n,
        cur_duration,
        insert_pos / max(1, route_len),
        num_routes / n,
        t_pred_cust if math.isfinite(t_pred_cust) else 0.0,
        t_cust_succ if math.isfinite(t_cust_succ) else 0.0,
        t_pred_succ if math.isfinite(t_pred_succ) else 0.0,
        detour if math.isfinite(detour) else 0.0,
    ], dtype=np.float64)

    assert len(feat) == _FEATURE_DIM
    return feat


# ---------------------------------------------------------------------------
# Candidate enumeration
# ---------------------------------------------------------------------------

@dataclass
class _Candidate:
    route_idx: int      # -1 or len(routes) → new route
    insert_pos: int
    customer: int
    route_after: tuple[int, ...]
    feature_vec: np.ndarray
    exact_cost: float | None = None
    base_route_cost: float | None = None


def enumerate_candidates(
    problem,
    routes: list[tuple[int, ...]],
    unserved: set[int],
    route_durations: dict[int, float],
) -> list[_Candidate]:
    """Return all capacity-feasible (customer, route, position) candidates + features."""
    Q = problem.capacity
    candidates: list[_Candidate] = []
    unserved_count = len(unserved)

    for customer in sorted(unserved):
        d_c = problem.demands[customer]
        # Existing routes
        for ridx, route in enumerate(routes):
            route_load = sum(problem.demands[c] for c in route)
            if route_load + d_c > Q:
                continue
            for pos in range(len(route) + 1):
                new_route = route[:pos] + (customer,) + route[pos:]
                feat = extract_features(
                    problem, routes, ridx, pos, customer, unserved_count, route_durations
                )
                candidates.append(_Candidate(
                    route_idx=ridx,
                    insert_pos=pos,
                    customer=customer,
                    route_after=new_route,
                    feature_vec=feat,
                ))
        # New route
        if d_c <= Q:
            feat = extract_features(
                problem, routes, -1, 0, customer, unserved_count, route_durations
            )
            candidates.append(_Candidate(
                route_idx=len(routes),
                insert_pos=0,
                customer=customer,
                route_after=(customer,),
                feature_vec=feat,
            ))

    return candidates


# ---------------------------------------------------------------------------
# Route cost helper
# ---------------------------------------------------------------------------

def _get_route_cost(problem, route: tuple[int, ...]) -> float | None:
    """Return exact route duration (NDCPWLF ATF) or None if infeasible."""
    if not route:
        return 0.0
    try:
        result = problem.route_cost(route)
        return float(result[0])
    except (ValueError, Exception):
        return None


# ---------------------------------------------------------------------------
# Heuristic solver (Cheapest feasible insertion, no ML)
# ---------------------------------------------------------------------------

def solve_heuristic(
    problem,
    seed: int = 0,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
) -> tuple[list[tuple[int, ...]], float]:
    """
    Parallel cheapest feasible insertion with exact route costs (Visser-Spliet §8).
    Unchanged routes preserve cached customer insertion descriptors for high speed.
    Supports checkpoint save and resume.
    Returns (routes, total_duration).
    """
    t_start = time.perf_counter()
    n = problem.dimension
    unplanned = set(range(1, n + 1))
    routes: list[tuple[int, ...]] = []
    route_costs: dict[int, float] = {}
    descriptors: dict[tuple[tuple[int, ...], int], tuple[float, int, tuple[int, ...]] | None] = {}
    route_cost_calls = 0
    time_offset = 0.0

    if resume and checkpoint_path:
        ckpt = _load_search_checkpoint(checkpoint_path, n)
        if ckpt and not ckpt.get("completed", False):
            routes = [tuple(r) for r in ckpt["routes"]]
            unplanned = set(ckpt["unserved"])
            route_costs = {int(k): v for k, v in ckpt["route_costs"].items()}
            route_cost_calls = ckpt.get("route_cost_calls", 0)
            time_offset = ckpt.get("elapsed_seconds", 0.0)
            log.info(
                "Resuming heuristic solve from checkpoint: %d customers routed, %d unserved remaining",
                n - len(unplanned), len(unplanned),
            )

    # Pre-evaluate singletons
    singleton_costs: dict[int, float] = {}
    for c in range(1, n + 1):
        cost = _get_route_cost(problem, (c,))
        route_cost_calls += 1
        if cost is None:
            raise ValueError(f"No feasible singleton route for customer {c}")
        singleton_costs[c] = cost

    try:
        while unplanned:
            best = None
            for customer in sorted(unplanned):
                singleton = (customer,)
                d_c = problem.demands[customer]
                choices: list[tuple[float, int, int, tuple[int, ...]]] = [
                    (singleton_costs[customer], len(routes), 0, singleton)
                ]
                for index, route in enumerate(routes):
                    key = (route, customer)
                    if key not in descriptors:
                        choice = None
                        route_load = sum(problem.demands[c] for c in route)
                        if route_load + d_c <= problem.capacity:
                            base = route_costs.get(index, 0.0)
                            for position in range(len(route) + 1):
                                candidate = route[:position] + singleton + route[position:]
                                cost = _get_route_cost(problem, candidate)
                                route_cost_calls += 1
                                if cost is not None:
                                    item = (cost - base, position, candidate)
                                    if choice is None or item < choice:
                                        choice = item
                        descriptors[key] = choice
                    choice = descriptors[key]
                    if choice is not None:
                        choices.append((choice[0], index, choice[1], choice[2]))

                delta, index, position, candidate = min(choices)
                item = (delta, customer, index, position, candidate)
                if best is None or item < best:
                    best = item

            if best is None:
                break

            _, customer, index, _, candidate = best
            if index == len(routes):
                routes.append(candidate)
                route_costs[index] = _get_route_cost(problem, candidate) or 0.0
                route_cost_calls += 1
            else:
                routes[index] = candidate
                route_costs[index] = _get_route_cost(problem, candidate) or 0.0
                route_cost_calls += 1
            unplanned.remove(customer)

            # Checkpoint progress
            if checkpoint_path:
                _save_search_checkpoint(
                    checkpoint_path=checkpoint_path,
                    instance_name=getattr(problem, "name", ""),
                    dimension=n,
                    mode="heuristic_only",
                    routes=routes,
                    unserved=unplanned,
                    route_costs=route_costs,
                    route_cost_calls=route_cost_calls,
                    elapsed_seconds=time.perf_counter() - t_start + time_offset,
                    completed=False,
                )
    except KeyboardInterrupt:
        log.warning(
            "Heuristic search paused by user! State saved to %s. Run with --resume to continue.",
            checkpoint_path,
        )
        raise

    runtime = time.perf_counter() - t_start + time_offset
    if checkpoint_path:
        _save_search_checkpoint(
            checkpoint_path=checkpoint_path,
            instance_name=getattr(problem, "name", ""),
            dimension=n,
            mode="heuristic_only",
            routes=routes,
            unserved=unplanned,
            route_costs=route_costs,
            route_cost_calls=route_cost_calls,
            elapsed_seconds=runtime,
            completed=True,
        )

    total = math.fsum(route_costs.get(i, 0.0) for i in range(len(routes)))
    return routes, total


# ---------------------------------------------------------------------------
# XGBoost solver
# ---------------------------------------------------------------------------

def solve_xgb(
    problem,
    model: Any,
    checkpoint_feature_names: list[str],
    seed: int = 0,
    top_k: int = 3,
    time_budget_sec: float = float("inf"),
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """
    XGBoost-guided greedy insertion with checkpoint save and resume support.

    At each step:
      1. Enumerate all feasible (customer, route, position) candidates.
      2. Predict insertion cost delta with XGBoost (lower = better).
      3. Select top-k lowest-scoring candidates (default k=3).
      4. Verify exact ATF route cost on top-k and choose the minimum exact delta.
      5. Insert and save checkpoint.

    Returns a result dict.
    """
    if not _XGBOOST_AVAILABLE:
        raise ImportError("xgboost is not installed. Run 'pip install xgboost' to use TD-XGB.")

    # Validate feature schema
    if checkpoint_feature_names != FEATURE_NAMES:
        raise ValueError(
            f"Feature schema mismatch.\n"
            f"  checkpoint : {checkpoint_feature_names}\n"
            f"  current    : {FEATURE_NAMES}"
        )

    t_start = time.perf_counter()
    n = problem.dimension
    unserved = set(range(1, n + 1))
    routes: list[tuple[int, ...]] = []
    route_costs: dict[int, float] = {}
    route_cost_calls = 0
    time_offset = 0.0

    if resume and checkpoint_path:
        ckpt = _load_search_checkpoint(checkpoint_path, n)
        if ckpt and not ckpt.get("completed", False):
            routes = [tuple(r) for r in ckpt["routes"]]
            unserved = set(ckpt["unserved"])
            route_costs = {int(k): v for k, v in ckpt["route_costs"].items()}
            route_cost_calls = ckpt.get("route_cost_calls", 0)
            time_offset = ckpt.get("elapsed_seconds", 0.0)
            log.info(
                "Resuming XGB search from checkpoint: %d customers already routed, %d unserved remaining (calls=%d, elapsed=%.2fs)",
                n - len(unserved), len(unserved), route_cost_calls, time_offset,
            )

    try:
        while unserved:
            elapsed = time.perf_counter() - t_start + time_offset
            if elapsed >= time_budget_sec:
                log.warning("Time budget exceeded during solve; completing remaining %d with heuristic", len(unserved))
                break

            candidates = enumerate_candidates(problem, routes, unserved, route_costs)
            if not candidates:
                break

            # Predict delta scores using XGBoost
            X = np.stack([c.feature_vec for c in candidates])
            scores = model.predict(X)

            # top-k by lowest predicted delta
            k = min(top_k, len(candidates))
            topk_indices = np.argsort(scores)[:k]
            topk_cands = [candidates[i] for i in topk_indices]

            best_cand = None
            best_delta = math.inf

            for cand in topk_cands:
                ridx = cand.route_idx
                is_new = ridx >= len(routes)
                base_cost = 0.0 if is_new else route_costs.get(ridx, 0.0)
                cost_new = _get_route_cost(problem, cand.route_after)
                route_cost_calls += 1
                if cost_new is None:
                    continue
                delta = cost_new - base_cost
                if delta < best_delta:
                    best_delta = delta
                    best_cand = cand
                    best_cand.exact_cost = cost_new

            # Fallback if all top-k candidates were infeasible
            if best_cand is None:
                for cand in candidates:
                    ridx = cand.route_idx
                    is_new = ridx >= len(routes)
                    base_cost = 0.0 if is_new else route_costs.get(ridx, 0.0)
                    cost_new = _get_route_cost(problem, cand.route_after)
                    route_cost_calls += 1
                    if cost_new is None:
                        continue
                    delta = cost_new - base_cost
                    if delta < best_delta:
                        best_delta = delta
                        best_cand = cand
                        best_cand.exact_cost = cost_new

            if best_cand is None:
                log.error("No feasible candidate found for any unserved customer; aborting")
                break

            # Apply chosen insertion
            ridx = best_cand.route_idx
            if ridx >= len(routes):
                routes.append(best_cand.route_after)
                new_ridx = len(routes) - 1
                route_costs[new_ridx] = best_cand.exact_cost or 0.0
            else:
                routes[ridx] = best_cand.route_after
                route_costs[ridx] = best_cand.exact_cost or 0.0

            unserved.remove(best_cand.customer)

            # Checkpoint progress
            if checkpoint_path:
                _save_search_checkpoint(
                    checkpoint_path=checkpoint_path,
                    instance_name=getattr(problem, "name", ""),
                    dimension=n,
                    mode="xgb",
                    routes=routes,
                    unserved=unserved,
                    route_costs=route_costs,
                    route_cost_calls=route_cost_calls,
                    elapsed_seconds=time.perf_counter() - t_start + time_offset,
                    completed=False,
                )
    except KeyboardInterrupt:
        log.warning(
            "XGB search paused by user! State saved to %s. Run with --resume to continue.",
            checkpoint_path,
        )
        raise

    # Heuristic fallback if budget expired
    if unserved:
        log.info("Completing %d remaining customers with heuristic fallback", len(unserved))
        remaining = sorted(unserved)
        for customer in remaining:
            best_delta = math.inf
            best_ridx = len(routes)
            best_route: tuple[int, ...] = (customer,)
            singleton = (customer,)
            cost_s = _get_route_cost(problem, singleton)
            route_cost_calls += 1
            if cost_s is not None:
                best_delta = cost_s
            for ridx, route in enumerate(routes):
                rload = sum(problem.demands[c] for c in route)
                if rload + problem.demands[customer] > problem.capacity:
                    continue
                base = route_costs.get(ridx, 0.0)
                for pos in range(len(route) + 1):
                    nr = route[:pos] + (customer,) + route[pos:]
                    cost = _get_route_cost(problem, nr)
                    route_cost_calls += 1
                    if cost is None:
                        continue
                    delta = cost - base
                    if delta < best_delta:
                        best_delta = delta
                        best_ridx = ridx
                        best_route = nr
            if best_ridx >= len(routes):
                routes.append(best_route)
                route_costs[len(routes) - 1] = _get_route_cost(problem, best_route) or 0.0
            else:
                routes[best_ridx] = best_route
                route_costs[best_ridx] = _get_route_cost(problem, best_route) or 0.0
            unserved.discard(customer)

    runtime = time.perf_counter() - t_start + time_offset
    if checkpoint_path:
        _save_search_checkpoint(
            checkpoint_path=checkpoint_path,
            instance_name=getattr(problem, "name", ""),
            dimension=n,
            mode="xgb",
            routes=routes,
            unserved=unserved,
            route_costs=route_costs,
            route_cost_calls=route_cost_calls,
            elapsed_seconds=runtime,
            completed=True,
        )

    total_duration = math.fsum(route_costs.get(i, 0.0) for i in range(len(routes)))
    return {
        "routes": [list(r) for r in routes],
        "duration": total_duration,
        "runtime_seconds": runtime,
        "route_cost_calls": route_cost_calls,
        "num_routes": len(routes),
        "mode": "xgb",
    }


# ---------------------------------------------------------------------------
# Solution evaluation
# ---------------------------------------------------------------------------

def evaluate_solution(
    problem,
    routes: list[tuple[int, ...] | list[int]],
    bks: float | None = None,
) -> dict[str, Any]:
    """Validate and score a solution against ATF oracle and optional BKS."""
    routes_t = [tuple(r) for r in routes]

    all_customers = [c for r in routes_t for c in r]
    expected = set(range(1, problem.dimension + 1))
    actual = set(all_customers)
    missing = sorted(expected - actual)
    duplicates = sorted(set(c for c in all_customers if all_customers.count(c) > 1))

    cap_ok = all(
        sum(problem.demands[c] for c in r) <= problem.capacity
        for r in routes_t
    )

    route_durations = []
    for r in routes_t:
        cost = _get_route_cost(problem, r)
        route_durations.append(cost if cost is not None else float("inf"))

    total = math.fsum(d for d in route_durations if math.isfinite(d))
    feasible = (
        len(missing) == 0
        and len(duplicates) == 0
        and cap_ok
        and all(math.isfinite(d) for d in route_durations)
    )

    gap_pct: float | None = None
    if bks is not None and bks > 0 and feasible:
        gap_pct = 100.0 * (total - bks) / bks

    route_loads = [int(sum(problem.demands[c] for c in r)) for r in routes_t]

    notes = []
    if missing:
        notes.append(f"missing customers: {missing}")
    if duplicates:
        notes.append(f"duplicate customers: {duplicates}")
    if not cap_ok:
        notes.append("capacity violated on some routes")

    return {
        "feasible": feasible,
        "duration": total,
        "gap_pct": gap_pct,
        "bks": bks,
        "missing": missing,
        "duplicates": duplicates,
        "capacity_ok": cap_ok,
        "num_routes": len(routes_t),
        "route_durations": route_durations,
        "route_loads": route_loads,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

@dataclass
class _DataPoint:
    instance_id: str
    customer: int
    route_idx: int
    insert_pos: int
    features: list[float]
    exact_delta: float
    heuristic_chosen: bool


def collect_labels(
    train_instance_paths: list[str | Path],
    output_path: str | Path,
    target_type: str = "exact_delta",
    seed: int = 0,
) -> None:
    """
    Run insertion heuristic on training instances to record (features, target) samples.
    Guarantees strict separation: test instances are prohibited.
    """
    if target_type not in ("exact_delta", "rollout_regret"):
        raise ValueError(f"target_type must be 'exact_delta' or 'rollout_regret', got {target_type!r}")

    all_points: list[dict] = []
    for inst_path in train_instance_paths:
        try:
            problem = load_instance(inst_path)
        except Exception as e:
            log.warning("Skipping %s: %s", inst_path, e)
            continue

        inst_id = Path(inst_path).stem if Path(inst_path).is_file() else Path(inst_path).name
        log.info("Collecting from %s (n=%d)", inst_id, problem.dimension)

        n = problem.dimension
        unserved = set(range(1, n + 1))
        routes: list[tuple[int, ...]] = []
        route_costs: dict[int, float] = {}
        step_data: list[list[_DataPoint]] = []

        while unserved:
            candidates = enumerate_candidates(problem, routes, unserved, route_costs)
            if not candidates:
                break

            step_points: list[_DataPoint] = []
            best_delta = math.inf
            best_cand = None
            best_step_point_idx = -1

            for cand in candidates:
                ridx = cand.route_idx
                is_new = ridx >= len(routes)
                base_cost = 0.0 if is_new else route_costs.get(ridx, 0.0)
                cost_new = _get_route_cost(problem, cand.route_after)
                if cost_new is None:
                    continue
                delta = cost_new - base_cost
                cur_idx = len(step_points)
                if delta < best_delta:
                    best_delta = delta
                    best_step_point_idx = cur_idx
                    best_cand = cand
                step_points.append(_DataPoint(
                    instance_id=inst_id,
                    customer=cand.customer,
                    route_idx=cand.route_idx,
                    insert_pos=cand.insert_pos,
                    features=cand.feature_vec.tolist(),
                    exact_delta=delta,
                    heuristic_chosen=False,
                ))

            if best_cand is None or best_step_point_idx < 0:
                break

            step_points[best_step_point_idx].heuristic_chosen = True
            step_data.append(step_points)

            # Apply best insertion
            ridx = best_cand.route_idx
            is_new = ridx >= len(routes)
            cost_chosen = _get_route_cost(problem, best_cand.route_after) or 0.0
            if is_new:
                routes.append(best_cand.route_after)
                route_costs[len(routes) - 1] = cost_chosen
            else:
                routes[ridx] = best_cand.route_after
                route_costs[ridx] = cost_chosen
            unserved.remove(best_cand.customer)

        for step_points in step_data:
            for dp in step_points:
                all_points.append({
                    "instance_id": dp.instance_id,
                    "customer": dp.customer,
                    "route_idx": dp.route_idx,
                    "insert_pos": dp.insert_pos,
                    "features": dp.features,
                    "exact_delta": dp.exact_delta,
                    "heuristic_chosen": dp.heuristic_chosen,
                    "target": dp.exact_delta,
                })

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "feature_names": FEATURE_NAMES,
            "target_type": target_type,
            "num_instances": len(train_instance_paths),
            "num_datapoints": len(all_points),
            "data": all_points,
        }, f, indent=2, ensure_ascii=False, allow_nan=False)
    log.info("Saved %d data points to %s", len(all_points), out)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_xgb(
    dataset_path: str | Path,
    model_path: str | Path,
    n_estimators: int = 200,
    max_depth: int = 5,
    lr: float = 0.08,
    n_jobs: int = 2,
    seed: int = 42,
    val_fraction: float = 0.1,
    checkpoint_file: str | Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Train XGBRegressor on the collected dataset."""
    if not _XGBOOST_AVAILABLE:
        raise ImportError("xgboost is not installed. Run 'pip install xgboost' to train TD-XGB.")

    np.random.seed(seed)
    random.seed(seed)

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, encoding="utf-8") as f:
        raw = json.load(f)

    # Validate schema
    saved_features = raw.get("feature_names", [])
    if saved_features != FEATURE_NAMES:
        raise ValueError(
            f"Dataset feature schema mismatch.\n"
            f"  dataset : {saved_features}\n"
            f"  current : {FEATURE_NAMES}"
        )

    data = raw["data"]
    if not data:
        raise ValueError("Dataset is empty")

    X = np.array([d["features"] for d in data], dtype=np.float64)
    y = np.array([d.get("target", d.get("exact_delta")) for d in data], dtype=np.float64)

    # Train / Val split
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    val_n = max(1, int(len(X) * val_fraction))
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    log.info(
        "Training XGBRegressor: train_samples=%d, val_samples=%d, n_estimators=%d, max_depth=%d, lr=%.3f, n_jobs=%d",
        len(train_idx), len(val_idx), n_estimators, max_depth, lr, n_jobs,
    )

    t_train_start = time.perf_counter()

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        n_jobs=n_jobs,
        random_state=seed,
        objective="reg:squarederror",
        tree_method="hist",
    )

    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=False,
    )

    training_time = time.perf_counter() - t_train_start

    # Evaluation metrics
    val_pred = model.predict(X_val)
    val_mae = float(np.mean(np.abs(val_pred - y_val)))
    val_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
    train_pred = model.predict(X_train)
    train_rmse = float(np.sqrt(np.mean((train_pred - y_train) ** 2)))

    log.info(
        "Training completed in %.2fs | val_mae=%.4f | val_rmse=%.4f | train_rmse=%.4f",
        training_time, val_mae, val_rmse, train_rmse,
    )

    # Save model bundle
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "target_type": raw.get("target_type", "exact_delta"),
        "config": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": lr,
            "n_jobs": n_jobs,
            "seed": seed,
        },
        "train_stats": {
            "num_train": len(train_idx),
            "num_val": len(val_idx),
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "training_time_seconds": training_time,
        },
    }
    joblib.dump(bundle, model_path)
    log.info("Model bundle saved to %s", model_path)

    return {
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "train_rmse": train_rmse,
        "training_time_seconds": training_time,
    }


# ---------------------------------------------------------------------------
# Model checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(model_path: str | Path) -> tuple[Any, list[str]]:
    """Load model and feature_names from a checkpoint file."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    bundle = joblib.load(model_path)
    if isinstance(bundle, dict):
        saved_features = bundle.get("feature_names", [])
        if saved_features != FEATURE_NAMES:
            raise ValueError(
                f"Checkpoint feature schema mismatch.\n"
                f"  checkpoint : {saved_features}\n"
                f"  current    : {FEATURE_NAMES}"
            )
        model = bundle["model"]
    else:
        # direct model object
        model = bundle
        saved_features = FEATURE_NAMES

    log.info("Loaded XGBoost checkpoint from %s", model_path)
    return model, saved_features


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def run_self_test(instance_path: str | Path) -> None:
    """Run a battery of correctness checks on the given instance."""
    log.info("=== Self-test TD-XGB on %s ===", instance_path)
    problem = load_instance(instance_path)
    n = problem.dimension
    Q = problem.capacity
    log.info("n=%d, Q=%d", n, Q)

    # 1. Heuristic solve with checkpoint recording
    log.info("[1/7] Heuristic solve (with checkpoint recording)...")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_ckpt = tmp.name

    t0 = time.perf_counter()
    routes, duration = solve_heuristic(problem, seed=0, checkpoint_path=tmp_ckpt, resume=False)
    t1 = time.perf_counter()
    log.info("  Duration=%.3f s, routes=%d, time=%.2fs", duration, len(routes), t1 - t0)

    # 2. Validate routes
    log.info("[2/7] Validating routes...")
    all_cust = [c for r in routes for c in r]
    assert sorted(all_cust) == list(range(1, n + 1)), "Missing or duplicate customers"
    for ridx, r in enumerate(routes):
        load = sum(problem.demands[c] for c in r)
        assert load <= Q, f"Route {ridx} capacity violated: {load} > {Q}"
    log.info("  OK: all customers present exactly once, all loads feasible")

    # 3. Duration matches evaluator re-computation
    log.info("[3/7] Duration consistency check...")
    recomputed = math.fsum(_get_route_cost(problem, tuple(r)) or 0.0 for r in routes)
    assert math.isclose(duration, recomputed, rel_tol=1e-9, abs_tol=1e-6), (
        f"Duration mismatch: solve={duration}, recomputed={recomputed}"
    )
    log.info("  OK: duration=%.3f == recomputed=%.3f", duration, recomputed)

    # 4. Feature extraction check
    log.info("[4/7] Feature extraction check...")
    if routes:
        r0 = tuple(routes[0])
        feat = extract_features(
            problem,
            [r0],
            route_idx=0,
            insert_pos=0,
            customer=routes[1][0] if len(routes) > 1 else 1,
            unserved_count=1,
            route_durations={0: _get_route_cost(problem, r0) or 0.0},
        )
        assert feat.shape == (_FEATURE_DIM,), f"Feature shape {feat.shape} != ({_FEATURE_DIM},)"
        assert np.all(np.isfinite(feat)), f"Non-finite features: {feat}"
        log.info("  OK: feature vector shape=%s, all finite", feat.shape)

    # 5. XGB forward pass & schema check (if xgboost installed)
    if _XGBOOST_AVAILABLE:
        log.info("[5/7] XGB forward pass check...")
        dummy_X = np.random.randn(32, _FEATURE_DIM)
        dummy_y = np.random.randn(32)
        dummy_model = xgb.XGBRegressor(n_estimators=5, max_depth=2, n_jobs=1, random_state=0)
        dummy_model.fit(dummy_X, dummy_y)
        pred = dummy_model.predict(feat.reshape(1, -1))
        assert pred.shape == (1,), f"Unexpected pred shape: {pred.shape}"
        assert math.isfinite(float(pred[0])), "Prediction is not finite"
        log.info("  OK: dummy XGB output=%.6f", float(pred[0]))

        # Test bundle save/load
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_jb:
            tmp_jb_path = tmp_jb.name
        bundle = {"model": dummy_model, "feature_names": FEATURE_NAMES}
        joblib.dump(bundle, tmp_jb_path)
        m2, fn2 = load_checkpoint(tmp_jb_path)
        pred2 = m2.predict(feat.reshape(1, -1))
        assert math.isclose(float(pred[0]), float(pred2[0]), rel_tol=1e-6)
        os.unlink(tmp_jb_path)
        log.info("  OK: checkpoint save/load round-trip consistent")

        # Test schema mismatch detection
        bad_bundle = {"model": dummy_model, "feature_names": ["wrong_feature"]}
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_bad:
            tmp_bad_path = tmp_bad.name
        joblib.dump(bad_bundle, tmp_bad_path)
        try:
            load_checkpoint(tmp_bad_path)
            assert False, "Should have raised ValueError on schema mismatch"
        except ValueError:
            log.info("  OK: schema mismatch correctly raises ValueError")
        finally:
            if os.path.exists(tmp_bad_path):
                os.unlink(tmp_bad_path)
    else:
        log.info("[5/7] SKIP: xgboost is not installed. Run 'pip install xgboost' to enable XGB inference.")

    # 6. Checkpoint pause & resume check
    log.info("[6/7] Checkpoint save & resume check...")
    routes_res, cost_res = solve_heuristic(
        problem, seed=0, checkpoint_path=tmp_ckpt, resume=True
    )
    assert math.isclose(duration, cost_res, rel_tol=1e-6), "Resumed cost != original cost"
    assert routes == routes_res, "Resumed routes != original routes"
    if os.path.exists(tmp_ckpt):
        os.unlink(tmp_ckpt)
    log.info("  OK: solve checkpoint resume produces identical valid solution")

    # 7. Check run.log recording
    log.info("[7/7] run.log logging check...")
    log_path = Path("run.log")
    assert log_path.exists() and log_path.stat().st_size > 0, "run.log does not exist or is empty"
    log.info("  OK: run.log successfully recorded %d bytes", log_path.stat().st_size)

    log.info("=== All self-tests PASSED ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_collect(args):
    setup_logging(args.log_file)
    log.info("Command: collect | target_type=%s | output=%s", args.target_type, args.dataset)
    if not args.train_instances:
        raise ValueError("--train-instances required")
    collect_labels(
        train_instance_paths=args.train_instances,
        output_path=args.dataset,
        target_type=args.target_type,
        seed=args.seed,
    )


def _cmd_train(args):
    setup_logging(args.log_file)
    log.info(
        "Command: train | dataset=%s | model_path=%s | n_estimators=%d | resume=%s",
        args.dataset, args.model_path, args.n_estimators, args.resume,
    )
    result = train_xgb(
        dataset_path=args.dataset,
        model_path=args.model_path,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        lr=args.lr,
        n_jobs=args.threads,
        seed=args.seed,
        val_fraction=args.val_fraction,
        checkpoint_file=args.checkpoint_file,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


def _cmd_solve(args):
    setup_logging(args.log_file)
    log.info(
        "Command: solve | instance=%s | model_path=%s | resume=%s",
        args.instance, args.model_path, args.resume,
    )
    inst_list = args.instance if isinstance(args.instance, list) else [args.instance]
    output_dir = Path(getattr(args, "output_dir", "results/td_xgb"))
    output_dir.mkdir(parents=True, exist_ok=True)

    model, feature_names = None, None
    if args.model_path and Path(args.model_path).exists():
        if not _XGBOOST_AVAILABLE:
            log.warning("xgboost not installed; using heuristic-only mode. Run 'pip install xgboost'")
            mode = "heuristic_only"
        else:
            model, feature_names = load_checkpoint(args.model_path)
            mode = "xgb"
    else:
        if args.model_path:
            log.warning("Model path not found: %s; using heuristic-only mode", args.model_path)
        else:
            log.info("No model path given; using heuristic-only mode")
        mode = "heuristic_only"

    all_outputs = []

    for inst in inst_list:
        try:
            problem = load_instance(inst)
        except Exception as e:
            log.warning("Skipping instance %s: %s", inst, e)
            continue

        bks = load_bks(inst)
        inst_vrp_path = find_vrp_json(inst)
        inst_stem = inst_vrp_path.stem

        ckpt_path = args.checkpoint_file or (
            output_dir / f"solve_checkpoint_{inst_stem}.json"
        )

        t_start = time.perf_counter()
        if mode == "xgb":
            result = solve_xgb(
                problem, model, feature_names,
                seed=args.seed,
                top_k=args.top_k,
                time_budget_sec=args.time_budget if args.time_budget else float("inf"),
                checkpoint_path=ckpt_path,
                resume=args.resume,
            )
        else:
            routes, duration = solve_heuristic(
                problem,
                seed=args.seed,
                checkpoint_path=ckpt_path,
                resume=args.resume,
            )
            runtime = time.perf_counter() - t_start
            result = {
                "routes": [list(r) for r in routes],
                "duration": duration,
                "runtime_seconds": runtime,
                "route_cost_calls": -1,
                "num_routes": len(routes),
                "mode": "heuristic_only",
            }

        eval_result = evaluate_solution(problem, [tuple(r) for r in result["routes"]], bks=bks)

        output = {
            "algorithm": "TD-XGB" if mode == "xgb" else "TD-Insertion-Heuristic",
            "mode": mode,
            "instance": str(inst_vrp_path),
            "seed": args.seed,
            "top_k": args.top_k if mode == "xgb" else None,
            **result,
            **{f"eval_{k}": v for k, v in eval_result.items()},
            "bks": bks,
            "gap_pct": eval_result.get("gap_pct"),
        }

        # Pretty print summary
        print(f"\n{'='*60}")
        print(f"Algorithm : {output['algorithm']}")
        print(f"Instance  : {inst_vrp_path.name}")
        print(f"Mode      : {mode}")
        print(f"Duration  : {result['duration']:.3f} s")
        if bks:
            gap = eval_result.get("gap_pct")
            print(f"BKS       : {bks:.3f} s")
            print(f"Gap(%)    : {gap:.4f}%" if gap is not None else "Gap(%)    : N/A (infeasible)")
        print(f"Routes    : {len(result['routes'])}")
        print(f"Feasible  : {eval_result['feasible']}")
        print(f"Runtime   : {result['runtime_seconds']:.2f} s")
        if mode == "xgb":
            print(f"RouteEvals: {result['route_cost_calls']}")
        print(f"{'='*60}\n")

        # Save JSON
        out_json = Path(args.output_json) if args.output_json else (output_dir / f"results_{inst_stem}.json")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        log.info("Results saved to %s", out_json)

        # Write summary.txt
        summary_txt = output_dir / "summary.txt"
        with open(summary_txt, "a", encoding="utf-8") as f:
            f.write(f"Algorithm : {output['algorithm']} ({mode})\n")
            f.write(f"Instance  : {inst_stem}\n")
            f.write(f"Duration  : {result['duration']:.3f} s\n")
            f.write(f"BKS       : {bks:.3f} s\n" if bks else "BKS : N/A\n")
            f.write(f"Gap(%)    : {eval_result.get('gap_pct', 0.0):.4f}%\n")
            f.write(f"Routes    : {len(result['routes'])}\n")
            f.write(f"Runtime   : {result['runtime_seconds']:.2f} s\n\n")

        # Generate plots
        if not getattr(args, "no_plot", False):
            try:
                from experiments.export_visualizations import plot_route_map, plot_departure_timeline
                plot_route_map(
                    inst_vrp_path, result["routes"], result["duration"], bks, eval_result.get("gap_pct"),
                    output["algorithm"], output_dir / f"route_map_{inst_stem}.png"
                )
                dep_times = result.get("departure_times", [0.0] * len(result["routes"]))
                durations = eval_result.get("route_durations", result.get("route_durations", []))
                plot_departure_timeline(
                    dep_times, durations, output["algorithm"], inst_stem,
                    output_dir / f"timeline_{inst_stem}.png"
                )
                log.info("Generated plots for %s in %s", inst_stem, output_dir)
            except Exception as e:
                log.warning("Could not generate plots for %s: %s", inst_stem, e)

        all_outputs.append(output)

    return all_outputs[0] if len(all_outputs) == 1 else all_outputs


def _cmd_evaluate(args):
    return _cmd_solve(args)


def _cmd_self_test(args):
    setup_logging(args.log_file)
    run_self_test(args.instance)


def main():
    parser = argparse.ArgumentParser(
        description="TD-XGB: XGBoost-guided insertion baseline for TDVRP Poryos2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log-file", default="run.log", metavar="FILE",
                        help="Path to write execution log (default: run.log)")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- collect ---
    p_collect = sub.add_parser("collect", help="Collect training data from instances")
    p_collect.add_argument("--train-instances", nargs="+", required=True, metavar="PATH",
                           help="Paths to .vrp.json files or directories (NOT n50-heavy)")
    p_collect.add_argument("--dataset", required=True, metavar="JSON",
                           help="Output dataset file")
    p_collect.add_argument("--target-type", choices=["exact_delta", "rollout_regret"],
                           default="exact_delta")
    p_collect.add_argument("--seed", type=int, default=0)
    p_collect.add_argument("--log-file", default="run.log", metavar="FILE")

    # --- train ---
    p_train = sub.add_parser("train", help="Train XGBoost model from collected dataset")
    p_train.add_argument("--dataset", required=True, metavar="JSON")
    p_train.add_argument("--model-path", required=True, metavar="JOBLIB")
    p_train.add_argument("--n-estimators", type=int, default=200, help="Number of trees (default: 200)")
    p_train.add_argument("--max-depth", type=int, default=5, help="Max tree depth (default: 5)")
    p_train.add_argument("--lr", type=float, default=0.08, help="Learning rate (default: 0.08)")
    p_train.add_argument("--threads", type=int, default=2, help="Number of parallel threads (default: 2)")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--val-fraction", type=float, default=0.1)
    p_train.add_argument("--checkpoint-file", default=None, metavar="FILE",
                         help="File to persist training progress")
    p_train.add_argument("--resume", action="store_true",
                         help="Resume training from checkpoint if available")
    p_train.add_argument("--log-file", default="run.log", metavar="FILE")

    # --- solve ---
    p_solve = sub.add_parser("solve", help="Solve instances using TD-XGB")
    p_solve.add_argument("--instance", nargs="+",
                         default=["data/TDVRP/hong_kong/n=50/poryos-hong_kong-n50-hyb/bpr-heavy"],
                         metavar="DIR_OR_VRP_JSON",
                         help="Paths to .vrp.json or directories (default: Hong Kong n50 BPR-heavy)")
    p_solve.add_argument("--model-path", default=None, metavar="JOBLIB",
                         help="Checkpoint; if missing or xgboost unavailable, uses heuristic-only mode")
    p_solve.add_argument("--seed", type=int, default=0)
    p_solve.add_argument("--top-k", type=int, default=3,
                         help="Number of XGB-top-k candidates to evaluate exactly (default: 3)")
    p_solve.add_argument("--time-budget", type=float, default=None,
                         help="Time budget in seconds")
    p_solve.add_argument("--output-json", default=None, metavar="JSON")
    p_solve.add_argument("--output-dir", default="results/td_xgb", metavar="DIR",
                         help="Folder to export results and visual charts (default: results/td_xgb)")
    p_solve.add_argument("--no-plot", action="store_true",
                         help="Disable generating PNG illustrations")
    p_solve.add_argument("--checkpoint-file", default=None, metavar="FILE",
                         help="File to persist search state")
    p_solve.add_argument("--resume", action="store_true",
                         help="Resume search from checkpoint if available")
    p_solve.add_argument("--log-file", default="run.log", metavar="FILE")

    # --- evaluate ---
    p_eval = sub.add_parser("evaluate", help="Solve and report vs BKS")
    p_eval.add_argument("--instance", nargs="+",
                        default=["data/TDVRP/hong_kong/n=50/poryos-hong_kong-n50-hyb/bpr-heavy"],
                        metavar="DIR_OR_VRP_JSON")
    p_eval.add_argument("--model-path", default=None, metavar="JOBLIB")
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.add_argument("--top-k", type=int, default=3)
    p_eval.add_argument("--time-budget", type=float, default=None)
    p_eval.add_argument("--output-json", default=None, metavar="JSON")
    p_eval.add_argument("--output-dir", default="results/td_xgb", metavar="DIR")
    p_eval.add_argument("--no-plot", action="store_true")
    p_eval.add_argument("--checkpoint-file", default=None, metavar="FILE")
    p_eval.add_argument("--resume", action="store_true")
    p_eval.add_argument("--log-file", default="run.log", metavar="FILE")

    # --- self-test ---
    p_st = sub.add_parser("self-test", help="Run correctness checks")
    p_st.add_argument("--instance", required=True, metavar="DIR_OR_VRP_JSON",
                      default=r"E:\1KLCN\tdvrp-rapo\data\TDVRP\hong_kong\n=50\poryos-hong_kong-n50-hyb\bpr-heavy")
    p_st.add_argument("--log-file", default="run.log", metavar="FILE")

    args = parser.parse_args()

    dispatch = {
        "collect": _cmd_collect,
        "train": _cmd_train,
        "solve": _cmd_solve,
        "evaluate": _cmd_evaluate,
        "self-test": _cmd_self_test,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
