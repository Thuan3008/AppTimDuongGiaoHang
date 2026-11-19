"""TD-MLP: MLP-guided greedy insertion baseline for TDVRP Poryos2026.

Standalone supervised-learning baseline. No shared imports with td_xgb.py.
Requires torch (CPU or CUDA). Uses exact ATF route_cost oracle (TDVRPProblem).

CLI usage:
  python algorithms/td_mlp.py collect --train-instances PATHS... --dataset out.json
  python algorithms/td_mlp.py train   --dataset data.json --model-path model.pt
  python algorithms/td_mlp.py solve   --instance DIR_OR_VRP_JSON --model-path model.pt
  python algorithms/td_mlp.py evaluate --instance DIR_OR_VRP_JSON --model-path model.pt
  python algorithms/td_mlp.py self-test --instance DIR_OR_VRP_JSON
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
# Optional torch import – graceful fallback to heuristic-only mode
# ---------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore", message=".*Failed to initialize NumPy.*")
warnings.filterwarnings("ignore", message=".*A module that was compiled using NumPy 1.x.*")

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Feature schema
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
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("td_mlp")


def setup_logging(log_file: str | Path | None = "run.log") -> None:
    """Configure logging to stream to stdout and append to log_file."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(p, mode="a", encoding="utf-8"))
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s td_mlp] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=handlers,
        force=True,
    )


# Initial logging setup to run.log
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
# MLP model
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:
    class InsertionMLP(nn.Module):
        """Linear(D,128) → ReLU → Linear(128,64) → ReLU → Linear(64,1)."""

        def __init__(self, in_features: int = _FEATURE_DIM, dropout: float = 0.0):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(64, 1),
            )

        def forward(self, x):  # type: ignore[override]
            return self.net(x).squeeze(-1)

else:
    class InsertionMLP:  # type: ignore[no-redef]
        """Placeholder when torch is not available."""
        def __init__(self, *a, **kw):
            raise ImportError("torch is required to use InsertionMLP")


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------

class StandardScaler:
    """Fit on train set only; clamp std > eps to avoid divide-by-zero."""

    _EPS = 1e-8

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = np.where(X.std(axis=0) < self._EPS, 1.0, X.std(axis=0))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Scaler not fitted")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "std": self.std_.tolist() if self.std_ is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StandardScaler":
        sc = cls()
        sc.mean_ = np.array(d["mean"], dtype=np.float64) if d.get("mean") else None
        sc.std_ = np.array(d["std"], dtype=np.float64) if d.get("std") else None
        return sc


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
    """Load TDVRPProblem from a .vrp.json file or directory.

    collection_root is set to data/TDVRP (where mamut-collection.json lives).
    """
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from core.tdvrp_common import TDVRPProblem  # noqa: PLC0415
    vrp_path = find_vrp_json(path_or_dir)
    collection_root = root / "data" / "TDVRP"
    log.info("Loading instance: %s", vrp_path.name)
    return TDVRPProblem(vrp_path, collection_root=collection_root)


def load_bks(path_or_dir: str | Path) -> float | None:
    """Try to read BKS Duration value from a .bks.Duration.json file."""
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
    # Try .txt
    for bks_txt in sorted(p.glob("*.bks.Duration.txt")):
        try:
            val = float(bks_txt.read_text(encoding="utf-8").strip())
            return val
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
    Build the feature vector for inserting `customer` at position `insert_pos`
    in routes[route_idx].

    Parameters
    ----------
    problem       : TDVRPProblem
    routes        : current routes (customer-only tuples, without depot)
    route_idx     : index of the target route (-1 = new empty route)
    insert_pos    : position within the route where customer is inserted
    customer      : customer ID (1-indexed)
    unserved_count: number of customers not yet routed
    route_durations: dict mapping route_idx → current route duration
    """
    n = problem.dimension
    Q = problem.capacity

    demand_c = problem.demands[customer]
    service_c = problem.services[customer]

    if route_idx < 0 or route_idx >= len(routes):
        # New route
        route = ()
        route_load = 0.0
        cur_duration = 0.0
    else:
        route = routes[route_idx]
        route_load = sum(problem.demands[c] for c in route)
        cur_duration = route_durations.get(route_idx, 0.0)

    route_len = len(route)
    num_routes = len(routes)

    # predecessor / successor in route after insertion
    if insert_pos == 0:
        pred = 0  # depot
    else:
        pred = route[insert_pos - 1]

    if insert_pos >= route_len:
        succ = 0  # depot
    else:
        succ = route[insert_pos]

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
    route_idx: int      # -1 → new route
    insert_pos: int
    customer: int
    route_after: tuple[int, ...]
    feature_vec: np.ndarray
    exact_cost: float | None = None   # filled lazily
    base_route_cost: float | None = None


def enumerate_candidates(
    problem,
    routes: list[tuple[int, ...]],
    unserved: set[int],
    route_durations: dict[int, float],
) -> list[_Candidate]:
    """Return all feasible (customer, route, position) triples + features."""
    n = problem.dimension
    Q = problem.capacity
    candidates: list[_Candidate] = []
    unserved_count = len(unserved)

    for customer in sorted(unserved):
        # Existing routes
        for ridx, route in enumerate(routes):
            route_load = sum(problem.demands[c] for c in route)
            if route_load + problem.demands[customer] > Q:
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
        if problem.demands[customer] <= Q:
            feat = extract_features(
                problem, routes, -1, 0, customer, unserved_count, route_durations
            )
            candidates.append(_Candidate(
                route_idx=len(routes),  # will become new route index
                insert_pos=0,
                customer=customer,
                route_after=(customer,),
                feature_vec=feat,
            ))

    return candidates


# ---------------------------------------------------------------------------
# Route cost helper (cached by problem.route_cost)
# ---------------------------------------------------------------------------

def _get_route_cost(problem, route: tuple[int, ...]) -> float | None:
    """Return route duration or None if infeasible."""
    if not route:
        return 0.0
    try:
        result = problem.route_cost(route)
        return float(result[0])
    except (ValueError, Exception):
        return None


# ---------------------------------------------------------------------------
# Heuristic solver (cheapest exact insertion, no ML)
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

            # Checkpoint after insertion
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
# MLP solver
# ---------------------------------------------------------------------------

def solve_mlp(
    problem,
    model: "InsertionMLP",
    scaler: StandardScaler,
    checkpoint_feature_names: list[str],
    seed: int = 0,
    top_k: int = 1,
    time_budget_sec: float = float("inf"),
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """
    MLP-guided greedy insertion with checkpoint save and resume support.

    At each step:
      1. Enumerate all feasible (customer, route, position) candidates.
      2. Score each with the MLP (lower = better).
      3. Among top-k lowest-scoring, pick the one with best exact cost delta.
      4. Insert and repeat.
      5. Persist progress to checkpoint_path if provided.

    Returns a result dict.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for solve_mlp")

    # Validate feature schema
    if checkpoint_feature_names != FEATURE_NAMES:
        raise ValueError(
            f"Feature schema mismatch.\n"
            f"  checkpoint : {checkpoint_feature_names}\n"
            f"  current    : {FEATURE_NAMES}"
        )

    t_start = time.perf_counter()
    model.eval()
    device = next(model.parameters()).device

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
                "Resuming MLP solve from checkpoint: %d customers already routed, %d unserved remaining (calls=%d, elapsed=%.2fs)",
                n - len(unserved), len(unserved), route_cost_calls, time_offset,
            )

    try:
        while unserved:
            elapsed = time.perf_counter() - t_start + time_offset
            if elapsed >= time_budget_sec:
                log.warning("Time budget exceeded during solve; falling back to heuristic for remaining %d customers", len(unserved))
                break

            candidates = enumerate_candidates(problem, routes, unserved, route_costs)
            if not candidates:
                break

            # Score with MLP
            X_raw = np.stack([c.feature_vec for c in candidates])
            X_scaled = scaler.transform(X_raw).astype(np.float32)
            with torch.no_grad():
                t_in = torch.tensor(X_scaled.tolist(), dtype=torch.float32, device=device)
                scores = np.array(model(t_in).detach().cpu().tolist(), dtype=np.float64)

            # top-k by lowest score
            k = min(top_k, len(candidates))
            topk_indices = np.argsort(scores)[:k]
            topk_cands = [candidates[i] for i in topk_indices]

            # exact cost evaluation for top-k
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
                    best_cand.base_route_cost = base_cost

            if best_cand is None:
                # All top-k infeasible; fall back to cheapest exact over all candidates
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

            # Apply insertion
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
                    mode="mlp",
                    routes=routes,
                    unserved=unserved,
                    route_costs=route_costs,
                    route_cost_calls=route_cost_calls,
                    elapsed_seconds=time.perf_counter() - t_start + time_offset,
                    completed=False,
                )
    except KeyboardInterrupt:
        log.warning(
            "MLP solve paused by user! State saved to %s. Run with --resume to continue.",
            checkpoint_path,
        )
        raise

    # If budget expired, use heuristic for remaining
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
            mode="mlp",
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
        "mode": "mlp",
    }


# ---------------------------------------------------------------------------
# Solution evaluation
# ---------------------------------------------------------------------------

def evaluate_solution(
    problem,
    routes: list[tuple[int, ...] | list[int]],
    bks: float | None = None,
) -> dict[str, Any]:
    """
    Validate and score a solution.

    Returns dict with keys: feasible, duration, gap_pct, missing, duplicate,
    capacity_ok, num_routes, route_durations, route_loads, notes.
    """
    routes_t = [tuple(r) for r in routes]

    # Count customers
    all_customers = [c for r in routes_t for c in r]
    expected = set(range(1, problem.dimension + 1))
    actual = set(all_customers)
    missing = sorted(expected - actual)
    duplicates = [c for c in all_customers if all_customers.count(c) > 1]
    duplicates = sorted(set(duplicates))

    # Capacity
    cap_ok = all(
        sum(problem.demands[c] for c in r) <= problem.capacity
        for r in routes_t
    )

    # Route durations
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

    route_loads = [
        int(sum(problem.demands[c] for c in r)) for r in routes_t
    ]

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
    exact_delta: float          # exact cost delta when inserting customer here
    heuristic_chosen: bool      # True if this was the choice the heuristic made
    rollout_regret: float = 0.0  # filled post-hoc


def collect_labels(
    train_instance_paths: list[str | Path],
    output_path: str | Path,
    target_type: str = "exact_delta",
    seed: int = 0,
) -> None:
    """
    For each training instance, run the insertion heuristic and record
    feature vectors + cost labels for every candidate at every step.

    Writes JSON dataset to output_path.
    target_type: "exact_delta" or "rollout_regret"
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

        rng = random.Random(seed)
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

        # Flatten into dataset
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
                    "target": dp.exact_delta,  # default target
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

def train_mlp(
    dataset_path: str | Path,
    model_path: str | Path,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    device_str: str = "cpu",
    seed: int = 42,
    val_fraction: float = 0.1,
    dropout: float = 0.0,
    checkpoint_file: str | Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Train the InsertionMLP on a collected dataset, with checkpoint and resume support."""
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for training")

    torch.manual_seed(seed)
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
    y = np.array([d["target"] for d in data], dtype=np.float64)

    # Train/val split (by shuffled index)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    val_n = max(1, int(len(X) * val_fraction))
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    y_train = y[train_idx]
    y_val = y[val_idx]

    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA not available; using CPU")
        device = torch.device("cpu")

    model = InsertionMLP(in_features=_FEATURE_DIM, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Normalize targets on train set
    y_mean = float(y_train.mean())
    y_std = float(y_train.std()) or 1.0
    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std

    X_tr_t = torch.tensor(X_train.tolist(), dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_train_norm.tolist(), dtype=torch.float32, device=device)
    X_va_t = torch.tensor(X_val.tolist(), dtype=torch.float32, device=device)
    y_va_t = torch.tensor(y_val_norm.tolist(), dtype=torch.float32, device=device)

    ckpt_file = Path(checkpoint_file) if checkpoint_file else Path(model_path).with_suffix(".train_ckpt.pt")
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_val_loss = math.inf
    best_state = None
    history = []
    time_offset = 0.0

    if resume and ckpt_file.exists():
        try:
            log.info("Loading training checkpoint from %s...", ckpt_file)
            ckpt = torch.load(ckpt_file, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_loss = ckpt.get("best_val_loss", math.inf)
            history = ckpt.get("history", [])
            time_offset = ckpt.get("training_time_seconds", 0.0)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            log.info("Resuming training from epoch %d/%d (best_val_loss=%.6f)", start_epoch, epochs, best_val_loss)
        except Exception as e:
            log.warning("Could not resume from training checkpoint %s: %s; starting fresh", ckpt_file, e)

    t_train_start = time.perf_counter()
    dataset_size = X_tr_t.shape[0]

    try:
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            perm = torch.randperm(dataset_size, device=device)
            train_loss = 0.0
            n_batches = 0
            for start in range(0, dataset_size, batch_size):
                idx_b = perm[start:start + batch_size]
                xb = X_tr_t[idx_b]
                yb = y_tr_t[idx_b]
                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1
            train_loss /= max(1, n_batches)

            model.eval()
            with torch.no_grad():
                val_pred = model(X_va_t)
                val_loss = loss_fn(val_pred, y_va_t).item()

            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if epoch % 10 == 0 or epoch == start_epoch or epoch == epochs:
                log.info("Epoch %3d | train_loss=%.6f | val_loss=%.6f", epoch, train_loss, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            # Save progress checkpoint each epoch
            total_elapsed = time.perf_counter() - t_train_start + time_offset
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "history": history,
                "training_time_seconds": total_elapsed,
            }, ckpt_file)
    except KeyboardInterrupt:
        log.warning("Training interrupted at epoch %d. State saved to %s. Run with --resume to continue.", epoch, ckpt_file)
        raise

    if best_state is not None:
        model.load_state_dict(best_state)

    training_time = time.perf_counter() - t_train_start + time_offset

    # Save final model checkpoint
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "scaler": scaler.to_dict(),
        "feature_names": FEATURE_NAMES,
        "config": {
            "in_features": _FEATURE_DIM,
            "dropout": dropout,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "y_mean": y_mean,
            "y_std": y_std,
        },
        "train_stats": {
            "num_train": len(train_idx),
            "num_val": len(val_idx),
            "best_val_loss": best_val_loss,
            "training_time_seconds": training_time,
            "history": history,
        },
    }
    torch.save(ckpt, model_path)
    log.info(
        "Model saved to %s | best_val_loss=%.6f | total_time=%.1fs",
        model_path, best_val_loss, training_time,
    )
    return {"best_val_loss": best_val_loss, "training_time_seconds": training_time, "history": history}


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(model_path: str | Path) -> tuple["InsertionMLP", StandardScaler, list[str]]:
    """Load model, scaler, and feature_names from a checkpoint file."""
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required to load checkpoints")
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    ckpt = torch.load(model_path, map_location="cpu")

    # Schema check
    saved_features = ckpt.get("feature_names", [])
    if saved_features != FEATURE_NAMES:
        raise ValueError(
            f"Checkpoint feature schema mismatch.\n"
            f"  checkpoint : {saved_features}\n"
            f"  current    : {FEATURE_NAMES}"
        )

    config = ckpt.get("config", {})
    in_features = config.get("in_features", _FEATURE_DIM)
    dropout = config.get("dropout", 0.0)

    model = InsertionMLP(in_features=in_features, dropout=dropout)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler = StandardScaler.from_dict(ckpt["scaler"])

    log.info("Loaded checkpoint from %s", model_path)
    return model, scaler, saved_features


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def run_self_test(instance_path: str | Path) -> None:
    """Run a battery of correctness checks on the given instance."""
    log.info("=== Self-test on %s ===", instance_path)
    problem = load_instance(instance_path)
    n = problem.dimension
    Q = problem.capacity
    log.info("n=%d, Q=%d", n, Q)

    # 1. Heuristic solve
    log.info("[1/7] Heuristic solve (with checkpoint recording)...")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_ckpt = tmp.name
    t0 = time.perf_counter()
    routes, duration = solve_heuristic(problem, seed=0, checkpoint_path=tmp_ckpt, resume=False)
    t1 = time.perf_counter()
    log.info("  Duration=%.3f s, routes=%d, time=%.2fs", duration, len(routes), t1 - t0)

    # 2. Validate routes
    log.info("[2/5] Validating routes...")
    all_cust = [c for r in routes for c in r]
    assert sorted(all_cust) == list(range(1, n + 1)), "Missing or duplicate customers"
    for ridx, r in enumerate(routes):
        load = sum(problem.demands[c] for c in r)
        assert load <= Q, f"Route {ridx} capacity violated: {load} > {Q}"
    log.info("  OK: all customers present exactly once, all loads feasible")

    # 3. Duration matches evaluator re-computation
    log.info("[3/5] Duration consistency check...")
    recomputed = math.fsum(
        _get_route_cost(problem, tuple(r)) or 0.0
        for r in routes
    )
    assert math.isclose(duration, recomputed, rel_tol=1e-9, abs_tol=1e-6), (
        f"Duration mismatch: solve={duration}, recomputed={recomputed}"
    )
    log.info("  OK: duration=%.3f == recomputed=%.3f", duration, recomputed)

    # 4. Feature extraction check
    log.info("[4/5] Feature extraction check...")
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

    # 5. MLP forward pass (if torch available)
    if _TORCH_AVAILABLE:
        log.info("[5/5] MLP forward pass check...")
        model = InsertionMLP(in_features=_FEATURE_DIM)
        model.eval()
        sc = StandardScaler()
        # fit scaler on dummy data
        dummy_X = np.random.randn(16, _FEATURE_DIM)
        sc.fit(dummy_X)
        feat_scaled = sc.transform(feat.reshape(1, -1)).astype(np.float32)
        with torch.no_grad():
            out = model(torch.tensor(feat_scaled.tolist(), dtype=torch.float32))
        assert out.shape == (1,) or out.shape == (), f"Unexpected output shape: {out.shape}"
        val_0 = float(out.flatten().tolist()[0])
        assert math.isfinite(val_0), "MLP output is not finite"
        log.info("  OK: MLP output=%.6f", val_0)

        # Reload from temp checkpoint
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        ckpt = {
            "model_state_dict": model.state_dict(),
            "scaler": sc.to_dict(),
            "feature_names": FEATURE_NAMES,
            "config": {"in_features": _FEATURE_DIM, "dropout": 0.0},
        }
        torch.save(ckpt, tmp_path)
        model2, sc2, fn2 = load_checkpoint(tmp_path)
        with torch.no_grad():
            out2 = model2(torch.tensor(feat_scaled.tolist(), dtype=torch.float32))
        val_2 = float(out2.flatten().tolist()[0])
        assert math.isclose(val_0, val_2, rel_tol=1e-6), (
            "Reloaded model gives different output"
        )
        os.unlink(tmp_path)
        log.info("  OK: checkpoint save/load round-trip consistent")

        # Schema mismatch should raise
        bad_ckpt = dict(ckpt)
        bad_ckpt["feature_names"] = ["wrong_feature"]
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path2 = tmp.name
        torch.save(bad_ckpt, tmp_path2)
        try:
            load_checkpoint(tmp_path2)
            assert False, "Should have raised ValueError"
        except ValueError:
            log.info("  OK: schema mismatch correctly raises ValueError")
        finally:
            os.unlink(tmp_path2)
    else:
        log.info("[5/5] SKIP: torch not available")

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
        "Command: train | dataset=%s | model_path=%s | epochs=%d | resume=%s",
        args.dataset, args.model_path, args.epochs, args.resume,
    )
    result = train_mlp(
        dataset_path=args.dataset,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
        seed=args.seed,
        val_fraction=args.val_fraction,
        dropout=args.dropout,
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
    output_dir = Path(getattr(args, "output_dir", "results/td_mlp"))
    output_dir.mkdir(parents=True, exist_ok=True)

    model, scaler, feature_names = None, None, None
    if args.model_path and Path(args.model_path).exists():
        if not _TORCH_AVAILABLE:
            log.warning("torch not available; using heuristic-only mode")
            mode = "heuristic_only"
        else:
            model, scaler, feature_names = load_checkpoint(args.model_path)
            if args.device and args.device != "cpu":
                device = torch.device(args.device if torch.cuda.is_available() else "cpu")
                model = model.to(device)
            mode = "mlp"
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
        if mode == "mlp":
            result = solve_mlp(
                problem, model, scaler, feature_names,
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
            "algorithm": "TD-MLP" if mode == "mlp" else "TD-Insertion-Heuristic",
            "mode": mode,
            "instance": str(inst_vrp_path),
            "seed": args.seed,
            "top_k": args.top_k if mode == "mlp" else None,
            **result,
            **{f"eval_{k}": v for k, v in eval_result.items()},
            "bks": bks,
            "gap_pct": eval_result.get("gap_pct"),
        }

        # Pretty print
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
        if mode == "mlp":
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
        description="TD-MLP: MLP-guided insertion baseline for TDVRP Poryos2026",
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
    p_train = sub.add_parser("train", help="Train MLP from collected dataset")
    p_train.add_argument("--dataset", required=True, metavar="JSON")
    p_train.add_argument("--model-path", required=True, metavar="PT")
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--batch-size", type=int, default=256)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--device", default="cpu")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--val-fraction", type=float, default=0.1)
    p_train.add_argument("--dropout", type=float, default=0.0)
    p_train.add_argument("--checkpoint-file", default=None, metavar="FILE",
                         help="File to persist training progress")
    p_train.add_argument("--resume", action="store_true",
                         help="Resume training from checkpoint if available")
    p_train.add_argument("--log-file", default="run.log", metavar="FILE")

    # --- solve ---
    p_solve = sub.add_parser("solve", help="Solve instances using TD-MLP")
    p_solve.add_argument("--instance", nargs="+",
                         default=["data/TDVRP/hong_kong/n=50/poryos-hong_kong-n50-hyb/bpr-heavy"],
                         metavar="DIR_OR_VRP_JSON",
                         help="Paths to .vrp.json or directories (default: Hong Kong n50 BPR-heavy)")
    p_solve.add_argument("--model-path", default=None, metavar="PT",
                         help="Checkpoint; if missing, uses heuristic-only mode")
    p_solve.add_argument("--seed", type=int, default=0)
    p_solve.add_argument("--top-k", type=int, default=5,
                         help="Number of MLP-top-k candidates to evaluate exactly")
    p_solve.add_argument("--device", default="cpu")
    p_solve.add_argument("--time-budget", type=float, default=None,
                         help="Time budget in seconds")
    p_solve.add_argument("--output-json", default=None, metavar="JSON")
    p_solve.add_argument("--output-dir", default="results/td_mlp", metavar="DIR",
                         help="Folder to export results and visual charts (default: results/td_mlp)")
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
    p_eval.add_argument("--model-path", default=None, metavar="PT")
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.add_argument("--top-k", type=int, default=5)
    p_eval.add_argument("--device", default="cpu")
    p_eval.add_argument("--time-budget", type=float, default=None)
    p_eval.add_argument("--output-json", default=None, metavar="JSON")
    p_eval.add_argument("--output-dir", default="results/td_mlp", metavar="DIR")
    p_eval.add_argument("--no-plot", action="store_true")
    p_eval.add_argument("--checkpoint-file", default=None, metavar="FILE")
    p_eval.add_argument("--resume", action="store_true")
    p_eval.add_argument("--log-file", default="run.log", metavar="FILE")

    # --- self-test ---
    p_st = sub.add_parser("self-test", help="Run correctness checks")
    p_st.add_argument("--instance", required=True, metavar="DIR_OR_VRP_JSON")
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
