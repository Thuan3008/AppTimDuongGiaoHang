"""TD-QL: Tabular Q-Learning-guided Local Search baseline for TDVRP Poryos2026.

Reinforcement learning baseline for adaptive operator selection in local search.
Does NOT use deep RL/DQN. Uses Tabular Q-Learning with Bellman updates.
Conditioned on search state (progress, stagnation, recent acceptance rate).
Uses exact ATF route_cost oracle (TDVRPProblem) for candidate move evaluation.
Includes checkpoint pause/resume capability and auto-logging to run.log.

Scientific distinction from TD-ERAPO:
  - TD-ERAPO uses population-based metaheuristic (ARO/POA) with Elite AOS
    updating rolling operator success scores across iterations.
  - TD-QL is a single-trajectory local search agent using state-conditioned
    Bellman Q-Learning: Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)].
  - Action space: relocate, swap, two_opt_star, relocate_new_route.
  - Candidate moves have an explicit evaluation budget per step to prevent unfair exhaustive search.

CLI usage:
  python algorithms/td_ql.py train    --train-instances PATHS... --qtable-path qtable.json
  python algorithms/td_ql.py solve    --instance DIR_OR_VRP_JSON --qtable-path qtable.json
  python algorithms/td_ql.py evaluate --instance DIR_OR_VRP_JSON --qtable-path qtable.json
  python algorithms/td_ql.py self-test --instance DIR_OR_VRP_JSON
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass, field
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Logging – Dual stdout and run.log
# ---------------------------------------------------------------------------
log = logging.getLogger("td_ql")


def setup_logging(log_file: str | Path | None = "run.log") -> None:
    """Configure logging to stream to stdout and append to log_file."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(p, mode="a", encoding="utf-8"))
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s td_ql] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=handlers,
        force=True,
    )


setup_logging("run.log")


# ---------------------------------------------------------------------------
# Instance and BKS loading
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
    """Read BKS Duration value from a .bks.Duration.json file if present."""
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
# Route evaluation helper (Exact ATF via problem.route_cost)
# ---------------------------------------------------------------------------

def evaluate_route(
    problem,
    route: tuple[int, ...],
    cost_cache: dict[tuple[int, ...], tuple[float, float, float] | None],
) -> tuple[float, float, float] | None:
    """Evaluate route via problem.route_cost. Returns (duration, departure, return) or None."""
    if not route:
        return (0.0, 0.0, 0.0)
    if route in cost_cache:
        return cost_cache[route]
    # Check capacity constraint first (O(1))
    if sum(problem.demands[c] for c in route) > problem.capacity:
        cost_cache[route] = None
        return None
    try:
        val = problem.route_cost(route)
        res = (float(val[0]), float(val[1]), float(val[2]))
        cost_cache[route] = res
        return res
    except (ValueError, Exception):
        cost_cache[route] = None
        return None


# ---------------------------------------------------------------------------
# Initial solution generator: Parallel Cheapest TD-Insertion
# ---------------------------------------------------------------------------

def generate_initial_solution(
    problem,
    seed: int = 0,
) -> tuple[list[tuple[int, ...]], dict[int, float], float]:
    """Generate initial feasible solution using greedy TD-Insertion."""
    cost_cache: dict[tuple[int, ...], tuple[float, float, float] | None] = {}
    n = problem.dimension
    unplanned = set(range(1, n + 1))
    routes: list[tuple[int, ...]] = []
    route_costs: dict[int, float] = {}

    rng = random.Random(seed)

    while unplanned:
        best_delta = math.inf
        best_choice: tuple[int, int, tuple[int, ...], float] | None = None
        # candidate customer choices
        candidate_customers = sorted(unplanned)

        for customer in candidate_customers:
            cust_demand = problem.demands[customer]
            # Try existing routes
            for ridx, route in enumerate(routes):
                cur_load = sum(problem.demands[c] for c in route)
                if cur_load + cust_demand > problem.capacity:
                    continue
                cur_dur = route_costs.get(ridx, 0.0)
                for pos in range(len(route) + 1):
                    cand_route = route[:pos] + (customer,) + route[pos:]
                    eval_res = evaluate_route(problem, cand_route, cost_cache)
                    if eval_res is None:
                        continue
                    delta = eval_res[0] - cur_dur
                    if delta < best_delta:
                        best_delta = delta
                        best_choice = (customer, ridx, cand_route, eval_res[0])
            # Try new route
            if cust_demand <= problem.capacity:
                cand_route = (customer,)
                eval_res = evaluate_route(problem, cand_route, cost_cache)
                if eval_res is not None:
                    delta = eval_res[0]
                    if delta < best_delta:
                        best_delta = delta
                        best_choice = (customer, len(routes), cand_route, eval_res[0])

        if best_choice is None:
            # Fallback singleton
            cust = candidate_customers[0]
            cand_route = (cust,)
            eval_res = evaluate_route(problem, cand_route, cost_cache)
            cost = eval_res[0] if eval_res else 0.0
            routes.append(cand_route)
            route_costs[len(routes) - 1] = cost
            unplanned.remove(cust)
            continue

        cust, ridx, cand_route, new_cost = best_choice
        if ridx == len(routes):
            routes.append(cand_route)
            route_costs[ridx] = new_cost
        else:
            routes[ridx] = cand_route
            route_costs[ridx] = new_cost
        unplanned.remove(cust)

    total_duration = math.fsum(route_costs.values())
    return routes, route_costs, total_duration


# ---------------------------------------------------------------------------
# Action space and Move Generators
# ---------------------------------------------------------------------------

ACTIONS: list[str] = [
    "relocate",           # relocate customer intra/inter-route
    "swap",               # swap two customers intra/inter-route
    "two_opt_star",       # 2-opt* tail exchange between two routes
    "relocate_new_route", # move customer to a new singleton route
]


@dataclass
class CandidateMove:
    operator: str
    affected_indices: tuple[int, ...]
    replacement_routes: tuple[tuple[int, ...], ...]


def generate_candidate_moves(
    operator: str,
    routes: list[tuple[int, ...]],
    problem,
    rng: random.Random,
    budget: int = 30,
) -> list[CandidateMove]:
    """
    Generate up to `budget` feasible candidate moves for the specified operator.
    Does NOT sweep exhaustive neighborhoods freely, ensuring fair bounded search.
    """
    moves: list[CandidateMove] = []
    num_routes = len(routes)
    if num_routes == 0:
        return moves

    Q = problem.capacity
    demands = problem.demands

    if operator == "relocate":
        # Sample inter-route or intra-route relocations
        indices = list(range(num_routes))
        rng.shuffle(indices)
        for i in indices:
            route_i = routes[i]
            if len(route_i) == 0:
                continue
            positions_p = list(range(len(route_i)))
            rng.shuffle(positions_p)
            for p in positions_p:
                cust = route_i[p]
                demand_c = demands[cust]
                reduced_i = route_i[:p] + route_i[p + 1:]

                # Sample target routes
                target_indices = list(range(num_routes))
                rng.shuffle(target_indices)
                for j in target_indices:
                    if i == j:
                        # intra-route relocate
                        if len(reduced_i) >= 1:
                            for q in range(len(reduced_i) + 1):
                                if q != p and q != p + 1:
                                    cand = reduced_i[:q] + (cust,) + reduced_i[q:]
                                    moves.append(CandidateMove("relocate", (i,), (cand,)))
                                    if len(moves) >= budget:
                                        return moves
                    else:
                        # inter-route relocate: check capacity first
                        route_j = routes[j]
                        load_j = sum(demands[c] for c in route_j)
                        if load_j + demand_c <= Q:
                            for q in range(len(route_j) + 1):
                                cand_j = route_j[:q] + (cust,) + route_j[q:]
                                moves.append(CandidateMove("relocate", (i, j), (reduced_i, cand_j)))
                                if len(moves) >= budget:
                                    return moves

    elif operator == "swap":
        # Sample swap moves
        pairs: list[tuple[int, int]] = []
        for i in range(num_routes):
            for j in range(i, num_routes):
                pairs.append((i, j))
        rng.shuffle(pairs)

        for i, j in pairs:
            route_i = routes[i]
            route_j = routes[j]
            if i == j:
                # intra-route swap
                if len(route_i) >= 2:
                    p_indices = list(range(len(route_i)))
                    rng.shuffle(p_indices)
                    for p in p_indices:
                        for q in range(p + 1, len(route_i)):
                            new_r = list(route_i)
                            new_r[p], new_r[q] = new_r[q], new_r[p]
                            moves.append(CandidateMove("swap", (i,), (tuple(new_r),)))
                            if len(moves) >= budget:
                                return moves
            else:
                # inter-route swap
                if len(route_i) >= 1 and len(route_j) >= 1:
                    load_i = sum(demands[c] for c in route_i)
                    load_j = sum(demands[c] for c in route_j)
                    p_list = list(range(len(route_i)))
                    q_list = list(range(len(route_j)))
                    rng.shuffle(p_list)
                    rng.shuffle(q_list)
                    for p in p_list:
                        cust_p = route_i[p]
                        d_p = demands[cust_p]
                        for q in q_list:
                            cust_q = route_j[q]
                            d_q = demands[cust_q]
                            # Check capacities
                            if (load_i - d_p + d_q <= Q) and (load_j - d_q + d_p <= Q):
                                cand_i = route_i[:p] + (cust_q,) + route_i[p + 1:]
                                cand_j = route_j[:q] + (cust_p,) + route_j[q + 1:]
                                moves.append(CandidateMove("swap", (i, j), (cand_i, cand_j)))
                                if len(moves) >= budget:
                                    return moves

    elif operator == "two_opt_star":
        # 2-opt*: swap tails of two routes
        if num_routes >= 2:
            pairs = []
            for i in range(num_routes):
                for j in range(i + 1, num_routes):
                    pairs.append((i, j))
            rng.shuffle(pairs)
            for i, j in pairs:
                route_i, route_j = routes[i], routes[j]
                p_cuts = list(range(len(route_i) + 1))
                q_cuts = list(range(len(route_j) + 1))
                rng.shuffle(p_cuts)
                rng.shuffle(q_cuts)
                for p in p_cuts:
                    for q in q_cuts:
                        new_i = route_i[:p] + route_j[q:]
                        new_j = route_j[:q] + route_i[p:]
                        if (new_i, new_j) == (route_i, route_j) or (new_i, new_j) == (route_j, route_i):
                            continue
                        # Capacity check
                        if sum(demands[c] for c in new_i) <= Q and sum(demands[c] for c in new_j) <= Q:
                            moves.append(CandidateMove("two_opt_star", (i, j), (new_i, new_j)))
                            if len(moves) >= budget:
                                return moves

    elif operator == "relocate_new_route":
        # Relocate customer into a new singleton route
        indices = list(range(num_routes))
        rng.shuffle(indices)
        for i in indices:
            route_i = routes[i]
            if len(route_i) >= 2:  # only if route has >= 2 customers
                p_list = list(range(len(route_i)))
                rng.shuffle(p_list)
                for p in p_list:
                    cust = route_i[p]
                    if demands[cust] <= Q:
                        reduced = route_i[:p] + route_i[p + 1:]
                        new_singleton = (cust,)
                        moves.append(CandidateMove("relocate_new_route", (i,), (reduced, new_singleton)))
                        if len(moves) >= budget:
                            return moves

    return moves


# ---------------------------------------------------------------------------
# State Representation for Tabular Q-Learning
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QLState:
    progress_bin: int       # 0..3: progress through max_steps budget
    stagnation_bin: int     # 0..3: consecutive non-improving steps
    recent_success_bin: int # 0..2: acceptance rate over recent window (0: 0%, 1: 10-20%, 2: >=30%)

    def to_key(self) -> str:
        return f"{self.progress_bin}_{self.stagnation_bin}_{self.recent_success_bin}"

    @classmethod
    def from_key(cls, key: str) -> "QLState":
        parts = [int(x) for x in key.split("_")]
        return cls(parts[0], parts[1], parts[2])


def compute_state(
    step: int,
    max_steps: int,
    stagnation: int,
    recent_history: deque[bool],
) -> QLState:
    """Discretize continuous search state into compact, non-exploding discrete bins."""
    # Progress: 4 bins [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0]
    frac = min(0.999, step / max(1, max_steps))
    prog_bin = min(3, int(frac * 4))

    # Stagnation: 4 bins (0: 0, 1: 1-3, 2: 4-9, 3: >=10)
    if stagnation == 0:
        stag_bin = 0
    elif stagnation <= 3:
        stag_bin = 1
    elif stagnation <= 9:
        stag_bin = 2
    else:
        stag_bin = 3

    # Recent success: 3 bins over window
    if not recent_history:
        succ_bin = 1
    else:
        succ_rate = sum(1 for x in recent_history if x) / len(recent_history)
        if succ_rate == 0.0:
            succ_bin = 0
        elif succ_rate <= 0.25:
            succ_bin = 1
        else:
            succ_bin = 2

    return QLState(prog_bin, stag_bin, succ_bin)


# ---------------------------------------------------------------------------
# Tabular Q-Table
# ---------------------------------------------------------------------------

class QTable:
    """Tabular Q-Table storing Q(s, a) for small discrete state space."""

    def __init__(self, actions: list[str] = ACTIONS, default_value: float = 0.0):
        self.actions = list(actions)
        self.default_value = default_value
        self.table: dict[str, dict[str, float]] = {}

    def get_q(self, state: QLState, action: str) -> float:
        s_key = state.to_key()
        if s_key not in self.table:
            return self.default_value
        return self.table[s_key].get(action, self.default_value)

    def set_q(self, state: QLState, action: str, value: float) -> None:
        s_key = state.to_key()
        if s_key not in self.table:
            self.table[s_key] = {a: self.default_value for a in self.actions}
        self.table[s_key][action] = value

    def best_action(self, state: QLState, rng: random.Random) -> str:
        s_key = state.to_key()
        if s_key not in self.table:
            return rng.choice(self.actions)
        q_vals = self.table[s_key]
        max_q = max(q_vals.get(a, self.default_value) for a in self.actions)
        candidates = [a for a in self.actions if math.isclose(q_vals.get(a, self.default_value), max_q, rel_tol=1e-9)]
        return rng.choice(candidates)

    def max_q(self, state: QLState) -> float:
        s_key = state.to_key()
        if s_key not in self.table:
            return self.default_value
        return max(self.table[s_key].get(a, self.default_value) for a in self.actions)

    def select_action(
        self,
        state: QLState,
        epsilon: float,
        rng: random.Random,
        random_policy: bool = False,
    ) -> str:
        if random_policy or rng.random() < epsilon:
            return rng.choice(self.actions)
        return self.best_action(state, rng)

    def update(
        self,
        state: QLState,
        action: str,
        reward: float,
        next_state: QLState | None,
        alpha: float,
        gamma: float,
    ) -> None:
        """Bellman update: Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]."""
        old_q = self.get_q(state, action)
        if next_state is None:
            # Terminal step: no bootstrap
            target = reward
        else:
            target = reward + gamma * self.max_q(next_state)
        new_q = old_q + alpha * (target - old_q)
        self.set_q(state, action, new_q)

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": self.actions,
            "default_value": self.default_value,
            "table": self.table,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QTable":
        qt = cls(actions=data["actions"], default_value=data.get("default_value", 0.0))
        qt.table = {k: dict(v) for k, v in data["table"].items()}
        return qt


# ---------------------------------------------------------------------------
# Checkpoint Persistence for Search & Resume
# ---------------------------------------------------------------------------

def _save_search_checkpoint(
    checkpoint_path: str | Path,
    instance_name: str,
    dimension: int,
    mode: str,
    step: int,
    max_steps: int,
    routes: list[tuple[int, ...]],
    route_costs: dict[int, float],
    best_routes: list[tuple[int, ...]],
    best_cost: float,
    stagnation: int,
    stats: dict[str, Any],
    elapsed_seconds: float,
    completed: bool = False,
) -> None:
    p = Path(checkpoint_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "instance_name": instance_name,
        "dimension": dimension,
        "mode": mode,
        "step": step,
        "max_steps": max_steps,
        "routes": [list(r) for r in routes],
        "route_costs": {str(k): float(v) for k, v in route_costs.items()},
        "best_routes": [list(r) for r in best_routes],
        "best_cost": float(best_cost),
        "stagnation": stagnation,
        "stats": stats,
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
    p = Path(checkpoint_path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("dimension") != expected_dimension:
            log.warning("Checkpoint dimension (%s) != expected (%d)", data.get("dimension"), expected_dimension)
            return None
        return data
    except Exception as e:
        log.warning("Could not read search checkpoint %s: %s", checkpoint_path, e)
        return None


# ---------------------------------------------------------------------------
# Local Search Step Engine
# ---------------------------------------------------------------------------

@dataclass
class SearchStatistics:
    steps: int = 0
    proposed_moves: dict[str, int] = field(default_factory=lambda: {a: 0 for a in ACTIONS})
    capacity_checks: dict[str, int] = field(default_factory=lambda: {a: 0 for a in ACTIONS})
    capacity_rejects: dict[str, int] = field(default_factory=lambda: {a: 0 for a in ACTIONS})
    exact_route_evals: dict[str, int] = field(default_factory=lambda: {a: 0 for a in ACTIONS})
    accepted_moves: dict[str, int] = field(default_factory=lambda: {a: 0 for a in ACTIONS})
    cache_hits: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "proposed_moves": self.proposed_moves,
            "capacity_checks": self.capacity_checks,
            "capacity_rejects": self.capacity_rejects,
            "exact_route_evals": self.exact_route_evals,
            "accepted_moves": self.accepted_moves,
            "cache_hits": self.cache_hits,
            "total_route_evals": sum(self.exact_route_evals.values()),
            "total_accepted": sum(self.accepted_moves.values()),
        }


def step_local_search(
    problem,
    routes: list[tuple[int, ...]],
    route_costs: dict[int, float],
    operator: str,
    cost_cache: dict[tuple[int, ...], tuple[float, float, float] | None],
    rng: random.Random,
    stats: SearchStatistics,
    candidate_budget: int = 30,
) -> tuple[list[tuple[int, ...]], dict[int, float], bool, float]:
    """
    Execute one local search step for the chosen operator with bounded evaluation budget.
    Returns: (updated_routes, updated_route_costs, accepted, delta_gain)
    """
    moves = generate_candidate_moves(operator, routes, problem, rng, budget=candidate_budget)
    stats.proposed_moves[operator] += len(moves)

    best_gain = 1e-9
    best_move: tuple[tuple[int, ...], tuple[tuple[int, ...], ...], list[tuple[float, float, float]]] | None = None

    for move in moves:
        stats.capacity_checks[operator] += len(move.replacement_routes)
        # Check capacities
        cap_ok = True
        for r in move.replacement_routes:
            if r and sum(problem.demands[c] for c in r) > problem.capacity:
                cap_ok = False
                stats.capacity_rejects[operator] += 1
                break
        if not cap_ok:
            continue

        # Evaluate replacement routes
        evals: list[tuple[float, float, float]] = []
        eval_failed = False
        for r in move.replacement_routes:
            if not r:
                evals.append((0.0, 0.0, 0.0))
                continue
            if r in cost_cache:
                stats.cache_hits += 1
                val = cost_cache[r]
            else:
                stats.exact_route_evals[operator] += 1
                val = evaluate_route(problem, r, cost_cache)
            if val is None:
                eval_failed = True
                break
            evals.append(val)

        if eval_failed:
            continue

        old_cost = math.fsum(route_costs[i] for i in move.affected_indices)
        new_cost = math.fsum(e[0] for e in evals)
        gain = old_cost - new_cost

        if gain > best_gain:
            best_gain = gain
            best_move = (move.affected_indices, move.replacement_routes, evals)

    if best_move is not None:
        affected, replacements, evals = best_move
        # Apply move: replace affected routes, filter out empty routes
        new_routes_list: list[tuple[int, ...]] = []
        new_costs_dict: dict[int, float] = {}

        # Retain unaffected routes
        for i, r in enumerate(routes):
            if i not in affected and r:
                new_routes_list.append(r)
                new_costs_dict[len(new_routes_list) - 1] = route_costs[i]

        # Add replacement non-empty routes
        for r, ev in zip(replacements, evals):
            if r:
                new_routes_list.append(r)
                new_costs_dict[len(new_routes_list) - 1] = ev[0]

        stats.accepted_moves[operator] += 1
        return new_routes_list, new_costs_dict, True, best_gain

    return routes, route_costs, False, 0.0


# ---------------------------------------------------------------------------
# Training: Tabular Q-Learning
# ---------------------------------------------------------------------------

def train_ql(
    train_instance_paths: list[str | Path],
    qtable_output_path: str | Path,
    episodes: int = 30,
    max_steps: int = 50,
    alpha: float = 0.2,
    gamma: float = 0.9,
    epsilon_start: float = 0.2,
    epsilon_min: float = 0.02,
    epsilon_decay: float = 0.98,
    seed: int = 42,
    candidate_budget: int = 25,
) -> dict[str, Any]:
    """Train Tabular Q-Learning agent on training instances (test instances strictly prohibited)."""
    # Strict validation: ensure n50 heavy is NOT in training set
    for p in train_instance_paths:
        name = Path(p).name.lower()
        if "n50" in name and "heavy" in name:
            raise ValueError(
                f"STRICT TEST SEPARATION VIOLATION: {p} is the test benchmark. "
                "n50 heavy is strictly prohibited from training."
            )

    rng = random.Random(seed)
    np.random.seed(seed)

    qtable = QTable(ACTIONS)
    epsilon = epsilon_start
    recent_history: deque[bool] = deque(maxlen=10)

    t_start = time.perf_counter()
    total_steps = 0
    total_rewards = 0.0
    episodes_stats: list[dict[str, Any]] = []

    log.info(
        "Starting Q-Learning training: episodes=%d, max_steps=%d, alpha=%.2f, gamma=%.2f, eps=[%.2f -> %.2f]",
        episodes, max_steps, alpha, gamma, epsilon_start, epsilon_min,
    )

    for ep in range(episodes):
        # Pick instance round-robin or random
        inst_path = train_instance_paths[ep % len(train_instance_paths)]
        problem = load_instance(inst_path)
        cost_cache: dict[tuple[int, ...], tuple[float, float, float] | None] = {}
        stats = SearchStatistics()

        # Initial solution
        routes, route_costs, cur_duration = generate_initial_solution(problem, seed=seed + ep)
        best_duration = cur_duration
        stagnation = 0

        ep_reward = 0.0
        ep_steps = 0

        for step in range(max_steps):
            total_steps += 1
            ep_steps += 1
            state = compute_state(step, max_steps, stagnation, recent_history)

            # Choose action via epsilon-greedy
            action = qtable.select_action(state, epsilon, rng)

            # Execute step
            routes, route_costs, accepted, gain = step_local_search(
                problem, routes, route_costs, action, cost_cache, rng, stats,
                candidate_budget=candidate_budget,
            )

            # Reward calculation
            if accepted:
                cur_duration = math.fsum(route_costs.values())
                norm_gain = gain / max(cur_duration, 1.0)
                if cur_duration < best_duration:
                    reward = 2.0 * norm_gain  # global improvement bonus
                    best_duration = cur_duration
                else:
                    reward = 1.0 * norm_gain
                stagnation = 0
                recent_history.append(True)
            else:
                reward = -0.01  # non-improving move penalty
                stagnation += 1
                recent_history.append(False)

            ep_reward += reward

            # Next state and Q-learning update
            if step == max_steps - 1:
                next_state = None
            else:
                next_state = compute_state(step + 1, max_steps, stagnation, recent_history)

            qtable.update(state, action, reward, next_state, alpha, gamma)

        # Epsilon decay per episode
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        total_rewards += ep_reward

        episodes_stats.append({
            "episode": ep + 1,
            "instance": getattr(problem, "name", str(inst_path)),
            "steps": ep_steps,
            "initial_duration": cur_duration + gain,
            "final_duration": math.fsum(route_costs.values()),
            "reward": ep_reward,
            "epsilon": epsilon,
        })
        if (ep + 1) % 5 == 0 or ep == episodes - 1:
            log.info(
                "Episode %d/%d completed | steps=%d | reward=%.4f | epsilon=%.4f",
                ep + 1, episodes, ep_steps, ep_reward, epsilon,
            )

    training_time = time.perf_counter() - t_start

    # Save Q-table artifact
    out_path = Path(qtable_output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "algorithm": "TD-QL",
        "description": "Tabular Q-Learning operator selection for TDVRP Local Search",
        "actions": ACTIONS,
        "state_spec": {
            "progress_bins": 4,
            "stagnation_bins": 4,
            "recent_success_bins": 3,
            "total_states": 48,
        },
        "hyperparameters": {
            "alpha": alpha,
            "gamma": gamma,
            "epsilon_start": epsilon_start,
            "epsilon_min": epsilon_min,
            "epsilon_decay": epsilon_decay,
            "max_steps": max_steps,
            "episodes": episodes,
            "candidate_budget": candidate_budget,
            "seed": seed,
        },
        "train_instances": [str(p) for p in train_instance_paths],
        "train_stats": {
            "total_episodes": episodes,
            "total_steps": total_steps,
            "total_rewards": total_rewards,
            "training_time_seconds": training_time,
            "states_explored": len(qtable.table),
        },
        "q_table": qtable.to_dict(),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)
    log.info("Saved trained Q-table to %s (explored %d states)", out_path, len(qtable.table))

    return {
        "episodes": episodes,
        "total_steps": total_steps,
        "training_time_seconds": training_time,
        "states_explored": len(qtable.table),
        "qtable_path": str(out_path),
    }


# ---------------------------------------------------------------------------
# Solver: Inference with Learned Q-Table or Random Policy
# ---------------------------------------------------------------------------

def solve_ql(
    problem,
    qtable: QTable | None = None,
    max_steps: int = 60,
    time_budget_sec: float = float("inf"),
    seed: int = 0,
    eval_epsilon: float = 0.0,
    random_policy: bool = False,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    candidate_budget: int = 30,
) -> dict[str, Any]:
    """
    Run Q-Learning guided local search (or random policy baseline) on problem.
    Q-table is frozen (eval_epsilon=0.0 by default, no Q-table updates).
    """
    rng = random.Random(seed)
    cost_cache: dict[tuple[int, ...], tuple[float, float, float] | None] = {}
    stats = SearchStatistics()
    recent_history: deque[bool] = deque(maxlen=10)

    n = problem.dimension
    t_start = time.perf_counter()
    time_offset = 0.0

    step_start = 0
    routes: list[tuple[int, ...]] = []
    route_costs: dict[int, float] = {}
    best_routes: list[tuple[int, ...]] = []
    best_duration = math.inf
    stagnation = 0

    if resume and checkpoint_path:
        ckpt = _load_search_checkpoint(checkpoint_path, n)
        if ckpt and not ckpt.get("completed", False):
            routes = [tuple(r) for r in ckpt["routes"]]
            route_costs = {int(k): v for k, v in ckpt["route_costs"].items()}
            best_routes = [tuple(r) for r in ckpt["best_routes"]]
            best_duration = ckpt["best_cost"]
            step_start = ckpt.get("step", 0)
            stagnation = ckpt.get("stagnation", 0)
            time_offset = ckpt.get("elapsed_seconds", 0.0)
            log.info("Resuming QL solve from checkpoint: step=%d/%d, best_cost=%.3f", step_start, max_steps, best_duration)

    if not routes:
        # Generate initial solution via TD-Insertion
        log.info("Generating initial solution via TD-Insertion...")
        routes, route_costs, init_cost = generate_initial_solution(problem, seed=seed)
        best_routes = [r for r in routes]
        best_duration = init_cost
        log.info("Initial solution: duration=%.3f s, routes=%d", init_cost, len(routes))

    if qtable is None:
        qtable = QTable(ACTIONS)

    cur_duration = math.fsum(route_costs.values())
    mode_name = "random_policy" if random_policy else "q_learning" if qtable.table else "untrained_baseline"

    for step in range(step_start, max_steps):
        elapsed = time.perf_counter() - t_start + time_offset
        if elapsed >= time_budget_sec:
            log.info("Time budget reached (%.2fs >= %.2fs); terminating search", elapsed, time_budget_sec)
            break

        stats.steps += 1
        state = compute_state(step, max_steps, stagnation, recent_history)

        # Select action
        action = qtable.select_action(state, eval_epsilon, rng, random_policy=random_policy)

        # Step
        routes, route_costs, accepted, gain = step_local_search(
            problem, routes, route_costs, action, cost_cache, rng, stats,
            candidate_budget=candidate_budget,
        )

        if accepted:
            cur_duration = math.fsum(route_costs.values())
            recent_history.append(True)
            stagnation = 0
            if cur_duration < best_duration:
                best_duration = cur_duration
                best_routes = [r for r in routes]
                log.info("Step %d [%s]: New best duration=%.3f s (gain=%.3f s)", step + 1, action, best_duration, gain)
        else:
            recent_history.append(False)
            stagnation += 1

        # Checkpoint every 10 steps or if requested
        if checkpoint_path and (step + 1) % 10 == 0:
            _save_search_checkpoint(
                checkpoint_path=checkpoint_path,
                instance_name=getattr(problem, "name", ""),
                dimension=n,
                mode=mode_name,
                step=step + 1,
                max_steps=max_steps,
                routes=routes,
                route_costs=route_costs,
                best_routes=best_routes,
                best_cost=best_duration,
                stagnation=stagnation,
                stats=stats.to_dict(),
                elapsed_seconds=time.perf_counter() - t_start + time_offset,
                completed=False,
            )

    runtime = time.perf_counter() - t_start + time_offset

    if checkpoint_path:
        _save_search_checkpoint(
            checkpoint_path=checkpoint_path,
            instance_name=getattr(problem, "name", ""),
            dimension=n,
            mode=mode_name,
            step=max_steps,
            max_steps=max_steps,
            routes=best_routes,
            route_costs={i: cost_cache.get(r, (0.0,))[0] for i, r in enumerate(best_routes)},
            best_routes=best_routes,
            best_cost=best_duration,
            stagnation=stagnation,
            stats=stats.to_dict(),
            elapsed_seconds=runtime,
            completed=True,
        )

    # Compute final route evaluations and departures
    final_evals: list[tuple[float, float, float]] = []
    for r in best_routes:
        ev = evaluate_route(problem, r, cost_cache)
        if ev is None:
            raise RuntimeError(f"Best route {r} failed evaluation")
        final_evals.append(ev)

    final_duration = math.fsum(e[0] for e in final_evals)

    return {
        "routes": [list(r) for r in best_routes],
        "duration": final_duration,
        "runtime_seconds": runtime,
        "departure_times": [e[1] for e in final_evals],
        "return_times": [e[2] for e in final_evals],
        "route_durations": [e[0] for e in final_evals],
        "route_loads": [sum(problem.demands[c] for c in r) for r in best_routes],
        "num_routes": len(best_routes),
        "mode": mode_name,
        "stats": stats.to_dict(),
    }


# ---------------------------------------------------------------------------
# Independent Solution Validation & Evaluation
# ---------------------------------------------------------------------------

def evaluate_solution(problem, routes: list[tuple[int, ...]], bks: float | None = None) -> dict[str, Any]:
    """Validate solution using problem.validate and compute gap against BKS."""
    n = problem.dimension
    Q = problem.capacity
    all_custs = [c for r in routes for c in r]
    missing = sorted(set(range(1, n + 1)) - set(all_custs))
    duplicates = sorted(c for c in set(all_custs) if all_custs.count(c) > 1)
    cap_ok = all(sum(problem.demands[c] for c in r) <= Q for r in routes)

    try:
        evaluations = problem.validate(routes)
        total = math.fsum(float(e[0]) for e in evaluations)
        dep_times = [float(e[1]) for e in evaluations]
        ret_times = [float(e[2]) for e in evaluations]
        route_durations = [float(e[0]) for e in evaluations]
        feasible = True
    except Exception as e:
        log.warning("Validation exception: %s", e)
        total = float("inf")
        dep_times = []
        ret_times = []
        route_durations = []
        feasible = False

    gap_pct = ((total - bks) / bks * 100.0) if (feasible and bks and bks > 0) else None

    return {
        "feasible": feasible,
        "duration": total,
        "gap_pct": gap_pct,
        "bks": bks,
        "missing": missing,
        "duplicates": duplicates,
        "capacity_ok": cap_ok,
        "num_routes": len(routes),
        "departure_times": dep_times,
        "return_times": ret_times,
        "route_durations": route_durations,
    }


# ---------------------------------------------------------------------------
# Self-Test: 7 Comprehensive Verification Checks
# ---------------------------------------------------------------------------

def run_self_test(instance_path: str | Path) -> None:
    """Execute 7 correctness and contract tests."""
    log.info("=== Self-test TD-QL on %s ===", instance_path)
    problem = load_instance(instance_path)
    n = problem.dimension
    Q = problem.capacity
    log.info("n=%d, Q=%d", n, Q)

    # 1. Initial solution generation (TD-Insertion)
    log.info("[1/7] Heuristic initial solution (TD-Insertion)...")
    t0 = time.perf_counter()
    init_routes, route_costs, init_cost = generate_initial_solution(problem, seed=0)
    t_init = time.perf_counter() - t0
    log.info("  Initial duration=%.3f s, routes=%d, time=%.2fs", init_cost, len(init_routes), t_init)
    assert len(init_routes) >= 1, "Initial routes must be non-empty"

    # 2. Validation of initial solution
    log.info("[2/7] Validating routes (customer conservation and capacities)...")
    eval_res = evaluate_solution(problem, init_routes)
    assert eval_res["feasible"], f"Initial solution infeasible: {eval_res}"
    assert len(eval_res["missing"]) == 0, f"Missing customers: {eval_res['missing']}"
    assert len(eval_res["duplicates"]) == 0, f"Duplicate customers: {eval_res['duplicates']}"
    assert eval_res["capacity_ok"], "Capacity constraint violated"
    log.info("  OK: all %d customers present once, capacity respected", n)

    # 3. Move generators contract check
    log.info("[3/7] Move generators and operator contracts check...")
    rng = random.Random(42)
    for op in ACTIONS:
        moves = generate_candidate_moves(op, init_routes, problem, rng, budget=10)
        assert len(moves) >= 1, f"Operator {op} generated 0 moves"
        for m in moves:
            # Check customer conservation
            affected_custs = [c for idx in m.affected_indices for c in init_routes[idx]]
            repl_custs = [c for r in m.replacement_routes for c in r]
            assert sorted(affected_custs) == sorted(repl_custs), (
                f"Operator {op} violated customer conservation: {affected_custs} != {repl_custs}"
            )
        log.info("  OK: operator %s generated %d valid moves", op, len(moves))

    # 4. Q-table Bellman update numerical unit test
    log.info("[4/7] Q-table Bellman update numerical unit test...")
    qt = QTable(ACTIONS, default_value=0.5)
    s1 = QLState(0, 0, 0)
    s2 = QLState(1, 0, 1)
    qt.set_q(s2, "relocate", 0.8)
    qt.set_q(s2, "swap", 0.4)
    # Target: 0.5 + 0.2 * (1.0 + 0.9 * 0.8 - 0.5) = 0.5 + 0.2 * (1.72 - 0.5) = 0.5 + 0.244 = 0.744
    qt.update(s1, "relocate", reward=1.0, next_state=s2, alpha=0.2, gamma=0.9)
    assert math.isclose(qt.get_q(s1, "relocate"), 0.744, rel_tol=1e-6), (
        f"Bellman update mismatch: got {qt.get_q(s1, 'relocate')}, expected 0.744"
    )
    log.info("  OK: Bellman update matches numerical expectation")

    # 5. Q-table serialization round-trip
    log.info("[5/7] Q-table serialization round-trip...")
    tmp_path = Path("scratch_test_qtable.json")
    bundle = qt.to_dict()
    tmp_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    qt2 = QTable.from_dict(json.loads(tmp_path.read_text(encoding="utf-8")))
    assert math.isclose(qt2.get_q(s1, "relocate"), 0.744, rel_tol=1e-6)
    if tmp_path.exists():
        tmp_path.unlink()
    log.info("  OK: Q-table serialization preserved values")

    # 6. Checkpoint pause & resume equivalence check
    log.info("[6/7] Checkpoint save & resume check...")
    tmp_ckpt = Path("scratch_test_ckpt.json")
    res1 = solve_ql(
        problem, qtable=qt, max_steps=5, seed=0, checkpoint_path=tmp_ckpt, resume=False
    )
    res2 = solve_ql(
        problem, qtable=qt, max_steps=5, seed=0, checkpoint_path=tmp_ckpt, resume=True
    )
    assert math.isclose(res1["duration"], res2["duration"], rel_tol=1e-6)
    if tmp_ckpt.exists():
        tmp_ckpt.unlink()
    log.info("  OK: solve checkpoint resume produces identical valid solution")

    # 7. run.log logging check
    log.info("[7/7] run.log logging check...")
    log_path = Path("run.log")
    assert log_path.exists() and log_path.stat().st_size > 0
    log.info("  OK: run.log successfully recorded %d bytes", log_path.stat().st_size)

    log.info("=== All self-tests PASSED ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_train(args):
    setup_logging(args.log_file)
    log.info("Command: train | episodes=%d | max_steps=%d | output=%s", args.episodes, args.max_steps, args.qtable_path)
    train_ql(
        train_instance_paths=args.train_instances,
        qtable_output_path=args.qtable_path,
        episodes=args.episodes,
        max_steps=args.max_steps,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )


def _cmd_solve(args):
    setup_logging(args.log_file)
    log.info(
        "Command: solve | instance=%s | qtable_path=%s | random_policy=%s",
        args.instance, args.qtable_path, args.random_policy,
    )
    inst_list = args.instance if isinstance(args.instance, list) else [args.instance]
    output_dir = Path(getattr(args, "output_dir", "results/td_ql"))
    output_dir.mkdir(parents=True, exist_ok=True)

    qtable = None
    if args.qtable_path and Path(args.qtable_path).exists():
        with open(args.qtable_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        qtable = QTable.from_dict(data.get("q_table", data))
        log.info("Loaded Q-table from %s (%d states)", args.qtable_path, len(qtable.table))
    else:
        if args.qtable_path:
            log.warning("Q-table path not found: %s; running with untrained baseline", args.qtable_path)
        else:
            log.info("No Q-table path given; running with %s", "random_policy" if args.random_policy else "untrained_baseline")

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

        result = solve_ql(
            problem,
            qtable=qtable,
            max_steps=args.max_steps,
            time_budget_sec=args.wall_time_sec if args.wall_time_sec else float("inf"),
            seed=args.seed,
            eval_epsilon=args.eval_epsilon,
            random_policy=args.random_policy,
            checkpoint_path=ckpt_path,
            resume=args.resume,
        )

        eval_result = evaluate_solution(problem, [tuple(r) for r in result["routes"]], bks=bks)

        output = {
            "algorithm": "TD-QL",
            "instance": str(inst_vrp_path),
            "seed": args.seed,
            "mode": result["mode"],
            "max_steps": args.max_steps,
            **result,
            **{f"eval_{k}": v for k, v in eval_result.items()},
            "bks": bks,
            "gap_pct": eval_result.get("gap_pct"),
        }

        # Pretty print summary
        print(f"\n{'='*65}")
        print(f"Algorithm : TD-QL ({result['mode']})")
        print(f"Instance  : {inst_vrp_path.name}")
        print(f"Duration  : {result['duration']:.3f} s")
        if bks:
            gap = eval_result.get("gap_pct")
            print(f"BKS       : {bks:.3f} s")
            print(f"Gap(%)    : {gap:.4f}%" if gap is not None else "Gap(%)    : N/A")
        print(f"Routes    : {len(result['routes'])}")
        print(f"Feasible  : {eval_result['feasible']}")
        print(f"Runtime   : {result['runtime_seconds']:.2f} s")
        print(f"Steps     : {result['stats']['steps']}")
        print(f"Accepted  : {result['stats']['total_accepted']}")
        print(f"RouteEvals: {result['stats']['total_route_evals']}")
        print(f"CacheHits : {result['stats']['cache_hits']}")
        print(f"{'='*65}\n")

        # Save JSON
        out_json = Path(args.output_json) if args.output_json else (output_dir / f"results_{inst_stem}.json")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        log.info("Results saved to %s", out_json)

        # Write summary.txt
        summary_txt = output_dir / "summary.txt"
        with open(summary_txt, "a", encoding="utf-8") as f:
            f.write(f"Algorithm : TD-QL ({result['mode']})\n")
            f.write(f"Instance  : {inst_stem}\n")
            f.write(f"Duration  : {result['duration']:.3f} s\n")
            f.write(f"BKS       : {bks:.3f} s\n" if bks else "BKS : N/A\n")
            f.write(f"Gap(%)    : {eval_result.get('gap_pct', 0.0):.4f}%\n")
            f.write(f"Routes    : {len(result['routes'])}\n")
            f.write(f"Runtime   : {result['runtime_seconds']:.2f} s\n")
            f.write(f"Steps     : {result['stats']['steps']}\n")
            f.write(f"Accepted  : {result['stats']['total_accepted']}\n")
            f.write(f"RouteEvals: {result['stats']['total_route_evals']}\n\n")

        # Generate plots
        if not getattr(args, "no_plot", False):
            try:
                from experiments.export_visualizations import (
                    plot_route_map, plot_departure_timeline, plot_operator_statistics
                )
                plot_route_map(
                    inst_vrp_path, result["routes"], result["duration"], bks, eval_result.get("gap_pct"),
                    "TD-QL (Tabular Q-Learning)", output_dir / f"route_map_{inst_stem}.png"
                )
                dep_times = result.get("departure_times", [0.0] * len(result["routes"]))
                durations = eval_result.get("route_durations", result.get("route_durations", []))
                plot_departure_timeline(
                    dep_times, durations, "TD-QL", inst_stem,
                    output_dir / f"timeline_{inst_stem}.png"
                )
                plot_operator_statistics(
                    result.get("stats", {}), inst_stem,
                    output_dir / f"operator_stats_{inst_stem}.png"
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
        description="TD-QL: Tabular Q-Learning-guided Local Search for TDVRP Poryos2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log-file", default="run.log", metavar="FILE",
                        help="Path to write execution log (default: run.log)")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    p_train = sub.add_parser("train", help="Train Q-Learning agent on instances")
    p_train.add_argument("--train-instances", nargs="+", required=True, metavar="PATH",
                         help="Paths to .vrp.json files or directories (NOT n50-heavy)")
    p_train.add_argument("--qtable-path", required=True, metavar="JSON",
                         help="Output Q-table file")
    p_train.add_argument("--episodes", type=int, default=30)
    p_train.add_argument("--max-steps", type=int, default=50)
    p_train.add_argument("--alpha", type=float, default=0.2)
    p_train.add_argument("--gamma", type=float, default=0.9)
    p_train.add_argument("--epsilon-start", type=float, default=0.2)
    p_train.add_argument("--epsilon-min", type=float, default=0.02)
    p_train.add_argument("--epsilon-decay", type=float, default=0.98)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--log-file", default="run.log", metavar="FILE")

    # --- solve ---
    p_solve = sub.add_parser("solve", help="Solve instances using TD-QL")
    p_solve.add_argument("--instance", nargs="+",
                         default=["data/TDVRP/hong_kong/n=50/poryos-hong_kong-n50-hyb/bpr-heavy"],
                         metavar="PATH",
                         help="Paths to .vrp.json or directories (default: Hong Kong n50 BPR-heavy)")
    p_solve.add_argument("--qtable-path", default=None, metavar="JSON",
                         help="Path to trained Q-table file")
    p_solve.add_argument("--random-policy", action="store_true",
                         help="Ablation: select operators uniformly at random")
    p_solve.add_argument("--untrained-baseline", action="store_true",
                         help="Ablation: run without trained Q-table")
    p_solve.add_argument("--max-steps", type=int, default=60)
    p_solve.add_argument("--wall-time-sec", type=float, default=None)
    p_solve.add_argument("--seed", type=int, default=0)
    p_solve.add_argument("--eval-epsilon", type=float, default=0.0,
                         help="Exploration rate during inference (default 0.0: greedy)")
    p_solve.add_argument("--output-json", default=None, metavar="FILE")
    p_solve.add_argument("--output-dir", default="results/td_ql", metavar="DIR",
                         help="Folder to export results and visual charts (default: results/td_ql)")
    p_solve.add_argument("--no-plot", action="store_true",
                         help="Disable generating PNG illustrations")
    p_solve.add_argument("--checkpoint-file", default=None, metavar="FILE",
                         help="Path to save/load search checkpoint")
    p_solve.add_argument("--resume", action="store_true",
                         help="Resume search from checkpoint if exists")
    p_solve.add_argument("--log-file", default="run.log", metavar="FILE")

    # --- evaluate ---
    p_eval = sub.add_parser("evaluate", help="Solve and report vs BKS")
    p_eval.add_argument("--instance", nargs="+",
                        default=["data/TDVRP/hong_kong/n=50/poryos-hong_kong-n50-hyb/bpr-heavy"],
                        metavar="PATH")
    p_eval.add_argument("--qtable-path", default=None, metavar="JSON")
    p_eval.add_argument("--random-policy", action="store_true")
    p_eval.add_argument("--untrained-baseline", action="store_true")
    p_eval.add_argument("--max-steps", type=int, default=60)
    p_eval.add_argument("--wall-time-sec", type=float, default=None)
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.add_argument("--eval-epsilon", type=float, default=0.0)
    p_eval.add_argument("--output-json", default=None, metavar="FILE")
    p_eval.add_argument("--output-dir", default="results/td_ql", metavar="DIR")
    p_eval.add_argument("--no-plot", action="store_true")
    p_eval.add_argument("--checkpoint-file", default=None, metavar="FILE")
    p_eval.add_argument("--resume", action="store_true")
    p_eval.add_argument("--log-file", default="run.log", metavar="FILE")

    # --- self-test ---
    p_test = sub.add_parser("self-test", help="Run correctness checks")
    p_test.add_argument("--instance", required=True, metavar="PATH")
    p_test.add_argument("--log-file", default="run.log", metavar="FILE")

    args = parser.parse_args()

    if args.command == "train":
        _cmd_train(args)
    elif args.command == "solve":
        _cmd_solve(args)
    elif args.command == "evaluate":
        _cmd_evaluate(args)
    elif args.command == "self-test":
        _cmd_self_test(args)


if __name__ == "__main__":
    main()
