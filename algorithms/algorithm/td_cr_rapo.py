"""TD-CR-RAPO: Time-Dependent Criticality-Guided ARO–POA with Adaptive Large-Neighborhood Reconstruction.

Architecture:
- ARO Core: Primary global exploration engine (>= 40% NFE budget)
- TD-Criticality: Zero-NFE route-trace criticality analysis (Travel, Temporal, Route contributions)
- POA Large-Neighborhood Reconstruction:
    - Adaptive perturbation radius (POA-SMALL: 5-10%n, POA-LARGE: 10-20%n)
    - Critical customer sampling probability p_i proportional to (epsilon + C_i)
    - TD-aware Regret-2 Reconstruction with 0-NFE pre-screening
    - Elite B structural guidance
- Elite AOS Refinement:
    - Operators: relocate, swap, 2-opt, 2-opt*
    - EMA relative improvement reward tracking (alpha=0.20, floor epsilon=0.10)
    - Hard cap <= 20% NFE budget
- Adaptive NFE Controller:
    - 20% NFE warmup favoring ARO (~70% ARO, POA-SMALL only)
    - Dynamic scheduling by efficiency: E_ARO, E_POA, E_ELITE
    - Strict budget enforcement: ARO >= 40%, POA >= 15%, Elite <= 20%
    - Stagnation tracking (S1 = 0.10 * max_nfe, S2 = 0.20 * max_nfe)
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field, fields
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.tdvrp_common import OptimizationResult, TDVRPProblem
from algorithms.rapo_operators import rank_keys, from_keys
from algorithms.route_aware_operators import (
    route_aware_relocate,
    route_aware_swap,
    route_aware_two_opt,
    route_aware_two_opt_star,
)
from algorithms.td_criticality import compute_customer_criticality, compute_removal_probabilities


@dataclass
class TDCRRAPOConfig:
    population_size: int = 10
    max_function_evaluations: int = 300
    max_iterations: int | None = None
    use_exact_parent_dedup: bool = False  # Section 20: locked to False for search architecture comparison

    # ARO target share: 40-60%, hard minimum 40%
    min_aro_share: float = 0.40
    target_aro_share: float = 0.50
    min_poa_share: float = 0.15
    target_poa_share: float = 0.30
    max_elite_share: float = 0.20

    # Warmup phase: first 20% NFE
    warmup_nfe_fraction: float = 0.20
    warmup_aro_target: float = 0.70

    # Stagnation thresholds
    s1_fraction: float = 0.10
    s2_fraction: float = 0.20

    # POA Perturbation bounds
    poa_small_min: float = 0.05
    poa_small_max: float = 0.10
    poa_large_min: float = 0.10
    poa_large_max: float = 0.20
    criticality_epsilon: float = 0.10

    # AOS parameters
    aos_alpha: float = 0.20
    aos_epsilon: float = 0.10

    # Elite B diversity threshold (Hamming distance)
    elite_b_min_distance: float = 0.20

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None = None) -> TDCRRAPOConfig:
        mapping = dict(mapping or {})
        mapping.pop("minimization", None)
        mapping.pop("use_evaluation_budget", None)
        mapping.pop("run_id", None)
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in mapping.items() if k in allowed}
        return cls(**filtered)


@dataclass(frozen=True)
class Candidate:
    permutation: tuple[int, ...]
    cost: float
    routes: tuple[tuple[int, ...], ...]
    evaluations: tuple[tuple[float, float, float], ...]


class EvaluationLedger:
    """Strict accounting ledger for TDVRP function evaluations.
    1 call to problem.evaluate_permutation = 1 NFE.
    """
    def __init__(self, problem: TDVRPProblem, limit: int):
        self.problem = problem
        self.limit = limit
        self.nfe = 0
        self.spending: dict[str, int] = {
            "initialization": 0,
            "aro": 0,
            "poa": 0,
            "elite": 0,
            "path_relink": 0,
        }
        self.best: Candidate | None = None
        self.best_nfe: int = 0
        self.convergence: list[float] = []

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.nfe)

    def evaluate(self, permutation: tuple[int, ...] | list[int], phase: str) -> Candidate:
        if self.nfe >= self.limit:
            raise RuntimeError(f"NFE budget exhausted: {self.nfe}/{self.limit}")
        perm_tuple = tuple(int(c) for c in permutation)
        self.nfe += 1
        self.spending[phase] = self.spending.get(phase, 0) + 1

        cost, routes = self.problem.evaluate_permutation(perm_tuple)
        details = tuple(self.problem.route_cost(r) for r in routes)
        candidate = Candidate(perm_tuple, cost, routes, details)

        if self.best is None or candidate.cost < self.best.cost:
            self.best = candidate
            self.best_nfe = self.nfe

        self.convergence.append(self.best.cost)
        return candidate


class SearchRunTDCRRAPO:
    """Execution engine for one run of TD-CR-RAPO."""

    def __init__(
        self,
        problem: TDVRPProblem,
        config: TDCRRAPOConfig,
        seed: int,
        observer: Any = None,
        initial_population: Any = None,
    ):
        self.problem = problem
        self.config = config
        self.seed = seed
        self.observer = observer
        self.initial_population = initial_population

        self.rng = np.random.default_rng(seed)
        self.ledger = EvaluationLedger(problem, config.max_function_evaluations)
        self.population: list[Candidate] = []
        self.archive: Candidate | None = None

        self.generation = 0
        self.stagnation_nfe = 0

        # Stagnation thresholds
        self.s1 = max(1, int(round(config.s1_fraction * config.max_function_evaluations)))
        self.s2 = max(2, int(round(config.s2_fraction * config.max_function_evaluations)))

        # Elite archive: Elite A (best) and Elite B (diverse high-quality)
        self.elite_a: Candidate | None = None
        self.elite_b: Candidate | None = None

        # AOS operators & statistics
        self.aos_operators = ["relocate", "swap", "two_opt", "two_opt_star"]
        self.aos_scores = {op: 1.0 for op in self.aos_operators}
        self.aos_usage = {op: 0 for op in self.aos_operators}
        self.aos_success = {op: 0 for op in self.aos_operators}

        # Efficiency tracking: improvement / NFE
        self.aro_improvement = 0.0
        self.poa_improvement = 0.0
        self.elite_improvement = 0.0

        # Diagnostic accounting counters
        self.criticality_calls = 0
        self.poa_small_calls = 0
        self.poa_large_calls = 0
        self.poa_success = 0
        self.poa_global_best = 0
        self.regret_reconstruction_calls = 0
        self.regret_success = 0

        # Warmup limit
        self.warmup_nfe_limit = int(round(config.warmup_nfe_fraction * config.max_function_evaluations))

    def initialize(self):
        """Initialize population using canonical random keys."""
        if self.initial_population is not None:
            for perm in self.initial_population:
                if not self.ledger.remaining:
                    break
                cand = self.ledger.evaluate(tuple(perm), "initialization")
                self.population.append(cand)
                if self.archive is None or cand.cost < self.archive.cost:
                    self.archive = cand
        else:
            keys = self.rng.random((self.config.population_size, self.problem.dimension))
            for row in keys:
                if not self.ledger.remaining:
                    break
                cand = self.ledger.evaluate(from_keys(row), "initialization")
                self.population.append(cand)
                if self.archive is None or cand.cost < self.archive.cost:
                    self.archive = cand

        self.update_elites()

    def update_elites(self):
        """Update Elite A (best) and Elite B (diverse high-quality)."""
        if not self.population:
            return
        sorted_pop = sorted(self.population, key=lambda c: c.cost)
        self.elite_a = sorted_pop[0]

        # Find Elite B: best candidate with Hamming distance >= threshold from Elite A
        dim = self.problem.dimension
        a_perm = np.array(self.elite_a.permutation)
        best_b = None
        for cand in sorted_pop[1:]:
            b_perm = np.array(cand.permutation)
            hamm = np.mean(a_perm != b_perm)
            if hamm >= self.config.elite_b_min_distance:
                best_b = cand
                break
        self.elite_b = best_b if best_b is not None else (sorted_pop[1] if len(sorted_pop) > 1 else self.elite_a)

    def _aro_explore(self, parent_idx: int) -> bool:
        """ARO Global Search Move.
        Continuous ARO proposal via rank keys or broad perturbation operators.
        Consumes exactly 1 NFE via canonical split & evaluator.
        """
        if not self.ledger.remaining:
            return False

        parent = self.population[parent_idx]
        n = self.problem.dimension
        op_type = int(self.rng.integers(0, 5))

        if op_type == 0:
            # Continuous ARO proposal adapted via rank keys
            from algorithms.aro import ARO
            positions = np.array([rank_keys(x.permutation) for x in self.population])
            values = np.array([x.cost for x in self.population])
            max_iter = max(1, (self.config.max_function_evaluations - self.config.population_size) // self.config.population_size)
            proposal = next(ARO.propose(positions, values, parent_idx, min(self.generation, max_iter - 1), max_iter, self.rng))
            cand_perm = from_keys(np.clip(proposal, 0, 1))
            if cand_perm == parent.permutation:
                # Diversify if tied
                p_list = list(cand_perm)
                self.rng.shuffle(p_list)
                cand_perm = tuple(p_list)
        elif op_type == 1:
            # Guided order injection from a good peer
            peers = [x for i, x in enumerate(self.population) if i != parent_idx]
            peers.sort(key=lambda x: x.cost)
            guide = peers[int(self.rng.integers(max(1, (len(peers) + 1) // 2)))]
            low = min(n, max(2, n // 5))
            high = min(n, max(low, n // 2))
            width = int(self.rng.integers(low, high + 1))
            start = int(self.rng.integers(n - width + 1))
            block = list(guide.permutation[start : start + width])
            block_set = set(block)
            remainder = [c for c in parent.permutation if c not in block_set]
            ins_pos = int(self.rng.integers(len(remainder) + 1))
            cand_perm = tuple(remainder[:ins_pos] + block + remainder[ins_pos:])
        elif op_type == 2:
            # Multi-swap
            p_list = list(parent.permutation)
            swaps = max(2, n // 3)
            indices = self.rng.choice(n, size=min(n, 2 * swaps), replace=False)
            for a, b in zip(indices[::2], indices[1::2]):
                p_list[a], p_list[b] = p_list[b], p_list[a]
            cand_perm = tuple(p_list)
        elif op_type == 3:
            # Segment exchange
            p_list = list(parent.permutation)
            width = max(2, n // 4)
            if n > 2 * width:
                a = int(self.rng.integers(n - 2 * width + 1))
                b = int(self.rng.integers(a + width, n - width + 1))
                p_list[a : a + width], p_list[b : b + width] = p_list[b : b + width], p_list[a : a + width]
            else:
                a, b = self.rng.choice(n, 2, replace=False)
                p_list[a], p_list[b] = p_list[b], p_list[a]
            cand_perm = tuple(p_list)
        else:
            # Large inversion
            p_list = list(parent.permutation)
            length = int(self.rng.integers(max(3, n // 2), n + 1))
            a = int(self.rng.integers(n - length + 1))
            p_list[a : a + length] = reversed(p_list[a : a + length])
            cand_perm = tuple(p_list)

        # Evaluate candidate (+1 NFE)
        cand = self.ledger.evaluate(cand_perm, "aro")

        improved = cand.cost < parent.cost
        if improved:
            delta = parent.cost - cand.cost
            self.aro_improvement += delta
            self.population[parent_idx] = cand

        if cand.cost < self.archive.cost:
            self.archive = cand
            self.stagnation_nfe = 0
        else:
            self.stagnation_nfe += 1

        return improved

    def _poa_reconstruct(self, parent_idx: int) -> bool:
        """POA Large-Neighborhood Reconstruction.
        Radius selected by stagnation & efficiency:
        - POA-SMALL: r = ceil(0.05n .. 0.10n)
        - POA-LARGE: r = ceil(0.10n .. 0.20n)
        Removal probability proportional to (epsilon + C_i) from TD-Criticality.
        Reconstruction via TD-aware Regret-2 insertion with 0-NFE heuristic pre-screening.
        Evaluates 1 candidate via canonical Split & Evaluator = 1 NFE.
        """
        if not self.ledger.remaining:
            return False

        parent = self.population[parent_idx]
        n = self.problem.dimension

        # 1. Compute TD-Criticality (0 NFE, uses cached route trace)
        self.criticality_calls += 1
        crit = compute_customer_criticality(
            self.problem, parent.routes, parent.evaluations
        )

        # 2. Determine perturbation radius r
        if self.stagnation_nfe >= self.s2:
            self.poa_large_calls += 1
            low, high = self.config.poa_large_min, self.config.poa_large_max
        else:
            self.poa_small_calls += 1
            low, high = self.config.poa_small_min, self.config.poa_small_max

        r_frac = float(self.rng.uniform(low, high))
        r = max(1, min(n - 1, int(np.ceil(r_frac * n))))

        # 3. Critical Customer Removal
        customers = list(parent.permutation)
        probs = compute_removal_probabilities(crit, customers, self.config.criticality_epsilon)
        removed_indices = set(
            int(idx) for idx in self.rng.choice(len(customers), size=r, replace=False, p=probs)
        )
        removed_customers = [customers[i] for i in sorted(removed_indices)]
        partial_perm = [c for i, c in enumerate(customers) if i not in removed_indices]

        # 4. TD-aware Regret-2 Reconstruction (0 NFE pre-screening)
        self.regret_reconstruction_calls += 1
        reconstructed = self._td_aware_regret_insertion(partial_perm, removed_customers)

        # 5. Canonical Split Evaluation (+1 NFE)
        cand = self.ledger.evaluate(tuple(reconstructed), "poa")

        improved = cand.cost < parent.cost
        is_global = cand.cost < self.archive.cost

        if improved:
            self.poa_success += 1
            self.regret_success += 1
            delta = parent.cost - cand.cost
            self.poa_improvement += delta
            self.population[parent_idx] = cand

        if is_global:
            self.poa_global_best += 1
            self.archive = cand
            self.stagnation_nfe = 0
        else:
            self.stagnation_nfe += 1

        return improved

    def _td_aware_regret_insertion(
        self,
        partial_perm: list[int],
        removed_customers: list[int],
    ) -> list[int]:
        """TD-Aware Regret-2 Insertion with Elite B fragment preference.
        Pre-screens insertion positions using local marginal arc deltas at 0 NFE.
        """
        perm = list(partial_perm)
        unassigned = list(removed_customers)
        arcs = getattr(self.problem, "arcs", {})
        services = getattr(self.problem, "services", (0,) * (self.problem.dimension + 1))

        # Build edge adjacency set from Elite B for fragment preservation bonus
        elite_b_edges: set[tuple[int, int]] = set()
        if self.elite_b is not None and self.elite_b.routes:
            for r in self.elite_b.routes:
                if not r:
                    continue
                elite_b_edges.add((0, r[0]))
                for i in range(len(r) - 1):
                    elite_b_edges.add((r[i], r[i + 1]))
                elite_b_edges.add((r[-1], 0))

        while unassigned:
            if len(unassigned) == 1:
                c = unassigned.pop()
                best_pos = self._best_insertion_position(perm, c, arcs, services, elite_b_edges)
                perm.insert(best_pos, c)
                break

            # Regret-2 calculation for all remaining unassigned customers
            best_regret = -1.0
            best_c_idx = 0
            best_c_pos = 0

            for idx, c in enumerate(unassigned):
                # Calculate cost at all insertion positions
                costs: list[tuple[float, int]] = []
                for pos in range(len(perm) + 1):
                    cost_delta = self._insertion_cost_delta(perm, pos, c, arcs, services, elite_b_edges)
                    costs.append((cost_delta, pos))

                costs.sort(key=lambda x: x[0])
                c_best_cost, c_best_pos = costs[0]
                c_second_cost = costs[1][0] if len(costs) > 1 else c_best_cost

                regret = c_second_cost - c_best_cost
                if regret > best_regret:
                    best_regret = regret
                    best_c_idx = idx
                    best_c_pos = c_best_pos

            c_star = unassigned.pop(best_c_idx)
            perm.insert(best_c_pos, c_star)

        return perm

    def _insertion_cost_delta(
        self,
        perm: list[int],
        pos: int,
        c: int,
        arcs: dict,
        services: tuple,
        elite_b_edges: set[tuple[int, int]],
    ) -> float:
        """Approximate local arc detour for inserting customer c at pos."""
        u = perm[pos - 1] if pos > 0 else 0
        v = perm[pos] if pos < len(perm) else 0

        # Base service time
        cost = float(services[c])

        # Arc travel time approximations
        if arcs:
            arc_uc = arcs.get((u, c))
            arc_cv = arcs.get((c, v))
            arc_uv = arcs.get((u, v)) if (u != 0 or v != 0) else None

            # Evaluate at nominal reference time 0.0 or free-flow
            tt_uc = (arc_uc(0.0) - 0.0) if arc_uc is not None else 10.0
            tt_cv = (arc_cv(0.0) - 0.0) if arc_cv is not None else 10.0
            tt_uv = (arc_uv(0.0) - 0.0) if arc_uv is not None else 0.0
            cost += (tt_uc + tt_cv - tt_uv)
        else:
            cost += 20.0

        # Elite B bonus: if (u, c) or (c, v) is an edge in Elite B, reduce cost
        if (u, c) in elite_b_edges:
            cost *= 0.90
        if (c, v) in elite_b_edges:
            cost *= 0.90

        return cost

    def _best_insertion_position(
        self,
        perm: list[int],
        c: int,
        arcs: dict,
        services: tuple,
        elite_b_edges: set[tuple[int, int]],
    ) -> int:
        best_pos = 0
        best_cost = float("inf")
        for pos in range(len(perm) + 1):
            cost = self._insertion_cost_delta(perm, pos, c, arcs, services, elite_b_edges)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        return best_pos

    def _elite_aos_refine(self) -> bool:
        """Elite AOS Refinement.
        Selects an operator from (relocate, swap, 2-opt, 2-opt*) using EMA reward scores.
        Hard-capped at <= 20% total NFE.
        Evaluates candidate via canonical Split & Evaluator = 1 NFE.
        """
        if not self.ledger.remaining or not self.population:
            return False

        # Guard: Elite NFE <= 20%
        if self.ledger.spending["elite"] >= int(self.config.max_elite_share * self.config.max_function_evaluations):
            return False

        # Target Elite A (best)
        parent = self.elite_a if self.elite_a is not None else min(self.population, key=lambda c: c.cost)
        parent_idx = self.population.index(parent) if parent in self.population else 0

        # Operator Selection via softmax/proportional with exploration floor
        epsilon = self.config.aos_epsilon
        raw_scores = np.array([max(0.0, self.aos_scores[op]) for op in self.aos_operators], dtype=float)
        score_sum = np.sum(raw_scores)
        if score_sum > 1e-9:
            base_probs = raw_scores / score_sum
        else:
            base_probs = np.ones(len(self.aos_operators)) / len(self.aos_operators)

        probs = (1.0 - len(self.aos_operators) * epsilon) * base_probs + epsilon
        probs /= np.sum(probs)

        op = str(self.rng.choice(self.aos_operators, p=probs))
        self.aos_usage[op] += 1

        # Apply operator
        if op == "relocate":
            cand_perm = route_aware_relocate(parent.permutation, parent.routes, self.rng)
        elif op == "swap":
            cand_perm = route_aware_swap(parent.permutation, parent.routes, self.rng)
        elif op == "two_opt":
            cand_perm = route_aware_two_opt(parent.permutation, parent.routes, self.rng)
        elif op == "two_opt_star":
            cand_perm = route_aware_two_opt_star(parent.permutation, parent.routes, self.rng)
        else:
            cand_perm = parent.permutation

        cand = self.ledger.evaluate(tuple(cand_perm), "elite")

        improved = cand.cost < parent.cost
        is_global = cand.cost < self.archive.cost

        # Relative improvement reward per NFE
        denom = max(abs(parent.cost), 1e-6)
        reward = max(0.0, (parent.cost - cand.cost) / denom)
        if is_global:
            reward *= 2.0

        # EMA update
        alpha = self.config.aos_alpha
        self.aos_scores[op] = (1.0 - alpha) * self.aos_scores[op] + alpha * reward

        if improved:
            self.aos_success[op] += 1
            delta = parent.cost - cand.cost
            self.elite_improvement += delta
            self.population[parent_idx] = cand

        if is_global:
            self.archive = cand
            self.stagnation_nfe = 0
        else:
            self.stagnation_nfe += 1

        return improved

    def run(self) -> OptimizationResult:
        start_time = time.perf_counter()
        self.initialize()

        # Search loop
        while self.ledger.remaining and (
            self.config.max_iterations is None or self.generation < self.config.max_iterations
        ):
            self.generation += 1

            # Adaptive NFE Controller scheduling
            nfe_used = self.ledger.nfe
            rem = self.ledger.remaining
            max_nfe = self.config.max_function_evaluations

            aro_nfe = self.ledger.spending["aro"]
            poa_nfe = self.ledger.spending["poa"]
            elite_nfe = self.ledger.spending["elite"]

            # Required NFE to satisfy hard minimum constraints by end of run
            aro_needed = max(0, int(np.ceil(self.config.min_aro_share * max_nfe)) - aro_nfe)
            poa_needed = max(0, int(np.ceil(self.config.min_poa_share * max_nfe)) - poa_nfe)

            in_warmup = nfe_used < self.warmup_nfe_limit

            # Strict deficit guards: if remaining evaluations are needed to hit minimums, enforce them
            if aro_needed >= rem:
                action = "aro"
            elif poa_needed >= rem:
                action = "poa"
            elif in_warmup:
                # Warmup: ~70% ARO, ~30% POA-SMALL, 0% Elite
                action = "aro" if self.rng.random() < self.config.warmup_aro_target else "poa"
            else:
                # Post-warmup: dynamic scheduling by efficiency
                e_aro = self.aro_improvement / max(1, aro_nfe)
                e_poa = self.poa_improvement / max(1, poa_nfe)
                e_elite = self.elite_improvement / max(1, elite_nfe)

                max_e = max(1e-6, e_aro, e_poa, e_elite)
                w_aro = 0.50 * (1.0 + e_aro / max_e)
                w_poa = 0.30 * (1.0 + e_poa / max_e)
                w_elite = 0.20 * (1.0 + e_elite / max_e)

                # Penalize or boost to stay on track with targets
                if aro_nfe / nfe_used < self.config.min_aro_share:
                    w_aro *= 3.0
                if poa_nfe / nfe_used < self.config.min_poa_share:
                    w_poa *= 3.0

                # Elite hard cap guard: Elite <= 20%
                if elite_nfe >= int(self.config.max_elite_share * max_nfe):
                    w_elite = 0.0

                weights = [w_aro, w_poa, w_elite]
                w_sum = sum(weights)
                if w_sum > 0:
                    probs = [w / w_sum for w in weights]
                    action = str(self.rng.choice(["aro", "poa", "elite"], p=probs))
                else:
                    action = "aro"

            # Execute chosen action
            target_idx = int(self.rng.integers(len(self.population)))
            if action == "aro":
                self._aro_explore(target_idx)
            elif action == "poa":
                self._poa_reconstruct(target_idx)
            else:
                self._elite_aos_refine()

            self.update_elites()

            if self.observer is not None:
                self.observer.on_generation(
                    self.population,
                    self.archive.cost if self.archive else 0.0,
                    self.generation,
                    self.ledger.nfe,
                    time.perf_counter() - start_time,
                )

        runtime = time.perf_counter() - start_time
        best = self.archive if self.archive is not None else self.population[0]

        # Invariant verification
        flat_custs = sorted(c for r in best.routes for c in r)
        if flat_custs != list(range(1, self.problem.dimension + 1)):
            raise AssertionError(f"Archive violates customer coverage: found {len(flat_custs)}, expected {self.problem.dimension}")
        if any(not r or sum(self.problem.demands[c] for c in r) > self.problem.capacity for r in best.routes):
            raise AssertionError("Archive violates capacity constraint")

        # NFE accounting & metadata
        tot_nfe = self.ledger.nfe
        aro_nfe = self.ledger.spending["aro"]
        poa_nfe = self.ledger.spending["poa"]
        elite_nfe = self.ledger.spending["elite"]
        path_relink_nfe = self.ledger.spending["path_relink"]

        aro_share = aro_nfe / tot_nfe if tot_nfe else 0.0
        poa_share = poa_nfe / tot_nfe if tot_nfe else 0.0
        elite_share = elite_nfe / tot_nfe if tot_nfe else 0.0

        # Architecture validation: ARO share >= 40% (Section 5)
        arch_valid = (aro_share >= 0.40) and (elite_share <= 0.20)
        arch_status = "PASS" if arch_valid else "ARCHITECTURE_VIOLATION"

        metadata = {
            "objective": "Duration",
            "units": "seconds",
            "feasible": True,
            "encoding": "permutation + shared minimum-duration Split",
            "departure_domain": [0, self.problem.horizon],
            "travel_time_domain": "route departure restricted to nonempty T(R); no extrapolation",
            "model_version": 4,
            "vehicle_limit_K": self.problem.vehicle_limit,
            "penalty_weights": {
                "miss": self.problem.alpha_miss,
                "dup": self.problem.alpha_dup,
                "cap": self.problem.alpha_cap,
                "fleet": self.problem.alpha_fleet,
                "M0": self.problem.penalty_fixed,
                "M1": self.problem.penalty_multiplier,
            },
            "evaluations": tot_nfe,
            "completed_iterations": self.generation,
            "total_nfe": tot_nfe,
            "aro_nfe": aro_nfe,
            "poa_nfe": poa_nfe,
            "elite_nfe": elite_nfe,
            "path_relink_nfe": path_relink_nfe,
            "aro_share": aro_share,
            "poa_share": poa_share,
            "elite_share": elite_share,
            "criticality_calls": self.criticality_calls,
            "poa_small_calls": self.poa_small_calls,
            "poa_large_calls": self.poa_large_calls,
            "poa_success": self.poa_success,
            "poa_global_best": self.poa_global_best,
            "regret_reconstruction_calls": self.regret_reconstruction_calls,
            "regret_success": self.regret_success,
            "elite_operator_usage": dict(self.aos_usage),
            "elite_operator_success": dict(self.aos_success),
            "nfe_by_phase": dict(self.ledger.spending),
            "architecture_valid": arch_valid,
            "architecture_status": arch_status,
            "stagnation_final": self.stagnation_nfe,
            "use_exact_parent_dedup": self.config.use_exact_parent_dedup,
        }

        res = OptimizationResult(
            algorithm="TD-CR-RAPO",
            benchmark=self.problem.name,
            seed=self.seed,
            best_fitness=best.cost,
            best_solution=rank_keys(best.permutation).tolist(),
            routes=[[0, *r, 0] for r in best.routes],
            departure_times=[e[1] for e in best.evaluations],
            return_times=[e[2] for e in best.evaluations],
            route_durations=[e[0] for e in best.evaluations],
            route_loads=[sum(self.problem.demands[c] for c in r) for r in best.routes],
            convergence_curve=self.ledger.convergence,
            runtime_seconds=runtime,
            population_size=self.config.population_size,
            max_iterations=self.generation,
            nfe=tot_nfe,
            metadata=metadata,
            dimension=self.problem.dimension,
            hybrid_type="ARO+POA+CR",
        )
        res.best_permutation = list(best.permutation)
        return res


class TDCRRAPO:
    """TD-CR-RAPO Solver.
    Full name: Time-Dependent Criticality-Guided ARO–POA with Adaptive Large-Neighborhood Reconstruction.
    """
    name = "TD-CR-RAPO"
    full_name = "Time-Dependent Criticality-Guided ARO–POA with Adaptive Large-Neighborhood Reconstruction"
    hybrid_type = "ARO+POA+CR"

    def optimize(
        self,
        problem: TDVRPProblem,
        config: dict[str, Any] | None = None,
        seed: int | None = None,
        observer: Any = None,
        initial_population: Any = None,
    ) -> OptimizationResult:
        cfg = TDCRRAPOConfig.from_mapping(config)
        actual_seed = 42 if seed is None else int(seed)
        runner = SearchRunTDCRRAPO(
            problem=problem,
            config=cfg,
            seed=actual_seed,
            observer=observer,
            initial_population=initial_population,
        )
        return runner.run()


def split_decoder(permutation, instance):
    """Tách giant tour thành các route liên tiếp thỏa capacity cho V3 mới."""
    from core.models import Solution

    routes = []
    current = [instance.depot]
    load = 0.0
    for customer_id in permutation:
        customer_id = int(customer_id)
        demand = instance.demands[customer_id]
        if current != [instance.depot] and load + demand > instance.vehicle_capacity + 1e-9:
            current.append(instance.depot)
            routes.append(current)
            current = [instance.depot]
            load = 0.0
        current.append(customer_id)
        load += demand
    if len(current) > 1:
        current.append(instance.depot)
        routes.append(current)
    return Solution(routes=routes)


def compute_customer_criticality(candidate, instance, graph, traffic):
    """Tính criticality zero-NFE; không gọi evaluator trong lúc prescreen."""
    del graph, traffic
    maximum_service = max(instance.service_times[1:], default=1.0)
    values = {}
    for position, customer_id in enumerate(candidate.permutation):
        demand_score = instance.demands[customer_id] / max(instance.vehicle_capacity, 1e-9)
        service_score = instance.service_times[customer_id] / max(maximum_service, 1e-9)
        boundary_score = float(position in {0, len(candidate.permutation) - 1})
        values[int(customer_id)] = 0.45 * demand_score + 0.45 * service_score + 0.10 * boundary_score
    return values


def optimize(instance, graph, traffic, config=None, seed=None, **kwargs):
    """API tương thích cho test và caller của data layer HCMC hiện tại.

    Bản TD-CR-RAPO nguồn dùng backend random-key cũ; caller Poryos hiện tại
    được chuyển sang V3 dùng chung evaluator ``core.data``.
    """
    from algorithms.td_cr_rapo_v3 import TDCRRAPOV3Config, optimize as optimize_v3

    source_config = config or TDCRRAPOConfig()
    v3_config = TDCRRAPOV3Config(
        population_size=source_config.population_size,
        max_function_evaluations=source_config.max_function_evaluations,
        max_iterations=source_config.max_iterations,
    )
    result = optimize_v3(
        instance,
        graph,
        traffic,
        v3_config,
        seed=seed,
        start_time=kwargs.get("start_time", 0.0),
        departure_time_mode=kwargs.get("departure_time_mode", "fixed"),
        departure_time_step=kwargs.get("departure_time_step"),
    )

    # V3 cung cấp property tương thích để benchmark cũ đọc cùng kết quả.
    return result
