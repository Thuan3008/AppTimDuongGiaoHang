"""TD-RAPO-Pufferfish: Hybrid Artificial Rabbits and Pufferfish Optimization for TDVRP.

Proposed Metaheuristic Architecture:
- Component 1: Artificial Rabbits Optimization (ARO-D, Wang et al. 2022)
  - Detour Foraging (macro-exploration) via Order Crossover (OX) with peer rabbits when energy factor alpha > 1.
  - Random Hiding (localized exploitation) via burrow neighborhood moves (relocate, inversion, swap) when energy alpha <= 1.
  - Transition governed by energy factor alpha(t, r) = 4.0 * (1.0 - t/T) * ln(1.0 / r).
- Component 2: Pufferfish Optimization Algorithm (PF-D, Al-Baik et al. 2024)
  - Phase 1: Predator Attack (directional exploration) via Path-Relinking towards strictly better peers (CP_i).
    Step size K = max(1, min(d_H, floor(I * r * d_H))), I in {1, 2}, r ~ U(0, 1).
  - Phase 2: Pufferfish Defense (contracting exploitation) via bounded contiguous segment reversal
    with hyperbolically shrinking radius R(t) = max(2, min(n, floor(W_max / t) + 1)).
- Adaptive Deficit-Balancing NFE Controller:
  - Coordinates ARO actions (1 NFE) and atomic Pufferfish cycles (1 Attack + 1 Defense = 2 NFE).
  - Enforces symmetric Pufferfish spending: pf_attack == pf_defense.
  - Odd remaining budget (remaining == 1) is absorbed gracefully by an ARO action.
  - Strict greedy acceptance (<) at every evaluation step.
  - 100% canonical NFE accounting via EvaluationLedger.
- Reproducibility & Checkpointing:
  - Provides state_dict() and load_state_dict() for seamless intra-run resumption.
"""
from __future__ import annotations
from dataclasses import dataclass, fields
import math
import time
from typing import Any, Sequence
import numpy as np

from core.tdvrp_common import TDVRPProblem, OptimizationResult
from algorithms.common.ledger import EvaluationLedger, Candidate, build_optimization_result
from algorithms.rapo_operators import from_keys
from algorithms.operators.permutation import (
    swap_random,
    relocate_random,
    inversion_random,
    order_crossover,
)
from algorithms.hcmc_poa_c import (
    HasCost,
    PufferfishOptimization,
    validate_customer_permutation,
)


@dataclass
class TDRAPOPufferfishConfig:
    """Configuration dataclass for TD-RAPO-Pufferfish proposed algorithm."""
    population_size: int = 10
    max_function_evaluations: int = 1000
    max_iterations: int | None = None
    target_aro_share: float = 0.50
    target_pf_share: float = 0.50

    @classmethod
    def from_mapping(cls, config: dict[str, Any] | None = None) -> tuple[TDRAPOPufferfishConfig, int]:
        values = dict(config or {})
        values.pop("minimization", None)
        values.pop("use_evaluation_budget", None)
        run_id_val = values.pop("run_id", 1)
        try:
            run_id = int(run_id_val)
        except (ValueError, TypeError):
            run_id = 1

        if "pop_size" in values:
            val = values.pop("pop_size")
            if "population_size" not in values:
                values["population_size"] = val
        if "max_nfe" in values:
            val = values.pop("max_nfe")
            if "max_function_evaluations" not in values:
                values["max_function_evaluations"] = val
        if "target_nfe" in values:
            val = values.pop("target_nfe")
            if "max_function_evaluations" not in values:
                values["max_function_evaluations"] = val

        allowed = {f.name for f in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown TD-RAPO-Pufferfish config keys: {sorted(unknown)}")

        cfg = cls(**values)

        if (
            not isinstance(cfg.population_size, (int, np.integer))
            or isinstance(cfg.population_size, bool)
            or cfg.population_size < 2
        ):
            raise ValueError(f"population_size must be an integer >= 2, got {cfg.population_size!r}")

        if (
            not isinstance(cfg.max_function_evaluations, (int, np.integer))
            or isinstance(cfg.max_function_evaluations, bool)
            or cfg.max_function_evaluations < 1
        ):
            raise ValueError(
                f"max_function_evaluations must be a positive integer, got {cfg.max_function_evaluations!r}"
            )

        if cfg.max_iterations is not None:
            if (
                not isinstance(cfg.max_iterations, (int, np.integer))
                or isinstance(cfg.max_iterations, bool)
                or cfg.max_iterations < 0
            ):
                raise ValueError(
                    f"max_iterations must be None or a non-negative integer, got {cfg.max_iterations!r}"
                )

        if not (0.0 <= cfg.target_aro_share <= 1.0):
            raise ValueError(f"target_aro_share must be in [0, 1], got {cfg.target_aro_share}")
        if not (0.0 <= cfg.target_pf_share <= 1.0):
            raise ValueError(f"target_pf_share must be in [0, 1], got {cfg.target_pf_share}")

        return cfg, run_id


def aro_generate_candidate(
    parent_perm: Sequence[int] | tuple[int, ...],
    population: Sequence[HasCost],
    parent_idx: int,
    t_ratio: float,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Generates an Artificial Rabbits Optimization candidate permutation.

    Mechanics:
    - Calculates energy factor alpha(t, r) = 4.0 * (1.0 - t_ratio) * ln(1.0 / r).
    - If energy > 1.0: Detour Foraging (exploration guided by peer rabbit).
      Performs Order Crossover (OX) with a randomly selected peer.
      If child equals parent, applies swap_random fallback to ensure non-zero exploration.
    - If energy <= 1.0: Random Hiding (localized exploitation).
      Samples uniformly from burrow neighborhood moves: {"relocate", "inversion", "swap"}.
    """
    parent_tuple = validate_customer_permutation(parent_perm)
    n = len(parent_tuple)
    if n <= 1:
        return parent_tuple

    r = max(1e-15, float(rng.random()))
    energy = 4.0 * (1.0 - min(1.0, max(0.0, float(t_ratio)))) * math.log(1.0 / r)
    pop_size = len(population)

    if energy > 1.0:
        # Detour foraging
        if pop_size > 1:
            peer_idx = int(rng.integers(pop_size - 1))
            if peer_idx >= parent_idx:
                peer_idx += 1
            peer_perm = getattr(population[peer_idx], "permutation", population[peer_idx])
            cand_perm = order_crossover(parent_tuple, peer_perm, rng)
            if cand_perm == parent_tuple:
                cand_perm = swap_random(cand_perm, rng)
            return cand_perm
        return swap_random(parent_tuple, rng)
    else:
        # Random hiding
        op = str(rng.choice(["relocate", "inversion", "swap"]))
        if op == "relocate":
            return relocate_random(parent_tuple, rng)
        elif op == "inversion":
            return inversion_random(parent_tuple, rng)
        else:
            return swap_random(parent_tuple, rng)


class TDRAPOPufferfish:
    """Proposed RAPO: Hybrid Artificial Rabbits and Pufferfish Optimization Algorithm."""
    name = "RAPO (Proposed)"
    registry_key = "td-rapo-pufferfish"

    def __init__(self, **kwargs: Any):
        self.default_config: dict[str, Any] = dict(kwargs)
        self._state: dict[str, Any] | None = None

    def state_dict(self) -> dict[str, Any]:
        """Returns serializable dictionary of current optimization state."""
        if self._state is None:
            return {}
        return self._state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Loads state dictionary for checkpoint resumption."""
        self._state = state

    def optimize(
        self,
        problem: TDVRPProblem,
        config: dict[str, Any] | None = None,
        seed: int | None = None,
        initial_population: Sequence[Sequence[int]] | None = None,
        checkpoint_callback=None,
        initial_state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Executes TD-RAPO-Pufferfish on the given TDVRPProblem instance."""
        if not isinstance(problem, TDVRPProblem):
            raise TypeError("Expected TDVRPProblem instance")

        merged = dict(self.default_config)
        if config:
            merged.update(config)
        merged.update(kwargs)

        cfg, run_id = TDRAPOPufferfishConfig.from_mapping(merged)
        rng = np.random.default_rng(seed)
        start_time = time.perf_counter()

        b_budget = cfg.max_function_evaluations
        p_effective = min(cfg.population_size, b_budget)

        ledger = EvaluationLedger(problem, b_budget)
        population: list[Candidate] = []

        generation = 0
        aro_accepted = 0
        pf_attack_accepted = 0
        pf_defense_accepted = 0
        elapsed_wall_offset = 0.0

        state_to_load = initial_state or self._state
        if state_to_load:
            # Resume from state
            generation = state_to_load["generation"]
            aro_accepted = state_to_load.get("aro_accepted", 0)
            pf_attack_accepted = state_to_load.get("pf_attack_accepted", 0)
            pf_defense_accepted = state_to_load.get("pf_defense_accepted", 0)
            elapsed_wall_offset = state_to_load.get("elapsed_wall_seconds", 0.0)

            # Restore RNG state
            if "rng_state" in state_to_load:
                rng.bit_generator.state = state_to_load["rng_state"]

            # Restore Ledger
            ledger.nfe = state_to_load["ledger_nfe"]
            ledger.spending = dict(state_to_load["ledger_spending"])
            ledger.best_nfe = state_to_load["ledger_best_nfe"]
            ledger.convergence = list(state_to_load.get("ledger_convergence", []))
            ledger.search_log = list(state_to_load.get("ledger_search_log", []))

            # Rebuild Candidates
            for item in state_to_load["population"]:
                cand = Candidate(
                    permutation=tuple(item["permutation"]),
                    cost=float(item["cost"]),
                    routes=tuple(tuple(r) for r in item["routes"]),
                    evaluations=tuple(tuple(e) for e in item["evaluations"]),
                )
                population.append(cand)

            b_item = state_to_load["best_candidate"]
            ledger.best = Candidate(
                permutation=tuple(b_item["permutation"]),
                cost=float(b_item["cost"]),
                routes=tuple(tuple(r) for r in b_item["routes"]),
                evaluations=tuple(tuple(e) for e in b_item["evaluations"]),
            )
        else:
            # Khởi tạo mới vẫn giữ random-keys cũ nếu không truyền P0 chung.
            if initial_population is None:
                initial_permutations = [
                    from_keys(row)
                    for row in rng.random((p_effective, problem.dimension))
                ]
            else:
                if len(initial_population) != p_effective:
                    raise ValueError(
                        "initial_population phải có đúng population_size hoán vị"
                    )
                initial_permutations = [
                    tuple(int(customer) for customer in permutation)
                    for permutation in initial_population
                ]

            # Mỗi cá thể khởi tạo vẫn tiêu thụ đúng một NFE như trước.
            for permutation in initial_permutations:
                cand = ledger.evaluate(permutation, phase="initialization")
                population.append(cand)

            if checkpoint_callback:
                self._save_state(generation, population, ledger, rng, time.perf_counter() - start_time,
                                 aro_accepted, pf_attack_accepted, pf_defense_accepted)
                checkpoint_callback(self.state_dict())

        # Early termination
        if b_budget <= p_effective or (cfg.max_iterations is not None and cfg.max_iterations == 0):
            runtime = (time.perf_counter() - start_time) + elapsed_wall_offset
            return build_optimization_result(
                algorithm=self.name,
                problem=problem,
                ledger=ledger,
                seed=seed,
                runtime_seconds=runtime,
                population_size=cfg.population_size,
                max_iterations=0,
                run_id=run_id,
                extra_metadata={
                    "config": cfg.__dict__,
                    "effective_population_size": p_effective,
                    "search_cycles": 0,
                    "aro_nfe": 0,
                    "aro_accepted": 0,
                    "pf_cycles": 0,
                    "pf_nfe": 0,
                    "pf_attack_accepted": 0,
                    "pf_defense_accepted": 0,
                    "target_aro_share": cfg.target_aro_share,
                    "target_pf_share": cfg.target_pf_share,
                    "unused_budget": ledger.remaining,
                },
            )

        b_search = b_budget - p_effective
        total_est_generations = cfg.max_iterations or max(
            1, math.ceil(b_search / max(1.0, 1.5 * p_effective))
        )

        last_ckpt_nfe = ledger.nfe

        # Main hybrid search loop
        while ledger.remaining >= 1 and (cfg.max_iterations is None or generation < cfg.max_iterations):
            generation += 1
            t_ratio = min(1.0, generation / max(1, total_est_generations))

            for i in range(len(population)):
                if ledger.remaining < 1:
                    break

                parent = population[i]

                # Deficit-balancing between ARO (share target_aro_share) and Pufferfish (share target_pf_share)
                w_aro = cfg.target_aro_share
                w_pf = cfg.target_pf_share
                total_w = max(1e-12, w_aro + w_pf)
                target_aro = w_aro / total_w
                target_pf = w_pf / total_w

                s_aro = ledger.spending.get("aro", 0)
                s_pf_att = ledger.spending.get("pf_attack", 0)
                s_pf_def = ledger.spending.get("pf_defense", 0)
                s_pf = s_pf_att + s_pf_def
                s_search = s_aro + s_pf

                can_pf = (ledger.remaining >= 2) and (target_pf > 0.0)
                can_aro = (ledger.remaining >= 1) and (target_aro > 0.0)

                if ledger.remaining == 1:
                    can_pf = False

                if s_search == 0:
                    action = "PF" if (can_pf and target_pf >= target_aro) else "ARO"
                else:
                    def_aro = (target_aro * s_search - s_aro) if can_aro else -float("inf")
                    def_pf = (target_pf * s_search - s_pf) if can_pf else -float("inf")
                    if def_pf >= def_aro and can_pf:
                        action = "PF"
                    elif can_aro:
                        action = "ARO"
                    else:
                        break

                if action == "ARO":
                    # --- ARO Action (Consumes exactly 1 NFE) ---
                    aro_perm = aro_generate_candidate(parent.permutation, population, i, t_ratio, rng)
                    aro_cand = ledger.evaluate(aro_perm, phase="aro")
                    if aro_cand.cost < parent.cost:
                        population[i] = aro_cand
                        aro_accepted += 1

                elif action == "PF":
                    # --- Pufferfish Full Cycle (Consumes exactly 2 NFE) ---
                    # Phase 1: Predator Attack (Exploration towards strictly better peer CP_i)
                    target = PufferfishOptimization.select_target(i, population, rng)
                    attack_perm = PufferfishOptimization.predator_attack(parent.permutation, target.permutation, rng)
                    attack_cand = ledger.evaluate(attack_perm, phase="pf_attack")

                    if attack_cand.cost < parent.cost:
                        state = attack_cand
                        pf_attack_accepted += 1
                    else:
                        state = parent

                    # Phase 2: Pufferfish Defense (Exploitation with shrinking radius R(t))
                    defense_perm = PufferfishOptimization.pufferfish_defense(state.permutation, t=generation, rng=rng)
                    defense_cand = ledger.evaluate(defense_perm, phase="pf_defense")

                    if defense_cand.cost < state.cost:
                        final_state = defense_cand
                        pf_defense_accepted += 1
                    else:
                        final_state = state

                    population[i] = final_state

                # Checkpoint trigger (every 50 NFE or on completion)
                if checkpoint_callback and (ledger.nfe - last_ckpt_nfe >= 50 or ledger.remaining == 0):
                    last_ckpt_nfe = ledger.nfe
                    self._save_state(generation, population, ledger, rng,
                                     (time.perf_counter() - start_time) + elapsed_wall_offset,
                                     aro_accepted, pf_attack_accepted, pf_defense_accepted)
                    checkpoint_callback(self.state_dict())

        # Invariants Check
        init_nfe = ledger.spending.get("initialization", 0)
        aro_nfe = ledger.spending.get("aro", 0)
        att_nfe = ledger.spending.get("pf_attack", 0)
        def_nfe = ledger.spending.get("pf_defense", 0)

        assert ledger.nfe <= b_budget, f"NFE {ledger.nfe} exceeded budget {b_budget}"
        assert att_nfe == def_nfe, f"Asymmetric Pufferfish cycles: attack={att_nfe}, defense={def_nfe}"
        assert (
            ledger.nfe == init_nfe + aro_nfe + att_nfe + def_nfe
        ), f"NFE mismatch: {ledger.nfe} != {init_nfe} + {aro_nfe} + {att_nfe} + {def_nfe}"

        runtime = (time.perf_counter() - start_time) + elapsed_wall_offset

        return build_optimization_result(
            algorithm=self.name,
            problem=problem,
            ledger=ledger,
            seed=seed,
            runtime_seconds=runtime,
            population_size=cfg.population_size,
            max_iterations=generation,
            run_id=run_id,
            extra_metadata={
                "config": cfg.__dict__,
                "effective_population_size": p_effective,
                "search_nfe": aro_nfe + att_nfe + def_nfe,
                "aro_nfe": aro_nfe,
                "aro_accepted": aro_accepted,
                "pf_cycles": att_nfe,
                "pf_nfe": att_nfe + def_nfe,
                "pf_attack_accepted": pf_attack_accepted,
                "pf_defense_accepted": pf_defense_accepted,
                "target_aro_share": cfg.target_aro_share,
                "target_pf_share": cfg.target_pf_share,
                "unused_budget": ledger.remaining,
            },
        )

    def _save_state(
        self,
        generation: int,
        population: list[Candidate],
        ledger: EvaluationLedger,
        rng: np.random.Generator,
        elapsed_wall_seconds: float,
        aro_accepted: int,
        pf_attack_accepted: int,
        pf_defense_accepted: int,
    ) -> None:
        best = ledger.best
        self._state = {
            "algorithm": self.registry_key,
            "generation": generation,
            "elapsed_wall_seconds": elapsed_wall_seconds,
            "aro_accepted": aro_accepted,
            "pf_attack_accepted": pf_attack_accepted,
            "pf_defense_accepted": pf_defense_accepted,
            "rng_state": rng.bit_generator.state,
            "ledger_nfe": ledger.nfe,
            "ledger_spending": dict(ledger.spending),
            "ledger_best_nfe": ledger.best_nfe,
            "ledger_convergence": list(ledger.convergence),
            "ledger_search_log": list(ledger.search_log),
            "population": [
                {
                    "permutation": list(c.permutation),
                    "cost": float(c.cost),
                    "routes": [list(r) for r in c.routes],
                    "evaluations": [list(e) for e in c.evaluations],
                }
                for c in population
            ],
            "best_candidate": {
                "permutation": list(best.permutation) if best else [],
                "cost": float(best.cost) if best else float("inf"),
                "routes": [list(r) for r in best.routes] if best else [],
                "evaluations": [list(e) for e in best.evaluations] if best else [],
            } if best else None,
        }


if __name__ == "__main__":
    from core.tdvrp_common import run_cli
    run_cli(TDRAPOPufferfish)
