"""Iterated Local Search with VND (ILS-VND) for TDVRP.

Trajectory-based metaheuristic combining:
- Perturbation moves: double relocate, block swap, segment exchange
- Variable Neighborhood Descent (VND) with relocate, swap, 2-opt, 2-opt*
- Acceptance criteria (better or simulated annealing threshold)
Strictly:
- perturbed candidate = +1 NFE
- VND neighbor = +1 NFE
No fake population.
"""
from __future__ import annotations
from dataclasses import dataclass, fields
import math
import time
import numpy as np

from core.tdvrp_common import TDVRPProblem, OptimizationResult
from algorithms.common.ledger import EvaluationLedger, Candidate, build_optimization_result
from algorithms.operators import (
    swap_random,
    relocate_random,
    two_opt,
    two_opt_star,
    double_relocate,
    block_swap,
    segment_exchange,
)


@dataclass
class ILSVNDConfig:
    max_function_evaluations: int = 1000
    max_iterations: int | None = None
    temperature_fraction: float = 0.05
    cooling_rate: float = 0.98
    vnd_max_steps: int = 10

    @classmethod
    def from_mapping(cls, config=None):
        values = dict(config or {})
        values.pop("minimization", None)
        values.pop("use_evaluation_budget", None)
        values.pop("run_id", None)
        values.pop("population_size", None)  # trajectory-based
        allowed = {f.name for f in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown ILS-VND config keys: {sorted(unknown)}")
        cfg = cls(**values)
        if cfg.max_function_evaluations < 2:
            raise ValueError("NFE budget must be >= 2")
        return cfg


class ILSVND:
    name = "ILS-VND"

    def optimize(
        self,
        problem: TDVRPProblem,
        config: dict | None = None,
        seed: int | None = None,
        observer=None,
        initial_population=None,
    ) -> OptimizationResult:
        if not isinstance(problem, TDVRPProblem):
            raise TypeError("Expected TDVRPProblem instance")
        c = ILSVNDConfig.from_mapping(config)
        rng = np.random.default_rng(seed)
        ledger = EvaluationLedger(problem, c.max_function_evaluations)
        start_time = time.perf_counter()

        # Initial solution
        if initial_population is not None and len(initial_population) > 0:
            init_perm = tuple(initial_population[0])
        else:
            init_perm = tuple(int(x) for x in rng.permutation(problem.dimension) + 1)

        current = ledger.evaluate(init_perm, "initialization")
        best = current

        def run_vnd(sol: Candidate) -> Candidate:
            """VND local search: relocate -> swap -> two_opt -> two_opt_star."""
            cur = sol
            step = 0
            k = 1
            while k <= 4 and ledger.remaining and step < c.vnd_max_steps:
                step += 1
                if k == 1:
                    nbr_perm = relocate_random(cur.permutation, rng)
                elif k == 2:
                    nbr_perm = swap_random(cur.permutation, rng)
                elif k == 3:
                    nbr_perm = two_opt(cur.permutation, cur.routes, rng)
                else:
                    nbr_perm = two_opt_star(cur.permutation, cur.routes, rng)

                if nbr_perm == cur.permutation:
                    k += 1
                    continue

                nbr_cand = ledger.evaluate(nbr_perm, "vnd")
                if nbr_cand.cost < cur.cost:
                    cur = nbr_cand
                    k = 1  # restart from neighborhood 1
                else:
                    k += 1
            return cur

        # Initial VND
        if ledger.remaining:
            current = run_vnd(current)
            if current.cost < best.cost:
                best = current

        temperature = current.cost * c.temperature_fraction
        generation = 0
        if observer is not None:
            observer.on_generation([current], best.cost, generation, ledger.nfe, time.perf_counter() - start_time)

        while ledger.remaining and (c.max_iterations is None or generation < c.max_iterations):
            generation += 1

            # 1. Perturbation: double relocate, block swap, or segment exchange
            pert_op = str(rng.choice(["double_relocate", "block_swap", "segment_exchange"]))
            if pert_op == "double_relocate":
                pert_perm = double_relocate(current.permutation, rng)
            elif pert_op == "block_swap":
                pert_perm = block_swap(current.permutation, rng)
            else:
                pert_perm = segment_exchange(current.permutation, rng)

            # Evaluate perturbed candidate = +1 NFE
            perturbed_cand = ledger.evaluate(pert_perm, "perturbation")

            # 2. Local Search: VND on perturbed candidate
            if ledger.remaining:
                local_opt = run_vnd(perturbed_cand)
            else:
                local_opt = perturbed_cand

            # 3. Acceptance criterion
            accepted = False
            if local_opt.cost < best.cost:
                best = local_opt
                accepted = True
            elif local_opt.cost < current.cost:
                accepted = True
            else:
                delta = local_opt.cost - current.cost
                if temperature > 1e-9:
                    prob = math.exp(-min(50.0, delta / temperature))
                    if rng.random() < prob:
                        accepted = True

            if accepted:
                current = local_opt

            temperature *= c.cooling_rate
            if observer is not None:
                observer.on_generation([current], best.cost, generation, ledger.nfe, time.perf_counter() - start_time)

        runtime = time.perf_counter() - start_time
        return build_optimization_result(
            algorithm=self.name,
            problem=problem,
            ledger=ledger,
            seed=seed,
            runtime_seconds=runtime,
            population_size=1,  # trajectory-based
            max_iterations=generation,
            run_id=(config or {}).get("run_id", 1),
            extra_metadata={"config": c.__dict__},
        )

