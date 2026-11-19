"""Discrete PSO with Variable Neighborhood Search (PSO-VNS) for TDVRP.

Combines DPSO global exploration with Variable Neighborhood Search (VNS)
intensification around global best using:
- N1: swap
- N2: relocate
- N3: two_opt
- N4: two_opt_star
Every neighbor evaluated counts strictly as 1 NFE in the ledger.
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
    inversion_random,
    two_opt,
    two_opt_star,
    align_towards,
)
from algorithms.rapo_operators import from_keys


@dataclass
class PSOVNSConfig:
    population_size: int = 30
    max_function_evaluations: int = 1000
    max_iterations: int | None = None
    w: float = 0.20
    c1: float = 0.50
    c2: float = 0.50
    step_size: int = 2
    vns_frequency: int = 1   # Apply VNS every N generations
    vns_max_k: int = 4

    @classmethod
    def from_mapping(cls, config=None):
        values = dict(config or {})
        values.pop("minimization", None)
        values.pop("use_evaluation_budget", None)
        values.pop("run_id", None)
        allowed = {f.name for f in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown PSO-VNS config keys: {sorted(unknown)}")
        cfg = cls(**values)
        if cfg.population_size < 2:
            raise ValueError("population_size must be >= 2")
        if cfg.max_function_evaluations < cfg.population_size:
            raise ValueError("NFE budget must be >= population_size")
        return cfg


class PSOVNS:
    name = "PSO-VNS"

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
        c = PSOVNSConfig.from_mapping(config)
        rng = np.random.default_rng(seed)
        ledger = EvaluationLedger(problem, c.max_function_evaluations)
        start_time = time.perf_counter()

        particles: list[Candidate] = []
        pbests: list[Candidate] = []
        gbest: Candidate | None = None

        # Initialization
        if initial_population is not None:
            _ = rng.random((c.population_size, problem.dimension))
            for perm in initial_population:
                if not ledger.remaining:
                    break
                cand = ledger.evaluate(tuple(perm), "initialization")
                particles.append(cand)
                pbests.append(cand)
                if gbest is None or cand.cost < gbest.cost:
                    gbest = cand
        else:
            keys = rng.random((c.population_size, problem.dimension))
            for row in keys:
                if not ledger.remaining:
                    break
                cand = ledger.evaluate(from_keys(row), "initialization")
                particles.append(cand)
                pbests.append(cand)
                if gbest is None or cand.cost < gbest.cost:
                    gbest = cand

        generation = 0
        if observer is not None and gbest is not None:
            observer.on_generation(particles, gbest.cost, generation, ledger.nfe, time.perf_counter() - start_time)

        def run_vns(current_best: Candidate) -> Candidate:
            """Variable Neighborhood Search over 4 ordered neighborhoods."""
            cur = current_best
            k = 1
            while k <= c.vns_max_k and ledger.remaining:
                # Select neighborhood
                if k == 1:
                    nbr_perm = swap_random(cur.permutation, rng)
                elif k == 2:
                    nbr_perm = relocate_random(cur.permutation, rng)
                elif k == 3:
                    nbr_perm = two_opt(cur.permutation, cur.routes, rng)
                else:
                    nbr_perm = two_opt_star(cur.permutation, cur.routes, rng)

                if nbr_perm == cur.permutation:
                    k += 1
                    continue

                nbr_cand = ledger.evaluate(nbr_perm, "vns")
                if nbr_cand.cost < cur.cost:
                    cur = nbr_cand
                    k = 1  # restart from first neighborhood
                else:
                    k += 1  # move to next neighborhood
            return cur

        while ledger.remaining and (c.max_iterations is None or generation < c.max_iterations):
            generation += 1
            # 1. Swarm movement
            for i in range(len(particles)):
                if not ledger.remaining:
                    break
                current_perm = particles[i].permutation

                if rng.random() < c.c1 and pbests[i].permutation != current_perm:
                    current_perm = align_towards(current_perm, pbests[i].permutation, c.step_size, rng)

                if rng.random() < c.c2 and gbest is not None and gbest.permutation != current_perm:
                    current_perm = align_towards(current_perm, gbest.permutation, c.step_size, rng)

                if rng.random() < c.w:
                    move_op = str(rng.choice(["swap", "relocate", "inversion"]))
                    if move_op == "swap":
                        current_perm = swap_random(current_perm, rng)
                    elif move_op == "relocate":
                        current_perm = relocate_random(current_perm, rng)
                    else:
                        current_perm = inversion_random(current_perm, rng)

                cand = ledger.evaluate(current_perm, "main")
                particles[i] = cand

                if cand.cost < pbests[i].cost:
                    pbests[i] = cand
                if gbest is None or cand.cost < gbest.cost:
                    gbest = cand

            # 2. VNS intensification on global best
            if generation % c.vns_frequency == 0 and gbest is not None and ledger.remaining:
                gbest = run_vns(gbest)

            if observer is not None and gbest is not None:
                observer.on_generation(particles, gbest.cost, generation, ledger.nfe, time.perf_counter() - start_time)

        runtime = time.perf_counter() - start_time
        return build_optimization_result(
            algorithm=self.name,
            problem=problem,
            ledger=ledger,
            seed=seed,
            runtime_seconds=runtime,
            population_size=c.population_size,
            max_iterations=generation,
            run_id=(config or {}).get("run_id", 1),
            extra_metadata={"config": c.__dict__},
        )

