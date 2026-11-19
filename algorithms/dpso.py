"""Discrete Particle Swarm Optimization (DPSO) for TDVRP.

Maintains permutation particles with discrete velocity/movement:
- Cognitive alignment towards pBest
- Social alignment towards gBest
- Inertia exploration via discrete swaps/relocations
No continuous vectors or key-sorting during iterations.
Strictly 1 candidate evaluation = 1 NFE.
"""
from __future__ import annotations
from dataclasses import dataclass, fields
import math
import time
import numpy as np

from core.tdvrp_common import TDVRPProblem, OptimizationResult
from algorithms.common.ledger import EvaluationLedger, Candidate, build_optimization_result
from algorithms.operators import swap_random, relocate_random, inversion_random, align_towards
from algorithms.rapo_operators import from_keys


@dataclass
class DPSOConfig:
    population_size: int = 30
    max_function_evaluations: int = 1000
    max_iterations: int | None = None
    w: float = 0.20        # Inertia / random exploration probability
    c1: float = 0.50       # Cognitive learning rate (pBest alignment)
    c2: float = 0.50       # Social learning rate (gBest alignment)
    step_size: int = 2     # Max alignment steps per component

    @classmethod
    def from_mapping(cls, config=None):
        values = dict(config or {})
        values.pop("minimization", None)
        values.pop("use_evaluation_budget", None)
        values.pop("run_id", None)
        allowed = {f.name for f in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown DPSO config keys: {sorted(unknown)}")
        cfg = cls(**values)
        if cfg.population_size < 2:
            raise ValueError("population_size must be >= 2")
        if cfg.max_function_evaluations < cfg.population_size:
            raise ValueError("NFE budget must be >= population_size")
        for key in ("w", "c1", "c2"):
            val = getattr(cfg, key)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{key} must lie in [0, 1]")
        return cfg


class DPSO:
    name = "DPSO"

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
        c = DPSOConfig.from_mapping(config)
        rng = np.random.default_rng(seed)
        ledger = EvaluationLedger(problem, c.max_function_evaluations)
        start_time = time.perf_counter()

        # Particles & pBest
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

        # Swarm iterations
        while ledger.remaining and (c.max_iterations is None or generation < c.max_iterations):
            generation += 1
            for i in range(len(particles)):
                if not ledger.remaining:
                    break
                current_perm = particles[i].permutation
                
                # Cognitive move towards pBest
                if rng.random() < c.c1 and pbests[i].permutation != current_perm:
                    current_perm = align_towards(current_perm, pbests[i].permutation, c.step_size, rng)

                # Social move towards gBest
                if rng.random() < c.c2 and gbest is not None and gbest.permutation != current_perm:
                    current_perm = align_towards(current_perm, gbest.permutation, c.step_size, rng)

                # Inertia / random exploration
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

