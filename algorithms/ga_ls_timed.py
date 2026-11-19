# Timed experiment copy; original source remains unchanged.
"""Genetic Algorithm with Local Search (GA-LS) for TDVRP.

Combines generational GA (OX crossover, mutation, tournament selection, elitism)
with explicit local search on offspring using:
- relocate
- swap
- two_opt
- two_opt_star
EVERY local search neighbor evaluated counts strictly as 1 NFE in the ledger.
"""
from __future__ import annotations
from dataclasses import dataclass, fields
import math
import time
import numpy as np

from core.tdvrp_common import TDVRPProblem, OptimizationResult
from algorithms.common.ledger import Candidate, build_optimization_result
from algorithms.timed_runtime import TimedLedger as EvaluationLedger
from algorithms.operators import (
    swap_random,
    relocate_random,
    two_opt,
    two_opt_star,
    order_crossover,
    inversion_random,
)
from algorithms.rapo_operators import from_keys


@dataclass
class GALSConfig:
    population_size: int = 30
    max_function_evaluations: int = 1000
    max_iterations: int | None = None
    crossover_rate: float = 0.90
    mutation_rate: float = 0.20
    tournament_size: int = 3
    elite_ratio: float = 0.10
    ls_probability: float = 0.50
    ls_max_steps: int = 5

    @classmethod
    def from_mapping(cls, config=None):
        values = dict(config or {})
        values.pop("minimization", None)
        values.pop("use_evaluation_budget", None)
        values.pop("run_id", None)
        allowed = {f.name for f in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown GA-LS config keys: {sorted(unknown)}")
        cfg = cls(**values)
        if cfg.population_size < 2:
            raise ValueError("population_size must be >= 2")
        if cfg.max_function_evaluations < cfg.population_size:
            raise ValueError("NFE budget must be >= population_size")
        return cfg


class GALS:
    name = "GA-LS"

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
        c = GALSConfig.from_mapping(config)
        rng = np.random.default_rng(seed)
        ledger = EvaluationLedger(problem, c.max_function_evaluations)
        start_time = time.perf_counter()

        population: list[Candidate] = []
        best: Candidate | None = None

        # Initialization
        if initial_population is not None:
            _ = rng.random((c.population_size, problem.dimension))
            for perm in initial_population:
                if not ledger.remaining:
                    break
                cand = ledger.evaluate(tuple(perm), "initialization")
                population.append(cand)
                if best is None or cand.cost < best.cost:
                    best = cand
        else:
            keys = rng.random((c.population_size, problem.dimension))
            for row in keys:
                if not ledger.remaining:
                    break
                cand = ledger.evaluate(from_keys(row), "initialization")
                population.append(cand)
                if best is None or cand.cost < best.cost:
                    best = cand

        generation = 0
        elite_count = min(c.population_size - 1, max(1, math.ceil(c.elite_ratio * c.population_size)))
        if observer is not None and best is not None:
            observer.on_generation(population, best.cost, generation, ledger.nfe, time.perf_counter() - start_time)

        def run_local_search(candidate: Candidate) -> Candidate:
            """Apply local search neighborhoods, counting each evaluation as 1 NFE."""
            cur = candidate
            for _ in range(c.ls_max_steps):
                if not ledger.remaining:
                    break
                op = str(rng.choice(["relocate", "swap", "two_opt", "two_opt_star"]))
                if op == "relocate":
                    nbr_perm = relocate_random(cur.permutation, rng)
                elif op == "swap":
                    nbr_perm = swap_random(cur.permutation, rng)
                elif op == "two_opt":
                    nbr_perm = two_opt(cur.permutation, cur.routes, rng)
                else:
                    nbr_perm = two_opt_star(cur.permutation, cur.routes, rng)

                if nbr_perm == cur.permutation:
                    continue

                nbr_cand = ledger.evaluate(nbr_perm, "local_search")
                if nbr_cand.cost < cur.cost:
                    cur = nbr_cand  # first improvement
                else:
                    # continue searching
                    pass
            return cur

        while ledger.remaining and (c.max_iterations is None or generation < c.max_iterations):
            generation += 1
            ranked = sorted(population, key=lambda x: x.cost)
            next_population = ranked[:elite_count]

            while len(next_population) < c.population_size and ledger.remaining:
                # Tournament selection
                t1 = rng.choice(len(population), min(c.tournament_size, len(population)), replace=False)
                parent1 = min((population[int(i)] for i in t1), key=lambda x: x.cost)
                child_perm = parent1.permutation

                if rng.random() < c.crossover_rate:
                    t2 = rng.choice(len(population), min(c.tournament_size, len(population)), replace=False)
                    parent2 = min((population[int(i)] for i in t2), key=lambda x: x.cost)
                    child_perm = order_crossover(parent1.permutation, parent2.permutation, rng)

                if rng.random() < c.mutation_rate:
                    mut_op = str(rng.choice(["swap", "inversion"]))
                    if mut_op == "swap":
                        child_perm = swap_random(child_perm, rng)
                    else:
                        child_perm = inversion_random(child_perm, rng)

                cand = ledger.evaluate(child_perm, "main")

                # Local Search step on offspring
                if rng.random() < c.ls_probability and ledger.remaining:
                    cand = run_local_search(cand)

                next_population.append(cand)
                if best is None or cand.cost < best.cost:
                    best = cand

            # Fill missing if budget exhausted mid-generation
            if len(next_population) < c.population_size:
                missing = c.population_size - len(next_population)
                next_population.extend(ranked[elite_count:elite_count + missing])

            population = next_population
            if observer is not None and best is not None:
                observer.on_generation(population, best.cost, generation, ledger.nfe, time.perf_counter() - start_time)

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

