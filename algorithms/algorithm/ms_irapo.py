"""Legacy Euclidean MS-IRAPO.

Module này vẫn được giữ để tái lập kết quả cũ. Nó dùng ``TDVRPProblem`` và
distance matrix, không thuộc pipeline HCMC road-graph mới. Không đưa lớp này
vào benchmark HCMC cho tới khi được chuyển sang ``HCMCTDVRPEvaluator``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from algorithms.base import BaseOptimizer
from algorithms.operators.exploration import apply_exploration_operator
from algorithms.operators.exploitation import apply_exploitation_operator
from algorithms.operators.repair import repair_solution
from algorithms.population import initialize_population, split_population
from algorithms.swarm import adaptive_probabilities, swarm_profiles
from core.evaluator import RouteEvaluator
from core.fitness import FitnessEvaluator
from core.models import MigrationEvent, OptimizationResult, Solution, SwarmState, TDVRPProblem
from core.validation import validate_solution


@dataclass
class MSIRAPOConfig:
    population_size: int = 60
    num_swarms: int = 3
    max_iterations: int = 150
    migration_interval: int = 10
    migration_count: int = 1
    seed: int = 42
    adaptive_probabilities: bool = True


class MSIRAPOOptimizer(BaseOptimizer):
    def __init__(self, start_time, progress_callback=None):
        self.start_time = start_time
        self.progress_callback = progress_callback

    def solve(self, problem: TDVRPProblem, config: MSIRAPOConfig) -> OptimizationResult:
        if config.population_size < config.num_swarms or config.num_swarms < 1: raise ValueError("population_size must be >= num_swarms >= 1")
        if config.migration_interval <= 0: raise ValueError("migration_interval must be positive")
        rng = np.random.default_rng(config.seed); started = time.perf_counter()
        evaluator = FitnessEvaluator(problem, RouteEvaluator(problem, self.start_time))
        population = initialize_population(problem, config.population_size, rng)
        for solution in population: evaluator.evaluate(solution)
        profiles = swarm_profiles(config.num_swarms)
        swarms = [SwarmState(i + 1, role, p, q, members) for i, ((role, p, q), members) in enumerate(zip(profiles, split_population(population, config.num_swarms)))]
        for swarm in swarms: swarm.local_best = min(swarm.individuals, key=lambda s: s.fitness).copy()
        global_best = min((s.local_best for s in swarms), key=lambda s: s.fitness).copy()
        convergence = [global_best.fitness]; local_history = [[s.local_best.fitness for s in swarms]]; migrations = []
        for iteration in range(1, config.max_iterations + 1):
            for swarm in swarms:
                p_aro, p_poa = adaptive_probabilities(swarm.p_aro, iteration, config.max_iterations, config.adaptive_probabilities)
                swarm.probability_history.append({"p_aro": p_aro, "p_poa": p_poa})
                for idx, parent in enumerate(swarm.individuals):
                    candidate = apply_exploration_operator(parent, problem, rng) if rng.random() < p_aro else apply_exploitation_operator(parent, problem, evaluator, rng)
                    candidate = repair_solution(candidate, problem); evaluator.evaluate(candidate)
                    if candidate.is_feasible and candidate.fitness < parent.fitness: swarm.individuals[idx] = candidate
                swarm.local_best = min(swarm.individuals, key=lambda s: s.fitness).copy()
            if config.num_swarms > 1 and iteration % config.migration_interval == 0:
                for source_idx, source in enumerate(swarms):
                    target = swarms[(source_idx + 1) % len(swarms)]
                    migrant = source.local_best.copy()
                    for _ in range(min(config.migration_count, len(target.individuals))):
                        worst = max(range(len(target.individuals)), key=lambda i: target.individuals[i].fitness)
                        if migrant.fitness < target.individuals[worst].fitness:
                            target.individuals[worst] = migrant.copy(); migrations.append(MigrationEvent(iteration, source.index, target.index, migrant.fitness))
            current = min((s.local_best for s in swarms), key=lambda s: s.fitness)
            if current.fitness < global_best.fitness: global_best = current.copy()
            convergence.append(global_best.fitness); local_history.append([s.local_best.fitness for s in swarms])
            if self.progress_callback: self.progress_callback(iteration, config.max_iterations)
        evaluation = RouteEvaluator(problem, self.start_time).evaluate(global_best)
        assert validate_solution(global_best, problem).feasible
        return OptimizationResult(global_best, evaluation, convergence, local_history, swarms, migrations, time.perf_counter() - started)
