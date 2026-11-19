"""Generational permutation GA with OX, mutation, tournament and elitism.

Uses the same evaluator, Split, NFE ledger and initial RNG draws as RAPO.
No local search, adaptive operator selection, or uncounted offspring scoring.
"""
from dataclasses import asdict, dataclass, fields
import argparse
import json
import math
from pathlib import Path
import time
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from algorithms.rapo import EvaluationLedger
from algorithms.rapo_operators import from_keys, rank_keys
from core.tdvrp_common import DATA, ROOT, TDVRPProblem, OptimizationResult, data_directory


@dataclass
class GAConfig:
    population_size: int = 30
    max_function_evaluations: int = 1000
    max_iterations: int | None = None
    crossover_rate: float = 0.90
    mutation_rate: float = 0.20
    mutation_operator: str = 'mixed'
    tournament_size: int = 3
    elite_ratio: float = 0.10

    @classmethod
    def from_mapping(cls, config=None):
        values = dict(config or {})
        if not values.pop('minimization', True):
            raise ValueError('TDVRP requires minimization')
        values.pop('use_evaluation_budget', None)  # explicit NFE limit always enforced
        values.pop('run_id', None)
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown GA configuration: {sorted(unknown)}')
        c = cls(**values)
        for key in ('population_size', 'max_function_evaluations', 'tournament_size'):
            if type(getattr(c, key)) is not int or getattr(c, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        if c.population_size < 2 or c.max_function_evaluations < c.population_size:
            raise ValueError('Need population_size >= 2 and NFE budget >= population_size')
        if c.max_iterations is not None and (type(c.max_iterations) is not int or c.max_iterations < 0):
            raise ValueError('max_iterations must be null or a nonnegative integer')
        for key in ('crossover_rate', 'mutation_rate', 'elite_ratio'):
            value = getattr(c, key)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{key} must lie in [0,1]')
        if not 0 < c.elite_ratio < 1:
            raise ValueError('elite_ratio must lie in (0,1)')
        if c.mutation_operator not in ('swap', 'inversion', 'mixed'):
            raise ValueError('mutation_operator must be swap, inversion or mixed')
        return c


def order_crossover(first, second, rng):
    """One OX child: retain [a:b], cyclically fill from the second parent."""
    if len(first) < 2 or len(first) != len(second) or len(set(first)) != len(first) or set(first) != set(second):
        raise ValueError('OX needs two permutations of the same distinct customers')
    n = len(first)
    a, b = sorted(int(i) for i in rng.choice(n+1, 2, replace=False))
    child = [None]*n
    child[a:b] = first[a:b]
    retained = set(first[a:b])
    donor = [second[(b+i) % n] for i in range(n) if second[(b+i) % n] not in retained]
    slots = [(b+i) % n for i in range(n) if child[(b+i) % n] is None]
    for slot, customer in zip(slots, donor):
        child[slot] = customer
    return tuple(child)


def mutate(permutation, rng, operator='mixed'):
    if operator == 'mixed':
        operator = str(rng.choice(('swap', 'inversion')))
    if operator not in ('swap', 'inversion'):
        raise ValueError('Unknown mutation operator')
    child = list(permutation)
    a, b = sorted(int(i) for i in rng.choice(len(child), 2, replace=False))
    if operator == 'swap':
        child[a], child[b] = child[b], child[a]
    else:
        child[a:b+1] = reversed(child[a:b+1])
    return tuple(child)


def tournament(population, size, rng):
    indices = rng.choice(len(population), min(size, len(population)), replace=False)
    # Random draw order provides seeded tie-breaking on equal objectives.
    return min((population[int(i)] for i in indices), key=lambda x: x.cost)


class GA:
    name = 'GA'
    hybrid_type = ''

    def optimize(self, problem, config=None, seed=None, observer=None, initial_population=None):
        if not isinstance(problem, TDVRPProblem):
            raise TypeError('Expected the shared TDVRPProblem')
        c = GAConfig.from_mapping(config)
        rng = np.random.default_rng(seed)
        ledger = EvaluationLedger(problem, c.max_function_evaluations)
        started = time.perf_counter()
        logs, history = [], []
        population = []
        best = None
        generation = 0

        def evaluate(permutation, phase):
            nonlocal best
            candidate = ledger.evaluate(permutation, phase)
            improved = best is None or candidate.cost < best.cost
            if improved:
                best = candidate
            logs.append({'nfe': ledger.nfe, 'generation': generation, 'phase': phase,
                         'candidate_cost': candidate.cost, 'global_best': best.cost,
                         'global_improvement': improved})
            return candidate

        # Same initial population as the other solvers for identical seed/size.
        if initial_population is not None:
            _ = rng.random((c.population_size, problem.dimension))
            for perm in initial_population:
                population.append(evaluate(tuple(perm), 'initialization'))
        else:
            for keys in rng.random((c.population_size, problem.dimension)):
                population.append(evaluate(from_keys(keys), 'initialization'))
        if observer is not None:
            observer.on_generation(population, best.cost, generation, ledger.nfe, time.perf_counter() - started)
        elite_count = min(c.population_size-1, max(1, math.ceil(c.elite_ratio*c.population_size)))
        while ledger.remaining and (c.max_iterations is None or generation < c.max_iterations):
            generation += 1
            ranked = sorted(population, key=lambda x: x.cost)
            next_population = ranked[:elite_count]
            while len(next_population) < c.population_size and ledger.remaining:
                first = tournament(population, c.tournament_size, rng)
                child = first.permutation
                if rng.random() < c.crossover_rate:
                    second = tournament(population, c.tournament_size, rng)
                    child = order_crossover(first.permutation, second.permutation, rng)
                if rng.random() < c.mutation_rate:
                    child = mutate(child, rng, c.mutation_operator)
                # Even an unchanged clone costs one NFE; copying elites does not.
                next_population.append(evaluate(child, 'main'))
            # A partially completed generation retains unevaluated old survivors.
            if len(next_population) < c.population_size:
                missing = c.population_size-len(next_population)
                next_population.extend(ranked[elite_count:elite_count+missing])
            population = next_population
            history.append(best.cost)
            if observer is not None:
                observer.on_generation(population, best.cost, generation, ledger.nfe, time.perf_counter() - started)
            logs.append({'nfe': ledger.nfe, 'generation': generation, 'phase': 'generation_end',
                         'global_best': best.cost, 'current_best': min(x.cost for x in population),
                         'median_population_objective': float(np.median([x.cost for x in population])),
                         'elite_count': elite_count})
        if (sorted(cust for r in best.routes for cust in r) != list(range(1, problem.dimension+1))
                or any(not r or sum(problem.demands[cust] for cust in r) > problem.capacity for r in best.routes)):
            raise AssertionError('GA archive must be feasible')
        return OptimizationResult(
            algorithm=self.name, benchmark=problem.name, seed=seed,
            best_fitness=best.cost, best_solution=rank_keys(best.permutation).tolist(),
            best_permutation=list(best.permutation), routes=[[0, *r, 0] for r in best.routes],
            departure_times=[e[1] for e in best.evaluations], return_times=[e[2] for e in best.evaluations],
            route_durations=[e[0] for e in best.evaluations],
            route_loads=[sum(problem.demands[cust] for cust in r) for r in best.routes],
            convergence_curve=history, runtime_seconds=time.perf_counter()-started,
            population_size=c.population_size,
            max_iterations=c.max_iterations if c.max_iterations is not None else generation,
            nfe=ledger.nfe, dimension=problem.dimension,
            run_id=(config or {}).get('run_id', 1),
            metadata={'objective': 'Duration', 'units': 'seconds', 'feasible': True,
                      'encoding': 'permutation + shared minimum-duration Split',
                      'implementation': 'permutation-ga-ox-v1', 'config': asdict(c),
                      'departure_domain': [0, problem.horizon],
                      'travel_time_domain': 'route departure restricted to nonempty T(R); no extrapolation',
                      'model_version': 4, 'vehicle_limit_K': problem.vehicle_limit,
                      'penalty_weights': {'miss': problem.alpha_miss, 'dup': problem.alpha_dup,
                                          'cap': problem.alpha_cap, 'fleet': problem.alpha_fleet,
                                          'M0': problem.penalty_fixed, 'M1': problem.penalty_multiplier},
                      'evaluations': ledger.nfe, 'nfe_by_phase': ledger.spending,
                      'completed_iterations': generation, 'elite_count': elite_count,
                      'stop_reason': 'nfe_budget' if not ledger.remaining else 'iteration_limit',
                      'search_log': logs})


def main():
    parser = argparse.ArgumentParser(description='Permutation GA: Hong Kong n=10 only')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--method', choices=['hyb', 'poi', 'all'], default='hyb')
    parser.add_argument('--customers', type=int, choices=[10, 50], default=10)
    parser.add_argument('--traffic', choices=['bpr-light', 'bpr-moderate', 'bpr-heavy',
                        'wave-light', 'wave-moderate', 'wave-heavy', 'all'], default='bpr-light')
    parser.add_argument('--population-size', type=int)
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--max-evaluations', type=int)
    parser.add_argument('--crossover-rate', type=float)
    parser.add_argument('--mutation-rate', type=float)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    values = json.loads(args.config.read_text(encoding='utf-8')) if args.config else {}
    for arg, field in [('population_size', 'population_size'), ('iterations', 'max_iterations'),
                       ('max_evaluations', 'max_function_evaluations'),
                       ('crossover_rate', 'crossover_rate'), ('mutation_rate', 'mutation_rate')]:
        if getattr(args, arg) is not None:
            values[field] = getattr(args, arg)
    settings = GAConfig.from_mapping(values)
    paths = [p for p in sorted(data_directory(args.customers).rglob('*.vrp.json'))
             if (args.method == 'all' or p.parent.parent.name.endswith('-'+args.method))
             and (args.traffic == 'all' or p.parent.name == args.traffic)]
    if not paths:
        parser.error(f'No Hong Kong n={args.customers} instances match the filters')
    results = []
    for path in paths:
        print(f'Loading {path.name}', flush=True)
        result = GA().optimize(TDVRPProblem(path), asdict(settings), args.seed)
        results.append(asdict(result))
        print(f'GA: {result.best_fitness:.6f} seconds; NFE={result.nfe}; routes={result.routes}', flush=True)
    output = args.output or ROOT / 'results' / 'ga_hong_kong_n10.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
