"""RAPO-V3: two-level AOS coordinating ARO exploration and POA refinement.

Uses TDVRPProblem.evaluate_permutation without changing the objective, Split,
time model, or the np.random.default_rng(seed) protocol. All search evaluations
pass through one strict NFE ledger, including initialization and cache hits.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import time
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from core.tdvrp_common import DATA, ROOT, OptimizationResult, TDVRPProblem, data_directory
from algorithms.rapo_operators import (FAMILIES, ELITE_OPERATORS, STRONG_OPERATORS,
                            from_keys, make_candidate, rank_keys)

PRESETS = {
    'RAPO-Base': (False, False, False),
    'RAPO-AOS': (True, False, False),
    'RAPO-ER': (False, True, False),
    'RAPO-V3': (True, True, True),
}


@dataclass
class RAPOConfig:
    population_size: int = 30
    max_function_evaluations: int = 1000
    max_iterations: int | None = None
    variant: str = 'RAPO-V3'
    adaptive: bool = True
    elite_refinement: bool = True
    stagnation_control: bool = True
    diversity_control: bool = True
    elite_ratio: float = 0.10
    initial_aro_probability: float = 0.50
    initial_poa_probability: float = 0.50
    reward_alpha: float = 0.20
    ucb_c: float = 1.20
    reward_global_best: float = 5.0
    reward_parent_improvement: float = 3.0
    reward_accepted: float = 1.0
    reward_rejected: float = 0.0
    stagnation_L1: int = 20
    stagnation_L2: int = 40
    restart_ratio: float = 0.20
    destroy_fraction_min: float = 0.10
    destroy_fraction_max: float = 0.20
    elite_max_neighbors: int = 5
    elite_budget_fraction: float = 0.20
    restart_budget_fraction: float = 0.05
    exploration_bias: float = 1.10  # log-odds; exp(1.10) ~ 3
    strong_operator_bias: float = 0.30
    family_temperature: float = 2.0
    family_probability_floor: float = 0.05
    diversity_threshold: float = 0.30
    diversity_bias: float = 0.50

    @classmethod
    def from_mapping(cls, mapping=None):
        mapping = dict(mapping or {})
        variant = mapping.get('variant', 'RAPO-V3')
        if variant not in PRESETS:
            raise ValueError(f'Unknown variant {variant}')
        adaptive, elite, stagnation = PRESETS[variant]
        values = dict(adaptive=adaptive, elite_refinement=elite,
                      stagnation_control=stagnation, diversity_control=stagnation)
        allowed = {f.name for f in fields(cls)}
        legacy = {'minimization', 'use_evaluation_budget', 'run_id'}
        unknown = set(mapping) - allowed - legacy
        if unknown:
            raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
        if not mapping.get('minimization', True):
            raise ValueError('TDVRP requires minimization.')
        values.update({k: v for k, v in mapping.items() if k in allowed})
        config = cls(**values)
        config.validate()
        return config

    def validate(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
                raise ValueError(f'{field.name} must be finite')
        for key in ('population_size', 'max_function_evaluations', 'stagnation_L1',
                    'stagnation_L2', 'elite_max_neighbors'):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if self.population_size < 2 or self.max_function_evaluations < self.population_size:
            raise ValueError('Need population_size >= 2 and NFE budget >= population_size')
        if self.max_iterations is not None and (type(self.max_iterations) is not int or self.max_iterations < 0):
            raise ValueError('max_iterations must be null or a nonnegative integer')
        for key in ('adaptive', 'elite_refinement', 'stagnation_control', 'diversity_control'):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f'{key} must be boolean')
        for key in ('elite_ratio', 'restart_ratio', 'reward_alpha', 'destroy_fraction_min',
                    'destroy_fraction_max'):
            if not 0 < getattr(self, key) <= 1:
                raise ValueError(f'{key} must lie in (0,1]')
        if self.elite_ratio >= 1:
            raise ValueError('elite_ratio must leave non-elite population members')
        if self.destroy_fraction_min > self.destroy_fraction_max:
            raise ValueError('destroy fractions are reversed')
        if self.stagnation_L2 <= self.stagnation_L1:
            raise ValueError('stagnation_L2 must exceed stagnation_L1')
        for key in ('elite_budget_fraction', 'restart_budget_fraction', 'diversity_threshold'):
            if not 0 <= getattr(self, key) < 1:
                raise ValueError(f'{key} must lie in [0,1)')
        if self.elite_budget_fraction + self.restart_budget_fraction >= 1:
            raise ValueError('Reserve a positive share for main search')
        if not 0 <= self.family_probability_floor < 0.5 or self.family_temperature <= 0:
            raise ValueError('Invalid family probability controls')
        if not (0 < self.initial_aro_probability < 1 and 0 < self.initial_poa_probability < 1
                and math.isclose(self.initial_aro_probability + self.initial_poa_probability, 1)):
            raise ValueError('Initial family probabilities must be positive and sum to 1')
        for key in ('ucb_c', 'exploration_bias', 'strong_operator_bias', 'diversity_bias',
                    'reward_global_best', 'reward_parent_improvement', 'reward_accepted', 'reward_rejected'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f'{key} must be finite and nonnegative')


@dataclass
class OperatorStats:
    usage_count: int = 0
    improvement_count: int = 0
    global_best_contributions: int = 0
    cumulative_reward: float = 0.0
    mean_reward: float = 0.0
    current_score: float = 0.0

    def update(self, reward, improvement, global_improvement, alpha):
        self.usage_count += 1
        self.improvement_count += int(improvement)
        self.global_best_contributions += int(global_improvement)
        self.cumulative_reward += reward
        self.mean_reward = self.cumulative_reward / self.usage_count
        self.current_score = (1-alpha)*self.current_score + alpha*reward


def candidate_reward(global_improvement, improvement, accepted, config):
    if global_improvement:
        return config.reward_global_best
    if improvement:
        return config.reward_parent_improvement
    return config.reward_accepted if accepted else config.reward_rejected


def choose_ucb(names, stats, rng, c, bonus=None):
    untried = [name for name in names if stats[name].usage_count == 0]
    if untried:
        return str(rng.choice(untried))
    total = sum(stats[name].usage_count for name in names)
    scores = {name: stats[name].mean_reward + c * math.sqrt(math.log(max(1, total)) /
              stats[name].usage_count) + (bonus or {}).get(name, 0) for name in names}
    maximum = max(scores.values())
    return str(rng.choice([name for name in names if scores[name] == maximum]))


def diversity(population):
    """Mean pairwise normalized Hamming distance; no objective contribution."""
    if len(population) < 2:
        return 0.0
    values = np.asarray([x.permutation for x in population])
    distances = [np.mean(values[i] != values[j]) for i in range(len(values))
                 for j in range(i+1, len(values))]
    return float(np.mean(distances))


class AdaptiveController:
    def __init__(self, config):
        self.config = config
        self.families = {name: OperatorStats() for name in FAMILIES}
        self.operators = {op: OperatorStats() for ops in FAMILIES.values() for op in ops}

    def probabilities(self, stagnation, population_diversity):
        c = self.config
        logit = math.log(c.initial_aro_probability / c.initial_poa_probability)
        if c.adaptive:
            logit += (self.families['ARO'].current_score - self.families['POA'].current_score) / c.family_temperature
        if c.stagnation_control and stagnation >= c.stagnation_L1:
            logit += c.exploration_bias
        if c.diversity_control and population_diversity < c.diversity_threshold:
            logit += c.diversity_bias
        probability = 1 / (1 + math.exp(-max(-50, min(50, logit))))
        probability = min(1-c.family_probability_floor, max(c.family_probability_floor, probability))
        return {'ARO': probability, 'POA': 1-probability}

    def select(self, rng, stagnation, population_diversity, forced_family=None):
        c = self.config
        probabilities = self.probabilities(stagnation, population_diversity)
        if forced_family:
            family = forced_family
        else:
            eligible = list(FAMILIES)
            # Warmup has precedence: every operator gets one trial if budget permits.
            if c.adaptive:
                untried_families = [f for f in eligible if any(
                    self.operators[op].usage_count == 0 for op in FAMILIES[f])]
                if untried_families:
                    eligible = untried_families
            weights = np.array([probabilities[f] for f in eligible])
            family = str(rng.choice(eligible, p=weights/weights.sum()))
        names = FAMILIES[family]
        if c.adaptive:
            strong = c.stagnation_control and stagnation >= c.stagnation_L1
            bonus = {op: c.strong_operator_bias for op in names if strong and op in STRONG_OPERATORS}
            operator = choose_ucb(names, self.operators, rng, c.ucb_c, bonus)
        else:
            operator = str(rng.choice(names))
        return family, operator, probabilities

    def update(self, family, operator, reward, improvement, global_improvement):
        for stats in (self.families[family], self.operators[operator]):
            stats.update(reward, improvement, global_improvement, self.config.reward_alpha)


@dataclass(frozen=True)
class Candidate:
    permutation: tuple[int, ...]
    cost: float
    routes: tuple[tuple[int, ...], ...]
    evaluations: tuple[tuple[float, float, float], ...]


class EvaluationLedger:
    def __init__(self, problem, limit):
        self.problem, self.limit, self.nfe = problem, limit, 0
        self.spending = dict(initialization=0, main=0, elite=0, restart=0)

    @property
    def remaining(self):
        return self.limit-self.nfe

    def evaluate(self, permutation, phase):
        if not self.remaining:
            raise RuntimeError('NFE budget exhausted')
        self.nfe += 1
        self.spending[phase] += 1
        cost, routes = self.problem.evaluate_permutation(permutation)
        # Split already evaluated these routes; collect its cached route details.
        details = tuple(self.problem.route_cost(r) for r in routes)
        return Candidate(tuple(permutation), cost, routes, details)


class SearchRun:
    """One isolated run; no shared mutable RNG, archive, statistics, or budget."""
    def __init__(self, problem, config, seed, forced_family=None, observer=None, initial_population=None):
        if not isinstance(problem, TDVRPProblem):
            raise TypeError('Expected the shared TDVRPProblem')
        self.problem, self.config, self.seed = problem, config, seed
        self.forced_family = forced_family
        self.observer = observer
        self.initial_population = initial_population
        self.rng = np.random.default_rng(seed)
        self.ledger = EvaluationLedger(problem, config.max_function_evaluations)
        self.controller = AdaptiveController(config)
        self.population = []
        self.archive = None
        self.generation = self.stagnation = 0
        self.last_restart_generation = 0
        self.elite_refinement_count = self.restart_count = 0
        self.restart_candidate_count = 0
        self.logs, self.restart_events, self.curve = [], [], []
        remaining = config.max_function_evaluations-config.population_size
        self.elite_cap = int(remaining*config.elite_budget_fraction)
        self.restart_cap = int(remaining*config.restart_budget_fraction)
        self.main_family_counts = {'ARO': 0, 'POA': 0}

    def _log(self, phase, family=None, operator=None, reward=None, accepted=None,
             probabilities=None):
        counts = {f: s.usage_count for f, s in self.controller.families.items()}
        total = sum(counts.values())
        stats = self.controller.operators.get(operator)
        self.logs.append({
            'nfe': self.ledger.nfe, 'generation': self.generation, 'phase': phase,
            'global_best': self.archive.cost, 'current_best': min(x.cost for x in self.population),
            'median_population_objective': float(np.median([x.cost for x in self.population])),
            'selected_family': family, 'selected_operator': operator, 'operator_reward': reward,
            'operator_mean_reward': stats.mean_reward if stats else None,
            'operator_usage_count': stats.usage_count if stats else None,
            'family_selection_ratio': {f: counts[f]/total if total else 0 for f in counts},
            'family_probabilities': probabilities, 'accepted': accepted,
            'stagnation_counter': self.stagnation,
            'elite_refinement_count': self.elite_refinement_count,
            'restart_count': self.restart_count,
            'restart_candidate_count': self.restart_candidate_count,
            'diversity': diversity(self.population),
        })

    def initialize(self):
        # Identical initial RNG protocol to the previous ARO/POA implementations.
        if self.initial_population is not None:
            _ = self.rng.random((self.config.population_size, self.problem.dimension))
            for perm in self.initial_population:
                candidate = self.ledger.evaluate(tuple(perm), 'initialization')
                self.population.append(candidate)
                if self.archive is None or candidate.cost < self.archive.cost:
                    self.archive = candidate
                self._log('initialization')
        else:
            keys = self.rng.random((self.config.population_size, self.problem.dimension))
            for row in keys:
                candidate = self.ledger.evaluate(from_keys(row), 'initialization')
                self.population.append(candidate)
                if self.archive is None or candidate.cost < self.archive.cost:
                    self.archive = candidate
                self._log('initialization')

    def attempt(self, index, family, operator, phase, probabilities=None):
        if not self.ledger.remaining:
            return False
        parent = self.population[index]
        # An NFE bound also bounds the number of cycles; avoid division by zero
        # and keep the original ARO energy schedule within its valid domain.
        limit = self.config.max_iterations or max(1, math.ceil(
            (self.ledger.limit-self.config.population_size) / self.config.population_size))
        permutation = make_candidate(operator, parent, self.population, index,
                                     min(self.generation-1, limit-1), limit, self.rng, self.config)
        candidate = self.ledger.evaluate(permutation, phase)
        improved = candidate.cost < parent.cost
        global_improvement = candidate.cost < self.archive.cost
        # Strict greedy: equal/worse candidates are rejected. The configurable
        # accepted-without-improvement reward is consequently unused in V3.
        if improved:
            self.population[index] = candidate
        if global_improvement:
            self.archive = candidate
            self.stagnation = 0
        reward = candidate_reward(global_improvement, improved, improved, self.config)
        self.controller.update(family, operator, reward, improved, global_improvement)
        if phase == 'main':
            self.main_family_counts[family] += 1
        elif phase == 'elite':
            self.elite_refinement_count += 1
        self._log(phase, family, operator, reward, improved, probabilities)
        return improved

    def elite_indices(self):
        count = min(len(self.population)-1, max(1, math.ceil(
            len(self.population)*self.config.elite_ratio)))
        return sorted(range(len(self.population)), key=lambda i: self.population[i].cost)[:count]

    def refine_elite(self):
        if not self.config.elite_refinement:
            return
        # At most ~20% of post-initialization NFE; additionally throttle each
        # cycle, so elites cannot consume the complete remaining budget early.
        main_share = 1-self.config.elite_budget_fraction-self.config.restart_budget_fraction
        cycle_cap = max(1, int(self.config.population_size * self.config.elite_budget_fraction / main_share))
        available = min(self.ledger.remaining, self.elite_cap-self.ledger.spending['elite'], cycle_cap)
        if available <= 0:
            return
        starting_nfe = self.ledger.nfe
        for index in self.elite_indices():
            for _ in range(self.config.elite_max_neighbors):
                if self.ledger.nfe-starting_nfe >= available:
                    return
                if self.config.adaptive:
                    operator = choose_ucb(ELITE_OPERATORS, self.controller.operators,
                                          self.rng, self.config.ucb_c)
                else:
                    operator = str(self.rng.choice(ELITE_OPERATORS))
                if self.attempt(index, 'POA', operator, 'elite'):
                    break  # first improvement; refine further in a later cycle

    def partial_restart(self):
        protected = self.elite_indices()
        protected_before = [self.population[i].permutation for i in protected]
        allowed = min(self.ledger.remaining, self.restart_cap-self.ledger.spending['restart'])
        count = min(max(1, math.ceil(len(self.population)*self.config.restart_ratio)),
                    len(self.population)-len(protected), allowed)
        if count <= 0:
            return
        worst = sorted((i for i in range(len(self.population)) if i not in protected),
                       key=lambda i: self.population[i].cost, reverse=True)[:count]
        archived_before = self.archive.cost
        self.restart_count += 1
        for index in worst:
            permutation = tuple(int(c) for c in self.rng.permutation(self.problem.dimension)+1)
            candidate = self.ledger.evaluate(permutation, 'restart')
            # Explicit diversification reset of worst members, not accept-worse
            # in the greedy search. Protected elites and archive remain intact.
            self.population[index] = candidate
            if candidate.cost < self.archive.cost:
                self.archive = candidate
                self.stagnation = 0
            self.restart_candidate_count += 1
            self._log('restart', accepted=True)
        self.restart_events.append({
            'generation': self.generation, 'nfe': self.ledger.nfe,
            'protected_before': protected_before,
            'protected_after': [self.population[i].permutation for i in protected],
            'replaced_indices': worst, 'archive_before': archived_before,
            'archive_after': self.archive.cost,
        })
        self.last_restart_generation = self.generation

    def run(self):
        start = time.perf_counter()
        self.initialize()
        if self.observer is not None:
            self.observer.on_generation(self.population, self.archive.cost, self.generation, self.ledger.nfe, time.perf_counter() - start)
        c = self.config
        while self.ledger.remaining and (c.max_iterations is None or self.generation < c.max_iterations):
            self.generation += 1
            best_before = self.archive.cost
            for index in range(len(self.population)):
                if not self.ledger.remaining:
                    break
                family, operator, probabilities = self.controller.select(
                    self.rng, self.stagnation, diversity(self.population), self.forced_family)
                self.attempt(index, family, operator, 'main', probabilities)
            self.refine_elite()
            if self.archive.cost < best_before:
                self.stagnation = 0
            else:
                self.stagnation += 1
            if (c.stagnation_control and self.stagnation >= c.stagnation_L2
                    and self.generation-self.last_restart_generation >= c.stagnation_L2):
                self.partial_restart()
            self.curve.append(self.archive.cost)
            if self.observer is not None:
                self.observer.on_generation(self.population, self.archive.cost, self.generation, self.ledger.nfe, time.perf_counter() - start)
            self._log('cycle_end', probabilities=self.controller.probabilities(
                self.stagnation, diversity(self.population)))
        best = self.archive
        # Validate only structural invariants, reuse stored objective and route
        # details. No uncounted candidate re-evaluation for final serialization.
        if (sorted(cust for route in best.routes for cust in route) != list(range(1, self.problem.dimension+1))
                or any(not r or sum(self.problem.demands[cust] for cust in r) > self.problem.capacity
                       for r in best.routes)):
            raise AssertionError('The global archive is not feasible')
        counts = {f: s.usage_count for f, s in self.controller.families.items()}
        total = sum(counts.values())
        main_total = sum(self.main_family_counts.values())
        metadata = {
            'objective': 'Duration', 'units': 'seconds', 'feasible': True,
            'encoding': 'permutation + shared minimum-duration Split',
            'departure_domain': [0, self.problem.horizon],
            'travel_time_domain': 'route departure restricted to nonempty T(R); no extrapolation',
            'model_version': 4, 'vehicle_limit_K': self.problem.vehicle_limit,
            'penalty_weights': {'miss': self.problem.alpha_miss, 'dup': self.problem.alpha_dup,
                                'cap': self.problem.alpha_cap, 'fleet': self.problem.alpha_fleet,
                                'M0': self.problem.penalty_fixed, 'M1': self.problem.penalty_multiplier},
            'evaluations': self.ledger.nfe, 'completed_iterations': self.generation,
            'stop_reason': 'nfe_budget' if not self.ledger.remaining else 'iteration_limit',
            'config': asdict(c), 'nfe_by_phase': dict(self.ledger.spending),
            'elite_nfe_cap': self.elite_cap, 'restart_nfe_cap': self.restart_cap,
            'operator_statistics': {op: asdict(stats) for op, stats in self.controller.operators.items()},
            'family_statistics': {f: asdict(s) for f, s in self.controller.families.items()},
            'family_selection_ratio': {f: counts[f]/total if total else 0 for f in counts},
            'main_family_selection_ratio': {f: count/main_total if main_total else 0
                                            for f, count in self.main_family_counts.items()},
            'elite_refinement_count': self.elite_refinement_count,
            'restart_count': self.restart_count, 'restart_candidate_count': self.restart_candidate_count,
            'restart_events': self.restart_events, 'search_log': self.logs,
            'baseline': 'discrete family adaptation' if self.forced_family else None,
            'implementation': 'permutation-aos-er-v3',
        }
        return OptimizationResult(
            algorithm=self.forced_family or c.variant, benchmark=self.problem.name, seed=self.seed,
            best_fitness=best.cost, best_solution=rank_keys(best.permutation).tolist(),
            best_permutation=list(best.permutation), routes=[[0, *r, 0] for r in best.routes],
            departure_times=[e[1] for e in best.evaluations],
            return_times=[e[2] for e in best.evaluations], route_durations=[e[0] for e in best.evaluations],
            route_loads=[sum(self.problem.demands[cust] for cust in r) for r in best.routes],
            convergence_curve=self.curve, runtime_seconds=time.perf_counter()-start,
            population_size=c.population_size, max_iterations=c.max_iterations if c.max_iterations is not None else self.generation,
            nfe=self.ledger.nfe, metadata=metadata, dimension=self.problem.dimension,
            hybrid_type='' if self.forced_family else 'ARO+POA')


class RAPO:
    name = 'RAPO-V3'
    hybrid_type = 'ARO+POA'

    def optimize(self, problem, config=None, seed=None, observer=None, initial_population=None):
        options = dict(config or {})
        profile = 'standard'
        if problem.dimension >= 50 and options.get('variant', 'RAPO-V3') == 'RAPO-V3':
            # Dimension-aware defaults derived from the n=50 diagnostic run.
            # Explicit caller values always take precedence.
            profile = 'n50-strong'
            strong_defaults = {
                'elite_ratio': 0.15,
                'elite_budget_fraction': 0.25,
                'elite_max_neighbors': 8,
                'destroy_fraction_min': 0.20,
                'destroy_fraction_max': 0.40,
                'initial_aro_probability': 0.55,
                'initial_poa_probability': 0.45,
                'ucb_c': 1.45,
            }
            for key, value in strong_defaults.items():
                options.setdefault(key, value)
        settings = RAPOConfig.from_mapping(options)
        result = SearchRun(problem, settings, seed, observer=observer, initial_population=initial_population).run()
        result.metadata['rapo_profile'] = profile
        result.metadata['implementation'] = 'permutation-aos-er-guided-v3.1'
        result.run_id = (config or {}).get('run_id', 1)
        return result


def optimize_family(family, problem, config=None, seed=None, observer=None, initial_population=None):
    """Minimal shared permutation adapter for standalone ARO and POA."""
    options = dict(config or {})
    if options.get('max_function_evaluations') is None:
        size = options.get('population_size', 30)
        iterations = options.get('max_iterations', 100)
        iterations = 100 if iterations is None else iterations
        options['max_function_evaluations'] = size * (1+iterations)
    options.update(variant='RAPO-Base', adaptive=False, elite_refinement=False,
                   stagnation_control=False, diversity_control=False)
    result = SearchRun(problem, RAPOConfig.from_mapping(options), seed, family, observer=observer, initial_population=initial_population).run()
    result.run_id = options.get('run_id', 1)
    return result


def main():
    parser = argparse.ArgumentParser(description='RAPO ablations, Hong Kong n=10 only')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--variant', choices=list(PRESETS))
    parser.add_argument('--method', choices=['hyb', 'poi', 'all'], default='hyb')
    parser.add_argument('--customers', type=int, choices=[10, 50], default=10)
    parser.add_argument('--traffic', choices=['bpr-light', 'bpr-moderate', 'bpr-heavy',
                        'wave-light', 'wave-moderate', 'wave-heavy', 'all'], default='bpr-light')
    parser.add_argument('--population-size', type=int)
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--max-evaluations', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    values = json.loads(args.config.read_text(encoding='utf-8')) if args.config else {}
    # A requested preset replaces flags from a V3 config, avoiding mislabeled ablations.
    if args.variant:
        for key in ('adaptive', 'elite_refinement', 'stagnation_control', 'diversity_control'):
            values.pop(key, None)
        values['variant'] = args.variant
    for arg, field_name in [('population_size', 'population_size'), ('iterations', 'max_iterations'),
                            ('max_evaluations', 'max_function_evaluations')]:
        if getattr(args, arg) is not None:
            values[field_name] = getattr(args, arg)
    settings = RAPOConfig.from_mapping(values)
    paths = [p for p in sorted(data_directory(args.customers).rglob('*.vrp.json'))
             if (args.method == 'all' or p.parent.parent.name.endswith('-'+args.method))
             and (args.traffic == 'all' or p.parent.name == args.traffic)]
    if not paths:
        parser.error(f'No Hong Kong n={args.customers} instances match the filters')
    results = []
    for path in paths:
        print(f'Loading {path.name}', flush=True)
        problem = TDVRPProblem(path)
        result = RAPO().optimize(problem, asdict(settings), args.seed)
        results.append(asdict(result))
        print(f'{result.algorithm}: {result.best_fitness:.6f} seconds; NFE={result.nfe}; routes={result.routes}', flush=True)
    output = args.output or ROOT / 'results' / f'{settings.variant.lower()}_hong_kong_n10.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
