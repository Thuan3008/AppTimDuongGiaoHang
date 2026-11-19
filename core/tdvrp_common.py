"""Shared feasible-only TDVRP encoding and piecewise-linear evaluation.

Equations (12)-(33) in MoHinhBaiToan_TDVRP.pdf: no waiting, time windows,
fleet penalty or return deadline. Random keys encode a permutation; dynamic
programming chooses its minimum-duration capacity-feasible partition.

Travel-time functions are never extrapolated beyond their declared domain.
Each route is minimized only over T(R), the departure times for which every
arc evaluation remains inside that domain (model revision 4, equations 27-33).
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLLECTION_ROOT = ROOT / 'data'
DATA_ROOT = COLLECTION_ROOT / 'TDVRP' / 'hong_kong'
DATA = DATA_ROOT / 'n=10'  # backward-compatible default used by existing tests


def data_directory(num_customers=10, city='hong_kong'):
    return COLLECTION_ROOT / 'TDVRP' / city / f'n={num_customers}'
# Use packages installed for the active interpreter. Never prepend the local
# .tdvrp-deps directory: its native wheels may target another Python version.
try:
    from mamut_routing_lib.td import load_td_instance, NDCPWLF
except ImportError as exc:
    raise ImportError(
        'TDVRP dependencies are missing or incompatible with the active Python. '
        'Run: python -m pip install --upgrade -r requirements.txt'
    ) from exc


@dataclass
class OptimizationResult:
    algorithm: str
    benchmark: str
    seed: int | None
    best_fitness: float
    best_solution: list[float]
    routes: list[list[int]]
    departure_times: list[float]
    return_times: list[float]
    route_durations: list[float]
    route_loads: list[int]
    convergence_curve: list[float]
    runtime_seconds: float
    population_size: int
    max_iterations: int
    nfe: int
    metadata: dict
    dimension: int = 10
    lower_bound: float = 0.0
    upper_bound: float = 1.0
    hybrid_type: str = ''
    run_id: int = 1
    best_permutation: list[int] | None = None


@dataclass(frozen=True)
class CandidateEvaluation:
    """Penalized model evaluation for an evaluator-domain-valid candidate."""
    routes: tuple[tuple[int, ...], ...]
    base_cost: float
    violation: float
    penalty: float
    fitness: float
    feasible: bool
    missing: int
    duplicates: int
    capacity_violation: float
    fleet_violation: int
    route_evaluations: tuple[tuple[float, float, float], ...]


class TDVRPProblem:
    dimension = 10  # fallback for small synthetic tests; real instances override it
    lower_bound = 0.0
    upper_bound = 1.0

    def __init__(self, instance_path, collection_root=COLLECTION_ROOT,
                 vehicle_limit=20, search_route_limit=None,
                 alpha_miss=1_000_000.0, alpha_dup=1_000_000.0,
                 alpha_cap=1_000_000.0, alpha_fleet=1_000_000.0,
                 penalty_multiplier=1.0):
        path = Path(instance_path).resolve()
        if not path.is_file() or not path.name.endswith('.vrp.json'):
            raise ValueError('Only existing .vrp.json instance files are allowed.')
        payload = json.loads(path.read_text(encoding='utf-8'))
        self.dimension = int(payload['num_customers'])
        for key in ('graph', 'traffic'):
            target = Path(collection_root) / payload['td'][key]['path']
            if not target.is_file():
                try:
                    from urllib.request import urlopen
                    BASE = 'https://raw.githubusercontent.com/ANR-MAMUT/MAMUT-routing-Poryos2026/main/'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(urlopen(BASE + payload['td'][key]['path'], timeout=120).read())
                except Exception as e:
                    raise FileNotFoundError(f'Missing {target} and auto-download failed: {e}. Run python scripts/prepare_tdvrp_data.py first.')
        loaded = load_td_instance(path, collection_root=collection_root, verify_sha256=True)
        self.loaded = loaded
        self.name = loaded.instance.instance_name
        self.demands = tuple(loaded.instance.demands)
        self.services = tuple(loaded.instance.service_times)
        self.capacity = loaded.instance.vehicle_capacity
        self.horizon = float(loaded.instance.horizon[1])
        self.vehicle_limit = int(vehicle_limit)
        self.search_route_limit = int(search_route_limit or max(2*self.vehicle_limit, self.dimension))
        self.alpha_miss = float(alpha_miss)
        self.alpha_dup = float(alpha_dup)
        self.alpha_cap = float(alpha_cap)
        self.alpha_fleet = float(alpha_fleet)
        self.penalty_multiplier = float(penalty_multiplier)
        if self.vehicle_limit < 1 or self.search_route_limit <= self.vehicle_limit:
            raise ValueError('Require 1 <= K < search_route_limit to make fleet violations searchable.')
        if any(not np.isfinite(x) or x <= 0 for x in
               (self.alpha_miss, self.alpha_dup, self.alpha_cap,
                self.alpha_fleet, self.penalty_multiplier)):
            raise ValueError('All penalty weights and M1 must be finite and positive.')
        if loaded.instance.depot != 0 or loaded.instance.horizon[0] != 0 or self.services[0] != 0:
            raise ValueError('The model requires depot=0, s0=0 and departure domain [0,H].')
        if any(d <= 0 or d > self.capacity for d in self.demands[1:]):
            raise ValueError('The model requires 0 < demand <= capacity.')
        self.arcs = dict(loaded.atfs.arcs)
        max_travel = max(max(y-x for x, y in zip(f.xs, f.ys)) for f in self.arcs.values())
        # For every evaluator-valid route q<=n, duration is bounded by this
        # expression. Bounding m makes the candidate space finite (PDF §1.7).
        self.base_cost_upper_bound = self.search_route_limit * (
            (self.dimension + 1) * max_travel + self.dimension * max(self.services))
        self.penalty_fixed = float(np.nextafter(self.base_cost_upper_bound, np.inf))
        self.route_cost = lru_cache(maxsize=100000)(self._route_cost)
        self.split = lru_cache(maxsize=100000)(self._split)

    def _route_cost(self, route):
        if not route or len(set(route)) != len(route) or any(c not in range(1, self.dimension+1) for c in route):
            raise ValueError(f'A route must contain distinct customer indices in 1..{self.dimension}.')
        if sum(self.demands[c] for c in route) > self.capacity:
            raise ValueError('Capacity exceeded.')
        composed = NDCPWLF.identity(0.0, self.horizon)
        previous = 0
        for customer in route:
            composed = self.arcs[previous, customer].compose(composed)
            composed = NDCPWLF(list(composed.xs), [y + self.services[customer] for y in composed.ys])
            previous = customer
        composed = self.arcs[previous, 0].compose(composed)
        if composed.is_empty():
            raise ValueError('Route has empty T(R): travel-time data cannot evaluate it.')
        duration, departure = composed.min_shifted_image()
        # Composition and sequential replay can differ by one ULP exactly at
        # a Harc boundary. Move only toward the interior until executable.
        for _ in range(256):
            returned = self.simulate_route(route, departure)
            if returned is not None:
                return returned-departure, departure, returned
            departure = float(np.nextafter(departure, -np.inf))
            if departure < composed.min_domain:
                break
        raise ValueError('Numerical boundary left no replayable departure in T(R).')

    def simulate_route(self, route, departure):
        """Replay one route; return depot arrival, or None outside Harc."""
        clock = float(departure)
        previous = 0
        for customer in route:
            arc = self.arcs.get((previous, customer))
            if arc is None or clock < arc.min_domain or clock > arc.max_domain:
                return None
            clock = arc(clock) + self.services[customer]
            previous = customer
        arc = self.arcs.get((previous, 0))
        if arc is None or clock < arc.min_domain or clock > arc.max_domain:
            return None
        return arc(clock)

    def _split(self, permutation):
        if sorted(permutation) != list(range(1, self.dimension+1)):
            raise ValueError('Each customer must appear exactly once.')
        costs = [0.0] + [float('inf')] * self.dimension
        partitions = [()] + [None] * self.dimension
        for end in range(1, self.dimension+1):
            load = 0
            for start in range(end - 1, -1, -1):
                load += self.demands[permutation[start]]
                if load > self.capacity:
                    break
                route = permutation[start:end]
                try:
                    r_cost = self.route_cost(route)[0]
                except ValueError:
                    continue
                candidate = costs[start] + r_cost
                if candidate < costs[end]:
                    costs[end] = candidate
                    partitions[end] = partitions[start] + (route,)
        if partitions[self.dimension] is None:
            raise ValueError('No feasible partition found.')
        routes = tuple(sorted(partitions[self.dimension], key=lambda r: r[0]))
        return sum(self.route_cost(r)[0] for r in routes), routes

    def evaluate(self, keys):
        keys = np.asarray(keys, dtype=float)
        if keys.shape != (self.dimension,) or not np.isfinite(keys).all():
            raise ValueError(f'Expected {self.dimension} finite random keys.')
        return self.evaluate_permutation(tuple(int(i)+1 for i in np.argsort(keys, kind='stable')))

    def evaluate_permutation(self, permutation):
        """Shared entry point; reuse the existing Split and route evaluator."""
        permutation = tuple(permutation)
        if any(not isinstance(c, (int, np.integer)) for c in permutation):
            raise ValueError('Customer indices must be integers.')
        return self.split(permutation)

    def evaluate_candidate(self, routes):
        """Evaluate C(S)+M0*1[V>0]+M1*V over the bounded candidate space.

        Domain-invalid candidates are rejected: illegal nodes/depot placement,
        empty routes, missing arcs, q>n, too many search routes, or empty T(R).
        Business violations are retained and penalized as specified by the PDF.
        """
        routes = tuple(tuple(route) for route in routes)
        if not routes or len(routes) > self.search_route_limit:
            raise ValueError('Candidate must have 1..search_route_limit routes.')
        if any(not route or len(route) > self.dimension for route in routes):
            raise ValueError('Every route must contain 1..n customers.')
        if any(type(customer) is not int or customer not in range(1, self.dimension+1)
               for route in routes for customer in route):
            raise ValueError('Routes may contain customer indices 1..n only; depot is implicit.')
        counts = np.bincount([c for route in routes for c in route], minlength=self.dimension+1)
        missing = int(np.maximum(0, 1-counts[1:]).sum())
        duplicates = int(np.maximum(0, counts[1:]-1).sum())
        capacity_violation = float(sum(max(0.0,
            (sum(self.demands[c] for c in route)-self.capacity)/self.capacity)
            for route in routes))
        fleet_violation = max(0, len(routes)-self.vehicle_limit)
        violation = (self.alpha_miss*missing + self.alpha_dup*duplicates +
                     self.alpha_cap*capacity_violation + self.alpha_fleet*fleet_violation)
        evaluations = tuple(self.route_cost(route) for route in routes)
        base_cost = sum(item[0] for item in evaluations)
        penalty = 0.0 if violation == 0 else self.penalty_fixed + self.penalty_multiplier*violation
        return CandidateEvaluation(routes, base_cost, violation, penalty,
                                   base_cost+penalty, violation == 0, missing,
                                   duplicates, capacity_violation,
                                   fleet_violation, evaluations)

    def validate(self, routes):
        if not routes or any(not r for r in routes):
            raise ValueError('Empty solution/route.')
        if sorted(c for route in routes for c in route) != list(range(1, self.dimension+1)):
            raise ValueError('Missing or repeated customer.')
        evaluation = self.evaluate_candidate(routes)
        if not evaluation.feasible:
            raise ValueError('Candidate violates customer, capacity, or fleet constraints.')
        return list(evaluation.route_evaluations)


def optimize_population(name, problem, config, seed, propose):
    if not isinstance(problem, TDVRPProblem):
        raise TypeError('Pass a TDVRPProblem, not a continuous benchmark.')
    if not config.get('minimization', True):
        raise ValueError('TDVRP minimizes total route duration.')
    size = int(config.get('population_size', 30))
    iterations = int(config.get('max_iterations', 100))
    if size < 2 or iterations < 0:
        raise ValueError('population_size >= 2 and max_iterations >= 0 are required.')
    budget = int(config.get('max_function_evaluations', 10000)) if config.get('use_evaluation_budget', False) else None
    if budget is not None and budget < size:
        raise ValueError('Evaluation budget must cover initialization.')
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    population = rng.random((size, 10))
    fitness = np.array([problem.evaluate(x)[0] for x in population])
    nfe = size
    history = []
    for t in range(iterations):
        if budget is not None and nfe >= budget:
            break
        for i in range(size):
            for candidate in propose(population, fitness, i, t, iterations, rng):
                if budget is not None and nfe >= budget:
                    break
                candidate = np.clip(candidate, 0, 1)
                value = problem.evaluate(candidate)[0]
                nfe += 1
                if value < fitness[i]:
                    population[i], fitness[i] = candidate, value
            if budget is not None and nfe >= budget:
                break
        history.append(float(np.min(fitness)))
    best = int(np.argmin(fitness))
    cost, routes = problem.evaluate(population[best])
    evaluations = problem.validate(routes)
    assert cost == sum(e[0] for e in evaluations)
    return OptimizationResult(
        algorithm=name, benchmark=problem.name, seed=seed, best_fitness=cost,
        best_solution=population[best].tolist(), routes=[[0, *r, 0] for r in routes],
        departure_times=[e[1] for e in evaluations], return_times=[e[2] for e in evaluations],
        route_durations=[e[0] for e in evaluations],
        route_loads=[sum(problem.demands[c] for c in r) for r in routes],
        convergence_curve=history, runtime_seconds=time.perf_counter()-start,
        population_size=size, max_iterations=iterations, nfe=nfe,
        dimension=problem.dimension,
        run_id=config.get('run_id', 1),
        metadata={'objective': 'Duration', 'units': 'seconds', 'feasible': True,
                  'departure_domain': [0, problem.horizon],
                  'travel_time_domain': 'route departure restricted to nonempty T(R); no extrapolation',
                  'model_version': 4, 'vehicle_limit_K': problem.vehicle_limit,
                  'encoding': 'random keys + capacity-feasible minimum-duration split',
                  'evaluations': nfe, 'completed_iterations': len(history)})


def run_cli(optimizer_class):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['hyb', 'poi', 'all'], default='all')
    parser.add_argument('--customers', type=int, choices=[10, 50], default=10)
    parser.add_argument('--traffic', choices=['bpr-light', 'bpr-moderate', 'bpr-heavy', 'wave-light', 'wave-moderate', 'wave-heavy', 'all'], default='all')
    parser.add_argument('--population-size', type=int, default=30)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-evaluations', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    paths = [p for p in sorted(data_directory(args.customers).rglob('*.vrp.json'))
             if (args.method == 'all' or p.parent.parent.name.endswith('-'+args.method))
             and (args.traffic == 'all' or p.parent.name == args.traffic)]
    if not paths:
        parser.error(f'No Hong Kong n={args.customers} instances found.')
    config = {'population_size': args.population_size, 'max_iterations': args.iterations,
              'use_evaluation_budget': args.max_evaluations is not None,
              'max_function_evaluations': args.max_evaluations}
    results = []
    for path in paths:
        print(f'Loading {path.name}', flush=True)
        problem = TDVRPProblem(path)
        result = optimizer_class().optimize(problem, config, args.seed)
        results.append(asdict(result))
        print(f'{result.algorithm}: {result.best_fitness:.6f} seconds; routes={result.routes}; NFE={result.nfe}', flush=True)
    output = args.output or ROOT / 'results' / f'{optimizer_class.name.lower()}_hong_kong_n10.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(f'Saved {output}')
