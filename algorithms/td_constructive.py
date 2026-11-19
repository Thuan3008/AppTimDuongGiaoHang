"""Deterministic TDVRP baselines; see docs/td_heuristics.md for source mapping.

Customer-only routes, exact departure-optimized Duration, no giant-tour Split.
These are transparent TD adaptations, not full TREE or BonnTour reproductions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from time import perf_counter
from typing import Iterable

Route = tuple[int, ...]
Routes = tuple[Route, ...]
ALGORITHMS = ('TD-Insertion', 'TD-Savings', 'TD-Local-Search')


@dataclass(frozen=True)
class TDHeuristicConfig:
    time_limit_seconds: float | None = None
    improvement_tolerance: float = 1e-9

    def __post_init__(self):
        if self.time_limit_seconds is not None and (not math.isfinite(self.time_limit_seconds) or self.time_limit_seconds <= 0):
            raise ValueError('time_limit_seconds must be finite and positive')
        if not math.isfinite(self.improvement_tolerance) or self.improvement_tolerance < 0:
            raise ValueError('improvement_tolerance must be finite and nonnegative')


@dataclass
class TDHeuristicResult:
    algorithm: str
    routes: list[list[int]]
    duration: float
    departure_times: list[float]
    return_times: list[float]
    route_durations: list[float]
    runtime_seconds: float
    feasible: bool
    termination: str
    statistics: dict
    history: list[dict]
    metadata: dict

    def to_dict(self):
        return asdict(self)


class _TimeLimit(Exception):
    pass


class _Search:
    def __init__(self, problem, config):
        self.problem, self.config = problem, config
        self.start = perf_counter()
        self.costs: dict[Route, tuple[float, float, float] | None] = {}
        self.requests = self.evaluations = self.hits = self.capacity_rejects = self.domain_rejects = 0
        self.moves = self.accepted = 0
        self.history = []
        self.best: Routes = ()
        self.best_cost = math.inf
        # The reference problem is the thesis no-TW, unrestricted-fleet model.
        loaded = getattr(problem, 'loaded', None)
        if loaded is not None and getattr(loaded.instance, 'time_windows', None) is not None:
            raise ValueError('This adapter supports TDVRP without time windows only')
        declared_fleet = getattr(loaded.instance, 'num_vehicles', None) if loaded is not None else None
        if declared_fleet is not None and declared_fleet < problem.dimension:
            raise ValueError('Instance declares a restricted fleet; this adaptation requires unrestricted fleet')
        if problem.vehicle_limit < problem.dimension:
            raise ValueError('These baselines require an unrestricted fleet (vehicle_limit >= n)')
        if problem.dimension < 1:
            raise ValueError('At least one customer is required')

    def check_time(self):
        if self.config.time_limit_seconds is not None and perf_counter() - self.start >= self.config.time_limit_seconds:
            raise _TimeLimit

    def evaluate_route(self, route, *, check_time=True):
        if check_time:
            self.check_time()
        route = tuple(route)
        if not route:
            return (0.0, 0.0, 0.0)
        self.requests += 1
        if route in self.costs:
            self.hits += 1
            return self.costs[route]
        if sum(self.problem.demands[c] for c in route) > self.problem.capacity:
            self.capacity_rejects += 1
            self.costs[route] = None
            return None
        self.evaluations += 1
        try:
            value = tuple(float(x) for x in self.problem.route_cost(route))
        except ValueError:
            self.domain_rejects += 1
            value = None
        if value is not None and (len(value) != 3 or not all(math.isfinite(x) for x in value) or value[0] < 0):
            raise ValueError('Route evaluator returned invalid duration/departure/return')
        self.costs[route] = value
        return value

    def remember(self, routes, phase):
        routes = tuple(sorted((tuple(r) for r in routes if r), key=lambda r: r))
        values = [self.costs[r] for r in routes]
        if any(v is None for v in values):
            raise ValueError('Attempted to accept an infeasible route')
        cost = math.fsum(v[0] for v in values)
        if cost < self.best_cost:
            self.best, self.best_cost = routes, cost
            self.history.append({'seconds': perf_counter() - self.start, 'duration': cost,
                                 'route_evaluations': self.evaluations, 'phase': phase})
        elif cost == self.best_cost:
            self.best = routes

    def singletons(self):
        routes = tuple((c,) for c in range(1, self.problem.dimension + 1))
        # Establish a feasible incumbent even for very short time limits.
        # These n startup evaluations are charged but cannot be interrupted.
        for route in routes:
            if self.evaluate_route(route, check_time=False) is None:
                raise ValueError(f'No feasible singleton route for customer {route[0]}')
        self.remember(routes, 'singleton_initialization')
        return routes


def _insertion(search):
    """Parallel cheapest feasible insertion, including opening a new route.

    Route/customer descriptors survive unchanged routes (Visser-Spliet §8).
    No static travel-time insertion proxy is used for ranking.
    """
    routes: Routes = ()
    unplanned = set(range(1, search.problem.dimension + 1))
    descriptors = {}
    while unplanned:
        search.check_time()
        best = None
        for customer in sorted(unplanned):
            singleton = (customer,)
            choices = [(search.costs[singleton][0], len(routes), 0, singleton)]
            for index, route in enumerate(routes):
                key = route, customer
                if key not in descriptors:
                    choice = None
                    for position in range(len(route) + 1):
                        candidate = route[:position] + singleton + route[position:]
                        value = search.evaluate_route(candidate)
                        search.moves += 1
                        if value is not None:
                            item = (value[0] - search.costs[route][0], position, candidate)
                            if choice is None or item < choice:
                                choice = item
                    descriptors[key] = choice
                choice = descriptors[key]
                if choice is not None:
                    choices.append((choice[0], index, choice[1], choice[2]))
            delta, index, position, candidate = min(choices)
            item = (delta, customer, index, position, candidate)
            if best is None or item < best:
                best = item
        _, customer, index, _, candidate = best
        updated = list(routes)
        if index == len(routes):
            updated.append(candidate)
        else:
            updated[index] = candidate
        routes = tuple(updated)
        unplanned.remove(customer)
        search.accepted += 1
        search.remember(routes + tuple((c,) for c in sorted(unplanned)), 'insertion')
    return routes


def _savings(search, routes):
    """Dynamic best-positive savings, ordered concatenation, no reversal.

    All ordered pairs cover A+B and B+A; descriptors are cached by route tuple.
    Savings are recomputed for new merged routes, unlike static CW edge savings.
    """
    descriptors = {}
    while True:
        search.check_time()
        best = None
        for i, left in enumerate(routes):
            for j, right in enumerate(routes):
                if i == j:
                    continue
                key = left, right
                if key not in descriptors:
                    value = search.evaluate_route(left + right)
                    search.moves += 1
                    descriptors[key] = (search.costs[left][0] + search.costs[right][0] - value[0]
                                        if value is not None else -math.inf)
                gain = descriptors[key]
                item = (-gain, i, j)
                if gain > search.config.improvement_tolerance and (best is None or item < best):
                    best = item
        if best is None:
            return routes
        _, i, j = best
        routes = tuple(r for k, r in enumerate(routes) if k not in (i, j)) + (routes[i] + routes[j],)
        search.accepted += 1
        search.remember(routes, 'savings_merge')


def neighborhood(routes: Routes):
    """Deterministic relocate (including new vehicle), swap, and 2-opt*.

    Yield only changed route indices and replacement routes, so unchanged
    routes need not be repriced. No reversal or asymmetric 2-opt shortcut.
    """
    for i, route in enumerate(routes):
        for p, customer in enumerate(route):
            reduced = route[:p] + route[p + 1:]
            for j, target in enumerate(routes):
                if i == j:
                    for q in range(len(reduced) + 1):
                        new = reduced[:q] + (customer,) + reduced[q:]
                        if new != route:
                            yield 'relocate', (i,), (new,)
                else:
                    for q in range(len(target) + 1):
                        yield 'relocate', (i, j), (reduced, target[:q] + (customer,) + target[q:])
            if reduced:
                yield 'relocate_new_route', (i,), (reduced, (customer,))
    for i in range(len(routes)):
        for j in range(i, len(routes)):
            left, right = routes[i], routes[j]
            for p in range(len(left)):
                for q in range(p + 1 if i == j else 0, len(right)):
                    if i == j:
                        new = list(left)
                        new[p], new[q] = new[q], new[p]
                        yield 'swap', (i,), (tuple(new),)
                    else:
                        yield 'swap', (i, j), (left[:p] + (right[q],) + left[p + 1:],
                                               right[:q] + (left[p],) + right[q + 1:])
            if i != j:
                for p in range(len(left) + 1):
                    for q in range(len(right) + 1):
                        new = (left[:p] + right[q:], right[:q] + left[p:])
                        if new != (left, right) and new != (right, left):
                            yield 'two_opt_star', (i, j), new


def _local_search(search, routes):
    """Best-improvement descent across the union of the stated neighborhoods."""
    while True:
        search.check_time()
        best = None
        best_gain = search.config.improvement_tolerance
        for name, indices, replacements in neighborhood(routes):
            search.check_time()
            search.moves += 1
            values = [search.evaluate_route(r) for r in replacements]
            if any(v is None for v in values):
                continue
            old = math.fsum(search.costs[routes[i]][0] for i in indices)
            new = math.fsum(v[0] for v in values)
            gain = old - new
            if gain > best_gain:
                best_gain, best = gain, (name, indices, replacements)
        if best is None:
            return routes
        name, indices, replacements = best
        routes = tuple(r for i, r in enumerate(routes) if i not in indices) + tuple(r for r in replacements if r)
        search.accepted += 1
        search.remember(routes, name)


def solve(problem, algorithm: str, config: TDHeuristicConfig | None = None,
          *, initial_routes: Iterable[Iterable[int]] | None = None) -> TDHeuristicResult:
    """Solve the thesis TDVRP via its existing TDVRPProblem route oracle.

    time_limit is cooperative (between route evaluations); startup/final
    validation are never skipped. Route evaluations are NOT giant-tour NFE.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f'Unknown algorithm: {algorithm}')
    if initial_routes is not None and algorithm != 'TD-Local-Search':
        raise ValueError('initial_routes is only supported by TD-Local-Search')
    search = _Search(problem, config or TDHeuristicConfig())
    phase = 'initialization'
    if initial_routes is not None:
        routes = tuple(tuple(r) for r in initial_routes)
        if any(type(c) is not int for r in routes for c in r):
            raise ValueError('Routes must contain integer customer IDs')
        problem.validate(routes)
        for r in routes:
            if search.evaluate_route(r, check_time=False) is None:
                raise ValueError('Invalid initial route')
        search.remember(routes, 'provided_initial_solution')
        initial_policy = 'provided_customer_routes'
    else:
        routes = search.singletons()
        initial_policy = 'parallel_cheapest_insertion' if algorithm == 'TD-Local-Search' else 'singleton_routes'
    try:
        if algorithm == 'TD-Savings':
            phase = 'savings'
            _savings(search, routes)
            termination = 'no_positive_savings'
        else:
            if initial_routes is None:
                phase = 'insertion'
                routes = _insertion(search)
            if algorithm == 'TD-Local-Search':
                phase = 'local_search'
                _local_search(search, routes)
                termination = 'local_optimum_in_stated_neighborhoods'
            else:
                termination = 'all_customers_inserted'
    except _TimeLimit:
        termination = f'time_limit_during_{phase}'
    # Independent full-solution validation, no extra Split/repartitioning.
    evaluations = problem.validate(search.best)
    duration = math.fsum(float(e[0]) for e in evaluations)
    if not math.isclose(duration, search.best_cost, abs_tol=1e-6, rel_tol=1e-12):
        raise RuntimeError('Final solution cost differs from cached route costs')
    return TDHeuristicResult(
        algorithm, [list(r) for r in search.best], duration,
        [float(e[1]) for e in evaluations], [float(e[2]) for e in evaluations],
        [float(e[0]) for e in evaluations], perf_counter() - search.start, True,
        termination, {'route_evaluation_requests': search.requests, 'route_evaluations': search.evaluations,
                      'cache_hits': search.hits, 'capacity_rejections': search.capacity_rejects,
                      'domain_rejections': search.domain_rejects, 'candidate_moves': search.moves,
                      'accepted_moves': search.accepted}, search.history,
        {'deterministic': True, 'initialization': initial_policy, 'objective': 'Duration',
         'route_encoding': 'customer IDs, implicit depot', 'departure': 'exact route oracle',
         'split': False, 'nfe_equivalence': False, 'config': asdict(search.config),
         'time_limit': 'cooperative; includes startup and final validation, may overrun',
         'implementation_scope': 'TD adaptations; not full TREE/BonnTour implementations'})
