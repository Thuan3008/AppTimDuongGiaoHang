from collections import Counter
import math

import pytest

from algorithms.td_constructive import (
    ALGORITHMS, TDHeuristicConfig, neighborhood, solve,
)


class AsymmetricProblem:
    """Small independent route oracle with an expensive 2->1 direction."""
    dimension = 4
    vehicle_limit = 4
    capacity = 2
    demands = (0, 1, 1, 1, 1)

    def route_cost(self, route):
        if len(set(route)) != len(route) or any(c not in range(1, 5) for c in route):
            raise ValueError('invalid route')
        if len(route) > self.capacity:
            raise ValueError('capacity')
        # Departure depends on route order: solvers must use the oracle.
        departure = sum((i + 1) * c for i, c in enumerate(route))
        duration = 20 + sum(30 if (a, b) == (2, 1) else 1 for a, b in zip(route, route[1:]))
        return duration, departure, departure + duration

    def validate(self, routes):
        if Counter(c for r in routes for c in r) != Counter(range(1, 5)) or any(not r for r in routes):
            raise ValueError('missing, duplicate, empty')
        return [self.route_cost(r) for r in routes]


@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_feasible_deterministic_exact_cost_and_departures(algorithm):
    problem = AsymmetricProblem()
    first, second = solve(problem, algorithm), solve(problem, algorithm)
    assert first.routes == second.routes
    assert first.duration == second.duration == 42
    assert first.feasible
    assert first.departure_times == [problem.route_cost(r)[1] for r in first.routes]
    assert sum(first.route_durations) == first.duration
    assert all(a['duration'] > b['duration'] for a, b in zip(first.history, first.history[1:]))
    assert first.statistics['route_evaluation_requests'] == first.statistics['route_evaluations'] + first.statistics['cache_hits'] + first.statistics['capacity_rejections']
    assert not first.metadata['nfe_equivalence']


def test_savings_keeps_direction_and_recalculates_after_merge():
    class Unlimited(AsymmetricProblem):
        capacity = 4
    result = solve(Unlimited(), 'TD-Savings')
    assert len(result.routes) == 1
    assert result.duration == 23
    assert all((a, b) != (2, 1) for a, b in zip(result.routes[0], result.routes[0][1:]))
    assert result.statistics['accepted_moves'] == 3
    assert result.termination == 'no_positive_savings'


def test_local_search_improves_initial_and_exhausts_declared_neighborhoods():
    problem = AsymmetricProblem()
    result = solve(problem, 'TD-Local-Search', initial_routes=[[2, 1], [4, 3]])
    assert result.duration == 42 < 71
    assert result.termination == 'local_optimum_in_stated_neighborhoods'
    routes = tuple(tuple(r) for r in result.routes)
    for _, indices, replacements in neighborhood(routes):
        candidate = [r for i, r in enumerate(routes) if i not in indices] + [r for r in replacements if r]
        try:
            values = problem.validate(candidate)
        except ValueError:
            continue
        assert math.fsum(v[0] for v in values) >= result.duration - 1e-9


def test_neighborhood_preserves_customers_including_empty_route_and_new_vehicle():
    routes = ((1,), (2, 3, 4))
    kinds = set()
    for kind, indices, replacements in neighborhood(routes):
        candidate = [r for i, r in enumerate(routes) if i not in indices] + [r for r in replacements if r]
        assert Counter(c for r in candidate for c in r) == Counter(range(1, 5))
        kinds.add(kind)
    assert kinds == {'relocate', 'relocate_new_route', 'swap', 'two_opt_star'}


@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_tiny_time_budget_returns_complete_incumbent_not_false_local_optimum(algorithm):
    result = solve(AsymmetricProblem(), algorithm, TDHeuristicConfig(time_limit_seconds=1e-12))
    assert result.feasible and result.duration == 80
    assert result.termination.startswith('time_limit')
    AsymmetricProblem().validate(result.routes)


def test_reject_invalid_input_and_unsupported_fleet():
    with pytest.raises(ValueError):
        solve(AsymmetricProblem(), 'TD-Local-Search', initial_routes=[[0, 1, 2], [3, 4]])
    problem = AsymmetricProblem()
    problem.vehicle_limit = 2
    with pytest.raises(ValueError, match='unrestricted'):
        solve(problem, 'TD-Savings')
    with pytest.raises(ValueError):
        TDHeuristicConfig(time_limit_seconds=float('nan'))


def test_unavailable_route_domain_is_rejected_without_static_fallback():
    class DomainRestricted(AsymmetricProblem):
        def route_cost(self, route):
            if tuple(route) == (1, 2):
                raise ValueError('empty departure domain')
            return super().route_cost(route)
    result = solve(DomainRestricted(), 'TD-Insertion')
    assert result.feasible
    assert [1, 2] not in result.routes
    assert result.statistics['domain_rejections'] > 0


@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_real_piecewise_linear_oracle_optimizes_departure_and_replays(algorithm):
    from functools import lru_cache
    from mamut_routing_lib.td import NDCPWLF
    from core.tdvrp_common import TDVRPProblem

    class TimeDependent(AsymmetricProblem):
        horizon = 100.0
        services = (0, 1, 1, 1, 1)
        arcs = {(a, b): NDCPWLF([0., 50., 100.], [10., 55., 110.])
                for a in range(5) for b in range(5) if a != b}
        simulate_route = TDVRPProblem.simulate_route

        def __init__(self):
            self.route_cost = lru_cache(maxsize=1000)(
                lambda route: TDVRPProblem._route_cost(self, tuple(route)))

    problem = TimeDependent()
    result = solve(problem, algorithm)
    assert result.feasible
    assert all(t > 0 for t in result.departure_times)
    for route, duration, departure, returned in zip(result.routes, result.route_durations,
                                                   result.departure_times, result.return_times):
        replay = problem.simulate_route(route, departure)
        assert replay == pytest.approx(returned)
        assert duration == pytest.approx(returned - departure)
        assert duration < problem.simulate_route(route, 0)
