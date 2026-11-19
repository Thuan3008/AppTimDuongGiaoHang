"""Isolated timed-run support. Never imported by the original NFE solvers."""
from collections import defaultdict
from time import perf_counter
from algorithms.common.ledger import Candidate


class TimeExpired(Exception):
    pass


class TimedLedger:
    def __init__(self, problem, limit):
        self.problem = problem
        self.limit = limit  # large safety guard only; deadline is the stop rule
        self.nfe = 0
        self.spending = defaultdict(int)

    @property
    def remaining(self):
        # Raise, rather than return zero: original NFE assertions must not run.
        self.problem.time_budget.check()
        return self.limit - self.nfe

    def evaluate(self, permutation, phase):
        self.problem.time_budget.check()
        self.nfe += 1
        self.spending[phase] += 1
        self.problem.time_budget.nfe = self.nfe
        cost, routes = self.problem.evaluate_permutation(tuple(permutation))
        details = tuple(self.problem.route_cost(r) for r in routes)
        candidate = Candidate(tuple(permutation), cost, routes, details)
        self.problem.time_budget.record(candidate)
        return candidate


class TimeBudget:
    def __init__(self, seconds):
        self.start = perf_counter()
        self.seconds = seconds
        self.nfe = 0
        self.best = None
        self.history = []

    @property
    def fraction(self):
        return min(1.0, (perf_counter() - self.start) / self.seconds)

    def check(self):
        if perf_counter() - self.start >= self.seconds:
            raise TimeExpired

    def record(self, candidate):
        if self.best is None or candidate.cost < self.best.cost:
            self.best = candidate
            elapsed = perf_counter() - self.start
            self.history.append({'seconds': elapsed, 'duration': candidate.cost,
                                 'nfe': self.nfe, 'completed_within_deadline': elapsed <= self.seconds})
