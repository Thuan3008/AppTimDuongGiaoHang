import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from algorithms import td_erapo_timed as timed
from algorithms.timed_runtime import TimeBudget, TimeExpired, TimedLedger


def test_original_td_erapo_operator_functions_are_preserved_in_copy():
    root = Path(__file__).resolve().parents[1]
    before = ast.parse((root / 'algorithms/rapo_2709.py').read_text(encoding='utf-8-sig'))
    after = ast.parse((root / 'algorithms/td_erapo_timed.py').read_text(encoding='utf-8'))
    functions = {node.name: node for node in after.body if isinstance(node, ast.FunctionDef)}
    for node in before.body:
        if isinstance(node, ast.FunctionDef) and node.name != 'optimize':
            old = ast.unparse(node).replace('RAPO2709Candidate', 'TDERAPOTimedCandidate').replace('RAPO2709Config', 'TDERAPOTimedConfig')
            assert old == ast.unparse(functions[node.name])


def test_time_ledger_does_not_evaluate_after_deadline(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(timed, 'perf_counter', lambda: clock[0])
    class Evaluator:
        calls = 0
        def evaluate(self, p):
            self.calls += 1
            return SimpleNamespace(fitness=10.)
    evaluator = Evaluator()
    ledger = timed._TimeLedger(evaluator, 2., 0.)
    assert ledger.evaluate((1,), 'initialization').fitness == 10
    clock[0] = 2.
    assert ledger.evaluate((1,), 'Elite') is None
    assert ledger.used == evaluator.calls == 1
    assert ledger.history[0]['completed_within_deadline']


def test_baseline_timed_ledger_keeps_incumbent_on_deadline(monkeypatch):
    import algorithms.timed_runtime as rt
    clock = [0.]
    monkeypatch.setattr(rt, 'perf_counter', lambda: clock[0])
    budget = TimeBudget(1.)
    class Problem:
        time_budget = budget
        def evaluate_permutation(self, p):
            return 7., ((1,),)
        def route_cost(self, r):
            return 7., 0., 7.
    ledger = TimedLedger(Problem(), 10**9)
    ledger.evaluate((1,), 'initialization')
    clock[0] = 1.
    with pytest.raises(TimeExpired):
        ledger.evaluate((1,), 'main')
    assert budget.best.cost == 7. and budget.nfe == 1


def test_timed_configuration_rejects_invalid_time():
    with pytest.raises(ValueError):
        timed.TDERAPOTimedConfig(time_limit_seconds=0).validate()
    assert not hasattr(timed.TDERAPOTimedConfig(), 'max_function_evaluations')


@pytest.mark.parametrize('name', ['GA', 'GA-LS', 'ARO-Original', 'POA-Original'])
def test_timed_baseline_search_continues_beyond_initial_population(name, monkeypatch):
    from core.tdvrp_common import TDVRPProblem
    from algorithms.ga_timed import GA
    from algorithms.ga_ls_timed import GALS
    from algorithms.aro_original_timed import AROOriginal
    from algorithms.poa_original_timed import POAOriginal
    import algorithms.timed_runtime as rt
    clock = [0.]
    def tick():
        clock[0] += .001
        return clock[0]
    monkeypatch.setattr(rt, 'perf_counter', tick)
    problem = object.__new__(TDVRPProblem)
    problem.dimension = 4
    problem.evaluate_permutation = lambda p: (float(sum(i * c for i, c in enumerate(p, 1))), (tuple(p),))
    problem.route_cost = lambda r: (float(sum(i * c for i, c in enumerate(r, 1))), 0., 10.)
    problem.time_budget = TimeBudget(.03)
    cls = {'GA': GA, 'GA-LS': GALS, 'ARO-Original': AROOriginal, 'POA-Original': POAOriginal}[name]
    with pytest.raises(TimeExpired):
        cls().optimize(problem, {'population_size': 2, 'max_function_evaluations': 10**9},
                       seed=0, initial_population=[(1, 2, 3, 4), (4, 3, 2, 1)])
    assert problem.time_budget.nfe > 2
    assert problem.time_budget.best is not None


def test_td_erapo_timed_retains_feasible_archive_and_online_elite_cap(monkeypatch):
    clock = [0.]
    def tick():
        clock[0] += .001
        return clock[0]
    monkeypatch.setattr(timed, 'perf_counter', tick)
    class Evaluator:
        def __init__(self, *args, **kwargs): pass
        def _validate_permutation(self, p): assert sorted(p) == [1, 2, 3]
        def evaluate(self, p):
            p = tuple(p)
            return timed.TDERAPOTimedCandidate(p, (p,), (), float(sum(i*c for i,c in enumerate(p, 1))))
        def materialize(self, c):
            return SimpleNamespace(routes=[[0, *c.permutation, 0]]), SimpleNamespace(fitness=c.fitness, feasible=True)
    monkeypatch.setattr(timed, '_ExactSplitEvaluator', Evaluator)
    monkeypatch.setattr(timed, 'poa_restructure', lambda p, e, c, r: tuple(reversed(p.permutation)))
    result = timed.optimize(SimpleNamespace(num_customers=3), None, None,
                           timed.TDERAPOTimedConfig(population_size=2, time_limit_seconds=.1),
                           seed=0, initial_population=[(1,2,3), (3,2,1)])
    assert result.best_evaluation.feasible
    assert result.nfe > 2
    assert result.metadata['termination'] == 'time_limit'
    assert result.nfe_by_branch['Elite'] <= int(.2 * result.nfe)
