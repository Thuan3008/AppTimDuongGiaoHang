import random
from pathlib import Path

from algorithms.operators.discrete import OPERATORS
from algorithms.operators.hcmc_pipeline import repair_validate_evaluate
from core.data.evaluator import HCMCTDVRPEvaluator
from core.data.load_instance import load_instance
from core.data.load_road_graph import load_road_graph
from core.data.load_traffic import load_traffic
from core.data.solution import repair_solution, validate_solution
from core.models import Solution


ROOT = Path(__file__).parents[1]


def test_all_discrete_operators_preserve_customers_after_pipeline():
    instance = load_instance(ROOT / "data/hcmc/instances/hcmc-n100-heavy.vrp.json")
    graph = load_road_graph(instance.graph_path)
    traffic = load_traffic(instance.traffic_path, graph[1])
    evaluator = HCMCTDVRPEvaluator(instance, graph, traffic)
    base = repair_solution(Solution([[0, 1, 2, 3, 4, 5], [0, 6, 7, 8, 9, 10]]), instance)
    for operator in OPERATORS:
        candidate = operator(base, random.Random(42))
        evaluated = repair_validate_evaluate(candidate, instance, evaluator)
        assert evaluated is not None, operator.__name__
        repaired, result = evaluated
        assert result.feasible
        assert validate_solution(repaired, instance).feasible
        served = [customer_id for route in repaired.routes for customer_id in route if customer_id != 0]
        assert sorted(served) == list(range(1, instance.num_customers + 1))
