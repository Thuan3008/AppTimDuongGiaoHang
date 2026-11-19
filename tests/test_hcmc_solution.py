from pathlib import Path

from core.data.load_instance import load_instance
from core.data.solution import empty_solution, repair_solution, validate_solution
from core.models import Solution


ROOT = Path(__file__).parents[1]


def hcmc_instance():
    return load_instance(ROOT / "data/hcmc/instances/hcmc-n100-heavy.vrp.json")


def test_empty_hcmc_solution_has_one_route_per_vehicle():
    instance = hcmc_instance()
    solution = empty_solution(instance)
    assert len(solution.routes) == instance.num_vehicles
    assert all(route == [0, 0] for route in solution.routes)
    assert not validate_solution(solution, instance).feasible


def test_valid_hcmc_solution_has_exactly_once_customers():
    instance = hcmc_instance()
    solution = repair_solution(Solution([[0, 1, 2, 3, 1, 0]]), instance)
    result = validate_solution(solution, instance)
    assert result.feasible, result.errors
    served = [customer_id for route in solution.routes for customer_id in route if customer_id != 0]
    assert sorted(served) == list(range(1, instance.num_customers + 1))
    assert len(served) == len(set(served))


def test_invalid_solution_reports_route_contract_errors():
    instance = hcmc_instance()
    result = validate_solution(Solution([[1, 1, 999], [0, 0]]), instance)
    assert not result.feasible
    assert any("depot" in error for error in result.errors)
    assert any("unknown customer" in error for error in result.errors)
    assert any("Duplicate" in error for error in result.errors)
    assert any("Missing" in error for error in result.errors)


def test_repair_is_deterministic_and_capacity_feasible():
    instance = hcmc_instance()
    candidate = Solution([[0, 7, 7, 4, 0], [0, 2, 999, 1]])
    repaired = repair_solution(candidate, instance)
    again = repair_solution(candidate, instance)
    assert repaired.routes == again.routes
    result = validate_solution(repaired, instance)
    assert result.feasible, result.errors
