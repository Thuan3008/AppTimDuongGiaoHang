from dataclasses import replace
from pathlib import Path

import pytest

from algorithms.hcmc_td_erapo import (
    HCMCTDERAPO,
    HCMCTDERAPOConfig,
    load_hcmc_dataset,
    optimize,
    optimize_hcmc_dataset,
)
from core.data.evaluator import HCMCTDVRPEvaluator
from core.data.solution import validate_solution


ROOT = Path(__file__).parents[1]
HCMC_N10 = ROOT / "data/hcmc/instances/hcmc-n10-heavy.vrp.json"
HCMC_N100 = ROOT / "data/hcmc/instances/hcmc-n100-heavy.vrp.json"


def test_hcmc_td_erapo_runs_on_n10_and_matches_evaluator():
    instance, graph, traffic = load_hcmc_dataset(HCMC_N10)
    for exact_split, dep in [(False, 0.0), (True, 28800.0)]:
        config = HCMCTDERAPOConfig(
            population_size=6,
            max_function_evaluations=40,
            earliest_departure=dep,
            exact_split=exact_split,
            seed=42,
        )
        result = optimize(instance, graph, traffic, config=config, seed=42)
        assert result.best_evaluation.feasible
        assert validate_solution(result.best_solution, instance).feasible
        assert result.evaluations == 40
        assert sum(result.nfe_by_branch.values()) == 40
        assert result.nfe_by_branch["Elite"] <= int(0.20 * 40)
        assert result.convergence[-1] <= result.convergence[0]
        assert result.convergence[-1] == pytest.approx(result.best_evaluation.fitness, rel=1e-9)
        assert len(result.best_permutation) == instance.num_customers

        # Đối chiếu trực tiếp từng tuyến tại đúng giờ xuất phát tối ưu bằng HCMCTDVRPEvaluator
        evaluator = HCMCTDVRPEvaluator(instance, graph, traffic, start_time=dep)
        eval_total = 0.0
        for idx, route in enumerate(result.best_solution.routes):
            r_eval = result.best_evaluation.routes[idx]
            assert r_eval.start_time >= dep - 1e-9
            checked = evaluator._evaluate_route(idx, route, r_eval.start_time)
            assert checked.feasible, checked.errors
            eval_total += checked.travel_time + checked.service_time
        assert result.best_evaluation.fitness == pytest.approx(eval_total, rel=0.02)


def test_hcmc_td_erapo_runs_on_n100_respecting_fleet_limit():
    instance, graph, traffic = load_hcmc_dataset(HCMC_N100)
    for exact_split in (False, True):
        budget = 40 if not exact_split else 20
        config = HCMCTDERAPOConfig(
            population_size=6,
            max_function_evaluations=budget,
            earliest_departure=28800.0,
            exact_split=exact_split,
            seed=7,
        )
        result = optimize(instance, graph, traffic, config=config, seed=7)
        assert result.best_evaluation.feasible
        assert validate_solution(result.best_solution, instance).feasible
        assert len(result.best_solution.routes) == instance.num_vehicles
        assert result.metadata["route_count"] <= instance.num_vehicles
        assert result.evaluations == budget
        assert sum(result.nfe_by_branch.values()) == budget
        assert result.convergence[-1] <= result.convergence[0]


def test_hcmc_td_erapo_reproducibility_with_common_population():
    instance, graph, traffic = load_hcmc_dataset(HCMC_N10)
    p0 = [
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        (10, 9, 8, 7, 6, 5, 4, 3, 2, 1),
        (2, 4, 6, 8, 10, 1, 3, 5, 7, 9),
        (9, 7, 5, 3, 1, 10, 8, 6, 4, 2),
    ]
    cfg = HCMCTDERAPOConfig(population_size=4, max_function_evaluations=24, exact_split=True)
    res1 = optimize(instance, graph, traffic, config=cfg, seed=99, initial_population=p0)
    res2 = optimize(instance, graph, traffic, config=cfg, seed=99, initial_population=p0)
    assert res1.best_evaluation.fitness == res2.best_evaluation.fitness
    assert res1.best_permutation == res2.best_permutation
    assert res1.convergence == res2.convergence
    assert res1.nfe_by_branch == res2.nfe_by_branch
    assert res1.metadata["initialization_policy"] == "provided_common_population"


def test_hcmc_td_erapo_supports_time_windows_pinned_policy_and_adapters():
    instance, graph, traffic = load_hcmc_dataset(HCMC_N10)
    # Ràng buộc thời gian phục vụ rộng hơn tổng thời lượng để kiểm tra apply_ready_time
    tw_instance = replace(
        instance,
        time_windows=[(0.0, 86400.0)] + [(32400.0, 72000.0) for _ in range(instance.num_customers)],
    )
    solver = HCMCTDERAPO()
    cfg = HCMCTDERAPOConfig(
        population_size=4,
        max_function_evaluations=16,
        exact_split=True,
        routing_policy="pinned_free_flow_paths",
    )
    res = solver.optimize(tw_instance, graph, traffic, config=cfg, seed=11)
    assert res.best_evaluation.feasible
    assert res.metadata["routing_policy"] == "pinned_free_flow_paths"
    assert res.best_evaluation.routes[0].stops[1].service_start_time >= 32400.0 - 1e-6

    helper_res = optimize_hcmc_dataset(HCMC_N10, HCMCTDERAPOConfig(population_size=4, max_function_evaluations=12), seed=5)
    assert helper_res.best_evaluation.feasible
    assert helper_res.nfe == 12
