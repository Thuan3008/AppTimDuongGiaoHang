from pathlib import Path

from experiments.runner import (
    HCMC_ABLATION_ALGORITHMS,
    ExperimentConfig,
    ExperimentRunner,
)


ROOT = Path(__file__).parents[1]
INSTANCE = ROOT / "data/hcmc/instances/hcmc-n10-heavy.vrp.json"


def test_hcmc_aro_poa_and_rapo_share_one_experiment_protocol():
    report = ExperimentRunner(
        ExperimentConfig(
            instance_path=INSTANCE,
            seeds=(42,),
            population_size=4,
            nfe_budget=8,
            algorithms=("ARO-goc", "POA-goc", "RAPO"),
            start_time=8 * 3600,
            departure_time_mode="fixed",
            resume=False,
        )
    ).run()
    assert report.protocol["objective"] == "total_route_duration = total_travel_time + total_service_time"
    assert {row["algorithm"] for row in report.runs} == {"ARO-goc", "POA-goc", "RAPO"}
    assert all(row["status"] == "completed" and row["nfe"] == 8 for row in report.runs)
    assert all(row["departure_time_mode"] == "fixed" for row in report.runs)
    for trace in report.best_route_traces.values():
        assert all("edge_path" in route for route in trace["routes"])


def test_hcmc_rapo_ablation_variants_are_runnable():
    report = ExperimentRunner(
        ExperimentConfig(
            instance_path=INSTANCE,
            seeds=(7,),
            population_size=4,
            nfe_budget=4,
            algorithms=HCMC_ABLATION_ALGORITHMS,
            start_time=8 * 3600,
            departure_time_mode="fixed",
            resume=False,
        )
    ).run()
    assert {row["algorithm"] for row in report.runs} == set(HCMC_ABLATION_ALGORITHMS)
    assert all(row["status"] == "completed" and row["nfe"] == 4 for row in report.runs)
