from pathlib import Path
import json

from core.tdvrp_common import TDVRPProblem
from experiments.rapo_contribution.analyze import paired_comparisons
from experiments.rapo_contribution.exact_greedy import run_exact_greedy
from experiments.rapo_contribution.merge import merge_dataset
from experiments.rapo_contribution.run import (
    bootstrap_shard_checkpoint,
    parse_seed_spec,
    seed_shard_name,
)


ROOT = Path(__file__).parents[1]
INSTANCE = (
    ROOT
    / "data/TDVRP/hong_kong/n=25/poryos-hong_kong-n25-hyb/bpr-heavy"
    / "poryos-hong_kong-n25-hyb-bpr-heavy.vrp.json"
)


def test_exact_greedy_is_feasible_and_deterministic():
    problem = TDVRPProblem(
        INSTANCE,
        collection_root=ROOT / "data/TDVRP",
        vehicle_limit=25,
    )
    first = run_exact_greedy(problem)
    second = run_exact_greedy(problem)
    assert first.feasible and second.feasible
    assert first.fitness == second.fitness
    assert first.routes == second.routes


def test_paired_comparison_uses_matching_seeds_and_positive_delta_for_rapo_win():
    runs = []
    for seed, rapo, baseline in ((0, 10.0, 12.0), (1, 11.0, 13.0), (2, 14.0, 13.0)):
        runs.extend(
            [
                {"algorithm": "RAPO-2709", "seed": seed, "fitness": rapo, "status": "completed"},
                {"algorithm": "GA", "seed": seed, "fitness": baseline, "status": "completed"},
            ]
        )
    rows = paired_comparisons(runs, {"fitness": 15.0})
    ga = next(row for row in rows if row["baseline"] == "GA")
    assert ga["paired_seeds"] == 3
    assert ga["primary_wins"] == 2
    assert ga["primary_losses"] == 1
    assert ga["mean_baseline_minus_primary_seconds"] == 1.0


def test_seed_shard_parser_accepts_ranges_and_rejects_outside_protocol():
    allowed = list(range(30))
    assert parse_seed_spec("0-9", allowed) == list(range(10))
    assert parse_seed_spec("10-19", allowed) == list(range(10, 20))
    assert seed_shard_name(list(range(20, 30))) == "seeds_20_29"

    import pytest

    with pytest.raises(ValueError):
        parse_seed_spec("29-30", allowed)


def test_merge_combines_disjoint_seed_shards(tmp_path):
    algorithms = ["ARO-Original", "POA-Original", "RAPO-2709", "GA", "GA-LS"]
    config = {
        "datasets": {"n25": str(INSTANCE.relative_to(ROOT)).replace("\\", "/")},
        "stochastic_algorithms": algorithms,
        "seeds": list(range(30)),
        "population_size": 10,
        "nfe_budget": 290,
    }
    stable_protocol = {
        "instance_name": "synthetic-n25",
        "benchmark_name": "Poryos2026",
        "data_source": "poryos-official-sidecars",
        "proxy_graph": False,
        "traffic_model": "bpr",
        "traffic_profile": "heavy",
        "objective": "total_route_duration = total_travel_time + total_service_time",
        "population_size": 10,
        "nfe_budget": 290,
        "algorithms": algorithms,
        "initialization_policy": "paired_common_population_per_seed",
        "stopping_condition": "strict NFE budget",
        "start_time": 0.0,
        "departure_time_mode": "exact_ndcpwlf",
        "departure_time_step": None,
        "bks_cost": None,
        "atf_sha256_verified": True,
    }
    input_dir = tmp_path / "shards"
    for start in (0, 10, 20):
        shard_dir = input_dir / f"seeds_{start:02d}_{start + 9:02d}"
        shard_dir.mkdir(parents=True)
        seeds = list(range(start, start + 10))
        runs = [
            {
                "algorithm": algorithm,
                "seed": seed,
                "status": "completed",
                "fitness": float(1000 + seed + index),
                "feasible": True,
                "runtime_seconds": 1.0,
                "nfe": 290,
            }
            for seed in seeds
            for index, algorithm in enumerate(algorithms)
        ]
        (shard_dir / "stochastic_report.json").write_text(
            json.dumps(
                {
                    "protocol": {**stable_protocol, "seeds": seeds},
                    "runs": runs,
                    "summary": [],
                    "best_route_traces": {},
                }
            ),
            encoding="utf-8",
        )
        (shard_dir / "greedy.json").write_text(
            json.dumps({"fitness": 2000.0, "runtime_seconds": 1.0, "feasible": True}),
            encoding="utf-8",
        )

    output_dir = tmp_path / "merged"
    manifest = merge_dataset("n25", config, input_dir, output_dir, plots=False)
    merged = json.loads((output_dir / "stochastic_report.json").read_text(encoding="utf-8"))
    assert manifest["run_count"] == 150
    assert len(merged["runs"]) == 150
    assert merged["protocol"]["seeds"] == list(range(30))


def test_bootstrap_shard_checkpoint_filters_legacy_runs(tmp_path):
    source = tmp_path / "legacy.jsonl"
    destination = tmp_path / "shard" / "checkpoint.jsonl"
    records = [
        {"type": "header", "checkpoint_version": 1, "protocol": {"seeds": list(range(30))}},
        {"type": "run", "row": {"algorithm": "GA", "seed": 0, "status": "completed"}},
        {"type": "run", "row": {"algorithm": "GA", "seed": 10, "status": "completed"}},
    ]
    source.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    records[0]["protocol"]["nfe_budget"] = 290
    source.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    imported = bootstrap_shard_checkpoint(source, destination, list(range(10)), ["GA"], 290)
    lines = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert imported == 1
    assert lines[0]["protocol"]["seeds"] == list(range(10))
    assert [record["row"]["seed"] for record in lines[1:]] == [0]

