import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from core.data.evaluator import HCMCTDVRPEvaluator
from core.data.load_poryos import find_poryos_bks, load_poryos, load_poryos_bks
from core.data.models import Customer, RoadEdge, RoadNode, TDVRPInstance, TrafficEdgeProfile, TrafficOverlay
from core.models import Solution
from experiments.runner import PRIMARY_ALGORITHM, ExperimentConfig, ExperimentRunner, summarize_runs


ROOT = Path(__file__).parents[1]
HK50 = ROOT / "data/TDVRP/hong_kong/n=50/poryos-hong_kong-n50-hyb/bpr-heavy/poryos-hong_kong-n50-hyb-bpr-heavy.vrp.json"
HK10 = ROOT / "data/TDVRP/hong_kong/n=10/poryos-hong_kong-n10-poi/bpr-light/poryos-hong_kong-n10-poi-bpr-light.vrp.json"


def test_official_poryos_loader_reads_and_validates_real_sidecars():
    instance, graph, traffic = load_poryos(HK50, materialize_atfs=False)
    assert instance.num_customers == 50
    assert instance.num_vehicles == 4
    assert len(graph[0]) == 10378
    assert len(graph[1]) == len(traffic.edges) == 11549
    assert instance.metadata["proxy_graph"] is False
    assert instance.metadata["sidecar_checksums_verified"] is True
    assert instance.metadata["routing_policy"] == "pinned_free_flow_paths"
    assert instance.road_node_ids[0] in graph[0]


def test_poryos_loader_preserves_time_windows_and_rejects_bad_checksum(tmp_path):
    payload = json.loads(HK50.read_text(encoding="utf-8"))
    payload["time_windows"] = [[0, 86400] for _ in range(payload["num_customers"] + 1)]
    with_windows = tmp_path / HK50.name
    with_windows.write_text(json.dumps(payload), encoding="utf-8")
    instance, _, _ = load_poryos(
        with_windows,
        collection_root=ROOT / "data/TDVRP",
        materialize_atfs=False,
    )
    assert instance.time_windows == [(0.0, 86400.0)] * 51

    payload["td"]["graph"]["sha256"] = "0" * 64
    with_windows.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_poryos(with_windows, collection_root=ROOT / "data/TDVRP", materialize_atfs=False)


def test_reference_atf_reproduces_published_bks_exactly():
    instance, graph, traffic = load_poryos(HK50)
    bks = load_poryos_bks(find_poryos_bks(HK50))
    solution = Solution([[0, *route, 0] for route in bks["routes"]])
    result = HCMCTDVRPEvaluator(instance, graph, traffic, start_time=0).evaluate(solution)
    assert result.feasible, result.errors
    assert result.fitness == bks["cost"]
    assert instance.metadata["atf_sha256_verified"] is True


def test_departure_optimization_and_detailed_route_trace():
    nodes = {0: RoadNode(0, 0, 0), 1: RoadNode(1, 0, 1)}
    edges = {
        "out": RoadEdge("out", 0, 1, "out", 1000, 60, False),
        "back": RoadEdge("back", 1, 0, "back", 1000, 60, False),
    }
    profiles = {
        edge_id: TrafficEdgeProfile(edge_id, edge_id, {"00": 10.0, "01": 60.0})
        for edge_id in edges
    }
    traffic = TrafficOverlay("test", profiles, bin_edges=(0.0, 3600.0, 7200.0))
    instance = TDVRPInstance(
        instance_name="departure-test",
        benchmark_name="test",
        num_customers=1,
        num_vehicles=1,
        vehicle_capacity=10,
        coordinates=[(0, 0), (0, 1)],
        demands=[0, 1],
        service_times=[0, 30],
        road_node_ids=[0, 1],
        depot=0,
        reference_lla={},
        horizon=(0.0, 7200.0),
        sample_step=3600.0,
        graph_path="test",
        traffic_path="test",
        metadata={"routing_policy": "dynamic_fastest_path"},
        customers=[Customer(1, 1, 1, 30, (0, 1))],
    )
    solution = Solution([[0, 1, 0]])
    fixed = HCMCTDVRPEvaluator(instance, (nodes, edges), traffic, start_time=0).evaluate(solution)
    optimized = HCMCTDVRPEvaluator(
        instance,
        (nodes, edges),
        traffic,
        start_time=0,
        departure_time_mode="optimize",
        departure_time_step=3600,
    ).evaluate(solution)
    route = optimized.routes[0]
    assert optimized.fitness < fixed.fitness
    assert route.start_time == 3600
    assert len(route.legs) == 2
    assert [stop.customer_id for stop in route.stops] == [0, 1, 0]
    assert route.stops[1].service_time == 30
    morning_evaluator = HCMCTDVRPEvaluator(
        replace(instance, horizon=(0.0, 86400.0)),
        (nodes, edges),
        traffic,
        departure_time_mode="optimize",
        departure_time_step=900,
        departure_time_window=(7 * 3600, 9 * 3600),
    )
    assert morning_evaluator._departure_candidates() == [
        7 * 3600 + index * 900 for index in range(9)
    ]


def test_statistics_and_proxy_experiment_runner_keep_protocol_fair():
    rows = [
        {"algorithm": "A", "status": "completed", "fitness": 10.0, "runtime_seconds": 2.0, "feasible": True},
        {"algorithm": "A", "status": "completed", "fitness": 14.0, "runtime_seconds": 4.0, "feasible": True},
    ]
    summary = summarize_runs(rows, bks_cost=8.0)[0]
    assert summary["best"] == 10.0
    assert summary["mean"] == summary["median"] == 12.0
    assert math.isclose(summary["standard_deviation"], math.sqrt(8.0))
    assert summary["best_gap_to_bks_percent"] == 25.0

    report = ExperimentRunner(
        ExperimentConfig(
            instance_path=HK10,
            seeds=(1, 2),
            population_size=2,
            nfe_budget=4,
            algorithms=("TD-CR-RAPO-V3",),
            departure_time_mode="fixed",
            allow_proxy=True,
        )
    ).run()
    assert len(report.runs) == 2
    assert all(row["nfe"] == 4 for row in report.runs)
    assert report.protocol["proxy_graph"] is True
    assert report.protocol["bks_cost"] is None


def test_rapo_2709_is_primary_and_mixed_evaluator_comparison_is_blocked():
    config = ExperimentConfig(instance_path=HK50)
    assert config.algorithms == (PRIMARY_ALGORITHM,) == ("RAPO-2709",)

    # So sánh paired phải đi qua runner chuyên dụng để dùng chung P0 và exact evaluator.
    with pytest.raises(ValueError, match="paired"):
        ExperimentConfig(
            instance_path=HK50,
            algorithms=("RAPO-2709", "TD-CR-RAPO-V3"),
        ).validate()
