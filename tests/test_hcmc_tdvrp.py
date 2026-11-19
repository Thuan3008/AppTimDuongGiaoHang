import json
from pathlib import Path

from core.data.load_instance import load_instance
from core.data.load_road_graph import load_road_graph
from core.data.load_traffic import load_traffic
from core.data.models import RoadEdge, RoadNode, TrafficEdgeProfile, TrafficOverlay
from core.routing import fastest_path as routing
from core.routing.fastest_path import find_fastest_path, find_fastest_paths_from_source
from core.routing.travel_time import get_edge_travel_time


ROOT = Path(__file__).parents[1]


def test_hcmc_sidecars_and_instance_load():
    instance = load_instance(ROOT / "data/hcmc/instances/hcmc-n100-heavy.vrp.json")
    nodes, edges = load_road_graph(ROOT / "data/hcmc/sidecars/hcmc.road.json")
    traffic = load_traffic(ROOT / "data/hcmc/sidecars/hcmc.traffic-heavy.json", edges)
    assert instance.num_customers == 100
    assert len(nodes) > 100 and len(edges) > len(nodes)
    assert len(traffic.edges) == len(edges)


def test_time_dependent_fastest_path_prefers_longer_light_route():
    nodes = {1: RoadNode(1, 0, 0), 2: RoadNode(2, 0, 1), 3: RoadNode(3, 1, 0)}
    edges = {
        "A": RoadEdge("A", 1, 2, "Cong Hoa", 5000, 30, False),
        "B": RoadEdge("B", 1, 3, "Local", 4000, 20, False),
        "C": RoadEdge("C", 3, 2, "Local", 4000, 20, False),
    }
    hours = {f"{hour:02d}": 30.0 for hour in range(24)}
    hours["08"] = 15.0
    traffic = TrafficOverlay("fixed_hourly_speed", {
        "A": TrafficEdgeProfile("A", "Cong Hoa", hours),
        "B": TrafficEdgeProfile("B", "Local", {f"{h:02d}": 40.0 for h in range(24)}),
        "C": TrafficEdgeProfile("C", "Local", {f"{h:02d}": 40.0 for h in range(24)}),
    })
    result = find_fastest_path((nodes, edges), traffic, 1, 2, 8 * 3600)
    assert result.edge_path == ["B", "C"]
    assert result.travel_time_seconds < get_edge_travel_time(edges["A"], 8 * 3600, traffic)


def test_traffic_lookup_changes_after_edge_crosses_hour():
    edge = RoadEdge("E", 1, 2, "Test", 1000, 60, False)
    traffic = TrafficOverlay("fixed_hourly_speed", {"E": TrafficEdgeProfile("E", "Test", {"00": 60.0, **{f"{h:02d}": 30.0 for h in range(1, 24)}})})
    assert get_edge_travel_time(edge, 0, traffic) < get_edge_travel_time(edge, 3600, traffic)


def _small_routing_case():
    nodes = {node_id: RoadNode(node_id, 0, node_id) for node_id in range(1, 5)}
    edges = {
        "A": RoadEdge("A", 1, 2, "A", 100, 30, False),
        "B": RoadEdge("B", 2, 3, "B", 100, 30, False),
        "C": RoadEdge("C", 3, 4, "C", 100, 30, False),
    }
    speeds = {f"{hour:02d}": 30.0 for hour in range(24)}
    traffic = TrafficOverlay("fixed_hourly_speed", {
        edge_id: TrafficEdgeProfile(edge_id, edge.road_name, speeds)
        for edge_id, edge in edges.items()
    })
    return (nodes, edges), traffic


def test_adjacency_is_built_once_per_graph(monkeypatch):
    graph, traffic = _small_routing_case()
    original = routing._build_adjacency
    calls = 0

    def counted(edges):
        nonlocal calls
        calls += 1
        return original(edges)

    monkeypatch.setattr(routing, "_build_adjacency", counted)
    find_fastest_path(graph, traffic, 1, 2, 0)
    find_fastest_path(graph, traffic, 1, 3, 0)
    assert calls == 1


def test_multi_target_routing_stops_and_reconstructs_only_requested_nodes(monkeypatch):
    graph, traffic = _small_routing_case()
    original = routing.get_edge_travel_time
    visited_edges = []

    def counted(edge, departure_time, overlay):
        visited_edges.append(edge.id)
        return original(edge, departure_time, overlay)

    monkeypatch.setattr(routing, "get_edge_travel_time", counted)
    results = find_fastest_paths_from_source(graph, traffic, 1, 0, target_nodes={2})
    assert set(results) == {2}
    assert results[2].edge_path == ["A"]
    assert visited_edges == ["A"]


def test_fastest_path_cache_keeps_source_target_and_time_bucket():
    graph, traffic = _small_routing_case()
    cache = {}
    first = find_fastest_path(graph, traffic, 1, 3, 0, sample_step=300, cache=cache)
    same_bucket = find_fastest_path(graph, traffic, 1, 3, 299, sample_step=300, cache=cache)
    next_bucket = find_fastest_path(graph, traffic, 1, 3, 300, sample_step=300, cache=cache)
    assert same_bucket is first
    assert next_bucket is not first
    assert set(cache) == {(1, 3, 0), (1, 3, 1)}


