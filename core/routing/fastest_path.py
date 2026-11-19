from __future__ import annotations

import heapq
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from threading import Lock

from core.data.models import RoadEdge, RoadNode, TrafficOverlay
from .travel_time import get_edge_travel_time


_ADJACENCY_CACHE_MAXSIZE = 4
_ADJACENCY_CACHE: OrderedDict[
    int,
    tuple[dict[str, RoadEdge], dict[int, list[tuple[int, RoadEdge]]]],
] = OrderedDict()
_ADJACENCY_LOCK = Lock()


@dataclass(frozen=True)
class FastestPathResult:
    node_path: list[int]
    edge_path: list[str]
    travel_time_seconds: float


def find_fastest_path(graph: tuple[dict[int, RoadNode], dict[str, RoadEdge]] | dict[str, object], traffic: TrafficOverlay, source_node: int, target_node: int, departure_time: float | datetime, sample_step: float = 300.0, cache: dict | None = None) -> FastestPathResult:
    nodes, edges = (graph if isinstance(graph, tuple) else (graph["nodes"], graph["edges"]))
    if source_node not in nodes or target_node not in nodes:
        raise ValueError("source_node and target_node must exist in graph")
    departure_seconds = _seconds_since_midnight(departure_time)
    bucket = int(departure_seconds // sample_step)
    cache_key = (source_node, target_node, bucket)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    adjacency = prepare_adjacency((nodes, edges))
    best: dict[int, float] = {source_node: departure_seconds}
    previous: dict[int, tuple[int, str]] = {}
    queue = [(departure_seconds, source_node)]
    while queue:
        arrival_time, node = heapq.heappop(queue)
        if arrival_time != best.get(node):
            continue
        if node == target_node:
            break
        for neighbour, edge in adjacency.get(node, ()):
            candidate = arrival_time + get_edge_travel_time(edge, arrival_time, traffic)
            if candidate < best.get(neighbour, float("inf")):
                best[neighbour] = candidate
                previous[neighbour] = (node, edge.id)
                heapq.heappush(queue, (candidate, neighbour))
    if target_node not in best:
        raise ValueError(f"No route from {source_node} to {target_node}")
    node_path = [target_node]
    edge_path: list[str] = []
    while node_path[-1] != source_node:
        parent, edge_id = previous[node_path[-1]]
        node_path.append(parent); edge_path.append(edge_id)
    node_path.reverse(); edge_path.reverse()
    result = FastestPathResult(node_path, edge_path, best[target_node] - departure_seconds)
    if cache is not None:
        cache[cache_key] = result
    return result


def find_fastest_paths_from_source(
    graph: tuple[dict[int, RoadNode], dict[str, RoadEdge]],
    traffic: TrafficOverlay,
    source_node: int,
    departure_time: float | datetime,
    target_nodes: set[int] | None = None,
) -> dict[int, FastestPathResult]:
    """Compute earliest-arrival paths from one source to every reachable node.

    With ``target_nodes``, stop as soon as all requested destinations are
    settled and reconstruct only those paths.
    """
    nodes, edges = graph
    if source_node not in nodes:
        raise ValueError("source_node must exist in graph")
    requested_targets = set(target_nodes) if target_nodes is not None else None
    if requested_targets is not None:
        unknown = requested_targets.difference(nodes)
        if unknown:
            raise ValueError(f"target_nodes must exist in graph: {sorted(unknown)}")
    departure_seconds = _seconds_since_midnight(departure_time)
    adjacency = prepare_adjacency(graph)
    best: dict[int, float] = {source_node: departure_seconds}
    previous: dict[int, tuple[int, str]] = {}
    queue = [(departure_seconds, source_node)]
    remaining_targets = set(requested_targets or ())
    while queue:
        arrival_time, node = heapq.heappop(queue)
        if arrival_time != best.get(node):
            continue
        remaining_targets.discard(node)
        if requested_targets is not None and not remaining_targets:
            break
        for neighbour, edge in adjacency.get(node, ()):
            candidate = arrival_time + get_edge_travel_time(edge, arrival_time, traffic)
            if candidate < best.get(neighbour, float("inf")):
                best[neighbour] = candidate
                previous[neighbour] = (node, edge.id)
                heapq.heappush(queue, (candidate, neighbour))
    results: dict[int, FastestPathResult] = {}
    selected_targets = sorted(requested_targets) if requested_targets is not None else best.keys()
    for target_node in selected_targets:
        if target_node not in best:
            continue
        arrival_time = best[target_node]
        node_path, edge_path = _reconstruct_path(source_node, target_node, previous)
        results[target_node] = FastestPathResult(node_path, edge_path, arrival_time - departure_seconds)
    return results


def find_pinned_fastest_path(
    graph: tuple[dict[int, RoadNode], dict[str, RoadEdge]],
    traffic: TrafficOverlay,
    source_node: int,
    target_node: int,
    departure_time: float | datetime,
    path_cache: dict[tuple[int, int], tuple[list[int], list[str]]] | None = None,
) -> FastestPathResult:
    """Đi theo đường nhanh nhất ở free-flow đã được ghim theo chuẩn Poryos.

    Traffic chỉ làm thay đổi thời gian đi trên các cạnh của đường đã ghim;
    nó không được phép chọn lại tuyến đường ở từng thời điểm.
    """
    nodes, edges = graph
    if source_node not in nodes or target_node not in nodes:
        raise ValueError("source_node and target_node must exist in graph")
    key = (source_node, target_node)
    cached = path_cache.get(key) if path_cache is not None else None
    if cached is None:
        node_path, edge_path = _free_flow_path(nodes, edges, source_node, target_node)
        if path_cache is not None:
            path_cache[key] = (node_path, edge_path)
    else:
        node_path, edge_path = cached
    clock = _seconds_since_midnight(departure_time)
    started = clock
    for edge_id in edge_path:
        clock += get_edge_travel_time(edges[edge_id], clock, traffic)
    return FastestPathResult(list(node_path), list(edge_path), clock - started)


def find_pinned_fastest_paths_from_source(
    graph: tuple[dict[int, RoadNode], dict[str, RoadEdge]],
    traffic: TrafficOverlay,
    source_node: int,
    departure_time: float | datetime,
    path_cache: dict[tuple[int, int], tuple[list[int], list[str]]] | None = None,
    target_nodes: set[int] | None = None,
) -> dict[int, FastestPathResult]:
    """Tạo đường free-flow cố định từ một nguồn rồi áp traffic cho từng đích."""
    nodes, edges = graph
    adjacency = _adjacency(nodes, edges)
    best: dict[int, float] = {source_node: 0.0}
    previous: dict[int, tuple[int, str]] = {}
    queue = [(0.0, source_node)]
    remaining_targets = set(target_nodes or ())
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != best.get(node):
            continue
        remaining_targets.discard(node)
        if target_nodes is not None and not remaining_targets:
            break
        for neighbour, edge in adjacency.get(node, ()):
            candidate = cost + edge.distance_m / max(edge.free_flow_speed_kmh / 3.6, 1e-12)
            if candidate < best.get(neighbour, float("inf")):
                best[neighbour] = candidate
                previous[neighbour] = (node, edge.id)
                heapq.heappush(queue, (candidate, neighbour))
    results: dict[int, FastestPathResult] = {}
    selected_targets = sorted(target_nodes) if target_nodes is not None else best.keys()
    for target_node in selected_targets:
        if target_node not in best:
            continue
        node_path, edge_path = _reconstruct_path(source_node, target_node, previous)
        if path_cache is not None:
            path_cache[(source_node, target_node)] = (node_path, edge_path)
        clock = _seconds_since_midnight(departure_time)
        started = clock
        for edge_id in edge_path:
            clock += get_edge_travel_time(edges[edge_id], clock, traffic)
        results[target_node] = FastestPathResult(node_path, edge_path, clock - started)
    return results


def _free_flow_path(
    nodes: dict[int, RoadNode],
    edges: dict[str, RoadEdge],
    source_node: int,
    target_node: int,
) -> tuple[list[int], list[str]]:
    """Dijkstra tĩnh với trọng số length/speed và tie-break theo node id."""
    adjacency = _adjacency(nodes, edges)
    best: dict[int, float] = {source_node: 0.0}
    previous: dict[int, tuple[int, str]] = {}
    queue = [(0.0, source_node)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != best.get(node):
            continue
        if node == target_node:
            break
        for neighbour, edge in adjacency.get(node, ()):
            candidate = cost + edge.distance_m / max(edge.free_flow_speed_kmh / 3.6, 1e-12)
            if candidate < best.get(neighbour, float("inf")):
                best[neighbour] = candidate
                previous[neighbour] = (node, edge.id)
                heapq.heappush(queue, (candidate, neighbour))
    if target_node not in best:
        raise ValueError(f"No route from {source_node} to {target_node}")
    return _reconstruct_path(source_node, target_node, previous)


def _adjacency(
    nodes: dict[int, RoadNode], edges: dict[str, RoadEdge]
) -> dict[int, list[tuple[int, RoadEdge]]]:
    """Backward-compatible wrapper around the shared adjacency cache."""
    return prepare_adjacency((nodes, edges))


def _build_adjacency(edges: dict[str, RoadEdge]) -> dict[int, list[tuple[int, RoadEdge]]]:
    adjacency: dict[int, list[tuple[int, RoadEdge]]] = {}
    for edge in edges.values():
        adjacency.setdefault(edge.from_node, []).append((edge.to_node, edge))
        if edge.bidirectional:
            adjacency.setdefault(edge.to_node, []).append((edge.from_node, edge))
    return adjacency


def prepare_adjacency(
    graph: tuple[dict[int, RoadNode], dict[str, RoadEdge]] | dict[str, object],
) -> dict[int, list[tuple[int, RoadEdge]]]:
    """Build one adjacency list per edge dictionary and reuse it across calls."""
    _, edges = (graph if isinstance(graph, tuple) else (graph["nodes"], graph["edges"]))
    key = id(edges)
    with _ADJACENCY_LOCK:
        cached = _ADJACENCY_CACHE.get(key)
        if cached is not None and cached[0] is edges:
            _ADJACENCY_CACHE.move_to_end(key)
            return cached[1]
        adjacency = _build_adjacency(edges)
        _ADJACENCY_CACHE[key] = (edges, adjacency)
        _ADJACENCY_CACHE.move_to_end(key)
        while len(_ADJACENCY_CACHE) > _ADJACENCY_CACHE_MAXSIZE:
            _ADJACENCY_CACHE.popitem(last=False)
        return adjacency


def _reconstruct_path(
    source_node: int,
    target_node: int,
    previous: dict[int, tuple[int, str]],
) -> tuple[list[int], list[str]]:
    node_path = [target_node]
    edge_path: list[str] = []
    while node_path[-1] != source_node:
        parent, edge_id = previous[node_path[-1]]
        node_path.append(parent)
        edge_path.append(edge_id)
    node_path.reverse()
    edge_path.reverse()
    return node_path, edge_path


def _seconds_since_midnight(value: float | datetime) -> float:
    if isinstance(value, datetime):
        return float(value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1_000_000)
    return float(value) % 86400
