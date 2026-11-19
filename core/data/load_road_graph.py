from __future__ import annotations

import gzip
import json
from collections import defaultdict, deque
from pathlib import Path

from .models import RoadEdge, RoadNode


def load_road_graph(path: str | Path) -> tuple[dict[int, RoadNode], dict[str, RoadEdge]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Road graph not found: {path}")
    payload = _read_json(path)
    if payload.get("format") == "mamut-road-graph":
        return _load_poryos_road_graph(payload)
    nodes = {int(row["id"]): RoadNode(int(row["id"]), float(row["lat"]), float(row["lon"])) for row in payload["nodes"]}
    edges: dict[str, RoadEdge] = {}
    for row in payload["edges"]:
        edge = RoadEdge(
            id=str(row["id"]), from_node=int(row["from"]), to_node=int(row["to"]),
            road_name=str(row["road_name"]), distance_m=float(row["distance_m"]),
            free_flow_speed_kmh=float(row["free_flow_speed_kmh"]),
            bidirectional=bool(row.get("bidirectional", True)), road_class=str(row.get("road_class", "local")),
            osm_way_ids=tuple(int(value) for value in row.get("osm_way_ids", [])),
        )
        if edge.id in edges or edge.from_node not in nodes or edge.to_node not in nodes:
            raise ValueError(f"Invalid or duplicate road edge: {edge.id}")
        if edge.distance_m <= 0 or edge.free_flow_speed_kmh <= 0:
            raise ValueError(f"Road edge {edge.id} must have positive distance and speed")
        edges[edge.id] = edge
    _validate_connected(nodes, edges)
    return nodes, edges


def _read_json(path: Path) -> dict:
    """Đọc JSON thường hoặc JSON nén gzip mà không làm thay đổi dữ liệu gốc."""
    raw = path.read_bytes()
    if path.name.endswith(".gz"):
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def _load_poryos_road_graph(payload: dict) -> tuple[dict[int, RoadNode], dict[str, RoadEdge]]:
    """Chuyển road sidecar Poryos v2 sang mô hình đồ thị nội bộ.

    Sidecar lưu tốc độ theo mét/giây và cạnh có hướng. Lớp nội bộ dùng km/giờ,
    vì vậy tốc độ được nhân 3.6 ngay tại biên đọc dữ liệu.
    """
    if int(payload.get("format_version", -1)) != 2:
        raise ValueError(f"Unsupported Poryos road format version: {payload.get('format_version')}")
    num_vertices = int(payload["num_vertices"])
    lonlat = payload["vertex_lonlat"]
    if len(lonlat) != num_vertices:
        raise ValueError("Poryos road graph has an invalid vertex_lonlat length")
    nodes = {
        index: RoadNode(index, float(pair[1]), float(pair[0]))
        for index, pair in enumerate(lonlat)
    }
    edges: dict[str, RoadEdge] = {}
    previous: tuple[int, int] | None = None
    for index, row in enumerate(payload["edges"]):
        if len(row) != 4:
            raise ValueError(f"Invalid Poryos road edge at index {index}")
        source, target, distance_m, speed_mps = row
        key = (int(source), int(target))
        if previous is not None and key <= previous:
            raise ValueError("Poryos road edges must be strictly sorted by (from, to)")
        previous = key
        edge_id = str(index)
        edge = RoadEdge(
            id=edge_id,
            from_node=key[0],
            to_node=key[1],
            road_name=f"OSM {key[0]}->{key[1]}",
            distance_m=float(distance_m),
            free_flow_speed_kmh=float(speed_mps) * 3.6,
            bidirectional=False,
            road_class="poryos-osm",
        )
        if edge.from_node not in nodes or edge.to_node not in nodes:
            raise ValueError(f"Poryos road edge {index} references an unknown vertex")
        if edge.distance_m <= 0 or edge.free_flow_speed_kmh <= 0:
            raise ValueError(f"Poryos road edge {index} must have positive distance and speed")
        edges[edge_id] = edge
    _validate_connected(nodes, edges)
    return nodes, edges


def _validate_connected(nodes: dict[int, RoadNode], edges: dict[str, RoadEdge]) -> None:
    if not nodes:
        raise ValueError("Road graph has no nodes")
    adjacency: dict[int, set[int]] = defaultdict(set)
    for edge in edges.values():
        adjacency[edge.from_node].add(edge.to_node)
        if edge.bidirectional:
            adjacency[edge.to_node].add(edge.from_node)
    seen = {next(iter(nodes))}
    queue = deque(seen)
    while queue:
        node = queue.popleft()
        for other in adjacency[node]:
            if other not in seen:
                seen.add(other)
                queue.append(other)
    if len(seen) != len(nodes):
        raise ValueError(f"Road graph is disconnected: reached {len(seen)}/{len(nodes)} nodes")
