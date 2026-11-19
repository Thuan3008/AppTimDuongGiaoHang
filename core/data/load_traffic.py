from __future__ import annotations

import gzip
import json
from pathlib import Path

from .models import RoadEdge, TrafficEdgeProfile, TrafficOverlay


def load_traffic(path: str | Path, road_edges: dict[str, RoadEdge] | None = None) -> TrafficOverlay:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Traffic overlay not found: {path}")
    raw = path.read_bytes()
    if path.name.endswith(".gz"):
        raw = gzip.decompress(raw)
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("format") == "mamut-traffic-overlay":
        return _load_poryos_traffic(payload, road_edges)
    profiles: dict[str, TrafficEdgeProfile] = {}
    for row in payload.get("edges", []):
        edge_id = str(row["edge_id"])
        speeds = {str(hour).zfill(2): float(speed) for hour, speed in row["hourly_speed_kmh"].items()}
        if set(speeds) != {f"{hour:02d}" for hour in range(24)} or any(speed <= 0 for speed in speeds.values()):
            raise ValueError(f"Traffic profile for {edge_id} must contain positive speeds for all 24 hours")
        profiles[edge_id] = TrafficEdgeProfile(edge_id, str(row.get("road_name", "")), speeds)
    if road_edges is not None:
        missing = set(road_edges) - set(profiles)
        unknown = set(profiles) - set(road_edges)
        if missing or unknown:
            raise ValueError(f"Traffic/road edge mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")
    return TrafficOverlay(
        str(payload["traffic_model"]), profiles, payload.get("seed"), payload.get("profile"), payload.get("version")
    )


def _load_poryos_traffic(payload: dict, road_edges: dict[str, RoadEdge] | None) -> TrafficOverlay:
    """Đọc traffic sidecar Poryos và giữ nguyên thứ tự cạnh đã được ghim."""
    if int(payload.get("format_version", -1)) != 1:
        raise ValueError(f"Unsupported Poryos traffic format version: {payload.get('format_version')}")
    bin_edges = tuple(float(value) for value in payload["bin_edges"])
    if len(bin_edges) < 2 or any(right <= left for left, right in zip(bin_edges, bin_edges[1:])):
        raise ValueError("Poryos traffic bin_edges must be strictly increasing")
    rows = payload["edge_speeds"]
    if road_edges is None:
        raise ValueError("Poryos traffic overlay requires its matching road graph")
    if len(rows) != len(road_edges):
        raise ValueError(f"Traffic/road edge mismatch; traffic={len(rows)}, road={len(road_edges)}")
    profiles: dict[str, TrafficEdgeProfile] = {}
    expected_bins = len(bin_edges) - 1
    for index, (edge_id, edge) in enumerate(road_edges.items()):
        speeds_mps = rows[index]
        if len(speeds_mps) != expected_bins:
            raise ValueError(f"Poryos traffic edge {index} has {len(speeds_mps)} bins, expected {expected_bins}")
        speeds_kmh = {f"{slot:02d}": float(speed) * 3.6 for slot, speed in enumerate(speeds_mps)}
        if any(speed <= 0 or speed > edge.free_flow_speed_kmh + 1e-9 for speed in speeds_kmh.values()):
            raise ValueError(f"Poryos traffic edge {index} has an invalid speed")
        profiles[edge_id] = TrafficEdgeProfile(edge_id, edge.road_name, speeds_kmh)
    generator = payload.get("generator", {})
    return TrafficOverlay(
        traffic_model=str(payload["traffic_model"]),
        edges=profiles,
        seed=generator.get("seed"),
        profile=str(payload.get("intensity", "")),
        version=str(payload.get("format_version", "")),
        bin_edges=bin_edges,
        base_name=str(payload["base_name"]),
        benchmark_name=str(payload["benchmark_name"]),
        source_kind="poryos-official",
    )
