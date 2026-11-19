"""Map observed OSM traffic ways onto the HCMC graph and build an overlay.

The HCMC graph is a synthetic grid, so this uses normalized road name plus
geometry distance and records the mapping quality.  Only mappings within the
configured threshold are applied; unmatched edges keep the simulated traffic
profile.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd


def normalize(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"^duong\s+", "", text)
    return re.sub(r"[^a-z0-9]", "", text)


def decode_polyline(encoded: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    index = lat = lon = 0
    while index < len(encoded):
        result = shift = 0
        while True:
            byte = ord(encoded[index]) - 63
            index += 1
            result |= (byte & 0x1F) << shift
            shift += 5
            if byte < 0x20:
                break
        lat += ~(result >> 1) if result & 1 else result >> 1
        result = shift = 0
        while True:
            byte = ord(encoded[index]) - 63
            index += 1
            result |= (byte & 0x1F) << shift
            shift += 5
            if byte < 0x20:
                break
        lon += ~(result >> 1) if result & 1 else result >> 1
        points.append((lat / 1e5, lon / 1e5))
    return points


def point_segment_distance_m(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    lat0 = math.radians(point[0])
    scale_x = 111320.0 * math.cos(lat0)
    scale_y = 110540.0
    px, py = point[1] * scale_x, point[0] * scale_y
    ax, ay = a[1] * scale_x, a[0] * scale_y
    bx, by = b[1] * scale_x, b[0] * scale_y
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def polyline_distance(point: tuple[float, float], line: list[tuple[float, float]]) -> float:
    return min(point_segment_distance_m(point, a, b) for a, b in zip(line, line[1:]))


def integrate(graph_path: Path, observations_path: Path, polylines_path: Path,
              fallback_traffic_path: Path, output_graph: Path, output_traffic: Path,
              mapping_path: Path, threshold_m: float) -> None:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    fallback = json.loads(fallback_traffic_path.read_text(encoding="utf-8"))
    observations = pd.read_parquet(observations_path)
    polylines = pd.read_parquet(polylines_path)
    lines = {int(row.way_id): decode_polyline(row.polyline) for row in polylines.itertuples()}
    names: dict[str, list[int]] = defaultdict(list)
    for row in observations.itertuples():
        if pd.notna(row.name):
            names[normalize(row.name)].append(int(row.way_id))
    names = {key: sorted(set(values)) for key, values in names.items()}

    edge_to_ways: dict[str, list[int]] = {}
    report: list[dict[str, object]] = []
    for edge in graph["edges"]:
        edge_id = str(edge["id"])
        road_key = normalize(edge["road_name"])
        candidates = names.get(road_key, [])
        midpoint = (
            (next(node["lat"] for node in graph["nodes"] if node["id"] == edge["from"]) +
             next(node["lat"] for node in graph["nodes"] if node["id"] == edge["to"])) / 2,
            (next(node["lon"] for node in graph["nodes"] if node["id"] == edge["from"]) +
             next(node["lon"] for node in graph["nodes"] if node["id"] == edge["to"])) / 2,
        )
        ranked = sorted(
            ((way_id, polyline_distance(midpoint, lines[way_id])) for way_id in candidates if way_id in lines),
            key=lambda item: item[1],
        )
        accepted = [way_id for way_id, distance in ranked if distance <= threshold_m]
        if accepted:
            edge_to_ways[edge_id] = accepted
            edge["osm_way_ids"] = accepted
        report.append({
            "edge_id": edge_id,
            "road_name": edge["road_name"],
            "candidate_count": len(candidates),
            "accepted_way_ids": accepted,
            "best_distance_m": ranked[0][1] if ranked else None,
        })

    # Build a 24-hour observed overlay; unmapped edges retain the old profile.
    observed = observations.copy()
    observed["timestamp"] = pd.to_datetime(observed["timestamp"], utc=True)
    observed["hour"] = observed["timestamp"].dt.tz_convert("Asia/Ho_Chi_Minh").dt.hour
    observed["way_id"] = pd.to_numeric(observed["way_id"], errors="coerce").astype("Int64")
    observed = observed[(observed["currentSpeed"] > 0) & (~observed["roadClosure"].astype(bool))]
    way_speeds = observed.groupby(["way_id", "hour"])["currentSpeed"].median().to_dict()
    fallback_rows = {str(row["edge_id"]): row for row in fallback["edges"]}
    traffic_edges = []
    for edge in graph["edges"]:
        edge_id = str(edge["id"])
        way_ids = edge_to_ways.get(edge_id, [])
        speeds = {}
        for hour in range(24):
            values = [way_speeds[(way_id, hour)] for way_id in way_ids if (way_id, hour) in way_speeds]
            speeds[f"{hour:02d}"] = round(float(sum(values) / len(values)), 3) if values else float(fallback_rows[edge_id]["hourly_speed_kmh"][f"{hour:02d}"])
        traffic_edges.append({"edge_id": edge_id, "road_name": edge["road_name"], "hourly_speed_kmh": speeds})

    output_graph.parent.mkdir(parents=True, exist_ok=True)
    output_graph.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    output_traffic.write_text(json.dumps({
        "traffic_model": "observed_hourly_with_simulated_fallback",
        "source": str(observations_path),
        "timezone": "Asia/Ho_Chi_Minh",
        "mapping_threshold_m": threshold_m,
        "edges": traffic_edges,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    mapping_path.write_text(json.dumps({
        "format": "hcmc-osm-way-edge-mapping",
        "threshold_m": threshold_m,
        "mapped_edges": len(edge_to_ways),
        "total_edges": len(graph["edges"]),
        "mapping": report,
        "warning": "Unmapped edges use simulated heavy traffic fallback.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"mapped_edges": len(edge_to_ways), "total_edges": len(graph["edges"]), "threshold_m": threshold_m}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold-m", type=float, default=500.0)
    parser.add_argument("--graph", type=Path, default=Path("data/hcmc/sidecars/hcmc.road.json"))
    parser.add_argument("--observations", type=Path, default=Path("data/vantoctb/vietnam-road-traffic-observations.parquet"))
    parser.add_argument("--polylines", type=Path, default=Path("data/vantoctb/vietnam-road-traffic-polylines.parquet"))
    parser.add_argument("--fallback", type=Path, default=Path("data/hcmc/sidecars/hcmc.traffic-heavy.json"))
    parser.add_argument("--output-graph", type=Path, default=Path("data/hcmc/sidecars/hcmc.road-observed.json"))
    parser.add_argument("--output-traffic", type=Path, default=Path("data/hcmc/sidecars/hcmc.traffic-observed.json"))
    parser.add_argument("--mapping", type=Path, default=Path("data/hcmc/sidecars/hcmc.osm-way-mapping.json"))
    args = parser.parse_args()
    integrate(args.graph, args.observations, args.polylines, args.fallback, args.output_graph, args.output_traffic, args.mapping, args.threshold_m)
