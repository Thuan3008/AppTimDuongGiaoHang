from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HCMC = ROOT / "data" / "hcmc"
GRAPH_PATH = HCMC / "sidecars" / "hcmc.road.json"
TRAFFIC_PATH = HCMC / "sidecars" / "hcmc.traffic-heavy.json"
ROAD_NAMES = ["Cong Hoa", "Truong Chinh", "Hoang Van Thu", "Au Co", "Lac Long Quan", "Ly Thuong Kiet", "Bac Hai", "Le Van Sy", "Nguyen Van Troi", "Pham Van Hai", "Tan Ky Tan Quy", "Le Trong Tan"]


def build_graph() -> tuple[list[dict], list[dict]]:
    rows, cols = 25, 42
    ref_lat, ref_lon = 10.792, 106.653
    nodes = []
    for y in range(rows):
        for x in range(cols):
            node_id = 1001 + y * cols + x
            nodes.append({"id": node_id, "lat": round(ref_lat + (y - rows / 2) * 0.00125, 7), "lon": round(ref_lon + (x - cols / 2) * 0.00135, 7)})
    edges = []
    edge_no = 1
    for y in range(rows):
        for x in range(cols):
            node = 1001 + y * cols + x
            for dx, dy in ((1, 0), (0, 1)):
                nx, ny = x + dx, y + dy
                if nx >= cols or ny >= rows:
                    continue
                other = 1001 + ny * cols + nx
                arterial = (dx == 1 and y in {0, 8, 16, 24}) or (dy == 1 and x in {0, 10, 20, 30, 40})
                road_class = "arterial" if arterial else ("main" if x % 5 == 0 or y % 5 == 0 else "local")
                road_name = ROAD_NAMES[(x + y * 3) % len(ROAD_NAMES)]
                distance = 150.0 if dx else 140.0
                speed = 48.0 if road_class == "arterial" else (38.0 if road_class == "main" else 25.0)
                edges.append({"id": f"E{edge_no:05d}", "from": node, "to": other, "road_name": road_name, "distance_m": distance, "free_flow_speed_kmh": speed, "bidirectional": True, "road_class": road_class})
                edge_no += 1
    return nodes, edges


def hourly_profile(edge: dict, profile: str) -> dict[str, float]:
    base = edge["free_flow_speed_kmh"]
    road_class = edge["road_class"]
    speeds = {}
    for hour in range(24):
        factor = 1.0
        if road_class == "arterial":
            factor = 0.36 if 7 <= hour <= 8 or 17 <= hour <= 18 else (0.78 if 6 <= hour <= 9 or 16 <= hour <= 19 else 0.96)
        elif road_class == "main":
            factor = 0.55 if 7 <= hour <= 8 or 17 <= hour <= 18 else (0.82 if 6 <= hour <= 9 or 16 <= hour <= 19 else 0.94)
        else:
            factor = 0.9 if 7 <= hour <= 8 or 17 <= hour <= 18 else 0.98
        if profile == "heavy" and road_class in {"arterial", "main"}:
            factor *= 0.9
        speeds[f"{hour:02d}"] = round(max(5.0, base * factor), 2)
    return speeds


def ensure_sidecars() -> tuple[list[dict], list[dict]]:
    HCMC.joinpath("instances").mkdir(parents=True, exist_ok=True)
    HCMC.joinpath("sidecars").mkdir(parents=True, exist_ok=True)
    if GRAPH_PATH.exists() and TRAFFIC_PATH.exists():
        graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        return graph["nodes"], graph["edges"]
    nodes, edges = build_graph()
    GRAPH_PATH.write_text(json.dumps({"schema_version": "1.0", "region": "hcmc-tan-binh-study-area", "nodes": nodes, "edges": edges}, indent=2), encoding="utf-8")
    traffic = {"traffic_model": "fixed_hourly_speed", "seed": 42, "profile": "heavy", "version": "1.0", "generated_at": "fixed-seed-generation", "edges": [{"edge_id": e["id"], "road_name": e["road_name"], "hourly_speed_kmh": hourly_profile(e, "heavy")} for e in edges]}
    TRAFFIC_PATH.write_text(json.dumps(traffic, indent=2), encoding="utf-8")
    return nodes, edges


def generate(customers: int, seed: int, traffic: str) -> Path:
    if customers < 1:
        raise ValueError("--customers must be positive")
    nodes, _ = ensure_sidecars()
    rng = random.Random(seed)
    depot_node = 1001 + (25 // 2) * 42 + (42 // 2)
    candidates = [node for node in nodes if node["id"] != depot_node]
    if customers > len(candidates):
        raise ValueError(f"graph has only {len(candidates)} non-depot nodes")
    selected = rng.sample(candidates, customers)
    by_id = {node["id"]: node for node in nodes}
    demands = [0.0] + [float(rng.randint(1, 10)) for _ in range(customers)]
    services = [0.0] + [float(rng.choice([60, 90, 120, 180, 240])) for _ in range(customers)]
    node_ids = [depot_node] + [node["id"] for node in selected]
    coordinates = [[by_id[node_id]["lat"], by_id[node_id]["lon"]] for node_id in node_ids]
    total_demand = sum(demands)
    capacity = 100.0
    vehicle_count = max(1, math.ceil(total_demand / capacity))
    name = f"hcmc-n{customers}-{traffic}"
    payload = {"instance_name": name, "benchmark_name": "HCMC-TDVRP", "num_customers": customers, "num_vehicles": vehicle_count, "vehicle_capacity": capacity, "coordinates": coordinates, "demands": demands, "service_times": services, "road_node_ids": node_ids, "depot": 0, "reference_lla": {"lat": 10.792, "lon": 106.653, "alt": 0.0}, "horizon": [0.0, 86400.0], "td": {"model": "road-graph", "graph": {"path": "sidecars/hcmc.road.json"}, "traffic": {"path": "sidecars/hcmc.traffic-heavy.json"}, "sample_step": 300.0}, "metadata": {"city": "ho_chi_minh", "study_area": "tan_binh_tan_phu_phu_nhuan_q10_q11_q3", "problem_type": "TDVRP", "traffic_mode": "fixed", "seed": seed, "traffic_profile": traffic, "generated_at": "fixed-seed-generation", "note": "Simulated fixed traffic; not real HCMC traffic."}}
    output = HCMC / "instances" / f"{name}.vrp.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a reproducible HCMC road-graph TDVRP instance")
    parser.add_argument("--customers", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--traffic", choices=["heavy"], default="heavy")
    args = parser.parse_args()
    print(generate(args.customers, args.seed, args.traffic))
