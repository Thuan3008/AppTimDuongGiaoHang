from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core.distance import build_distance_matrix
from core.models import Customer, Depot, TDVRPProblem, Vehicle


def load_problem(data_dir: Path = Path("data")) -> TDVRPProblem:
    benchmark = data_dir / "TDVRP" / "hong_kong" / "n=10" / "poryos-hong_kong-n10-poi" / "bpr-light" / "poryos-hong_kong-n10-poi-bpr-light.vrp.json"
    if benchmark.exists():
        return load_tdvrp_json(benchmark)
    config = json.loads((data_dir / "sample_config.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(data_dir / "sample_customers.csv")
    required = {"id", "x", "y", "demand", "service_time"}
    if not required.issubset(frame.columns): raise ValueError(f"CSV must contain: {sorted(required)}")
    if frame.id.duplicated().any(): raise ValueError("Customer IDs must be unique")
    if (frame.demand < 0).any(): raise ValueError("Demand cannot be negative")
    if config["vehicle_capacity"] <= 0: raise ValueError("Vehicle capacity must be positive")
    customers = [Customer(int(r.id), float(r.x), float(r.y), float(r.demand), float(r.service_time)) for r in frame.itertuples()]
    depot = Depot(**config["depot"])
    vehicles = [Vehicle(i + 1, config["vehicle_capacity"]) for i in range(config["vehicle_count"])]
    problem = TDVRPProblem(depot, customers, vehicles, config["base_speed"], config["traffic_multipliers"], instance_name="sample")
    problem.distance_matrix = build_distance_matrix(problem.nodes)
    if sum(c.demand for c in customers) > sum(v.capacity for v in vehicles): raise ValueError("Dataset demand exceeds total capacity")
    return problem


def load_tdvrp_json(path: Path, vehicle_count: int = 4) -> TDVRPProblem:
    """Load the repository's Poryos TDVRP JSON format.

    The benchmark stores service times in seconds and coordinates in meters.
    The app normalizes service time to minutes and uses a simulated traffic
    provider because the referenced road/traffic sidecars are not included.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    coords = payload["coordinates"]
    depot_index = int(payload.get("depot", 0))
    depot = Depot(0, float(coords[depot_index][0]), float(coords[depot_index][1]))
    customers = []
    for idx, (coord, demand, service_seconds) in enumerate(zip(coords, payload["demands"], payload["service_times"])):
        if idx == depot_index:
            continue
        customers.append(Customer(idx, float(coord[0]), float(coord[1]), float(demand), float(service_seconds) / 60.0))
    capacity = float(payload["vehicle_capacity"])
    vehicles = [Vehicle(i + 1, capacity) for i in range(vehicle_count)]
    traffic = payload.get("metadata", {}).get("traffic", {}).get("intensity", "light")
    multipliers = {"normal": 1.0, "morning_peak": 1.35, "evening_peak": 1.55}
    if traffic == "moderate": multipliers = {"normal": 1.0, "morning_peak": 1.55, "evening_peak": 1.8}
    if traffic == "heavy": multipliers = {"normal": 1.0, "morning_peak": 1.8, "evening_peak": 2.1}
    problem = TDVRPProblem(depot, customers, vehicles, base_speed=18000.0, traffic_multipliers=multipliers, instance_name=payload.get("instance_name", path.stem))
    problem.distance_matrix = build_distance_matrix(problem.nodes)
    if sum(c.demand for c in customers) > sum(v.capacity for v in vehicles):
        raise ValueError("Benchmark demand exceeds selected vehicle capacity")
    return problem
