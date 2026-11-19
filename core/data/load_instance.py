from __future__ import annotations

import json
from pathlib import Path

from .load_road_graph import load_road_graph
from .load_traffic import load_traffic
from .models import Customer, TDVRPInstance


def load_instance(path: str | Path, validate_sidecars: bool = True) -> TDVRPInstance:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Instance not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    n = int(payload["num_customers"])
    coordinates = [tuple(map(float, value)) for value in payload["coordinates"]]
    demands = [float(value) for value in payload["demands"]]
    service_times = [float(value) for value in payload["service_times"]]
    road_node_ids = [int(value) for value in payload["road_node_ids"]]
    if len(coordinates) != n + 1 or any(len(values) != 2 for values in coordinates):
        raise ValueError("coordinates must contain depot plus every customer")
    if not (len(demands) == len(service_times) == len(road_node_ids) == n + 1):
        raise ValueError("demands, service_times and road_node_ids must contain depot plus every customer")
    if int(payload.get("depot", 0)) != 0 or demands[0] != 0:
        raise ValueError("customer 0 must be the depot with zero demand")
    if any(value <= 0 for value in demands[1:]) or any(value < 0 for value in service_times):
        raise ValueError("customers need positive demand and non-negative service time")
    capacity = float(payload["vehicle_capacity"])
    vehicles = int(payload["num_vehicles"])
    if capacity <= 0 or vehicles <= 0 or sum(demands) > vehicles * capacity + 1e-9:
        raise ValueError("vehicle capacity is insufficient for all demand")
    graph_path = path.parent.parent / payload["td"]["graph"]["path"]
    traffic_path = path.parent.parent / payload["td"]["traffic"]["path"]
    if validate_sidecars:
        nodes, edges = load_road_graph(graph_path)
        load_traffic(traffic_path, edges)
        if any(node_id not in nodes for node_id in road_node_ids):
            raise ValueError("Instance references a road node missing from graph")
    customers = [Customer(i, road_node_ids[i], demands[i], service_times[i], coordinates[i]) for i in range(1, n + 1)]
    return TDVRPInstance(
        instance_name=str(payload["instance_name"]), benchmark_name=str(payload.get("benchmark_name", "HCMC-TDVRP")),
        num_customers=n, num_vehicles=vehicles, vehicle_capacity=capacity, coordinates=coordinates,
        demands=demands, service_times=service_times, road_node_ids=road_node_ids, depot=0,
        reference_lla=dict(payload["reference_lla"]), horizon=tuple(map(float, payload["horizon"])),
        sample_step=float(payload["td"].get("sample_step", 300.0)), graph_path=str(graph_path),
        traffic_path=str(traffic_path), metadata=dict(payload.get("metadata", {})), customers=customers,
    )
