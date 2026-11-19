from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.data.load_instance import load_instance
from core.data.load_road_graph import load_road_graph
from core.data.load_traffic import load_traffic
from core.routing.fastest_path import find_fastest_path


def precompute(instance_path: Path, output_path: Path, step: float = 300.0) -> None:
    instance = load_instance(instance_path)
    graph = load_road_graph(instance.graph_path)
    traffic = load_traffic(instance.traffic_path, graph[1])
    nodes = [0, *range(1, instance.num_customers + 1)]
    road_nodes = [instance.road_node_ids[index] for index in nodes]
    table = {}
    for departure in range(int(instance.horizon[0]), int(instance.horizon[1]), int(step)):
        slot = str(departure)
        table[slot] = {}
        for source_index, source in enumerate(road_nodes):
            for target_index, target in enumerate(road_nodes):
                if source == target:
                    table[slot][f"{source_index},{target_index}"] = 0.0
                else:
                    result = find_fastest_path(graph, traffic, source, target, departure, step)
                    table[slot][f"{source_index},{target_index}"] = result.travel_time_seconds
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"instance_name": instance.instance_name, "sample_step": step, "travel_time_seconds": table}, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=float, default=300.0)
    args = parser.parse_args()
    precompute(args.instance, args.output, args.step)
