from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RoadNode:
    id: int
    lat: float
    lon: float


@dataclass(frozen=True)
class RoadEdge:
    id: str
    from_node: int
    to_node: int
    road_name: str
    distance_m: float
    free_flow_speed_kmh: float
    bidirectional: bool = True
    road_class: str = "local"
    # Các OSM way tương ứng; graph cũ không có field này nên mặc định rỗng.
    osm_way_ids: tuple[int, ...] = tuple()


@dataclass(frozen=True)
class Customer:
    id: int
    road_node_id: int
    demand: float
    service_time: float
    coordinate: tuple[float, float]


@dataclass(frozen=True)
class Vehicle:
    id: int
    capacity: float


@dataclass(frozen=True)
class TrafficEdgeProfile:
    edge_id: str
    road_name: str
    hourly_speed_kmh: dict[str, float]


@dataclass(frozen=True)
class TrafficOverlay:
    traffic_model: str
    edges: dict[str, TrafficEdgeProfile]
    seed: int | None = None
    profile: str | None = None
    version: str | None = None
    bin_edges: tuple[float, ...] = tuple(float(hour * 3600) for hour in range(25))
    base_name: str | None = None
    benchmark_name: str | None = None
    source_kind: str = "local"

    def speed_for(self, edge: RoadEdge, slot: int) -> float:
        """Return the configured speed for one road edge and traffic bin."""
        return float(self.edges[edge.id].hourly_speed_kmh[f"{slot:02d}"])


@dataclass(frozen=True)
class ProceduralTrafficOverlay:
    """Compact traffic overlay for a large OSM graph.

    Storing 24 numbers for every edge of the city graph would create a very
    large sidecar and an equally large in-memory object.  This overlay keeps a
    shared hourly congestion profile and only stores explicit observations for
    OSM ways that are present in the source traffic dataset.
    """

    observed_way_speeds: dict[int, dict[str, float]]
    hourly_alpha: tuple[float, ...]
    road_class_weights: dict[str, float]
    traffic_model: str = "osm_observed_with_procedural_fallback"
    seed: int | None = None
    profile: str | None = "heavy"
    version: str | None = "1.0"
    bin_edges: tuple[float, ...] = tuple(float(hour * 3600) for hour in range(25))
    source_kind: str = "hcmc-osm-procedural"

    def speed_for(self, edge: RoadEdge, slot: int) -> float:
        hour = f"{slot:02d}"
        observed = next(
            (self.observed_way_speeds[way_id] for way_id in edge.osm_way_ids
             if way_id in self.observed_way_speeds),
            None,
        )
        if observed is not None:
            return max(1.0, min(edge.free_flow_speed_kmh, float(observed[hour])))
        base_alpha = self.hourly_alpha[slot]
        class_weight = self.road_class_weights.get(edge.road_class, 1.0)
        alpha = 1.0 + (base_alpha - 1.0) * class_weight
        return max(1.0, edge.free_flow_speed_kmh / alpha)


@dataclass(frozen=True)
class TDVRPInstance:
    instance_name: str
    benchmark_name: str
    num_customers: int
    num_vehicles: int
    vehicle_capacity: float
    coordinates: list[tuple[float, float]]
    demands: list[float]
    service_times: list[float]
    road_node_ids: list[int]
    depot: int
    reference_lla: dict[str, float]
    horizon: tuple[float, float]
    sample_step: float
    graph_path: str
    traffic_path: str
    metadata: dict[str, Any]
    customers: list[Customer]
    time_windows: list[tuple[float, float]] | None = None
    arrival_time_functions: Any = None
