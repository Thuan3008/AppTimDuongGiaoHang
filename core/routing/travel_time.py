from __future__ import annotations

from datetime import datetime

from core.data.models import RoadEdge, TrafficOverlay


def get_edge_travel_time(edge: RoadEdge, departure_time: float | datetime, traffic_data: TrafficOverlay | object) -> float:
    """Tính thời gian qua cạnh, kể cả khi hành trình cắt qua ranh giới traffic bin."""
    if isinstance(departure_time, datetime):
        seconds = departure_time.hour * 3600 + departure_time.minute * 60 + departure_time.second
    else:
        seconds = float(departure_time)
    bin_edges = tuple(getattr(traffic_data, "bin_edges", tuple(float(hour * 3600) for hour in range(25))))
    if len(bin_edges) < 2:
        raise ValueError("Traffic overlay must define at least one time bin")

    # Profile được lặp theo chu kỳ của horizon để các route qua nửa đêm vẫn xác định.
    cycle_start, cycle_end = bin_edges[0], bin_edges[-1]
    cycle = cycle_end - cycle_start
    remaining_m = float(edge.distance_m)
    elapsed = 0.0
    while remaining_m > 1e-9:
        local_time = cycle_start + ((seconds + elapsed - cycle_start) % cycle)
        slot = _find_bin(bin_edges, local_time)
        if hasattr(traffic_data, "speed_for"):
            speed_kmh = float(traffic_data.speed_for(edge, slot))
        else:
            profile = traffic_data.edges[edge.id] if hasattr(traffic_data, "edges") else traffic_data[edge.id]
            speed_kmh = float(profile.hourly_speed_kmh[f"{slot:02d}"])
        speed_mps = speed_kmh / 3.6
        seconds_to_boundary = bin_edges[slot + 1] - local_time
        if seconds_to_boundary <= 1e-12:
            seconds_to_boundary = cycle
        possible_m = speed_mps * seconds_to_boundary
        if possible_m + 1e-9 >= remaining_m:
            elapsed += remaining_m / speed_mps
            break
        remaining_m -= possible_m
        elapsed += seconds_to_boundary
    return elapsed


def _find_bin(bin_edges: tuple[float, ...], local_time: float) -> int:
    """Tìm traffic bin bằng tìm kiếm nhị phân, ưu tiên bin bên phải tại biên."""
    low, high = 0, len(bin_edges) - 1
    while low + 1 < high:
        middle = (low + high) // 2
        if local_time < bin_edges[middle]:
            high = middle
        else:
            low = middle
    return min(low, len(bin_edges) - 2)
