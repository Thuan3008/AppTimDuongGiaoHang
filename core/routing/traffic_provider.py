from __future__ import annotations

from datetime import datetime

from core.data.models import TrafficOverlay


class TrafficProvider:
    def get_speed(self, edge_id: str, departure_time: float | datetime) -> float:
        raise NotImplementedError


class FixedTrafficProvider(TrafficProvider):
    def __init__(self, traffic: TrafficOverlay):
        self.traffic = traffic

    def get_speed(self, edge_id: str, departure_time: float | datetime) -> float:
        if isinstance(departure_time, datetime):
            seconds = departure_time.hour * 3600 + departure_time.minute * 60 + departure_time.second
        else:
            seconds = float(departure_time)
        bin_edges = self.traffic.bin_edges
        cycle = bin_edges[-1] - bin_edges[0]
        local_time = bin_edges[0] + ((seconds - bin_edges[0]) % cycle)
        slot = 0
        while slot + 1 < len(bin_edges) - 1 and local_time >= bin_edges[slot + 1]:
            slot += 1
        return self.traffic.edges[edge_id].hourly_speed_kmh[f"{slot:02d}"]
