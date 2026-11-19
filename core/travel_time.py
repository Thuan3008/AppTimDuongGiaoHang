from __future__ import annotations

from datetime import datetime, time

from .models import TDVRPProblem


class TravelTimeProvider:
    def get_travel_time(self, from_node: int, to_node: int, departure_time: datetime) -> float:
        raise NotImplementedError


class SimulatedTravelTimeProvider(TravelTimeProvider):
    def __init__(self, problem: TDVRPProblem):
        self.problem = problem
        self._index = {node.id: i for i, node in enumerate(problem.nodes)}

    def traffic_scenario(self, departure_time: datetime) -> str:
        current = departure_time.time()
        if time(7, 0) <= current < time(9, 0):
            return "morning_peak"
        if time(16, 0) <= current < time(19, 0):
            return "evening_peak"
        return "normal"

    def multiplier(self, departure_time: datetime) -> float:
        return self.problem.traffic_multipliers[self.traffic_scenario(departure_time)]

    def get_travel_time(self, from_node: int, to_node: int, departure_time: datetime) -> float:
        distance = self.problem.distance_matrix[self._index[from_node], self._index[to_node]]
        return float(distance / self.problem.base_speed * self.multiplier(departure_time))
