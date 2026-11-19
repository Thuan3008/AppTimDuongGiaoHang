from __future__ import annotations

from datetime import datetime, timedelta

from .models import EvaluationResult, RouteTimeline, Solution, TDVRPProblem
from .travel_time import SimulatedTravelTimeProvider


class RouteEvaluator:
    def __init__(self, problem: TDVRPProblem, start_time: datetime):
        self.problem = problem
        self.start_time = start_time
        self.travel_provider = SimulatedTravelTimeProvider(problem)
        self.node_by_id = {n.id: n for n in problem.nodes}

    def evaluate(self, solution: Solution) -> EvaluationResult:
        total_distance = total_travel = total_service = 0.0
        timelines = []
        for vehicle_idx, route in enumerate(solution.routes):
            clock = self.start_time
            load = 0.0
            rows = []
            route_distance = route_travel = route_service = 0.0
            previous = route[0]
            for node_id in route:
                travel = self.travel_provider.get_travel_time(previous, node_id, clock) if node_id != previous else 0.0
                distance = self.problem.distance_matrix[self._idx(previous), self._idx(node_id)]
                arrival = clock + timedelta(hours=travel)
                customer = self.node_by_id[node_id]
                service = getattr(customer, "service_time", 0.0)
                demand = getattr(customer, "demand", 0.0)
                if node_id != 0:
                    load += demand
                departure = arrival + timedelta(minutes=service)
                rows.append({"Vehicle": vehicle_idx + 1, "Node": "Depot" if node_id == 0 else f"C{node_id}", "Arrival Time": arrival.strftime("%H:%M"), "Service Time": service, "Departure Time": departure.strftime("%H:%M"), "Demand": demand, "Current Load": load, "Travel Time From Previous Node": travel})
                route_distance += distance
                route_travel += travel
                route_service += service
                clock = departure
                previous = node_id
            timelines.append(RouteTimeline(vehicle_idx + 1, rows, route_distance, route_travel, route_service))
            total_distance += route_distance; total_travel += route_travel; total_service += route_service
        return EvaluationResult(total_distance, total_travel, total_service, total_travel + total_service / 60, timelines)

    def _idx(self, node_id: int) -> int:
        return next(i for i, n in enumerate(self.problem.nodes) if n.id == node_id)
