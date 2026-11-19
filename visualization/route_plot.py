from __future__ import annotations

import plotly.graph_objects as go

from core.models import EvaluationResult, Solution, TDVRPProblem


def route_figure(problem: TDVRPProblem, solution: Solution, evaluation: EvaluationResult, vehicle_filter: str = "All Vehicles") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[problem.depot.x], y=[problem.depot.y], mode="markers+text", text=["Depot"], textposition="top center", marker={"size": 16, "symbol": "star", "color": "black"}, name="Depot"))
    fig.add_trace(go.Scatter(x=[c.x for c in problem.customers], y=[c.y for c in problem.customers], mode="markers+text", text=[f"C{c.id}" for c in problem.customers], textposition="top center", marker={"size": 9, "color": "#94a3b8"}, name="Customers", hovertext=[f"C{c.id} · demand {c.demand} · service {c.service_time} min" for c in problem.customers], hoverinfo="text"))
    palette = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    for timeline, route in zip(evaluation.routes, solution.routes):
        if vehicle_filter != "All Vehicles" and vehicle_filter != f"Vehicle {timeline.vehicle_id}": continue
        coords = [problem.depot if n == 0 else next(c for c in problem.customers if c.id == n) for n in route]
        fig.add_trace(go.Scatter(x=[n.x for n in coords], y=[n.y for n in coords], mode="lines+markers", line={"color": palette[(timeline.vehicle_id - 1) % len(palette)], "width": 3}, marker={"size": 7}, name=f"Vehicle {timeline.vehicle_id}"))
    fig.update_layout(height=560, margin={"l": 10, "r": 10, "t": 20, "b": 10}, xaxis_title="X", yaxis_title="Y", yaxis_scaleanchor="x", legend_title="Routes")
    return fig
