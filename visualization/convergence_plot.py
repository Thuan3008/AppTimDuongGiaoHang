import plotly.graph_objects as go


def convergence_figure(convergence, local_history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=convergence, mode="lines", name="Global best", line={"width": 4, "color": "#2563eb"}))
    if local_history:
        for idx in range(len(local_history[0])):
            fig.add_trace(go.Scatter(y=[row[idx] for row in local_history], mode="lines", name=f"Swarm {idx + 1}", opacity=0.45))
    fig.update_layout(height=360, xaxis_title="Iteration", yaxis_title="Fitness (minimize)", margin={"l": 10, "r": 10, "t": 20, "b": 10})
    return fig
