"""Giao diện Streamlit cho hai thuật toán RAPO được chốt của đề tài."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from algorithms.poryos_baseline_adapter import run_baseline
from algorithms.rapo_2709 import RAPO2709Config, optimize
from core.data.load_poryos import find_poryos_bks, load_poryos, load_poryos_bks


ROOT = Path(__file__).resolve().parent
ALGORITHMS = ("RAPO-2709", "RAPO-Pufferfish")
ALGORITHM_LABELS = {
    "RAPO-2709": "TD-ERAPO",
}


def algorithm_label(name: str) -> str:
    """Tên hiển thị trên giao diện; định danh nội bộ vẫn giữ nguyên."""
    return ALGORITHM_LABELS.get(name, name)

# Chỉ cho phép app chạy các instance Poryos đã có trong repository.
DATASETS = {
    "Hong Kong n=25 · HYB · BPR-heavy": ROOT / "data/TDVRP/hong_kong/n=25/poryos-hong_kong-n25-hyb/bpr-heavy/poryos-hong_kong-n25-hyb-bpr-heavy.vrp.json",
    "Hong Kong n=50 · HYB · BPR-heavy": ROOT / "data/TDVRP/hong_kong/n=50/poryos-hong_kong-n50-hyb/bpr-heavy/poryos-hong_kong-n50-hyb-bpr-heavy.vrp.json",
    "Hong Kong n=100 · HYB · BPR-heavy": ROOT / "data/TDVRP/hong_kong/n=100/poryos-hong_kong-n100-hyb/bpr-heavy/poryos-hong_kong-n100-hyb-bpr-heavy.vrp.json",
}

st.set_page_config(page_title="TDVRP RAPO", page_icon="🚚", layout="wide")


@st.cache_resource(show_spinner="Đang nạp instance, road graph, traffic và ATF...")
def load_dataset(path: str):
    """Nạp dữ liệu chính thức và kiểm tra checksum sidecar."""
    return load_poryos(Path(path), verify_checksums=True, allow_proxy=False, materialize_atfs=True)


def make_common_population(num_customers: int, population_size: int, seed: int) -> list[tuple[int, ...]]:
    """Tạo P0 chung để hai thuật toán bắt đầu từ cùng điều kiện."""
    rng = random.Random(seed)
    customers = list(range(1, num_customers + 1))
    return [tuple(rng.sample(customers, len(customers))) for _ in range(population_size)]


def run_algorithm(algorithm: str, dataset_path: Path, instance: Any, graph: Any, traffic: Any,
                  population_size: int, nfe_budget: int, seed: int,
                  common_population: list[tuple[int, ...]]) -> Any:
    """Chạy đúng một trong hai thuật toán đã chốt cho app."""
    if algorithm == "RAPO-2709":
        config = RAPO2709Config(population_size=population_size, max_function_evaluations=nfe_budget, seed=seed)
        return optimize(instance, graph, traffic, config=config, seed=seed, initial_population=common_population)
    if algorithm == "RAPO-Pufferfish":
        # Adapter giữ nguyên implementation Pufferfish và chỉ chuẩn hóa output.
        return run_baseline("RAPO-Pufferfish", instance, graph, traffic,
                            instance_path=dataset_path, population_size=population_size,
                            nfe_budget=nfe_budget, seed=seed,
                            initial_population=common_population)
    raise ValueError(f"Thuật toán không được hỗ trợ: {algorithm}")


def dataset_figure(instance) -> go.Figure:
    """Vẽ depot và khách hàng theo tọa độ instance."""
    figure = go.Figure()
    points = instance.coordinates[1:]
    figure.add_trace(go.Scattergl(x=[p[0] for p in points], y=[p[1] for p in points], mode="markers",
                                  text=[str(i) for i in range(1, instance.num_customers + 1)],
                                  hovertemplate="Khách hàng %{text}<extra></extra>",
                                  marker={"size": 8, "color": "#2563eb"}, name="Khách hàng"))
    depot = instance.coordinates[instance.depot]
    figure.add_trace(go.Scatter(x=[depot[0]], y=[depot[1]], mode="markers",
                                marker={"size": 15, "color": "#111827", "symbol": "star"}, name="Depot"))
    figure.update_layout(title="Phân bố điểm giao hàng", height=520,
                         xaxis_title="Tọa độ X", yaxis_title="Tọa độ Y")
    return figure


def route_figure(instance, graph, result) -> go.Figure:
    """Vẽ route trace trên road graph thật."""
    nodes, _ = graph
    figure = go.Figure()
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    for index, route in enumerate(result.best_evaluation.routes):
        path_nodes = [nodes[node_id] for node_id in route.node_path if node_id in nodes]
        figure.add_trace(go.Scattergl(x=[node.lon for node in path_nodes], y=[node.lat for node in path_nodes],
                                      mode="lines", line={"width": 2, "color": colors[index % len(colors)]},
                                      name=f"Route {index + 1}",
                                      hovertemplate=f"Route {index + 1}<br>Khách: {len(route.customer_ids)}<br>Duration: {route.duration:.2f}s<extra></extra>"))
    customer_nodes = [nodes[node_id] for node_id in instance.road_node_ids[1:] if node_id in nodes]
    depot_node = nodes[instance.road_node_ids[instance.depot]]
    figure.add_trace(go.Scattergl(x=[node.lon for node in customer_nodes], y=[node.lat for node in customer_nodes],
                                  mode="markers", marker={"size": 6, "color": "#f8fafc", "line": {"width": 1, "color": "#111827"}}, name="Khách hàng"))
    figure.add_trace(go.Scatter(x=[depot_node.lon], y=[depot_node.lat], mode="markers",
                                marker={"size": 14, "color": "#111827", "symbol": "star"}, name="Depot"))
    figure.update_layout(title="Lộ trình tốt nhất trên road graph", height=640,
                         xaxis_title="Longitude", yaxis_title="Latitude")
    return figure


def convergence_figure(results: dict[str, Any]) -> go.Figure:
    """Vẽ đường hội tụ của các thuật toán đang được chọn."""
    figure = go.Figure()
    for algorithm, result in results.items():
        values = list(result.convergence)
        figure.add_trace(go.Scatter(x=list(range(1, len(values) + 1)), y=values,
                                    mode="lines", name=algorithm_label(algorithm)))
    figure.update_layout(title="Hội tụ theo NFE", xaxis_title="NFE",
                         yaxis_title="Fitness (giây)", height=440)
    return figure


def route_rows(result) -> list[dict[str, object]]:
    """Chuẩn hóa route trace thành bảng hiển thị và CSV."""
    return [{"route": i + 1, "customers": " → ".join(map(str, route.customer_ids)),
             "customer_count": len(route.customer_ids), "load": route.load,
             "start_time": route.start_time, "end_time": route.end_time,
             "travel_time": route.travel_time, "service_time": route.service_time,
             "waiting_time": route.waiting_time, "duration": route.duration,
             "distance_m": route.distance}
            for i, route in enumerate(result.best_evaluation.routes)]


def config_payload(config: Any) -> dict[str, Any]:
    """Chuyển dataclass hoặc mapping config thành dictionary."""
    return asdict(config) if is_dataclass(config) else dict(config or {})


def result_payload(dataset_path: Path, algorithm: str, config: Any, result: Any) -> dict[str, Any]:
    """Tạo JSON gồm metric, nghiệm, hội tụ và route trace."""
    return {"algorithm": algorithm_label(algorithm), "dataset": str(dataset_path), "config": config_payload(config),
            "fitness": result.best_evaluation.fitness, "runtime_seconds": result.runtime_seconds,
            "evaluations": result.evaluations, "metadata": result.metadata,
            "best_solution": result.best_solution.routes, "evaluation": asdict(result.best_evaluation),
            "convergence": list(result.convergence)}


def main() -> None:
    st.sidebar.title("TDVRP RAPO")
    st.sidebar.caption("Ứng dụng chạy hai thuật toán đã chốt")
    dataset_name = st.sidebar.selectbox("Dataset", list(DATASETS))
    dataset_path = DATASETS[dataset_name]
    selected = st.sidebar.multiselect("Thuật toán", ALGORITHMS, default=list(ALGORITHMS),
                                      format_func=algorithm_label)
    population_size = int(st.sidebar.number_input("Population size", 2, 50, 10, 1))
    nfe_budget = int(st.sidebar.number_input("Ngân sách NFE", min_value=2, max_value=10000, value=290, step=10))
    seed = int(st.sidebar.number_input("Seed", 0, 999999, 42, 1))
    run = st.sidebar.button("Chạy thuật toán", type="primary", width="stretch")
    st.sidebar.caption("Hai thuật toán dùng chung dataset, P0, seed, population và NFE.")

    st.title("TDVRP — TD-ERAPO vs RAPO-Pufferfish")
    st.write("Objective: `total_route_duration = total_travel_time + total_service_time`.")
    try:
        instance, graph, traffic = load_dataset(str(dataset_path))
        bks = load_poryos_bks(find_poryos_bks(dataset_path))
    except Exception as error:
        st.error(f"Không thể nạp dữ liệu Poryos chính thức: {error}")
        return

    if not selected:
        st.warning("Hãy chọn ít nhất một thuật toán ở thanh bên.")
    common_population = make_common_population(instance.num_customers, population_size, seed)
    run_key = f"results::{dataset_path}::{population_size}::{nfe_budget}::{seed}"
    config_key = f"configs::{dataset_path}::{population_size}::{nfe_budget}::{seed}"
    if run and selected:
        with st.spinner("Đang chạy thuật toán; n lớn có thể cần thời gian..."):
            current_results: dict[str, Any] = {}
            current_configs: dict[str, Any] = {}
            for algorithm in selected:
                try:
                    current_results[algorithm] = run_algorithm(algorithm, dataset_path, instance, graph, traffic,
                                                              population_size, nfe_budget, seed, common_population)
                    current_configs[algorithm] = (RAPO2709Config(population_size=population_size,
                                                                 max_function_evaluations=nfe_budget, seed=seed)
                                                  if algorithm == "RAPO-2709"
                                                  else {"population_size": population_size,
                                                        "max_function_evaluations": nfe_budget, "seed": seed})
                    st.toast(f"{algorithm_label(algorithm)} đã chạy xong")
                except Exception as error:
                    st.error(f"{algorithm_label(algorithm)} lỗi: {error}")
            st.session_state[run_key] = current_results
            st.session_state[config_key] = current_configs

    results = st.session_state.get(run_key, {})
    configs = st.session_state.get(config_key, {})
    dataset_tab, result_tab, trace_tab, algorithm_tab = st.tabs(["Dataset", "Kết quả", "Route trace & Export", "Cơ chế"])
    with dataset_tab:
        metrics = st.columns(5)
        metrics[0].metric("Khách hàng", instance.num_customers)
        metrics[1].metric("Capacity", f"{instance.vehicle_capacity:g}")
        metrics[2].metric("Traffic", traffic.traffic_model)
        metrics[3].metric("BKS Duration", f"{float(bks['cost']):,.2f}")
        metrics[4].metric("Sidecar", "Official" if not instance.metadata.get("proxy_graph") else "Proxy")
        st.success("Đã kiểm tra checksum road graph, traffic sidecar và ATF; proxy graph bị tắt.")
        st.plotly_chart(dataset_figure(instance), width="stretch")

    with result_tab:
        if not results:
            st.info("Chọn thuật toán rồi bấm ‘Chạy thuật toán’ ở thanh bên.")
        else:
            bks_cost = float(bks["cost"])
            rows = [{"Thuật toán": algorithm_label(algorithm), "Fitness": result.best_evaluation.fitness,
                     "Gap BKS (%)": 100.0 * (result.best_evaluation.fitness - bks_cost) / bks_cost,
                     "Travel time": result.best_evaluation.total_travel_time,
                     "Service time": result.best_evaluation.total_service_time,
                     "Routes": len(result.best_evaluation.routes), "NFE": result.evaluations,
                     "Runtime (s)": result.runtime_seconds}
                    for algorithm, result in results.items()]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.plotly_chart(convergence_figure(results), width="stretch")
            if len(results) == 2:
                winner = min(results, key=lambda name: results[name].best_evaluation.fitness)
                st.success(f"Kết quả tốt hơn theo fitness của lần chạy này: {algorithm_label(winner)}")

    with trace_tab:
        if not results:
            st.info("Route trace và nút export sẽ xuất hiện sau khi chạy.")
        else:
            algorithm = st.selectbox("Thuật toán xem route", list(results), format_func=algorithm_label)
            result = results[algorithm]
            st.plotly_chart(route_figure(instance, graph, result), width="stretch")
            st.dataframe(pd.DataFrame(route_rows(result)), width="stretch", hide_index=True)
            route_number = st.selectbox("Chọn route để xem timeline", range(1, len(result.best_evaluation.routes) + 1))
            selected_route = result.best_evaluation.routes[route_number - 1]
            st.dataframe(pd.DataFrame(asdict(stop) for stop in selected_route.stops), width="stretch", hide_index=True)
            payload = result_payload(dataset_path, algorithm, configs[algorithm], result)
            export_columns = st.columns(2)
            safe_name = algorithm_label(algorithm).lower().replace("-", "_")
            export_columns[0].download_button("Tải kết quả JSON", json.dumps(payload, ensure_ascii=False, indent=2),
                                              file_name=f"{safe_name}_{instance.instance_name}_seed{seed}.json",
                                              mime="application/json", width="stretch")
            export_columns[1].download_button("Tải route CSV", pd.DataFrame(route_rows(result)).to_csv(index=False),
                                              file_name=f"{safe_name}_{instance.instance_name}_routes.csv",
                                              mime="text/csv", width="stretch")

    with algorithm_tab:
        st.subheader("TD-ERAPO")
        st.caption("Time-Dependent Elite-Adaptive Rabbit-and-Pufferfish Optimization")
        st.markdown("Giant tour + Split động, ARO/POA/Elite, Elite AOS, strict NFE và tối ưu departure time bằng NDCPWLF.")
        st.subheader("RAPO-Pufferfish")
        st.markdown("ARO kết hợp Pufferfish Attack/Defense, giữ cân bằng NFE giữa hai pha và dùng adapter để xuất route trace chung.")
        st.info("Hai thuật toán dùng chung instance, P0, seed, population size và NFE; kết quả được chuẩn hóa về cùng route trace/evaluator hiển thị.")


if __name__ == "__main__":
    main()
