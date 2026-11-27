"""Loader chính thức và chế độ proxy tách biệt cho bộ Poryos2026."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .load_road_graph import load_road_graph
from .load_traffic import load_traffic
from .models import Customer, RoadEdge, RoadNode, TDVRPInstance, TrafficEdgeProfile, TrafficOverlay


class PoryosDataError(ValueError):
    """Báo lỗi dữ liệu Poryos sai schema, sai checksum hoặc thiếu sidecar."""


def load_poryos(
    path: str | Path,
    *,
    collection_root: str | Path | None = None,
    verify_checksums: bool = True,
    allow_proxy: bool = False,
    materialize_atfs: bool = True,
) -> tuple[TDVRPInstance, tuple[dict[int, RoadNode], dict[str, RoadEdge]], TrafficOverlay]:
    """Đọc instance Poryos cùng road graph và traffic sidecar thật.

    Loader chỉ chuyển sang proxy khi ``allow_proxy=True``. Nhờ vậy benchmark
    không thể vô tình gắn nhãn dữ liệu proxy thành dữ liệu road-network thật.
    """
    source = Path(path)
    payload = _read_json(source)
    normalized = _validate_instance_payload(payload)
    graph_ref = payload["td"]["graph"]
    traffic_ref = payload["td"]["traffic"]
    root = _resolve_collection_root(source, graph_ref["path"], collection_root)
    if root is None:
        if allow_proxy:
            return load_poryos_proxy(source)
        expected_root = next((parent for parent in source.parents if parent.name == "TDVRP"), source.parent)
        raise FileNotFoundError(
            "Không tìm thấy road sidecar Poryos chính thức. "
            f"Cần file: {expected_root / graph_ref['path']}. "
            "Chỉ dùng allow_proxy=True cho thử nghiệm proxy."
        )

    graph_path = root / graph_ref["path"]
    traffic_path = root / traffic_ref["path"]
    if not traffic_path.exists():
        if allow_proxy:
            return load_poryos_proxy(source)
        raise FileNotFoundError(f"Không tìm thấy traffic sidecar Poryos chính thức: {traffic_path}")
    if verify_checksums:
        _verify_sha256(graph_path, str(graph_ref.get("sha256", "")), "road graph")
        _verify_sha256(traffic_path, str(traffic_ref.get("sha256", "")), "traffic overlay")

    road_payload = _read_json(graph_path)
    traffic_payload = _read_json(traffic_path)
    _validate_sidecar_pair(payload, road_payload, traffic_payload)
    graph = load_road_graph(graph_path)
    traffic = load_traffic(traffic_path, graph[1])
    arrival_time_functions = None
    if materialize_atfs:
        try:
            from mamut_routing_lib.td import load_td_instance
        except ImportError as error:
            raise PoryosDataError(
                "Thiếu mamut-routing-lib; hãy cài requirements.txt để materialize ATF chính thức"
            ) from error
        loaded = load_td_instance(source, verify_sha256=verify_checksums, collection_root=root)
        arrival_time_functions = loaded.atfs.arcs
    road_node_ids = [int(value) for value in road_payload["node_vertices"]]
    instance = _build_instance(
        payload,
        normalized,
        road_node_ids=road_node_ids,
        graph_path=str(graph_path),
        traffic_path=str(traffic_path),
        metadata_extra={
            "proxy_graph": False,
            "data_source": "poryos-official-sidecars",
            "sidecar_checksums_verified": verify_checksums,
            "routing_policy": "pinned_free_flow_paths",
            "traffic_speed_unit": "m/s (đổi sang km/h khi nạp)",
            "collection_root": str(root),
            "atf_sha256_verified": bool(materialize_atfs and verify_checksums),
        },
        arrival_time_functions=arrival_time_functions,
    )
    return instance, graph, traffic


def load_poryos_bks(path: str | Path) -> dict[str, Any]:
    """Đọc BKS Duration và kiểm tra các trường tối thiểu dùng để tính gap."""
    payload = _read_json(Path(path))
    if payload.get("objective_function") != "Duration":
        raise PoryosDataError("BKS phải dùng objective_function=Duration")
    cost = float(payload["cost"])
    routes = payload.get("routes")
    if not math.isfinite(cost) or cost <= 0 or not isinstance(routes, list):
        raise PoryosDataError("BKS có cost hoặc routes không hợp lệ")
    return payload


def find_poryos_bks(instance_path: str | Path) -> Path:
    """Suy ra đường dẫn BKS Duration nằm cạnh instance."""
    source = Path(instance_path)
    name = source.name.removesuffix(".vrp.json") + ".bks.Duration.json"
    return source.with_name(name)


def load_poryos_proxy(
    path: str | Path,
) -> tuple[TDVRPInstance, tuple[dict[int, RoadNode], dict[str, RoadEdge]], TrafficOverlay]:
    """Dựng complete coordinate graph để smoke test khi sidecar thật chưa có."""
    source = Path(path)
    payload = _read_json(source)
    normalized = _validate_instance_payload(payload)
    n = normalized["num_customers"]
    coordinates = normalized["coordinates"]
    demands = normalized["demands"]
    service_times = normalized["service_times"]
    instance = _build_instance(
        payload,
        normalized,
        road_node_ids=list(range(n + 1)),
        graph_path="proxy://complete-coordinate-graph",
        traffic_path="proxy://profile-derived-traffic",
        metadata_extra={
            "proxy_graph": True,
            "data_source": "coordinate-proxy",
            "sidecar_checksums_verified": False,
            "routing_policy": "dynamic_fastest_path",
        },
        arrival_time_functions=None,
    )
    nodes = {index: RoadNode(index, coordinates[index][0], coordinates[index][1]) for index in range(n + 1)}
    profile_name = source.stem.lower()
    if "light" in profile_name:
        peak_factor = 0.88
    elif "moderate" in profile_name:
        peak_factor = 0.72
    else:
        peak_factor = 0.58
    edges: dict[str, RoadEdge] = {}
    profiles: dict[str, TrafficEdgeProfile] = {}
    for source_id in range(n + 1):
        for target_id in range(source_id + 1, n + 1):
            x1, y1 = coordinates[source_id]
            x2, y2 = coordinates[target_id]
            distance = max(1.0, math.hypot(x1 - x2, y1 - y2))
            edge_id = f"proxy-{source_id}-{target_id}"
            edges[edge_id] = RoadEdge(edge_id, source_id, target_id, "proxy-coordinate", distance, 36.0, True, "proxy")
            speeds: dict[str, float] = {}
            for hour in range(24):
                congestion = peak_factor if hour in (7, 8, 17, 18) else (
                    1.0 - (1.0 - peak_factor) * 0.35 if hour in (6, 9, 16, 19) else 1.0
                )
                speeds[f"{hour:02d}"] = 36.0 * congestion
            profiles[edge_id] = TrafficEdgeProfile(edge_id, "proxy-coordinate", speeds)
    traffic = TrafficOverlay(
        "proxy-time-dependent",
        profiles,
        seed=0,
        profile=profile_name,
        version="proxy-v1",
        source_kind="proxy",
    )
    return instance, (nodes, edges), traffic


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")
    raw = path.read_bytes()
    if path.name.endswith(".gz"):
        raw = gzip.decompress(raw)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise PoryosDataError(f"JSON root phải là object: {path}")
    return payload


def _validate_instance_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Chuẩn hóa và kiểm tra instance trước khi tạo object miền."""
    required = {
        "instance_name", "benchmark_name", "num_customers", "vehicle_capacity",
        "coordinates", "demands", "service_times", "depot", "horizon", "td",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise PoryosDataError(f"Instance thiếu trường bắt buộc: {missing}")
    if str(payload["benchmark_name"]) != "Poryos2026":
        raise PoryosDataError(f"Benchmark không phải Poryos2026: {payload['benchmark_name']!r}")
    n = int(payload["num_customers"])
    coordinates = [tuple(map(float, pair)) for pair in payload["coordinates"]]
    demands = [float(value) for value in payload["demands"]]
    service_times = [float(value) for value in payload["service_times"]]
    if n <= 0 or len(coordinates) != n + 1 or any(len(pair) != 2 for pair in coordinates):
        raise PoryosDataError("coordinates phải chứa depot và đúng n khách hàng")
    if not (len(demands) == len(service_times) == n + 1):
        raise PoryosDataError("demands và service_times phải có đúng n + 1 phần tử")
    if any(not math.isfinite(value) for pair in coordinates for value in pair):
        raise PoryosDataError("coordinates chứa giá trị không hữu hạn")
    if int(payload["depot"]) != 0 or demands[0] != 0 or service_times[0] != 0:
        raise PoryosDataError("Poryos yêu cầu depot=0 với demand và service_time bằng 0")
    capacity = float(payload["vehicle_capacity"])
    if capacity <= 0 or any(value <= 0 or value > capacity for value in demands[1:]):
        raise PoryosDataError("Demand khách hàng phải dương và không vượt vehicle_capacity")
    if any(value < 0 or not math.isfinite(value) for value in service_times):
        raise PoryosDataError("service_times phải hữu hạn và không âm")
    horizon = tuple(map(float, payload["horizon"]))
    if len(horizon) != 2 or not horizon[0] < horizon[1]:
        raise PoryosDataError("horizon phải là [start, end] hợp lệ")
    td = payload["td"]
    if td.get("model") != "road-graph" or "graph" not in td or "traffic" not in td:
        raise PoryosDataError("Loader hiện yêu cầu Poryos td.model=road-graph")
    time_windows = payload.get("time_windows")
    normalized_windows: list[tuple[float, float]] | None = None
    if time_windows is not None:
        normalized_windows = [tuple(map(float, window)) for window in time_windows]
        if len(normalized_windows) != n + 1 or any(len(window) != 2 or window[0] > window[1] for window in normalized_windows):
            raise PoryosDataError("time_windows phải có n + 1 cặp [earliest, latest] hợp lệ")
    return {
        "num_customers": n,
        "coordinates": coordinates,
        "demands": demands,
        "service_times": service_times,
        "capacity": capacity,
        "horizon": horizon,
        "time_windows": normalized_windows,
    }


def _build_instance(
    payload: dict[str, Any],
    normalized: dict[str, Any],
    *,
    road_node_ids: list[int],
    graph_path: str,
    traffic_path: str,
    metadata_extra: dict[str, Any],
    arrival_time_functions: Any,
) -> TDVRPInstance:
    n = normalized["num_customers"]
    if len(road_node_ids) != n + 1:
        raise PoryosDataError("road node mapping phải có đúng n + 1 phần tử")
    capacity = normalized["capacity"]
    demands = normalized["demands"]
    declared_vehicles = payload.get("num_vehicles")
    vehicles = int(declared_vehicles) if declared_vehicles is not None else math.ceil(sum(demands[1:]) / capacity)
    if vehicles <= 0 or sum(demands[1:]) > vehicles * capacity + 1e-9:
        raise PoryosDataError("Số xe không đủ tải trọng để phục vụ toàn bộ demand")
    coordinates = normalized["coordinates"]
    service_times = normalized["service_times"]
    customers = [
        Customer(index, road_node_ids[index], demands[index], service_times[index], coordinates[index])
        for index in range(1, n + 1)
    ]
    return TDVRPInstance(
        instance_name=str(payload["instance_name"]),
        benchmark_name=str(payload["benchmark_name"]),
        num_customers=n,
        num_vehicles=vehicles,
        vehicle_capacity=capacity,
        coordinates=coordinates,
        demands=demands,
        service_times=service_times,
        road_node_ids=road_node_ids,
        depot=0,
        reference_lla=dict(payload.get("reference_lla", {})),
        horizon=normalized["horizon"],
        sample_step=float(payload["td"].get("sample_step", 60.0)),
        graph_path=graph_path,
        traffic_path=traffic_path,
        metadata={**dict(payload.get("metadata", {})), **metadata_extra},
        customers=customers,
        time_windows=normalized["time_windows"],
        arrival_time_functions=arrival_time_functions,
    )


def resolve_poryos_sidecar_root(
    source: Path,
    relative_path: str | Path,
    explicit_root: str | Path | None = None,
    fallback_roots: tuple[str | Path, ...] = (),
    *,
    require_manifest: bool = True,
) -> Path | None:
    """Tìm thư mục gốc chứa sidecar (kiểm tra mamut-collection.json cho Poryos)."""
    rel = Path(relative_path)

    def _valid(root: Path) -> bool:
        return (root / rel).exists() and (
            not require_manifest or (root / "mamut-collection.json").is_file()
        )

    if explicit_root is not None:
        root = Path(explicit_root)
        return root if _valid(root) else None
    for ancestor in source.parents:
        if _valid(ancestor):
            return ancestor
    for candidate in fallback_roots:
        root = Path(candidate)
        if _valid(root):
            return root
    return None


def _resolve_collection_root(source: Path, graph_ref: str, explicit_root: str | Path | None) -> Path | None:
    return resolve_poryos_sidecar_root(source, graph_ref, explicit_root)


def _verify_sha256(path: Path, expected: str, label: str) -> None:
    if len(expected) != 64:
        raise PoryosDataError(f"Instance thiếu SHA-256 hợp lệ cho {label}")
    raw = path.read_bytes()
    if path.name.endswith(".gz"):
        raw = gzip.decompress(raw)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise PoryosDataError(f"SHA-256 {label} không khớp: expected={expected}, actual={actual}")


def _validate_sidecar_pair(instance: dict[str, Any], road: dict[str, Any], traffic: dict[str, Any]) -> None:
    if road.get("format") != "mamut-road-graph" or int(road.get("format_version", -1)) != 2:
        raise PoryosDataError("Road sidecar không đúng mamut-road-graph v2")
    if traffic.get("format") != "mamut-traffic-overlay" or int(traffic.get("format_version", -1)) != 1:
        raise PoryosDataError("Traffic sidecar không đúng mamut-traffic-overlay v1")
    expected_base = instance.get("metadata", {}).get("base_instance_name")
    if not expected_base or road.get("base_name") != expected_base or traffic.get("base_name") != expected_base:
        raise PoryosDataError("Instance, road graph và traffic overlay không cùng base_name")
    if road.get("benchmark_name") != instance["benchmark_name"] or traffic.get("benchmark_name") != instance["benchmark_name"]:
        raise PoryosDataError("Benchmark name giữa instance và sidecar không khớp")
    if int(road.get("num_customers", -1)) != int(instance["num_customers"]):
        raise PoryosDataError("Số khách hàng giữa instance và road sidecar không khớp")
    if [float(value) for value in road.get("horizon", [])] != [float(value) for value in instance["horizon"]]:
        raise PoryosDataError("Horizon giữa instance và road sidecar không khớp")
    if [float(value) for value in road.get("bin_edges", [])] != [float(value) for value in traffic.get("bin_edges", [])]:
        raise PoryosDataError("Traffic bin_edges không khớp road graph")
    if len(road.get("edges", [])) != len(traffic.get("edge_speeds", [])):
        raise PoryosDataError("Số dòng tốc độ không khớp số cạnh road graph")
