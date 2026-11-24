"""TD-ERAPO (theo cơ chế ``rapo_2709.py``) chạy trực tiếp trên bộ dữ liệu HCMC.

Tệp này giữ nguyên toàn bộ cơ chế thuật toán của ``rapo_2709.py`` (giant tour,
Split động, NDCPWLF + tối ưu giờ xuất phát, nhánh ARO dùng Order Crossover,
nhánh POA dùng Criticality D-K-W + Regret-2, nhánh Elite dùng AOS và đếm đúng
1 NFE mỗi ứng viên), đồng thời bổ sung bộ dựng hàm thời gian đến ``NDCPWLF``
trực tiếp từ đồ thị đường bộ và dữ liệu giao thông của ``data/hcmc`` cũng như
ràng buộc số lượng phương tiện ``num_vehicles`` của bài toán HCMC.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import OrderedDict
import hashlib
import heapq
import json
import math
import pickle
import random
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Callable, Iterable

from mamut_routing_lib.td import NDCPWLF, apply_ready_time, restrict_domain
from mamut_routing_lib.td.igp import build_arc_atf
from mamut_routing_lib.td.roadgraph import _drop_exactly_collinear, _simplify_points

if __package__ in (None, ""):
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from core.data.evaluator import (
    HCMCEvaluationResult,
    HCMCTDVRPEvaluator,
    RouteEvaluation,
    RouteLegTrace,
    RouteStopTrace,
)
from core.data.load_instance import load_instance
from core.data.load_road_graph import load_road_graph
from core.data.load_traffic import load_traffic
from core.data.models import ProceduralTrafficOverlay, RoadEdge, RoadNode, TDVRPInstance, TrafficOverlay
from core.data.solution import repair_solution
from core.models import Solution
from core.routing.fastest_path import prepare_adjacency
from core.routing.travel_time import get_edge_travel_time


ELITE_OPERATORS = ("relocate", "swap", "two_opt", "two_opt_star")
DEFAULT_HCMC_INSTANCE = (
    Path(__file__).resolve().parents[1] / "data/hcmc/instances/hcmc-n10-heavy.vrp.json"
)
_ATF_CACHE_DIR = Path(__file__).resolve().parents[1] / "results" / ".atf_cache"
_ATF_CACHE_VERSION = 1
_MEM_ATF_CACHE: OrderedDict[
    str, tuple[list[list[float]], list[list[float]], dict[int, dict[int, NDCPWLF]], Any, Any]
] = OrderedDict()
_MEM_CACHE_LOCK = Lock()
_MEM_CACHE_MAXSIZE = 8
_COMPACT_GRAPH_CACHE: OrderedDict[tuple[int, int, int, int, int], "_CompactGraphData"] = OrderedDict()
_COMPACT_GRAPH_CACHE_LOCK = Lock()
_COMPACT_GRAPH_CACHE_MAXSIZE = 4


@dataclass
class _CompactGraphData:
    nodes_ref: dict[int, RoadNode]
    edges_ref: dict[str, RoadEdge]
    traffic_ref: Any
    node_ids: tuple[int, ...]
    node_to_idx: dict[int, int]
    num_v: int
    bin_edges: tuple[float, ...]
    num_bins: int
    cycle_start: float
    cycle_end: float
    cycle: float
    uniform_bin_width: float | None
    adj: list[list[tuple[int, float, float, list[float], list[float]]]]
    compact_edges: list[tuple[int, int, float, float, list[float], list[float], bool]]
    transitions: tuple[float, ...]
    worst_slowdown: float
    bin_max_alpha: tuple[float, ...] | None = None
    core_v: int = 0
    orig_to_core: list[int] | None = None
    core_adj: list[list[tuple[int, float, float, list[float], list[float]]]] | None = None
    peeled: list[bool] | None = None
    peel_parent: list[int] | None = None
    node_to_chain: list[int] | None = None
    chains: list[tuple[int, int, tuple[int, ...], Any, Any]] | None = None
    chain_shortcuts: list[tuple[int, int, Any, Any]] | None = None
    pinned_adj: list[list[tuple[int, float, float, int]]] | None = None
    directed_edge_data: list[tuple[int, int, float, list[float]]] | None = None


@dataclass(frozen=True)
class HCMCTDERAPOConfig:
    """Tham số chuẩn của TD-ERAPO trên tập dữ liệu HCMC."""

    population_size: int = 10
    max_function_evaluations: int = 100
    destroy_fraction: float = 0.10
    criticality_floor: float = 0.10
    elite_budget_fraction: float = 0.20
    aos_epsilon: float = 0.10
    aos_alpha: float = 0.20
    earliest_departure: float = 0.0
    latest_departure: float | None = None
    exact_split: bool = False
    sample_step: float | None = None
    simplify_tolerance: float = 1.0
    routing_policy: str | None = None
    seed: int | None = None

    def validate(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size phải từ 2 trở lên")
        if self.max_function_evaluations < self.population_size:
            raise ValueError("Ngân sách NFE phải đủ để đánh giá toàn bộ quần thể ban đầu")
        if not 0 < self.destroy_fraction <= 1:
            raise ValueError("destroy_fraction phải thuộc (0, 1]")
        if self.criticality_floor < 0:
            raise ValueError("criticality_floor không được âm")
        if not 0 <= self.elite_budget_fraction <= 1:
            raise ValueError("elite_budget_fraction phải thuộc [0, 1]")
        if self.latest_departure is not None and self.latest_departure < self.earliest_departure:
            raise ValueError("latest_departure phải lớn hơn hoặc bằng earliest_departure")
        if not 0 <= self.aos_epsilon <= 1 / len(ELITE_OPERATORS):
            raise ValueError("aos_epsilon quá lớn so với số toán tử Elite")
        if not 0 < self.aos_alpha <= 1:
            raise ValueError("aos_alpha phải thuộc (0, 1]")
        if type(self.exact_split) is not bool:
            raise ValueError("exact_split phải là bool")
        if self.sample_step is not None and self.sample_step <= 0:
            raise ValueError("sample_step phải lớn hơn 0")
        if self.simplify_tolerance < 0:
            raise ValueError("simplify_tolerance không được âm")
        if self.routing_policy is not None and self.routing_policy not in {
            "dynamic_fastest_path",
            "pinned_free_flow_paths",
        }:
            raise ValueError(
                "routing_policy phải là 'dynamic_fastest_path' hoặc 'pinned_free_flow_paths'"
            )


@dataclass(frozen=True)
class _LegState:
    source: int
    target: int
    departure: float
    arrival: float
    travel_time: float


@dataclass(frozen=True)
class _StopState:
    customer: int
    arrival: float
    service_start: float
    departure: float
    waiting: float
    service: float
    load: float


@dataclass(frozen=True)
class _RouteState:
    customers: tuple[int, ...]
    departure: float
    end_time: float
    duration: float
    travel_time: float
    service_time: float
    waiting_time: float
    load: float
    legs: tuple[_LegState, ...]
    stops: tuple[_StopState, ...]


@dataclass(frozen=True)
class HCMCTDERAPOCandidate:
    permutation: tuple[int, ...]
    routes: tuple[tuple[int, ...], ...]
    route_states: tuple[_RouteState, ...]
    fitness: float


@dataclass(frozen=True)
class HCMCTDERAPOResult:
    best_solution: Solution
    best_evaluation: HCMCEvaluationResult
    best_permutation: tuple[int, ...]
    convergence: list[float]
    runtime_seconds: float
    evaluations: int
    nfe_by_branch: dict[str, int]
    operator_scores: dict[str, float]
    operator_uses: dict[str, int]
    operator_successes: dict[str, int]
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def nfe(self) -> int:
        """Tên thay thế để tương thích với ExperimentRunner."""
        return self.evaluations


class _HCMCATFProvider:
    """Cung cấp hàm thời gian đến NDCPWLF từ đồ thị đường và giao thông HCMC.

    - Nếu ``instance.arrival_time_functions`` đã có sẵn, dùng trực tiếp.
    - Ngược lại, dựng đồ thị nén và tính sẵn ``travel_at_zero`` + ``free_flow``
      cho toàn bộ cặp (kho/khách) chỉ bằng 1 lần quét Dijkstra/nguồn; các hàm
      ``NDCPWLF`` đầy đủ cho từng nguồn được dựng lười (lazy per source) theo
      lưới chuyển pha giao thông để giảm bộ nhớ và thời gian khởi động.
    """

    def __init__(
        self,
        instance: TDVRPInstance,
        graph: tuple[dict[int, RoadNode], dict[str, RoadEdge]],
        traffic: TrafficOverlay | Any,
        sample_step: float | None = None,
        simplify_tolerance: float = 1.0,
        routing_policy: str | None = None,
        earliest_departure: float = 0.0,
        latest_departure: float | None = None,
    ) -> None:
        self.instance = instance
        self.graph = graph
        self.nodes, self.edges = graph
        self.traffic = traffic
        self.num_nodes = instance.num_customers + 1
        self.routing_policy = routing_policy or str(
            instance.metadata.get("routing_policy", "dynamic_fastest_path")
        )
        self.horizon = (float(instance.horizon[0]), float(instance.horizon[1]))
        self.step = float(sample_step or instance.sample_step or 300.0)
        self.simplify_tolerance = float(simplify_tolerance)
        self.earliest_departure = float(earliest_departure)
        self.latest_departure = (
            float(latest_departure) if latest_departure is not None else None
        )

        self._external_atfs = instance.arrival_time_functions
        self._atfs_by_source: dict[int, dict[int, NDCPWLF]] = {}

        if self._external_atfs is not None:
            self.atf_source = "instance_atfs"
            self.t0_matrix: list[list[float]] | None = None
            self.ff_matrix: list[list[float]] | None = None
            self._active_end: float | None = None
            return

        self.atf_source = f"hcmc_{self.routing_policy}"
        self._init_compact_graph()
        self._instantiate_run_graph()
        self._tree_cache: dict[tuple[int, float], dict[int, float]] = {}

        ref_zero = self._reference_zero_time()
        self.t0_matrix = self._compute_scalar_matrix(departure=ref_zero)
        self.ff_matrix = self._compute_free_flow_matrix()

        # Xác định giới hạn trên an toàn cho các cung khi biết latest_departure
        if self.latest_departure is not None:
            end_h = min(self.horizon[1], self.latest_departure)
            self._active_end = self._estimate_active_end(end_h)
        else:
            self._active_end = None

        self._build_grid()

        # Nếu max hành trình vượt quá cửa sổ tìm trước mặc định, mở rộng lookback và dựng lại lưới.
        max_tt = max(
            max(row)
            for row in self.t0_matrix
        )
        if max_tt > self._lookback:
            self._lookback = max_tt * 1.5
            self._build_grid()

    def _reference_zero_time(self) -> float:
        if self.horizon[0] <= 0.0 <= self.horizon[1]:
            return 0.0
        return self.horizon[0]

    def _bin_alpha_at(self, t: float) -> float:
        if self._bin_max_alpha is None:
            return self._worst_slowdown
        rel = (t - self._cycle_start) % self._cycle
        if self._uniform_bin_width is not None:
            slot = min(int(rel // self._uniform_bin_width), self._num_bins - 1)
        else:
            slot = min(bisect_right(self._bin_edges, self._cycle_start + rel) - 1, self._num_bins - 1)
        return self._bin_max_alpha[max(0, slot)]

    def _max_single_route_travel(self, matrix: list[list[float]], k_fit: int) -> float:
        n = self.instance.num_customers
        depot = self.instance.depot
        if n <= 0 or k_fit <= 0:
            return 0.0
        if n <= 8:
            cap = self.instance.vehicle_capacity + 1e-9
            demands = self.instance.demands
            full = 1 << n
            # dp[(mask << 4) | last_idx] lưu chi phí đường đi dài nhất từ depot tới last (0-based 0..n-1)
            dp = [-1.0] * (full * n)
            mask_load = [0.0] * full
            for mask in range(1, full):
                lsb = mask & -mask
                bit = lsb.bit_length() - 1
                mask_load[mask] = mask_load[mask ^ lsb] + float(demands[bit + 1])
                if mask == lsb:
                    if mask_load[mask] <= cap:
                        dp[(mask * n) + bit] = matrix[depot][bit + 1]
                    continue
                if mask_load[mask] > cap:
                    continue
                tmp = mask
                while tmp:
                    lb = tmp & -tmp
                    last = lb.bit_length() - 1
                    prev_mask = mask ^ lb
                    best_val = -1.0
                    tmp_p = prev_mask
                    base_prev = prev_mask * n
                    c_last = last + 1
                    while tmp_p:
                        pb = tmp_p & -tmp_p
                        prev_i = pb.bit_length() - 1
                        pv = dp[base_prev + prev_i]
                        if pv >= 0.0:
                            cand = pv + matrix[prev_i + 1][c_last]
                            if cand > best_val:
                                best_val = cand
                        tmp_p ^= pb
                    dp[(mask * n) + last] = best_val
                    tmp ^= lb
            best_cycle = 0.0
            for mask in range(1, full):
                if mask_load[mask] > cap:
                    continue
                base_m = mask * n
                for last in range(n):
                    val = dp[base_m + last]
                    if val >= 0.0:
                        cand = val + matrix[last + 1][depot]
                        if cand > best_cycle:
                            best_cycle = cand
            return best_cycle

        max_depot_out = max((matrix[depot][c] for c in range(self.num_nodes) if c != depot), default=0.0)
        row_maxes = sorted(
            (
                max((matrix[i][j] for j in range(self.num_nodes) if j != i), default=0.0)
                for i in range(self.num_nodes)
                if i != depot
            ),
            reverse=True,
        )
        return max_depot_out + sum(row_maxes[:k_fit])

    def _estimate_active_end(self, end_h: float) -> float:
        cap = self.instance.vehicle_capacity + 1e-9
        demands = sorted(float(self.instance.demands[c]) for c in range(1, self.num_nodes))
        acc_d = 0.0
        k_fit = 0
        for d in demands:
            if acc_d + d <= cap:
                acc_d += d
                k_fit += 1
            else:
                break
        k_fit = max(1, min(self.instance.num_customers, k_fit))
        service_bound = sum(
            sorted((float(self.instance.service_times[c]) for c in range(1, self.num_nodes)), reverse=True)[:k_fit]
        )

        if self._bin_max_alpha is None:
            assert self.ff_matrix is not None
            max_tour_ff = self._max_single_route_travel(self.ff_matrix, k_fit)
            span_bound = max_tour_ff * self._worst_slowdown + service_bound + 3600.0
            active_end = min(self.horizon[1], end_h + span_bound)
            if self.instance.time_windows is not None:
                max_ready = max(
                    (min(self.horizon[1], float(w[0])) for w in self.instance.time_windows if w),
                    default=self.earliest_departure,
                )
                active_end = min(self.horizon[1], max(active_end, max_ready + span_bound))
            return max(self.earliest_departure, active_end)

        # Tính ma trận thời gian thực tại end_h (được cache lại để tái sử dụng cho _grid)
        end_matrix = self._compute_scalar_matrix(departure=end_h)
        max_tour_end = self._max_single_route_travel(end_matrix, k_fit)
        # Bù biên độ biến thiên cục bộ của observed_way_speeds + service của một tuyến
        rem_work = max_tour_end * 1.25 + service_bound

        def _propagate_from(start_t: float, work: float) -> float:
            cur_t = min(self.horizon[1], start_t)
            rem = work
            min_alpha = self._bin_alpha_at(cur_t)
            guard = 0
            while rem > 1e-6 and cur_t < self.horizon[1] - 1e-9 and guard < self._num_bins * 4:
                guard += 1
                rel = (cur_t - self._cycle_start) % self._cycle
                if self._uniform_bin_width is not None:
                    slot = min(int(rel // self._uniform_bin_width), self._num_bins - 1)
                else:
                    slot = min(bisect_right(self._bin_edges, self._cycle_start + rel) - 1, self._num_bins - 1)
                to_b = self._bin_edges[slot + 1] - (self._cycle_start + rel)
                if to_b <= 1e-9:
                    to_b = self._cycle
                b_end = min(self.horizon[1], cur_t + to_b)
                a_bin = self._bin_max_alpha[slot] if self._bin_max_alpha is not None else self._worst_slowdown
                if a_bin < min_alpha:
                    min_alpha = a_bin
                scale = max(1.0, a_bin / max(min_alpha, 1e-6))
                avail = b_end - cur_t
                if rem * scale <= avail + 1e-9:
                    cur_t += rem * scale
                    rem = 0.0
                    break
                rem -= avail / scale
                cur_t = b_end
            return min(self.horizon[1], cur_t + 1800.0)

        active_end = _propagate_from(end_h, rem_work)
        if self.instance.time_windows is not None:
            max_ready = max(
                (min(self.horizon[1], float(w[0])) for w in self.instance.time_windows if w),
                default=self.earliest_departure,
            )
            if max_ready > end_h:
                assert self.ff_matrix is not None
                ff_span = self._max_single_route_travel(self.ff_matrix, k_fit) * self._worst_slowdown + service_bound + 1800.0
                active_end = min(self.horizon[1], max(active_end, max_ready + ff_span))
        return max(self.earliest_departure, active_end)

    def _init_compact_graph(self) -> None:
        cache_key = (
            id(self.nodes),
            len(self.nodes),
            id(self.edges),
            len(self.edges),
            id(self.traffic),
        )
        use_cache = len(self.nodes) >= 1_000
        if use_cache:
            with _COMPACT_GRAPH_CACHE_LOCK:
                cached = _COMPACT_GRAPH_CACHE.get(cache_key)
                if (
                    cached is not None
                    and cached.nodes_ref is self.nodes
                    and cached.edges_ref is self.edges
                    and cached.traffic_ref is self.traffic
                ):
                    _COMPACT_GRAPH_CACHE.move_to_end(cache_key)
                    self._apply_compact_data(cached)
                    return

        node_ids = tuple(sorted(self.nodes.keys()))
        node_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
        num_v = len(node_ids)

        bin_edges = tuple(
            float(x)
            for x in getattr(self.traffic, "bin_edges", tuple(float(h * 3600) for h in range(25)))
        )
        if len(bin_edges) < 2:
            raise ValueError("Traffic overlay cần ít nhất một bin thời gian")
        num_bins = len(bin_edges) - 1
        cycle_start = bin_edges[0]
        cycle_end = bin_edges[-1]
        cycle = cycle_end - cycle_start

        widths = [bin_edges[k + 1] - bin_edges[k] for k in range(num_bins)]
        uniform_bin_width: float | None = (
            widths[0]
            if all(abs(w - widths[0]) <= 1e-9 for w in widths) and widths[0] > 0
            else None
        )

        adj: list[list[tuple[int, float, float, list[float], list[float]]]] = [
            [] for _ in range(num_v)
        ]
        compact_edges: list[tuple[int, int, float, float, list[float], list[float], bool]] = []
        worst_slowdown = 1.0
        bin_max_alpha: tuple[float, ...] | None = None

        if type(self.traffic) is ProceduralTrafficOverlay:
            obs_map = self.traffic.observed_way_speeds
            hourly_alpha = self.traffic.hourly_alpha
            class_weights = self.traffic.road_class_weights
            hour_keys = tuple(f"{s:02d}" for s in range(num_bins))
            alphas_by_class: dict[str, list[float]] = {}
            fallback_cache: dict[tuple[float, str], tuple[list[float], list[float]]] = {}
            max_class_weight = max(class_weights.values(), default=1.0)
            max_alpha = max(hourly_alpha, default=1.0)
            worst_slowdown = max(3.5, 1.0 + max(0.0, max_alpha - 1.0) * max_class_weight)
            bin_max_alpha = tuple(
                max(1.0, 1.0 + (hourly_alpha[s] - 1.0) * max_class_weight)
                for s in range(num_bins)
            )

            for edge in self.edges.values():
                u = node_to_idx[edge.from_node]
                v = node_to_idx[edge.to_node]
                dist = float(edge.distance_m)
                ff_kmh = float(edge.free_flow_speed_kmh)
                ff_speed = ff_kmh / 3.6
                observed = None
                if edge.osm_way_ids:
                    for way_id in edge.osm_way_ids:
                        if way_id in obs_map:
                            observed = obs_map[way_id]
                            break
                if observed is not None:
                    speeds_mps = [
                        max(1.0, min(ff_kmh, float(observed[hour_keys[s]]))) / 3.6
                        for s in range(num_bins)
                    ]
                    inv_speeds = [1.0 / val for val in speeds_mps]
                else:
                    r_class = edge.road_class
                    f_key = (ff_kmh, r_class)
                    cached_sp = fallback_cache.get(f_key)
                    if cached_sp is None:
                        alphas = alphas_by_class.get(r_class)
                        if alphas is None:
                            w_class = class_weights.get(r_class, 1.0)
                            alphas = [1.0 + (hourly_alpha[s] - 1.0) * w_class for s in range(num_bins)]
                            alphas_by_class[r_class] = alphas
                        speeds_mps = [max(1.0, ff_kmh / alphas[s]) / 3.6 for s in range(num_bins)]
                        inv_speeds = [1.0 / val for val in speeds_mps]
                        cached_sp = (speeds_mps, inv_speeds)
                        fallback_cache[f_key] = cached_sp
                    speeds_mps, inv_speeds = cached_sp
                compact_edges.append((u, v, dist, ff_speed, speeds_mps, inv_speeds, bool(edge.bidirectional)))
                adj[u].append((v, dist, ff_speed, speeds_mps, inv_speeds))
                if edge.bidirectional:
                    adj[v].append((u, dist, ff_speed, speeds_mps, inv_speeds))
        else:
            has_speed_for = hasattr(self.traffic, "speed_for")
            traffic_edges = getattr(self.traffic, "edges", None)
            hour_keys = tuple(f"{s:02d}" for s in range(num_bins))
            for edge in self.edges.values():
                u = node_to_idx[edge.from_node]
                v = node_to_idx[edge.to_node]
                dist = float(edge.distance_m)
                ff_speed = float(edge.free_flow_speed_kmh) / 3.6
                if has_speed_for:
                    speeds_mps = [
                        max(1e-6, float(self.traffic.speed_for(edge, s)) / 3.6)
                        for s in range(num_bins)
                    ]
                else:
                    prof = (
                        traffic_edges[edge.id]
                        if traffic_edges is not None
                        else self.traffic[edge.id]
                    )
                    speeds_mps = [
                        max(1e-6, float(prof.hourly_speed_kmh[hour_keys[s]]) / 3.6)
                        for s in range(num_bins)
                    ]
                min_sp = min(speeds_mps)
                if min_sp > 0:
                    worst_slowdown = max(worst_slowdown, ff_speed / min_sp)
                inv_speeds = [1.0 / s for s in speeds_mps]
                compact_edges.append((u, v, dist, ff_speed, speeds_mps, inv_speeds, bool(edge.bidirectional)))
                adj[u].append((v, dist, ff_speed, speeds_mps, inv_speeds))
                if edge.bidirectional:
                    adj[v].append((u, dist, ff_speed, speeds_mps, inv_speeds))

        # Xác định sẵn mọi mốc chuyển pha trên miền toàn cục của traffic
        transitions: list[float] = []
        if len(compact_edges) > 20_000 and hasattr(self.traffic, "hourly_alpha"):
            alphas = self.traffic.hourly_alpha
            for b in range(1, num_bins):
                boundary = bin_edges[b]
                if alphas[b - 1] != alphas[b]:
                    transitions.append(boundary)
        if not transitions:
            for b in range(1, num_bins):
                boundary = bin_edges[b]
                if any(row[4][b - 1] != row[4][b] for row in compact_edges):
                    transitions.append(boundary)

        core_v = 0
        orig_to_core = None
        core_adj = None
        peeled = None
        peel_parent = None
        node_to_chain = None
        chains = None
        chain_shortcuts = None
        if use_cache:
            (
                core_v,
                orig_to_core,
                core_adj,
                peeled,
                peel_parent,
                node_to_chain,
                chains,
                chain_shortcuts,
            ) = self._build_core_contraction(num_v, adj)

        compact_data = _CompactGraphData(
            nodes_ref=self.nodes,
            edges_ref=self.edges,
            traffic_ref=self.traffic,
            node_ids=node_ids,
            node_to_idx=node_to_idx,
            num_v=num_v,
            bin_edges=bin_edges,
            num_bins=num_bins,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            cycle=cycle,
            uniform_bin_width=uniform_bin_width,
            adj=adj,
            compact_edges=compact_edges,
            transitions=tuple(transitions),
            worst_slowdown=worst_slowdown,
            bin_max_alpha=bin_max_alpha,
            core_v=core_v,
            orig_to_core=orig_to_core,
            core_adj=core_adj,
            peeled=peeled,
            peel_parent=peel_parent,
            node_to_chain=node_to_chain,
            chains=chains,
            chain_shortcuts=chain_shortcuts,
        )
        if use_cache:
            with _COMPACT_GRAPH_CACHE_LOCK:
                _COMPACT_GRAPH_CACHE[cache_key] = compact_data
                if len(_COMPACT_GRAPH_CACHE) > _COMPACT_GRAPH_CACHE_MAXSIZE:
                    _COMPACT_GRAPH_CACHE.popitem(last=False)
        self._apply_compact_data(compact_data)

    @staticmethod
    def _build_core_contraction(
        num_v: int,
        adj: list[list[tuple[int, float, float, list[float], list[float]]]],
    ) -> tuple[
        int,
        list[int],
        list[list[tuple[int, float, float, list[float], list[float]]]],
        list[bool],
        list[int],
        list[int],
        list[tuple[int, int, tuple[int, ...], Any, Any]],
        list[tuple[int, int, Any, Any]],
    ]:
        """Rút gọn nhánh cụt (hẻm cụt) và nén chuỗi bậc 2 đồng tốc mà vẫn khôi phục chính xác cho tập đích."""
        undir_nbrs: list[list[int]] = [[] for _ in range(num_v)]
        for u in range(num_v):
            edges_u = adj[u]
            if not edges_u:
                continue
            un_u = undir_nbrs[u]
            for v, _, _, _, _ in edges_u:
                if u != v and v not in un_u:
                    un_u.append(v)
                    undir_nbrs[v].append(u)

        peel_parent = [-1] * num_v
        peeled = [False] * num_v
        q = [u for u in range(num_v) if len(undir_nbrs[u]) <= 1]
        while q:
            u = q.pop()
            if peeled[u] or len(undir_nbrs[u]) > 1:
                continue
            peeled[u] = True
            if undir_nbrs[u]:
                w = undir_nbrs[u][0]
                peel_parent[u] = w
                undir_nbrs[w].remove(u)
                if not peeled[w] and len(undir_nbrs[w]) <= 1:
                    q.append(w)
                undir_nbrs[u].clear()

        contractable = [False] * num_v
        nbr_pair: list[tuple[int, int] | None] = [None] * num_v
        for x in range(num_v):
            if peeled[x] or len(undir_nbrs[x]) != 2:
                continue
            a, b = undir_nbrs[x]
            ax = [row for row in adj[a] if row[0] == x]
            if len(ax) > 1:
                continue
            xb = [row for row in adj[x] if row[0] == b]
            if len(xb) > 1:
                continue
            bx = [row for row in adj[b] if row[0] == x]
            if len(bx) > 1:
                continue
            xa = [row for row in adj[x] if row[0] == a]
            if len(xa) > 1:
                continue
            fwd_ok = (not ax and not xb) or (
                len(ax) == 1 and len(xb) == 1 and (ax[0][3] is xb[0][3] or ax[0][3] == xb[0][3])
            )
            rev_ok = (not bx and not xa) or (
                len(bx) == 1 and len(xa) == 1 and (bx[0][3] is xa[0][3] or bx[0][3] == xa[0][3])
            )
            if fwd_ok and rev_ok and (ax or bx):
                contractable[x] = True
                nbr_pair[x] = (a, b)

        node_to_chain = [-1] * num_v
        chains: list[tuple[int, int, tuple[int, ...], Any, Any]] = []
        visited_c = [False] * num_v
        for x in range(num_v):
            if not contractable[x] or visited_c[x]:
                continue
            pair0 = nbr_pair[x]
            assert pair0 is not None
            a0, b0 = pair0
            left_nodes: list[int] = []
            cur, prev = a0, x
            is_cycle = False
            while contractable[cur]:
                if cur == x:
                    is_cycle = True
                    break
                left_nodes.append(cur)
                pair_cur = nbr_pair[cur]
                assert pair_cur is not None
                na, nb = pair_cur
                nxt = na if nb == prev else nb
                prev, cur = cur, nxt
            if is_cycle:
                contractable[x] = False
                for z in left_nodes:
                    contractable[z] = False
                continue
            end_a = cur
            right_nodes: list[int] = []
            cur, prev = b0, x
            while contractable[cur]:
                right_nodes.append(cur)
                pair_cur = nbr_pair[cur]
                assert pair_cur is not None
                na, nb = pair_cur
                nxt = na if nb == prev else nb
                prev, cur = cur, nxt
            end_b = cur
            if end_a == end_b:
                contractable[x] = False
                for z in left_nodes:
                    contractable[z] = False
                for z in right_nodes:
                    contractable[z] = False
                continue

            interior = tuple(reversed(left_nodes)) + (x,) + tuple(right_nodes)
            for z in interior:
                visited_c[z] = True
            full_seq = (end_a,) + interior + (end_b,)

            def _compose_chain(seq: tuple[int, ...]) -> tuple[float, float, list[float], list[float]] | None:
                d_sum = 0.0
                ff_t_sum = 0.0
                sp_ref = inv_ref = None
                for idx in range(len(seq) - 1):
                    u_i, v_i = seq[idx], seq[idx + 1]
                    edge_match = next((row for row in adj[u_i] if row[0] == v_i), None)
                    if edge_match is None:
                        return None
                    _, d_e, ff_e, sp_e, inv_e = edge_match
                    if sp_ref is None:
                        sp_ref, inv_ref = sp_e, inv_e
                    elif sp_ref is not sp_e and sp_ref != sp_e:
                        return None
                    d_sum += d_e
                    ff_t_sum += d_e / max(ff_e, 1e-12)
                if sp_ref is None or inv_ref is None:
                    return None
                return (d_sum, ff_t_sum, sp_ref, inv_ref)

            fwd_edge = _compose_chain(full_seq)
            rev_edge = _compose_chain(full_seq[::-1])
            if fwd_edge is None and rev_edge is None:
                for z in interior:
                    contractable[z] = False
                continue
            cid = len(chains)
            for z in interior:
                node_to_chain[z] = cid
            chains.append((end_a, end_b, interior, fwd_edge, rev_edge))

        orig_to_core = [-1] * num_v
        core_nodes: list[int] = []
        for u in range(num_v):
            if not peeled[u] and node_to_chain[u] < 0:
                orig_to_core[u] = len(core_nodes)
                core_nodes.append(u)

        core_v = len(core_nodes)
        core_adj: list[list[tuple[int, float, float, list[float], list[float]]]] = [
            [] for _ in range(core_v)
        ]
        for cu, u in enumerate(core_nodes):
            for v, dist, ff_sp, sp, inv_sp in adj[u]:
                cv = orig_to_core[v]
                if cv >= 0:
                    core_adj[cu].append((cv, dist, dist / max(ff_sp, 1e-12), sp, inv_sp))

        chain_shortcuts: list[tuple[int, int, Any, Any]] = []
        for ea, eb, _, fwd_e, rev_e in chains:
            cea, ceb = orig_to_core[ea], orig_to_core[eb]
            f_tup = (ceb, fwd_e[0], fwd_e[1], fwd_e[2], fwd_e[3]) if fwd_e is not None else None
            r_tup = (cea, rev_e[0], rev_e[1], rev_e[2], rev_e[3]) if rev_e is not None else None
            if f_tup is not None:
                core_adj[cea].append(f_tup)
            if r_tup is not None:
                core_adj[ceb].append(r_tup)
            chain_shortcuts.append((cea, ceb, f_tup, r_tup))

        return core_v, orig_to_core, core_adj, peeled, peel_parent, node_to_chain, chains, chain_shortcuts

    def _apply_compact_data(self, data: _CompactGraphData) -> None:
        self._shared_graph = data
        self._node_ids = data.node_ids
        self._node_to_idx = data.node_to_idx
        self._num_v = data.num_v
        self._cust_v = [self._node_to_idx[nid] for nid in self.instance.road_node_ids]
        self._target_v_set = set(self._cust_v)
        self._bin_edges = data.bin_edges
        self._num_bins = data.num_bins
        self._cycle_start = data.cycle_start
        self._cycle_end = data.cycle_end
        self._cycle = data.cycle
        self._uniform_bin_width = data.uniform_bin_width
        self._adj = data.adj
        self._compact_edges = data.compact_edges
        self._all_transitions = data.transitions
        self._worst_slowdown = data.worst_slowdown
        self._bin_max_alpha = data.bin_max_alpha
        self._pinned_adj = data.pinned_adj
        self._directed_edge_data = data.directed_edge_data
        self._edge_atfs: dict[int, NDCPWLF] = {}
        self._lookback = 1800.0

    def _instantiate_run_graph(self) -> None:
        data = self._shared_graph
        if data.core_adj is None or data.orig_to_core is None:
            self._run_adj = [
                [(v, d, d / max(ff, 1e-12), sp, inv_sp) for v, d, ff, sp, inv_sp in row]
                for row in self._adj
            ]
            self._orig_to_run = list(range(self._num_v))
            self._run_num_v = self._num_v
            self._run_targets = set(self._cust_v)
            return

        assert (
            data.peeled is not None
            and data.peel_parent is not None
            and data.node_to_chain is not None
            and data.chains is not None
            and data.chain_shortcuts is not None
        )
        peeled = data.peeled
        peel_parent = data.peel_parent
        node_to_chain = data.node_to_chain
        chains = data.chains
        chain_shortcuts = data.chain_shortcuts

        run_adj = [row[:] for row in data.core_adj]
        v_map = data.orig_to_core[:]
        core_v = data.core_v
        adj = self._adj
        extra_count = 0

        def _ensure_node(orig_u: int) -> int:
            nonlocal extra_count
            cu = v_map[orig_u]
            if cu >= 0:
                return cu
            cu = core_v + extra_count
            extra_count += 1
            v_map[orig_u] = cu
            run_adj.append([])
            return cu

        broken_chains: set[int] = set()
        restored_peeled: set[int] = set()
        pending_stems: list[list[int]] = []

        for tv in self._cust_v:
            if peeled[tv]:
                cur = tv
                stem: list[int] = []
                while peeled[cur]:
                    if cur in restored_peeled:
                        break
                    stem.append(cur)
                    nxt = peel_parent[cur]
                    if nxt < 0:
                        break
                    cur = nxt
                if cur >= 0 and node_to_chain[cur] >= 0:
                    broken_chains.add(node_to_chain[cur])
                for z in stem:
                    restored_peeled.add(z)
                    _ensure_node(z)
                if stem:
                    pending_stems.append(stem)
            elif node_to_chain[tv] >= 0:
                broken_chains.add(node_to_chain[tv])

        for cid in broken_chains:
            ea, eb, interior, _, _ = chains[cid]
            cea, ceb, f_tup, r_tup = chain_shortcuts[cid]
            if f_tup is not None:
                run_adj[cea] = [e for e in run_adj[cea] if e is not f_tup]
            if r_tup is not None:
                run_adj[ceb] = [e for e in run_adj[ceb] if e is not r_tup]
            for z in interior:
                _ensure_node(z)
            full_seq = (ea,) + interior + (eb,)
            for i in range(len(full_seq) - 1):
                u_i, v_i = full_seq[i], full_seq[i + 1]
                cu, cv = v_map[u_i], v_map[v_i]
                for vv, d, ff, sp, inv_sp in adj[u_i]:
                    if vv == v_i:
                        run_adj[cu].append((cv, d, d / max(ff, 1e-12), sp, inv_sp))
                for vv, d, ff, sp, inv_sp in adj[v_i]:
                    if vv == u_i:
                        run_adj[cv].append((cu, d, d / max(ff, 1e-12), sp, inv_sp))

        for stem in pending_stems:
            for z in stem:
                cz = v_map[z]
                for v, d, ff, sp, inv_sp in adj[z]:
                    if v_map[v] >= 0:
                        run_adj[cz].append((v_map[v], d, d / max(ff, 1e-12), sp, inv_sp))
                w = peel_parent[z]
                if w >= 0 and w not in restored_peeled and v_map[w] >= 0:
                    cw = v_map[w]
                    for vv, d, ff, sp, inv_sp in adj[w]:
                        if vv == z:
                            run_adj[cw].append((cz, d, d / max(ff, 1e-12), sp, inv_sp))

        self._run_adj = run_adj
        self._orig_to_run = v_map
        self._run_num_v = core_v + extra_count
        self._run_targets = {v_map[v] for v in self._cust_v}

    def _ensure_pinned_structures(self) -> None:
        if self._pinned_adj is not None and self._directed_edge_data is not None:
            return
        pinned_adj: list[list[tuple[int, float, float, int]]] = [[] for _ in range(self._num_v)]
        directed_edge_data: list[tuple[int, int, float, list[float]]] = []
        for u, v, dist, ff_speed, speeds_mps, _, bidir in self._compact_edges:
            ff_cost = dist / max(ff_speed, 1e-12)
            d_idx = len(directed_edge_data)
            directed_edge_data.append((u, v, dist, speeds_mps))
            pinned_adj[u].append((v, dist, ff_cost, d_idx))
            if bidir:
                d_idx_rev = len(directed_edge_data)
                directed_edge_data.append((v, u, dist, speeds_mps))
                pinned_adj[v].append((u, dist, ff_cost, d_idx_rev))
        self._pinned_adj = pinned_adj
        self._directed_edge_data = directed_edge_data
        self._shared_graph.pinned_adj = pinned_adj
        self._shared_graph.directed_edge_data = directed_edge_data

    def _find_transitions(self) -> list[float]:
        h_start, h_end = self.horizon
        return [b for b in self._all_transitions if h_start < b < h_end]

    def _build_grid(self) -> None:
        h_start, h_end = self.horizon
        span = h_end - h_start
        step = max(60.0, min(self.step, span))
        if span / step <= 64:
            pts: list[float] = []
            cur = h_start
            while cur < h_end - 1e-9:
                pts.append(cur)
                cur += step
            pts.append(h_end)
            self._grid = pts
            return

        active_start = max(h_start, self.earliest_departure)
        active_end = min(h_end, self._active_end) if self._active_end is not None else h_end
        if active_start > active_end:
            active_start = h_start
            active_end = h_end

        transitions = self._find_transitions()
        grid_set = {h_start, active_start, active_end, h_end}
        active_transitions = [tr for tr in transitions if active_start < tr <= active_end]
        for tr in active_transitions:
            grid_set.add(tr)
            t = tr - step
            earliest_win = max(active_start, tr - self._lookback)
            while t >= earliest_win - 1e-9:
                if t > active_start:
                    grid_set.add(t)
                t -= step
            # Thêm điểm lấy mẫu giữa nếu lookback nhỏ hơn transition
            mid = tr - min(0.5 * step, 150.0)
            if active_start < mid < tr:
                grid_set.add(mid)
        self._grid = sorted(grid_set)

    def _td_dijkstra_from_vertex(self, src_v: int, t0: float) -> dict[int, float]:
        cached = self._tree_cache.get((src_v, t0))
        if cached is not None:
            return cached

        ref_t = self._reference_zero_time()
        if self._cycle > 0 and abs(t0 - ref_t) > 1e-9:
            cycles = round((t0 - ref_t) / self._cycle)
            if abs((t0 - ref_t) - cycles * self._cycle) <= 1e-9:
                base_map = self._td_dijkstra_from_vertex(src_v, ref_t)
                delta = t0 - ref_t
                shifted = {k: val + delta for k, val in base_map.items()}
                self._tree_cache[(src_v, t0)] = shifted
                return shifted

        run_src = self._orig_to_run[src_v]
        if run_src < 0:
            raise ValueError(f"Đỉnh nguồn {src_v} không nằm trong đồ thị chạy")
        num_v = self._run_num_v
        adj = self._run_adj
        bin_edges = self._bin_edges
        num_bins = self._num_bins
        cycle_start = self._cycle_start
        cycle = self._cycle
        uniform_w = self._uniform_bin_width

        best = [math.inf] * num_v
        best[run_src] = t0
        pq: list[tuple[float, int]] = [(t0, run_src)]
        visited = [False] * num_v
        targets = self._run_targets
        rem_targets = len(targets)
        seen_targets = 0
        heappop = heapq.heappop
        heappush = heapq.heappush

        while pq:
            t, u = heappop(pq)
            if visited[u]:
                continue
            visited[u] = True
            if u in targets:
                seen_targets += 1
                if seen_targets >= rem_targets:
                    break

            rel = (t - cycle_start) % cycle
            if uniform_w is not None:
                slot = int(rel // uniform_w)
                if slot >= num_bins:
                    slot = num_bins - 1
            else:
                slot = min(bisect_right(bin_edges, cycle_start + rel) - 1, num_bins - 1)
            b_end = bin_edges[slot + 1]
            cur_local = cycle_start + rel

            for v, d, _, sp, inv_sp in adj[u]:
                if visited[v]:
                    continue
                dt = d * inv_sp[slot]
                if cur_local + dt <= b_end + 1e-9:
                    arr = t + dt
                else:
                    rem = d
                    cur = t
                    while True:
                        r_loc = (cur - cycle_start) % cycle
                        if uniform_w is not None:
                            c_slot = int(r_loc // uniform_w)
                            if c_slot >= num_bins:
                                c_slot = num_bins - 1
                        else:
                            c_slot = min(
                                bisect_right(bin_edges, cycle_start + r_loc) - 1,
                                num_bins - 1,
                            )
                        s_mps = sp[c_slot]
                        to_b = bin_edges[c_slot + 1] - (cycle_start + r_loc)
                        if to_b <= 1e-12:
                            to_b = cycle
                        poss = s_mps * to_b
                        if poss + 1e-9 >= rem:
                            arr = cur + rem * inv_sp[c_slot]
                            break
                        rem -= poss
                        cur += to_b
                if arr < best[v]:
                    best[v] = arr
                    heappush(pq, (arr, v))

        orig_to_run = self._orig_to_run
        result = {tgt_v: best[orig_to_run[tgt_v]] for tgt_v in self._target_v_set}
        self._tree_cache[(src_v, t0)] = result
        return result

    def _free_flow_dijkstra_from_vertex(self, src_v: int, *, need_pred: bool = True) -> tuple[Any, list[int]]:
        heappop = heapq.heappop
        heappush = heapq.heappush

        if not need_pred:
            run_src = self._orig_to_run[src_v]
            num_v = self._run_num_v
            best = [math.inf] * num_v
            best[run_src] = 0.0
            pq: list[tuple[float, int]] = [(0.0, run_src)]
            visited = [False] * num_v
            targets = self._run_targets
            rem_targets = len(targets)
            seen_targets = 0
            adj = self._run_adj
            while pq:
                cost, u = heappop(pq)
                if visited[u]:
                    continue
                visited[u] = True
                if u in targets:
                    seen_targets += 1
                    if seen_targets >= rem_targets:
                        break
                for v, _, ff_cost, _, _ in adj[u]:
                    if visited[v]:
                        continue
                    cand = cost + ff_cost
                    if cand < best[v]:
                        best[v] = cand
                        heappush(pq, (cand, v))
            orig_to_run = self._orig_to_run
            return {tgt_v: best[orig_to_run[tgt_v]] for tgt_v in self._target_v_set}, []

        num_v = self._num_v
        best = [math.inf] * num_v
        best[src_v] = 0.0
        pq = [(0.0, src_v)]
        visited = [False] * num_v
        targets = self._target_v_set
        rem_targets = len(targets)
        seen_targets = 0
        self._ensure_pinned_structures()
        assert self._pinned_adj is not None
        p_adj = self._pinned_adj
        pred_edge = [-1] * num_v
        while pq:
            cost, u = heappop(pq)
            if visited[u]:
                continue
            visited[u] = True
            if u in targets:
                seen_targets += 1
                if seen_targets >= rem_targets:
                    break
            for v, _, ff_cost, d_idx in p_adj[u]:
                if visited[v]:
                    continue
                cand = cost + ff_cost
                if cand < best[v]:
                    best[v] = cand
                    pred_edge[v] = d_idx
                    heappush(pq, (cand, v))
        return best, pred_edge

    def _trace_fastest_path(
        self, source_node_id: int, target_node_id: int, departure: float
    ) -> tuple[list[int], list[str], float]:
        src_v = self._node_to_idx[source_node_id]
        tgt_v = self._node_to_idx[target_node_id]
        if src_v == tgt_v:
            return [source_node_id], [], 0.0

        num_v = self._num_v
        adj = self._adj
        bin_edges = self._bin_edges
        num_bins = self._num_bins
        cycle_start = self._cycle_start
        cycle = self._cycle
        uniform_w = self._uniform_bin_width

        t0 = float(departure)
        best = [math.inf] * num_v
        best[src_v] = t0
        pred_v = [-1] * num_v
        pq: list[tuple[float, int]] = [(t0, src_v)]
        visited = [False] * num_v
        heappop = heapq.heappop
        heappush = heapq.heappush

        while pq:
            t, u = heappop(pq)
            if visited[u]:
                continue
            visited[u] = True
            if u == tgt_v:
                break

            rel = (t - cycle_start) % cycle
            if uniform_w is not None:
                slot = int(rel // uniform_w)
                if slot >= num_bins:
                    slot = num_bins - 1
            else:
                slot = min(bisect_right(bin_edges, cycle_start + rel) - 1, num_bins - 1)
            b_end = bin_edges[slot + 1]
            cur_local = cycle_start + rel

            for v, d, _, sp, inv_sp in adj[u]:
                if visited[v]:
                    continue
                dt = d * inv_sp[slot]
                if cur_local + dt <= b_end + 1e-9:
                    arr = t + dt
                else:
                    rem = d
                    cur = t
                    while True:
                        r_loc = (cur - cycle_start) % cycle
                        if uniform_w is not None:
                            c_slot = int(r_loc // uniform_w)
                            if c_slot >= num_bins:
                                c_slot = num_bins - 1
                        else:
                            c_slot = min(
                                bisect_right(bin_edges, cycle_start + r_loc) - 1,
                                num_bins - 1,
                            )
                        s_mps = sp[c_slot]
                        to_b = bin_edges[c_slot + 1] - (cycle_start + r_loc)
                        if to_b <= 1e-12:
                            to_b = cycle
                        poss = s_mps * to_b
                        if poss + 1e-9 >= rem:
                            arr = cur + rem * inv_sp[c_slot]
                            break
                        rem -= poss
                        cur += to_b
                if arr < best[v]:
                    best[v] = arr
                    pred_v[v] = u
                    heappush(pq, (arr, v))

        if not math.isfinite(best[tgt_v]):
            raise ValueError(f"No route from {source_node_id} to {target_node_id}")

        idx_path = [tgt_v]
        cur_v = tgt_v
        while cur_v != src_v:
            cur_v = pred_v[cur_v]
            if cur_v < 0:
                raise ValueError(f"No route from {source_node_id} to {target_node_id}")
            idx_path.append(cur_v)
        idx_path.reverse()

        node_ids = self._node_ids
        node_path = [node_ids[v] for v in idx_path]
        adjacency = prepare_adjacency(self.graph)
        edge_path: list[str] = []
        clock = t0
        total_dist = 0.0
        for i in range(len(node_path) - 1):
            nu, nv = node_path[i], node_path[i + 1]
            cands = [edge for nb, edge in adjacency.get(nu, ()) if nb == nv]
            if len(cands) == 1:
                chosen = cands[0]
                dt = get_edge_travel_time(chosen, clock, self.traffic)
            elif cands:
                best_pair = min(
                    ((get_edge_travel_time(edge, clock, self.traffic), edge) for edge in cands),
                    key=lambda item: item[0],
                )
                dt, chosen = best_pair
            else:
                raise ValueError(f"Missing edge {nu} -> {nv}")
            edge_path.append(chosen.id)
            total_dist += float(chosen.distance_m)
            clock += dt
        return node_path, edge_path, total_dist

    def _compute_scalar_matrix(self, departure: float) -> list[list[float]]:
        matrix: list[list[float]] = []
        cust_v = self._cust_v
        cache_by_v: dict[int, dict[int, float]] = {}
        for src in range(self.num_nodes):
            sv = cust_v[src]
            if sv not in cache_by_v:
                cache_by_v[sv] = self._td_dijkstra_from_vertex(sv, departure)
            arrs = cache_by_v[sv]
            row = [
                0.0 if tgt == src or cust_v[tgt] == sv else max(0.0, arrs[cust_v[tgt]] - departure)
                for tgt in range(self.num_nodes)
            ]
            matrix.append(row)
        return matrix

    def _compute_free_flow_matrix(self) -> list[list[float]]:
        matrix: list[list[float]] = []
        cust_v = self._cust_v
        cache_by_v: dict[int, dict[int, float]] = {}
        for src in range(self.num_nodes):
            sv = cust_v[src]
            if sv not in cache_by_v:
                dist, _ = self._free_flow_dijkstra_from_vertex(sv, need_pred=False)
                cache_by_v[sv] = dist
            dist = cache_by_v[sv]
            row = [
                0.0 if tgt == src or cust_v[tgt] == sv else dist[cust_v[tgt]]
                for tgt in range(self.num_nodes)
            ]
            matrix.append(row)
        return matrix

    def _build_source_atfs_dynamic(self, source_cust: int) -> dict[int, NDCPWLF]:
        src_v = self._cust_v[source_cust]
        grid = self._grid
        trees = [self._td_dijkstra_from_vertex(src_v, t0) for t0 in grid]
        source_arcs: dict[int, NDCPWLF] = {}
        cust_v = self._cust_v
        for target_cust in range(self.num_nodes):
            if target_cust == source_cust:
                continue
            tgt_v = cust_v[target_cust]
            if tgt_v == src_v:
                source_arcs[target_cust] = NDCPWLF.identity(self.horizon[0], self.horizon[1])
                continue
            ys = [tr[tgt_v] for tr in trees]
            if any(not math.isfinite(val) for val in ys):
                raise ValueError(f"Không có đường đi từ khách {source_cust} tới {target_cust}")
            for k in range(1, len(ys)):
                if ys[k] < ys[k - 1]:
                    ys[k] = ys[k - 1]
            xs, c_ys = _drop_exactly_collinear(grid, ys)
            if self.simplify_tolerance > 0:
                xs, c_ys = _simplify_points(xs, c_ys, self.simplify_tolerance)
            source_arcs[target_cust] = NDCPWLF(list(xs), list(c_ys), validate=False)
        return source_arcs

    def _get_directed_edge_atf(self, d_idx: int) -> NDCPWLF:
        self._ensure_pinned_structures()
        assert self._directed_edge_data is not None
        atf = self._edge_atfs.get(d_idx)
        if atf is None:
            _, _, dist, speeds_mps = self._directed_edge_data[d_idx]
            zones = [(self._bin_edges[k], self._bin_edges[k + 1]) for k in range(self._num_bins)]
            ext_end = self.horizon[1] + max(3600.0, self._lookback)
            atf = build_arc_atf(zones, speeds_mps, dist, (self.horizon[0], ext_end))
            self._edge_atfs[d_idx] = atf
        return atf

    def _build_source_atfs_pinned(self, source_cust: int) -> dict[int, NDCPWLF]:
        self._ensure_pinned_structures()
        assert self._directed_edge_data is not None
        src_v = self._cust_v[source_cust]
        _, pred_edge = self._free_flow_dijkstra_from_vertex(src_v, need_pred=True)
        grid = self._grid
        arrivals: dict[int, list[float]] = {src_v: list(grid)}
        source_arcs: dict[int, NDCPWLF] = {}
        cust_v = self._cust_v
        for target_cust in range(self.num_nodes):
            if target_cust == source_cust:
                continue
            tgt_v = cust_v[target_cust]
            if tgt_v == src_v:
                source_arcs[target_cust] = NDCPWLF.identity(self.horizon[0], self.horizon[1])
                continue
            chain: list[int] = []
            vertex = tgt_v
            while vertex not in arrivals:
                d_idx = pred_edge[vertex]
                if d_idx < 0:
                    raise ValueError(f"Không có đường đi free-flow từ {source_cust} tới {target_cust}")
                chain.append(vertex)
                vertex = self._directed_edge_data[d_idx][0]
            for vertex in reversed(chain):
                d_idx = pred_edge[vertex]
                parent_v = self._directed_edge_data[d_idx][0]
                edge_atf = self._get_directed_edge_atf(d_idx)
                arrivals[vertex] = [edge_atf.evaluate(val) for val in arrivals[parent_v]]
            ys = list(arrivals[tgt_v])
            for k in range(1, len(ys)):
                if ys[k] < ys[k - 1]:
                    ys[k] = ys[k - 1]
            xs, c_ys = _drop_exactly_collinear(grid, ys)
            if self.simplify_tolerance > 0:
                xs, c_ys = _simplify_points(xs, c_ys, self.simplify_tolerance)
            source_arcs[target_cust] = NDCPWLF(list(xs), list(c_ys), validate=False)
        return source_arcs

    def get_atf(self, source_cust: int, target_cust: int) -> NDCPWLF:
        if self._external_atfs is not None:
            return self._external_atfs[(source_cust, target_cust)]
        source_arcs = self._atfs_by_source.get(source_cust)
        if source_arcs is None:
            if self.routing_policy == "pinned_free_flow_paths":
                source_arcs = self._build_source_atfs_pinned(source_cust)
            else:
                source_arcs = self._build_source_atfs_dynamic(source_cust)
            self._atfs_by_source[source_cust] = source_arcs
        return source_arcs[target_cust]

    def has_arc(self, source_cust: int, target_cust: int) -> bool:
        if self._external_atfs is not None:
            return (source_cust, target_cust) in self._external_atfs
        return source_cust != target_cust and 0 <= source_cust < self.num_nodes and 0 <= target_cust < self.num_nodes

    def travel_at_zero(self, source_cust: int, target_cust: int) -> float:
        if source_cust == target_cust:
            return 0.0
        if self.t0_matrix is not None:
            return self.t0_matrix[source_cust][target_cust]
        atf = self.get_atf(source_cust, target_cust)
        dep = self._reference_zero_time()
        return float(atf.evaluate(dep)) - dep

    def free_flow(self, source_cust: int, target_cust: int) -> float:
        if source_cust == target_cust:
            return 0.0
        if self.ff_matrix is not None:
            return self.ff_matrix[source_cust][target_cust]
        return float(self.get_atf(source_cust, target_cust).min_shifted_image()[0])


class _HCMCExactSplitEvaluator:
    """Split tối ưu và định giá thời gian xuất phát chính xác bằng NDCPWLF trên HCMC."""

    def __init__(
        self,
        instance: TDVRPInstance,
        graph: tuple[dict[int, RoadNode], dict[str, RoadEdge]],
        traffic: TrafficOverlay | Any,
        earliest_departure: float = 0.0,
        latest_departure: float | None = None,
        exact_split: bool = False,
        sample_step: float | None = None,
        simplify_tolerance: float = 1.0,
        routing_policy: str | None = None,
    ) -> None:
        self.instance = instance
        self.graph = graph
        self.traffic = traffic
        self.earliest_departure = float(earliest_departure)
        self.latest_departure = float(latest_departure) if latest_departure is not None else None
        self.exact_split = bool(exact_split)
        self.atf_provider = _HCMCATFProvider(
            instance,
            graph,
            traffic,
            sample_step=sample_step,
            simplify_tolerance=simplify_tolerance,
            routing_policy=routing_policy,
            earliest_departure=self.earliest_departure,
            latest_departure=self.latest_departure,
        )
        self.routing_policy = self.atf_provider.routing_policy
        self.route_cache: dict[tuple[int, ...], _RouteState | None] = {}
        self.duration_cache: dict[tuple[int, ...], tuple[float, float] | None] = {}
        self.free_flow_cache: dict[tuple[int, int], float] = {}
        self.time_zero_cache: dict[tuple[int, int], float] = {}

    def evaluate(self, permutation: Iterable[int]) -> HCMCTDERAPOCandidate:
        order = tuple(permutation)
        self._validate_permutation(order)
        routes, states, fitness = self._split(order)
        return HCMCTDERAPOCandidate(order, routes, states, fitness)

    def _validate_permutation(self, order: tuple[int, ...]) -> None:
        expected = set(range(1, self.instance.num_customers + 1))
        if len(order) != self.instance.num_customers or set(order) != expected:
            raise ValueError("Ứng viên HCMC-TD-ERAPO không phải hoán vị đầy đủ của khách hàng")

    def _contiguous_min_routes(self, order: tuple[int, ...]) -> int:
        count = 1
        load = 0.0
        cap = self.instance.vehicle_capacity + 1e-9
        demands = self.instance.demands
        for cust in order:
            d = demands[cust]
            if load > 0.0 and load + d > cap:
                count += 1
                load = 0.0
            load += d
        return count

    def _split(
        self,
        order: tuple[int, ...],
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[_RouteState, ...], float]:
        """Order-first, split-second bằng quy hoạch động có tôn trọng ``num_vehicles``."""
        if self.exact_split:
            return self._split_exact(order)
        n = len(order)
        max_vehicles = self.instance.num_vehicles
        if max_vehicles == 1:
            if sum(self.instance.demands[c] for c in order) <= self.instance.vehicle_capacity + 1e-9:
                state = self.route_state(order)
                if state is not None:
                    return (order,), (state,), state.duration
            raise ValueError("Split không thể xếp toàn bộ khách hàng lên 1 xe")

        best_proxy_cost = [math.inf] * (n + 1)
        best_route_count = [n + 1] * (n + 1)
        predecessor = [-1] * (n + 1)
        best_proxy_cost[0] = 0.0
        best_route_count[0] = 0
        cap = self.instance.vehicle_capacity + 1e-9
        demands = self.instance.demands
        service_times = self.instance.service_times
        depot = self.instance.depot

        for end in range(1, n + 1):
            load = 0.0
            proxy_route_cost = 0.0
            for start in range(end - 1, -1, -1):
                customer = order[start]
                load += demands[customer]
                if load > cap:
                    break
                if start == end - 1:
                    proxy_route_cost = (
                        service_times[customer]
                        + self.travel_at_zero(depot, customer)
                        + self.travel_at_zero(customer, depot)
                    )
                else:
                    next_customer = order[start + 1]
                    proxy_route_cost += (
                        service_times[customer]
                        + self.travel_at_zero(depot, customer)
                        + self.travel_at_zero(customer, next_customer)
                        - self.travel_at_zero(depot, next_customer)
                    )
                if not math.isfinite(best_proxy_cost[start]):
                    continue
                candidate_cost = best_proxy_cost[start] + proxy_route_cost
                candidate_routes = best_route_count[start] + 1
                key = (candidate_cost, candidate_routes, start)
                current = (best_proxy_cost[end], best_route_count[end], predecessor[end])
                if key < current:
                    best_proxy_cost[end] = candidate_cost
                    best_route_count[end] = candidate_routes
                    predecessor[end] = start

        if predecessor[n] < 0:
            raise ValueError("Split Decoder không tìm được cách phân tuyến khả thi")

        routes = self._reconstruct_routes(order, predecessor)
        if len(routes) > max_vehicles:
            routes = self._enforce_vehicle_limit(order, max_vehicles, exact=False)

        states_list = [self.route_state(route) for route in routes]
        if any(state is None for state in states_list):
            raise ValueError("Split Decoder tạo ra tuyến không khả thi theo thời gian")
        states = tuple(state for state in states_list if state is not None)
        return routes, states, sum(state.duration for state in states)

    def _split_exact(
        self,
        order: tuple[int, ...],
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[_RouteState, ...], float]:
        """Minimum-duration capacity-feasible Exact Split tôn trọng ``num_vehicles``.

        Duy trì cộng dồn NDCPWLF tiến dần từ ``start`` để mỗi đoạn con mới chỉ
        thực hiện đúng một phép hợp thành ``compose`` thay vì tính lại từ đầu.
        """
        n = len(order)
        max_vehicles = self.instance.num_vehicles
        if max_vehicles == 1:
            if sum(self.instance.demands[c] for c in order) <= self.instance.vehicle_capacity + 1e-9:
                state = self.route_state(order)
                if state is not None:
                    return (order,), (state,), state.duration
            raise ValueError("Exact Split không thể xếp toàn bộ khách hàng lên 1 xe")

        horizon_start, horizon_end = self.instance.horizon
        start_h = max(float(horizon_start), self.earliest_departure)
        if self.latest_departure is not None:
            end_h = min(float(horizon_end), self.latest_departure)
        else:
            end_h = float(horizon_end)
        end_h = float(end_h)
        if start_h > end_h:
            raise ValueError("earliest_departure phải nhỏ hơn hoặc bằng latest_departure (và cuối horizon)")

        cap = self.instance.vehicle_capacity + 1e-9
        demands = self.instance.demands
        service_times = self.instance.service_times
        depot = self.instance.depot
        depot_window = self._time_window(depot)
        depot_latest = min(float(horizon_end), depot_window[1] if depot_window is not None else float(horizon_end))
        get_atf = self.atf_provider.get_atf
        dur_cache = self.duration_cache

        seg_cost: list[list[tuple[int, float]]] = [[] for _ in range(n + 1)]
        for start in range(n):
            load = 0.0
            acc: NDCPWLF | None = None
            acc_pos = start
            prev = depot
            prefix_failed = False

            for end in range(start + 1, n + 1):
                c_end = order[end - 1]
                load += demands[c_end]
                if load > cap:
                    break
                sub = order[start:end]
                if sub in dur_cache:
                    cached_dur = dur_cache[sub]
                    if cached_dur is None:
                        break
                    seg_cost[end].append((start, cached_dur[0]))
                    continue

                while acc_pos < end:
                    c_step = order[acc_pos]
                    arc = get_atf(prev, c_step)
                    acc = (
                        restrict_domain(arc, start_h, end_h)
                        if acc is None
                        else arc.compose(acc, slope_one_exact=True)
                    )
                    if acc.is_empty():
                        dur_cache[order[start : acc_pos + 1]] = None
                        prefix_failed = True
                        break
                    win = self._time_window(c_step)
                    acc = apply_ready_time(
                        acc,
                        earliest=win[0] if win is not None else None,
                        latest=win[1] if win is not None else None,
                        service_time=float(service_times[c_step]),
                    )
                    if acc.is_empty():
                        dur_cache[order[start : acc_pos + 1]] = None
                        prefix_failed = True
                        break
                    prev = c_step
                    acc_pos += 1

                if prefix_failed:
                    break

                assert acc is not None
                ret_acc = get_atf(prev, depot).compose(acc, slope_one_exact=True)
                if ret_acc.is_empty():
                    dur_cache[sub] = None
                    break
                ret_acc = apply_ready_time(
                    ret_acc,
                    earliest=None,
                    latest=depot_latest,
                    service_time=0.0,
                )
                if ret_acc.is_empty():
                    dur_cache[sub] = None
                    break
                dur, dep = ret_acc.min_shifted_image()
                dur_val = float(dur)
                dur_cache[sub] = (dur_val, float(dep))
                seg_cost[end].append((start, dur_val))

        costs = [0.0] + [math.inf] * n
        predecessor = [-1] * (n + 1)
        for end in range(1, n + 1):
            best_c = math.inf
            best_s = -1
            for start, sc in seg_cost[end]:
                if not math.isfinite(costs[start]):
                    continue
                cand = costs[start] + sc
                if cand < best_c:
                    best_c = cand
                    best_s = start
            costs[end] = best_c
            predecessor[end] = best_s

        if predecessor[n] < 0:
            raise ValueError("Exact Split không tìm được cách phân tuyến khả thi")

        routes = tuple(sorted(self._reconstruct_routes(order, predecessor), key=lambda route: route[0]))
        if len(routes) > max_vehicles:
            routes = self._enforce_vehicle_limit(
                order, max_vehicles, exact=True, seg_cost=seg_cost
            )

        states_list = [self.route_state(route) for route in routes]
        if any(state is None for state in states_list):
            raise ValueError("Exact Split tạo ra tuyến không khả thi theo thời gian")
        states = tuple(state for state in states_list if state is not None)
        return routes, states, sum(state.duration for state in states)

    def _reconstruct_routes(
        self, order: tuple[int, ...], predecessor: list[int]
    ) -> tuple[tuple[int, ...], ...]:
        routes_reversed: list[tuple[int, ...]] = []
        cursor = len(order)
        while cursor > 0:
            start = predecessor[cursor]
            if start < 0:
                raise RuntimeError("Split Decoder mất trạng thái truy vết")
            routes_reversed.append(order[start:cursor])
            cursor = start
        return tuple(reversed(routes_reversed))

    def _enforce_vehicle_limit(
        self,
        order: tuple[int, ...],
        max_vehicles: int,
        *,
        exact: bool,
        seg_cost: list[list[tuple[int, float]]] | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        """Ràng buộc số tuyến tối đa về ``max_vehicles`` bằng DP hữu hạn xe hoặc First-Fit."""
        if self._contiguous_min_routes(order) <= max_vehicles:
            dp_routes = self._bounded_split_dp(
                order, max_vehicles, exact=exact, seg_cost=seg_cost
            )
            if dp_routes is not None:
                return dp_routes
        repaired = repair_solution(Solution([[self.instance.depot, *order, self.instance.depot]]), self.instance)
        return tuple(tuple(route[1:-1]) for route in repaired.routes if len(route) > 2)

    def _bounded_split_dp(
        self,
        order: tuple[int, ...],
        max_vehicles: int,
        *,
        exact: bool,
        seg_cost: list[list[tuple[int, float]]] | None = None,
    ) -> tuple[tuple[int, ...], ...] | None:
        n = len(order)
        if seg_cost is None:
            cap = self.instance.vehicle_capacity + 1e-9
            demands = self.instance.demands
            service_times = self.instance.service_times
            depot = self.instance.depot

            seg_cost = [[] for _ in range(n + 1)]
            for end in range(1, n + 1):
                load = 0.0
                proxy_cost = 0.0
                for start in range(end - 1, -1, -1):
                    c = order[start]
                    load += demands[c]
                    if load > cap:
                        break
                    if exact:
                        st = self.route_state(order[start:end])
                        if st is None:
                            continue
                        cost = st.duration
                    else:
                        if start == end - 1:
                            proxy_cost = (
                                service_times[c]
                                + self.travel_at_zero(depot, c)
                                + self.travel_at_zero(c, depot)
                            )
                        else:
                            nxt = order[start + 1]
                            proxy_cost += (
                                service_times[c]
                                + self.travel_at_zero(depot, c)
                                + self.travel_at_zero(c, nxt)
                                - self.travel_at_zero(depot, nxt)
                            )
                        cost = proxy_cost
                    seg_cost[end].append((start, cost))

        prev_layer = [math.inf] * (n + 1)
        prev_layer[0] = 0.0
        best_cost = math.inf
        best_k = -1
        preds: list[list[int]] = []

        for k in range(1, max_vehicles + 1):
            cur_layer = [math.inf] * (n + 1)
            pred = [-1] * (n + 1)
            for end in range(1, n + 1):
                best_e = math.inf
                best_s = -1
                for start, sc in seg_cost[end]:
                    val = prev_layer[start] + sc
                    if val < best_e:
                        best_e = val
                        best_s = start
                cur_layer[end] = best_e
                pred[end] = best_s
            preds.append(pred)
            if cur_layer[n] < best_cost:
                best_cost = cur_layer[n]
                best_k = k
            prev_layer = cur_layer

        if not math.isfinite(best_cost) or best_k < 1:
            return None

        routes_rev: list[tuple[int, ...]] = []
        cursor = n
        for layer in range(best_k - 1, -1, -1):
            start = preds[layer][cursor]
            if start < 0:
                return None
            routes_rev.append(order[start:cursor])
            cursor = start
        routes = tuple(reversed(routes_rev))
        if exact:
            routes = tuple(sorted(routes, key=lambda r: r[0]))
        return routes

    def route_state(self, customers: tuple[int, ...]) -> _RouteState | None:
        cached = self.route_cache.get(customers)
        if customers in self.route_cache:
            return cached
        if customers in self.duration_cache:
            dur_dep = self.duration_cache[customers]
            state = (
                None
                if dur_dep is None
                else self._schedule(customers, dur_dep[1], dur_dep[0])
            )
            self.route_cache[customers] = state
            return state
        state = self._compose_and_schedule(customers)
        self.route_cache[customers] = state
        return state

    def _compose_and_schedule(self, customers: tuple[int, ...]) -> _RouteState | None:
        if not customers:
            return None
        start, end = self.instance.horizon
        start = max(float(start), self.earliest_departure)
        if self.latest_departure is not None:
            end = min(float(end), self.latest_departure)
        end = float(end)
        if start > end:
            raise ValueError("earliest_departure phải nhỏ hơn hoặc bằng latest_departure (và cuối horizon)")
        depot = self.instance.depot
        accumulator: NDCPWLF | None = None
        previous = depot

        for customer in customers:
            arc = self.atf_provider.get_atf(previous, customer)
            accumulator = (
                restrict_domain(arc, start, end)
                if accumulator is None
                else arc.compose(accumulator, slope_one_exact=True)
            )
            if accumulator.is_empty():
                return None
            window = self._time_window(customer)
            accumulator = apply_ready_time(
                accumulator,
                earliest=window[0] if window is not None else None,
                latest=window[1] if window is not None else None,
                service_time=float(self.instance.service_times[customer]),
            )
            if accumulator.is_empty():
                return None
            previous = customer

        return_arc = self.atf_provider.get_atf(previous, depot)
        accumulator = return_arc.compose(accumulator, slope_one_exact=True)
        if accumulator.is_empty():
            return None
        horizon_end = float(self.instance.horizon[1])
        depot_window = self._time_window(depot)
        depot_latest = depot_window[1] if depot_window is not None else horizon_end
        accumulator = apply_ready_time(
            accumulator,
            earliest=None,
            latest=min(horizon_end, depot_latest),
            service_time=0.0,
        )
        if accumulator.is_empty():
            return None

        duration, departure = accumulator.min_shifted_image()
        return self._schedule(customers, float(departure), float(duration))

    def _schedule(
        self,
        customers: tuple[int, ...],
        departure: float,
        exact_duration: float,
    ) -> _RouteState:
        depot = self.instance.depot
        clock = departure
        load = travel = service = waiting = 0.0
        legs: list[_LegState] = []
        stops: list[_StopState] = [
            _StopState(depot, departure, departure, departure, 0.0, 0.0, 0.0)
        ]
        previous = depot

        for target in (*customers, depot):
            arrival = self.arrival_at(previous, target, clock)
            leg_travel = arrival - clock
            legs.append(_LegState(previous, target, clock, arrival, leg_travel))
            travel += leg_travel
            service_start = arrival
            stop_waiting = stop_service = 0.0
            if target != depot:
                window = self._time_window(target)
                if window is not None:
                    service_start = max(arrival, window[0])
                stop_waiting = service_start - arrival
                stop_service = float(self.instance.service_times[target])
                load += self.instance.demands[target]
                waiting += stop_waiting
                service += stop_service
            clock = service_start + stop_service
            stops.append(
                _StopState(
                    target,
                    arrival,
                    service_start,
                    clock,
                    stop_waiting,
                    stop_service,
                    load,
                )
            )
            previous = target

        return _RouteState(
            customers,
            departure,
            departure + exact_duration,
            exact_duration,
            travel,
            service,
            waiting,
            load,
            tuple(legs),
            tuple(stops),
        )

    def _time_window(self, customer: int) -> tuple[float, float] | None:
        if self.instance.time_windows is None:
            return None
        return self.instance.time_windows[customer]

    def travel_at_zero(self, source: int, target: int) -> float:
        if source == target:
            return 0.0
        key = (source, target)
        cached = self.time_zero_cache.get(key)
        if cached is None:
            cached = self.atf_provider.travel_at_zero(source, target)
            self.time_zero_cache[key] = cached
        return cached

    def arrival_at(self, source: int, target: int, departure: float) -> float:
        """Đánh giá ATF và khử sai số ulp rất nhỏ tại biên miền xác định."""
        if source == target:
            return departure
        atf = self.atf_provider.get_atf(source, target)
        value = float(departure)
        low, high = float(atf.xs[0]), float(atf.xs[-1])
        tolerance = 1e-9 * max(1.0, abs(low), abs(high))
        if low - tolerance <= value < low:
            value = low
        elif high < value <= high + tolerance:
            value = high
        return float(atf.evaluate(value))

    def free_flow(self, source: int, target: int) -> float:
        if source == target:
            return 0.0
        key = (source, target)
        cached = self.free_flow_cache.get(key)
        if cached is None:
            cached = self.atf_provider.free_flow(source, target)
            self.free_flow_cache[key] = cached
        return cached

    def materialize(
        self, candidate: HCMCTDERAPOCandidate
    ) -> tuple[Solution, HCMCEvaluationResult]:
        """Tạo nghiệm và route trace đầy đủ trên mạng lưới HCMC."""
        depot = self.instance.depot
        solution_routes = [[depot, *route, depot] for route in candidate.routes]
        while len(solution_routes) < self.instance.num_vehicles:
            solution_routes.append([depot, depot])
        solution = Solution(solution_routes)
        path_evaluator: HCMCTDVRPEvaluator | None = None
        use_fast_trace = (
            self.atf_provider._external_atfs is None
            and self.routing_policy != "pinned_free_flow_paths"
        )
        if not use_fast_trace:
            path_evaluator = HCMCTDVRPEvaluator(
                self.instance,
                self.graph,
                self.traffic,
                start_time=self.earliest_departure,
            )
        route_evaluations: list[RouteEvaluation] = []

        for vehicle_index, state in enumerate(candidate.route_states):
            node_path: list[int] = [self.instance.road_node_ids[depot]]
            edge_path: list[str] = []
            leg_traces: list[RouteLegTrace] = []
            total_distance = 0.0
            for leg in state.legs:
                source_node = self.instance.road_node_ids[leg.source]
                target_node = self.instance.road_node_ids[leg.target]
                try:
                    if use_fast_trace:
                        leg_nodes, leg_edges, distance = self.atf_provider._trace_fastest_path(
                            source_node, target_node, leg.departure
                        )
                    else:
                        assert path_evaluator is not None
                        path = path_evaluator._find_path(source_node, target_node, leg.departure)
                        leg_nodes = list(path.node_path)
                        leg_edges = list(path.edge_path)
                        distance = sum(self.graph[1][edge_id].distance_m for edge_id in leg_edges)
                except (KeyError, ValueError):
                    leg_nodes = [source_node, target_node]
                    leg_edges = []
                    x1, y1 = self.instance.coordinates[leg.source]
                    x2, y2 = self.instance.coordinates[leg.target]
                    distance = math.hypot(x1 - x2, y1 - y2)
                node_path.extend(leg_nodes[1:])
                edge_path.extend(leg_edges)
                total_distance += distance
                leg_traces.append(
                    RouteLegTrace(
                        leg.source,
                        leg.target,
                        leg.departure,
                        leg.arrival,
                        leg.travel_time,
                        distance,
                        leg_nodes,
                        leg_edges,
                    )
                )

            stop_traces = [
                RouteStopTrace(
                    stop.customer,
                    self.instance.road_node_ids[stop.customer],
                    stop.arrival,
                    stop.service_start,
                    stop.departure,
                    stop.waiting,
                    stop.service,
                    self.instance.demands[stop.customer],
                    stop.load,
                )
                for stop in state.stops
            ]
            route_evaluations.append(
                RouteEvaluation(
                    vehicle_index,
                    list(state.customers),
                    node_path,
                    edge_path,
                    state.departure,
                    state.end_time,
                    state.travel_time,
                    state.service_time,
                    total_distance,
                    state.load,
                    state.waiting_time,
                    True,
                    [],
                    leg_traces,
                    stop_traces,
                )
            )

        # Đảm bảo đủ số route theo hợp đồng Solution HCMC nếu có xe không sử dụng
        depot_node = self.instance.road_node_ids[depot]
        dep_t = max(float(self.instance.horizon[0]), self.earliest_departure)
        while len(route_evaluations) < self.instance.num_vehicles:
            v_idx = len(route_evaluations)
            empty_leg = RouteLegTrace(depot, depot, dep_t, dep_t, 0.0, 0.0, [depot_node], [])
            empty_stop = RouteStopTrace(depot, depot_node, dep_t, dep_t, dep_t, 0.0, 0.0, 0.0, 0.0)
            route_evaluations.append(
                RouteEvaluation(
                    v_idx,
                    [],
                    [depot_node],
                    [],
                    dep_t,
                    dep_t,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    True,
                    [],
                    [empty_leg],
                    [empty_stop, empty_stop],
                )
            )

        total_travel = sum(route.travel_time for route in route_evaluations)
        total_service = sum(route.service_time for route in route_evaluations)
        total_waiting = sum(route.waiting_time for route in route_evaluations)
        total_distance = sum(route.distance for route in route_evaluations)
        evaluation = HCMCEvaluationResult(
            total_travel,
            total_service,
            total_distance,
            candidate.fitness,
            candidate.fitness,
            True,
            [],
            route_evaluations,
            total_waiting,
            "exact_ndcpwlf",
        )
        return solution, evaluation


class _NFELedger:
    """Mọi lần gọi evaluate đều tiêu thụ đúng một NFE, kể cả ứng viên lỗi."""

    def __init__(self, evaluator: _HCMCExactSplitEvaluator, budget: int) -> None:
        self.evaluator = evaluator
        self.budget = budget
        self.used = 0
        self.by_branch = {"initialization": 0, "ARO": 0, "POA": 0, "Elite": 0}

    def evaluate(
        self, permutation: Iterable[int], branch: str
    ) -> HCMCTDERAPOCandidate | None:
        if self.used >= self.budget:
            return None
        self.used += 1
        self.by_branch[branch] += 1
        try:
            return self.evaluator.evaluate(permutation)
        except (KeyError, ValueError):
            return None


def order_crossover(
    first_parent: tuple[int, ...],
    second_parent: tuple[int, ...],
    rng: random.Random,
) -> tuple[int, ...]:
    """Order Crossover: giữ một đoạn của cha thứ nhất, điền theo cha thứ hai."""
    n = len(first_parent)
    if n < 2:
        return first_parent
    left, right = sorted(rng.sample(range(n), 2))
    child: list[int | None] = [None] * n
    child[left : right + 1] = first_parent[left : right + 1]
    preserved = set(first_parent[left : right + 1])
    rotated_second = second_parent[right + 1 :] + second_parent[: right + 1]
    remaining = [value for value in rotated_second if value not in preserved]
    positions = list(range(right + 1, n)) + list(range(0, left))
    for position, value in zip(positions, remaining):
        child[position] = value
    result = tuple(int(value) for value in child if value is not None)
    if result == first_parent:
        mutable = list(result)
        i, j = rng.sample(range(n), 2)
        mutable[i], mutable[j] = mutable[j], mutable[i]
        result = tuple(mutable)
    return result


def compute_criticality(
    candidate: HCMCTDERAPOCandidate,
    evaluator: _HCMCExactSplitEvaluator,
) -> dict[int, float]:
    """Tính D, K, W đúng trên lịch chạy của nghiệm đã Split."""
    detour: dict[int, float] = {}
    congestion: dict[int, float] = {}
    route_weight: dict[int, float] = {}

    for route in candidate.route_states:
        for index, customer in enumerate(route.customers):
            incoming = route.legs[index]
            outgoing = route.legs[index + 1]
            direct = 0.0
            if incoming.source != outgoing.target and evaluator.atf_provider.has_arc(
                incoming.source, outgoing.target
            ):
                direct_arrival = evaluator.arrival_at(
                    incoming.source,
                    outgoing.target,
                    incoming.departure,
                )
                direct = direct_arrival - incoming.departure
            detour[customer] = max(0.0, incoming.travel_time + outgoing.travel_time - direct)

            free_in = evaluator.free_flow(incoming.source, customer)
            free_out = evaluator.free_flow(customer, outgoing.target)
            eta_in = max(0.0, (incoming.travel_time - free_in) / max(1e-6, free_in))
            eta_out = max(0.0, (outgoing.travel_time - free_out) / max(1e-6, free_out))
            congestion[customer] = 0.5 * (eta_in + eta_out)
            route_weight[customer] = (
                evaluator.instance.service_times[customer]
                + incoming.travel_time
                + outgoing.travel_time
            ) / max(1e-6, route.duration)

    normalized = [_minmax(values) for values in (detour, congestion, route_weight)]
    return {
        customer: sum(values.get(customer, 0.5) for values in normalized) / 3.0
        for customer in candidate.permutation
    }


def _minmax(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if high - low <= 1e-12:
        return {key: 0.5 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def _weighted_sample_without_replacement(
    population: tuple[int, ...],
    criticality: dict[int, float],
    count: int,
    floor: float,
    rng: random.Random,
) -> list[int]:
    remaining = list(population)
    selected: list[int] = []
    for _ in range(min(count, len(remaining))):
        weights = [criticality.get(customer, 0.0) + floor for customer in remaining]
        threshold = rng.random() * sum(weights)
        cumulative = 0.0
        chosen_index = len(remaining) - 1
        for index, weight in enumerate(weights):
            cumulative += weight
            if threshold <= cumulative:
                chosen_index = index
                break
        selected.append(remaining.pop(chosen_index))
    return selected


def regret2_reconstruct(
    partial: list[int],
    removed: list[int],
    evaluator: _HCMCExactSplitEvaluator,
) -> tuple[int, ...]:
    """Chèn lại toàn cục theo regret-2 và chi phí cung tại thời điểm 0."""
    pending = removed[:]
    t0_mat = evaluator.atf_provider.t0_matrix
    service_times = evaluator.instance.service_times
    depot = evaluator.instance.depot

    while pending:
        choice: tuple[float, float, int, int] | None = None
        len_p = len(partial)
        if t0_mat is not None:
            for customer in pending:
                s_c = service_times[customer]
                row_c = t0_mat[customer]
                options: list[tuple[float, int]] = []
                before = depot
                for position in range(len_p + 1):
                    after = partial[position] if position < len_p else depot
                    delta = s_c + t0_mat[before][customer] + row_c[after] - t0_mat[before][after]
                    options.append((delta, position))
                    before = after
                options.sort(key=lambda item: (item[0], item[1]))
                regret = options[1][0] - options[0][0] if len(options) > 1 else 0.0
                candidate_choice = (regret, -options[0][0], -customer, options[0][1])
                if choice is None or candidate_choice > choice:
                    choice = candidate_choice
        else:
            for customer in pending:
                options = []
                for position in range(len_p + 1):
                    before = partial[position - 1] if position else depot
                    after = partial[position] if position < len_p else depot
                    delta = (
                        service_times[customer]
                        + evaluator.travel_at_zero(before, customer)
                        + evaluator.travel_at_zero(customer, after)
                        - evaluator.travel_at_zero(before, after)
                    )
                    options.append((delta, position))
                options.sort(key=lambda item: (item[0], item[1]))
                regret = options[1][0] - options[0][0] if len(options) > 1 else 0.0
                candidate_choice = (regret, -options[0][0], -customer, options[0][1])
                if choice is None or candidate_choice > choice:
                    choice = candidate_choice
        if choice is None:
            raise RuntimeError("Regret-2 không chọn được khách hàng để chèn")
        customer = -choice[2]
        partial.insert(choice[3], customer)
        pending.remove(customer)
    return tuple(partial)


def poa_restructure(
    parent: HCMCTDERAPOCandidate,
    evaluator: _HCMCExactSplitEvaluator,
    config: HCMCTDERAPOConfig,
    rng: random.Random,
) -> tuple[int, ...]:
    criticality = compute_criticality(parent, evaluator)
    n = len(parent.permutation)
    remove_count = max(1, min(n - 1, math.ceil(config.destroy_fraction * n))) if n > 1 else 1
    removed = _weighted_sample_without_replacement(
        parent.permutation,
        criticality,
        remove_count,
        config.criticality_floor,
        rng,
    )
    removed_set = set(removed)
    partial = [customer for customer in parent.permutation if customer not in removed_set]
    return regret2_reconstruct(partial, removed, evaluator)


def _elite_probabilities(scores: dict[str, float], epsilon: float) -> list[float]:
    nonnegative = [max(0.0, scores[name]) for name in ELITE_OPERATORS]
    total = sum(nonnegative)
    base = [value / total for value in nonnegative] if total > 0 else [0.25] * 4
    multiplier = 1.0 - len(ELITE_OPERATORS) * epsilon
    return [multiplier * probability + epsilon for probability in base]


def _elite_move(
    permutation: tuple[int, ...],
    operator: str,
    rng: random.Random,
) -> tuple[int, ...]:
    result = list(permutation)
    if len(result) < 2:
        return tuple(result)
    if operator == "swap":
        left, right = rng.sample(range(len(result)), 2)
        result[left], result[right] = result[right], result[left]
    elif operator == "relocate":
        source, target = rng.sample(range(len(result)), 2)
        value = result.pop(source)
        result.insert(target, value)
    else:
        left, right = sorted(rng.sample(range(len(result)), 2))
        result[left : right + 1] = reversed(result[left : right + 1])
    return tuple(result)


def _best_index(population: list[HCMCTDERAPOCandidate]) -> int:
    return min(range(len(population)), key=lambda index: population[index].fitness)


def optimize(
    instance: TDVRPInstance,
    graph: tuple[dict[int, RoadNode], dict[str, RoadEdge]],
    traffic: TrafficOverlay | Any,
    config: HCMCTDERAPOConfig | None = None,
    seed: int | None = None,
    initial_population: Iterable[Iterable[int]] | None = None,
    start_time: float | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
    **_: object,
) -> HCMCTDERAPOResult:
    """Chạy TD-ERAPO trên tập dữ liệu HCMC với bộ điều phối ARO/POA/Elite lấy mẫu đều."""
    if config is None:
        config = HCMCTDERAPOConfig(
            earliest_departure=0.0 if start_time is None else float(start_time)
        )
    elif start_time is not None and config.earliest_departure == 0.0:
        config = replace(config, earliest_departure=float(start_time))
    config.validate()
    rng = random.Random(config.seed if seed is None else seed)
    started = perf_counter()

    evaluator = _HCMCExactSplitEvaluator(
        instance,
        graph,
        traffic,
        earliest_departure=config.earliest_departure,
        latest_departure=config.latest_departure,
        exact_split=config.exact_split,
        sample_step=config.sample_step,
        simplify_tolerance=config.simplify_tolerance,
        routing_policy=config.routing_policy,
    )
    ledger = _NFELedger(evaluator, config.max_function_evaluations)

    population: list[HCMCTDERAPOCandidate] = []
    archive: HCMCTDERAPOCandidate | None = None
    convergence: list[float] = []
    base = list(range(1, instance.num_customers + 1))
    if initial_population is None:
        initial_orders = [
            tuple(rng.sample(base, len(base))) for _ in range(config.population_size)
        ]
        initialization_policy = "independent_random"
    else:
        initial_orders = [
            tuple(int(customer) for customer in order) for order in initial_population
        ]
        if len(initial_orders) != config.population_size:
            raise ValueError("Quần thể P0 phải có đúng population_size hoán vị")
        for order in initial_orders:
            evaluator._validate_permutation(order)
        initialization_policy = "provided_common_population"

    for permutation in initial_orders:
        candidate = ledger.evaluate(permutation, "initialization")
        if candidate is None:
            raise RuntimeError("Không thể khởi tạo quần thể HCMC-TD-ERAPO khả thi")
        population.append(candidate)
        if archive is None or candidate.fitness < archive.fitness:
            archive = candidate
        convergence.append(archive.fitness)
        if progress_callback is not None:
            progress_callback(ledger.used, ledger.budget, archive.fitness)

    assert archive is not None
    scores = {name: 1.0 for name in ELITE_OPERATORS}
    uses = {name: 0 for name in ELITE_OPERATORS}
    successes = {name: 0 for name in ELITE_OPERATORS}
    elite_quota = math.floor(config.elite_budget_fraction * config.max_function_evaluations)

    while ledger.used < ledger.budget:
        available = ["ARO", "POA"]
        if ledger.by_branch["Elite"] < elite_quota:
            available.append("Elite")
        branch = rng.choice(available)

        if branch == "ARO":
            parent_index = rng.randrange(len(population))
            partner_index = rng.choice(
                [index for index in range(len(population)) if index != parent_index]
            )
            parent = population[parent_index]
            child_order = order_crossover(
                parent.permutation, population[partner_index].permutation, rng
            )
            child = ledger.evaluate(child_order, "ARO")
            if child is not None and child.fitness < parent.fitness:
                population[parent_index] = child
                if child.fitness < archive.fitness:
                    archive = child

        elif branch == "POA":
            parent_index = rng.randrange(len(population))
            parent = population[parent_index]
            child_order = poa_restructure(parent, evaluator, config, rng)
            child = ledger.evaluate(child_order, "POA")
            if child is not None and child.fitness < parent.fitness:
                population[parent_index] = child
                if child.fitness < archive.fitness:
                    archive = child

        else:
            parent_index = _best_index(population)
            parent = population[parent_index]
            probabilities = _elite_probabilities(scores, config.aos_epsilon)
            operator = rng.choices(ELITE_OPERATORS, weights=probabilities, k=1)[0]
            uses[operator] += 1
            child_order = _elite_move(parent.permutation, operator, rng)
            child = ledger.evaluate(child_order, "Elite")
            reward = 0.0
            if child is not None:
                relative = max(
                    0.0,
                    (parent.fitness - child.fitness) / max(parent.fitness, 1e-6),
                )
                if child.fitness < archive.fitness:
                    reward = 2.0 * relative
                elif child.fitness < parent.fitness:
                    reward = relative
                if child.fitness < parent.fitness:
                    population[parent_index] = child
                    successes[operator] += 1
                if child.fitness < archive.fitness:
                    archive = child
            scores[operator] = (
                (1.0 - config.aos_alpha) * scores[operator]
                + config.aos_alpha * reward
            )

        convergence.append(archive.fitness)
        if progress_callback is not None:
            progress_callback(ledger.used, ledger.budget, archive.fitness)

    solution, evaluation = evaluator.materialize(archive)
    return HCMCTDERAPOResult(
        solution,
        evaluation,
        archive.permutation,
        convergence,
        perf_counter() - started,
        ledger.used,
        dict(ledger.by_branch),
        scores,
        uses,
        successes,
        {
            "variant": "HCMC-TD-ERAPO",
            "dataset": instance.instance_name,
            "benchmark": instance.benchmark_name,
            "objective": "sum of exact optimal route durations",
            "split_decoder": (
                "exact minimum-duration order-first split-second"
                if config.exact_split
                else "time-zero proxy order-first split-second"
            ),
            "departure_time_mode": "exact_ndcpwlf",
            "atf_source": evaluator.atf_provider.atf_source,
            "routing_policy": evaluator.routing_policy,
            "fleet_policy": f"bounded_max_{instance.num_vehicles}_vehicles",
            "initialization_policy": initialization_policy,
            "elite_quota": elite_quota,
            "route_count": len(archive.routes),
            "strict_nfe": ledger.used == ledger.budget,
        },
    )


def load_hcmc_dataset(
    instance_path: str | Path = DEFAULT_HCMC_INSTANCE,
) -> tuple[TDVRPInstance, tuple[dict[int, RoadNode], dict[str, RoadEdge]], TrafficOverlay]:
    """Nạp một instance từ ``data/hcmc/instances`` cùng đồ thị đường và giao thông."""
    instance = load_instance(instance_path, validate_sidecars=True)
    graph = load_road_graph(instance.graph_path)
    traffic = load_traffic(instance.traffic_path, graph[1])
    return instance, graph, traffic


def optimize_hcmc_dataset(
    instance_path: str | Path = DEFAULT_HCMC_INSTANCE,
    config: HCMCTDERAPOConfig | None = None,
    seed: int | None = None,
) -> HCMCTDERAPOResult:
    """Nạp bộ dữ liệu HCMC và chạy thẳng thuật toán HCMC-TD-ERAPO."""
    instance, graph, traffic = load_hcmc_dataset(instance_path)
    return optimize(instance, graph, traffic, config=config, seed=seed)


class HCMCTDERAPO:
    """Adapter hướng đối tượng đồng nhất với các thuật toán còn lại."""

    name = "HCMC-TD-ERAPO"

    def optimize(self, instance, graph, traffic, config=None, seed=None, **kwargs):
        return optimize(instance, graph, traffic, config=config, seed=seed, **kwargs)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Chạy HCMC-TD-ERAPO (theo cơ chế rapo_2709) trên bộ dữ liệu HCMC."
    )
    parser.add_argument(
        "--instance",
        type=Path,
        default=DEFAULT_HCMC_INSTANCE,
        help="Đường dẫn tới file .vrp.json trong data/hcmc/instances",
    )
    parser.add_argument("--pop", type=int, default=10, help="Kích thước quần thể")
    parser.add_argument("--nfe", type=int, default=100, help="Ngân sách đánh giá NFE")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên")
    parser.add_argument(
        "--earliest-departure",
        type=float,
        default=0.0,
        help="Giờ xuất phát sớm nhất (giây tính từ 00:00)",
    )
    parser.add_argument(
        "--exact-split",
        action="store_true",
        help="Dùng Exact Split bằng NDCPWLF đầy đủ trong quá trình tách tuyến",
    )
    args = parser.parse_args(argv)

    cfg = HCMCTDERAPOConfig(
        population_size=args.pop,
        max_function_evaluations=args.nfe,
        earliest_departure=args.earliest_departure,
        exact_split=args.exact_split,
        seed=args.seed,
    )
    result = optimize_hcmc_dataset(args.instance, config=cfg, seed=args.seed)
    summary = {
        "algorithm": "HCMC-TD-ERAPO",
        "instance": result.metadata["dataset"],
        "nfe": result.evaluations,
        "runtime_seconds": round(result.runtime_seconds, 4),
        "fitness": round(result.best_evaluation.fitness, 4),
        "total_travel_time": round(result.best_evaluation.total_travel_time, 4),
        "total_service_time": round(result.best_evaluation.total_service_time, 4),
        "route_count": result.metadata["route_count"],
        "nfe_by_branch": result.nfe_by_branch,
        "routes": [route.customer_ids for route in result.best_evaluation.routes if route.customer_ids],
        "departures": [round(route.start_time, 2) for route in result.best_evaluation.routes if route.customer_ids],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
