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
from core.data.models import RoadEdge, RoadNode, TDVRPInstance, TrafficOverlay
from core.data.solution import repair_solution
from core.models import Solution


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

        self._external_atfs = instance.arrival_time_functions
        self._atfs_by_source: dict[int, dict[int, NDCPWLF]] = {}

        if self._external_atfs is not None:
            self.atf_source = "instance_atfs"
            self.t0_matrix: list[list[float]] | None = None
            self.ff_matrix: list[list[float]] | None = None
            return

        self.atf_source = f"hcmc_{self.routing_policy}"
        self._init_compact_graph()
        self._build_grid()
        self.t0_matrix = self._compute_scalar_matrix(departure=self._reference_zero_time())
        self.ff_matrix = self._compute_free_flow_matrix()

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

    def _init_compact_graph(self) -> None:
        node_ids = sorted(self.nodes.keys())
        self._node_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
        self._num_v = len(node_ids)
        self._cust_v = [self._node_to_idx[nid] for nid in self.instance.road_node_ids]
        self._target_v_set = set(self._cust_v)

        bin_edges = tuple(
            float(x)
            for x in getattr(self.traffic, "bin_edges", tuple(float(h * 3600) for h in range(25)))
        )
        if len(bin_edges) < 2:
            raise ValueError("Traffic overlay cần ít nhất một bin thời gian")
        self._bin_edges = bin_edges
        self._num_bins = len(bin_edges) - 1
        self._cycle_start = bin_edges[0]
        self._cycle_end = bin_edges[-1]
        self._cycle = self._cycle_end - self._cycle_start

        widths = [bin_edges[k + 1] - bin_edges[k] for k in range(self._num_bins)]
        self._uniform_bin_width: float | None = (
            widths[0]
            if all(abs(w - widths[0]) <= 1e-9 for w in widths) and widths[0] > 0
            else None
        )

        adj: list[list[tuple[int, float, float, list[float], list[float]]]] = [
            [] for _ in range(self._num_v)
        ]
        # Lưu edge list riêng nếu dùng pinned_free_flow_paths với Edge ATF
        self._compact_edges: list[tuple[int, int, float, float, list[float], list[float]]] = []
        has_speed_for = hasattr(self.traffic, "speed_for")
        traffic_edges = getattr(self.traffic, "edges", None)

        for edge in self.edges.values():
            u = self._node_to_idx[edge.from_node]
            v = self._node_to_idx[edge.to_node]
            dist = float(edge.distance_m)
            ff_speed = float(edge.free_flow_speed_kmh) / 3.6
            if has_speed_for:
                speeds_mps = [
                    max(1e-6, float(self.traffic.speed_for(edge, s)) / 3.6)
                    for s in range(self._num_bins)
                ]
            else:
                prof = (
                    traffic_edges[edge.id]
                    if traffic_edges is not None
                    else self.traffic[edge.id]
                )
                speeds_mps = [
                    max(1e-6, float(prof.hourly_speed_kmh[f"{s:02d}"]) / 3.6)
                    for s in range(self._num_bins)
                ]
            inv_speeds = [1.0 / s for s in speeds_mps]
            entry_idx = len(self._compact_edges)
            self._compact_edges.append((u, v, dist, ff_speed, speeds_mps, inv_speeds))
            adj[u].append((v, dist, ff_speed, speeds_mps, inv_speeds))
            if edge.bidirectional:
                adj[v].append((u, dist, ff_speed, speeds_mps, inv_speeds))

        self._adj = adj
        self._pinned_adj: list[list[tuple[int, float, float, int]]] = [
            [] for _ in range(self._num_v)
        ]
        for eid_idx, (u, v, dist, ff_speed, _, _) in enumerate(self._compact_edges):
            self._pinned_adj[u].append((v, dist, dist / max(ff_speed, 1e-12), eid_idx))
            # Kiểm tra nếu đồ thị gốc là 2 chiều thông qua adj
        # Re-build _pinned_adj trực tiếp cho cả hai chiều để tránh mất cạnh ngược:
        self._pinned_adj = [[] for _ in range(self._num_v)]
        self._directed_edge_data: list[tuple[int, int, float, list[float]]] = []
        for edge in self.edges.values():
            u = self._node_to_idx[edge.from_node]
            v = self._node_to_idx[edge.to_node]
            dist = float(edge.distance_m)
            ff_cost = dist / max(float(edge.free_flow_speed_kmh) / 3.6, 1e-12)
            if has_speed_for:
                speeds_mps = [
                    max(1e-6, float(self.traffic.speed_for(edge, s)) / 3.6)
                    for s in range(self._num_bins)
                ]
            else:
                prof = (
                    traffic_edges[edge.id]
                    if traffic_edges is not None
                    else self.traffic[edge.id]
                )
                speeds_mps = [
                    max(1e-6, float(prof.hourly_speed_kmh[f"{s:02d}"]) / 3.6)
                    for s in range(self._num_bins)
                ]
            d_idx = len(self._directed_edge_data)
            self._directed_edge_data.append((u, v, dist, speeds_mps))
            self._pinned_adj[u].append((v, dist, ff_cost, d_idx))
            if edge.bidirectional:
                d_idx_rev = len(self._directed_edge_data)
                self._directed_edge_data.append((v, u, dist, speeds_mps))
                self._pinned_adj[v].append((u, dist, ff_cost, d_idx_rev))

        self._edge_atfs: dict[int, NDCPWLF] = {}
        self._lookback = 1800.0

    def _find_transitions(self) -> list[float]:
        transitions: list[float] = []
        h_start, h_end = self.horizon
        if len(self._compact_edges) > 20_000 and hasattr(self.traffic, "hourly_alpha"):
            alphas = self.traffic.hourly_alpha
            for b in range(1, self._num_bins):
                boundary = self._bin_edges[b]
                if h_start < boundary < h_end and alphas[b - 1] != alphas[b]:
                    transitions.append(boundary)
            if transitions:
                return transitions
        for b in range(1, self._num_bins):
            boundary = self._bin_edges[b]
            if not (h_start < boundary < h_end):
                continue
            changed = any(
                row[4][b - 1] != row[4][b] for row in self._compact_edges
            )
            if changed:
                transitions.append(boundary)
        return transitions

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

        transitions = self._find_transitions()
        grid_set = {h_start, h_end}
        if not transitions:
            self._grid = [h_start, h_end]
            return
        for tr in transitions:
            grid_set.add(tr)
            t = tr - step
            earliest_win = max(h_start, tr - self._lookback)
            while t >= earliest_win - 1e-9:
                if t > h_start:
                    grid_set.add(t)
                t -= step
            # Thêm điểm lấy mẫu giữa nếu lookback nhỏ hơn transition
            mid = tr - min(0.5 * step, 150.0)
            if h_start < mid < tr:
                grid_set.add(mid)
        self._grid = sorted(grid_set)

    def _td_dijkstra_from_vertex(self, src_v: int, t0: float) -> list[float]:
        num_v = self._num_v
        adj = self._adj
        bin_edges = self._bin_edges
        num_bins = self._num_bins
        cycle_start = self._cycle_start
        cycle = self._cycle
        uniform_w = self._uniform_bin_width

        best = [math.inf] * num_v
        best[src_v] = t0
        pq: list[tuple[float, int]] = [(t0, src_v)]
        visited = [False] * num_v
        targets = self._target_v_set
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
        return best

    def _free_flow_dijkstra_from_vertex(self, src_v: int) -> tuple[list[float], list[int]]:
        num_v = self._num_v
        adj = self._pinned_adj
        best = [math.inf] * num_v
        pred_edge = [-1] * num_v
        best[src_v] = 0.0
        pq: list[tuple[float, int]] = [(0.0, src_v)]
        visited = [False] * num_v
        targets = self._target_v_set
        rem_targets = len(targets)
        seen_targets = 0
        heappop = heapq.heappop
        heappush = heapq.heappush

        while pq:
            cost, u = heappop(pq)
            if visited[u]:
                continue
            visited[u] = True
            if u in targets:
                seen_targets += 1
                if seen_targets >= rem_targets:
                    break
            for v, _, ff_cost, d_idx in adj[u]:
                if visited[v]:
                    continue
                cand = cost + ff_cost
                if cand < best[v]:
                    best[v] = cand
                    pred_edge[v] = d_idx
                    heappush(pq, (cand, v))
        return best, pred_edge

    def _compute_scalar_matrix(self, departure: float) -> list[list[float]]:
        matrix: list[list[float]] = []
        cust_v = self._cust_v
        cache_by_v: dict[int, list[float]] = {}
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
        cache_by_v: dict[int, list[float]] = {}
        for src in range(self.num_nodes):
            sv = cust_v[src]
            if sv not in cache_by_v:
                dist, _ = self._free_flow_dijkstra_from_vertex(sv)
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
        atf = self._edge_atfs.get(d_idx)
        if atf is None:
            _, _, dist, speeds_mps = self._directed_edge_data[d_idx]
            zones = [(self._bin_edges[k], self._bin_edges[k + 1]) for k in range(self._num_bins)]
            ext_end = self.horizon[1] + max(3600.0, self._lookback)
            atf = build_arc_atf(zones, speeds_mps, dist, (self.horizon[0], ext_end))
            self._edge_atfs[d_idx] = atf
        return atf

    def _build_source_atfs_pinned(self, source_cust: int) -> dict[int, NDCPWLF]:
        src_v = self._cust_v[source_cust]
        _, pred_edge = self._free_flow_dijkstra_from_vertex(src_v)
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
        exact_split: bool = False,
        sample_step: float | None = None,
        simplify_tolerance: float = 1.0,
        routing_policy: str | None = None,
    ) -> None:
        self.instance = instance
        self.graph = graph
        self.traffic = traffic
        self.earliest_departure = float(earliest_departure)
        self.exact_split = bool(exact_split)
        self.atf_provider = _HCMCATFProvider(
            instance,
            graph,
            traffic,
            sample_step=sample_step,
            simplify_tolerance=simplify_tolerance,
            routing_policy=routing_policy,
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

        start_h, end_h = self.instance.horizon
        start_h = max(float(start_h), self.earliest_departure)
        end_h = float(end_h)
        if start_h >= end_h:
            raise ValueError("earliest_departure phải nhỏ hơn cuối horizon")

        cap = self.instance.vehicle_capacity + 1e-9
        demands = self.instance.demands
        service_times = self.instance.service_times
        depot = self.instance.depot
        depot_window = self._time_window(depot)
        depot_latest = min(end_h, depot_window[1] if depot_window is not None else end_h)
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
        end = float(end)
        if start >= end:
            raise ValueError("earliest_departure phải nhỏ hơn cuối horizon")
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
        depot_window = self._time_window(depot)
        depot_latest = depot_window[1] if depot_window is not None else end
        accumulator = apply_ready_time(
            accumulator,
            earliest=None,
            latest=min(end, depot_latest),
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
