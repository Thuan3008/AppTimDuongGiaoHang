"""ExperimentRunner nhiều seed, thống kê và kiểm tra fairness."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from algorithms.hcmc_aro_goc import AROConfig, run_aro
from algorithms.hcmc_poa_goc import POAConfig, run_poa
from algorithms.hcmc_rapo import RAPOConfig, run_rapo
from algorithms.poryos_baseline_adapter import BASELINE_ALGORITHMS, run_baseline
from algorithms.rapo_2709 import RAPO2709Config, optimize as optimize_rapo_2709
from algorithms.td_cr_rapo_v3 import TDCRRAPOV3Config, optimize as optimize_v3
from core.data.load_instance import load_instance
from core.data.load_poryos import find_poryos_bks, load_poryos, load_poryos_bks
from core.data.load_road_graph import load_road_graph
from core.data.load_traffic import load_traffic


PRIMARY_ALGORITHM = "RAPO-2709"
HCMC_ABLATION_ALGORITHMS = (
    "RAPO-no-adaptive",
    "RAPO-no-elite",
    "RAPO-no-restart",
)
SUPPORTED_ALGORITHMS = (
    PRIMARY_ALGORITHM,
    *BASELINE_ALGORITHMS,
    "TD-CR-RAPO-V3",
    "RAPO",
    "ARO-goc",
    "POA-goc",
    *HCMC_ABLATION_ALGORITHMS,
)


@dataclass(frozen=True)
class ExperimentConfig:
    instance_path: Path
    seeds: tuple[int, ...] = tuple(range(10))
    population_size: int = 10
    nfe_budget: int = 290
    algorithms: tuple[str, ...] = (PRIMARY_ALGORITHM,)
    start_time: float = 0.0
    departure_time_mode: str = "optimize"
    departure_time_step: float = 900.0
    allow_proxy: bool = False
    verify_checksums: bool = True
    bks_path: Path | None = None
    checkpoint_path: Path | None = None
    resume: bool = True

    def validate(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Danh sách seed phải khác nhau và không được rỗng")
        if self.population_size < 2 or self.nfe_budget < self.population_size:
            raise ValueError("population_size >= 2 và nfe_budget >= population_size")
        unknown = sorted(set(self.algorithms) - set(SUPPORTED_ALGORITHMS))
        if unknown:
            raise ValueError(f"Thuật toán chưa được hỗ trợ: {unknown}")
        exact_suite = {PRIMARY_ALGORITHM, *BASELINE_ALGORITHMS}
        if PRIMARY_ALGORITHM in self.algorithms and any(
            algorithm not in exact_suite for algorithm in self.algorithms
        ):
            raise ValueError(
                "Không trộn RAPO-2709 với evaluator lưới trong runner tổng quát. "
                "Hãy dùng compare_td_cr_rapo_v3_vs_rapo_2709.py để so sánh paired "
                "trên cùng P0 và exact NDCPWLF."
            )
        if self.departure_time_mode not in {"fixed", "optimize"}:
            raise ValueError("departure_time_mode phải là fixed hoặc optimize")
        if self.departure_time_step <= 0:
            raise ValueError("departure_time_step phải dương")


@dataclass
class ExperimentReport:
    protocol: dict[str, Any]
    runs: list[dict[str, Any]]
    summary: list[dict[str, Any]]
    best_route_traces: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "runs": self.runs,
            "summary": self.summary,
            "best_route_traces": self.best_route_traces,
        }

    def save(self, output_json: str | Path) -> tuple[Path, Path, Path]:
        """Xuất JSON đầy đủ, CSV từng lần chạy và CSV thống kê."""
        json_path = Path(output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        runs_path = json_path.with_name(json_path.stem + "_runs.csv")
        summary_path = json_path.with_name(json_path.stem + "_summary.csv")
        _write_csv(runs_path, self.runs)
        _write_csv(summary_path, self.summary)
        return json_path, runs_path, summary_path


class ExperimentRunner:
    """Chạy mọi thuật toán dưới một protocol duy nhất và kiểm tra hậu nghiệm."""

    def __init__(
        self,
        config: ExperimentConfig,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        config.validate()
        self.config = config
        self.progress_callback = progress_callback

    def run(self) -> ExperimentReport:
        config = self.config
        instance, graph, traffic = _load_experiment_bundle(config)
        is_poryos = instance.benchmark_name == "Poryos2026"
        bks_path = config.bks_path or (find_poryos_bks(config.instance_path) if is_poryos else None)
        # BKS chính thức không được dùng để tính gap cho benchmark proxy.
        bks = (
            load_poryos_bks(bks_path)
            if bks_path is not None and bks_path.exists() and not instance.metadata.get("proxy_graph")
            else None
        )
        if bks is not None and bks.get("instance_name") != instance.instance_name:
            raise ValueError("BKS không thuộc instance đang chạy")
        bks_cost = float(bks["cost"]) if bks is not None else None
        protocol = self._protocol(instance, traffic, bks_path if bks is not None else None, bks_cost)
        runs, checkpoint_traces = _load_checkpoint(
            config.checkpoint_path,
            protocol,
            resume=config.resume,
        )
        best_fitness: dict[str, float] = {}
        best_traces: dict[str, dict[str, Any]] = {}
        for saved_row in runs:
            algorithm = str(saved_row["algorithm"])
            fitness = saved_row.get("fitness")
            if saved_row.get("status") == "completed" and fitness is not None:
                if float(fitness) < best_fitness.get(algorithm, math.inf):
                    best_fitness[algorithm] = float(fitness)
                    if algorithm in checkpoint_traces:
                        best_traces[algorithm] = checkpoint_traces[algorithm]
        exact_suite = {PRIMARY_ALGORITHM, *BASELINE_ALGORITHMS}
        total_runs = len(config.seeds) * len(config.algorithms)
        completed_keys = {
            (str(row["algorithm"]), int(row["seed"]))
            for row in runs
            if row.get("status") == "completed"
        }
        completed_runs = len(completed_keys)

        for seed in config.seeds:
            for algorithm in config.algorithms:
                if (algorithm, seed) in completed_keys:
                    continue
                common = self._common_run_fields(protocol, algorithm, seed)
                try:
                    initial_population = (
                        _common_initial_population(instance.num_customers, config.population_size, seed)
                        if algorithm in exact_suite
                        else None
                    )
                    result = self._run_one(
                        algorithm,
                        seed,
                        instance,
                        graph,
                        traffic,
                        initial_population,
                    )
                    nfe = int(getattr(result, "nfe", getattr(result, "evaluations", 0)))
                    evaluation = result.best_evaluation
                    row = {
                        **common,
                        "status": "completed",
                        "fitness": float(evaluation.fitness),
                        "feasible": bool(evaluation.feasible),
                        "runtime_seconds": float(result.runtime_seconds),
                        "nfe": nfe,
                        "gap_to_bks_percent": _gap(float(evaluation.fitness), bks_cost),
                        "solution_routes": result.best_solution.routes,
                        "route_departure_times": [route.start_time for route in evaluation.routes],
                        "convergence": list(getattr(result, "convergence", [])),
                        "error_type": None,
                        "error_message": None,
                    }
                    if algorithm == "TD-CR-RAPO-V3":
                        row["architecture_status"] = result.metadata.get("architecture_status")
                        row["nfe_spending"] = result.spending
                    elif algorithm == "RAPO-2709":
                        row["architecture_status"] = "PRIMARY"
                        row["nfe_spending"] = result.nfe_by_branch
                    elif algorithm in BASELINE_ALGORITHMS:
                        row["architecture_status"] = "BASELINE"
                        row["nfe_spending"] = result.metadata.get("nfe_by_phase")
                    runs = _replace_run(runs, row)
                    trace = _serialize_trace(evaluation)
                    _append_checkpoint(config.checkpoint_path, row, trace)
                    completed_runs += 1
                    self._notify_progress(completed_runs, total_runs, row)
                    if evaluation.fitness < best_fitness.get(algorithm, math.inf):
                        best_fitness[algorithm] = float(evaluation.fitness)
                        best_traces[algorithm] = trace
                except Exception as error:  # Lỗi từng run được ghi lại, không làm mất các seed còn lại.
                    row = {
                        **common,
                        "status": "error",
                        "fitness": None,
                        "feasible": False,
                        "runtime_seconds": None,
                        "nfe": None,
                        "gap_to_bks_percent": None,
                        "solution_routes": None,
                        "route_departure_times": None,
                        "convergence": [],
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                    runs = _replace_run(runs, row)
                    _append_checkpoint(config.checkpoint_path, row, None)
                    completed_runs += 1
                    self._notify_progress(completed_runs, total_runs, row)

        self._validate_fairness(runs)
        summary = summarize_runs(runs, bks_cost)
        return ExperimentReport(protocol, runs, summary, best_traces)

    def _notify_progress(self, completed: int, total: int, row: dict[str, Any]) -> None:
        """Gửi tiến độ tùy chọn cho CLI; không ảnh hưởng logic benchmark."""
        if self.progress_callback is not None:
            self.progress_callback(
                {
                    "completed": completed,
                    "total": total,
                    "algorithm": row.get("algorithm"),
                    "seed": row.get("seed"),
                    "status": row.get("status"),
                    "fitness": row.get("fitness"),
                    "nfe": row.get("nfe"),
                    "runtime_seconds": row.get("runtime_seconds"),
                    "error_message": row.get("error_message"),
                }
            )

    def _run_one(self, algorithm: str, seed: int, instance, graph, traffic, initial_population=None):
        config = self.config
        departure = {
            "departure_time_mode": config.departure_time_mode,
            "departure_time_step": config.departure_time_step,
        }
        if algorithm == "ARO-goc":
            return run_aro(
                instance,
                graph,
                traffic,
                start_time=config.start_time,
                config=AROConfig(config.population_size, 10_000, seed, config.nfe_budget),
                **departure,
            )
        if algorithm == "POA-goc":
            return run_poa(
                instance,
                graph,
                traffic,
                start_time=config.start_time,
                config=POAConfig(config.population_size, 10_000, seed, config.nfe_budget),
                **departure,
            )
        if algorithm == "RAPO" or algorithm in HCMC_ABLATION_ALGORITHMS:
            ablation = {
                "RAPO-no-adaptive": {"adaptive_family_selection": False},
                "RAPO-no-elite": {"enable_elite_refinement": False},
                "RAPO-no-restart": {"enable_restarts": False},
            }.get(algorithm, {})
            return run_rapo(
                instance,
                graph,
                traffic,
                RAPOConfig(
                    population_size=config.population_size,
                    max_function_evaluations=config.nfe_budget,
                    max_iterations=10_000,
                    variant=algorithm,
                    **ablation,
                ),
                seed=seed,
                start_time=config.start_time,
                **departure,
            )
        if algorithm == "RAPO-2709":
            return optimize_rapo_2709(
                instance,
                graph,
                traffic,
                RAPO2709Config(
                    population_size=config.population_size,
                    max_function_evaluations=config.nfe_budget,
                    exact_split=True,
                ),
                seed=seed,
                initial_population=initial_population,
            )
        if algorithm in BASELINE_ALGORITHMS:
            return run_baseline(
                algorithm,
                instance,
                graph,
                traffic,
                instance_path=config.instance_path,
                population_size=config.population_size,
                nfe_budget=config.nfe_budget,
                seed=seed,
                initial_population=initial_population,
            )
        return optimize_v3(
            instance,
            graph,
            traffic,
            TDCRRAPOV3Config(
                population_size=config.population_size,
                max_function_evaluations=config.nfe_budget,
            ),
            seed=seed,
            start_time=config.start_time,
            **departure,
        )

    def _protocol(self, instance, traffic, bks_path: Path | None, bks_cost: float | None) -> dict[str, Any]:
        exact_suite = {PRIMARY_ALGORITHM, *BASELINE_ALGORITHMS}
        exact_evaluator_suite = all(algorithm in exact_suite for algorithm in self.config.algorithms)
        return {
            "instance_name": instance.instance_name,
            "instance_path": str(self.config.instance_path),
            "benchmark_name": instance.benchmark_name,
            "data_source": instance.metadata.get("data_source"),
            "proxy_graph": bool(instance.metadata.get("proxy_graph", False)),
            "graph_path": instance.graph_path,
            "traffic_path": instance.traffic_path,
            "traffic_model": traffic.traffic_model,
            "traffic_profile": traffic.profile,
            "objective": "total_route_duration = total_travel_time + total_service_time",
            "population_size": self.config.population_size,
            "nfe_budget": self.config.nfe_budget,
            "seeds": list(self.config.seeds),
            "algorithms": list(self.config.algorithms),
            "initialization_policy": (
                "paired_common_population_per_seed"
                if exact_evaluator_suite
                else "algorithm-native; use paired comparison runner for head-to-head claims"
            ),
            "stopping_condition": "strict NFE budget",
            "start_time": self.config.start_time,
            "departure_time_mode": "exact_ndcpwlf" if exact_evaluator_suite else self.config.departure_time_mode,
            "departure_time_step": None if exact_evaluator_suite else self.config.departure_time_step,
            "bks_path": str(bks_path) if bks_path is not None else None,
            "bks_cost": bks_cost,
            "atf_sha256_verified": bool(instance.metadata.get("atf_sha256_verified", False)),
        }

    def _common_run_fields(self, protocol: dict[str, Any], algorithm: str, seed: int) -> dict[str, Any]:
        keys = (
            "instance_name", "data_source", "traffic_model", "traffic_profile", "objective",
            "population_size", "nfe_budget", "initialization_policy", "stopping_condition",
            "start_time", "departure_time_mode", "departure_time_step",
        )
        return {"algorithm": algorithm, "seed": seed, **{key: protocol[key] for key in keys}}

    def _validate_fairness(self, runs: list[dict[str, Any]]) -> None:
        """Chặn báo cáo nếu một run hoàn tất nhưng vi phạm protocol chung."""
        comparison_keys = (
            "instance_name", "data_source", "traffic_model", "traffic_profile", "objective",
            "population_size", "nfe_budget", "initialization_policy", "stopping_condition",
            "start_time", "departure_time_mode", "departure_time_step",
        )
        completed = [row for row in runs if row["status"] == "completed"]
        if completed:
            reference = tuple(completed[0][key] for key in comparison_keys)
            if any(tuple(row[key] for key in comparison_keys) != reference for row in completed[1:]):
                raise RuntimeError("Fairness guard: các run không dùng cùng benchmark protocol")
        for row in completed:
            if row["nfe"] != self.config.nfe_budget:
                raise RuntimeError(
                    f"Strict NFE guard: {row['algorithm']} seed={row['seed']} dùng {row['nfe']}, "
                    f"yêu cầu {self.config.nfe_budget}"
                )
        expected_seeds = set(self.config.seeds)
        for algorithm in self.config.algorithms:
            actual = {row["seed"] for row in runs if row["algorithm"] == algorithm}
            if actual != expected_seeds:
                raise RuntimeError(f"Seed-set guard thất bại cho {algorithm}: {sorted(actual)}")


def summarize_runs(runs: list[dict[str, Any]], bks_cost: float | None = None) -> list[dict[str, Any]]:
    """Tính best, mean, median, standard deviation, runtime và feasible rate."""
    summary: list[dict[str, Any]] = []
    algorithms = list(dict.fromkeys(row["algorithm"] for row in runs))
    for algorithm in algorithms:
        group = [row for row in runs if row["algorithm"] == algorithm]
        successful = [row for row in group if row["status"] == "completed" and row["fitness"] is not None]
        fitnesses = [float(row["fitness"]) for row in successful]
        runtimes = [float(row["runtime_seconds"]) for row in successful if row["runtime_seconds"] is not None]
        feasible_count = sum(bool(row["feasible"]) for row in successful)
        summary.append(
            {
                "algorithm": algorithm,
                "runs_requested": len(group),
                "runs_completed": len(successful),
                "runs_failed": len(group) - len(successful),
                "feasible_rate": feasible_count / len(group) if group else 0.0,
                "best": min(fitnesses) if fitnesses else None,
                "mean": statistics.fmean(fitnesses) if fitnesses else None,
                "median": statistics.median(fitnesses) if fitnesses else None,
                "standard_deviation": statistics.stdev(fitnesses) if len(fitnesses) > 1 else 0.0 if fitnesses else None,
                "runtime_mean_seconds": statistics.fmean(runtimes) if runtimes else None,
                "runtime_median_seconds": statistics.median(runtimes) if runtimes else None,
                "runtime_standard_deviation": statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0 if runtimes else None,
                "best_gap_to_bks_percent": _gap(min(fitnesses), bks_cost) if fitnesses else None,
                "mean_gap_to_bks_percent": _gap(statistics.fmean(fitnesses), bks_cost) if fitnesses else None,
            }
        )
    return summary


def _common_initial_population(
    dimension: int,
    population_size: int,
    seed: int,
) -> list[tuple[int, ...]]:
    """Sinh P0 chung theo seed để các baseline exact dùng cùng điểm xuất phát."""
    rng = random.Random(seed)
    customers = list(range(1, dimension + 1))
    return [tuple(rng.sample(customers, dimension)) for _ in range(population_size)]


def _run_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["algorithm"]), int(row["seed"])


def _replace_run(rows: list[dict[str, Any]], row: dict[str, Any]) -> list[dict[str, Any]]:
    """Thay bản ghi lỗi cũ bằng kết quả mới khi resume và retry."""
    key = _run_key(row)
    return [existing for existing in rows if _run_key(existing) != key] + [row]


def _load_checkpoint(
    path: Path | None,
    protocol: dict[str, Any],
    *,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Đọc checkpoint JSONL và kiểm tra protocol trước khi tiếp tục."""
    if path is None or not resume or not path.exists():
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            header = {"type": "header", "checkpoint_version": 1, "protocol": protocol}
            path.write_text(json.dumps(header, ensure_ascii=False) + "\n", encoding="utf-8")
        return [], {}

    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise RuntimeError(f"Checkpoint rỗng, không thể resume an toàn: {path}")
    header = json.loads(lines[0])
    if header.get("type") != "header" or header.get("protocol") != protocol:
        raise RuntimeError(
            "Checkpoint không khớp protocol hiện tại. "
            "Dùng đường dẫn checkpoint mới hoặc chạy với --no-resume."
        )
    rows: list[dict[str, Any]] = []
    traces: dict[str, dict[str, Any]] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("type") != "run":
            continue
        row = record["row"]
        rows = _replace_run(rows, row)
        if record.get("trace") is not None:
            traces[str(row["algorithm"])] = record["trace"]
    return rows, traces


def _append_checkpoint(
    path: Path | None,
    row: dict[str, Any],
    trace: dict[str, Any] | None,
) -> None:
    """Append atomically một run để Ctrl+C không làm mất các run trước đó."""
    if path is None:
        return
    record = {"type": "run", "row": row, "trace": trace}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def _gap(value: float, reference: float | None) -> float | None:
    if reference is None or reference <= 0 or not math.isfinite(value):
        return None
    return (value - reference) / reference * 100.0


def _serialize_trace(evaluation) -> dict[str, Any]:
    """Chỉ lưu trace của run tốt nhất mỗi thuật toán để tránh file kết quả phình lớn."""
    return {
        "fitness": evaluation.fitness,
        "total_travel_time": evaluation.total_travel_time,
        "total_service_time": evaluation.total_service_time,
        "total_waiting_time": evaluation.total_waiting_time,
        "total_distance": evaluation.total_distance,
        "routes": [
            {
                "vehicle_index": route.vehicle_index,
                "customer_ids": route.customer_ids,
                "start_time": route.start_time,
                "end_time": route.end_time,
                "travel_time": route.travel_time,
                "service_time": route.service_time,
                "waiting_time": route.waiting_time,
                "distance": route.distance,
                "load": route.load,
                "node_path": list(route.node_path),
                "edge_path": list(route.edge_path),
                "stops": [asdict(stop) for stop in route.stops],
                "legs": [asdict(leg) for leg in route.legs],
            }
            for route in evaluation.routes
        ],
    }


def _load_experiment_bundle(config: ExperimentConfig):
    """Nạp HCMC hoặc Poryos nhưng giữ chung evaluator và benchmark protocol."""
    payload = json.loads(config.instance_path.read_text(encoding="utf-8"))
    if payload.get("benchmark_name") == "Poryos2026":
        return load_poryos(
            config.instance_path,
            verify_checksums=config.verify_checksums,
            allow_proxy=config.allow_proxy,
            materialize_atfs=True,
        )
    instance = load_instance(config.instance_path)
    sidecar_root = config.instance_path.parent.parent / "sidecars"
    observed_graph = sidecar_root / "hcmc.road-observed.json"
    observed_traffic = sidecar_root / "hcmc.traffic-observed.json"
    graph_path = observed_graph if observed_graph.is_file() and observed_traffic.is_file() else Path(instance.graph_path)
    traffic_path = observed_traffic if graph_path == observed_graph else Path(instance.traffic_path)
    graph = load_road_graph(graph_path)
    traffic = load_traffic(traffic_path, graph[1])
    if graph_path == observed_graph:
        instance = replace(
            instance,
            graph_path=str(graph_path),
            traffic_path=str(traffic_path),
            metadata={**instance.metadata, "data_source": "hcmc-observed-hourly-with-simulated-fallback"},
        )
    return instance, graph, traffic


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value
                    for key, value in row.items()
                }
            )
