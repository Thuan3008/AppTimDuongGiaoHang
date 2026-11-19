"""So sánh công bằng TD-CR-RAPO-V3 và RAPO-2709 trên Poryos2026.

Hai thuật toán dùng chung quần thể P0, Split Decoder, đội xe không giới hạn,
evaluator NDCPWLF chính xác, seed, NFE và objective. Vì vậy phần khác nhau
chỉ còn chính sách tìm kiếm của từng thuật toán.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.rapo_2709 import (  # noqa: E402
    RAPO2709Config,
    _ExactSplitEvaluator,
    optimize as optimize_rapo_2709,
)
from algorithms.td_cr_rapo_v3 import (  # noqa: E402
    Candidate as V3Candidate,
    SearchRunTDCRRAPOV3,
    TDCRRAPOV3Config,
)
from core.data.load_poryos import find_poryos_bks, load_poryos, load_poryos_bks  # noqa: E402


ALGORITHMS = ("TD-CR-RAPO-V3", "RAPO-2709")
DEFAULT_INSTANCE = (
    ROOT
    / "data/TDVRP/hong_kong/n=50/poryos-hong_kong-n50-hyb/"
    "bpr-heavy/poryos-hong_kong-n50-hyb-bpr-heavy.vrp.json"
)
DEFAULT_OUTPUT = ROOT / "experiments/td_cr_rapo_v3_vs_rapo_2709.json"
DEFAULT_CHECKPOINT = ROOT / "experiments/td_cr_rapo_v3_vs_rapo_2709.checkpoint.jsonl"
CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class FairComparisonConfig:
    instance_path: Path = DEFAULT_INSTANCE
    seeds: tuple[int, ...] = tuple(range(20))
    population_size: int = 10
    nfe_budget: int = 100
    output_path: Path = DEFAULT_OUTPUT
    verify_checksums: bool = True

    def validate(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Danh sách seed phải khác nhau và không được rỗng")
        if self.population_size < 2:
            raise ValueError("population_size phải từ 2 trở lên")
        if self.nfe_budget < self.population_size:
            raise ValueError("NFE phải đủ để đánh giá toàn bộ quần thể P0")


class _CheckpointLog:
    """Nhật ký máy đọc được, bền vững sau từng run thuật toán hoàn tất."""

    def __init__(
        self,
        path: Path,
        config: FairComparisonConfig,
        fresh: bool = False,
    ) -> None:
        self.path = path
        self.expected_header = _checkpoint_header(config)
        self.rows: list[dict[str, object]] = []
        self.warnings: list[str] = []
        path.parent.mkdir(parents=True, exist_ok=True)
        if fresh and path.exists():
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            backup = path.with_name(f"{path.name}.bak-{stamp}")
            path.replace(backup)
            self.warnings.append(f"Checkpoint cũ đã được lưu tại {backup}")
        if path.exists():
            self._load()
        else:
            self._append(self.expected_header)

    def _load(self) -> None:
        header: dict[str, object] | None = None
        seen: set[tuple[int, str]] = set()
        lines = self.path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                self.warnings.append(
                    f"Bỏ qua dòng checkpoint hỏng số {line_number}; run chưa ghi xong sẽ được chạy lại"
                )
                continue
            if record.get("type") == "header" and header is None:
                header = record
            elif record.get("type") == "run":
                row = record.get("row")
                if not isinstance(row, dict):
                    continue
                key = (int(row["seed"]), str(row["algorithm"]))
                if key not in seen:
                    seen.add(key)
                    self.rows.append(row)
        if header is None:
            raise RuntimeError(f"Checkpoint thiếu header: {self.path}")
        if header != self.expected_header:
            raise RuntimeError(
                "Checkpoint không khớp protocol hiện tại. "
                "Dùng --fresh để lưu checkpoint cũ thành bản backup và chạy lại."
            )

    def append_run(self, row: dict[str, object]) -> None:
        self.rows.append(row)
        self._append(
            {
                "type": "run",
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                "row": row,
            }
        )

    def append_event(self, event: str, **values: object) -> None:
        self._append(
            {
                "type": "event",
                "event": event,
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                **values,
            }
        )

    def _append(self, record: dict[str, object]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _checkpoint_header(config: FairComparisonConfig) -> dict[str, object]:
    return {
        "type": "header",
        "checkpoint_version": CHECKPOINT_VERSION,
        "instance_path": str(config.instance_path.resolve()),
        "algorithms": list(ALGORITHMS),
        "seeds": list(config.seeds),
        "population_size": config.population_size,
        "nfe_budget": config.nfe_budget,
        "verify_checksums": config.verify_checksums,
        "objective": "total_route_duration = total_travel_time + total_service_time",
        "evaluator": "exact_ndcpwlf",
        "split_decoder": "dynamic order-first split-second",
        "fleet_policy": "unlimited_nonempty_routes",
    }


class _SharedExactLedger:
    """Adapter để V3 dùng đúng evaluator và Split của RAPO-2709."""

    def __init__(self, evaluator: _ExactSplitEvaluator, limit: int) -> None:
        self.evaluator = evaluator
        self.limit = limit
        self.nfe = 0
        self.spending = {"initialization": 0, "aro": 0, "poa": 0, "elite": 0}
        self.recovery_nfe = 0
        self.best: V3Candidate | None = None
        self.convergence: list[float] = []

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.nfe)

    def evaluate(
        self,
        permutation: Iterable[int],
        phase: str,
        recovery: bool = False,
    ) -> V3Candidate | None:
        if not self.remaining:
            return None
        self.nfe += 1
        self.spending[phase] += 1
        if recovery:
            self.recovery_nfe += 1
        try:
            exact_candidate = self.evaluator.evaluate(permutation)
            solution, evaluation = self.evaluator.materialize(exact_candidate)
        except (KeyError, ValueError):
            if self.best is not None:
                self.convergence.append(self.best.cost)
            return None
        candidate = V3Candidate(exact_candidate.permutation, solution, evaluation)
        if self.best is None or candidate.cost < self.best.cost:
            self.best = candidate
        self.convergence.append(self.best.cost)
        return candidate


class _FairV3Run(SearchRunTDCRRAPOV3):
    """Giữ cơ chế V3 nhưng thay lớp giải mã–đánh giá bằng protocol chung."""

    def __init__(self, instance, graph, traffic, config, seed, initial_population) -> None:
        super().__init__(instance, graph, traffic, config, seed)
        self.initial_population = tuple(tuple(order) for order in initial_population)
        self.exact_evaluator = _ExactSplitEvaluator(instance, graph, traffic)
        self.ledger = _SharedExactLedger(self.exact_evaluator, config.max_function_evaluations)

    def initialize_population(self) -> None:
        for permutation in self.initial_population:
            candidate = self.ledger.evaluate(permutation, "initialization")
            if candidate is None:
                raise RuntimeError("V3 không đánh giá được quần thể P0 chung")
            self.population.append(candidate)
        if len(self.population) < 2:
            raise RuntimeError("Quần thể P0 chung phải có ít nhất hai cá thể")
        self.archive = min(self.population, key=lambda item: item.cost)
        self.update_elites()

    def run(self):
        result = super().run()
        result.metadata.update(
            {
                "fair_adapter": True,
                "departure_time_mode": "exact_ndcpwlf",
                "split_decoder": "dynamic order-first split-second",
                "fleet_policy": "unlimited_nonempty_routes",
                "initialization_policy": "provided_common_population",
                "feasible": _validate_variable_routes(result.best_solution.routes, self.instance),
            }
        )
        return result


def _common_initial_population(instance, population_size: int, seed: int) -> tuple[tuple[int, ...], ...]:
    """Sinh P0 một lần rồi truyền nguyên vẹn cho cả hai thuật toán."""
    rng = random.Random(seed)
    customers = list(range(1, instance.num_customers + 1))
    population: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    while len(population) < population_size:
        permutation = tuple(rng.sample(customers, len(customers)))
        if permutation not in seen:
            seen.add(permutation)
            population.append(permutation)
    return tuple(population)


def _population_hash(population: tuple[tuple[int, ...], ...]) -> str:
    raw = json.dumps(population, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_variable_routes(routes: list[list[int]], instance) -> bool:
    customers: list[int] = []
    for route in routes:
        if len(route) < 3 or route[0] != instance.depot or route[-1] != instance.depot:
            return False
        route_customers = route[1:-1]
        if sum(instance.demands[customer] for customer in route_customers) > instance.vehicle_capacity + 1e-9:
            return False
        customers.extend(route_customers)
    return sorted(customers) == list(range(1, instance.num_customers + 1))


def _run_v3(instance, graph, traffic, config, seed, initial_population):
    algorithm_config = TDCRRAPOV3Config(
        population_size=config.population_size,
        max_function_evaluations=config.nfe_budget,
    )
    return _FairV3Run(
        instance,
        graph,
        traffic,
        algorithm_config,
        seed,
        initial_population,
    ).run()


def _run_2709(instance, graph, traffic, config, seed, initial_population):
    return optimize_rapo_2709(
        instance,
        graph,
        traffic,
        RAPO2709Config(
            population_size=config.population_size,
            max_function_evaluations=config.nfe_budget,
        ),
        seed=seed,
        initial_population=initial_population,
    )


def _result_row(
    algorithm: str,
    result,
    seed: int,
    pair_index: int,
    execution_order: int,
    p0_hash: str,
    population_size: int,
    bks_cost: float | None,
) -> dict[str, object]:
    evaluation = result.best_evaluation
    nfe = int(getattr(result, "nfe", getattr(result, "evaluations", -1)))
    row: dict[str, object] = {
        "pair_index": pair_index,
        "execution_order": execution_order,
        "algorithm": algorithm,
        "seed": seed,
        "initial_population_sha256": p0_hash,
        "fitness": float(evaluation.fitness),
        "gap_to_bks_percent": _gap(float(evaluation.fitness), bks_cost),
        "runtime_seconds": float(result.runtime_seconds),
        "nfe": nfe,
        "feasible": bool(evaluation.feasible),
        "route_count": len(result.best_solution.routes),
        "total_travel_time": float(evaluation.total_travel_time),
        "total_service_time": float(evaluation.total_service_time),
        "total_route_duration": float(evaluation.total_route_duration),
        "departure_time_mode": evaluation.departure_time_mode,
        "split_decoder": "dynamic order-first split-second",
        "fleet_policy": "unlimited_nonempty_routes",
        "objective": "total_route_duration = total_travel_time + total_service_time",
        "population_size": population_size,
    }
    if algorithm == "TD-CR-RAPO-V3":
        row["architecture_status"] = result.metadata.get("architecture_status")
        row["nfe_spending"] = result.spending
    else:
        row["architecture_status"] = "NOT_APPLICABLE"
        row["nfe_spending"] = result.nfe_by_branch
    return row


def run_comparison(
    config: FairComparisonConfig,
    progress: Callable[[str], None] | None = None,
    checkpoint_path: Path | None = None,
    fresh: bool = False,
) -> dict[str, object]:
    config.validate()
    started = perf_counter()
    total_runs = len(config.seeds) * len(ALGORITHMS)
    checkpoint = _CheckpointLog(checkpoint_path, config, fresh=fresh) if checkpoint_path else None
    rows: list[dict[str, object]] = list(checkpoint.rows) if checkpoint else []
    completed_runs = len(rows)
    session_completed = 0

    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    if checkpoint is not None:
        for warning in checkpoint.warnings:
            report(f"[CHECKPOINT] {warning}")
        report(
            f"[CHECKPOINT] path={checkpoint.path} | đã hoàn tất "
            f"{completed_runs}/{total_runs} run"
        )

    report(
        f"[SETUP] Đang nạp instance chính thức: {config.instance_path} | "
        f"seeds={len(config.seeds)} | P={config.population_size} | NFE={config.nfe_budget}"
    )
    instance, graph, traffic = load_poryos(
        config.instance_path,
        verify_checksums=config.verify_checksums,
        allow_proxy=False,
        materialize_atfs=True,
    )
    if instance.metadata.get("proxy_graph"):
        raise RuntimeError("Fairness guard: không được dùng graph hoặc traffic proxy")
    if instance.arrival_time_functions is None:
        raise RuntimeError("Fairness guard: thiếu ATF chính thức")
    if instance.time_windows is not None:
        raise RuntimeError("Protocol này chỉ so sánh TDVRP không có time window")

    bks_path = find_poryos_bks(config.instance_path)
    bks = load_poryos_bks(bks_path) if bks_path.exists() else None
    bks_cost = float(bks["cost"]) if bks is not None else None
    report(
        f"[SETUP] Nạp dữ liệu xong | instance={instance.instance_name} | "
        f"traffic={traffic.traffic_model} | BKS={bks_cost if bks_cost is not None else 'N/A'}"
    )

    for pair_index, seed in enumerate(config.seeds):
        initial_population = _common_initial_population(instance, config.population_size, seed)
        p0_hash = _population_hash(initial_population)
        report(
            f"[PAIR {pair_index + 1}/{len(config.seeds)}] seed={seed} | "
            f"P0_SHA256={p0_hash[:12]}..."
        )
        runners = {
            "TD-CR-RAPO-V3": lambda: _run_v3(
                instance, graph, traffic, config, seed, initial_population
            ),
            "RAPO-2709": lambda: _run_2709(
                instance, graph, traffic, config, seed, initial_population
            ),
        }
        # Đảo thứ tự chạy theo cặp để giảm thiên lệch do nhiệt độ hoặc cache hệ thống.
        order = ALGORITHMS if pair_index % 2 == 0 else tuple(reversed(ALGORITHMS))
        for execution_order, algorithm in enumerate(order, 1):
            existing = next(
                (
                    row
                    for row in rows
                    if int(row["seed"]) == seed and str(row["algorithm"]) == algorithm
                ),
                None,
            )
            if existing is not None:
                if existing.get("initial_population_sha256") != p0_hash:
                    raise RuntimeError(
                        f"Checkpoint P0 không khớp tại seed={seed}, algorithm={algorithm}"
                    )
                if int(existing.get("nfe", -1)) != config.nfe_budget:
                    raise RuntimeError(
                        f"Checkpoint NFE không khớp tại seed={seed}, algorithm={algorithm}"
                    )
                report(
                    f"[RESUME] SKIP | seed={seed} | algorithm={algorithm} | "
                    f"fitness={float(existing['fitness']):.6f} | checkpoint=OK"
                )
                continue
            run_number = completed_runs + 1
            run_started = perf_counter()
            report(
                f"[RUN {run_number}/{total_runs}] START | seed={seed} | "
                f"algorithm={algorithm} | order={execution_order}/2"
            )
            result = runners[algorithm]()
            row = _result_row(
                algorithm,
                result,
                seed,
                pair_index,
                execution_order,
                p0_hash,
                config.population_size,
                bks_cost,
            )
            _guard_result(row, result, instance, config)
            if checkpoint is not None:
                checkpoint.append_run(row)
                rows = list(checkpoint.rows)
            else:
                rows.append(row)
            completed_runs += 1
            session_completed += 1
            elapsed = perf_counter() - started
            average = elapsed / session_completed
            eta = average * (total_runs - completed_runs)
            report(
                f"[RUN {completed_runs}/{total_runs}] DONE  | seed={seed} | "
                f"algorithm={algorithm} | fitness={float(row['fitness']):.6f} | "
                f"gap={_format_gap(row['gap_to_bks_percent'])} | routes={row['route_count']} | "
                f"NFE={row['nfe']}/{config.nfe_budget} | run={_format_seconds(perf_counter() - run_started)} | "
                f"total={_format_seconds(elapsed)} | ETA={_format_seconds(eta)} | "
                f"progress={100.0 * completed_runs / total_runs:.1f}%"
            )

        pair_rows = [row for row in rows if row["seed"] == seed]
        pair_costs = {str(row["algorithm"]): float(row["fitness"]) for row in pair_rows}
        pair_winner = min(pair_costs, key=pair_costs.get)
        report(
            f"[PAIR {pair_index + 1}/{len(config.seeds)}] DONE | seed={seed} | "
            f"winner={pair_winner} | delta(V3-R2709)="
            f"{pair_costs['TD-CR-RAPO-V3'] - pair_costs['RAPO-2709']:.6f}"
        )

    summary = _summarize(rows, config.seeds)
    protocol = {
        "fairness_status": "PASS",
        "instance_name": instance.instance_name,
        "instance_path": str(config.instance_path),
        "data_source": instance.metadata.get("data_source"),
        "proxy_graph": False,
        "traffic_model": traffic.traffic_model,
        "traffic_profile": traffic.profile,
        "algorithms": list(ALGORITHMS),
        "seeds": list(config.seeds),
        "population_size": config.population_size,
        "nfe_budget": config.nfe_budget,
        "initialization": "paired common P0 per seed, verified by SHA-256",
        "execution_order": "alternating by seed pair",
        "objective": "total_route_duration = total_travel_time + total_service_time",
        "evaluator": "exact global departure optimization over NDCPWLF",
        "split_decoder": "shared dynamic order-first split-second",
        "fleet_policy": "shared unlimited nonempty routes",
        "stopping_condition": "strict NFE budget",
        "bks_path": str(bks_path) if bks is not None else None,
        "bks_cost": bks_cost,
        "checksum_verified": config.verify_checksums,
    }
    _guard_pairs(rows, config)
    if checkpoint is not None:
        checkpoint.append_event(
            "completed",
            completed_runs=completed_runs,
            total_runs=total_runs,
            fairness_status="PASS",
        )
    report(
        f"[DONE] Fairness=PASS | runs={completed_runs}/{total_runs} | "
        f"elapsed={_format_seconds(perf_counter() - started)}"
    )
    return {"protocol": protocol, "runs": rows, "summary": summary}


def _format_seconds(value: float) -> str:
    seconds = max(0, round(value))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _format_gap(value: object) -> str:
    return "N/A" if value is None else f"{float(value):.4f}%"


def _guard_result(row, result, instance, config) -> None:
    if row["nfe"] != config.nfe_budget:
        raise RuntimeError(
            f"Strict NFE guard: {row['algorithm']} seed={row['seed']} dùng {row['nfe']} NFE"
        )
    if len(result.convergence) != config.nfe_budget:
        raise RuntimeError(f"Convergence guard thất bại cho {row['algorithm']} seed={row['seed']}")
    if not row["feasible"] or not _validate_variable_routes(result.best_solution.routes, instance):
        raise RuntimeError(f"Feasibility guard thất bại cho {row['algorithm']} seed={row['seed']}")
    if row["departure_time_mode"] != "exact_ndcpwlf":
        raise RuntimeError(f"Evaluator guard thất bại cho {row['algorithm']} seed={row['seed']}")
    objective_sum = float(row["total_travel_time"]) + float(row["total_service_time"])
    if not math.isclose(float(row["fitness"]), objective_sum, rel_tol=1e-9, abs_tol=1e-6):
        raise RuntimeError(f"Objective guard thất bại cho {row['algorithm']} seed={row['seed']}")
    if row["algorithm"] == "TD-CR-RAPO-V3" and row["architecture_status"] != "PASS":
        raise RuntimeError(f"Architecture guard V3 thất bại tại seed={row['seed']}")


def _guard_pairs(rows: list[dict[str, object]], config: FairComparisonConfig) -> None:
    for seed in config.seeds:
        pair = [row for row in rows if row["seed"] == seed]
        if {row["algorithm"] for row in pair} != set(ALGORITHMS):
            raise RuntimeError(f"Pairing guard: seed={seed} không có đủ hai thuật toán")
        hashes = {row["initial_population_sha256"] for row in pair}
        shared = {
            (
                row["nfe"],
                row["population_size"],
                row["departure_time_mode"],
                row["split_decoder"],
                row["fleet_policy"],
                row["objective"],
            )
            for row in pair
        }
        if len(hashes) != 1 or len(shared) != 1:
            raise RuntimeError(f"Fairness guard: protocol của cặp seed={seed} không đồng nhất")


def _summarize(rows: list[dict[str, object]], seeds: tuple[int, ...]) -> dict[str, object]:
    by_algorithm: dict[str, dict[str, object]] = {}
    for algorithm in ALGORITHMS:
        group = [row for row in rows if row["algorithm"] == algorithm]
        fitnesses = [float(row["fitness"]) for row in group]
        runtimes = [float(row["runtime_seconds"]) for row in group]
        by_algorithm[algorithm] = {
            "runs": len(group),
            "best": min(fitnesses),
            "mean": statistics.fmean(fitnesses),
            "median": statistics.median(fitnesses),
            "standard_deviation": statistics.stdev(fitnesses) if len(fitnesses) > 1 else 0.0,
            "runtime_mean_seconds": statistics.fmean(runtimes),
            "runtime_median_seconds": statistics.median(runtimes),
            "runtime_standard_deviation": statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0,
        }

    v3_values: list[float] = []
    rapo_values: list[float] = []
    wins = {"TD-CR-RAPO-V3": 0, "RAPO-2709": 0, "ties": 0}
    for seed in seeds:
        costs = {
            str(row["algorithm"]): float(row["fitness"])
            for row in rows
            if row["seed"] == seed
        }
        v3_values.append(costs["TD-CR-RAPO-V3"])
        rapo_values.append(costs["RAPO-2709"])
        if math.isclose(v3_values[-1], rapo_values[-1], rel_tol=1e-12, abs_tol=1e-9):
            wins["ties"] += 1
        elif v3_values[-1] < rapo_values[-1]:
            wins["TD-CR-RAPO-V3"] += 1
        else:
            wins["RAPO-2709"] += 1

    differences = [v3 - rapo for v3, rapo in zip(v3_values, rapo_values)]
    return {
        "by_algorithm": by_algorithm,
        "paired": {
            "wins": wins,
            "mean_fitness_difference_v3_minus_rapo2709": statistics.fmean(differences),
            "median_fitness_difference_v3_minus_rapo2709": statistics.median(differences),
            "wilcoxon_two_sided_p_value": _wilcoxon(v3_values, rapo_values),
            "interpretation": "Giá trị chênh lệch dương nghĩa là RAPO-2709 tốt hơn.",
        },
    }


def _wilcoxon(left: list[float], right: list[float]) -> float | None:
    differences = [a - b for a, b in zip(left, right) if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)]
    if not differences:
        return 1.0
    if len(differences) < 2:
        return None
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        # Tính phân phối exact bằng quy hoạch động trên tổng hạng có dấu.
        # Nhân hạng với 2 để biểu diễn chính xác các hạng trung bình khi có tie.
        ordered = sorted(enumerate(differences), key=lambda item: abs(item[1]))
        doubled_ranks = [0] * len(differences)
        cursor = 0
        while cursor < len(ordered):
            end = cursor + 1
            while end < len(ordered) and math.isclose(
                abs(ordered[end][1]), abs(ordered[cursor][1]), rel_tol=1e-12, abs_tol=1e-9
            ):
                end += 1
            doubled_average_rank = (cursor + 1) + end
            for index in range(cursor, end):
                doubled_ranks[ordered[index][0]] = doubled_average_rank
            cursor = end
        observed_positive = sum(
            rank for rank, difference in zip(doubled_ranks, differences) if difference > 0
        )
        total_rank = sum(doubled_ranks)
        observed_minimum = min(observed_positive, total_rank - observed_positive)
        counts = {0: 1}
        for rank in doubled_ranks:
            updated = dict(counts)
            for subtotal, count in counts.items():
                updated[subtotal + rank] = updated.get(subtotal + rank, 0) + count
            counts = updated
        extreme = sum(
            count
            for subtotal, count in counts.items()
            if subtotal <= observed_minimum or subtotal >= total_rank - observed_minimum
        )
        return min(1.0, extreme / (2 ** len(doubled_ranks)))
    return float(wilcoxon(left, right, alternative="two-sided").pvalue)


def _gap(value: float, reference: float | None) -> float | None:
    if reference is None or reference <= 0:
        return None
    return (value - reference) / reference * 100.0


def save_report(report: dict[str, object], output_path: Path) -> tuple[Path, Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    runs_path = output_path.with_name(output_path.stem + "_runs.csv")
    summary_path = output_path.with_name(output_path.stem + "_summary.csv")
    runs = list(report["runs"])
    with runs_path.open("w", newline="", encoding="utf-8-sig") as handle:
        columns = list(dict.fromkeys(key for row in runs for key in row))
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in runs:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )
    summary_rows = [
        {"algorithm": algorithm, **values}
        for algorithm, values in report["summary"]["by_algorithm"].items()
    ]
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    return output_path, runs_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="So sánh công bằng TD-CR-RAPO-V3 và RAPO-2709"
    )
    parser.add_argument("--instance", type=Path, default=DEFAULT_INSTANCE)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="File JSONL dùng để tự động tiếp tục lần chạy bị dừng",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Chạy lại từ đầu; checkpoint cũ được đổi tên thành bản backup",
    )
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Không in progress log ra terminal")
    args = parser.parse_args()
    if args.num_seeds <= 0:
        parser.error("--num-seeds phải dương")
    seeds = tuple(args.seeds) if args.seeds else tuple(
        range(args.seed_start, args.seed_start + args.num_seeds)
    )
    config = FairComparisonConfig(
        instance_path=args.instance,
        seeds=seeds,
        population_size=args.population_size,
        nfe_budget=args.budget,
        output_path=args.output,
        verify_checksums=not args.skip_checksums,
    )
    terminal_progress = None if args.quiet else lambda message: print(message, flush=True)
    try:
        report = run_comparison(
            config,
            progress=terminal_progress,
            checkpoint_path=args.checkpoint,
            fresh=args.fresh,
        )
    except KeyboardInterrupt:
        try:
            interrupted_checkpoint = _CheckpointLog(args.checkpoint, config)
            interrupted_checkpoint.append_event(
                "interrupted",
                message="Run đang thực hiện sẽ được chạy lại; các run hoàn tất được giữ nguyên",
            )
        except (OSError, RuntimeError, ValueError):
            pass
        print(
            f"\n[STOPPED] Đã giữ checkpoint tại {args.checkpoint}. "
            "Chạy lại với cùng protocol và không dùng --fresh để tiếp tục.",
            flush=True,
        )
        raise SystemExit(130)
    paths = save_report(report, config.output_path)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    for path in paths:
        print(f"Đã lưu: {path}")


if __name__ == "__main__":
    main()
