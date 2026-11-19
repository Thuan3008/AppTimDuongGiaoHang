"""Run RAPO-2709 and three independent ablation variants on official Poryos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from datetime import datetime
from pathlib import Path

from algorithms import rapo_2709
from algorithms.rapo_ablation import ABLATION_VARIANTS, optimize_ablation
from core.data.load_poryos import find_poryos_bks, load_poryos, load_poryos_bks


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INSTANCE = ROOT / (
    "data/TDVRP/hong_kong/n=25/poryos-hong_kong-n25-hyb/"
    "bpr-heavy/poryos-hong_kong-n25-hyb-bpr-heavy.vrp.json"
)
VARIANTS = ("RAPO-2709", *ABLATION_VARIANTS)


def result_paths(output: Path) -> tuple[Path, Path, Path, Path]:
    """Resolve a report name into a self-contained result directory."""
    requested = output.resolve()
    if requested.suffix.lower() != ".json":
        raise ValueError("--output phải là tên file .json")
    directory = requested.parent if requested.parent.name == requested.stem else requested.parent / requested.stem
    return directory, directory / requested.name, directory / f"{requested.stem}.checkpoint.jsonl", directory / "run.log"


def _log_line(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}\n")


def _progress_line(index: int, total: int, row: dict) -> str:
    return (
        f"[{index}/{total}] {row['variant']} seed={row['seed']} "
        f"fitness={row['fitness']:.6f} NFE={row['nfe']} "
        f"runtime={row['runtime_seconds']:.2f}s"
    )


def organize_existing(output: Path) -> Path:
    """Move a completed run from the old flat layout into its result directory."""
    requested = output.resolve()
    directory, report, checkpoint, log_path = result_paths(requested)
    legacy_checkpoint = requested.with_suffix(".checkpoint.jsonl")
    legacy_files = {
        requested: report,
        legacy_checkpoint: checkpoint,
        **{
            requested.with_name(f"{requested.stem}_{kind}.csv"): directory / f"{requested.stem}_{kind}.csv"
            for kind in ("runs", "summary", "paired")
        },
    }
    if requested.parent == directory:
        raise ValueError("Kết quả đã ở trong thư mục riêng")
    missing = [str(source) for source in legacy_files if not source.is_file()]
    if missing:
        raise ValueError("Chưa đủ file kết quả; hãy đợi lệnh đang chạy hoàn tất: " + ", ".join(missing))
    collisions = [str(target) for target in (*legacy_files.values(), log_path) if target.exists()]
    if collisions:
        raise FileExistsError("Không ghi đè file trong thư mục đích: " + ", ".join(collisions))
    lines = legacy_checkpoint.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("Checkpoint rỗng")
    header = json.loads(lines[0])
    if header.get("type") != "header":
        raise ValueError("Checkpoint thiếu header")
    protocol = header["protocol"]
    expected = {(variant, seed) for variant in protocol["variants"] for seed in protocol["seeds"]}
    rows = [json.loads(line)["row"] for line in lines[1:] if line.strip()]
    actual = [(row["variant"], row["seed"]) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(f"Checkpoint chưa hoàn tất: {len(actual)}/{len(expected)} run")
    saved_report = json.loads(requested.read_text(encoding="utf-8"))
    if saved_report.get("protocol") != protocol or len(saved_report.get("runs", [])) != len(expected):
        raise ValueError("JSON kết quả không khớp checkpoint")
    directory.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "Recovered log from completed legacy checkpoint (timestamps of earlier runs unavailable).\n"
        + "\n".join(_progress_line(index, len(expected), row) for index, row in enumerate(rows, 1))
        + "\n",
        encoding="utf-8",
    )
    for source, target in legacy_files.items():
        source.rename(target)
    _log_line(log_path, f"DONE: moved {len(expected)} runs and result files into {directory}")
    return directory


def parse_seeds(spec: str) -> list[int]:
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            raise ValueError("Danh sách seed không hợp lệ")
        if "-" in part:
            first, last = map(int, part.split("-", 1))
            if first > last:
                raise ValueError("Khoảng seed phải tăng dần")
            seeds.extend(range(first, last + 1))
        else:
            seeds.append(int(part))
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("Seed phải không âm và không trùng")
    return sorted(seeds)


def common_population(n: int, size: int, seed: int) -> list[tuple[int, ...]]:
    rng = random.Random(seed)
    customers = list(range(1, n + 1))
    return [tuple(rng.sample(customers, n)) for _ in range(size)]


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict], bks_cost: float | None) -> tuple[list[dict], list[dict]]:
    summaries: list[dict] = []
    values: dict[str, dict[int, float]] = {}
    for variant in VARIANTS:
        group = [row for row in rows if row["variant"] == variant]
        fitnesses = [float(row["fitness"]) for row in group]
        values[variant] = {int(row["seed"]): float(row["fitness"]) for row in group}
        if fitnesses:
            summaries.append({
                "variant": variant,
                "runs": len(group),
                "best": min(fitnesses),
                "mean": statistics.fmean(fitnesses),
                "median": statistics.median(fitnesses),
                "standard_deviation": statistics.stdev(fitnesses) if len(fitnesses) > 1 else None,
                "mean_gap_to_bks_percent": (
                    (statistics.fmean(fitnesses) - bks_cost) / bks_cost * 100
                    if bks_cost is not None else None
                ),
                "mean_runtime_seconds": statistics.fmean(float(row["runtime_seconds"]) for row in group),
            })

    full = values["RAPO-2709"]
    comparisons: list[dict] = []
    for variant in ABLATION_VARIANTS:
        shared = sorted(full.keys() & values[variant].keys())
        deltas = [values[variant][seed] - full[seed] for seed in shared]
        comparisons.append({
            "variant": variant,
            "paired_seeds": len(shared),
            "full_wins": sum(delta > 1e-9 for delta in deltas),
            "ties": sum(abs(delta) <= 1e-9 for delta in deltas),
            "variant_wins": sum(delta < -1e-9 for delta in deltas),
            "mean_variant_minus_full_seconds": statistics.fmean(deltas) if deltas else None,
        })
    return summaries, comparisons


def run_ablation(
    instance_path: Path,
    seeds: list[int],
    population_size: int,
    nfe_budget: int,
    output: Path,
) -> dict:
    if not seeds or population_size < 2 or nfe_budget < population_size:
        raise ValueError("Cần seed, P >= 2 và NFE >= P")
    directory, output, checkpoint, log_path = result_paths(output)
    legacy_checkpoint = output.parent.parent / checkpoint.name if output.parent == directory else None
    if legacy_checkpoint is not None and legacy_checkpoint.is_file() and not checkpoint.is_file():
        raise ValueError(
            f"Có checkpoint theo cấu trúc cũ: {legacy_checkpoint}. "
            "Đợi run hiện tại xong rồi dùng --organize-existing --output <đường dẫn JSON cũ>."
        )
    instance_path = instance_path.resolve()
    instance, graph, traffic = load_poryos(instance_path)
    if instance.metadata.get("proxy_graph") or instance.arrival_time_functions is None:
        raise ValueError("Ablation yêu cầu Poryos sidecar và ATF chính thức")
    bks_path = find_poryos_bks(instance_path)
    bks_cost = float(load_poryos_bks(bks_path)["cost"]) if bks_path.is_file() else None
    protocol = {
        "instance_path": str(instance_path),
        "instance_name": instance.instance_name,
        "variants": list(VARIANTS),
        "seeds": seeds,
        "population_size": population_size,
        "nfe_budget": nfe_budget,
        "objective": "total_route_duration = total_travel_time + total_service_time",
        "split_decoder": "exact minimum-duration",
        "departure_time_mode": "exact_ndcpwlf",
        "initialization_policy": "paired_common_population_per_seed",
        "atf_sha256_verified": instance.metadata.get("atf_sha256_verified"),
        "bks_cost": bks_cost,
        "rapo_source_sha256": _source_sha256(Path(rapo_2709.__file__)),
        "ablation_source_sha256": _source_sha256(ROOT / "algorithms/rapo_ablation.py"),
    }
    directory.mkdir(parents=True, exist_ok=True)
    if checkpoint.is_file():
        lines = checkpoint.read_text(encoding="utf-8").splitlines()
        if not lines or json.loads(lines[0]) != {"type": "header", "protocol": protocol}:
            raise ValueError(f"Checkpoint khác protocol: {checkpoint}")
        rows = [json.loads(line)["row"] for line in lines[1:] if line.strip()]
    else:
        checkpoint.write_text(
            json.dumps({"type": "header", "protocol": protocol}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        rows = []
    if rows and not log_path.exists():
        log_path.write_text(
            "Recovered progress from checkpoint (timestamps of earlier runs unavailable).\n"
            + "\n".join(_progress_line(index, len(seeds) * len(VARIANTS), row) for index, row in enumerate(rows, 1))
            + "\n",
            encoding="utf-8",
        )
    _log_line(log_path, f"START: instance={instance_path.name} seeds={seeds} P={population_size} NFE={nfe_budget} resumed={len(rows)}")
    completed = {(row["variant"], row["seed"]) for row in rows}

    for seed in seeds:
        initial = common_population(instance.num_customers, population_size, seed)
        p0_hash = hashlib.sha256(json.dumps(initial).encode("utf-8")).hexdigest()
        config = rapo_2709.RAPO2709Config(
            population_size=population_size,
            max_function_evaluations=nfe_budget,
            exact_split=True,
        )
        for variant in VARIANTS:
            if (variant, seed) in completed:
                continue
            if variant == "RAPO-2709":
                result = rapo_2709.optimize(
                    instance, graph, traffic, config=config,
                    seed=seed, initial_population=initial,
                )
            else:
                result = optimize_ablation(
                    instance, graph, traffic, variant=variant, config=config,
                    seed=seed, initial_population=initial,
                )
            if (
                result.evaluations != nfe_budget
                or not result.best_evaluation.feasible
                or not math.isfinite(result.best_evaluation.fitness)
            ):
                raise RuntimeError(f"NFE/fitness không hợp lệ: {variant}, seed={seed}")
            row = {
                "variant": variant,
                "seed": seed,
                "fitness": float(result.best_evaluation.fitness),
                "gap_to_bks_percent": (
                    (result.best_evaluation.fitness - bks_cost) / bks_cost * 100
                    if bks_cost is not None else None
                ),
                "runtime_seconds": result.runtime_seconds,
                "feasible": True,
                "nfe": result.evaluations,
                "nfe_by_branch": result.nfe_by_branch,
                "operator_uses": result.operator_uses,
                "initial_population_sha256": p0_hash,
                "convergence": result.convergence,
            }
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"type": "run", "row": row}, ensure_ascii=False) + "\n")
                handle.flush()
            rows.append(row)
            progress = _progress_line(len(rows), len(seeds) * len(VARIANTS), row)
            print(progress, flush=True)
            _log_line(log_path, progress)

    summary, comparisons = _summarize(rows, bks_cost)
    report = {"protocol": protocol, "runs": rows, "summary": summary, "paired_comparisons": comparisons}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(output.with_name(output.stem + "_runs.csv"), [
        {
            **{key: value for key, value in row.items() if key not in {"convergence", "nfe_by_branch", "operator_uses"}},
            "nfe_by_branch": json.dumps(row["nfe_by_branch"]),
            "operator_uses": json.dumps(row["operator_uses"]),
        }
        for row in rows
    ])
    _write_csv(output.with_name(output.stem + "_summary.csv"), summary)
    _write_csv(output.with_name(output.stem + "_paired.csv"), comparisons)
    _log_line(log_path, f"DONE: {len(rows)}/{len(seeds) * len(VARIANTS)} runs; report={output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired RAPO-2709 ablation")
    parser.add_argument("--instance", type=Path, default=DEFAULT_INSTANCE)
    parser.add_argument("--seeds", default="0-9", help="Ví dụ: 0-9 hoặc 0,2,4")
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--nfe", type=int, default=290)
    parser.add_argument("--output", type=Path, default=ROOT / "results/rapo_ablation_n25.json")
    parser.add_argument("--organize-existing", action="store_true", help="Gom kết quả hoàn tất từ cấu trúc cũ vào một thư mục")
    args = parser.parse_args()
    if args.organize_existing:
        print(f"Đã gom kết quả vào: {organize_existing(args.output)}")
        return
    report = run_ablation(args.instance, parse_seeds(args.seeds), args.population_size, args.nfe, args.output)
    print(json.dumps(report["paired_comparisons"], ensure_ascii=False, indent=2))
    print(f"Đã lưu: {result_paths(args.output)[0]}")


if __name__ == "__main__":
    main()
