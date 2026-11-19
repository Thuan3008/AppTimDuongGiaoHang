"""Run the canonical RAPO contribution experiment on Hong Kong benchmarks."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.load_poryos import find_poryos_bks, load_poryos_bks
from core.tdvrp_common import TDVRPProblem
from experiments.plot_experiment_report import generate_plots
from experiments.rapo_contribution.analyze import (
    greedy_summary,
    paired_comparisons,
    write_rows,
)
from experiments.rapo_contribution.exact_greedy import run_exact_greedy
from experiments.runner import ExperimentConfig, ExperimentRunner


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.json"


def parse_seed_spec(spec: str, allowed: list[int]) -> list[int]:
    """Parse ``0-9,12`` and reject seeds outside the canonical protocol."""
    selected: list[int] = []
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError(f"Seed specification không hợp lệ: {spec!r}")
        if "-" in part:
            bounds = part.split("-", 1)
            if len(bounds) != 2:
                raise ValueError(f"Khoảng seed không hợp lệ: {part!r}")
            start, end = (int(value.strip()) for value in bounds)
            if start > end:
                raise ValueError(f"Khoảng seed phải tăng dần: {part!r}")
            selected.extend(range(start, end + 1))
        else:
            selected.append(int(part))
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Danh sách seed không được rỗng hoặc trùng nhau")
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise ValueError(f"Seed nằm ngoài protocol canonical: {unknown}")
    return sorted(selected)


def seed_shard_name(seeds: list[int]) -> str:
    """Stable directory name used when moving shards between computers."""
    contiguous = seeds == list(range(min(seeds), max(seeds) + 1))
    if contiguous:
        return f"seeds_{min(seeds):02d}_{max(seeds):02d}"
    return "seeds_" + "_".join(f"{seed:02d}" for seed in seeds)


def bootstrap_shard_checkpoint(
    source: Path,
    destination: Path,
    seeds: list[int],
    algorithms: list[str],
    nfe_budget: int,
) -> int:
    """Reuse matching completed work from a legacy full-seed checkpoint."""
    if destination.exists() or not source.is_file():
        return 0
    lines = source.read_text(encoding="utf-8").splitlines()
    if not lines:
        return 0
    header = json.loads(lines[0])
    if header.get("type") != "header" or not isinstance(header.get("protocol"), dict):
        return 0
    if int(header["protocol"].get("nfe_budget", -1)) != int(nfe_budget):
        return 0
    header = {**header, "protocol": {**header["protocol"], "seeds": list(seeds)}}
    allowed_keys = {(algorithm, seed) for algorithm in algorithms for seed in seeds}
    records = []
    imported_keys: set[tuple[str, int]] = set()
    for line in lines[1:]:
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("type") != "run" or not isinstance(record.get("row"), dict):
            continue
        row = record["row"]
        key = (str(row.get("algorithm")), int(row.get("seed", -1)))
        if key in allowed_keys:
            records.append(record)
            if row.get("status") == "completed":
                imported_keys.add(key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = [json.dumps(header, ensure_ascii=False)]
    serialized.extend(json.dumps(record, ensure_ascii=False) for record in records)
    destination.write_text("\n".join(serialized) + "\n", encoding="utf-8")
    return len(imported_keys)


def _load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "datasets",
        "stochastic_algorithms",
        "population_size",
        "nfe_budget",
        "seeds",
        "departure_policy",
        "earliest_departure_seconds",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Experiment config is missing: {missing}")
    if int(config["nfe_budget"]) < int(config["population_size"]):
        raise ValueError("NFE phải lớn hơn hoặc bằng population size")
    if config["seeds"] != list(range(30)):
        raise ValueError("Canonical contribution experiment requires seeds 0..29")
    if config["departure_policy"] != "exact_ndcpwlf_optimize":
        raise ValueError("Hong Kong BKS comparison requires exact_ndcpwlf_optimize")
    return config


def _progress_logger(log_path: Path):
    def progress(event: dict) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if event["status"] == "completed":
            detail = (
                f"fitness={event['fitness']:.6f} | NFE={event['nfe']} | "
                f"runtime={event['runtime_seconds']:.2f}s"
            )
        else:
            detail = f"ERROR={event['error_message']}"
        line = (
            f"[{timestamp}] [{event['completed']}/{event['total']}] "
            f"{event['algorithm']} seed={event['seed']} | "
            f"{event['status']} | {detail}"
        )
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    return progress


def run_dataset(
    label: str,
    instance_path: Path,
    config: dict,
    output_root: Path,
    *,
    resume: bool,
    plots: bool,
    shard_name: str | None = None,
    budget_name: str | None = None,
) -> dict:
    dataset_root = output_root / label
    if budget_name is not None:
        dataset_root = dataset_root / budget_name
    output_dir = dataset_root / "shards" / shard_name if shard_name is not None else dataset_root
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "stochastic_report.json"
    checkpoint_path = output_dir / "checkpoint.jsonl"
    log_path = output_dir / "run.log"
    if not resume:
        for old in (report_path, checkpoint_path, log_path):
            if old.exists():
                old.unlink()
    elif shard_name is not None:
        imported = bootstrap_shard_checkpoint(
            output_root / label / "checkpoint.jsonl",
            checkpoint_path,
            list(config["seeds"]),
            list(config["stochastic_algorithms"]),
            int(config["nfe_budget"]),
        )
        if imported:
            message = f"Imported {imported} completed runs from legacy checkpoint"
            print(message, flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")

    setup = (
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SETUP | "
        f"dataset={label} | algorithms={','.join(config['stochastic_algorithms'])} | "
        f"P={config['population_size']} | NFE={config['nfe_budget']} | "
        f"seeds={min(config['seeds'])}..{max(config['seeds'])} | "
        f"departure={config['departure_policy']}"
    )
    print(setup, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(setup + "\n")

    runner = ExperimentRunner(
        ExperimentConfig(
            instance_path=instance_path,
            seeds=tuple(config["seeds"]),
            population_size=int(config["population_size"]),
            nfe_budget=int(config["nfe_budget"]),
            algorithms=tuple(config["stochastic_algorithms"]),
            start_time=float(config["earliest_departure_seconds"]),
            departure_time_mode="optimize",
            departure_time_step=60.0,
            allow_proxy=False,
            verify_checksums=True,
            checkpoint_path=checkpoint_path,
            resume=resume,
        ),
        progress_callback=_progress_logger(log_path),
    )
    report = runner.run()
    report.save(report_path)

    problem = TDVRPProblem(
        instance_path,
        collection_root=ROOT / "data/TDVRP",
        vehicle_limit=json.loads(instance_path.read_text(encoding="utf-8"))["num_customers"],
    )
    greedy = run_exact_greedy(problem).to_dict()
    greedy_path = output_dir / "greedy.json"
    greedy_path.write_text(json.dumps(greedy, indent=2, ensure_ascii=False), encoding="utf-8")

    bks_path = find_poryos_bks(instance_path)
    bks = load_poryos_bks(bks_path) if bks_path is not None else None
    bks_cost = float(bks["cost"]) if bks is not None else None
    combined = [
        {"dataset": label, **row}
        for row in [greedy_summary(greedy, bks_cost), *report.summary]
    ]
    comparisons = [
        {"dataset": label, **row}
        for row in paired_comparisons(report.runs, greedy)
    ]
    write_rows(output_dir / "summary.json", combined)
    write_rows(output_dir / "summary.csv", combined)
    write_rows(output_dir / "paired_comparisons.json", comparisons)
    write_rows(output_dir / "paired_comparisons.csv", comparisons)
    if plots:
        generate_plots(report_path)
    return {
        "dataset": label,
        "instance": str(instance_path),
        "summary": combined,
        "paired_comparisons": comparisons,
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("n25", "n50", "n100", "all"),
        default=("all",),
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--seeds",
        help="Chạy một shard seed, ví dụ 0-9, 10-19 hoặc 20-29",
    )
    parser.add_argument(
        "--nfe",
        type=int,
        help="Ghi đè ngân sách đánh giá, ví dụ --nfe 580",
    )
    args = parser.parse_args()

    config = _load_config(args.config.resolve())
    budget_name = None
    if args.nfe is not None:
        if args.nfe < int(config["population_size"]):
            parser.error(f"--nfe phải >= population size ({config['population_size']})")
        config = {**config, "nfe_budget": int(args.nfe)}
        budget_name = f"nfe_{args.nfe}"
    canonical_seeds = list(config["seeds"])
    shard_name = None
    if args.seeds:
        selected_seeds = parse_seed_spec(args.seeds, canonical_seeds)
        config = {**config, "seeds": selected_seeds}
        shard_name = seed_shard_name(selected_seeds)
    selected = list(config["datasets"]) if "all" in args.datasets else list(args.datasets)
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifests = []
    for label in selected:
        instance_path = (ROOT / config["datasets"][label]).resolve()
        if not instance_path.is_file():
            raise FileNotFoundError(instance_path)
        manifests.append(
            run_dataset(
                label,
                instance_path,
                config,
                output_root,
                resume=not args.no_resume,
                plots=not args.no_plots,
                shard_name=shard_name,
                budget_name=budget_name,
            )
        )

    all_summary = [row for item in manifests for row in item["summary"]]
    all_comparisons = [row for item in manifests for row in item["paired_comparisons"]]
    aggregate_root = output_root
    if shard_name is not None:
        aggregate_root = output_root / "shard_manifests"
        if budget_name is not None:
            aggregate_root = aggregate_root / budget_name
        aggregate_root = aggregate_root / shard_name
    aggregate_root.mkdir(parents=True, exist_ok=True)
    write_rows(aggregate_root / "all_datasets_summary.json", all_summary)
    write_rows(aggregate_root / "all_datasets_summary.csv", all_summary)
    write_rows(aggregate_root / "all_paired_comparisons.json", all_comparisons)
    write_rows(aggregate_root / "all_paired_comparisons.csv", all_comparisons)
    (aggregate_root / "manifest.json").write_text(
        json.dumps(
            {
                "config": config,
                "datasets": manifests,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result_dirs = ", ".join(item["output_dir"] for item in manifests)
    print(f"Hoàn tất. Kết quả: {result_dirs}")


if __name__ == "__main__":
    main()

