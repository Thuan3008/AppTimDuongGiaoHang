"""Merge seed shards produced on multiple computers into one canonical report."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.load_poryos import find_poryos_bks, load_poryos_bks
from experiments.plot_experiment_report import generate_plots
from experiments.rapo_contribution.analyze import greedy_summary, paired_comparisons, write_rows
from experiments.rapo_contribution.run import DEFAULT_CONFIG, _load_config
from experiments.runner import ExperimentReport, summarize_runs


PROTOCOL_KEYS = (
    "instance_name",
    "benchmark_name",
    "data_source",
    "proxy_graph",
    "traffic_model",
    "traffic_profile",
    "objective",
    "population_size",
    "nfe_budget",
    "algorithms",
    "initialization_policy",
    "stopping_condition",
    "start_time",
    "departure_time_mode",
    "departure_time_step",
    "bks_cost",
    "atf_sha256_verified",
)


def _stable_protocol(protocol: dict) -> dict:
    return {key: protocol.get(key) for key in PROTOCOL_KEYS}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _same_number(left: object, right: object) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-9)
    except (TypeError, ValueError):
        return left == right


def merge_dataset(
    dataset: str,
    config: dict,
    input_dir: Path,
    output_dir: Path,
    *,
    plots: bool = True,
) -> dict:
    report_paths = sorted(input_dir.glob("*/stochastic_report.json"))
    if not report_paths:
        raise FileNotFoundError(f"Không tìm thấy shard stochastic_report.json trong {input_dir}")

    expected_seeds = set(int(seed) for seed in config["seeds"])
    expected_algorithms = list(config["stochastic_algorithms"])
    merged_runs: dict[tuple[str, int], dict] = {}
    source_reports: list[tuple[Path, dict]] = []
    reference_protocol: dict | None = None
    seed_owners: dict[int, Path] = {}

    for report_path in report_paths:
        payload = _read_json(report_path)
        protocol = payload["protocol"]
        if int(protocol.get("nfe_budget", -1)) != int(config["nfe_budget"]):
            raise RuntimeError(
                f"NFE của shard {report_path.parent} là {protocol.get('nfe_budget')}, "
                f"nhưng lệnh merge yêu cầu {config['nfe_budget']}"
            )
        if int(protocol.get("population_size", -1)) != int(config["population_size"]):
            raise RuntimeError(f"Population size không khớp trong {report_path.parent}")
        if reference_protocol is None:
            reference_protocol = protocol
        elif _stable_protocol(protocol) != _stable_protocol(reference_protocol):
            raise RuntimeError(f"Protocol shard không khớp: {report_path}")
        shard_seeds = {int(seed) for seed in protocol.get("seeds", [])}
        for seed in shard_seeds:
            if seed in seed_owners:
                raise RuntimeError(
                    f"Seed {seed} xuất hiện trong cả {seed_owners[seed].parent} và {report_path.parent}"
                )
            seed_owners[seed] = report_path
        for row in payload.get("runs", []):
            key = (str(row["algorithm"]), int(row["seed"]))
            if key in merged_runs:
                raise RuntimeError(f"Run bị trùng: algorithm={key[0]}, seed={key[1]}")
            merged_runs[key] = row
        source_reports.append((report_path, payload))

    actual_seeds = set(seed_owners)
    missing = sorted(expected_seeds - actual_seeds)
    unexpected = sorted(actual_seeds - expected_seeds)
    if missing or unexpected:
        raise RuntimeError(f"Seed-set chưa đủ. Thiếu={missing}; ngoài protocol={unexpected}")

    expected_keys = {(algorithm, seed) for seed in expected_seeds for algorithm in expected_algorithms}
    actual_keys = set(merged_runs)
    if expected_keys != actual_keys:
        missing_runs = sorted(expected_keys - actual_keys)
        extra_runs = sorted(actual_keys - expected_keys)
        raise RuntimeError(f"Run-set chưa đủ. Thiếu={missing_runs}; dư={extra_runs}")
    failed = [key for key, row in merged_runs.items() if row.get("status") != "completed"]
    if failed:
        raise RuntimeError(f"Các run chưa hoàn thành, hãy resume shard tương ứng: {sorted(failed)}")

    runs = [merged_runs[(algorithm, seed)] for seed in sorted(expected_seeds) for algorithm in expected_algorithms]
    bks_cost = reference_protocol.get("bks_cost") if reference_protocol is not None else None
    summary = summarize_runs(runs, bks_cost)
    protocol = dict(reference_protocol or {})
    protocol["seeds"] = sorted(expected_seeds)

    best_traces: dict[str, dict] = {}
    for algorithm in expected_algorithms:
        best_fitness = min(float(row["fitness"]) for row in runs if row["algorithm"] == algorithm)
        for _, payload in source_reports:
            trace = payload.get("best_route_traces", {}).get(algorithm)
            if trace is not None and _same_number(trace.get("fitness"), best_fitness):
                best_traces[algorithm] = trace
                break

    output_dir.mkdir(parents=True, exist_ok=True)
    report = ExperimentReport(protocol, runs, summary, best_traces)
    report_path = output_dir / "stochastic_report.json"
    report.save(report_path)

    greedy_payloads = []
    for report_path, _ in source_reports:
        greedy_path = report_path.parent / "greedy.json"
        if greedy_path.is_file():
            greedy_payloads.append((greedy_path, _read_json(greedy_path)))
    if not greedy_payloads:
        raise FileNotFoundError("Không shard nào có greedy.json")
    greedy = greedy_payloads[0][1]
    for greedy_path, candidate in greedy_payloads[1:]:
        if not _same_number(candidate.get("fitness"), greedy.get("fitness")):
            raise RuntimeError(f"Greedy không đồng nhất giữa các máy: {greedy_path}")
    (output_dir / "greedy.json").write_text(
        json.dumps(greedy, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    combined = [
        {"dataset": dataset, **row}
        for row in [greedy_summary(greedy, bks_cost), *summary]
    ]
    comparisons = [
        {"dataset": dataset, **row}
        for row in paired_comparisons(runs, greedy)
    ]
    write_rows(output_dir / "summary.json", combined)
    write_rows(output_dir / "summary.csv", combined)
    write_rows(output_dir / "paired_comparisons.json", comparisons)
    write_rows(output_dir / "paired_comparisons.csv", comparisons)

    instance_path = (ROOT / config["datasets"][dataset]).resolve()
    bks_path = find_poryos_bks(instance_path)
    if bks_path is not None:
        local_bks = load_poryos_bks(bks_path)
        if bks_cost is not None and not _same_number(local_bks.get("cost"), bks_cost):
            raise RuntimeError("BKS local không khớp BKS trong các shard")

    manifest = {
        "dataset": dataset,
        "merged_at": datetime.now().isoformat(timespec="seconds"),
        "source_reports": [str(path) for path, _ in source_reports],
        "seeds": sorted(expected_seeds),
        "algorithms": expected_algorithms,
        "run_count": len(runs),
    }
    (output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if plots:
        generate_plots(report_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("n25", "n50", "n100"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--nfe",
        type=int,
        help="Ngân sách NFE của các shard cần merge, ví dụ --nfe 580",
    )
    args = parser.parse_args()

    config = _load_config(args.config.resolve())
    if args.nfe is not None:
        if args.nfe < int(config["population_size"]):
            parser.error(f"--nfe phải >= population size ({config['population_size']})")
        config = {**config, "nfe_budget": int(args.nfe)}
    default_root = Path(__file__).resolve().parent / "results" / args.dataset
    if args.nfe is not None:
        default_root = default_root / f"nfe_{args.nfe}"
    dataset_root = Path(args.output_dir).resolve() if args.output_dir else default_root
    input_dir = Path(args.input_dir).resolve() if args.input_dir else dataset_root / "shards"
    manifest = merge_dataset(
        args.dataset,
        config,
        input_dir,
        dataset_root,
        plots=not args.no_plots,
    )
    print(
        f"Merge hoàn tất: {manifest['run_count']} runs, "
        f"seed {min(manifest['seeds'])}..{max(manifest['seeds'])} -> {dataset_root}"
    )


if __name__ == "__main__":
    main()
