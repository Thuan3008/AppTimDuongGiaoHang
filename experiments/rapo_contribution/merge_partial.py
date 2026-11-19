"""Merge completed runs from overlapping contribution checkpoints without rerunning.

This creates an explicitly partial report in a separate output directory. It
never edits the source checkpoints and never claims that the 30-seed dataset is
complete.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from core.data.load_poryos import find_poryos_bks, load_poryos_bks
from experiments.plot_experiment_report import generate_plots
from experiments.rapo_contribution.analyze import greedy_summary, paired_comparisons, write_rows
from experiments.rapo_contribution.merge import _stable_protocol
from experiments.runner import ExperimentReport, summarize_runs


def _records(path: Path) -> tuple[dict, list[dict]]:
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines or lines[0].get("type") != "header":
        raise ValueError(f"Checkpoint thiếu header: {path}")
    records = [item for item in lines[1:] if item.get("type") == "run"]
    return lines[0]["protocol"], records


def _equivalent_run(left: dict, right: dict) -> bool:
    left_row, right_row = left["row"], right["row"]
    return (
        {key: value for key, value in left_row.items() if key != "runtime_seconds"}
        == {key: value for key, value in right_row.items() if key != "runtime_seconds"}
        and left.get("trace") == right.get("trace")
    )


def merge_partial(
    checkpoints: list[Path], expected_seeds: list[int], output_dir: Path,
    greedy_path: Path, instance_path: Path, *, plots: bool = True,
) -> dict:
    if len(checkpoints) < 2 or not expected_seeds or len(expected_seeds) != len(set(expected_seeds)):
        raise ValueError("Cần ít nhất hai checkpoint và danh sách seed không trùng")
    reference_protocol = None
    selected: dict[tuple[str, int], dict] = {}
    duplicate_runs = []
    for path in checkpoints:
        protocol, records = _records(path)
        if reference_protocol is None:
            reference_protocol = protocol
        elif _stable_protocol(protocol) != _stable_protocol(reference_protocol):
            raise ValueError(f"Protocol không khớp: {path}")
        allowed = set(protocol["seeds"])
        for record in records:
            row = record["row"]
            key = str(row["algorithm"]), int(row["seed"])
            if key[1] not in allowed:
                raise ValueError(f"Run ngoài seed protocol tại {path}: {key}")
            if key in selected:
                if not _equivalent_run(selected[key], record):
                    raise ValueError(f"Run trùng nhưng nghiệm khác nhau: {key}")
                duplicate_runs.append({
                    "algorithm": key[0], "seed": key[1],
                    "kept_runtime_seconds": selected[key]["row"]["runtime_seconds"],
                    "excluded_runtime_seconds": row["runtime_seconds"],
                    "excluded_from": str(path),
                })
                continue
            selected[key] = record
    assert reference_protocol is not None
    algorithms = list(reference_protocol["algorithms"])
    expected = {(algorithm, seed) for seed in expected_seeds for algorithm in algorithms}
    if set(selected) != expected:
        raise ValueError(f"Run-set chưa đủ: thiếu={sorted(expected - set(selected))}; dư={sorted(set(selected) - expected)}")
    nfe_budget = int(reference_protocol["nfe_budget"])
    for key, record in selected.items():
        row = record["row"]
        if row.get("status") != "completed" or not row.get("feasible") or int(row["nfe"]) != nfe_budget:
            raise ValueError(f"Run chưa hoàn tất hoặc sai NFE: {key}")
        if not math.isfinite(float(row["fitness"])) or not row.get("convergence"):
            raise ValueError(f"Fitness/convergence không hợp lệ: {key}")
    runs = [selected[(algorithm, seed)]["row"] for seed in sorted(expected_seeds) for algorithm in algorithms]
    best_traces = {}
    for algorithm in algorithms:
        record = min((selected[(algorithm, seed)] for seed in expected_seeds),
                     key=lambda item: float(item["row"]["fitness"]))
        if record.get("trace") is None:
            raise ValueError(f"Thiếu trace nghiệm tốt nhất: {algorithm}")
        best_traces[algorithm] = record["trace"]
    bks_cost = reference_protocol.get("bks_cost")
    bks_path = find_poryos_bks(instance_path)
    if bks_path is not None and bks_cost is not None:
        local_cost = float(load_poryos_bks(bks_path)["cost"])
        if not math.isclose(float(bks_cost), local_cost, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError("BKS local khác BKS checkpoint")
    greedy = json.loads(greedy_path.read_text(encoding="utf-8"))
    report_protocol = {**reference_protocol, "seeds": sorted(expected_seeds)}
    summary = summarize_runs(runs, bks_cost)
    report = ExperimentReport(report_protocol, runs, summary, best_traces)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "stochastic_report.json"
    report.save(report_path)
    (output_dir / "greedy.json").write_text(json.dumps(greedy, ensure_ascii=False, indent=2), encoding="utf-8")
    combined = [{"dataset": "n100", **row} for row in [greedy_summary(greedy, bks_cost), *summary]]
    comparisons = [{"dataset": "n100", **row} for row in paired_comparisons(runs, greedy)]
    for filename, rows in (("summary", combined), ("paired_comparisons", comparisons)):
        write_rows(output_dir / f"{filename}.json", rows)
        write_rows(output_dir / f"{filename}.csv", rows)
    manifest = {
        "dataset": "n100", "status": "partial_10_of_30_seeds",
        "merged_at": datetime.now().isoformat(timespec="seconds"),
        "source_checkpoints": [str(path) for path in checkpoints],
        "selected_seeds": sorted(expected_seeds), "selected_run_count": len(runs),
        "duplicate_runs_excluded": duplicate_runs,
        "selection_rule": "first checkpoint wins when duplicate solutions match except runtime",
        "note": "This is a 10-seed report; do not combine it with the 30-seed n25/n50 summaries as if sample sizes matched.",
    }
    (output_dir / "merge_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if plots:
        generate_plots(report_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--seeds", default="0-9", help="Inclusive seed range, e.g. 0-9")
    parser.add_argument("--greedy", required=True, type=Path)
    parser.add_argument("--instance", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    first, last = map(int, args.seeds.split("-", 1))
    manifest = merge_partial(
        [path.resolve() for path in args.checkpoints], list(range(first, last + 1)),
        args.output_dir.resolve(), args.greedy.resolve(), args.instance.resolve(),
        plots=not args.no_plots,
    )
    print(f"Đã gộp {manifest['selected_run_count']} run, seed {first}–{last}; "
          f"loại {len(manifest['duplicate_runs_excluded'])} run trùng vào {args.output_dir}")


if __name__ == "__main__":
    main()
