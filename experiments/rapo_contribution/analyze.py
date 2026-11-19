"""Aggregate descriptive and paired statistics for the RAPO contribution study."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics


def _gap(value: float, reference: float | None) -> float | None:
    if reference is None or reference <= 0:
        return None
    return (value - reference) / reference * 100.0


def greedy_summary(greedy: dict, bks_cost: float | None) -> dict:
    fitness = float(greedy["fitness"])
    return {
        "algorithm": "Greedy",
        "runs_requested": 1,
        "runs_completed": 1,
        "runs_failed": 0,
        "feasible_rate": 1.0 if greedy.get("feasible") else 0.0,
        "best": fitness,
        "mean": fitness,
        "median": fitness,
        "standard_deviation": None,
        "runtime_mean_seconds": float(greedy["runtime_seconds"]),
        "runtime_median_seconds": float(greedy["runtime_seconds"]),
        "runtime_standard_deviation": None,
        "best_gap_to_bks_percent": _gap(fitness, bks_cost),
        "mean_gap_to_bks_percent": _gap(fitness, bks_cost),
        "statistical_note": "deterministic single run; population, seed and NFE do not apply",
    }


def paired_comparisons(
    runs: list[dict],
    greedy: dict,
    primary: str = "RAPO-2709",
) -> list[dict]:
    completed = [
        row
        for row in runs
        if row.get("status") == "completed" and row.get("fitness") is not None
    ]
    by_algorithm = {
        algorithm: {int(row["seed"]): float(row["fitness"]) for row in completed if row["algorithm"] == algorithm}
        for algorithm in sorted({str(row["algorithm"]) for row in completed})
    }
    if primary not in by_algorithm:
        raise ValueError(f"Missing primary algorithm {primary}")
    primary_values = by_algorithm[primary]
    comparisons: list[dict] = []
    for baseline, baseline_values in by_algorithm.items():
        if baseline == primary:
            continue
        seeds = sorted(set(primary_values) & set(baseline_values))
        deltas = [baseline_values[seed] - primary_values[seed] for seed in seeds]
        tolerance = 1e-9
        wins = sum(delta > tolerance for delta in deltas)
        losses = sum(delta < -tolerance for delta in deltas)
        ties = len(deltas) - wins - losses
        comparisons.append(
            {
                "primary": primary,
                "baseline": baseline,
                "paired_seeds": len(seeds),
                "primary_wins": wins,
                "ties": ties,
                "primary_losses": losses,
                "mean_baseline_minus_primary_seconds": statistics.fmean(deltas),
                "median_baseline_minus_primary_seconds": statistics.median(deltas),
                "two_sided_sign_test_p": _two_sided_sign_test(wins, losses),
            }
        )

    greedy_fitness = float(greedy["fitness"])
    greedy_deltas = [greedy_fitness - value for value in primary_values.values()]
    comparisons.append(
        {
            "primary": primary,
            "baseline": "Greedy",
            "paired_seeds": len(greedy_deltas),
            "primary_wins": sum(delta > 1e-9 for delta in greedy_deltas),
            "ties": sum(abs(delta) <= 1e-9 for delta in greedy_deltas),
            "primary_losses": sum(delta < -1e-9 for delta in greedy_deltas),
            "mean_baseline_minus_primary_seconds": statistics.fmean(greedy_deltas),
            "median_baseline_minus_primary_seconds": statistics.median(greedy_deltas),
            "two_sided_sign_test_p": None,
            "statistical_note": "Greedy is one deterministic reference, so no paired significance test is claimed",
        }
    )
    return comparisons


def _two_sided_sign_test(wins: int, losses: int) -> float | None:
    trials = wins + losses
    if not trials:
        return None
    tail = sum(math.comb(trials, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**trials))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

