"""Run the same Hong Kong n=50 proxy instance across the main algorithms.

No BKS file is loaded here.  The script only reads the VRP instance and uses
the same proxy road graph, traffic profile, seed and NFE-equivalent budget for
every algorithm.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.hcmc_aro_goc import AROConfig, run_aro
from algorithms.hcmc_poa_goc import POAConfig, run_poa
from algorithms.hcmc_rapo import RAPOConfig, run_rapo
from algorithms.td_cr_rapo_v3 import TDCRRAPOV3Config, optimize as optimize_v3
from core.data.load_poryos import load_poryos_proxy


def run_comparison(instance_path: Path, budget: int, seed: int) -> list[dict[str, object]]:
    instance, graph, traffic = load_poryos_proxy(instance_path)
    iterations = max(1, (budget - 10) // 10)
    rows: list[dict[str, object]] = []

    aro = run_aro(instance, graph, traffic, config=AROConfig(population_size=10, iterations=iterations, seed=seed))
    rows.append({"algorithm": "ARO-goc", "fitness": aro.best_evaluation.fitness, "runtime_seconds": aro.runtime_seconds, "feasible": aro.best_evaluation.feasible, "nfe": 10 * (iterations + 1), "seed": seed})

    poa = run_poa(instance, graph, traffic, config=POAConfig(population_size=10, iterations=iterations, seed=seed))
    rows.append({"algorithm": "POA-goc", "fitness": poa.best_evaluation.fitness, "runtime_seconds": poa.runtime_seconds, "feasible": poa.best_evaluation.feasible, "nfe": 10 * (iterations + 1), "seed": seed})

    rapo = run_rapo(instance, graph, traffic, RAPOConfig(population_size=10, max_function_evaluations=budget, max_iterations=budget), seed=seed)
    rows.append({"algorithm": "RAPO", "fitness": rapo.best_evaluation.fitness, "runtime_seconds": rapo.runtime_seconds, "feasible": rapo.best_evaluation.feasible, "nfe": rapo.evaluations, "seed": seed})

    v3 = optimize_v3(instance, graph, traffic, TDCRRAPOV3Config(population_size=10, max_function_evaluations=budget), seed=seed)
    rows.append({"algorithm": "TD-CR-RAPO-V3", "fitness": v3.best_evaluation.fitness, "runtime_seconds": v3.runtime_seconds, "feasible": v3.best_evaluation.feasible, "nfe": v3.nfe, "seed": seed, "architecture": v3.metadata["architecture_status"], "spending": v3.spending})
    return sorted(rows, key=lambda row: float(row["fitness"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("data/TDVRP/hong_kong/n=50/poryos-hong_kong-n50-hyb/bpr-heavy/poryos-hong_kong-n50-hyb-bpr-heavy.vrp.json"))
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/hong_kong_n50_comparison.json"))
    args = parser.parse_args()
    rows = run_comparison(args.instance, args.budget, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["algorithm", "fitness", "runtime_seconds", "feasible", "nfe", "seed", "architecture", "spending"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(json.dumps(rows, indent=2))
    print(f"saved: {args.output}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
