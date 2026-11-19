"""Fair comparison harness for Hong Kong n=50.

All algorithms use the same greedy-seeded initialization policy, population
size, proxy traffic graph, seed, exact evaluator budget and objective.  BKS
files are intentionally not read.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from algorithms.hcmc_aro_goc import AROConfig, run_aro
from algorithms.hcmc_poa_goc import POAConfig, run_poa
from algorithms.hcmc_rapo import RAPOConfig, run_rapo
from algorithms.td_cr_rapo_v3 import TDCRRAPOV3Config, optimize as optimize_v3
from core.data.load_poryos import load_poryos_proxy


DEFAULT_INSTANCE = ROOT / "data/TDVRP/hong_kong/n=50/poryos-hong_kong-n50-hyb/bpr-heavy/poryos-hong_kong-n50-hyb-bpr-heavy.vrp.json"


def compare(instance_path: Path, population_size: int, budget: int, seed: int) -> list[dict[str, object]]:
    if budget < population_size:
        raise ValueError("budget must be at least population_size")
    instance, graph, traffic = load_poryos_proxy(instance_path)
    rows: list[dict[str, object]] = []

    aro = run_aro(instance, graph, traffic, config=AROConfig(population_size=population_size, iterations=10_000, seed=seed, max_function_evaluations=budget))
    rows.append({"algorithm": "ARO-goc", "fitness": aro.best_evaluation.fitness, "runtime_seconds": aro.runtime_seconds, "nfe": aro.evaluations, "feasible": aro.best_evaluation.feasible, "seed": seed})

    poa = run_poa(instance, graph, traffic, config=POAConfig(population_size=population_size, iterations=10_000, seed=seed, max_function_evaluations=budget))
    rows.append({"algorithm": "POA-goc", "fitness": poa.best_evaluation.fitness, "runtime_seconds": poa.runtime_seconds, "nfe": poa.evaluations, "feasible": poa.best_evaluation.feasible, "seed": seed})

    rapo = run_rapo(instance, graph, traffic, RAPOConfig(population_size=population_size, max_function_evaluations=budget, max_iterations=10_000), seed=seed)
    rows.append({"algorithm": "RAPO", "fitness": rapo.best_evaluation.fitness, "runtime_seconds": rapo.runtime_seconds, "nfe": rapo.evaluations, "feasible": rapo.best_evaluation.feasible, "seed": seed})

    v3 = optimize_v3(instance, graph, traffic, TDCRRAPOV3Config(population_size=population_size, max_function_evaluations=budget), seed=seed)
    rows.append({"algorithm": "TD-CR-RAPO-V3", "fitness": v3.best_evaluation.fitness, "runtime_seconds": v3.runtime_seconds, "nfe": v3.nfe, "feasible": v3.best_evaluation.feasible, "seed": seed, "architecture": v3.metadata["architecture_status"], "spending": v3.spending})
    return sorted(rows, key=lambda row: float(row["fitness"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=DEFAULT_INSTANCE)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--budget", type=int, default=290)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results_seed42.json")
    args = parser.parse_args()
    rows = compare(args.instance, args.population_size, args.budget, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        columns = ["algorithm", "fitness", "runtime_seconds", "nfe", "feasible", "seed", "architecture", "spending"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    print(f"saved {args.output}")
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()

