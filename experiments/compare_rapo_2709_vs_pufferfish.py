"""So sánh riêng RAPO-2709 và RAPO-Pufferfish trên cùng protocol."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.plot_experiment_report import generate_plots
from experiments.runner import ExperimentConfig, ExperimentRunner


ALGORITHMS = ("RAPO-2709", "RAPO-Pufferfish")
DEFAULT_INSTANCE = (
    ROOT
    / "data/TDVRP/hong_kong/n=100/poryos-hong_kong-n100-hyb/bpr-heavy/"
    / "poryos-hong_kong-n100-hyb-bpr-heavy.vrp.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="So sánh riêng RAPO-2709 và RAPO-Pufferfish"
    )
    parser.add_argument("--instance", type=Path, default=DEFAULT_INSTANCE)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--budget", type=int, default=290)
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Bỏ qua checkpoint cũ và chạy lại toàn bộ",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Không sinh ảnh sau khi benchmark hoàn tất",
    )
    args = parser.parse_args()

    if args.num_seeds <= 0:
        parser.error("--num-seeds phải dương")

    output = args.output or (
        ROOT / "experiments/rapo_2709_vs_pufferfish_n100_nfe290_20seeds.json"
    )
    log_path = args.log or output.with_suffix(".log")
    checkpoint_path = args.checkpoint or output.with_name(
        output.stem + ".checkpoint.jsonl"
    )
    seeds = tuple(range(args.seed_start, args.seed_start + args.num_seeds))

    # Ghi tiến độ ra cả terminal và file để có thể theo dõi hoặc resume.
    log_path.parent.mkdir(parents=True, exist_ok=True)

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

    setup_line = (
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SETUP | "
        f"algorithms={','.join(ALGORITHMS)} | seeds={args.num_seeds} | "
        f"population={args.population_size} | NFE={args.budget}"
    )
    print(setup_line, flush=True)
    if args.no_resume or not log_path.exists():
        log_path.write_text(setup_line + "\n", encoding="utf-8")
    else:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(setup_line + " | RESUME\n")

    report = ExperimentRunner(
        ExperimentConfig(
            instance_path=args.instance,
            seeds=seeds,
            population_size=args.population_size,
            nfe_budget=args.budget,
            algorithms=ALGORITHMS,
            start_time=args.start_time,
            departure_time_mode="optimize",
            allow_proxy=False,
            checkpoint_path=checkpoint_path,
            resume=not args.no_resume,
        ),
        progress_callback=progress,
    ).run()

    for path in report.save(output):
        print(f"Đã lưu: {path}")
    print(f"Đã lưu log: {log_path}")
    print(f"Đã lưu checkpoint: {checkpoint_path}")
    print(json.dumps(report.summary, indent=2, ensure_ascii=False))

    if not args.no_plots:
        for path in generate_plots(output):
            print(f"Đã lưu ảnh: {path}")


if __name__ == "__main__":
    main()
