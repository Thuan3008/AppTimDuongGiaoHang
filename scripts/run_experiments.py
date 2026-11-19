"""CLI chạy benchmark nhiều seed theo một protocol công bằng."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.plot_experiment_report import generate_plots
from experiments.runner import ExperimentConfig, ExperimentRunner, PRIMARY_ALGORITHM, SUPPORTED_ALGORITHMS


DEFAULT_INSTANCE = ROOT / "data/TDVRP/hong_kong/n=50/poryos-hong_kong-n50-hyb/bpr-heavy/poryos-hong_kong-n50-hyb-bpr-heavy.vrp.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy TDVRP benchmark nhiều seed")
    parser.add_argument("--instance", type=Path, default=DEFAULT_INSTANCE)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--budget", type=int, default=290)
    parser.add_argument("--algorithms", nargs="+", choices=SUPPORTED_ALGORITHMS, default=[PRIMARY_ALGORITHM])
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--departure-mode", choices=("fixed", "optimize"), default="optimize")
    parser.add_argument("--departure-step", type=float, default=900.0)
    parser.add_argument("--allow-proxy", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results/multi_seed_experiment.json")
    parser.add_argument("--log", type=Path, default=None, help="File log tiến độ; mặc định cùng tên output với đuôi .log")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint JSONL; mặc định cùng tên output với đuôi .checkpoint.jsonl")
    parser.add_argument("--no-resume", action="store_true", help="Bỏ qua checkpoint cũ và chạy lại từ đầu")
    parser.add_argument("--no-plots", action="store_true", help="Không tạo ảnh PNG sau khi hoàn tất")
    args = parser.parse_args()
    if args.num_seeds <= 0:
        parser.error("--num-seeds phải dương")
    seeds = tuple(range(args.seed_start, args.seed_start + args.num_seeds))
    log_path = args.log or args.output.with_suffix(".log")
    checkpoint_path = args.checkpoint or args.output.with_name(args.output.stem + ".checkpoint.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def progress(event: dict) -> None:
        """In và lưu tiến độ từng run để người dùng theo dõi trực tiếp."""
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
            f"{event['algorithm']} seed={event['seed']} | {event['status']} | {detail}"
        )
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    setup_line = (
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SETUP | "
        f"algorithms={','.join(args.algorithms)} | seeds={args.num_seeds} | "
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
            algorithms=tuple(args.algorithms),
            start_time=args.start_time,
            departure_time_mode=args.departure_mode,
            departure_time_step=args.departure_step,
            allow_proxy=args.allow_proxy,
            checkpoint_path=checkpoint_path,
            resume=not args.no_resume,
        ),
        progress_callback=progress,
    ).run()
    paths = report.save(args.output)
    print(json.dumps(report.summary, indent=2, ensure_ascii=False))
    for path in paths:
        print(f"Đã lưu: {path}")
    print(f"Đã lưu log tiến độ: {log_path}")
    if not args.no_plots:
        for path in generate_plots(args.output):
            print(f"Đã lưu ảnh: {path}")


if __name__ == "__main__":
    main()
