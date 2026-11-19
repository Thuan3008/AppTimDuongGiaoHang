"""Tạo các ảnh PNG so sánh từ report ExperimentRunner."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]


def _font(size: int, bold: bool = False):
    candidates = (
        ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf"]
        if bold
        else ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"]
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _canvas(title: str, width: int = 1600, height: int = 900):
    image = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(image)
    draw.text((55, 30), title, fill="#111827", font=_font(32, True))
    return image, draw


def _axes(draw, left=130, top=110, right=1510, bottom=790):
    draw.line((left, top, left, bottom), fill="#475569", width=2)
    draw.line((left, bottom, right, bottom), fill="#475569", width=2)
    return left, top, right, bottom


def _scale(value, low, high, start, end):
    if math.isclose(low, high):
        return (start + end) / 2
    return start + (value - low) / (high - low) * (end - start)


def _bar_chart(path: Path, title: str, values: dict[str, float], unit: str):
    image, draw = _canvas(title)
    left, top, right, bottom = _axes(draw)
    finite = [value for value in values.values() if math.isfinite(value)]
    if not finite:
        image.save(path)
        return
    low, high = min(0.0, min(finite)), max(finite) * 1.12
    slot = (right - left) / max(1, len(values))
    for index, (name, value) in enumerate(values.items()):
        x0 = left + slot * index + slot * 0.18
        x1 = left + slot * (index + 1) - slot * 0.18
        y = _scale(value, low, high, bottom, top)
        draw.rectangle((x0, y, x1, bottom), fill=COLORS[index % len(COLORS)])
        label = name
        draw.text(((x0 + x1) / 2, bottom + 18), label, fill="#111827", anchor="ma", font=_font(19))
        draw.text(((x0 + x1) / 2, max(top, y - 28)), f"{value:.2f}", fill="#111827", anchor="ms", font=_font(18))
    draw.text((right, bottom + 55), unit, fill="#475569", anchor="ra", font=_font(17))
    image.save(path)


def _fitness_by_seed(path: Path, rows: list[dict]):
    image, draw = _canvas("Fitness theo seed · thấp hơn tốt hơn")
    left, top, right, bottom = _axes(draw)
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "completed" and row.get("fitness") is not None:
            grouped[str(row["algorithm"])].append((int(row["seed"]), float(row["fitness"])))
    values = [value for pairs in grouped.values() for _, value in pairs]
    if not values:
        image.save(path)
        return
    low, high = min(values) * 0.98, max(values) * 1.02
    seeds = sorted({seed for pairs in grouped.values() for seed, _ in pairs})
    for tick in seeds:
        x = _scale(tick, min(seeds), max(seeds) if len(seeds) > 1 else min(seeds) + 1, left, right)
        draw.line((x, bottom, x, bottom + 8), fill="#475569")
        draw.text((x, bottom + 18), str(tick), fill="#111827", anchor="ma", font=_font(15))
    for index, (algorithm, pairs) in enumerate(grouped.items()):
        points = [
            (
                _scale(seed, min(seeds), max(seeds) if len(seeds) > 1 else min(seeds) + 1, left, right),
                _scale(value, low, high, bottom, top),
            )
            for seed, value in sorted(pairs)
        ]
        if len(points) > 1:
            draw.line(points, fill=COLORS[index % len(COLORS)], width=4)
        for point in points:
            draw.ellipse((point[0] - 6, point[1] - 6, point[0] + 6, point[1] + 6), fill=COLORS[index % len(COLORS)])
        draw.text((right - 10, top + 35 * index), algorithm, fill=COLORS[index % len(COLORS)], anchor="ra", font=_font(18, True))
    draw.text((left, top - 35), f"Fitness: {low:.0f}–{high:.0f}", fill="#475569", font=_font(17))
    image.save(path)


def _normalized_curve(histories: list[list[float]], points: int = 501) -> list[float]:
    """Average best-so-far histories at equal fractions of recorded progress."""
    if not histories or points < 2:
        raise ValueError("Cần ít nhất một history và hai điểm vẽ")
    return [
        sum(
            history[min(len(history) - 1, int(index / (points - 1) * (len(history) - 1)))]
            for history in histories
        ) / len(histories)
        for index in range(points)
    ]


def _convergence(path: Path, rows: list[dict]):
    image, draw = _canvas("Hội tụ trung bình theo tiến độ ghi nhận (%)")
    left, top, right, bottom = _axes(draw)
    grouped: dict[str, list[list[float]]] = defaultdict(list)
    for row in rows:
        history = row.get("convergence") or []
        if row.get("status") == "completed" and history:
            grouped[str(row["algorithm"])].append([float(value) for value in history])
    if not grouped:
        draw.text((left, top + 20), "Report không có convergence history.", fill="#b91c1c", font=_font(22))
        image.save(path)
        return
    curves = {algorithm: _normalized_curve(histories) for algorithm, histories in grouped.items()}
    values = [value for curve in curves.values() for value in curve]
    low, high = min(values) * 0.98, max(values) * 1.02
    for tick in (0, 25, 50, 75, 100):
        x = _scale(tick, 0, 100, left, right)
        draw.line((x, bottom, x, bottom + 8), fill="#475569")
        draw.text((x, bottom + 18), str(tick), fill="#111827", anchor="ma", font=_font(16))
    for index, (algorithm, curve) in enumerate(curves.items()):
        points = [
            (_scale(position, 0, len(curve) - 1, left, right), _scale(value, low, high, bottom, top))
            for position, value in enumerate(curve)
        ]
        if len(points) > 1:
            draw.line(points, fill=COLORS[index % len(COLORS)], width=3)
        draw.text((right - 10, top + 30 * index), algorithm, fill=COLORS[index % len(COLORS)], anchor="ra", font=_font(17, True))
    draw.text((left, bottom + 50), "Tiến độ ghi nhận (%)", fill="#475569", font=_font(18))
    draw.text(
        (left, bottom + 75),
        "Trục X chuẩn hóa theo số điểm convergence; không biểu diễn NFE chính xác.",
        fill="#475569", font=_font(15),
    )
    draw.text((left, top - 35), "Fitness trung bình", fill="#475569", font=_font(18))
    image.save(path)


def generate_plots(report_path: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    """Tạo bộ PNG dùng trực tiếp trong báo cáo thực nghiệm."""
    report_path = Path(report_path)
    output_dir = Path(output_dir or report_path.with_name(report_path.stem + "_plots"))
    output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report.get("runs", [])
    summary = {row["algorithm"]: row for row in report.get("summary", [])}
    outputs: list[Path] = []
    for key, label, unit in (
        ("mean", "Fitness trung bình theo thuật toán", "fitness"),
        ("best", "Best fitness theo thuật toán", "fitness"),
        ("mean_gap_to_bks_percent", "Gap BKS trung bình", "%"),
        ("runtime_mean_seconds", "Runtime trung bình", "giây"),
    ):
        path = output_dir / f"{key}.png"
        _bar_chart(path, label, {name: float(row[key]) for name, row in summary.items() if row.get(key) is not None}, unit)
        outputs.append(path)
    fitness_path = output_dir / "fitness_by_seed.png"
    _fitness_by_seed(fitness_path, rows)
    outputs.append(fitness_path)
    convergence_path = output_dir / "convergence_mean.png"
    _convergence(convergence_path, rows)
    outputs.append(convergence_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Vẽ ảnh so sánh từ report TDVRP")
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    for path in generate_plots(args.report, args.output_dir):
        print(f"Đã lưu ảnh: {path}")


if __name__ == "__main__":
    main()
