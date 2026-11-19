"""Merge the two n=25 RAPO ablation shards and make a compact PDF report."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image as PdfImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("RAPO-2709", "ARO-only", "POA-only", "RAPO-no-AOS")
SOURCES = (
    ROOT / "results/rapo_ablation_n25_seed09/rapo_ablation_n25.json",
    ROOT / "results/rapo_ablation_n25_seed1029/rapo_ablation_n25_seed1029.json",
)
OUTPUT = ROOT / "results/rapo_ablation_n25_merged"
PALETTE = {
    "RAPO-2709": "#138a60",
    "ARO-only": "#2563eb",
    "POA-only": "#e4572e",
    "RAPO-no-AOS": "#8b5cf6",
}


def _summarize(rows, bks_cost):
    summary = []
    by_variant = {}
    for variant in VARIANTS:
        group = [row for row in rows if row["variant"] == variant]
        fitness = [float(row["fitness"]) for row in group]
        by_variant[variant] = {int(row["seed"]): float(row["fitness"]) for row in group}
        mean = statistics.fmean(fitness)
        summary.append({
            "variant": variant, "runs": len(group), "best": min(fitness), "mean": mean,
            "median": statistics.median(fitness),
            "standard_deviation": statistics.stdev(fitness),
            "mean_gap_to_bks_percent": (mean - bks_cost) / bks_cost * 100,
            "mean_runtime_seconds": statistics.fmean(row["runtime_seconds"] for row in group),
        })
    full = by_variant["RAPO-2709"]
    paired = []
    for variant in VARIANTS[1:]:
        deltas = [by_variant[variant][seed] - full[seed] for seed in sorted(full)]
        paired.append({
            "variant": variant, "paired_seeds": len(deltas),
            "full_wins": sum(delta > 1e-9 for delta in deltas),
            "ties": sum(abs(delta) <= 1e-9 for delta in deltas),
            "variant_wins": sum(delta < -1e-9 for delta in deltas),
            "mean_variant_minus_full_seconds": statistics.fmean(deltas),
        })
    return summary, paired


def _font(size: int, bold: bool = False):
    filename = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / filename), size)


def _load_and_validate():
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in SOURCES]
    base = {k: v for k, v in reports[0]["protocol"].items() if k != "seeds"}
    for report in reports:
        if {k: v for k, v in report["protocol"].items() if k != "seeds"} != base:
            raise ValueError("Hai đợt chạy khác protocol; không thể gộp")
    rows = [row for report in reports for row in report["runs"]]
    expected_seeds = sorted(seed for report in reports for seed in report["protocol"]["seeds"])
    if expected_seeds != list(range(30)):
        raise ValueError("Cần đúng seed 0–29, không trùng")
    expected = {(variant, seed) for variant in VARIANTS for seed in expected_seeds}
    actual = [(row["variant"], int(row["seed"])) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Thiếu hoặc trùng cặp biến thể–seed")
    for seed in expected_seeds:
        if len({row["initial_population_sha256"] for row in rows if row["seed"] == seed}) != 1:
            raise ValueError(f"P0 không khớp tại seed {seed}")
    nfe = int(base["nfe_budget"])
    for row in rows:
        if row["nfe"] != nfe or len(row["convergence"]) != nfe or not row["feasible"]:
            raise ValueError(f"NFE/convergence/feasibility không hợp lệ: {row['variant']} {row['seed']}")
        if not math.isfinite(row["fitness"]):
            raise ValueError("Fitness không hữu hạn")
    rows.sort(key=lambda row: (row["seed"], VARIANTS.index(row["variant"])))
    protocol = {**reports[0]["protocol"], "seeds": expected_seeds}
    summary, comparisons = _summarize(rows, protocol["bks_cost"])
    return {"protocol": protocol, "runs": rows, "summary": summary,
            "paired_comparisons": comparisons,
            "source_reports": [str(path) for path in SOURCES]}


def _csv(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _base_chart(title: str, subtitle: str, legend=VARIANTS):
    image = Image.new("RGB", (1600, 850), "#ffffff")
    draw = ImageDraw.Draw(image)
    draw.text((95, 42), title, font=_font(34, True), fill="#172033")
    draw.text((95, 92), subtitle, font=_font(19), fill="#526075")
    left, top, right, bottom = 150, 180, 1400, 690
    for index in range(6):
        y = top + (bottom - top) * index / 5
        draw.line((left, y, right, y), fill="#e6ebf2", width=2)
    draw.line((left, top, left, bottom), fill="#65748b", width=2)
    draw.line((left, bottom, right, bottom), fill="#65748b", width=2)
    for index, name in enumerate(legend):
        x = 150 + (index % 2) * 570
        y = 770 + (index // 2) * 29
        draw.line((x, y + 11, x + 35, y + 11), fill=PALETTE[name], width=5)
        draw.text((x + 45, y), name, font=_font(18), fill="#263449")
    return image, draw, (left, top, right, bottom)


def _y(value: float, low: float, high: float, top: int, bottom: int) -> float:
    return bottom - (value - low) / (high - low) * (bottom - top)


def _axis_labels(draw, bounds, low, high, xlabel):
    left, top, right, bottom = bounds
    for index in range(6):
        value = low + (high - low) * (5 - index) / 5
        y = top + (bottom - top) * index / 5
        draw.text((left - 12, y), f"{value:,.0f}", font=_font(17), fill="#526075", anchor="rm")
    draw.text(((left + right) / 2, bottom + 52), xlabel,
              font=_font(19), fill="#526075", anchor="mm")


def _mean_fitness_chart(report, path):
    image, draw, bounds = _base_chart("Fitness trung bình và độ lệch chuẩn", "30 seed · fitness thấp hơn là tốt hơn")
    left, top, right, bottom = bounds
    summary = report["summary"]
    bks = report["protocol"]["bks_cost"]
    low = min(bks, *(row["mean"] - row["standard_deviation"] for row in summary)) * .98
    high = max(row["mean"] + row["standard_deviation"] for row in summary) * 1.03
    _axis_labels(draw, bounds, low, high, "Biến thể")
    by = _y(bks, low, high, top, bottom)
    draw.line((left, by, right, by), fill="#888e99", width=3)
    draw.text((right - 5, by - 10), f"BKS {bks:,.1f}", font=_font(17), fill="#576272", anchor="rb")
    width = 165
    for index, row in enumerate(summary):
        x = left + 155 + index * 300
        value = row["mean"]
        y = _y(value, low, high, top, bottom)
        draw.rectangle((x - width / 2, y, x + width / 2, bottom), fill=PALETTE[row["variant"]])
        lo = _y(value - row["standard_deviation"], low, high, top, bottom)
        hi = _y(value + row["standard_deviation"], low, high, top, bottom)
        draw.line((x, lo, x, hi), fill="#172033", width=4)
        draw.line((x - 17, lo, x + 17, lo), fill="#172033", width=4)
        draw.line((x - 17, hi, x + 17, hi), fill="#172033", width=4)
        draw.text((x, y - 14), f"{value:,.0f}", font=_font(19, True), fill="#172033", anchor="mb")
    image.save(path)


def _paired_chart(report, path):
    image, draw, bounds = _base_chart("Chênh lệch fitness theo seed", "Giá trị = biến thể trừ RAPO đầy đủ · âm nghĩa là biến thể tốt hơn", VARIANTS[1:])
    left, top, right, bottom = bounds
    by_key = {(row["variant"], row["seed"]): row["fitness"] for row in report["runs"]}
    series = {variant: [by_key[(variant, seed)] - by_key[("RAPO-2709", seed)]
                        for seed in range(30)] for variant in VARIANTS[1:]}
    values = [value for curve in series.values() for value in curve] + [0]
    low, high = min(values) - 180, max(values) + 180
    _axis_labels(draw, bounds, low, high, "Seed 0–29")
    zero = _y(0, low, high, top, bottom)
    draw.line((left, zero, right, zero), fill="#172033", width=3)
    for tick in (0, 5, 10, 15, 20, 25, 29):
        x = left + (right - left) * tick / 29
        draw.text((x, bottom + 15), str(tick), font=_font(17), fill="#526075", anchor="mt")
    for variant, curve in series.items():
        points = [(left + (right - left) * seed / 29, _y(value, low, high, top, bottom))
                  for seed, value in enumerate(curve)]
        draw.line(points, fill=PALETTE[variant], width=4, joint="curve")
        for x, y in points:
            draw.ellipse((x-3, y-3, x+3, y+3), fill=PALETTE[variant])
    image.save(path)


def _convergence_chart(report, path):
    image, draw, bounds = _base_chart("Hội tụ fitness trung bình theo NFE", "Trung bình 30 seed · mọi history có đúng 290 điểm NFE")
    left, top, right, bottom = bounds
    curves = {variant: [statistics.fmean(row["convergence"][step]
                            for row in report["runs"] if row["variant"] == variant)
                        for step in range(290)] for variant in VARIANTS}
    values = [value for curve in curves.values() for value in curve]
    low, high = min(values) * .985, max(values) * 1.015
    _axis_labels(draw, bounds, low, high, "NFE")
    for tick in (1, 50, 100, 150, 200, 250, 290):
        x = left + (right - left) * (tick - 1) / 289
        draw.text((x, bottom + 15), str(tick), font=_font(17), fill="#526075", anchor="mt")
    for variant, curve in curves.items():
        points = [(left + (right - left) * step / 289, _y(value, low, high, top, bottom))
                  for step, value in enumerate(curve)]
        draw.line(points, fill=PALETTE[variant], width=5, joint="curve")
    image.save(path)


def _pdf(report, output: Path):
    pdfmetrics.registerFont(TTFont("ArialVN", "C:/Windows/Fonts/arial.ttf"))
    pdfmetrics.registerFont(TTFont("ArialVNBold", "C:/Windows/Fonts/arialbd.ttf"))
    styles = getSampleStyleSheet()
    title = ParagraphStyle("titleVN", parent=styles["Title"], fontName="ArialVNBold", fontSize=17,
                           leading=23, textColor=colors.HexColor("#172033"), spaceAfter=12)
    h = ParagraphStyle("hVN", parent=styles["Heading2"], fontName="ArialVNBold", fontSize=11,
                       leading=15, textColor=colors.HexColor("#172033"), spaceBefore=14, spaceAfter=7)
    body = ParagraphStyle("bodyVN", parent=styles["BodyText"], fontName="ArialVN", fontSize=9.4,
                          leading=14, spaceAfter=8)
    small = ParagraphStyle("smallVN", parent=body, fontSize=8.2, leading=12)
    centered = ParagraphStyle("centerVN", parent=body, alignment=TA_CENTER, spaceAfter=4)
    doc = SimpleDocTemplate(str(output), pagesize=(21*cm, 29.7*cm), rightMargin=2*cm,
                            leftMargin=2*cm, topMargin=1.8*cm, bottomMargin=1.7*cm)
    story = [Paragraph("Báo cáo gộp ablation RAPO n25", title),
             Paragraph("Hong Kong · poryos-hong_kong-n25-hyb-bpr-heavy · 30 seed (0–29)", body),
             Paragraph("Kết quả chính", h)]
    best = min(report["summary"], key=lambda row: row["mean"])
    story.append(Paragraph(
        f"Trong bốn biến thể, {best['variant']} có fitness trung bình thấp nhất "
        f"({best['mean']:,.2f}). Đây là kết quả của một instance n=25, P=10 và ngân sách 290 NFE; "
        "không suy rộng thành kết luận cho toàn bộ bộ dữ liệu.", body))
    story.append(Paragraph("Thiết kế và kiểm tra dữ liệu", h))
    story.append(Paragraph(
        "Hai đợt seed 0–9 và 10–29 có cùng protocol. Báo cáo gộp 120 run: 30 seed × 4 biến thể. "
        "Mọi run khả thi, dừng đúng 290 NFE và có 290 điểm hội tụ. Trong từng seed, bốn biến thể "
        "dùng cùng quần thể khởi tạo P0. Fitness là tổng thời gian di chuyển và phục vụ; giá trị thấp hơn tốt hơn.", body))
    story.append(Paragraph(
        "Tên biến thể: RAPO-2709 là bản đầy đủ; ARO-only chỉ dùng nhánh ARO; POA-only chỉ dùng "
        "nhánh POA; RAPO-no-AOS giữ ba nhánh nhưng chọn toán tử Elite với trọng số cố định. "
        "ARO-only và POA-only cũng tắt Elite, nên đây không phải hai phép chỉ bỏ riêng POA hoặc ARO.", small))
    story.append(Paragraph("Tổng hợp 30 seed", h))
    header = ["Biến thể", "Mean", "Median", "SD", "Gap BKS", "Runtime"]
    data = [header] + [[row["variant"], f"{row['mean']:,.1f}", f"{row['median']:,.1f}",
                        f"{row['standard_deviation']:,.1f}", f"{row['mean_gap_to_bks_percent']:.2f}%",
                        f"{row['mean_runtime_seconds']:.1f} s"] for row in report["summary"]]
    table = Table(data, colWidths=[3.4*cm, 2.4*cm, 2.4*cm, 1.8*cm, 2.3*cm, 2.4*cm], repeatRows=1)
    table.setStyle(TableStyle([("FONTNAME", (0,0), (-1,0), "ArialVNBold"),
                               ("FONTNAME", (0,1), (-1,-1), "ArialVN"),
                               ("FONTSIZE", (0,0), (-1,-1), 8.5),
                               ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#dfe8f1")),
                               ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f6f8fb")]),
                               ("GRID", (0,0), (-1,-1), .4, colors.HexColor("#d9d9d9")),
                               ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                               ("ALIGN", (1,1), (-1,-1), "RIGHT"),
                               ("TOPPADDING", (0,0), (-1,-1), 7),
                               ("BOTTOMPADDING", (0,0), (-1,-1), 7)]))
    story.extend([table, Spacer(1, 8), Paragraph(
        f"BKS = {report['protocol']['bks_cost']:,.2f}. Gap = (fitness trung bình − BKS) / BKS × 100%; "
        "runtime là trung bình giây mỗi run.", small)])
    story.append(Paragraph("So sánh ghép cặp với RAPO đầy đủ", h))
    paired_data = [["Biến thể", "Full thắng", "Hòa", "Biến thể thắng", "Δ fitness TB"]] + [
        [row["variant"], str(row["full_wins"]), str(row["ties"]), str(row["variant_wins"]),
         f"{row['mean_variant_minus_full_seconds']:+,.1f}"]
        for row in report["paired_comparisons"]]
    paired = Table(paired_data, colWidths=[3.4*cm, 2.4*cm, 1.5*cm, 3.2*cm, 3.2*cm], repeatRows=1)
    paired.setStyle(TableStyle([("FONTNAME", (0,0), (-1,0), "ArialVNBold"),
                                ("FONTNAME", (0,1), (-1,-1), "ArialVN"),
                                ("FONTSIZE", (0,0), (-1,-1), 8.5),
                                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#dfe8f1")),
                                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f6f8fb")]),
                                ("GRID", (0,0), (-1,-1), .4, colors.HexColor("#d9d9d9")),
                                ("TOPPADDING", (0,0), (-1,-1), 7),
                                ("BOTTOMPADDING", (0,0), (-1,-1), 7)]))
    story.extend([paired, Spacer(1, 7), Paragraph(
        "Δ fitness TB = fitness biến thể trừ fitness RAPO đầy đủ trên cùng seed. "
        "Số âm nghĩa là biến thể có fitness thấp hơn. Bảng chỉ mô tả kết quả ghép cặp, "
        "không phải kiểm định ý nghĩa thống kê.", small)])
    story.append(PageBreak())
    for filename, caption in [
        ("fitness_mean_sd.png", "Hình 1. Fitness trung bình ± độ lệch chuẩn; đường ngang là BKS."),
        ("paired_delta_by_seed.png", "Hình 2. Chênh lệch fitness của từng biến thể so với RAPO đầy đủ theo seed."),
        ("convergence_mean.png", "Hình 3. Hội tụ fitness trung bình trên 30 seed theo đúng số NFE đã ghi.")
    ]:
        story.append(PdfImage(str(OUTPUT / filename), width=16.4*cm, height=8.71*cm))
        story.append(Paragraph(caption, centered))
        story.append(Spacer(1, 9))
    story.extend([Paragraph("Nguồn dữ liệu", h), Paragraph(
        "Gộp từ hai report JSON ablation tại results/rapo_ablation_n25_seed09 và "
        "results/rapo_ablation_n25_seed1029. File merged.json, runs.csv, summary.csv và paired.csv "
        "được lưu cùng thư mục với báo cáo này.", small)])
    def footer(canvas, pdf_doc):
        canvas.setFont("ArialVN", 8)
        canvas.setFillColor(colors.HexColor("#65748b"))
        canvas.drawRightString(19*cm, 1*cm, f"Trang {pdf_doc.page}")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = _load_and_validate()
    (OUTPUT / "merged.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _csv(OUTPUT / "summary.csv", report["summary"])
    _csv(OUTPUT / "paired.csv", report["paired_comparisons"])
    _csv(OUTPUT / "runs.csv", [{**{k: v for k, v in row.items() if k not in {"convergence", "nfe_by_branch", "operator_uses"}},
                                 "nfe_by_branch": json.dumps(row["nfe_by_branch"]),
                                 "operator_uses": json.dumps(row["operator_uses"])} for row in report["runs"]])
    _mean_fitness_chart(report, OUTPUT / "fitness_mean_sd.png")
    _paired_chart(report, OUTPUT / "paired_delta_by_seed.png")
    _convergence_chart(report, OUTPUT / "convergence_mean.png")
    _pdf(report, OUTPUT / "bao_cao_ablation_n25_30_seed.pdf")
    print(OUTPUT / "bao_cao_ablation_n25_30_seed.pdf")


if __name__ == "__main__":
    main()
