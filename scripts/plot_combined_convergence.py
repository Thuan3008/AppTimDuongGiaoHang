"""Script vẽ biểu đồ hội tụ kết hợp 3 quy mô (n=25, n=50, n=100) chuẩn báo chí khoa học (Colab/Matplotlib/Seaborn).

Chữ to, nét đậm, trục rõ ràng, không dùng đồ họa thô sơ kiểu PIL nháp.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

# Tránh lỗi ghi cache của matplotlib trên Windows
os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "mpl_convergence")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

# Thiết lập phong cách học thuật hiện đại
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "Tahoma"]
plt.rcParams["axes.edgecolor"] = "#334155"
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["mathtext.fontset"] = "dejavusans"


ALGO_CONFIG = {
    "RAPO-2709": {
        "label": "RAPO",
        "color": "#059669",  # Xanh ngọc lục bảo nổi bật
        "linewidth": 3.4,
        "zorder": 6,
        "linestyle": "-",
    },
    "GA-LS": {
        "label": "GA-LS",
        "color": "#ea580c",  # Cam đậm
        "linewidth": 2.4,
        "zorder": 5,
        "linestyle": "-",
    },
    "GA": {
        "label": "GA",
        "color": "#8b5cf6",  # Tím
        "linewidth": 2.2,
        "zorder": 4,
        "linestyle": "-",
    },
    "ARO-Original": {
        "label": "ARO-Original",
        "color": "#2563eb",  # Xanh dương
        "linewidth": 2.2,
        "zorder": 3,
        "linestyle": "-",
    },
    "POA-Original": {
        "label": "POA-Original",
        "color": "#dc2626",  # Đỏ
        "linewidth": 2.2,
        "zorder": 2,
        "linestyle": "-",
    },
}


def load_convergence_curves(report_path: Path, points: int = 501):
    data = json.loads(report_path.read_text(encoding="utf-8"))
    runs = [r for r in data.get("runs", []) if r.get("status") == "completed"]
    budget = data.get("protocol", {}).get("nfe_budget")

    grouped = {}
    for r in runs:
        algo = r["algorithm"]
        h = r.get("convergence") or []
        if h:
            if algo not in grouped:
                grouped[algo] = []
            grouped[algo].append([float(x) for x in h])

    curves = {}
    for algo, hists in grouped.items():
        arr = []
        for h in hists:
            indices = np.linspace(0, len(h) - 1, points).astype(int)
            arr.append([h[i] for i in indices])
        curves[algo] = np.mean(arr, axis=0)

    return curves, budget


def plot_horizontal(reports, output_path: Path):
    """Vẽ 3 subplots nằm ngang (1 x 3) chuẩn bài báo IEEE/Elsevier."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.2), dpi=300)

    for i, (key, info) in enumerate(reports.items()):
        ax = axes[i]
        curves, budget = load_convergence_curves(info["path"])
        x_nfe = np.linspace(0, budget, len(next(iter(curves.values()))))

        for algo_name, cfg in ALGO_CONFIG.items():
            if algo_name in curves:
                ax.plot(
                    x_nfe,
                    curves[algo_name],
                    label=cfg["label"],
                    color=cfg["color"],
                    linewidth=cfg["linewidth"],
                    linestyle=cfg["linestyle"],
                    zorder=cfg["zorder"],
                )

        ax.set_title(info["title"], fontsize=15, fontweight="bold", pad=12, color="#0f172a")
        ax.set_xlabel("Ngân sách đánh giá (NFE)", fontsize=13, fontweight="bold", labelpad=8, color="#1e293b")
        if i == 0:
            ax.set_ylabel("Fitness trung bình (Duration - giây)", fontsize=13, fontweight="bold", labelpad=8, color="#1e293b")
        else:
            ax.set_ylabel("Fitness trung bình (giây)", fontsize=12, fontweight="bold", labelpad=6, color="#475569")

        ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f"{int(val):d}"))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f"{int(val):,}"))
        ax.tick_params(axis="both", which="major", labelsize=12, length=5, width=1.2)
        ax.grid(True, linestyle="--", alpha=0.6, color="#cbd5e1")
        ax.set_facecolor("#f8fafc")

        # Đánh dấu BKS nếu có
        if "bks" in info:
            ax.axhline(info["bks"], color="#475569", linestyle=":", linewidth=1.5, alpha=0.8)
            ax.text(
                budget * 0.98,
                info["bks"] * 1.01,
                f"BKS: {info['bks']:,}",
                color="#475569",
                fontsize=11,
                fontweight="bold",
                ha="right",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.9),
            )

    # Legend dùng chung ở trên đầu
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        fontsize=13,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#94a3b8",
        shadow=True,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_vertical(reports, output_path: Path):
    """Vẽ 3 subplots xếp dọc (3 x 1) cho trang tài liệu khổ đứng."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 14), dpi=300)

    for i, (key, info) in enumerate(reports.items()):
        ax = axes[i]
        curves, budget = load_convergence_curves(info["path"])
        x_nfe = np.linspace(0, budget, len(next(iter(curves.values()))))

        for algo_name, cfg in ALGO_CONFIG.items():
            if algo_name in curves:
                ax.plot(
                    x_nfe,
                    curves[algo_name],
                    label=cfg["label"],
                    color=cfg["color"],
                    linewidth=cfg["linewidth"],
                    linestyle=cfg["linestyle"],
                    zorder=cfg["zorder"],
                )

        ax.set_title(info["title"], fontsize=14, fontweight="bold", pad=8, color="#0f172a")
        ax.set_xlabel("Ngân sách đánh giá (NFE)", fontsize=13, fontweight="bold", labelpad=6, color="#1e293b")
        ax.set_ylabel("Fitness trung bình (s)", fontsize=13, fontweight="bold", labelpad=6, color="#1e293b")

        ax.xaxis.set_major_locator(ticker.MaxNLocator(7))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f"{int(val):d}"))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f"{int(val):,}"))
        ax.tick_params(axis="both", which="major", labelsize=12, length=5, width=1.2)
        ax.grid(True, linestyle="--", alpha=0.6, color="#cbd5e1")
        ax.set_facecolor("#f8fafc")

        if i == 0:
            ax.legend(
                loc="upper right",
                fontsize=12,
                frameon=True,
                facecolor="#ffffff",
                edgecolor="#cbd5e1",
            )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    base_dir = Path("experiments/rapo_contribution/results")
    reports = {
        "n25": {
            "path": base_dir / "n25/stochastic_report.json",
            "title": "(a) Quy mô n = 25 Khách hàng (Ngân sách: 290 NFE · 30 Seeds)",
            "bks": 21425.5,
        },
        "n50": {
            "path": base_dir / "n50/nfe_580/stochastic_report.json",
            "title": "(b) Quy mô n = 50 Khách hàng (Ngân sách: 580 NFE · 30 Seeds)",
            "bks": 56491.3,
        },
        "n100": {
            "path": base_dir / "n100/nfe_1060/partial_seeds_00_09/stochastic_report.json",
            "title": "(c) Quy mô n = 100 Khách hàng (Ngân sách: 1060 NFE · 10 Seeds)",
            "bks": 101499.3,
        },
    }

    out_horiz = Path("results/comparison/convergence_comparison_3scales_horizontal.png")
    out_vert = Path("results/comparison/convergence_comparison_3scales_vertical.png")

    print("[1/2] Generating horizontal 3-panel figure...")
    plot_horizontal(reports, out_horiz)

    print("[2/2] Generating vertical 3-panel figure...")
    plot_vertical(reports, out_vert)

    # Copy ảnh sang thư mục artifact để nhúng trực tiếp xem trong chat
    artifact_dir = Path(r"C:\Users\thuan\.gemini\antigravity\brain\7db38a9e-84ba-4742-9161-cf1548ab8ad3")
    if artifact_dir.exists():
        import shutil
        shutil.copy2(out_horiz, artifact_dir / "convergence_comparison_3scales_horizontal.png")
        shutil.copy2(out_vert, artifact_dir / "convergence_comparison_3scales_vertical.png")
        print("Copied to artifact directory successfully.")


if __name__ == "__main__":
    main()
