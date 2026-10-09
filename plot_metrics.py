"""
plot_metrics.py (수정본)
Plot average flow time and total fleet travel distance across order arrival
rates (lambda), one grouped bar chart per metric, one bar per policy within
each lambda group. Error bars show the [min, max] range observed over the
20 evaluation instances (seeds) at that (policy, lambda) combination.

Produces two independent figures, each saved as both PDF and PNG:
  - fig/flow_time_by_lambda.pdf / .png
  - fig/total_distance_by_lambda.pdf / .png
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------
# 논문 본문 서체(Times 계열 serif)에 맞춘 전역 폰트 설정
# ---------------------------------------------------------------
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [
    "Times New Roman", "Nimbus Roman No9 L", "Nimbus Roman",
    "Liberation Serif", "DejaVu Serif",
]
plt.rcParams["mathtext.fontset"] = "stix"  # Times 스타일 수식 폰트
plt.rcParams["axes.unicode_minus"] = False

CSV_PATH = "eval_all_lambdas.csv"
OUT_DIR = "fig"

# orders/sec -> orders/hour, matching the lambda grid used in run.py
LAM_MAP = {
    700 / 3600: 700,
    800 / 3600: 800,
    900 / 3600: 900,
    1000 / 3600: 1000,
}
LAM_VALUES = [700, 800, 900, 1000]

LABELS_ORDER = [
    "RL", "BL1_FullBuffer", "BL2_Partial_k3",
    "BL3_Random", "BL4_Nearest", "BL5_MinInsertion",
]
DISPLAY_NAMES = [
    "RL (proposed)", "BL1 Full-Buffer", "BL2 Partial-Buffer",
    "BL3 Random", "BL4 Nearest", "BL5 Min-Insertion",
]
COLORS = ["#0d1b4c", "#3b5fc4", "#8fb8f0", "#bfe8e3", "#3fae7f", "#e8b93f"]

BAR_WIDTH = 0.13
GROUP_GAP = 0.35

# 확대된 폰트 크기
LABEL_FONTSIZE = 15
TICK_FONTSIZE = 13
LEGEND_FONTSIZE = 11


def _match_lambda(lam_str: str) -> int:
    lam = float(lam_str)
    closest = min(LAM_MAP, key=lambda k: abs(k - lam))
    return LAM_MAP[closest]


def load_eval_results(path: str):
    with open(path) as f:
        rows = list(csv.DictReader(f))

    agg = defaultdict(list)
    for row in rows:
        lam_hr = _match_lambda(row["lam"])
        agg[(row["label"], lam_hr)].append(row)
    return agg


def compute_stats(agg, field: str, scale: float = 1.0):
    """Return {label: {'mean': [...], 'lo_err': [...], 'hi_err': [...]}} over LAM_VALUES."""
    stats = {}
    for label in LABELS_ORDER:
        means, lo_err, hi_err = [], [], []
        for lam_hr in LAM_VALUES:
            vals = np.array(
                [float(r[field]) for r in agg[(label, lam_hr)]]
            ) * scale
            m, lo, hi = vals.mean(), vals.min(), vals.max()
            means.append(m)
            lo_err.append(m - lo)
            hi_err.append(hi - m)
        stats[label] = {"mean": means, "lo_err": lo_err, "hi_err": hi_err}
    return stats


def plot_grouped_bars(stats, ylabel: str, out_pdf: str, out_png: str, legend_loc: str = "upper right"):
    n_lam = len(LAM_VALUES)
    n_alg = len(LABELS_ORDER)
    x_base = np.arange(n_lam) * (n_alg * BAR_WIDTH + GROUP_GAP)

    fig, ax = plt.subplots(figsize=(9, 6))

    for i, (label, name) in enumerate(zip(LABELS_ORDER, DISPLAY_NAMES)):
        xs = x_base + i * BAR_WIDTH
        means = stats[label]["mean"]
        yerr = [stats[label]["lo_err"], stats[label]["hi_err"]]
        ax.bar(
            xs, means, width=BAR_WIDTH, label=name,
            color=COLORS[i], edgecolor="black", linewidth=0.4,
        )
        ax.errorbar(
            xs, means, yerr=yerr, fmt="none",
            ecolor="black", elinewidth=1.0, capsize=3, capthick=1.0,
        )

    ax.set_xticks(x_base + BAR_WIDTH * (n_alg - 1) / 2)
    ax.set_xticklabels([f"{v}/hr" for v in LAM_VALUES], fontsize=TICK_FONTSIZE)
    ax.set_xlabel("Order arrival rate $\\lambda$ (orders/hour)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    # 그래프별로 막대와 겹치지 않는 위치에 범례 배치
    ax.legend(loc=legend_loc, fontsize=LEGEND_FONTSIZE, ncol=2, frameon=True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    agg = load_eval_results(CSV_PATH)

    flow_stats = compute_stats(agg, "avg_flow_time", scale=1.0)
    plot_grouped_bars(
        flow_stats,
        ylabel="Avg. flow time (sec)",
        out_pdf=os.path.join(OUT_DIR, "flow_time_by_lambda.pdf"),
        out_png=os.path.join(OUT_DIR, "flow_time_by_lambda.png"),
        legend_loc="upper right",
    )

    dist_stats = compute_stats(agg, "total_distance", scale=1.0 / 1000.0)
    plot_grouped_bars(
        dist_stats,
        ylabel="Total fleet travel distance (km)",
        out_pdf=os.path.join(OUT_DIR, "total_distance_by_lambda.pdf"),
        out_png=os.path.join(OUT_DIR, "total_distance_by_lambda.png"),
        legend_loc="upper left",
    )

    print(f"Saved figures to {OUT_DIR}/:")
    print("  flow_time_by_lambda.pdf / .png")
    print("  total_distance_by_lambda.pdf / .png")