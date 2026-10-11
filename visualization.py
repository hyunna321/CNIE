"""논문용 그래프 생성

- 학습 보상 수렴 곡선: training_curves_all_lambdas.csv -> fig/reward_convergence.pdf/.png
- λ별 평균 체류시간 / 총 이동거리 막대그래프: eval_all_lambdas.csv
  -> fig/flow_time_by_lambda.pdf/.png, fig/total_distance_by_lambda.pdf/.png

사용법:
    python visualization.py            # 전체
    python visualization.py reward     # 보상 수렴 곡선만
    python visualization.py metrics    # 막대그래프만
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from constant import (
    PLOT_FONT_FAMILY, PLOT_FONT_SERIF, PLOT_MATHTEXT_FONTSET,
    PLOT_LABEL_FONTSIZE, PLOT_TICK_FONTSIZE, PLOT_DPI, PLOT_LAM_VALUES, FIG_DIR,
    POLICY_LABELS,
    REWARD_PLOT_CSV_PATH, REWARD_PLOT_OUT_PDF, REWARD_PLOT_OUT_PNG,
    REWARD_PLOT_COLORS, REWARD_PLOT_SMOOTH_WINDOW, REWARD_PLOT_LEGEND_FONTSIZE,
    REWARD_PLOT_FIGSIZE,
    METRIC_PLOT_CSV_PATH, METRIC_PLOT_DISPLAY_NAMES, METRIC_PLOT_COLORS,
    METRIC_PLOT_BAR_WIDTH, METRIC_PLOT_GROUP_GAP, METRIC_PLOT_LEGEND_FONTSIZE,
    METRIC_PLOT_FIGSIZE,
)

plt.rcParams["font.family"] = PLOT_FONT_FAMILY
plt.rcParams["font.serif"] = PLOT_FONT_SERIF
plt.rcParams["mathtext.fontset"] = PLOT_MATHTEXT_FONTSET
plt.rcParams["axes.unicode_minus"] = False

# orders/s -> orders/hr
LAM_MAP = {hr / 3600: hr for hr in PLOT_LAM_VALUES}


def _style_axes(ax):
    """위/오른쪽 테두리 숨김

    입력:
        ax: matplotlib Axes
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, out_pdf: str, out_png: str):
    """그림을 PDF와 PNG로 저장 후 닫음

    입력:
        fig: matplotlib Figure
        out_pdf: PDF 저장 경로
        out_png: PNG 저장 경로
    """
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=PLOT_DPI)
    plt.close(fig)


# 학습 보상 수렴 곡선

def load_training_curves(path: str):
    """학습 곡선 CSV를 λ별로 묶어 읽음 (λ는 label("RL_lam{λ}_train")에서 추출)

    입력:
        path: CSV 경로
    출력:
        {λ(orders/hr): {"episode": [...], "reward": [...]}}
    """
    with open(path) as f:
        rows = list(csv.DictReader(f))

    data = defaultdict(lambda: defaultdict(list))
    for row in rows:
        lam_hr = int(row["label"].split("lam")[1].split("_")[0])
        data[lam_hr]["episode"].append(int(row["episode"]))
        data[lam_hr]["reward"].append(float(row["reward"]))
    return data


def rolling_mean(x, window: int = REWARD_PLOT_SMOOTH_WINDOW):
    """이동평균 (valid 구간만)

    입력:
        x: 1차원 수열
        window: 창 크기
    출력:
        길이 len(x) - window + 1 numpy 배열
    """
    x = np.asarray(x, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_reward_convergence(data, out_pdf=REWARD_PLOT_OUT_PDF, out_png=REWARD_PLOT_OUT_PNG):
    """λ별 원본 보상(흐리게)과 이동평균 곡선을 한 그림에 그려 PDF/PNG로 저장

    입력:
        data: load_training_curves 결과
        out_pdf: PDF 저장 경로
        out_png: PNG 저장 경로
    """
    fig, ax = plt.subplots(figsize=REWARD_PLOT_FIGSIZE)

    for lam_hr in PLOT_LAM_VALUES:
        episodes = data[lam_hr]["episode"]
        reward = data[lam_hr]["reward"]

        reward_smooth = rolling_mean(reward, REWARD_PLOT_SMOOTH_WINDOW)
        episodes_smooth = episodes[len(episodes) - len(reward_smooth):]

        ax.plot(episodes, reward, color=REWARD_PLOT_COLORS[lam_hr], alpha=0.15, linewidth=0.8)
        ax.plot(
            episodes_smooth, reward_smooth,
            color=REWARD_PLOT_COLORS[lam_hr], linewidth=2.0,
            label=f"$\\lambda$={lam_hr}/hr",
        )

    ax.set_xlabel("Training episode", fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_ylabel("Average reward per completed order", fontsize=PLOT_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PLOT_TICK_FONTSIZE)
    ax.legend(loc="lower right", fontsize=REWARD_PLOT_LEGEND_FONTSIZE)
    _style_axes(ax)
    _save(fig, out_pdf, out_png)


def plot_reward_figures(csv_path: str = REWARD_PLOT_CSV_PATH):
    """학습 곡선 CSV를 읽어 보상 수렴 곡선 저장

    입력:
        csv_path: 학습 곡선 CSV 경로
    """
    curves = load_training_curves(csv_path)
    plot_reward_convergence(curves)
    print(f"Saved {REWARD_PLOT_OUT_PDF} and {REWARD_PLOT_OUT_PNG}")


# λ별 정책 비교 막대그래프

def _match_lambda(lam_str: str) -> int:
    """CSV의 λ(orders/s) 문자열을 가장 가까운 그리드 값(orders/hr)으로 변환

    입력:
        lam_str: λ 문자열
    출력:
        λ (orders/hr)
    """
    lam = float(lam_str)
    closest = min(LAM_MAP, key=lambda k: abs(k - lam))
    return LAM_MAP[closest]


def load_eval_results(path: str):
    """평가 CSV를 (정책, λ)별로 묶어 읽음

    입력:
        path: CSV 경로
    출력:
        {(label, λ_hr): [CSV 행 dict, ...]}
    """
    with open(path) as f:
        rows = list(csv.DictReader(f))

    agg = defaultdict(list)
    for row in rows:
        lam_hr = _match_lambda(row["lam"])
        agg[(row["label"], lam_hr)].append(row)
    return agg


def compute_stats(agg, field: str, scale: float = 1.0):
    """(정책, λ)별 지표의 평균과 [min, max] 오차 막대 길이 계산

    입력:
        agg: load_eval_results 결과
        field: CSV 지표 열 이름
        scale: 값에 곱할 배율 (단위 변환용)
    출력:
        {label: {"mean": [...], "lo_err": [...], "hi_err": [...]}} (PLOT_LAM_VALUES 순서)
    """
    stats = {}
    for label in POLICY_LABELS:
        means, lo_err, hi_err = [], [], []
        for lam_hr in PLOT_LAM_VALUES:
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
    """λ를 그룹, 정책을 막대로 하는 묶음 막대그래프를 그려 PDF/PNG로 저장

    오차 막대: 평가 인스턴스(시드)들의 [min, max] 범위

    입력:
        stats: compute_stats 결과
        ylabel: y축 라벨
        out_pdf: PDF 저장 경로
        out_png: PNG 저장 경로
        legend_loc: 범례 위치
    """
    n_lam = len(PLOT_LAM_VALUES)
    n_alg = len(POLICY_LABELS)
    x_base = np.arange(n_lam) * (n_alg * METRIC_PLOT_BAR_WIDTH + METRIC_PLOT_GROUP_GAP)

    fig, ax = plt.subplots(figsize=METRIC_PLOT_FIGSIZE)

    for i, (label, name) in enumerate(zip(POLICY_LABELS, METRIC_PLOT_DISPLAY_NAMES)):
        xs = x_base + i * METRIC_PLOT_BAR_WIDTH
        means = stats[label]["mean"]
        yerr = [stats[label]["lo_err"], stats[label]["hi_err"]]
        ax.bar(
            xs, means, width=METRIC_PLOT_BAR_WIDTH, label=name,
            color=METRIC_PLOT_COLORS[i], edgecolor="black", linewidth=0.4,
        )
        ax.errorbar(
            xs, means, yerr=yerr, fmt="none",
            ecolor="black", elinewidth=1.0, capsize=3, capthick=1.0,
        )

    ax.set_xticks(x_base + METRIC_PLOT_BAR_WIDTH * (n_alg - 1) / 2)
    ax.set_xticklabels([f"{v}/hr" for v in PLOT_LAM_VALUES], fontsize=PLOT_TICK_FONTSIZE)
    ax.set_xlabel("Order arrival rate $\\lambda$ (orders/hour)", fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=PLOT_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PLOT_TICK_FONTSIZE)
    ax.legend(loc=legend_loc, fontsize=METRIC_PLOT_LEGEND_FONTSIZE, ncol=2, frameon=True)
    _style_axes(ax)
    _save(fig, out_pdf, out_png)


def plot_metric_figures(csv_path: str = METRIC_PLOT_CSV_PATH):
    """평가 CSV를 읽어 평균 체류시간 / 총 이동거리(km) 막대그래프 저장

    입력:
        csv_path: 평가 결과 CSV 경로
    """
    agg = load_eval_results(csv_path)

    flow_stats = compute_stats(agg, "avg_flow_time", scale=1.0)
    plot_grouped_bars(
        flow_stats,
        ylabel="Avg. flow time (sec)",
        out_pdf=os.path.join(FIG_DIR, "flow_time_by_lambda.pdf"),
        out_png=os.path.join(FIG_DIR, "flow_time_by_lambda.png"),
        legend_loc="upper right",
    )

    dist_stats = compute_stats(agg, "total_distance", scale=1.0 / 1000.0)
    plot_grouped_bars(
        dist_stats,
        ylabel="Total fleet travel distance (km)",
        out_pdf=os.path.join(FIG_DIR, "total_distance_by_lambda.pdf"),
        out_png=os.path.join(FIG_DIR, "total_distance_by_lambda.png"),
        legend_loc="upper left",
    )

    print(f"Saved figures to {FIG_DIR}/:")
    print("  flow_time_by_lambda.pdf / .png")
    print("  total_distance_by_lambda.pdf / .png")


def main():
    """명령행 인자에 따라 그래프 생성 (reward / metrics / all)"""
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("which", nargs="?", default="all",
                        choices=["all", "reward", "metrics"])
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    if args.which in ("all", "reward"):
        plot_reward_figures()
    if args.which in ("all", "metrics"):
        plot_metric_figures()


if __name__ == "__main__":
    main()
