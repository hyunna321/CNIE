"""
plot_reward.py (수정본)
Plot reward convergence curves during training, one line per order arrival
rate (lambda), from training_curves_all_lambdas.csv.
"""
import csv
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
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

CSV_PATH = "training_curves_all_lambdas.csv"
OUT_PDF = "fig/reward_convergence.pdf"
OUT_PNG = "fig/reward_convergence.png"

LAM_VALUES = [700, 800, 900, 1000]
COLORS = {700: "#1a2a6c", 800: "#3b5fc4", 900: "#3fae7f", 1000: "#e8b93f"}
SMOOTH_WINDOW = 25

# 확대된 폰트 크기
LABEL_FONTSIZE = 15
TICK_FONTSIZE = 13
LEGEND_FONTSIZE = 12


def load_training_curves(path: str):
    with open(path) as f:
        rows = list(csv.DictReader(f))

    data = defaultdict(lambda: defaultdict(list))
    for row in rows:
        lam_hr = int(row["label"].split("lam")[1].split("_")[0])
        data[lam_hr]["episode"].append(int(row["episode"]))
        data[lam_hr]["reward"].append(float(row["reward"]))
    return data


def rolling_mean(x, window: int = SMOOTH_WINDOW):
    x = np.asarray(x, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_reward_convergence(data, out_pdf=OUT_PDF, out_png=OUT_PNG):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for lam_hr in LAM_VALUES:
        episodes = data[lam_hr]["episode"]
        reward = data[lam_hr]["reward"]

        reward_smooth = rolling_mean(reward, SMOOTH_WINDOW)
        episodes_smooth = episodes[len(episodes) - len(reward_smooth):]

        ax.plot(episodes, reward, color=COLORS[lam_hr], alpha=0.15, linewidth=0.8)
        ax.plot(
            episodes_smooth, reward_smooth,
            color=COLORS[lam_hr], linewidth=2.0,
            label=f"$\\lambda$={lam_hr}/hr",
        )

    ax.set_xlabel("Training episode", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Average reward per completed order", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    # 제목 제거 (요청 반영). 곡선이 하단에서 상승해 우하단에 여유 공간이 있어
    # 범례 위치는 겹침이 없으므로 기존 위치 유지.
    ax.legend(loc="lower right", fontsize=LEGEND_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs("fig", exist_ok=True)
    curves = load_training_curves(CSV_PATH)
    plot_reward_convergence(curves)
    print(f"Saved {OUT_PDF} and {OUT_PNG}")