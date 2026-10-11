"""보상 정식화 비교 실험: legacy 보상 RL vs SMDP 보상 RL vs BL1~BL6.

조건(setting) 하나와 RL 학습 방식(arm) 하나를 받아 학습/평가 후 CSV로 저장
조건별, arm별 별도 프로세스 병렬 실행 전제

사용법:
    python run_reward_compare.py --robots 14 --lam 800 --arm legacy
    python run_reward_compare.py --robots 14 --lam 800 --arm smdp
    python run_reward_compare.py --robots 14 --lam 800 --arm smdp_u50
    python run_reward_compare.py --robots 14 --lam 800 --arm baselines
    python run_reward_compare.py --summarize

arm:
    legacy    : 기존 보상, 에피소드당 train_step 1회
    smdp      : SMDP 시간 적분 보상, 에피소드당 train_step 1회 (보상만 바꾼 효과)
    smdp_u50  : SMDP 보상, 에피소드당 train_step 50회
    legacy_u50: 기존 보상, 에피소드당 train_step 50회 (smdp_u50과 업데이트 수를 맞춘 대조군)
    baselines : BL1~BL6만 평가
    greedy_r  : Greedy-R (legacy 즉시 보상 greedy 최대화)만 평가
"""
import argparse
import csv
import glob
import math
import os
from collections import defaultdict

import torch

from environment import WarehouseLayout
from agents import DQNAgent
from metrics import MetricsStore, TrainingCurveStore
from training import train, evaluate_all
from baselines import GreedyRewardSimulator
from constant import (
    SIM_TIME, DEFAULT_SEED, ROBOT_CAPACITY, TOTAL_EPISODES, N_EVAL,
    EVAL_SEED_BASE, NUM_WORKSTATIONS,
)

OUT_DIR = "logs_reward_compare"
ARMS = {
    "legacy":   dict(reward_mode="legacy", updates_per_episode=1),
    "smdp":     dict(reward_mode="smdp",   updates_per_episode=1),
    "smdp_u50": dict(reward_mode="smdp",   updates_per_episode=50),
    "legacy_u50": dict(reward_mode="legacy", updates_per_episode=50),
}
SUMMARY_METRICS = ["avg_flow_time", "dist_per_order", "avg_totes_per_trip",
                   "mid_trip_rate", "n_completed"]


def run_arm(num_robots: int, lam_hr: int, arm: str, total_episodes: int, n_eval: int):
    """조건 하나에서 arm 하나를 실행해 setting 디렉터리에 저장

    입력:
        num_robots: 로봇 수
        lam_hr: 주문 도착률 (orders/hr)
        arm: ARMS 키 또는 "baselines"
        total_episodes: 학습 에피소드 수
        n_eval: 평가 시드 수
    """
    torch.set_num_threads(1)
    lam = lam_hr / 3600
    setting_dir = os.path.join(OUT_DIR, f"M{num_robots}_lam{lam_hr}")
    os.makedirs(setting_dir, exist_ok=True)
    layout     = WarehouseLayout(seed=DEFAULT_SEED, num_workstations=NUM_WORKSTATIONS)
    eval_seeds = list(range(EVAL_SEED_BASE, EVAL_SEED_BASE + n_eval))
    eval_store = MetricsStore()

    if arm == "greedy_r":
        for i, s in enumerate(eval_seeds):
            sim = GreedyRewardSimulator(layout, lam=lam, seed=s, sim_time=SIM_TIME,
                                        num_robots=num_robots,
                                        robot_capacity=ROBOT_CAPACITY)
            sim.reset(s)
            sim.run()
            eval_store.add(sim.build_result("Greedy_R", i, lam))
    elif arm == "baselines":
        # RL 자리를 학습 없는 에이전트로 채우지 않도록 baselines 전용 경로 사용
        agent = DQNAgent(num_robots=num_robots, robot_capacity=ROBOT_CAPACITY,
                         seed=DEFAULT_SEED, sim_time=SIM_TIME,
                         num_workstations=NUM_WORKSTATIONS)
        evaluate_all(layout, agent, eval_store, lam=lam,
                     num_robots=num_robots, robot_capacity=ROBOT_CAPACITY,
                     eval_seeds=eval_seeds, n_eval=n_eval,
                     rl_label="_untrained")
        eval_store.records = [r for r in eval_store.records if r.label != "_untrained"]
    else:
        cfg   = ARMS[arm]
        agent = DQNAgent(num_robots=num_robots, robot_capacity=ROBOT_CAPACITY,
                         seed=DEFAULT_SEED, sim_time=SIM_TIME,
                         num_workstations=NUM_WORKSTATIONS)
        train_store = MetricsStore()
        curve_store = TrainingCurveStore()
        train(layout, agent, train_store, lam=lam, seed=DEFAULT_SEED,
              total_episodes=total_episodes, num_robots=num_robots,
              robot_capacity=ROBOT_CAPACITY, label=f"RL_{arm}_train",
              curve_store=curve_store, **cfg)
        curve_store.save_csv(os.path.join(setting_dir, f"curve_{arm}.csv"))
        agent.save(os.path.join(setting_dir, f"agent_{arm}.pt"))
        evaluate_all(layout, agent, eval_store, lam=lam,
                     num_robots=num_robots, robot_capacity=ROBOT_CAPACITY,
                     eval_seeds=eval_seeds, n_eval=n_eval,
                     rl_label=f"RL_{arm}", reward_mode=cfg["reward_mode"],
                     include_baselines=False)

    path = os.path.join(setting_dir, f"eval_{arm}.csv")
    eval_store.save_csv(path)
    print(f"Saved: {path}")


def _paired_t(a, b):
    """대응표본 t-검정의 (평균 차이, p값). scipy가 없으면 p값은 nan."""
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    mean = sum(diffs) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1)) if n > 1 else 0.0
    if sd == 0:
        return mean, float("nan")
    t = mean / (sd / math.sqrt(n))
    try:
        from scipy import stats
        return mean, float(2 * stats.t.sf(abs(t), n - 1))
    except Exception:
        return mean, float("nan")


def summarize():
    """setting마다 eval_*.csv를 합쳐 정책별 평균과 BL6 대비 대응표본 검정 출력/저장"""
    rows_out = []
    for setting_dir in sorted(glob.glob(os.path.join(OUT_DIR, "M*_lam*"))):
        by_label = defaultdict(dict)
        for path in glob.glob(os.path.join(setting_dir, "eval_*.csv")):
            with open(path) as f:
                for r in csv.DictReader(f):
                    by_label[r["label"]][int(r["episode"])] = r
        if not by_label:
            continue
        setting = os.path.basename(setting_dir)
        print(f"\n=== {setting} ===")
        print(f"  {'policy':<20}" + "".join(f"{m:>20}" for m in SUMMARY_METRICS)
              + f"{'Δflow vs BL6':>16}{'p':>10}")
        ref = by_label.get("BL6_MarginalCost")
        for label in sorted(by_label):
            recs = by_label[label]
            means = {m: sum(float(r[m]) for r in recs.values()) / len(recs)
                     for m in SUMMARY_METRICS}
            d, p = float("nan"), float("nan")
            if ref is not None and label != "BL6_MarginalCost":
                common = sorted(set(recs) & set(ref))
                d, p = _paired_t([float(recs[i]["avg_flow_time"]) for i in common],
                                 [float(ref[i]["avg_flow_time"]) for i in common])
            print(f"  {label:<20}" + "".join(f"{means[m]:>20.2f}" for m in SUMMARY_METRICS)
                  + f"{d:>16.2f}{p:>10.3g}")
            rows_out.append({"setting": setting, "label": label, "n": len(recs),
                             **means, "dflow_vs_bl6": d, "p_vs_bl6": p})
    if rows_out:
        path = os.path.join(OUT_DIR, "summary.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)
        print(f"\nSaved: {path}")


def main():
    """명령행 인자로 arm 하나 실행 또는 결과 요약"""
    parser = argparse.ArgumentParser(description="Reward formulation comparison")
    parser.add_argument("--robots", type=int, default=14)
    parser.add_argument("--lam", type=int, default=800, help="orders/hr")
    parser.add_argument("--arm", choices=list(ARMS) + ["baselines", "greedy_r"])
    parser.add_argument("--episodes", type=int, default=TOTAL_EPISODES)
    parser.add_argument("--n-eval", type=int, default=N_EVAL)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    if args.summarize:
        summarize()
    else:
        run_arm(args.robots, args.lam, args.arm, args.episodes, args.n_eval)


if __name__ == "__main__":
    main()
