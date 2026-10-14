"""RL 개선 실험: 업데이트 비율과 BL6 한계비용 특징의 효과를 BL6와 비교

arm, 환경, 시드 하나를 학습/평가해 저장
조합마다 프로세스를 따로 띄워 병렬 실행

사용법:
    python run_rl_improve.py --env grid --arm smdp_r8_bl6 --seed 42
    python run_rl_improve.py --env cont --arm baselines
    python run_rl_improve.py --summarize

arm:
    smdp_u1_base   : SMDP 보상, 에피소드당 업데이트 1회, 기본 특징 (run.py 기본값)
    smdp_r8_base   : SMDP 보상, 결정 8개당 업데이트 1회, 기본 특징
    smdp_r8_bl6    : SMDP 보상, 결정 8개당 업데이트 1회, BL6 한계비용 특징 추가
    legacy_r8_base : legacy 보상, 결정 8개당 업데이트 1회, 기본 특징
    legacy_r8_bl6  : legacy 보상, 결정 8개당 업데이트 1회, BL6 특징 추가
    smdpw_r8       : SMDP + wait 행동 + shaping (RLWaitSimulator), 결정 8개당 업데이트 1회
    baselines      : BL6, BL4, Greedy-R만 평가

env:
    cont : 연속 레이아웃
    grid : 그리드 free + reserve (기본 시나리오)

저장: logs_rl_improve/{env}_M{M}_lam{λ}/{arm}_s{seed}/ (train.csv, curve.csv, eval.csv, agent.pt)
"""
import argparse
import csv
import glob
import math
import os
from collections import defaultdict

from environment import WarehouseLayout
from grid import GridWarehouseLayout
from agents import DQNAgent
from metrics import MetricsStore, TrainingCurveStore
from simulator import rl_simulator_cls
from scenario import Scenario
from training import train
from baselines import MarginalCostSimulator, NearestRobotSimulator, GreedyRewardSimulator
from constant import (SIM_TIME, DEFAULT_SEED, ROBOT_CAPACITY, TOTAL_EPISODES, N_EVAL, EVAL_SEED_BASE,
                      PICK_TIME, WS_PROC_TIME)

OUT_DIR = "logs_rl_improve"
ARMS = {
    "smdp_u1_base":   dict(reward_mode="smdp",   update_ratio=None, feature_set="base"),
    "smdp_r8_base":   dict(reward_mode="smdp",   update_ratio=8,    feature_set="base"),
    "smdp_r8_bl6":    dict(reward_mode="smdp",   update_ratio=8,    feature_set="bl6"),
    "legacy_r8_base": dict(reward_mode="legacy", update_ratio=8,    feature_set="base"),
    "legacy_r8_bl6":  dict(reward_mode="legacy", update_ratio=8,    feature_set="bl6"),
    "smdpw_r8":       dict(reward_mode="smdp_w", update_ratio=8,    feature_set="smdp_w"),
}
BASELINES = {"BL6_MarginalCost": MarginalCostSimulator, "BL4_Nearest": NearestRobotSimulator,
             "Greedy_R": GreedyRewardSimulator}


def make_layout(env: str, num_robots: int, num_workstations: int,
                pick_time: float = PICK_TIME, proc_time: float = WS_PROC_TIME):
    sc = Scenario(pick_time=pick_time, ws_proc_time=proc_time)
    if env == "grid":
        return GridWarehouseLayout(seed=DEFAULT_SEED, num_robots=num_robots,
                                   num_workstations=num_workstations, traffic="free", motion="reserve",
                                   scenario=sc)
    return WarehouseLayout(seed=DEFAULT_SEED, num_robots=num_robots, num_workstations=num_workstations,
                           scenario=sc)


def run(env: str, arm: str, seed: int, num_robots: int, num_workstations: int, lam_hr: int,
        episodes: int, n_eval: int, pick_time: float = PICK_TIME, proc_time: float = WS_PROC_TIME):
    """arm 하나를 학습/평가 후 저장"""
    lam      = lam_hr / 3600.0
    layout   = make_layout(env, num_robots, num_workstations, pick_time, proc_time)
    tag      = f"_pick{pick_time:g}_proc{proc_time:g}" if (pick_time or proc_time) else ""
    out      = os.path.join(OUT_DIR, f"{env}_M{num_robots}_lam{lam_hr}{tag}", f"{arm}_s{seed}")
    os.makedirs(out, exist_ok=True)
    seeds    = list(range(EVAL_SEED_BASE, EVAL_SEED_BASE + n_eval))
    ev_store = MetricsStore()

    if arm == "baselines":
        for i, s in enumerate(seeds):
            for label, cls in BASELINES.items():
                sim = cls(layout, lam=lam, seed=s, sim_time=SIM_TIME, num_robots=num_robots,
                          robot_capacity=ROBOT_CAPACITY)
                sim.reset(s); sim.run()
                ev_store.add(sim.build_result(label, i, lam))
        ev_store.save_csv(os.path.join(out, "eval.csv"))
        return

    cfg   = ARMS[arm]
    agent = DQNAgent(robot_capacity=ROBOT_CAPACITY, seed=seed, sim_time=SIM_TIME,
                     num_robots=num_robots, num_workstations=num_workstations,
                     feature_set=cfg["feature_set"])
    tr_store, curve = MetricsStore(), TrainingCurveStore()
    train(layout, agent, tr_store, lam=lam, seed=seed, total_episodes=episodes,
          num_robots=num_robots, robot_capacity=ROBOT_CAPACITY, label=arm,
          curve_store=curve, reward_mode=cfg["reward_mode"], update_ratio=cfg["update_ratio"],
          verbose_every=max(1, episodes // 20))
    tr_store.save_csv(os.path.join(out, "train.csv"))
    curve.save_csv(os.path.join(out, "curve.csv"))
    agent.save(os.path.join(out, "agent.pt"), {
        "lam": lam, "num_robots": num_robots, "num_workstations": num_workstations,
        "robot_capacity": ROBOT_CAPACITY, "episodes": episodes, "grid": env == "grid",
        "traffic": "free" if env == "grid" else None, "motion": "reserve" if env == "grid" else None,
        "scenario": layout.scenario.to_dict(), "arm": arm, "train_seed": seed, **cfg})
    for i, s in enumerate(seeds):
        sim = rl_simulator_cls(cfg["reward_mode"])(
            layout, agent, lam=lam, seed=s, num_robots=num_robots,
            robot_capacity=ROBOT_CAPACITY, eval_mode=True, reward_mode=cfg["reward_mode"])
        sim.reset(s); sim.run()
        ev_store.add(sim.build_result(arm, i, lam))
    ev_store.save_csv(os.path.join(out, "eval.csv"))


def summarize():
    """설정별로 RL arm과 베이스라인의 평균 흐름시간, BL6 대비 차이(같은 평가 시드 대응) 출력"""
    for setting in sorted(glob.glob(os.path.join(OUT_DIR, "*"))):
        flows = defaultdict(dict)        # label/seed -> {eval episode: flow}
        for path in glob.glob(os.path.join(setting, "*", "eval.csv")):
            run_name = os.path.basename(os.path.dirname(path))
            for row in csv.DictReader(open(path)):
                key = row["label"] if run_name.startswith("baselines") else run_name
                flows[key][int(row["episode"])] = float(row["avg_flow_time"])
        if "BL6_MarginalCost" not in flows:
            continue
        bl6 = flows["BL6_MarginalCost"]
        print(f"\n{os.path.basename(setting)}  (평균 흐름시간 s, BL6 대비 차이 = 같은 평가 시드 대응 평균)")
        for key in sorted(flows, key=lambda k: sum(flows[k].values()) / len(flows[k])):
            v = flows[key]
            common = sorted(set(v) & set(bl6))
            diff = [v[e] - bl6[e] for e in common]
            m = sum(diff) / len(diff)
            sd = math.sqrt(sum((d - m) ** 2 for d in diff) / max(1, len(diff) - 1))
            print(f"  {key:22s} {sum(v.values()) / len(v):7.1f}   vs BL6 {m:+7.1f} ± {sd / math.sqrt(len(diff)):.1f} (n={len(diff)})")


def main():
    ap = argparse.ArgumentParser(description="RL 개선 실험 (학습량, BL6 특징)")
    ap.add_argument("--env", choices=["cont", "grid"], default="cont")
    ap.add_argument("--arm", choices=list(ARMS) + ["baselines"])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--robots", type=int, default=14)
    ap.add_argument("--workstations", type=int, default=3)
    ap.add_argument("--lam", type=int, default=800, help="orders/hr")
    ap.add_argument("--episodes", type=int, default=TOTAL_EPISODES)
    ap.add_argument("--n-eval", type=int, default=N_EVAL)
    ap.add_argument("--pick-time", type=float, default=PICK_TIME, help="s/토트, 피킹 시간 (그리드 모드)")
    ap.add_argument("--proc-time", type=float, default=WS_PROC_TIME, help="s/토트, 워크스테이션 처리 시간 (그리드 모드)")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        summarize()
    else:
        run(a.env, a.arm, a.seed, a.robots, a.workstations, a.lam, a.episodes, a.n_eval,
            a.pick_time, a.proc_time)


if __name__ == "__main__":
    main()
