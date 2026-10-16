"""RL 학습/평가 진입점: arm 하나를 학습한 뒤 같은 평가 시드로 베이스라인과 비교

사용법:
    python run_rl.py                                   # smdpw_r8, λ=320 주문/hr, M=14, K=3, 1000 에피소드
    python run_rl.py --arm smdp_r8_bl6 --episodes 200 --baselines BL4 BL6 --n-eval 5
    python run_rl.py --arm smdpw_r8 --load logs_rl/.../agent.pt   # 학습 없이 체크포인트 평가
    python run_rl.py --summarize                       # 설정별 arm/베이스라인 평균 흐름시간, BL6 대비 차이

arm (보상 모드, 업데이트 비율, 특징 세트):
    smdp_u1_base   : smdp,   에피소드당 1회,  base
    smdp_r8_base   : smdp,   결정 8개당 1회, base
    smdp_r8_bl6    : smdp,   결정 8개당 1회, bl6
    legacy_r8_base : legacy, 결정 8개당 1회, base
    legacy_r8_bl6  : legacy, 결정 8개당 1회, bl6
    smdpw_r8       : smdp_w, 결정 8개당 1회, smdp_w (wait 행동 + shaping)

    --traffic, --motion, --scenario는 run.py와 같음
    결과: {out}/grid_{traffic}_{motion}[_sc해시]/lam{λ}_M{M}_K{K}/{arm}_s{seed}/
          (train.csv, curve.csv, agent.pt, eval.csv), 베이스라인 평가는 같은 폴더의 baselines/eval.csv
"""
import argparse
import csv
import glob
import hashlib
import json
import math
import os
from collections import defaultdict

from grid import GridWarehouseLayout
from scenario import Scenario
from agents import DQNAgent
from metrics import MetricsStore, TrainingCurveStore, print_comparison
from training import train, evaluate as evaluate_rl
from run import POLICIES, evaluate as evaluate_baselines
from constant import (
    GRID_TRAFFIC, GRID_MOTION, SIM_TIME, LAMBDA_DEFAULT, DEFAULT_SEED,
    NUM_ROBOTS, NUM_WORKSTATIONS, ROBOT_CAPACITY, N_EVAL, EVAL_SEED_BASE,
    TOTAL_EPISODES, RL_OUT_DIR,
)

ARMS = {
    "smdp_u1_base":   dict(reward_mode="smdp",   update_ratio=None, feature_set="base"),
    "smdp_r8_base":   dict(reward_mode="smdp",   update_ratio=8,    feature_set="base"),
    "smdp_r8_bl6":    dict(reward_mode="smdp",   update_ratio=8,    feature_set="bl6"),
    "legacy_r8_base": dict(reward_mode="legacy", update_ratio=8,    feature_set="base"),
    "legacy_r8_bl6":  dict(reward_mode="legacy", update_ratio=8,    feature_set="bl6"),
    "smdpw_r8":       dict(reward_mode="smdp_w", update_ratio=8,    feature_set="smdp_w"),
}


def summarize(root: str):
    """설정별로 arm과 베이스라인의 평균 흐름시간, BL6 대비 차이(같은 평가 시드 대응 평균 ± 표준오차) 출력"""
    for setting in sorted(glob.glob(os.path.join(root, "*", "*"))):
        flows = defaultdict(dict)        # 라벨 -> {평가 에피소드: 흐름시간}
        for path in glob.glob(os.path.join(setting, "*", "eval.csv")):
            run_name = os.path.basename(os.path.dirname(path))
            for row in csv.DictReader(open(path)):
                key = run_name if row["label"] == "RL" else row["label"]
                flows[key][int(row["episode"])] = float(row["avg_flow_time"])
        if "BL6_MarginalCost" not in flows:
            continue
        bl6 = flows["BL6_MarginalCost"]
        print(f"\n{os.path.relpath(setting, root)}  (평균 흐름시간 s, BL6 대비 차이 = 같은 평가 시드 대응 평균)")
        for key in sorted(flows, key=lambda k: sum(flows[k].values()) / len(flows[k])):
            v = flows[key]
            diff = [v[e] - bl6[e] for e in sorted(set(v) & set(bl6))]
            m  = sum(diff) / len(diff)
            sd = math.sqrt(sum((d - m) ** 2 for d in diff) / max(1, len(diff) - 1))
            print(f"  {key:22s} {sum(v.values()) / len(v):7.1f}   vs BL6 {m:+7.1f} ± {sd / math.sqrt(len(diff)):.1f} (n={len(diff)})")


def main():
    """명령행 인자로 고른 설정에서 RL arm 학습/평가 후 베이스라인과 비교"""
    p = argparse.ArgumentParser(description="MTSR grid RL")
    p.add_argument("--arm", choices=list(ARMS), default="smdpw_r8", help="보상 모드/업데이트 비율/특징 세트 조합")
    p.add_argument("--lam", type=float, default=LAMBDA_DEFAULT * 3600, help="주문 도착률 (orders/hr)")
    p.add_argument("--robots", type=int, default=NUM_ROBOTS, help="로봇 수 M")
    p.add_argument("--ws", type=int, default=NUM_WORKSTATIONS, help="워크스테이션 수 K")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="학습 시드")
    p.add_argument("--episodes", type=int, default=TOTAL_EPISODES, help="학습 에피소드 수")
    p.add_argument("--sim-time", type=float, default=SIM_TIME, help="에피소드 길이 (s), 학습/평가 공통")
    p.add_argument("--n-eval", type=int, default=N_EVAL, help="평가 시드 수")
    p.add_argument("--baselines", nargs="*", choices=list(POLICIES), default=["BL4", "BL5", "BL6"],
                   help="함께 평가할 베이스라인 (빈 목록이면 RL만)")
    p.add_argument("--load", default=None, help="학습 없이 평가할 체크포인트 경로")
    p.add_argument("--traffic", choices=["free", "lanes"], default=GRID_TRAFFIC)
    p.add_argument("--motion", choices=["reserve", "reactive"], default=GRID_MOTION)
    p.add_argument("--scenario", default=None,
                   help="시나리오 JSON 문자열 또는 JSON 파일 경로 (scenario.Scenario 필드)")
    p.add_argument("--out", default=RL_OUT_DIR, help="결과 상위 폴더")
    p.add_argument("--summarize", action="store_true", help="저장된 결과 요약만 출력")
    args = p.parse_args()

    if args.summarize:
        summarize(args.out)
        return

    scenario = Scenario()
    if args.scenario:
        text = open(args.scenario).read() if os.path.isfile(args.scenario) else args.scenario
        scenario = Scenario(**json.loads(text))
    setting = f"grid_{args.traffic}_{args.motion}"
    if not scenario.is_default:
        setting += "_sc" + hashlib.sha1(json.dumps(scenario.to_dict(), sort_keys=True).encode()).hexdigest()[:6]
    base_dir = os.path.join(args.out, setting, f"lam{args.lam:.0f}_M{args.robots}_K{args.ws}")
    out      = os.path.join(base_dir, f"{args.arm}_s{args.seed}")
    os.makedirs(out, exist_ok=True)

    cfg    = ARMS[args.arm]
    lam    = args.lam / 3600.0
    layout = GridWarehouseLayout(seed=DEFAULT_SEED, num_robots=args.robots,
                                 robot_capacity=ROBOT_CAPACITY, num_workstations=args.ws,
                                 traffic=args.traffic, motion=args.motion, scenario=scenario)
    agent  = DQNAgent(robot_capacity=ROBOT_CAPACITY, seed=args.seed, sim_time=args.sim_time,
                      num_robots=args.robots, num_workstations=args.ws, feature_set=cfg["feature_set"])
    print(f"arm={args.arm} ({cfg}), λ={args.lam:.0f}/hr, M={args.robots}, K={args.ws}, "
          f"{args.traffic}/{args.motion}, 특징 {agent.feat_dim}차원, 결과 {out}")

    if args.load:
        agent.load(args.load)
    else:
        tr_store, curve = MetricsStore(), TrainingCurveStore()
        train(layout, agent, tr_store, lam=lam, seed=args.seed, total_episodes=args.episodes,
              sim_time=args.sim_time, label=args.arm, curve_store=curve,
              update_ratio=cfg["update_ratio"], reward_mode=cfg["reward_mode"],
              verbose_every=max(1, args.episodes // 20))
        tr_store.save_csv(os.path.join(out, "train.csv"))
        curve.save_csv(os.path.join(out, "curve.csv"))
        agent.save(os.path.join(out, "agent.pt"), {
            "lam": lam, "num_robots": args.robots, "num_workstations": args.ws,
            "robot_capacity": ROBOT_CAPACITY, "episodes": args.episodes, "train_seed": args.seed,
            "arm": args.arm, **cfg, "traffic": args.traffic, "motion": args.motion,
            "scenario": scenario.to_dict()})

    seeds = list(range(EVAL_SEED_BASE, EVAL_SEED_BASE + args.n_eval))
    store = MetricsStore()
    evaluate_rl(layout, agent, store, lam, seeds, sim_time=args.sim_time, reward_mode=cfg["reward_mode"])
    store.save_csv(os.path.join(out, "eval.csv"))
    labels = ["RL"]
    if args.baselines:
        bl = evaluate_baselines(layout, args.baselines, lam, args.robots, args.n_eval, args.sim_time)
        bl.save_csv(os.path.join(base_dir, "baselines", "eval.csv"))
        for r in bl.records:
            store.add(r)
        labels += [POLICIES[k][0] for k in args.baselines]
    print_comparison(store, labels, lam=lam, num_robots=args.robots, last_n=args.n_eval,
                     title=f"RL ({args.arm}) vs baselines")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
