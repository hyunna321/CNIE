"""RL(SMDP + wait, DQN) 학습/평가 진입점: 학습 후 같은 평가 시드로 베이스라인과 비교

사용법:
    python run_rl.py                                   # λ=320 주문/hr, M=14, K=3, 1000 에피소드 학습 후 평가
    python run_rl.py --episodes 200 --baselines BL4 BL6 --n-eval 5
    python run_rl.py --load logs_rl/.../agent.pt       # 학습 없이 체크포인트 평가

    --traffic, --motion, --scenario는 run.py와 같음
    결과: {out}/grid_{traffic}_{motion}[_sc해시]/lam{λ}_M{M}_K{K}_s{seed}/
          (train.csv, curve.csv, agent.pt, eval.csv)
"""
import argparse
import hashlib
import json
import os

from grid import GridWarehouseLayout
from scenario import Scenario
from agents import DQNAgent
from metrics import MetricsStore, TrainingCurveStore, print_comparison
from training import train, evaluate as evaluate_rl
from run import POLICIES, evaluate as evaluate_baselines
from constant import (
    GRID_TRAFFIC, GRID_MOTION, SIM_TIME, LAMBDA_DEFAULT, DEFAULT_SEED,
    NUM_ROBOTS, NUM_WORKSTATIONS, ROBOT_CAPACITY, N_EVAL, EVAL_SEED_BASE,
    TOTAL_EPISODES, UPDATE_RATIO, RL_OUT_DIR,
)


def main():
    """명령행 인자로 고른 설정에서 RL 학습/평가 후 베이스라인과 비교"""
    p = argparse.ArgumentParser(description="MTSR grid RL (SMDP + wait)")
    p.add_argument("--lam", type=float, default=LAMBDA_DEFAULT * 3600, help="주문 도착률 (orders/hr)")
    p.add_argument("--robots", type=int, default=NUM_ROBOTS, help="로봇 수 M")
    p.add_argument("--ws", type=int, default=NUM_WORKSTATIONS, help="워크스테이션 수 K")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="학습 시드")
    p.add_argument("--episodes", type=int, default=TOTAL_EPISODES, help="학습 에피소드 수")
    p.add_argument("--update-ratio", type=float, default=UPDATE_RATIO, help="결정 N개당 업데이트 1회")
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
    args = p.parse_args()

    scenario = Scenario()
    if args.scenario:
        text = open(args.scenario).read() if os.path.isfile(args.scenario) else args.scenario
        scenario = Scenario(**json.loads(text))
    setting = f"grid_{args.traffic}_{args.motion}"
    if not scenario.is_default:
        setting += "_sc" + hashlib.sha1(json.dumps(scenario.to_dict(), sort_keys=True).encode()).hexdigest()[:6]
    out = os.path.join(args.out, setting, f"lam{args.lam:.0f}_M{args.robots}_K{args.ws}_s{args.seed}")
    os.makedirs(out, exist_ok=True)

    lam    = args.lam / 3600.0
    layout = GridWarehouseLayout(seed=DEFAULT_SEED, num_robots=args.robots,
                                 robot_capacity=ROBOT_CAPACITY, num_workstations=args.ws,
                                 traffic=args.traffic, motion=args.motion, scenario=scenario)
    agent  = DQNAgent(robot_capacity=ROBOT_CAPACITY, seed=args.seed, sim_time=args.sim_time,
                      num_robots=args.robots, num_workstations=args.ws)
    print(f"λ={args.lam:.0f}/hr, M={args.robots}, K={args.ws}, {args.traffic}/{args.motion}, "
          f"특징 {agent.feat_dim}차원, 결과 {out}")

    if args.load:
        agent.load(args.load)
    else:
        tr_store, curve = MetricsStore(), TrainingCurveStore()
        train(layout, agent, tr_store, lam=lam, seed=args.seed, total_episodes=args.episodes,
              sim_time=args.sim_time, label="RL", curve_store=curve,
              update_ratio=args.update_ratio, verbose_every=max(1, args.episodes // 20))
        tr_store.save_csv(os.path.join(out, "train.csv"))
        curve.save_csv(os.path.join(out, "curve.csv"))
        agent.save(os.path.join(out, "agent.pt"), {
            "lam": lam, "num_robots": args.robots, "num_workstations": args.ws,
            "robot_capacity": ROBOT_CAPACITY, "episodes": args.episodes, "train_seed": args.seed,
            "update_ratio": args.update_ratio, "traffic": args.traffic, "motion": args.motion,
            "scenario": scenario.to_dict()})

    seeds = list(range(EVAL_SEED_BASE, EVAL_SEED_BASE + args.n_eval))
    store = MetricsStore()
    evaluate_rl(layout, agent, store, lam, seeds, sim_time=args.sim_time)
    if args.baselines:
        for r in evaluate_baselines(layout, args.baselines, lam, args.robots,
                                    args.n_eval, args.sim_time).records:
            store.add(r)
    store.save_csv(os.path.join(out, "eval.csv"))
    print_comparison(store, ["RL"] + [POLICIES[k][0] for k in args.baselines], lam=lam,
                     num_robots=args.robots, last_n=args.n_eval, title="RL vs baselines")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
