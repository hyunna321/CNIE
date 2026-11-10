"""비교 실험: 학습한 체크포인트(RL)와 베이스라인을 같은 테스트 시드로 평가

사용법 (저장소 최상위에서):
    python -m experiments.run_compare --config experiments/configs/base.yaml          # 베이스라인만 (BL8 제외 전부)
    python -m experiments.run_compare --baselines BL4 BL6 --lam 280 --robots 18 --n-eval 5
    python -m experiments.run_compare --load runs/<이름>/.../agent.pt --baselines BL4 BL5 BL6   # RL vs 베이스라인
    python -m experiments.run_compare --load runs/<이름>/.../agent.pt --robots 18 --tag xfer    # 재학습 없는 일반화
    python -m experiments.run_compare --summarize --out runs/<이름>       # 셀별 run/베이스라인 평균, BL6 대비 차이

    --route fixed_dock|fixed_return  경로 구성 규칙 (RL·베이스라인 공통) → setting과 RL run 이름에 _route-* 꼬리표
    --no-collisions                  reactive에서 로봇 간 셀 점유를 무시 (회귀 검증용)

--load 시 arm/학습 시드/학습 M/학습 λ는 체크포인트 기록을 따름
--load 시 BL6 prior 척도: 평가 레이아웃의 dist_norm으로 mc_scale을 다시 설정 (기본, 주차장 행 수가 M·K에 따라
    달라 학습 레이아웃 값이면 prior가 정확히 mc/Z가 아님). --keep-train-mc-scale이면 체크포인트 값 유지.
    적용 여부와 두 값은 run.json의 mc_scale에 기록
--load 결과 폴더가 체크포인트 학습 폴더와 같고 eval.csv가 이미 있으면 --tag가 필요 (원래 eval.csv 보호)
--load 시 경과시간 특징의 정규화 기준은 체크포인트 학습 sim_time (평가 --sim-time과 달라도 특징 척도 유지)

결과
    테스트: {out}/{setting}/lam{λ}_M{M'}_K{K}/{run}/eval.csv, run.json
          run = {arm}_s{seed}[_tr{spec}][_trlam{학습 λ}][_route-*][_{tag}]  (학습 M = M'이면 _tr 없음)
    베이스라인: {out}/{setting}/lam{λ}_M{M'}_K{K}/baselines/eval.csv (이미 있는 (정책, 시드)는 다시 돌리지 않음)
"""
import argparse
import csv
import glob
import json
import math
import os
import time
from collections import defaultdict

from agents import DQNAgent, read_checkpoint
from env.layout import NUM_ROBOTS, ROBOT_CAPACITY
from env.orders import Scenario
from env.rl import ARMS
from evaluation import (MetricsStore, N_EVAL, POLICIES, evaluate_baselines_cached, evaluate_policy, policy_arg, policy_label,
                        print_comparison, test_seeds)
from train import add_common_args, m_spec, make_condition, parse_args, runtime_env, save_config

RL_DEFAULT_BASELINES = ["BL4", "BL5", "BL6"]


def summarize(root: str):
    """셀별로 run과 베이스라인의 평균 흐름시간, BL6 대비 차이(같은 테스트 시드 대응 평균 ± 표준오차) 출력"""
    for cell in sorted(glob.glob(os.path.join(root, "*", "*"))):
        flows = defaultdict(dict)        # 라벨 -> {테스트 시드: 흐름시간}
        for path in glob.glob(os.path.join(cell, "*", "eval.csv")):
            run_name = os.path.basename(os.path.dirname(path))
            for row in csv.DictReader(open(path)):
                key = run_name if row["label"] == "RL" else row["label"]
                flows[key][int(row["seed"])] = float(row["avg_flow_time"])
        if "BL6_MarginalCost" not in flows:
            continue
        bl6 = flows["BL6_MarginalCost"]
        print(f"\n{os.path.relpath(cell, root)}  (평균 흐름시간 s, BL6 대비 차이 = 같은 테스트 시드 대응 평균)")
        for key in sorted(flows, key=lambda k: sum(flows[k].values()) / len(flows[k])):
            v = flows[key]
            diff = [v[e] - bl6[e] for e in sorted(set(v) & set(bl6))]
            if not diff:
                continue
            m  = sum(diff) / len(diff)
            sd = math.sqrt(sum((d - m) ** 2 for d in diff) / max(1, len(diff) - 1))
            print(f"  {key:30s} {sum(v.values()) / len(v):7.1f}   vs BL6 {m:+7.1f} ± {sd / math.sqrt(len(diff)):.1f} (n={len(diff)})")


def main(argv=None):
    p = argparse.ArgumentParser(description="MTSR 비교 실험", formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    add_common_args(p)
    p.add_argument("--load", default=None, help="평가할 체크포인트 경로 (없으면 베이스라인만)")
    p.add_argument("--baselines", nargs="*", type=policy_arg, default=None,
                   help="베이스라인 (기본: --load이면 BL4 BL5 BL6, 아니면 BL8 제외 전부, 빈 목록이면 RL만)")
    p.add_argument("--eval-robots", type=int, nargs="+", default=None, help="평가 M (기본: --robots 또는 학습 M)")
    p.add_argument("--n-eval", type=int, default=N_EVAL, help="테스트 시드 수")
    p.add_argument("--no-collisions", action="store_true", help="reactive에서 로봇 간 셀 점유를 무시 (회귀 검증용)")
    p.add_argument("--keep-train-mc-scale", action="store_true",
                   help="--load 시 BL6 prior 척도를 평가 레이아웃 기준으로 다시 맞추지 않음")
    p.add_argument("--summarize", action="store_true", help="저장된 결과 요약만 출력")
    args = parse_args(p, argv)
    if args.summarize:
        summarize(args.out)
        return

    scenario = Scenario.from_arg(args.scenario)
    cond     = make_condition(args)
    cond.collisions = not args.no_collisions
    setting  = cond.setting(scenario)
    lam      = args.lam / 3600.0
    tag_s    = f"_{args.tag}" if args.tag else ""
    seeds    = test_seeds(args.n_eval)
    layouts  = {}

    def layout_for(M: int):
        """M대용 레이아웃 (주차·대기열 셀 크기가 M에 따라 달라짐, M마다 한 번 생성)"""
        if M not in layouts:
            layouts[M] = cond.layout(M, args.ws, scenario)
        return layouts[M]

    def base_dir(M: int) -> str:
        return os.path.join(args.out, setting, f"lam{args.lam:.0f}_M{M}_K{args.ws}")

    baselines = args.baselines
    if baselines is None:
        baselines = RL_DEFAULT_BASELINES if args.load else [k for k in POLICIES if k != "BL8"]

    agent, arm, meta, train_Ms = None, None, {}, []
    if args.load:
        import torch
        meta = torch.load(args.load, map_location="cpu").get("meta", {})
        arm_name, train_seed, train_lam = meta["arm"], meta["train_seed"], meta.get("lam", lam)
        train_Ms = sorted(meta.get("train_robots") or [meta["num_robots"]])
        if arm_name not in ARMS:
            raise SystemExit(f"체크포인트 arm '{arm_name}'은 지금 코드의 env.rl.ARMS에 없습니다")
        arm = ARMS[arm_name]
        spec = m_spec(train_Ms)
    default_Ms = [args.robots] if args.robots is not None else (train_Ms or [NUM_ROBOTS])
    eval_Ms = sorted(set(args.eval_robots or default_Ms))

    if args.load:
        norm_time = read_checkpoint(args.load).get("sim_time") or args.sim_time
        agent = DQNAgent(robot_capacity=ROBOT_CAPACITY, dist_norm=layout_for(train_Ms[0]).dist_norm, seed=train_seed,
                         sim_time=norm_time, num_robots=max(train_Ms + eval_Ms), num_workstations=args.ws,
                         n_step=arm.n_step, features=arm.features)
        agent.load(args.load)
        train_mc_scale = agent.mc_scale

    for M in eval_Ms:
        L, bdir = layout_for(M), base_dir(M)
        store, labels = MetricsStore(), []
        title = "Baselines"
        if args.load:
            run_name = f"{arm_name}_s{train_seed}"
            if spec != str(M):
                run_name += f"_tr{spec}"
            if abs(train_lam - lam) * 3600 > 0.5:
                run_name += f"_trlam{train_lam * 3600:.0f}"
            run_name += cond.tag + tag_s
            out = os.path.join(bdir, run_name)
            same_dir = os.path.abspath(out) == os.path.dirname(os.path.abspath(args.load))
            if same_dir and os.path.isfile(os.path.join(out, "eval.csv")) and not args.tag:
                raise SystemExit(f"--load 결과 폴더 {out}에 eval.csv가 이미 있습니다. --tag를 지정하세요 (원래 eval.csv 보호)")
            save_config(args, out, "compare")
            mc_info = {"train": train_mc_scale, "eval_layout": agent.mc_scale_for(L.dist_norm),
                       "applied": not args.keep_train_mc_scale}
            if mc_info["applied"]:
                agent.set_mc_scale(L.dist_norm)
            mc_info["used"] = agent.mc_scale
            wall  = time.perf_counter()
            store = evaluate_policy(L, "RL", lam, seeds, args.sim_time, agent=agent, arm=arm, sim_kw=cond.sim_kw)
            eval_wall = time.perf_counter() - wall
            store.save_csv(os.path.join(out, "eval.csv"))
            rs  = store.records
            lat = {k: sum(getattr(r, k) for r in rs) / len(rs)
                   for k in ("decision_ms_mean", "decision_ms_p95", "decision_full_ms_mean", "decision_full_ms_p95")}
            print(f"결정 지연 평균 {lat['decision_ms_mean']:.2f} ms (p95 {lat['decision_ms_p95']:.2f}), "
                  f"배정·재계획 포함 {lat['decision_full_ms_mean']:.2f} ms (p95 {lat['decision_full_ms_p95']:.2f}), "
                  f"평가 {eval_wall:.0f} s")
            path = os.path.join(out, "run.json")
            prev = {}                                    # 학습 폴더 평가면 학습 기록 유지
            if same_dir and os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    prev = json.load(f)
            info = {**prev, "arm": arm_name, **arm.to_dict(), "train_seed": train_seed, "train_robots": train_Ms,
                    "train_lam_hr": train_lam * 3600, "eval_lam_hr": args.lam, "num_workstations": args.ws,
                    "traffic": cond.traffic, "motion": cond.motion, "scenario": scenario.to_dict(),
                    "condition": cond.to_dict(), "setting": setting, "sim_time": args.sim_time, "tag": args.tag,
                    "n_eval": args.n_eval, "env": runtime_env(), "checkpoint": os.path.abspath(args.load),
                    "select": meta.get("select"), "selected_episode": meta.get("selected_episode"),
                    "eval_robots": M, "run": run_name, "test_seeds": seeds, "eval_wall_s": eval_wall,
                    "eval_latency_seed_mean": lat, "mc_scale": mc_info,
                    "train_num_workstations": meta.get("num_workstations", args.ws)}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=1, ensure_ascii=False)
            labels, title = ["RL"], f"RL ({run_name}) vs baselines"
        if baselines:
            save_config(args, os.path.join(bdir, "baselines"), "compare")
            bl = evaluate_baselines_cached(L, baselines, lam, seeds, args.sim_time,
                                           os.path.join(bdir, "baselines", "eval.csv"), sim_kw=cond.sim_kw)
            for r in bl.records:
                store.add(r)
            labels += [policy_label(k) for k in baselines]
        if labels:
            print_comparison(store, labels, lam=lam, num_robots=M, last_n=args.n_eval, title=title)
        print(f"저장: {bdir}")


if __name__ == "__main__":
    main()
