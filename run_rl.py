"""RL 학습/평가 진입점: arm 하나를 학습한 뒤 같은 테스트 시드로 베이스라인과 비교

사용법:
    python run_rl.py                                   # smdp_r8_bl6, λ=320 주문/hr, M=14, K=3, 1000 에피소드
    python run_rl.py --arm smdp_r8_bl6 --episodes 200 --baselines BL4 BL6 --n-eval 5
    python run_rl.py --train-robots 10 14 18 --eval-robots 10 12 14 16 18   # 다중 M 학습, 여러 M에서 평가
    python run_rl.py --select val --val-every 50 --n-val 5                  # 검증 시드로 체크포인트 선택
    python run_rl.py --load logs_rl/.../agent.pt --eval-robots 18 --tag xfer # 학습 없이 체크포인트 평가
    python run_rl.py --summarize                       # 설정별 arm/베이스라인 평균 흐름시간, BL6 대비 차이

arm (보상 모드, 업데이트 비율, 특징 세트):
    smdp_u1_base   : smdp,   에피소드당 1회,  base
    smdp_r8_base   : smdp,   결정 8개당 1회, base
    smdp_r8_bl6    : smdp,   결정 8개당 1회, bl6
    legacy_r8_base : legacy, 결정 8개당 1회, base
    legacy_r8_bl6  : legacy, 결정 8개당 1회, bl6
    local_r8_ddqn  : local,  결정 8개당 1회, bl6 + 전역 상태 (GlobalDQNAgent: dueling 없음, BL6 잔차 α 학습, ε 고정)
                     v1: α 상한 없음, 보상 기준점 = 학습 누적 평균, MSE (logs_exp/v1_snapshot)
    local_r8_ddqn_v2: v1 + 보상 기준점 고정 μ(λ) (학습 전 BL6 prior 롤아웃, 셀마다 local_mu.json 공유), α ≤ 1, Huber 손실
    (이전 local_r8_glob은 dueling 구조로 제거됨, 결과 CSV만 남음)
    리비전 변형 (local_r8_ddqn_v2에서 한 요소만 바꿈, 에이전트/시뮬레이터 인자로 전달)
      local_r8_ddqn_v2_gamma  : 연속시간 할인 e^(−βτ) 대신 결정당 γ = 0.99, local 비용 유지
                                (RLSimulator discount="decision", μ(λ)는 기본 local_mu.json 공유)
      legacy_r8_ddqn_v2       : legacy shaped reward + 결정당 γ (원 논문 방식) + GlobalDQNAgent(α ≤ 1, Huber),
                                μ(λ) 없음. BL6 prior 척도 1/Z는 그대로 (agents.GlobalDQNAgent 설명)

평가 옵션
    --route fixed_dock 도킹 순서(복귀 WS)를 트립 출발 시 고정 (RL·베이스라인 공통)
                       → setting 뒤 _route-fixeddock, RL run 이름에도 같은 꼬리표
    --log-reconstructions PATH  RL 테스트 중 NNH 경로 구성 호출마다 입력·D(ρ)·호출 시간을 JSONL로 기록
                       (.gz이면 gzip, 결과 지표는 그대로, experiments/route_gap.py 입력)

--load 시 BL6 prior 척도: 평가 레이아웃의 dist_norm으로 mc_scale을 다시 설정 (기본, 주차장 행 수가 M·K에 따라
    달라 학습 레이아웃 값이면 prior가 정확히 mc/Z가 아님). --keep-train-mc-scale이면 체크포인트 값 유지.
    적용 여부와 두 값은 run.json의 mc_scale에 기록
--load 결과 폴더가 체크포인트의 학습 폴더와 같으면 (같은 셀·조건 재평가) --tag가 필요 (원래 eval.csv 보호)

이어하기: 학습 폴더에 resume.pt를 CKPT_EVERY 에피소드마다 저장. 같은 명령을 다시 실행하면 이어서 학습
    (--fresh이면 무시하고 처음부터). 테스트 평가까지 끝나면 resume.pt 삭제

로봇 수
    --train-robots M [M ...]  학습 M (여러 개면 에피소드마다 균등 추출, 레이아웃도 그 M으로)
    --eval-robots  M [M ...]  평가 M (기본: 학습 M, --load이면 --robots 또는 체크포인트 학습 M)
    --robots M                둘 다 M (위 두 인자가 없을 때만)
    --load이면 arm/학습 시드/학습 M/학습 λ는 체크포인트 기록을 따름

시드 (evaluation.py): 학습 에피소드 1,000,000 + 시드 × 100,000 + ep, 검증 20000 + i, 테스트 10000 + i
    --select last : 마지막 가중치로 테스트 (기본)
    --select val  : val_every마다 검증 시드로만 평가해 평균 흐름시간이 가장 작은 가중치로 테스트

    --traffic, --motion, --scenario는 run.py와 같음

결과 (spec = 학습 M, 여러 개면 "10-14-18")
    학습: {out}/grid_{traffic}_{motion}[_sc해시]/lam{λ}_M{spec}_K{K}/{arm}_s{seed}[_{tag}]/
          train.csv, curve.csv, val.csv, agent.pt (테스트에 쓴 가중치), agent_last.pt, run.json
    테스트: .../lam{λ}_M{M'}_K{K}/{run}/eval.csv, run.json
          run = {arm}_s{seed}[_tr{spec}][_trlam{학습 λ}][_{tag}]  (학습 M = M'인 specialist는 _tr 없음)
    베이스라인: .../lam{λ}_M{M'}_K{K}/baselines/eval.csv (이미 있는 (정책, 시드)는 다시 돌리지 않음)
"""
import argparse
import csv
import glob
import hashlib
import json
import math
import os
import time
from collections import defaultdict

from grid import GridWarehouseLayout
from scenario import Scenario
from agents import DQNAgent, GlobalDQNAgent
from metrics import MetricsStore, TrainingCurveStore, print_comparison
from training import train, make_validator, load_or_estimate_local_mu
from evaluation import (POLICIES, POLICY_ALIASES, Condition, evaluate_policy, evaluate_baselines_cached,
                        policy_key, test_seeds, val_seeds)
from constant import (
    GRID_TRAFFIC, GRID_MOTION, SIM_TIME, LAMBDA_DEFAULT, DEFAULT_SEED,
    NUM_ROBOTS, NUM_WORKSTATIONS, ROBOT_CAPACITY, N_EVAL, N_VAL,
    TOTAL_EPISODES, RL_OUT_DIR, ROUTE_MODES,
)

ARMS = {
    "smdp_u1_base":   dict(reward_mode="smdp",   update_ratio=None, feature_set="base"),
    "smdp_r8_base":   dict(reward_mode="smdp",   update_ratio=8,    feature_set="base"),
    "smdp_r8_bl6":    dict(reward_mode="smdp",   update_ratio=8,    feature_set="bl6"),
    "legacy_r8_base": dict(reward_mode="legacy", update_ratio=8,    feature_set="base"),
    "legacy_r8_bl6":  dict(reward_mode="legacy", update_ratio=8,    feature_set="bl6"),
    "local_r8_ddqn":  dict(reward_mode="local",  update_ratio=8,    feature_set="bl6", agent="global"),
    "local_r8_ddqn_v2": dict(reward_mode="local", update_ratio=8,   feature_set="bl6", agent="global",
                             mu="bl6", alpha_max=1.0, loss="huber"),
}
_V2 = ARMS["local_r8_ddqn_v2"]
ARMS.update({
    "local_r8_ddqn_v2_gamma": dict(_V2, discount="decision"),
    "legacy_r8_ddqn_v2":      dict(reward_mode="legacy", update_ratio=8, feature_set="bl6", agent="global",
                                   alpha_max=1.0, loss="huber"),
})
SIM_KEYS = ("discount",)    # arm 설정 중 RLSimulator 인자 (학습·검증)


def summarize(root: str):
    """설정별로 arm과 베이스라인의 평균 흐름시간, BL6 대비 차이(같은 테스트 시드 대응 평균 ± 표준오차) 출력"""
    for setting in sorted(glob.glob(os.path.join(root, "*", "*"))):
        flows = defaultdict(dict)        # 라벨 -> {테스트 시드: 흐름시간}
        for path in glob.glob(os.path.join(setting, "*", "eval.csv")):
            run_name = os.path.basename(os.path.dirname(path))
            for row in csv.DictReader(open(path)):
                key = run_name if row["label"] == "RL" else row["label"]
                # 시드 열이 없는 옛 CSV는 episode 번호로 대응
                seed = int(row["seed"]) if row.get("seed") not in (None, "", "-1") else -1 - int(row["episode"])
                flows[key][seed] = float(row["avg_flow_time"])
        if "BL6_MarginalCost" not in flows:
            continue
        bl6 = flows["BL6_MarginalCost"]
        print(f"\n{os.path.relpath(setting, root)}  (평균 흐름시간 s, BL6 대비 차이 = 같은 테스트 시드 대응 평균)")
        for key in sorted(flows, key=lambda k: sum(flows[k].values()) / len(flows[k])):
            v = flows[key]
            diff = [v[e] - bl6[e] for e in sorted(set(v) & set(bl6))]
            if not diff:
                continue
            m  = sum(diff) / len(diff)
            sd = math.sqrt(sum((d - m) ** 2 for d in diff) / max(1, len(diff) - 1))
            print(f"  {key:30s} {sum(v.values()) / len(v):7.1f}   vs BL6 {m:+7.1f} ± {sd / math.sqrt(len(diff)):.1f} (n={len(diff)})")


def runtime_env() -> dict:
    """실행 환경 기록 (학습 시간/결정 지연 보고용): CPU, 코어/스레드 수, Python·PyTorch 버전"""
    import platform
    import torch
    cpu = platform.processor()
    if platform.system() == "Windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif os.path.isfile("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            cpu = next((l.split(":", 1)[1].strip() for l in f if l.startswith("model name")), cpu)
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "torch_threads": torch.get_num_threads(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "python": platform.python_version(),
            "torch": torch.__version__, "platform": platform.platform()}


def m_spec(Ms) -> str:
    """학습 M 목록의 폴더 표기 (14, 10-14-18)"""
    return "-".join(str(m) for m in sorted(set(Ms)))


def main():
    """명령행 인자로 고른 설정에서 RL arm 학습/평가 후 베이스라인과 비교"""
    p = argparse.ArgumentParser(description="MTSR grid RL")
    p.add_argument("--arm", choices=list(ARMS), default="smdp_r8_bl6", help="보상 모드/업데이트 비율/특징 세트 조합")
    p.add_argument("--lam", type=float, default=LAMBDA_DEFAULT * 3600, help="주문 도착률 (orders/hr)")
    p.add_argument("--robots", type=int, default=None, help="학습/평가 공통 로봇 수 M")
    p.add_argument("--train-robots", type=int, nargs="+", default=None, help="학습 M (여러 개면 에피소드마다 추출)")
    p.add_argument("--eval-robots", type=int, nargs="+", default=None, help="평가 M (기본: 학습 M)")
    p.add_argument("--ws", type=int, default=NUM_WORKSTATIONS, help="워크스테이션 수 K")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="학습 시드")
    p.add_argument("--episodes", type=int, default=TOTAL_EPISODES, help="학습 에피소드 수")
    p.add_argument("--sim-time", type=float, default=SIM_TIME, help="에피소드 길이 (s), 학습/평가 공통")
    p.add_argument("--n-eval", type=int, default=N_EVAL, help="테스트 시드 수")
    p.add_argument("--select", choices=["last", "val"], default="last", help="테스트에 쓸 체크포인트 선택 방식")
    p.add_argument("--val-every", type=int, default=50, help="--select val 검증 주기 (에피소드)")
    p.add_argument("--n-val", type=int, default=N_VAL, help="검증 시드 수")
    p.add_argument("--baselines", nargs="*", choices=list(POLICIES) + list(POLICY_ALIASES),
                   default=["BL4", "BL5", "BL6"], help="함께 평가할 베이스라인 (빈 목록이면 RL만, BL7 = GR)")
    p.add_argument("--load", default=None, help="학습 없이 평가할 체크포인트 경로")
    p.add_argument("--tag", default=None, help="결과 폴더 이름 뒤에 붙일 꼬리표")
    p.add_argument("--traffic", choices=["free", "lanes"], default=GRID_TRAFFIC)
    p.add_argument("--motion", choices=["reserve", "reactive"], default=GRID_MOTION)
    p.add_argument("--scenario", default=None,
                   help="시나리오 JSON 문자열 또는 JSON 파일 경로 (scenario.Scenario 필드)")
    p.add_argument("--out", default=RL_OUT_DIR, help="결과 상위 폴더")
    p.add_argument("--summarize", action="store_true", help="저장된 결과 요약만 출력")
    p.add_argument("--fresh", action="store_true", help="resume.pt가 있어도 무시하고 처음부터 학습")
    p.add_argument("--mu-only", action="store_true",
                   help="arm이 mu=bl6이면 셀의 local_mu.json만 만들고 종료 (병렬 학습 전 1회, stage1이 호출)")
    p.add_argument("--route", choices=list(ROUTE_MODES), default="nnh",
                   help="경로 구성 규칙 (nnh / fixed_dock, RL·베이스라인 공통)")
    p.add_argument("--keep-train-mc-scale", action="store_true",
                   help="--load 시 BL6 prior 척도를 평가 레이아웃 기준으로 다시 맞추지 않음")
    p.add_argument("--log-reconstructions", default=None, metavar="PATH",
                   help="RL 테스트 중 경로 구성 호출을 JSONL로 기록 (.gz이면 gzip, route_gap.py 입력)")
    args = p.parse_args()

    if args.summarize:
        summarize(args.out)
        return

    scenario = Scenario()
    if args.scenario:
        text = open(args.scenario).read() if os.path.isfile(args.scenario) else args.scenario
        scenario = Scenario(**json.loads(text))
    cond = Condition(traffic=args.traffic, motion=args.motion, route=args.route)
    setting     = cond.setting(scenario)
    setting_dir = os.path.join(args.out, setting)
    lam = args.lam / 3600.0

    # 학습 조건 (--load이면 체크포인트 기록)
    default_M = [args.robots if args.robots is not None else NUM_ROBOTS]
    train_Ms  = sorted(set(args.train_robots or default_M))
    arm, train_seed, train_lam = args.arm, args.seed, lam
    ckpt_meta = {}
    if args.load:
        import torch
        ckpt_meta  = torch.load(args.load, map_location="cpu").get("meta", {})
        arm        = ckpt_meta.get("arm", arm)
        train_seed = ckpt_meta.get("train_seed", train_seed)
        train_lam  = ckpt_meta.get("lam", train_lam)
        train_Ms   = sorted(ckpt_meta.get("train_robots") or [ckpt_meta.get("num_robots") or train_Ms[0]])
        for name, mine, theirs in (("arm", args.arm, arm), ("seed", args.seed, train_seed)):
            if mine != theirs and mine != p.get_default(name):
                print(f"경고: --{name} {mine} 대신 체크포인트 기록 {theirs}를 씁니다")
    if args.eval_robots:
        eval_Ms = sorted(set(args.eval_robots))
    elif args.robots is not None:
        eval_Ms = [args.robots]
    else:
        eval_Ms = train_Ms
    spec  = m_spec(train_Ms)
    tag_s = f"_{args.tag}" if args.tag else ""

    cfg = ARMS[arm]
    layouts = {}

    def layout_for(M: int) -> GridWarehouseLayout:
        """M대용 레이아웃 (주차·대기열 셀 크기가 M에 따라 달라짐, M마다 한 번 생성)"""
        if M not in layouts:
            layouts[M] = cond.layout(M, args.ws, scenario)
        return layouts[M]

    # max_candidates는 학습/평가 M 중 최댓값 기준
    agent_kw = dict(robot_capacity=ROBOT_CAPACITY, seed=train_seed, sim_time=args.sim_time,
                    num_robots=max(train_Ms + eval_Ms), num_workstations=args.ws,
                    feature_set=cfg["feature_set"])
    if cfg.get("agent") == "global":
        agent = GlobalDQNAgent(dist_norm=layout_for(train_Ms[0]).dist_norm, alpha_max=cfg.get("alpha_max"),
                               huber=cfg.get("loss") == "huber", **agent_kw)
    else:
        agent = DQNAgent(**agent_kw)
    train_sim_kw = {k: cfg[k] for k in SIM_KEYS if k in cfg}
    run_info = {"arm": arm, **cfg, "train_seed": train_seed, "train_robots": train_Ms,
                "train_lam_hr": train_lam * 3600, "eval_lam_hr": args.lam, "num_workstations": args.ws,
                "traffic": cond.traffic, "motion": cond.motion, "scenario": scenario.to_dict(),
                "condition": cond.to_dict(), "setting": setting,
                "sim_time": args.sim_time, "tag": args.tag, "n_eval": args.n_eval, "env": runtime_env()}
    print(f"arm={arm} ({cfg}), λ={args.lam:.0f}/hr, 학습 M={train_Ms}, 평가 M={eval_Ms}, K={args.ws}, "
          f"{setting}, 특징 {agent.feat_dim}차원")
    cell_dir = os.path.join(setting_dir, f"lam{args.lam:.0f}_M{spec}_K{args.ws}")
    if args.mu_only:
        if cfg.get("mu") == "bl6":
            load_or_estimate_local_mu(cell_dir, {M: layout_for(M) for M in train_Ms}, agent, lam, args.sim_time)
        return

    if args.load:
        agent.load(args.load)
        run_info.update(checkpoint=os.path.abspath(args.load), select=ckpt_meta.get("select"),
                        selected_episode=ckpt_meta.get("selected_episode"))
    else:
        train_dir = os.path.join(cell_dir, f"{arm}_s{train_seed}{tag_s}")
        os.makedirs(train_dir, exist_ok=True)
        print(f"학습 결과 {train_dir}")
        resume_path = os.path.join(train_dir, "resume.pt")
        if args.fresh and os.path.isfile(resume_path):
            os.remove(resume_path)
        tr_layouts = {M: layout_for(M) for M in train_Ms}
        if cfg.get("mu") == "bl6" and not os.path.isfile(resume_path):
            agent.mu_table = load_or_estimate_local_mu(cell_dir, tr_layouts, agent, lam, args.sim_time)
        run_info["resumed"] = os.path.isfile(resume_path)      # True이면 train_wall_s는 이어한 구간만
        tr_store, curve, val_store = MetricsStore(), TrainingCurveStore(), MetricsStore()
        validate = None
        if args.select == "val":
            validate = make_validator(tr_layouts, agent, lam, val_seeds(args.n_val), args.sim_time,
                                      cfg["reward_mode"], val_store, sim_kw={**train_sim_kw, **cond.sim_kw})
        wall = time.perf_counter()
        best = train(tr_layouts, agent, tr_store, lam=lam, seed=train_seed, total_episodes=args.episodes,
                     sim_time=args.sim_time, label=arm, curve_store=curve,
                     update_ratio=cfg["update_ratio"], reward_mode=cfg["reward_mode"],
                     verbose_every=max(1, args.episodes // 20), validate=validate, val_every=args.val_every,
                     resume_path=resume_path, val_store=val_store, sim_kw={**train_sim_kw, **cond.sim_kw})
        train_wall = time.perf_counter() - wall
        if agent.mu_table is not None:
            run_info["local_mu_table"] = {"lam_hr": [l * 3600 for l in agent.mu_table[0]], "mu_s": agent.mu_table[1]}
        if hasattr(agent.policy_net, "alpha"):
            run_info["alpha_final"] = agent.policy_net.alpha
        ep_walls   = [p.wall_s for p in curve.records]
        run_info.update(train_wall_s=train_wall,
                        train_episode_wall_s_mean=sum(ep_walls) / max(1, len(ep_walls)),
                        train_sim_s_total=sum(p.sim_s for p in curve.records),
                        train_n_updates=sum(p.n_updates for p in curve.records))
        print(f"학습 시간 {train_wall / 3600:.2f} h (에피소드당 {run_info['train_episode_wall_s_mean']:.1f} s, "
              f"검증 포함 총계)")
        tr_store.save_csv(os.path.join(train_dir, "train.csv"))
        curve.save_csv(os.path.join(train_dir, "curve.csv"))
        val_store.save_csv(os.path.join(train_dir, "val.csv"))
        meta = {"lam": lam, "num_robots": train_Ms[0] if len(train_Ms) == 1 else None,
                "train_robots": train_Ms, "num_workstations": args.ws, "robot_capacity": ROBOT_CAPACITY,
                "episodes": args.episodes, "train_seed": train_seed, "arm": arm, **cfg,
                "traffic": cond.traffic, "motion": cond.motion, "scenario": scenario.to_dict(),
                "condition": cond.to_dict(),
                "tag": args.tag, "select": args.select, "selected_episode": args.episodes}
        agent.save(os.path.join(train_dir, "agent_last.pt"), meta)
        if best is not None:
            agent.policy_net.load_state_dict(best["policy_state"])
            agent.target_net.load_state_dict(best["target_state"])
            meta.update(selected_episode=best["episode"], val_score=best["score"])
            print(f"검증 선택: 에피소드 {best['episode']} (검증 평균 흐름시간 {best['score']:.1f} s)")
        agent.save(os.path.join(train_dir, "agent.pt"), meta)
        run_info.update(checkpoint=os.path.abspath(os.path.join(train_dir, "agent.pt")),
                        select=args.select, selected_episode=meta["selected_episode"])
        with open(os.path.join(train_dir, "run.json"), "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=1, ensure_ascii=False)

    seeds = test_seeds(args.n_eval)
    if args.log_reconstructions:
        if os.path.isfile(args.log_reconstructions):
            os.remove(args.log_reconstructions)          # 이번 실행 기록만 남김 (시드별로 덧붙임)
        run_info["log_reconstructions"] = os.path.abspath(args.log_reconstructions)
    train_K = ckpt_meta.get("num_workstations", args.ws)   # 전역 상태 차원이 K에 의존하므로 K가 다르면 로드 불가
    train_mc_scale = getattr(agent, "mc_scale", None)       # GlobalDQNAgent만 (체크포인트/학습 레이아웃 값)
    for M in eval_Ms:
        base_dir = os.path.join(setting_dir, f"lam{args.lam:.0f}_M{M}_K{args.ws}")
        run_name = f"{arm}_s{train_seed}"
        if spec != str(M):
            run_name += f"_tr{spec}"
        if abs(train_lam - lam) * 3600 > 0.5:
            run_name += f"_trlam{train_lam * 3600:.0f}"
        run_name += cond.tag + tag_s
        out = os.path.join(base_dir, run_name)
        if args.load and os.path.abspath(out) == os.path.dirname(os.path.abspath(args.load)) and not args.tag:
            raise SystemExit(f"--load 결과 폴더 {out}가 체크포인트 학습 폴더와 같습니다. --tag를 지정하세요 "
                             f"(원래 eval.csv 보호)")
        os.makedirs(out, exist_ok=True)
        L = layout_for(M)
        mc_info = None
        if train_mc_scale is not None:
            mc_info = {"train": train_mc_scale, "eval_layout": agent.mc_scale_for(L.dist_norm),
                       "applied": bool(args.load) and not args.keep_train_mc_scale}
            if mc_info["applied"]:
                agent.set_mc_scale(L.dist_norm)
            mc_info["used"] = agent.mc_scale
        wall  = time.perf_counter()
        store = evaluate_policy(L, "RL", lam, seeds, args.sim_time, agent=agent, reward_mode=cfg["reward_mode"],
                                sim_kw=cond.sim_kw, log_reconstructions=args.log_reconstructions)
        eval_wall = time.perf_counter() - wall
        store.save_csv(os.path.join(out, "eval.csv"))
        rs = store.records
        lat = {k: sum(getattr(r, k) for r in rs) / len(rs)
               for k in ("decision_ms_mean", "decision_ms_p95", "decision_full_ms_mean", "decision_full_ms_p95")}
        print(f"결정 지연 평균 {lat['decision_ms_mean']:.2f} ms (p95 {lat['decision_ms_p95']:.2f}), "
              f"배정·재계획 포함 {lat['decision_full_ms_mean']:.2f} ms (p95 {lat['decision_full_ms_p95']:.2f}), "
              f"평가 {eval_wall:.0f} s")
        with open(os.path.join(out, "run.json"), "w", encoding="utf-8") as f:
            json.dump({**run_info, "eval_robots": M, "run": run_name, "test_seeds": seeds,
                       "eval_wall_s": eval_wall, "eval_latency_seed_mean": lat, "mc_scale": mc_info,
                       "train_num_workstations": train_K}, f,
                      indent=1, ensure_ascii=False)
        labels = ["RL"]
        if args.baselines:
            bl = evaluate_baselines_cached(L, args.baselines, lam, seeds, args.sim_time,
                                           os.path.join(base_dir, "baselines", "eval.csv"), sim_kw=cond.sim_kw)
            for r in bl.records:
                store.add(r)
            labels += [POLICIES[policy_key(k)][0] for k in args.baselines]
        print_comparison(store, labels, lam=lam, num_robots=M, last_n=args.n_eval,
                         title=f"RL ({run_name}) vs baselines")
        print(f"저장: {out}")

    # 테스트 평가까지 끝났으므로 이어하기 파일 삭제 (도중에 끊기면 남아 있어 재실행 시 학습을 건너뜀)
    if not args.load and os.path.isfile(resume_path):
        os.remove(resume_path)


if __name__ == "__main__":
    main()
