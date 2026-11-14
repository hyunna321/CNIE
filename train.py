"""제안 모델(RL 배정) 학습: 학습 루프, 검증 시드 기반 체크포인트 선택, 보상 중심화 기준 μ(λ) / ρ(λ) 추정

사용법:
    python train.py --config experiments/configs/base.yaml                # YAML 설정으로 학습 (명령행 인자가 우선)
    python train.py --arm local_r8_ddqn_v2 --episodes 200 --seed 1        # 인자만으로 학습 → runs/default
    python train.py --select val --val-every 50 --n-val 5                 # 검증 시드로 체크포인트 선택
    python train.py --train-robots 10 14 18                               # 다중 M 학습

arm = 보상 설정 (env.rl.ARMS): 보상 종류, 할인 방식, local 비용 항 계수
실험 이름(prefix) = YAML name (없으면 파일 이름). 결과는 runs/<prefix>/에 저장하고 실행 설정을 학습 폴더의 config_train.yaml로 남김
같은 학습 폴더에 다른 설정으로 다시 학습하면 중단 (민감도 분석은 값마다 실험 이름을 다르게)
테스트 시드 평가와 베이스라인 비교는 experiments/run_compare.py

이어하기: 학습 폴더에 resume.pt를 CKPT_EVERY 에피소드마다 저장. 같은 명령을 다시 실행하면 이어서 학습
    (--fresh이면 무시하고 처음부터). 학습이 끝나면 resume.pt 삭제

결과 (spec = 학습 M, 여러 개면 "10-14-18")
    {out}/{setting}/lam{λ}_M{spec}_K{K}/{arm}_s{seed}[_{tag}]/
        train.csv, curve.csv, val.csv, agent.pt (테스트에 쓸 가중치), agent_last.pt, run.json
    μ(λ) 표: {out}/{setting}/lam{λ}_M{spec}_K{K}/local_mu*.json (env.rl.mu_filename, 같은 셀의 학습 시드 공유)
"""
import argparse
import copy
import json
import math
import os
import platform
import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import yaml

from agents import (BATCH_SIZE, DQNAgent, EPSILON, LR, REPLAY_BUFFER_SIZE, TARGET_UPDATE_STEPS, TOTAL_EPISODES)
from evaluation import (Condition, MetricsStore, N_VAL, TrainingCurveStore, TrainingPoint, evaluate_policy,
                        train_episode_seed, val_seeds)
from env.battery import BatteryConfig
from env.layout import GRID_MOTION, GRID_TRAFFIC, NUM_ROBOTS, NUM_WORKSTATIONS, ROBOT_CAPACITY
from env.orders import DEFAULT_SEED, LAMBDA_DEFAULT, Scenario, SIM_TIME
from env.rl import ARMS, Arm, DEFAULT_ARM, RLSimulator, mu_filename
from env.robot import ROUTE_MODES
from td import TDAgent, rl_sim_class

# 학습 계수
VERBOSE_EVERY         = 50
CURRICULUM_LAM_MARGIN = 0.7    # 앞쪽 절반 에피소드의 λ 하한 = lam × margin
UPDATE_RATIO          = 8      # 결정 N개당 train_step 1회
CKPT_EVERY            = 50     # 이어하기 체크포인트 저장 주기 (에피소드)
# local 보상 고정 기준점 μ(λ): 학습 전 BL6 prior 정책으로 [λ·CURRICULUM_LAM_MARGIN, λ] 격자 롤아웃
# 셀(λ, M, K, 조건)·계수마다 한 번 계산해 local_mu*.json에 캐시, 모든 학습 시드가 같은 표 사용
LOCAL_MU_POINTS  = 5      # λ 격자 점 수 (사이 λ는 선형 보간)
LOCAL_MU_SEEDS   = 10     # 격자 점당 에피소드 수 (학습 시드 0 구간의 별도 번호, 학습/검증/테스트와 겹치지 않음)
LOCAL_MU_WORKERS = 5      # 병렬 프로세스 수 (격자 점 단위)

MU_SEED           = 0         # μ(λ) 추정 에피소드 시드 구간 (학습 시드 0, 실제 학습 시드 1~와 겹치지 않음)
MU_EPISODE_OFFSET = 90_000    # μ(λ) 추정 에피소드 번호 (학습 에피소드 1..TOTAL과 겹치지 않음)


# 보상 중심화 기준 (local: μ(λ) = 결정당 평균 비용, order_increment: ρ(λ) = 시간 평균 N(t))

def _mu_point(job) -> tuple:
    """작업자 프로세스: λ 격자 점 하나의 에피소드별 결정당 평균 국소 비용과 항별 평균 (계수 곱 전)"""
    agent, layouts, l, i, n_seeds, sim_time, arm, sim_kw = job
    costs, terms = [], []
    for M, L in sorted(layouts.items()):
        sim = rl_sim_class(arm, train=False)(L, agent, lam=l, seed=MU_SEED, sim_time=sim_time, num_robots=M,
                                             robot_capacity=L.robot_capacity, eval_mode=True, arm=arm, **sim_kw)
        for j in range(n_seeds):
            sim.reset(train_episode_seed(MU_SEED, MU_EPISODE_OFFSET + i * n_seeds + j))
            sim.run()
            costs.append(sim.reward.center_stat())
            terms.append(sim.last_cost_terms)
    return costs, terms


def estimate_local_mu(layouts: Dict[int, object], agent: DQNAgent, lam: float, sim_time: float, arm: Arm,
                      n_points: int = LOCAL_MU_POINTS, n_seeds: int = LOCAL_MU_SEEDS,
                      workers: int = LOCAL_MU_WORKERS, verbose: bool = True,
                      margin: float = CURRICULUM_LAM_MARGIN, sim_kw: Optional[Dict] = None) -> dict:
    """보상 중심화 기준표 μ(λ) / ρ(λ) (학습 전 호출, 기준 종류는 arm 보상의 Reward.center)

    학습 전 에이전트(f의 마지막 층 = 0이라 argmax = BL6 prior, 가중치 초기화와 무관)를 ε = 0으로
    [λ·margin, λ] 격자에서 실행해 에피소드마다 Reward.center_stat()을 구함 (local: 결정당 평균 국소 비용). 에피소드 시드는 학습 시드와 무관한 공통 구간이라
    같은 셀의 모든 학습 시드가 같은 표를 씀. 학습 중 보상 기준점이 움직이지 않음
    BL6 prior 궤적은 arm 계수와 무관하고 μ만 계수(arm.local_w)에 따라 달라짐

    출력:
        {"lam": [orders/s], "mu": [s], "se": [s, 표준오차], "n_seeds", "sim_time", ...}  ("lam", "mu" → agent.mu_table)
        "terms": 격자 점마다 결정당 평균 [자기 지연, 탑재 지연 추정, 큐잉 지연] (계수 곱 전, 항 크기 비교용)
    """
    assert agent.mu_table is None and agent.steps == 0, "학습 전에 호출해야 합니다"
    lams = [float(l) for l in np.linspace(lam * margin, lam, n_points)] if n_points > 1 else [lam]
    jobs = [(agent, layouts, l, i, n_seeds, sim_time, arm, sim_kw or {}) for i, l in enumerate(lams)]
    if workers > 1:
        with ProcessPoolExecutor(min(workers, len(jobs))) as ex:
            res = list(ex.map(_mu_point, jobs))
    else:
        res = [_mu_point(j) for j in jobs]
    all_costs = [c for c, _ in res]
    terms = [np.mean(np.asarray(t, dtype=float), axis=0).tolist() for _, t in res]
    mus = [float(np.mean(c)) for c in all_costs]
    ses = [float(np.std(c, ddof=1) / math.sqrt(len(c))) if len(c) > 1 else float("nan") for c in all_costs]
    if verbose:
        for l, m, s, c in zip(lams, mus, ses, all_costs):
            print(f"  μ(λ={l * 3600:5.1f}/hr) = {m:7.2f} ± {s:5.2f} s  ({len(c)} 에피소드)", flush=True)
    out = {"lam": lams, "mu": mus, "se": ses, "n_seeds": n_seeds, "n_points": n_points,
           "sim_time": sim_time, "margin": margin, "Ms": sorted(layouts), "terms": terms}
    if arm.local_w != (1.0, 1.0):
        out["local_w"] = list(arm.local_w)
    return out


def load_or_estimate_local_mu(cell_dir: str, layouts: Dict[int, object], agent: DQNAgent, lam: float,
                              sim_time: float, arm: Arm, margin: float = CURRICULUM_LAM_MARGIN,
                              sim_kw: Optional[Dict] = None) -> tuple:
    """셀 폴더의 μ(λ) 표(env.rl.mu_filename)를 읽거나 (조건이 같을 때) 새로 추정해 저장, agent.mu_table 형식으로 반환"""
    path = os.path.join(cell_dir, mu_filename(arm))
    want = {"n_seeds": LOCAL_MU_SEEDS, "n_points": LOCAL_MU_POINTS, "sim_time": sim_time,
            "margin": margin, "Ms": sorted(layouts)}
    if arm.local_w != (1.0, 1.0):          # 기본 계수 파일에는 local_w 키가 없음
        want["local_w"] = list(arm.local_w)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            tab = json.load(f)
        if all(tab.get(k) == v for k, v in want.items()) and abs(tab["lam"][-1] - lam) < 1e-9:
            print(f"local 보상 기준점 μ(λ): {path} 사용", flush=True)
            return tab["lam"], tab["mu"]
        print(f"local 보상 기준점 μ(λ): {path} 조건 불일치, 다시 추정", flush=True)
    print(f"local 보상 기준점 μ(λ) 추정 (BL6 prior 정책, ε = 0, 격자 {LOCAL_MU_POINTS}점 × {LOCAL_MU_SEEDS} 에피소드)",
          flush=True)
    tab = estimate_local_mu(layouts, agent, lam, sim_time, arm, margin=margin, sim_kw=sim_kw)
    os.makedirs(cell_dir, exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(tab, f, indent=1)
    os.replace(tmp, path)
    return tab["lam"], tab["mu"]


# 학습

def save_resume(path: str, state: Dict):
    """이어하기 체크포인트를 원자적으로 저장 (임시 파일에 쓴 뒤 교체)"""
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_resume(path: str) -> Dict:
    """save_resume으로 저장한 체크포인트 (리플레이 버퍼 등 파이썬 객체 포함)"""
    return torch.load(path, map_location="cpu", weights_only=False)


def train(layouts: Union[Dict[int, object], object], agent: DQNAgent, store: MetricsStore, arm: Arm,
          lam: float = LAMBDA_DEFAULT, seed: int = DEFAULT_SEED,
          total_episodes: int = TOTAL_EPISODES, sim_time: float = SIM_TIME,
          label: str = "RL", verbose_every: int = VERBOSE_EVERY,
          curve_store: Optional[TrainingCurveStore] = None,
          validate: Optional[Callable[[int], float]] = None, val_every: int = 0,
          resume_path: Optional[str] = None, ckpt_every: int = CKPT_EVERY,
          val_store: Optional[MetricsStore] = None, sim_kw: Optional[Dict] = None,
          update_ratio: int = UPDATE_RATIO, curriculum_margin: float = CURRICULUM_LAM_MARGIN) -> Optional[dict]:
    """RL 에이전트 학습

    - 앞쪽 절반 에피소드: λ를 [lam × CURRICULUM_LAM_MARGIN, lam]에서 무작위로 뽑는 커리큘럼, 나머지는 lam
      (arm.curriculum = False이면 모든 에피소드 lam, 비정상 도착 시나리오는 lam이 근무 시간 평균)
    - 다중 M: 에피소드마다 M을 layouts 키에서 균등하게 뽑고 그 M의 레이아웃(주차·대기열 셀)으로 실행
    - 에피소드 주문 시드는 evaluation.train_episode_seed (검증/테스트 시드 구간과 겹치지 않음)
    - 에피소드마다 전이를 버퍼에 넣고 (전이 수 / UPDATE_RATIO)회 train_step
    - validate가 있으면 val_every 에피소드마다 (그리고 마지막에) 호출해 값이 가장 작은 시점의 가중치 보관
    - resume_path가 있으면 ckpt_every 에피소드마다 (그리고 마지막에) 전체 학습 상태를 저장하고,
      시작 시 그 파일이 있으면 저장된 다음 에피소드부터 이어서 학습 (에피소드 시드·λ·M 순서 동일)

    입력:
        layouts: {M: GridWarehouseLayout} 또는 GridWarehouseLayout 하나 (단일 M)
        agent: 학습할 DQNAgent (in-place 갱신, max_candidates는 최대 M 기준이어야 함)
        store: 에피소드별 결과를 기록할 MetricsStore
        arm: 보상 설정 (env.rl.Arm)
        lam: 목표 주문 도착률 (orders/s)
        seed: 학습 시드 (에피소드 시드 구간, λ/M 샘플링 난수)
        validate: 에피소드 번호 -> 검증 점수 (작을수록 좋음, 검증 시드로만 계산, make_validator)
        val_store: 검증 결과 store (체크포인트에 함께 저장해 이어하기 후에도 val.csv가 이어지도록)
        sim_kw: SimulatorBase 추가 인자 (route)
    출력:
        validate가 있으면 {"episode", "score", "policy_state", "target_state"} (최고 검증 시점), 없으면 None
    """
    if not isinstance(layouts, dict):
        layouts = {layouts.num_robots: layouts}
    Ms = sorted(layouts)
    assert agent.max_candidates >= max(Ms), (agent.max_candidates, Ms)

    lam_lo, lam_hi = lam * curriculum_margin, lam
    ring = arm.buffer == "ring"
    online = arm.update == "online"
    sims = {M: rl_sim_class(arm)(L, agent, lam=lam, seed=seed, sim_time=sim_time, num_robots=M,
                                 robot_capacity=L.robot_capacity, arm=arm, **(sim_kw or {}))
            for M, L in layouts.items()}
    if ring:
        for s in sims.values():
            s.update_every = update_ratio
    rng   = random.Random(seed)
    best  = None
    start = 1
    if resume_path and os.path.isfile(resume_path):
        st = load_resume(resume_path)
        assert st["total_episodes"] == total_episodes, (st["total_episodes"], total_episodes)
        agent.load_full_state(st["agent"])
        rng.setstate(st["train_rng"])
        best, start = st["best"], st["episode"] + 1
        store.records = st["store"]
        if curve_store is not None:
            curve_store.records = st["curve"]
        if val_store is not None:
            val_store.records = st["val"]
        print(f"  이어하기: {resume_path} (에피소드 {st['episode']}까지 완료, {start}부터)", flush=True)

    def checkpoint(ep: int):
        save_resume(resume_path, {
            "episode": ep, "total_episodes": total_episodes, "agent": agent.full_state(),
            "train_rng": rng.getstate(), "best": best, "store": store.records,
            "curve": curve_store.records if curve_store is not None else [],
            "val": val_store.records if val_store is not None else []})

    for ep in range(start, total_episodes + 1):
        wall    = time.perf_counter()
        ep_seed = train_episode_seed(seed, ep)
        ep_lam  = rng.uniform(lam_lo, lam_hi) if arm.curriculum and ep <= total_episodes // 2 else lam
        M       = Ms[0] if len(Ms) == 1 else rng.choice(Ms)

        sim = sims[M]
        sim.lam = sim.order_mgr.lam = ep_lam
        sim.reset(ep_seed)
        transitions = sim.run()
        sim_s = time.perf_counter() - wall

        for trans in transitions:
            agent.push(*trans)
        if arm.truncate and not ring:
            agent.flush_nstep()
        loss = None
        n_updates = max(1, int((sim.n_transitions if ring else len(transitions)) / update_ratio))
        q_sum = td_sum = 0.0
        n_done = 0
        if online:             # 에피소드 도중 이미 업데이트함 (TDSimulator)
            n_updates, n_done, loss, q_sum, td_sum = sim.n_updates, sim.n_trained, sim.last_loss, sim.q_sum, sim.td_sum
        for _ in range(0 if online else n_updates):
            step_loss = agent.train_step()
            if step_loss is not None:
                loss = step_loss
                q_sum  += agent.last_stats[0]
                td_sum += agent.last_stats[1]
                n_done += 1
        agent.end_episode(total_episodes)
        alpha = agent.policy_net.alpha

        result = sim.build_result(label, ep, ep_lam)
        result.seed   = ep_seed
        result.wall_s = time.perf_counter() - wall
        store.add(result)
        if curve_store is not None:
            curve_store.add(TrainingPoint(
                label=label, episode=ep, reward=result.reward,
                loss=loss if loss is not None else float("nan"),
                epsilon=agent.epsilon, n_completed=result.n_completed,
                avg_flow_time=result.avg_flow_time, num_robots=M,
                wall_s=result.wall_s, sim_s=sim_s, n_updates=n_updates,
                lam=ep_lam * 3600, alpha=alpha,
                q_mean=q_sum / n_done if n_done else float("nan"),
                td_abs=td_sum / n_done if n_done else float("nan"),
                local_mu=sim.last_mu, local_cost=sim.last_mean_cost,
                cost_own=sim.last_cost_terms[0], cost_ext=sim.last_cost_terms[1],
                cost_queue=sim.last_cost_terms[2]))

        if ep % verbose_every == 0:
            loss_s = f"{loss:.4f}" if loss is not None else "warmup"
            print(f"  [Ep {ep:4d}] {label}  M={M}  λ={ep_lam * 3600:4.0f}/hr  "
                  f"완료={result.n_completed:5d}  flow={result.avg_flow_time:6.1f}s  "
                  f"totes/trip={result.avg_totes_per_trip:.2f}  "
                  f"reward/order={result.reward:.4f}  ε={agent.epsilon:.3f}  loss={loss_s}  α={alpha:.3f}",
                  flush=True)

        if validate is not None and ((val_every and ep % val_every == 0) or ep == total_episodes):
            # 검증이 탐색 난수를 소모해 학습 궤적이 바뀌지 않도록 보존
            rng_state = agent.rng.getstate()
            score = validate(ep)
            agent.rng.setstate(rng_state)
            is_best = best is None or score < best["score"]
            print(f"  [Val {ep:4d}] {label}  score={score:.2f}" + ("  *best" if is_best else ""), flush=True)
            if is_best:
                best = {"episode": ep, "score": score,
                        "policy_state": copy.deepcopy(agent.policy_net.state_dict()),
                        "target_state": copy.deepcopy(agent.target_net.state_dict())}

        if resume_path and (ep % ckpt_every == 0 or ep == total_episodes):
            checkpoint(ep)
    return best


def make_validator(layouts: Dict[int, object], agent: DQNAgent, lam: float, seeds: List[int],
                   sim_time: float, arm: Arm, store: MetricsStore,
                   metric: str = "avg_flow_time", sim_kw: Optional[Dict] = None) -> Callable[[int], float]:
    """검증 시드로만 평가하는 train(validate=...)용 함수

    점수 = 모든 학습 M × 검증 시드의 metric 평균 (작을수록 좋음). 결과는 store에 쌓임 (episode = 학습 에피소드)
    """
    def validate(ep: int) -> float:
        vals = []
        for M, L in sorted(layouts.items()):
            res = evaluate_policy(L, "RL", lam, seeds, sim_time, label=f"val_ep{ep}", agent=agent, arm=arm,
                                  track_decisions=False, verbose=False, sim_kw=sim_kw)
            for r in res.records:
                r.episode = ep
                store.add(r)
                vals.append(getattr(r, metric))
        return sum(vals) / len(vals)
    return validate


# 실행 설정

# 다른 실행 파일 전용 키 (같은 YAML을 train.py와 experiments/run_compare.py가 함께 씀, 여기서는 무시)
_OTHER_TOOL_KEYS = {"n_eval", "baselines", "eval_robots", "train_robots", "episodes", "select", "val_every", "n_val", "lr",
                    "batch_size", "buffer_size", "target_update", "epsilon", "update_ratio", "curriculum_margin",
                    "arm", "seed", "keep_train_mc_scale", "no_collisions"}
_NOT_SAVED = {"config", "fresh", "mu_only", "summarize"}


def add_common_args(p: argparse.ArgumentParser):
    """train.py와 비교 실험이 공유하는 실험 조건 인자 (YAML 키 = 인자 이름의 - 를 _ 로)"""
    p.add_argument("--config", default=None, help="실험 설정 YAML (experiments/configs/)")
    p.add_argument("--out", default=None, help="결과 폴더 (기본 runs/<실험 이름>)")
    p.add_argument("--tag", default=None, help="결과 폴더 이름 뒤에 붙일 꼬리표")
    p.add_argument("--lam", type=float, default=LAMBDA_DEFAULT * 3600, help="주문 도착률 (orders/hr)")
    p.add_argument("--robots", type=int, default=None, help="로봇 수 M")
    p.add_argument("--ws", type=int, default=NUM_WORKSTATIONS, help="워크스테이션 수 K")
    p.add_argument("--sim-time", type=float, default=SIM_TIME, help="에피소드 길이 (s)")
    p.add_argument("--traffic", choices=["free", "lanes"], default=GRID_TRAFFIC)
    p.add_argument("--motion", choices=["reserve", "reactive"], default=GRID_MOTION)
    p.add_argument("--route", choices=list(ROUTE_MODES), default="nnh", help="경로 구성 규칙")
    p.add_argument("--grid", type=int, nargs=3, default=None, metavar=("BLOCKS", "AISLES", "SHELVES"),
                   help="그리드 크기 (블록 수, 블록당 통로 수, 선반 행 수). 기본 3 14 20")
    p.add_argument("--scenario", default=None, help="시나리오 JSON 문자열 또는 JSON 파일 경로 (env.orders.Scenario 필드)")
    p.add_argument("--battery", default=None, help="배터리 제약 JSON 문자열 또는 JSON 파일 경로 (env.battery.BatteryConfig 필드)")


def parse_args(p: argparse.ArgumentParser, argv=None) -> argparse.Namespace:
    """--config YAML을 인자 기본값으로 적용해 파싱 (명령행 우선). --out 기본값 = runs/<실험 이름>"""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    config = pre.parse_known_args(argv)[0].config
    name = "default"
    if config:
        with open(config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        name = cfg.pop("name", os.path.splitext(os.path.basename(config))[0])
        for key in ("scenario", "battery"):
            if isinstance(cfg.get(key), dict):
                cfg[key] = json.dumps(cfg[key], sort_keys=True)
        dests = {a.dest for a in p._actions}
        bad = set(cfg) - dests - _OTHER_TOOL_KEYS
        if bad:
            raise SystemExit(f"{config}: 알 수 없는 설정 키 {sorted(bad)}")
        p.set_defaults(**{k: v for k, v in cfg.items() if k in dests})
    args = p.parse_args(argv)
    args.out = args.out or os.path.join("runs", name)
    return args


def resolved_config(args: argparse.Namespace) -> dict:
    """실제 사용한 설정 (config_*.yaml 내용)"""
    return {k: v for k, v in vars(args).items() if k not in _NOT_SAVED}


def save_config(args: argparse.Namespace, directory: str, tool: str):
    """실제 사용한 설정을 실행 폴더의 config_{tool}.yaml로 저장"""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"config_{tool}.yaml")
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_config(args), f, allow_unicode=True, sort_keys=False)
    os.replace(tmp, path)


def check_same_config(args: argparse.Namespace, directory: str):
    """이 학습 폴더의 이전 설정과 지금 설정이 다르면 중단 (다른 계수의 결과를 덮어쓰거나 이어받지 않도록)"""
    path = os.path.join(directory, "config_train.yaml")
    if args.fresh or not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        old = yaml.safe_load(f) or {}
    new = yaml.safe_load(yaml.safe_dump(resolved_config(args)))
    diff = sorted(k for k in set(old) | set(new) if k != "out" and old.get(k) != new.get(k))
    if diff:
        raise SystemExit(f"{directory}에 다른 설정의 학습 기록이 있습니다 (다른 키: {diff}). "
                         f"실험 이름(name, --out)이나 --tag를 바꾸거나 --fresh로 처음부터 학습하세요")


def make_condition(args: argparse.Namespace) -> Condition:
    return Condition(traffic=args.traffic, motion=args.motion, route=args.route,
                     grid=tuple(args.grid) if args.grid else None, battery=BatteryConfig.from_arg(args.battery))


def runtime_env() -> dict:
    """실행 환경 기록 (학습 시간/결정 지연 보고용): CPU, 코어/스레드 수, Python·PyTorch 버전"""
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


def main(argv=None):
    """설정에 따라 RL arm 하나를 학습해 체크포인트 저장"""
    p = argparse.ArgumentParser(description="MTSR RL 학습", formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    add_common_args(p)
    p.add_argument("--arm", choices=list(ARMS), default=DEFAULT_ARM, help="보상 설정 (env.rl.ARMS)")
    p.add_argument("--train-robots", type=int, nargs="+", default=None, help="학습 M (여러 개면 에피소드마다 추출)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="학습 시드")
    p.add_argument("--episodes", type=int, default=TOTAL_EPISODES, help="학습 에피소드 수")
    p.add_argument("--select", choices=["last", "val"], default="last", help="테스트에 쓸 체크포인트 선택 방식")
    p.add_argument("--val-every", type=int, default=50, help="--select val 검증 주기 (에피소드)")
    p.add_argument("--n-val", type=int, default=N_VAL, help="검증 시드 수")
    p.add_argument("--fresh", action="store_true", help="resume.pt가 있어도 무시하고 처음부터 학습")
    p.add_argument("--mu-only", action="store_true", help="arm이 μ(λ)를 쓰면 셀의 μ 표만 만들고 종료 (병렬 학습 전 1회)")
    # 학습 계수 (민감도 분석용)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--buffer-size", type=int, default=REPLAY_BUFFER_SIZE)
    p.add_argument("--target-update", type=int, default=TARGET_UPDATE_STEPS)
    p.add_argument("--epsilon", type=float, default=EPSILON)
    p.add_argument("--update-ratio", type=int, default=UPDATE_RATIO)
    p.add_argument("--curriculum-margin", type=float, default=CURRICULUM_LAM_MARGIN)
    args = parse_args(p, argv)

    scenario = Scenario.from_arg(args.scenario)
    cond     = make_condition(args)
    setting  = cond.setting(scenario)
    lam      = args.lam / 3600.0
    train_Ms = sorted(set(args.train_robots or [args.robots if args.robots is not None else NUM_ROBOTS]))
    spec     = m_spec(train_Ms)
    tag_s    = f"_{args.tag}" if args.tag else ""
    arm      = ARMS[args.arm]
    layouts  = {M: cond.layout(M, args.ws, scenario) for M in train_Ms}
    if arm.buffer == "ring" and len(train_Ms) > 1:
        raise SystemExit("ring buffer는 단일 M 학습만 지원합니다")
    agent_kw = dict(robot_capacity=ROBOT_CAPACITY, dist_norm=layouts[train_Ms[0]].dist_norm, seed=args.seed,
                    sim_time=args.sim_time, num_robots=max(train_Ms), num_workstations=args.ws,
                    n_step=arm.n_step, features=arm.features, lr=args.lr, batch_size=args.batch_size,
                    buffer_size=args.buffer_size, target_update=args.target_update, epsilon=args.epsilon)
    agent = TDAgent(beta=arm.beta_value, **agent_kw) if arm.buffer == "ring" else DQNAgent(**agent_kw)
    print(f"arm={args.arm} ({arm}), λ={args.lam:.0f}/hr, 학습 M={train_Ms}, K={args.ws}, {setting}, 특징 {agent.feat_dim}차원")
    cell_dir = os.path.join(args.out, setting, f"lam{args.lam:.0f}_M{spec}_K{args.ws}")
    mu_kw = {"battery": cond.battery} if cond.battery is not None else {}     # BL6 prior 롤아웃도 같은 배터리 조건
    if args.mu_only:
        if arm.uses_mu:
            load_or_estimate_local_mu(cell_dir, layouts, agent, lam, args.sim_time, arm, margin=args.curriculum_margin,
                                      sim_kw=mu_kw)
        return

    train_dir = os.path.join(cell_dir, f"{args.arm}_s{args.seed}{tag_s}")
    check_same_config(args, train_dir)
    save_config(args, train_dir, "train")
    print(f"학습 결과 {train_dir}")
    resume_path = os.path.join(train_dir, "resume.pt")
    if args.fresh and os.path.isfile(resume_path):
        os.remove(resume_path)
    if arm.uses_mu and not os.path.isfile(resume_path):
        agent.mu_table = load_or_estimate_local_mu(cell_dir, layouts, agent, lam, args.sim_time, arm,
                                                   margin=args.curriculum_margin, sim_kw=mu_kw)
    run_info = {"arm": args.arm, **arm.to_dict(), "train_seed": args.seed, "train_robots": train_Ms,
                "train_lam_hr": args.lam, "num_workstations": args.ws, "traffic": cond.traffic, "motion": cond.motion,
                "scenario": scenario.to_dict(), "condition": cond.to_dict(), "setting": setting,
                "sim_time": args.sim_time, "tag": args.tag, "env": runtime_env(),
                "hyper": {k: getattr(args, k) for k in ("lr", "batch_size", "buffer_size", "target_update", "epsilon",
                                                        "update_ratio", "curriculum_margin")},
                "resumed": os.path.isfile(resume_path)}      # True이면 train_wall_s는 이어한 구간만
    tr_store, curve, val_store = MetricsStore(), TrainingCurveStore(), MetricsStore()
    validate = None
    if args.select == "val":
        validate = make_validator(layouts, agent, lam, val_seeds(args.n_val), args.sim_time, arm, val_store,
                                  sim_kw=cond.sim_kw)
    wall = time.perf_counter()
    best = train(layouts, agent, tr_store, arm, lam=lam, seed=args.seed, total_episodes=args.episodes,
                 sim_time=args.sim_time, label=args.arm, curve_store=curve,
                 verbose_every=max(1, args.episodes // 20), validate=validate, val_every=args.val_every,
                 resume_path=resume_path, val_store=val_store, sim_kw=cond.sim_kw,
                 update_ratio=args.update_ratio, curriculum_margin=args.curriculum_margin)
    train_wall = time.perf_counter() - wall
    if agent.mu_table is not None:
        run_info["local_mu_table"] = {"lam_hr": [l * 3600 for l in agent.mu_table[0]], "mu_s": agent.mu_table[1]}
    run_info["alpha_final"] = agent.policy_net.alpha
    ep_walls = [p.wall_s for p in curve.records]
    run_info.update(train_wall_s=train_wall, train_episode_wall_s_mean=sum(ep_walls) / max(1, len(ep_walls)),
                    train_sim_s_total=sum(p.sim_s for p in curve.records),
                    train_n_updates=sum(p.n_updates for p in curve.records))
    print(f"학습 시간 {train_wall / 3600:.2f} h (에피소드당 {run_info['train_episode_wall_s_mean']:.1f} s, 검증 포함 총계)")
    tr_store.save_csv(os.path.join(train_dir, "train.csv"))
    curve.save_csv(os.path.join(train_dir, "curve.csv"))
    val_store.save_csv(os.path.join(train_dir, "val.csv"))
    meta = {"lam": lam, "num_robots": train_Ms[0] if len(train_Ms) == 1 else None, "train_robots": train_Ms,
            "num_workstations": args.ws, "robot_capacity": ROBOT_CAPACITY, "episodes": args.episodes,
            "train_seed": args.seed, "arm": args.arm, **arm.to_dict(), "traffic": cond.traffic, "motion": cond.motion,
            "scenario": scenario.to_dict(), "condition": cond.to_dict(), "tag": args.tag, "select": args.select,
            "selected_episode": args.episodes}
    agent.save(os.path.join(train_dir, "agent_last.pt"), meta)
    if best is not None:
        agent.policy_net.load_state_dict(best["policy_state"])
        agent.target_net.load_state_dict(best["target_state"])
        meta.update(selected_episode=best["episode"], val_score=best["score"])
        print(f"검증 선택: 에피소드 {best['episode']} (검증 평균 흐름시간 {best['score']:.1f} s)")
    agent.save(os.path.join(train_dir, "agent.pt"), meta)
    run_info.update(checkpoint=os.path.abspath(os.path.join(train_dir, "agent.pt")), select=args.select,
                    selected_episode=meta["selected_episode"])
    with open(os.path.join(train_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=1, ensure_ascii=False)
    # 이전 학습의 평가 결과는 지금 가중치와 맞지 않음. 이어하기 파일도 더 필요 없음
    for name in ("eval.csv", "resume.pt"):
        if os.path.isfile(os.path.join(train_dir, name)):
            os.remove(os.path.join(train_dir, name))


if __name__ == "__main__":
    main()
