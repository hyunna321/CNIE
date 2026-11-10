"""RL 학습 루프, 검증 시드 기반 체크포인트 선택, 보상 중심화 기준 μ(λ) / ρ(λ) 추정"""
import copy
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch

from agents import DQNAgent
from metrics import MetricsStore, TrainingPoint, TrainingCurveStore
from rewards import Arm, mu_filename
from rl_simulator import RLSimulator
from evaluation import evaluate_policy, train_episode_seed
from constant import (
    SIM_TIME, LAMBDA_DEFAULT, DEFAULT_SEED, TOTAL_EPISODES, VERBOSE_EVERY,
    CURRICULUM_LAM_MARGIN, UPDATE_RATIO, LOCAL_MU_POINTS, LOCAL_MU_SEEDS, LOCAL_MU_WORKERS, CKPT_EVERY,
)

MU_SEED           = 0         # μ(λ) 추정 에피소드 시드 구간 (학습 시드 0, 실제 학습 시드 1~와 겹치지 않음)
MU_EPISODE_OFFSET = 90_000    # μ(λ) 추정 에피소드 번호 (학습 에피소드 1..TOTAL과 겹치지 않음)


# 보상 중심화 기준 (local: μ(λ) = 결정당 평균 비용, order_increment: ρ(λ) = 시간 평균 N(t))

def _mu_point(job) -> tuple:
    """작업자 프로세스: λ 격자 점 하나의 에피소드별 결정당 평균 국소 비용과 항별 평균 (계수 곱 전)"""
    agent, layouts, l, i, n_seeds, sim_time, arm = job
    costs, terms = [], []
    for M, L in sorted(layouts.items()):
        sim = RLSimulator(L, agent, lam=l, seed=MU_SEED, sim_time=sim_time, num_robots=M,
                          robot_capacity=L.robot_capacity, eval_mode=True, arm=arm)
        for j in range(n_seeds):
            sim.reset(train_episode_seed(MU_SEED, MU_EPISODE_OFFSET + i * n_seeds + j))
            sim.run()
            costs.append(sim.reward.center_stat())
            terms.append(sim.last_cost_terms)
    return costs, terms


def estimate_local_mu(layouts: Dict[int, object], agent: DQNAgent, lam: float, sim_time: float, arm: Arm,
                      n_points: int = LOCAL_MU_POINTS, n_seeds: int = LOCAL_MU_SEEDS,
                      workers: int = LOCAL_MU_WORKERS, verbose: bool = True) -> dict:
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
    lams = [float(l) for l in np.linspace(lam * CURRICULUM_LAM_MARGIN, lam, n_points)] if n_points > 1 else [lam]
    jobs = [(agent, layouts, l, i, n_seeds, sim_time, arm) for i, l in enumerate(lams)]
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
           "sim_time": sim_time, "margin": CURRICULUM_LAM_MARGIN, "Ms": sorted(layouts), "terms": terms}
    if arm.local_w != (1.0, 1.0):
        out["local_w"] = list(arm.local_w)
    return out


def load_or_estimate_local_mu(cell_dir: str, layouts: Dict[int, object], agent: DQNAgent, lam: float,
                              sim_time: float, arm: Arm) -> tuple:
    """셀 폴더의 μ(λ) 표(rewards.mu_filename)를 읽거나 (조건이 같을 때) 새로 추정해 저장, agent.mu_table 형식으로 반환"""
    path = os.path.join(cell_dir, mu_filename(arm))
    want = {"n_seeds": LOCAL_MU_SEEDS, "n_points": LOCAL_MU_POINTS, "sim_time": sim_time,
            "margin": CURRICULUM_LAM_MARGIN, "Ms": sorted(layouts)}
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
    tab = estimate_local_mu(layouts, agent, lam, sim_time, arm)
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
          val_store: Optional[MetricsStore] = None, sim_kw: Optional[Dict] = None) -> Optional[dict]:
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
        arm: 보상 설정 (rewards.Arm)
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

    lam_lo, lam_hi = lam * CURRICULUM_LAM_MARGIN, lam
    sims = {M: RLSimulator(L, agent, lam=lam, seed=seed, sim_time=sim_time, num_robots=M,
                           robot_capacity=L.robot_capacity, arm=arm, **(sim_kw or {}))
            for M, L in layouts.items()}
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
        loss = None
        n_updates = max(1, int(len(transitions) / UPDATE_RATIO))
        q_sum = td_sum = 0.0
        n_done = 0
        for _ in range(n_updates):
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
