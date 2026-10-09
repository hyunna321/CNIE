"""RL 에이전트 학습, RL·베이스라인 공통 평가, 부하(λ) 민감도 분석."""
import csv
import os
import random
from typing import List, Dict, Tuple, Optional

import numpy as np

from environment import WarehouseLayout
from agents import DQNAgent
from metrics import MetricsStore, TrainingPoint, TrainingCurveStore, print_comparison
from simulator import RLSimulator
from baselines import (
    FullBufferSimulator, PartialSimulator, RandomAssignSimulator,
    NearestRobotSimulator, MinInsertionSimulator,
)
from constant import (
    SIM_TIME, LAMBDA_DEFAULT, DEFAULT_SEED, NUM_ROBOTS, ROBOT_CAPACITY,
    TOTAL_EPISODES, VERBOSE_EVERY, CURRICULUM_LAM_MARGIN, N_EVAL,
    EVAL_SEED_BASE, PARTIAL_K, ROBOT_SPEED, LOAD_SWEEP_SEED_BASE,
    P_VALUE_THRESHOLD, POLICY_LABELS,
)


def train(layout:         WarehouseLayout,
          agent:          DQNAgent,
          store:          MetricsStore,
          lam:            float = LAMBDA_DEFAULT,
          seed:           int   = DEFAULT_SEED,
          total_episodes: int   = TOTAL_EPISODES,
          num_robots:     int   = NUM_ROBOTS,
          robot_capacity: int   = ROBOT_CAPACITY,
          reward_cfg:     Dict  = None,
          label:          str   = "RL",
          verbose_every:  int   = VERBOSE_EVERY,
          curve_store:    Optional["TrainingCurveStore"] = None) -> None:
    """RL 에이전트를 학습한다.

    앞쪽 절반 에피소드는 λ를 [lam × CURRICULUM_LAM_MARGIN, lam]에서 무작위로
    뽑는 커리큘럼으로, 나머지는 lam으로 학습한다. 에피소드마다 전이를
    버퍼에 넣고 train_step을 1회 수행한다.

    입력:
        layout: WarehouseLayout
        agent: 학습할 DQNAgent (in-place 갱신)
        store: 에피소드별 결과를 기록할 MetricsStore
        lam: 목표 주문 도착률 (orders/s)
        seed: 에피소드 시드/λ 샘플링 난수 시드
        total_episodes: 학습 에피소드 수
        num_robots: 로봇 수
        robot_capacity: 로봇 용량
        reward_cfg: compute_reward 가중치 덮어쓰기용 kwargs
        label: 결과 라벨
        verbose_every: 진행 로그 출력 주기 (에피소드)
        curve_store: 학습 곡선 기록용 TrainingCurveStore (선택)
    """
    reward_cfg = reward_cfg or {}
    lam_lo, lam_hi = lam * CURRICULUM_LAM_MARGIN, lam

    rl_sim = RLSimulator(layout, agent, lam=lam, seed=seed,
                         num_robots=num_robots, robot_capacity=robot_capacity,
                         reward_cfg=reward_cfg)
    rng = random.Random(seed)

    for ep in range(1, total_episodes + 1):
        ep_seed = rng.randint(0, 2**31 - 1)
        ep_lam  = (rng.uniform(lam_lo, lam_hi)
                   if ep <= total_episodes // 2 else lam)

        rl_sim.lam             = ep_lam
        rl_sim.order_mgr.lam   = ep_lam
        rl_sim.reset(ep_seed)
        transitions = rl_sim.run()

        for trans in transitions:
            agent.push(*trans)
        loss = agent.train_step()
        agent.decay_epsilon(total_episodes)

        # 누적 보상은 완료 주문 수에 비례해 커지므로, λ/K 설정 간 비교가
        # 가능하도록 완료 주문 1건당 평균으로 기록한다.
        n_done     = max(1, rl_sim.order_mgr.completed_count())
        avg_reward = rl_sim.total_reward / n_done

        result = rl_sim.build_result(label, ep, ep_lam,
                                     reward=avg_reward,
                                     epsilon=agent.epsilon)
        store.add(result)

        if curve_store is not None:
            curve_store.add(TrainingPoint(
                label=label,
                episode=ep,
                reward=avg_reward,
                loss=loss if loss is not None else float("nan"),
                epsilon=agent.epsilon,
                n_completed=result.n_completed,
                avg_flow_time=result.avg_flow_time,
            ))

        if ep % verbose_every == 0:
            loss_s = f"{loss:.4f}" if loss is not None else "warmup"
            print(f"  [Ep {ep:4d}] {label}  M={num_robots}  "
                  f"λ={int(ep_lam*3600):3d}/hr  "
                  f"completed={result.n_completed:5d}  "
                  f"flow={result.avg_flow_time:.0f}s  "
                  f"totes/trip={result.avg_totes_per_trip:.2f}  "
                  f"save/ins={result.saving_per_insertion:.1f}m  "
                  f"reward/order={avg_reward:.4f}  "
                  f"ε={agent.epsilon:.3f}  loss={loss_s}")


def evaluate_all(layout:         WarehouseLayout,
                 agent:          DQNAgent,
                 store:          MetricsStore,
                 lam:            float,
                 num_robots:     int = NUM_ROBOTS,
                 robot_capacity: int = ROBOT_CAPACITY,
                 eval_seeds:     List[int] = None,
                 n_eval:         int = N_EVAL,
                 partial_k:      int = PARTIAL_K) -> None:
    """같은 평가 시드로 RL(ε=0)과 BL1~BL5를 모두 실행해 결과를 기록한다.

    입력:
        layout: WarehouseLayout
        agent: 평가할 DQNAgent
        store: 결과를 기록할 MetricsStore
        lam: 주문 도착률 (orders/s)
        num_robots: 로봇 수
        robot_capacity: 로봇 용량
        eval_seeds: 평가 시드 리스트 (None이면 EVAL_SEED_BASE부터 n_eval개)
        n_eval: eval_seeds가 None일 때 사용할 시드 수
        partial_k: BL2 출발 기준 주문 수
    """
    if eval_seeds is None:
        eval_seeds = list(range(EVAL_SEED_BASE, EVAL_SEED_BASE + n_eval))

    for i, s in enumerate(eval_seeds):

        sim = RLSimulator(layout, agent, lam=lam, seed=s,
                          num_robots=num_robots, robot_capacity=robot_capacity,
                          eval_mode=True)
        sim.reset(s)
        sim.run()
        n_done     = max(1, sim.order_mgr.completed_count())
        avg_reward = sim.total_reward / n_done
        store.add(sim.build_result("RL", i, lam,
                                   reward=avg_reward,
                                   epsilon=0.0))

        sim = FullBufferSimulator(layout, lam=lam, seed=s, sim_time=SIM_TIME,
                                  num_robots=num_robots, robot_capacity=robot_capacity)
        sim.reset(s)
        sim.run()
        store.add(sim.build_result("BL1_FullBuffer", i, lam))

        sim = PartialSimulator(layout, lam=lam, seed=s, sim_time=SIM_TIME,
                               num_robots=num_robots, robot_capacity=robot_capacity,
                               partial_k=partial_k)
        sim.reset(s)
        sim.run()
        store.add(sim.build_result(f"BL2_Partial_k{partial_k}", i, lam))

        sim = RandomAssignSimulator(layout, lam=lam, seed=s, sim_time=SIM_TIME,
                                    num_robots=num_robots, robot_capacity=robot_capacity)
        sim.reset(s)
        sim.run()
        store.add(sim.build_result("BL3_Random", i, lam))

        sim = NearestRobotSimulator(layout, lam=lam, seed=s, sim_time=SIM_TIME,
                                    num_robots=num_robots, robot_capacity=robot_capacity)
        sim.reset(s)
        sim.run()
        store.add(sim.build_result("BL4_Nearest", i, lam))

        sim = MinInsertionSimulator(layout, lam=lam, seed=s, sim_time=SIM_TIME,
                                    num_robots=num_robots, robot_capacity=robot_capacity)
        sim.reset(s)
        sim.run()
        store.add(sim.build_result("BL5_MinInsertion", i, lam))


# 부하(λ) 민감도 분석
#
# 시스템이 포화되면 정책과 무관하게 처리량이 이동 capacity로 수렴하므로,
# 여러 λ에서 로봇 가동률과 RL 대비 베이스라인의 대응표본 t-검정 결과를 보고
# RL이 유의하게 나은 부하 구간을 찾는다.

def _robot_utilization(store: MetricsStore, label: str, lam: float,
                       num_robots: int) -> float:
    """로봇 가동률 = 평균 총 이동거리 / (num_robots · 속도 · SIM_TIME).

    입력:
        store: 결과 저장소
        label: 정책 라벨
        lam: 주문 도착률
        num_robots: 로봇 수
    출력:
        가동률 (0~1), 결과가 없으면 nan
    """
    recs = store.filter(label=label, lam=lam, num_robots=num_robots)
    if not recs:
        return float("nan")
    avg_dist = sum(r.total_distance for r in recs) / len(recs)
    max_dist = num_robots * ROBOT_SPEED * SIM_TIME
    return avg_dist / max_dist if max_dist > 0 else float("nan")


def _paired_ttest(a: List[float], b: List[float]) -> Tuple[float, float]:
    """대응표본 t-검정. scipy가 없으면 정규근사로 p-value를 계산한다.

    입력:
        a, b: 같은 시드 순서로 짝지어진 두 표본
    출력:
        (mean(a − b), p-value), 표본이 부족하면 (nan, nan)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return float("nan"), float("nan")
    diff      = a - b
    mean_diff = float(diff.mean())
    try:
        from scipy import stats
        _, p = stats.ttest_rel(a, b)
        return mean_diff, float(p)
    except ImportError:
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        if se == 0:
            return mean_diff, float("nan")
        t = mean_diff / se
        from math import erf, sqrt
        p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t) / sqrt(2.0))))
        return mean_diff, float(p)


def _print_advantage_summary(rows: List[Dict], baseline_labels: List[str]) -> None:
    """λ별로 RL이 대기시간에서 유의하게(p < 0.05) 앞선 베이스라인 수를 출력한다.

    입력:
        rows: analyze_load_sensitivity의 λ별 요약 행
        baseline_labels: 비교 대상 베이스라인 라벨
    """
    print(f"\n{'-'*78}")
    print("RL 우위 구간 요약 (p<0.05 이면서 RL 평균 대기시간이 더 짧은 경우 기준)")
    print(f"{'-'*78}")
    hdr = f"  {'λ(/hr)':>8}  {'util':>7}  {'RL이 유의하게 더 나은 베이스라인 수':>18}"
    print(hdr)
    for row in rows:
        wins = sum(
            1 for b in baseline_labels
            if row[f"wait_p_vs_{b}"] == row[f"wait_p_vs_{b}"]     # NaN 제외
            and row[f"wait_p_vs_{b}"] < P_VALUE_THRESHOLD
            and row[f"wait_diff_vs_{b}"] < 0
        )
        print(f"  {row['lam_per_hr']:>8d}  {row['utilization']:>6.1%}  "
              f"{wins:>10d} / {len(baseline_labels)}")


def analyze_load_sensitivity(
    layout:         WarehouseLayout,
    agent:          DQNAgent,
    num_robots:     int,
    robot_capacity: int,
    lam_values:     List[float],
    n_eval:         int = N_EVAL,
    seed_base:      int = LOAD_SWEEP_SEED_BASE,
    out_csv:        Optional[str] = None,
) -> MetricsStore:
    """학습된 에이전트를 여러 λ에서 평가해 RL이 베이스라인보다 나은 부하 구간을 찾는다.

    λ마다 로봇 가동률과, RL 대비 각 베이스라인의 대기시간/완료율 대응표본
    t-검정 결과를 계산한다.

    입력:
        layout: WarehouseLayout
        agent: 평가할 DQNAgent
        num_robots: 로봇 수
        robot_capacity: 로봇 용량
        lam_values: 평가할 λ 리스트 (orders/s)
        n_eval: λ당 평가 시드 수
        seed_base: 평가 시드 시작값
        out_csv: λ별 요약을 저장할 CSV 경로 (선택)
    출력:
        모든 λ의 평가 결과를 합친 MetricsStore
    """
    labels     = POLICY_LABELS
    combined   = MetricsStore()
    eval_seeds = list(range(seed_base, seed_base + n_eval))

    print(f"\n{'='*78}")
    print(f"Load sensitivity sweep  (M={num_robots}, K={layout.num_workstations})")
    print(f"{'='*78}")

    summary_rows: List[Dict] = []
    for lam in lam_values:
        store = MetricsStore()
        evaluate_all(layout, agent, store, lam=lam,
                     num_robots=num_robots, robot_capacity=robot_capacity,
                     eval_seeds=eval_seeds, n_eval=n_eval)
        for r in store.records:
            combined.add(r)

        util = _robot_utilization(store, "RL", lam, num_robots)
        print_comparison(store, labels, lam=lam, num_robots=num_robots,
                         last_n=n_eval,
                         title=f"M={num_robots}  λ={int(lam*3600)}/hr  "
                               f"robot util≈{util:.1%}")

        rl_recs = store.filter(label="RL", lam=lam, num_robots=num_robots)
        row: Dict = {"lam_per_hr": int(lam * 3600), "utilization": util}
        for other in labels[1:]:
            other_recs = store.filter(label=other, lam=lam, num_robots=num_robots)
            d_wait, p_wait = _paired_ttest(
                [r.avg_wait_time for r in rl_recs],
                [r.avg_wait_time for r in other_recs])
            d_comp, p_comp = _paired_ttest(
                [r.completion_rate for r in rl_recs],
                [r.completion_rate for r in other_recs])
            row[f"wait_diff_vs_{other}"] = d_wait   # RL − other, 음수면 RL 대기시간이 짧음
            row[f"wait_p_vs_{other}"]    = p_wait
            row[f"comp_diff_vs_{other}"] = d_comp   # RL − other, 양수면 RL 완료율이 높음
            row[f"comp_p_vs_{other}"]    = p_comp
        summary_rows.append(row)

    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            for row in summary_rows:
                w.writerow(row)
        print(f"\n부하 민감도 요약(t-test 포함) 저장 → {out_csv}")

    _print_advantage_summary(summary_rows, labels[1:])
    return combined
