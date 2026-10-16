"""RL 에이전트 학습과 평가 (ε = 0)"""
import random
from typing import List, Optional

from agents import DQNAgent
from metrics import MetricsStore, TrainingPoint, TrainingCurveStore
from rl_simulator import RLSimulator
from constant import (
    SIM_TIME, LAMBDA_DEFAULT, DEFAULT_SEED, TOTAL_EPISODES, VERBOSE_EVERY,
    CURRICULUM_LAM_MARGIN, UPDATE_RATIO,
)


def train(layout, agent: DQNAgent, store: MetricsStore,
          lam: float = LAMBDA_DEFAULT, seed: int = DEFAULT_SEED,
          total_episodes: int = TOTAL_EPISODES, sim_time: float = SIM_TIME,
          label: str = "RL", verbose_every: int = VERBOSE_EVERY,
          curve_store: Optional[TrainingCurveStore] = None,
          update_ratio: Optional[float] = UPDATE_RATIO,
          reward_mode: str = "smdp_w") -> None:
    """RL 에이전트 학습

    - 앞쪽 절반 에피소드: λ를 [lam × CURRICULUM_LAM_MARGIN, lam]에서 무작위로 뽑는 커리큘럼
    - 나머지: lam으로 학습
    - 에피소드마다 전이를 버퍼에 넣고 (전이 수 / update_ratio)회 train_step (None이면 1회)

    입력:
        layout: GridWarehouseLayout (num_robots, robot_capacity 사용)
        agent: 학습할 DQNAgent (in-place 갱신)
        store: 에피소드별 결과를 기록할 MetricsStore
        lam: 목표 주문 도착률 (orders/s)
        seed: 에피소드 시드/λ 샘플링 난수 시드
        total_episodes: 학습 에피소드 수
        sim_time: 에피소드 길이 (s)
        label: 결과 라벨
        verbose_every: 진행 로그 출력 주기 (에피소드)
        curve_store: 학습 곡선 기록용 TrainingCurveStore (선택)
        update_ratio: 결정 N개당 업데이트 1회 (None이면 에피소드당 1회)
        reward_mode: "legacy", "smdp", "smdp_w" (rl_simulator 참고)
    """
    lam_lo, lam_hi = lam * CURRICULUM_LAM_MARGIN, lam
    sim = RLSimulator(layout, agent, lam=lam, seed=seed, sim_time=sim_time,
                      num_robots=layout.num_robots, robot_capacity=layout.robot_capacity,
                      reward_mode=reward_mode)
    rng = random.Random(seed)

    for ep in range(1, total_episodes + 1):
        ep_seed = rng.randint(0, 2**31 - 1)
        ep_lam  = rng.uniform(lam_lo, lam_hi) if ep <= total_episodes // 2 else lam

        sim.lam = sim.order_mgr.lam = ep_lam
        sim.reset(ep_seed)
        transitions = sim.run()

        for trans in transitions:
            agent.push(*trans)
        loss = None
        n_updates = 1 if update_ratio is None else max(1, int(len(transitions) / update_ratio))
        for _ in range(n_updates):
            step_loss = agent.train_step()
            if step_loss is not None:
                loss = step_loss
        agent.decay_epsilon(total_episodes)

        result = sim.build_result(label, ep, ep_lam)
        store.add(result)
        if curve_store is not None:
            curve_store.add(TrainingPoint(
                label=label, episode=ep, reward=result.reward,
                loss=loss if loss is not None else float("nan"),
                epsilon=agent.epsilon, n_completed=result.n_completed,
                avg_flow_time=result.avg_flow_time))

        if ep % verbose_every == 0:
            loss_s = f"{loss:.4f}" if loss is not None else "warmup"
            print(f"  [Ep {ep:4d}] {label}  λ={ep_lam * 3600:4.0f}/hr  "
                  f"완료={result.n_completed:5d}  flow={result.avg_flow_time:6.1f}s  "
                  f"totes/trip={result.avg_totes_per_trip:.2f}  wait={result.n_wait_action}  "
                  f"reward/order={result.reward:.4f}  ε={agent.epsilon:.3f}  loss={loss_s}")


def evaluate(layout, agent: DQNAgent, store: MetricsStore, lam: float,
             seeds: List[int], sim_time: float = SIM_TIME, label: str = "RL",
             reward_mode: str = "smdp_w") -> None:
    """학습된 에이전트를 ε = 0으로 평가 시드마다 한 번씩 실행해 기록

    입력:
        layout: GridWarehouseLayout
        agent: 평가할 DQNAgent
        store: 결과를 기록할 MetricsStore
        lam: 주문 도착률 (orders/s)
        seeds: 평가 시드 리스트 (episode 번호 = 리스트 인덱스, run.evaluate와 같은 대응)
        sim_time: 시뮬레이션 시간 (s)
        label: 결과 라벨
        reward_mode: 보상 기록 방식 (smdp_w면 wait 행동 허용, 정책 선택에 영향)
    """
    for i, s in enumerate(seeds):
        sim = RLSimulator(layout, agent, lam=lam, seed=s, sim_time=sim_time,
                          num_robots=layout.num_robots, robot_capacity=layout.robot_capacity,
                          eval_mode=True, reward_mode=reward_mode)
        sim.reset(s)
        sim.run()
        r = sim.build_result(label, i, lam)
        store.add(r)
        print(f"  {label:<18} seed {s}: flow {r.avg_flow_time:7.1f} s, "
              f"완료 {r.n_completed}/{r.n_arrived}, 거리/주문 {r.dist_per_order:5.1f} m, wait {r.n_wait_action}")
