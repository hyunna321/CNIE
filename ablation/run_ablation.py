"""멀티 워크스테이션 복귀와 최근접 이웃(NNH) 경로 구성의 기여도를 보는 대응 어블레이션.

비교 조건
--------
RL-MultiStation-NNH
    기본 정책. 복귀 워크스테이션을 동적으로 고르고, NNH로 경로를 구성/재구성한다.
RL-NoNNH
    멀티 워크스테이션은 유지하되 배정 순서(FIFO)대로 방문한다. mid-trip 주문은
    남은 경로 끝에 붙이고 재정렬하지 않는다.
RL-SingleStation-NNH
    NNH는 유지하되 모든 트립이 가운데 워크스테이션으로 복귀한다.

모든 조건은 같은 학습 에피소드 시드, 커리큘럼, 평가 시드를 쓰며,
각각 새로 시드를 고정한 에이전트로 학습한다.
"""

import argparse
import csv
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from agents import DQNAgent, Robot
from environment import (
    WarehouseLayout,
    nearest_neighbor_route,
    route_total_distance,
)
from metrics import MetricsStore, print_comparison
from run import RLSimulator
from constant import (
    LAMBDA_DEFAULT,
    ROBOT_CAPACITY,
    SIM_TIME,
    NUM_ROBOTS,
    DEFAULT_SEED,
    TOTAL_EPISODES,
    N_EVAL,
    EVAL_SEED_BASE,
    CURRICULUM_LAM_MARGIN,
    ABLATION_OUT_DIR,
    PAIRED_SUMMARY_METRICS,
)


@dataclass(frozen=True)
class AblationSpec:
    """어블레이션 조건 1개.

    label: 결과 라벨
    use_nnh: NNH로 방문 순서를 정할지 여부 (False면 배정 순서 유지)
    fixed_workstation_id: 복귀 워크스테이션 고정 인덱스 (None이면 동적 선택)
    """
    label: str
    use_nnh: bool
    fixed_workstation_id: Optional[int] = None


class RoutingAblationRobot(Robot):
    """방문 순서 방식과 복귀 워크스테이션 선택 방식을 바꿀 수 있는 Robot."""

    def __init__(self, *args, use_nnh: bool,
                 fixed_workstation_id: Optional[int], **kwargs):
        """
        입력:
            *args, **kwargs: Robot 인자
            use_nnh: NNH 사용 여부
            fixed_workstation_id: 고정 복귀 워크스테이션 (None이면 동적 선택)
        """
        super().__init__(*args, **kwargs)
        self.use_nnh = use_nnh
        self.fixed_workstation_id = fixed_workstation_id

    def _route(self, start: Tuple[float, float], destinations: Sequence[Tuple[float, float]],
               layout: WarehouseLayout) -> List[Tuple[float, float]]:
        """방문 순서를 정한다.

        입력:
            start: 출발 좌표
            destinations: 방문할 좌표
            layout: WarehouseLayout
        출력:
            use_nnh면 NNH 순서, 아니면 입력 순서 그대로의 좌표 리스트
        """
        if self.use_nnh:
            return nearest_neighbor_route(start, list(destinations), layout.aisle_distance)
        return list(destinations)

    def _workstation_for(self, start: Tuple[float, float], route: Sequence[Tuple[float, float]],
                         layout: WarehouseLayout) -> int:
        """주어진 경로의 복귀 워크스테이션을 정한다.

        입력:
            start: 출발 좌표
            route: 방문 순서가 정해진 경로
            layout: WarehouseLayout
        출력:
            고정 워크스테이션 인덱스, 또는 총 이동거리가 최소인 워크스테이션 인덱스
        """
        if self.fixed_workstation_id is not None:
            return self.fixed_workstation_id
        return min(
            range(len(layout.workstations)),
            key=lambda wid: route_total_distance(
                start, list(route), layout.workstations[wid], layout.aisle_distance),
        )

    def assign_trip(self, destinations: List[Tuple[float, float]], layout: WarehouseLayout):
        """현재 위치에서 출발하는 새 트립을 배정하고 이동을 시작한다.

        입력:
            destinations: 방문할 좌표 리스트
            layout: WarehouseLayout
        """
        route = self._route(self.pos, destinations, layout)
        wid = self._workstation_for(self.pos, route, layout)
        self.workstation_id = wid
        self.workstation = layout.workstations[wid]
        self.route = route
        self.idle_slots = self.capacity - len(destinations)
        self.is_idle = False
        self._start_move()

    def insert_and_reorder(self, loc: Tuple[float, float], layout: WarehouseLayout):
        """이동 중인 트립에 주문을 추가하고 남은 경로와 복귀 워크스테이션을 다시 정한다.

        현재 목적지(current_dest)는 이미 이동 중이므로 바꾸지 않는다.

        입력:
            loc: 추가할 주문 위치
            layout: WarehouseLayout
        """
        start = self.current_dest if self.current_dest else self.pos
        route = self._route(start, list(self.route) + [loc], layout)
        wid = self._workstation_for(start, route, layout)
        self.workstation_id = wid
        self.workstation = layout.workstations[wid]
        self.route = route
        self.idle_slots = max(0, self.idle_slots - 1)

    def delta_distance_insert(self, loc: Tuple[float, float], layout: WarehouseLayout,
                              target_ws_coords=None, dynamic_ws: bool = False) -> float:
        """이 로봇이 실제로 수행할 삽입 방식 기준으로 늘어나는 이동거리를 계산한다.

        target_ws_coords, dynamic_ws는 DQNAgent.build_candidates와 시그니처를
        맞추기 위한 인자로, 여기서는 사용하지 않는다.

        입력:
            loc: 추가할 주문 위치
            layout: WarehouseLayout
        출력:
            삽입 후 남은 경로 거리 − 삽입 전 남은 경로 거리 (m)
        """
        start = self.current_dest if self.current_dest else self.pos
        before = route_total_distance(start, self.route, self.workstation,
                                      layout.aisle_distance)
        route = self._route(start, list(self.route) + [loc], layout)
        wid = self._workstation_for(start, route, layout)
        after = route_total_distance(start, route, layout.workstations[wid],
                                     layout.aisle_distance)
        return after - before


class AblationRLSimulator(RLSimulator):
    """로봇만 RoutingAblationRobot으로 바꾼 RLSimulator."""

    def __init__(self, *args, spec: AblationSpec, **kwargs):
        """
        입력:
            *args, **kwargs: RLSimulator 인자
            spec: 어블레이션 조건
        """
        # super().__init__ 안에서 _build_robots가 호출되므로 먼저 설정한다.
        self._ablation_spec = spec
        super().__init__(*args, **kwargs)

    def _build_robots(self) -> List[Robot]:
        """조건에 맞는 RoutingAblationRobot을 생성한다.

        단일 워크스테이션 조건이면 모든 로봇을 그 워크스테이션에서 출발시킨다.

        출력:
            로봇 리스트
        """
        fixed = self._ablation_spec.fixed_workstation_id
        robots: List[Robot] = []
        for rid in range(self.num_robots):
            wid = fixed if fixed is not None else rid % self.num_workstations
            robots.append(RoutingAblationRobot(
                rid, self.layout.workstations[wid], self.layout, self.robot_capacity,
                workstation_id=wid,
                use_nnh=self._ablation_spec.use_nnh,
                fixed_workstation_id=fixed,
            ))
        return robots


def make_simulator(spec: AblationSpec, layout: WarehouseLayout, agent: DQNAgent,
                   lam: float, seed: int, sim_time: float, num_robots: int,
                   robot_capacity: int, eval_mode: bool):
    """조건에 맞는 시뮬레이터를 만든다. 기본 조건이면 원래 RLSimulator를 쓴다.

    입력:
        spec: 어블레이션 조건
        layout: WarehouseLayout
        agent: DQNAgent
        lam: 주문 도착률 (orders/s)
        seed: 주문 생성 난수 시드
        sim_time: 시뮬레이션 시간 (s)
        num_robots: 로봇 수
        robot_capacity: 로봇 용량
        eval_mode: 평가 모드 여부
    출력:
        RLSimulator 또는 AblationRLSimulator
    """
    if spec.use_nnh and spec.fixed_workstation_id is None:
        return RLSimulator(layout, agent, lam=lam, seed=seed, sim_time=sim_time,
                           num_robots=num_robots, robot_capacity=robot_capacity,
                           eval_mode=eval_mode)
    return AblationRLSimulator(layout, agent, lam=lam, seed=seed, sim_time=sim_time,
                                num_robots=num_robots, robot_capacity=robot_capacity,
                                eval_mode=eval_mode, spec=spec)


def train_condition(spec: AblationSpec, layout: WarehouseLayout, agent: DQNAgent,
                    lam: float, seed: int, episodes: int, sim_time: float,
                    num_robots: int, robot_capacity: int) -> MetricsStore:
    """조건 1개를 학습한다. 지역 RNG를 써서 모든 조건이 같은 에피소드 시드를 받는다.

    입력:
        spec: 어블레이션 조건
        layout: WarehouseLayout
        agent: 학습할 DQNAgent (in-place 갱신)
        lam: 목표 주문 도착률 (orders/s)
        seed: 에피소드 시드/λ 샘플링 난수 시드
        episodes: 학습 에피소드 수
        sim_time: 시뮬레이션 시간 (s)
        num_robots: 로봇 수
        robot_capacity: 로봇 용량
    출력:
        에피소드별 학습 결과 MetricsStore
    """
    store = MetricsStore()
    sim = make_simulator(spec, layout, agent, lam, seed, sim_time, num_robots,
                         robot_capacity, eval_mode=False)
    rng = random.Random(seed)
    lam_lo = lam * CURRICULUM_LAM_MARGIN

    for episode in range(1, episodes + 1):
        episode_seed = rng.randint(0, 2**31 - 1)
        episode_lam = rng.uniform(lam_lo, lam) if episode <= episodes // 2 else lam
        sim.lam = episode_lam
        sim.order_mgr.lam = episode_lam
        sim.reset(episode_seed)
        for transition in sim.run():
            agent.push(*transition)
        agent.train_step()
        agent.decay_epsilon(episodes)
        completed = max(1, sim.order_mgr.completed_count())
        store.add(sim.build_result(spec.label, episode, episode_lam,
                                   reward=sim.total_reward / completed,
                                   epsilon=agent.epsilon))
    return store


def evaluate_condition(spec: AblationSpec, layout: WarehouseLayout, agent: DQNAgent,
                       lam: float, seeds: Sequence[int], sim_time: float,
                       num_robots: int, robot_capacity: int) -> MetricsStore:
    """조건 1개를 평가 시드마다 ε=0으로 실행한다.

    입력:
        spec: 어블레이션 조건
        layout: WarehouseLayout
        agent: 평가할 DQNAgent
        lam: 주문 도착률 (orders/s)
        seeds: 평가 시드 리스트
        sim_time: 시뮬레이션 시간 (s)
        num_robots: 로봇 수
        robot_capacity: 로봇 용량
    출력:
        시드별 평가 결과 MetricsStore
    """
    store = MetricsStore()
    for episode, seed in enumerate(seeds):
        sim = make_simulator(spec, layout, agent, lam, seed, sim_time, num_robots,
                             robot_capacity, eval_mode=True)
        sim.reset(seed)
        sim.run()
        completed = max(1, sim.order_mgr.completed_count())
        store.add(sim.build_result(spec.label, episode, lam,
                                   reward=sim.total_reward / completed, epsilon=0.0))
    return store


def write_paired_summary(store: MetricsStore, specs: Sequence[AblationSpec], lam: float,
                         num_robots: int, path: str) -> None:
    """시드별로 (기본 조건 − 어블레이션 조건) 지표 차이를 CSV로 저장한다.

    입력:
        store: 모든 조건의 평가 결과
        specs: 조건 리스트 (첫 번째가 기본 조건)
        lam: 주문 도착률
        num_robots: 로봇 수
        path: 저장 경로
    """
    full = specs[0].label
    by_label = {
        spec.label: store.filter(label=spec.label, lam=lam, num_robots=num_robots)
        for spec in specs
    }
    rows = []
    for i, ref in enumerate(by_label[full]):
        for spec in specs[1:]:
            other = by_label[spec.label][i]
            row = {"seed_index": i, "reference": full, "ablation": spec.label}
            for metric in PAIRED_SUMMARY_METRICS:
                ref_value = getattr(ref, metric)
                other_value = getattr(other, metric)
                row[f"{metric}_full_minus_ablation"] = ref_value - other_value
            rows.append(row)
    if not rows:
        return
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """CLI 인자로 세 조건을 학습/평가하고 결과 CSV와 비교표를 출력한다."""
    parser = argparse.ArgumentParser(description="Paired routing ablation runner")
    parser.add_argument("--episodes", type=int, default=TOTAL_EPISODES)
    parser.add_argument("--n-eval", type=int, default=N_EVAL)
    parser.add_argument("--lam", type=float, default=LAMBDA_DEFAULT,
                        help="Arrival rate in orders/second (e.g. 850/3600).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-robots", type=int, default=NUM_ROBOTS)
    parser.add_argument("--robot-capacity", type=int, default=ROBOT_CAPACITY)
    parser.add_argument("--sim-time", type=float, default=SIM_TIME)
    parser.add_argument("--out-dir", default=ABLATION_OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    layout = WarehouseLayout(seed=args.seed, num_robots=args.num_robots,
                             robot_capacity=args.robot_capacity)
    centre_station = (layout.num_workstations - 1) // 2
    specs = [
        AblationSpec("RL-MultiStation-NNH", use_nnh=True),
        AblationSpec("RL-NoNNH", use_nnh=False),
        AblationSpec("RL-SingleStation-NNH", use_nnh=True,
                      fixed_workstation_id=centre_station),
    ]
    eval_seeds = list(range(EVAL_SEED_BASE, EVAL_SEED_BASE + args.n_eval))
    all_eval = MetricsStore()

    print("Paired ablation: every arm uses the same training/evaluation seeds.")
    print(f"load={args.lam * 3600:.0f}/hr, episodes={args.episodes}, eval={args.n_eval}")
    for spec in specs:
        print(f"\nTraining {spec.label} ...")
        # 네트워크 초기화(torch)와 PER 샘플링(np.random)이 전역 RNG를 쓰므로,
        # 모든 조건이 같은 초기 상태에서 시작하도록 매번 다시 고정한다.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        agent = DQNAgent(num_robots=args.num_robots,
                         robot_capacity=args.robot_capacity, seed=args.seed,
                         sim_time=args.sim_time,
                         num_workstations=layout.num_workstations)
        train_store = train_condition(spec, layout, agent, args.lam, args.seed,
                                      args.episodes, args.sim_time, args.num_robots,
                                      args.robot_capacity)
        train_store.save_csv(os.path.join(args.out_dir, f"train_{spec.label}.csv"))
        eval_store = evaluate_condition(spec, layout, agent, args.lam, eval_seeds,
                                        args.sim_time, args.num_robots,
                                        args.robot_capacity)
        for record in eval_store.records:
            all_eval.add(record)

    all_eval.save_csv(os.path.join(args.out_dir, "evaluation_all.csv"))
    write_paired_summary(all_eval, specs, args.lam, args.num_robots,
                         os.path.join(args.out_dir, "paired_full_minus_ablation.csv"))
    print_comparison(all_eval, [s.label for s in specs], lam=args.lam,
                     num_robots=args.num_robots, last_n=args.n_eval,
                     title="Routing ablation")
    print(f"\nSaved results to: {args.out_dir}")


if __name__ == "__main__":
    main()
