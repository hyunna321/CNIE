import csv
import random
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from environment import WarehouseLayout, nearest_neighbor_route, nearest_workstation
from agents import Robot, DQNAgent
from manager import (
    OrderManager, EventQueue, Event, Order,
    EVENT_ORDER_ARRIVAL, EVENT_NODE_ARRIVAL, EVENT_TRIP_COMPLETE,
)
from metrics import MetricsStore, EpisodeResult, print_comparison
from constant import (
    SIM_TIME, LAMBDA_DEFAULT, LAMBDA_GRID, ROBOT_SPEED, DEFAULT_SEED,
    NUM_ROBOTS, ROBOT_CAPACITY, SOLO_RT_M,
    REWARD_D_MAX, REWARD_GAMMA_Q, REWARD_ALPHA_MID, REWARD_ALPHA_IDLE,
    REWARD_BETA_DEPLETION,
    TOTAL_EPISODES, VERBOSE_EVERY, CURRICULUM_LAM_MARGIN, N_EVAL,
    EVAL_SEED_BASE, LOAD_SWEEP_SEED_BASE, PARTIAL_K, P_VALUE_THRESHOLD,
    OUT_DIR, POLICY_LABELS,
)


@dataclass
class TrainingPoint:
    """학습 곡선의 에피소드 1개 지점."""
    label:         str
    episode:       int
    reward:        float
    loss:          float
    epsilon:       float
    n_completed:   int
    avg_flow_time: float


class TrainingCurveStore:
    """학습 곡선 지점(TrainingPoint)을 모아 CSV로 저장한다."""

    def __init__(self):
        self.records: List[TrainingPoint] = []

    def add(self, p: TrainingPoint):
        """지점 1개를 추가한다.

        입력:
            p: 추가할 TrainingPoint
        """
        self.records.append(p)

    def save_csv(self, path: str):
        """전체 지점을 CSV로 저장한다.

        입력:
            path: 저장 경로 (상위 디렉터리는 자동 생성)
        """
        if not self.records:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fields = ["label", "episode", "reward", "loss", "epsilon",
                  "n_completed", "avg_flow_time"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in self.records:
                w.writerow({k: getattr(r, k) for k in fields})


# 보상 함수

def compute_reward(
    robot:          Robot,
    order_loc:      Tuple[float, float],
    layout:         WarehouseLayout,
    delta_d:        float,
    robots:         Optional[List[Robot]] = None,
    n_wait:         int   = 0,
    num_robots:     int   = NUM_ROBOTS,
    robot_capacity: int   = ROBOT_CAPACITY,
    gamma_q:        float = REWARD_GAMMA_Q,
    alpha_mid:      float = REWARD_ALPHA_MID,
    alpha_idle:     float = REWARD_ALPHA_IDLE,
    beta_depletion: float = REWARD_BETA_DEPLETION,
) -> float:
    """주문 배정 1건의 보상을 계산한다.

    보상 = exp(-alpha · detour / D_MAX)
           − gamma_q · 큐 압박
           − beta_depletion (idle 로봇을 썼는데 빈 idle 로봇이 더 남지 않은 경우)

    입력:
        robot: 주문을 배정받은 로봇 (배정 직전 상태)
        order_loc: 주문 위치
        layout: WarehouseLayout
        delta_d: 배정으로 늘어난 이동거리 (m)
        robots: 후보 로봇 리스트 (소진 페널티 판단용)
        n_wait: 현재 대기 큐 길이
        num_robots: 큐 압박 정규화용 로봇 수
        robot_capacity: 로봇 용량
        gamma_q: 큐 압박 페널티 가중치
        alpha_mid: mid-trip 삽입 시 detour 감쇠 계수
        alpha_idle: idle 로봇 배정 시 detour 감쇠 계수
        beta_depletion: idle 로봇 소진 페널티
    출력:
        보상값
    """
    alpha   = alpha_idle if robot.is_idle else alpha_mid

    reward  = float(np.exp(-alpha * delta_d / REWARD_D_MAX))

    if gamma_q > 0.0:
        queue_pressure = min(n_wait / max(1, num_robots * robot_capacity), 1.0)
        reward        -= gamma_q * queue_pressure

    if beta_depletion > 0.0 and robot.is_idle and robots is not None:
        free_idle_left = sum(
            1 for r in robots
            if r.id != robot.id and r.is_idle and r.idle_slots == robot_capacity
        )
        if free_idle_left == 0:
            reward -= beta_depletion

    return reward


# 시뮬레이터

class SimulatorBase:
    """이산 사건 기반 창고 시뮬레이터의 공통 부분.

    - 로봇 풀은 모든 워크스테이션이 공유하며, 초기 위치만 라운드로빈으로 분산한다.
    - 대기 주문은 단일 FIFO 큐(OrderManager.wait_queue)로 관리한다.
    - 트립 시작/주문 삽입 때마다 총 이동거리가 최소가 되는 복귀
      워크스테이션을 다시 정한다 (environment.best_workstation_route).

    하위 클래스는 reset()과 run()을 구현하고, 주문 도착 시 배정 정책을 정의한다.
    """

    def __init__(self, layout: WarehouseLayout, lam: float, seed: int,
                 sim_time: float, num_robots: int, robot_capacity: int):
        """
        입력:
            layout: WarehouseLayout
            lam: 주문 도착률 (orders/s)
            seed: 주문 생성 난수 시드
            sim_time: 시뮬레이션 시간 (s)
            num_robots: 로봇 수
            robot_capacity: 로봇 용량
        """
        self.layout           = layout
        self.lam              = lam
        self.seed             = seed
        self.sim_time         = sim_time
        self.num_robots       = num_robots
        self.robot_capacity   = robot_capacity
        self.num_workstations = layout.num_workstations

        self.robots    = self._build_robots()
        self.order_mgr = OrderManager(layout, lam=lam, seed=seed)
        self.eq        = EventQueue()
        self.now       = 0.0

        self._robot_orders:          Dict[int, List[Order]] = {i: [] for i in range(num_robots)}
        self._robot_trip_distance:   Dict[int, float]       = {i: 0.0 for i in range(num_robots)}
        self._total_sim_distance:    float                  = 0.0
        self._total_trips:           int                    = 0
        self._mid_trip_count:        int                    = 0
        self._extra_travel:          float                  = 0.0

    def _build_robots(self) -> List[Robot]:
        """로봇을 생성하고 워크스테이션에 라운드로빈으로 초기 배치한다.

        출력:
            Robot 리스트
        """
        K = self.num_workstations
        robots = []
        for rid in range(self.num_robots):
            k = rid % K
            robots.append(Robot(rid, self.layout.workstations[k],
                                self.layout, self.robot_capacity,
                                workstation_id=k))
        return robots

    def _reset_common(self, seed: int):
        """에피소드 공통 상태와 로봇을 초기화하고 첫 주문 도착 사건을 넣는다.

        입력:
            seed: 주문 생성 난수 시드
        """
        self.now = 0.0
        self.order_mgr.reset(seed)
        self.eq  = EventQueue()
        self._robot_orders          = {i: [] for i in range(self.num_robots)}
        self._robot_trip_distance   = {i: 0.0 for i in range(self.num_robots)}
        self._total_sim_distance    = 0.0
        self._total_trips           = 0
        self._mid_trip_count        = 0
        self._extra_travel          = 0.0
        for r in self.robots:
            r.reset()
        t0 = self.order_mgr.generate_next_arrival(0.0)
        self.eq.push(Event(t0, EVENT_ORDER_ARRIVAL))

    def _record_service_start(self, order: Order, t: float):
        """주문의 로봇 배정 시각을 기록한다.

        입력:
            order: 대상 주문
            t: 배정 시각 (s)
        """
        self.order_mgr.record_service_start(order, t)

    def _handle_node_arrival(self, ev: Event):
        """로봇이 경유지에 도착했을 때 다음 이동 또는 트립 종료를 처리한다.

        남은 경유지가 있으면 다음 지점으로, 없으면 복귀 워크스테이션으로 보내고,
        이미 워크스테이션에 도착했으면 트립 완료 사건을 넣는다.

        입력:
            ev: EVENT_NODE_ARRIVAL 사건 (data["robot_id"])
        """
        robot = self.robots[ev.data["robot_id"]]
        robot.end_travel(self.now)
        still_going = robot.arrive_at_node()

        if still_going:
            self._schedule_next(robot)
            return

        if robot.pos == robot.workstation:
            self.eq.push(Event(self.now, EVENT_TRIP_COMPLETE, {"robot_id": robot.id}))
        else:
            robot.current_dest = robot.workstation
            self._schedule_next(robot)

    def _complete_trip(self, robot: Robot):
        """트립을 마친 로봇을 idle로 되돌리고 실었던 주문을 완료 처리한다.

        입력:
            robot: 트립을 마친 로봇
        """
        robot.pos          = robot.workstation
        robot.is_idle      = True
        robot.idle_slots   = self.robot_capacity
        robot.route        = []
        robot.current_dest = None
        robot.progress     = 0.0
        completed = list(self._robot_orders.pop(robot.id, []))
        for o in completed:
            self.order_mgr.record_completion(o, self.now)
        self._total_trips            += 1
        self._robot_orders[robot.id]  = []

    def _schedule_next(self, robot: Robot):
        """현재 목적지까지 이동을 시작하고 도착 사건을 예약한다. 이동거리도 누적한다.

        입력:
            robot: 이동할 로봇
        """
        if robot.current_dest is None:
            return
        seg = max(0.0, self.layout.aisle_distance(robot.pos, robot.current_dest) - robot.progress)
        self._robot_trip_distance[robot.id] += seg
        self._total_sim_distance            += seg
        t = self.now + seg / ROBOT_SPEED
        robot.start_travel(self.now)
        self.eq.push(Event(t, EVENT_NODE_ARRIVAL,
                           {"robot_id": robot.id, "dest": robot.current_dest}))

    def _dispatch(self, robot: Robot, orders: List[Order]):
        """대기 큐에서 꺼낸 주문들로 로봇의 새 트립을 시작한다.

        입력:
            robot: idle 로봇
            orders: 배정할 주문 리스트
        """
        robot.assign_trip([o.loc for o in orders], self.layout)
        self._robot_orders[robot.id] = list(orders)
        for o in orders:
            self._record_service_start(o, self.now)
        self._schedule_next(robot)

    def distance_per_completed_order(self) -> float:
        """완료 주문 1건당 총 이동거리 (m). 완료 주문이 없으면 0.0."""
        n = self.order_mgr.completed_count()
        return self._total_sim_distance / n if n > 0 else 0.0

    def avg_totes_per_trip(self) -> float:
        """트립당 평균 완료 주문 수."""
        return self.order_mgr.completed_count() / max(1, self._total_trips)

    def saving_per_insertion(self) -> float:
        """mid-trip 삽입 1건당 절감 이동거리 (m) = 단독 왕복거리 − 삽입당 추가 이동거리.

        출력:
            절감 이동거리, 삽입이 없으면 0.0
        """
        if self._mid_trip_count == 0:
            return 0.0
        extra_per = self._extra_travel / self._mid_trip_count
        return SOLO_RT_M - extra_per

    def build_result(self, label: str, episode: int, lam: float,
                     reward: float = 0.0, epsilon: float = 0.0) -> EpisodeResult:
        """현재 에피소드 지표를 EpisodeResult로 묶는다.

        입력:
            label: 정책 라벨
            episode: 에피소드 번호
            lam: 이번 에피소드의 주문 도착률
            reward: 기록할 보상값
            epsilon: 기록할 ε
        출력:
            EpisodeResult
        """
        return EpisodeResult(
            label                = label,
            episode              = episode,
            lam                  = lam,
            num_robots           = self.num_robots,
            n_completed          = self.order_mgr.completed_count(),
            n_arrived            = len(self.order_mgr.all_orders),
            avg_flow_time        = self.order_mgr.avg_flow_time(),
            avg_wait_time        = self.order_mgr.avg_wait_time(),
            dist_per_order       = self.distance_per_completed_order(),
            total_distance       = self._total_sim_distance,
            avg_totes_per_trip   = self.avg_totes_per_trip(),
            mid_trip_rate        = self._mid_trip_count / max(1, self.order_mgr.completed_count()),
            extra_travel_m       = self._extra_travel,
            saving_per_insertion = self.saving_per_insertion(),
            reward               = reward,
            epsilon              = epsilon,
        )


class RLSimulator(SimulatorBase):
    """DQNAgent가 주문마다 배정할 로봇을 고르는 시뮬레이터.

    후보는 빈 슬롯이 있는 모든 로봇(idle + mid-trip)이고, 행동은 후보 리스트
    내 인덱스다. 한 결정의 전이는 다음 결정 시점의 후보 특징이 나와야
    완성되므로 직전 결정을 _pending에 보관해 둔다.
    """

    def __init__(self, layout: WarehouseLayout, agent: DQNAgent,
                 lam: float, seed: int, sim_time: float = SIM_TIME,
                 num_robots: int = NUM_ROBOTS, robot_capacity: int = ROBOT_CAPACITY,
                 reward_cfg: Optional[Dict] = None,
                 eval_mode: bool = False):
        """
        입력:
            layout: WarehouseLayout
            agent: 배정 정책 DQNAgent
            lam: 주문 도착률 (orders/s)
            seed: 주문 생성 난수 시드
            sim_time: 시뮬레이션 시간 (s)
            num_robots: 로봇 수
            robot_capacity: 로봇 용량
            reward_cfg: compute_reward 가중치 덮어쓰기용 kwargs
            eval_mode: True이면 ε=0으로 실행하고 전이를 수집하지 않음
        """
        super().__init__(layout, lam, seed, sim_time, num_robots, robot_capacity)
        self.agent        = agent
        self.reward_cfg   = reward_cfg or {}
        self.total_reward = 0.0
        self.eval_mode    = eval_mode
        self._pending: Optional[Tuple[np.ndarray, float]] = None

    def reset(self, seed: int):
        """에피소드 상태, 누적 보상, 보류 전이를 초기화한다.

        입력:
            seed: 주문 생성 난수 시드
        """
        self._reset_common(seed)
        self.total_reward = 0.0
        self._pending = None

    def run(self) -> List[tuple]:
        """에피소드 1회를 끝까지 실행한다.

        출력:
            학습용 전이 리스트 [(feat, reward, next_feats, done), ...]
            (eval_mode이면 빈 리스트)
        """
        transitions = []
        saved_epsilon = self.agent.epsilon
        if self.eval_mode:
            self.agent.epsilon = 0.0

        while self.eq:
            ev = self.eq.pop()
            if ev.time > self.sim_time:
                break
            self.now = ev.time
            if ev.etype == EVENT_ORDER_ARRIVAL:
                t = self._handle_order_arrival(ev)
                if t:
                    transitions.append(t)
                nxt = self.order_mgr.generate_next_arrival(self.now)
                if nxt <= self.sim_time:
                    self.eq.push(Event(nxt, EVENT_ORDER_ARRIVAL))
            elif ev.etype == EVENT_NODE_ARRIVAL:
                self._handle_node_arrival(ev)
            elif ev.etype == EVENT_TRIP_COMPLETE:
                self._handle_trip_complete(ev)

        # 마지막 결정은 다음 상태가 없으므로 done=True로 마감한다.
        if not self.eval_mode and self._pending is not None:
            feat, r = self._pending
            transitions.append((feat, r, None, True))
            self._pending = None

        if self.eval_mode:
            self.agent.epsilon = saved_epsilon
        return transitions

    def _handle_order_arrival(self, ev: Event) -> Optional[tuple]:
        """새 주문을 에이전트가 고른 로봇에 배정하고 보상을 계산한다.

        후보가 없으면 주문을 대기 큐에 넣는다.

        입력:
            ev: EVENT_ORDER_ARRIVAL 사건
        출력:
            직전 결정의 완성된 전이 (feat, reward, next_feats, False),
            보류 중인 결정이 없거나 eval_mode이면 None
        """
        order = self.order_mgr.create_order(self.now)

        busy     = [r for r in self.robots if not r.is_idle and r.idle_slots > 0]
        idle_any = [r for r in self.robots if r.is_idle and r.idle_slots > 0]
        eligible = busy + idle_any

        if not eligible:
            self.order_mgr.enqueue(order, self.now)
            return None

        n_wait      = self.order_mgr.queue_size()
        ws_capacity = self.num_robots * self.robot_capacity
        cand_feats  = self.agent.build_candidates(
            order.loc, eligible, n_wait, layout=self.layout, now=self.now,
            ws_capacity=ws_capacity,
        )
        nearest_ws_id = nearest_workstation(order.loc, self.layout.workstations,
                                            self.layout.aisle_distance)
        target_coords = self.layout.workstations[nearest_ws_id]

        idx = self.agent.select_index(cand_feats)
        if idx < 0:
            idx = int(np.argmin([
                r.delta_distance_insert(
                    order.loc, self.layout,
                    target_ws_coords=(target_coords if r.is_idle else None),
                    dynamic_ws=(not r.is_idle))
                for r in eligible
            ]))

        robot        = eligible[idx]
        chosen_feat  = cand_feats[idx]
        was_idle     = robot.is_idle

        # 배정 전 상태 기준으로 detour를 측정한다 (보상 및 extra_travel 집계용).
        delta_d = robot.delta_distance_insert(
            order.loc, self.layout,
            target_ws_coords=(target_coords if was_idle else None),
            dynamic_ws=(not was_idle))
        self._extra_travel += delta_d

        reward = compute_reward(
            robot, order.loc, self.layout, delta_d,
            robots=eligible,
            n_wait=n_wait,
            num_robots=len(eligible),
            robot_capacity=self.robot_capacity,
            **self.reward_cfg,
        )
        self.total_reward += reward

        if was_idle:
            robot.assign_trip([order.loc], self.layout)
        else:
            robot.insert_and_reorder(order.loc, self.layout)
            self._mid_trip_count += 1

        self._robot_orders[robot.id].append(order)
        self._record_service_start(order, self.now)

        if was_idle:
            self._schedule_next(robot)

        if self.eval_mode:
            return None

        # 직전 결정을 지금의 후보 특징으로 완성하고, 이번 결정은 보류한다.
        completed = None
        if self._pending is not None:
            feat_prev, r_prev = self._pending
            completed = (feat_prev, r_prev, cand_feats, False)
        self._pending = (chosen_feat, reward)
        return completed

    def _handle_trip_complete(self, ev: Event):
        """트립 완료 처리 후, 대기 주문이 있으면 최대 용량만큼 꺼내 새 트립을 시작한다.

        입력:
            ev: EVENT_TRIP_COMPLETE 사건 (data["robot_id"])
        """
        robot  = self.robots[ev.data["robot_id"]]
        self._complete_trip(robot)
        orders = self.order_mgr.dequeue_up_to(self.robot_capacity)
        if orders:
            self._dispatch(robot, orders)


class BaselineSimulator(SimulatorBase):
    """BL1 Full-Buffer: idle 로봇의 슬롯이 모두 찬 뒤에 출발한다. mid-trip 삽입 없음.

    주문은 빈 슬롯이 가장 많은 idle 로봇에 쌓이고, 출발 시점에 모인 목적지
    전체 기준으로 복귀 워크스테이션이 정해진다.
    """

    def reset(self, seed: int):
        """에피소드 상태를 초기화한다.

        입력:
            seed: 주문 생성 난수 시드
        """
        self._reset_common(seed)

    def run(self):
        """에피소드 1회를 끝까지 실행한다."""
        while self.eq:
            ev = self.eq.pop()
            if ev.time > self.sim_time:
                break
            self.now = ev.time
            if ev.etype == EVENT_ORDER_ARRIVAL:
                self._handle_bl(ev)
                nxt = self.order_mgr.generate_next_arrival(self.now)
                if nxt <= self.sim_time:
                    self.eq.push(Event(nxt, EVENT_ORDER_ARRIVAL))
            elif ev.etype == EVENT_NODE_ARRIVAL:
                self._handle_node_arrival(ev)
            elif ev.etype == EVENT_TRIP_COMPLETE:
                robot = self.robots[ev.data["robot_id"]]
                self._complete_trip(robot)
                orders = self.order_mgr.dequeue_up_to(self.robot_capacity)
                if orders:
                    self._robot_orders[robot.id] = list(orders)
                    robot.idle_slots = self.robot_capacity - len(orders)
                    if robot.idle_slots == 0:
                        robot.assign_trip([o.loc for o in orders], self.layout)
                        self._schedule_next(robot)

    def _handle_bl(self, ev: Event):
        """새 주문을 빈 슬롯이 가장 많은 idle 로봇에 쌓고, 슬롯이 다 차면 출발시킨다.

        idle 로봇이 없으면 대기 큐에 넣는다.

        입력:
            ev: EVENT_ORDER_ARRIVAL 사건
        """
        order = self.order_mgr.create_order(self.now)
        idle  = [r for r in self.robots if r.is_idle and r.idle_slots > 0]
        if idle:
            robot = max(idle, key=lambda r: r.idle_slots)
            self._robot_orders[robot.id].append(order)
            self._record_service_start(order, self.now)
            robot.idle_slots -= 1
            if robot.idle_slots == 0:
                robot.assign_trip([o.loc for o in self._robot_orders[robot.id]],
                                  self.layout)
                self._schedule_next(robot)
        else:
            self.order_mgr.enqueue(order, self.now)


class PartialSimulator(SimulatorBase):
    """BL2 Partial-Buffer: idle 로봇에 주문이 partial_k개 쌓이면 바로 출발한다. mid-trip 삽입 없음."""

    def __init__(self, *args, partial_k: int = PARTIAL_K, **kwargs):
        """
        입력:
            *args, **kwargs: SimulatorBase 인자
            partial_k: 출발 기준 주문 수
        """
        super().__init__(*args, **kwargs)
        self.partial_k = partial_k

    def reset(self, seed: int):
        """에피소드 상태를 초기화한다.

        입력:
            seed: 주문 생성 난수 시드
        """
        self._reset_common(seed)

    def run(self):
        """에피소드 1회를 끝까지 실행한다."""
        while self.eq:
            ev = self.eq.pop()
            if ev.time > self.sim_time:
                break
            self.now = ev.time
            if ev.etype == EVENT_ORDER_ARRIVAL:
                self._handle_partial(ev)
                nxt = self.order_mgr.generate_next_arrival(self.now)
                if nxt <= self.sim_time:
                    self.eq.push(Event(nxt, EVENT_ORDER_ARRIVAL))
            elif ev.etype == EVENT_NODE_ARRIVAL:
                self._handle_node_arrival(ev)
            elif ev.etype == EVENT_TRIP_COMPLETE:
                robot = self.robots[ev.data["robot_id"]]
                self._complete_trip(robot)
                orders = self.order_mgr.dequeue_up_to(self.robot_capacity)
                if orders:
                    self._robot_orders[robot.id] = list(orders)
                    robot.idle_slots = self.robot_capacity - len(orders)
                    if len(orders) >= self.partial_k:
                        robot.assign_trip([o.loc for o in orders], self.layout)
                        self._schedule_next(robot)

    def _handle_partial(self, ev: Event):
        """새 주문을 빈 슬롯이 가장 많은 idle 로봇에 쌓고, partial_k개가 되면 출발시킨다.

        idle 로봇이 없으면 대기 큐에 넣는다.

        입력:
            ev: EVENT_ORDER_ARRIVAL 사건
        """
        order = self.order_mgr.create_order(self.now)
        idle  =[r for r in self.robots if r.is_idle and r.idle_slots > 0]
        if idle:
            robot = max(idle, key=lambda r: r.idle_slots)
            self._robot_orders[robot.id].append(order)
            self._record_service_start(order, self.now)
            robot.idle_slots -= 1
            if self.robot_capacity - robot.idle_slots >= self.partial_k:
                robot.assign_trip([o.loc for o in self._robot_orders[robot.id]],
                                  self.layout)
                self._schedule_next(robot)
        else:
            self.order_mgr.enqueue(order, self.now)


class RandomAssignSimulator(SimulatorBase):
    """BL3 Random: 빈 슬롯이 있는 로봇(idle + mid-trip) 중 무작위로 배정한다. mid-trip 삽입 허용."""

    def reset(self, seed: int):
        """에피소드 상태와 배정용 난수 생성기를 초기화한다.

        입력:
            seed: 주문 생성 및 배정 난수 시드
        """
        self._reset_common(seed)
        self._assign_rng = random.Random(seed)

    def run(self):
        """에피소드 1회를 끝까지 실행한다."""
        while self.eq:
            ev = self.eq.pop()
            if ev.time > self.sim_time:
                break
            self.now = ev.time
            if ev.etype == EVENT_ORDER_ARRIVAL:
                self._handle_random(ev)
                nxt = self.order_mgr.generate_next_arrival(self.now)
                if nxt <= self.sim_time:
                    self.eq.push(Event(nxt, EVENT_ORDER_ARRIVAL))
            elif ev.etype == EVENT_NODE_ARRIVAL:
                self._handle_node_arrival(ev)
            elif ev.etype == EVENT_TRIP_COMPLETE:
                robot = self.robots[ev.data["robot_id"]]
                self._complete_trip(robot)
                orders = self.order_mgr.dequeue_up_to(self.robot_capacity)
                if orders:
                    self._dispatch(robot, orders)

    def _handle_random(self, ev: Event):
        """새 주문을 후보 로봇 중 무작위 1대에 배정한다. 후보가 없으면 대기 큐에 넣는다.

        입력:
            ev: EVENT_ORDER_ARRIVAL 사건
        """
        order    =self.order_mgr.create_order(self.now)
        busy     = [r for r in self.robots if not r.is_idle and r.idle_slots > 0]
        idle_any = [r for r in self.robots if r.is_idle and r.idle_slots > 0]
        eligible = busy + idle_any
        if not eligible:
            self.order_mgr.enqueue(order, self.now)
            return
        robot = self._assign_rng.choice(eligible)

        nearest_ws_id = nearest_workstation(order.loc, self.layout.workstations,
                                            self.layout.aisle_distance)
        target_coords = self.layout.workstations[nearest_ws_id]
        delta_d = robot.delta_distance_insert(
            order.loc, self.layout,
            target_ws_coords=(target_coords if robot.is_idle else None),
            dynamic_ws=(not robot.is_idle))
        self._extra_travel += delta_d

        self._robot_orders[robot.id].append(order)
        self._record_service_start(order, self.now)
        if robot.is_idle:
            robot.assign_trip([order.loc], self.layout)
            self._schedule_next(robot)
        else:
            robot.insert_and_reorder(order.loc, self.layout)
            self._mid_trip_count += 1


class NearestRobotSimulator(SimulatorBase):
    """BL4 Nearest: 빈 슬롯이 있는 로봇 중 주문 위치까지 현재 거리가 가장 짧은 로봇에 배정한다. mid-trip 삽입 허용."""

    def reset(self, seed: int):
        """에피소드 상태를 초기화한다.

        입력:
            seed: 주문 생성 난수 시드
        """
        self._reset_common(seed)

    def run(self):
        """에피소드 1회를 끝까지 실행한다."""
        while self.eq:
            ev = self.eq.pop()
            if ev.time > self.sim_time:
                break
            self.now = ev.time
            if ev.etype == EVENT_ORDER_ARRIVAL:
                self._handle_nearest(ev)
                nxt = self.order_mgr.generate_next_arrival(self.now)
                if nxt <= self.sim_time:
                    self.eq.push(Event(nxt, EVENT_ORDER_ARRIVAL))
            elif ev.etype == EVENT_NODE_ARRIVAL:
                self._handle_node_arrival(ev)
            elif ev.etype == EVENT_TRIP_COMPLETE:
                robot = self.robots[ev.data["robot_id"]]
                self._complete_trip(robot)
                orders = self.order_mgr.dequeue_up_to(self.robot_capacity)
                if orders:
                    self._dispatch(robot, orders)

    def _handle_nearest(self, ev: Event):
        """새 주문을 현재 위치가 가장 가까운 후보 로봇에 배정한다. 후보가 없으면 대기 큐에 넣는다.

        입력:
            ev: EVENT_ORDER_ARRIVAL 사건
        """
        order    =self.order_mgr.create_order(self.now)
        busy     = [r for r in self.robots if not r.is_idle and r.idle_slots > 0]
        idle_any = [r for r in self.robots if r.is_idle and r.idle_slots > 0]
        eligible = busy + idle_any
        if not eligible:
            self.order_mgr.enqueue(order, self.now)
            return
        robot = min(eligible,
                    key=lambda r: self.layout.aisle_distance(r.pos, order.loc))

        nearest_ws_id = nearest_workstation(order.loc, self.layout.workstations,
                                            self.layout.aisle_distance)
        target_coords = self.layout.workstations[nearest_ws_id]
        delta_d = robot.delta_distance_insert(
            order.loc, self.layout,
            target_ws_coords=(target_coords if robot.is_idle else None),
            dynamic_ws=(not robot.is_idle))
        self._extra_travel += delta_d

        self._robot_orders[robot.id].append(order)
        self._record_service_start(order, self.now)
        if robot.is_idle:
            robot.assign_trip([order.loc], self.layout)
            self._schedule_next(robot)
        else:
            robot.insert_and_reorder(order.loc, self.layout)
            self._mid_trip_count += 1


class MinInsertionSimulator(SimulatorBase):
    """BL5 Min-Insertion: 빈 슬롯이 있는 로봇 중 삽입 detour가 최소인 로봇에 배정한다. mid-trip 삽입 허용.

    detour 계산 기준은 RLSimulator와 같다 (idle 후보는 주문에서 가장 가까운
    워크스테이션 복귀 가정, mid-trip 후보는 최적 복귀 워크스테이션 재탐색).
    """

    def reset(self, seed: int):
        """에피소드 상태를 초기화한다.

        입력:
            seed: 주문 생성 난수 시드
        """
        self._reset_common(seed)

    def run(self):
        """에피소드 1회를 끝까지 실행한다."""
        while self.eq:
            ev = self.eq.pop()
            if ev.time > self.sim_time:
                break
            self.now = ev.time
            if ev.etype == EVENT_ORDER_ARRIVAL:
                self._handle_min_insertion(ev)
                nxt = self.order_mgr.generate_next_arrival(self.now)
                if nxt <= self.sim_time:
                    self.eq.push(Event(nxt, EVENT_ORDER_ARRIVAL))
            elif ev.etype == EVENT_NODE_ARRIVAL:
                self._handle_node_arrival(ev)
            elif ev.etype == EVENT_TRIP_COMPLETE:
                robot = self.robots[ev.data["robot_id"]]
                self._complete_trip(robot)
                orders = self.order_mgr.dequeue_up_to(self.robot_capacity)
                if orders:
                    self._dispatch(robot, orders)

    def _handle_min_insertion(self, ev: Event):
        """새 주문을 삽입 detour가 최소인 후보 로봇에 배정한다. 후보가 없으면 대기 큐에 넣는다.

        입력:
            ev: EVENT_ORDER_ARRIVAL 사건
        """
        order    =self.order_mgr.create_order(self.now)
        busy     = [r for r in self.robots if not r.is_idle and r.idle_slots > 0]
        idle_any = [r for r in self.robots if r.is_idle and r.idle_slots > 0]
        eligible = busy + idle_any
        if not eligible:
            self.order_mgr.enqueue(order, self.now)
            return

        nearest_ws_id = nearest_workstation(order.loc, self.layout.workstations,
                                            self.layout.aisle_distance)
        target_coords = self.layout.workstations[nearest_ws_id]

        def _delta(r: Robot) -> float:
            """로봇 r에 이 주문을 넣을 때 늘어나는 이동거리 (m)."""
            return r.delta_distance_insert(
                order.loc, self.layout,
                target_ws_coords=(target_coords if r.is_idle else None),
                dynamic_ws=(not r.is_idle))

        robot   = min(eligible, key=_delta)
        delta_d = _delta(robot)
        self._extra_travel += delta_d

        self._robot_orders[robot.id].append(order)
        self._record_service_start(order, self.now)
        if robot.is_idle:
            robot.assign_trip([order.loc], self.layout)
            self._schedule_next(robot)
        else:
            robot.insert_and_reorder(order.loc, self.layout)
            self._mid_trip_count += 1


# 학습 / 평가

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
        ep_lam  = (random.uniform(lam_lo, lam_hi)
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

        sim = BaselineSimulator(layout, lam=lam, seed=s, sim_time=SIM_TIME,
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


# λ 그리드 실험

def experiment_lambda_grid(
    layout:         WarehouseLayout,
    store:          MetricsStore,
    lam_grid:       List[float] = LAMBDA_GRID,
    seed:           int = DEFAULT_SEED,
    total_episodes: int = TOTAL_EPISODES,
    n_eval:         int = N_EVAL,
    num_robots:     int = NUM_ROBOTS,
    robot_capacity: int = ROBOT_CAPACITY,
    out_dir:        str = OUT_DIR,
    curve_store:    Optional[TrainingCurveStore] = None,
) -> Dict[float, DQNAgent]:
    """λ마다 새 에이전트를 학습하고 RL/베이스라인을 평가해 CSV로 저장한다.

    저장 파일 (out_dir):
        train_lam{λ}_M{M}.csv, eval_lam{λ}_M{M}_n{n_eval}.csv

    입력:
        layout: WarehouseLayout
        store: 모든 λ의 평가 결과를 누적할 MetricsStore
        lam_grid: λ 리스트 (orders/s)
        seed: 에이전트/학습 시드
        total_episodes: λ당 학습 에피소드 수
        n_eval: λ당 평가 시드 수
        num_robots: 로봇 수
        robot_capacity: 로봇 용량
        out_dir: CSV 저장 디렉터리
        curve_store: 학습 곡선 기록용 TrainingCurveStore (선택)
    출력:
        {λ: 학습된 DQNAgent}
    """
    labels     = POLICY_LABELS
    eval_seeds = list(range(EVAL_SEED_BASE, EVAL_SEED_BASE + n_eval))
    agents: Dict[float, DQNAgent] = {}

    os.makedirs(out_dir, exist_ok=True)

    for lam in lam_grid:
        lam_hr = int(round(lam * 3600))
        print(f"\n{'='*60}")
        print(f"λ = {lam_hr}/hr  (M={num_robots} 고정, K={layout.num_workstations})")
        print(f"{'='*60}")

        agent = DQNAgent(num_robots=num_robots, robot_capacity=robot_capacity,
                         seed=seed, sim_time=SIM_TIME,
                         num_workstations=layout.num_workstations)

        tr_store = MetricsStore()
        train(layout, agent, tr_store,
              lam=lam, seed=seed,
              total_episodes=total_episodes,
              num_robots=num_robots, robot_capacity=robot_capacity,
              label=f"RL_lam{lam_hr}_train",
              curve_store=curve_store)
        tr_store.save_csv(os.path.join(out_dir, f"train_lam{lam_hr}_M{num_robots}.csv"))
        agents[lam] = agent

        ev_store = MetricsStore()
        evaluate_all(layout, agent, ev_store, lam=lam,
                     num_robots=num_robots, robot_capacity=robot_capacity,
                     eval_seeds=eval_seeds, n_eval=n_eval)
        for r in ev_store.records:
            store.add(r)

        print_comparison(ev_store, labels, lam=lam, num_robots=num_robots,
                         last_n=n_eval, title=f"M={num_robots}  λ={lam_hr}/hr")

        ev_csv = os.path.join(out_dir, f"eval_lam{lam_hr}_M{num_robots}_n{n_eval}.csv")
        ev_store.save_csv(ev_csv)
        print(f"→ 평가 결과 저장: {ev_csv}")

    return agents


def main():
    """λ 그리드 실험을 실행하고 전체 평가 결과와 학습 곡선을 OUT_DIR에 저장한다."""
    os.makedirs(OUT_DIR, exist_ok=True)

    layout = WarehouseLayout(seed=DEFAULT_SEED)

    print("=" * 60)
    print(f"λ grid experiment  (M={NUM_ROBOTS} 고정, K={layout.num_workstations})")
    print("=" * 60)

    store = MetricsStore()
    curve = TrainingCurveStore()
    agents = experiment_lambda_grid(
        layout, store,
        lam_grid=LAMBDA_GRID,
        seed=DEFAULT_SEED,
        total_episodes=TOTAL_EPISODES,
        n_eval=N_EVAL,
        num_robots=NUM_ROBOTS,
        robot_capacity=ROBOT_CAPACITY,
        out_dir=OUT_DIR,
        curve_store=curve,
    )
    store.save_csv(os.path.join(OUT_DIR, "eval_all_lambdas.csv"))
    curve.save_csv(os.path.join(OUT_DIR, "training_curves_all_lambdas.csv"))

    # 선택 사항: 가장 높은 λ로 학습한 에이전트로 부하 민감도 분석
    # analyze_load_sensitivity(
    #     layout, agents[max(LAMBDA_GRID)],
    #     num_robots=NUM_ROBOTS, robot_capacity=ROBOT_CAPACITY,
    #     lam_values=LAMBDA_GRID, n_eval=N_EVAL,
    #     out_csv=os.path.join(OUT_DIR, f"load_sensitivity_summary_M{NUM_ROBOTS}.csv"),
    # )

    print(f"\nDone. CSVs → {OUT_DIR}/")


if __name__ == "__main__":
    main()
