import csv
import random
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from environment import (
    WarehouseLayout, nearest_neighbor_route, nearest_workstation,
    SIM_TIME, LAMBDA_DEFAULT, ROBOT_SPEED,
)
from agents import Robot, DQNAgent
from manager import (
    OrderManager, EventQueue, Event, Order,
    EVENT_ORDER_ARRIVAL, EVENT_NODE_ARRIVAL, EVENT_TRIP_COMPLETE,
)
from metrics import MetricsStore, EpisodeResult, print_comparison, _SOLO_RT_M


@dataclass
class TrainingPoint:
    label:         str
    episode:       int
    reward:        float
    loss:          float
    epsilon:       float
    n_completed:   int
    avg_flow_time: float


class TrainingCurveStore:
    def __init__(self):
        self.records: List[TrainingPoint] = []

    def add(self, p: TrainingPoint):
        self.records.append(p)

    def save_csv(self, path: str):
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


# ─────────────────────────────────────────────────────────────────────────────
# 보상 함수
# ─────────────────────────────────────────────────────────────────────────────

# 삽입 detour 정규화 상수 ≈ 2 × (aisle-graph 최대 leg).
# K=3 워크스테이션 / 3-block 확장 레이아웃(56m x 56m)에 맞춰 재산정
# (기존 2-block/40x36 레이아웃 기준 147.4 -> 치수 비율(~1.48배) 반영).
_D_MAX = 218.0


def compute_reward(
    robot:          Robot,
    order_loc:      Tuple[float, float],
    layout:         WarehouseLayout,
    delta_d:        float,
    robots:         Optional[List[Robot]] = None,
    n_wait:         int   = 0,
    num_robots:     int   = 4,
    robot_capacity: int   = 5,
    gamma_q:        float = 0.3,
    alpha_mid:      float = 2.0,
    alpha_idle:     float = 1.0,
    beta_depletion: float = 0.2,
) -> float:

    alpha   = alpha_idle if robot.is_idle else alpha_mid

    reward  = float(np.exp(-alpha * delta_d / _D_MAX))

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


# ─────────────────────────────────────────────────────────────────────────────
# 기본 시뮬레이터
# ─────────────────────────────────────────────────────────────────────────────

class SimulatorBase:
    """
    K개의 워크스테이션과 로봇 풀을 관리한다.
    - 로봇은 특정 워크스테이션에 영구 고정되지 않는다. 초기 도킹 위치만
      라운드로빈으로 균등 분산해두고(Robot.workstation_id), 이후에는 idle
      상태일 때마다 어떤 워크스테이션의 주문이든 받아 그쪽으로 재도킹
      (Robot.redock)될 수 있다. 즉 특정 워크스테이션에 큐가 몰리면 idle
      로봇들이 그쪽으로 쏠릴 수 있다.
    - [수정] 주문은 생성 시점에 특정 워크스테이션에 고정되지 않으며, 대기
      큐도 워크스테이션 구분 없는 단일 큐(OrderManager.wait_queue) 하나이다
      (problem description #2). 
    - [수정] 로봇의 한 트립은 여러 워크스테이션에 속한 주문을 함께 서비스할
      수 있다(problem description #3, shared fleet). 트립을 시작하거나
      (Robot.assign_trip) mid-trip 로봇에 새 주문을 삽입할 때마다
      (Robot.insert_and_reorder) environment.best_workstation_route 로 그
      시점까지 모인 목적지 전체에 대해 총 이동거리가 최소가 되는 복귀
      워크스테이션을 다시 계산한다. 즉 복귀 지점은 트립 시작 시 한 번
      영구적으로 고정되는 것이 아니라 트립 구성이 바뀔 때마다 갱신된다.
    """

    def __init__(self, layout: WarehouseLayout, lam: float, seed: int,
                 sim_time: float, num_robots: int, robot_capacity: int):
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
        K = self.num_workstations
        robots = []
        for rid in range(self.num_robots):
            k = rid % K
            robots.append(Robot(rid, self.layout.workstations[k],
                                self.layout, self.robot_capacity,
                                workstation_id=k))
        return robots

    def _reset_common(self, seed: int):
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
        self.order_mgr.record_service_start(order, t)

    def _handle_node_arrival(self, ev: Event):
        robot = self.robots[ev.data["robot_id"]]
        robot.end_travel(self.now)
        still_going = robot.arrive_at_node()

        if still_going:
            self._schedule_next(robot)
            return

        # route 가 비었고 더 갈 곳이 없는 상태.
        if robot.pos == robot.workstation:
            # 자신의 home 워크스테이션까지 도착했고, 그 사이 새로 삽입된 정차지도 없었다 → 트립 종료
            self.eq.push(Event(self.now, EVENT_TRIP_COMPLETE, {"robot_id": robot.id}))
        else:
            robot.current_dest = robot.workstation
            self._schedule_next(robot)

    def _complete_trip(self, robot: Robot):
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
        robot.assign_trip([o.loc for o in orders], self.layout)
        self._robot_orders[robot.id] = list(orders)
        for o in orders:
            self._record_service_start(o, self.now)
        self._schedule_next(robot)

    # ── 공통 지표 ─────────────────────────────────────────────────────────────
    def distance_per_completed_order(self) -> float:
        n = self.order_mgr.completed_count()
        return self._total_sim_distance / n if n > 0 else 0.0

    def avg_totes_per_trip(self) -> float:
        return self.order_mgr.completed_count() / max(1, self._total_trips)

    def saving_per_insertion(self) -> float:
        """삽입 1건당 절감 이동거리 = solo_rt − extra_per_insertion."""
        if self._mid_trip_count == 0:
            return 0.0
        extra_per = self._extra_travel / self._mid_trip_count
        return _SOLO_RT_M - extra_per

    def build_result(self, label: str, episode: int, lam: float,
                     reward: float = 0.0, epsilon: float = 0.0) -> EpisodeResult:
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


# ─────────────────────────────────────────────────────────────────────────────
# RL 시뮬레이터
# ─────────────────────────────────────────────────────────────────────────────

class RLSimulator(SimulatorBase):
    """
    Factored 에이전트(agents.DQNAgent) 전용 시뮬레이터.

    핵심 설계:
      1. 행동 = '전역 로봇 인덱스'가 아니라 '이 주문에 대한 eligible 후보
         로봇 리스트 내 인덱스'. eligible 은 idle_slots>0 인 모든 로봇이다
         (mid-trip 로봇이든 idle 로봇이든, 워크스테이션 무관). [수정] 이전
         버전은 mid-trip 로봇을 "이 워크스테이션을 향해 이동 중인" 로봇으로
         제한했는데, 이는 주문이 워크스테이션에 고정되지 않는다는 problem
         description #2/#3 요구사항과 모순되는 오류였다. 이제 mid-trip
         로봇도 목적지 워크스테이션과 무관하게 어떤 주문이든 받을 수 있고,
         받으면 insert_and_reorder()가 복귀 워크스테이션을 다시 최적화한다.
         에이전트는 각 후보를 스칼라로 스코어링하므로 전체 로봇 수 M 이
         커져도 행동공간이 커지지 않는다.
      2. [수정] 전이(transition) pending 체인을 워크스테이션별(dict[ws_id])로
         분리했었으나, 주문 큐 자체가 이제 단일 큐이므로(problem description
         #2) 워크스테이션별로 분리할 근거가 없다. 단일 pending 체인
         (self._pending: Optional[(feat, reward)]) 하나로 통합한다.
      3. build_candidates 호출 시 ws_capacity 를 항상 함께 넘겨 큐 압박
         인코딩의 스케일을 일관되게 유지한다(num_robots × robot_capacity로
         정규화). [수정] 이전 코드는 build_candidates 를 존재하지 않는
         키워드 인자 target_ws_id 로 호출하여 TypeError 로 실행 자체가
         불가능한 상태였다(agents.DQNAgent.build_candidates 시그니처에는
         target_ws_id 파라미터가 없음). build_candidates 는 내부적으로
         nearest_workstation(order_loc, ...) 으로 이미 목표 워크스테이션을
         스스로 계산하므로 해당 인자를 제거한다.
    """

    def __init__(self, layout: WarehouseLayout, agent: DQNAgent,
                 lam: float, seed: int, sim_time: float = SIM_TIME,
                 num_robots: int = 4, robot_capacity: int = 5,
                 reward_cfg: Optional[Dict] = None,
                 eval_mode: bool = False):
        super().__init__(layout, lam, seed, sim_time, num_robots, robot_capacity)
        self.agent        = agent
        self.reward_cfg   = reward_cfg or {}
        self.total_reward = 0.0
        self.eval_mode    = eval_mode
        # [수정] 단일 큐 구조에 맞춘 단일 pending 체인.
        self._pending: Optional[Tuple[np.ndarray, float]] = None

    def reset(self, seed: int):
        self._reset_common(seed)
        self.total_reward = 0.0
        self._pending = None

    def run(self) -> List[tuple]:
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

        # 에피소드 종료: 남아있는 마지막 결정은 다음 결정 상태가 없으므로
        # done=True 로 마감한다. (단일 큐/단일 pending 체인으로 통합됨.)
        if not self.eval_mode and self._pending is not None:
            feat, r = self._pending
            transitions.append((feat, r, None, True))
            self._pending = None

        if self.eval_mode:
            self.agent.epsilon = saved_epsilon
        return transitions

    def _handle_order_arrival(self, ev: Event) -> Optional[tuple]:
        order = self.order_mgr.create_order(self.now)

        # [수정] eligible = idle_slots>0 인 모든 로봇(mid-trip이든 idle이든),
        # 워크스테이션 제약 없음.
        busy     = [r for r in self.robots if not r.is_idle and r.idle_slots > 0]
        idle_any = [r for r in self.robots if r.is_idle and r.idle_slots > 0]
        eligible = busy + idle_any

        if not eligible:
            self.order_mgr.enqueue(order, self.now)
            return None

        # [수정] 단일 큐 기준 대기 길이(queue_size()에 인자 없음).
        n_wait      = self.order_mgr.queue_size()
        ws_capacity = self.num_robots * self.robot_capacity
        # [수정] 존재하지 않는 target_ws_id 인자 제거 (기존 코드는 TypeError
        # 로 실행 불가능했다). build_candidates 는 내부에서 order_loc 기준
        # nearest_workstation 으로 idle 후보의 목표 워크스테이션을 스스로
        # 계산한다.
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

        # 실제 삽입 전에, 삽입될 경우의 detour 를 측정한다(집계용).
        # idle 후보는 '이 주문 위치와 가장 가까운 워크스테이션으로 복귀한다면'
        # 을 가정하고, mid-trip 후보는 삽입 후 최적 복귀 워크스테이션을 다시
        # 탐색(dynamic_ws=True)해 실제 insert_and_reorder()가 확정할 값과
        # 동일한 기준으로 계산한다.
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
            # [수정] 로봇의 워크스테이션을 임의 고정(redock(ws_id)) 하지 않고
            # assign_trip() 이 best_workstation_route 로 최적 복귀지를 확정한다.
            robot.assign_trip([order.loc], self.layout)
            self._mid_trip_count += 0  # (idle → 신규 트립은 mid-trip 삽입이 아님)
        else:
            robot.insert_and_reorder(order.loc, self.layout)
            self._mid_trip_count += 1

        self._robot_orders[robot.id].append(order)
        self._record_service_start(order, self.now)

        if was_idle:
            self._schedule_next(robot)

        if self.eval_mode:
            return None

        # [수정] 단일 큐이므로 pending 체인도 단일 체인이다. 이전 결정을
        # 지금(다음 결정 시점) 후보들로 완성하고, 이번 결정은 pending 으로
        # 보류한다.
        completed = None
        if self._pending is not None:
            feat_prev, r_prev = self._pending
            completed = (feat_prev, r_prev, cand_feats, False)
        self._pending = (chosen_feat, reward)
        return completed

    def _handle_trip_complete(self, ev: Event):
        robot  = self.robots[ev.data["robot_id"]]
        self._complete_trip(robot)
        # [수정] 단일 큐이므로 워크스테이션 인자 없이 FIFO로 최대
        # robot_capacity개까지 꺼낸다(problem description #5).
        orders = self.order_mgr.dequeue_up_to(self.robot_capacity)
        if orders:
            self._dispatch(robot, orders)


# ─────────────────────────────────────────────────────────────────────────────
# BL-1: Full-Buffer
# ─────────────────────────────────────────────────────────────────────────────

class BaselineSimulator(SimulatorBase):
    """K개 슬롯을 모두 채운 후 출발. mid-trip 삽입 없음.

    [수정] 주문이 워크스테이션에 고정되지 않으므로(problem description #2),
    버퍼링 중인 idle 로봇을 "같은 워크스테이션(ws_id)" 조건으로 제한하지
    않는다. 어떤 idle 로봇이든 슬롯이 남아 있으면 버퍼링 후보가 될 수 있고,
    버퍼가 다 찬 시점(assign_trip 호출 시)에 비로소 그동안 모인 목적지
    전체에 대해 best_workstation_route 로 최적 복귀 워크스테이션이
    확정된다(problem description #3) — 따라서 이전 버전처럼 버퍼링 시작
    시점에 미리 redock() 으로 워크스테이션을 고정할 필요가 없다."""

    def reset(self, seed: int):
        self._reset_common(seed)

    def run(self):
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
                # [수정] 단일 큐: 워크스테이션 인자 없이 FIFO로 dequeue.
                orders = self.order_mgr.dequeue_up_to(self.robot_capacity)
                if orders:
                    self._robot_orders[robot.id] = list(orders)
                    robot.idle_slots = self.robot_capacity - len(orders)
                    if robot.idle_slots == 0:
                        robot.assign_trip([o.loc for o in orders], self.layout)
                        self._schedule_next(robot)

    def _handle_bl(self, ev: Event):
        order = self.order_mgr.create_order(self.now)
        # [수정] 워크스테이션 무관: idle_slots>0 인 모든 idle 로봇이 후보.
        idle  = [r for r in self.robots if r.is_idle and r.idle_slots > 0]
        if idle:
            robot = max(idle, key=lambda r: r.idle_slots)
            self._robot_orders[robot.id].append(order)
            self._record_service_start(order, self.now)
            robot.idle_slots -= 1
            if robot.idle_slots == 0:
                # assign_trip() 이 지금까지 모인 목적지 전체를 기준으로
                # 최적 복귀 워크스테이션을 확정한다(is_idle=False 로 갱신됨).
                robot.assign_trip([o.loc for o in self._robot_orders[robot.id]],
                                  self.layout)
                self._schedule_next(robot)
        else:
            self.order_mgr.enqueue(order, self.now)


# ─────────────────────────────────────────────────────────────────────────────
# BL-2: Partial-Buffer
# ─────────────────────────────────────────────────────────────────────────────

class PartialSimulator(SimulatorBase):
    """k개 슬롯만 채워도 즉시 출발. mid-trip 삽입 없음.

    [수정] BaselineSimulator와 동일한 이유로 "같은 워크스테이션" 제약을
    제거했다 — 주문이 워크스테이션에 고정되지 않으므로(problem description
    #2) 버퍼링 후보를 워크스테이션으로 제한할 근거가 없고, 복귀 워크스테이션은
    assign_trip() 시점에 best_workstation_route 로 동적으로 정해진다
    (problem description #3)."""

    def __init__(self, *args, partial_k: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.partial_k = partial_k

    def reset(self, seed: int):
        self._reset_common(seed)

    def run(self):
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
                # [수정] 단일 큐: 워크스테이션 인자 없이 FIFO로 dequeue.
                orders = self.order_mgr.dequeue_up_to(self.robot_capacity)
                if orders:
                    self._robot_orders[robot.id] = list(orders)
                    robot.idle_slots = self.robot_capacity - len(orders)
                    if len(orders) >= self.partial_k:
                        robot.assign_trip([o.loc for o in orders], self.layout)
                        self._schedule_next(robot)

    def _handle_partial(self, ev: Event):
        order = self.order_mgr.create_order(self.now)
        # [수정] 워크스테이션 무관: idle_slots>0 인 모든 idle 로봇이 후보.
        idle  = [r for r in self.robots if r.is_idle and r.idle_slots > 0]
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


# ─────────────────────────────────────────────────────────────────────────────
# BL-3: Random Assignment
# ─────────────────────────────────────────────────────────────────────────────

class RandomAssignSimulator(SimulatorBase):
    """가용 슬롯이 있는 로봇(mid-trip + idle, 워크스테이션 무관) 중 무작위로
    배정. mid-trip 삽입 허용.

    [수정] "동일 워크스테이션" 제약 제거 — 주문이 워크스테이션에 고정되지
    않으므로(problem description #2) mid-trip 로봇도 목적지 워크스테이션과
    무관하게 후보가 된다. idle 로봇이 선택되면 assign_trip() 이, mid-trip
    로봇이 선택되면 insert_and_reorder() 이 각각 best_workstation_route 로
    복귀 워크스테이션을 동적으로 확정한다(problem description #3)."""

    def reset(self, seed: int):
        self._reset_common(seed)
        self._assign_rng = random.Random(seed)

    def run(self):
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
        order    = self.order_mgr.create_order(self.now)
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


# ─────────────────────────────────────────────────────────────────────────────
# BL-4: Greedy Nearest-Robot
# ─────────────────────────────────────────────────────────────────────────────

class NearestRobotSimulator(SimulatorBase):
    """가용 로봇(mid-trip + idle, 워크스테이션 무관) 중 주문 위치까지 현재
    거리가 가장 짧은 로봇에 배정. mid-trip 삽입 허용.

    [수정] "동일 워크스테이션" 제약 제거(RandomAssignSimulator와 동일한
    사유, problem description #2/#3)."""

    def reset(self, seed: int):
        self._reset_common(seed)

    def run(self):
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
        order    = self.order_mgr.create_order(self.now)
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


# ─────────────────────────────────────────────────────────────────────────────
# BL-5: Min-Insertion Detour Greedy
# ─────────────────────────────────────────────────────────────────────────────

class MinInsertionSimulator(SimulatorBase):
    """가용 로봇(mid-trip + idle, 워크스테이션 무관) 중 삽입 detour가 최소인
    로봇에 배정. mid-trip 삽입 허용.

    [수정] "동일 워크스테이션" 제약 제거. idle 후보는 '주문 위치에서 가장
    가까운 워크스테이션으로 복귀한다면'을 가정해 detour를 계산하고, busy
    (mid-trip) 후보는 삽입 후 최적 복귀 워크스테이션을 다시 탐색
    (dynamic_ws=True)해 detour를 계산한다(problem description #2/#3)."""

    def reset(self, seed: int):
        self._reset_common(seed)

    def run(self):
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
        order    = self.order_mgr.create_order(self.now)
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


# ─────────────────────────────────────────────────────────────────────────────
# 학습 루프
# ─────────────────────────────────────────────────────────────────────────────

# λ 그리드(700~1000/hr) 범위 내에서의 커리큘럼 학습용 λ 하한/상한.
# (로봇 수는 14대로 고정하므로 M별 딕셔너리는 더 이상 필요 없다.)
_CURRICULUM_LAM_MARGIN = 0.7   # 학습 λ 대비 하한 = lam * margin


def train(layout:         WarehouseLayout,
          agent:          DQNAgent,
          store:          MetricsStore,
          lam:            float = LAMBDA_DEFAULT,
          seed:           int   = 42,
          total_episodes: int   = 1000,
          num_robots:     int   = 14,
          robot_capacity: int   = 5,
          reward_cfg:     Dict  = None,
          label:          str   = "RL",
          verbose_every:  int   = 50,
          curve_store:    Optional["TrainingCurveStore"] = None) -> None:

    reward_cfg = reward_cfg or {}
    lam_lo, lam_hi = lam * _CURRICULUM_LAM_MARGIN, lam

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

        # reward 는 에피소드 누적 합이 아니라 '완료 주문 1건당 평균'으로 정규화해
        # 기록한다. 누적 합은 완료 주문 수(부하·시뮬레이션 시간에 좌우)에 비례해
        # 커지므로, 서로 다른 부하(λ)나 워크스테이션 수(K) 설정 간 reward 값을
        # 곧바로 비교하면 착시가 생긴다. per-order 평균은 이 스케일 문제를 없앤다.
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


# ─────────────────────────────────────────────────────────────────────────────
# 평가
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(layout:         WarehouseLayout,
                 agent:          DQNAgent,
                 store:          MetricsStore,
                 lam:            float,
                 num_robots:     int = 14,
                 robot_capacity: int = 5,
                 eval_seeds:     List[int] = None,
                 n_eval:         int = 20,
                 partial_k:      int = 3) -> None:

    if eval_seeds is None:
        eval_seeds = list(range(10000, 10000 + n_eval))

    for i, s in enumerate(eval_seeds):

        sim = RLSimulator(layout, agent, lam=lam, seed=s,
                          num_robots=num_robots, robot_capacity=robot_capacity,
                          eval_mode=True)
        sim.reset(s)
        sim.run()
        n_done     = max(1, sim.order_mgr.completed_count())
        avg_reward = sim.total_reward / n_done   # 학습부와 동일하게 per-order 정규화
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


# ─────────────────────────────────────────────────────────────────────────────
# 부하(λ) 민감도 분석 — RL이 베이스라인 대비 이점을 보이는 부하 구간 식별
#
# 시스템이 포화(로봇 가동률 ≈ 100%) 상태이면 배정 정책과 무관하게 처리량이
# 물리적 이동거리 capacity 로 수렴하므로, RL의 우위는 관측되지 않는다.
# 따라서 훈련 λ 뿐 아니라 그보다 낮은 λ(비포화 구간)까지 스윕하여, 각 λ에서
#   (1) 로봇 가동률(utilization) = 실측 총이동거리 / (num_robots·speed·sim_time)
#   (2) RL vs 각 베이스라인의 paired t-test (평균 대기시간, 완료율)
# 을 계산해 RL이 통계적으로 유의하게 나은 부하 구간을 찾는다.
# ─────────────────────────────────────────────────────────────────────────────

def _robot_utilization(store: MetricsStore, label: str, lam: float,
                       num_robots: int) -> float:
    """실측 총이동거리 / 이론적 최대 총이동거리(로봇이 쉬지 않고 이동했을 때)."""
    recs = store.filter(label=label, lam=lam, num_robots=num_robots)
    if not recs:
        return float("nan")
    avg_dist = sum(r.total_distance for r in recs) / len(recs)
    max_dist = num_robots * ROBOT_SPEED * SIM_TIME
    return avg_dist / max_dist if max_dist > 0 else float("nan")


def _paired_ttest(a: List[float], b: List[float]) -> Tuple[float, float]:
    """대응표본 t-검정. (평균차이, p-value) 반환. scipy 없으면 정규근사로 대체."""
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
    print(f"\n{'-'*78}")
    print("RL 우위 구간 요약 (p<0.05 이면서 RL 평균 대기시간이 더 짧은 경우 기준)")
    print(f"{'-'*78}")
    hdr = f"  {'λ(/hr)':>8}  {'util':>7}  {'RL이 유의하게 더 나은 베이스라인 수':>18}"
    print(hdr)
    for row in rows:
        wins = sum(
            1 for b in baseline_labels
            if row[f"wait_p_vs_{b}"] == row[f"wait_p_vs_{b}"]     # NaN 체크
            and row[f"wait_p_vs_{b}"] < 0.05
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
    n_eval:         int = 20,
    seed_base:      int = 30000,
    out_csv:        Optional[str] = None,
) -> MetricsStore:
    """
    학습된 에이전트를 여러 부하(λ) 조건에서 평가하여, RL이 베이스라인 대비
    이점을 보이는 시스템 부하 구간을 식별한다.
    """
    labels = ["RL", "BL1_FullBuffer", "BL2_Partial_k3", "BL3_Random",
              "BL4_Nearest", "BL5_MinInsertion"]
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
            row[f"wait_diff_vs_{other}"] = d_wait   # RL - other (음수면 RL이 더 짧음)
            row[f"wait_p_vs_{other}"]    = p_wait
            row[f"comp_diff_vs_{other}"] = d_comp   # RL - other (양수면 RL이 더 높음)
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


# ─────────────────────────────────────────────────────────────────────────────
# λ 그리드 실험 (로봇 수 M=14 고정, K=3 워크스테이션)
#
# 요청사항 반영:
#   - λ = 700 / 800 / 900 / 1000 (orders/hr) 각각에 대해 학습 후,
#     20개(n_eval) 인스턴스(시드)로 평가하고 결과를 CSV로 저장한다.
#   - 로봇 수는 14대로 고정한다(추후 실험에서 M을 바꿔가며 재사용 예정).
# ─────────────────────────────────────────────────────────────────────────────

def experiment_lambda_grid(
    layout:         WarehouseLayout,
    store:          MetricsStore,
    lam_grid:       List[float] = [700/3600, 800/3600, 900/3600, 1000/3600],
    seed:           int = 42,
    total_episodes: int = 1000,
    n_eval:         int = 20,
    num_robots:     int = 14,
    robot_capacity: int = 5,
    out_dir:        str = "logs",
    curve_store:    Optional[TrainingCurveStore] = None,
) -> Dict[float, DQNAgent]:
    """λ 그리드별로 독립적인 에이전트를 학습·평가하고, 부하별 20 인스턴스
    평가 결과를 각각의 CSV 파일로 저장한다."""
    labels     = ["RL", "BL1_FullBuffer", "BL2_Partial_k3", "BL3_Random",
                  "BL4_Nearest", "BL5_MinInsertion"]
    eval_seeds = list(range(10000, 10000 + n_eval))
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

        # 부하(λ)별 20 인스턴스 평가 결과 CSV 저장
        ev_csv = os.path.join(out_dir, f"eval_lam{lam_hr}_M{num_robots}_n{n_eval}.csv")
        ev_store.save_csv(ev_csv)
        print(f"→ 평가 결과 저장: {ev_csv}")

    return agents


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    SEED           = 42
    ROBOT_CAPACITY = 5
    NUM_ROBOTS     = 14                                        # 로봇 수 14대 고정
    LAM_GRID       = [700/3600, 800/3600, 900/3600, 1000/3600] # λ 그리드
    TOTAL_EPISODES = 1000
    N_EVAL         = 20
    OUT_DIR        = "logs"
    os.makedirs(OUT_DIR, exist_ok=True)

    # K=3 워크스테이션, 3-block x 14-aisle 확장 레이아웃
    layout = WarehouseLayout(seed=SEED)

    print("=" * 60)
    print(f"λ grid experiment  (M={NUM_ROBOTS} 고정, K={layout.num_workstations})")
    print("=" * 60)

    store = MetricsStore()
    curve = TrainingCurveStore()
    agents = experiment_lambda_grid(
        layout, store,
        lam_grid=LAM_GRID,
        seed=SEED,
        total_episodes=TOTAL_EPISODES,
        n_eval=N_EVAL,
        num_robots=NUM_ROBOTS,
        robot_capacity=ROBOT_CAPACITY,
        out_dir=OUT_DIR,
        curve_store=curve,
    )
    store.save_csv(os.path.join(OUT_DIR, "eval_all_lambdas.csv"))
    curve.save_csv(os.path.join(OUT_DIR, "training_curves_all_lambdas.csv"))

    # ── 부하(λ) 민감도 분석 (선택 사항: 학습된 λ=1000/hr 에이전트로 추가 스윕) ──
    # λ 그리드(700~1000/hr) 자체가 이미 부하 스윕이므로, 필요 시 아래처럼
    # 가장 높은 부하로 학습된 에이전트를 기준으로 추가 민감도 분석을 수행할 수
    # 있다(선택 사항이며 기본 실행에서는 생략).
    #
    # analyze_load_sensitivity(
    #     layout, agents[max(LAM_GRID)],
    #     num_robots=NUM_ROBOTS, robot_capacity=ROBOT_CAPACITY,
    #     lam_values=LAM_GRID, n_eval=N_EVAL,
    #     out_csv=os.path.join(OUT_DIR, "load_sensitivity_summary_M14.csv"),
    # )

    print(f"\nDone. CSVs → {OUT_DIR}/")


if __name__ == "__main__":
    main()