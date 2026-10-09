"""이산 사건 창고 시뮬레이터: 주문/사건 관리, 공통 엔진, RL 배정 시뮬레이터."""
import heapq
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np

from environment import WarehouseLayout, Robot, nearest_workstation
from agents import DQNAgent
from metrics import EpisodeResult
from constant import (
    SIM_TIME, ROBOT_SPEED, NUM_ROBOTS, ROBOT_CAPACITY, SOLO_RT_M, DEFAULT_SEED,
    REWARD_D_MAX, REWARD_GAMMA_Q, REWARD_ALPHA_MID, REWARD_ALPHA_IDLE,
    REWARD_BETA_DEPLETION, REWARD_MODE_DEFAULT, GAMMA,
    SMDP_BETA, SMDP_COST_DIST, SMDP_REWARD_SCALE,
)


# 주문 / 사건 관리

@dataclass
class Order:
    """주문 1건.

    주문은 특정 워크스테이션에 귀속되지 않으며, 복귀 워크스테이션은
    주문을 담은 로봇의 트립 구성에 따라 결정된다.
    """
    order_id:        int
    sku:             int
    loc_idx:         int
    loc:             tuple
    arrival_time:    float
    completion_time:    Optional[float] = None
    service_start_time: Optional[float] = None   # 로봇에 배정된 시각


EVENT_ORDER_ARRIVAL = "ORDER_ARRIVAL"
EVENT_NODE_ARRIVAL  = "ROBOT_NODE_ARRIVAL"
EVENT_TRIP_COMPLETE = "ROBOT_TRIP_COMPLETE"


@dataclass(order=True)
class Event:
    """이산 사건 시뮬레이션의 사건. time 기준으로만 정렬된다."""
    time:  float
    etype: str  = field(compare=False)
    data:  dict = field(compare=False, default_factory=dict)


class EventQueue:
    """사건 발생 시각 순으로 꺼내는 우선순위 큐."""

    def __init__(self):
        self._q: list = []

    def push(self, ev: Event):
        """사건을 큐에 넣는다.

        입력:
            ev: 추가할 사건
        """
        heapq.heappush(self._q, ev)

    def pop(self) -> Event:
        """가장 이른 사건을 꺼낸다.

        출력:
            발생 시각이 가장 이른 사건
        """
        return heapq.heappop(self._q)

    def __len__(self):
        return len(self._q)


class OrderManager:
    """주문 생성(포아송 도착)과 단일 FIFO 대기 큐, 주문 단위 지표를 관리한다."""

    def __init__(self, layout, lam: float, seed: int = DEFAULT_SEED):
        """
        입력:
            layout: WarehouseLayout
            lam: 주문 도착률 (orders/s)
            seed: 주문 생성 난수 시드
        """
        self.layout           = layout
        self.lam               = lam
        self._rng               = random.Random(seed)
        self._cnt               = 0
        self.num_workstations   = layout.num_workstations

        self.wait_queue:          List[Order]  = []
        self.all_orders:          List[Order]  = []
        self._queue_entry_times:  List[float]  = []

    def reset(self, seed: int):
        """에피소드 시작 전 상태를 초기화한다.

        입력:
            seed: 새 주문 생성 난수 시드
        """
        self._rng    = random.Random(seed)
        self._cnt    = 0
        self.wait_queue          = []
        self._queue_entry_times  = []
        self.all_orders.clear()

    def generate_next_arrival(self, current_time: float) -> float:
        """지수분포 도착 간격으로 다음 주문 도착 시각을 만든다.

        입력:
            current_time: 현재 시각 (s)
        출력:
            다음 주문 도착 시각 (s)
        """
        return current_time + self._rng.expovariate(self.lam)

    def create_order(self, arrival_time: float) -> Order:
        """SKU와 보관 위치를 샘플링해 새 주문을 만들고 전체 주문 목록에 기록한다.

        입력:
            arrival_time: 주문 도착 시각 (s)
        출력:
            생성된 Order
        """
        sku          = self.layout.sample_order_sku(self._rng)
        loc_idx, loc = self.layout.find_tote_location(sku, self._rng)
        order        = Order(self._cnt, sku, loc_idx, loc, arrival_time)
        self._cnt   += 1
        self.all_orders.append(order)
        return order

    def enqueue(self, order: Order, enqueue_time: float):
        """배정할 로봇이 없는 주문을 대기 큐 끝에 넣는다.

        입력:
            order: 대기시킬 주문
            enqueue_time: 큐 진입 시각 (s)
        """
        self.wait_queue.append(order)
        self._queue_entry_times.append(enqueue_time)

    def dequeue_up_to(self, n: int) -> List[Order]:
        """대기 큐 앞에서 FIFO 순서로 최대 n개를 꺼낸다.

        입력:
            n: 꺼낼 최대 주문 수
        출력:
            꺼낸 주문 리스트
        """
        k  = min(n, len(self.wait_queue))
        orders = self.wait_queue[:k]
        self.last_batch_entry_times = list(self._queue_entry_times[:k])
        self.wait_queue             = self.wait_queue[k:]
        self._queue_entry_times     = self._queue_entry_times[k:]
        return orders

    def queue_size(self) -> int:
        """현재 대기 큐 길이."""
        return len(self.wait_queue)

    def record_service_start(self, order: Order, time: float):
        """주문이 처음 로봇에 배정된 시각을 기록한다. 이미 기록돼 있으면 무시한다.

        입력:
            order: 대상 주문
            time: 배정 시각 (s)
        """
        if order.service_start_time is None:
            order.service_start_time = time

    def record_completion(self, order: Order, time: float):
        """주문 완료(워크스테이션 도착) 시각을 기록한다.

        입력:
            order: 대상 주문
            time: 완료 시각 (s)
        """
        order.completion_time = time

    def avg_wait_time(self) -> float:
        """평균 대기시간(배정 시각 − 도착 시각). 이동/서비스 시간은 제외한다.

        출력:
            배정된 주문들의 평균 대기시간 (s), 없으면 0.0
        """
        waits = [
            o.service_start_time - o.arrival_time
            for o in self.all_orders if o.service_start_time is not None
        ]
        return sum(waits) / len(waits) if waits else 0.0

    def avg_flow_time(self) -> float:
        """평균 체류시간(완료 시각 − 도착 시각). 대기 + 이동 시간을 모두 포함한다.

        출력:
            완료된 주문들의 평균 체류시간 (s), 없으면 0.0
        """
        flows = [
            o.completion_time - o.arrival_time
            for o in self.all_orders if o.completion_time is not None
        ]
        return sum(flows) / len(flows) if flows else 0.0

    def completed_count(self) -> int:
        """완료된 주문 수."""
        return sum(1 for o in self.all_orders if o.completion_time is not None)


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

    하위 클래스는 _on_order_arrival()로 배정 정책을 정의하고, 필요하면
    _on_trip_complete()를 바꾼다.
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
        self._n_in_system:           int                    = 0

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

    def reset(self, seed: int):
        """에피소드 상태와 로봇을 초기화하고 첫 주문 도착 사건을 넣는다.

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
        self._n_in_system           = 0
        for r in self.robots:
            r.reset()
        t0 = self.order_mgr.generate_next_arrival(0.0)
        self.eq.push(Event(t0, EVENT_ORDER_ARRIVAL))

    def run(self):
        """sim_time까지 사건을 시각 순으로 처리한다."""
        while self.eq:
            ev = self.eq.pop()
            if ev.time > self.sim_time:
                break
            self._advance_clock(ev.time)
            if ev.etype == EVENT_ORDER_ARRIVAL:
                self._n_in_system += 1
                self._on_order_arrival()
                nxt = self.order_mgr.generate_next_arrival(self.now)
                if nxt <= self.sim_time:
                    self.eq.push(Event(nxt, EVENT_ORDER_ARRIVAL))
            elif ev.etype == EVENT_NODE_ARRIVAL:
                self._handle_node_arrival(ev)
            elif ev.etype == EVENT_TRIP_COMPLETE:
                self._on_trip_complete(self.robots[ev.data["robot_id"]])
        self._advance_clock(self.sim_time)

    def _advance_clock(self, t: float):
        """시각을 t로 옮기며 [now, t] 구간의 상태를 _on_interval()에 넘긴다.

        사건 사이에는 시스템 내 주문 수와 이동 중 로봇 수가 바뀌지 않는다.

        입력:
            t: 새 시각 (s)
        """
        if t > self.now:
            n_busy = sum(1 for r in self.robots if not r.is_idle)
            self._on_interval(self.now, t, self._n_in_system, n_busy)
        self.now = t

    def _on_interval(self, t0: float, t1: float, n_in_system: int, n_busy: int):
        """사건 사이 구간 [t0, t1] 훅. 기본은 아무 것도 하지 않는다.

        입력:
            t0, t1: 구간 시작/끝 시각 (s)
            n_in_system: 구간 동안 시스템 내 주문 수 (도착 − 완료)
            n_busy: 구간 동안 이동 중인 로봇 수
        """
        pass

    def _on_order_arrival(self):
        """새 주문 도착 시 배정 정책. 하위 클래스가 구현한다."""
        raise NotImplementedError

    def _on_trip_complete(self, robot: Robot):
        """트립 완료 처리 후, 대기 주문이 있으면 최대 용량만큼 꺼내 새 트립을 시작한다.

        입력:
            robot: 트립을 마친 로봇
        """
        self._complete_trip(robot)
        orders = self.order_mgr.dequeue_up_to(self.robot_capacity)
        if orders:
            self._dispatch(robot, orders)

    def _eligible_robots(self) -> List[Robot]:
        """빈 슬롯이 있는 로봇 목록 (mid-trip 로봇 먼저, 그다음 idle 로봇).

        출력:
            Robot 리스트
        """
        busy     = [r for r in self.robots if not r.is_idle and r.idle_slots > 0]
        idle_any = [r for r in self.robots if r.is_idle and r.idle_slots > 0]
        return busy + idle_any

    def _idle_return_target(self, loc: Tuple[float, float]) -> Tuple[float, float]:
        """idle 로봇에 배정할 때 가정하는 복귀 지점 (주문 위치에서 가장 가까운 워크스테이션).

        입력:
            loc: 주문 위치
        출력:
            워크스테이션 좌표
        """
        ws_id = nearest_workstation(loc, self.layout.workstations, self.layout.aisle_distance)
        return self.layout.workstations[ws_id]

    def _insertion_delta(self, robot: Robot, loc: Tuple[float, float],
                         target_coords: Tuple[float, float]) -> float:
        """로봇에 주문을 넣을 때 늘어나는 이동거리.

        idle 로봇은 target_coords로 복귀한다고 가정하고, mid-trip 로봇은
        최적 복귀 워크스테이션을 재탐색한다.

        입력:
            robot: 후보 로봇
            loc: 주문 위치
            target_coords: idle 로봇의 가정 복귀 좌표
        출력:
            추가 이동거리 (m)
        """
        return robot.delta_distance_insert(
            loc, self.layout,
            target_ws_coords=(target_coords if robot.is_idle else None),
            dynamic_ws=(not robot.is_idle))

    def _assign_order(self, robot: Robot, order: Order):
        """주문을 로봇에 배정한다. idle이면 새 트립을 시작하고, 이동 중이면 경로에 삽입한다.

        입력:
            robot: 배정할 로봇
            order: 주문
        """
        self._robot_orders[robot.id].append(order)
        self._record_service_start(order, self.now)
        if robot.is_idle:
            robot.assign_trip([order.loc], self.layout)
            self._schedule_next(robot)
        else:
            robot.insert_and_reorder(order.loc, self.layout)
            self._mid_trip_count += 1

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
        self._n_in_system -= len(completed)
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
                 eval_mode: bool = False,
                 reward_mode: str = REWARD_MODE_DEFAULT):
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
            reward_mode: "legacy" (배정 시점 보상, 결정당 γ 할인) 또는
                         "smdp" (결정 epoch 사이 시간 적분 비용, e^{-βτ} 할인)
        """
        assert reward_mode in ("legacy", "smdp"), reward_mode
        super().__init__(layout, lam, seed, sim_time, num_robots, robot_capacity)
        self.agent        = agent
        self.reward_cfg   = reward_cfg or {}
        self.total_reward = 0.0
        self.eval_mode    = eval_mode
        self.reward_mode  = reward_mode
        # legacy: (feat, reward), smdp: (feat, 결정 시각)
        self._pending: Optional[Tuple[np.ndarray, float]] = None
        self._transitions: List[tuple] = []
        self._smdp_cost = 0.0   # 직전 결정 이후 누적된 할인 비용

    def reset(self, seed: int):
        """에피소드 상태, 누적 보상, 보류 전이를 초기화한다.

        입력:
            seed: 주문 생성 난수 시드
        """
        super().reset(seed)
        self.total_reward = 0.0
        self._pending = None
        self._smdp_cost = 0.0

    def run(self) -> List[tuple]:
        """에피소드 1회를 끝까지 실행한다.

        출력:
            학습용 전이 리스트 [(feat, reward, next_feats, done, discount), ...]
            (eval_mode이면 빈 리스트)
        """
        self._transitions = []
        saved_epsilon = self.agent.epsilon
        if self.eval_mode:
            self.agent.epsilon = 0.0

        super().run()

        # 마지막 결정은 다음 상태가 없으므로 done=True로 마감한다.
        # smdp이면 sim_time까지 누적된 비용이 마지막 보상이 된다.
        if self.reward_mode == "smdp":
            self._close_smdp_interval()
        if not self.eval_mode and self._pending is not None:
            feat, r, g = self._pending_transition(self.sim_time)
            self._transitions.append((feat, r, None, True, g))
            self._pending = None

        if self.eval_mode:
            self.agent.epsilon = saved_epsilon
        return self._transitions

    def _on_order_arrival(self):
        """새 주문을 에이전트가 고른 로봇에 배정하고 보상을 계산한다.

        후보가 없으면 주문을 대기 큐에 넣는다. 학습 모드이면 직전 결정의 전이를
        완성해 _transitions에 추가하고, 이번 결정은 _pending에 보류한다.
        """
        order    = self.order_mgr.create_order(self.now)
        eligible = self._eligible_robots()

        if not eligible:
            self.order_mgr.enqueue(order, self.now)
            return

        n_wait      = self.order_mgr.queue_size()
        ws_capacity = self.num_robots * self.robot_capacity
        cand_feats  = self.agent.build_candidates(
            order.loc, eligible, n_wait, layout=self.layout, now=self.now,
            ws_capacity=ws_capacity,
        )
        target_coords = self._idle_return_target(order.loc)

        idx = self.agent.select_index(cand_feats)
        if idx < 0:
            idx = int(np.argmin([
                self._insertion_delta(r, order.loc, target_coords) for r in eligible
            ]))

        robot       = eligible[idx]
        chosen_feat = cand_feats[idx]

        # 배정 전 상태 기준으로 detour를 측정한다 (보상 및 extra_travel 집계용).
        delta_d = self._insertion_delta(robot, order.loc, target_coords)
        self._extra_travel += delta_d

        if self.reward_mode == "legacy":
            reward = compute_reward(
                robot, order.loc, self.layout, delta_d,
                robots=eligible,
                n_wait=n_wait,
                num_robots=len(eligible),
                robot_capacity=self.robot_capacity,
                **self.reward_cfg,
            )
            self.total_reward += reward
        else:
            self._close_smdp_interval()

        self._assign_order(robot, order)

        # 직전 결정을 지금의 후보 특징으로 완성하고, 이번 결정은 보류한다.
        if not self.eval_mode and self._pending is not None:
            feat_prev, r_prev, g_prev = self._pending_transition(self.now)
            self._transitions.append((feat_prev, r_prev, cand_feats, False, g_prev))
        if self.reward_mode == "legacy":
            self._pending = (chosen_feat, reward)
        else:
            self._pending = (chosen_feat, self.now)
            self._smdp_cost = 0.0

    def _on_interval(self, t0: float, t1: float, n_in_system: int, n_busy: int):
        """smdp 모드에서 [t0, t1] 구간 비용을 직전 결정 시각 기준으로 할인해 누적한다.

        구간 비용률 c = N(t) + c_d · v · busy(t) 는 구간 안에서 상수이므로
        ∫_{t0}^{t1} e^{-β(t-t_k)} c dt = c · (e^{-β(t0-t_k)} − e^{-β(t1-t_k)}) / β.

        입력:
            t0, t1: 구간 시작/끝 시각 (s)
            n_in_system: 시스템 내 주문 수
            n_busy: 이동 중 로봇 수
        """
        if self.reward_mode != "smdp" or self._pending is None:
            return
        t_k  = self._pending[1]
        rate = n_in_system + SMDP_COST_DIST * ROBOT_SPEED * n_busy
        self._smdp_cost += rate * (math.exp(-SMDP_BETA * (t0 - t_k))
                                   - math.exp(-SMDP_BETA * (t1 - t_k))) / SMDP_BETA

    def _close_smdp_interval(self):
        """직전 결정 이후 누적된 비용을 보상으로 확정해 total_reward에 더한다."""
        if self._pending is None:
            return
        r = -self._smdp_cost / SMDP_REWARD_SCALE
        self.total_reward += r
        self._pending = (self._pending[0], self._pending[1], r)

    def _pending_transition(self, t_next: float) -> Tuple[np.ndarray, float, float]:
        """보류 중인 직전 결정을 (feat, reward, discount)로 만든다.

        입력:
            t_next: 다음 결정 시각 (s)
        출력:
            (선택 후보 특징, 보상, 이 전이의 할인 계수)
        """
        if self.reward_mode == "legacy":
            feat, r = self._pending
            return feat, r, GAMMA
        feat, t_k, r = self._pending
        return feat, r, math.exp(-SMDP_BETA * (t_next - t_k))
