"""그리드 창고 시뮬레이터: 주문/토트/사건 관리, 셀 단위 틱 이동 엔진 (reactive / reserve)

주문 1건 = 서로 다른 SKU 여러 개. 토트 1개는 SKU 1개만 담으므로 SKU마다 토트 1개를 워크스테이션으로 가져와야 함
  - 주문은 도착 시 워크스테이션 하나에 배정 (_choose_ws), 그 주문의 토트는 모두 그 WS에서 처리
  - 배정 정책은 토트 단위로 로봇을 고름. 한 주문의 토트가 여러 로봇·트립에 나뉠 수 있음
  - 주문 완료 = 마지막 토트의 WS 처리 완료
  - 토트 재고/반납은 모델링하지 않음 (가져간 저장 위치도 계속 사용 가능)
"""
import heapq
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from environment import Robot
from grid import GridWarehouseLayout
from reservation import ReservationTable, plan_leg
from metrics import EpisodeResult
from constant import (
    SIM_TIME, ROBOT_SPEED, DEFAULT_SEED, CELL_SIZE,
    GRID_REPLAN_WAIT, GRID_DETOUR_MAX_EXTRA, GRID_DEADLOCK_TICKS,
)

Cell = Tuple[int, int]


# 주문 / 토트 / 사건 관리

@dataclass(eq=False)
class Tote:
    """주문의 SKU 1개를 담은 토트 1개 (배정 단위)"""
    tote_id:      int
    order:        "Order"
    sku:          int
    loc_idx:      int
    loc:          Cell
    pick_s:       float = 0.0             # 랙에서 이 토트를 꺼내는 시간 (s)
    proc_s:       float = 0.0             # 워크스테이션에서 이 토트를 처리하는 시간 (s)
    assign_time:  Optional[float] = None  # 로봇에 배정된 시각
    done_time:    Optional[float] = None  # WS 처리 완료 시각
    picked:       bool  = False           # 로봇이 이미 집었는지

    @property
    def ws_id(self) -> int:
        return self.order.ws_id


@dataclass(eq=False)
class Order:
    """주문 1건: 서로 다른 SKU의 토트 목록, 처리할 워크스테이션"""
    order_id:           int
    arrival_time:       float
    totes:              List[Tote] = field(default_factory=list)
    ws_id:              int = -1
    service_start_time: Optional[float] = None   # 첫 토트가 로봇에 배정된 시각
    completion_time:    Optional[float] = None   # 마지막 토트의 WS 처리 완료 시각
    n_done:             int = 0


EVENT_ORDER_ARRIVAL = "ORDER_ARRIVAL"


@dataclass(order=True)
class Event:
    """이산 사건 시뮬레이션의 사건. time 기준으로만 정렬"""
    time:  float
    etype: str  = field(compare=False)
    data:  dict = field(compare=False, default_factory=dict)


class EventQueue:
    """사건 발생 시각 순으로 꺼내는 우선순위 큐"""

    def __init__(self):
        self._q: list = []

    def push(self, ev: Event):
        """사건을 큐에 추가

        입력:
            ev: 추가할 사건
        """
        heapq.heappush(self._q, ev)

    def pop(self) -> Event:
        """가장 이른 사건 꺼내기

        출력:
            발생 시각이 가장 이른 사건
        """
        return heapq.heappop(self._q)

    def __len__(self):
        return len(self._q)

    def peek_time(self) -> float:
        """가장 이른 사건의 시각 (비어 있으면 inf)"""
        return self._q[0].time if self._q else math.inf


class OrderManager:
    """주문 생성(layout.scenario의 분포)과 토트 단위 FIFO 대기 큐, 주문 단위 지표 관리"""

    def __init__(self, layout, lam: float, seed: int = DEFAULT_SEED):
        """
        입력:
            layout: GridWarehouseLayout
            lam: 주문 도착률 (orders/s)
            seed: 주문 생성 난수 시드
        """
        self.layout           = layout
        self.lam               = lam
        self._rng               = random.Random(seed)
        self._cnt               = 0
        self._tote_cnt          = 0
        self.num_workstations   = layout.num_workstations

        self.wait_queue:          List[Tote]   = []
        self.all_orders:          List[Order]  = []
        self.all_totes:           List[Tote]   = []
        self._queue_entry_times:  List[float]  = []
        self.horizon            = SIM_TIME      # 비정상 도착률 프로파일의 기준 시간 (SimulatorBase가 설정)
        self._seed_streams(seed)

    def _seed_streams(self, seed: int):
        """기본 분포가 아닌 도착 과정/처리 시간용 난수 생성기와 분포 샘플러 생성"""
        from scenario import OrderStream
        self.stream       = OrderStream(self.layout.scenario, self.lam, self.horizon, len(self.layout.sku_ids))
        self._arrival_rng = random.Random(f"{seed}-arrival")
        self._time_rng    = random.Random(f"{seed}-time")

    def reset(self, seed: int):
        """에피소드 시작 전 상태 초기화

        입력:
            seed: 새 주문 생성 난수 시드
        """
        self._rng      = random.Random(seed)
        self._cnt      = 0
        self._tote_cnt = 0
        self.wait_queue          = []
        self._queue_entry_times  = []
        self.all_orders.clear()
        self.all_totes.clear()
        self._seed_streams(seed)

    def next_arrival(self, current_time: float) -> Tuple[float, int]:
        """다음 주문 도착 시각과 그때 들어오는 주문 수 (묶음 도착이 아니면 1)

        입력:
            current_time: 현재 시각 (s)
        출력:
            (다음 도착 시각 (s), 주문 수)
        """
        return self.stream.next_arrival(current_time, self._rng, self._arrival_rng)

    def create_order(self, arrival_time: float) -> Order:
        """SKU 수, 서로 다른 SKU들, SKU마다 토트 위치를 샘플링해 새 주문 생성

        입력:
            arrival_time: 주문 도착 시각 (s)
        출력:
            생성된 Order (ws_id는 시뮬레이터가 정함)
        """
        order = Order(self._cnt, arrival_time)
        n     = self.stream.sample_lines(self._rng)
        for sku in self.stream.sample_skus(self._rng, self.layout.sku_ids, n):
            loc_idx, loc = self.layout.find_tote_location(sku, self._rng)
            tote = Tote(self._tote_cnt, order, sku, loc_idx, loc,
                        pick_s=self.stream.sample_pick(self._time_rng),
                        proc_s=self.stream.sample_proc(self._time_rng))
            self._tote_cnt += 1
            order.totes.append(tote)
            self.all_totes.append(tote)
        self._cnt += 1
        self.all_orders.append(order)
        return order

    def enqueue(self, tote: Tote, enqueue_time: float):
        """배정할 로봇이 없는 토트를 대기 큐 끝에 추가

        입력:
            tote: 대기시킬 토트
            enqueue_time: 큐 진입 시각 (s)
        """
        self.wait_queue.append(tote)
        self._queue_entry_times.append(enqueue_time)

    def dequeue_up_to(self, n: int) -> List[Tote]:
        """대기 큐 앞에서 FIFO 순서로 최대 n개 꺼내기

        입력:
            n: 꺼낼 최대 토트 수
        출력:
            꺼낸 토트 리스트
        """
        k  = min(n, len(self.wait_queue))
        totes = self.wait_queue[:k]
        self.wait_queue         = self.wait_queue[k:]
        self._queue_entry_times = self._queue_entry_times[k:]
        return totes

    def queue_size(self) -> int:
        """현재 대기 큐 길이 (토트 수)"""
        return len(self.wait_queue)

    def record_assign(self, tote: Tote, time: float):
        """토트가 로봇에 배정된 시각 기록 (주문의 첫 배정 시각도 갱신)

        입력:
            tote: 대상 토트
            time: 배정 시각 (s)
        """
        if tote.assign_time is None:
            tote.assign_time = time
        o = tote.order
        if o.service_start_time is None:
            o.service_start_time = time

    def record_tote_done(self, tote: Tote, time: float) -> bool:
        """토트 WS 처리 완료 기록. 주문의 마지막 토트였으면 주문 완료 처리 후 True

        입력:
            tote: 처리를 마친 토트
            time: 완료 시각 (s)
        """
        tote.done_time = time
        o = tote.order
        o.n_done += 1
        if o.n_done == len(o.totes):
            o.completion_time = time
            return True
        return False

    def avg_wait_time(self) -> float:
        """주문 평균 대기시간 (첫 토트 배정 시각 − 도착 시각). 이동/서비스 시간 제외

        출력:
            배정된 주문들의 평균 대기시간 (s), 없으면 0.0
        """
        waits = [
            o.service_start_time - o.arrival_time
            for o in self.all_orders if o.service_start_time is not None
        ]
        return sum(waits) / len(waits) if waits else 0.0

    def avg_flow_time(self) -> float:
        """주문 평균 체류시간 (완료 시각 − 도착 시각). 대기 + 이동 + 처리 시간 모두 포함

        출력:
            완료된 주문들의 평균 체류시간 (s), 없으면 0.0
        """
        flows = [
            o.completion_time - o.arrival_time
            for o in self.all_orders if o.completion_time is not None
        ]
        return sum(flows) / len(flows) if flows else 0.0

    def completed_count(self) -> int:
        """완료된 주문 수"""
        return sum(1 for o in self.all_orders if o.completion_time is not None)

    def totes_done_count(self) -> int:
        """WS 처리를 마친 토트 수"""
        return sum(1 for t in self.all_totes if t.done_time is not None)


# 시뮬레이터

class SimulatorBase:
    """그리드 창고 시뮬레이터의 공통 부분

    - 로봇 풀은 모든 워크스테이션이 공유, 초기 위치만 라운드로빈으로 분산
    - 대기 토트는 단일 FIFO 큐(OrderManager.wait_queue)로 관리
    - 트립: 실은 토트의 피킹 위치를 NNH로 돈 뒤, 그 토트들의 WS 도킹 셀을 NNH로 돌며 하역
      (environment.plan_trip). 도킹 셀에서는 그 WS로 가는 토트의 처리 시간 합만큼 머묾

    하위 클래스는 _on_tote_arrival()로 배정 정책 정의, 필요하면
    _on_trip_complete(), _choose_ws() 재정의

    로봇 이동은 셀 단위 틱(CELL_SIZE / ROBOT_SPEED)으로 시뮬레이션 (run)
    - 배정 결정은 주문 도착 사건 시점, 결정 시점의 로봇 위치는 직전 틱 기준
    - 이동거리는 실제로 이동한 셀 수
    - 충돌 처리는 layout.motion으로 선택
      - "reactive": 매 틱 다음 셀이 비어 있으면 이동, 막히면 대기/우회
      - "reserve": 로봇의 경로가 바뀔 때(배정, 삽입, 재출발) 그 로봇의 트립 전체를
        시공간 경로로 계획해 예약표에 올리고, 매 틱 예약대로 이동 (_reserve_plan)
    """

    def __init__(self, layout: GridWarehouseLayout, lam: float, seed: int,
                 sim_time: float, num_robots: int, robot_capacity: int):
        """
        입력:
            layout: GridWarehouseLayout
            lam: 주문 도착률 (orders/s)
            seed: 주문 생성 난수 시드
            sim_time: 시뮬레이션 시간 (s)
            num_robots: 로봇 수
            robot_capacity: 로봇 용량 (토트)
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
        self.order_mgr.horizon = sim_time
        self.eq        = EventQueue()
        self.now       = 0.0
        self.reserve   = layout.motion == "reserve"
        self._reset_stats()
        # 시각화 기록기 (viewer.recorder.Recorder). None이면 기록 안 함
        self.recorder = None

    def _reset_stats(self):
        # 로봇별 배정됐지만 아직 하역하지 않은 토트 (배정 순서)
        self._robot_totes:   Dict[int, List[Tote]] = {i: [] for i in range(self.num_robots)}
        self._dropping:      Dict[int, List[Tote]] = {i: [] for i in range(self.num_robots)}   # WS 처리 중인 토트
        self._ws_load:       List[int]             = [0] * self.num_workstations   # WS별 처리 전 토트 수
        self._completions:   List[Tuple[float, float]] = []    # (완료 시각, 흐름시간), 주문 완료 순
        self._total_sim_distance = 0.0
        self._total_trips        = 0
        self._trip_totes         = [0] * self.num_robots   # 로봇별 이번 트립에 하역한 토트 수
        self._totes_in_trips     = 0                       # 완료된 트립의 하역 토트 합
        self._mid_trip_count     = 0
        self._extra_travel       = 0.0
        self._n_in_system        = 0
        self._blocked_time       = 0.0
        self._n_replans          = 0
        self._n_plans            = 0
        self._n_plan_fail        = 0
        self._plan_time          = 0.0

    def _build_robots(self) -> List[Robot]:
        """로봇 생성 및 워크스테이션에 라운드로빈으로 초기 배치

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
        """에피소드 상태와 로봇 초기화, 첫 주문 도착 사건 추가

        입력:
            seed: 주문 생성 난수 시드
        """
        self.now = 0.0
        self.order_mgr.reset(seed)
        self.eq  = EventQueue()
        self._reset_stats()
        for r in self.robots:
            r.reset()
        for r, pos in zip(self.robots, self.layout.initial_positions(self.num_robots)):
            r.pos, r.park_spot = pos, pos
        self._spots_taken = {r.park_spot for r in self.robots}
        self._occ: Dict[Cell, int] = {r.pos: r.id for r in self.robots}
        self._last_move_time = 0.0
        # 예약 모드: 지금 계획을 세우면 경로가 시작되는 틱 (주문 도착 시 직전 틱, 틱 처리 중이면 그 틱)
        self._t0_ctx = 0
        if self.reserve:
            self._res = ReservationTable()
            for r in self.robots:
                self._res.set_rest(r.id, self.layout.cell_index[r.pos], 0)
        t0, n0 = self.order_mgr.next_arrival(0.0)
        self.eq.push(Event(t0, EVENT_ORDER_ARRIVAL, {"count": n0}))
        if self.recorder is not None:
            self.recorder.on_reset(self)

    def run(self):
        """sim_time까지 진행. 고정 간격 틱으로 로봇 이동, 틱 사이에 도착한 주문은 도착 시각에 처리

        틱 시각은 k · tick으로 주문 도착과 무관하게 고정
        움직일 로봇도, 진행 중인 피킹/처리도 없으면 다음 주문 도착 직후 틱으로 건너뜀
        """
        dt = self.layout.tick
        k  = 1
        while True:
            t_ev = self.eq.peek_time()
            if not self._grid_active():
                if math.isinf(t_ev):
                    break
                k = max(k, math.ceil(t_ev / dt))
            t_tick = k * dt
            if min(t_ev, t_tick) > self.sim_time:
                break
            if t_ev <= t_tick:
                ev = self.eq.pop()
                self._advance_clock(t_ev)
                self._t0_ctx = k - 1
                self._handle_order_event(ev)
            else:
                self._advance_clock(t_tick)
                self._t0_ctx = k
                if self.reserve:
                    self._reserve_tick(k)
                else:
                    self._grid_tick()
                if self.recorder is not None:
                    self.recorder.on_tick(self, k)
                k += 1
        self._advance_clock(self.sim_time)

    def _handle_order_event(self, ev: Event):
        """주문 도착 사건: 주문마다 WS를 정하고 토트마다 배정 정책 호출, 다음 도착 사건 예약"""
        for _ in range(ev.data.get("count", 1)):
            order = self.order_mgr.create_order(self.now)
            order.ws_id = self._choose_ws(order)
            self._ws_load[order.ws_id] += len(order.totes)
            self._n_in_system += 1
            for tote in order.totes:
                self._on_tote_arrival(tote)
        nxt, n = self.order_mgr.next_arrival(self.now)
        if nxt <= self.sim_time:
            self.eq.push(Event(nxt, EVENT_ORDER_ARRIVAL, {"count": n}))

    def _advance_clock(self, t: float):
        """시각을 t로 옮김

        입력:
            t: 새 시각 (s)
        """
        self.now = t

    def _choose_ws(self, order: Order) -> int:
        """주문을 처리할 워크스테이션: 처리 전 토트가 가장 적은 WS (동률이면 번호가 작은 WS)"""
        return min(range(self.num_workstations), key=lambda k: (self._ws_load[k], k))

    def _on_tote_arrival(self, tote: Tote):
        """새 토트의 배정 정책. 하위 클래스에서 구현"""
        raise NotImplementedError

    def _on_trip_complete(self, robot: Robot):
        """트립 완료 처리 후, 대기 토트가 있으면 최대 용량만큼 꺼내 새 트립 시작

        입력:
            robot: 트립을 마친 로봇
        """
        self._complete_trip(robot)
        totes = self.order_mgr.dequeue_up_to(self.robot_capacity)
        if totes:
            self._dispatch(robot, totes)

    def _eligible_robots(self) -> List[Robot]:
        """빈 슬롯이 있는 로봇 목록 (mid-trip 로봇 먼저, 그다음 idle 로봇)

        출력:
            Robot 리스트
        """
        # 워크스테이션 처리 중인 로봇은 도킹 셀에 묶여 있으므로 삽입 후보에서 제외
        busy     = [r for r in self.robots
                    if not r.is_idle and r.idle_slots > 0 and r.timer_kind != "ws"]
        idle_any = [r for r in self.robots if r.is_idle and r.idle_slots > 0]
        return busy + idle_any

    def _needed_docks(self, r: Robot) -> List[Cell]:
        """r이 아직 하역하지 않은 토트의 WS 도킹 셀 (WS 번호 순)

        current_dest가 도킹 셀 k이고 k로 갈 토트를 모두 집었으면 그 도착에서 하역하므로 제외
        """
        ws = {t.ws_id for t in self._robot_totes[r.id]}
        c = r.current_dest
        if c is not None and self.layout.is_dock(c):
            k = self.layout.dock_ws[c]
            if k in ws and all(t.picked for t in self._robot_totes[r.id] if t.ws_id == k):
                ws.discard(k)
        return [self.layout.workstations[k] for k in sorted(ws)]

    def _robot_time_left(self, r: Robot) -> float:
        """트립 완료까지 남은 예상 시간 (s): 이동 + 남은 피킹 + WS 처리. 혼잡 대기는 미포함

        입력:
            r: 대상 로봇
        출력:
            예상 잔여 시간 (s), idle이면 0.0
        """
        if r.is_idle:
            return 0.0
        totes = self._robot_totes[r.id]
        return (r.remaining_distance / ROBOT_SPEED + r.timer
                + sum(t.pick_s for t in totes if not t.picked) + sum(t.proc_s for t in totes))

    def _insertion_delta(self, robot: Robot, tote: Tote) -> float:
        """로봇에 토트를 넣을 때 늘어나는 이동거리 (m)

        삽입 후 경로 = 현재 목적지 유지 + (남은 피킹 셀 + 새 토트 위치) NNH + (필요 도킹 셀 + 새 토트 WS) NNH
        idle 로봇은 현재 위치에서 출발하는 새 트립 전체 거리

        입력:
            robot: 후보 로봇
            tote: 새 토트
        출력:
            추가 이동거리 (m)
        """
        dock  = self.layout.workstations[tote.ws_id]
        docks = self._needed_docks(robot)
        if dock not in docks:
            docks.append(dock)
        after = robot.trip_distance_with(robot.route + [tote.loc], docks)
        return after - robot.remaining_distance

    def _assign_tote(self, robot: Robot, tote: Tote):
        """토트를 로봇에 배정. idle이면 새 트립 시작, 이동 중이면 경로에 삽입

        입력:
            robot: 배정할 로봇
            tote: 토트
        """
        self._robot_totes[robot.id].append(tote)
        self.order_mgr.record_assign(tote, self.now)
        if robot.is_idle:
            robot.assign_trip([tote.loc], self._needed_docks(robot))
            self._schedule_next(robot)
        elif self.reserve:
            robot.idle_slots = max(0, robot.idle_slots - 1)
            self._reserve_insert(robot, robot.pending_insert + [tote.loc])
            self._mid_trip_count += 1
        else:
            robot.idle_slots = max(0, robot.idle_slots - 1)
            robot.reroute(robot.route + [tote.loc], self._needed_docks(robot))
            self._mid_trip_count += 1

    def _complete_trip(self, robot: Robot):
        """트립을 마친 로봇을 idle로 되돌림 (토트는 도킹 셀마다 이미 하역)

        입력:
            robot: 트립을 마친 로봇
        """
        if self._robot_totes[robot.id]:
            raise RuntimeError(f"로봇 {robot.id}: 하역하지 않은 토트가 남은 채 트립 완료 (t={self.now:.0f})")
        robot.is_idle      = True
        robot.idle_slots   = self.robot_capacity
        robot.route        = []
        robot.drops        = []
        robot.current_dest = None
        self._total_trips    += 1
        self._totes_in_trips += self._trip_totes[robot.id]
        self._trip_totes[robot.id] = 0

    def _schedule_next(self, robot: Robot):
        """현재 목적지로 출발. 실제 이동과 거리 집계는 매 틱 (_grid_tick / _reserve_tick)

        입력:
            robot: 이동할 로봇
        """
        if robot.current_dest is None:
            return
        if self.reserve:
            # 경로가 바뀐 경우에만 트립 전체 재계획
            if robot.route_version != robot.plan_version:
                self._reserve_plan(robot)
        else:
            # 목적지가 바뀌었으므로 이전 우회 경로와 주차 칸 예약은 폐기
            robot.detour = []
            self._grid_release_spot(robot)
        if robot.pos == robot.current_dest:
            self._grid_on_arrive(robot)

    def _dispatch(self, robot: Robot, totes: List[Tote]):
        """대기 큐에서 꺼낸 토트들로 로봇의 새 트립 시작

        입력:
            robot: idle 로봇
            totes: 배정할 토트 리스트
        """
        self._robot_totes[robot.id] = list(totes)
        for t in totes:
            self.order_mgr.record_assign(t, self.now)
        robot.assign_trip([t.loc for t in totes], self._needed_docks(robot))
        self._schedule_next(robot)

    def distance_per_completed_order(self) -> float:
        """완료 주문 1건당 총 이동거리 (m). 완료 주문이 없으면 0.0."""
        n = self.order_mgr.completed_count()
        return self._total_sim_distance / n if n > 0 else 0.0

    def avg_totes_per_trip(self) -> float:
        """완료된 트립당 평균 하역 토트 수"""
        return self._totes_in_trips / max(1, self._total_trips)

    def saving_per_insertion(self) -> float:
        """mid-trip 삽입 1건당 절감 이동거리 (m) = 토트 1개 단독 왕복거리 − 삽입당 추가 이동거리

        출력:
            절감 이동거리, 삽입이 없으면 0.0
        """
        if self._mid_trip_count == 0:
            return 0.0
        extra_per = self._extra_travel / self._mid_trip_count
        return self.layout.solo_rt_m - extra_per

    def build_result(self, label: str, episode: int, lam: float) -> EpisodeResult:
        """현재 에피소드 지표를 EpisodeResult로 묶음

        입력:
            label: 정책 라벨
            episode: 에피소드 번호
            lam: 이번 에피소드의 주문 도착률
        출력:
            EpisodeResult
        """
        om = self.order_mgr
        return EpisodeResult(
            label                = label,
            episode              = episode,
            lam                  = lam,
            num_robots           = self.num_robots,
            n_completed          = om.completed_count(),
            n_arrived            = len(om.all_orders),
            avg_flow_time        = om.avg_flow_time(),
            avg_wait_time        = om.avg_wait_time(),
            dist_per_order       = self.distance_per_completed_order(),
            n_totes              = len(om.all_totes),
            n_totes_done         = om.totes_done_count(),
            total_distance       = self._total_sim_distance,
            avg_totes_per_trip   = self.avg_totes_per_trip(),
            mid_trip_rate        = self._mid_trip_count / max(1, om.totes_done_count()),
            extra_travel_m       = self._extra_travel,
            saving_per_insertion = self.saving_per_insertion(),
            blocked_time         = self._blocked_time,
            n_replans            = self._n_replans,
            n_plans              = self._n_plans,
            n_plan_fail          = self._n_plan_fail,
            plan_time_s          = self._plan_time,
            sim_time             = self.sim_time,
        )

    # 이동 엔진

    def _grid_wants_move(self, r: Robot) -> bool:
        """트립 중이고 목적지로 이동해야 하는 로봇인지"""
        return r.timer == 0.0 and r.current_dest is not None and r.pos != r.current_dest

    def _grid_can_park(self, r: Robot) -> bool:
        """idle 로봇이 예약한 주차 칸으로 가는 중인지"""
        return (r.is_idle and r.current_dest is None
                and r.park_spot is not None and r.pos != r.park_spot)

    def _grid_assign_spot(self, r: Robot):
        """트립을 마친 idle 로봇에 현재 워크스테이션의 빈 주차 칸 중 도킹 셀에서 가장 가까운 칸 예약

        빈 칸이 없으면 (WS_PARK_CELLS < M일 때만 가능) 도킹 셀에 그대로 머묾
        """
        for c in self.layout.spots_by_dist[r.workstation_id]:
            if c not in self._spots_taken:
                r.park_spot = c
                self._spots_taken.add(c)
                return

    def _grid_release_spot(self, r: Robot):
        """r의 주차 칸 예약 해제"""
        if r.park_spot is not None:
            self._spots_taken.discard(r.park_spot)
            r.park_spot = None

    def _grid_active(self) -> bool:
        """다음 틱에 진행할 일이 있는지 (이동, 피킹/처리 타이머, 주차 칸으로 이동, 재계획 대기)"""
        if self.reserve:
            return any(r.plan or r.timer > 0.0 or r.replan_pending for r in self.robots)
        return any(r.timer > 0.0 or self._grid_wants_move(r) or self._grid_can_park(r)
                   for r in self.robots)

    def _grid_free(self, cell) -> bool:
        return (not self.layout.collisions) or cell not in self._occ

    def _grid_next_cell(self, r: Robot):
        """이번 틱에 r이 들어갈 셀. 막혀 있으면 None.

        최단경로상 다음 셀 후보 중 비어 있는 것을 선택
        GRID_REPLAN_WAIT 틱 이상 막혔으면 점유 셀을 피하는 우회 경로 탐색 (최단 대비 GRID_DETOUR_MAX_EXTRA칸 이내)
        """
        L = self.layout
        goal = r.current_dest if r.current_dest is not None else r.park_spot
        if r.detour:
            if self._grid_free(r.detour[0]):
                return r.detour.pop(0)
            r.detour = []
        hops = L.next_hops(r.pos, goal)
        for c in hops:
            if self._grid_free(c):
                return c
        if r.wait_ticks + 1 >= GRID_REPLAN_WAIT and goal not in hops:
            base = L.aisle_distance(r.pos, goal) / CELL_SIZE
            path = L.detour_path(r.pos, goal, set(self._occ),
                                 max_len=int(base) + GRID_DETOUR_MAX_EXTRA)
            if path:
                self._n_replans += 1
                r.detour = path
                return r.detour.pop(0)
        return None

    def _grid_tick(self):
        """한 틱 진행: 타이머 감소 -> 이동 -> 도착 처리 -> 교착 감시"""
        dt = self.layout.tick
        done = self._grid_run_timers()
        movers  = [r for r in self.robots
                   if r.id not in done and (self._grid_wants_move(r) or self._grid_can_park(r))]
        moved   = set()
        changed = True
        # 앞 로봇이 비운 셀로 뒤 로봇이 같은 틱에 따라 들어갈 수 있도록 변화가 없을 때까지 반복
        while changed:
            changed = False
            for r in movers:
                if r.id in moved:
                    continue
                nxt = self._grid_next_cell(r)
                if nxt is None:
                    continue
                if self.layout.collisions:
                    del self._occ[r.pos]
                    self._occ[nxt] = r.id
                r.pos = nxt
                self._total_sim_distance += CELL_SIZE
                moved.add(r.id)
                changed = True

        for r in movers:
            if r.id in moved:
                r.wait_ticks = 0
            else:
                r.wait_ticks += 1
                if r.current_dest is not None:          # 주차하러 가는 idle 로봇의 대기는 미집계
                    self._blocked_time += dt
        for r in movers:
            if r.id in moved and r.current_dest is not None and r.pos == r.current_dest:
                self._grid_on_arrive(r)

        if moved:
            self._last_move_time = self.now
        elif (self.now - self._last_move_time > GRID_DEADLOCK_TICKS * dt
              and not any(r.timer > 0.0 for r in self.robots)
              and any(self._grid_wants_move(r) for r in self.robots)):
            stuck = [(r.id, r.pos, r.current_dest) for r in self.robots if self._grid_wants_move(r)]
            raise RuntimeError(f"그리드 교착: t={self.now:.0f}s, {GRID_DEADLOCK_TICKS}틱 동안 이동 없음. "
                               f"(로봇, 위치, 목적지) = {stuck}")

    def _grid_run_timers(self) -> set:
        """피킹/워크스테이션 처리 타이머를 한 틱 줄이고, 끝난 로봇의 다음 단계 처리

        출력:
            이번 틱에 타이머가 끝난 로봇 id 집합 (이번 틱에는 이동 안 함)
        """
        dt   = self.layout.tick
        done = set()
        for r in self.robots:
            if r.timer > 0.0:
                r.timer -= dt
                if r.timer <= 1e-9:
                    r.timer, kind = 0.0, r.timer_kind
                    r.timer_kind = None
                    done.add(r.id)
                    if kind == "ws":
                        self._grid_finish_drop(r)
                    self._grid_after_node(r)
        return done

    def _grid_next_pick(self, r: Robot, loc, skip: int = 0) -> Optional[Tote]:
        """r이 loc에서 아직 집지 않은 토트 중 배정 순서로 (skip+1)번째"""
        for t in self._robot_totes[r.id]:
            if not t.picked and t.loc == loc:
                if skip == 0:
                    return t
                skip -= 1
        return None

    def _grid_ticks(self, seconds: float) -> int:
        """머무는 시간(s)을 틱 수로 올림"""
        return int(math.ceil(seconds / self.layout.tick - 1e-9)) if seconds > 0.0 else 0

    def _grid_on_arrive(self, r: Robot):
        """r이 current_dest 셀에 들어섰을 때

        - 피킹 셀: 아직 집지 않은 토트 하나를 집고 피킹 시간 시작
        - 도킹 셀: 이 WS로 갈 토트 중 이미 집은 것을 모두 내리고 처리 시간 합만큼 머묾
        그 밖에는 바로 다음 단계로
        """
        L = self.layout
        if L.is_dock(r.pos):
            k    = L.dock_ws[r.pos]
            drop = [t for t in self._robot_totes[r.id] if t.picked and t.ws_id == k]
            if drop:
                self._robot_totes[r.id] = [t for t in self._robot_totes[r.id] if t not in drop]
                self._dropping[r.id] = drop
                proc = sum(t.proc_s for t in drop)
                if proc > 0.0:
                    r.timer, r.timer_kind = proc, "ws"
                    return
                self._grid_finish_drop(r)
        else:
            t = self._grid_next_pick(r, r.pos)
            if t is not None:
                t.picked = True
                if t.pick_s > 0.0:
                    r.timer, r.timer_kind = t.pick_s, "pick"
                    return
        self._grid_after_node(r)

    def _grid_finish_drop(self, r: Robot):
        """도킹 셀에서 처리를 마친 토트 완료 기록 (주문의 마지막 토트면 주문 완료)"""
        self._trip_totes[r.id] += len(self._dropping[r.id])
        for t in self._dropping[r.id]:
            self._ws_load[t.ws_id] -= 1
            if self.order_mgr.record_tote_done(t, self.now):
                o = t.order
                self._n_in_system -= 1
                self._completions.append((self.now, o.completion_time - o.arrival_time))
        self._dropping[r.id] = []

    def _grid_after_node(self, r: Robot):
        """방문 지점 처리를 마친 뒤 다음 목적지로 보내거나, 남은 지점이 없으면 트립 완료"""
        if r.arrive_at_node():
            self._schedule_next(r)
            return
        if self.reserve and r.pending_insert:
            # 삽입 계획에 실패한 채 마지막 도킹 셀에 오면 하역 후 대기 토트를 경로에 넣어 재출발
            locs, r.pending_insert = r.pending_insert, []
            r.reroute(locs, self._needed_docks(r))
            r._start_move()
            self._reserve_plan(r)
            return
        if self._robot_totes[r.id]:
            raise RuntimeError(f"로봇 {r.id}: 방문 지점이 끝났는데 하역하지 않은 토트가 남음 (t={self.now:.0f})")
        self._grid_finish_trip(r)

    def _grid_finish_trip(self, r: Robot):
        """마지막 도킹 셀 처리 후 트립 완료 (대기 토트가 있으면 바로 새 트립)

        새 트립이 없으면 idle 로봇은 주차 칸으로 이동
        """
        if r.pending_insert:
            raise RuntimeError(f"로봇 {r.id}: 삽입 대기 토트 {r.pending_insert}이 남은 채 트립 완료 (t={self.now:.0f})")
        self._on_trip_complete(r)
        if r.is_idle and r.current_dest is None and not self.reserve:
            self._grid_assign_spot(r)

    # 예약 모드 (motion="reserve")

    def _reserve_tick(self, k: int):
        """예약 모드 한 틱: 타이머 -> 예약대로 이동 -> 도착 처리 -> 재계획 대기 로봇 재시도"""
        dt = self.layout.tick
        self._grid_run_timers()

        moved = []
        for r in self.robots:
            if r.plan and r.plan[0][0] == k:
                _, cell = r.plan.popleft()
                if cell != r.pos:
                    if r.timer > 0.0:
                        raise RuntimeError(f"예약 불일치: 로봇 {r.id}가 {r.timer_kind} 중인데 {r.pos}->{cell} 이동 계획 (t={self.now:.0f})")
                    r.pos = cell
                    self._total_sim_distance += CELL_SIZE
                    moved.append(r)
                    continue
            elif r.plan and r.plan[0][0] < k:
                raise RuntimeError(f"예약 불일치: 로봇 {r.id}의 계획 틱 {r.plan[0][0]} < 현재 틱 {k}")
            if r.current_dest is not None and r.timer == 0.0 and r.pos != r.current_dest:
                self._blocked_time += dt                # 예약 때문에 제자리에서 기다린 시간
        for r in moved:
            if r.current_dest is not None and r.pos == r.current_dest:
                self._grid_on_arrive(r)
        # 이 틱의 위치가 확정된 뒤 재시도 (계획 시작 틱 = k)
        for r in self.robots:
            if r.pending_insert:
                self._reserve_insert(r, r.pending_insert)
            elif r.replan_pending:
                self._reserve_plan(r)
        if k % 600 == 0:
            self._res.prune(k - 1)

    def _reserve_legs(self, r: Robot) -> Tuple[int, List[Tuple[Cell, int]]]:
        """r의 남은 방문 지점을 (셀, 머무는 틱) 목록으로 변환

        - 피킹 셀: 그 셀에서 집을 토트의 피킹 시간
        - 도킹 셀: 그 시점까지 집은 토트 중 그 WS로 갈 것의 처리 시간 합 (_grid_on_arrive와 같은 규칙)
        - 같은 피킹 셀을 연달아 방문하면 한 번 들러 피킹 시간 합만큼 머묾
        출력:
            (지금 위치에서 먼저 머물 틱 (피킹/처리 중이면 남은 시간), legs)
        """
        L = self.layout
        stay0  = self._grid_ticks(r.timer) if r.timer > 0.0 else 0
        visits = ([r.current_dest] if r.current_dest is not None and r.timer == 0.0 else []) + r.stops
        held   = [t for t in self._robot_totes[r.id] if t.picked]
        seen: Dict[Cell, int] = {}
        legs: List[Tuple[Cell, int]] = []
        for c in visits:
            if L.is_dock(c):
                k = L.dock_ws[c]
                d = self._grid_ticks(sum(t.proc_s for t in held if t.ws_id == k))
                held = [t for t in held if t.ws_id != k]
                legs.append((c, d))
                continue
            t = self._grid_next_pick(r, c, seen.get(c, 0))
            seen[c] = seen.get(c, 0) + 1
            d = 0
            if t is not None:
                d = self._grid_ticks(t.pick_s)
                held.append(t)
            if legs and legs[-1][0] == c:
                legs[-1] = (c, legs[-1][1] + d)
            elif not legs and r.timer_kind == "pick" and c == r.pos:
                stay0 += d
            else:
                legs.append((c, d))
        return stay0, legs

    def _reserve_plan(self, r: Robot) -> bool:
        """r의 남은 트립 전체를 시공간 경로로 계획해 예약표에 등록. 성공 여부 반환

        경로: (피킹/처리 중이면 남은 시간만큼 대기) -> current_dest -> 남은 피킹 셀 (NNH 순서)
              -> 도킹 셀들 (각각 하역 토트 처리 시간만큼 머묾) -> 마지막 WS의 빈 주차 칸 (무기한 머묾)
        계획 시작 틱(self._t0_ctx) 이후 예약은 지우고 다시 계획
        실패하면 지운 예약을 복원하고 replan_pending으로 표시해 다음 틱에 재시도
        """
        L, tab = self.layout, self._res
        t0 = self._t0_ctx
        wall = time.perf_counter()
        idx = L.cell_index
        removed, old_rest = tab.clear_after(r.id, t0)
        old_spot = r.park_spot
        self._spots_taken.discard(old_spot)

        stay0, legs = self._reserve_legs(r)
        cells, cur, ok = [], idx[r.pos], True
        for tt in range(t0 + 1, t0 + stay0 + 1):
            if not tab.can_enter(cur, cur, tt, r.id):
                ok = False
                break
            cells.append(cur)
        t = t0 + stay0
        if ok:
            for c, d in legs:
                res = plan_leg(tab, L.out_nbrs, L.dist_to(c), r.id, cur, t, idx[c], d, final=False)
                if res is None:
                    ok = False
                    break
                path, t = res
                cells += path
                cur = idx[c]
        spot = None
        if ok:
            ok = False
            for cand in L.spots_by_dist[r.workstation_id]:
                if cand in self._spots_taken:
                    continue
                res = plan_leg(tab, L.out_nbrs, L.dist_to(cand), r.id, cur, t, idx[cand], 0, final=True)
                if res is not None:
                    cells += res[0]
                    spot, ok = cand, True
                    break

        if ok:
            tab.commit(r.id, t0, idx[r.pos], cells, idx[spot])
            if self.recorder is not None:
                self.recorder.on_plan(self, r, t0, cells)
            r.plan = deque((t0 + 1 + i, L.cells[c]) for i, c in enumerate(cells))
            r.park_spot = spot
            self._spots_taken.add(spot)
            r.plan_version   = r.route_version
            r.replan_pending = False
            self._n_plans   += 1
        else:
            tab.restore(r.id, removed, old_rest)
            r.park_spot = old_spot
            if old_spot is not None:
                self._spots_taken.add(old_spot)
            r.replan_pending = True
            self._n_plan_fail += 1
        self._plan_time += time.perf_counter() - wall
        return ok

    def _reserve_insert(self, r: Robot, locs: List[Cell]) -> bool:
        """이동 중인 r의 경로에 locs를 삽입하고 재계획

        계획에 실패하면 경로 변경을 되돌리고 locs를 삽입 대기(pending_insert)로 둠
        r은 이전 계획대로 계속 움직이고 다음 틱에 재시도
        빈 슬롯 수는 배정 시점에 이미 줄여 두므로 되돌리지 않음
        """
        snap = (list(r.route), list(r.drops), r.workstation, r.workstation_id, r.route_version)
        r.reroute(r.route + list(locs), self._needed_docks(r))
        if self._reserve_plan(r):
            r.pending_insert = []
            return True
        r.route, r.drops, r.workstation, r.workstation_id, r.route_version = snap
        r.pending_insert = list(locs)
        r.replan_pending = True
        return False
