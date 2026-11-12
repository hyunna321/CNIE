"""창고 시뮬레이터 본체: 주문 도착 처리, 토트 배정, 트립 관리, 에피소드 지표, 계획 완료시각 추정

배정 정책은 SimulatorBase를 상속해 구현 (baselines.policies, env.rl.RLSimulator). 같은 엔진 위에서 정책만 다름
로봇 이동은 env.motion.MotionEngine
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from env.battery import Battery, BatteryConfig
from env.layout import Cell, GridWarehouseLayout, ROBOT_SPEED
from env.motion import MotionEngine, ReservationTable, to_ticks
from env.orders import Order, OrderManager, SIM_TIME, Tote
from env.robot import ROUTE_MODES, Robot, path_distance

COST_DIST   = 1.0    # order·s / m, BL6 한계비용에서 이동거리 1m를 흐름시간 몇 초로 환산할지 (J 지표에도 사용)
TOTE_ORDERS = ("arrival", "bottleneck_first")   # 주문 내 토트 배정 순서 (SimulatorBase tote_order, 기본 arrival)
# 계획 완료시각 추정 (아래 "계획 완료시각" 절)
# 확정된 계획의 완료시각 출처: "estimate" (NNH 경로 + 이동·피킹·처리 시간, 다른 로봇 예약 무시) /
#   "sipp" (reserve 모드에서 예약 경로의 도착 틱, 없거나 맞지 않으면 estimate). 후보 가상 삽입과
#   Delta_order는 배정 전후를 같은 기준으로 비교하려고 항상 estimate
PC_CONFIRMED = "estimate"
# 같은 도킹 셀에서 내리는 토트들의 완료시각: "batch" (모두 처리 합 끝 = 시뮬레이터 동작) /
#   "sequential" (j번째 토트 = j번째까지 처리 시간 합)
PC_DOCK      = "batch"


def nearest_rank(xs: List[float], q: float) -> float:
    """nearest-rank 분위수 (비어 있으면 0)"""
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(math.ceil(q * len(s))) - 1)]


def marginal_cost_terms(robot: Robot, delta_d: float, onboard: int,
                        cost_dist: float = COST_DIST) -> Tuple[float, float, float]:
    """BL6 한계비용 구성요소 (배정 후 남은 거리 m, 적재 토트 지연 tote·s, 한계비용 합계 tote·s)

    BL6 베이스라인(baselines.policies.bl6_marginal_cost)과 RL prior·특징(agents.DQNAgent._bl6_features)이 공유
    입력:
        robot: 후보 로봇
        delta_d: 삽입 추가 이동거리 (m, SimulatorBase._insertion_delta)
        onboard: 아직 하역하지 않은 토트 수
        cost_dist: 거리비용 c_d (order·s / m)
    """
    if robot.is_idle:
        remaining_after, delay_onboard = delta_d, 0.0
    else:
        remaining_after = robot.remaining_distance + delta_d
        delay_onboard   = onboard * delta_d / ROBOT_SPEED
    return remaining_after, delay_onboard, delay_onboard + remaining_after / ROBOT_SPEED + cost_dist * delta_d


# 에피소드 결과

@dataclass
class EpisodeResult:
    """에피소드 1회의 결과 지표 (주문 단위: n_completed, n_arrived, 흐름/대기 시간, 주문당 거리)"""
    label:                  str
    episode:                int
    lam:                    float
    num_robots:             int
    n_completed:            int
    n_arrived:              int
    avg_flow_time:          float         # s, 주문 도착 -> 마지막 토트 WS 처리 완료
    avg_wait_time:          float         # s, 주문 도착 -> 첫 토트 로봇 배정
    dist_per_order:         float
    n_totes:                int   = 0     # 도착 주문의 토트 수
    n_totes_done:           int   = 0     # WS 처리를 마친 토트 수
    total_distance:         float = 0.0   # m, 총 이동거리
    avg_totes_per_trip:     float = 0.0   # 트립당 평균 처리 토트 수
    mid_trip_rate:          float = 0.0   # mid-trip 삽입 수 / 처리 토트 수
    extra_travel_m:         float = 0.0
    saving_per_insertion:   float = 0.0   # m, 삽입 1건당 절감 이동거리
    blocked_time:           float = 0.0   # robot·s, 다른 로봇에 막혀 기다린 시간
    n_replans:              int   = 0     # 우회 경로 재계획 횟수 (reactive 모드)
    n_plans:                int   = 0     # 시공간 경로계획 횟수 (reserve 모드)
    n_plan_fail:            int   = 0     # 경로계획 실패 횟수 (reserve 모드)
    plan_time_s:            float = 0.0   # 경로계획에 쓴 실행 시간 (reserve 모드)
    sim_time:               float = SIM_TIME   # s, 시뮬레이션 시간
    reward:                 float = 0.0   # RL: 완료 주문 1건당 누적 보상 (shaping 제외)
    epsilon:                float = 0.0   # RL: 에피소드 종료 시 ε
    seed:                   int   = -1    # 주문 생성 시드 (평가: 테스트/검증 시드)
    n_decisions:            int   = 0     # 도착 시 정책이 바로 배정한 토트 수 (RL: 배정 결정 수)
    bl6_agree:              float = float("nan")   # RL 평가: 선택 로봇이 BL6 argmin과 같은 결정 비율
    avg_flow_time_censored: float = 0.0   # s, 미완료 주문을 sim_time에서 자른 도착 주문 전체 평균
    avg_unfinished_age:     float = 0.0   # s, sim_time 시점 미완료 주문의 평균 경과시간
    n_queued_totes:         int   = 0     # 도착 시 후보가 없어 대기 큐로 간 토트 수 (나중에 결정 없이 FIFO 방출)
    avg_candidates:         float = 0.0   # RL: 결정당 평균 후보 로봇 수
    n_qpen_active:          int   = 0     # RL: 결정 시점 대기 큐가 비어 있지 않았던 결정 수 (legacy 큐 압박 항 활성)
    n_depletion_active:     int   = 0     # RL: 마지막 빈 idle 로봇을 쓴 결정 수 (legacy 소진 페널티 활성)
    # 실행 시간 (삽입형 정책과 RL만, 버퍼형은 0)
    decision_ms_mean:       float = 0.0   # ms, 결정 1건 선택 지연 (후보 탐색·detour·특징·추론, 기록기 제외)
    decision_ms_p95:        float = 0.0
    decision_full_ms_mean:  float = 0.0   # ms, 선택 + 배정·경로 재계획(예약표) 포함
    decision_full_ms_p95:   float = 0.0
    wall_s:                 float = 0.0   # s, 에피소드 1회 실행 시간 (학습이면 업데이트 포함)
    flow_p99:               float = 0.0   # s, 완료 주문 흐름시간 99백분위 (nearest-rank)
    utilization:            float = 0.0   # 가동률 = 운행 중(idle 아님) 로봇 시간 / (M · 시뮬레이션 시간)
    idle_at_decision:       float = 0.0   # 결정 가능한 토트 도착 시점(후보 ≥ 1)의 평균 idle 로봇 수
    # RL 평가 결정 분류 (viewer.recorder.DecisionStats, 그림 6 재분석)
    agree_cd0:              float = float("nan")   # 선택 = BL6(c_d = 0) argmin 비율
    agree_bl8:              float = float("nan")   # 선택 = BL8 argmin 비율
    cd_reproducible:        float = float("nan")   # 어떤 c_d ≥ 0의 BL6 argmin으로도 재현되는 결정 비율
    # 배터리 제약 (env.battery, 제약이 없으면 기본값)
    n_charge_required:      int   = 0              # 필수 충전 횟수
    charge_time:            float = 0.0            # robot·s, 실제 충전에 쓴 시간 (회복 주행거리 / 충전 속도)
    min_soc:                float = float("nan")   # 에피소드 최소 SoC 비율
    n_soc_depleted:         int   = 0              # SoC가 음수인 상태로 이동한 셀 수 (0이어야 함)
    n_battery_rejects:      int   = 0              # SoC 부족으로 후보에서 빠진 (토트, 로봇) 수

    @property
    def decision_rate(self) -> float:
        """도착 토트 중 정책이 바로 배정한 비율 (나머지는 대기 큐 경유)"""
        return self.n_decisions / max(1, self.n_totes)

    @property
    def completion_rate(self) -> float:
        """완료 주문 수 / 도착 주문 수"""
        return self.n_completed / max(1, self.n_arrived)

    @property
    def throughput(self) -> float:
        """초당 완료 주문 수 (orders/s)"""
        return self.n_completed / self.sim_time


# 사건

EVENT_ORDER_ARRIVAL = "ORDER_ARRIVAL"
EVENT_CHARGE_DONE   = "CHARGE_DONE"      # 필수 충전 완료 (env.battery)


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


# 시뮬레이터

class SimulatorBase(MotionEngine):
    """그리드 창고 시뮬레이터의 공통 부분

    - 로봇 풀은 모든 워크스테이션이 공유, 초기 위치만 라운드로빈으로 분산
    - 대기 토트는 단일 FIFO 큐(OrderManager.wait_queue)로 관리
    - 트립: 실은 토트의 피킹 위치를 NNH로 돈 뒤, 그 토트들의 WS 도킹 셀을 NNH로 돌며 하역
      (env.robot.plan_trip). 도킹 셀에서는 그 WS로 가는 토트의 처리 시간 합만큼 머묾

    하위 클래스는 _on_tote_arrival()로 배정 정책 정의, 필요하면
    _serve_queue(), _choose_ws() 재정의

    로봇 이동은 셀 단위 틱(CELL_SIZE / ROBOT_SPEED)으로 시뮬레이션 (run)
    - 배정 결정은 주문 도착 사건 시점, 결정 시점의 로봇 위치는 직전 틱 기준
    - 이동거리는 실제로 이동한 셀 수
    - 충돌 처리는 layout.motion으로 선택
      - "reactive": 매 틱 다음 셀이 비어 있으면 이동, 막히면 대기/우회
      - "reserve": 로봇의 경로가 바뀔 때(배정, 삽입, 재출발) 그 로봇의 트립 전체를
        시공간 경로로 계획해 예약표에 올리고, 매 틱 예약대로 이동 (_reserve_plan)

    평가 옵션 (라우팅 오차 분리용, 기본값이면 동작 변화 없음)
    - route: 경로 구성 규칙 (env.robot.ROUTE_MODES, env.robot 모듈 설명). _insertion_delta도 같은 규칙
    - battery: 배터리 제약 (env.battery 모듈 설명)
    """

    def __init__(self, layout: GridWarehouseLayout, lam: float, seed: int,
                 sim_time: float, num_robots: int, robot_capacity: int, route: str = "nnh",
                 tote_order: str = "arrival", battery: Optional[BatteryConfig] = None):
        """
        입력:
            layout: GridWarehouseLayout
            lam: 주문 도착률 (orders/s)
            seed: 주문 생성 난수 시드
            sim_time: 시뮬레이션 시간 (s)
            num_robots: 로봇 수
            robot_capacity: 로봇 용량 (토트)
            route: 경로 구성 규칙 ROUTE_MODES (기본 "nnh")
            tote_order: 주문 내 토트 배정 순서 TOTE_ORDERS. "arrival" (기본, 주문 생성 순서) /
                        "bottleneck_first" (도착 시 토트마다 min_m Ť_{k,m}을 구해 큰 것부터, tote_completion)
            battery: env.battery.BatteryConfig (None이면 배터리 제약 없음)
        """
        assert route in ROUTE_MODES, route
        assert tote_order in TOTE_ORDERS, tote_order
        self.route_mode       = route
        self.tote_order       = tote_order
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
        self.battery   = Battery(self, battery) if battery is not None else None
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
        self._lat_sel:  List[float] = []   # s, 결정별 선택 지연 (후보 탐색 ~ 로봇 선택)
        self._lat_full: List[float] = []   # s, 결정별 전체 지연 (선택 + 배정·경로 재계획)
        self._busy_time  = 0.0             # robot·s, 운행 중(idle 아님) 로봇 시간 적분 (가동률)
        self._idle_sum   = 0                # 결정 가능한 토트 도착 시점의 idle 로봇 수 합
        self._idle_n     = 0                # 그 도착 수

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
                                workstation_id=k, route_mode=self.route_mode))
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
        if self.battery is not None:
            self.battery.reset()
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
            if not self._motion_active():
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
                if ev.etype == EVENT_CHARGE_DONE:
                    self._handle_charge_done(ev)
                else:
                    self._handle_order_event(ev)
            else:
                self._advance_clock(t_tick)
                self._t0_ctx = k
                if self.reserve:
                    self._reserve_tick(k)
                else:
                    self._reactive_tick()
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
            for tote in self._tote_sequence(order):
                if self._eligible_robots():
                    self._idle_sum += sum(r.is_idle for r in self.robots)
                    self._idle_n   += 1
                self._on_tote_arrival(tote)
        nxt, n = self.order_mgr.next_arrival(self.now)
        if nxt <= self.sim_time:
            self.eq.push(Event(nxt, EVENT_ORDER_ARRIVAL, {"count": n}))

    def _push_charge_done(self, robot: Robot, t: float):
        """필수 충전 완료 사건 예약 (env.battery)"""
        self.eq.push(Event(t, EVENT_CHARGE_DONE, {"robot": robot.id}))

    def _handle_charge_done(self, ev: Event):
        """필수 충전 완료: 로봇을 후보로 되돌리고 대기 큐 방출"""
        robot = self.robots[ev.data["robot"]]
        if robot.must_charge:
            self.battery.finish_required(robot)
            self._serve_queue(robot)

    def _advance_clock(self, t: float):
        """시각을 t로 옮기며 [now, t] 구간의 상태를 _on_interval()에 전달

        구간 안에서는 시스템 내 주문 수와 운행 중 로봇 수가 불변 (틱과 주문 도착 사이에서만 바뀜)

        입력:
            t: 새 시각 (s)
        """
        if t > self.now:
            n_busy = sum(1 for r in self.robots if not r.is_idle)
            self._busy_time += n_busy * (t - self.now)
            self._on_interval(self.now, t, self._n_in_system, n_busy)
        self.now = t

    def _tote_sequence(self, order: Order) -> List[Tote]:
        """주문 도착 시 토트를 정책에 넘기는 순서 (tote_order, __init__ 설명). 후보가 없으면 생성 순서"""
        if self.tote_order == "arrival" or len(order.totes) < 2:
            return order.totes
        eligible = self._eligible_robots()
        if not eligible:
            return order.totes
        key = {t.tote_id: min(tote_completion(self, t, r) for r in eligible) for t in order.totes}
        return sorted(order.totes, key=lambda t: -key[t.tote_id])

    def _on_interval(self, t0: float, t1: float, n_in_system: int, n_busy: int):
        """사건 사이 구간 [t0, t1] 훅 (RLSimulator가 보상 객체에 전달, 구간 적분형 보상용). 기본 구현은 no-op

        입력:
            t0, t1: 구간 시작/끝 시각 (s)
            n_in_system: 구간 동안 시스템 내 주문 수 (도착 − 완료)
            n_busy: 구간 동안 운행 중인 로봇 수
        """

    def _choose_ws(self, order: Order) -> int:
        """주문을 처리할 워크스테이션: 처리 전 토트가 가장 적은 WS (동률이면 번호가 작은 WS)"""
        return min(range(self.num_workstations), key=lambda k: (self._ws_load[k], k))

    def _on_tote_arrival(self, tote: Tote):
        """새 토트의 배정 정책. 하위 클래스에서 구현"""
        raise NotImplementedError

    def _on_trip_complete(self, robot: Robot):
        """트립 완료 처리 후 대기 큐 방출

        입력:
            robot: 트립을 마친 로봇
        """
        self._complete_trip(robot)
        self._serve_queue(robot)

    def _take_from_queue(self, robot: Robot) -> List[Tote]:
        """idle 로봇이 맡을 대기 토트를 FIFO로 꺼냄 (최대 용량, 배터리 제약이 있으면 SoC로 맡을 수 있는 만큼)"""
        n = self.robot_capacity if self.battery is None else self.battery.take_count(robot)
        return self.order_mgr.dequeue_up_to(n)

    def _serve_queue(self, robot: Robot):
        """idle 로봇에 대기 토트가 있으면 새 트립 시작 (트립 완료, 필수 충전 완료 시)"""
        totes = self._take_from_queue(robot)
        if totes:
            self._dispatch(robot, totes)

    def _enqueue_tote(self, tote: Tote):
        """배정할 로봇이 없는 토트를 대기 큐에 추가"""
        self.order_mgr.enqueue(tote, self.now)
        if self.battery is not None:
            self.battery.on_enqueue()

    def _eligible_robots(self, tote: Optional[Tote] = None) -> List[Robot]:
        """빈 슬롯이 있는 로봇 목록 (mid-trip 로봇 먼저, 그다음 idle 로봇)

        입력:
            tote: 배정할 토트. 배터리 제약이 있으면 이 토트를 맡을 수 있는 로봇만 남김 (None이면 필수 충전만 제외)
        출력:
            Robot 리스트
        """
        # 워크스테이션 처리 중인 로봇은 도킹 셀에 묶여 있으므로 삽입 후보에서 제외
        busy     = [r for r in self.robots
                    if not r.is_idle and r.idle_slots > 0 and r.timer_kind != "ws"]
        idle_any = [r for r in self.robots if r.is_idle and r.idle_slots > 0]
        if self.battery is not None:
            return self.battery.filter(busy + idle_any, tote)
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
        (route가 nnh가 아니면 Robot._plan의 같은 규칙, env.robot 모듈 설명)
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

    def _route_after(self, robot: Robot, tote: Optional[Tote]) -> Tuple[float, Cell]:
        """tote를 더 맡았을 때 남은 경로의 (이동거리 m, 마지막 방문 셀). 삽입 대기 토트 포함 (env.battery)
        tote가 None이면 지금 남은 경로. 남은 방문 지점이 없으면 (0, 현재 위치)"""
        visits, _ = _visits(self, robot, tote)
        if not visits:
            return 0.0, robot.pos
        return path_distance(robot.pos, visits, self.layout.distance), visits[-1]

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
        else:
            robot.idle_slots = max(0, robot.idle_slots - 1)
            if self.reserve:
                self._reserve_insert(robot, robot.pending_insert + [tote.loc])
            else:
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
        """현재 목적지로 출발. 실제 이동과 거리 집계는 매 틱 (_reactive_tick / _reserve_tick)

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
            self._release_spot(robot)
        if robot.pos == robot.current_dest:
            self._on_arrive(robot)

    def _dispatch(self, robot: Robot, totes: List[Tote]):
        """대기 큐에서 꺼낸 토트들로 로봇의 새 트립 시작

        입력:
            robot: idle 로봇
            totes: 배정할 토트 리스트
        """
        self._load_totes(robot, totes)
        robot.assign_trip([t.loc for t in totes], self._needed_docks(robot))
        self._schedule_next(robot)

    def _load_totes(self, robot: Robot, totes: List[Tote]):
        """대기 큐에서 꺼낸 토트들을 로봇에 싣고 배정 시각 기록"""
        self._robot_totes[robot.id] = list(totes)
        for t in totes:
            self.order_mgr.record_assign(t, self.now)

    def _record_latency(self, t0: float, t_sel: float, t_end: float):
        """결정 1건의 실행 시간 기록 (time.perf_counter 값)

        입력:
            t0: 결정 시작 (후보 탐색 전)
            t_sel: 로봇 선택 완료
            t_end: 배정과 경로 재계획 완료
        """
        self._lat_sel.append(t_sel - t0)
        self._lat_full.append(t_end - t0)

    @staticmethod
    def _ms_stats(xs: List[float]) -> Tuple[float, float]:
        """초 단위 목록의 (평균, 95백분위) ms, 비어 있으면 (0, 0)"""
        if not xs:
            return 0.0, 0.0
        return 1e3 * sum(sorted(xs)) / len(xs), 1e3 * nearest_rank(xs, 0.95)

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
        sel_mean, sel_p95   = self._ms_stats(self._lat_sel)
        full_mean, full_p95 = self._ms_stats(self._lat_full)
        res = EpisodeResult(
            label                = label,
            episode              = episode,
            lam                  = lam,
            num_robots           = self.num_robots,
            n_completed          = om.completed_count(),
            n_arrived            = len(om.all_orders),
            avg_flow_time        = om.avg_flow_time(),
            avg_flow_time_censored = om.censored_flow_time(self.sim_time),
            avg_unfinished_age   = om.avg_unfinished_age(self.sim_time),
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
            n_queued_totes       = om.n_enqueued,
            n_decisions          = len(om.all_totes) - om.n_enqueued,
            decision_ms_mean     = sel_mean,
            decision_ms_p95      = sel_p95,
            decision_full_ms_mean = full_mean,
            decision_full_ms_p95 = full_p95,
            flow_p99             = nearest_rank([f for _, f in self._completions], 0.99),
            utilization          = self._busy_time / max(1e-9, self.num_robots * self.now),
            idle_at_decision     = self._idle_sum / self._idle_n if self._idle_n else 0.0,
        )
        if self.battery is not None:
            self.battery.fill_result(res)
        return res


# 계획 완료시각

# 계획 완료시각 추정기와 주문 단위 계획 비용 Delta_order
# 
# BL8, RL 주문 특징, tote_order = bottleneck_first, little_pbrs 보상의 Φ가 모두 이 모듈의 같은 함수를 씀
# 
# 토트 j의 계획 처리 완료시각 Ť_j(s) (로봇의 남은 트립을 따라 걷기, robot_timeline)
#   t_base = 결정 시점 로봇 상태의 틱 시각 (sim._t0_ctx · tick, 경로 계획 시작 기준과 같음)
#   + 진행 중인 피킹/WS 처리의 남은 틱
#   + 방문 지점마다 이동시간 d / v
#   + 피킹 셀: 그 셀에서 집는 토트마다 피킹 시간 (틱 올림, 시뮬레이터와 같은 규칙)
#   + 도킹 셀: 도착 시점의 WS 잔여 backlog 대기 max(0, backlog_w − (도착 − t_base))
#              + 그 WS로 내리는 토트의 처리 시간 (PC_DOCK: batch = 처리 합 끝에 모두 완료 (시뮬레이터 동작),
#                sequential = j번째 토트는 j번째까지의 처리 합)
#   backlog_w (ws_backlog) = w 도킹 셀에서 처리 중인 다른 로봇의 남은 처리 틱
#                            + w 대기열 셀에서 w 도킹을 기다리는 다른 로봇이 내릴 토트의 처리 틱
#   다른 로봇의 예약·혼잡은 무시 (후보 평가의 Δd와 같은 가정)
# 확정된 계획 (planned_completions): PC_CONFIRMED = "estimate"이면 위 추정, "sipp"이면 reserve 모드 예약 경로의
#   도착 틱 (sipp_timeline, 맞는 경로가 없으면 추정). 가상 삽입과 Delta_order는 항상 추정
# 주문 계획 완료시각 Ť_o(s) = max(t, max_{j∈o, 배정·미처리} Ť_j(s)), 미배정 토트(대기 큐 포함)는 t
# Delta_order(k, m) = Σ_o [Ť_o(배정 후) − Ť_o(배정 전)]: 후보 m에 토트 k를 가상 삽입(NNH 재구성, 실제 배정과 같은
#   경로 규칙)한 결과. 영향받는 주문 = k의 주문과 m에 실린 토트들의 주문
# 모든 함수는 시뮬레이터 상태를 바꾸지 않음

def ws_backlog(sim, exclude_id: Optional[int] = None) -> List[float]:
    """WS별 잔여 backlog (s): 도킹 중인 로봇의 남은 처리 + 대기열 셀에서 그 도킹을 기다리는 로봇의 처리 (exclude_id 제외)"""
    L, tick = sim.layout, sim.layout.tick
    b = [0.0] * sim.num_workstations
    qmap = L.queue_ws
    for r in sim.robots:
        if r.id == exclude_id:
            continue
        if r.timer_kind == "ws" and r.pos in L.dock_ws:
            b[L.dock_ws[r.pos]] += to_ticks(r.timer, tick) * tick
        elif r.pos in qmap and r.current_dest == L.workstations[qmap[r.pos]]:
            k = qmap[r.pos]
            b[k] += to_ticks(sum(t.proc_s for t in sim._robot_totes[r.id] if t.picked and t.ws_id == k), tick) * tick
    return b


def _visits(sim, r, extra=None) -> Tuple[List, List]:
    """r의 남은 방문 지점과 그 트립의 미하역 토트 (배정 순서). extra가 있으면 실제 배정과 같은 규칙으로 가상 삽입

    삽입 실패로 미뤄 둔 토트 위치(pending_insert)는 다음 재계획처럼 경로에 넣어 계산
    """
    totes = list(sim._robot_totes[r.id]) + ([extra] if extra is not None else [])
    if extra is None and not r.pending_insert:
        head = [r.current_dest] if r.current_dest is not None and r.timer == 0.0 else []
        return head + r.stops, totes
    docks = sim._needed_docks(r)
    if extra is not None:
        dock = sim.layout.workstations[extra.ws_id]
        if dock not in docks:
            docks.append(dock)
    new = [extra.loc] if extra is not None else []
    if r.is_idle:
        picks, drops, _ = r._plan(r.pos, [t.loc for t in totes if not t.picked], docks, keep_docks=False)
        return picks + drops, totes
    start = r.current_dest if r.current_dest is not None else r.pos
    picks, drops, _ = r._plan(start, r.route + list(r.pending_insert) + new, docks, keep_docks=True)
    head = [r.current_dest] if r.current_dest is not None and r.timer == 0.0 else []
    return head + picks + drops, totes


def robot_timeline(sim, r, extra=None, backlog: Optional[List[float]] = None) -> Dict[object, float]:
    """r에 배정된 미처리 토트(+ 가상 삽입 extra)의 계획 처리 완료시각 {토트: Ť (s)} (모듈 설명의 추정)"""
    L, tick = sim.layout, sim.layout.tick
    t_base = sim._t0_ctx * tick
    backlog = backlog if backlog is not None else ws_backlog(sim, r.id)
    visits, totes = _visits(sim, r, extra)
    done: Dict[object, float] = {}
    t = t_base
    if r.timer > 0.0:
        t += to_ticks(r.timer, tick) * tick
        if r.timer_kind == "ws":
            for x in sim._dropping[r.id]:
                done[x] = t
    held = [x for x in totes if x.picked]
    left = [x for x in totes if not x.picked]
    p = r.pos
    for c in visits:
        t += L.distance(p, c) / ROBOT_SPEED
        p = c
        if L.is_dock(c):
            k = L.dock_ws[c]
            drop = [x for x in held if x.ws_id == k]
            if not drop:
                continue
            t = max(t, t_base + backlog[k])
            if PC_DOCK == "sequential":
                acc = 0.0
                for x in drop:
                    acc += x.proc_s
                    done[x] = t + to_ticks(acc, tick) * tick
            t += to_ticks(sum(x.proc_s for x in drop), tick) * tick
            if PC_DOCK != "sequential":
                for x in drop:
                    done[x] = t
            held = [x for x in held if x.ws_id != k]
        else:
            x = next((y for y in left if y.loc == c), None)
            if x is not None:
                left.remove(x)
                held.append(x)
                t += to_ticks(x.pick_s, tick) * tick
    for x in totes:                      # 경로에 없는 토트 (있으면 안 됨): 트립 끝 시각
        done.setdefault(x, t)
    return done


def sipp_timeline(sim, r) -> Optional[Dict[object, float]]:
    """reserve 모드의 확정 예약 경로에서 읽은 완료시각 (예약이 지금 경로와 맞지 않으면 None)"""
    if not sim.reserve or not r.plan or r.plan_version != r.route_version or r.pending_insert:
        return None
    L, tick = sim.layout, sim.layout.tick
    stay0, legs = sim._reserve_legs(r)
    t0 = sim._t0_ctx
    done: Dict[object, float] = {}
    if r.timer > 0.0 and r.timer_kind == "ws":
        for x in sim._dropping[r.id]:
            done[x] = (t0 + stay0) * tick
    plan = list(r.plan)
    held = [x for x in sim._robot_totes[r.id] if x.picked]
    left = [x for x in sim._robot_totes[r.id] if not x.picked]
    i = 0
    for c, d in legs:
        while i < len(plan) and not (plan[i][1] == c and all(plan[j][1] == c for j in range(i + 1, min(len(plan), i + 1 + d)))):
            i += 1
        if i >= len(plan):
            return None
        arrive = plan[i][0]
        if L.is_dock(c):
            k = L.dock_ws[c]
            for x in [x for x in held if x.ws_id == k]:
                done[x] = (arrive + d) * tick
            held = [x for x in held if x.ws_id != k]
        else:
            x = next((y for y in left if y.loc == c), None)
            if x is not None:
                left.remove(x)
                held.append(x)
        i += 1 + d if d else 1
    if len(done) < len(sim._robot_totes[r.id]) + (len(sim._dropping[r.id]) if r.timer_kind == "ws" else 0):
        return None
    return done


def planned_completions(sim, mode: str = None) -> Dict[object, float]:
    """모든 로봇의 배정·미처리 토트 계획 완료시각 {토트: Ť}. mode: "estimate" / "sipp" (기본 PC_CONFIRMED)"""
    mode = mode or PC_CONFIRMED
    out: Dict[object, float] = {}
    for r in sim.robots:
        tl = sipp_timeline(sim, r) if mode == "sipp" else None
        out.update(tl if tl is not None else robot_timeline(sim, r))
    return out


def order_completion(order, pc: Dict[object, float], t: float) -> float:
    """주문 계획 완료시각 Ť_o = max(t, 배정·미처리 토트의 Ť)"""
    return max([t] + [pc[x] for x in order.totes if x in pc])


def tote_completion(sim, tote, r) -> float:
    """토트 tote를 로봇 r에 가상 삽입했을 때 그 토트의 계획 완료시각 Ť_{k,m}"""
    return robot_timeline(sim, r, extra=tote)[tote]


def delta_order(sim, tote, r, pc: Optional[Dict[object, float]] = None,
                detail: Optional[dict] = None) -> Tuple[float, float, int]:
    """토트 tote를 후보 r에 가상 삽입했을 때 (Delta_order, Ť_{k,m}, 계획 완료가 늦어지는 다른 주문 수)

    pc: 배정 전 계획 완료시각 (planned_completions(sim, "estimate"), 후보마다 다시 계산하지 않도록 넘김)
    detail: dict를 넘기면 {order_id: 그 주문의 Ť_o 증가}를 채움
    """
    if pc is None:
        pc = planned_completions(sim, "estimate")
    t = sim.now
    new = robot_timeline(sim, r, extra=tote)
    after = dict(pc)
    for x in sim._robot_totes[r.id]:
        after.pop(x, None)
    after.update(new)
    total, n_late = 0.0, 0
    for o in {tote.order} | {x.order for x in new}:
        d = order_completion(o, after, t) - order_completion(o, pc, t)
        total += d
        if detail is not None:
            detail[o.order_id] = d
        if o is not tote.order and d > 1e-9:
            n_late += 1
    return total, new[tote], n_late
