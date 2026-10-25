"""로봇 상태와 트립 경로 구성 (피킹 위치 NNH 순서 -> 도킹 셀 NNH 순서)

좌표는 grid.GridWarehouseLayout의 셀 (col, row), 거리는 layout.aisle_distance (유향 셀 그래프 최단거리, m)

트립 = 피킹 단계 (route) + 하역 단계 (drops)
  - 실은 토트마다 가야 할 워크스테이션이 주문 도착 시 정해져 있으므로 하역할 도킹 셀은 정책이 아니라
    실은 토트가 결정 (simulator._needed_docks)
  - NNH는 피킹 위치 방문 순서와, 마지막 피킹 위치에서 출발하는 도킹 셀 방문 순서만 정함
"""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Callable, List, Tuple

from constant import ROBOT_SPEED

if TYPE_CHECKING:
    from grid import GridWarehouseLayout

Cell = Tuple[int, int]


def nearest_neighbor_route(
    start:        Cell,
    destinations: List[Cell],
    dist_func:    Callable[[Cell, Cell], float],
) -> List[Cell]:
    """최근접 이웃(NNH) 방식의 방문 순서 결정

    입력:
        start: 출발 좌표
        destinations: 방문할 좌표 리스트
        dist_func: 두 좌표 간 거리 함수
    출력:
        방문 순서대로 정렬된 좌표 리스트 (start 미포함)
    """
    remaining = list(destinations)
    route, current = [], start
    while remaining:
        nearest = min(remaining, key=lambda d: dist_func(current, d))
        route.append(nearest)
        remaining.remove(nearest)
        current = nearest
    return route


def path_distance(start: Cell, stops: List[Cell],
                  dist_func: Callable[[Cell, Cell], float]) -> float:
    """start -> stops 순서대로 방문하는 총 이동거리"""
    total, current = 0.0, start
    for p in stops:
        total  += dist_func(current, p)
        current = p
    return total


def plan_trip(start: Cell, picks: List[Cell], docks: List[Cell],
              dist_func: Callable[[Cell, Cell], float]) -> Tuple[List[Cell], List[Cell], float]:
    """피킹 위치를 NNH로 방문한 뒤 도킹 셀을 NNH로 방문하는 경로

    입력:
        start: 출발 좌표
        picks: 방문할 피킹 셀 (중복 가능)
        docks: 방문할 도킹 셀 (중복 없음)
        dist_func: 두 좌표 간 거리 함수
    출력:
        (피킹 순서, 도킹 순서, start부터 마지막 도킹 셀까지 총 이동거리)
    """
    pick_route = nearest_neighbor_route(start, picks, dist_func)
    last       = pick_route[-1] if pick_route else start
    dock_route = nearest_neighbor_route(last, docks, dist_func)
    return pick_route, dock_route, path_distance(start, pick_route + dock_route, dist_func)


class Robot:
    """피킹 로봇의 위치, 남은 경로, 적재 슬롯 상태 관리

    route: 남은 피킹 셀 (current_dest 제외), drops: 피킹 후 들를 도킹 셀
    workstation / workstation_id: 트립의 마지막 도킹 셀 (트립 후 그 WS 주차장에 주차)
    """

    def __init__(self, robot_id: int, workstation: Cell,
                 layout: GridWarehouseLayout, capacity: int,
                 workstation_id: int = 0):
        """
        입력:
            robot_id: 로봇 id
            workstation: 초기 워크스테이션 도킹 셀
            layout: GridWarehouseLayout
            capacity: 적재 가능한 토트 수
            workstation_id: 초기 워크스테이션 인덱스
        """
        self.id             = robot_id
        self.home_workstation    = workstation      # reset 시 되돌릴 초기 워크스테이션
        self.home_workstation_id = workstation_id
        self.layout          = layout
        self.capacity        = capacity
        self.reset()

    def reset(self):
        """로봇을 초기 워크스테이션에 대기 중인 빈 상태로 되돌림

        workstation은 트립마다 바뀌므로 초기값으로 복원 (안 하면 같은 시뮬레이터로 연속 실행한
        학습 에피소드가 직전 에피소드 끝의 WS 배정을 물려받음)
        """
        self.workstation       = self.home_workstation
        self.workstation_id    = self.home_workstation_id
        self.pos               = self.workstation
        self.route             = []
        self.drops             = []
        self.idle_slots        = self.capacity
        self.is_idle           = True
        self.current_dest      = None
        # 피킹/워크스테이션 처리 잔여 시간, 우회 경로(reactive), 연속 대기 틱, 주차 칸
        self.timer             = 0.0
        self.timer_kind        = None
        self.detour            = []
        self.wait_ticks        = 0
        self.park_spot         = None
        # reserve 모드: 경로가 바뀔 때마다 route_version이 오르고, 계획은 plan_version 기준
        self.route_version     = 0
        self.plan_version      = 0
        self.plan              = deque()     # (틱, 셀) 순서의 예약 경로
        self.replan_pending    = False
        self.pending_insert    = []          # 계획 실패로 아직 경로에 넣지 못한 삽입 토트 위치

    @property
    def stops(self) -> List[Cell]:
        """current_dest 이후 남은 방문 지점 (피킹 셀 -> 도킹 셀)"""
        return self.route + self.drops

    @property
    def remaining_distance(self) -> float:
        """현재 위치 -> current_dest -> 남은 피킹 셀 -> 도킹 셀까지 남은 이동거리 (m)"""
        if self.current_dest is None:
            return 0.0
        return self._md(self.pos, self.current_dest) + \
            path_distance(self.current_dest, self.stops, self._md)

    def trip_distance_with(self, picks: List[Cell], docks: List[Cell]) -> float:
        """현재 목적지는 유지하고 그 뒤를 picks, docks로 다시 짤 때의 남은 이동거리 (m)"""
        if self.current_dest is None:
            return plan_trip(self.pos, picks, docks, self._md)[2]
        return self._md(self.pos, self.current_dest) + \
            plan_trip(self.current_dest, picks, docks, self._md)[2]

    def arrive_at_node(self) -> bool:
        """현재 목적지 도착 처리 후 다음 목적지 꺼냄 (피킹 셀 먼저, 그다음 도킹 셀)

        출력:
            남은 방문 지점이 있어 다음 목적지가 설정되면 True, 없으면 False
        """
        self.pos = self.current_dest
        self.current_dest = None
        self._start_move()
        return self.current_dest is not None

    def assign_trip(self, picks: List[Cell], docks: List[Cell]):
        """현재 위치에서 출발하는 새 트립 배정 및 이동 시작

        입력:
            picks: 방문할 피킹 셀
            docks: 하역할 도킹 셀
        """
        self.route, self.drops, _ = plan_trip(self.pos, picks, docks, self._md)
        self._set_final_ws()
        self.idle_slots  = self.capacity - len(picks)
        self.is_idle     = False
        self.route_version += 1
        self._start_move()

    def reroute(self, picks: List[Cell], docks: List[Cell]):
        """이동 중인 트립의 남은 경로를 picks, docks로 다시 구성 (current_dest는 유지, 슬롯 수는 그대로)"""
        start = self.current_dest if self.current_dest is not None else self.pos
        self.route, self.drops, _ = plan_trip(start, picks, docks, self._md)
        self._set_final_ws()
        self.route_version += 1

    def _set_final_ws(self):
        """마지막 도킹 셀을 트립의 워크스테이션으로 설정"""
        last = self.drops[-1] if self.drops else (
            self.current_dest if self.current_dest is not None and self.layout.is_dock(self.current_dest) else None)
        if last is not None:
            self.workstation    = last
            self.workstation_id = self.layout.dock_ws[last]

    def _start_move(self):
        """남은 방문 지점의 첫 지점을 현재 목적지로 설정"""
        if self.route:
            self.current_dest = self.route.pop(0)
        elif self.drops:
            self.current_dest = self.drops.pop(0)

    def state_vector(self, max_time: float) -> List[float]:
        """RL 특징용 로봇 상태 (트립 마지막 워크스테이션 기준 상대좌표, layout.coord_norm으로 정규화)

        입력:
            max_time: 잔여 이동시간 정규화 기준 (s)
        출력:
            길이 5 + capacity*2 리스트
            [상대 x, 상대 y, 빈 슬롯 비율, idle 여부, 잔여 이동시간, 남은 피킹 셀(current_dest 제외) 상대좌표 (0 패딩)...]
        """
        wx, wy = self.workstation
        norm   = self.layout.coord_norm
        vec = [
            (self.pos[0] - wx) / norm,
            (self.pos[1] - wy) / norm,
            self.idle_slots / self.capacity,
            float(self.is_idle),
            min(self.remaining_distance / ROBOT_SPEED / max_time, 1.0),
        ]
        for p in self.route[:self.capacity]:
            vec += [(p[0] - wx) / norm, (p[1] - wy) / norm]
        vec += [0.0] * (5 + 2 * self.capacity - len(vec))
        return vec

    def _md(self, p1, p2):
        """통로 그래프 거리 (layout.aisle_distance 단축)"""
        return self.layout.aisle_distance(p1, p2)
