"""로봇 상태와 트립 경로 구성 (NNH 방문 순서, 복귀 워크스테이션 선택)

좌표는 grid.GridWarehouseLayout의 셀 (col, row), 거리는 layout.aisle_distance (유향 셀 그래프 최단거리, m)
"""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

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


def route_total_distance(
    start:     Cell,
    route:     List[Cell],
    end:       Cell,
    dist_func: Callable[[Cell, Cell], float],
) -> float:
    """start -> route -> end 경로의 총 이동거리

    입력:
        start: 출발 좌표
        route: 경유 좌표 리스트 (순서대로 방문)
        end: 도착 좌표
        dist_func: 두 좌표 간 거리 함수
    출력:
        총 이동거리
    """
    if not route:
        return dist_func(start, end)
    total, current = 0.0, start
    for point in route:
        total   += dist_func(current, point)
        current  = point
    total += dist_func(current, end)
    return total


def nearest_workstation(
    loc: Cell,
    workstations: List[Cell],
    dist_func: Callable[[Cell, Cell], float],
) -> int:
    """주어진 위치에서 가장 가까운 워크스테이션 탐색

    입력:
        loc: 기준 좌표
        workstations: 워크스테이션 좌표 리스트
        dist_func: 두 좌표 간 거리 함수
    출력:
        가장 가까운 워크스테이션 인덱스
    """
    return min(range(len(workstations)), key=lambda i: dist_func(loc, workstations[i]))


def best_workstation_route(
    start: Cell,
    destinations: List[Cell],
    workstations: List[Cell],
    dist_func: Callable[[Cell, Cell], float],
) -> Tuple[int, List[Cell], float]:
    """NNH 경로 생성 후 총 이동거리가 최소가 되는 복귀 워크스테이션 선택

    입력:
        start: 출발 좌표
        destinations: 방문할 좌표 리스트
        workstations: 복귀 후보 워크스테이션 좌표 리스트
        dist_func: 두 좌표 간 거리 함수
    출력:
        (workstation_id, route, total_distance)
    """
    route = nearest_neighbor_route(start, destinations, dist_func)
    best_id, best_dist = 0, float("inf")
    for wid, ws in enumerate(workstations):
        d = route_total_distance(start, route, ws, dist_func)
        if d < best_dist:
            best_dist, best_id = d, wid
    return best_id, route, best_dist


class Robot:
    """피킹 로봇의 위치, 남은 경로, 적재 슬롯 상태 관리

    workstation / workstation_id: 현재 트립의 복귀 워크스테이션
    (트립 시작 또는 주문 삽입 때마다 총 이동거리가 최소가 되도록 재결정)
    """

    def __init__(self, robot_id: int, workstation: Cell,
                 layout: GridWarehouseLayout, capacity: int,
                 workstation_id: int = 0):
        """
        입력:
            robot_id: 로봇 id
            workstation: 초기 워크스테이션 좌표
            layout: GridWarehouseLayout
            capacity: 적재 가능한 토트 수
            workstation_id: 초기 워크스테이션 인덱스
        """
        self.id             = robot_id
        self.workstation     = workstation
        self.workstation_id  = workstation_id
        self.layout          = layout
        self.capacity        = capacity
        self.reset()

    def reset(self):
        """로봇을 워크스테이션에 대기 중인 빈 상태로 되돌림"""
        self.pos               = self.workstation
        self.route             = []
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
        self.pending_insert    = []          # 계획 실패로 아직 경로에 넣지 못한 삽입 주문 위치

    @property
    def remaining_distance(self) -> float:
        """현재 목적지 -> 남은 경로 -> 복귀 워크스테이션까지 남은 이동거리 (m)"""
        d = 0.0
        if self.current_dest:
            d += self._md(self.pos, self.current_dest)
        if self.route:
            prev = self.current_dest if self.current_dest else self.pos
            for nxt in self.route:
                d += self._md(prev, nxt)
                prev = nxt
            d += self._md(prev, self.workstation)
        elif self.current_dest:
            d += self._md(self.current_dest, self.workstation)
        return d

    def arrive_at_node(self) -> bool:
        """현재 목적지 도착 처리 후 다음 목적지 꺼냄

        출력:
            남은 경유지가 있어 다음 목적지가 설정되면 True, 없으면 False
        """
        self.pos = self.current_dest
        if self.route:
            self.current_dest = self.route.pop(0)
            return True
        self.current_dest = None
        return False

    def assign_trip(self, destinations: List[Cell],
                    layout: GridWarehouseLayout):
        """현재 위치에서 출발하는 새 트립 배정 및 이동 시작

        - 방문 순서: NNH
        - 복귀 워크스테이션: 총 이동거리가 최소인 곳

        입력:
            destinations: 방문할 좌표 리스트
            layout: GridWarehouseLayout
        """
        best_ws_id, route, _ = best_workstation_route(
            self.pos, destinations, layout.workstations, layout.aisle_distance)
        self.workstation_id = best_ws_id
        self.workstation     = layout.workstations[best_ws_id]
        self.route       = route
        self.idle_slots  = self.capacity - len(destinations)
        self.is_idle     = False
        self.route_version += 1
        self._start_move()

    def insert_and_reorder(self, loc: Cell,
                            layout: GridWarehouseLayout):
        """이동 중인 트립에 주문 삽입 후 남은 경로와 복귀 워크스테이션 재계산

        현재 목적지(current_dest)는 유지하고 그 이후 경로만 재구성

        입력:
            loc: 삽입할 주문 위치
            layout: GridWarehouseLayout
        """
        pending = list(self.route) + [loc]
        start   = self.current_dest if self.current_dest else self.pos
        best_ws_id, new_route, _ = best_workstation_route(
            start, pending, layout.workstations, layout.aisle_distance)
        self.workstation_id = best_ws_id
        self.workstation     = layout.workstations[best_ws_id]
        self.route       = new_route
        self.idle_slots  = max(0, self.idle_slots - 1)
        self.route_version += 1

    def delta_distance_insert(self, loc: Cell,
                               layout: GridWarehouseLayout,
                               target_ws_coords: Optional[Cell] = None,
                               dynamic_ws: bool = False,
                               ) -> float:
        """주문을 이 로봇에 추가했을 때 늘어나는 이동거리

        복귀 지점 결정 우선순위:
          1) target_ws_coords가 주어지면 그 좌표로 고정 (idle 후보 평가용)
          2) dynamic_ws=True이면 주문 포함 후 최적 워크스테이션으로 재탐색
             (insert_and_reorder와 같은 기준, mid-trip 후보 평가용)
          3) 그 외에는 현재 self.workstation 사용

        입력:
            loc: 추가할 주문 위치
            layout: GridWarehouseLayout
            target_ws_coords: 고정할 복귀 좌표
            dynamic_ws: 복귀 워크스테이션 재탐색 여부
        출력:
            삽입 후 남은 경로 거리 − 삽입 전 남은 경로 거리 (m)
        """
        pending = list(self.route) + [loc]
        start   = self.current_dest if self.current_dest else self.pos

        if target_ws_coords is not None:
            end       = target_ws_coords
            before    = self._remaining_route_dist(layout, end)
            new_route = nearest_neighbor_route(start, pending, layout.aisle_distance)
            after     = route_total_distance(start, new_route, end, layout.aisle_distance)
        elif dynamic_ws:
            before          = self._remaining_route_dist(layout, self.workstation)
            _, _, after_ws  = best_workstation_route(
                start, pending, layout.workstations, layout.aisle_distance)
            after = after_ws
        else:
            end       = self.workstation
            before    = self._remaining_route_dist(layout, end)
            new_route = nearest_neighbor_route(start, pending, layout.aisle_distance)
            after     = route_total_distance(start, new_route, end, layout.aisle_distance)

        return after - before

    def _remaining_route_dist(self, layout: GridWarehouseLayout,
                               end: Optional[Cell] = None) -> float:
        """현재 목적지(없으면 현재 위치)부터 남은 경로를 거쳐 end까지의 거리

        입력:
            layout: GridWarehouseLayout
            end: 도착 좌표 (None이면 self.workstation)
        출력:
            이동거리 (m)
        """
        end   = end if end is not None else self.workstation
        start = self.current_dest if self.current_dest else self.pos
        return route_total_distance(start, self.route, end,
                                    layout.aisle_distance)

    def _start_move(self):
        """경로의 첫 지점을 현재 목적지로 설정"""
        if self.route:
            self.current_dest = self.route.pop(0)

    def _md(self, p1, p2):
        """통로 그래프 거리 (layout.aisle_distance 단축)"""
        return self.layout.aisle_distance(p1, p2)
