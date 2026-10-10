import random
from collections import deque
from typing import List, Tuple, Dict, Callable, Optional

import numpy as np

from constant import (
    BN, AN, SN, Wa, Wc, Wt, Dt, STORAGE_LOCS, NUM_SKUS,
    NUM_ROBOTS, ROBOT_CAPACITY, NUM_WORKSTATIONS, DEFAULT_SEED,
    ROBOT_SPEED, STATE_MAX_TIME, REWARD_D_MAX, SOLO_RT_M,
)


class WarehouseLayout:
    """블록/통로/선반으로 구성된 창고 레이아웃과 통로 기반 거리 계산.

    워크스테이션 K개는 창고 하단(y=0)에 폭 방향으로 균등 배치되고,
    각 저장 위치에는 SKU가 무작위로 할당된다.
    """

    def __init__(self, seed: int = DEFAULT_SEED,
                 num_robots: int = NUM_ROBOTS,
                 robot_capacity: int = ROBOT_CAPACITY,
                 num_workstations: int = NUM_WORKSTATIONS,
                 scenario=None):
        """
        입력:
            seed: SKU 배치용 난수 시드
            num_robots: 로봇 수
            robot_capacity: 로봇 1대가 한 트립에 담을 수 있는 토트 수
            num_workstations: 워크스테이션 수 K
            scenario: scenario.Scenario (None이면 기본값: 포아송/균등/무작위 저장/고정 시간)
        """
        from scenario import Scenario
        self.scenario         = scenario if scenario is not None else Scenario()
        self.num_robots       = num_robots
        self.robot_capacity   = robot_capacity
        self.num_workstations = num_workstations
        self.rng              = random.Random(seed)

        self.aisle_block_w  = Wa + 2 * Dt
        self.block_h        = Wc + SN * Wt
        self.total_width    = AN * self.aisle_block_w
        self.total_height   = BN * self.block_h + (BN - 1) * Wc

        # coord_norm: 좌표(로봇/주문의 워크스테이션 기준 상대좌표) 정규화 기준
        # dist_norm : 이동거리(삽입 detour 등) 정규화 기준
        self.coord_norm = max(self.total_width, self.total_height)
        self.dist_norm  = self.total_width + self.total_height
        # reward_d_max: 삽입 detour 보상 정규화 기준, solo_rt_m: 주문 1건 단독 왕복 거리 추정치
        self.reward_d_max = REWARD_D_MAX
        self.solo_rt_m    = SOLO_RT_M

        self.workstations = self._build_workstations()
        self.workstation  = self.workstations[0]

        # 교차통로 밴드 (y_lo, y_hi): 최하단 밴드 + 블록 사이 밴드
        self.cross_bands = [(0.0, Wc)]
        for b in range(BN - 1):
            top_b            = b * (self.block_h + Wc) + self.block_h
            shelf_start_next = (b + 1) * (self.block_h + Wc) + Wc
            self.cross_bands.append((top_b, shelf_start_next))

        self.storage_coords, self.access_coords = self._build_storage_coords()
        self.sku_ids        = list(range(NUM_SKUS))
        self.loc_to_sku, self.sku_to_locs = self._assign_skus()
        if self.scenario.storage == "class_based" and not getattr(self, "is_grid", False):
            self.apply_class_storage()

    def _build_workstations(self) -> List[Tuple[float, float]]:
        """워크스테이션 좌표를 생성한다.

        출력:
            창고 하단(y=0)에 폭 방향으로 균등 배치된 K개 좌표 리스트
        """
        xs = [(i + 0.5) / self.num_workstations * self.total_width
              for i in range(self.num_workstations)]
        return [(x, 0.0) for x in xs]

    def _build_storage_coords(self):
        """모든 저장 위치의 좌표와 통로 접근 좌표를 생성한다.

        출력:
            (coords, access)
            coords: 랙 위 토트 좌표 리스트
            access: 각 토트에 접근하는 통로 중앙선 위 좌표 리스트 (coords와 같은 순서)
        """
        coords, access = [], []
        for b in range(BN):
            block_y0 = b * (self.block_h + Wc)
            for a in range(AN):
                cx = a * self.aisle_block_w + Dt + Wa / 2
                x_left  = cx - Wa / 2 - Dt / 2
                x_right = cx + Wa / 2 + Dt / 2
                for side_x in (x_left, x_right):
                    for row in range(SN):
                        y = block_y0 + Wc + (row + 0.5) * Wt
                        coords.append((side_x, y))
                        access.append((cx, y))

        assert len(coords) == STORAGE_LOCS, (
            f"Coordinate count mismatch: expected {STORAGE_LOCS}, got {len(coords)}"
        )
        return coords, access

    def _assign_skus(self) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
        """저장 위치마다 SKU를 무작위로 할당한다.

        위치가 하나도 할당되지 않은 SKU는 임의 위치 하나를 추가로 매핑한다.

        출력:
            (loc_to_sku, sku_to_locs)
            loc_to_sku: 위치 인덱스 -> SKU
            sku_to_locs: SKU -> 위치 인덱스 리스트
        """
        indices = list(range(len(self.storage_coords)))
        self.rng.shuffle(indices)
        loc_to_sku: Dict[int, int]       = {}
        sku_to_locs: Dict[int, List[int]] = {s: [] for s in self.sku_ids}

        for idx in indices:
            sku = self.rng.choice(self.sku_ids)
            loc_to_sku[idx] = sku
            sku_to_locs[sku].append(idx)

        for sku in self.sku_ids:
            if not sku_to_locs[sku]:
                fallback = self.rng.choice(indices)
                sku_to_locs[sku].append(fallback)

        return loc_to_sku, sku_to_locs

    def apply_class_storage(self):
        """클래스 기반 저장: 인기 SKU(id가 작은 SKU)를 워크스테이션에서 가까운 위치부터 배정한다.

        위치를 가장 가까운 워크스테이션까지 거리(워크스테이션 -> 위치) 순으로 정렬하고,
        SKU마다 같은 수의 위치를 순위대로 나눠 준다 (Hausman et al., 1976의 class-based storage를
        SKU 순위 단위로 세분한 형태).
        """
        n_loc = len(self.access_coords)
        dist  = [min(self.aisle_distance(w, self.access_coords[i]) for w in self.workstations)
                 for i in range(n_loc)]
        order = sorted(range(n_loc), key=lambda i: (dist[i], i))
        n_sku = len(self.sku_ids)
        self.loc_to_sku  = {}
        self.sku_to_locs = {s: [] for s in self.sku_ids}
        for rank, i in enumerate(order):
            sku = self.sku_ids[min(n_sku - 1, rank * n_sku // n_loc)]
            self.loc_to_sku[i] = sku
            self.sku_to_locs[sku].append(i)

    def manhattan(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """두 점의 맨해튼(L1) 거리. 랙 통과를 허용하므로 이동 비용 계산에는 쓰지 않는다.

        입력:
            p1, p2: (x, y) 좌표
        출력:
            |Δx| + |Δy|
        """
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    @staticmethod
    def _band_vert(ya: float, yb: float, lo: float, hi: float) -> float:
        """교차통로 밴드 [lo, hi]를 경유해 ya에서 yb로 갈 때의 세로 이동거리.

        입력:
            ya, yb: 출발/도착 y 좌표
            lo, hi: 밴드의 y 범위
        출력:
            세로 방향 이동거리
        """
        lo_ab, hi_ab = (ya, yb) if ya <= yb else (yb, ya)
        if hi < lo_ab:
            yc = hi
        elif lo > hi_ab:
            yc = lo
        else:
            return abs(ya - yb)
        return abs(ya - yc) + abs(yb - yc)

    def aisle_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """통로 그래프 위 최단 이동거리.

        같은 세로 통로(x 동일)이면 |Δy|, 다른 통로이면 가장 유리한
        교차통로 밴드를 경유하는 거리(세로 우회 + |Δx|)를 쓴다.

        입력:
            p1, p2: (x, y) 좌표
        출력:
            이동거리 (m)
        """
        x1, y1 = p1
        x2, y2 = p2
        if abs(x1 - x2) < 1e-9:
            return abs(y1 - y2)
        dx   = abs(x1 - x2)
        best = float("inf")
        for lo, hi in self.cross_bands:
            best = min(best, self._band_vert(y1, y2, lo, hi) + dx)
        return best

    def sample_order_sku(self, rng: random.Random) -> int:
        """주문 SKU를 균등 확률로 하나 뽑는다.

        입력:
            rng: 난수 생성기
        출력:
            SKU id
        """
        return rng.choice(self.sku_ids)

    def find_tote_location(
        self, sku: int, rng: random.Random
    ) -> Tuple[int, Tuple[float, float]]:
        """SKU가 보관된 위치 중 하나를 무작위로 고른다.

        입력:
            sku: SKU id
            rng: 난수 생성기
        출력:
            (위치 인덱스, 통로 접근 좌표)
        """
        locs = self.sku_to_locs[sku]
        idx  = rng.choice(locs)
        return idx, self.access_coords[idx]

    @property
    def num_locations(self) -> int:
        """전체 저장 위치 수."""
        return len(self.storage_coords)

    @property
    def num_skus(self) -> int:
        """전체 SKU 수."""
        return len(self.sku_ids)


def nearest_neighbor_route(
    start:        Tuple[float, float],
    destinations: List[Tuple[float, float]],
    dist_func:    Callable[[Tuple[float, float], Tuple[float, float]], float],
) -> List[Tuple[float, float]]:
    """최근접 이웃(NNH) 방식으로 방문 순서를 정한다.

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
    start:     Tuple[float, float],
    route:     List[Tuple[float, float]],
    end:       Tuple[float, float],
    dist_func: Callable[[Tuple[float, float], Tuple[float, float]], float],
) -> float:
    """start -> route -> end 경로의 총 이동거리.

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
    loc: Tuple[float, float],
    workstations: List[Tuple[float, float]],
    dist_func: Callable[[Tuple[float, float], Tuple[float, float]], float],
) -> int:
    """주어진 위치에서 가장 가까운 워크스테이션을 찾는다.

    입력:
        loc: 기준 좌표
        workstations: 워크스테이션 좌표 리스트
        dist_func: 두 좌표 간 거리 함수
    출력:
        가장 가까운 워크스테이션 인덱스
    """
    return min(range(len(workstations)), key=lambda i: dist_func(loc, workstations[i]))


def best_workstation_route(
    start: Tuple[float, float],
    destinations: List[Tuple[float, float]],
    workstations: List[Tuple[float, float]],
    dist_func: Callable[[Tuple[float, float], Tuple[float, float]], float],
) -> Tuple[int, List[Tuple[float, float]], float]:
    """NNH 경로를 만들고, 총 이동거리가 최소가 되는 복귀 워크스테이션을 고른다.

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
    """피킹 로봇의 위치, 남은 경로, 적재 슬롯 상태를 관리한다.

    workstation / workstation_id는 현재 트립의 복귀 워크스테이션이며,
    트립을 시작하거나 주문이 삽입될 때마다 총 이동거리가 최소가 되도록
    다시 정해진다.
    """

    def __init__(self, robot_id: int, workstation: Tuple[float, float],
                 layout: WarehouseLayout, capacity: int,
                 workstation_id: int = 0):
        """
        입력:
            robot_id: 로봇 id
            workstation: 초기 워크스테이션 좌표
            layout: WarehouseLayout
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
        """로봇을 워크스테이션에 대기 중인 빈 상태로 되돌린다."""
        self.pos               = self.workstation
        self.route             = []
        self.idle_slots        = self.capacity
        self.is_idle           = True
        self.current_dest      = None
        self.progress          = 0.0
        self.total_travel_time = 0.0
        self._travel_start     = None
        # 그리드 이동 모드 전용: 피킹/워크스테이션 처리 잔여 시간, 우회 경로, 연속 대기 틱
        self.timer             = 0.0
        self.timer_kind        = None
        self.detour            = []
        self.wait_ticks        = 0
        self.park_spot         = None
        # 그리드 예약 모드 전용: 경로가 바뀔 때마다 route_version이 오르고, 계획은 plan_version 기준
        self.route_version     = 0
        self.plan_version      = 0
        self.plan              = deque()     # (틱, 셀) 순서의 예약 경로
        self.replan_pending    = False
        self.pending_insert    = []          # 계획 실패로 아직 경로에 넣지 못한 삽입 주문 위치

    def start_travel(self, now: float):
        """이동 시작 시각을 기록한다 (이미 이동 중이면 무시).

        입력:
            now: 현재 시각 (s)
        """
        if self._travel_start is None:
            self._travel_start = now

    def end_travel(self, now: float):
        """이동 종료 시 누적 이동시간을 갱신한다.

        입력:
            now: 현재 시각 (s)
        """
        if self._travel_start is not None:
            self.total_travel_time += max(0.0, now - self._travel_start)
            self._travel_start = None

    def time_to_complete_current(self) -> float:
        """현재 트립을 마치고 복귀할 때까지 남은 시간.

        출력:
            남은 이동시간 (s)
        """
        return self.remaining_distance / ROBOT_SPEED

    @property
    def remaining_distance(self) -> float:
        """현재 목적지 -> 남은 경로 -> 복귀 워크스테이션까지 남은 이동거리 (m)."""
        d = 0.0
        if self.current_dest:
            d += max(0.0, self._md(self.pos, self.current_dest) - self.progress)
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
        """현재 목적지에 도착 처리하고 다음 목적지를 꺼낸다.

        출력:
            남은 경유지가 있어 다음 목적지가 설정되면 True, 없으면 False
        """
        self.pos      = self.current_dest
        self.progress = 0.0
        if self.route:
            self.current_dest = self.route.pop(0)
            return True
        self.current_dest = None
        return False

    def redock(self, ws_id: int, layout: WarehouseLayout):
        """idle 로봇의 복귀 워크스테이션만 바꾼다. 현재 위치(pos)는 그대로 둔다.

        입력:
            ws_id: 새 워크스테이션 인덱스
            layout: WarehouseLayout
        """
        assert self.is_idle, "redock()은 idle 로봇에서만 호출할 수 있습니다."
        self.workstation_id = ws_id
        self.workstation     = layout.workstations[ws_id]

    def assign_trip(self, destinations: List[Tuple[float, float]],
                    layout: WarehouseLayout):
        """현재 위치에서 출발하는 새 트립을 배정하고 이동을 시작한다.

        방문 순서는 NNH로 정하고, 복귀 워크스테이션은 총 이동거리가 최소인 곳으로 정한다.

        입력:
            destinations: 방문할 좌표 리스트
            layout: WarehouseLayout
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

    def insert_and_reorder(self, loc: Tuple[float, float],
                            layout: WarehouseLayout):
        """이동 중인 트립에 주문을 삽입하고 남은 경로와 복귀 워크스테이션을 다시 계산한다.

        현재 목적지(current_dest)는 바꾸지 않고, 그 이후 경로만 재구성한다.

        입력:
            loc: 삽입할 주문 위치
            layout: WarehouseLayout
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

    def delta_distance_insert(self, loc: Tuple[float, float],
                               layout: WarehouseLayout,
                               target_ws_coords: Optional[Tuple[float, float]] = None,
                               dynamic_ws: bool = False,
                               ) -> float:
        """주문을 이 로봇에 추가했을 때 늘어나는 이동거리를 계산한다.

        복귀 지점은 다음 우선순위로 정한다.
          1) target_ws_coords가 주어지면 그 좌표로 고정 (idle 후보 평가용)
          2) dynamic_ws=True이면 주문 포함 후 최적 워크스테이션으로 재탐색
             (insert_and_reorder와 같은 기준, mid-trip 후보 평가용)
          3) 그 외에는 현재 self.workstation 사용

        입력:
            loc: 추가할 주문 위치
            layout: WarehouseLayout
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

    def state_vector(self, max_cap: int, max_time: float = STATE_MAX_TIME) -> np.ndarray:
        """로봇 상태를 현재 복귀 워크스테이션 기준 상대좌표로 인코딩한다.

        입력:
            max_cap: 로봇 용량 (슬롯/경로 정규화 및 패딩 길이)
            max_time: 잔여 이동시간 정규화 기준 (s)
        출력:
            길이 5 + max_cap*2 벡터
            [상대 x, 상대 y, 빈 슬롯 비율, idle 여부, 잔여시간, 경유지 상대좌표...]
        """
        wx, wy     = self.workstation
        norm       = self.layout.coord_norm
        rem_t_norm = min(self.time_to_complete_current() / max_time, 1.0)
        vec = [
            (self.pos[0] - wx) / norm,
            (self.pos[1] - wy) / norm,
            float(self.idle_slots) / max_cap,
            float(self.is_idle),
            rem_t_norm,
        ]
        route_flat = []
        for p in self.route[:max_cap]:
            route_flat.extend([(p[0] - wx) / norm, (p[1] - wy) / norm])
        while len(route_flat) < max_cap * 2:
            route_flat.extend([0.0, 0.0])
        vec.extend(route_flat)
        return np.array(vec, dtype=np.float32)

    def _remaining_route_dist(self, layout: WarehouseLayout,
                               end: Optional[Tuple[float, float]] = None) -> float:
        """현재 목적지(없으면 현재 위치)부터 남은 경로를 거쳐 end까지의 거리.

        입력:
            layout: WarehouseLayout
            end: 도착 좌표 (None이면 self.workstation)
        출력:
            이동거리 (m)
        """
        end   = end if end is not None else self.workstation
        start = self.current_dest if self.current_dest else self.pos
        return route_total_distance(start, self.route, end,
                                    layout.aisle_distance)

    def _start_move(self):
        """경로의 첫 지점을 현재 목적지로 설정한다."""
        if self.route:
            self.current_dest = self.route.pop(0)

    def _md(self, p1, p2):
        """통로 그래프 거리 (layout.aisle_distance 단축)."""
        return self.layout.aisle_distance(p1, p2)
