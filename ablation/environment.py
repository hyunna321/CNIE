import random
from typing import List, Tuple, Dict, Callable

from constant import (
    BN, AN, SN, Wa, Wc, Wt, Dt, STORAGE_LOCS, NUM_SKUS,
    NUM_ROBOTS, ROBOT_CAPACITY, NUM_WORKSTATIONS, DEFAULT_SEED,
)


class WarehouseLayout:
    """블록/통로/선반으로 구성된 창고 레이아웃과 통로 기반 거리 계산.

    워크스테이션 K개는 창고 하단(y=0)에 폭 방향으로 균등 배치되고,
    각 저장 위치에는 SKU가 무작위로 할당된다.
    """

    def __init__(self, seed: int = DEFAULT_SEED,
                 num_robots: int = NUM_ROBOTS,
                 robot_capacity: int = ROBOT_CAPACITY,
                 num_workstations: int = NUM_WORKSTATIONS):
        """
        입력:
            seed: SKU 배치용 난수 시드
            num_robots: 로봇 수
            robot_capacity: 로봇 1대가 한 트립에 담을 수 있는 토트 수
            num_workstations: 워크스테이션 수 K
        """
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
