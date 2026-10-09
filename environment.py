import random
from typing import List, Tuple, Dict, Callable

BN            = 3       # number of blocks
AN            = 14      # aisles per block
SN            = 20      # shelves on ONE side of each aisle (rows)

Wa            = 3.0     # m  — aisle (corridor) width
Wc            = 4.0     # m  — cross-aisle width
Wt            = 0.6     # m  — tote width  (row pitch along y)
Dt            = 0.5     # m  — tote depth  (rack depth along x), single-deep

NUM_AISLES    = BN * AN
STORAGE_LOCS  = BN * AN * 2 * SN

NUM_SKUS      = 350

NUM_ROBOTS        = 14
ROBOT_CAPACITY     = 5
ROBOT_SPEED        = 1.0    # m/s

NUM_WORKSTATIONS   = 3      # K=3, 창고 하단에 일정 간격 배치

SIM_TIME        = 10 * 3600          # 10-hour shift
# 그리드 실험 밖에서 단발성 시뮬레이션을 돌릴 때 참고할 일반적 기본값
LAMBDA_DEFAULT   = 850 / 3600         # orders/s (700~1000/hr 그리드의 중앙값)


class WarehouseLayout:
    def __init__(self, seed: int = 42,
                 num_robots: int = NUM_ROBOTS,
                 robot_capacity: int = ROBOT_CAPACITY,
                 num_workstations: int = NUM_WORKSTATIONS):
        self.num_robots       = num_robots
        self.robot_capacity   = robot_capacity
        self.num_workstations = num_workstations
        self.rng              = random.Random(seed)

        # Derived geometry
        self.aisle_block_w  = Wa + 2 * Dt
        self.block_h        = Wc + SN * Wt
        self.total_width    = AN * self.aisle_block_w
        self.total_height   = BN * self.block_h + (BN - 1) * Wc

        # 상대좌표/거리 정규화 기준. 창고의 실제 치수에서 유도하므로 레이아웃
        # 크기가 바뀌어도(K, 블록 수 등) 항상 그 시점 창고 규모에 상대적이다.
        #   coord_norm : 좌표 하나(예: pos-workstation)를 정규화할 때 쓰는 기준
        #   dist_norm  : 트립 전체 이동거리류(삽입 detour, 보상 등)를 정규화할 때 쓰는 기준
        self.coord_norm = max(self.total_width, self.total_height)
        self.dist_norm  = self.total_width + self.total_height

        # 워크스테이션: 창고 하단(y=0)에 K개를 폭 방향으로 균등 배치.
        self.workstations = self._build_workstations()
        self.workstation   = self.workstations[0]   # 단일 지점 참조가 필요한 레거시 코드용

        # Cross-aisle bands
        self.cross_bands = [(0.0, Wc)]
        for b in range(BN - 1):
            top_b            = b * (self.block_h + Wc) + self.block_h
            shelf_start_next = (b + 1) * (self.block_h + Wc) + Wc
            self.cross_bands.append((top_b, shelf_start_next))

        self.storage_coords, self.access_coords = self._build_storage_coords()
        self.sku_ids        = list(range(NUM_SKUS))
        self.loc_to_sku, self.sku_to_locs = self._assign_skus()

    def _build_workstations(self) -> List[Tuple[float, float]]:
        xs = [(i + 0.5) / self.num_workstations * self.total_width
              for i in range(self.num_workstations)]
        return [(x, 0.0) for x in xs]

    def _build_storage_coords(self):
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
                        access.append((cx, y))       # 통로 접근 노드

        assert len(coords) == STORAGE_LOCS, (
            f"Coordinate count mismatch: expected {STORAGE_LOCS}, got {len(coords)}"
        )
        return coords, access

    def _assign_skus(self) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
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
        """Manhattan (L1) distance — 참고용(랙 통과 허용). 이동 비용엔 쓰지 않는다."""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    @staticmethod
    def _band_vert(ya: float, yb: float, lo: float, hi: float) -> float:
        lo_ab, hi_ab = (ya, yb) if ya <= yb else (yb, ya)
        if hi < lo_ab:
            yc = hi
        elif lo > hi_ab:
            yc = lo
        else:
            return abs(ya - yb)
        return abs(ya - yc) + abs(yb - yc)

    def aisle_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """통로 그래프 최단거리. 같은 세로 통로(x 동일) → |Δy|.
        다른 통로 → 최근접 교차통로 밴드를 경유:  밴드 세로우회 + |Δx|."""
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
        return rng.choice(self.sku_ids)

    def find_tote_location(
        self, sku: int, rng: random.Random
    ) -> Tuple[int, Tuple[float, float]]:
        locs = self.sku_to_locs[sku]
        idx  = rng.choice(locs)
        return idx, self.access_coords[idx]

    @property
    def num_locations(self) -> int:
        return len(self.storage_coords)

    @property
    def num_skus(self) -> int:
        return len(self.sku_ids)


# Routing utilities

def nearest_neighbor_route(
    start:        Tuple[float, float],
    destinations: List[Tuple[float, float]],
    dist_func:    Callable[[Tuple[float, float], Tuple[float, float]], float],
) -> List[Tuple[float, float]]:
    """Greedy nearest-neighbour tour starting from *start*."""
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
    """주어진 위치에서 가장 가까운 워크스테이션의 인덱스를 반환한다.

    주문이 더 이상 특정 워크스테이션에 고정 배정되지 않으므로, idle 로봇이
    주문 하나를 새로 받을 때 '어느 워크스테이션으로 복귀할지'를 결정하는
    기본 규칙으로 사용한다.
    """
    return min(range(len(workstations)), key=lambda i: dist_func(loc, workstations[i]))


def best_workstation_route(
    start: Tuple[float, float],
    destinations: List[Tuple[float, float]],
    workstations: List[Tuple[float, float]],
    dist_func: Callable[[Tuple[float, float], Tuple[float, float]], float],
) -> Tuple[int, List[Tuple[float, float]], float]:
    """*destinations* 를 도는 트립을 시작할 때, 복귀 워크스테이션을 어디로
    잡아야 총 이동거리가 최소가 되는지 K개 후보 전부를 비교해 고른다.

    로봇이 특정 워크스테이션에 고정되지 않는 모델에서, 새 트립을 시작하는
    모든 시뮬레이터(RL/baseline 공통)가 이 함수로 복귀지를 동적으로 정한다.
    반환값: (workstation_id, route, total_distance)
    """
    route = nearest_neighbor_route(start, destinations, dist_func)
    best_id, best_dist = 0, float("inf")
    for wid, ws in enumerate(workstations):
        d = route_total_distance(start, route, ws, dist_func)
        if d < best_dist:
            best_dist, best_id = d, wid
    return best_id, route, best_dist