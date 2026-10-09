"""그리드(셀) 기반 창고 레이아웃: 단방향 통로 그래프, 워크스테이션 대기열/주차 셀, 거리 테이블.

연속 좌표 레이아웃(environment.WarehouseLayout)과 같은 상수로 셀 지도를 만든다.
CELL_SIZE = 1 m 기준 열 구성은 [랙][통로][통로][통로] × AN + [랙] (등을 맞댄 랙 2개가 1칸).
행은 아래부터 워크스테이션 구역, 하단 교차통로, (블록 랙 행, 교차통로) × BN 이고
맨 위에 교차통로를 하나 더 둔다 (단방향 통로가 막다른 길이 되지 않도록).

이동 규칙 (유향 그래프, traffic 인자):
  - "free": 통로와 교차통로 안에서 상하좌우 모든 방향 이동.
  - "lanes": 통로는 짝수 번째가 위(+y), 홀수 번째가 아래(−y)로만 이동하고 같은 통로 안
    좌우 이동은 허용. 교차통로는 아래 절반 행이 동쪽, 위 절반 행이 서쪽으로만 가로 이동하고
    세로 이동은 양방향.
  - 워크스테이션 구역은 두 모드 모두 아래와 같이 한 방향이다.
  - 워크스테이션 k: 대기열 셀(동쪽으로 한 줄) -> 도킹 셀 -> 주차장.
    대기열 입구는 맨 서쪽 셀 하나다. 주차장은 도킹 셀 동쪽에 주행 차선과 주차 칸 행을
    번갈아 쌓은 구조로, 모든 주차 칸이 차선에 바로 붙어 있어 다른 주차 로봇에 막히지 않는다.

        교차통로 (하단)         ↑    ↑  ↑  ↑  ↑    ↑
        대기열 → → → 도킹 →  [ → →  →  →  →  ↓ ]   차선 0 (동쪽, 각 칸에서 교차통로로 출구)
                              [ ↑  P  P  P  P  ↓ ]   주차 칸 (위 차선과 양방향)
                              [ ↑ ← ← ← ← ← ← ]   차선 1 (서쪽)
                              [    P  P  P  P    ]   ...

좌표는 정수 셀 (col, row) 튜플이고, 거리 = 셀 수 × CELL_SIZE (m).
"""
import math
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from environment import WarehouseLayout
from constant import (
    BN, AN, SN, Wa, Wc, Wt, Dt, NUM_ROBOTS, ROBOT_CAPACITY, NUM_WORKSTATIONS,
    DEFAULT_SEED, ROBOT_SPEED, CELL_SIZE, WS_QUEUE_CELLS, WS_PARK_CELLS,
    GRID_TRAFFIC, GRID_MOTION,
)

Cell = Tuple[int, int]

# 셀 종류
RACK, AISLE, CROSS, WS_ZONE = 0, 1, 2, 3

# (WS 수, 대기열, 주차장 폭, 차선 수, 이동 규칙) -> 셀 그래프와 거리 테이블.
# 시드와 무관하므로 레이아웃끼리 공유한다.
_GRAPH_CACHE: Dict[tuple, dict] = {}


class GridWarehouseLayout(WarehouseLayout):
    """셀 기반 레이아웃. WarehouseLayout과 같은 인터페이스를 유지한다.

    SKU 배치는 같은 시드의 연속 레이아웃과 동일하고, 저장 위치의 접근 좌표
    (access_coords)와 워크스테이션 좌표만 셀 좌표로 바뀐다.
    aisle_distance()는 유향 그래프 최단거리라 d(a, b) != d(b, a)일 수 있다.
    """

    is_grid = True

    def __init__(self, seed: int = DEFAULT_SEED,
                 num_robots: int = NUM_ROBOTS,
                 robot_capacity: int = ROBOT_CAPACITY,
                 num_workstations: int = NUM_WORKSTATIONS,
                 queue_cells: Optional[int] = WS_QUEUE_CELLS,
                 park_cells: Optional[int] = WS_PARK_CELLS,
                 collisions: bool = True,
                 traffic: str = GRID_TRAFFIC,
                 motion: str = GRID_MOTION):
        """
        입력:
            seed: SKU 배치용 난수 시드
            num_robots: 로봇 수 M (대기열/주차 셀 크기 결정에 사용)
            robot_capacity: 로봇 용량
            num_workstations: 워크스테이션 수 K
            queue_cells: 워크스테이션별 대기열 셀 수 (None이면 ceil(M/K))
            park_cells: 워크스테이션별 주차 칸 수 (None이면 M, 어느 WS에든 전원 주차 가능)
            collisions: False이면 로봇끼리 셀 점유를 무시한다 (회귀 검증용, motion="reactive"에서만)
            traffic: "free" (모든 방향) 또는 "lanes" (일방통행 통로 + 차선 교차통로)
            motion: "reserve" (배정 시 시공간 경로 예약) 또는 "reactive" (막히면 대기/우회)
        """
        assert traffic in ("free", "lanes"), traffic
        assert motion in ("reserve", "reactive"), motion
        super().__init__(seed=seed, num_robots=num_robots,
                         robot_capacity=robot_capacity,
                         num_workstations=num_workstations)
        self.collisions = collisions or motion == "reserve"
        self.traffic    = traffic
        self.motion     = motion
        self.tick       = CELL_SIZE / ROBOT_SPEED

        self._size_ws_zone(queue_cells, park_cells)
        self._build_cell_map()
        self._build_ws_zone()

        # 저장 위치 인덱스 순서는 연속 레이아웃의 _build_storage_coords와 같다.
        self.access_coords = self._build_pick_cells()
        self.workstations  = list(self.docks)
        self.workstation   = self.workstations[0]

        key = (num_workstations, self.queue_len, self.lot_w, self.n_lanes, traffic)
        if key not in _GRAPH_CACHE:
            _GRAPH_CACHE[key] = self._build_graph()
        g = _GRAPH_CACHE[key]
        self.cell_index: Dict[Cell, int]  = g["index"]
        self.cells:      List[Cell]       = g["cells"]
        self.out_nbrs:   List[List[int]]  = g["out"]
        self._D:         np.ndarray       = g["dist"]
        self._h_cache:   Dict[Cell, List[int]] = g.setdefault("h_cache", {})

        # 주차 칸: 도킹 셀에서 가까운 순
        self.spots_by_dist = [
            sorted(spots, key=lambda c, d=self.docks[k]: (self.aisle_distance(d, c), c))
            for k, spots in enumerate(self.park_spots)]

        self.coord_norm = max(self.n_cols, self.n_rows) * CELL_SIZE
        self.dist_norm  = (self.n_cols + self.n_rows) * CELL_SIZE

        # 연속 레이아웃의 REWARD_D_MAX(≈ 2 × 최대 구간), SOLO_RT_M(단독 왕복)에 해당하는 값
        key_pts = [self.cell_index[c] for c in set(self.access_coords)] + \
                  [self.cell_index[d] for d in self.docks]
        sub = self._D[np.ix_(key_pts, key_pts)]
        self.reward_d_max = 2.0 * float(sub.max()) * CELL_SIZE
        picks = [self.cell_index[c] for c in self.access_coords]
        docks = [self.cell_index[d] for d in self.docks]
        rt = self._D[np.ix_(docks, picks)] + self._D[np.ix_(picks, docks)].T
        self.solo_rt_m = float(rt.min(axis=0).mean()) * CELL_SIZE

    # 셀 지도

    def _build_cell_map(self):
        """셀 종류 배열과 통로 방향을 만든다."""
        aisle_w = self.aisle_w
        rack_w  = max(1, int(round(2 * Dt / CELL_SIZE)))
        wc      = int(round(Wc / CELL_SIZE))
        shelf_h = int(round(SN * Wt / CELL_SIZE))
        assert rack_w == 1, "현재 열 배치는 등을 맞댄 랙 2개가 1칸이라고 가정한다."

        # 행 구간: (종류, 시작 행, 행 수). 블록 사이 교차통로는 연속 레이아웃처럼 2·Wc.
        bands, blocks = [], []
        row = self.ws_rows                         # 아래 ws_rows행은 워크스테이션 구역
        bands.append((row, wc)); row += wc
        for b in range(BN):
            blocks.append((row, shelf_h)); row += shelf_h
            h = 2 * wc if b < BN - 1 else wc       # 맨 위 교차통로 추가
            bands.append((row, h)); row += h
        self.n_rows   = row
        self.bands    = bands
        self.blocks   = blocks
        self.shelf_h  = shelf_h

        kind = np.full((self.n_cols, self.n_rows), RACK, dtype=np.int8)
        kind[:, :self.ws_rows] = -1                # 워크스테이션 구역은 _build_ws_zone에서 채운다
        for r0, h in bands:
            kind[:, r0:r0 + h] = CROSS
        for r0, h in blocks:
            for a in range(AN):
                x0 = a * self.module + rack_w
                kind[x0:x0 + aisle_w, r0:r0 + h] = AISLE
        self.kind = kind

        # 열 -> 통로 방향 (+1 위, −1 아래, 0 랙 열)
        self.col_dir = np.zeros(self.n_cols, dtype=np.int8)
        for a in range(AN):
            x0 = a * self.module + rack_w
            self.col_dir[x0:x0 + aisle_w] = 1 if a % 2 == 0 else -1

        # 행 -> 교차통로 가로 방향 (+1 동, −1 서, 0 교차통로 아님)
        self.row_dir = np.zeros(self.n_rows, dtype=np.int8)
        for r0, h in bands:
            half = h // 2
            self.row_dir[r0:r0 + half]     = 1
            self.row_dir[r0 + half:r0 + h] = -1

    def _size_ws_zone(self, queue_cells: Optional[int], park_cells: Optional[int]):
        """열 수와 워크스테이션 구역 크기(대기열 길이, 주차장 폭/차선 수, 행 수)를 정한다."""
        aisle_w = int(round(Wa / CELL_SIZE))
        self.aisle_w = aisle_w
        self.module  = 1 + aisle_w
        self.n_cols  = AN * self.module + 1

        # 도킹 셀은 연속 레이아웃처럼 폭 방향 균등 위치에 두고, 서쪽에 대기열, 동쪽에 주차장을 둔다.
        K, M   = self.num_workstations, self.num_robots
        per_ws = math.ceil(M / K)
        span   = self.n_cols // K
        q_max, w_max = span // 2, span - span // 2 - 1
        q   = per_ws if queue_cells is None else queue_cells
        cap = M if park_cells is None else park_cells
        w   = min(w_max, cap + 2)                  # 주차장 폭 (양 끝 연결 열 포함)
        if not (1 <= q <= q_max) or w < 3 or cap < per_ws:
            raise ValueError(
                f"워크스테이션 구역이 들어가지 않습니다: WS당 폭 {span}칸, 대기열 {q} (최대 {q_max}), "
                f"주차장 폭 {w} (최소 3), 주차 {cap}칸 (초기 배치에 최소 {per_ws} 필요)")
        self.span, self.offset = span, (self.n_cols - K * span) // 2
        self.queue_len, self.lot_w = q, w
        self.n_lanes  = math.ceil(cap / (w - 2))
        self.park_cap = self.n_lanes * (w - 2)
        self.ws_rows  = 2 * self.n_lanes

    def _build_ws_zone(self):
        """워크스테이션별 대기열, 도킹 셀, 주차장(차선/주차 칸) 셀을 배치한다."""
        K, q, w, n = self.num_workstations, self.queue_len, self.lot_w, self.n_lanes
        r0 = self.ws_rows - 1                      # 교차통로 바로 아래 행

        self.docks:       List[Cell]       = []
        self.queue_cells: List[List[Cell]] = []
        self.lot_lanes:   List[List[List[Cell]]] = []   # [k][차선 i] = 서->동 셀 목록
        self.park_spots:  List[List[Cell]] = []         # [k] = 도킹 셀에서 가까운 순
        self.dock_ws:     Dict[Cell, int]  = {}
        for k in range(K):
            xd   = self.offset + k * self.span + self.span // 2
            dock = (xd, r0)
            cw, ce = xd + 1, xd + w
            self.docks.append(dock)
            self.dock_ws[dock] = k
            self.queue_cells.append([(x, r0) for x in range(xd - q, xd)])
            lanes = [[(x, r0 - 2 * i) for x in range(cw, ce + 1)] for i in range(n)]
            spots = [(x, r0 - 2 * i - 1) for i in range(n) for x in range(cw + 1, ce)]
            conns = [(x, r0 - 2 * i - 1) for i in range(n - 1) for x in (cw, ce)]
            self.lot_lanes.append(lanes)
            self.park_spots.append(spots)
            for c in self.queue_cells[k] + [dock] + sum(lanes, []) + spots + conns:
                self.kind[c] = WS_ZONE

    def _build_pick_cells(self) -> List[Cell]:
        """저장 위치마다 토트를 집는 통로 셀 (랙 바로 옆 통로 열)."""
        cells = []
        for b in range(BN):
            r0, _ = self.blocks[b]
            for a in range(AN):
                x_left  = a * self.module + 1
                x_right = x_left + self.aisle_w - 1
                for x in (x_left, x_right):
                    for r in range(SN):
                        cells.append((x, r0 + int((r + 0.5) * Wt / CELL_SIZE)))
        assert len(cells) == len(self.storage_coords)
        return cells

    # 그래프

    def _passable(self, c: Cell) -> bool:
        x, y = c
        return 0 <= x < self.n_cols and 0 <= y < self.n_rows and self.kind[x, y] in (AISLE, CROSS, WS_ZONE)

    def _edges(self) -> List[Tuple[Cell, Cell]]:
        """이동 규칙에 맞는 유향 간선 목록."""
        edges = []
        for x in range(self.n_cols):
            for y in range(self.ws_rows, self.n_rows):
                u = (x, y)
                ku = self.kind[x, y]
                if ku not in (AISLE, CROSS):
                    continue
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    v = (x + dx, y + dy)
                    if not self._passable(v) or v[1] < self.ws_rows:
                        continue
                    kv = self.kind[v]
                    if self.traffic == "free":
                        edges.append((u, v))
                        continue
                    if dy != 0:
                        # 세로: 통로가 끼면 통로 방향만, 교차통로끼리는 양방향
                        if (ku == AISLE or kv == AISLE) and dy != self.col_dir[x]:
                            continue
                    else:
                        if ku == CROSS and kv == CROSS:
                            if dx != self.row_dir[y]:
                                continue
                        elif not (ku == AISLE and kv == AISLE):
                            continue            # 통로 <-> 교차통로 가로 연결은 없음
                    edges.append((u, v))
        band = self.ws_rows                                       # 하단 교차통로 첫 행
        for k in range(self.num_workstations):
            q, dock, lanes = self.queue_cells[k], self.docks[k], self.lot_lanes[k]
            edges.append(((q[0][0], band), q[0]))                 # 대기열 입구
            chain = q + [dock, lanes[0][0]]
            edges += list(zip(chain, chain[1:]))                  # 대기열 -> 도킹 -> 주차장
            edges.append((dock, (dock[0], band)))                 # 도킹 -> 교차통로
            for i, lane in enumerate(lanes):
                if i == 0:
                    edges += list(zip(lane, lane[1:]))            # 차선 0: 동쪽
                    edges += [(c, (c[0], band)) for c in lane]    # 차선 0 -> 교차통로
                else:
                    edges += list(zip(lane[::-1], lane[-2::-1]))  # 차선 1..: 서쪽
                    (cw, y), (ce, _) = lane[0], lane[-1]
                    edges += [((ce, y + 2), (ce, y + 1)), ((ce, y + 1), (ce, y)),   # 동쪽 끝 내려감
                              ((cw, y), (cw, y + 1)), ((cw, y + 1), (cw, y + 2))]   # 서쪽 끝 올라감
                for (x, y) in lane[1:-1]:                         # 주차 칸 <-> 위 차선
                    edges += [((x, y), (x, y - 1)), ((x, y - 1), (x, y))]
        return edges

    def _build_graph(self) -> dict:
        """셀 인덱스, 이웃 목록, 전체 쌍 최단거리(셀 수) 테이블을 만든다."""
        cells = [(x, y) for x in range(self.n_cols) for y in range(self.n_rows)
                 if self._passable((x, y))]
        index = {c: i for i, c in enumerate(cells)}
        edges = self._edges()
        src = [index[u] for u, _ in edges]
        dst = [index[v] for _, v in edges]
        n   = len(cells)
        G   = csr_matrix((np.ones(len(edges)), (src, dst)), shape=(n, n))
        D   = shortest_path(G, directed=True, unweighted=True)
        if np.isinf(D).any():
            bad = cells[int(np.argwhere(np.isinf(D))[0][0])]
            raise RuntimeError(f"셀 그래프가 강연결이 아닙니다 (예: {bad}에서 도달 불가 셀 존재)")
        out = [[] for _ in range(n)]
        for s, d in zip(src, dst):
            out[s].append(d)
        return {"cells": cells, "index": index, "out": out, "dist": D.astype(np.int16)}

    # 거리 / 이동

    def aisle_distance(self, p1: Cell, p2: Cell) -> float:
        """p1에서 p2까지 유향 셀 그래프 최단거리 (m)."""
        return float(self._D[self.cell_index[p1], self.cell_index[p2]]) * CELL_SIZE

    def next_hops(self, u: Cell, goal: Cell) -> List[Cell]:
        """u에서 goal 쪽 최단경로상의 다음 셀 후보 (거리가 1 줄어드는 이웃)."""
        iu, ig = self.cell_index[u], self.cell_index[goal]
        d = self._D[iu, ig] - 1
        col = self._D[:, ig]
        return [self.cells[w] for w in self.out_nbrs[iu] if col[w] == d]

    def detour_path(self, u: Cell, goal: Cell, blocked: Set[Cell],
                    max_len: int) -> Optional[List[Cell]]:
        """blocked 셀을 피해 u -> goal로 가는 최단경로 (u 제외). max_len 칸을 넘으면 None."""
        iu, ig = self.cell_index[u], self.cell_index[goal]
        bl = {self.cell_index[c] for c in blocked}
        bl.discard(ig)
        prev = {iu: -1}
        dq = deque([(iu, 0)])
        while dq:
            v, dv = dq.popleft()
            if v == ig:
                path = []
                while v != iu:
                    path.append(self.cells[v])
                    v = prev[v]
                return path[::-1]
            if dv >= max_len:
                continue
            for w in self.out_nbrs[v]:
                if w not in prev and w not in bl:
                    prev[w] = v
                    dq.append((w, dv + 1))
        return None

    def dist_to(self, goal: Cell) -> List[int]:
        """셀 인덱스 -> goal까지 정적 최단거리(셀 수) 목록. 경로 탐색 휴리스틱용으로 캐시한다."""
        h = self._h_cache.get(goal)
        if h is None:
            h = self._D[:, self.cell_index[goal]].tolist()
            self._h_cache[goal] = h
        return h

    def is_dock(self, c: Cell) -> bool:
        return c in self.dock_ws

    def initial_positions(self, num_robots: int) -> List[Cell]:
        """로봇 rid를 워크스테이션 rid % K의 주차 칸에 도킹 셀에서 가까운 순으로 배치한다."""
        used = [0] * self.num_workstations
        pos = []
        for rid in range(num_robots):
            k = rid % self.num_workstations
            pos.append(self.spots_by_dist[k][used[k]])
            used[k] += 1
        return pos
