"""그리드 이동 모드의 시공간 예약표와 우선순위 기반 경로계획 (Prioritized Planning + SIPP).

로봇의 경로를 "틱 t에 어느 셀"의 목록으로 계획해 예약표에 올린다. 나중에 계획하는
로봇은 이미 올라간 예약을 피해 경로를 짜므로, 계획 순서가 곧 우선순위다.

예약 규칙:
  - 같은 틱에 같은 셀에는 한 로봇만 있을 수 있다 (vertex conflict).
  - 같은 틱에 두 로봇이 서로 자리를 맞바꿀 수 없다 (swap conflict).
  - 계획이 끝난 로봇은 마지막 셀에 무기한 머문다 (rest). 그 셀은 이후 누구도 쓸 수 없다.

셀은 GridWarehouseLayout.cell_index의 정수 인덱스로 다룬다.
"""
import heapq
from typing import Dict, List, Optional, Tuple

from constant import RESERVE_HORIZON_EXTRA, RESERVE_MAX_EXPAND


class ReservationTable:
    """(셀, 틱) -> 로봇 예약과 셀별 무기한 점유(rest)를 관리한다."""

    def __init__(self):
        self.cell_res: Dict[int, Dict[int, int]] = {}         # 셀 -> {틱: 로봇}
        self.rest:     Dict[int, Tuple[int, int]] = {}        # 셀 -> (시작 틱, 로봇)
        self.robot_res:  Dict[int, List[Tuple[int, int]]] = {}   # 로봇 -> [(셀, 틱)]
        self.robot_rest: Dict[int, Tuple[int, int]] = {}         # 로봇 -> (셀, 시작 틱)

    # 조회

    def owner(self, c: int, t: int) -> Optional[int]:
        """틱 t에 셀 c를 쓰는 로봇 (없으면 None)."""
        r = self.cell_res.get(c, {}).get(t)
        if r is not None:
            return r
        rs = self.rest.get(c)
        if rs is not None and t >= rs[0]:
            return rs[1]
        return None

    def can_enter(self, u: int, v: int, t: int, rid: int) -> bool:
        """rid가 틱 t-1의 셀 u에서 틱 t에 셀 v로 갈 수 있는지 (u == v이면 제자리 대기)."""
        o = self.owner(v, t)
        if o is not None and o != rid:
            return False
        if u != v:
            # 맞바꾸기: 다른 로봇이 t-1에 v에 있다가 t에 u로 오는 경우
            o2 = self.cell_res.get(v, {}).get(t - 1)
            if o2 is not None and o2 != rid and self.cell_res.get(u, {}).get(t) == o2:
                return False
        return True

    def free_from(self, c: int, t: int, rid: int) -> bool:
        """셀 c가 틱 t부터 무기한 rid 외에 아무도 쓰지 않는지."""
        rs = self.rest.get(c)
        if rs is not None and rs[1] != rid:
            return False
        for tt, r in self.cell_res.get(c, {}).items():
            if tt >= t and r != rid:
                return False
        return True

    # 갱신

    def commit(self, rid: int, t0: int, cells: List[int], rest_cell: int):
        """rid의 경로(틱 t0+1부터 cells 순서)와 그 뒤 rest_cell 무기한 점유를 예약한다."""
        lst = self.robot_res.setdefault(rid, [])
        for i, c in enumerate(cells):
            t = t0 + 1 + i
            self.cell_res.setdefault(c, {})[t] = rid
            lst.append((c, t))
        t_rest = t0 + len(cells)
        self.rest[rest_cell] = (t_rest, rid)
        self.robot_rest[rid] = (rest_cell, t_rest)

    def set_rest(self, rid: int, c: int, t: int):
        """rid가 틱 t부터 셀 c에 무기한 머문다 (초기 배치용)."""
        self.rest[c] = (t, rid)
        self.robot_rest[rid] = (c, t)

    def clear_after(self, rid: int, t: int) -> Tuple[List[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """rid의 틱 t 이후 예약과 rest를 지우고, 되돌리기용으로 지운 내용을 돌려준다."""
        keep, removed = [], []
        for c, tt in self.robot_res.get(rid, []):
            if tt > t:
                d = self.cell_res.get(c)
                if d is not None and d.get(tt) == rid:
                    del d[tt]
                removed.append((c, tt))
            else:
                keep.append((c, tt))
        self.robot_res[rid] = keep
        rest = self.robot_rest.pop(rid, None)
        if rest is not None and self.rest.get(rest[0], (None, None))[1] == rid:
            del self.rest[rest[0]]
        return removed, rest

    def restore(self, rid: int, removed: List[Tuple[int, int]], rest: Optional[Tuple[int, int]]):
        """clear_after로 지운 예약을 되돌린다."""
        lst = self.robot_res.setdefault(rid, [])
        for c, t in removed:
            self.cell_res.setdefault(c, {})[t] = rid
            lst.append((c, t))
        if rest is not None:
            self.rest[rest[0]] = (rest[1], rid)
            self.robot_rest[rid] = rest

    def prune(self, t: int):
        """틱 t 이전 예약을 지워 메모리를 정리한다."""
        for c in list(self.cell_res):
            d = self.cell_res[c]
            for tt in [tt for tt in d if tt < t]:
                del d[tt]
            if not d:
                del self.cell_res[c]
        for rid, lst in self.robot_res.items():
            self.robot_res[rid] = [(c, tt) for c, tt in lst if tt >= t]


INF = float("inf")


def _safe_intervals(table: ReservationTable, c: int, t_lo: int, rid: int) -> List[Tuple[int, float]]:
    """셀 c가 틱 t_lo 이후 rid 외 로봇에게 점유되지 않는 최대 연속 구간 [s, e] 목록 (e는 inf 가능)."""
    occ = sorted(t for t, r in table.cell_res.get(c, {}).items() if t >= t_lo and r != rid)
    rs = table.rest.get(c)
    rest_t = max(rs[0], t_lo) if (rs is not None and rs[1] != rid) else INF
    out, s = [], t_lo
    for t in occ:
        if t >= rest_t:
            break
        if t > s:
            out.append((s, t - 1))
        s = t + 1
    if s < rest_t:
        out.append((s, rest_t - 1 if rest_t != INF else INF))
    return out


def plan_leg(table: ReservationTable, out_nbrs: List[List[int]], h: List[int],
             rid: int, start: int, t0: int, goal: int, dwell: int,
             final: bool) -> Optional[Tuple[List[int], int]]:
    """SIPP(Safe Interval Path Planning)로 start(틱 t0) -> goal 구간 경로를 찾는다.

    상태를 (셀, 그 셀이 비어 있는 시간 구간)으로 잡아, 기다림이 길어도 상태 수가 늘지 않는다.
    각 상태에는 가장 이른 도착 틱을 기록하고, 이동 시 다음 셀의 안전 구간에 맞춰 출발 셀에서
    기다린다. goal에 도착한 틱 ta부터 ta + dwell까지 goal에 머물 수 있어야 하고, final이면
    도착 이후 무기한 머물 수 있어야 한다. 경로는 도중에 goal을 지나가지 않는다
    (시뮬레이터는 로봇이 current_dest 셀에 들어서는 순간을 도착으로 처리하므로).

    입력:
        table: 예약표
        out_nbrs: 셀 인덱스 -> 나가는 이웃 셀 인덱스 목록
        h: 셀 인덱스 -> goal까지 정적 최단거리 (휴리스틱)
        rid: 계획하는 로봇
        start, t0: 출발 셀과 틱
        goal: 목표 셀
        dwell: goal에서 머무는 틱 수 (피킹/처리)
        final: True이면 goal에서 무기한 머무름
    출력:
        (틱 t0+1부터의 셀 목록 (머무는 틱 포함), 구간 종료 틱) 또는 실패 시 None
    """
    if start == goal:
        if not _can_stay(table, rid, goal, t0, dwell, final):
            return None                             # 이미 도착한 셀에서 머물 수 없으면 다음 틱에 재시도
        return [goal] * dwell, t0 + dwell
    t_max = t0 + h[start] + RESERVE_HORIZON_EXTRA
    ivs: Dict[int, List[Tuple[int, float]]] = {}

    def intervals(c):
        v = ivs.get(c)
        if v is None:
            v = ivs[c] = _safe_intervals(table, c, t0, rid)
        return v

    s_iv = [i for i, (a, b) in enumerate(intervals(start)) if a <= t0 <= b]
    if not s_iv:
        return None
    start_state = (start, s_iv[0])
    best = {start_state: t0}
    parent = {start_state: None}
    open_ = [(t0 + h[start], t0, start, s_iv[0])]
    expand = 0
    cell_res = table.cell_res
    while open_:
        _, t, c, ci = heapq.heappop(open_)
        if best.get((c, ci), INF) < t:
            continue
        if c == goal:
            # 경로 복원: 부모 셀에서 (도착 틱 - 1)까지 기다렸다가 이동
            seq, node, ta = [], (c, ci), t
            while parent[node] is not None:
                pnode = parent[node]
                tp = best[pnode]
                seq.append([node[0]] + [pnode[0]] * (ta - 1 - tp))
                node, ta = pnode, tp
            path = []
            for chunk in reversed(seq):
                path += list(reversed(chunk))
            path += [goal] * dwell
            return path, t + dwell
        expand += 1
        if expand > RESERVE_MAX_EXPAND:
            return None
        c_end = intervals(c)[ci][1]
        for v in out_nbrs[c]:
            for vi, (vs, ve) in enumerate(intervals(v)):
                ta = max(t + 1, vs)
                if ta > c_end + 1 or ta > t_max:
                    break                           # 구간은 시간 순이라 뒤 구간은 더 늦다
                if ta > ve:
                    continue
                # 맞바꾸기 금지: 다른 로봇이 ta-1에 v, ta에 c에 있으면 한 틱 늦춘다
                while ta <= ve and ta <= c_end + 1:
                    o = cell_res.get(v, {}).get(ta - 1)
                    if o is None or o == rid or cell_res.get(c, {}).get(ta) != o:
                        break
                    ta += 1
                if ta > ve or ta > c_end + 1:
                    continue
                if v == goal and (ve != INF if final else ve < ta + dwell):
                    continue                        # goal은 머물 수 있는 구간에만 들어간다
                key = (v, vi)
                if ta < best.get(key, INF):
                    best[key] = ta
                    parent[key] = (c, ci)
                    heapq.heappush(open_, (ta + h[v], ta, v, vi))
    return None


def _can_stay(table: ReservationTable, rid: int, c: int, ta: int, dwell: int, final: bool) -> bool:
    """틱 ta에 셀 c에 도착한 뒤 dwell틱 (final이면 무기한) 머물 수 있는지."""
    for t in range(ta + 1, ta + dwell + 1):
        if not table.can_enter(c, c, t, rid):
            return False
    if final:
        return table.free_from(c, ta + dwell, rid)
    return True
