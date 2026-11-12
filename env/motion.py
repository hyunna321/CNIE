"""로봇 이동 엔진: 셀 단위 틱으로 로봇을 움직이는 두 방식과 시공간 예약표

충돌 처리는 layout.motion으로 선택
  - "reactive": 매 틱 다음 셀이 비어 있으면 이동, 막히면 대기/우회 (MotionEngine._reactive_tick)
  - "reserve": 로봇의 경로가 바뀔 때(배정, 삽입, 재출발) 그 로봇의 트립 전체를 시공간 경로로 계획해
    예약표에 올리고, 매 틱 예약대로 이동 (MotionEngine._reserve_tick, _reserve_plan)

예약표 (Prioritized Planning + SIPP): 로봇 경로를 "틱 t에 어느 셀"의 목록으로 계획해 등록
나중에 계획하는 로봇은 이미 등록된 예약을 피해 경로를 짜므로 계획 순서가 곧 우선순위
  - 같은 틱, 같은 셀에는 한 로봇만 (vertex conflict)
  - 같은 틱에 두 로봇의 자리 맞바꾸기 금지 (swap conflict)
  - 계획이 끝난 로봇은 마지막 셀에 무기한 정지 (rest), 이후 그 셀은 누구도 사용 불가
셀은 GridWarehouseLayout.cell_index의 정수 인덱스로 표현
"""
import heapq
import math
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

from env.layout import Cell, CELL_SIZE
from env.orders import Tote
from env.robot import Robot

GRID_REPLAN_WAIT      = 3       # 틱, 이만큼 막히면 점유 셀을 피하는 우회 경로 탐색
GRID_DETOUR_MAX_EXTRA = 10      # 셀, 최단거리 대비 허용하는 최대 우회 길이
GRID_DEADLOCK_TICKS   = 1800    # 틱, 이동 대기 로봇이 있는데 아무도 못 움직이면 교착으로 보고 중단
RESERVE_HORIZON_EXTRA = 300     # 틱, 구간 경로 탐색에서 최단거리 대비 허용하는 최대 지연
RESERVE_MAX_EXPAND    = 60000   # 구간 경로 탐색 1회의 최대 상태 확장 수 (넘으면 계획 실패)


def to_ticks(seconds: float, tick: float) -> int:
    """머무는 시간(s)을 틱 수로 올림"""
    return int(math.ceil(seconds / tick - 1e-9)) if seconds > 0.0 else 0


class ReservationTable:
    """(셀, 틱) -> 로봇 예약과 셀별 무기한 점유(rest) 관리"""

    def __init__(self):
        self.cell_res: Dict[int, Dict[int, int]] = {}         # 셀 -> {틱: 로봇}
        self.rest:     Dict[int, Tuple[int, int]] = {}        # 셀 -> (시작 틱, 로봇)
        self.robot_res:  Dict[int, List[Tuple[int, int]]] = {}   # 로봇 -> [(셀, 틱)]
        self.robot_rest: Dict[int, Tuple[int, int]] = {}         # 로봇 -> (셀, 시작 틱)

    # 조회

    def owner(self, c: int, t: int) -> Optional[int]:
        """틱 t에 셀 c를 쓰는 로봇 (없으면 None)"""
        r = self.cell_res.get(c, {}).get(t)
        if r is not None:
            return r
        rs = self.rest.get(c)
        if rs is not None and t >= rs[0]:
            return rs[1]
        return None

    def can_enter(self, u: int, v: int, t: int, rid: int) -> bool:
        """rid가 틱 t-1의 셀 u에서 틱 t에 셀 v로 갈 수 있는지 (u == v이면 제자리 대기)"""
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
        """셀 c가 틱 t부터 무기한 rid 외에 아무도 쓰지 않는지"""
        rs = self.rest.get(c)
        if rs is not None and rs[1] != rid:
            return False
        for tt, r in self.cell_res.get(c, {}).items():
            if tt >= t and r != rid:
                return False
        return True

    # 갱신

    def commit(self, rid: int, t0: int, start: int, cells: List[int], rest_cell: int):
        """rid의 출발 셀(틱 t0), 경로(틱 t0+1부터 cells 순서), 그 뒤 rest_cell 무기한 점유 예약"""
        lst = self.robot_res.setdefault(rid, [])
        self.cell_res.setdefault(start, {})[t0] = rid
        lst.append((start, t0))
        for i, c in enumerate(cells):
            t = t0 + 1 + i
            self.cell_res.setdefault(c, {})[t] = rid
            lst.append((c, t))
        t_rest = t0 + len(cells)
        self.rest[rest_cell] = (t_rest, rid)
        self.robot_rest[rid] = (rest_cell, t_rest)

    def set_rest(self, rid: int, c: int, t: int):
        """rid의 틱 t부터 셀 c 무기한 점유 설정 (초기 배치용)"""
        self.rest[c] = (t, rid)
        self.robot_rest[rid] = (c, t)

    def clear_after(self, rid: int, t: int) -> Tuple[List[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """rid의 틱 t 이후 예약과 rest를 지우고 지운 내용 반환 (되돌리기용)"""
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
        """clear_after로 지운 예약 복원"""
        lst = self.robot_res.setdefault(rid, [])
        for c, t in removed:
            self.cell_res.setdefault(c, {})[t] = rid
            lst.append((c, t))
        if rest is not None:
            self.rest[rest[0]] = (rest[1], rid)
            self.robot_rest[rid] = rest

    def prune(self, t: int):
        """틱 t 이전 예약 삭제 (메모리 정리)"""
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
    """셀 c가 틱 t_lo 이후 rid 외 로봇에게 점유되지 않는 최대 연속 구간 [s, e] 목록 (e는 inf 가능)"""
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
    """SIPP(Safe Interval Path Planning)로 start(틱 t0) -> goal 구간 경로 탐색

    - 상태 = (셀, 그 셀이 비어 있는 시간 구간)이라 기다림이 길어도 상태 수가 늘지 않음
    - 각 상태에 가장 이른 도착 틱을 기록, 이동 시 다음 셀의 안전 구간에 맞춰 출발 셀에서 대기
    - goal 도착 틱 ta부터 ta + dwell까지 goal에 머물 수 있어야 함 (final이면 도착 이후 무기한)
    - 셀에 들어서는 순간이 도착이므로 경로는 도중에 goal을 지나지 않음

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
            return None
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
                    break                           # 구간은 시간 순이라 뒤 구간은 더 늦음
                if ta > ve:
                    continue
                # 맞바꾸기 금지: 다른 로봇이 ta-1에 v, ta에 c에 있으면 한 틱 늦춤
                while ta <= ve and ta <= c_end + 1:
                    o = cell_res.get(v, {}).get(ta - 1)
                    if o is None or o == rid or cell_res.get(c, {}).get(ta) != o:
                        break
                    ta += 1
                if ta > ve or ta > c_end + 1:
                    continue
                if v == goal and (ve != INF if final else ve < ta + dwell):
                    continue                        # goal은 머물 수 있는 구간에만 진입
                key = (v, vi)
                if ta < best.get(key, INF):
                    best[key] = ta
                    parent[key] = (c, ci)
                    heapq.heappush(open_, (ta + h[v], ta, v, vi))
    return None


def _can_stay(table: ReservationTable, rid: int, c: int, ta: int, dwell: int, final: bool) -> bool:
    """틱 ta에 셀 c에 도착한 뒤 dwell틱 (final이면 무기한) 머물 수 있는지"""
    for t in range(ta + 1, ta + dwell + 1):
        if not table.can_enter(c, c, t, rid):
            return False
    if final:
        return table.free_from(c, ta + dwell, rid)
    return True


class MotionEngine:
    """SimulatorBase의 이동 부분 (self.robots, self.layout, self._robot_totes 등은 SimulatorBase가 소유)"""

    def _wants_move(self, r: Robot) -> bool:
        """트립 중이고 목적지로 이동해야 하는 로봇인지"""
        return r.timer == 0.0 and r.current_dest is not None and r.pos != r.current_dest

    def _can_park(self, r: Robot) -> bool:
        """idle 로봇이 예약한 주차 칸으로 가는 중인지"""
        return (r.is_idle and r.current_dest is None
                and r.park_spot is not None and r.pos != r.park_spot)

    def _assign_spot(self, r: Robot):
        """트립을 마친 idle 로봇에 현재 워크스테이션의 빈 주차 칸 중 도킹 셀에서 가장 가까운 칸 예약

        빈 칸이 없으면 (WS_PARK_CELLS < M일 때만 가능) 도킹 셀에 그대로 머묾
        """
        for c in self.layout.spots_by_dist[r.workstation_id]:
            if c not in self._spots_taken:
                r.park_spot = c
                self._spots_taken.add(c)
                return

    def _release_spot(self, r: Robot):
        """r의 주차 칸 예약 해제"""
        if r.park_spot is not None:
            self._spots_taken.discard(r.park_spot)
            r.park_spot = None

    def _motion_active(self) -> bool:
        """다음 틱에 진행할 일이 있는지 (이동, 피킹/처리 타이머, 주차 칸으로 이동, 재계획 대기)"""
        if self.reserve:
            return any(r.plan or r.timer > 0.0 or r.replan_pending for r in self.robots)
        return any(r.timer > 0.0 or self._wants_move(r) or self._can_park(r)
                   for r in self.robots)

    def _cell_free(self, cell) -> bool:
        return (not self.layout.collisions) or cell not in self._occ

    def _next_cell(self, r: Robot):
        """이번 틱에 r이 들어갈 셀. 막혀 있으면 None.

        최단경로상 다음 셀 후보 중 비어 있는 것을 선택
        GRID_REPLAN_WAIT 틱 이상 막혔으면 점유 셀을 피하는 우회 경로 탐색 (최단 대비 GRID_DETOUR_MAX_EXTRA칸 이내)
        """
        L = self.layout
        goal = r.current_dest if r.current_dest is not None else r.park_spot
        if r.detour:
            if self._cell_free(r.detour[0]):
                return r.detour.pop(0)
            r.detour = []
        hops = L.next_hops(r.pos, goal)
        for c in hops:
            if self._cell_free(c):
                return c
        if r.wait_ticks + 1 >= GRID_REPLAN_WAIT and goal not in hops:
            base = L.distance(r.pos, goal) / CELL_SIZE
            path = L.detour_path(r.pos, goal, set(self._occ),
                                 max_len=int(base) + GRID_DETOUR_MAX_EXTRA)
            if path:
                self._n_replans += 1
                r.detour = path
                return r.detour.pop(0)
        return None

    def _reactive_tick(self):
        """한 틱 진행: 타이머 감소 -> 이동 -> 도착 처리 -> 교착 감시"""
        dt = self.layout.tick
        done = self._run_timers()
        movers  = [r for r in self.robots
                   if r.id not in done and (self._wants_move(r) or self._can_park(r))]
        moved   = set()
        changed = True
        # 앞 로봇이 비운 셀로 뒤 로봇이 같은 틱에 따라 들어갈 수 있도록 변화가 없을 때까지 반복
        while changed:
            changed = False
            for r in movers:
                if r.id in moved:
                    continue
                nxt = self._next_cell(r)
                if nxt is None:
                    continue
                if self.layout.collisions:
                    del self._occ[r.pos]
                    self._occ[nxt] = r.id
                r.pos = nxt
                self._total_sim_distance += CELL_SIZE
                if self.battery is not None:
                    self.battery.on_move(r)
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
                self._on_arrive(r)

        if moved:
            self._last_move_time = self.now
        elif (self.now - self._last_move_time > GRID_DEADLOCK_TICKS * dt
              and not any(r.timer > 0.0 for r in self.robots)
              and any(self._wants_move(r) for r in self.robots)):
            stuck = [(r.id, r.pos, r.current_dest) for r in self.robots if self._wants_move(r)]
            raise RuntimeError(f"그리드 교착: t={self.now:.0f}s, {GRID_DEADLOCK_TICKS}틱 동안 이동 없음. "
                               f"(로봇, 위치, 목적지) = {stuck}")

    def _run_timers(self) -> set:
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
                        self._finish_drop(r)
                    self._after_node(r)
        return done

    def _next_pick(self, r: Robot, loc, skip: int = 0) -> Optional[Tote]:
        """r이 loc에서 아직 집지 않은 토트 중 배정 순서로 (skip+1)번째"""
        for t in self._robot_totes[r.id]:
            if not t.picked and t.loc == loc:
                if skip == 0:
                    return t
                skip -= 1
        return None

    def _on_arrive(self, r: Robot):
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
                self._finish_drop(r)
        else:
            t = self._next_pick(r, r.pos)
            if t is not None:
                t.picked = True
                if t.pick_s > 0.0:
                    r.timer, r.timer_kind = t.pick_s, "pick"
                    return
        self._after_node(r)

    def _finish_drop(self, r: Robot):
        """도킹 셀에서 처리를 마친 토트 완료 기록 (주문의 마지막 토트면 주문 완료)"""
        self._trip_totes[r.id] += len(self._dropping[r.id])
        for t in self._dropping[r.id]:
            self._ws_load[t.ws_id] -= 1
            if self.order_mgr.record_tote_done(t, self.now):
                o = t.order
                self._n_in_system -= 1
                self._completions.append((self.now, o.completion_time - o.arrival_time))
        self._dropping[r.id] = []

    def _after_node(self, r: Robot):
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
        self._finish_trip(r)

    def _finish_trip(self, r: Robot):
        """마지막 도킹 셀 처리 후 트립 완료 (대기 토트가 있으면 바로 새 트립)

        새 트립이 없으면 idle 로봇은 주차 칸으로 이동
        """
        if r.pending_insert:
            raise RuntimeError(f"로봇 {r.id}: 삽입 대기 토트 {r.pending_insert}이 남은 채 트립 완료 (t={self.now:.0f})")
        self._on_trip_complete(r)
        if r.is_idle and r.current_dest is None and not self.reserve:
            self._assign_spot(r)

    # 예약 모드 (motion="reserve")

    def _reserve_tick(self, k: int):
        """예약 모드 한 틱: 타이머 -> 예약대로 이동 -> 도착 처리 -> 재계획 대기 로봇 재시도"""
        dt = self.layout.tick
        self._run_timers()

        moved = []
        for r in self.robots:
            if r.plan and r.plan[0][0] == k:
                _, cell = r.plan.popleft()
                if cell != r.pos:
                    if r.timer > 0.0:
                        raise RuntimeError(f"예약 불일치: 로봇 {r.id}가 {r.timer_kind} 중인데 {r.pos}->{cell} 이동 계획 (t={self.now:.0f})")
                    r.pos = cell
                    self._total_sim_distance += CELL_SIZE
                    if self.battery is not None:
                        self.battery.on_move(r)
                    moved.append(r)
                    continue
            elif r.plan and r.plan[0][0] < k:
                raise RuntimeError(f"예약 불일치: 로봇 {r.id}의 계획 틱 {r.plan[0][0]} < 현재 틱 {k}")
            if r.current_dest is not None and r.timer == 0.0 and r.pos != r.current_dest:
                self._blocked_time += dt                # 예약 때문에 제자리에서 기다린 시간
        for r in moved:
            if r.current_dest is not None and r.pos == r.current_dest:
                self._on_arrive(r)
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
        stay0  = to_ticks(r.timer, L.tick) if r.timer > 0.0 else 0
        visits = ([r.current_dest] if r.current_dest is not None and r.timer == 0.0 else []) + r.stops
        held   = [t for t in self._robot_totes[r.id] if t.picked]
        seen: Dict[Cell, int] = {}
        legs: List[Tuple[Cell, int]] = []
        for c in visits:
            if L.is_dock(c):
                k = L.dock_ws[c]
                d = to_ticks(sum(t.proc_s for t in held if t.ws_id == k), L.tick)
                held = [t for t in held if t.ws_id != k]
                legs.append((c, d))
                continue
            t = self._next_pick(r, c, seen.get(c, 0))
            seen[c] = seen.get(c, 0) + 1
            d = 0
            if t is not None:
                d = to_ticks(t.pick_s, L.tick)
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
