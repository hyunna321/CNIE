"""그리드 시뮬레이션 기록기: 매 틱 로봇 위치/상태, 경로·예약 변경, 주문 수명, 지표, 혼잡 히트맵

SimulatorBase.recorder에 붙이면 시뮬레이터가 on_reset / on_tick / on_plan을 호출
"""
import base64
from collections import deque
from array import array
from typing import Dict, List, Optional

import numpy as np

# 로봇 상태 코드 (뷰어와 공유)
ST_PARKED, ST_TO_PARK, ST_MOVING, ST_WAITING, ST_PICKING, ST_WS = range(6)
STATE_NAMES = ["parked", "to_park", "moving", "waiting", "picking", "ws_processing"]

METRIC_EVERY = 10   # 틱, 시계열 지표 기록 간격


class Recorder:
    """매 틱 상태를 바이트 배열로 모으고, 끝나면 뷰어용 dict로 묶음"""

    def on_reset(self, sim):
        L = sim.layout
        self.M        = sim.num_robots
        self.dt       = L.tick
        self.frames   = bytearray()          # 틱마다 로봇별 [x, y, 상태, 적재 수]
        self.last_k   = -1
        self.prev_pos = [r.pos for r in sim.robots]
        self.route_ev: List[list] = []       # [틱, 로봇, [방문 셀 인덱스...]]
        self.plan_ev:  List[list] = []       # [시작 틱, 로봇, [셀 인덱스...]]
        self.assigned: Dict[int, int] = {}  # 토트 id -> 로봇
        self.metrics = {k: array("f") for k in
                        ("t", "in_system", "queue", "busy", "waiting", "completed", "avg_flow")}
        self.occ  = np.zeros(len(L.cells), dtype=np.int32)
        self.wait = np.zeros(len(L.cells), dtype=np.int32)
        self._route_key = [None] * self.M
        self._n_done    = 0                                  # sim._completions에서 읽은 개수
        self._recent    = deque()                            # (완료 시각, 흐름시간), 최근 10분
        self._append_frame(sim, 0, moved=None)

    # 시뮬레이터 훅

    def on_tick(self, sim, k: int):
        # 아무 일도 없어 건너뛴 틱은 직전 프레임으로 채움
        if k > self.last_k + 1:
            last = self.frames[-self.M * 4:]
            for _ in range(k - self.last_k - 1):
                self.frames += last
            self.last_k = k - 1
        moved = [r.pos != p for r, p in zip(sim.robots, self.prev_pos)]
        self._append_frame(sim, k, moved)
        self.prev_pos = [r.pos for r in sim.robots]

    def on_plan(self, sim, r, t0: int, cells: List[int]):
        self.plan_ev.append([t0, r.id, list(cells)])

    # 내부

    def _append_frame(self, sim, k: int, moved: Optional[list]):
        L = sim.layout
        n_wait = 0
        for i, r in enumerate(sim.robots):
            if r.timer_kind == "pick":
                st = ST_PICKING
            elif r.timer_kind == "ws":
                st = ST_WS
            elif r.current_dest is None:
                st = ST_PARKED if (moved is None or not moved[i]) else ST_TO_PARK
            elif moved is not None and not moved[i] and r.pos != r.current_dest:
                st = ST_WAITING
            else:
                st = ST_MOVING
            if st == ST_WAITING:
                n_wait += 1
                self.wait[L.cell_index[r.pos]] += 1
            self.occ[L.cell_index[r.pos]] += 1
            totes = sim._robot_totes.get(r.id, [])
            self.frames += bytes((r.pos[0], r.pos[1], st, min(255, len(totes))))
            # 경로 변경 기록
            key = (r.current_dest, tuple(r.stops), r.workstation)
            if key != self._route_key[i]:
                self._route_key[i] = key
                visits = ([r.current_dest] if r.current_dest is not None else []) + r.stops
                self.route_ev.append([k, r.id, [L.cell_index[c] for c in visits], L.cell_index[r.workstation]])
            for t in totes:
                if t.tote_id not in self.assigned:
                    self.assigned[t.tote_id] = r.id
        comp = sim._completions
        self._recent.extend(comp[self._n_done:])
        self._n_done = len(comp)
        self.last_k = k
        if k % METRIC_EVERY == 0:
            om = sim.order_mgr
            done = om.completed_count()
            m = self.metrics
            m["t"].append(k * self.dt)
            m["in_system"].append(sim._n_in_system)
            m["queue"].append(om.queue_size())
            m["busy"].append(sum(1 for r in sim.robots if not r.is_idle))
            m["waiting"].append(n_wait)
            m["completed"].append(done)
            # 최근 10분 완료 주문의 평균 흐름시간
            while self._recent and self._recent[0][0] < k * self.dt - 600:
                self._recent.popleft()
            rec = self._recent
            m["avg_flow"].append(sum(f for _, f in rec) / len(rec) if rec else float("nan"))

    def result(self, sim) -> dict:
        """뷰어로 보낼 기록 (위치 프레임은 base64 바이트)

        orders: 토트마다 [주문 id, 도착, 배정, WS 처리 완료, 피킹 셀, SKU, 로봇, WS] (도착 순)
        """
        L = sim.layout
        r2 = lambda v: None if v is None else round(v, 2)
        orders = [[t.order.order_id, round(t.order.arrival_time, 2), r2(t.assign_time), r2(t.done_time),
                   L.cell_index[t.loc], t.sku, self.assigned.get(t.tote_id, -1), t.ws_id]
                  for t in sim.order_mgr.all_totes]
        return {
            "n_ticks": self.last_k + 1, "dt": self.dt, "M": self.M,
            "frames": base64.b64encode(bytes(self.frames)).decode(),
            "route_ev": self.route_ev, "plan_ev": self.plan_ev,
            "orders": orders,
            "metrics": {k: [None if (isinstance(x, float) and x != x) else round(x, 2) for x in v]
                        for k, v in self.metrics.items()},
            "heat_occ": self.occ.tolist(), "heat_wait": self.wait.tolist(),
            "state_names": STATE_NAMES,
        }
