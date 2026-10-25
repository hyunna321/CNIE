"""그리드 시뮬레이션 기록기: 매 틱 로봇 위치/상태, 경로·예약 변경, 주문 수명, 지표, 혼잡 히트맵, RL 결정

SimulatorBase.recorder에 붙이면 시뮬레이터가 on_reset / on_tick / on_plan을 호출 (RLSimulator는 on_rl_decision도)
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

    def __init__(self, record_decisions: bool = True):
        self.record_decisions = record_decisions

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
        self.decisions: List[dict] = []
        self.agreement = DecisionStats()
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

    def on_rl_decision(self, sim, tote, eligible, cand_feats, idx, deltas):
        """RL 결정 1건: 후보별 Q값, 삽입 detour, BL6 한계비용, 선택한 로봇"""
        self.agreement.on_rl_decision(sim, tote, eligible, cand_feats, idx, deltas)
        if not self.record_decisions:
            return
        import torch
        with torch.no_grad():
            q = sim.agent.policy_net(torch.tensor(cand_feats)).numpy().reshape(-1)
        mc = self.agreement.last_mc
        self.decisions.append({
            "t": round(sim.now, 2), "tote": tote.tote_id, "order": tote.order.order_id,
            "loc": sim.layout.cell_index[tote.loc], "ws": tote.ws_id,
            "cands": [{"robot": r.id, "q": round(float(q[i]), 4), "dd": round(deltas[i], 1),
                       "mc": round(mc[i], 1), "idle": bool(r.is_idle), "load": len(sim._robot_totes[r.id])}
                      for i, r in enumerate(eligible)],
            "chosen": eligible[idx].id,
            "bl6": eligible[int(np.argmin(mc))].id,
            "queue": sim.order_mgr.queue_size(),
        })

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
            "orders": orders, "decisions": self.decisions,
            "metrics": {k: [None if (isinstance(x, float) and x != x) else round(x, 2) for x in v]
                        for k, v in self.metrics.items()},
            "heat_occ": self.occ.tolist(), "heat_wait": self.wait.tolist(),
            "state_names": STATE_NAMES, "bl6_agreement": self.agreement.summary(),
        }


class DecisionStats:
    """RL 결정과 BL6 argmin(한계비용 최소 후보)의 일치율만 집계하는 가벼운 기록기

    SimulatorBase.recorder에 붙이면 on_reset / on_tick / on_plan은 무시하고 on_rl_decision만 집계
    동률(한계비용 차 < tie_tol)인 후보를 골라도 일치로 셈
    """

    def __init__(self, tie_tol: float = 1e-6):
        self.tie_tol = tie_tol
        self._clear()

    def _clear(self):
        self.n = self.agree = 0
        self.n_idle = self.agree_idle = 0      # BL6 argmin이 idle 로봇인 결정
        self.regret = 0.0                       # Σ (선택 후보 한계비용 − 최소 한계비용), tote·s
        self.n_cands = 0
        self.last_mc: List[float] = []

    def on_reset(self, sim):
        self._clear()

    def on_tick(self, sim, k: int):
        pass

    def on_plan(self, sim, r, t0: int, cells):
        pass

    def on_rl_decision(self, sim, tote, eligible, cand_feats, idx, deltas):
        from baselines import bl6_marginal_cost
        mc  = [bl6_marginal_cost(sim, r, d) for r, d in zip(eligible, deltas)]
        lo  = min(mc)
        hit = mc[idx] - lo < self.tie_tol
        self.last_mc  = mc
        self.n       += 1
        self.agree   += hit
        self.regret  += mc[idx] - lo
        self.n_cands += len(eligible)
        if eligible[int(np.argmin(mc))].is_idle:
            self.n_idle     += 1
            self.agree_idle += hit

    def summary(self) -> dict:
        """결정 수, 일치율 (전체 / BL6 argmin이 idle·mid-trip인 결정), 결정당 평균 regret, 평균 후보 수"""
        nan = float("nan")
        n_mid = self.n - self.n_idle
        return {
            "n_decisions": self.n,
            "agree":       self.agree / self.n if self.n else nan,
            "agree_idle":  self.agree_idle / self.n_idle if self.n_idle else nan,
            "agree_mid":   (self.agree - self.agree_idle) / n_mid if n_mid else nan,
            "regret_mc":   self.regret / self.n if self.n else nan,
            "avg_cands":   self.n_cands / self.n if self.n else nan,
        }
