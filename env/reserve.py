"""로봇별 다음 트립 예약: 후보 로봇이 없을 때 공용 FIFO 큐 대신 쓰는 구조 (휴리스틱 baselines.policies.*R과 RL이 공유)

후보 로봇(빈 슬롯이 있는 로봇)이 있으면 원래 정책과 같음. 없으면 운행 중인 로봇 하나의 다음 트립에 예약
  - 어느 로봇에 예약할지: 휴리스틱은 _reserve_cost 최소, RL은 에이전트가 고름 (env.rl.RLSimulator)
  - 로봇당 예약은 용량까지 (다음 트립 1회분), 예약할 로봇이 없으면 공용 FIFO 큐
  - 트립을 마친 로봇은 예약 토트를 먼저 싣고 남는 슬롯은 공용 큐에서 채워 출발
  - 배정 시각(대기시간 기준)은 예약 시각이 아니라 실제 출발 시각 (공용 큐와 같은 기준)
  - 다음 트립의 출발점 = 지금 트립의 마지막 도킹 셀 (robot.workstation)
배터리 제약이 있으면 (env.battery)
  - 필수 충전 로봇과, 지금 트립 후 SoC로 다음 트립을 감당 못 하는 로봇은 예약 대상에서 제외 (Battery.reserve_ok)
  - 트립을 마친 로봇의 예약 토트는 공용 큐 앞으로 돌려 SoC 규칙(Battery.take_count)으로 다시 실음.
    충전에 들어가면 다른 로봇이 가져감
"""
from typing import List, Tuple

from env.layout import ROBOT_SPEED


def reserve_cost_terms(sim, robot, tote, cost_dist: float) -> Tuple[int, float, float, float, float]:
    """BL6 한계비용 형태의 예약 비용 구성요소 (예약 수, 예약 토트 지연 s, 새 토트 체류 s, Δd m, 합계 s)

      합계 = n_예약 · Δd / v + (T_free + 예약 트립 거리 / v) + c_d · Δd
      T_free = 지금 트립 잔여 시간 (SimulatorBase._robot_time_left: 이동 + 남은 피킹 + WS 처리)
    BL6R 휴리스틱의 예약 기준이자 RL 예약 후보의 prior (agents.ScoreNet의 마지막 특징 열)
    """
    after, dd = sim._next_trip_delta(robot, tote)
    n = len(sim._resv[robot.id])
    delay = n * dd / ROBOT_SPEED
    stay = sim._robot_time_left(robot) + after / ROBOT_SPEED
    return n, delay, stay, dd, delay + stay + cost_dist * dd


class ReserveMixin:
    """SimulatorBase 하위 클래스 앞에 섞어 쓰는 예약 구조 (모듈 설명)"""

    def reset(self, seed: int):
        super().reset(seed)
        self._resv: dict = {r.id: [] for r in self.robots}
        self.n_reserved = 0           # 예약으로 간 토트 수
        self.n_fifo = 0               # 예약할 로봇이 없어 공용 큐로 간 토트 수

    def _reserve_candidates(self, tote) -> List:
        """tote를 다음 트립에 예약할 수 있는 운행 중 로봇"""
        C, bat = self.robot_capacity, self.battery
        return [r for r in self.robots if not r.is_idle and len(self._resv[r.id]) < C
                and (bat is None or (not r.must_charge and bat.reserve_ok(r, self._resv[r.id] + [tote])))]

    def _reserve(self, robot, tote):
        self._resv[robot.id].append(tote)
        self.n_reserved += 1

    def _on_tote_arrival(self, tote):
        if self._eligible_robots(tote):
            super()._on_tote_arrival(tote)
            return
        cands = self._reserve_candidates(tote)
        if not cands:
            self.n_fifo += 1
            self._enqueue_tote(tote)
            return
        self._reserve(min(cands, key=lambda r: self._reserve_cost(r, tote)), tote)

    def _next_trip_distance(self, robot, totes) -> float:
        """robot이 지금 트립의 마지막 도킹 셀에서 totes로 새 트립을 할 때 거리 (m)"""
        if not totes:
            return 0.0
        docks = [self.layout.workstations[k] for k in sorted({t.ws_id for t in totes})]
        return robot._plan(robot.workstation, [t.loc for t in totes], docks)[2]

    def _next_trip_delta(self, robot, tote) -> Tuple[float, float]:
        """(새 토트를 더한 다음 트립 거리, 늘어나는 거리 Δd)"""
        res = self._resv[robot.id]
        before = self._next_trip_distance(robot, res)
        after  = self._next_trip_distance(robot, res + [tote])
        return after, after - before

    def _reserve_cost(self, robot, tote) -> float:
        raise NotImplementedError

    def _on_trip_complete(self, robot):
        self._complete_trip(robot)
        res, self._resv[robot.id] = self._resv[robot.id], []
        if self.battery is None:
            totes = res + self.order_mgr.dequeue_up_to(self.robot_capacity - len(res))
            if totes:
                self._dispatch(robot, totes)
            return
        self.order_mgr.requeue_front(res, self.now)
        self._serve_queue(robot)
        if self.order_mgr.queue_size() > 0:
            self.battery.on_enqueue()      # 남은 큐를 맡을 idle 로봇이 충전 완료 사건에서 가져가도록

    def build_result(self, label, episode, lam):
        """예약으로 간 토트도 도착 즉시 배정되지 않은 토트로 집계 (n_queued_totes, n_decisions)"""
        res = super().build_result(label, episode, lam)
        res.n_queued_totes += self.n_reserved
        res.n_decisions    -= self.n_reserved
        return res
