"""배터리 제약: 로봇 SoC, 충전 규칙, 배정 실행 가능성 (SimulatorBase의 battery 인자, 기본 None이면 동작 변화 없음)

에너지 단위는 이동 거리(m)로, 만충 주행거리 range_m를 쓰고 셀 1칸 이동마다 CELL_SIZE만큼 소모. 대기·피킹·처리 중에는 소모 없음

충전 (주차 칸 = 충전 칸, 충전기 수 제약 없음)
  - idle 로봇이 주차 칸에 들어서면 충전 시작, 그 칸에서 움직이면 끝. 충전량은 경과시간으로 계산 (틱 건너뛰기와 무관)
  - 필수 충전 (must_charge): 트립 완료 시 SoC < low_soc · range_m이거나, 대기 큐 토트를 어느 것도 맡을 수 없을 때.
    SoC가 target_soc · range_m에 이를 때까지 배정 후보에서 빠지고 큐 방출도 받지 않음
  - 필수 충전이 끝나면 (EVENT_CHARGE_DONE) 대기 큐를 다시 방출. 그 뒤로도 주차 칸에 있는 동안 만충까지 계속 충전

실행 가능성 (hard): 배정 후 SoC ≥ 남은 경로 + 마지막 도킹 셀에서 그 WS의 가장 먼 주차 칸까지 + reserve_m
  - 남은 경로는 정책 평가와 같은 경로 규칙 (env.simulator._visits, 삽입 대기 pending_insert 포함). 혼잡 우회는 reserve_m으로 흡수
  - 삽입형 정책과 RL은 후보에서, 버퍼형은 idle 로봇 선택에서, 큐 방출은 FIFO 앞쪽에서 맡을 수 있는 토트 수를 줄여서 적용
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from env.layout import CELL_SIZE

BATTERY_CAND_DIM = 2    # features="battery" 후보 특징: SoC 비율, 배정 후 여유 비율 (Battery.candidate_features)
BATTERY_CTX_DIM  = 3    # features="battery" 전역 특징: 평균 SoC, 최소 SoC, 필수 충전 로봇 비율 (Battery.global_features)


@dataclass(frozen=True)
class BatteryConfig:
    """배터리 파라미터. 거리 m, 시간 s, 비율 0~1

    기본값: MTSR AMR 시스템 연구 (Shan, Jin, Li, Kondo, arXiv 2407.06394)의 설정을 이 시뮬레이터 단위로 환산
      - 이동 중 소모 0.5 %/분 → 만충으로 이동 200분, 로봇 속도 1 m/s이므로 12,000 m
      - 충전 시작 임계값 20 % (트립 완료 시 판정), 완충까지 충전
      - 충전 시간 25~35분 (평균 30분)을 고정 30분으로: 20 % → 100 % (9,600 m)를 1,800 s에 → 16/3 m/s
      - 충전은 워크스테이션 구역(주차 칸)에서, 충전기 수 제약 없음 (논문은 충전소 1곳에 충전기 4대)
    """
    range_m:     float = 12000.0         # 만충 주행거리
    charge_rate: float = 9600.0 / 1800.0  # 충전 속도 (m/s, 충전 1초당 회복하는 주행거리)
    reserve_m:   float = 50.0            # 실행 가능성 여유 (혼잡 우회 흡수)
    low_soc:     float = 0.2             # 트립 완료 시 이 비율 미만이면 필수 충전
    target_soc:  float = 1.0             # 필수 충전 목표
    initial_soc: float = 1.0      # 에피소드 시작 SoC

    def __post_init__(self):
        if self.range_m <= 0 or self.charge_rate <= 0 or self.reserve_m < 0:
            raise ValueError("range_m, charge_rate는 양수, reserve_m은 0 이상이어야 합니다")
        if not 0.0 <= self.low_soc <= self.target_soc <= 1.0 or not 0.0 <= self.initial_soc <= 1.0:
            raise ValueError("0 ≤ low_soc ≤ target_soc ≤ 1, 0 ≤ initial_soc ≤ 1이어야 합니다")

    @property
    def tag(self) -> str:
        """결과 폴더 꼬리표 (같은 값이면 같은 폴더)"""
        return (f"_bat-R{self.range_m:g}-c{self.charge_rate:g}-res{self.reserve_m:g}"
                f"-l{self.low_soc:g}-t{self.target_soc:g}-i{self.initial_soc:g}")

    @classmethod
    def from_arg(cls, text) -> Optional["BatteryConfig"]:
        """명령행 값(JSON 문자열 또는 JSON 파일 경로)으로 만듦. None/빈 문자열이면 배터리 제약 없음"""
        if not text:
            return None
        return cls(**json.loads(open(text).read() if os.path.isfile(text) else text))


class Battery:
    """시뮬레이터 한 대의 SoC·충전 상태 관리. 로봇별 상태는 Robot.soc / charge_t0 / must_charge

    r.soc는 마지막 정산 시점 값이고 지금 값은 soc(r). 주차 칸에서 충전 중이면 r.charge_t0 = 마지막 정산 시각, 아니면 None
    """

    def __init__(self, sim, cfg: BatteryConfig):
        self.sim, self.cfg = sim, cfg
        L = sim.layout
        if L.park_cap < sim.num_robots:
            raise ValueError(f"배터리 제약에는 WS마다 로봇 수만큼 주차 칸이 필요합니다 ({L.park_cap} < {sim.num_robots})")
        # 어느 주차 칸에서 출발해도 토트 1개짜리 트립을 끝낼 수 있어야 필수 충전 후 큐가 막히지 않음
        picks = [L.cell_index[c] for c in L.access_coords]
        docks = [L.cell_index[d] for d in L.docks]
        spots = [L.cell_index[c] for s in L.park_spots for c in s]
        worst = float(L._D[np.ix_(spots, picks)].max() + L._D[np.ix_(picks, docks)].max()
                      + L._D[np.ix_(docks, spots)].max()) * CELL_SIZE
        if cfg.target_soc * cfg.range_m < worst + cfg.reserve_m:
            raise ValueError(f"target_soc · range_m = {cfg.target_soc * cfg.range_m:g} m가 "
                             f"최장 단일 트립 {worst:g} m + reserve {cfg.reserve_m:g} m보다 작습니다")

    def reset(self):
        """에피소드 시작: 주차 칸에서 initial_soc로 시작 (충전 중 상태), low_soc 미만이면 바로 필수 충전"""
        self.charged_m  = 0.0            # 회복한 주행거리 합
        self.n_required = 0              # 필수 충전 횟수
        self.n_rejects  = 0              # SoC 부족으로 후보에서 빠진 (토트, 로봇) 수
        self.n_depleted = 0              # SoC가 음수인 상태로 이동한 셀 수 (0이어야 함)
        c = self.cfg
        self.min_soc = c.initial_soc * c.range_m
        for r in self.sim.robots:
            r.soc, r.charge_t0, r.must_charge = c.initial_soc * c.range_m, self.sim.now, False
            if r.soc < c.low_soc * c.range_m:
                self.require_charge(r)

    # SoC

    def soc(self, r) -> float:
        """지금 SoC (m)"""
        if r.charge_t0 is None:
            return r.soc
        return min(self.cfg.range_m, r.soc + self.cfg.charge_rate * (self.sim.now - r.charge_t0))

    def _settle(self, r):
        """진행 중인 충전량을 r.soc에 반영"""
        if r.charge_t0 is not None:
            new = self.soc(r)
            self.charged_m += new - r.soc
            r.soc, r.charge_t0 = new, self.sim.now

    def on_move(self, r):
        """r이 셀 1칸 이동한 직후 (거리 집계와 같은 곳에서 호출): 충전 종료, 소모, 주차 칸 도착이면 충전 시작"""
        self._settle(r)
        r.charge_t0 = None
        r.soc -= CELL_SIZE
        self.min_soc = min(self.min_soc, r.soc)
        self.n_depleted += r.soc < 0.0
        if r.is_idle and r.current_dest is None and r.pos == r.park_spot:
            r.charge_t0 = self.sim.now
            if r.must_charge:
                self._schedule_done(r)

    # 충전 규칙

    def require_charge(self, r):
        """r을 필수 충전으로 표시. 이미 주차 칸이면 완료 사건 예약, 아니면 주차 칸 도착 시 예약 (on_move)"""
        if r.must_charge:
            return
        r.must_charge = True
        self.n_required += 1
        if r.charge_t0 is not None:
            self._schedule_done(r)

    def _schedule_done(self, r):
        self._settle(r)
        wait = max(0.0, self.cfg.target_soc * self.cfg.range_m - r.soc) / self.cfg.charge_rate
        self.sim._push_charge_done(r, self.sim.now + wait)

    def finish_required(self, r):
        """필수 충전 완료 사건 처리"""
        self._settle(r)
        r.must_charge = False

    def on_enqueue(self):
        """토트가 대기 큐로 갔을 때: 그 토트를 못 맡은 빈 idle 로봇은 충전시켜 큐가 다음 트립 완료까지 묶이지 않게 함

        트립이 끝나 가는 로봇이 없어도 충전 완료 사건에서 큐를 다시 방출하므로 토트가 영구히 남지 않음
        """
        for r in self.sim.robots:
            if r.is_idle and not self.sim._robot_totes[r.id]:
                self.require_charge(r)

    # 실행 가능성

    def _to_spot(self, last) -> float:
        """경로 끝 셀에서 주차(충전) 칸까지 거리 상한 (m)"""
        L = self.sim.layout
        return L.distance(last, L.spots_by_dist[L.dock_ws[last]][-1]) if L.is_dock(last) else 0.0

    def need(self, r, tote) -> float:
        """r에 tote를 더 맡길 때 필요한 SoC (m)"""
        dist, last = self.sim._route_after(r, tote)
        return dist + self._to_spot(last) + self.cfg.reserve_m

    def filter(self, robots: List, tote=None) -> List:
        """배정 가능한 로봇만 남김 (필수 충전 제외, tote가 있으면 실행 가능성까지)"""
        out = [r for r in robots if not r.must_charge]
        if tote is None:
            return out
        ok = [r for r in out if self.soc(r) >= self.need(r, tote)]
        self.n_rejects += len(out) - len(ok)
        return ok

    def take_count(self, r) -> int:
        """트립을 마쳤거나 충전을 끝낸 idle 로봇 r이 대기 큐 앞에서 맡을 토트 수 (맡을 수 없으면 필수 충전으로 표시)"""
        soc = self.soc(r)
        if soc < self.cfg.low_soc * self.cfg.range_m:
            self.require_charge(r)
            return 0
        sim = self.sim
        queue = sim.order_mgr.wait_queue
        n = min(sim.robot_capacity, len(queue))
        while n > 0 and soc < self._new_trip_need(r, queue[:n]):
            n -= 1
        if n == 0 and queue:
            self.require_charge(r)
        return n

    def reserve_ok(self, r, totes) -> bool:
        """운행 중인 r의 다음 트립에 totes를 예약해도 되는지 (baselines.policies.ReserveMixin)

        지금 트립을 마친 뒤의 예상 SoC가 low_soc 이상이고 (아니면 트립 후 충전으로 예약이 무산됨),
        마지막 도킹 셀에서 출발하는 다음 트립 + 주차 칸 + reserve_m을 감당해야 함
        """
        left, need = self._reserve_need(r, totes)
        return left >= self.cfg.low_soc * self.cfg.range_m and left >= need

    def _reserve_need(self, r, totes):
        """(지금 트립을 마친 뒤 예상 SoC, totes로 하는 다음 트립 + 주차 칸 + reserve_m에 필요한 SoC) (m)"""
        cur, _ = self.sim._route_after(r, None)
        L = self.sim.layout
        docks = [L.workstations[k] for k in sorted({t.ws_id for t in totes})]
        picks, drops, dist = r._plan(r.workstation, [t.loc for t in totes], docks, keep_docks=False)
        return self.soc(r) - cur, dist + self._to_spot((picks + drops)[-1]) + self.cfg.reserve_m

    def reserve_features(self, tote, robots: List) -> List[List[float]]:
        """예약 후보별 [SoC / range, (지금 트립 후 SoC − 다음 트립 필요량) / range] (candidate_features와 같은 형식)"""
        R = self.cfg.range_m
        out = []
        for r in robots:
            left, need = self._reserve_need(r, self.sim._resv[r.id] + [tote])
            out.append([self.soc(r) / R, (left - need) / R])
        return out

    def _new_trip_need(self, r, totes) -> float:
        """r이 지금 위치에서 totes로 새 트립을 할 때 필요한 SoC (m)"""
        L = self.sim.layout
        docks = [L.workstations[k] for k in sorted({t.ws_id for t in totes})]
        picks, drops, dist = r._plan(r.pos, [t.loc for t in totes], docks, keep_docks=False)
        return dist + self._to_spot((picks + drops)[-1]) + self.cfg.reserve_m

    # RL 특징 / 지표

    def candidate_features(self, tote, robots: List) -> List[List[float]]:
        """후보별 [SoC / range, (SoC − 배정 후 필요량) / range]"""
        R = self.cfg.range_m
        return [[self.soc(r) / R, (self.soc(r) - self.need(r, tote)) / R] for r in robots]

    def global_features(self) -> List[float]:
        """[평균 SoC, 최소 SoC, 필수 충전 로봇 비율] (range 대비)"""
        socs = [self.soc(r) / self.cfg.range_m for r in self.sim.robots]
        return [sum(socs) / len(socs), min(socs), sum(r.must_charge for r in self.sim.robots) / len(socs)]

    def fill_result(self, res):
        """EpisodeResult에 배터리 지표 기록"""
        for r in self.sim.robots:
            self._settle(r)
        res.n_charge_required = self.n_required
        res.charge_time       = self.charged_m / self.cfg.charge_rate
        res.min_soc           = self.min_soc / self.cfg.range_m
        res.n_soc_depleted    = self.n_depleted
        res.n_battery_rejects = self.n_rejects
