"""휴리스틱 배정 정책 BL1~BL6, Greedy_R (토트 단위 배정)

- 버퍼형 (mid-trip 삽입 없음): BL1 Full-Buffer, BL1-Fill (한 배치를 채운 뒤 다음 배치), BL2 Partial-Buffer
- 삽입형 (mid-trip 삽입 허용): BL3 Random, BL4 Nearest, BL5 Min-Insertion, BL6 Marginal-Cost,
  Greedy_R (legacy RL 보상 compute_reward의 즉시값이 최대인 후보, 학습 없는 근시안 기준)

주문의 WS는 도착 시 SimulatorBase._choose_ws가 정하고, 정책은 토트마다 로봇만 고름
"""
import random
import time
from typing import List, Optional

import numpy as np

from environment import Robot
from simulator import SimulatorBase, Tote
from constant import (
    PARTIAL_K, ROBOT_SPEED, COST_DIST,
    REWARD_GAMMA_Q, REWARD_ALPHA_MID, REWARD_ALPHA_IDLE, REWARD_BETA_DEPLETION,
)


def compute_reward(robot: Robot, layout, delta_d: float, robots: List[Robot],
                   n_wait: int, num_robots: int, robot_capacity: int,
                   gamma_q: float = REWARD_GAMMA_Q, alpha_mid: float = REWARD_ALPHA_MID,
                   alpha_idle: float = REWARD_ALPHA_IDLE,
                   beta_depletion: float = REWARD_BETA_DEPLETION) -> float:
    """legacy 보상: 토트 배정 1건의 보상

    보상 = exp(−α · detour / D_max) − γ_q · 큐 압박
           − β_dep (idle 로봇을 썼는데 빈 idle 로봇이 더 남지 않은 경우)

    입력:
        robot: 토트를 배정받은 로봇 (배정 직전 상태)
        layout: GridWarehouseLayout (reward_d_max 사용)
        delta_d: 배정으로 늘어난 이동거리 (m)
        robots: 후보 로봇 리스트 (소진 페널티 판단용)
        n_wait: 현재 대기 큐 길이 (토트)
        num_robots: 큐 압박 정규화용 로봇 수
        robot_capacity: 로봇 용량
    출력:
        보상값
    """
    alpha  = alpha_idle if robot.is_idle else alpha_mid
    reward = float(np.exp(-alpha * delta_d / layout.reward_d_max))
    if gamma_q > 0.0:
        reward -= gamma_q * min(n_wait / max(1, num_robots * robot_capacity), 1.0)
    if beta_depletion > 0.0 and robot.is_idle:
        free_idle_left = sum(1 for r in robots
                             if r.id != robot.id and r.is_idle and r.idle_slots == robot_capacity)
        if free_idle_left == 0:
            reward -= beta_depletion
    return reward


def bl6_marginal_cost(sim: SimulatorBase, robot: Robot, delta_d: float) -> float:
    """BL6 한계 비용 (tote·s): 삽입 detour delta_d가 이미 계산된 경우 (MarginalCostSimulator 설명 참고)

    입력:
        sim: 시뮬레이터 (_robot_totes 사용)
        robot: 후보 로봇
        delta_d: 삽입 추가 이동거리 (m, SimulatorBase._insertion_delta)
    출력:
        한계 비용
    """
    if robot.is_idle:
        remaining_after = delta_d
        delay_onboard   = 0.0
    else:
        remaining_after = robot.remaining_distance + delta_d
        delay_onboard   = len(sim._robot_totes[robot.id]) * delta_d / ROBOT_SPEED
    return delay_onboard + remaining_after / ROBOT_SPEED + COST_DIST * delta_d


class BufferSimulator(SimulatorBase):
    """idle 로봇에 토트를 쌓아 두었다가 depart_k개가 모이면 출발시키는 버퍼형 정책

    - 토트는 빈 슬롯이 가장 많은 idle 로봇에 적재
    - mid-trip 삽입 없음
    """

    def __init__(self, *args, depart_k: Optional[int] = None, **kwargs):
        """
        입력:
            *args, **kwargs: SimulatorBase 인자
            depart_k: 출발 기준 토트 수 (None이면 로봇 용량)
        """
        super().__init__(*args, **kwargs)
        self.depart_k = depart_k if depart_k is not None else self.robot_capacity

    def _pick_idle(self, idle: List[Robot]) -> Robot:
        """토트를 쌓을 idle 로봇: 빈 슬롯이 가장 많은 로봇 (동률이면 번호가 작은 로봇)"""
        return max(idle, key=lambda r: r.idle_slots)

    def _depart(self, robot: Robot):
        robot.assign_trip([t.loc for t in self._robot_totes[robot.id]], self._needed_docks(robot))
        self._schedule_next(robot)

    def _on_tote_arrival(self, tote: Tote):
        """새 토트를 빈 슬롯이 가장 많은 idle 로봇에 쌓고 depart_k개가 되면 출발

        idle 로봇이 없으면 대기 큐에 넣음
        """
        idle = [r for r in self.robots if r.is_idle and r.idle_slots > 0]
        if not idle:
            self.order_mgr.enqueue(tote, self.now)
            return
        robot = self._pick_idle(idle)
        self._robot_totes[robot.id].append(tote)
        self.order_mgr.record_assign(tote, self.now)
        robot.idle_slots -= 1
        if self.robot_capacity - robot.idle_slots >= self.depart_k:
            self._depart(robot)

    def _on_trip_complete(self, robot: Robot):
        """트립 완료 후 대기 큐에서 최대 용량만큼 꺼내 싣고 depart_k개 이상이면 바로 출발

        입력:
            robot: 트립을 마친 로봇
        """
        self._complete_trip(robot)
        totes = self.order_mgr.dequeue_up_to(self.robot_capacity)
        if totes:
            self._robot_totes[robot.id] = list(totes)
            for t in totes:
                self.order_mgr.record_assign(t, self.now)
            robot.idle_slots = self.robot_capacity - len(totes)
            if len(totes) >= self.depart_k:
                self._depart(robot)


class FullBufferSimulator(BufferSimulator):
    """BL1 Full-Buffer: idle 로봇의 슬롯이 모두 찬 뒤에 출발 (depart_k = 로봇 용량)"""


class FillBufferSimulator(FullBufferSimulator):
    """BL1-Fill: BL1과 같지만 이미 토트를 쌓기 시작한 idle 로봇부터 채움 (배치 하나를 채운 뒤 다음 배치)

    BL1은 빈 슬롯이 가장 많은 로봇에 쌓아 idle 로봇마다 토트가 흩어지므로 출발이 늦어질 수 있음
    """

    def _pick_idle(self, idle: List[Robot]) -> Robot:
        """빈 슬롯이 가장 적은 idle 로봇 (동률이면 번호가 작은 로봇)"""
        return min(idle, key=lambda r: r.idle_slots)


class PartialSimulator(BufferSimulator):
    """BL2 Partial-Buffer: idle 로봇에 토트가 partial_k개 쌓이면 바로 출발"""

    def __init__(self, *args, partial_k: int = PARTIAL_K, **kwargs):
        """
        입력:
            *args, **kwargs: SimulatorBase 인자
            partial_k: 출발 기준 토트 수
        """
        super().__init__(*args, depart_k=partial_k, **kwargs)
        self.partial_k = partial_k


class InsertionSimulator(SimulatorBase):
    """빈 슬롯이 있는 로봇(idle + mid-trip) 중 하나를 골라 즉시 배정하는 삽입형 정책

    하위 클래스는 _select_robot()으로 로봇 선택 규칙만 정의
    """

    def _select_robot(self, tote: Tote, eligible: List[Robot]) -> Robot:
        """배정할 로봇 선택 (하위 클래스에서 구현)

        입력:
            tote: 새 토트 (tote.ws_id = 가야 할 WS)
            eligible: 후보 로봇 리스트 (비어 있지 않음)
        출력:
            선택한 로봇
        """
        raise NotImplementedError

    def _on_tote_arrival(self, tote: Tote):
        """새 토트를 _select_robot()이 고른 로봇에 배정 (후보가 없으면 대기 큐에 넣음)"""
        t0 = time.perf_counter()
        eligible = self._eligible_robots()
        if not eligible:
            self.order_mgr.enqueue(tote, self.now)
            return
        robot = self._select_robot(tote, eligible)
        t_sel = time.perf_counter()
        self._extra_travel += self._insertion_delta(robot, tote)
        self._assign_tote(robot, tote)
        self._record_latency(t0, t_sel, time.perf_counter())


class RandomAssignSimulator(InsertionSimulator):
    """BL3 Random: 후보 로봇 중 무작위 배정"""

    def reset(self, seed: int):
        """에피소드 상태와 배정용 난수 생성기 초기화

        입력:
            seed: 주문 생성 및 배정 난수 시드
        """
        super().reset(seed)
        self._assign_rng = random.Random(seed)

    def _select_robot(self, tote, eligible):
        """후보 중 무작위 1대"""
        return self._assign_rng.choice(eligible)


class NearestRobotSimulator(InsertionSimulator):
    """BL4 Nearest: 토트 위치까지 현재 거리가 가장 짧은 로봇에 배정"""

    def _select_robot(self, tote, eligible):
        """현재 위치(pos)가 토트 위치와 가장 가까운 후보"""
        return min(eligible,
                   key=lambda r: self.layout.aisle_distance(r.pos, tote.loc))


class MinInsertionSimulator(InsertionSimulator):
    """BL5 Min-Insertion: 삽입 detour가 최소인 로봇에 배정 (SimulatorBase._insertion_delta)"""

    def _select_robot(self, tote, eligible):
        """삽입 시 추가 이동거리가 가장 작은 후보"""
        return min(eligible, key=lambda r: self._insertion_delta(r, tote))


class MarginalCostSimulator(InsertionSimulator):
    """BL6 Marginal-Cost: 배정으로 늘어나는 (토트 체류시간 + c_d · 이동거리)가 최소인 로봇에 배정

    이후 삽입이 없다고 보고 토트 단위로 근사
      - 기존 탑재 토트 지연: n_onboard · Δd / v
      - 새 토트 체류시간  : 삽입 후 남은 경로 거리 / v
      - 이동거리 증가     : mid-trip이면 Δd, idle이면 새 트립 전체 거리
    주문 완료가 마지막 토트에 달린 점, 피킹/WS 처리 시간, 혼잡 대기는 반영하지 않음
    """

    def _marginal_cost(self, robot: Robot, tote: Tote) -> float:
        """로봇에 토트를 넣을 때의 한계 비용 (tote·s 단위)

        입력:
            robot: 후보 로봇
            tote: 새 토트
        출력:
            한계 비용
        """
        return bl6_marginal_cost(self, robot, self._insertion_delta(robot, tote))

    def _select_robot(self, tote, eligible):
        """한계 비용이 가장 작은 후보"""
        return min(eligible, key=lambda r: self._marginal_cost(r, tote))


class GreedyRewardSimulator(InsertionSimulator):
    """Greedy_R: legacy 보상 compute_reward (RLSimulator와 같은 인자)가 최대인 로봇에 배정

    큐 압박 항은 후보 공통이라 선택에는 detour 감쇠(idle/mid-trip 계수 다름)와 소진 페널티만 작용
    """

    def _select_robot(self, tote, eligible):
        """즉시 legacy 보상이 가장 큰 후보 (동률이면 앞쪽 = mid-trip 우선)"""
        n_wait = self.order_mgr.queue_size()
        return max(eligible, key=lambda r: compute_reward(
            r, self.layout, self._insertion_delta(r, tote), eligible,
            n_wait, len(eligible), self.robot_capacity))
