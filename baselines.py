"""휴리스틱 배정 정책 BL1~BL6

- 버퍼형 (mid-trip 삽입 없음): BL1 Full-Buffer, BL2 Partial-Buffer
- 삽입형 (mid-trip 삽입 허용): BL3 Random, BL4 Nearest, BL5 Min-Insertion, BL6 Marginal-Cost
"""
import random
from typing import List, Optional, Tuple

from environment import Robot
from simulator import SimulatorBase, Order
from constant import PARTIAL_K, ROBOT_SPEED, COST_DIST


class BufferSimulator(SimulatorBase):
    """idle 로봇에 주문을 쌓아 두었다가 depart_k개가 모이면 출발시키는 버퍼형 정책

    - 주문은 빈 슬롯이 가장 많은 idle 로봇에 적재
    - 복귀 워크스테이션은 출발 시점에 모인 목적지 전체 기준으로 결정
    - mid-trip 삽입 없음
    """

    def __init__(self, *args, depart_k: Optional[int] = None, **kwargs):
        """
        입력:
            *args, **kwargs: SimulatorBase 인자
            depart_k: 출발 기준 주문 수 (None이면 로봇 용량)
        """
        super().__init__(*args, **kwargs)
        self.depart_k = depart_k if depart_k is not None else self.robot_capacity

    def _on_order_arrival(self):
        """새 주문을 빈 슬롯이 가장 많은 idle 로봇에 쌓고 depart_k개가 되면 출발

        idle 로봇이 없으면 대기 큐에 넣음
        """
        order = self.order_mgr.create_order(self.now)
        idle  = [r for r in self.robots if r.is_idle and r.idle_slots > 0]
        if idle:
            robot = max(idle, key=lambda r: r.idle_slots)
            self._robot_orders[robot.id].append(order)
            self._record_service_start(order, self.now)
            robot.idle_slots -= 1
            if self.robot_capacity - robot.idle_slots >= self.depart_k:
                robot.assign_trip([o.loc for o in self._robot_orders[robot.id]],
                                  self.layout)
                self._schedule_next(robot)
        else:
            self.order_mgr.enqueue(order, self.now)

    def _on_trip_complete(self, robot: Robot):
        """트립 완료 후 대기 큐에서 최대 용량만큼 꺼내 싣고 depart_k개 이상이면 바로 출발

        입력:
            robot: 트립을 마친 로봇
        """
        self._complete_trip(robot)
        orders = self.order_mgr.dequeue_up_to(self.robot_capacity)
        if orders:
            self._robot_orders[robot.id] = list(orders)
            for o in orders:
                self._record_service_start(o, self.now)
            robot.idle_slots = self.robot_capacity - len(orders)
            if len(orders) >= self.depart_k:
                robot.assign_trip([o.loc for o in orders], self.layout)
                self._schedule_next(robot)


class FullBufferSimulator(BufferSimulator):
    """BL1 Full-Buffer: idle 로봇의 슬롯이 모두 찬 뒤에 출발 (depart_k = 로봇 용량)"""


class PartialSimulator(BufferSimulator):
    """BL2 Partial-Buffer: idle 로봇에 주문이 partial_k개 쌓이면 바로 출발"""

    def __init__(self, *args, partial_k: int = PARTIAL_K, **kwargs):
        """
        입력:
            *args, **kwargs: SimulatorBase 인자
            partial_k: 출발 기준 주문 수
        """
        super().__init__(*args, depart_k=partial_k, **kwargs)
        self.partial_k = partial_k


class InsertionSimulator(SimulatorBase):
    """빈 슬롯이 있는 로봇(idle + mid-trip) 중 하나를 골라 즉시 배정하는 삽입형 정책

    하위 클래스는 _select_robot()으로 로봇 선택 규칙만 정의
    """

    def _select_robot(self, order: Order, eligible: List[Robot],
                      target_coords: Tuple[int, int]) -> Robot:
        """배정할 로봇 선택 (하위 클래스에서 구현)

        입력:
            order: 새 주문
            eligible: 후보 로봇 리스트 (비어 있지 않음)
            target_coords: idle 로봇의 가정 복귀 좌표
        출력:
            선택한 로봇
        """
        raise NotImplementedError

    def _on_order_arrival(self):
        """새 주문을 _select_robot()이 고른 로봇에 배정 (후보가 없으면 대기 큐에 넣음)"""
        order    = self.order_mgr.create_order(self.now)
        eligible = self._eligible_robots()
        if not eligible:
            self.order_mgr.enqueue(order, self.now)
            return

        target_coords = self._idle_return_target(order.loc)
        robot = self._select_robot(order, eligible, target_coords)
        self._extra_travel += self._insertion_delta(robot, order.loc, target_coords)
        self._assign_order(robot, order)


class RandomAssignSimulator(InsertionSimulator):
    """BL3 Random: 후보 로봇 중 무작위 배정"""

    def reset(self, seed: int):
        """에피소드 상태와 배정용 난수 생성기 초기화

        입력:
            seed: 주문 생성 및 배정 난수 시드
        """
        super().reset(seed)
        self._assign_rng = random.Random(seed)

    def _select_robot(self, order, eligible, target_coords):
        """후보 중 무작위 1대"""
        return self._assign_rng.choice(eligible)


class NearestRobotSimulator(InsertionSimulator):
    """BL4 Nearest: 주문 위치까지 현재 거리가 가장 짧은 로봇에 배정"""

    def _select_robot(self, order, eligible, target_coords):
        """현재 위치(pos)가 주문 위치와 가장 가까운 후보"""
        return min(eligible,
                   key=lambda r: self.layout.aisle_distance(r.pos, order.loc))


class MinInsertionSimulator(InsertionSimulator):
    """BL5 Min-Insertion: 삽입 detour가 최소인 로봇에 배정

    idle 후보는 주문에서 가장 가까운 워크스테이션 복귀 가정,
    mid-trip 후보는 최적 복귀 워크스테이션 재탐색 (SimulatorBase._insertion_delta)
    """

    def _select_robot(self, order, eligible, target_coords):
        """삽입 시 추가 이동거리가 가장 작은 후보"""
        return min(eligible,
                   key=lambda r: self._insertion_delta(r, order.loc, target_coords))


class MarginalCostSimulator(InsertionSimulator):
    """BL6 Marginal-Cost: 배정으로 늘어나는 (flow time + c_d · 이동거리)가 최소인 로봇에 배정

    탑재 주문은 모두 트립 종료 시점에 함께 완료되므로, 이후 삽입이 없다고 보면
    한계 비용을 정확히 계산 가능
      - 기존 탑재 주문 지연: n_onboard · Δd / v
      - 새 주문 flow time : 삽입 후 남은 경로 거리 / v
      - 이동거리 증가     : mid-trip이면 Δd, idle이면 새 트립 전체 거리
    피킹/WS 처리 시간과 혼잡 대기는 반영하지 않음
    """

    def _marginal_cost(self, robot: Robot, order: Order,
                       target_coords: Tuple[int, int]) -> float:
        """로봇에 주문을 넣을 때의 한계 비용 (order·s 단위)

        입력:
            robot: 후보 로봇
            order: 새 주문
            target_coords: idle 로봇의 가정 복귀 좌표
        출력:
            한계 비용
        """
        dist = self.layout.aisle_distance
        if robot.is_idle:
            remaining_after = dist(robot.pos, order.loc) + dist(order.loc, target_coords)
            delay_onboard   = 0.0
            added_distance  = remaining_after
        else:
            delta_d         = self._insertion_delta(robot, order.loc, target_coords)
            remaining_after = robot.remaining_distance + delta_d
            delay_onboard   = len(self._robot_orders[robot.id]) * delta_d / ROBOT_SPEED
            added_distance  = delta_d
        return delay_onboard + remaining_after / ROBOT_SPEED + COST_DIST * added_distance

    def _select_robot(self, order, eligible, target_coords):
        """한계 비용이 가장 작은 후보"""
        return min(eligible,
                   key=lambda r: self._marginal_cost(r, order, target_coords))
