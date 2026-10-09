"""RL과 비교하는 휴리스틱 배정 정책 BL1~BL5.

- 버퍼형 (mid-trip 삽입 없음): BL1 Full-Buffer, BL2 Partial-Buffer
- 삽입형 (mid-trip 삽입 허용): BL3 Random, BL4 Nearest, BL5 Min-Insertion
"""
import random
from typing import List, Optional, Tuple

from environment import Robot
from simulator import SimulatorBase, Order
from constant import PARTIAL_K


class BufferSimulator(SimulatorBase):
    """idle 로봇에 주문을 쌓아 두었다가 depart_k개가 모이면 출발시키는 버퍼형 정책.

    주문은 빈 슬롯이 가장 많은 idle 로봇에 쌓이고, 출발 시점에 모인 목적지
    전체 기준으로 복귀 워크스테이션이 정해진다. mid-trip 삽입은 없다.
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
        """새 주문을 빈 슬롯이 가장 많은 idle 로봇에 쌓고, depart_k개가 되면 출발시킨다.

        idle 로봇이 없으면 대기 큐에 넣는다.
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
        """트립 완료 후 대기 큐에서 최대 용량만큼 꺼내 싣고, depart_k개 이상이면 바로 출발시킨다.

        입력:
            robot: 트립을 마친 로봇
        """
        self._complete_trip(robot)
        orders = self.order_mgr.dequeue_up_to(self.robot_capacity)
        if orders:
            self._robot_orders[robot.id] = list(orders)
            robot.idle_slots = self.robot_capacity - len(orders)
            if len(orders) >= self.depart_k:
                robot.assign_trip([o.loc for o in orders], self.layout)
                self._schedule_next(robot)


class FullBufferSimulator(BufferSimulator):
    """BL1 Full-Buffer: idle 로봇의 슬롯이 모두 찬 뒤에 출발한다 (depart_k = 로봇 용량)."""


class PartialSimulator(BufferSimulator):
    """BL2 Partial-Buffer: idle 로봇에 주문이 partial_k개 쌓이면 바로 출발한다."""

    def __init__(self, *args, partial_k: int = PARTIAL_K, **kwargs):
        """
        입력:
            *args, **kwargs: SimulatorBase 인자
            partial_k: 출발 기준 주문 수
        """
        super().__init__(*args, depart_k=partial_k, **kwargs)
        self.partial_k = partial_k


class InsertionSimulator(SimulatorBase):
    """빈 슬롯이 있는 로봇(idle + mid-trip) 중 하나를 골라 즉시 배정하는 삽입형 정책.

    하위 클래스는 _select_robot()으로 로봇 선택 규칙만 정의한다.
    """

    def _select_robot(self, order: Order, eligible: List[Robot],
                      target_coords: Tuple[float, float]) -> Robot:
        """배정할 로봇을 고른다. 하위 클래스가 구현한다.

        입력:
            order: 새 주문
            eligible: 후보 로봇 리스트 (비어 있지 않음)
            target_coords: idle 로봇의 가정 복귀 좌표
        출력:
            선택한 로봇
        """
        raise NotImplementedError

    def _on_order_arrival(self):
        """새 주문을 _select_robot()이 고른 로봇에 배정한다. 후보가 없으면 대기 큐에 넣는다."""
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
    """BL3 Random: 후보 로봇 중 무작위로 배정한다."""

    def reset(self, seed: int):
        """에피소드 상태와 배정용 난수 생성기를 초기화한다.

        입력:
            seed: 주문 생성 및 배정 난수 시드
        """
        super().reset(seed)
        self._assign_rng = random.Random(seed)

    def _select_robot(self, order, eligible, target_coords):
        """후보 중 무작위 1대."""
        return self._assign_rng.choice(eligible)


class NearestRobotSimulator(InsertionSimulator):
    """BL4 Nearest: 주문 위치까지 현재 거리가 가장 짧은 로봇에 배정한다."""

    def _select_robot(self, order, eligible, target_coords):
        """현재 위치(pos)가 주문 위치와 가장 가까운 후보."""
        return min(eligible,
                   key=lambda r: self.layout.aisle_distance(r.pos, order.loc))


class MinInsertionSimulator(InsertionSimulator):
    """BL5 Min-Insertion: 삽입 detour가 최소인 로봇에 배정한다.

    detour 계산 기준은 RLSimulator와 같다 (idle 후보는 주문에서 가장 가까운
    워크스테이션 복귀 가정, mid-trip 후보는 최적 복귀 워크스테이션 재탐색).
    """

    def _select_robot(self, order, eligible, target_coords):
        """삽입 시 추가 이동거리가 가장 작은 후보."""
        return min(eligible,
                   key=lambda r: self._insertion_delta(r, order.loc, target_coords))
