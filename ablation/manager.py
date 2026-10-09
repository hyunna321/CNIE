import heapq
import random
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Order:
    order_id:        int
    sku:             int
    loc_idx:         int
    loc:             tuple
    # 주문은 생성 시점에 특정 워크스테이션에 귀속되지
    # 않으며, 어느 워크스테이션으로 복귀(드롭오프)할지는 실제로 그 주문을
    # 담게 되는 로봇의 트립 구성에 따라 동적으로(environment.
    # best_workstation_route) 결정된다.
    arrival_time:    float
    completion_time:    Optional[float] = None
    service_start_time: Optional[float] = None   # 로봇에 배정(트립 디스패치/삽입)된 시각


EVENT_ORDER_ARRIVAL = "ORDER_ARRIVAL"
EVENT_NODE_ARRIVAL  = "ROBOT_NODE_ARRIVAL"
EVENT_TRIP_COMPLETE = "ROBOT_TRIP_COMPLETE"


@dataclass(order=True)
class Event:
    time:  float
    etype: str  = field(compare=False)
    data:  dict = field(compare=False, default_factory=dict)


class EventQueue:
    def __init__(self):
        self._q: list = []

    def push(self, ev: Event):
        heapq.heappush(self._q, ev)

    def pop(self) -> Event:
        return heapq.heappop(self._q)

    def __len__(self):
        return len(self._q)


class OrderManager:
    """
    주문은 어떤 워크스테이션에도 고정되지 않으며, 대기 큐도 워크스테이션
    구분 없는 단일 큐(wait_queue) 하나로 통합한다. 대기 주문은 FIFO 순서로
    큐에 쌓이고, 로봇이 트립을 완료해 새 트립을 받을 수 있게 되는 시점(재배정
    의사결정 시점)마다 큐 맨 앞에서 최대 robot_capacity개까지 꺼내어 배정한다.
    """

    def __init__(self, layout, lam: float, seed: int = 42):
        self.layout           = layout
        self.lam               = lam
        self._rng               = random.Random(seed)
        self._cnt               = 0
        self.num_workstations   = layout.num_workstations

        self.wait_queue:          List[Order]  = []
        self.all_orders:          List[Order]  = []
        self._queue_entry_times:  List[float]  = []

    def reset(self, seed: int):
        self._rng    = random.Random(seed)
        self._cnt    = 0
        self.wait_queue          = []
        self._queue_entry_times  = []
        self.all_orders.clear()

    def generate_next_arrival(self, current_time: float) -> float:
        return current_time + self._rng.expovariate(self.lam)

    def create_order(self, arrival_time: float) -> Order:
        sku          = self.layout.sample_order_sku(self._rng)
        loc_idx, loc = self.layout.find_tote_location(sku, self._rng)
        order        = Order(self._cnt, sku, loc_idx, loc, arrival_time)
        self._cnt   += 1
        self.all_orders.append(order)
        return order

    def enqueue(self, order: Order, enqueue_time: float):
        self.wait_queue.append(order)
        self._queue_entry_times.append(enqueue_time)

    def dequeue_up_to(self, n: int) -> List[Order]:
        """단일 큐에서 FIFO 순서로 최대 n개까지 꺼낸다(problem description #5)."""
        k  = min(n, len(self.wait_queue))
        orders = self.wait_queue[:k]
        self.last_batch_entry_times = list(self._queue_entry_times[:k])
        self.wait_queue             = self.wait_queue[k:]
        self._queue_entry_times     = self._queue_entry_times[k:]
        return orders

    def queue_size(self) -> int:
        return len(self.wait_queue)

    def record_service_start(self, order: Order, time: float):
        """주문이 로봇에 배정(트립 디스패치 또는 mid-trip 삽입)된 시각을 기록한다.
        큐잉 지연(wait_time) 계산의 기준점이 된다."""
        if order.service_start_time is None:
            order.service_start_time = time

    def record_completion(self, order: Order, time: float):
        order.completion_time = time

    def avg_wait_time(self) -> float:
        """평균 큐잉 지연 = 로봇에 배정된 시각 − 도착 시각.
        (대기열에서 실제로 기다린 시간만 측정. 이동/서비스 시간은 제외.)"""
        waits = [
            o.service_start_time - o.arrival_time
            for o in self.all_orders if o.service_start_time is not None
        ]
        return sum(waits) / len(waits) if waits else 0.0

    def avg_flow_time(self) -> float:
        """평균 체류시간(flow time) = 완료 시각 − 도착 시각.
        (큐잉 지연 + 이동/서비스 시간을 모두 포함한 전체 체류시간.)"""
        flows = [
            o.completion_time - o.arrival_time
            for o in self.all_orders if o.completion_time is not None
        ]
        return sum(flows) / len(flows) if flows else 0.0

    def completed_count(self) -> int:
        return sum(1 for o in self.all_orders if o.completion_time is not None)