import heapq
import random
from dataclasses import dataclass, field
from typing import List, Optional

from constant import DEFAULT_SEED


@dataclass
class Order:
    """주문 1건.

    주문은 특정 워크스테이션에 귀속되지 않으며, 복귀 워크스테이션은
    주문을 담은 로봇의 트립 구성에 따라 결정된다.
    """
    order_id:        int
    sku:             int
    loc_idx:         int
    loc:             tuple
    arrival_time:    float
    completion_time:    Optional[float] = None
    service_start_time: Optional[float] = None   # 로봇에 배정된 시각


EVENT_ORDER_ARRIVAL = "ORDER_ARRIVAL"
EVENT_NODE_ARRIVAL  = "ROBOT_NODE_ARRIVAL"
EVENT_TRIP_COMPLETE = "ROBOT_TRIP_COMPLETE"


@dataclass(order=True)
class Event:
    """이산 사건 시뮬레이션의 사건. time 기준으로만 정렬된다."""
    time:  float
    etype: str  = field(compare=False)
    data:  dict = field(compare=False, default_factory=dict)


class EventQueue:
    """사건 발생 시각 순으로 꺼내는 우선순위 큐."""

    def __init__(self):
        self._q: list = []

    def push(self, ev: Event):
        """사건을 큐에 넣는다.

        입력:
            ev: 추가할 사건
        """
        heapq.heappush(self._q, ev)

    def pop(self) -> Event:
        """가장 이른 사건을 꺼낸다.

        출력:
            발생 시각이 가장 이른 사건
        """
        return heapq.heappop(self._q)

    def __len__(self):
        return len(self._q)


class OrderManager:
    """주문 생성(포아송 도착)과 단일 FIFO 대기 큐, 주문 단위 지표를 관리한다."""

    def __init__(self, layout, lam: float, seed: int = DEFAULT_SEED):
        """
        입력:
            layout: WarehouseLayout
            lam: 주문 도착률 (orders/s)
            seed: 주문 생성 난수 시드
        """
        self.layout           = layout
        self.lam               = lam
        self._rng               = random.Random(seed)
        self._cnt               = 0
        self.num_workstations   = layout.num_workstations

        self.wait_queue:          List[Order]  = []
        self.all_orders:          List[Order]  = []
        self._queue_entry_times:  List[float]  = []

    def reset(self, seed: int):
        """에피소드 시작 전 상태를 초기화한다.

        입력:
            seed: 새 주문 생성 난수 시드
        """
        self._rng    = random.Random(seed)
        self._cnt    = 0
        self.wait_queue          = []
        self._queue_entry_times  = []
        self.all_orders.clear()

    def generate_next_arrival(self, current_time: float) -> float:
        """지수분포 도착 간격으로 다음 주문 도착 시각을 만든다.

        입력:
            current_time: 현재 시각 (s)
        출력:
            다음 주문 도착 시각 (s)
        """
        return current_time + self._rng.expovariate(self.lam)

    def create_order(self, arrival_time: float) -> Order:
        """SKU와 보관 위치를 샘플링해 새 주문을 만들고 전체 주문 목록에 기록한다.

        입력:
            arrival_time: 주문 도착 시각 (s)
        출력:
            생성된 Order
        """
        sku          = self.layout.sample_order_sku(self._rng)
        loc_idx, loc = self.layout.find_tote_location(sku, self._rng)
        order        = Order(self._cnt, sku, loc_idx, loc, arrival_time)
        self._cnt   += 1
        self.all_orders.append(order)
        return order

    def enqueue(self, order: Order, enqueue_time: float):
        """배정할 로봇이 없는 주문을 대기 큐 끝에 넣는다.

        입력:
            order: 대기시킬 주문
            enqueue_time: 큐 진입 시각 (s)
        """
        self.wait_queue.append(order)
        self._queue_entry_times.append(enqueue_time)

    def dequeue_up_to(self, n: int) -> List[Order]:
        """대기 큐 앞에서 FIFO 순서로 최대 n개를 꺼낸다.

        입력:
            n: 꺼낼 최대 주문 수
        출력:
            꺼낸 주문 리스트
        """
        k  = min(n, len(self.wait_queue))
        orders = self.wait_queue[:k]
        self.last_batch_entry_times = list(self._queue_entry_times[:k])
        self.wait_queue             = self.wait_queue[k:]
        self._queue_entry_times     = self._queue_entry_times[k:]
        return orders

    def queue_size(self) -> int:
        """현재 대기 큐 길이."""
        return len(self.wait_queue)

    def record_service_start(self, order: Order, time: float):
        """주문이 처음 로봇에 배정된 시각을 기록한다. 이미 기록돼 있으면 무시한다.

        입력:
            order: 대상 주문
            time: 배정 시각 (s)
        """
        if order.service_start_time is None:
            order.service_start_time = time

    def record_completion(self, order: Order, time: float):
        """주문 완료(워크스테이션 도착) 시각을 기록한다.

        입력:
            order: 대상 주문
            time: 완료 시각 (s)
        """
        order.completion_time = time

    def avg_wait_time(self) -> float:
        """평균 대기시간(배정 시각 − 도착 시각). 이동/서비스 시간은 제외한다.

        출력:
            배정된 주문들의 평균 대기시간 (s), 없으면 0.0
        """
        waits = [
            o.service_start_time - o.arrival_time
            for o in self.all_orders if o.service_start_time is not None
        ]
        return sum(waits) / len(waits) if waits else 0.0

    def avg_flow_time(self) -> float:
        """평균 체류시간(완료 시각 − 도착 시각). 대기 + 이동 시간을 모두 포함한다.

        출력:
            완료된 주문들의 평균 체류시간 (s), 없으면 0.0
        """
        flows = [
            o.completion_time - o.arrival_time
            for o in self.all_orders if o.completion_time is not None
        ]
        return sum(flows) / len(flows) if flows else 0.0

    def completed_count(self) -> int:
        """완료된 주문 수."""
        return sum(1 for o in self.all_orders if o.completion_time is not None)
