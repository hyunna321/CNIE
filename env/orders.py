"""주문 쪽 환경: 시나리오(도착/수요/저장/처리 시간 분포)와 주문·토트 관리

주문 1건 = 서로 다른 SKU 여러 개. 토트 1개는 SKU 1개만 담으므로 SKU마다 토트 1개를 워크스테이션으로 가져와야 함
  - 주문은 도착 시 워크스테이션 하나에 배정, 그 주문의 토트는 모두 그 WS에서 처리
  - 주문 완료 = 마지막 토트의 WS 처리 완료. 토트 재고/반납은 모델링하지 않음

Scenario() 기본값: 포아송 도착, 주문당 SKU 수 기하분포 (평균 ORDER_LINES_MEAN),
균등 SKU 수요, 무작위 저장, 고정 피킹/처리 시간 (PICK_TIME, WS_PROC_TIME)
주문 크기/SKU/위치와 포아송 도착 간격은 OrderManager._rng에서,
그 밖의 도착 과정과 처리 시간은 별도 생성기(arrival_rng, time_rng)에서 추출

참고 문헌:
  - ABC 수요 곡선 G(x) = x^s: Hausman, Schwarz & Graves (1976), Management Science 22(6)
  - 클래스 기반 저장(인기 SKU를 출고 지점 가까이): 같은 논문의 class-based storage.
  - 로봇 창고(RMFS) 대기행렬 모형의 포아송 주문 도착: Lamballais, Roy & De Koster (2017), EJOR.
  - 전자상거래 창고의 시간대별 수요 변동과 몰림(피크, 웨이브): Boysen, de Koster & Weidinger (2019), EJOR.
"""
import bisect
import math
import random
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple

Cell = Tuple[int, int]

DEFAULT_SEED     = 42
SIM_TIME         = 10 * 3600     # s, 10시간 근무
LAMBDA_DEFAULT   = 320 / 3600    # orders/s (평균 2.5토트 → 약 800토트/hr)
ORDER_LINES_MEAN = 2.5           # 주문당 평균 SKU(토트) 수, 기하분포
ORDER_LINES_MAX  = 10            # 주문당 최대 SKU 수 (기하분포 절단)
PICK_TIME        = 5.2           # s/토트, 랙 앞에서 토트를 꺼내 싣는 시간
WS_PROC_TIME     = 4.9           # s/토트, 워크스테이션에서 토트를 처리하는 시간

ARRIVALS      = ("poisson", "nhpp_peak", "nhpp_sine", "nhpp_steps", "batch", "renewal")
SKU_DISTS     = ("uniform", "abc", "zipf")
STORAGES      = ("random", "class_based")
TIME_DISTS    = ("fixed", "uniform", "exponential", "lognormal")


@dataclass
class Scenario:
    """시나리오 파라미터. 시간 단위는 초, 비율은 0~1."""
    # 주문 도착 과정 (평균 도착률은 모두 시뮬레이터의 λ와 동일)
    arrival:        str   = "poisson"
    peak_mult:      float = 2.0    # nhpp_peak: 피크 구간 도착률 = peak_mult × 평균 λ
    peak_start:     float = 0.3    # nhpp_peak: 피크 시작 (시뮬레이션 시간 대비 비율)
    peak_len:       float = 0.2    # nhpp_peak: 피크 길이 (비율). peak_mult × peak_len ≤ 1 필요
    sine_amp:       float = 0.5    # nhpp_sine: λ(t) = λ (1 + A sin(2πt / P)), 0 ≤ A ≤ 1
    sine_period:    float = 3600.0 # nhpp_sine: 주기 P (s)
    batch_mean:     float = 5.0    # batch: 한 번에 들어오는 평균 주문 수 (묶음 도착 = 웨이브)
    batch_fixed:    bool  = False  # batch: True면 묶음 크기 고정, False면 기하분포
    scv:            float = 1.0    # renewal: 도착 간격의 변동계수 제곱 (0 일정, <1 Erlang, 1 지수, >1 초지수)
    step_rates:     Tuple[float, ...] = ()   # nhpp_steps: 시뮬레이션 시간을 같은 길이 구간으로 나눈 구간별 λ 배수
                                             #   (평균 1이어야 함). 예: λ = 240, (200, 320, 200, 280, 200) / 240
    # 주문 크기: 주문 1건의 서로 다른 SKU 수 (SKU 1개 = 토트 1개). 1 + 기하분포, lines_max에서 절단
    lines_mean:     float = ORDER_LINES_MEAN   # 절단 전 평균 (1이면 단일 SKU 주문)
    lines_max:      int   = ORDER_LINES_MAX
    # SKU 수요 분포 (SKU id 0이 가장 인기)
    sku_dist:       str   = "uniform"
    abc_items:      float = 0.2    # abc: 상위 abc_items 비율의 SKU가
    abc_demand:     float = 0.8    #      abc_demand 비율의 수요를 차지 (G(x) = x^s)
    zipf_alpha:     float = 1.0    # zipf: p_i ∝ 1 / i^alpha
    # 저장 위치 배정
    storage:        str   = "random"
    # 피킹 (토트 1개) / 워크스테이션 처리 (토트 1개) 시간 분포
    pick_time:      float = PICK_TIME
    pick_dist:      str   = "fixed"
    pick_cv:        float = 0.5
    ws_proc_time:   float = WS_PROC_TIME
    ws_proc_dist:   str   = "fixed"
    ws_proc_cv:     float = 0.5

    def __post_init__(self):
        assert self.arrival in ARRIVALS, self.arrival
        assert self.sku_dist in SKU_DISTS, self.sku_dist
        assert self.storage in STORAGES, self.storage
        assert self.pick_dist in TIME_DISTS and self.ws_proc_dist in TIME_DISTS
        if self.arrival == "nhpp_peak" and self.peak_mult * self.peak_len > 1.0 + 1e-9:
            raise ValueError("nhpp_peak: peak_mult × peak_len이 1 이하여야 평균 λ를 맞출 수 있습니다")
        self.step_rates = tuple(float(x) for x in self.step_rates)
        if self.arrival == "nhpp_steps":
            if not self.step_rates or min(self.step_rates) < 0:
                raise ValueError("nhpp_steps: step_rates(구간별 λ 배수)가 필요합니다")
            if abs(sum(self.step_rates) / len(self.step_rates) - 1.0) > 1e-6:
                raise ValueError("nhpp_steps: step_rates 평균이 1이어야 평균 λ를 맞출 수 있습니다")
        if self.arrival == "nhpp_sine" and not (0.0 <= self.sine_amp <= 1.0):
            raise ValueError("nhpp_sine: sine_amp는 0~1이어야 합니다")
        self.lines_max = int(self.lines_max)
        if self.lines_mean < 1.0 or self.lines_max < 1:
            raise ValueError("lines_mean은 1 이상, lines_max는 1 이상이어야 합니다")
        if self.sku_dist == "abc" and not (0 < self.abc_items < 1 and 0 < self.abc_demand < 1):
            raise ValueError("abc: abc_items, abc_demand는 0과 1 사이여야 합니다")

    def to_dict(self) -> dict:
        """필드 dict (step_rates가 비어 있으면 빼서 이전 시나리오 기록·해시와 같게)"""
        d = asdict(self)
        if not d["step_rates"]:
            del d["step_rates"]
        else:
            d["step_rates"] = list(d["step_rates"])
        return d

    @property
    def is_default(self) -> bool:
        return self == Scenario()

    @classmethod
    def from_arg(cls, text: Optional[str]) -> "Scenario":
        """명령행 값(JSON 문자열 또는 JSON 파일 경로)으로 만듦. None/빈 문자열이면 기본 시나리오"""
        import json
        import os
        if not text:
            return cls()
        return cls(**json.loads(open(text).read() if os.path.isfile(text) else text))


class OrderStream:
    """시나리오에 따라 주문 도착 시각/묶음 크기, 주문 크기, SKU, 처리 시간 샘플링

    order_rng: 주문 크기/SKU/위치, 포아송 도착 간격 (OrderManager._rng)
    arrival_rng: 포아송이 아닌 도착 과정의 난수
    time_rng: 확률적 피킹/처리 시간의 난수
    """

    def __init__(self, sc: Scenario, lam: float, horizon: float, num_skus: int):
        self.sc, self.lam, self.horizon, self.num_skus = sc, lam, horizon, num_skus
        if sc.lines_max > num_skus:
            raise ValueError(f"lines_max {sc.lines_max} > SKU 수 {num_skus}")
        if sc.sku_dist == "abc":
            self._abc_s = math.log(sc.abc_demand) / math.log(sc.abc_items)
        if sc.sku_dist == "zipf":
            w = [1.0 / (i + 1) ** sc.zipf_alpha for i in range(num_skus)]
            tot = sum(w)
            acc, self._zipf_cdf = 0.0, []
            for x in w:
                acc += x / tot
                self._zipf_cdf.append(acc)

    # 도착

    def rate_factor(self, t: float) -> float:
        """비정상 포아송 도착률 배수 f(t) (시간 평균 1)"""
        sc = self.sc
        if sc.arrival == "nhpp_peak":
            s, l = sc.peak_start * self.horizon, sc.peak_len * self.horizon
            if s <= t < s + l:
                return sc.peak_mult
            return (1.0 - sc.peak_mult * sc.peak_len) / max(1e-9, 1.0 - sc.peak_len)
        if sc.arrival == "nhpp_sine":
            return 1.0 + sc.sine_amp * math.sin(2 * math.pi * t / sc.sine_period)
        if sc.arrival == "nhpp_steps":
            n = len(sc.step_rates)
            return sc.step_rates[min(n - 1, max(0, int(t / self.horizon * n)))]
        return 1.0

    def next_arrival(self, t: float, order_rng: random.Random,
                     arrival_rng: random.Random) -> Tuple[float, int]:
        """다음 도착 시각과 그때 들어오는 주문 수"""
        sc, lam = self.sc, self.lam
        if sc.arrival == "poisson":
            return t + order_rng.expovariate(lam), 1
        if sc.arrival in ("nhpp_peak", "nhpp_sine", "nhpp_steps"):
            if sc.arrival == "nhpp_peak":
                base  = (1.0 - sc.peak_mult * sc.peak_len) / max(1e-9, 1.0 - sc.peak_len)
                f_max = max(sc.peak_mult, base)
            elif sc.arrival == "nhpp_steps":
                f_max = max(sc.step_rates)
            else:
                f_max = 1.0 + sc.sine_amp
            while True:                                         # thinning
                t += arrival_rng.expovariate(lam * f_max)
                if arrival_rng.random() * f_max <= self.rate_factor(t):
                    return t, 1
        if sc.arrival == "batch":
            b = max(1.0, sc.batch_mean)
            t += arrival_rng.expovariate(lam / b)
            if sc.batch_fixed:
                return t, int(round(b))
            p = 1.0 / b                                         # 기하분포 (평균 b)
            k = 1 if p >= 1.0 else 1 + int(math.log(1.0 - arrival_rng.random()) / math.log(1.0 - p))
            return t, k
        # renewal
        scv, mean = sc.scv, 1.0 / lam
        if scv <= 1e-9:
            return t + mean, 1
        if scv < 1.0:
            k = max(1, round(1.0 / scv))
            return t + arrival_rng.gammavariate(k, mean / k), 1
        if abs(scv - 1.0) < 1e-9:
            return t + arrival_rng.expovariate(lam), 1
        p1 = 0.5 * (1.0 + math.sqrt((scv - 1.0) / (scv + 1.0)))  # 평균 균형 초지수(H2)
        mu = 2.0 * p1 * lam if arrival_rng.random() < p1 else 2.0 * (1.0 - p1) * lam
        return t + arrival_rng.expovariate(mu), 1

    # 주문 크기 / SKU

    def sample_lines(self, order_rng: random.Random) -> int:
        """주문 1건의 SKU 수: 1 + 기하분포 (평균 lines_mean), lines_max에서 절단"""
        m = self.sc.lines_mean
        if m <= 1.0:
            return 1
        p = 1.0 / m
        k = 1 + int(math.log(1.0 - order_rng.random()) / math.log(1.0 - p))
        return min(k, self.sc.lines_max)

    def sample_skus(self, order_rng: random.Random, sku_ids: List[int], n: int) -> List[int]:
        """수요 분포에서 서로 다른 SKU n개 (중복이면 다시 뽑음)"""
        out, seen = [], set()
        while len(out) < n:
            s = self.sample_sku(order_rng, sku_ids)
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def sample_sku(self, order_rng: random.Random, sku_ids: List[int]) -> int:
        sc = self.sc
        if sc.sku_dist == "uniform":
            return order_rng.choice(sku_ids)
        u = order_rng.random()
        if sc.sku_dist == "abc":
            idx = int(u ** (1.0 / self._abc_s) * self.num_skus)
        else:
            idx = bisect.bisect_left(self._zipf_cdf, u)
        return sku_ids[min(idx, self.num_skus - 1)]

    # 처리 시간

    def sample_time(self, mean: float, dist: str, cv: float, rng: random.Random) -> float:
        if dist == "fixed" or mean <= 0.0:
            return mean
        if dist == "uniform":
            h = math.sqrt(3.0) * cv * mean
            return max(0.0, rng.uniform(mean - h, mean + h))
        if dist == "exponential":
            return rng.expovariate(1.0 / mean)
        sig2 = math.log(1.0 + cv * cv)                           # lognormal
        return rng.lognormvariate(math.log(mean) - sig2 / 2.0, math.sqrt(sig2))

    def sample_pick(self, rng: random.Random) -> float:
        return self.sample_time(self.sc.pick_time, self.sc.pick_dist, self.sc.pick_cv, rng)

    def sample_proc(self, rng: random.Random) -> float:
        return self.sample_time(self.sc.ws_proc_time, self.sc.ws_proc_dist, self.sc.ws_proc_cv, rng)


# 주문 / 토트 관리

@dataclass(eq=False)
class Tote:
    """주문의 SKU 1개를 담은 토트 1개 (배정 단위)"""
    tote_id:      int
    order:        "Order"
    sku:          int
    loc_idx:      int
    loc:          Cell
    pick_s:       float = 0.0             # 랙에서 이 토트를 꺼내는 시간 (s)
    proc_s:       float = 0.0             # 워크스테이션에서 이 토트를 처리하는 시간 (s)
    assign_time:  Optional[float] = None  # 로봇에 배정된 시각
    done_time:    Optional[float] = None  # WS 처리 완료 시각
    picked:       bool  = False           # 로봇이 이미 집었는지

    @property
    def ws_id(self) -> int:
        return self.order.ws_id


@dataclass(eq=False)
class Order:
    """주문 1건: 서로 다른 SKU의 토트 목록, 처리할 워크스테이션"""
    order_id:           int
    arrival_time:       float
    totes:              List[Tote] = field(default_factory=list)
    ws_id:              int = -1
    service_start_time: Optional[float] = None   # 첫 토트가 로봇에 배정된 시각
    completion_time:    Optional[float] = None   # 마지막 토트의 WS 처리 완료 시각
    n_done:             int = 0


class OrderManager:
    """주문 생성(layout.scenario의 분포)과 토트 단위 FIFO 대기 큐, 주문 단위 지표 관리"""

    def __init__(self, layout, lam: float, seed: int = DEFAULT_SEED):
        """
        입력:
            layout: GridWarehouseLayout
            lam: 주문 도착률 (orders/s)
            seed: 주문 생성 난수 시드
        """
        self.layout           = layout
        self.lam               = lam
        self._rng               = random.Random(seed)
        self._cnt               = 0
        self._tote_cnt          = 0
        self.num_workstations   = layout.num_workstations

        self.wait_queue:          List[Tote]   = []
        self.all_orders:          List[Order]  = []
        self.all_totes:           List[Tote]   = []
        self._queue_entry_times:  List[float]  = []
        self.n_enqueued         = 0             # 도착 시 배정되지 못하고 대기 큐에 들어간 토트 수 (누적)
        self.horizon            = SIM_TIME      # 비정상 도착률 프로파일의 기준 시간 (SimulatorBase가 설정)
        self._seed_streams(seed)

    def _seed_streams(self, seed: int):
        """기본 분포가 아닌 도착 과정/처리 시간용 난수 생성기와 분포 샘플러 생성"""
        self.stream       = OrderStream(self.layout.scenario, self.lam, self.horizon, len(self.layout.sku_ids))
        self._arrival_rng = random.Random(f"{seed}-arrival")
        self._time_rng    = random.Random(f"{seed}-time")

    def reset(self, seed: int):
        """에피소드 시작 전 상태 초기화

        입력:
            seed: 새 주문 생성 난수 시드
        """
        self._rng      = random.Random(seed)
        self._cnt      = 0
        self._tote_cnt = 0
        self.wait_queue          = []
        self._queue_entry_times  = []
        self.n_enqueued          = 0
        self.all_orders.clear()
        self.all_totes.clear()
        self._seed_streams(seed)

    def next_arrival(self, current_time: float) -> Tuple[float, int]:
        """다음 주문 도착 시각과 그때 들어오는 주문 수 (묶음 도착이 아니면 1)

        입력:
            current_time: 현재 시각 (s)
        출력:
            (다음 도착 시각 (s), 주문 수)
        """
        return self.stream.next_arrival(current_time, self._rng, self._arrival_rng)

    def create_order(self, arrival_time: float) -> Order:
        """SKU 수, 서로 다른 SKU들, SKU마다 토트 위치를 샘플링해 새 주문 생성

        입력:
            arrival_time: 주문 도착 시각 (s)
        출력:
            생성된 Order (ws_id는 시뮬레이터가 정함)
        """
        order = Order(self._cnt, arrival_time)
        n     = self.stream.sample_lines(self._rng)
        for sku in self.stream.sample_skus(self._rng, self.layout.sku_ids, n):
            loc_idx, loc = self.layout.find_tote_location(sku, self._rng)
            tote = Tote(self._tote_cnt, order, sku, loc_idx, loc,
                        pick_s=self.stream.sample_pick(self._time_rng),
                        proc_s=self.stream.sample_proc(self._time_rng))
            self._tote_cnt += 1
            order.totes.append(tote)
            self.all_totes.append(tote)
        self._cnt += 1
        self.all_orders.append(order)
        return order

    def enqueue(self, tote: Tote, enqueue_time: float):
        """배정할 로봇이 없는 토트를 대기 큐 끝에 추가

        입력:
            tote: 대기시킬 토트
            enqueue_time: 큐 진입 시각 (s)
        """
        self.wait_queue.append(tote)
        self._queue_entry_times.append(enqueue_time)
        self.n_enqueued += 1

    def dequeue_up_to(self, n: int) -> List[Tote]:
        """대기 큐 앞에서 FIFO 순서로 최대 n개 꺼내기

        입력:
            n: 꺼낼 최대 토트 수
        출력:
            꺼낸 토트 리스트
        """
        k  = min(n, len(self.wait_queue))
        totes = self.wait_queue[:k]
        self.wait_queue         = self.wait_queue[k:]
        self._queue_entry_times = self._queue_entry_times[k:]
        return totes

    def queue_size(self) -> int:
        """현재 대기 큐 길이 (토트 수)"""
        return len(self.wait_queue)

    def record_assign(self, tote: Tote, time: float):
        """토트가 로봇에 배정된 시각 기록 (주문의 첫 배정 시각도 갱신)

        입력:
            tote: 대상 토트
            time: 배정 시각 (s)
        """
        if tote.assign_time is None:
            tote.assign_time = time
        o = tote.order
        if o.service_start_time is None:
            o.service_start_time = time

    def record_tote_done(self, tote: Tote, time: float) -> bool:
        """토트 WS 처리 완료 기록. 주문의 마지막 토트였으면 주문 완료 처리 후 True

        입력:
            tote: 처리를 마친 토트
            time: 완료 시각 (s)
        """
        tote.done_time = time
        o = tote.order
        o.n_done += 1
        if o.n_done == len(o.totes):
            o.completion_time = time
            return True
        return False

    def avg_wait_time(self) -> float:
        """주문 평균 대기시간 (첫 토트 배정 시각 − 도착 시각). 이동/서비스 시간 제외

        출력:
            배정된 주문들의 평균 대기시간 (s), 없으면 0.0
        """
        waits = [
            o.service_start_time - o.arrival_time
            for o in self.all_orders if o.service_start_time is not None
        ]
        return sum(waits) / len(waits) if waits else 0.0

    def avg_flow_time(self) -> float:
        """주문 평균 체류시간 (완료 시각 − 도착 시각). 대기 + 이동 + 처리 시간 모두 포함

        출력:
            완료된 주문들의 평균 체류시간 (s), 없으면 0.0
        """
        flows = [
            o.completion_time - o.arrival_time
            for o in self.all_orders if o.completion_time is not None
        ]
        return sum(flows) / len(flows) if flows else 0.0

    def censored_flow_time(self, t_end: float) -> float:
        """미완료 주문까지 포함한 평균 체류시간: 미완료 주문은 t_end − 도착 시각으로 자름 (하한 추정)

        avg_flow_time은 완료 주문만 평균하므로 미완료가 많은 과부하 조건에서 오래 걸리는 주문이 빠져 작게 나옴

        입력:
            t_end: 관측 종료 시각 (s, 보통 sim_time)
        출력:
            도착 주문 전체의 평균 (s), 도착 주문이 없으면 0.0
        """
        flows = [(o.completion_time if o.completion_time is not None else t_end) - o.arrival_time
                 for o in self.all_orders]
        return sum(flows) / len(flows) if flows else 0.0

    def avg_unfinished_age(self, t_end: float) -> float:
        """t_end 시점 미완료 주문의 평균 경과시간 (s), 미완료 주문이 없으면 0.0"""
        ages = [t_end - o.arrival_time for o in self.all_orders if o.completion_time is None]
        return sum(ages) / len(ages) if ages else 0.0

    def completed_count(self) -> int:
        """완료된 주문 수"""
        return sum(1 for o in self.all_orders if o.completion_time is not None)

    def totes_done_count(self) -> int:
        """WS 처리를 마친 토트 수"""
        return sum(1 for t in self.all_totes if t.done_time is not None)
