"""주문 도착 과정, SKU 수요, 저장 위치, 피킹/워크스테이션 처리 시간 분포 설정.

Scenario() 기본값: 포아송 도착, 균등 SKU 수요, 무작위 저장, 고정 피킹/처리 시간
(constant.PICK_TIME, WS_PROC_TIME). SKU/위치와 포아송 도착 간격은 OrderManager._rng에서,
그 밖의 도착 과정과 처리 시간은 별도 생성기(arrival_rng, time_rng)에서 뽑는다.

참고 문헌:
  - ABC 수요 곡선 G(x) = x^s: Hausman, Schwarz & Graves (1976), Management Science 22(6).
  - 클래스 기반 저장(인기 SKU를 출고 지점 가까이): 같은 논문의 class-based storage.
  - 로봇 창고(RMFS) 대기행렬 모형의 포아송 주문 도착: Lamballais, Roy & De Koster (2017), EJOR.
  - 전자상거래 창고의 시간대별 수요 변동과 몰림(피크, 웨이브): Boysen, de Koster & Weidinger (2019), EJOR.
"""
import bisect
import math
import random
from dataclasses import dataclass, asdict
from typing import List, Tuple

from constant import PICK_TIME, WS_PROC_TIME

ARRIVALS      = ("poisson", "nhpp_peak", "nhpp_sine", "batch", "renewal")
SKU_DISTS     = ("uniform", "abc", "zipf")
STORAGES      = ("random", "class_based")
TIME_DISTS    = ("fixed", "uniform", "exponential", "lognormal")


@dataclass
class Scenario:
    """시나리오 파라미터. 시간 단위는 초, 비율은 0~1."""
    # 주문 도착 과정 (평균 도착률은 모두 시뮬레이터의 λ와 같게 맞춘다)
    arrival:        str   = "poisson"
    peak_mult:      float = 2.0    # nhpp_peak: 피크 구간 도착률 = peak_mult × 평균 λ
    peak_start:     float = 0.3    # nhpp_peak: 피크 시작 (시뮬레이션 시간 대비 비율)
    peak_len:       float = 0.2    # nhpp_peak: 피크 길이 (비율). peak_mult × peak_len ≤ 1이어야 한다
    sine_amp:       float = 0.5    # nhpp_sine: λ(t) = λ (1 + A sin(2πt / P)), 0 ≤ A ≤ 1
    sine_period:    float = 3600.0 # nhpp_sine: 주기 P (s)
    batch_mean:     float = 5.0    # batch: 한 번에 들어오는 평균 주문 수 (묶음 도착 = 웨이브)
    batch_fixed:    bool  = False  # batch: True면 묶음 크기 고정, False면 기하분포
    scv:            float = 1.0    # renewal: 도착 간격의 변동계수 제곱 (0 일정, <1 Erlang, 1 지수, >1 초지수)
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
        if self.arrival == "nhpp_sine" and not (0.0 <= self.sine_amp <= 1.0):
            raise ValueError("nhpp_sine: sine_amp는 0~1이어야 합니다")
        if self.sku_dist == "abc" and not (0 < self.abc_items < 1 and 0 < self.abc_demand < 1):
            raise ValueError("abc: abc_items, abc_demand는 0과 1 사이여야 합니다")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_default(self) -> bool:
        return self == Scenario()


class OrderStream:
    """시나리오에 따라 주문 도착 시각/묶음 크기, SKU, 처리 시간을 뽑는다.

    order_rng: SKU/위치, 포아송 도착 간격 (OrderManager._rng)
    arrival_rng: 포아송이 아닌 도착 과정의 난수
    time_rng: 확률적 피킹/처리 시간의 난수
    """

    def __init__(self, sc: Scenario, lam: float, horizon: float, num_skus: int):
        self.sc, self.lam, self.horizon, self.num_skus = sc, lam, horizon, num_skus
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
        """비정상 포아송 도착률 배수 f(t) (시간 평균 1)."""
        sc = self.sc
        if sc.arrival == "nhpp_peak":
            s, l = sc.peak_start * self.horizon, sc.peak_len * self.horizon
            if s <= t < s + l:
                return sc.peak_mult
            return (1.0 - sc.peak_mult * sc.peak_len) / max(1e-9, 1.0 - sc.peak_len)
        if sc.arrival == "nhpp_sine":
            return 1.0 + sc.sine_amp * math.sin(2 * math.pi * t / sc.sine_period)
        return 1.0

    def next_arrival(self, t: float, order_rng: random.Random,
                     arrival_rng: random.Random) -> Tuple[float, int]:
        """다음 도착 시각과 그때 들어오는 주문 수."""
        sc, lam = self.sc, self.lam
        if sc.arrival == "poisson":
            return t + order_rng.expovariate(lam), 1
        if sc.arrival in ("nhpp_peak", "nhpp_sine"):
            if sc.arrival == "nhpp_peak":
                base  = (1.0 - sc.peak_mult * sc.peak_len) / max(1e-9, 1.0 - sc.peak_len)
                f_max = max(sc.peak_mult, base)
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

    # SKU

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
