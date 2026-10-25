"""에피소드 결과 컨테이너, CSV 저장, 정책 간 비교표 출력"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, fields
from typing import Dict, List, Optional

from constant import SIM_TIME


@dataclass
class EpisodeResult:
    """에피소드 1회의 결과 지표 (주문 단위: n_completed, n_arrived, 흐름/대기 시간, 주문당 거리)"""
    label:                  str
    episode:                int
    lam:                    float
    num_robots:             int
    n_completed:            int
    n_arrived:              int
    avg_flow_time:          float         # s, 주문 도착 -> 마지막 토트 WS 처리 완료
    avg_wait_time:          float         # s, 주문 도착 -> 첫 토트 로봇 배정
    dist_per_order:         float
    n_totes:                int   = 0     # 도착 주문의 토트 수
    n_totes_done:           int   = 0     # WS 처리를 마친 토트 수
    total_distance:         float = 0.0   # m, 총 이동거리
    avg_totes_per_trip:     float = 0.0   # 트립당 평균 처리 토트 수
    mid_trip_rate:          float = 0.0   # mid-trip 삽입 수 / 처리 토트 수
    extra_travel_m:         float = 0.0
    saving_per_insertion:   float = 0.0   # m, 삽입 1건당 절감 이동거리
    blocked_time:           float = 0.0   # robot·s, 다른 로봇에 막혀 기다린 시간
    n_replans:              int   = 0     # 우회 경로 재계획 횟수 (reactive 모드)
    n_plans:                int   = 0     # 시공간 경로계획 횟수 (reserve 모드)
    n_plan_fail:            int   = 0     # 경로계획 실패 횟수 (reserve 모드)
    plan_time_s:            float = 0.0   # 경로계획에 쓴 실행 시간 (reserve 모드)
    sim_time:               float = SIM_TIME   # s, 시뮬레이션 시간
    reward:                 float = 0.0   # RL: 완료 주문 1건당 누적 보상 (shaping 제외)
    epsilon:                float = 0.0   # RL: 에피소드 종료 시 ε
    seed:                   int   = -1    # 주문 생성 시드 (평가: 테스트/검증 시드)
    n_decisions:            int   = 0     # 도착 시 정책이 바로 배정한 토트 수 (RL: 배정 결정 수)
    bl6_agree:              float = float("nan")   # RL 평가: 선택 로봇이 BL6 argmin과 같은 결정 비율
    avg_flow_time_censored: float = 0.0   # s, 미완료 주문을 sim_time에서 자른 도착 주문 전체 평균
    avg_unfinished_age:     float = 0.0   # s, sim_time 시점 미완료 주문의 평균 경과시간
    n_queued_totes:         int   = 0     # 도착 시 후보가 없어 대기 큐로 간 토트 수 (나중에 결정 없이 FIFO 방출)
    avg_candidates:         float = 0.0   # RL: 결정당 평균 후보 로봇 수
    n_qpen_active:          int   = 0     # RL: 결정 시점 대기 큐가 비어 있지 않았던 결정 수 (legacy 큐 압박 항 활성)
    n_depletion_active:     int   = 0     # RL: 마지막 빈 idle 로봇을 쓴 결정 수 (legacy 소진 페널티 활성)
    # 실행 시간 (삽입형 정책과 RL만, 버퍼형은 0)
    decision_ms_mean:       float = 0.0   # ms, 결정 1건 선택 지연 (후보 탐색·detour·특징·추론, 기록기 제외)
    decision_ms_p95:        float = 0.0
    decision_full_ms_mean:  float = 0.0   # ms, 선택 + 배정·경로 재계획(예약표) 포함
    decision_full_ms_p95:   float = 0.0
    wall_s:                 float = 0.0   # s, 에피소드 1회 실행 시간 (학습이면 업데이트 포함)

    @property
    def decision_rate(self) -> float:
        """도착 토트 중 정책이 바로 배정한 비율 (나머지는 대기 큐 경유)"""
        return self.n_decisions / max(1, self.n_totes)

    @property
    def completion_rate(self) -> float:
        """완료 주문 수 / 도착 주문 수"""
        return self.n_completed / max(1, self.n_arrived)

    @property
    def throughput(self) -> float:
        """초당 완료 주문 수 (orders/s)"""
        return self.n_completed / self.sim_time


class MetricsStore:
    """EpisodeResult 목록 수집, 필터링, CSV 저장"""

    def __init__(self):
        self.records: List[EpisodeResult] = []

    def add(self, r: EpisodeResult):
        """결과 1건 추가

        입력:
            r: 추가할 EpisodeResult
        """
        self.records.append(r)

    def save_csv(self, path: str):
        """전체 결과 CSV 저장 (completion_rate, throughput 열 포함)

        입력:
            path: 저장 경로 (상위 디렉터리는 자동 생성)
        """
        if not self.records:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        extra  = ["completion_rate", "throughput", "decision_rate"]
        fields = list(vars(self.records[0]).keys()) + extra
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in self.records:
                row = vars(r).copy()
                row["completion_rate"] = r.completion_rate
                row["throughput"]      = r.throughput
                row["decision_rate"]   = r.decision_rate
                w.writerow(row)

    @classmethod
    def load_csv(cls, path: str) -> "MetricsStore":
        """save_csv로 저장한 CSV를 다시 읽음 (없는 열은 기본값, 모르는 열은 무시)"""
        types = {f.name: f.type for f in fields(EpisodeResult)}
        conv  = {"int": lambda v: int(float(v)), "float": float, "str": str}
        store = cls()
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                kw = {k: conv[types[k]](v) for k, v in row.items() if k in types and v != ""}
                store.add(EpisodeResult(**kw))
        return store

    def filter(self, label: Optional[str] = None,
               lam: Optional[float] = None,
               num_robots: Optional[int] = None,
               last_n: Optional[int] = None) -> List[EpisodeResult]:
        """조건에 맞는 결과만 선별 (None인 조건은 무시)

        입력:
            label: 정책 라벨
            lam: 주문 도착률
            num_robots: 로봇 수
            last_n: 조건을 적용한 뒤 마지막 n개만 사용
        출력:
            조건에 맞는 EpisodeResult 리스트
        """
        rs = self.records
        if label is not None:
            rs = [r for r in rs if r.label == label]
        if lam is not None:
            rs = [r for r in rs if abs(r.lam - lam) < 1e-9]
        if num_robots is not None:
            rs = [r for r in rs if r.num_robots == num_robots]
        if last_n is not None:
            rs = rs[-last_n:]
        return rs


@dataclass
class TrainingPoint:
    """학습 곡선의 에피소드 1개 지점"""
    label:         str
    episode:       int
    reward:        float
    loss:          float
    epsilon:       float
    n_completed:   int
    avg_flow_time: float
    num_robots:    int = 0     # 이 에피소드의 로봇 수 M (다중 M 학습)
    wall_s:        float = 0.0 # s, 에피소드 전체 실행 시간 (시뮬레이션 + 버퍼 적재 + 업데이트, 검증 제외)
    sim_s:         float = 0.0 # s, 그중 시뮬레이션 (결정 추론 포함)
    n_updates:     int = 0     # 이 에피소드의 train_step 호출 수
    lam:           float = float("nan")   # orders/hr, 이 에피소드의 주문 도착률 (커리큘럼)
    alpha:         float = float("nan")   # 에피소드 끝 잔차 계수 α (GlobalDQNAgent)
    q_mean:        float = float("nan")   # 이 에피소드 업데이트들의 평균 Q (배치 평균)
    td_abs:        float = float("nan")   # 이 에피소드 업데이트들의 평균 |TD 오차|
    local_mu:      float = float("nan")   # s, local 보상 기준점 μ
    local_cost:    float = float("nan")   # s, 결정당 평균 국소 비용 (μ와 무관, 보상 추세 대신 제시용)


class TrainingCurveStore:
    """학습 곡선 지점(TrainingPoint) 수집 및 CSV 저장"""

    def __init__(self):
        self.records: List[TrainingPoint] = []

    def add(self, p: TrainingPoint):
        """지점 1개 추가"""
        self.records.append(p)

    def save_csv(self, path: str):
        """전체 지점 CSV 저장

        입력:
            path: 저장 경로 (상위 디렉터리는 자동 생성)
        """
        if not self.records:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fields = list(vars(self.records[0]).keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in self.records:
                w.writerow(vars(r))


# 지표 이름 -> (값이 클수록 좋은지, 표시 이름)
METRIC_DISPLAY = {
    "throughput":           (True,  "Throughput (orders/s)"),
    "completion_rate":      (True,  "Completion Rate"),
    "avg_flow_time":        (False, "Avg Flow Time (s)"),
    "avg_flow_time_censored": (False, "Avg Flow Time, censored (s)"),
    "avg_wait_time":        (False, "Avg Wait Time (s)"),
    "decision_rate":        (True,  "Decision Rate"),
    "dist_per_order":       (False, "Dist/Order (m)"),
    "total_distance":       (False, "Total Distance (m)"),
    "avg_totes_per_trip":   (True,  "Totes/Trip"),
    "mid_trip_rate":        (True,  "Mid-Trip Rate"),
    "saving_per_insertion": (True,  "Saving/Insertion (m)"),
}

# 코드 라벨 -> 논문 표기 (라벨 자체는 바꾸지 않고 비교표/분석 표에서만 사용)
PAPER_NAMES = {
    "BL1F_FillBuffer":  "BL1",
    "BL1_FullBuffer":   "BL1-Spread",
    "BL2_Partial_k3":   "BL2",
    "BL3_Random":       "BL3",
    "BL4_Nearest":      "BL4",
    "BL5_MinInsertion": "BL5",
    "BL6_MarginalCost": "BL6",
    "Greedy_R":         "BL7",
}


def display_name(label: str) -> str:
    """코드 라벨의 논문 표기 (매핑에 없으면 그대로)"""
    return PAPER_NAMES.get(label, label)


def _get_field(r: EpisodeResult, fname: str) -> float:
    """EpisodeResult의 지표 값 읽기 (property 지표 포함)

    입력:
        r: 대상 결과
        fname: 지표 이름
    출력:
        지표 값
    """
    if fname == "completion_rate":
        return r.completion_rate
    if fname == "throughput":
        return r.throughput
    if fname == "decision_rate":
        return r.decision_rate
    return getattr(r, fname)


def compare_labels(store: MetricsStore, labels: List[str],
                   lam: Optional[float] = None,
                   num_robots: Optional[int] = None,
                   last_n: int = 10) -> Dict[str, Dict[str, float]]:
    """정책별 각 지표의 평균 계산

    입력:
        store: 결과 저장소
        labels: 비교할 정책 라벨 리스트
        lam, num_robots: 필터 조건
        last_n: 정책별로 마지막 n개 결과만 평균
    출력:
        {지표 이름: {라벨: 평균값}}, 결과가 없으면 nan
    """
    result: Dict[str, Dict[str, float]] = {}
    for fname in METRIC_DISPLAY:
        result[fname] = {}
        for lbl in labels:
            rs = store.filter(label=lbl, lam=lam, num_robots=num_robots,
                              last_n=last_n)
            if rs:
                vals = [_get_field(r, fname) for r in rs]
                result[fname][lbl] = sum(vals) / len(vals)
            else:
                result[fname][lbl] = float("nan")
    return result


def print_comparison(store: MetricsStore, labels: List[str],
                     lam: Optional[float] = None,
                     num_robots: Optional[int] = None,
                     last_n: int = 10, title: str = "Comparison", paper_names: bool = True):
    """정책별 지표 평균을 표로 출력 (지표마다 가장 좋은 값에 * 표시)

    입력:
        store: 결과 저장소
        labels: 비교할 정책 라벨 리스트
        lam, num_robots: 필터 조건
        last_n: 정책별로 마지막 n개 결과만 평균
        title: 표 제목
        paper_names: True이면 열 제목을 논문 표기(PAPER_NAMES)로
    """
    data  = compare_labels(store, labels, lam=lam,
                           num_robots=num_robots, last_n=last_n)
    lam_s = f"  λ={int(lam*3600)}/hr" if lam is not None else ""
    m_s   = f"  M={num_robots}" if num_robots is not None else ""
    shown = {l: (display_name(l) if paper_names else l) for l in labels}
    cw    = max(max(len(l) for l in shown.values()), 10) + 2
    hw    = 36
    sep   = "=" * (hw + cw * len(labels) + 12)

    print(f"\n{sep}")
    print(f"  {title}{lam_s}{m_s}  [last {last_n} ep]")
    print(sep)
    hdr = f"  {'Metric':<{hw}}" + "".join(f"{shown[l]:>{cw}}" for l in labels)
    print(hdr)
    print("  " + "-" * (hw + cw * len(labels) + 10))

    for fname, (higher_better, display) in METRIC_DISPLAY.items():
        vals  = data[fname]
        bests = [l for l in labels if not math.isnan(vals.get(l, float("nan")))]
        ref   = None
        if bests:
            ref = max(bests, key=lambda l: vals[l]) if higher_better \
                  else min(bests, key=lambda l: vals[l])
        row = f"  {display:<{hw}}"
        for lbl in labels:
            v  = vals.get(lbl, float("nan"))
            s  = _fmt(v)
            mk = "*" if lbl == ref and len(labels) > 1 else " "
            row += f"{(s+mk):>{cw}}"
        print(row)
    print(sep)


def _fmt(v: float) -> str:
    """값 크기에 따라 자릿수를 달리해 문자열로 변환

    입력:
        v: 숫자
    출력:
        포맷된 문자열
    """
    if math.isnan(v):
        return "nan"
    if abs(v) >= 1e4:
        return f"{v:,.0f}"
    if abs(v) >= 1e2:
        return f"{v:,.1f}"
    if abs(v) >= 1.0:
        return f"{v:.3f}"
    return f"{v:.4f}"
