"""
metrics.py  —  실험 결과 컨테이너 및 비교 출력.

변경 사항
──────────
- EpisodeResult에 num_robots, saving_per_insertion 필드 추가
  (이전 실행에서 num_robots=0으로 유실되던 문제 수정)
- saving_per_insertion: extra_travel 기반 삽입 효율 지표
- robot_utilization, partial_trip_rate, mid_trip_count 필드 제거
- total_distance(총 이동거리), avg_totes_per_trip(trip당 평균 tote 수) 필드 추가
- n_completed는 throughput(완료 주문 수 기반 처리량)으로 표시
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EpisodeResult:
    label:                  str
    episode:                int
    lam:                    float
    num_robots:             int        # M 값 (이전에는 항상 0으로 기록되던 버그 수정)
    n_completed:            int
    n_arrived:              int
    avg_flow_time:          float
    avg_wait_time:          float
    dist_per_order:         float
    total_distance:         float = 0.0   # 총 이동 거리 (m)
    avg_totes_per_trip:     float = 0.0   # trip당 평균 tote(주문) 수
    mid_trip_rate:          float = 0.0
    extra_travel_m:         float = 0.0
    saving_per_insertion:   float = 0.0  # 삽입 1건당 절감 이동거리 (m)
    reward:                 float = 0.0
    epsilon:                float = 0.0

    @property
    def completion_rate(self) -> float:
        return self.n_completed / max(1, self.n_arrived)

    @property
    def throughput(self) -> float:
        return self.n_completed / (10 * 3600)


# 단독 왕복 거리 추정 기준 (창고 평균 픽업 거리 × 2)
_SOLO_RT_M = 80.0


class MetricsStore:
    def __init__(self):
        self.records: List[EpisodeResult] = []

    def add(self, r: EpisodeResult):
        self.records.append(r)

    def save_csv(self, path: str):
        if not self.records:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        extra  = ["completion_rate", "throughput"]
        fields = list(vars(self.records[0]).keys()) + extra
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in self.records:
                row = vars(r).copy()
                row["completion_rate"] = r.completion_rate
                row["throughput"]      = r.throughput
                w.writerow(row)

    def filter(self, label: Optional[str] = None,
               lam: Optional[float] = None,
               num_robots: Optional[int] = None,
               last_n: Optional[int] = None) -> List[EpisodeResult]:
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


# ── 테이블 출력 ──────────────────────────────────────────────────────────────

METRIC_DISPLAY = {
    "throughput":           (True,  "Throughput (orders/s)"),
    "completion_rate":      (True,  "Completion Rate"),
    "avg_flow_time":        (False, "Avg Flow Time (s)"),
    "avg_wait_time":        (False, "Avg Wait Time (s)"),
    "dist_per_order":       (False, "Dist/Order (m)"),
    "total_distance":       (False, "Total Distance (m)"),
    "avg_totes_per_trip":   (True,  "Totes/Trip"),
    "mid_trip_rate":        (True,  "Mid-Trip Rate"),
    "saving_per_insertion": (True,  "Saving/Insertion (m)"),
}


def _get_field(r: EpisodeResult, fname: str) -> float:
    if fname == "completion_rate":
        return r.completion_rate
    if fname == "throughput":
        return r.throughput
    return getattr(r, fname)


def compare_labels(store: MetricsStore, labels: List[str],
                   lam: Optional[float] = None,
                   num_robots: Optional[int] = None,
                   last_n: int = 10) -> Dict[str, Dict[str, float]]:
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
                     last_n: int = 10, title: str = "Comparison"):
    data  = compare_labels(store, labels, lam=lam,
                           num_robots=num_robots, last_n=last_n)
    lam_s = f"  λ={int(lam*3600)}/hr" if lam is not None else ""
    m_s   = f"  M={num_robots}" if num_robots is not None else ""
    cw    = max(max(len(l) for l in labels), 10) + 2
    hw    = 36
    sep   = "=" * (hw + cw * len(labels) + 12)

    print(f"\n{sep}")
    print(f"  {title}{lam_s}{m_s}  [last {last_n} ep]")
    print(sep)
    hdr = f"  {'Metric':<{hw}}" + "".join(f"{l:>{cw}}" for l in labels)
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
    if math.isnan(v):
        return "nan"
    if abs(v) >= 1e4:
        return f"{v:,.0f}"
    if abs(v) >= 1e2:
        return f"{v:,.1f}"
    if abs(v) >= 1.0:
        return f"{v:.3f}"
    return f"{v:.4f}"