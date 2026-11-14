"""정책 평가 하네스와 결과 기록: 베이스라인과 RL 체크포인트를 같은 함수, 같은 시드 규칙으로 평가

시드 구간 (서로 겹치지 않음)
  - 학습 에피소드: TRAIN_SEED_BASE + 학습 시드 × TRAIN_SEED_STRIDE + 에피소드 번호
  - 검증 (체크포인트·베이스라인 선택): VAL_SEED_BASE + i
  - 최종 테스트 (보고): EVAL_SEED_BASE + i — 모든 정책·학습 시드가 같은 테스트 인스턴스 (대응 비교)
모든 평가 결과는 EpisodeResult.seed에 주문 생성 시드를 남기고, episode = 시드 리스트 인덱스

평가 조건 (Condition): 이동 가정(traffic, motion)과 경로 구성 규칙(route)
  - 결과 폴더 이름 setting = grid_{traffic}_{motion}[_sc해시][_route-{규칙}]  (route = nnh이면 꼬리표 없음)
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Optional, Tuple

from baselines import (BL6_CD_GRID, POLICIES, POLICY_ALIASES, BF_SUFFIX, resolve_policy,  # noqa: F401
                       policy_arg, policy_key, policy_label)
from env.battery import BatteryConfig
from env.layout import GRID_MOTION, GRID_TRAFFIC, ROBOT_CAPACITY
from env.orders import DEFAULT_SEED, SIM_TIME
from env.simulator import EpisodeResult

N_EVAL            = 20
EVAL_SEED_BASE    = 10000        # 최종 테스트 시드 10000 + i (보고용, 선택/튜닝에 쓰지 않음)
VAL_SEED_BASE     = 20000        # 검증 시드 20000 + i (체크포인트·베이스라인 선택 전용)
N_VAL             = 5
TRAIN_SEED_BASE   = 1_000_000    # 학습 에피소드 시드 = BASE + 학습 시드 × STRIDE + 에피소드 번호
TRAIN_SEED_STRIDE = 100_000


def write_csv(rows: List[dict], path: str, columns: Optional[List[str]] = None):
    """dict 행 목록을 CSV로 (columns가 없으면 행들의 열 합집합, 첫 행 순서 우선). 행이 없으면 저장 안 함"""
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    columns = columns or list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


# 결과 기록

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
        write_csv([{**vars(r), "completion_rate": r.completion_rate, "throughput": r.throughput,
                    "decision_rate": r.decision_rate} for r in self.records], path)

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
    alpha:         float = float("nan")   # 에피소드 끝 잔차 계수 α (agents.ScoreNet)
    q_mean:        float = float("nan")   # 이 에피소드 업데이트들의 평균 Q (배치 평균)
    td_abs:        float = float("nan")   # 이 에피소드 업데이트들의 평균 |TD 오차|
    local_mu:      float = float("nan")   # s, local 보상 기준점 μ
    local_cost:    float = float("nan")   # s, 결정당 평균 국소 비용 (μ와 무관, 보상 추세 대신 제시용)
    cost_own:      float = float("nan")   # s, local 비용 항별 결정당 평균 (계수 곱 전): 자기 토트 지연
    cost_ext:      float = float("nan")   # s, 탑재 토트 지연 추정
    cost_queue:    float = float("nan")   # s, 직후 큐잉 토트 지연


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
        write_csv([vars(r) for r in self.records], path)


# 지표 이름 -> (값이 클수록 좋은지, 표시 이름)
METRIC_DISPLAY = {
    "throughput":           (True,  "Throughput (orders/s)"),
    "completion_rate":      (True,  "Completion Rate"),
    "avg_flow_time":        (False, "Avg Flow Time (s)"),
    "avg_flow_time_censored": (False, "Avg Flow Time, censored (s)"),
    "flow_p99":             (False, "Flow Time p99 (s)"),
    "avg_wait_time":        (False, "Avg Wait Time (s)"),
    "decision_rate":        (True,  "Decision Rate"),
    "dist_per_order":       (False, "Dist/Order (m)"),
    "total_distance":       (False, "Total Distance (m)"),
    "avg_totes_per_trip":   (True,  "Totes/Trip"),
    "mid_trip_rate":        (True,  "Mid-Trip Rate"),
    "saving_per_insertion": (True,  "Saving/Insertion (m)"),
    "utilization":          (False, "Utilization"),
    "idle_at_decision":     (True,  "Idle Robots at Decision"),
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


# 평가

@dataclass
class Condition:
    """평가 조건 (이동 가정 + 경로 구성 규칙). 기본값 = 본실험 조건"""
    traffic: str = GRID_TRAFFIC
    motion:  str = GRID_MOTION
    route:   str = "nnh"          # 경로 구성 규칙 (env.robot.ROUTE_MODES)
    grid:    Optional[Tuple[int, int, int]] = None   # (블록 수, 블록당 통로 수, 선반 행 수), None = 기본 레이아웃
    collisions: bool = True       # False이면 reactive에서 로봇 간 점유 무시 (회귀 검증용)
    battery: Optional[BatteryConfig] = None   # 배터리 제약 (env.battery), None = 제약 없음

    @property
    def sim_kw(self) -> Dict:
        """SimulatorBase 추가 인자 (기본이면 빈 dict)"""
        kw = {"route": self.route} if self.route != "nnh" else {}
        if self.battery is not None:
            kw["battery"] = self.battery
        return kw

    @property
    def tag(self) -> str:
        """조건 꼬리표 (기본이면 빈 문자열, fixed_dock → _route-fixeddock, 배터리 → BatteryConfig.tag)"""
        route = f"_route-{self.route.replace('_', '')}" if self.route != "nnh" else ""
        return route + (self.battery.tag if self.battery is not None else "")

    def layout(self, num_robots: int, num_workstations: int, scenario=None):
        """이 조건의 GridWarehouseLayout"""
        from env.layout import GridWarehouseLayout
        return GridWarehouseLayout(seed=DEFAULT_SEED, num_robots=num_robots, robot_capacity=ROBOT_CAPACITY,
                                   num_workstations=num_workstations, traffic=self.traffic,
                                   motion=self.motion, scenario=scenario, dims=self.grid,
                                   collisions=self.collisions)

    def setting(self, scenario=None) -> str:
        """결과 폴더 이름 (모듈 설명)"""
        name = f"grid_{self.traffic}_{self.motion}" + ("" if self.collisions else "_nocoll")
        if self.grid is not None:
            name += "_g" + "x".join(str(x) for x in self.grid)
        if scenario is not None and not scenario.is_default:
            name += "_sc" + hashlib.sha1(json.dumps(scenario.to_dict(), sort_keys=True).encode()).hexdigest()[:6]
        return name + self.tag

    def to_dict(self) -> Dict:
        """필드 dict (기본값인 grid, collisions, battery는 빼서 이전 기록과 같게)"""
        d = asdict(self)
        if self.battery is None:
            del d["battery"]
        if self.grid is None:
            del d["grid"]
        if self.collisions:
            del d["collisions"]
        return d


def test_seeds(n: int) -> List[int]:
    """최종 테스트 시드 n개"""
    return list(range(EVAL_SEED_BASE, EVAL_SEED_BASE + n))


def val_seeds(n: int) -> List[int]:
    """검증 시드 n개 (체크포인트·베이스라인 선택 전용)"""
    return list(range(VAL_SEED_BASE, VAL_SEED_BASE + n))


def train_episode_seed(train_seed: int, episode: int) -> int:
    """학습 시드 train_seed의 episode번째 (1부터) 에피소드 주문 생성 시드"""
    assert 0 <= episode < TRAIN_SEED_STRIDE, episode
    return TRAIN_SEED_BASE + train_seed * TRAIN_SEED_STRIDE + episode


def make_sim(policy: str, layout, lam: float, seed: int, sim_time: float = SIM_TIME,
             agent=None, arm=None, sim_kw: Optional[Dict] = None):
    """정책 키로 시뮬레이터 생성 (로봇 수/용량은 layout을 따름)

    입력:
        policy: POLICIES 키(별칭 BL7 포함) 또는 "RL"
        layout: GridWarehouseLayout
        lam: 주문 도착률 (orders/s)
        seed: 주문 생성 시드
        sim_time: 시뮬레이션 시간 (s)
        agent, arm: policy == "RL"일 때 DQNAgent와 보상 설정 (arm은 기록되는 reward에만 영향)
        sim_kw: 시뮬레이터 추가 인자 (Condition.sim_kw의 route)
    """
    kw = dict(lam=lam, seed=seed, sim_time=sim_time,
              num_robots=layout.num_robots, robot_capacity=layout.robot_capacity, **(sim_kw or {}))
    if policy == "RL":
        from td import rl_sim_class
        assert agent is not None, "RL 평가에는 agent가 필요합니다"
        return rl_sim_class(arm, train=False)(layout, agent, eval_mode=True, arm=arm, **kw)
    return resolve_policy(policy)[1](layout, **kw)


def evaluate_policy(layout, policy: str, lam: float, seeds: List[int],
                    sim_time: float = SIM_TIME, label: Optional[str] = None,
                    agent=None, arm=None, store: Optional[MetricsStore] = None,
                    track_decisions: bool = True, verbose: bool = True,
                    sim_kw: Optional[Dict] = None) -> MetricsStore:
    """정책 하나를 시드마다 한 번씩 실행해 기록 (RL은 ε = 0)

    입력:
        layout: GridWarehouseLayout
        policy: POLICIES 키 또는 "RL"
        lam: 주문 도착률 (orders/s)
        seeds: 주문 생성 시드 리스트 (test_seeds / val_seeds)
        sim_time: 시뮬레이션 시간 (s)
        label: 결과 라벨 (None이면 POLICIES 라벨, RL은 "RL")
        agent, arm: RL 전용 (make_sim 참고)
        store: 결과를 추가할 MetricsStore (None이면 새로 만듦)
        track_decisions: RL이면 DecisionStats로 BL6 argmin 일치율 기록
        verbose: 시드별 한 줄 출력
        sim_kw: 시뮬레이터 추가 인자 (make_sim 참고)
    출력:
        결과가 담긴 MetricsStore
    """
    if label is None:
        label = "RL" if policy == "RL" else policy_label(policy)
    store = store if store is not None else MetricsStore()
    for i, s in enumerate(seeds):
        sim = make_sim(policy, layout, lam, s, sim_time, agent=agent, arm=arm, sim_kw=sim_kw)
        stats = None
        if policy == "RL" and track_decisions:
            from viewer.recorder import DecisionStats
            stats = sim.recorder = DecisionStats()
        wall = time.perf_counter()
        sim.reset(s)
        sim.run()
        r = sim.build_result(label, i, lam)
        r.seed   = s
        r.wall_s = time.perf_counter() - wall
        if stats is not None:
            summ = stats.summary()
            r.n_decisions, r.bl6_agree = summ["n_decisions"], summ["agree"]
            r.agree_cd0, r.agree_bl8, r.cd_reproducible = summ["agree_cd0"], summ["agree_bl8"], summ["cd_reproducible"]
        store.add(r)
        if verbose:
            agree = f", BL6 일치 {r.bl6_agree:.3f}" if stats is not None else ""
            print(f"  {label:<18} M={layout.num_robots} seed {s}: flow {r.avg_flow_time:7.1f} s, "
                  f"완료 {r.n_completed}/{r.n_arrived}, 거리/주문 {r.dist_per_order:5.1f} m{agree}")
    return store


def evaluate_baselines_cached(layout, policies: List[str], lam: float, seeds: List[int],
                              sim_time: float, path: str, sim_kw: Optional[Dict] = None) -> MetricsStore:
    """베이스라인 평가 결과를 path(CSV)에 누적: 이미 있는 (라벨, 시드)는 건너뛰고 빠진 것만 실행

    입력:
        layout, lam, seeds, sim_time, sim_kw: evaluate_policy와 같음 (path는 같은 조건의 폴더여야 함)
        policies: POLICIES 키 리스트
        path: 결과 CSV (같은 셀의 baselines/eval.csv 또는 val.csv)
    출력:
        요청한 정책 × 시드 결과만 담은 MetricsStore (episode = seeds 인덱스)
    """
    old  = MetricsStore.load_csv(path) if os.path.isfile(path) else MetricsStore()
    have = {(r.label, r.seed): r for r in old.records
            if r.seed >= 0 and abs(r.sim_time - sim_time) < 1e-6}
    for key in policies:
        label = policy_label(key)
        todo  = [s for s in seeds if (label, s) not in have]
        if todo:
            for r in evaluate_policy(layout, key, lam, todo, sim_time, sim_kw=sim_kw).records:
                have[(label, r.seed)] = r
    # 시뮬레이션 시간이 다른 행은 그대로 둠
    merged = MetricsStore()
    for r in old.records:
        if r.seed >= 0 and abs(r.sim_time - sim_time) >= 1e-6:
            merged.add(r)
    for (label, s), r in sorted(have.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        merged.add(r)
    merged.save_csv(path)

    out = MetricsStore()
    for key in policies:
        for i, s in enumerate(seeds):
            r = have[(policy_label(key), s)]
            r.episode = i
            out.add(r)
    return out
