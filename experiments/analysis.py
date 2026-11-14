"""사후 분석: 저장된 실험 결과의 통계와 논문 표 생성 (runs/<실험 이름> → results/<실험 이름>)

# 통계

입력: experiments/run_compare.py / run_plans.py가 남긴 {root}/{setting}/lam{λ}_M{M}_K{K}/{run}/eval.csv와 baselines/eval.csv
  - 정책 그룹: 베이스라인은 라벨, RL은 run 이름에서 학습 시드(_s{seed})를 뺀 것
    (예: local_r8_ddqn_v2_s3_tr14_trlam240 → local_r8_ddqn_v2_tr14_trlam240, 학습 시드 3)
  - 모든 정책·학습 시드가 같은 테스트 인스턴스(시드)를 쓰므로 인스턴스 대응 비교

비교 (정책 − 기준, 작을수록 정책이 좋음)
  - seed_level_compare: RL (학습 시드 있음). 학습 시드 = 반복 단위
      시드별 평균 차이 m_i (공통 인스턴스), 보고 평균 = mean_i m_i
      CI: 시드→인스턴스 nested 부트스트랩 (ci_*), 시드 간 t 구간 (ci_t_*, 자유도 n_seed − 1)
      ΔF%: 평균 비율 (nested 부트스트랩 rel_ci_*, 시드별 log 비의 t 구간 rel_ci_t_*)
      p: 시드 간 대응 t검정 (학습 시드 1개면 인스턴스 간)
      시드가 적으면 percentile 부트스트랩은 시드 값 범위에 가까워 포함률이 낮음 → t 구간을 함께 보고
  - matched_compare: 같은 네트워크의 두 조건 (예: 경로 규칙 fixed_dock vs nnh), (학습 시드, 인스턴스) 대응
  - rule_compare: 규칙 기반끼리 인스턴스 간 대응 t
  - holm_table: 표 단위 Holm 보정
  - factorial_effects_by_unit: 2^k 요인 효과, 반복 단위 = 네트워크 (네트워크 내 요인과 네트워크 간 요인 분리)


# 표

논문 표 CSV와 학습 곡선 요약 (runs/<실험 이름> 입력) → results/<실험 이름>/tables/

사용법:
    python -m experiments.analysis                                   # 모든 표
    python -m experiments.analysis --tables tab_main tab_fleet --B 2000
    python -m experiments.analysis --arm <새 arm>                      # 기본 arm 대신 다른 arm을 RL 행으로

지표: F = avg_flow_time (완료 주문 평균 흐름시간), J = F + c_d · 주문당 이동거리 (metric_value)
기준(ref): 셀의 select.json best_label (검증 시드로 고른 최고 삽입형 규칙, BL4–BL7), 없으면 BL6_MarginalCost
  --pool new: best_label_new (BL4–BL8 + BL6*, run_stages stage0이 기록)
통계 (위 설명): RL은 학습 시드 단위, 규칙끼리는 인스턴스 대응, 같은 네트워크의 조건 비교는
  (학습 시드, 인스턴스) 대응. Holm 보정은 표 단위 (표의 모든 비교 × F, J)
출력 표 (run_plans 실험 블록)
  tab_main            : main 셀마다 RL·모든 베이스라인 vs 기준
  tab_sources         : 개선 출처 — 삽입 (BL6 − 최고 배치 규칙), 학습 (RL − BL6), 합계 (RL − 최고 배치 규칙)
  tab_variants        : 학습 변형 arm vs BL6 (+ 기본 arm과 Welch t), 라우팅 블록 (경로 규칙별 같은 네트워크 재평가)
  tab_fleet           : 로봇 수 일반화 (전이)
  tab_conditions      : 2³ 요인 설계 8셀 + 기준 행, _effects = 네트워크 단위 요인 효과
                        (y = log(F̄_RL,seed / F̄_ref), within = λ·M, between = K, 단위 = 네트워크), _obs = 관측
  tab_app_diagnostics : 소형 플릿 진단 (λ = 240, M ∈ SMALL_M + 기준 셀)
  fig_curves (+ _summary): 기본 네트워크 학습 곡선, 50 에피소드 창 요약
  tab_reward_sens (+ _summary): 보상 계수 민감도 vs BL6, vs 중심점 (같은 학습 시드), 계수별 F와 로그 탄력성
"""
import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict
from itertools import combinations
from typing import List, Optional, Sequence

import numpy as np

from env.rl import ARMS
from env.simulator import COST_DIST
from evaluation import POLICIES, display_name, write_csv
from experiments import run_plans as RP
from experiments.run_plans import Cell, Ckpt

_CELL_RE = re.compile(r"^lam(?P<lam>[\d.]+)_M(?P<M>[\d-]+)_K(?P<K>\d+)$")
_RUN_RE  = re.compile(r"^(?P<arm>.+?)_s(?P<seed>\d+)(?P<rest>.*)$")


def metric_value(rec: dict, metric: str) -> float:
    """eval.csv 행(dict)에서 지표 값. 파생 지표 포함

    J                 : avg_flow_time + COST_DIST · dist_per_order (주문당 흐름시간 + c_d · 주문당 이동거리, s)
    blocked_per_robot : blocked_time / M (s)
    decision_rate     : n_decisions / n_totes (큐에서 방출된 토트는 결정 아님)
    completion_rate   : n_completed / n_arrived
    그 밖: 같은 이름 열
    """
    f = lambda k: float(rec[k])
    if metric == "J":
        return f("avg_flow_time") + COST_DIST * f("dist_per_order")
    if metric == "blocked_per_robot":
        return f("blocked_time") / max(1.0, f("num_robots"))
    if metric == "decision_rate":
        return f("n_decisions") / max(1.0, f("n_totes"))
    if metric == "completion_rate":
        return f("n_completed") / max(1.0, f("n_arrived"))
    return f(metric)


def load_records(root: str) -> List[dict]:
    """root 아래 모든 셀의 eval.csv 행 (다중 M 학습 폴더 제외)

    출력 행: setting, lam, M, K, policy, run, train_seed, seed (테스트), rec (CSV 행 dict)
    같은 (셀, run, 테스트 시드)가 여러 번 있으면 마지막 것만
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "*", "*", "*", "eval.csv"))):
        run_dir  = os.path.dirname(path)
        cell_dir = os.path.dirname(run_dir)
        m = _CELL_RE.match(os.path.basename(cell_dir))
        if not m or "-" in m["M"]:
            continue
        setting, run_name = os.path.basename(os.path.dirname(cell_dir)), os.path.basename(run_dir)
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("seed") in (None, "", "-1"):
                    continue
                if r["label"] == "RL":
                    rm = _RUN_RE.match(run_name)
                    policy, ts, run = (rm["arm"] + rm["rest"], int(rm["seed"]), run_name) if rm else (run_name, 0, run_name)
                else:
                    policy, ts, run = r["label"], 0, r["label"]
                rows.append({"setting": setting, "lam": float(m["lam"]), "M": int(m["M"]), "K": int(m["K"]),
                             "policy": policy, "run": run, "train_seed": ts, "seed": int(r["seed"]), "rec": r})
    keep = {}
    for r in rows:
        keep[(r["setting"], r["lam"], r["M"], r["K"], r["run"], r["seed"])] = r
    return list(keep.values())


def with_metric(rows: List[dict], metric: str) -> List[dict]:
    """load_records 행에 y = metric 값을 붙인 복사본 (비교 함수 입력 형식)"""
    return [{**r, "y": metric_value(r["rec"], metric)} for r in rows]


# 대응 행렬 / 부트스트랩

def paired_matrix(rows: List[dict], ref_rows: List[dict]):
    """정책 그룹 행과 기준 행으로 차이 행렬 D (학습 시드 × 공통 테스트 시드)

    출력:
        (D (n_train, n_test), 학습 시드 리스트, 테스트 시드 리스트), 공통 시드가 없으면 None
    """
    ref = {r["seed"]: r["y"] for r in ref_rows}
    by_ts = defaultdict(dict)
    for r in rows:
        by_ts[r["train_seed"]][r["seed"]] = r["y"]
    common = set(ref)
    for v in by_ts.values():
        common &= set(v)
    if not common:
        return None
    ts, es = sorted(by_ts), sorted(common)
    D = np.array([[by_ts[t][e] - ref[e] for e in es] for t in ts], dtype=float)
    return D, ts, es


def hierarchical_bootstrap(D: np.ndarray, B: int = 10000, mode: str = "nested",
                           rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """차이 행렬 D (학습 시드 × 테스트 시드)의 평균에 대한 부트스트랩 분포 (B,)

    mode: "nested" (학습 시드를 뽑은 뒤 그 안에서 테스트 시드를 따로 복원추출) /
          "crossed" (두 축 독립 복원추출, 테스트 시드 공유)
    """
    rng = rng or np.random.default_rng(0)
    n_t, n_e = D.shape
    ti = rng.integers(0, n_t, size=(B, n_t))
    if mode == "crossed":
        ei = rng.integers(0, n_e, size=(B, n_e))
        return D[ti[:, :, None], ei[:, None, :]].mean(axis=(1, 2))
    if mode == "nested":
        ei = rng.integers(0, n_e, size=(B, n_t, n_e))
        return D[ti[:, :, None], ei].mean(axis=(1, 2))
    raise ValueError(mode)


def nested_ratio_bootstrap(Y: np.ndarray, R: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    """상대 차이 100 · (mean Y / mean R − 1)의 시드→인스턴스 계층 부트스트랩 분포

    Y: (n_train, n_test) 정책 값, R: (n_test,) 기준 값 (같은 테스트 인스턴스)
    학습 시드를 복원추출한 뒤 시드마다 테스트 인스턴스를 따로 복원추출 (기준도 같은 인스턴스로)
    """
    n_t, n_e = Y.shape
    ti = rng.integers(0, n_t, size=(B, n_t))
    ei = rng.integers(0, n_e, size=(B, n_t, n_e))
    num = Y[ti[:, :, None], ei].mean(axis=(1, 2))
    den = R[ei].mean(axis=(1, 2))
    return 100.0 * (num / den - 1.0)


def _t_interval(x: np.ndarray, alpha: float):
    """평균의 t 구간 (자유도 n − 1). n < 2이면 nan"""
    from scipy import stats as st
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    h = st.t.ppf(1 - alpha / 2, n - 1) * x.std(ddof=1) / math.sqrt(n)
    return float(x.mean() - h), float(x.mean() + h)


# 비교

def seed_level_compare(pol_rows: List[dict], ref_rows: List[dict], B: int = 10000,
                       rng: Optional[np.random.Generator] = None, alpha: float = 0.05) -> Optional[dict]:
    """학습 시드 단위 비교 (모듈 설명). pol_rows의 학습 시드별로 기준(ref_rows, 학습 시드 없음)과 대응"""
    from scipy import stats as st
    rng = rng or np.random.default_rng(0)
    pm = paired_matrix(pol_rows, ref_rows)
    if pm is None:
        return None
    D, ts, es = pm
    ref = {r["seed"]: r["y"] for r in ref_rows}
    R = np.array([ref[e] for e in es])
    Y = D + R[None, :]
    m_i = D.mean(axis=1)
    if len(ts) > 1:
        p, test = float(st.ttest_1samp(m_i, 0.0).pvalue), "seed_t"
    else:
        p, test = (float(st.ttest_1samp(D[0], 0.0).pvalue) if len(es) > 1 else float("nan")), "instance_t"
    boot = hierarchical_bootstrap(D, B, "nested", rng)
    rb   = nested_ratio_bootstrap(Y, R, B, rng)
    t_lo, t_hi = _t_interval(m_i, alpha)
    lr_lo, lr_hi = _t_interval(np.log(Y.mean(axis=1) / R.mean()), alpha)
    return {"n_train": len(ts), "n_test": len(es), "mean_pol": float(Y.mean()), "mean_ref": float(R.mean()),
            "diff": float(m_i.mean()), "ci_lo": float(np.quantile(boot, alpha / 2)),
            "ci_hi": float(np.quantile(boot, 1 - alpha / 2)), "ci_t_lo": t_lo, "ci_t_hi": t_hi,
            "p": p, "test": test, "n_better": int(np.sum(m_i < 0)), "seed_diffs": ";".join(f"{x:.3f}" for x in m_i),
            "rel_pct": float(100.0 * (Y.mean() / R.mean() - 1.0)),
            "rel_ci_lo": float(np.quantile(rb, alpha / 2)), "rel_ci_hi": float(np.quantile(rb, 1 - alpha / 2)),
            "rel_ci_t_lo": 100.0 * math.expm1(lr_lo) if not math.isnan(lr_lo) else float("nan"),
            "rel_ci_t_hi": 100.0 * math.expm1(lr_hi) if not math.isnan(lr_hi) else float("nan")}


def matched_compare(pol_rows: List[dict], base_rows: List[dict], B: int = 10000,
                    rng: Optional[np.random.Generator] = None, alpha: float = 0.05) -> Optional[dict]:
    """같은 (학습 시드, 테스트 인스턴스) 대응 차이 (예: 같은 네트워크의 fixed_dock − nnh, 계수 변형 − 중심점)"""
    from scipy import stats as st
    rng = rng or np.random.default_rng(0)
    b = {(r["train_seed"], r["seed"]): r["y"] for r in base_rows}
    by = defaultdict(dict)
    for r in pol_rows:
        if (r["train_seed"], r["seed"]) in b:
            by[r["train_seed"]][r["seed"]] = r["y"]
    if not by:
        return None
    ts = sorted(by)
    es = sorted(set.intersection(*[set(v) for v in by.values()]))
    P = np.array([[by[t][e] for e in es] for t in ts])
    Q = np.array([[b[(t, e)] for e in es] for t in ts])
    Dm = P - Q
    m_i = Dm.mean(axis=1)
    boot = hierarchical_bootstrap(Dm, B, "nested", rng)
    p = float(st.ttest_1samp(m_i, 0).pvalue) if len(ts) > 1 else float(st.ttest_1samp(Dm[0], 0).pvalue)
    ti = rng.integers(0, len(ts), size=(B, len(ts)))
    ei = rng.integers(0, len(es), size=(B, len(ts), len(es)))
    rb = 100.0 * (P[ti[:, :, None], ei].mean(axis=(1, 2)) / Q[ti[:, :, None], ei].mean(axis=(1, 2)) - 1)
    t_lo, t_hi = _t_interval(m_i, alpha)
    return {"n_train": len(ts), "n_test": len(es), "mean_pol": float(P.mean()), "mean_ref": float(Q.mean()),
            "diff": float(m_i.mean()), "ci_lo": float(np.quantile(boot, alpha / 2)),
            "ci_hi": float(np.quantile(boot, 1 - alpha / 2)), "ci_t_lo": t_lo, "ci_t_hi": t_hi, "p": p,
            "test": "matched_seed_t" if len(ts) > 1 else "matched_instance_t",
            "n_better": int(np.sum(m_i < 0)), "seed_diffs": ";".join(f"{x:.3f}" for x in m_i),
            "rel_pct": float(100 * (P.mean() / Q.mean() - 1)),
            "rel_ci_lo": float(np.quantile(rb, alpha / 2)), "rel_ci_hi": float(np.quantile(rb, 1 - alpha / 2))}


def rule_compare(pol_rows: List[dict], ref_rows: List[dict], alpha: float = 0.05) -> Optional[dict]:
    """규칙 기반 정책끼리 비교 (학습 시드 없음): 인스턴스 간 대응 t검정, t 기반 95% CI (상대 차이는 델타법)"""
    from scipy import stats as st
    pm = paired_matrix(pol_rows, ref_rows)
    if pm is None:
        return None
    D, _, es = pm
    d = D[0]
    ref = {r["seed"]: r["y"] for r in ref_rows}
    R = np.array([ref[e] for e in es])
    n = len(d)
    se = d.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    tc = st.t.ppf(1 - alpha / 2, max(1, n - 1))
    rel = 100.0 * d.mean() / R.mean()
    return {"n_train": 1, "n_test": n, "mean_pol": float((d + R).mean()), "mean_ref": float(R.mean()),
            "diff": float(d.mean()), "ci_lo": float(d.mean() - tc * se), "ci_hi": float(d.mean() + tc * se),
            "p": float(st.ttest_rel(d + R, R).pvalue) if n > 1 else float("nan"), "test": "instance_t",
            "n_better": int(d.mean() < 0), "seed_diffs": "",
            "rel_pct": float(rel), "rel_ci_lo": float(rel - 100.0 * tc * se / R.mean()),
            "rel_ci_hi": float(rel + 100.0 * tc * se / R.mean())}


def holm(pvals: Sequence[float]) -> List[float]:
    """Holm–Bonferroni 보정 p값 (입력 순서 유지, nan은 그대로)"""
    idx = [i for i, p in enumerate(pvals) if not math.isnan(p)]
    m   = len(idx)
    out = [float("nan")] * len(pvals)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda i: pvals[i])):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        out[i]  = running
    return out


def holm_table(rows: List[dict], key: str = "p", out_key: str = "p_holm") -> List[dict]:
    """표 하나의 모든 비교(행 × 지표)에 Holm 보정 (in-place)"""
    for r, ph in zip(rows, holm([r.get(key, float("nan")) for r in rows])):
        r[out_key] = ph
    return rows


# 요인 효과

def factorial_effects_by_unit(obs: List[dict], within: Sequence[str], between: Sequence[str],
                              unit: str = "unit", alpha: float = 0.05) -> List[dict]:
    """2^k 요인 설계 효과, 반복 단위 = unit (예: 네트워크)

    obs: {"y", unit, 요인: ±1, ...}. within 요인은 단위 안에서 모든 조합이 관측되고 (예: 같은 네트워크를 λ·M 셀로 전이),
         between 요인은 단위마다 한 값 (예: K마다 따로 학습한 네트워크)
    단계 1: 단위마다 within 요인 부분집합 S의 대비 c_S = mean(y · Π_{f∈S} f)  (S = ∅이면 단위 평균)
    단계 2: 대비마다 단위 간 최소제곱 c_S ~ 1 + between 요인 (±1 코딩) → 절편 = 항 S, 기울기 = 항 S × between
            자유도 = 단위 수 − (1 + between 수). 같은 단위의 셀을 독립 반복으로 세지 않음
    출력 행: term, coef (±1 코딩 계수, 효과 = 2 · coef), se, ci_lo, ci_hi, p, df, n_units
    """
    from scipy import stats as st
    by = defaultdict(list)
    for o in obs:
        by[o[unit]].append(o)
    units = sorted(by)
    n_cells = 2 ** len(within)
    for u in units:
        if len({tuple(o[f] for f in within) for o in by[u]}) != n_cells:
            raise ValueError(f"단위 {u}에 within 요인 조합 {n_cells}개가 모두 있어야 합니다")
        if len({tuple(o[f] for f in between) for o in by[u]}) != 1:
            raise ValueError(f"단위 {u}의 between 요인 값이 셀마다 다릅니다")
    X = np.array([[1.0] + [float(by[u][0][f]) for f in between] for u in units])
    names_b = [""] + list(between)
    df = len(units) - X.shape[1]
    tc = st.t.ppf(1 - alpha / 2, df) if df > 0 else float("nan")
    XtX_inv = np.linalg.pinv(X.T @ X)
    out = []
    for k in range(len(within) + 1):
        for S in combinations(within, k):
            c = np.array([np.mean([o["y"] * np.prod([o[f] for f in S]) for o in by[u]]) for u in units])
            beta, *_ = np.linalg.lstsq(X, c, rcond=None)
            resid = c - X @ beta
            s2 = float(resid @ resid / df) if df > 0 else float("nan")
            for j, b in enumerate(names_b):
                term = ":".join([*S, b] if b else S) or "Intercept"
                se = math.sqrt(s2 * XtX_inv[j, j]) if df > 0 else float("nan")
                tval = beta[j] / se if se > 0 else float("nan")
                out.append({"term": term, "coef": float(beta[j]), "se": se,
                            "ci_lo": float(beta[j] - tc * se), "ci_hi": float(beta[j] + tc * se),
                            "p": float(2 * st.t.sf(abs(tval), df)) if df > 0 and not math.isnan(tval) else float("nan"),
                            "df": df, "n_units": len(units)})
    return out


# 논문 표

BL6 = "BL6_MarginalCost"
BASELINE_LABELS = [label for label, _ in POLICIES.values()]
DIAG = ["avg_flow_time", "J", "avg_flow_time_censored", "completion_rate", "avg_unfinished_age",
        "decision_rate", "blocked_per_robot", "mid_trip_rate", "dist_per_order", "avg_totes_per_trip",
        "avg_wait_time", "n_qpen_active", "n_depletion_active"]
METRICS = ("avg_flow_time", "J")


class Data:
    """결과 행 색인: (setting, λ, M, K, policy) -> 행"""

    def __init__(self, root: str, pool: str = "old"):
        self.root, self.pool = root, pool
        self.idx = defaultdict(list)
        for r in load_records(root):
            self.idx[(r["setting"], r["lam"], r["M"], r["K"], r["policy"])].append(r)

    @staticmethod
    def key(cell: Cell):
        return cell.setting, float(cell.lam), cell.M, cell.K

    def get(self, cell: Cell, policy: str, metric: str):
        return with_metric(self.idx.get((*self.key(cell), policy), []), metric)

    def policies(self, cell: Cell):
        return sorted({k[4] for k in self.idx if k[:4] == self.key(cell)})

    def select(self, cell: Cell) -> dict:
        p = os.path.join(cell.dir(self.root), "select.json")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return {"best_label": BL6}

    def ref(self, cell: Cell) -> str:
        """셀 기준: pool = "old"이면 BL4–BL7 중 검증 최고, "new"이면 BL4–BL8 + BL6* (run_stages stage0의 best_label_new)"""
        sel = self.select(cell)
        return sel.get("best_label_new", sel["best_label"]) if self.pool == "new" else sel["best_label"]

    def mean(self, cell: Cell, policy: str, metric: str) -> float:
        vals = [r["y"] for r in self.get(cell, policy, metric) if not math.isnan(r["y"])]
        return float(np.mean(vals)) if vals else float("nan")


def compare(Dt: Data, cell: Cell, pol: str, ref: str, metric: str, B, rng, ref_cell: Cell = None):
    """정책 그룹 vs 기준 (RL이면 학습 시드 단위, 규칙이면 인스턴스 대응)"""
    pr = Dt.get(cell, pol, metric)
    rr = Dt.get(ref_cell or cell, ref, metric)
    if not pr or not rr:
        return None
    is_rl = any(r["rec"]["label"] == "RL" for r in pr)
    return seed_level_compare(pr, rr, B, rng) if is_rl else rule_compare(pr, rr)


def _row(table, cell: Cell, pol, ref, metric, res, Dt: Data, extra=None):
    out = {"table": table, "setting": cell.setting, "lam": float(cell.lam), "M": cell.M, "K": cell.K,
           "policy": pol, "policy_name": display_name(pol), "ref": ref, "ref_name": display_name(ref),
           "metric": metric, **(res or {})}
    for d in DIAG:
        out[f"mean_{d}"] = Dt.mean(cell, pol, d)
    out.update(extra or {})
    return out


def _vs_ref(table, Dt, cell, pols, B, rng, extra=None, ref=None):
    """pols 각각 vs 셀 기준 (F, J) 행 목록"""
    ref = ref or Dt.ref(cell)
    rows = []
    for pol in pols:
        if pol == ref:
            continue
        for m in METRICS:
            res = compare(Dt, cell, pol, ref, m, B, rng)
            if res:
                rows.append(_row(table, cell, pol, ref, m, res, Dt, extra))
    return rows


def _seed_means(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["train_seed"]].append(r["y"])
    return [float(np.mean(v)) for _, v in sorted(by.items())]


# 표

def tab_main(Dt, B, rng, arm):
    rows = []
    for cell in RP.MAIN_CELLS:
        ref = Dt.ref(cell)
        rows += _vs_ref("tab_main", Dt, cell, [arm] + BASELINE_LABELS, B, rng)
        rows.append(_row("tab_main", cell, ref, ref, "avg_flow_time", None, Dt))   # 기준 행 (평균만)
    return holm_table(rows)


def tab_sources(Dt, B, rng, arm):
    rows = []
    for cell in RP.MAIN_CELLS:
        batch = Dt.select(cell).get("best_batching")
        pairs = [("learning", arm, BL6)]
        if batch:
            pairs = [("insertion", BL6, batch)] + pairs + [("total", arm, batch)]
        else:
            print(f"tab_sources {cell}: select.json에 best_batching 없음, 학습 효과만")
        for src, pol, ref in pairs:
            for m in METRICS:
                res = compare(Dt, cell, pol, ref, m, B, rng)
                if res:
                    rows.append(_row("tab_sources", cell, pol, ref, m, res, Dt, {"source": src}))
    return holm_table(rows)


def _train_rates(root, cell: Cell, arm):
    """학습 train.csv에서 큐 압박/소진 페널티 조건 비율 (결정당, 에피소드 평균), 시드 평균"""
    out = defaultdict(list)
    for p in glob.glob(os.path.join(cell.dir(root), f"{arm}_s*", "train.csv")):
        rs = list(csv.DictReader(open(p, newline="")))
        nd = [max(1.0, float(r["n_decisions"])) for r in rs]
        out["train_qpen_rate"].append(np.mean([float(r["n_qpen_active"]) / n for r, n in zip(rs, nd)]))
        out["train_depletion_rate"].append(np.mean([float(r["n_depletion_active"]) / n for r, n in zip(rs, nd)]))
        out["train_decision_rate"].append(np.mean([float(r["n_decisions"]) / max(1.0, float(r["n_totes"]))
                                                   for r in rs]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def tab_variants(Dt, B, rng, arm):
    from scipy import stats as st
    cell = RP.BASE_CELL
    rows = []
    for a in [arm] + RP.VARIANT_ARMS:
        extra = {"block": "training", **_train_rates(Dt.root, cell, a)}
        for m in METRICS:
            res = compare(Dt, cell, a, BL6, m, B, rng)
            if res is None:
                continue
            if a != arm:      # 기본 arm과의 차이: 시드 평균끼리 Welch t (학습 시드가 독립)
                x, y = _seed_means(Dt.get(cell, a, m)), _seed_means(Dt.get(cell, arm, m))
                if len(x) > 1 and len(y) > 1:
                    extra = {**extra, "vs_base_diff": float(np.mean(x) - np.mean(y)),
                             "vs_base_p_welch": float(st.ttest_ind(x, y, equal_var=False).pvalue)}
            rows.append(_row("tab_variants", cell, a, BL6, m, res, Dt, extra))
    # 라우팅 블록 (가중치 고정): 변형 RL vs 변형 BL6, 변형 RL vs nnh RL (같은 네트워크), 변형 BL6 vs nnh BL6
    for rc in RP.ROUTE_CELLS:
        block = f"routing_{rc.route}"
        pol = Ckpt(cell, arm, 0).policy(rc)
        for m in METRICS:
            res = compare(Dt, rc, pol, BL6, m, B, rng)
            if res:
                rows.append(_row("tab_variants", rc, pol, BL6, m, res, Dt, {"block": block}))
            res = matched_compare(Dt.get(rc, pol, m), Dt.get(cell, arm, m), B, rng)
            if res:
                rows.append(_row("tab_variants", rc, pol, f"{arm} (nnh)", m, res, Dt, {"block": block}))
            res = compare(Dt, rc, BL6, BL6, m, B, rng, ref_cell=cell)
            if res:
                rows.append(_row("tab_variants", rc, BL6, f"{BL6} (nnh)", m, res, Dt, {"block": block}))
    return holm_table(rows)


def tab_fleet(Dt, B, rng, arm):
    rows = []
    for cell in sorted([RP.BASE_CELL] + RP.FLEET_CELLS, key=lambda c: c.M):
        pol = Ckpt(RP.BASE_CELL, arm, 0).policy(cell)
        rows += _vs_ref("tab_fleet", Dt, cell, [pol], B, rng)
    return holm_table(rows)


def tab_conditions(Dt, B, rng, arm):
    """요인 설계 8셀 + 기준 행, 네트워크 단위 요인 효과 (반환: 표 행, 효과 행, 관측)"""
    rows, obs = [], []
    for cell in RP.FACTORIAL_CELLS:
        pol = RP.factorial_nets(cell.K, arm)[0].policy(cell)
        rows += _vs_ref("tab_conditions", Dt, cell, [pol], B, rng, {"kind": "design"})
        # 요인 효과 관측: 학습 시드(네트워크)별 log(F̄_RL / F̄_ref), 공통 인스턴스
        pr, rr = Dt.get(cell, pol, "avg_flow_time"), Dt.get(cell, Dt.ref(cell), "avg_flow_time")
        pm = paired_matrix(pr, rr) if pr and rr else None
        if pm is not None:
            Dm, ts, es = pm
            ref_by = {r["seed"]: r["y"] for r in rr}
            Rm = np.mean([ref_by[e] for e in es])
            for i, t in enumerate(ts):
                obs.append({"y": math.log((Dm[i].mean() + Rm) / Rm), "lam": 1 if cell.lam == 320 else -1,
                            "M": 1 if cell.M == 18 else -1, "K": 1 if cell.K == 4 else -1,
                            "cell": f"{cell.lam:.0f}-{cell.M}-{cell.K}", "train_seed": t,
                            "unit": f"K{cell.K}_s{t}"})
    refs = [(RP.BASE_CELL, RP.factorial_ref_nets(arm)[0].policy(RP.BASE_CELL), "ref_lam320_net"),
            (Cell(320, 14, 3), arm, "ref_lam320_net"), (RP.BASE_CELL, arm, "ref_lam240_net")]
    for cell, pol, kind in refs:
        rows += _vs_ref("tab_conditions", Dt, cell, [pol], B, rng, {"kind": kind})
    eff = []
    try:
        eff = [{"table": "tab_conditions_effects", **e}
               for e in factorial_effects_by_unit(obs, within=["lam", "M"], between=["K"])] if obs else []
    except ValueError as e:
        print(f"tab_conditions: 요인 효과 생략 ({e})")
    return holm_table(rows), eff, obs


def tab_app_diagnostics(Dt, B, rng, arm):
    rows = []
    for cell in sorted(RP.SMALL_CELLS + [RP.BASE_CELL], key=lambda c: c.M):
        ref = Dt.ref(cell)
        pol = Ckpt(RP.BASE_CELL, arm, 0).policy(cell)
        for p in [pol] + BASELINE_LABELS:
            if not Dt.get(cell, p, "avg_flow_time"):
                continue
            res = compare(Dt, cell, p, ref, "avg_flow_time", B, rng) if p != ref else None
            rows.append(_row("tab_app_diagnostics", cell, p, ref, "avg_flow_time", res, Dt))
    return holm_table(rows)


def fig_curves(Dt, arm):
    """기본 네트워크 학습 곡선 (보상, loss, |TD|, Q, α, λ, flow, 국소 비용) + 검증 flow, 50 에피소드 창 요약"""
    long, val = [], defaultdict(dict)
    f = lambda r, k: float(r.get(k, "nan") or "nan")
    for ck in RP.base_nets(arm):
        run = ck.dir(Dt.root)
        cp = os.path.join(run, "curve.csv")
        if not os.path.isfile(cp):
            continue
        vp = os.path.join(run, "val.csv")
        if os.path.isfile(vp):
            by = defaultdict(list)
            for r in csv.DictReader(open(vp, newline="")):
                by[int(r["episode"])].append(float(r["avg_flow_time"]))
            val[ck.seed] = {e: float(np.mean(v)) for e, v in by.items()}
        for r in csv.DictReader(open(cp, newline="")):
            ep = int(r["episode"])
            long.append({"train_seed": ck.seed, "episode": ep,
                         **{k: f(r, k) for k in ("lam", "reward", "loss", "td_abs", "q_mean", "alpha",
                                                 "avg_flow_time", "local_mu", "local_cost")},
                         "val_flow": val.get(ck.seed, {}).get(ep, float("nan"))})
    summ = []
    keys = ["lam", "reward", "loss", "td_abs", "q_mean", "alpha", "avg_flow_time", "local_cost"]
    for w0 in range(1, max([r["episode"] for r in long] or [0]) + 1, 50):
        win = [r for r in long if w0 <= r["episode"] < w0 + 50]
        if not win:
            continue
        seeds = sorted({r["train_seed"] for r in win})
        row = {"ep_from": w0, "ep_to": w0 + 49, "n_seeds": len(seeds)}
        for k in keys:
            v = [r[k] for r in win if not math.isnan(r[k])]
            row[f"{k}_mean"] = float(np.mean(v)) if v else float("nan")
            sm = [np.mean([r[k] for r in win if r["train_seed"] == s and not math.isnan(r[k])] or [np.nan])
                  for s in seeds]
            row[f"{k}_sd_seeds"] = float(np.nanstd(sm, ddof=1)) if len(sm) > 1 else float("nan")
        vf = [v[e] for v in val.values() for e in v if w0 <= e < w0 + 50]
        row["val_flow_mean"] = float(np.mean(vf)) if vf else float("nan")
        summ.append(row)
    return long, summ


def _prior_terms(root, cell: Cell):
    """BL6 prior 정책의 결정당 비용 항 평균 (μ 추정 파일의 목표 λ 격자점, 계수와 무관). 없으면 NaN"""
    for path in sorted(glob.glob(os.path.join(cell.dir(root), "local_mu*.json"))):
        tab = json.load(open(path, encoding="utf-8"))
        if tab.get("terms"):
            return dict(zip(("prior_own", "prior_ext", "prior_queue"), tab["terms"][-1]))
    return {"prior_own": float("nan"), "prior_ext": float("nan"), "prior_queue": float("nan")}


def _train_terms(root, cell: Cell, arm, seeds, last=500):
    """학습 후반 last 에피소드 curve.csv의 결정당 비용 항 평균 (시드 평균, 계수 곱 전)"""
    vals = defaultdict(list)
    for s in seeds:
        p = os.path.join(Ckpt(cell, arm, s).dir(root), "curve.csv")
        if not os.path.isfile(p):
            continue
        rs = list(csv.DictReader(open(p, newline="")))[-last:]
        for k in ("cost_own", "cost_ext", "cost_queue", "local_cost"):
            if rs and k in rs[0]:
                vals[k].append(np.mean([float(r[k]) for r in rs]))
    return {f"train_{k}": float(np.mean(v)) for k, v in vals.items()}


def tab_reward_sens(Dt, B, rng, arm, seeds=tuple(RP.SUB_SEEDS)):
    """보상 계수 민감도 표와 요약 (중심점 = 기본 arm의 같은 학습 시드)"""
    rows, summ = [], []
    sel = lambda rs: [r for r in rs if r["train_seed"] in seeds]
    for lam in sorted({lam for lam, _ in RP.SENS_DESIGN}):
        cell = Cell(lam, 14, 3)
        arms = []
        for pol in Dt.policies(cell):
            m = re.fullmatch(re.escape(arm) + r"_(wext|wq)([\d.]+)", pol)
            if m:
                arms.append((pol, m[1], float(m[2])))
        if not arms:
            continue
        prior = _prior_terms(Dt.root, cell)
        F = {}
        for a, term, w in [(arm, None, 1.0)] + sorted(arms, key=lambda x: (x[1], x[2])):
            info = {"term": term or "center", "w_ext": w if term == "wext" else 1.0,
                    "w_q": w if term == "wq" else 1.0, **prior, **_train_terms(Dt.root, cell, a, seeds)}
            for m in METRICS:
                pr = sel(Dt.get(cell, a, m))
                if not pr:
                    continue
                diag = {}
                for d in DIAG:
                    v = [r["y"] for r in sel(Dt.get(cell, a, d)) if not math.isnan(r["y"])]
                    diag[f"mean_{d}"] = float(np.mean(v)) if v else float("nan")
                if m == "avg_flow_time":
                    F[(term, w)] = diag["mean_avg_flow_time"]
                base = {"table": "tab_reward_sens", "setting": cell.setting, "lam": float(lam), "M": 14, "K": 3,
                        "policy": a, "policy_name": display_name(a), "metric": m, **info, **diag}
                bl6 = Dt.get(cell, BL6, m)
                if bl6:
                    rows.append({**base, "ref": BL6, **(seed_level_compare(pr, bl6, B, rng) or {})})
                if term is not None:
                    res = matched_compare(pr, sel(Dt.get(cell, arm, m)), B, rng)
                    if res:
                        rows.append({**base, "ref": f"{arm} (w = 1, 1)", **res})
        for term in ("wext", "wq"):
            f = {w: F.get((term, w), F.get((None, 1.0), float("nan")) if w == 1.0 else float("nan"))
                 for w in (0.0, 0.5, 1.0, 2.0)}
            ok = all(not math.isnan(f[w]) for w in (0.5, 1.0, 2.0))
            summ.append({"lam": float(lam), "term": term, "F_w0": f[0.0], "F_w0.5": f[0.5], "F_w1": f[1.0],
                         "F_w2": f[2.0], "log_elasticity": (f[2.0] - f[0.5]) / f[1.0] / math.log(4) if ok else float("nan"),
                         "n_train_seeds": len(seeds), **prior})
    return holm_table(rows), summ


SIMPLE = {"tab_main": tab_main, "tab_sources": tab_sources, "tab_variants": tab_variants,
          "tab_fleet": tab_fleet, "tab_app_diagnostics": tab_app_diagnostics}
ALL_TABLES = list(SIMPLE) + ["tab_conditions", "fig_curves", "tab_reward_sens"]


def _print_rows(rows):
    for r in rows:
        if r.get("metric") == "avg_flow_time" and "diff" in r:
            print(f"  λ={r['lam']:.0f} M={r['M']} K={r['K']} {r['policy_name']:30s} vs {r['ref_name']:16s} "
                  f"ΔF {r['diff']:+7.2f} s [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] ({r['rel_pct']:+.1f}%)  "
                  f"p={r['p']:.4f} p_holm={r['p_holm']:.4f}  나은 시드 {r['n_better']}/{r['n_train']}")


def main():
    p = argparse.ArgumentParser(description="논문 표 CSV")
    p.add_argument("--out", default=os.path.join(RP.REPO, "runs", "exp"), help="실험 폴더 (runs/<실험 이름>)")
    p.add_argument("--tables", nargs="*", choices=ALL_TABLES, default=None, help="만들 표 (기본 전부)")
    p.add_argument("--arm", choices=list(ARMS), default=RP.BASE_ARM, help="RL 행으로 쓸 arm")
    p.add_argument("--pool", choices=["old", "new"], default="old", help="기준 선택 풀 (모듈 설명)")
    p.add_argument("--B", type=int, default=10000, help="부트스트랩 반복 수")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    Dt  = Data(os.path.abspath(args.out), args.pool)
    rng = np.random.default_rng(args.seed)
    tdir = os.path.join(RP.results_dir(Dt.root), "tables")
    want = args.tables or ALL_TABLES
    for name, fn in SIMPLE.items():
        if name in want:
            rows = fn(Dt, args.B, rng, args.arm)
            write_csv(rows, os.path.join(tdir, f"{name}.csv"))
            print(f"{name}: {len(rows)}행")
            _print_rows(rows)
    if "tab_conditions" in want:
        rows, eff, obs = tab_conditions(Dt, args.B, rng, args.arm)
        write_csv(rows, os.path.join(tdir, "tab_conditions.csv"))
        write_csv(eff, os.path.join(tdir, "tab_conditions_effects.csv"))
        write_csv(obs, os.path.join(tdir, "tab_conditions_obs.csv"))
        print(f"tab_conditions: {len(rows)}행, 요인 효과 {len(eff)}항")
        _print_rows(rows)
        for e in eff:
            print(f"  {e['term']:10s} {e['coef']:+.4f} [{e['ci_lo']:+.4f}, {e['ci_hi']:+.4f}] p={e['p']:.4f} "
                  f"(df {e['df']}, 네트워크 {e['n_units']})")
    if "tab_reward_sens" in want:
        rows, summ = tab_reward_sens(Dt, args.B, rng, args.arm)
        write_csv(rows, os.path.join(tdir, "tab_reward_sens.csv"))
        write_csv(summ, os.path.join(tdir, "tab_reward_sens_summary.csv"))
        print(f"tab_reward_sens: {len(rows)}행")
        for r in summ:
            print(f"  λ={r['lam']:.0f} {r['term']:4s} F(0, ½, 1, 2) = {r['F_w0']:.1f}, {r['F_w0.5']:.1f}, "
                  f"{r['F_w1']:.1f}, {r['F_w2']:.1f}  탄력성 {r['log_elasticity']:+.4f}")
    if "fig_curves" in want:
        long, summ = fig_curves(Dt, args.arm)
        write_csv(long, os.path.join(tdir, "fig_curves.csv"))
        write_csv(summ, os.path.join(tdir, "fig_curves_summary.csv"))
        print(f"fig_curves: {len(long)}행, 요약 {len(summ)}창")
    print(f"저장: {tdir}")


if __name__ == "__main__":
    main()
