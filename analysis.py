"""평가 결과 통계 (experiments/tables.py가 사용)

입력: run_rl.py / experiments/runner.py가 남긴 {root}/{setting}/lam{λ}_M{M}_K{K}/{run}/eval.csv와 baselines/eval.csv
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
"""
import csv
import glob
import math
import os
import re
from collections import defaultdict
from itertools import combinations
from typing import List, Optional, Sequence

import numpy as np

_CELL_RE = re.compile(r"^lam(?P<lam>[\d.]+)_M(?P<M>[\d-]+)_K(?P<K>\d+)$")
_RUN_RE  = re.compile(r"^(?P<arm>.+?)_s(?P<seed>\d+)(?P<rest>.*)$")


# 데이터 읽기

def metric_value(rec: dict, metric: str) -> float:
    """eval.csv 행(dict)에서 지표 값. 파생 지표 포함

    J                 : avg_flow_time + COST_DIST · dist_per_order (주문당 흐름시간 + c_d · 주문당 이동거리, s)
    blocked_per_robot : blocked_time / M (s)
    decision_rate     : n_decisions / n_totes (큐에서 방출된 토트는 결정 아님)
    completion_rate   : n_completed / n_arrived
    그 밖: 같은 이름 열
    """
    from constant import COST_DIST
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


# 출력

def save_csv(rows: List[dict], path: str):
    """dict 행 목록을 CSV로 (행마다 열이 달라도 합집합, 첫 행 순서 우선). 행이 없으면 저장 안 함"""
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
