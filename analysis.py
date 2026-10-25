"""평가 결과 통계: 계층(학습 시드 × 테스트 시드) 부트스트랩, Holm 보정, 혼합효과 모형

입력은 run_rl.py가 남긴 {root}/{setting}/lam{λ}_M{M}_K{K}/{run}/eval.csv (+ run.json)와 baselines/eval.csv
  - 정책 그룹: 베이스라인은 라벨, RL은 run 이름에서 학습 시드(_s{seed})를 뺀 것
    (예: smdp_r8_bl6_s7_tr10-14-18_xfer -> smdp_r8_bl6_tr10-14-18_xfer, 학습 시드 7)
  - 비교 셀: (setting, λ, M, K). 셀마다 정책 그룹 vs 기준(ref, 기본 BL6)의 테스트 시드 대응 차이

부트스트랩 (셀 × 그룹마다)
  D[i, j] = y(그룹, 학습 시드 i, 테스트 시드 j) − y(ref, 테스트 시드 j)
  - crossed (기본): 학습 시드와 테스트 시드를 각각 복원추출해 부분행렬 평균. 모든 학습 시드가 같은
    테스트 시드를 쓰는 교차 설계이므로 테스트 시드 변동을 학습 시드 사이에 공유
  - nested: 학습 시드를 뽑은 뒤 그 안에서 테스트 시드를 따로 복원추출 (전형적 계층 부트스트랩)
  p값 = 부트스트랩 분포가 0의 반대쪽에 놓인 비율 × 2, Holm으로 전체 비교에 대해 보정

리비전 (experiments/revision_analyze.py가 사용)
  load_records / metric_value: eval.csv 행 전체와 파생 지표 (J, blocked_per_robot, decision_rate, completion_rate)
  seed_level_compare: 학습 시드 단위 (시드별 평균 차이, nested 부트스트랩 CI, 시드 간 대응 t, 나은 시드 수, ΔF%)
  rule_compare: 규칙 기반끼리 인스턴스 간 대응 t, holm_table: 표 단위 Holm
  factorial_effects: 2^k 요인 설계 주효과·2요인 상호작용 (±1 코딩 최소제곱)

혼합효과 모형 (statsmodels MixedLM, --mixed)
  y ~ C(policy, Treatment(ref)) + 분산성분 (1|테스트 시드) + (1|run)   (셀마다 적합, --mixed-by로 묶음 변경)
  run = 정책 그룹 × 학습 시드 (베이스라인은 run 1개). 학습 시드가 2개 이상인 그룹이 없으면 run 성분 생략
  고정효과 대비의 p값도 Holm 보정

사용법:
    python analysis.py --root logs_rl                       # 흐름시간, BL6 기준, crossed 부트스트랩
    python analysis.py --root logs_rl --relative --mixed --out logs_rl/stats.csv
    python analysis.py --root logs_rl --mixed --mixed-by setting lam K \\
        --mixed-formula "y ~ C(policy, Treatment('BL6_MarginalCost')) * C(M)"
"""
import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

REF_DEFAULT = "BL6_MarginalCost"
_CELL_RE    = re.compile(r"^lam(?P<lam>[\d.]+)_M(?P<M>[\d-]+)_K(?P<K>\d+)$")
_RUN_RE     = re.compile(r"^(?P<arm>.+?)_s(?P<seed>\d+)(?P<rest>.*)$")


# 데이터 읽기

def load_results(root: str, metric: str = "avg_flow_time") -> List[dict]:
    """root 아래 모든 eval.csv를 행 단위 dict 리스트로 (긴 형식)

    출력 행: setting, lam, M, K, policy (그룹), run, train_seed, seed (테스트), y, bl6_agree
    seed 열이 없는 옛 CSV는 건너뜀 (테스트 시드 대응이 불확실)
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "*", "*", "*", "eval.csv"))):
        run_dir  = os.path.dirname(path)
        cell_dir = os.path.dirname(run_dir)
        m = _CELL_RE.match(os.path.basename(cell_dir))
        if not m or "-" in m["M"]:                   # 다중 M 학습 폴더에는 eval.csv가 없음
            continue
        setting  = os.path.basename(os.path.dirname(cell_dir))
        run_name = os.path.basename(run_dir)
        info = {}
        if os.path.isfile(os.path.join(run_dir, "run.json")):
            with open(os.path.join(run_dir, "run.json"), encoding="utf-8") as f:
                info = json.load(f)
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("seed") in (None, "", "-1"):
                    continue
                if r["label"] == "RL":
                    rm = _RUN_RE.match(run_name)
                    policy = rm["arm"] + rm["rest"] if rm else run_name
                    train_seed = info.get("train_seed", int(rm["seed"]) if rm else 0)
                    run = run_name
                else:
                    policy, train_seed, run = r["label"], 0, r["label"]
                agree = r.get("bl6_agree", "")
                rows.append({
                    "setting": setting, "lam": float(m["lam"]), "M": int(m["M"]), "K": int(m["K"]),
                    "policy": policy, "run": run, "train_seed": int(train_seed), "seed": int(r["seed"]),
                    "y": float(r[metric]),
                    "bl6_agree": float(agree) if agree not in ("", "nan") else float("nan"),
                })
    return _dedup(rows)


def _dedup(rows: List[dict]) -> List[dict]:
    """같은 (셀, run, 테스트 시드)가 여러 번 있으면 마지막 것만 (베이스라인 중복 실행 등)"""
    keep = {}
    for r in rows:
        keep[(r["setting"], r["lam"], r["M"], r["K"], r["run"], r["seed"])] = r
    return list(keep.values())


# 부트스트랩 / 다중비교

def paired_matrix(rows: List[dict], ref_rows: List[dict], relative: bool = False):
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
    if relative:
        D = 100.0 * D / np.array([ref[e] for e in es])[None, :]
    return D, ts, es


def hierarchical_bootstrap(D: np.ndarray, B: int = 10000, mode: str = "crossed",
                           rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """차이 행렬 D (학습 시드 × 테스트 시드)의 평균에 대한 부트스트랩 분포

    입력:
        D: (n_train, n_test) 차이 행렬 (베이스라인 비교면 n_train = 1)
        B: 부트스트랩 반복 수
        mode: "crossed" (두 축 독립 복원추출, 테스트 시드 공유) / "nested" (학습 시드마다 테스트 시드 따로)
        rng: numpy Generator
    출력:
        (B,) 부트스트랩 평균
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


def bootstrap_p(boot: np.ndarray) -> float:
    """양측 부트스트랩 p값: 2 × min(P(θ* ≤ 0), P(θ* ≥ 0)) (+1 보정)"""
    B  = len(boot)
    lo = (np.sum(boot <= 0) + 1) / (B + 1)
    hi = (np.sum(boot >= 0) + 1) / (B + 1)
    return float(min(1.0, 2 * min(lo, hi)))


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


def compare(rows: List[dict], ref: str = REF_DEFAULT, B: int = 10000, mode: str = "crossed",
            relative: bool = False, alpha: float = 0.05, seed: int = 0) -> List[dict]:
    """셀 × 정책 그룹마다 ref 대비 차이의 부트스트랩 CI/p값, 전체 비교에 Holm 보정

    출력 행: setting, lam, M, K, policy, n_train, n_test, mean_y, diff, ci_lo, ci_hi, p, p_holm, bl6_agree
    """
    rng   = np.random.default_rng(seed)
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cells[(r["setting"], r["lam"], r["M"], r["K"])][r["policy"]].append(r)
    out = []
    for cell in sorted(cells):
        groups = cells[cell]
        if ref not in groups:
            continue
        for pol in sorted(groups):
            if pol == ref:
                continue
            pm = paired_matrix(groups[pol], groups[ref], relative)
            if pm is None:
                continue
            D, ts, es = pm
            boot = hierarchical_bootstrap(D, B, mode, rng)
            agree = [r["bl6_agree"] for r in groups[pol] if not math.isnan(r["bl6_agree"])]
            out.append({
                "setting": cell[0], "lam": cell[1], "M": cell[2], "K": cell[3], "policy": pol,
                "n_train": len(ts), "n_test": len(es),
                "mean_y": float(np.mean([r["y"] for r in groups[pol]])),
                "diff": float(D.mean()),
                "ci_lo": float(np.quantile(boot, alpha / 2)), "ci_hi": float(np.quantile(boot, 1 - alpha / 2)),
                "p": bootstrap_p(boot),
                "bl6_agree": float(np.mean(agree)) if agree else float("nan"),
            })
    for r, ph in zip(out, holm([r["p"] for r in out])):
        r["p_holm"] = ph
    return out


# 검정력 분석

def variance_components(D: np.ndarray) -> Dict[str, float]:
    """차이 행렬 D (학습 시드 × 테스트 시드)의 2원 분산분석(반복 없음) 분산성분

    D[i, j] = μ + a_i (학습 시드) + b_j (테스트 시드) + e_ij
    출력: mean, sd_train_means (학습 시드별 평균의 표준편차), var_train (σ_a²), var_test (σ_b²), var_resid (σ_e²)
    음수 추정치는 0으로 자름
    """
    n_t, n_e = D.shape
    mu  = D.mean()
    rm  = D.mean(axis=1)
    cm  = D.mean(axis=0)
    out = {"mean": float(mu), "n_train": n_t, "n_test": n_e,
           "sd_train_means": float(rm.std(ddof=1)) if n_t > 1 else float("nan")}
    if n_t < 2 or n_e < 2:
        return {**out, "var_train": float("nan"), "var_test": float("nan"), "var_resid": float("nan")}
    resid  = D - rm[:, None] - cm[None, :] + mu
    ms_row = n_e * np.sum((rm - mu) ** 2) / (n_t - 1)
    ms_col = n_t * np.sum((cm - mu) ** 2) / (n_e - 1)
    ms_res = np.sum(resid ** 2) / ((n_t - 1) * (n_e - 1))
    return {**out, "var_train": float(max(0.0, (ms_row - ms_res) / n_e)),
            "var_test": float(max(0.0, (ms_col - ms_res) / n_t)), "var_resid": float(ms_res)}


def power_curve(vc: Dict[str, float], deltas: Sequence[float], n_train: Sequence[int], n_test: int,
                alpha: float = 0.05, n_tests: int = 1, include_test_var: bool = True) -> List[dict]:
    """학습 시드 수별 양측 t검정 검정력 (학습 시드를 실험 단위로, 자유도 n_train − 1)

    SE² = σ_a²/n_t + σ_e²/(n_t·n_e) [+ σ_b²/n_e: 테스트 시드 집합도 표본으로 볼 때]
    n_tests > 1이면 유의수준 α / n_tests (Holm의 가장 보수적인 첫 단계와 같음)
    출력 행: delta, n_train, se, power
    """
    from scipy import stats as st
    a   = alpha / max(1, n_tests)
    out = []
    for d in deltas:
        for n in n_train:
            var = vc["var_train"] / n + vc["var_resid"] / (n * n_test)
            if include_test_var:
                var += vc["var_test"] / n_test
            se   = math.sqrt(var)
            df   = max(1, n - 1)
            tc   = st.t.ppf(1 - a / 2, df)
            ncp  = abs(d) / se if se > 0 else float("inf")
            pw   = 1.0 if math.isinf(ncp) else float(1 - st.nct.cdf(tc, df, ncp) + st.nct.cdf(-tc, df, ncp))
            out.append({"delta": float(d), "n_train": n, "se": se, "power": pw})
    return out


def power_report(rows: List[dict], policy: str, ref: str, rel_deltas: Sequence[float] = (2, 5, 10),
                 n_train: Sequence[int] = range(3, 11), n_test: Optional[int] = None,
                 alpha: float = 0.05, n_tests: int = 1, target: float = 0.8) -> dict:
    """셀마다 policy − ref 차이 행렬로 분산성분을 추정하고 학습 시드 수별 검정력 계산

    검출할 효과 크기 δ: 관측된 평균 차이 |diff|와 ref 평균의 rel_deltas %
    출력: {셀: {"vc", "ref_mean", "curve", "n_needed": {δ 라벨: 검정력 ≥ target인 최소 n_train (없으면 None)}}}
    """
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cells[(r["setting"], r["lam"], r["M"], r["K"])][r["policy"]].append(r)
    out = {}
    for cell, groups in sorted(cells.items()):
        if policy not in groups or ref not in groups:
            continue
        pm = paired_matrix(groups[policy], groups[ref])
        if pm is None:
            continue
        D, ts, es = pm
        vc  = variance_components(D)
        if math.isnan(vc["var_train"]):
            continue
        ref_mean = float(np.mean([r["y"] for r in groups[ref] if r["seed"] in set(es)]))
        deltas = {"observed": abs(vc["mean"]), **{f"{p:g}%": ref_mean * p / 100 for p in rel_deltas}}
        ne     = n_test or len(es)
        curve  = []
        need   = {}
        for lbl, d in deltas.items():
            c = power_curve(vc, [d], list(n_train), ne, alpha, n_tests)
            for r in c:
                r["label"] = lbl
            curve += c
            ok = [r["n_train"] for r in c if r["power"] >= target]
            need[lbl] = min(ok) if ok else None
        out[cell] = {"vc": vc, "ref_mean": ref_mean, "curve": curve, "n_needed": need, "n_test": ne}
    return out


# 혼합효과 모형

def fit_mixed(rows: List[dict], ref: str = REF_DEFAULT, formula: Optional[str] = None,
              by: Sequence[str] = ("setting", "lam", "M", "K")) -> List[dict]:
    """by 열로 나눈 묶음마다 MixedLM 적합: 고정효과 정책(기준 ref) + 분산성분 테스트 시드, run

    출력 행: by 값들, term, coef, se, z, p, ci_lo, ci_hi, n_obs, vc (분산성분 추정), converged (Holm p_holm 포함)
    """
    import warnings
    import pandas as pd
    import statsmodels.formula.api as smf

    formula = formula or f"y ~ C(policy, Treatment('{ref}'))"
    df  = pd.DataFrame(rows)
    out = []
    for key, g in df.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        if ref not in set(g["policy"]) or g["policy"].nunique() < 2:
            continue
        vc = {"seed": "0 + C(seed)"}
        if (g.groupby("policy")["train_seed"].nunique() > 1).any():
            vc["run"] = "0 + C(run)"
        g = g.assign(_grp=1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = smf.mixedlm(formula, g, groups="_grp", vc_formula=vc, re_formula="0").fit(reml=True)
        converged = bool(getattr(res, "converged", True)) and not any(
            "converge" in str(w.message).lower() for w in caught)
        ci  = res.conf_int()
        vcs = dict(zip(vc, np.atleast_1d(res.vcomp)))
        for term in res.fe_params.index:
            if term == "Intercept":
                continue
            out.append({**dict(zip(by, key)), "term": term, "coef": float(res.fe_params[term]),
                        "se": float(res.bse_fe[term]), "z": float(res.tvalues[term]),
                        "p": float(res.pvalues[term]), "ci_lo": float(ci.loc[term, 0]),
                        "ci_hi": float(ci.loc[term, 1]), "n_obs": len(g),
                        "vc": "; ".join(f"{k}={v:.3g}" for k, v in vcs.items()), "converged": converged})
    for r, ph in zip(out, holm([r["p"] for r in out])):
        r["p_holm"] = ph
    return out


# 리비전: 학습 시드 단위 통계, 규칙 기반 비교, 2³ 요인 효과

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
    """load_results와 같은 탐색 규칙으로 eval.csv 행 전체를 읽음 (지표는 metric_value로 꺼냄)

    출력 행: setting, lam, M, K, policy, run, train_seed, seed, rec (CSV 행 dict)
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
    """load_records 행에 y = metric 값을 붙인 복사본 (paired_matrix 입력 형식)"""
    return [{**r, "y": metric_value(r["rec"], metric)} for r in rows]


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


def seed_level_compare(pol_rows: List[dict], ref_rows: List[dict], B: int = 10000,
                       rng: Optional[np.random.Generator] = None, alpha: float = 0.05) -> Optional[dict]:
    """학습 시드 단위 비교 (정책 − 기준, 작을수록 좋음)

    - 시드별 평균 차이 m_i = mean_j (y_ij − ref_j) (공통 테스트 인스턴스), 보고 평균 = mean_i m_i
    - 95% CI: 시드→인스턴스 계층(nested) 부트스트랩
    - p: 시드 간 대응 t검정 (m_i의 일표본 t, 자유도 n_seed − 1). 학습 시드가 1개면 인스턴스 간 대응 t검정
    - n_better: m_i < 0인 시드 수
    - 상대 차이 ΔF% = 100 (F̄_pol − F̄_ref) / F̄_ref, 시드 단위 CI (nested_ratio_bootstrap)
    """
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
        p = float(st.ttest_1samp(m_i, 0.0).pvalue)
        test = "seed_t"
    else:
        p = float(st.ttest_1samp(D[0], 0.0).pvalue) if len(es) > 1 else float("nan")
        test = "instance_t"
    boot = hierarchical_bootstrap(D, B, "nested", rng)
    rb   = nested_ratio_bootstrap(Y, R, B, rng)
    return {"n_train": len(ts), "n_test": len(es), "mean_pol": float(Y.mean()), "mean_ref": float(R.mean()),
            "diff": float(m_i.mean()), "ci_lo": float(np.quantile(boot, alpha / 2)),
            "ci_hi": float(np.quantile(boot, 1 - alpha / 2)), "p": p, "test": test,
            "n_better": int(np.sum(m_i < 0)), "seed_diffs": ";".join(f"{x:.3f}" for x in m_i),
            "rel_pct": float(100.0 * (Y.mean() / R.mean() - 1.0)),
            "rel_ci_lo": float(np.quantile(rb, alpha / 2)), "rel_ci_hi": float(np.quantile(rb, 1 - alpha / 2))}


def rule_compare(pol_rows: List[dict], ref_rows: List[dict], alpha: float = 0.05) -> Optional[dict]:
    """규칙 기반 정책끼리 비교 (학습 시드 없음): 인스턴스 간 대응 t검정, t 기반 95% CI"""
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
    # 상대 차이 CI: 델타법 (평균 비율)
    rel = 100.0 * d.mean() / R.mean()
    return {"n_train": 1, "n_test": n, "mean_pol": float((d + R).mean()), "mean_ref": float(R.mean()),
            "diff": float(d.mean()), "ci_lo": float(d.mean() - tc * se), "ci_hi": float(d.mean() + tc * se),
            "p": float(st.ttest_rel(d + R, R).pvalue) if n > 1 else float("nan"), "test": "instance_t",
            "n_better": int(d.mean() < 0), "seed_diffs": "",
            "rel_pct": float(rel), "rel_ci_lo": float(rel - 100.0 * tc * se / R.mean()),
            "rel_ci_hi": float(rel + 100.0 * tc * se / R.mean())}


def holm_table(rows: List[dict], key: str = "p", out_key: str = "p_holm"):
    """표 하나의 모든 비교(행 × 지표)에 Holm 보정 (in-place)"""
    for r, ph in zip(rows, holm([r.get(key, float("nan")) for r in rows])):
        r[out_key] = ph
    return rows


def factorial_effects(obs: List[dict], factors: Sequence[str], alpha: float = 0.05) -> List[dict]:
    """2^k 요인 설계의 주효과와 2요인 상호작용 (±1 코딩 최소제곱, 학습 시드 = 반복)

    obs: {"y": 값, factor: ±1, ...} 리스트 (예: y = log(F̄_RL,seed / F̄_best))
    출력 행: term, coef (±1 코딩 계수, 효과 = 2 · coef), se, ci_lo, ci_hi, p, df
    """
    from itertools import combinations
    from scipy import stats as st
    terms = list(factors) + [f"{a}:{b}" for a, b in combinations(factors, 2)]
    X = np.array([[1.0] + [o[f] for f in factors] + [o[a] * o[b] for a, b in combinations(factors, 2)]
                  for o in obs])
    y = np.array([o["y"] for o in obs])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    df = len(y) - X.shape[1]
    resid = y - X @ beta
    s2 = float(resid @ resid / df) if df > 0 else float("nan")
    cov = s2 * np.linalg.pinv(X.T @ X)
    tc = st.t.ppf(1 - alpha / 2, df) if df > 0 else float("nan")
    out = []
    for i, name in enumerate(["Intercept"] + terms):
        se = math.sqrt(cov[i, i]) if df > 0 else float("nan")
        tval = beta[i] / se if se and se > 0 else float("nan")
        out.append({"term": name, "coef": float(beta[i]), "se": se, "ci_lo": float(beta[i] - tc * se),
                    "ci_hi": float(beta[i] + tc * se),
                    "p": float(2 * st.t.sf(abs(tval), df)) if df > 0 and not math.isnan(tval) else float("nan"),
                    "df": df, "n_obs": len(y)})
    return out


# 출력

def _save(rows: List[dict], path: str):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))     # 행마다 열이 달라도 합집합 (첫 행 순서 우선)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description="MTSR 평가 결과 통계 (부트스트랩, Holm, 혼합효과)")
    p.add_argument("--root", default="logs_rl", help="run_rl.py 결과 상위 폴더")
    p.add_argument("--metric", default="avg_flow_time", help="eval.csv 지표 열")
    p.add_argument("--ref", default=REF_DEFAULT, help="기준 정책 라벨")
    p.add_argument("--B", type=int, default=10000, help="부트스트랩 반복 수")
    p.add_argument("--mode", choices=["crossed", "nested"], default="crossed", help="부트스트랩 재표집 방식")
    p.add_argument("--relative", action="store_true", help="차이를 기준 대비 %%로")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--policies", nargs="*", default=None, help="이 정책 그룹(와 ref)만 분석")
    p.add_argument("--mixed", action="store_true", help="혼합효과 모형도 적합")
    p.add_argument("--mixed-formula", default=None, help="MixedLM 고정효과 식 (y, policy, M, lam 열 사용 가능)")
    p.add_argument("--mixed-by", nargs="*", default=["setting", "lam", "M", "K"], help="모형을 따로 적합할 묶음 열")
    p.add_argument("--out", default=None, help="부트스트랩 결과 CSV 경로 (혼합효과는 _mixed.csv, 검정력은 _power.csv)")
    p.add_argument("--power", default=None, metavar="POLICY", help="이 정책 그룹 − ref로 검정력 분석")
    p.add_argument("--power-rel", type=float, nargs="*", default=[2, 5, 10], help="검출할 효과 크기 (ref 평균의 %%)")
    p.add_argument("--power-n-train", type=int, nargs=2, default=[3, 10], help="학습 시드 수 범위")
    p.add_argument("--power-n-tests", type=int, default=1, help="다중비교 수 (α / n, Holm 첫 단계 기준)")
    p.add_argument("--power-target", type=float, default=0.8, help="목표 검정력")
    args = p.parse_args()

    rows = load_results(args.root, args.metric)
    if args.policies:
        keep = set(args.policies) | {args.ref}
        rows = [r for r in rows if r["policy"] in keep]
    if not rows:
        print("결과가 없습니다 (seed 열이 있는 eval.csv 필요)")
        return

    unit = "%" if args.relative else ""
    res = compare(rows, args.ref, args.B, args.mode, args.relative, args.alpha)
    print(f"\n{args.metric}: 정책 − {args.ref} (테스트 시드 대응, {args.mode} 부트스트랩 B={args.B}, "
          f"{100 * (1 - args.alpha):.0f}% CI, Holm 보정 {len(res)}개 비교)")
    last = None
    for r in res:
        cell = (r["setting"], r["lam"], r["M"], r["K"])
        if cell != last:
            print(f"\n  {r['setting']}  λ={r['lam']:.0f}  M={r['M']}  K={r['K']}")
            last = cell
        agree = f"  BL6일치 {r['bl6_agree']:.3f}" if not math.isnan(r["bl6_agree"]) else ""
        print(f"    {r['policy']:32s} {r['diff']:+8.2f}{unit} [{r['ci_lo']:+8.2f}, {r['ci_hi']:+8.2f}]  "
              f"p={r['p']:.4f}  p_holm={r['p_holm']:.4f}  (학습 {r['n_train']} × 테스트 {r['n_test']}){agree}")
    if args.out:
        _save(res, args.out)

    if args.mixed:
        mixed = fit_mixed(rows, args.ref, args.mixed_formula, args.mixed_by)
        print(f"\n혼합효과 모형 ({args.mixed_formula or 'y ~ policy'} + (1|seed) [+ (1|run)], Holm {len(mixed)}개 항)")
        for r in mixed:
            key = "  ".join(f"{b}={r[b]}" for b in args.mixed_by)
            flag = "" if r["converged"] else "  [수렴 경고]"
            print(f"  {key}  {r['term']:55s} {r['coef']:+8.2f} ± {r['se']:.2f}  "
                  f"p={r['p']:.4f}  p_holm={r['p_holm']:.4f}  ({r['vc']}){flag}")
        if args.out:
            _save(mixed, os.path.splitext(args.out)[0] + "_mixed.csv")

    if args.power:
        lo, hi = args.power_n_train
        rep = power_report(rows, args.power, args.ref, args.power_rel, range(lo, hi + 1),
                           alpha=args.alpha, n_tests=args.power_n_tests, target=args.power_target)
        flat = []
        for cell, r in rep.items():
            vc = r["vc"]
            print(f"\n검정력: {args.power} − {args.ref}  {cell[0]}  λ={cell[1]:.0f}  M={cell[2]}  K={cell[3]}")
            print(f"  관측 차이 {vc['mean']:+.2f} (학습 {vc['n_train']} × 테스트 {vc['n_test']}), "
                  f"학습 시드 평균의 SD {vc['sd_train_means']:.2f}")
            print(f"  분산성분: 학습 σ_a={math.sqrt(vc['var_train']):.2f}, 테스트 σ_b={math.sqrt(vc['var_test']):.2f}, "
                  f"잔차 σ_e={math.sqrt(vc['var_resid']):.2f}  (ref 평균 {r['ref_mean']:.1f})")
            labels = list(r["n_needed"])
            ns = sorted({c["n_train"] for c in r["curve"]})
            print("  n_train " + "".join(f"{l:>10s}" for l in labels))
            for n in ns:
                vals = {c["label"]: c["power"] for c in r["curve"] if c["n_train"] == n}
                print(f"  {n:7d} " + "".join(f"{vals[l]:10.3f}" for l in labels))
            print(f"  검정력 ≥ {args.power_target}: " + ", ".join(
                f"{l} (δ={d['delta']:.1f}) → {r['n_needed'][l] or f'>{hi}'}"
                for l in labels for d in [next(c for c in r["curve"] if c["label"] == l)]))
            for c in r["curve"]:
                flat.append({"setting": cell[0], "lam": cell[1], "M": cell[2], "K": cell[3],
                             "policy": args.power, "ref": args.ref, **c, "n_test": r["n_test"],
                             "sd_train_means": vc["sd_train_means"], "var_train": vc["var_train"],
                             "var_test": vc["var_test"], "var_resid": vc["var_resid"]})
        if args.out:
            _save(flat, os.path.splitext(args.out)[0] + "_power.csv")


if __name__ == "__main__":
    main()
