"""리비전 분석: 논문 표별 CSV와 학습 곡선 요약 (revision.py 결과 폴더 입력)

사용법:
    python experiments/revision_analyze.py --out logs_exp            # 모든 표 → logs_exp/revision_tables/
    python experiments/revision_analyze.py --out logs_exp --tables tab_main tab_fleet --B 2000

지표: F = avg_flow_time (완료 주문 평균 흐름시간), J = F + c_d · 주문당 이동거리 (analysis.metric_value)
기준(ref): 셀의 select.json best_label (검증 시드 선택), 없으면 BL6_MarginalCost
통계 (analysis.py)
  - RL (학습 시드 있음): 시드별 평균 차이 → 평균, 시드→인스턴스 nested 부트스트랩 95% CI, 시드 간 대응 t검정 p,
    기준보다 나은 시드 수, ΔF% (시드 단위 CI)
  - 규칙 기반끼리: 인스턴스 간 대응 t검정, t 기반 CI
  - 같은 네트워크의 조건 비교 (route fixed_dock vs nnh): (학습 시드, 인스턴스) 대응 차이에 같은 통계
  - Holm 보정: 표 단위 (표의 모든 비교 × F, J)
  - 2³ 요인 효과 (tab_conditions_effects): y = log(F̄_RL,seed / F̄_best), 반복 = 학습 시드, ±1 코딩 최소제곱
출력 표
  tab_main, tab_sources, tab_variants, tab_routing_gap, tab_fleet, tab_conditions (+ _effects),
  tab_app_diagnostics, fig_curves (+ _summary)
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import analysis as A                       # noqa: E402
from metrics import display_name           # noqa: E402

BASE   = "local_r8_ddqn_v2"
SET    = "grid_free_reserve"
SET_FD = "grid_free_reserve_route-fixeddock"
DIAG   = ["avg_flow_time", "J", "avg_flow_time_censored", "completion_rate", "avg_unfinished_age",
          "decision_rate", "blocked_per_robot", "mid_trip_rate", "dist_per_order", "avg_totes_per_trip",
          "n_qpen_active", "n_depletion_active"]
BASELINE_LABELS = ["BL1F_FillBuffer", "BL1_FullBuffer", "BL2_Partial_k3", "BL3_Random", "BL4_Nearest",
                   "BL5_MinInsertion", "BL6_MarginalCost", "Greedy_R"]


class Data:
    """결과 행 색인: (setting, λ, M, K, policy) -> 행"""

    def __init__(self, root: str):
        self.root = root
        self.rows = A.load_records(root)
        self.idx = defaultdict(list)
        for r in self.rows:
            self.idx[(r["setting"], r["lam"], r["M"], r["K"], r["policy"])].append(r)

    def get(self, setting, lam, M, K, policy, metric):
        return A.with_metric(self.idx.get((setting, float(lam), M, K, policy), []), metric)

    def select(self, setting, lam, M, K) -> dict:
        p = os.path.join(self.root, setting, f"lam{lam:.0f}_M{M}_K{K}", "select.json")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return {"best_label": "BL6_MarginalCost"}

    def mean(self, setting, lam, M, K, policy, metric):
        rs = self.get(setting, lam, M, K, policy, metric)
        vals = [r["y"] for r in rs if not math.isnan(r["y"])]
        return float(np.mean(vals)) if vals else float("nan")


def _is_rl(rows):
    return any(r["rec"]["label"] == "RL" for r in rows)


def compare(D: Data, cell, pol, ref, metric, B, rng, ref_setting=None, pol_setting=None):
    """정책 그룹 vs 기준 (RL이면 시드 단위, 아니면 인스턴스 대응)"""
    s, lam, M, K = cell
    pr = D.get(pol_setting or s, lam, M, K, pol, metric)
    rr = D.get(ref_setting or s, lam, M, K, ref, metric)
    if not pr or not rr:
        return None
    return A.seed_level_compare(pr, rr, B, rng) if _is_rl(pr) else A.rule_compare(pr, rr)


def matched_compare(pol_rows, base_rows, B, rng, alpha=0.05):
    """같은 (학습 시드, 테스트 인스턴스) 대응 차이 (예: 같은 네트워크의 fixed_dock − nnh)"""
    from scipy import stats as st
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
    boot = A.hierarchical_bootstrap(Dm, B, "nested", rng)
    p = float(st.ttest_1samp(m_i, 0).pvalue) if len(ts) > 1 else float(st.ttest_1samp(Dm[0], 0).pvalue)
    # 상대 차이: 시드·인스턴스 nested 재추출로 mean P / mean Q
    ti = rng.integers(0, len(ts), size=(B, len(ts)))
    ei = rng.integers(0, len(es), size=(B, len(ts), len(es)))
    rb = 100.0 * (P[ti[:, :, None], ei].mean(axis=(1, 2)) / Q[ti[:, :, None], ei].mean(axis=(1, 2)) - 1)
    return {"n_train": len(ts), "n_test": len(es), "mean_pol": float(P.mean()), "mean_ref": float(Q.mean()),
            "diff": float(m_i.mean()), "ci_lo": float(np.quantile(boot, alpha / 2)),
            "ci_hi": float(np.quantile(boot, 1 - alpha / 2)), "p": p,
            "test": "matched_seed_t" if len(ts) > 1 else "matched_instance_t",
            "n_better": int(np.sum(m_i < 0)), "seed_diffs": ";".join(f"{x:.3f}" for x in m_i),
            "rel_pct": float(100 * (P.mean() / Q.mean() - 1)),
            "rel_ci_lo": float(np.quantile(rb, alpha / 2)), "rel_ci_hi": float(np.quantile(rb, 1 - alpha / 2))}


def _row(table, cell, pol, ref, metric, res, D, extra=None, pol_setting=None):
    s, lam, M, K = cell
    out = {"table": table, "setting": pol_setting or s, "lam": lam, "M": M, "K": K,
           "policy": pol, "policy_name": display_name(pol), "ref": ref, "ref_name": display_name(ref),
           "metric": metric, **(res or {})}
    for d in DIAG:
        out[f"mean_{d}"] = D.mean(pol_setting or s, lam, M, K, pol, d)
    out.update(extra or {})
    return out


def _policies_in(D, cell):
    s, lam, M, K = cell
    return sorted({k[4] for k in D.idx if k[:4] == (s, float(lam), M, K)})


# 표

def tab_main(D, B, rng):
    cell = (SET, 240.0, 14, 3)
    ref = D.select(*cell)["best_label"]
    rows = []
    for pol in [BASE] + [l for l in BASELINE_LABELS if l != ref]:
        for m in ("avg_flow_time", "J"):
            res = compare(D, cell, pol, ref, m, B, rng)
            if res:
                rows.append(_row("tab_main", cell, pol, ref, m, res, D))
    rows.append(_row("tab_main", cell, ref, ref, "avg_flow_time", None, D))   # 기준 행 (평균만)
    return A.holm_table(rows)


def tab_sources(D, B, rng):
    """개선의 출처: 삽입 (BL6 − 최고 배치 규칙), 학습 (RL − BL6), 합계 (RL − 최고 배치 규칙)"""
    cell = (SET, 240.0, 14, 3)
    sel = D.select(*cell)
    batch = sel.get("best_batching")
    rows = []
    pairs = [("learning", BASE, "BL6_MarginalCost")]
    if batch:
        pairs = [("insertion", "BL6_MarginalCost", batch)] + pairs + [("total", BASE, batch)]
    else:
        print("tab_sources: select.json에 best_batching 없음 (revision.py T1 필요), 학습 효과만")
    for src, pol, ref in pairs:
        for m in ("avg_flow_time", "J"):
            res = compare(D, cell, pol, ref, m, B, rng)
            if res:
                rows.append(_row("tab_sources", cell, pol, ref, m, res, D, {"source": src}))
    return A.holm_table(rows)


def _train_rates(root, arm):
    """학습 train.csv에서 큐 압박/소진 페널티 활성 비율 (결정당, 에피소드 평균), 시드 평균"""
    out = defaultdict(list)
    for p in glob.glob(os.path.join(root, SET, "lam240_M14_K3", f"{arm}_s*", "train.csv")):
        rs = list(csv.DictReader(open(p, newline="")))
        nd = [max(1.0, float(r["n_decisions"])) for r in rs]
        out["train_qpen_rate"].append(np.mean([float(r["n_qpen_active"]) / n for r, n in zip(rs, nd)]))
        out["train_depletion_rate"].append(np.mean([float(r["n_depletion_active"]) / n for r, n in zip(rs, nd)]))
        out["train_decision_rate"].append(np.mean([float(r["n_decisions"]) / max(1.0, float(r["n_totes"]))
                                                   for r in rs]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def tab_variants(D, B, rng):
    from scipy import stats as st
    cell = (SET, 240.0, 14, 3)
    rows = []
    for arm in (BASE, "local_r8_ddqn_v2_gamma", "legacy_r8_ddqn_v2"):
        extra = {"block": "training", **_train_rates(D.root, arm)}
        for m in ("avg_flow_time", "J"):
            res = compare(D, cell, arm, "BL6_MarginalCost", m, B, rng)
            if res is None:
                continue
            if arm != BASE:      # 기본 arm과의 차이: 시드 평균끼리 Welch t (학습 시드가 독립)
                a = _seed_means(D.get(*cell, arm, m))
                b = _seed_means(D.get(*cell, BASE, m))
                if len(a) > 1 and len(b) > 1:
                    extra = {**extra, "vs_base_diff": float(np.mean(a) - np.mean(b)),
                             "vs_base_p_welch": float(st.ttest_ind(a, b, equal_var=False).pvalue)}
            rows.append(_row("tab_variants", cell, arm, "BL6_MarginalCost", m, res, D, extra))
    # 라우팅 블록 (가중치 고정): fixed_dock RL vs fixed_dock BL6, fixed_dock RL vs nnh RL (같은 네트워크)
    fd = (SET_FD, 240.0, 14, 3)
    pol_fd = BASE + "_route-fixeddock"
    for m in ("avg_flow_time", "J"):
        res = compare(D, fd, pol_fd, "BL6_MarginalCost", m, B, rng)
        if res:
            rows.append(_row("tab_variants", fd, pol_fd, "BL6_MarginalCost", m, res, D, {"block": "routing"}))
        res = matched_compare(D.get(*fd, pol_fd, m), D.get(*cell, BASE, m), B, rng)
        if res:
            rows.append(_row("tab_variants", fd, pol_fd, BASE + " (nnh)", m, res, D, {"block": "routing"}))
        res = A.rule_compare(D.get(*fd, "BL6_MarginalCost", m), D.get(*cell, "BL6_MarginalCost", m)) \
            if D.get(*fd, "BL6_MarginalCost", m) else None
        if res:
            rows.append(_row("tab_variants", fd, "BL6_MarginalCost", "BL6_MarginalCost (nnh)", m, res, D,
                             {"block": "routing"}))
    return A.holm_table(rows)


def _seed_means(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["train_seed"]].append(r["y"])
    return [float(np.mean(v)) for _, v in sorted(by.items())]


def tab_routing_gap(D, B, rng):
    p = os.path.join(D.root, "revision_tables", "tab_routing_gap_summary.csv")
    if not os.path.isfile(p):
        print("tab_routing_gap: route_gap.py 결과 없음 (revision.py T5)")
        return []
    return [{"table": "tab_routing_gap", **r} for r in csv.DictReader(open(p, newline="", encoding="utf-8"))]


def tab_fleet(D, B, rng):
    rows = []
    for M in (8, 10, 12, 14, 16, 18, 20):
        lam = 240.0 if M == 14 else float(round(240 * M / 14))
        cell = (SET, lam, M, 3)
        pol = BASE if M == 14 else f"{BASE}_tr14_trlam240"
        ref = D.select(*cell)["best_label"]
        for m in ("avg_flow_time", "J"):
            res = compare(D, cell, pol, ref, m, B, rng)
            if res:
                rows.append(_row("tab_fleet", cell, pol, ref, m, res, D))
    return A.holm_table(rows)


def tab_conditions(D, B, rng):
    """다요인 설계점 8개 + 기준 행, 그리고 2³ 요인 효과 (반환: 표 행, 효과 행)"""
    rows, obs = [], []
    design = [(lam, M, K) for lam in (240.0, 320.0) for M in (10, 18) for K in (2, 4)]
    refs = [(240.0, 14, 3, f"{BASE}_trlam320", "ref_lam320_net"), (320.0, 14, 3, BASE, "ref_lam320_net"),
            (240.0, 14, 3, BASE, "ref_lam240_net")]
    for lam, M, K in design:
        cell = (SET, lam, M, K)
        pol = f"{BASE}_tr14" + ("_trlam320" if lam == 240.0 else "")
        ref = D.select(*cell)["best_label"]
        for m in ("avg_flow_time", "J"):
            res = compare(D, cell, pol, ref, m, B, rng)
            if res:
                rows.append(_row("tab_conditions", cell, pol, ref, m, res, D, {"kind": "design"}))
        # 요인 효과 관측: 시드별 log(F̄_RL / F̄_best)
        pr = D.get(*cell, pol, "avg_flow_time")
        rr = D.get(*cell, ref, "avg_flow_time")
        pm = A.paired_matrix(pr, rr) if pr and rr else None
        if pm is not None:
            Dm, ts, es = pm
            ref_by = {r["seed"]: r["y"] for r in rr}
            Rm = np.mean([ref_by[e] for e in es])
            for i, t in enumerate(ts):
                obs.append({"y": math.log((Dm[i].mean() + Rm) / Rm), "lam": 1 if lam == 320 else -1,
                            "M": 1 if M == 18 else -1, "K": 1 if K == 4 else -1,
                            "cell": f"{lam:.0f}-{M}-{K}", "train_seed": t})
    for lam, M, K, pol, kind in refs:
        cell = (SET, lam, M, K)
        ref = D.select(*cell)["best_label"]
        for m in ("avg_flow_time", "J"):
            res = compare(D, cell, pol, ref, m, B, rng)
            if res:
                rows.append(_row("tab_conditions", cell, pol, ref, m, res, D, {"kind": kind}))
    eff = []
    if len({o["cell"] for o in obs}) == 8:
        eff = [{"table": "tab_conditions_effects", **e} for e in A.factorial_effects(obs, ["lam", "M", "K"])]
    else:
        print(f"tab_conditions: 설계점 {len({o['cell'] for o in obs})}/8개만 있어 요인 효과 생략")
    return A.holm_table(rows), eff, obs


def tab_app_diagnostics(D, B, rng):
    rows = []
    for M in (6, 10, 14):
        cell = (SET, 240.0, M, 3)
        ref = D.select(*cell)["best_label"]
        pol = BASE if M == 14 else f"{BASE}_tr14"
        for p in [pol] + BASELINE_LABELS:
            if not D.get(*cell, p, "avg_flow_time"):
                continue
            res = compare(D, cell, p, ref, "avg_flow_time", B, rng) if p != ref else None
            rows.append(_row("tab_app_diagnostics", cell, p, ref, "avg_flow_time", res, D))
    return A.holm_table(rows)


def fig_curves(D):
    """기본 네트워크 5개 학습 곡선 (보상, loss, |TD|, Q, α, λ, flow) + 검증 flow, 50 에피소드 창 요약"""
    long, val = [], defaultdict(dict)
    cdir = os.path.join(D.root, SET, "lam240_M14_K3")
    for s in range(1, 6):
        run = os.path.join(cdir, f"{BASE}_s{s}")
        cp = os.path.join(run, "curve.csv")
        if not os.path.isfile(cp):
            continue
        vp = os.path.join(run, "val.csv")
        if os.path.isfile(vp):
            by = defaultdict(list)
            for r in csv.DictReader(open(vp, newline="")):
                by[int(r["episode"])].append(float(r["avg_flow_time"]))
            val[s] = {e: float(np.mean(v)) for e, v in by.items()}
        for r in csv.DictReader(open(cp, newline="")):
            ep = int(r["episode"])
            long.append({"train_seed": s, "episode": ep, "lam": float(r["lam"]), "reward": float(r["reward"]),
                         "loss": float(r["loss"]), "td_abs": float(r["td_abs"]), "q_mean": float(r["q_mean"]),
                         "alpha": float(r["alpha"]), "avg_flow_time": float(r["avg_flow_time"]),
                         "local_mu": float(r.get("local_mu", "nan")), "local_cost": float(r.get("local_cost", "nan")),
                         "val_flow": val.get(s, {}).get(ep, float("nan"))})
    summ = []
    keys = ["lam", "reward", "loss", "td_abs", "q_mean", "alpha", "avg_flow_time", "local_cost"]
    for w0 in range(1, max([r["episode"] for r in long] or [0]) + 1, 50):
        win = [r for r in long if w0 <= r["episode"] < w0 + 50]
        if not win:
            continue
        row = {"ep_from": w0, "ep_to": w0 + 49, "n_seeds": len({r["train_seed"] for r in win})}
        for k in keys:
            v = [r[k] for r in win if not math.isnan(r[k])]
            row[f"{k}_mean"] = float(np.mean(v)) if v else float("nan")
            seed_means = [np.mean([r[k] for r in win if r["train_seed"] == s and not math.isnan(r[k])] or [np.nan])
                          for s in sorted({r["train_seed"] for r in win})]
            row[f"{k}_sd_seeds"] = float(np.nanstd(seed_means, ddof=1)) if len(seed_means) > 1 else float("nan")
        vf = [v[e] for v in val.values() for e in v if w0 <= e < w0 + 50]
        row["val_flow_mean"] = float(np.mean(vf)) if vf else float("nan")
        summ.append(row)
    return long, summ


def main():
    p = argparse.ArgumentParser(description="리비전 논문 표 CSV")
    p.add_argument("--out", default=os.path.join(REPO, "logs_exp"), help="결과 상위 폴더 (revision.py --out)")
    p.add_argument("--tables", nargs="*", default=None, help="만들 표 (기본 전부)")
    p.add_argument("--B", type=int, default=10000, help="부트스트랩 반복 수")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    D = Data(os.path.abspath(args.out))
    rng = np.random.default_rng(args.seed)
    tdir = os.path.join(D.root, "revision_tables")
    want = set(args.tables or ["tab_main", "tab_sources", "tab_variants", "tab_routing_gap", "tab_fleet",
                               "tab_conditions", "tab_app_diagnostics", "fig_curves"])
    simple = {"tab_main": tab_main, "tab_sources": tab_sources, "tab_variants": tab_variants,
              "tab_routing_gap": tab_routing_gap, "tab_fleet": tab_fleet, "tab_app_diagnostics": tab_app_diagnostics}
    for name, fn in simple.items():
        if name in want:
            rows = fn(D, args.B, rng)
            if name != "tab_routing_gap" or rows:
                A._save(rows, os.path.join(tdir, f"{name}.csv"))
            print(f"{name}: {len(rows)}행")
            for r in rows:
                if r.get("metric") == "avg_flow_time" and "diff" in r:
                    print(f"  λ={r['lam']:.0f} M={r['M']} K={r['K']} {r['policy_name']:28s} vs {r['ref_name']:14s} "
                          f"ΔF {r['diff']:+7.2f} s [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] "
                          f"({r['rel_pct']:+.1f}%)  p={r['p']:.4f} p_holm={r['p_holm']:.4f}  "
                          f"나은 시드 {r['n_better']}/{r['n_train']}")
    if "tab_conditions" in want:
        rows, eff, obs = tab_conditions(D, args.B, rng)
        A._save(rows, os.path.join(tdir, "tab_conditions.csv"))
        A._save(eff, os.path.join(tdir, "tab_conditions_effects.csv"))
        A._save(obs, os.path.join(tdir, "tab_conditions_obs.csv"))
        print(f"tab_conditions: {len(rows)}행, 요인 효과 {len(eff)}항")
        for e in eff:
            print(f"  {e['term']:10s} {e['coef']:+.4f} [{e['ci_lo']:+.4f}, {e['ci_hi']:+.4f}] p={e['p']:.4f}")
    if "fig_curves" in want:
        long, summ = fig_curves(D)
        A._save(long, os.path.join(tdir, "fig_curves.csv"))
        A._save(summ, os.path.join(tdir, "fig_curves_summary.csv"))
        print(f"fig_curves: {len(long)}행, 요약 {len(summ)}창")
    print(f"저장: {tdir}")


if __name__ == "__main__":
    main()
