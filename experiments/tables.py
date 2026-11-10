"""논문 표 CSV와 학습 곡선 요약 (runner.py 결과 폴더 입력) → {out}/tables/

사용법:
    python experiments/tables.py                                   # 모든 표
    python experiments/tables.py --tables tab_main tab_fleet --B 2000
    python experiments/tables.py --arm <새 arm>                      # 기본 arm 대신 다른 arm을 RL 행으로

지표: F = avg_flow_time (완료 주문 평균 흐름시간), J = F + c_d · 주문당 이동거리 (analysis.metric_value)
기준(ref): 셀의 select.json best_label (검증 시드로 고른 최고 삽입형 규칙, BL4–BL7), 없으면 BL6_MarginalCost
  --pool new: best_label_new (BL4–BL8 + BL6*, experiments/stages.py stage0이 기록)
통계 (analysis.py 모듈 설명): RL은 학습 시드 단위, 규칙끼리는 인스턴스 대응, 같은 네트워크의 조건 비교는
  (학습 시드, 인스턴스) 대응. Holm 보정은 표 단위 (표의 모든 비교 × F, J)
출력 표 (design.py 실험 블록)
  tab_main            : main 셀마다 RL·모든 베이스라인 vs 기준
  tab_sources         : 개선 출처 — 삽입 (BL6 − 최고 배치 규칙), 학습 (RL − BL6), 합계 (RL − 최고 배치 규칙)
  tab_variants        : 학습 변형 arm vs BL6 (+ 기본 arm과 Welch t), 라우팅 블록 (경로 규칙별 같은 네트워크 재평가)
  tab_routing_gap     : route_gap.py 요약 (NNH vs exact)
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
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analysis as A                       # noqa: E402
import design as D                         # noqa: E402
from design import Cell, Ckpt              # noqa: E402
from evaluation import POLICIES            # noqa: E402
from metrics import display_name           # noqa: E402

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
        for r in A.load_records(root):
            self.idx[(r["setting"], r["lam"], r["M"], r["K"], r["policy"])].append(r)

    @staticmethod
    def key(cell: Cell):
        return cell.setting, float(cell.lam), cell.M, cell.K

    def get(self, cell: Cell, policy: str, metric: str):
        return A.with_metric(self.idx.get((*self.key(cell), policy), []), metric)

    def policies(self, cell: Cell):
        return sorted({k[4] for k in self.idx if k[:4] == self.key(cell)})

    def select(self, cell: Cell) -> dict:
        p = os.path.join(cell.dir(self.root), "select.json")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return {"best_label": BL6}

    def ref(self, cell: Cell) -> str:
        """셀 기준: pool = "old"이면 BL4–BL7 중 검증 최고, "new"이면 BL4–BL8 + BL6* (stages.py stage0의 best_label_new)"""
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
    return A.seed_level_compare(pr, rr, B, rng) if is_rl else A.rule_compare(pr, rr)


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
    for cell in D.MAIN_CELLS:
        ref = Dt.ref(cell)
        rows += _vs_ref("tab_main", Dt, cell, [arm] + BASELINE_LABELS, B, rng)
        rows.append(_row("tab_main", cell, ref, ref, "avg_flow_time", None, Dt))   # 기준 행 (평균만)
    return A.holm_table(rows)


def tab_sources(Dt, B, rng, arm):
    rows = []
    for cell in D.MAIN_CELLS:
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
    return A.holm_table(rows)


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
    cell = D.BASE_CELL
    rows = []
    for a in [arm] + D.VARIANT_ARMS:
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
    for rc in D.ROUTE_CELLS:
        block = f"routing_{rc.route}"
        pol = Ckpt(cell, arm, 0).policy(rc)
        for m in METRICS:
            res = compare(Dt, rc, pol, BL6, m, B, rng)
            if res:
                rows.append(_row("tab_variants", rc, pol, BL6, m, res, Dt, {"block": block}))
            res = A.matched_compare(Dt.get(rc, pol, m), Dt.get(cell, arm, m), B, rng)
            if res:
                rows.append(_row("tab_variants", rc, pol, f"{arm} (nnh)", m, res, Dt, {"block": block}))
            res = compare(Dt, rc, BL6, BL6, m, B, rng, ref_cell=cell)
            if res:
                rows.append(_row("tab_variants", rc, BL6, f"{BL6} (nnh)", m, res, Dt, {"block": block}))
    return A.holm_table(rows)


def tab_routing_gap(Dt, B, rng, arm):
    p = os.path.join(Dt.root, "tables", "tab_routing_gap_summary.csv")
    if not os.path.isfile(p):
        print("tab_routing_gap: route_gap.py 결과 없음 (runner.py routing --stage recon)")
        return []
    return [{"table": "tab_routing_gap", **r} for r in csv.DictReader(open(p, newline="", encoding="utf-8"))]


def tab_fleet(Dt, B, rng, arm):
    rows = []
    for cell in sorted([D.BASE_CELL] + D.FLEET_CELLS, key=lambda c: c.M):
        pol = Ckpt(D.BASE_CELL, arm, 0).policy(cell)
        rows += _vs_ref("tab_fleet", Dt, cell, [pol], B, rng)
    return A.holm_table(rows)


def tab_conditions(Dt, B, rng, arm):
    """요인 설계 8셀 + 기준 행, 네트워크 단위 요인 효과 (반환: 표 행, 효과 행, 관측)"""
    rows, obs = [], []
    for cell in D.FACTORIAL_CELLS:
        pol = D.factorial_nets(cell.K, arm)[0].policy(cell)
        rows += _vs_ref("tab_conditions", Dt, cell, [pol], B, rng, {"kind": "design"})
        # 요인 효과 관측: 학습 시드(네트워크)별 log(F̄_RL / F̄_ref), 공통 인스턴스
        pr, rr = Dt.get(cell, pol, "avg_flow_time"), Dt.get(cell, Dt.ref(cell), "avg_flow_time")
        pm = A.paired_matrix(pr, rr) if pr and rr else None
        if pm is not None:
            Dm, ts, es = pm
            ref_by = {r["seed"]: r["y"] for r in rr}
            Rm = np.mean([ref_by[e] for e in es])
            for i, t in enumerate(ts):
                obs.append({"y": math.log((Dm[i].mean() + Rm) / Rm), "lam": 1 if cell.lam == 320 else -1,
                            "M": 1 if cell.M == 18 else -1, "K": 1 if cell.K == 4 else -1,
                            "cell": f"{cell.lam:.0f}-{cell.M}-{cell.K}", "train_seed": t,
                            "unit": f"K{cell.K}_s{t}"})
    refs = [(D.BASE_CELL, D.factorial_ref_nets(arm)[0].policy(D.BASE_CELL), "ref_lam320_net"),
            (Cell(320, 14, 3), arm, "ref_lam320_net"), (D.BASE_CELL, arm, "ref_lam240_net")]
    for cell, pol, kind in refs:
        rows += _vs_ref("tab_conditions", Dt, cell, [pol], B, rng, {"kind": kind})
    eff = []
    try:
        eff = [{"table": "tab_conditions_effects", **e}
               for e in A.factorial_effects_by_unit(obs, within=["lam", "M"], between=["K"])] if obs else []
    except ValueError as e:
        print(f"tab_conditions: 요인 효과 생략 ({e})")
    return A.holm_table(rows), eff, obs


def tab_app_diagnostics(Dt, B, rng, arm):
    rows = []
    for cell in sorted(D.SMALL_CELLS + [D.BASE_CELL], key=lambda c: c.M):
        ref = Dt.ref(cell)
        pol = Ckpt(D.BASE_CELL, arm, 0).policy(cell)
        for p in [pol] + BASELINE_LABELS:
            if not Dt.get(cell, p, "avg_flow_time"):
                continue
            res = compare(Dt, cell, p, ref, "avg_flow_time", B, rng) if p != ref else None
            rows.append(_row("tab_app_diagnostics", cell, p, ref, "avg_flow_time", res, Dt))
    return A.holm_table(rows)


def fig_curves(Dt, arm):
    """기본 네트워크 학습 곡선 (보상, loss, |TD|, Q, α, λ, flow, 국소 비용) + 검증 flow, 50 에피소드 창 요약"""
    long, val = [], defaultdict(dict)
    f = lambda r, k: float(r.get(k, "nan") or "nan")
    for ck in D.base_nets(arm):
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


def tab_reward_sens(Dt, B, rng, arm, seeds=tuple(D.SUB_SEEDS)):
    """보상 계수 민감도 표와 요약 (중심점 = 기본 arm의 같은 학습 시드)"""
    rows, summ = [], []
    sel = lambda rs: [r for r in rs if r["train_seed"] in seeds]
    for lam in sorted({lam for lam, _ in D.SENS_DESIGN}):
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
                    rows.append({**base, "ref": BL6, **(A.seed_level_compare(pr, bl6, B, rng) or {})})
                if term is not None:
                    res = A.matched_compare(pr, sel(Dt.get(cell, arm, m)), B, rng)
                    if res:
                        rows.append({**base, "ref": f"{arm} (w = 1, 1)", **res})
        for term in ("wext", "wq"):
            f = {w: F.get((term, w), F.get((None, 1.0), float("nan")) if w == 1.0 else float("nan"))
                 for w in (0.0, 0.5, 1.0, 2.0)}
            ok = all(not math.isnan(f[w]) for w in (0.5, 1.0, 2.0))
            summ.append({"lam": float(lam), "term": term, "F_w0": f[0.0], "F_w0.5": f[0.5], "F_w1": f[1.0],
                         "F_w2": f[2.0], "log_elasticity": (f[2.0] - f[0.5]) / f[1.0] / math.log(4) if ok else float("nan"),
                         "n_train_seeds": len(seeds), **prior})
    return A.holm_table(rows), summ


SIMPLE = {"tab_main": tab_main, "tab_sources": tab_sources, "tab_variants": tab_variants,
          "tab_routing_gap": tab_routing_gap, "tab_fleet": tab_fleet, "tab_app_diagnostics": tab_app_diagnostics}
ALL_TABLES = list(SIMPLE) + ["tab_conditions", "fig_curves", "tab_reward_sens"]


def _print_rows(rows):
    for r in rows:
        if r.get("metric") == "avg_flow_time" and "diff" in r:
            print(f"  λ={r['lam']:.0f} M={r['M']} K={r['K']} {r['policy_name']:30s} vs {r['ref_name']:16s} "
                  f"ΔF {r['diff']:+7.2f} s [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] ({r['rel_pct']:+.1f}%)  "
                  f"p={r['p']:.4f} p_holm={r['p_holm']:.4f}  나은 시드 {r['n_better']}/{r['n_train']}")


def main():
    from rewards import ARMS
    p = argparse.ArgumentParser(description="논문 표 CSV")
    p.add_argument("--out", default=os.path.join(REPO, "logs_exp"), help="결과 상위 폴더 (runner.py --out)")
    p.add_argument("--tables", nargs="*", choices=ALL_TABLES, default=None, help="만들 표 (기본 전부)")
    p.add_argument("--arm", choices=list(ARMS), default=D.BASE_ARM, help="RL 행으로 쓸 arm")
    p.add_argument("--pool", choices=["old", "new"], default="old", help="기준 선택 풀 (모듈 설명)")
    p.add_argument("--B", type=int, default=10000, help="부트스트랩 반복 수")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    Dt  = Data(os.path.abspath(args.out), args.pool)
    rng = np.random.default_rng(args.seed)
    tdir = os.path.join(Dt.root, "tables")
    want = args.tables or ALL_TABLES
    for name, fn in SIMPLE.items():
        if name in want:
            rows = fn(Dt, args.B, rng, args.arm)
            if name != "tab_routing_gap":
                A.save_csv(rows, os.path.join(tdir, f"{name}.csv"))
            print(f"{name}: {len(rows)}행")
            _print_rows(rows)
    if "tab_conditions" in want:
        rows, eff, obs = tab_conditions(Dt, args.B, rng, args.arm)
        A.save_csv(rows, os.path.join(tdir, "tab_conditions.csv"))
        A.save_csv(eff, os.path.join(tdir, "tab_conditions_effects.csv"))
        A.save_csv(obs, os.path.join(tdir, "tab_conditions_obs.csv"))
        print(f"tab_conditions: {len(rows)}행, 요인 효과 {len(eff)}항")
        _print_rows(rows)
        for e in eff:
            print(f"  {e['term']:10s} {e['coef']:+.4f} [{e['ci_lo']:+.4f}, {e['ci_hi']:+.4f}] p={e['p']:.4f} "
                  f"(df {e['df']}, 네트워크 {e['n_units']})")
    if "tab_reward_sens" in want:
        rows, summ = tab_reward_sens(Dt, args.B, rng, args.arm)
        A.save_csv(rows, os.path.join(tdir, "tab_reward_sens.csv"))
        A.save_csv(summ, os.path.join(tdir, "tab_reward_sens_summary.csv"))
        print(f"tab_reward_sens: {len(rows)}행")
        for r in summ:
            print(f"  λ={r['lam']:.0f} {r['term']:4s} F(0, ½, 1, 2) = {r['F_w0']:.1f}, {r['F_w0.5']:.1f}, "
                  f"{r['F_w1']:.1f}, {r['F_w2']:.1f}  탄력성 {r['log_elasticity']:+.4f}")
    if "fig_curves" in want:
        long, summ = fig_curves(Dt, args.arm)
        A.save_csv(long, os.path.join(tdir, "fig_curves.csv"))
        A.save_csv(summ, os.path.join(tdir, "fig_curves_summary.csv"))
        print(f"fig_curves: {len(long)}행, 요약 {len(summ)}창")
    print(f"저장: {tdir}")


if __name__ == "__main__":
    main()
