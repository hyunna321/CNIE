"""작업 지시 단계별 실행과 보고 (베이스라인 보강, 보상 재설계)

사용법 (scripts/run_stage*.sh가 호출, --dry-run이면 명령만 출력)
    python experiments/stages.py stage0 [--cells all|main|base] [--scenario NAME] [--rl-root DIR]
    python experiments/stages.py train  --stage stage2 --arms order_inc [--mode smoke|screen|final] [--scenario NAME]
    python experiments/stages.py report --stage stage2 --arms local_r8_ddqn_v2 local_noterm2 order_inc little_pbrs

stage0 (학습 없음): 셀마다
    1. 검증 시드 N_VAL_SELECT개: BL6 c_d ∈ BL6_CD_GRID × tote_order 2가지, BL8 × 2, BL4, BL5, BL7
       → BL6* c_d (tote_order마다 검증 평균 흐름시간 최소, 동률이면 격자 앞쪽), 기준 선택
         best_label     = 기존 풀 BL4–BL7 (runner.py와 같음)
         best_label_new = 새 풀 BL4–BL8 + BL6*, BL6·BL6*·BL8은 tote_order 두 가지 모두 후보
       → select.json에 병합 (bl6_star, best_label, best_label_new, 순위)
    2. 테스트 시드 N_TEST개: design.ALL_KEYS + BL8, BL6@bf, BL8@bf, BL6*(두 순서) → baselines/eval.csv
    3. results/stage0/: summary.md, metrics.csv (정책별 지표 평균), pairs.csv (대응 비교), select.csv (선택 결과)
       RL(현행) 행은 --rl-root의 eval.csv (현행 arm을 runner.py main/fleet/factorial로 학습·전이한 결과)
    셀: all = main 3 + fleet 6 (Table 11) + factorial 8 (Table 12), main, base
train (단계 1~5): arm마다 μ/ρ 표(필요하면) → 학습 시드를 병렬 프로세스로 run_rl.py 학습 + 기준 셀 테스트
    mode: smoke = 20 에피소드, 시드 1 / screen = 1000 에피소드, 시드 1~3 / final = 1000 에피소드, 시드 1~5
    결과 runs/<stage>_<arm>/ (run_rl.py --out), 로그 runs/<stage>_<arm>/s<seed>/train.log
    프로토콜 (모든 arm 공통, 바꾸지 않음): --select val --val-every 50 --n-val N_VAL_RL, 테스트 N_TEST
report (STOP 2 등): arm마다 RL − BL6, RL − BL6*, RL − BL8 (학습 시드 단위 t 95% CI, 이긴 시드 수),
    현행 arm과의 Welch t, 학습 진단 (Table 6 형식: 구간별 loss, |TD|, mean Q, κ_θ = α), 결정 분류 (그림 6),
    검증 곡선 → results/<stage>/
    비교 베이스라인의 tote_order는 arm과 같은 것 (arm이 bottleneck_first이면 BL6@bf, BL6*@bf, BL8@bf)
시나리오 (--scenario, 단계 5): multi = 주문당 SKU 기하분포 평균 4 (10에서 절단),
    steps = 2시간 구간 5개 λ = [200, 320, 200, 280, 200] /hr (평균 240, thinning). steps 학습은 *_nocurr arm만
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analysis as A                                   # noqa: E402
import design as D                                     # noqa: E402
import runner as RN                                    # noqa: E402
from design import Cell, Ckpt                          # noqa: E402
from constant import SIM_TIME, BL6_CD_GRID             # noqa: E402
from evaluation import policy_label, test_seeds, val_seeds   # noqa: E402
from metrics import display_name                       # noqa: E402

SCENARIOS = {
    "multi": {"lines_mean": 4.0, "lines_max": 10},
    "steps": {"arrival": "nhpp_steps", "step_rates": [x / 240 for x in (200, 320, 200, 280, 200)]},
}
ORDERS = {"arrival": "", "bottleneck_first": "@bf"}
MODES = {"smoke": (20, [1]), "screen": (1000, D.SUB_SEEDS), "final": (1000, D.TRAIN_SEEDS)}
REPORT_METRICS = ["avg_flow_time", "avg_flow_time_censored", "flow_p99", "dist_per_order", "avg_wait_time",
                  "avg_totes_per_trip", "mid_trip_rate", "blocked_per_robot", "utilization", "idle_at_decision",
                  "decision_rate"]
PHASES = [("1–500", 1, 500), ("501–600", 501, 600), ("901–1000", 901, 1000)]


def cd_key(c: float, bf: str = "") -> str:
    return f"BL6c{c:g}{bf}"


def scenario_json(name: Optional[str]) -> str:
    return json.dumps(SCENARIOS[name], sort_keys=True) if name else ""


def stage0_cells(which: str, sc: str) -> List[Cell]:
    if which == "base":
        cells = [D.BASE_CELL]
    elif which == "main":
        cells = D.MAIN_CELLS
    else:
        cells = D.MAIN_CELLS + D.FLEET_CELLS + D.FACTORIAL_CELLS
    return [Cell(c.lam, c.M, c.K, c.route, c.traffic, c.motion, sc) for c in cells]


def _mean_flow(st, label: str) -> float:
    v = [r.avg_flow_time for r in st.records if r.label == label]
    return float(np.mean(v)) if v else float("nan")


def _md_table(rows: List[dict], cols: List[str], fmt: Dict[str, str] = None) -> str:
    fmt = fmt or {}
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(format(v, fmt[c]) if c in fmt and isinstance(v, (int, float)) and not
                         (isinstance(v, float) and math.isnan(v)) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# 단계 0

def stage0(args):
    cells = stage0_cells(args.cells, scenario_json(args.scenario))
    vkeys = [cd_key(c, bf) for c in BL6_CD_GRID for bf in ORDERS.values()] + ["BL8", "BL8@bf", "BL4", "BL5", "BL7"]
    res = RN.run_baselines(args, cells, vkeys, val_seeds(D.N_VAL_SELECT), "val.csv")
    if args.dry_run:
        return
    stars = {}
    for c, st in res.items():
        star = {o: min(BL6_CD_GRID, key=lambda x: _mean_flow(st, policy_label(cd_key(x, bf))))
                for o, bf in ORDERS.items()}
        stars[c] = star
        old = ["BL4", "BL5", "BL6", "BL7"]
        new = old + ["BL8", "BL6@bf", "BL8@bf", cd_key(star["arrival"]), cd_key(star["bottleneck_first"], "@bf")]
        rank = lambda ks: sorted(({"key": k, "label": policy_label(k), "avg_flow_time": _mean_flow(st, policy_label(k))}
                                  for k in dict.fromkeys(ks)), key=lambda d: d["avg_flow_time"])
        r_old, r_new = rank(old), rank(new)
        path = os.path.join(c.dir(args.out), "select.json")
        sel = json.load(open(path, encoding="utf-8")) if os.path.isfile(path) else {}
        sel.update(best_key=r_old[0]["key"], best_label=r_old[0]["label"], ranking=r_old,
                   best_key_new=r_new[0]["key"], best_label_new=r_new[0]["label"], ranking_new=r_new,
                   bl6_star=star, bl6_cd_val={o: {f"{x:g}": _mean_flow(st, policy_label(cd_key(x, bf)))
                                                  for x in BL6_CD_GRID} for o, bf in ORDERS.items()},
                   val_seeds=val_seeds(D.N_VAL_SELECT), sim_time=args.sim_time, cell=vars(c))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sel, f, indent=1, ensure_ascii=False)
        print(f"{c}: BL6* c_d = {star}, 기존 풀 최고 {r_old[0]['label']}, 새 풀 최고 {r_new[0]['label']}")
    tkeys = list(D.ALL_KEYS) + ["BL8", "BL6@bf", "BL8@bf"]
    for c in cells:
        keys = tkeys + [cd_key(stars[c]["arrival"]), cd_key(stars[c]["bottleneck_first"], "@bf")]
        RN.run_baselines(args, [c], list(dict.fromkeys(keys)), test_seeds(args.n_test), "eval.csv")
    RN.progress(args, "stage0", [f"{c}: BL6* {stars[c]}" for c in cells])
    report0(args, cells)


def _rl_policy(cell: Cell, arm: str) -> str:
    """현행 RL의 셀별 정책 그룹 이름 (main: 그 셀 학습, fleet: 기본 네트워크 전이, factorial: K별 네트워크 전이)"""
    base = Cell(cell.lam, cell.M, cell.K)
    if base in D.MAIN_CELLS:
        return arm
    if base in D.FACTORIAL_CELLS:
        return D.factorial_nets(cell.K, arm)[0].policy(base)
    return Ckpt(D.BASE_CELL, arm, 0).policy(base)


def report0(args, cells: List[Cell]):
    from tables import Data
    Db = Data(args.out)
    Dr = Data(args.rl_root) if args.rl_root and os.path.isdir(args.rl_root) else None
    met_rows, pair_rows, sel_rows = [], [], []
    for c in cells:
        sel = Db.select(c)
        star = sel.get("bl6_star", {"arrival": 1.0, "bottleneck_first": 1.0})
        sel_rows.append({"cell": str(c), "lam": c.lam, "M": c.M, "K": c.K, "cd_star": star["arrival"],
                         "cd_star_bf": star["bottleneck_first"], "best_old": display_name(sel.get("best_label", "")),
                         "best_new": display_name(sel.get("best_label_new", ""))})
        labels = {display_name(policy_label(k)): policy_label(k) for k in D.ALL_KEYS + ["BL8", "BL6@bf", "BL8@bf"]}
        labels[f"BL6* (c_d={star['arrival']:g})"] = policy_label(cd_key(star["arrival"]))
        labels[f"BL6*_bf (c_d={star['bottleneck_first']:g})"] = policy_label(cd_key(star["bottleneck_first"], "@bf"))
        rl_pol = _rl_policy(c, D.BASE_ARM)
        for name, lab in labels.items():
            row = {"cell": str(c), "policy": name, "label": lab}
            for m in REPORT_METRICS:
                row[m] = Db.mean(c, lab, m)
            met_rows.append(row)
        if Dr is not None and Dr.get(c, rl_pol, "avg_flow_time"):
            row = {"cell": str(c), "policy": "RL (현행)", "label": rl_pol}
            for m in REPORT_METRICS:
                row[m] = Dr.mean(c, rl_pol, m)
            met_rows.append(row)
        s_arr, s_bf = labels[f"BL6* (c_d={star['arrival']:g})"], labels[f"BL6*_bf (c_d={star['bottleneck_first']:g})"]
        pairs = [("BL8 − BL6", "BL8_OrderPlanned", "BL6_MarginalCost"), ("BL8 − BL6*", "BL8_OrderPlanned", s_arr),
                 ("BL8_bf − BL6_bf", "BL8_OrderPlanned_bf", "BL6_MarginalCost_bf"),
                 ("BL8_bf − BL6*_bf", "BL8_OrderPlanned_bf", s_bf), ("BL6* − BL6", s_arr, "BL6_MarginalCost")]
        for name, a, b in pairs:
            for m in ("avg_flow_time", "J"):
                res = A.rule_compare(Db.get(c, a, m), Db.get(c, b, m)) if Db.get(c, a, m) and Db.get(c, b, m) else None
                if res:
                    pair_rows.append({"cell": str(c), "pair": name, "metric": m, **res})
        if Dr is not None:
            for name, b in (("RL(현행) − BL6*", s_arr), ("RL(현행) − BL6*_bf", s_bf), ("RL(현행) − BL6", "BL6_MarginalCost")):
                for m in ("avg_flow_time", "J"):
                    pr, rr = Dr.get(c, rl_pol, m), Db.get(c, b, m)
                    res = A.seed_level_compare(pr, rr) if pr and rr else None
                    if res:
                        pair_rows.append({"cell": str(c), "pair": name, "metric": m, **res,
                                          "ci_lo": res["ci_t_lo"], "ci_hi": res["ci_t_hi"]})
    A.holm_table(pair_rows)
    rdir = os.path.join(args.results, "stage0")
    for name, rows in (("metrics", met_rows), ("pairs", pair_rows), ("select", sel_rows)):
        A.save_csv(rows, os.path.join(rdir, f"{name}.csv"))
    # 갈림길 판정 (95% CI 상한 < 0 = 유의하게 빠름)
    F = [r for r in pair_rows if r["metric"] == "avg_flow_time"]
    faster = lambda name: [r["cell"] for r in F if r["pair"] == name and r["ci_hi"] < 0]
    n_cells = len(cells)
    g1 = {n: faster(n) for n in ("BL8 − BL6*", "BL8_bf − BL6*_bf")}
    g2 = {n: faster(n) for n in ("RL(현행) − BL6*", "RL(현행) − BL6*_bf")}
    lines = ["# 단계 0 — 학습 없는 베이스라인 보강\n",
             f"- 결과 폴더 `{args.out}`, 테스트 {args.n_test}개 / 검증 {D.N_VAL_SELECT}개 인스턴스, 셀 {n_cells}개",
             "- CI: 규칙끼리 인스턴스 대응 t 95%, RL은 학습 시드 단위 t 95% (Holm은 pairs.csv의 p_holm)\n",
             "## 갈림길 판정 (평균 흐름시간, CI 상한 < 0)\n",
             f"- G1 (BL8이 BL6*보다 유의하게 빠름): arrival {len(g1['BL8 − BL6*'])}/{n_cells}셀, "
             f"bottleneck_first {len(g1['BL8_bf − BL6*_bf'])}/{n_cells}셀",
             f"  - arrival: {', '.join(g1['BL8 − BL6*']) or '없음'}",
             f"  - bottleneck_first: {', '.join(g1['BL8_bf − BL6*_bf']) or '없음'}",
             (f"- G2 (현행 RL이 BL6*보다 유의하게 빠른 설정): BL6* {len(g2['RL(현행) − BL6*'])}셀, "
              f"BL6*_bf {len(g2['RL(현행) − BL6*_bf'])}셀 — {', '.join(g2['RL(현행) − BL6*']) or '없음'}"
              if Dr is not None else "- G2: RL(현행) 결과 없음 (--rl-root)"),
             "\n## BL6* c_d와 기준 선택 (검증 시드)\n",
             _md_table(sel_rows, ["cell", "cd_star", "cd_star_bf", "best_old", "best_new"]),
             "\n## 대응 비교 (평균 흐름시간 차이 s, 95% CI)\n",
             _md_table([{**r, "CI": f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]"} for r in F],
                       ["cell", "pair", "diff", "CI", "rel_pct", "p", "p_holm"],
                       {"diff": "+.2f", "rel_pct": "+.1f", "p": ".4f", "p_holm": ".4f"}),
             "\n## 지표 (main 셀, 테스트 평균)\n",
             _md_table([r for r in met_rows if any(r["cell"] == str(c) for c in cells if Cell(c.lam, c.M, c.K) in D.MAIN_CELLS)],
                       ["cell", "policy"] + REPORT_METRICS, {m: ".2f" for m in REPORT_METRICS}),
             "\n전체 셀 지표: metrics.csv"]
    os.makedirs(rdir, exist_ok=True)
    with open(os.path.join(rdir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"저장: {rdir}/summary.md")


# 학습 (단계 1~5)

def train(args):
    from rewards import ARMS
    episodes, seeds = MODES[args.mode]
    sc = scenario_json(args.scenario)
    cell = Cell(D.BASE_CELL.lam, D.BASE_CELL.M, D.BASE_CELL.K, scenario=sc)
    val_every = 10 if args.mode == "smoke" else 50
    for arm in args.arms:
        if args.scenario == "steps" and ARMS[arm].curriculum:
            raise SystemExit(f"steps 시나리오는 커리큘럼 없는 arm만 ({arm}_nocurr)")
        exp = os.path.join(args.runs, f"{args.stage}_{arm}")
        base = [sys.executable, "-u", os.path.join(REPO, "run_rl.py"), "--arm", arm, *cell.args(),
                "--episodes", str(episodes), "--sim-time", str(args.sim_time), "--n-eval", str(args.n_test),
                "--select", "val", "--val-every", str(val_every), "--n-val", str(D.N_VAL_RL), "--baselines",
                "--out", exp]
        if ARMS[arm].uses_mu:
            fails = RN.run_procs([{"cmd": base + ["--mu-only"], "name": f"μ/ρ {arm}", "dir": exp,
                                   "log": os.path.join(exp, "mu.log")}], 1, args.dry_run)
            if fails:
                print(f"{arm}: 중심화 기준 추정 실패, 로그 {exp}/mu.log")
                continue
        items = []
        for s in seeds:
            done = os.path.join(Ckpt(cell, arm, s).dir(exp), "eval.csv")
            if os.path.isfile(done) and args.mode != "smoke":
                print(f"{arm} s{s}: 완료, 건너뜀")
                continue
            items.append({"cmd": base + ["--seed", str(s)] + (["--fresh"] if args.mode == "smoke" else []),
                          "name": f"{arm} s{s}", "dir": exp, "log": os.path.join(exp, f"s{s}", "train.log")})
        fails = RN.run_procs(items, args.jobs, args.dry_run)
        RN.progress(argparse.Namespace(out=exp, dry_run=args.dry_run), f"train {args.stage} {arm} ({args.mode})",
                    RN._summary_lines(items, fails))


# 보고 (STOP 2 등)

def _find_root(stage: str, arm: str, runs: str, rl_root: Optional[str]) -> Optional[str]:
    from rewards import DEFAULT_ARM
    p = os.path.join(runs, f"{stage}_{arm}")
    if os.path.isdir(p):
        return p
    hits = sorted(glob.glob(os.path.join(runs, f"*_{arm}")), key=os.path.getmtime)
    if hits:
        return hits[-1]
    return rl_root if arm == DEFAULT_ARM else None


def _phase_stats(run_dirs: List[str]) -> List[dict]:
    """Table 6 형식: 구간별 (loss, |TD|, mean Q, κ_θ) 시드 평균 ± SD"""
    out = []
    for name, a, b in PHASES:
        row = {"phase": name}
        for k, col in (("loss", "loss"), ("td_abs", "abs TD"), ("q_mean", "mean Q"), ("alpha", "κ_θ")):
            per = []
            for d in run_dirs:
                p = os.path.join(d, "curve.csv")
                if os.path.isfile(p):
                    v = [float(r[k]) for r in csv.DictReader(open(p)) if a <= int(r["episode"]) <= b
                         and r.get(k) not in (None, "", "nan")]
                    v = [x for x in v if not math.isnan(x)]
                    if v:
                        per.append(np.mean(v))
            row[col] = (f"{np.mean(per):.4f} ± {np.std(per, ddof=1):.4f}" if len(per) > 1
                        else (f"{per[0]:.4f}" if per else "-"))
        out.append(row)
    return out


def report(args):
    from scipy import stats as st
    from rewards import ARMS, DEFAULT_ARM
    from tables import Data
    sc = scenario_json(args.scenario)
    cell = Cell(D.BASE_CELL.lam, D.BASE_CELL.M, D.BASE_CELL.K, scenario=sc)
    Db = Data(args.out)
    star = Db.select(cell).get("bl6_star", {"arrival": 1.0, "bottleneck_first": 1.0})
    rows, diag_md, fig6, curves = [], [], [], []
    seed_means = {}
    for arm in args.arms:
        root = _find_root(args.stage, arm, args.runs, args.rl_root)
        if root is None:
            print(f"{arm}: 결과 폴더 없음, 건너뜀")
            continue
        Dr = Data(root)
        bf = "@bf" if ARMS[arm].tote_order == "bottleneck_first" else ""
        refs = {"BL6": policy_label("BL6" + bf), "BL6*": policy_label(cd_key(star["bottleneck_first" if bf else "arrival"], bf)),
                "BL8": policy_label("BL8" + bf)}
        pr = Dr.get(cell, arm, "avg_flow_time")
        if not pr:
            print(f"{arm}: {root}에 {cell} 테스트 결과 없음")
            continue
        seed_means[arm] = defaultdict(list)
        for r in pr:
            seed_means[arm][r["train_seed"]].append(r["y"])
        row = {"arm": arm, "root": root, "n_seeds": len({r["train_seed"] for r in pr}),
               "flow": float(np.mean([r["y"] for r in pr]))}
        for name, lab in refs.items():
            rr = Db.get(cell, lab, "avg_flow_time")
            res = A.seed_level_compare(pr, rr) if rr else None
            if res:
                row[f"vs {name}"] = f"{res['diff']:+.2f} [{res['ci_t_lo']:+.2f}, {res['ci_t_hi']:+.2f}]"
                row[f"wins {name}"] = f"{res['n_better']}/{res['n_train']}"
        recs = Dr.idx.get((*Dr.key(cell), arm), [])
        mean = lambda k: float(np.nanmean([float(r["rec"].get(k, "nan") or "nan") for r in recs]))
        for k in ("dist_per_order", "flow_p99", "avg_totes_per_trip", "mid_trip_rate", "idle_at_decision"):
            row[k] = mean(k)
        rows.append(row)
        fig6.append({"arm": arm, "agree BL6(c_d=1)": mean("bl6_agree"), "agree BL6(c_d=0)": mean("agree_cd0"),
                     "agree BL8": mean("agree_bl8"), "c_d 재현 가능": mean("cd_reproducible"),
                     "어떤 c_d로도 재현 안 됨": 1 - mean("cd_reproducible")})
        run_dirs = sorted(glob.glob(os.path.join(cell.dir(root), f"{arm}_s[0-9]*")))
        run_dirs = [d for d in run_dirs if os.path.basename(d)[len(arm) + 2:].isdigit()]
        diag_md += [f"\n### {arm}\n", _md_table(_phase_stats(run_dirs), ["phase", "loss", "abs TD", "mean Q", "κ_θ"])]
        by_ep = defaultdict(list)
        for d in run_dirs:
            p = os.path.join(d, "val.csv")
            if os.path.isfile(p):
                for r in csv.DictReader(open(p)):
                    by_ep[int(r["episode"])].append(float(r["avg_flow_time"]))
        curves += [{"arm": arm, "episode": e, "val_flow": float(np.mean(v)), "n": len(v)} for e, v in sorted(by_ep.items())]
    if DEFAULT_ARM in seed_means:      # 현행 arm과 Welch t (학습 조건이 다른 그룹)
        base = [np.mean(v) for v in seed_means[DEFAULT_ARM].values()]
        for row in rows:
            x = [np.mean(v) for v in seed_means[row["arm"]].values()]
            if row["arm"] != DEFAULT_ARM and len(x) > 1 and len(base) > 1:
                t = st.ttest_ind(x, base, equal_var=False)
                row["vs 현행 (Welch)"] = f"{np.mean(x) - np.mean(base):+.2f}, p={t.pvalue:.3f}"
    rdir = os.path.join(args.results, args.stage)
    for name, rs in (("rl_vs_baselines", rows), ("fig6_decisions", fig6), ("val_curves", curves)):
        A.save_csv(rs, os.path.join(rdir, f"{name}.csv"))
    cols = ["arm", "n_seeds", "flow", "vs BL6", "wins BL6", "vs BL6*", "wins BL6*", "vs BL8", "wins BL8", "vs 현행 (Welch)",
            "dist_per_order", "flow_p99", "avg_totes_per_trip", "mid_trip_rate", "idle_at_decision"]
    fmt = {k: ".2f" for k in ("flow", "dist_per_order", "flow_p99", "avg_totes_per_trip", "idle_at_decision")}
    fmt["mid_trip_rate"] = ".3f"
    f6 = ["arm", "agree BL6(c_d=1)", "agree BL6(c_d=0)", "agree BL8", "c_d 재현 가능", "어떤 c_d로도 재현 안 됨"]
    lines = [f"# {args.stage} — RL arm 비교 ({cell})\n",
             "- 차이 = RL − 기준 평균 흐름시간 (s), 학습 시드 단위 t 95% CI, wins = 기준보다 나은 시드 수",
             f"- 기준의 tote_order는 arm과 같음, BL6* c_d = {star}",
             "\n## 성능\n", _md_table(rows, cols, fmt), "\n## 학습 진단 (Table 6 형식)", *diag_md,
             "\n## 결정 분류 (그림 6 재분석, 테스트 결정 비율)\n", _md_table(fig6, f6, {k: ".3f" for k in f6[1:]}),
             "\n검증 곡선: val_curves.csv"]
    os.makedirs(rdir, exist_ok=True)
    with open(os.path.join(rdir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"저장: {rdir}/summary.md")


def main():
    p = argparse.ArgumentParser(description="단계별 실행·보고", formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("what", choices=["stage0", "train", "report"])
    p.add_argument("--stage", default="stage0", help="단계 이름 (train/report: stage1 … stage5)")
    p.add_argument("--arms", nargs="+", default=[D.BASE_ARM])
    p.add_argument("--mode", choices=list(MODES), default="screen")
    p.add_argument("--cells", choices=["all", "main", "base"], default="all", help="stage0 셀")
    p.add_argument("--scenario", choices=list(SCENARIOS), default=None, help="단계 5 시나리오")
    p.add_argument("--out", default=os.path.join(REPO, "runs", "stage0"), help="베이스라인 결과 폴더 (stage0)")
    p.add_argument("--runs", default=os.path.join(REPO, "runs"), help="학습 결과 상위 폴더")
    p.add_argument("--rl-root", default=os.path.join(REPO, "runs", "rl_current"),
                   help="현행 RL 결과 폴더 (runner.py main/fleet/factorial --out)")
    p.add_argument("--results", default=os.path.join(REPO, "results"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--jobs", type=int, default=5, help="동시 학습 프로세스 수 (시드 병렬)")
    p.add_argument("--sim-time", type=float, default=SIM_TIME)
    p.add_argument("--n-test", type=int, default=D.N_TEST)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    for k in ("out", "runs", "rl_root", "results"):
        setattr(args, k, os.path.abspath(getattr(args, k)))
    if args.scenario and args.out == os.path.abspath(os.path.join(REPO, "runs", "stage0")):
        args.out = os.path.join(REPO, "runs", f"stage5_{args.scenario}_baselines")
    {"stage0": stage0, "train": train, "report": report}[args.what](args)


if __name__ == "__main__":
    main()
