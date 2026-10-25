"""NNH vs exact 경로 비교 (오프라인, 리뷰 R3-6, 논문 tab:routing-gap)

입력: run_rl.py --log-reconstructions PATH가 남긴 JSONL (.gz 가능). 한 줄 = NNH 경로 구성 호출 1번
      {"seed", "i", "kind" ("start" 트립 출발 / "reroute" 운행 중 재구성), "start", "picks", "docks",
       "dist_nnh" (m), "ms_nnh"}
처리: 같은 입력(시작 노드 고정)에서 피킹 셀 순열(≤ 5!, 같은 셀 중복은 순열 중복 제거) × 도킹 셀 순열(≤ K!)을
      모두 열거해 D(ρ) = d(start, ρ_1) + Σ d(ρ_i, ρ_{i+1}) (피킹 → 도킹, 논문 Eq. routedist) 최소값과 계산 시간
      (environment.plan_trip_exact). 거리 차이 = (D_NNH − D_exact) / D_exact (%)
      ms_nnh는 시뮬레이션 중 잰 값, ms_exact는 이 스크립트에서 잰 값 (같은 PC에서 비교할 것)
      closed-loop 결과에는 영향 없음 (기록 입력은 NNH로 굴러간 궤적의 것)
출력: --out 접두어로 _calls.csv (호출별), _summary.csv (파일 × 종류 × 피킹 수, 그리고 전체)
      요약 열: n_calls, n_calls_logged, gap_pct_mean/p95, gap_pct_total (거리 합 기준), frac_nnh_optimal,
              ms_nnh_mean/p95, ms_exact_mean/p95

사용법:
    python experiments/route_gap.py logs_exp/.../recon_s1.jsonl.gz ... --M 14 --K 3 --out logs_exp/.../tab_routing_gap
    --max-calls N : 파일당 무작위 표본 N개 (0이면 전부, 표본 수는 n_calls / n_calls_logged로 남음)
"""
import argparse
import csv
import gzip
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _read(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _q(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(math.ceil(q * len(s))) - 1)]


def _summ(rows, **key):
    g = [r["gap_pct"] for r in rows]
    dn, de = sum(r["dist_nnh"] for r in rows), sum(r["dist_exact"] for r in rows)
    return {**key, "n_calls": len(rows),
            "gap_pct_mean": sum(g) / len(g), "gap_pct_p95": _q(g, 0.95),
            "gap_pct_total": 100.0 * (dn - de) / de if de > 0 else 0.0,
            "frac_nnh_optimal": sum(x < 1e-9 for x in g) / len(g),
            "ms_nnh_mean": sum(r["ms_nnh"] for r in rows) / len(rows), "ms_nnh_p95": _q([r["ms_nnh"] for r in rows], 0.95),
            "ms_exact_mean": sum(r["ms_exact"] for r in rows) / len(rows),
            "ms_exact_p95": _q([r["ms_exact"] for r in rows], 0.95)}


def main():
    p = argparse.ArgumentParser(description="NNH vs exact 경로 비교 (오프라인)")
    p.add_argument("logs", nargs="+", help="--log-reconstructions JSONL 파일")
    p.add_argument("--M", type=int, default=14, help="기록한 평가 레이아웃의 로봇 수")
    p.add_argument("--K", type=int, default=3, help="워크스테이션 수")
    p.add_argument("--traffic", default="free")
    p.add_argument("--motion", default="reserve")
    p.add_argument("--max-calls", type=int, default=0, help="파일당 무작위 표본 수 (0이면 전부)")
    p.add_argument("--kinds", nargs="*", default=["start", "reroute"], help="포함할 호출 종류")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="출력 접두어 (_calls.csv, _summary.csv)")
    args = p.parse_args()

    from evaluation import Condition
    from environment import plan_trip_exact
    L = Condition(traffic=args.traffic, motion=args.motion).layout(args.M, args.K)
    d = L.aisle_distance
    rng = random.Random(args.seed)
    calls, summ = [], []
    for path in args.logs:
        recs = [r for r in _read(path) if r["kind"] in args.kinds]
        n_logged = len(recs)
        if args.max_calls and n_logged > args.max_calls:
            recs = rng.sample(recs, args.max_calls)
        rows = []
        for r in recs:
            start = tuple(r["start"])
            picks = [tuple(c) for c in r["picks"]]
            docks = [tuple(c) for c in r["docks"]]
            t0 = time.perf_counter()
            ex = plan_trip_exact(start, picks, docks, d)[2]
            ms = 1e3 * (time.perf_counter() - t0)
            if ex > r["dist_nnh"] + 1e-6:
                raise RuntimeError(f"exact가 NNH보다 김: {path} {r}")
            rows.append({"file": os.path.basename(path), "seed": r["seed"], "i": r["i"], "kind": r["kind"],
                         "n_picks": len(picks), "n_docks": len(docks), "dist_nnh": r["dist_nnh"], "dist_exact": ex,
                         "gap_pct": 100.0 * (r["dist_nnh"] - ex) / ex if ex > 0 else 0.0,
                         "ms_nnh": r["ms_nnh"], "ms_exact": ms})
        calls += rows
        groups = defaultdict(list)
        for r in rows:
            groups[(r["kind"], r["n_picks"])].append(r)
            groups[(r["kind"], "all")].append(r)
        for (kind, npk), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
            summ.append(_summ(rs, file=os.path.basename(path), kind=kind, n_picks=npk, n_calls_logged=n_logged))
        print(f"{path}: 기록 {n_logged}개 중 {len(rows)}개 비교")
    for kind in sorted({r["kind"] for r in calls}) + ["all"]:
        rs = [r for r in calls if kind == "all" or r["kind"] == kind]
        s = _summ(rs, file="ALL", kind=kind, n_picks="all", n_calls_logged=float("nan"))
        summ.append(s)
        print(f"  {kind:8s} n={s['n_calls']:7d}  거리 차이 평균 {s['gap_pct_mean']:.2f}% (p95 {s['gap_pct_p95']:.2f}%, "
              f"합계 {s['gap_pct_total']:.2f}%, NNH 최적 {s['frac_nnh_optimal']:.1%})  "
              f"시간 NNH {s['ms_nnh_mean']:.3f} ms (p95 {s['ms_nnh_p95']:.3f}) vs exact {s['ms_exact_mean']:.3f} ms "
              f"(p95 {s['ms_exact_p95']:.3f})")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for suffix, rows in (("_calls.csv", calls), ("_summary.csv", summ)):
        with open(args.out + suffix, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"저장: {args.out}_summary.csv, {args.out}_calls.csv")


if __name__ == "__main__":
    main()
