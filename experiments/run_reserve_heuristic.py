"""로봇별 다음 트립 예약(…R) vs 공용 FIFO 큐 비교, 학습 없음

python -m experiments.run_reserve_heuristic --workers 44                       # BL6 c_d 격자 × 셀 5개
python -m experiments.run_reserve_heuristic --pairs BL3 BL4 BL5 BL6c1 --cells 10,3,240 12,3,280 14,3,320 \
    --out runs/reserve_all --results results/reserve_all                          # 베이스라인별
  기준 키와 예약 키의 쌍: BL6c{c} ↔ BL6Rc{c}, BL3 ↔ BL3R 처럼 R을 붙인 키. 같은 셀·시드끼리 대응 비교
결과: {out}/eval.csv (에피소드별), {results}/summary.csv (셀·정책별 평균, 기준 대비 차이)
"""
import argparse
import math
import os
import statistics as st
import time
from multiprocessing import Pool

from baselines import resolve_policy
from env.battery import BatteryConfig
from evaluation import Condition, test_seeds, write_csv

CELLS = [(20, 3, 320), (14, 3, 280), (14, 3, 320), (12, 3, 280), (10, 3, 240)]   # (M, K, λ orders/hr)
PAIRS = ["BL6c0", "BL6c0.25", "BL6c1"]
METRICS = ["avg_flow_time", "flow_p99", "avg_wait_time", "dist_per_order", "avg_totes_per_trip", "completion_rate",
           "throughput_hr", "queued_frac", "reserved_frac", "fifo_frac", "utilization", "blocked_time",
           "charge_h_per_robot", "n_charge_required", "n_soc_depleted"]
DIFFS = ("avg_flow_time", "flow_p99", "avg_wait_time", "dist_per_order", "avg_totes_per_trip")


def reserve_key(base: str) -> str:
    """기준 키의 예약 버전 (BL6c0.25 → BL6Rc0.25, BL4 → BL4R)"""
    return base.replace("BL6c", "BL6Rc") if base.startswith("BL6c") else base + "R"


def one(job):
    M, K, lam, key, base, seed, sim_time, bat = job
    L = Condition().layout(M, K)
    wall = time.perf_counter()
    sim = resolve_policy(key)[1](L, lam=lam / 3600, seed=seed, sim_time=sim_time, num_robots=M,
                                 robot_capacity=L.robot_capacity, battery=BatteryConfig() if bat else None)
    sim.reset(seed)
    sim.run()
    r = sim.build_result(key, 0, lam / 3600)
    n = max(1, r.n_totes)
    extra = {"battery": bat, "charge_h_per_robot": r.charge_time / 3600 / M, "n_charge_required": r.n_charge_required,
             "min_soc": r.min_soc, "n_soc_depleted": r.n_soc_depleted}
    return {**extra, "M": M, "K": K, "lam_hr": lam, "base": base, "policy": key, "reserve": key != base, "seed": seed,
            "avg_flow_time": r.avg_flow_time, "flow_p99": r.flow_p99, "avg_wait_time": r.avg_wait_time,
            "dist_per_order": r.dist_per_order, "avg_totes_per_trip": r.avg_totes_per_trip,
            "completion_rate": r.completion_rate, "throughput_hr": r.throughput * 3600,
            "queued_frac": r.n_queued_totes / n, "reserved_frac": getattr(sim, "n_reserved", 0) / n,
            "fifo_frac": getattr(sim, "n_fifo", r.n_queued_totes) / n, "utilization": r.utilization,
            "blocked_time": r.blocked_time, "wall_s": time.perf_counter() - wall}


def summarize(rows):
    out, groups = [], {}
    for r in rows:
        groups.setdefault((r["battery"], r["M"], r["K"], r["lam_hr"], r["base"]), {}).setdefault(r["reserve"], {})[r["seed"]] = r
    for (bat, M, K, lam, base), pol in sorted(groups.items()):
        for rv, by in sorted(pol.items()):
            row = {"battery": bat, "M": M, "K": K, "lam_hr": lam, "base": base,
                   "policy": next(iter(by.values()))["policy"], "reserve": rv, "n": len(by)}
            row.update({m: round(sum(float(x[m]) for x in by.values()) / len(by), 4) for m in METRICS})
            if rv and False in pol:
                ref = pol[False]
                for m in DIFFS:
                    d = [float(by[s][m]) - float(ref[s][m]) for s in by if s in ref]
                    row[f"d_{m}"] = round(sum(d) / len(d), 3)
                    row[f"d_{m}_se"] = round(st.stdev(d) / math.sqrt(len(d)), 3) if len(d) > 1 else float("nan")
            out.append(row)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--sim-time", type=float, default=36000)
    p.add_argument("--pairs", nargs="+", default=PAIRS, help="기준 정책 키 (예약 버전은 reserve_key)")
    p.add_argument("--cells", nargs="+", default=None, help="M,K,λ 목록 (예: 14,3,320)")
    p.add_argument("--battery", nargs="+", choices=["off", "on"], default=["off"],
                   help="배터리 제약 (on = env.battery.BatteryConfig 기본값), 둘 다 주면 둘 다 실행")
    p.add_argument("--out", default="runs/reserve_heuristic")
    p.add_argument("--results", default="results/reserve_heuristic")
    a = p.parse_args()
    cells = [tuple(int(x) for x in c.split(",")) for c in a.cells] if a.cells else CELLS
    jobs = [(M, K, lam, key, base, s, a.sim_time, b == "on") for b in a.battery for (M, K, lam) in cells
            for base in a.pairs for key in (base, reserve_key(base)) for s in test_seeds(a.n_seeds)]
    wall = time.perf_counter()
    with Pool(a.workers) as pool:
        rows = pool.map(one, jobs, chunksize=1)
    write_csv(rows, os.path.join(a.out, "eval.csv"))
    write_csv(summarize(rows), os.path.join(a.results, "summary.csv"))
    print(f"{len(rows)} runs, {time.perf_counter() - wall:.0f} s, 에피소드 평균 {sum(r['wall_s'] for r in rows) / len(rows):.1f} s")


if __name__ == "__main__":
    main()
