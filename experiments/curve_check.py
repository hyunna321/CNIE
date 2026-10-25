"""학습 안정성 점검: run 폴더의 curve.csv / val.csv를 구간별로 요약

사용법:
    python experiments/curve_check.py <run 폴더> [<run 폴더> ...] [--window 50]
    예: python experiments/curve_check.py logs_exp/grid_free_reserve/lam320_M14_K3/local_r8_ddqn_v2_s1_pilot

출력 (구간 평균): λ, 흐름시간, 결정당 국소 비용 (μ와 무관), 보상, loss, |TD|, Q, α
    + 검증 점수 (val.csv, ε = 0, 고정 검증 시드)와 같은 셀의 BL6 검증 평균 (baselines/val.csv)
판정 기준 (파일럿): 커리큘럼 종료(전체의 절반) 직후 loss/|TD| 계단 없음, α가 상한(1)에 붙지 않음,
    뒤반부 국소 비용이 증가하지 않음
"""
import argparse
import csv
import math
import os


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(run_dir: str, window: int):
    rows = list(csv.DictReader(open(os.path.join(run_dir, "curve.csv"))))
    print(f"\n{run_dir}  ({len(rows)} 에피소드)")
    cols = [("lam", "λ/hr", "{:6.0f}"), ("avg_flow_time", "flow", "{:6.1f}"), ("local_cost", "cost", "{:6.1f}"),
            ("reward", "reward", "{:+7.3f}"), ("loss", "loss", "{:7.3f}"), ("td_abs", "|TD|", "{:6.3f}"),
            ("q_mean", "Q", "{:+7.3f}"), ("alpha", "α", "{:6.3f}")]
    print("  에피소드   " + " ".join(f"{h:>7s}" for _, h, _ in cols))
    for i in range(0, len(rows), window):
        part = rows[i:i + window]
        vals = [fmt.format(_mean([_f(r.get(k)) for r in part])) for k, _, fmt in cols]
        print(f"  {i + 1:4d}-{i + len(part):4d}  " + " ".join(f"{v:>7s}" for v in vals))

    val_path = os.path.join(run_dir, "val.csv")
    if os.path.isfile(val_path):
        vs = {}
        for r in csv.DictReader(open(val_path)):
            vs.setdefault(int(r["episode"]), []).append(_f(r["avg_flow_time"]))
        bl_path = os.path.join(os.path.dirname(run_dir), "baselines", "val.csv")
        n_val = len(next(iter(vs.values()))) if vs else 0
        bl = float("nan")
        if os.path.isfile(bl_path):
            # 같은 검증 시드 (20000 + i, i < n_val)의 BL6 평균
            bl = _mean([_f(r["avg_flow_time"]) for r in csv.DictReader(open(bl_path))
                        if r["label"] == "BL6_MarginalCost" and 0 <= int(r["seed"]) - 20000 < n_val])
        print(f"  검증 (ε = 0, 시드 {n_val}개, BL6 같은 시드 평균 {bl:.1f} s): " +
              "  ".join(f"ep{e}={_mean(v):.1f}" for e, v in sorted(vs.items())))


def main():
    p = argparse.ArgumentParser(description="학습 곡선 구간 요약")
    p.add_argument("runs", nargs="+")
    p.add_argument("--window", type=int, default=50)
    args = p.parse_args()
    for d in args.runs:
        summarize(d, args.window)


if __name__ == "__main__":
    main()
