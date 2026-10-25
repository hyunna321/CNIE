"""부록용 레이아웃 정보: 치수, 도킹 셀 좌표, 대기열/주차 칸 수, 정규화 기준 (dist_norm, legacy D_max)

사용법:
    python experiments/layout_info.py                      # M = 14, K ∈ {2, 3, 4}, 그리고 E7/E8의 M
    python experiments/layout_info.py --M 6 14 18 --K 3 --csv logs_exp/layout_info.csv

D_max (legacy 보상의 detour 정규화 기준) = 2 × (피킹 셀·도킹 셀 사이 최대 거리), 논문 표 값(--paper-dmax, 기본 218.0)과 비교
"""
import argparse
import csv
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def info(M: int, K: int) -> dict:
    from evaluation import Condition
    from constant import CELL_SIZE
    L = Condition().layout(M, K)
    return {"M": M, "K": K, "n_cols": L.n_cols, "n_rows": L.n_rows,
            "width_m": L.n_cols * CELL_SIZE, "depth_m": L.n_rows * CELL_SIZE,
            "ws_rows": L.ws_rows, "docks": " ".join(f"({x},{y})" for x, y in L.docks),
            "queue_cells_per_ws": L.queue_len, "park_spots_per_ws": len(L.park_spots[0]),
            "park_lanes": L.n_lanes, "lot_width": L.lot_w,
            "n_pick_cells": len(set(L.access_coords)), "n_storage": len(L.access_coords),
            "dist_norm": L.dist_norm, "coord_norm": L.coord_norm, "reward_d_max": L.reward_d_max,
            "solo_rt_m": round(L.solo_rt_m, 2)}


def main():
    p = argparse.ArgumentParser(description="레이아웃 정보 (부록)")
    p.add_argument("--M", type=int, nargs="+", default=[6, 8, 10, 12, 14, 16, 18, 20])
    p.add_argument("--K", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--paper-dmax", type=float, default=218.0, help="논문 표의 D_max (M = 14, K = 3)")
    p.add_argument("--csv", default=None)
    args = p.parse_args()
    rows = [info(M, K) for K in args.K for M in args.M]
    keys = list(rows[0])
    print("  ".join(keys))
    for r in rows:
        print("  ".join(str(r[k]) for k in keys))
    base = next((r for r in rows if r["M"] == 14 and r["K"] == 3), None) or info(14, 3)
    same = abs(base["reward_d_max"] - args.paper_dmax) < 1e-9
    print(f"\nD_max (M = 14, K = 3) = {base['reward_d_max']:.1f}, 논문 표 {args.paper_dmax:.1f} → "
          + ("일치" if same else "불일치 (논문 표 수정 필요)"))
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
