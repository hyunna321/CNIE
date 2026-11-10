"""부록용 레이아웃 정보: 치수, 도킹 셀 좌표, 대기열/주차 칸 수, 정규화 기준 (dist_norm, legacy D_max)

사용법:
    python -m experiments.check_layout                      # 실험에 쓰는 M (6~20) × K ∈ {2, 3, 4}
    python -m experiments.check_layout --M 6 14 18 --K 3 --csv runs/exp/layout_info.csv

D_max (legacy 보상의 detour 정규화 기준) = 2 × (피킹 셀·도킹 셀 사이 최대 거리)
"""
import argparse
import csv
import os



def info(M: int, K: int) -> dict:
    from evaluation import Condition
    from env.layout import CELL_SIZE
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
    p.add_argument("--csv", default=None)
    args = p.parse_args()
    rows = [info(M, K) for K in args.K for M in args.M]
    keys = list(rows[0])
    print("  ".join(keys))
    for r in rows:
        print("  ".join(str(r[k]) for k in keys))
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
