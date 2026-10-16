"""뷰어 설정으로 그리드 시뮬레이션을 한 번 실행하고 기록 반환"""
import os
import subprocess
import time
from typing import Dict, Optional

from constant import DEFAULT_SEED, ROBOT_CAPACITY
from grid import GridWarehouseLayout, RACK, AISLE, CROSS, WS_ZONE
from scenario import Scenario, ARRIVALS, SKU_DISTS, STORAGES, TIME_DISTS
from viewer.recorder import Recorder

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

POLICIES = {
    "BL1": "BL1 Full-Buffer",
    "BL2": "BL2 Partial-Buffer",
    "BL3": "BL3 Random",
    "BL4": "BL4 Nearest",
    "BL5": "BL5 Min-Insertion",
    "BL6": "BL6 Marginal-Cost",
}


def meta() -> dict:
    """뷰어 설정 화면에 필요한 선택지와 기본값"""
    return {
        "policies": POLICIES,
        "scenario_defaults": Scenario().to_dict(),
        "enums": {"arrival": ARRIVALS, "sku_dist": SKU_DISTS, "storage": STORAGES,
                  "pick_dist": TIME_DISTS, "ws_proc_dist": TIME_DISTS,
                  "traffic": ["free", "lanes"], "motion": ["reserve", "reactive"]},
        "git": _git_rev(),
    }


def layout_info(L: GridWarehouseLayout) -> dict:
    """정적 지도: 셀 종류, 셀 인덱스 -> 좌표, 워크스테이션 구역"""
    kinds = []
    for y in range(L.n_rows):
        row = []
        for x in range(L.n_cols):
            k = int(L.kind[x, y])
            row.append(k if k in (RACK, AISLE, CROSS, WS_ZONE) else -1)
        kinds.append(row)
    pick = sorted({L.cell_index[c] for c in L.access_coords})
    return {
        "n_cols": L.n_cols, "n_rows": L.n_rows, "kinds": kinds,
        "cells": [list(c) for c in L.cells],
        "docks": [list(d) for d in L.docks],
        "queues": [[list(c) for c in q] for q in L.queue_cells],
        "spots": [[list(c) for c in s] for s in L.park_spots],
        "lanes": [[list(c) for lane in ls for c in lane] for ls in L.lot_lanes],
        "pick_cells": pick, "ws_rows": L.ws_rows, "traffic": L.traffic, "motion": L.motion,
        "col_dir": [int(v) for v in L.col_dir], "row_dir": [int(v) for v in L.row_dir],
    }


def run(cfg: Dict) -> dict:
    """뷰어 설정 cfg로 시뮬레이션을 돌리고 지도, 기록, 요약 지표 반환"""
    from baselines import (FullBufferSimulator, PartialSimulator, RandomAssignSimulator,
                           NearestRobotSimulator, MinInsertionSimulator, MarginalCostSimulator)
    t_wall = time.time()
    cfg = {"policy": "BL6", "num_robots": 14, "num_workstations": 3, "robot_capacity": ROBOT_CAPACITY,
           "lam_hr": 320, "sim_time": 3600, "seed": 10000, "traffic": "free", "motion": "reserve",
           "scenario": {}, **cfg}
    M, K   = int(cfg.get("num_robots", 14)), int(cfg.get("num_workstations", 3))
    cap    = int(cfg.get("robot_capacity", ROBOT_CAPACITY))
    lam    = float(cfg.get("lam_hr", 320)) / 3600.0
    T      = float(cfg.get("sim_time", 3600))
    seed   = int(cfg.get("seed", 10000))
    sc     = _scenario_from(cfg.get("scenario") or {})
    L = GridWarehouseLayout(seed=DEFAULT_SEED, num_robots=M, robot_capacity=cap, num_workstations=K,
                            traffic=cfg.get("traffic", "free"), motion=cfg.get("motion", "reserve"),
                            scenario=sc)
    policy   = cfg.get("policy", "BL6")
    warnings = []
    kw = dict(lam=lam, seed=seed, sim_time=T, num_robots=M, robot_capacity=cap)
    cls = {"BL1": FullBufferSimulator, "BL2": PartialSimulator, "BL3": RandomAssignSimulator,
           "BL4": NearestRobotSimulator, "BL5": MinInsertionSimulator, "BL6": MarginalCostSimulator}[policy]
    sim = cls(L, **kw)
    rec = Recorder()
    sim.recorder = rec
    sim.reset(seed)
    sim.run()
    r = sim.build_result(policy, 0, lam)
    summary = {k: v for k, v in vars(r).items() if isinstance(v, (int, float))}
    summary["throughput_per_h"] = r.n_completed / (T / 3600.0)
    if r.n_plan_fail:
        warnings.append(f"경로계획 실패 {r.n_plan_fail}회 (재시도로 처리됨). 혼잡이 심하면 대기가 탐색 범위를 넘습니다.")
    return {
        "config": {**cfg, "scenario": sc.to_dict()}, "layout": layout_info(L),
        "trace": rec.result(sim), "summary": summary, "warnings": warnings,
        "git": _git_rev(), "wall_s": round(time.time() - t_wall, 2),
    }


def _scenario_from(d: dict) -> Scenario:
    """화면 입력을 Scenario로 변환 (빈 칸(None)은 기본값, 모르는 키는 무시)"""
    from dataclasses import fields
    names = {f.name for f in fields(Scenario)}
    clean = {k: v for k, v in d.items() if k in names and v is not None}
    try:
        return Scenario(**clean)
    except (TypeError, ValueError, AssertionError) as e:
        raise ValueError(f"시나리오 설정이 올바르지 않습니다: {e}")


def _git_rev() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None
