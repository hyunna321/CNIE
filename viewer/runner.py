"""뷰어 설정으로 그리드 시뮬레이션을 한 번 실행하고 기록 반환"""
import glob
import os
import subprocess
import time
from typing import Dict, List, Optional

from env.orders import DEFAULT_SEED, SIM_TIME
from env.layout import ROBOT_CAPACITY
from env.layout import GridWarehouseLayout, RACK, AISLE, CROSS, WS_ZONE
from env.orders import Scenario, ARRIVALS, SKU_DISTS, STORAGES, TIME_DISTS
from evaluation import POLICIES as EVAL_POLICIES, make_sim
from viewer.recorder import Recorder

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 화면 선택지: 정책 키 -> 표시 이름 (evaluation.POLICIES + RL 체크포인트)
POLICIES = {k: label for k, (label, _) in EVAL_POLICIES.items()} | {"RL": "RL (checkpoint)"}


def find_checkpoints() -> List[str]:
    """저장소 안의 체크포인트(*.pt, 이어하기 파일 제외) 경로 (가상환경 제외, 최근 수정 순)"""
    paths = [p for p in glob.glob(os.path.join(REPO, "**", "*.pt"), recursive=True)
             if "/.venv/" not in p and "/site-packages/" not in p and os.path.basename(p) != "resume.pt"]
    paths.sort(key=os.path.getmtime, reverse=True)
    return [os.path.relpath(p, REPO) for p in paths]


def read_checkpoint_meta(path: str) -> dict:
    """체크포인트의 용량/로봇 수/WS 수와 학습 조건(meta)"""
    from agents import read_checkpoint
    return read_checkpoint(_abs(path))


def meta() -> dict:
    """뷰어 설정 화면에 필요한 선택지와 기본값"""
    return {
        "policies": POLICIES,
        "scenario_defaults": Scenario().to_dict(),
        "enums": {"arrival": ARRIVALS, "sku_dist": SKU_DISTS, "storage": STORAGES,
                  "pick_dist": TIME_DISTS, "ws_proc_dist": TIME_DISTS,
                  "traffic": ["free", "lanes"], "motion": ["reserve", "reactive"]},
        "checkpoints": find_checkpoints(),
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
    if policy == "RL":
        sim = _rl_simulator(cfg, L, sc, M, K, cap, lam, seed, T, warnings)
    else:
        sim = make_sim(policy, L, lam, seed, T)
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


def _rl_simulator(cfg: Dict, L, sc: Scenario, M: int, K: int, cap: int, lam: float, seed: int, T: float,
                  warnings: List[str]):
    """체크포인트를 불러와 ε = 0 RLSimulator 생성 (보상 설정은 체크포인트 arm을 따름)

    BL6 prior 척도는 지금 레이아웃 기준으로 다시 맞춤 (run_compare --load 기본 동작과 같음)
    """
    from agents import DQNAgent
    from env.rl import ARMS
    path = cfg.get("checkpoint")
    if not path:
        raise ValueError("RL 정책은 체크포인트 경로가 필요합니다")
    info = read_checkpoint_meta(path)
    if info["robot_capacity"] != cap:
        raise ValueError(f"체크포인트 로봇 용량 {info['robot_capacity']} ≠ 설정 {cap} (특징 차원이 달라 불러올 수 없습니다)")
    if info["num_workstations"] != K:
        raise ValueError(f"체크포인트 WS 수 {info['num_workstations']} ≠ 설정 {K} (전역 상태 차원이 달라 불러올 수 없습니다)")
    m = info.get("meta") or {}
    arm = ARMS.get(m.get("arm"))
    if arm is None:
        raise ValueError(f"체크포인트 arm '{m.get('arm')}'은 지금 코드의 env.rl.ARMS에 없습니다")
    agent = DQNAgent(robot_capacity=cap, dist_norm=L.dist_norm, num_robots=M, num_workstations=K,
                     sim_time=info.get("sim_time") or SIM_TIME, n_step=arm.n_step, features=arm.features)
    try:
        agent.load(_abs(path))
    except RuntimeError:
        raise ValueError("체크포인트 네트워크 구조가 지금 코드와 다릅니다 (이전 버전 체크포인트일 수 있음)")
    agent.set_mc_scale(L.dist_norm)
    warnings += _ckpt_warnings(m, cfg, sc, M, K)
    return make_sim("RL", L, lam, seed, T, agent=agent, arm=arm)


def _ckpt_warnings(m: dict, cfg: dict, sc: Scenario, M: int, K: int) -> List[str]:
    """체크포인트 학습 조건과 지금 설정이 다르면 경고"""
    if not m:
        return ["학습 조건 기록이 없는 체크포인트입니다."]
    out = []
    if (m.get("traffic"), m.get("motion")) != (cfg.get("traffic", "free"), cfg.get("motion", "reserve")):
        out.append(f"학습 이동 규칙 {m.get('traffic')}/{m.get('motion')}과 지금 설정이 다릅니다.")
    train_Ms = m.get("train_robots") or [m.get("num_robots")]
    if M not in train_Ms or m.get("num_workstations") != K:
        out.append(f"학습 조건 M={'/'.join(map(str, train_Ms))}, K={m.get('num_workstations')}와 지금 설정이 다릅니다.")
    if m.get("lam") and abs(m["lam"] * 3600 - float(cfg.get("lam_hr", 320))) > 1:
        out.append(f"학습 λ={m['lam'] * 3600:.0f}/hr와 지금 λ가 다릅니다.")
    if m.get("scenario") and m["scenario"] != sc.to_dict():
        out.append("학습 시나리오(분포)와 지금 시나리오가 다릅니다.")
    return out


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO, path)


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
