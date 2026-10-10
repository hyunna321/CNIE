"""실험 실행 진입점.

사용법:
    python run.py                  # λ 그리드 실험 (기본값)
    python run.py lambda-grid      # λ = 700/800/900/1000 /hr, M=14, K=3 -> logs/
    python run.py robot-sweep      # λ = 800/hr, K=3, M=6~22   -> logs_experiment/robot_sweep_k3/
    python run.py ws-sweep         # λ = 800/hr, M=14, K=1,3,5 -> logs_experiment/workstation_sweep_m14/
    python run.py all              # 위 세 실험 전부

    --episodes, --n-eval 로 학습 에피소드 수와 평가 시드 수를 바꿀 수 있다.
    --grid 를 주면 셀 그리드 이동 모드(grid.GridWarehouseLayout)로 실행하고, 결과는
    logs_grid_{traffic}_{motion}/, logs_experiment_grid_{traffic}_{motion}/ 에 따로 저장한다.
      --traffic free|lanes     모든 방향 이동 / 일방통행 통로 + 차선 교차통로 (기본 free)
      --motion reserve|reactive 배정 시 시공간 경로 예약 / 막히면 대기·우회 (기본 reserve)
      --no-collisions           reactive에서 로봇 간 점유를 무시 (회귀 검증용)
    --scenario 로 주문 도착/SKU 수요/저장/처리 시간 분포를 JSON 문자열이나 파일로 준다 (scenario.Scenario 필드).
      예: --scenario '{"arrival": "batch", "batch_mean": 6, "sku_dist": "abc", "storage": "class_based"}'
    학습한 에이전트는 결과 폴더에 agent_*.pt 체크포인트로 저장된다 (시각화 도구에서 불러올 수 있다).
"""
import argparse
import json
import os
from typing import Dict, Iterable, List, Optional

from environment import WarehouseLayout
from grid import GridWarehouseLayout
from scenario import Scenario
from agents import DQNAgent
from metrics import MetricsStore, TrainingCurveStore, print_comparison
from training import train, evaluate_all
from constant import (
    GRID_TRAFFIC, GRID_MOTION,
    SIM_TIME, LAMBDA_GRID, DEFAULT_SEED, NUM_ROBOTS, ROBOT_CAPACITY,
    TOTAL_EPISODES, N_EVAL, EVAL_SEED_BASE, OUT_DIR, POLICY_LABELS,
    EXPERIMENT_LAMBDA, EXPERIMENT_OUT_DIR,
    EXPERIMENT_ROBOT_SWEEP, EXPERIMENT_ROBOT_SWEEP_K,
    EXPERIMENT_WS_SWEEP, EXPERIMENT_WS_SWEEP_M,
)


# 레이아웃 선택 (main()에서 --grid, --no-collisions로 설정)
GRID_MODE       = False
GRID_COLLISIONS = True
GRID_OPTS       = {"traffic": GRID_TRAFFIC, "motion": GRID_MOTION}
SCENARIO        = Scenario()


def checkpoint_meta(lam: float, num_robots: int, num_workstations: int,
                    total_episodes: int) -> dict:
    """체크포인트에 함께 저장할 학습 조건."""
    return {
        "lam": lam, "num_robots": num_robots, "num_workstations": num_workstations,
        "robot_capacity": ROBOT_CAPACITY, "episodes": total_episodes,
        "grid": GRID_MODE, "traffic": GRID_OPTS["traffic"] if GRID_MODE else None,
        "motion": GRID_OPTS["motion"] if GRID_MODE else None,
        "scenario": SCENARIO.to_dict(),
    }


def make_layout(num_robots: int = NUM_ROBOTS,
                num_workstations: Optional[int] = None) -> WarehouseLayout:
    """현재 모드에 맞는 레이아웃을 만든다.

    입력:
        num_robots: 로봇 수 M (그리드 모드의 대기열/주차 셀 크기 결정에 사용)
        num_workstations: 워크스테이션 수 K (None이면 기본값)
    출력:
        WarehouseLayout 또는 GridWarehouseLayout
    """
    kw = {} if num_workstations is None else {"num_workstations": num_workstations}
    if GRID_MODE:
        return GridWarehouseLayout(seed=DEFAULT_SEED, num_robots=num_robots,
                                   collisions=GRID_COLLISIONS, scenario=SCENARIO, **GRID_OPTS, **kw)
    return WarehouseLayout(seed=DEFAULT_SEED, scenario=SCENARIO, **kw)


# λ 그리드 실험

def experiment_lambda_grid(
    layout:         WarehouseLayout,
    store:          MetricsStore,
    lam_grid:       List[float] = LAMBDA_GRID,
    seed:           int = DEFAULT_SEED,
    total_episodes: int = TOTAL_EPISODES,
    n_eval:         int = N_EVAL,
    num_robots:     int = NUM_ROBOTS,
    robot_capacity: int = ROBOT_CAPACITY,
    out_dir:        str = OUT_DIR,
    curve_store:    Optional[TrainingCurveStore] = None,
) -> Dict[float, DQNAgent]:
    """λ마다 새 에이전트를 학습하고 RL/베이스라인을 평가해 CSV로 저장한다.

    저장 파일 (out_dir):
        train_lam{λ}_M{M}.csv, eval_lam{λ}_M{M}_n{n_eval}.csv

    입력:
        layout: WarehouseLayout
        store: 모든 λ의 평가 결과를 누적할 MetricsStore
        lam_grid: λ 리스트 (orders/s)
        seed: 에이전트/학습 시드
        total_episodes: λ당 학습 에피소드 수
        n_eval: λ당 평가 시드 수
        num_robots: 로봇 수
        robot_capacity: 로봇 용량
        out_dir: CSV 저장 디렉터리
        curve_store: 학습 곡선 기록용 TrainingCurveStore (선택)
    출력:
        {λ: 학습된 DQNAgent}
    """
    labels     = POLICY_LABELS
    eval_seeds = list(range(EVAL_SEED_BASE, EVAL_SEED_BASE + n_eval))
    agents: Dict[float, DQNAgent] = {}

    os.makedirs(out_dir, exist_ok=True)

    for lam in lam_grid:
        lam_hr = int(round(lam * 3600))
        print(f"\n{'='*60}")
        print(f"λ = {lam_hr}/hr  (M={num_robots} 고정, K={layout.num_workstations})")
        print(f"{'='*60}")

        agent = DQNAgent(num_robots=num_robots, robot_capacity=robot_capacity,
                         seed=seed, sim_time=SIM_TIME,
                         num_workstations=layout.num_workstations)

        tr_store = MetricsStore()
        train(layout, agent, tr_store,
              lam=lam, seed=seed,
              total_episodes=total_episodes,
              num_robots=num_robots, robot_capacity=robot_capacity,
              label=f"RL_lam{lam_hr}_train",
              curve_store=curve_store)
        tr_store.save_csv(os.path.join(out_dir, f"train_lam{lam_hr}_M{num_robots}.csv"))
        agent.save(os.path.join(out_dir, f"agent_lam{lam_hr}_M{num_robots}.pt"),
                   checkpoint_meta(lam, num_robots, layout.num_workstations, total_episodes))
        agents[lam] = agent

        ev_store = MetricsStore()
        evaluate_all(layout, agent, ev_store, lam=lam,
                     num_robots=num_robots, robot_capacity=robot_capacity,
                     eval_seeds=eval_seeds, n_eval=n_eval)
        for r in ev_store.records:
            store.add(r)

        print_comparison(ev_store, labels, lam=lam, num_robots=num_robots,
                         last_n=n_eval, title=f"M={num_robots}  λ={lam_hr}/hr")

        ev_csv = os.path.join(out_dir, f"eval_lam{lam_hr}_M{num_robots}_n{n_eval}.csv")
        ev_store.save_csv(ev_csv)
        print(f"→ 평가 결과 저장: {ev_csv}")

    return agents


def run_lambda_grid(total_episodes: int = TOTAL_EPISODES, n_eval: int = N_EVAL,
                    out_dir: str = OUT_DIR):
    """λ 그리드 실험을 실행하고 전체 평가 결과와 학습 곡선을 out_dir에 저장한다.

    저장 파일: eval_all_lambdas.csv, training_curves_all_lambdas.csv (+ λ별 CSV)

    입력:
        total_episodes: λ당 학습 에피소드 수
        n_eval: λ당 평가 시드 수
        out_dir: CSV 저장 디렉터리
    """
    os.makedirs(out_dir, exist_ok=True)

    layout = make_layout(NUM_ROBOTS)

    print("=" * 60)
    print(f"λ grid experiment  (M={NUM_ROBOTS} 고정, K={layout.num_workstations})")
    print("=" * 60)

    store = MetricsStore()
    curve = TrainingCurveStore()
    experiment_lambda_grid(
        layout, store,
        lam_grid=LAMBDA_GRID,
        seed=DEFAULT_SEED,
        total_episodes=total_episodes,
        n_eval=n_eval,
        num_robots=NUM_ROBOTS,
        robot_capacity=ROBOT_CAPACITY,
        out_dir=out_dir,
        curve_store=curve,
    )
    store.save_csv(os.path.join(out_dir, "eval_all_lambdas.csv"))
    curve.save_csv(os.path.join(out_dir, "training_curves_all_lambdas.csv"))
    print(f"\nDone. CSVs → {out_dir}/")


# 로봇 수 / 워크스테이션 수 스윕 (λ = EXPERIMENT_LAMBDA)

def run_setting(experiment_name: str, num_robots: int, num_workstations: int,
                total_episodes: int = TOTAL_EPISODES, n_eval: int = N_EVAL,
                out_dir: str = EXPERIMENT_OUT_DIR) -> None:
    """(M, K) 조건 하나에서 RL을 학습하고 RL/BL1~BL6를 평가해 저장한다.

    저장 파일 (out_dir/experiment_name/M{m}_K{k}/):
        train.csv, training_curve.csv, evaluation.csv

    입력:
        experiment_name: 실험 이름 (저장 하위 디렉터리)
        num_robots: 로봇 수 M
        num_workstations: 워크스테이션 수 K
        total_episodes: 학습 에피소드 수
        n_eval: 평가 시드 수
        out_dir: 결과 최상위 디렉터리
    """
    lam_hr = int(EXPERIMENT_LAMBDA * 3600)
    setting_name = f"M{num_robots}_K{num_workstations}"
    setting_dir = os.path.join(out_dir, experiment_name, setting_name)
    os.makedirs(setting_dir, exist_ok=True)

    layout = make_layout(num_robots, num_workstations)
    agent = DQNAgent(
        num_robots=num_robots,
        robot_capacity=ROBOT_CAPACITY,
        seed=DEFAULT_SEED,
        sim_time=SIM_TIME,
        num_workstations=num_workstations,
    )

    print("\n" + "=" * 72)
    print(f"{experiment_name}: lambda={lam_hr}/hr, M={num_robots}, K={num_workstations}")
    print("=" * 72)

    train_store = MetricsStore()
    curve_store = TrainingCurveStore()
    train(
        layout, agent, train_store,
        lam=EXPERIMENT_LAMBDA,
        seed=DEFAULT_SEED,
        total_episodes=total_episodes,
        num_robots=num_robots,
        robot_capacity=ROBOT_CAPACITY,
        label=f"RL_{setting_name}_train",
        curve_store=curve_store,
    )
    train_store.save_csv(os.path.join(setting_dir, "train.csv"))
    curve_store.save_csv(os.path.join(setting_dir, "training_curve.csv"))
    agent.save(os.path.join(setting_dir, "agent.pt"),
               checkpoint_meta(EXPERIMENT_LAMBDA, num_robots, num_workstations, total_episodes))

    eval_store = MetricsStore()
    eval_seeds = list(range(EVAL_SEED_BASE, EVAL_SEED_BASE + n_eval))
    evaluate_all(
        layout, agent, eval_store,
        lam=EXPERIMENT_LAMBDA,
        num_robots=num_robots,
        robot_capacity=ROBOT_CAPACITY,
        eval_seeds=eval_seeds,
        n_eval=n_eval,
    )
    eval_path = os.path.join(setting_dir, "evaluation.csv")
    eval_store.save_csv(eval_path)
    print_comparison(
        eval_store, POLICY_LABELS,
        lam=EXPERIMENT_LAMBDA,
        num_robots=num_robots,
        last_n=n_eval,
        title=f"lambda={lam_hr}/hr, M={num_robots}, K={num_workstations}",
    )
    print(f"Saved: {eval_path}")


def run_grid(experiment_name: str, settings: Iterable[tuple[int, int]],
             total_episodes: int = TOTAL_EPISODES, n_eval: int = N_EVAL,
             out_dir: str = EXPERIMENT_OUT_DIR) -> None:
    """여러 (M, K) 조건을 차례로 실행한다.

    입력:
        experiment_name: 실험 이름
        settings: (num_robots, num_workstations) 리스트
        total_episodes: 조건당 학습 에피소드 수
        n_eval: 조건당 평가 시드 수
        out_dir: 결과 최상위 디렉터리
    """
    for num_robots, num_workstations in settings:
        run_setting(experiment_name, num_robots, num_workstations,
                    total_episodes=total_episodes, n_eval=n_eval, out_dir=out_dir)


def run_robot_sweep(total_episodes: int = TOTAL_EPISODES, n_eval: int = N_EVAL,
                    out_dir: str = EXPERIMENT_OUT_DIR):
    """워크스테이션 수 K를 고정하고 로봇 수 M을 바꿔가며 실행한다.

    입력:
        total_episodes: 조건당 학습 에피소드 수
        n_eval: 조건당 평가 시드 수
        out_dir: 결과 최상위 디렉터리
    """
    run_grid(f"robot_sweep_k{EXPERIMENT_ROBOT_SWEEP_K}",
             [(m, EXPERIMENT_ROBOT_SWEEP_K) for m in EXPERIMENT_ROBOT_SWEEP],
             total_episodes, n_eval, out_dir)


def run_ws_sweep(total_episodes: int = TOTAL_EPISODES, n_eval: int = N_EVAL,
                 out_dir: str = EXPERIMENT_OUT_DIR):
    """로봇 수 M을 고정하고 워크스테이션 수 K를 바꿔가며 실행한다.

    입력:
        total_episodes: 조건당 학습 에피소드 수
        n_eval: 조건당 평가 시드 수
        out_dir: 결과 최상위 디렉터리
    """
    run_grid(f"workstation_sweep_m{EXPERIMENT_WS_SWEEP_M}",
             [(EXPERIMENT_WS_SWEEP_M, k) for k in EXPERIMENT_WS_SWEEP],
             total_episodes, n_eval, out_dir)


def main():
    """명령행 인자로 고른 실험을 실행한다."""
    parser = argparse.ArgumentParser(description="MTSR experiments")
    parser.add_argument("experiment", nargs="?", default="lambda-grid",
                        choices=["lambda-grid", "robot-sweep", "ws-sweep", "all"])
    parser.add_argument("--episodes", type=int, default=TOTAL_EPISODES,
                        help="학습 에피소드 수")
    parser.add_argument("--n-eval", type=int, default=N_EVAL,
                        help="평가 시드 수")
    parser.add_argument("--grid", action="store_true",
                        help="셀 그리드 이동 모드 (충돌/혼잡 반영)")
    parser.add_argument("--traffic", choices=["free", "lanes"], default=GRID_TRAFFIC,
                        help="그리드 이동 규칙: 모든 방향(free) / 일방통행 통로 + 차선 교차통로(lanes)")
    parser.add_argument("--motion", choices=["reserve", "reactive"], default=GRID_MOTION,
                        help="그리드 충돌 처리: 시공간 경로 예약(reserve) / 막히면 대기·우회(reactive)")
    parser.add_argument("--no-collisions", action="store_true",
                        help="reactive에서 로봇 간 셀 점유를 무시 (회귀 검증용)")
    parser.add_argument("--scenario", default=None,
                        help="시나리오 JSON 문자열 또는 JSON 파일 경로 (scenario.Scenario 필드)")
    args = parser.parse_args()

    global GRID_MODE, GRID_COLLISIONS, SCENARIO
    if args.scenario:
        text = open(args.scenario).read() if os.path.isfile(args.scenario) else args.scenario
        SCENARIO = Scenario(**json.loads(text))
    GRID_MODE       = args.grid
    GRID_COLLISIONS = not args.no_collisions
    GRID_OPTS.update(traffic=args.traffic, motion=args.motion)
    suffix = ""
    if GRID_MODE:
        suffix = f"_grid_{args.traffic}_{args.motion}" + ("" if GRID_COLLISIONS else "_nocoll")
    if not SCENARIO.is_default:
        import hashlib
        tag = hashlib.sha1(json.dumps(SCENARIO.to_dict(), sort_keys=True).encode()).hexdigest()[:6]
        suffix += f"_sc{tag}"
        print(f"시나리오 {tag}: {SCENARIO.to_dict()}")

    out_dir = OUT_DIR + suffix
    exp_dir = EXPERIMENT_OUT_DIR + suffix
    if not SCENARIO.is_default:
        for d in (out_dir, exp_dir):
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "scenario.json"), "w") as f:
                json.dump(SCENARIO.to_dict(), f, indent=1, ensure_ascii=False)

    if args.experiment in ("lambda-grid", "all"):
        run_lambda_grid(args.episodes, args.n_eval, out_dir=out_dir)
    if args.experiment in ("robot-sweep", "all"):
        run_robot_sweep(args.episodes, args.n_eval, out_dir=exp_dir)
    if args.experiment in ("ws-sweep", "all"):
        run_ws_sweep(args.episodes, args.n_eval, out_dir=exp_dir)
    if args.experiment in ("robot-sweep", "ws-sweep", "all"):
        print(f"\nDone. Results saved under: {exp_dir}")


if __name__ == "__main__":
    main()
