"""λ=800/hr에서 로봇/워크스테이션 수를 비교하는 실험 실행 스크립트.

실험 1: K=3 고정, 로봇 수 M=6, 10, 14, 18, 22
실험 2: M=14 고정, 워크스테이션 수 K=1, 3, 5

각 조건에서 별도의 RL 정책을 학습한 뒤, 동일한 평가 시드로 RL과
BL1~BL5를 평가한다. 결과는 logs_experiment/<experiment>/<config>/에 저장된다.
"""

import os
from typing import Iterable

from agents import DQNAgent
from environment import SIM_TIME, WarehouseLayout
from metrics import MetricsStore, print_comparison
from run import TrainingCurveStore, evaluate_all, train


SEED = 42
LAMBDA = 800 / 3600       # 800 orders/hour
ROBOT_CAPACITY = 5
TOTAL_EPISODES = 1000
N_EVAL = 20
OUT_DIR = "logs_experiment"

LABELS = [
    "RL",
    "BL1_FullBuffer",
    "BL2_Partial_k3",
    "BL3_Random",
    "BL4_Nearest",
    "BL5_MinInsertion",
]


def run_setting(experiment_name: str, num_robots: int,
                num_workstations: int) -> None:
    """Train and evaluate all six agents for one (M, K) setting."""
    lam_hr = int(LAMBDA * 3600)
    setting_name = f"M{num_robots}_K{num_workstations}"
    setting_dir = os.path.join(OUT_DIR, experiment_name, setting_name)
    os.makedirs(setting_dir, exist_ok=True)

    layout = WarehouseLayout(seed=SEED, num_workstations=num_workstations)
    agent = DQNAgent(
        num_robots=num_robots,
        robot_capacity=ROBOT_CAPACITY,
        seed=SEED,
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
        lam=LAMBDA,
        seed=SEED,
        total_episodes=TOTAL_EPISODES,
        num_robots=num_robots,
        robot_capacity=ROBOT_CAPACITY,
        label=f"RL_{setting_name}_train",
        curve_store=curve_store,
    )
    train_store.save_csv(os.path.join(setting_dir, "train.csv"))
    curve_store.save_csv(os.path.join(setting_dir, "training_curve.csv"))

    eval_store = MetricsStore()
    eval_seeds = list(range(10000, 10000 + N_EVAL))
    evaluate_all(
        layout, agent, eval_store,
        lam=LAMBDA,
        num_robots=num_robots,
        robot_capacity=ROBOT_CAPACITY,
        eval_seeds=eval_seeds,
        n_eval=N_EVAL,
    )
    eval_path = os.path.join(setting_dir, "evaluation.csv")
    eval_store.save_csv(eval_path)
    print_comparison(
        eval_store, LABELS,
        lam=LAMBDA,
        num_robots=num_robots,
        last_n=N_EVAL,
        title=f"lambda={lam_hr}/hr, M={num_robots}, K={num_workstations}",
    )
    print(f"Saved: {eval_path}")


def run_grid(experiment_name: str,
             settings: Iterable[tuple[int, int]]) -> None:
    for num_robots, num_workstations in settings:
        run_setting(experiment_name, num_robots, num_workstations)


def main() -> None:
    # 1) 워크스테이션 3대 고정, 적정 로봇 수 범위 비교.
    run_grid(
        "robot_sweep_k3",
        [(m, 3) for m in (6, 10, 14, 18, 22)],
    )

    # 2) 로봇 14대 고정, 워크스테이션 수 비교.
    run_grid(
        "workstation_sweep_m14",
        [(14, k) for k in (1, 3, 5)],
    )

    print(f"\nDone. Results saved under: {OUT_DIR}")


if __name__ == "__main__":
    main()
