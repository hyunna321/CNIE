"""그리드 시뮬레이션 평가 진입점: 배정 정책(BL1~BL6)을 같은 평가 시드로 돌려 비교

사용법:
    python run.py                              # BL1~BL6, λ=320 주문/hr, M=14, K=3, 평가 시드 20개
    python run.py --policies BL4 BL6 --lam 280 --robots 18 --n-eval 5

    --traffic free|lanes       모든 방향 이동 / 일방통행 통로 + 차선 교차통로 (기본 free)
    --motion reserve|reactive  배정 시 시공간 경로 예약 / 막히면 대기·우회 (기본 reserve)
    --no-collisions            reactive에서 로봇 간 점유를 무시 (회귀 검증용)
    --scenario: 주문 도착/SKU 수요/저장/처리 시간 분포를 JSON 문자열이나 파일로 지정 (scenario.Scenario 필드)
      예: --scenario '{"arrival": "batch", "batch_mean": 6, "sku_dist": "abc", "storage": "class_based"}'
    결과: {out}/grid_{traffic}_{motion}[_sc해시]/eval_lam{λ}_M{M}_K{K}.csv
"""
import argparse
import hashlib
import json
import os

from grid import GridWarehouseLayout
from scenario import Scenario
from metrics import MetricsStore, print_comparison
from baselines import (FullBufferSimulator, PartialSimulator, RandomAssignSimulator,
                       NearestRobotSimulator, MinInsertionSimulator, MarginalCostSimulator)
from constant import (
    GRID_TRAFFIC, GRID_MOTION, SIM_TIME, LAMBDA_DEFAULT, DEFAULT_SEED,
    NUM_ROBOTS, NUM_WORKSTATIONS, ROBOT_CAPACITY, PARTIAL_K,
    N_EVAL, EVAL_SEED_BASE, OUT_DIR,
)

# 정책 키 -> (라벨, 시뮬레이터 클래스)
POLICIES = {
    "BL1": ("BL1_FullBuffer",          FullBufferSimulator),
    "BL2": (f"BL2_Partial_k{PARTIAL_K}", PartialSimulator),
    "BL3": ("BL3_Random",              RandomAssignSimulator),
    "BL4": ("BL4_Nearest",             NearestRobotSimulator),
    "BL5": ("BL5_MinInsertion",        MinInsertionSimulator),
    "BL6": ("BL6_MarginalCost",        MarginalCostSimulator),
}


def evaluate(layout: GridWarehouseLayout, policies, lam: float, num_robots: int,
             n_eval: int, sim_time: float) -> MetricsStore:
    """정책마다 평가 시드 EVAL_SEED_BASE + i (i < n_eval)로 한 번씩 실행

    입력:
        layout: GridWarehouseLayout
        policies: POLICIES 키 리스트
        lam: 주문 도착률 (orders/s)
        num_robots: 로봇 수
        n_eval: 평가 시드 수
        sim_time: 시뮬레이션 시간 (s)
    출력:
        결과가 담긴 MetricsStore
    """
    store = MetricsStore()
    for key in policies:
        label, cls = POLICIES[key]
        for i in range(n_eval):
            seed = EVAL_SEED_BASE + i
            sim = cls(layout, lam=lam, seed=seed, sim_time=sim_time,
                      num_robots=num_robots, robot_capacity=layout.robot_capacity)
            sim.reset(seed)
            sim.run()
            r = sim.build_result(label, i, lam)
            store.add(r)
            print(f"  {label:<18} seed {seed}: flow {r.avg_flow_time:7.1f} s, "
                  f"완료 {r.n_completed}/{r.n_arrived}, 거리/주문 {r.dist_per_order:5.1f} m")
    return store


def main():
    """명령행 인자로 고른 설정에서 정책 평가"""
    parser = argparse.ArgumentParser(description="MTSR grid simulation")
    parser.add_argument("--policies", nargs="+", choices=list(POLICIES), default=list(POLICIES))
    parser.add_argument("--lam", type=float, default=LAMBDA_DEFAULT * 3600, help="주문 도착률 (orders/hr)")
    parser.add_argument("--robots", type=int, default=NUM_ROBOTS, help="로봇 수 M")
    parser.add_argument("--ws", type=int, default=NUM_WORKSTATIONS, help="워크스테이션 수 K")
    parser.add_argument("--n-eval", type=int, default=N_EVAL, help="평가 시드 수")
    parser.add_argument("--sim-time", type=float, default=SIM_TIME, help="시뮬레이션 시간 (s)")
    parser.add_argument("--traffic", choices=["free", "lanes"], default=GRID_TRAFFIC,
                        help="이동 규칙: 모든 방향(free) / 일방통행 통로 + 차선 교차통로(lanes)")
    parser.add_argument("--motion", choices=["reserve", "reactive"], default=GRID_MOTION,
                        help="충돌 처리: 시공간 경로 예약(reserve) / 막히면 대기·우회(reactive)")
    parser.add_argument("--no-collisions", action="store_true",
                        help="reactive에서 로봇 간 셀 점유를 무시 (회귀 검증용)")
    parser.add_argument("--scenario", default=None,
                        help="시나리오 JSON 문자열 또는 JSON 파일 경로 (scenario.Scenario 필드)")
    parser.add_argument("--out", default=OUT_DIR, help="결과 상위 폴더")
    args = parser.parse_args()

    scenario = Scenario()
    if args.scenario:
        text = open(args.scenario).read() if os.path.isfile(args.scenario) else args.scenario
        scenario = Scenario(**json.loads(text))

    out_dir = os.path.join(args.out, f"grid_{args.traffic}_{args.motion}"
                           + ("_nocoll" if args.no_collisions else ""))
    if not scenario.is_default:
        tag = hashlib.sha1(json.dumps(scenario.to_dict(), sort_keys=True).encode()).hexdigest()[:6]
        out_dir += f"_sc{tag}"
        print(f"시나리오 {tag}: {scenario.to_dict()}")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "scenario.json"), "w") as f:
            json.dump(scenario.to_dict(), f, indent=1, ensure_ascii=False)

    lam = args.lam / 3600.0
    layout = GridWarehouseLayout(seed=DEFAULT_SEED, num_robots=args.robots,
                                 robot_capacity=ROBOT_CAPACITY, num_workstations=args.ws,
                                 collisions=not args.no_collisions, traffic=args.traffic,
                                 motion=args.motion, scenario=scenario)
    print(f"λ={args.lam:.0f}/hr, M={args.robots}, K={args.ws}, {args.traffic}/{args.motion}, "
          f"평가 시드 {args.n_eval}개 × {args.sim_time:.0f} s")
    store = evaluate(layout, args.policies, lam, args.robots, args.n_eval, args.sim_time)

    path = os.path.join(out_dir, f"eval_lam{args.lam:.0f}_M{args.robots}_K{args.ws}.csv")
    store.save_csv(path)
    print_comparison(store, [POLICIES[k][0] for k in args.policies], lam=lam,
                     num_robots=args.robots, last_n=args.n_eval, title="Grid evaluation")
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
