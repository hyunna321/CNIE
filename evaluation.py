"""정책 평가 하네스: 베이스라인(BL1~BL6, Greedy_R)과 RL 체크포인트를 같은 함수, 같은 시드 규칙으로 평가

시드 구간 (constant.py, 서로 겹치지 않음)
  - 학습 에피소드: TRAIN_SEED_BASE + 학습 시드 × TRAIN_SEED_STRIDE + 에피소드 번호
  - 검증 (체크포인트 선택/튜닝): VAL_SEED_BASE + i
  - 최종 테스트 (보고): EVAL_SEED_BASE + i
모든 평가 결과는 EpisodeResult.seed에 주문 생성 시드를 남기고, episode = 시드 리스트 인덱스

평가 조건 (Condition): 이동 가정(traffic, motion)과 경로 구성 규칙(route)
  - 결과 폴더 이름 setting = grid_{traffic}_{motion}[_sc해시][_route-fixeddock]
  - 기본 조건(route = nnh)이면 기존 이름 그대로 (grid_free_reserve 등)
"""
import gzip
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from metrics import MetricsStore
from baselines import (FullBufferSimulator, FillBufferSimulator, PartialSimulator, RandomAssignSimulator,
                       NearestRobotSimulator, MinInsertionSimulator, MarginalCostSimulator,
                       GreedyRewardSimulator)
from constant import (
    SIM_TIME, PARTIAL_K, EVAL_SEED_BASE, VAL_SEED_BASE, TRAIN_SEED_BASE, TRAIN_SEED_STRIDE,
    GRID_TRAFFIC, GRID_MOTION, DEFAULT_SEED, ROBOT_CAPACITY,
)

# 정책 키 -> (라벨, 시뮬레이터 클래스)
POLICIES = {
    "BL1": ("BL1_FullBuffer",          FullBufferSimulator),
    "BL1F": ("BL1F_FillBuffer",        FillBufferSimulator),
    "BL2": (f"BL2_Partial_k{PARTIAL_K}", PartialSimulator),
    "BL3": ("BL3_Random",              RandomAssignSimulator),
    "BL4": ("BL4_Nearest",             NearestRobotSimulator),
    "BL5": ("BL5_MinInsertion",        MinInsertionSimulator),
    "BL6": ("BL6_MarginalCost",        MarginalCostSimulator),
    "GR":  ("Greedy_R",                GreedyRewardSimulator),
}
# 논문 번호 BL7 = Greedy_R (정책 키 GR)
POLICY_ALIASES = {"BL7": "GR"}


@dataclass
class Condition:
    """평가 조건 (이동 가정 + 경로 구성 규칙). 기본값 = 본실험 조건"""
    traffic: str = GRID_TRAFFIC
    motion:  str = GRID_MOTION
    route:   str = "nnh"          # 경로 구성 규칙 (environment 모듈 설명, "nnh" / "fixed_dock")

    @property
    def sim_kw(self) -> Dict:
        """SimulatorBase 추가 인자 (기본이면 빈 dict라 기존 호출과 같음)"""
        return {"route": self.route} if self.route != "nnh" else {}

    @property
    def tag(self) -> str:
        """경로 변형 꼬리표 (기본이면 빈 문자열, fixed_dock → _route-fixeddock)"""
        return f"_route-{self.route.replace('_', '')}" if self.route != "nnh" else ""

    def layout(self, num_robots: int, num_workstations: int, scenario=None):
        """이 조건의 GridWarehouseLayout"""
        from grid import GridWarehouseLayout
        return GridWarehouseLayout(seed=DEFAULT_SEED, num_robots=num_robots, robot_capacity=ROBOT_CAPACITY,
                                   num_workstations=num_workstations, traffic=self.traffic,
                                   motion=self.motion, scenario=scenario)

    def setting(self, scenario=None) -> str:
        """결과 폴더 이름 (모듈 설명)"""
        name = f"grid_{self.traffic}_{self.motion}"
        if scenario is not None and not scenario.is_default:
            name += "_sc" + hashlib.sha1(json.dumps(scenario.to_dict(), sort_keys=True).encode()).hexdigest()[:6]
        return name + self.tag

    def to_dict(self) -> Dict:
        return asdict(self)


def policy_key(key: str) -> str:
    """논문 번호 별칭(BL7)을 POLICIES 키로"""
    return POLICY_ALIASES.get(key, key)


def test_seeds(n: int) -> List[int]:
    """최종 테스트 시드 n개"""
    return list(range(EVAL_SEED_BASE, EVAL_SEED_BASE + n))


def val_seeds(n: int) -> List[int]:
    """검증 시드 n개 (체크포인트 선택/튜닝 전용)"""
    return list(range(VAL_SEED_BASE, VAL_SEED_BASE + n))


def train_episode_seed(train_seed: int, episode: int) -> int:
    """학습 시드 train_seed의 episode번째 (1부터) 에피소드 주문 생성 시드"""
    assert 0 <= episode < TRAIN_SEED_STRIDE, episode
    return TRAIN_SEED_BASE + train_seed * TRAIN_SEED_STRIDE + episode


def make_sim(policy: str, layout, lam: float, seed: int, sim_time: float = SIM_TIME,
             agent=None, reward_mode: str = "smdp", sim_kw: Optional[Dict] = None):
    """정책 키로 시뮬레이터 생성 (로봇 수/용량은 layout을 따름)

    입력:
        policy: POLICIES 키(별칭 BL7 포함) 또는 "RL"
        layout: GridWarehouseLayout
        lam: 주문 도착률 (orders/s)
        seed: 주문 생성 시드
        sim_time: 시뮬레이션 시간 (s)
        agent: policy == "RL"일 때 DQNAgent
        reward_mode: RL 보상 기록 방식 (정책 선택에는 영향 없음)
        sim_kw: 시뮬레이터 추가 인자 (Condition.sim_kw의 route, RL이면 RLSimulator 인자도 가능)
    """
    kw = dict(lam=lam, seed=seed, sim_time=sim_time,
              num_robots=layout.num_robots, robot_capacity=layout.robot_capacity, **(sim_kw or {}))
    if policy == "RL":
        from rl_simulator import RLSimulator
        assert agent is not None, "RL 평가에는 agent가 필요합니다"
        return RLSimulator(layout, agent, eval_mode=True, reward_mode=reward_mode, **kw)
    return POLICIES[policy_key(policy)][1](layout, **kw)


def _write_reconstructions(path: str, seed: int, log: list):
    """경로 구성 기록을 JSONL로 덧붙임 (path가 .gz로 끝나면 gzip)

    한 줄 = {"seed", "i", "kind" ("start"/"reroute"), "start", "picks", "docks", "dist_nnh" (m), "ms_nnh"}
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "at", encoding="utf-8") as f:
        for i, (kind, start, picks, docks, dist, sec) in enumerate(log):
            f.write(json.dumps({"seed": seed, "i": i, "kind": kind, "start": list(start),
                                "picks": [list(c) for c in picks], "docks": [list(c) for c in docks],
                                "dist_nnh": dist, "ms_nnh": 1e3 * sec}) + "\n")


def evaluate_policy(layout, policy: str, lam: float, seeds: List[int],
                    sim_time: float = SIM_TIME, label: Optional[str] = None,
                    agent=None, reward_mode: str = "smdp",
                    store: Optional[MetricsStore] = None,
                    track_decisions: bool = True, verbose: bool = True,
                    sim_kw: Optional[Dict] = None, log_reconstructions: Optional[str] = None) -> MetricsStore:
    """정책 하나를 시드마다 한 번씩 실행해 기록 (RL은 ε = 0)

    입력:
        layout: GridWarehouseLayout
        policy: POLICIES 키 또는 "RL"
        lam: 주문 도착률 (orders/s)
        seeds: 주문 생성 시드 리스트 (test_seeds / val_seeds)
        sim_time: 시뮬레이션 시간 (s)
        label: 결과 라벨 (None이면 POLICIES 라벨, RL은 "RL")
        agent, reward_mode: RL 전용 (make_sim 참고)
        store: 결과를 추가할 MetricsStore (None이면 새로 만듦)
        track_decisions: RL이면 DecisionStats로 BL6 argmin 일치율 기록
        verbose: 시드별 한 줄 출력
        sim_kw: 시뮬레이터 추가 인자 (make_sim 참고)
        log_reconstructions: JSONL 경로. 있으면 경로 구성 호출을 시드별로 덧붙여 기록 (결과 지표는 그대로,
                             experiments/route_gap.py 입력)
    출력:
        결과가 담긴 MetricsStore
    """
    if label is None:
        label = "RL" if policy == "RL" else POLICIES[policy_key(policy)][0]
    store = store if store is not None else MetricsStore()
    for i, s in enumerate(seeds):
        sim = make_sim(policy, layout, lam, s, sim_time, agent=agent, reward_mode=reward_mode, sim_kw=sim_kw)
        stats = None
        if policy == "RL" and track_decisions:
            from viewer.recorder import DecisionStats
            stats = sim.recorder = DecisionStats()
        wall = time.perf_counter()
        sim.reset(s)
        recon = None
        if log_reconstructions:
            recon = []
            sim.log_routes(recon)
        sim.run()
        r = sim.build_result(label, i, lam)
        r.seed   = s
        r.wall_s = time.perf_counter() - wall
        if recon is not None:
            _write_reconstructions(log_reconstructions, s, recon)
        if stats is not None:
            summ = stats.summary()
            r.n_decisions, r.bl6_agree = summ["n_decisions"], summ["agree"]
        store.add(r)
        if verbose:
            agree = f", BL6 일치 {r.bl6_agree:.3f}" if stats is not None else ""
            print(f"  {label:<18} M={layout.num_robots} seed {s}: flow {r.avg_flow_time:7.1f} s, "
                  f"완료 {r.n_completed}/{r.n_arrived}, 거리/주문 {r.dist_per_order:5.1f} m{agree}")
    return store


def evaluate_baselines_cached(layout, policies: List[str], lam: float, seeds: List[int],
                              sim_time: float, path: str, sim_kw: Optional[Dict] = None) -> MetricsStore:
    """베이스라인 평가 결과를 path(CSV)에 누적: 이미 있는 (라벨, 시드)는 건너뛰고 빠진 것만 실행

    입력:
        layout, lam, seeds, sim_time, sim_kw: evaluate_policy와 같음 (path는 같은 조건의 폴더여야 함)
        policies: POLICIES 키 리스트
        path: 결과 CSV (같은 설정의 baselines/eval.csv)
    출력:
        요청한 정책 × 시드 결과만 담은 MetricsStore (episode = seeds 인덱스)
    """
    old  = MetricsStore.load_csv(path) if os.path.isfile(path) else MetricsStore()
    have = {(r.label, r.seed): r for r in old.records
            if r.seed >= 0 and abs(r.sim_time - sim_time) < 1e-6}
    for key in policies:
        label = POLICIES[policy_key(key)][0]
        todo  = [s for s in seeds if (label, s) not in have]
        if todo:
            for r in evaluate_policy(layout, key, lam, todo, sim_time, sim_kw=sim_kw).records:
                have[(label, r.seed)] = r
    # 시뮬레이션 시간이 다른 행은 그대로 두고, 시드 기록이 없는 옛 형식 행은 버림
    merged = MetricsStore()
    for r in old.records:
        if r.seed >= 0 and abs(r.sim_time - sim_time) >= 1e-6:
            merged.add(r)
    for (label, s), r in sorted(have.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        merged.add(r)
    merged.save_csv(path)

    out = MetricsStore()
    for key in policies:
        for i, s in enumerate(seeds):
            r = have[(POLICIES[policy_key(key)][0], s)]
            r.episode = i
            out.add(r)
    return out
