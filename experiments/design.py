"""실험 설계: 셀, 학습 시드, arm, 정책 묶음 (runner.py와 tables.py가 공유)

셀 = (λ orders/hr, M, K, 경로 규칙[, 이동 가정]) → {out}/{setting}/lam{λ}_M{M}_K{K}/
모든 비교는 같은 테스트 인스턴스(evaluation.test_seeds) 위의 대응 비교, RL은 학습 시드 단위로 집계

실험 블록 (runner.py 계획 이름)
  main      : λ ∈ MAIN_LAMS, M = 14, K = 3. 베이스라인 선택(검증 시드) + 전체 베이스라인 테스트 + 기본 arm 학습 (시드 1~5)
  variants  : 기준 셀에서 학습 변형 arm (VARIANT_ARMS) 학습 (시드 1~3)
  sens      : 보상 계수 민감도 (SENS_DESIGN) 학습 (시드 1~3, 중심점 = main의 기본 arm 같은 시드)
  fleet     : 재학습 없는 일반화 — M ∈ FLEET_M, λ = 240·M/14 셀에 기본 네트워크 전이
  small     : 재학습 없는 일반화 — λ = 240, M ∈ SMALL_M 셀에 기본 네트워크 전이
  factorial : 2³ 요인 실험 — (λ, M, K) ∈ FACTORIAL_CELLS. K마다 (320, 14, K)에서 학습한 네트워크를 전이
              (λ·M = 네트워크 내 요인, K = 네트워크 간 요인), 기준 행 (240, 14, 3) ← λ = 320 main 네트워크
  routing   : 라우팅 오차 분리 — 경로 규칙 ROUTE_VARIANTS에서 같은 네트워크 재평가 (가중치 고정),
              기준 셀 NNH 경로 구성 기록 → route_gap.py로 NNH vs exact
기본 네트워크 = 기준 셀(BASE_CELL)의 기본 arm 시드 TRAIN_SEEDS
"""
import json
import os
from dataclasses import dataclass
from typing import List, Optional

from rewards import DEFAULT_ARM
from evaluation import Condition

BASE_ARM     = DEFAULT_ARM
TRAIN_SEEDS  = [1, 2, 3, 4, 5]          # main 학습 시드 (기본 네트워크)
SUB_SEEDS    = [1, 2, 3]                # 변형·민감도·요인 실험 학습 시드
N_TEST       = 30                       # 테스트 인스턴스 수
N_VAL_SELECT = 10                       # 베이스라인 선택용 검증 인스턴스 수
N_VAL_RL     = 5                        # RL 체크포인트 선택용 검증 인스턴스 수

SELECT_KEYS = ["BL4", "BL5", "BL6", "BL7"]           # 기준(최고 삽입형 규칙) 선택 후보, BL7 = Greedy_R
BATCH_KEYS  = ["BL1", "BL1F", "BL2"]                 # 배치 규칙 (개선 출처 분해용 최고 배치 규칙 선택)
ALL_KEYS    = ["BL1", "BL1F", "BL2", "BL3", "BL4", "BL5", "BL6", "BL6D0", "BL7"]   # 셀마다 테스트할 베이스라인

MAIN_LAMS = [240, 280, 320]
FLEET_M   = [8, 10, 12, 16, 18, 20]
SMALL_M   = [6, 10]
VARIANT_ARMS = ["local_r8_ddqn_v2_gamma", "legacy_r8_ddqn_v2"]
# 보상 계수 민감도: (λ, arm), 앞에서부터 실행 (λ = 320 먼저, 항 제거 w = 0은 뒤)
SENS_DESIGN = ([(lam, f"{BASE_ARM}_{t}") for lam in (320, 240) for t in ("wext0.5", "wext2", "wq0.5", "wq2")]
               + [(lam, f"{BASE_ARM}_{t}") for lam in (320, 240) for t in ("wext0", "wq0")])
ROUTE_VARIANTS = ["fixed_dock", "fixed_return"]


@dataclass(frozen=True)
class Cell:
    """평가/학습 셀 (λ orders/hr, M, K, 경로 규칙, 이동 가정, 시나리오 JSON (빈 문자열 = 기본 시나리오))"""
    lam:      float
    M:        int
    K:        int
    route:    str = "nnh"
    traffic:  str = "free"
    motion:   str = "reserve"
    scenario: str = ""

    @property
    def sc(self):
        """scenario.Scenario (기본 시나리오이면 None)"""
        if not self.scenario:
            return None
        from scenario import Scenario
        return Scenario(**json.loads(self.scenario))

    def layout(self):
        return self.cond.layout(self.M, self.K, self.sc)

    @property
    def cond(self) -> Condition:
        return Condition(traffic=self.traffic, motion=self.motion, route=self.route)

    @property
    def setting(self) -> str:
        return self.cond.setting(self.sc)

    @property
    def name(self) -> str:
        return f"lam{self.lam:.0f}_M{self.M}_K{self.K}"

    def dir(self, out: str) -> str:
        return os.path.join(out, self.setting, self.name)

    def args(self) -> List[str]:
        """run_rl.py 셀 인자"""
        a = ["--lam", str(self.lam), "--robots", str(self.M), "--ws", str(self.K),
             "--traffic", self.traffic, "--motion", self.motion]
        a += ["--route", self.route] if self.route != "nnh" else []
        return a + (["--scenario", self.scenario] if self.scenario else [])

    def __str__(self):
        return (f"λ={self.lam:.0f} M={self.M} K={self.K}" + ("" if self.route == "nnh" else f" route={self.route}")
                + (f" sc={self.setting}" if self.scenario else ""))


@dataclass(frozen=True)
class Ckpt:
    """체크포인트 (학습 셀, arm, 학습 시드)"""
    cell: Cell
    arm:  str
    seed: int

    def dir(self, out: str) -> str:
        return os.path.join(self.cell.dir(out), f"{self.arm}_s{self.seed}")

    def path(self, out: str) -> str:
        return os.path.join(self.dir(out), "agent.pt")

    def _suffix(self, target: Cell, tag: Optional[str]) -> str:
        s = ""
        if self.cell.M != target.M:
            s += f"_tr{self.cell.M}"
        if abs(self.cell.lam - target.lam) > 0.5:
            s += f"_trlam{self.cell.lam:.0f}"
        return s + target.cond.tag + (f"_{tag}" if tag else "")

    def run_name(self, target: Cell, tag: Optional[str] = None) -> str:
        """run_rl.py --load가 target 셀에서 만드는 run 이름 (같은 규칙)"""
        return f"{self.arm}_s{self.seed}" + self._suffix(target, tag)

    def policy(self, target: Cell, tag: Optional[str] = None) -> str:
        """target 셀에서 이 네트워크 그룹의 정책 이름 (analysis.load_records: run 이름에서 _s{시드}를 뺀 것)"""
        return self.arm + self._suffix(target, tag)


BASE_CELL = Cell(240, 14, 3)
MAIN_CELLS = [Cell(lam, 14, 3) for lam in MAIN_LAMS]
FLEET_CELLS = [Cell(round(240 * M / 14), M, 3) for M in FLEET_M]
SMALL_CELLS = [Cell(240, M, 3) for M in SMALL_M]
FACTORIAL_CELLS = [Cell(lam, M, K) for lam in (240, 320) for M in (10, 18) for K in (2, 4)]
FACTORIAL_TRAIN = {K: Cell(320, 14, K) for K in (2, 4)}      # K마다 네트워크 학습 셀
ROUTE_CELLS = [Cell(BASE_CELL.lam, BASE_CELL.M, BASE_CELL.K, route=r) for r in ROUTE_VARIANTS]


def base_nets(arm: str = BASE_ARM) -> List[Ckpt]:
    """기본 네트워크 (기준 셀, 시드 TRAIN_SEEDS)"""
    return [Ckpt(BASE_CELL, arm, s) for s in TRAIN_SEEDS]


def factorial_nets(K: int, arm: str = BASE_ARM) -> List[Ckpt]:
    """요인 실험 네트워크 (λ = 320, M = 14, 해당 K, 시드 SUB_SEEDS)"""
    return [Ckpt(FACTORIAL_TRAIN[K], arm, s) for s in SUB_SEEDS]


def factorial_ref_nets(arm: str = BASE_ARM) -> List[Ckpt]:
    """요인 실험 기준 행 네트워크 (main λ = 320 셀, 시드 TRAIN_SEEDS) → 기준 셀로 전이"""
    return [Ckpt(Cell(320, 14, 3), arm, s) for s in TRAIN_SEEDS]
