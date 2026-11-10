"""RL 보상 함수와 arm(보상·학습 설정) 등록표

결정 단위는 토트 (rl_simulator 모듈 설명). 보상 객체는 RLSimulator 한 에피소드 동안 아래 훅을 받음
  - on_decision(robot, tote, delta, eligible): 배정 직전 상태에서 호출. 즉시 보상이면 값, 사후 계산이면 None
  - on_queue(tote): 후보가 없어 결정 없이 대기 큐로 간 토트
  - on_interval(t0, t1, n_in_system, n_busy): 사건 사이 구간 [t0, t1] (상태가 구간 안에서 일정)
  - finalize(): 에피소드 종료 후 결정별 보상 리스트 (사후 계산 보상만, 즉시 보상이면 None)
새 보상은 Reward를 상속해 REWARDS에 등록하고 ARMS에 arm을 추가하면 학습·평가·실험 실행기에서 그대로 쓸 수 있음

보상 종류 (Arm.reward). 아래 c_k의 시간은 s, Z = REWARD_SCALE, β = Arm.beta (기본 SMDP_BETA), τ_k = t_{k+1} − t_k
  (마지막 결정은 T(근무 종료) − t_k). 미완료 토트의 완료시각은 T로 자름
  - "local" (현행): c_k = (T_k^done − t_k) + w_ext · n_onboard · Δd / v + w_q · Σ_{q∈Q_k} (T_q^done − t_q^enq)
      Q_k = 결정 k 직후(다음 결정 전) 후보가 없어 큐에 들어간 토트. r_k = −(c_k − μ(λ)) / Z
      μ(λ) = 학습 전 BL6 prior 롤아웃의 결정당 평균 c_k (training.load_or_estimate_center), 없으면 에피소드 평균
      local_w = (0, 1)이면 2항 없는 버전 (단계 1, arm local_noterm2)
  - "order_increment" (안 A): 주문 누적 증분 비용
      주문 o의 토트를 처리 순서(배정 시각, 큐잉 토트는 큐 진입 시각, 같으면 결정 순서)로 놓고
      inc_j = max(a_o, max_{i≤j} T_i^done) − max(a_o, max_{i<j} T_i^done)  (Σ_j inc_j = o의 censored 흐름시간)
      c_k = 결정 k가 배정한 토트의 inc + 구간 [t_k, t_{k+1})에 큐에 들어간 토트들의 inc
      r_k = −(c_k − ρ(λ) · τ_k) / Z, ρ(λ) = BL6 prior 롤아웃의 시간 평균 N(t) (μ와 같은 절차)
  - "little_pbrs" (안 B): Little 비용 + 잠재 기반 shaping
      Φ(s) = −(1/Z) Σ_{o∈O(t)} (Ť_o(s) − t)   (O(t): 시스템 내 주문, Ť_o: planning.order_completion)
      r_k = −(1/Z) ∫_{t_k}^{t_{k+1}} e^{−β(t−t_k)} N(t) dt + c_phi · (e^{−βτ_k} Φ(s_{k+1}) − Φ(s_k))
      Φ(s_k)는 결정 k의 배정 직전, Φ(s_{k+1})은 다음 결정의 배정 직전 상태. 마지막 전이는 T까지 적분, Φ(종단) = 0
      N(t)는 구간별 상수라 닫힌 형태로 적분. 중심화 없음
  - "legacy": 원 논문 배정 시점 보상 (compute_legacy_reward), 결정당 γ 할인과 함께 씀
할인 (Arm.discount): "time"이면 e^{−βτ} (τ = 다음 결정까지 시간, SMDP), "decision"이면 결정당 γ
"""
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from constant import (
    ROBOT_SPEED, REWARD_SCALE, SMDP_BETA, N_STEP, TOTE_ORDERS,
    REWARD_GAMMA_Q, REWARD_ALPHA_MID, REWARD_ALPHA_IDLE, REWARD_BETA_DEPLETION,
)

FEATURE_SETS = ("base", "order")     # Arm.features (agents.order_feature_dims)


def compute_legacy_reward(robot, layout, delta_d: float, robots: list, n_wait: int, num_robots: int,
                          robot_capacity: int, gamma_q: float = REWARD_GAMMA_Q,
                          alpha_mid: float = REWARD_ALPHA_MID, alpha_idle: float = REWARD_ALPHA_IDLE,
                          beta_depletion: float = REWARD_BETA_DEPLETION) -> float:
    """legacy 보상: 토트 배정 1건의 보상 (Greedy_R 베이스라인도 사용)

    보상 = exp(−α · detour / D_max) − γ_q · 큐 압박
           − β_dep (idle 로봇을 썼는데 빈 idle 로봇이 더 남지 않은 경우)

    입력:
        robot: 토트를 배정받은 로봇 (배정 직전 상태)
        layout: GridWarehouseLayout (reward_d_max 사용)
        delta_d: 배정으로 늘어난 이동거리 (m)
        robots: 후보 로봇 리스트 (소진 페널티 판단용)
        n_wait: 현재 대기 큐 길이 (토트)
        num_robots: 큐 압박 정규화용 로봇 수
        robot_capacity: 로봇 용량
    """
    alpha  = alpha_idle if robot.is_idle else alpha_mid
    reward = float(np.exp(-alpha * delta_d / layout.reward_d_max))
    if gamma_q > 0.0:
        reward -= gamma_q * min(n_wait / max(1, num_robots * robot_capacity), 1.0)
    if beta_depletion > 0.0 and robot.is_idle:
        free_idle_left = sum(1 for r in robots
                             if r.id != robot.id and r.is_idle and r.idle_slots == robot_capacity)
        if free_idle_left == 0:
            reward -= beta_depletion
    return reward


@dataclass(frozen=True)
class Arm:
    """RL arm = 보상·학습 설정. 기본값 = 현행 (에이전트 구조·나머지 하이퍼파라미터는 모든 arm 공통, agents.DQNAgent)

    reward:     REWARDS 키
    discount:   "time" (e^{−βτ}) / "decision" (결정당 γ)
    local_w:    local 비용 항 계수 (w_ext 탑재 토트 지연 추정, w_q 큐잉 토트 지연), 자기 토트 지연 계수는 1
    n_step:     n-step 전이 길이
    features:   "base" (현행 후보 특징) / "order" (+ 주문 계획 특징, agents.order_feature_dims)
    tote_order: 주문 내 토트 배정 순서 (constant.TOTE_ORDERS)
    curriculum: True이면 앞쪽 절반 에피소드 λ ∈ [0.7λ, λ] 커리큘럼
    c_phi:      little_pbrs의 shaping 계수
    beta:       연속시간 할인율 (None = SMDP_BETA). 0이면 할인 없음 (진단)
    """
    reward:     str = "local"
    discount:   str = "time"
    local_w:    Tuple[float, float] = (1.0, 1.0)
    n_step:     int = N_STEP
    features:   str = "base"
    tote_order: str = "arrival"
    curriculum: bool = True
    c_phi:      float = 1.0
    beta:       Optional[float] = None

    def __post_init__(self):
        assert self.reward in REWARDS, self.reward
        assert self.discount in ("time", "decision"), self.discount
        assert self.features in FEATURE_SETS, self.features
        assert self.tote_order in TOTE_ORDERS, self.tote_order
        object.__setattr__(self, "local_w", tuple(float(x) for x in self.local_w))

    @property
    def uses_mu(self) -> bool:
        """학습 전에 BL6 prior 롤아웃으로 중심화 기준(μ 또는 ρ)을 추정해야 하는지"""
        return REWARDS[self.reward].center is not None

    @property
    def beta_value(self) -> float:
        return SMDP_BETA if self.beta is None else float(self.beta)

    def to_dict(self) -> Dict:
        return {**asdict(self), "local_w": list(self.local_w)}


class Reward:
    """보상 기본 클래스 (모듈 설명의 훅). RLSimulator 생성 시 한 번 만들고 에피소드마다 reset

    center: 중심화 기준 종류 (None / "mu" / "rho"). None이 아니면 학습 전 BL6 prior 롤아웃으로 표를 만들고
            center_stat()이 에피소드 하나의 기준 표본 (training.estimate_center)
    """
    center: Optional[str] = None

    def __init__(self, sim, arm: Arm):
        self.sim, self.arm = sim, arm
        self.reset()

    def reset(self):
        """에피소드 시작 시 상태 초기화. 학습 곡선 기록용 값(last_*)도 비움"""
        self.last_mu         = float("nan")         # 중심화 기준 (μ: s, ρ: 주문 수)
        self.last_mean_cost  = float("nan")         # 결정당 평균 비용 (s, 기준과 무관)
        self.last_cost_terms = (float("nan"),) * 3  # 결정당 평균 비용 항 (보상마다 의미 다름, 계수 곱 전)

    def on_decision(self, robot, tote, delta: float, eligible: list) -> Optional[float]:
        return None

    def on_queue(self, tote):
        pass

    def on_interval(self, t0: float, t1: float, n_in_system: int, n_busy: int):
        pass

    def finalize(self) -> Optional[List[float]]:
        return None

    def center_stat(self) -> float:
        """에피소드 하나의 중심화 기준 표본 (center가 None이 아니면 구현)"""
        raise NotImplementedError


class LocalReward(Reward):
    """local 보상 (모듈 설명). 결정 i의 보상 = finalize() 결과의 i번째. 항 = (자기 지연, 탑재 지연 추정, 큐잉 지연)"""
    center = "mu"

    def reset(self):
        super().reset()
        self._decisions: List[tuple] = []     # (토트, 결정 시각, 탑재 토트 지연 추정)
        self._queued:    List[tuple] = []     # (직전 결정 번호, 토트, 큐 진입 시각)

    def on_decision(self, robot, tote, delta, eligible):
        sim = self.sim
        ext = 0.0 if robot.is_idle else len(sim._robot_totes[robot.id]) * delta / ROBOT_SPEED
        self._decisions.append((tote, sim.now, ext))
        return None

    def on_queue(self, tote):
        if self._decisions:
            self._queued.append((len(self._decisions) - 1, tote, self.sim.now))

    def finalize(self):
        T = self.sim.sim_time
        w_ext, w_q = self.arm.local_w
        done_t = lambda t: t.done_time if t.done_time is not None else T
        own  = [done_t(t) - t_k for t, t_k, _ in self._decisions]
        cost = [o + w_ext * ext for o, (_, _, ext) in zip(own, self._decisions)]
        q_sum = 0.0
        for k, t, t_q in self._queued:
            q = done_t(t) - t_q
            cost[k] += w_q * q
            q_sum += q
        if not cost:
            return []
        n = len(cost)
        self.last_cost_terms = (sum(own) / n, sum(ext for _, _, ext in self._decisions) / n, q_sum / n)
        mu = self.sim.agent.local_mu(self.sim.lam)
        if mu is None:
            mu = sum(cost) / n
        self.last_mu, self.last_mean_cost = mu, sum(cost) / n
        return [-(c - mu) / REWARD_SCALE for c in cost]

    def center_stat(self) -> float:
        return self.last_mean_cost


class OrderIncrementReward(Reward):
    """안 A 주문 누적 증분 비용 (모듈 설명). 항 = (배정 토트 inc, 0, 큐잉 토트 inc), 결정당 평균"""
    center = "rho"

    def reset(self):
        super().reset()
        self._times: List[float] = []          # 결정 시각 t_k
        self._seen:  List[tuple] = []          # (토트, 귀속 결정 번호 또는 None, 처리 순서 키 시각, 큐잉 여부)
        self._n_area = 0.0                     # ∫ N(t) dt (center_stat, 진단)
        self.unattributed = 0                  # 첫 결정 전에 큐로 간 토트 수 (어느 결정에도 귀속 안 됨)

    def on_decision(self, robot, tote, delta, eligible):
        self._times.append(self.sim.now)
        self._seen.append((tote, len(self._times) - 1, self.sim.now, False))
        return None

    def on_queue(self, tote):
        k = len(self._times) - 1 if self._times else None
        self.unattributed += k is None
        self._seen.append((tote, k, self.sim.now, True))

    def on_interval(self, t0, t1, n_in_system, n_busy):
        self._n_area += n_in_system * (t1 - t0)

    def increments(self) -> Dict[int, float]:
        """토트 id → inc_j (주문마다 처리 순서로 정렬해 누적 최대 완료시각의 증분)"""
        T = self.sim.sim_time
        by_order: Dict[int, list] = {}
        for seq, (t, k, key, q) in enumerate(self._seen):
            by_order.setdefault(t.order.order_id, []).append((key, seq, t))
        inc = {}
        for items in by_order.values():
            items.sort(key=lambda x: (x[0], x[1]))
            a = items[0][2].order.arrival_time
            run = a
            for _, _, t in items:
                done = t.done_time if t.done_time is not None else T
                new = max(run, done)
                inc[t.tote_id] = new - run
                run = new
        return inc

    def costs(self) -> Tuple[List[float], float, float]:
        """결정별 c_k와 (배정 토트 inc 합, 큐잉 토트 inc 합)"""
        inc = self.increments()
        c = [0.0] * len(self._times)
        own = queue = 0.0
        for t, k, _, q in self._seen:
            if k is None:
                continue
            c[k] += inc[t.tote_id]
            if q:
                queue += inc[t.tote_id]
            else:
                own += inc[t.tote_id]
        return c, own, queue

    def finalize(self):
        if not self._times:
            return []
        T = self.sim.sim_time
        c, own, queue = self.costs()
        n = len(c)
        tau = [b - a for a, b in zip(self._times, self._times[1:])] + [T - self._times[-1]]
        rho = self.sim.agent.local_mu(self.sim.lam)
        if rho is None:
            rho = sum(c) / max(1e-9, sum(tau))
        self.last_cost_terms = (own / n, 0.0, queue / n)
        self.last_mu, self.last_mean_cost = rho, sum(c) / n
        return [-(ck - rho * tk) / REWARD_SCALE for ck, tk in zip(c, tau)]

    def center_stat(self) -> float:
        """시간 평균 N(t) = ∫_0^T N dt / T"""
        return self._n_area / self.sim.sim_time


class LittlePBRSReward(Reward):
    """안 B Little 비용 + PBRS (모듈 설명). 항 = (결정당 평균 적분항 (s), 평균 c_phi·Φ 차분 · Z, 0)"""

    def reset(self):
        super().reset()
        self._times: List[float] = []      # 결정 시각
        self._phi:   List[float] = []      # Φ(s_k), 배정 직전
        self._area:  List[float] = []      # 결정 k의 할인 적분 ∫ e^{−β(t−t_k)} N dt
        self._open:  Dict[int, object] = {}   # 시스템 내 주문 (order_id → Order)

    def potential(self) -> float:
        """Φ(s) = −(1/Z) Σ_{o∈O(t)} (Ť_o − t) (지금 상태)"""
        from planning import planned_completions, order_completion
        sim, t = self.sim, self.sim.now
        for oid in [i for i, o in self._open.items() if o.completion_time is not None]:
            del self._open[oid]
        if not self._open:
            return 0.0
        pc = planned_completions(sim)
        return -sum(order_completion(o, pc, t) - t for o in self._open.values()) / REWARD_SCALE

    def _track(self, tote):
        self._open.setdefault(tote.order.order_id, tote.order)

    def on_decision(self, robot, tote, delta, eligible):
        self._track(tote)
        self._times.append(self.sim.now)
        self._phi.append(self.potential())
        self._area.append(0.0)
        return None

    def on_queue(self, tote):
        self._track(tote)

    def on_interval(self, t0, t1, n_in_system, n_busy):
        if not self._times:
            return
        t_k, b = self._times[-1], self.arm.beta_value
        if b > 0:
            self._area[-1] += n_in_system * (math.exp(-b * (t0 - t_k)) - math.exp(-b * (t1 - t_k))) / b
        else:
            self._area[-1] += n_in_system * (t1 - t0)

    def shaping_terms(self) -> List[float]:
        """결정별 c_phi · (e^{−βτ_k} Φ(s_{k+1}) − Φ(s_k)), Φ(종단) = 0"""
        T, b, c = self.sim.sim_time, self.arm.beta_value, self.arm.c_phi
        nxt = self._phi[1:] + [0.0]
        tau = [y - x for x, y in zip(self._times, self._times[1:])] + [T - self._times[-1]]
        return [c * (math.exp(-b * tk) * p1 - p0) for p0, p1, tk in zip(self._phi, nxt, tau)]

    def finalize(self):
        if not self._times:
            return []
        sh = self.shaping_terms()
        n = len(sh)
        self.last_cost_terms = (sum(self._area) / n, sum(sh) * REWARD_SCALE / n, 0.0)
        self.last_mean_cost = sum(self._area) / n
        return [-a / REWARD_SCALE + s for a, s in zip(self._area, sh)]


class LegacyReward(Reward):
    """legacy 보상 (compute_legacy_reward, 배정 시점 즉시 보상)"""

    def on_decision(self, robot, tote, delta, eligible):
        sim = self.sim
        return compute_legacy_reward(robot, sim.layout, delta, eligible, sim.order_mgr.queue_size(),
                                     len(eligible), sim.robot_capacity)


REWARDS = {"local": LocalReward, "legacy": LegacyReward,
           "order_increment": OrderIncrementReward, "little_pbrs": LittlePBRSReward}

# arm 이름 → 설정. 기본(현행) arm은 DEFAULT_ARM
DEFAULT_ARM = "local_r8_ddqn_v2"
_NEW = dict(features="order", tote_order="bottleneck_first")      # 안 A·B 공통 (단계 2·3)
ARMS: Dict[str, Arm] = {
    DEFAULT_ARM:              Arm(),
    "local_r8_ddqn_v2_gamma": Arm(discount="decision"),                 # 할인만 결정당 γ
    "legacy_r8_ddqn_v2":      Arm(reward="legacy", discount="decision"),  # 원 논문 보상 + 결정당 γ
    # 단계 1: 현행 보상에서 2항(탑재 토트 지연 추정)만 제거 (= local_w (0, 1))
    "local_noterm2":          Arm(local_w=(0.0, 1.0)),
    # 단계 2: 안 A 주문 누적 증분 비용 (n = 5, 보조 n = 10)
    "order_inc":              Arm(reward="order_increment", n_step=5, **_NEW),
    "order_inc_n10":          Arm(reward="order_increment", n_step=10, **_NEW),
    # 단계 3·4: 안 B Little 비용 + PBRS (c_phi = 1, n = 3), 민감도 c_phi ∈ {0, 0.5}, n = 10
    "little_pbrs":            Arm(reward="little_pbrs", n_step=3, c_phi=1.0, **_NEW),
    "little_pbrs_cphi0":      Arm(reward="little_pbrs", n_step=3, c_phi=0.0, **_NEW),
    "little_pbrs_cphi0.5":    Arm(reward="little_pbrs", n_step=3, c_phi=0.5, **_NEW),
    "little_pbrs_n10":        Arm(reward="little_pbrs", n_step=10, c_phi=1.0, **_NEW),
}
for _w in (0.0, 0.25, 0.5, 2.0, 4.0):     # local 비용 항 계수 민감도 (계수 하나만 바꿈)
    ARMS[f"{DEFAULT_ARM}_wext{_w:g}"] = Arm(local_w=(_w, 1.0))
    ARMS[f"{DEFAULT_ARM}_wq{_w:g}"]   = Arm(local_w=(1.0, _w))
# 단계 5 (비정상 도착): 커리큘럼 없는 버전
for _name in ("order_inc", "little_pbrs", "little_pbrs_cphi0", "little_pbrs_cphi0.5", "little_pbrs_n10"):
    ARMS[f"{_name}_nocurr"] = Arm(**{**asdict(ARMS[_name]), "curriculum": False})


def mu_filename(arm: Arm) -> str:
    """중심화 기준 표 파일 이름 (같은 셀에서 같은 보상·계수·토트 순서의 arm끼리 공유)

    local 기본 계수는 local_mu.json, 계수가 다르면 local_mu_wext{a}_wq{b}.json, order_increment는 rho_order_increment.json.
    tote_order가 bottleneck_first이면 BL6 prior 궤적이 달라지므로 _bf를 붙임
    """
    bf = "_bf" if arm.tote_order == "bottleneck_first" else ""
    if arm.reward == "order_increment":
        return f"rho_order_increment{bf}.json"
    w = arm.local_w
    return (f"local_mu{bf}.json" if w == (1.0, 1.0) else f"local_mu_wext{w[0]:g}_wq{w[1]:g}{bf}.json")
