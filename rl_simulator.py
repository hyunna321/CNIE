"""DQNAgent가 토트마다 로봇을 고르는 RL 시뮬레이터

결정 단위는 토트 (주문 도착 시 그 주문의 토트마다 1번). 한 주문의 토트들은 같은 시각의 연속 결정
행동 = 후보 로봇 (빈 슬롯이 있는 로봇) 중 하나. 후보마다 Q값을 추정해 최댓값 로봇에 배정
후보가 없으면 결정 없이 토트를 대기 큐에 넣고, 트립 완료 시 FIFO로 방출 (결정 아님)

보상 모드 (reward_mode)
  - "legacy": 배정 시점 보상, 결정당 γ 할인
      exp(−α · Δd / D_max) − γ_q · 큐 압박 − (idle 로봇을 썼는데 빈 idle 로봇이 더 없으면) β_dep
  - "smdp": 결정 사이 구간의 비용률을 시간 적분, 할인 e^{−βτ}
      비용률 = N(t) + c_d · v · 운행 로봇 수 (N(t): 시스템 내 주문 수)
  - "local": 구간 적분 없이 결정 k에 귀속되는 국소 비용, 할인 e^{−βτ}
      비용 = (토트 k WS 처리 완료 − 결정 시각) + 탑재 토트 지연 추정 (n_onboard · Δd / v)
             + 결정 k 직후(다음 결정 전) 후보가 없어 큐에 들어간 토트들의 (완료 − 큐 진입)
      보상 = −(비용 − μ) / scale. 완료 시각을 알아야 하므로 에피소드 종료 후 일괄 계산
      μ: 에이전트에 고정 기준점 μ(λ)가 있으면 그 값 (agent.local_mu, v2), 없으면 학습 누적 평균 (v1)
      (에피소드 끝까지 완료되지 않은 토트는 sim_time까지로 자름)

에이전트가 uses_global(GlobalDQNAgent)이면 결정마다 _global_state()를 set_global로 넘기고,
전이의 상태/다음 상태로 후보 특징 대신 agent.last_key를 씀
"""
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from agents import DQNAgent
from environment import Robot
from simulator import SimulatorBase, Event, Tote
from metrics import EpisodeResult
from baselines import compute_reward
from constant import (
    SIM_TIME, NUM_ROBOTS, ROBOT_CAPACITY, ROBOT_SPEED, GAMMA, REWARD_MODES,
    SMDP_BETA, SMDP_COST_DIST, SMDP_REWARD_SCALE, STATE_MAX_TIME, GLOBAL_ROBOT_DIM,
)


class RLSimulator(SimulatorBase):
    """DQNAgent 배정 시뮬레이터

    한 결정의 전이는 다음 결정 시점의 후보 특징이 나와야 완성되므로 직전 결정을 _pending에 보관
    _pending = (선택 특징, 결정 시각, 보상)  (smdp는 보상이 다음 결정 때 확정)
    """

    def __init__(self, layout, agent: DQNAgent, lam: float, seed: int,
                 sim_time: float = SIM_TIME, num_robots: int = NUM_ROBOTS,
                 robot_capacity: int = ROBOT_CAPACITY, eval_mode: bool = False,
                 reward_mode: str = "smdp", reward_cfg: Optional[Dict] = None):
        """
        입력:
            layout: GridWarehouseLayout
            agent: 배정 정책 DQNAgent
            lam: 주문 도착률 (orders/s)
            seed: 주문 생성 난수 시드
            sim_time: 시뮬레이션 시간 (s)
            num_robots: 로봇 수
            robot_capacity: 로봇 용량
            eval_mode: True이면 ε=0으로 실행하고 전이를 수집하지 않음
            reward_mode: "legacy", "smdp" (모듈 설명 참고)
            reward_cfg: legacy compute_reward 가중치 덮어쓰기용 kwargs
        """
        assert reward_mode in REWARD_MODES, reward_mode
        super().__init__(layout, lam, seed, sim_time, num_robots, robot_capacity)
        self.agent       = agent
        self.eval_mode   = eval_mode
        self.reward_mode = reward_mode
        self.reward_cfg  = reward_cfg or {}
        self.use_global  = getattr(agent, "uses_global", False)
        self._init_rl_state()

    def _init_rl_state(self):
        self.total_reward = 0.0
        self._pending: Optional[tuple] = None
        self._transitions: List[tuple] = []
        self._smdp_cost = 0.0                 # 직전 결정 이후 누적된 할인 비용
        self._decisions: List[tuple] = []     # local: (토트, 결정 시각, 탑재 토트 지연 추정)
        self._queued: List[tuple] = []        # local: (직전 결정 번호, 토트, 큐 진입 시각)
        self.last_mu        = float("nan")    # local: 이번 에피소드 보상 기준점 μ (s)
        self.last_mean_cost = float("nan")    # local: 이번 에피소드 결정당 평균 국소 비용 (s, μ와 무관)
        # 결정 통계 (보상 모드와 무관하게 집계): 결정 수, 후보 수 합, 큐 압박 항 / 소진 페널티 활성 횟수
        self._n_dec = self._n_cands = self._n_qpen = self._n_depl = 0

    def reset(self, seed: int):
        """에피소드 상태, 누적 보상, 보류 전이 초기화"""
        super().reset(seed)
        self._init_rl_state()

    def run(self) -> List[tuple]:
        """에피소드 1회를 끝까지 실행

        출력:
            학습용 전이 리스트 [(feat, reward, next_feats, done, discount), ...] (eval_mode이면 빈 리스트)
        """
        self._transitions = []
        saved_epsilon = self.agent.epsilon
        if self.eval_mode:
            self.agent.epsilon = 0.0

        super().run()

        # 마지막 결정은 다음 상태가 없으므로 done=True로 마감
        self._close_interval()
        if not self.eval_mode and self._pending is not None:
            feat, r, g = self._pending_transition(self.sim_time)
            self._transitions.append((feat, r, None, True, g))
        self._pending = None
        if self.reward_mode == "local":
            self._assign_local_rewards()

        if self.eval_mode:
            self.agent.epsilon = saved_epsilon
        return self._transitions

    def _assign_local_rewards(self):
        """local: 결정별 국소 비용을 계산해 전이 보상을 채움 (전이 i = 결정 i)"""
        T = self.sim_time
        done_t = lambda t: t.done_time if t.done_time is not None else T
        cost = [done_t(t) - t_k + ext for t, t_k, ext in self._decisions]
        for k, t, t_q in self._queued:
            cost[k] += done_t(t) - t_q
        if not cost:
            return
        a = self.agent
        if not self.eval_mode:
            a.local_cost_sum = getattr(a, "local_cost_sum", 0.0) + sum(cost)
            a.local_cost_n   = getattr(a, "local_cost_n", 0) + len(cost)
        mu = a.local_mu(self.lam) if hasattr(a, "local_mu") else None
        if mu is None:
            mu = a.local_cost_sum / a.local_cost_n if getattr(a, "local_cost_n", 0) else sum(cost) / len(cost)
        self.last_mu, self.last_mean_cost = mu, sum(cost) / len(cost)
        rewards = [-(c - mu) / SMDP_REWARD_SCALE for c in cost]
        self.total_reward = sum(rewards)
        if self._transitions:
            assert len(self._transitions) == len(rewards), (len(self._transitions), len(rewards))
            self._transitions = [(f, r, nf, d, g) for (f, _, nf, d, g), r in zip(self._transitions, rewards)]

    # 상태

    def _context(self) -> Dict:
        """결정 시점의 상태 (DQNAgent.build_candidates 입력)

        출력:
            n_wait: 대기 큐 길이 (토트)
            n_in_system: 시스템 내 주문 수 N(t) (도착 − 완료, smdp 비용률과 같은 값)
            n_busy: 운행 로봇 수 (idle 아님, smdp 비용률과 같은 값)
            num_robots: 로봇 수
            onboard: {로봇 id: 아직 하역하지 않은 토트 수} (BL6 한계비용의 적재 토트 수와 같은 값)
        """
        return {"n_wait": self.order_mgr.queue_size(), "n_in_system": self._n_in_system,
                "n_busy": sum(not r.is_idle for r in self.robots), "num_robots": self.num_robots,
                "onboard": {r.id: len(self._robot_totes[r.id]) for r in self.robots}}

    def _global_state(self, tote: Tote, eligible: List[Robot]) -> Tuple[np.ndarray, np.ndarray]:
        """GlobalDQNAgent 입력: 전 로봇 특징 집합 R과 WS·토트·컨텍스트 벡터 W

        R 행 (로봇마다 GLOBAL_ROBOT_DIM): idle 여부, 빈 슬롯 비율, 미하역 토트 비율, 잔여 거리, 트립 잔여 시간,
            토트 기준 상대좌표 2, WS 처리 중 여부, 후보 여부, 남은 도킹 WS 수 비율
        W: WS별 [도킹 잔여 처리 시간, 대기열 셀 점유율, 그 WS로 갈 탑재 토트 비율, 그 토트들의 처리 시간 합]
           + 토트 WS one-hot + 컨텍스트 4 (build_candidates와 같은 정규화)
        """
        L, C, M, K = self.layout, self.robot_capacity, self.num_robots, self.num_workstations
        tx, ty = tote.loc
        cand   = {r.id for r in eligible}
        R  = np.zeros((M, GLOBAL_ROBOT_DIM), dtype=np.float32)
        ws = np.zeros((K, 4), dtype=np.float32)
        for i, r in enumerate(self.robots):
            totes = self._robot_totes[r.id]
            R[i] = [float(r.is_idle), r.idle_slots / C, len(totes) / C,
                    min(r.remaining_distance / L.dist_norm, 5.0),
                    min(self._robot_time_left(r) / STATE_MAX_TIME, 3.0),
                    (r.pos[0] - tx) / L.coord_norm, (r.pos[1] - ty) / L.coord_norm,
                    float(r.timer_kind == "ws"), float(r.id in cand), len(self._needed_docks(r)) / K]
            if r.timer_kind == "ws" and r.pos in L.dock_ws:
                ws[L.dock_ws[r.pos], 0] = min(r.timer / STATE_MAX_TIME, 3.0)
            for t in totes:
                ws[t.ws_id, 2] += 1.0 / (M * C)
                ws[t.ws_id, 3] += t.proc_s / STATE_MAX_TIME
        occ = {r.pos for r in self.robots}
        for k, cells in enumerate(L.queue_cells):
            ws[k, 1] = sum(c in occ for c in cells) / max(1, len(cells))
        ctx = [min(self.order_mgr.queue_size() / (M * C), 1.0), min(self.now / self.sim_time, 1.0),
               min(self._n_in_system / (M * C), 5.0), sum(not r.is_idle for r in self.robots) / M]
        W = np.concatenate([ws.ravel(), np.eye(K, dtype=np.float32)[tote.ws_id],
                            np.asarray(ctx, dtype=np.float32)])
        return R, W

    # 결정

    def _on_tote_arrival(self, tote: Tote):
        """새 토트를 에이전트가 고른 후보 로봇에 배정 (후보가 없으면 결정 없이 대기 큐)

        학습 모드이면 직전 결정의 전이를 완성해 _transitions에 추가, 이번 결정은 _pending에 보류
        """
        t0 = time.perf_counter()
        eligible = self._eligible_robots()
        if not eligible:
            if self.reward_mode == "local" and self._decisions:
                self._queued.append((len(self._decisions) - 1, tote, self.now))
            self.order_mgr.enqueue(tote, self.now)
            return
        ctx    = self._context()
        deltas = [self._insertion_delta(r, tote) for r in eligible]
        feats  = self.agent.build_candidates(tote.loc, tote.ws_id, eligible, deltas,
                                             self.layout, ctx, now=self.now)
        if self.use_global:
            self.agent.set_global(*self._global_state(tote, eligible))
        idx = self.agent.select_index(feats)
        t_sel = time.perf_counter()
        if self.recorder is not None and hasattr(self.recorder, "on_rl_decision"):
            self.recorder.on_rl_decision(self, tote, eligible, feats, idx, deltas)
        t_rec = time.perf_counter() - t_sel           # 기록기 시간은 결정 지연에서 뺌

        self._close_interval()

        robot  = eligible[idx]
        self._count_decision(robot, eligible)
        reward = None
        if self.reward_mode == "legacy":
            reward = compute_reward(robot, self.layout, deltas[idx], eligible,
                                    self.order_mgr.queue_size(), len(eligible),
                                    self.robot_capacity, **self.reward_cfg)
            self.total_reward += reward
        elif self.reward_mode == "local":
            ext = 0.0 if robot.is_idle else len(self._robot_totes[robot.id]) * deltas[idx] / ROBOT_SPEED
            self._decisions.append((tote, self.now, ext))
        self._extra_travel += deltas[idx]
        self._assign_tote(robot, tote)
        self._record_latency(t0, t_sel, time.perf_counter() - t_rec)

        state = self.agent.last_key if self.use_global else None
        if not self.eval_mode and self._pending is not None:
            feat_prev, r_prev, g_prev = self._pending_transition(self.now)
            self._transitions.append((feat_prev, r_prev, state if self.use_global else feats, False, g_prev))
        self._pending   = (state if self.use_global else feats[idx], self.now, reward)
        self._smdp_cost = 0.0

    def _count_decision(self, robot: Robot, eligible: List[Robot]):
        """결정 통계 집계 (배정 직전 상태, compute_reward의 큐 압박 / 소진 페널티 조건과 같음)"""
        C = self.robot_capacity
        self._n_dec   += 1
        self._n_cands += len(eligible)
        self._n_qpen  += self.order_mgr.queue_size() > 0
        if robot.is_idle and not any(r.id != robot.id and r.is_idle and r.idle_slots == C for r in eligible):
            self._n_depl += 1

    # 보상

    def _on_interval(self, t0: float, t1: float, n_in_system: int, n_busy: int):
        """smdp: [t0, t1] 구간 비용을 직전 결정 시각 t_k 기준으로 할인해 누적

        구간 안에서 비용률 c가 일정하므로 ∫ e^{−β(t−t_k)} c dt = c · (E0 − E1) / β
        """
        if self.reward_mode != "smdp" or self._pending is None:
            return
        t_k = self._pending[1]
        c   = n_in_system + SMDP_COST_DIST * ROBOT_SPEED * n_busy
        b   = SMDP_BETA
        self._smdp_cost += c * (math.exp(-b * (t0 - t_k)) - math.exp(-b * (t1 - t_k))) / b

    def _close_interval(self):
        """smdp: 직전 결정 이후 누적 비용을 보상으로 확정"""
        if self.reward_mode != "smdp" or self._pending is None:
            return
        feat, t_k = self._pending[0], self._pending[1]
        r = -self._smdp_cost / SMDP_REWARD_SCALE
        self.total_reward += r
        self._pending = (feat, t_k, r)

    def _pending_transition(self, t_next: float) -> Tuple[np.ndarray, float, float]:
        """보류 중인 직전 결정을 (feat, reward, discount)로 변환

        입력:
            t_next: 다음 결정 시각 (s)
        출력:
            (선택 후보 특징, 보상, 할인 계수 (legacy: γ, smdp: e^{−βτ}))
        """
        feat, t_k, r = self._pending
        if self.reward_mode == "legacy":
            return feat, r, GAMMA
        return feat, r, math.exp(-SMDP_BETA * (t_next - t_k))

    def build_result(self, label: str, episode: int, lam: float) -> EpisodeResult:
        """SimulatorBase.build_result()에 보상(완료 주문 1건당), ε, 결정 통계 추가"""
        res = super().build_result(label, episode, lam)
        res.reward  = self.total_reward / max(1, self.order_mgr.completed_count())
        res.epsilon = 0.0 if self.eval_mode else self.agent.epsilon
        res.n_decisions        = self._n_dec
        res.avg_candidates     = self._n_cands / self._n_dec if self._n_dec else 0.0
        res.n_qpen_active      = self._n_qpen
        res.n_depletion_active = self._n_depl
        return res
