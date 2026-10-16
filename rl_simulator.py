"""DQNAgent가 토트마다 로봇(또는 wait)을 고르는 RL 시뮬레이터

결정 단위는 토트 (주문 도착 시 그 주문의 토트마다 1번). 한 주문의 토트들은 같은 시각의 연속 결정

보상 모드 (reward_mode)
  - "legacy": 배정 시점 보상, 결정당 γ 할인
      exp(−α · Δd / D_max) − γ_q · 큐 압박 − (idle 로봇을 썼는데 빈 idle 로봇이 더 없으면) β_dep
  - "smdp": 결정 사이 구간의 비용률을 시간 적분, 할인 e^{−βτ}
      비용률 = N(t) + c_d · v · 운행 로봇 수 (N(t): 시스템 내 주문 수)
  - "smdp_w": smdp + wait 행동 + 대기 비용 + 퍼텐셜 shaping (feature_set="smdp_w" 필요)
      - 결정 epoch: 모든 토트 도착 (후보가 없어도 epoch)
      - 행동: 후보 로봇 ∪ {wait}. wait는 후보 특징의 마지막 행, 고르면 토트를 대기 큐에 넣음
        운행 중 로봇이 있을 때만 허용 (대기 토트는 트립 완료 시 FIFO로 방출, 결정 아님)
        후보가 없으면 wait만 남음 (강제 대기)
      - 비용률 = N(t) + (κ / T_w) · Σ 대기 토트 나이 + c_d · v · 운행 로봇 수
      - 보상 = −(구간 할인 적분 비용) / scale + e^{−βτ} Φ(s') − Φ(s)
  legacy, smdp는 후보가 없으면 결정 없이 토트를 대기 큐에 넣음
"""
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from agents import DQNAgent, WS_FEAT_DIM
from environment import Robot
from simulator import SimulatorBase, Event, Tote
from metrics import EpisodeResult
from constant import (
    SIM_TIME, NUM_ROBOTS, ROBOT_CAPACITY, ROBOT_SPEED, GAMMA, REWARD_MODES,
    REWARD_GAMMA_Q, REWARD_ALPHA_MID, REWARD_ALPHA_IDLE, REWARD_BETA_DEPLETION,
    SMDP_BETA, SMDP_COST_DIST, SMDP_REWARD_SCALE,
    SMDP_WAIT_KAPPA, SMDP_WAIT_TREF, SMDP_IA_EMA, SMDP_IA_NORM, STATE_MAX_TIME,
)


def compute_reward(robot: Robot, layout, delta_d: float, robots: List[Robot],
                   n_wait: int, num_robots: int, robot_capacity: int,
                   gamma_q: float = REWARD_GAMMA_Q, alpha_mid: float = REWARD_ALPHA_MID,
                   alpha_idle: float = REWARD_ALPHA_IDLE,
                   beta_depletion: float = REWARD_BETA_DEPLETION) -> float:
    """legacy 보상: 토트 배정 1건의 보상

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
    출력:
        보상값
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


class RLSimulator(SimulatorBase):
    """DQNAgent 배정 시뮬레이터

    한 결정의 전이는 다음 결정 시점의 후보 특징이 나와야 완성되므로 직전 결정을 _pending에 보관
    _pending = (선택 특징, 결정 시각, 보상)  (smdp 계열은 보상이 다음 결정 때 확정)
    """

    def __init__(self, layout, agent: DQNAgent, lam: float, seed: int,
                 sim_time: float = SIM_TIME, num_robots: int = NUM_ROBOTS,
                 robot_capacity: int = ROBOT_CAPACITY, eval_mode: bool = False,
                 reward_mode: str = "smdp_w", reward_cfg: Optional[Dict] = None):
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
            reward_mode: "legacy", "smdp", "smdp_w" (모듈 설명 참고)
            reward_cfg: legacy compute_reward 가중치 덮어쓰기용 kwargs
        """
        assert reward_mode in REWARD_MODES, reward_mode
        assert (reward_mode == "smdp_w") == (agent.feature_set == "smdp_w"), \
            f"reward_mode={reward_mode}와 feature_set={agent.feature_set} 조합 불가 (smdp_w는 서로 짝)"
        super().__init__(layout, lam, seed, sim_time, num_robots, robot_capacity)
        self.agent       = agent
        self.eval_mode   = eval_mode
        self.reward_mode = reward_mode
        self.reward_cfg  = reward_cfg or {}
        self.with_wait   = reward_mode == "smdp_w"
        self._init_rl_state()

    def _init_rl_state(self):
        self.total_reward = 0.0
        self._pending: Optional[tuple] = None
        self._transitions: List[tuple] = []
        self._smdp_cost = 0.0                 # 직전 결정 이후 누적된 할인 비용
        self._phi_prev  = 0.0
        self._ia_ema    = 1.0 / self.lam      # 주문 도착 간격 지수이동평균 (s)
        self._last_arr  = 0.0
        self._n_wait    = 0
        self._n_forced  = 0

    def reset(self, seed: int):
        """에피소드 상태, 누적 보상, 보류 전이, 퍼텐셜, 도착 간격 추정, wait 집계 초기화"""
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

        # 마지막 결정은 다음 상태가 없으므로 Φ = 0, done=True로 마감
        self._close_interval(0.0)
        if not self.eval_mode and self._pending is not None:
            feat, r, g = self._pending_transition(self.sim_time)
            self._transitions.append((feat, r, None, True, g))
        self._pending = None

        if self.eval_mode:
            self.agent.epsilon = saved_epsilon
        return self._transitions

    # 상태

    def _ws_state(self) -> np.ndarray:
        """워크스테이션별 상태 (정규화 전)

        [도킹 잔여 처리 시간 (s), 대기열 셀 점유율, 복귀 예정 로봇 수, 복귀 예정 작업량 (s)]
        - 도킹 잔여 처리 시간: 그 WS 도킹 셀에서 처리 중인 로봇의 남은 시간
        - 복귀 예정: 그 WS로 갈 토트를 싣고(또는 배정받고) 있는 운행 로봇 수, 그 토트들의 처리 시간 합

        출력:
            (K, WS_FEAT_DIM) float32 배열
        """
        L  = self.layout
        ws = np.zeros((self.num_workstations, WS_FEAT_DIM), dtype=np.float32)
        for r in self.robots:
            if r.timer_kind == "ws":
                ws[L.dock_ws[r.pos], 0] = r.timer
            for k, proc in self._robot_ws_work(r).items():
                ws[k, 2] += 1.0
                ws[k, 3] += proc
        occ = {r.pos for r in self.robots}
        for k, cells in enumerate(L.queue_cells):
            ws[k, 1] = sum(c in occ for c in cells) / len(cells)
        return ws

    def _robot_ws_work(self, r) -> Dict[int, float]:
        """로봇이 아직 하역하지 않은 토트의 WS별 처리 시간 합 (s)"""
        work: Dict[int, float] = {}
        for t in self._robot_totes[r.id]:
            work[t.ws_id] = work.get(t.ws_id, 0.0) + t.proc_s
        return work

    def _context(self) -> Dict:
        """결정 시점의 상태 (DQNAgent.build_candidates 입력). base/bl6는 n_wait, num_robots만 채움

        출력:
            n_wait: 대기 큐 길이 (토트)
            num_robots: 로봇 수
            (smdp_w만)
            glob: [대기 토트 평균 나이, 운행 로봇 비율, 빈 슬롯 비율, 주문 도착 간격, 가장 빠른 트립 완료까지 시간]
            robot: {운행 로봇 id: (남은 피킹 수, 트립 잔여시간 (s), WS별 처리 시간 합)}
            ws: _ws_state() 결과
        """
        q = self.order_mgr.wait_queue
        if self.agent.feature_set != "smdp_w":
            return {"n_wait": len(q), "num_robots": self.num_robots}
        cap   = self.num_robots * self.robot_capacity
        ages  = [self.now - t.order.arrival_time for t in q]
        robot = {}
        for r in self.robots:
            if r.is_idle:
                continue
            totes = self._robot_totes[r.id]
            robot[r.id] = (sum(not t.picked for t in totes), self._robot_time_left(r),
                           self._robot_ws_work(r))
        t_free = min((v[1] for v in robot.values()), default=0.0)
        glob = [
            min((sum(ages) / len(ages) if ages else 0.0) / SMDP_WAIT_TREF, 5.0),
            len(robot) / self.num_robots,
            sum(r.idle_slots for r in self.robots) / cap,
            min(self._ia_ema / SMDP_IA_NORM, 3.0),
            min(t_free / STATE_MAX_TIME, 1.0),
        ]
        return {"glob": glob, "robot": robot, "ws": self._ws_state(),
                "n_wait": len(ages), "num_robots": self.num_robots}

    def _potential(self, ctx: Dict) -> float:
        """Φ(s) = −(배정/대기 토트의 예상 잔여 비용) / scale (smdp_w 전용, 그 외 0)

        - 운행 로봇: 실은 토트 수 × 트립 잔여시간 + c_d · 잔여 이동거리
        - 대기 토트: 단독 왕복 (시간 + c_d · 거리). 대기 자체의 비용은 환경 보상으로만 반영
        방금 도착한 미배정 토트는 행동과 무관한 상수라 제외
        """
        if self.reward_mode != "smdp_w":
            return 0.0
        cost = 0.0
        for r in self.robots:
            if r.id in ctx["robot"]:
                cost += (len(self._robot_totes[r.id]) * ctx["robot"][r.id][1]
                         + SMDP_COST_DIST * r.remaining_distance)
        solo  = self.layout.solo_rt_m * (1.0 / ROBOT_SPEED + SMDP_COST_DIST)
        cost += ctx["n_wait"] * solo
        return -cost / SMDP_REWARD_SCALE

    # 결정

    def _handle_order_event(self, ev: Event):
        """주문 도착 간격 지수이동평균 갱신 후 기본 처리 (주문마다 WS 배정, 토트마다 _on_tote_arrival)"""
        self._ia_ema  += SMDP_IA_EMA * ((self.now - self._last_arr) - self._ia_ema)
        self._last_arr = self.now
        super()._handle_order_event(ev)

    def _on_tote_arrival(self, tote: Tote):
        """새 토트를 에이전트가 고른 로봇에 배정하거나 대기 큐에 넣음

        학습 모드이면 직전 결정의 전이를 완성해 _transitions에 추가, 이번 결정은 _pending에 보류
        """
        eligible = self._eligible_robots()
        can_wait = self.with_wait and any(not r.is_idle for r in self.robots)
        if not eligible and not can_wait:
            # legacy/smdp: 후보가 없으면 결정 아님. smdp_w: 후보가 없으면 운행 로봇이 있어 can_wait
            self.order_mgr.enqueue(tote, self.now)
            return
        ctx    = self._context()
        deltas = [self._insertion_delta(r, tote) for r in eligible]
        feats  = self.agent.build_candidates(tote.loc, tote.ws_id, eligible, deltas,
                                             self.layout, ctx, now=self.now, with_wait=can_wait)
        idx = self.agent.select_index(feats)
        if self.recorder is not None and hasattr(self.recorder, "on_rl_decision"):
            self.recorder.on_rl_decision(self, tote, eligible, feats, idx, deltas)

        phi = self._potential(ctx)
        self._close_interval(phi)

        reward = None
        if idx == len(eligible):
            self.order_mgr.enqueue(tote, self.now)
            self._n_wait   += 1
            self._n_forced += int(not eligible)
        else:
            robot = eligible[idx]
            if self.reward_mode == "legacy":
                reward = compute_reward(robot, self.layout, deltas[idx], eligible,
                                        self.order_mgr.queue_size(), len(eligible),
                                        self.robot_capacity, **self.reward_cfg)
                self.total_reward += reward
            self._extra_travel += deltas[idx]
            self._assign_tote(robot, tote)

        if not self.eval_mode and self._pending is not None:
            feat_prev, r_prev, g_prev = self._pending_transition(self.now)
            self._transitions.append((feat_prev, r_prev, feats, False, g_prev))
        self._pending   = (feats[idx], self.now, reward)
        self._smdp_cost = 0.0
        self._phi_prev  = phi

    # 보상

    def _on_interval(self, t0: float, t1: float, n_in_system: int, n_busy: int):
        """smdp 계열: [t0, t1] 구간 비용을 직전 결정 시각 t_k 기준으로 할인해 누적

        smdp_w는 대기 토트 나이가 선형 증가해 비용률이 c0 + m · (t − t0)
        ∫ e^{−β(t−t_k)} (c0 + m · (t − t0)) dt
          = c0 · (E0 − E1) / β + m · (E0 − E1 · (1 + β · (t1 − t0))) / β²
        smdp는 m = 0, 대기 항 없음
        """
        if self.reward_mode == "legacy" or self._pending is None:
            return
        t_k = self._pending[1]
        c0  = n_in_system + SMDP_COST_DIST * ROBOT_SPEED * n_busy
        m   = 0.0
        if self.with_wait:
            q   = self.order_mgr.wait_queue
            w   = SMDP_WAIT_KAPPA / SMDP_WAIT_TREF
            c0 += w * sum(t0 - t.order.arrival_time for t in q)
            m   = w * len(q)
        b   = SMDP_BETA
        e0  = math.exp(-b * (t0 - t_k))
        e1  = math.exp(-b * (t1 - t_k))
        self._smdp_cost += c0 * (e0 - e1) / b + m * (e0 - e1 * (1.0 + b * (t1 - t0))) / (b * b)

    def _close_interval(self, phi_next: float):
        """smdp 계열: 직전 결정 이후 누적 비용(+ smdp_w shaping)을 보상으로 확정

        total_reward에는 shaping을 뺀 환경 보상만 합산

        입력:
            phi_next: 다음 결정 전 상태의 퍼텐셜 (에피소드 종료 시 0, smdp_w 외에는 항상 0)
        """
        if self.reward_mode == "legacy" or self._pending is None:
            return
        feat, t_k = self._pending[0], self._pending[1]
        r_env = -self._smdp_cost / SMDP_REWARD_SCALE
        self.total_reward += r_env
        g = math.exp(-SMDP_BETA * (self.now - t_k))
        self._pending = (feat, t_k, r_env + g * phi_next - self._phi_prev)

    def _pending_transition(self, t_next: float) -> Tuple[np.ndarray, float, float]:
        """보류 중인 직전 결정을 (feat, reward, discount)로 변환

        입력:
            t_next: 다음 결정 시각 (s)
        출력:
            (선택 후보 특징, 보상, 할인 계수 (legacy: γ, smdp 계열: e^{−βτ}))
        """
        feat, t_k, r = self._pending
        if self.reward_mode == "legacy":
            return feat, r, GAMMA
        return feat, r, math.exp(-SMDP_BETA * (t_next - t_k))

    def build_result(self, label: str, episode: int, lam: float) -> EpisodeResult:
        """SimulatorBase.build_result()에 보상(완료 주문 1건당), ε, wait 집계 추가"""
        res = super().build_result(label, episode, lam)
        res.reward        = self.total_reward / max(1, self.order_mgr.completed_count())
        res.epsilon       = 0.0 if self.eval_mode else self.agent.epsilon
        res.n_wait_action = self._n_wait
        res.n_forced_wait = self._n_forced
        return res
