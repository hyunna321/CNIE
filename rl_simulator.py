"""DQNAgent가 토트마다 로봇을 고르는 RL 시뮬레이터 (SMDP)

결정 단위는 토트 (주문 도착 시 그 주문의 토트마다 1번). 한 주문의 토트들은 같은 시각의 연속 결정
행동 = 후보 로봇 (빈 슬롯이 있는 로봇, SimulatorBase._eligible_robots) 중 하나
후보가 없으면 결정 없이 토트를 대기 큐에 넣고, 트립 완료 시 FIFO로 방출 (결정 아님)

보상은 arm(rewards.Arm)이 정함: 보상 객체(rewards.REWARDS)가 결정/큐/구간 훅을 받고,
사후 계산 보상(local)은 에피소드 종료 후 전이에 채움. 전이 할인은 arm.discount
  - "time": e^{−βτ} (τ = 다음 결정까지 시간), "decision": 결정당 γ
기록 통계: n_decisions, avg_candidates, n_qpen_active (대기 큐가 비어 있지 않은 결정 = legacy 큐 압박 항 활성),
  n_depletion_active (legacy 소진 페널티 조건) — 보상과 무관하게 train.csv / eval.csv에 기록
"""
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from agents import DQNAgent
from environment import Robot
from simulator import SimulatorBase, Tote
from metrics import EpisodeResult
from rewards import ARMS, DEFAULT_ARM, REWARDS, Arm
from constant import (
    SIM_TIME, NUM_ROBOTS, ROBOT_CAPACITY, GAMMA, STATE_MAX_TIME, GLOBAL_ROBOT_DIM,
)


class RLSimulator(SimulatorBase):
    """DQNAgent 배정 시뮬레이터

    한 결정의 전이는 다음 결정 시점의 상태가 나와야 완성되므로 직전 결정을 _pending에 보관
    _pending = (상태 key, 결정 시각, 보상)  (사후 계산 보상이면 보상 자리는 None, 에피소드 끝에 채움)
    """

    def __init__(self, layout, agent: DQNAgent, lam: float, seed: int,
                 sim_time: float = SIM_TIME, num_robots: int = NUM_ROBOTS,
                 robot_capacity: int = ROBOT_CAPACITY, eval_mode: bool = False,
                 arm: Optional[Arm] = None, **sim_opts):
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
            arm: 보상 설정 (None이면 기본 arm)
            sim_opts: SimulatorBase 추가 인자 (route)
        """
        arm = arm or ARMS[DEFAULT_ARM]
        super().__init__(layout, lam, seed, sim_time, num_robots, robot_capacity,
                         **{"tote_order": arm.tote_order, **sim_opts})
        self.agent     = agent
        self.eval_mode = eval_mode
        self.arm       = arm
        self.order_feats = arm.features == "order"
        self.reward    = REWARDS[self.arm.reward](self, self.arm)
        self._init_rl_state()

    def _init_rl_state(self):
        self.total_reward = 0.0
        self._pending: Optional[tuple] = None
        self._transitions: List[tuple] = []
        self.reward.reset()
        # 결정 통계: 결정 수, 후보 수 합, 큐 압박 항 / 소진 페널티 조건 횟수
        self._n_dec = self._n_cands = self._n_qpen = self._n_depl = 0

    def reset(self, seed: int):
        """에피소드 상태, 누적 보상, 보류 전이 초기화"""
        super().reset(seed)
        self._init_rl_state()

    # 학습 곡선용 보상 통계 (rewards.Reward 참고)
    @property
    def last_mu(self) -> float:
        return self.reward.last_mu

    @property
    def last_mean_cost(self) -> float:
        return self.reward.last_mean_cost

    @property
    def last_cost_terms(self) -> Tuple[float, float, float]:
        return self.reward.last_cost_terms

    def run(self) -> List[tuple]:
        """에피소드 1회를 끝까지 실행

        출력:
            학습용 전이 리스트 [(key, reward, next_key, done, discount), ...] (eval_mode이면 빈 리스트)
        """
        self._transitions = []
        saved_epsilon = self.agent.epsilon
        if self.eval_mode:
            self.agent.epsilon = 0.0

        super().run()

        # 마지막 결정은 다음 상태가 없으므로 done=True로 마감
        if not self.eval_mode and self._pending is not None:
            key, r, g = self._pending_transition(self.sim_time)
            self._transitions.append((key, r, None, True, g))
        self._pending = None
        rewards = self.reward.finalize()
        if rewards is not None:
            self.total_reward = sum(rewards)
            if self._transitions:
                assert len(self._transitions) == len(rewards), (len(self._transitions), len(rewards))
                self._transitions = [(k, r, nk, d, g) for (k, _, nk, d, g), r in zip(self._transitions, rewards)]

        if self.eval_mode:
            self.agent.epsilon = saved_epsilon
        return self._transitions

    # 상태

    def _context(self) -> Dict:
        """결정 시점의 상태 (DQNAgent.build_candidates 입력)

        출력:
            n_wait: 대기 큐 길이 (토트)
            n_in_system: 시스템 내 주문 수 N(t) (도착 − 완료)
            n_busy: 운행 로봇 수 (idle 아님)
            num_robots: 로봇 수
            onboard: {로봇 id: 아직 하역하지 않은 토트 수} (BL6 한계비용의 적재 토트 수와 같은 값)
        """
        return {"n_wait": self.order_mgr.queue_size(), "n_in_system": self._n_in_system,
                "n_busy": sum(not r.is_idle for r in self.robots), "num_robots": self.num_robots,
                "onboard": {r.id: len(self._robot_totes[r.id]) for r in self.robots}}

    def _global_state(self, tote: Tote, eligible: List[Robot]) -> Tuple[np.ndarray, np.ndarray]:
        """전역 상태: 전 로봇 특징 집합 R과 WS·토트·컨텍스트 벡터 W

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
        if self.order_feats:
            ctx += self._order_globals()
        W = np.concatenate([ws.ravel(), np.eye(K, dtype=np.float32)[tote.ws_id],
                            np.asarray(ctx, dtype=np.float32)])
        return R, W

    # 주문 특징 (arm.features = "order")

    def _order_globals(self) -> List[float]:
        """전역: 최근 5분(300 s) 도착 주문 수 / (M·C) (상한 3), 빈 idle 로봇 수 / M"""
        import bisect
        M, C = self.num_robots, self.robot_capacity
        orders = self.order_mgr.all_orders
        recent = len(orders) - bisect.bisect_left(orders, self.now - 300.0, key=lambda o: o.arrival_time)
        empty_idle = sum(r.is_idle and r.idle_slots == C for r in self.robots)
        return [min(recent / (M * C), 3.0), empty_idle / M]

    def _order_features(self, tote: Tote, eligible: List[Robot]) -> List[List[float]]:
        """후보별 주문 계획 특징 (planning 모듈, 정규화 STATE_MAX_TIME)

        [ (Ť_{k,m} − t) (0~5), slack = Ť_{k,m} − Ť_sib_max (−3~3), is_bottleneck (Ť_{k,m} > Ť_sib_max),
          이 배정으로 계획 완료가 늦어지는 다른 주문 수 / C (상한 2), Delta_order (−3~5) ]
        Ť_sib_max = max(t, k의 다른 토트 중 배정·미처리 토트의 Ť) (planning.order_completion과 같은 정의)
        """
        from planning import planned_completions, delta_order, order_completion
        t, S, C = self.now, STATE_MAX_TIME, self.robot_capacity
        pc = planned_completions(self, "estimate")
        sib = order_completion(tote.order, pc, t)
        out = []
        for r in eligible:
            d, tk, n_late = delta_order(self, tote, r, pc)
            out.append([min(max((tk - t) / S, 0.0), 5.0), min(max((tk - sib) / S, -3.0), 3.0),
                        float(tk > sib + 1e-9), min(n_late / C, 2.0), min(max(d / S, -3.0), 5.0)])
        return out

    # 결정

    def _on_tote_arrival(self, tote: Tote):
        """새 토트를 에이전트가 고른 후보 로봇에 배정 (후보가 없으면 결정 없이 대기 큐)

        학습 모드이면 직전 결정의 전이를 완성해 _transitions에 추가, 이번 결정은 _pending에 보류
        """
        t0 = time.perf_counter()
        eligible = self._eligible_robots()
        if not eligible:
            self.reward.on_queue(tote)
            self.order_mgr.enqueue(tote, self.now)
            return
        ctx    = self._context()
        deltas = [self._insertion_delta(r, tote) for r in eligible]
        extra  = self._order_features(tote, eligible) if self.order_feats else None
        feats  = self.agent.build_candidates(tote.loc, eligible, deltas, self.layout, ctx, now=self.now,
                                             order_feats=extra)
        self.agent.set_global(*self._global_state(tote, eligible))
        idx = self.agent.select_index(feats)
        t_sel = time.perf_counter()
        if self.recorder is not None and hasattr(self.recorder, "on_rl_decision"):
            self.recorder.on_rl_decision(self, tote, eligible, feats, idx, deltas)
        t_rec = time.perf_counter() - t_sel           # 기록기 시간은 결정 지연에서 뺌

        robot = eligible[idx]
        self._count_decision(robot, eligible)
        reward = self.reward.on_decision(robot, tote, deltas[idx], eligible)   # 배정 직전 상태
        if reward is not None:
            self.total_reward += reward
        self._extra_travel += deltas[idx]
        self._assign_tote(robot, tote)
        self._record_latency(t0, t_sel, time.perf_counter() - t_rec)

        key = self.agent.last_key
        if not self.eval_mode and self._pending is not None:
            key_prev, r_prev, g_prev = self._pending_transition(self.now)
            self._transitions.append((key_prev, r_prev, key, False, g_prev))
        self._pending = (key, self.now, reward)

    def _count_decision(self, robot: Robot, eligible: List[Robot]):
        """결정 통계 집계 (배정 직전 상태, legacy 보상의 큐 압박 / 소진 페널티 조건과 같음)"""
        C = self.robot_capacity
        self._n_dec   += 1
        self._n_cands += len(eligible)
        self._n_qpen  += self.order_mgr.queue_size() > 0
        if robot.is_idle and not any(r.id != robot.id and r.is_idle and r.idle_slots == C for r in eligible):
            self._n_depl += 1

    def _on_interval(self, t0: float, t1: float, n_in_system: int, n_busy: int):
        """사건 사이 구간을 보상 객체에 전달 (구간 적분형 보상용)"""
        self.reward.on_interval(t0, t1, n_in_system, n_busy)

    def _pending_transition(self, t_next: float) -> Tuple[tuple, Optional[float], float]:
        """보류 중인 직전 결정을 (key, reward, discount)로 변환 (t_next: 다음 결정 시각)"""
        key, t_k, r = self._pending
        if self.arm.discount == "decision":
            return key, r, GAMMA
        return key, r, math.exp(-self.arm.beta_value * (t_next - t_k))

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
