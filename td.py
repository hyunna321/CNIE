"""관측 기반 TD 학습: ring replay buffer, TDAgent (BL6 잔차 네트워크는 agents.ScoreNet 그대로), TDSimulator

전이는 결정 시점의 관측만으로 만듦 (env.rl의 "interval" 보상과 함께 씀)
"local" 보상(A0)이면 τ_k는 다음 결정에서, r_k는 결정 k의 토트와 k 뒤 큐잉 토트가 끝날 때 채우고, 둘 다 채워진 칸만 샘플링
  (값은 에피소드 후 LocalReward.finalize()와 같음. 에피소드 끝에 남은 칸은 finalize() 값으로 채움)
  s_k: 결정 k 직전 관측, a_k: 고른 후보, r_k: [t_k, t_k+1] 구간의 실제 비용, τ_k = t_k+1 − t_k
  r_k는 다음 결정 시점에 확정되므로 토트 완료시각이나 에피소드 종료를 기다리지 않음

n-step target (Double DQN)
  G = Σ_{j<n} e^{−β(t_i+j − t_i)} r_i+j,  y = G + e^{−β(t_i+n − t_i)} Q_target(s_i+n, argmax_{a∈A(s_i+n)} Q_online(s_i+n, a))
  terminal 없음: 에피소드는 데이터 구간일 뿐이고, 다음 관측이 없는 마지막 결정은 샘플링되지 않음 (절단)
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents import DQNAgent, GRAD_CLIP, HUBER_DELTA, REPLAY_BUFFER_SIZE
from env.reserve import ReserveMixin
from env.rl import RLReserveSimulator, RLSimulator


class RingReplay:
    """관측을 시간순으로 저장하는 ring buffer

    s'을 따로 저장하지 않고 다음 칸(i + n)을 읽으며, n-step 값은 샘플링 시점에 계산
    open(key)로 관측을 넣고 다음 결정에서 close(r, τ)로 그 전이를 확정. new_segment()는 에피소드 경계
    시작 칸 i는 i .. i+n이 같은 구간에 연속으로 있고 i .. i+n−1이 확정되었을 때만 샘플링됨
    """

    def __init__(self, capacity: int, n_step: int, beta: float, seed: int):
        self.capacity, self.n_step, self.beta = capacity, n_step, beta
        self.rng = np.random.default_rng(seed)
        self.cand = None         # 첫 open에서 할당 (학습 전 프로세스 풀로 넘길 때 비어 있도록)
        self.head = self.count = 0
        self.seg_id = 0
        self.pending: Optional[int] = None

    def _alloc(self, pad: np.ndarray, mask: np.ndarray, R: np.ndarray, W: np.ndarray):
        c = self.capacity
        self.cand   = np.zeros((c, *pad.shape), np.float32)
        self.mask   = np.zeros((c, *mask.shape), np.bool_)
        self.R      = np.zeros((c, *R.shape), np.float32)
        self.W      = np.zeros((c, *W.shape), np.float32)
        self.act    = np.zeros(c, np.int64)
        self.r      = np.zeros(c, np.float64)
        self.tau    = np.zeros(c, np.float64)
        self.seg    = np.full(c, -1, np.int64)
        self.seq    = np.full(c, -1, np.int64)      # 쓰인 순번 (연속 여부 판정)
        self.closed = np.zeros(c, np.bool_)
        self.has_r  = np.zeros(c, np.bool_)       # 지연 확정 보상 (local): 보상과 τ를 따로 채움
        self.has_t  = np.zeros(c, np.bool_)

    def __len__(self):
        return min(self.count, self.capacity)

    def new_segment(self):
        """새 에피소드 시작. 직전 구간의 미확정 관측은 그대로 두며 샘플링되지 않음"""
        self.seg_id += 1
        self.pending = None

    def open(self, key: tuple):
        """결정 관측 저장. key = DQNAgent.last_key (후보 특징 패딩, 마스크, 선택 인덱스, R, W)"""
        pad, mask, a, R, W = key
        if self.cand is None:
            self._alloc(pad, mask, R, W)
        i = self.head
        self.cand[i], self.mask[i], self.act[i], self.R[i], self.W[i] = pad, mask, a, R, W
        self.seg[i], self.seq[i], self.closed[i] = self.seg_id, self.count, False
        self.has_r[i] = self.has_t[i] = False
        self.pending = i
        self.head = (i + 1) % self.capacity
        self.count += 1
        return i

    def close(self, reward: float, tau: float):
        """가장 최근 open한 전이의 보상과 다음 결정까지 시간 확정"""
        i = self.pending
        self.r[i], self.tau[i], self.closed[i] = reward, tau, True

    def set_tau(self, i: int, tau: float):
        """지연 확정용: 칸 i의 다음 결정까지 시간 (보상이 이미 있으면 확정)"""
        self.tau[i], self.has_t[i] = tau, True
        self.closed[i] = self.has_r[i]

    def set_reward(self, i: int, reward: float):
        """지연 확정용: 칸 i의 보상 (τ가 이미 있으면 확정)"""
        self.r[i], self.has_r[i] = reward, True
        self.closed[i] = self.has_t[i]

    def _valid(self, idx: np.ndarray) -> np.ndarray:
        n = self.n_step
        offs = (idx[:, None] + np.arange(n + 1)) % self.capacity
        s = self.seq[offs]
        return ((s[:, 0] >= 0) & (s == s[:, [0]] + np.arange(n + 1)).all(1)
                & (self.seg[offs] == self.seg[offs[:, [0]]]).all(1) & self.closed[offs[:, :-1]].all(1))

    def sample(self, batch_size: int) -> Optional[dict]:
        """유효한 시작 칸에서 균등 복원 추출. 부족하면 None"""
        if self.count < batch_size + self.n_step + 1:
            return None
        got, need = [], batch_size
        for _ in range(100):
            draw = self.rng.integers(0, len(self), 4 * batch_size)
            ok = draw[self._valid(draw)]
            got.append(ok[:need])
            need -= len(got[-1])
            if need == 0:
                break
        if need > 0:
            return None
        return self._batch(np.concatenate(got))

    def _batch(self, idx: np.ndarray) -> dict:
        n, c = self.n_step, self.capacity
        offs = (idx[:, None] + np.arange(n)) % c
        dec  = np.exp(-self.beta * np.cumsum(self.tau[offs], axis=1))
        disc = np.concatenate([np.ones((len(idx), 1)), dec[:, :-1]], axis=1)
        nxt  = (idx + n) % c
        return {"cand": self.cand[idx], "R": self.R[idx], "W": self.W[idx], "act": self.act[idx],
                "G": (disc * self.r[offs]).sum(1).astype(np.float32), "boot": dec[:, -1].astype(np.float32),
                "next_cand": self.cand[nxt], "next_mask": self.mask[nxt], "next_R": self.R[nxt], "next_W": self.W[nxt]}

    def state(self) -> dict:
        names = ("cand", "mask", "R", "W", "act", "r", "tau", "seg", "seq", "closed", "has_r", "has_t")
        return {"arrays": None if self.cand is None else {k: getattr(self, k) for k in names},
                "head": self.head, "count": self.count, "seg_id": self.seg_id, "pending": self.pending,
                "rng": self.rng.bit_generator.state}

    def load_state(self, s: dict):
        for k, v in (s["arrays"] or {}).items():
            setattr(self, k, v)
        if self.cand is not None and not hasattr(self, "has_r"):       # 이전 체크포인트
            self.has_r, self.has_t = self.closed.copy(), self.closed.copy()
        self.head, self.count, self.seg_id, self.pending = s["head"], s["count"], s["seg_id"], s["pending"]
        self.rng.bit_generator.state = s["rng"]


class TDAgent(DQNAgent):
    """DQNAgent의 네트워크·특징·행동 선택을 그대로 쓰고 replay와 업데이트만 RingReplay로 교체"""

    def __init__(self, robot_capacity: int, dist_norm: float, beta: float, **kw):
        super().__init__(robot_capacity, dist_norm, **kw)
        self.replay = RingReplay(kw.get("buffer_size", REPLAY_BUFFER_SIZE), kw["n_step"], beta, kw["seed"])

    def train_step(self) -> Optional[float]:
        """uniform 배치 하나로 Double DQN 업데이트 1회. 샘플이 부족하면 None"""
        b = self.replay.sample(self.batch_size)
        if b is None:
            return None
        t = {k: torch.as_tensor(v) for k, v in b.items()}
        q = self.policy_net(t["cand"], t["R"], t["W"]).gather(1, t["act"][:, None]).squeeze(1)
        with torch.no_grad():
            nx, nm, nR, nW = t["next_cand"], t["next_mask"], t["next_R"], t["next_W"]
            best   = self.policy_net(nx, nR, nW).masked_fill(~nm, float("-inf")).argmax(1)
            target = t["G"] + t["boot"] * self.target_net(nx, nR, nW).gather(1, best[:, None]).squeeze(1)
        self.last_stats = (q.mean().item(), (q - target).abs().mean().item())
        loss = F.huber_loss(q, target, delta=HUBER_DELTA)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), GRAD_CLIP)
        self.optimizer.step()
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()

    def full_state(self) -> dict:
        return {**super().full_state(), "replay": self.replay.state()}

    def load_full_state(self, s: dict):
        super().load_full_state(s)
        self.replay.load_state(s["replay"])


class TDSimulator(RLSimulator):
    """결정마다 직전 전이를 확정해 agent.replay에 넣는 학습 시뮬레이터 (평가는 RLSimulator)

    arm.update == "online"이면 결정 update_every개마다 에피소드 도중 train_step
    """
    update_every = 8

    def _init_rl_state(self):
        super()._init_rl_state()
        self._open = False
        self._t_prev = 0.0
        self.n_transitions = 0
        self.n_updates = self.n_trained = 0
        self.q_sum = self.td_sum = 0.0
        self.last_loss = None

    def run(self):
        self.agent.replay.new_segment()
        self._slots = []          # local 보상: [(결정 번호, ring 칸, 칸 순번)] 보상 미확정
        self._k = -1
        out = super().run()
        if self._slots:           # 에피소드 끝: 미완료 토트는 T로 자른 finalize() 값 (PER 버전과 같음)
            fin = self.reward.finalize()
            for k, i, q in self._slots:
                self._set_reward(i, q, fin[k])
            self._slots = []
        return out

    def _set_reward(self, i: int, seq: int, r: float):
        replay = self.agent.replay
        if replay.seq[i] == seq:          # ring이 한 바퀴 돌아 덮어쓴 칸이면 버림
            replay.set_reward(i, r)

    def _after_decision(self, reward):
        replay = self.agent.replay
        closed = self._open
        delayed = self.arm.reward == "local"
        if closed:
            if delayed:
                replay.set_tau(replay.pending, self.now - self._t_prev)
            else:
                replay.close(self.reward.closed_value(), self.now - self._t_prev)
            self.n_transitions += 1
        i = replay.open(self.agent.last_key)
        self._open, self._t_prev = True, self.now
        if delayed:
            self._k += 1
            self._slots.append((self._k, i, int(replay.seq[i])))
            keep = []
            for k, j, q in self._slots:
                if self.reward.ready(k):
                    self._set_reward(j, q, self.reward.value(k))
                else:
                    keep.append((k, j, q))
            self._slots = keep
        if closed and self.arm.update == "online" and self.n_transitions % self.update_every == 0:
            self.n_updates += 1
            loss = self.agent.train_step()
            if loss is not None:
                self.last_loss = loss
                self.q_sum  += self.agent.last_stats[0]
                self.td_sum += self.agent.last_stats[1]
                self.n_trained += 1


class TDReserveSimulator(ReserveMixin, TDSimulator):
    """예약 행동(arm.reserve)이 있는 TDSimulator"""
    _on_tote_arrival = RLSimulator._on_tote_arrival
    build_result = RLReserveSimulator.build_result


def rl_sim_class(arm, train: bool = True):
    """arm에 맞는 RL 시뮬레이터 클래스 (train=False이면 평가용: 전이를 모으지 않으므로 ring buffer 구분 없음)"""
    if arm is None:
        return RLSimulator
    if train and arm.buffer == "ring":
        return TDReserveSimulator if arm.reserve else TDSimulator
    return RLReserveSimulator if arm.reserve else RLSimulator
