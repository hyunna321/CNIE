"""전역 상태 + BL6 잔차 Double DQN 에이전트 (dueling 없음)와 (토트, 후보 로봇) 특징

- 결정 단위는 토트 (BL1~BL6과 같음). 토트의 WS는 주문 도착 시 정해져 있음
- 행동: 후보 로봇 리스트 내 인덱스 (행동 공간 = 후보 로봇)
- Q_i = α · f([후보 특징_i, s]) − k · mc_i   (ScoreNet 설명)
  후보마다 같은 네트워크로 Q값을 내므로 로봇 수와 무관하게 네트워크 크기 고정
- 후보 특징 (candidate_feature_dim): 토트 상대좌표, 삽입 detour, 로봇 상태, 컨텍스트, BL6 한계비용 구성요소
- 전역 상태: RLSimulator._global_state가 결정마다 set_global(R, W)로 넘김
- 전이의 상태 = key (후보 특징 패딩, 마스크, 선택 인덱스, R, W). 다음 상태도 key (같은 객체 공유)
"""
import math
import random
import warnings
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from env.battery import BATTERY_CAND_DIM, BATTERY_CTX_DIM
from env.robot import Robot
from env.orders import SIM_TIME, DEFAULT_SEED
from env.layout import NUM_WORKSTATIONS, ROBOT_SPEED
from env.simulator import marginal_cost_terms
from env.rl import GAMMA, N_STEP, GLOBAL_ROBOT_DIM, STATE_MAX_TIME, REWARD_SCALE

# 하이퍼파라미터 (모든 arm 공통)
LR                    = 3e-4
BATCH_SIZE            = 128
REPLAY_BUFFER_SIZE    = 50_000
TARGET_UPDATE_STEPS   = 200
EPSILON               = 0.05    # 고정 ε (BL6 근처에서 데이터 수집)
GRAD_CLIP             = 10.0
HUBER_DELTA           = 1.0     # Huber 손실 전환점 (보상 단위)
MAX_CANDIDATES        = 8
MAX_CANDIDATES_MARGIN = 3       # max_candidates = max(MAX_CANDIDATES, num_robots + margin)
RESIDUAL_SCALE        = 0.1     # Q = α · f − k · mc_BL6 의 α 초기값 (BL6에서 천천히 벗어나도록)
ALPHA_MAX             = 1.0     # α 상한 (α = ALPHA_MAX · sigmoid(logit))
PER_ALPHA             = 0.6
PER_BETA_START        = 0.4
PER_BETA_END          = 1.0
PER_PRIORITY_EPS      = 1e-6
DELTA_NORM_CLIP       = 3.0     # 정규화된 삽입 detour 상한
TOTAL_EPISODES        = 1000

BL6_EXTRA_DIM = 5     # 후보 특징 끝의 BL6 한계비용 구성요소 수 (마지막 열 = 한계비용 합계, prior에 사용)
ORDER_CAND_DIM = 5    # features="order" 후보 특징 (RLSimulator._order_features): Ť_{k,m}, slack, is_bottleneck,
                      #   늦어지는 다른 주문 수, Delta_order. BL6 구성요소 바로 앞에 들어감
ORDER_CTX_DIM = 2     # features="order" 전역 특징 (W 끝): 최근 5분 도착 수, 빈 idle 로봇 수


RESERVE_CAND_DIM = 3  # features="reserve*" 후보 특징 (RLSimulator._reserve_features): 예약 후보 여부, 해제까지 시간, 예약 수
RESERVE_CTX_DIM  = 1  # features="reserve*" 전역 특징: 전체 예약 토트 수 / (M·C)


def order_feature_dims(features: str) -> Tuple[int, int]:
    """특징 세트별 (후보 특징 추가 차원, 전역 특징 추가 차원)"""
    return {"order": (ORDER_CAND_DIM, ORDER_CTX_DIM),
            "battery": (BATTERY_CAND_DIM, BATTERY_CTX_DIM),
            "reserve": (RESERVE_CAND_DIM, RESERVE_CTX_DIM),
            "reserve_battery": (RESERVE_CAND_DIM + BATTERY_CAND_DIM, RESERVE_CTX_DIM + BATTERY_CTX_DIM),
            }.get(features, (0, 0))


def candidate_feature_dim(robot_capacity: int, features: str = "base") -> int:
    """(토트, 후보 로봇) 쌍 특징벡터의 차원

    토트 상대좌표 2 + 삽입 detour 1 + 로봇 상태 (5 + capacity*2)
    + 컨텍스트 4 (큐 길이, 경과시간, 시스템 내 주문 수, 운행 로봇 비율) [+ 주문 특징 5] + BL6 구성요소 5
    """
    return 2 + 1 + (5 + robot_capacity * 2) + 4 + order_feature_dims(features)[0] + BL6_EXTRA_DIM


def global_ctx_dim(num_workstations: int, features: str = "base") -> int:
    """전역 상태 W 벡터 차원: WS별 4 + 토트 WS one-hot K + 컨텍스트 4 [+ 주문 전역 특징 2]"""
    return num_workstations * 4 + num_workstations + 4 + order_feature_dims(features)[1]


def _mlp(*dims: int) -> nn.Sequential:
    """Linear-ReLU 반복, 마지막 Linear 뒤에는 활성화 없음"""
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])


class ScoreNet(nn.Module):
    """전역 상태 + BL6 잔차 후보 스코어링 네트워크

    s   = enc([mean_i φ(R_i), max_i φ(R_i), W])        (R: 전 로봇 특징 집합, W: WS 상태 + 토트 WS + 컨텍스트)
    Q_i = α · f([후보 특징_i, s]) − k · mc_i            (mc_i: 후보 특징 마지막 열 = BL6 한계비용, k = mc_scale)
    α = ALPHA_MAX · sigmoid(alpha_logit), 초기값 RESIDUAL_SCALE
    f의 마지막 층을 0으로 초기화해 학습 시작 시 argmax = BL6
    """

    def __init__(self, feat_dim: int, ctx_dim: int, mc_scale: float,
                 residual_scale: float = RESIDUAL_SCALE, alpha_max: float = ALPHA_MAX):
        super().__init__()
        self.phi = _mlp(GLOBAL_ROBOT_DIM, 64, 64)
        self.enc = nn.Sequential(_mlp(128 + ctx_dim, 128, 64), nn.ReLU())
        self.f   = _mlp(feat_dim + 64, 128, 128, 64, 1)
        nn.init.zeros_(self.f[-1].weight)
        nn.init.zeros_(self.f[-1].bias)
        self.register_buffer("mc_scale", torch.tensor(float(mc_scale)))
        assert 0 < residual_scale < alpha_max, (residual_scale, alpha_max)
        self.alpha_max = alpha_max
        p = residual_scale / alpha_max
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(p / (1 - p))))

    def _alpha(self) -> torch.Tensor:
        return self.alpha_max * torch.sigmoid(self.alpha_logit)

    @property
    def alpha(self) -> float:
        """현재 잔차 계수 α"""
        return float(self._alpha().detach())

    def state(self, R: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """R (B, M, GLOBAL_ROBOT_DIM), W (B, ctx_dim) -> (B, 64)"""
        h = torch.relu(self.phi(R))
        return self.enc(torch.cat([h.mean(1), h.max(1).values, W], -1))

    def forward(self, x: torch.Tensor, R: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """x (B, N, feat_dim), R, W -> 후보 Q값 (B, N). 패딩 후보는 호출 쪽에서 마스크"""
        s = self.state(R, W)
        a = self.f(torch.cat([x, s.unsqueeze(1).expand(-1, x.shape[1], -1)], -1)).squeeze(-1)
        return self._alpha() * a - self.mc_scale * x[..., -1]


class PrioritizedReplayBuffer:
    """TD 오차 기반 우선순위 경험 재생 버퍼 (PER). 항목은 n-step 전이 (key, G, next_key, _, done, discount)"""

    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE, alpha: float = PER_ALPHA, seed: int = DEFAULT_SEED):
        self.capacity      = capacity
        self.alpha         = alpha
        self.buf           = [None] * capacity
        self.priorities    = np.zeros(capacity, dtype=np.float32)
        self.pos           = 0
        self.size          = 0
        self._max_priority = 1.0
        self.rng           = np.random.default_rng(seed)

    def push(self, *item):
        """전이 1건을 현재까지의 최대 우선순위로 저장"""
        self.buf[self.pos]        = item
        self.priorities[self.pos] = self._max_priority
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float):
        """우선순위 비례 확률로 batch_size개 (비복원) 추출

        출력:
            (항목 리스트, 인덱스, 중요도 가중치 (최댓값 1로 정규화))
        """
        prios   = self.priorities[:self.size] ** self.alpha
        probs   = prios / prios.sum()
        idxs    = self.rng.choice(self.size, batch_size, replace=False, p=probs)
        weights = (self.size * probs[idxs]) ** (-beta)
        weights /= weights.max()
        return [self.buf[i] for i in idxs], idxs, weights

    def update_priorities(self, idxs, td_errors):
        """샘플된 항목의 우선순위를 |TD 오차|로 갱신"""
        prios = np.abs(td_errors) + PER_PRIORITY_EPS
        self.priorities[idxs] = prios
        self._max_priority = max(self._max_priority, prios.max())

    def __len__(self):
        return self.size


class NStepBuffer:
    """n-step 누적 보상을 만들기 위한 임시 버퍼

    항목 형식: (key, reward, next_key, _, done, discount)
    discount: 그 전이 하나의 할인 계수 (e^{−βτ} 또는 γ)
    G = r_0 + g_0·r_1 + g_0·g_1·r_2 + ..., 부트스트랩 할인 = Π g_i
    """

    def __init__(self, n: int = N_STEP):
        self.n   = n
        self.buf = deque()

    def push(self, transition):
        self.buf.append(transition)

    def ready(self) -> bool:
        """n개 이상 쌓여 n-step 전이를 만들 수 있으면 True"""
        return len(self.buf) >= self.n

    def get(self):
        """가장 오래된 전이 기준 n-step 전이를 만들고 그 전이를 버퍼에서 제거"""
        key0 = self.buf[0][0]
        _, _, nk, nm, d, _ = self.buf[-1]
        G, disc = 0.0, 1.0
        for (_, r_i, _, _, d_i, g_i) in self.buf:
            G    += disc * r_i
            disc *= g_i
            if d_i:
                break
        self.buf.popleft()
        return (key0, G, nk, nm, d, disc)

    def flush(self):
        """에피소드 종료 시 남은 전이를 모두 n-step 전이로 변환 후 비움"""
        results = []
        while self.buf:
            results.append(self.get())
        return results


class DQNAgent:
    """전역 상태 + BL6 잔차 Double DQN (모듈 설명)"""

    def __init__(self, robot_capacity: int, dist_norm: float, seed: int = DEFAULT_SEED,
                 sim_time: float = SIM_TIME, num_robots: Optional[int] = None,
                 num_workstations: int = NUM_WORKSTATIONS, n_step: int = N_STEP, features: str = "base",
                 lr: float = LR, batch_size: int = BATCH_SIZE, buffer_size: int = REPLAY_BUFFER_SIZE,
                 target_update: int = TARGET_UPDATE_STEPS, epsilon: float = EPSILON):
        """
        입력:
            robot_capacity: 로봇 용량
            dist_norm: layout.dist_norm (BL6 특징 정규화 기준, prior 척도 mc_scale_for)
            seed: 네트워크 초기화 / 탐색 / 버퍼 샘플링 난수 시드
            sim_time: 경과시간 특징 정규화 기준 (s)
            num_robots: 로봇 수 (최대 후보 수 결정, 다중 M이면 최댓값)
            num_workstations: 워크스테이션 수 (전역 상태 차원)
            n_step: n-step 전이 길이 (arm.n_step)
            features: 특징 세트 "base" / "order" (arm.features, order_feature_dims)
            lr, batch_size, buffer_size, target_update, epsilon: 학습 계수 (train.py, 실험 설정 YAML)
        """
        self.batch_size       = batch_size
        self.target_update    = target_update
        self.robot_capacity   = robot_capacity
        self.sim_time         = sim_time
        self.num_robots       = num_robots
        self.num_workstations = num_workstations
        self.features         = features
        self.feat_dim         = candidate_feature_dim(robot_capacity, features)
        self.ctx_dim          = global_ctx_dim(num_workstations, features)
        self.max_candidates   = (max(MAX_CANDIDATES, num_robots + MAX_CANDIDATES_MARGIN)
                                 if num_robots is not None else MAX_CANDIDATES)

        # 네트워크 초기화가 전역 torch RNG를 쓰므로 생성 전에 시드 고정
        torch.manual_seed(seed)
        self.device     = torch.device("cpu")
        mc_scale        = self.mc_scale_for(dist_norm)
        self.policy_net = ScoreNet(self.feat_dim, self.ctx_dim, mc_scale).to(self.device)
        self.target_net = ScoreNet(self.feat_dim, self.ctx_dim, mc_scale).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer    = PrioritizedReplayBuffer(buffer_size, seed=seed)
        self.nstep_buf = NStepBuffer(n=n_step)

        self.epsilon  = epsilon
        self.beta     = PER_BETA_START
        self.steps    = 0
        self.rng      = random.Random(seed)
        self.mu_table: Optional[Tuple[List[float], List[float]]] = None   # local 보상 기준점 (λ 격자, μ)
        self.last_stats = (float("nan"), float("nan"))                     # 마지막 train_step (평균 Q, 평균 |TD|)
        self._global  = None
        self.last_key = None

    # BL6 prior 척도

    @staticmethod
    def mc_scale_for(dist_norm: float) -> float:
        """레이아웃 dist_norm에서 BL6 prior 척도 (prior = mc_scale · mc / (dist_norm / v) = mc / Z)"""
        return dist_norm / ROBOT_SPEED / REWARD_SCALE

    @property
    def mc_scale(self) -> float:
        """현재 policy_net의 BL6 prior 척도"""
        return float(self.policy_net.mc_scale)

    def set_mc_scale(self, dist_norm: float):
        """평가 레이아웃의 dist_norm으로 policy/target의 mc_scale을 다시 설정

        mc_scale은 학습 레이아웃 값으로 체크포인트에 저장되지만, 주차장 행 수가 M·K에 따라 달라 평가 레이아웃의
        dist_norm이 다르면 prior가 정확히 mc / Z가 아님. 학습 레이아웃과 dist_norm이 같으면 같은 값이 들어감
        """
        v = self.mc_scale_for(dist_norm)
        for net in (self.policy_net, self.target_net):
            net.mc_scale.fill_(v)

    def local_mu(self, lam: float) -> Optional[float]:
        """local 보상 고정 기준점 μ(λ) (mu_table 선형 보간, 범위 밖은 끝값). 없으면 None"""
        if self.mu_table is None:
            return None
        return float(np.interp(lam, *self.mu_table))

    # 특징

    def build_candidates(self, tote_loc: Tuple[int, int], robots: List[Robot], deltas: List[float],
                         layout, ctx: Dict, now: float = 0.0,
                         order_feats: Optional[List[List[float]]] = None,
                         bl6_terms: Optional[List[Tuple[float, float, float, float, float]]] = None) -> np.ndarray:
        """후보 로봇마다 (토트, 로봇) 특징벡터

        - 토트 위치의 상대좌표 2 (후보 로봇의 트립 마지막 WS 기준), 삽입 detour 1
        - 로봇 상태 5 + capacity*2 (Robot.state_vector)
        - 컨텍스트 4 (후보 공통): 대기 큐 길이 / (M·C), 경과시간 / sim_time, 시스템 내 주문 수 / (M·C) (상한 5),
          운행 로봇 수 / M
        - (features="order") 주문 계획 특징 ORDER_CAND_DIM (order_feats, RLSimulator._order_features)
        - BL6 한계비용 구성요소 5 (_bl6_features)

        입력:
            tote_loc: 토트 피킹 셀
            robots: 후보 로봇 리스트
            deltas: 후보별 삽입 추가 이동거리 (m, SimulatorBase._insertion_delta)
            layout: GridWarehouseLayout
            ctx: RLSimulator._context() 결과 (n_wait, n_in_system, n_busy, num_robots, onboard)
            now: 현재 시각 (s)
            bl6_terms: 후보별 BL6 구성요소 원값 (적재 비율, 적재 지연 s, 남은 거리 m, 추가 거리 m, 비용 합계 s).
                       주면 marginal_cost_terms 대신 사용 (예약 후보의 env.reserve.reserve_cost_terms, prior도 이 합계)
        출력:
            (len(robots), feat_dim) float32 배열
        """
        C, cn, dn = self.robot_capacity, layout.coord_norm, layout.dist_norm
        M         = ctx["num_robots"]
        base_ctx  = [min(ctx["n_wait"] / (M * C), 1.0), min(now / self.sim_time, 1.0),
                     min(ctx["n_in_system"] / (M * C), 5.0), ctx["n_busy"] / M]
        feats = np.zeros((len(robots), self.feat_dim), dtype=np.float32)
        for i, (r, delta) in enumerate(zip(robots, deltas)):
            wx, wy = r.workstation
            feats[i] = ([(tote_loc[0] - wx) / cn, (tote_loc[1] - wy) / cn, min(delta / dn, DELTA_NORM_CLIP)]
                        + r.state_vector(STATE_MAX_TIME) + base_ctx
                        + (list(order_feats[i]) if order_feats is not None else [])
                        + (self._norm_bl6(*bl6_terms[i], layout) if bl6_terms is not None
                           else self._bl6_features(r, delta, layout, ctx["onboard"][r.id])))
        return feats

    def _norm_bl6(self, load: float, delay: float, remaining: float, delta: float, cost: float, layout) -> List[float]:
        """BL6 구성요소 원값을 _bl6_features와 같은 기준으로 정규화"""
        dn, tn = layout.dist_norm, layout.dist_norm / ROBOT_SPEED
        return [load, min(delay / tn, 5.0), min(remaining / dn, 5.0), min(delta / dn, 5.0), min(cost / tn, 10.0)]

    def _bl6_features(self, r: Robot, delta: float, layout, onboard: int) -> List[float]:
        """BL6와 같은 식(env.simulator.marginal_cost_terms)의 한계비용 구성요소 (정규화)

        [적재 토트 비율, 적재 토트 지연, 배정 후 남은 거리, 추가 이동거리, 한계비용 합계]
        onboard: 아직 하역하지 않은 토트 수 (capacity − idle_slots는 트립 중 하역분을 빼지 않아 과대)
        정규화 기준: 거리는 layout.dist_norm, 시간은 dist_norm / 속도
        """
        remaining_after, delay_onboard, mc = marginal_cost_terms(r, delta, onboard)
        dn, tn = layout.dist_norm, layout.dist_norm / ROBOT_SPEED
        return [onboard / self.robot_capacity, min(delay_onboard / tn, 5.0), min(remaining_after / dn, 5.0),
                min(delta / dn, 5.0), min(mc / tn, 10.0)]

    def _pad(self, feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """후보 특징을 max_candidates 길이로 0-패딩 (초과분은 경고 후 버림)"""
        n      = feats.shape[0]
        padded = np.zeros((self.max_candidates, self.feat_dim), dtype=np.float32)
        mask   = np.zeros(self.max_candidates, dtype=np.bool_)
        k      = min(n, self.max_candidates)
        padded[:k] = feats[:k]
        mask[:k]   = True
        if n > self.max_candidates:
            warnings.warn(f"[DQNAgent] candidate count {n} exceeds max_candidates={self.max_candidates}; "
                          f"extra candidates are being dropped.")
        return padded, mask

    # 행동

    def set_global(self, R: np.ndarray, W: np.ndarray):
        """이번 결정의 전역 상태 (RLSimulator._global_state 결과)"""
        self._global = (R, W)

    def q_values(self, cand_feats: np.ndarray) -> np.ndarray:
        """이번 결정(set_global 상태)의 후보별 Q값"""
        R, W = self._global
        with torch.no_grad():
            q = self.policy_net(torch.tensor(cand_feats)[None], torch.tensor(R)[None], torch.tensor(W)[None])
        return q[0].numpy()

    def select_index(self, cand_feats: np.ndarray) -> int:
        """ε-greedy로 후보 하나 선택 (후보가 없으면 -1). 선택 결과를 last_key에 보관"""
        n = cand_feats.shape[0]
        if n == 0:
            return -1
        R, W = self._global
        if self.rng.random() < self.epsilon:
            idx = self.rng.randrange(n)
        else:
            with torch.no_grad():
                q   = self.policy_net(torch.tensor(cand_feats)[None], torch.tensor(R)[None], torch.tensor(W)[None])
                idx = int(q[0].argmax())
        pad, mask = self._pad(cand_feats)
        self.last_key = (pad, mask, min(idx, self.max_candidates - 1), R, W)
        return idx

    # 학습

    def push(self, key, reward: float, next_key, done: bool, discount: float = GAMMA):
        """key 전이를 n-step 버퍼를 거쳐 리플레이 버퍼에 저장 (next_key가 None이면 다음 상태 없음)"""
        self.nstep_buf.push((key, reward, next_key, None, done, discount))
        if self.nstep_buf.ready():
            self.buffer.push(*self.nstep_buf.get())
        if done:
            for t in self.nstep_buf.flush():
                self.buffer.push(*t)

    def flush_nstep(self):
        """에피소드 절단 (terminal 아님): n-step 버퍼에 남은 전이를 짧은 return + 부트스트랩으로 저장 (arm.truncate)"""
        for t in self.nstep_buf.flush():
            self.buffer.push(*t)

    def _stack(self, keys, j):
        return torch.tensor(np.stack([k[j] for k in keys]))

    def train_step(self) -> Optional[float]:
        """PER 배치 하나로 Double DQN 업데이트 1회 (다음 상태 행동은 policy_net, 값은 target_net), Huber 손실

        출력:
            가중 손실값, 버퍼가 batch_size보다 작으면 None
        """
        if len(self.buffer) < self.batch_size:
            return None
        batch, idxs, weights = self.buffer.sample(self.batch_size, self.beta)
        zero = (np.zeros((self.max_candidates, self.feat_dim), np.float32),
                np.zeros(self.max_candidates, np.bool_), 0,
                np.zeros_like(batch[0][0][3]), np.zeros_like(batch[0][0][4]))
        cur = [b[0] for b in batch]
        nxt = [b[2] if b[2] is not None else zero for b in batch]
        r   = torch.tensor(np.array([b[1] for b in batch], np.float32))
        d   = torch.tensor(np.array([b[4] for b in batch], np.float32))
        g   = torch.tensor(np.array([b[5] for b in batch], np.float32))
        ai  = torch.tensor([k[2] for k in cur])
        w_t = torch.tensor(weights, dtype=torch.float32)

        q = self.policy_net(*(self._stack(cur, j) for j in (0, 3, 4))).gather(1, ai[:, None]).squeeze(1)
        with torch.no_grad():
            nx, nm, nR, nW = (self._stack(nxt, j) for j in (0, 1, 3, 4))
            best   = self.policy_net(nx, nR, nW).masked_fill(~nm, float("-inf")).argmax(1)
            next_q = self.target_net(nx, nR, nW).gather(1, best[:, None]).squeeze(1)
            next_q = torch.where(~nm.any(1), torch.zeros_like(next_q), next_q)
            target = r + g * next_q * (1 - d)

        self.buffer.update_priorities(idxs, (q - target).detach().cpu().numpy())
        self.last_stats = (q.mean().item(), (q - target).abs().mean().item())
        loss = (w_t * F.huber_loss(q, target, reduction="none", delta=HUBER_DELTA)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), GRAD_CLIP)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()

    def end_episode(self, total_episodes: int = TOTAL_EPISODES):
        """에피소드 종료마다 PER β 선형 증가 (ε는 고정)"""
        self.beta = min(PER_BETA_END, self.beta + (PER_BETA_END - PER_BETA_START) / total_episodes)

    # 저장 / 불러오기

    def full_state(self) -> Dict:
        """이어하기용 전체 학습 상태 (가중치, 옵티마이저, 리플레이 버퍼, 탐색/난수 상태)"""
        B = self.buffer
        return {
            "policy": self.policy_net.state_dict(), "target": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "buffer": {"buf": B.buf[:B.size] if B.size < B.capacity else B.buf, "priorities": B.priorities,
                       "pos": B.pos, "size": B.size, "max_priority": B._max_priority,
                       "rng": B.rng.bit_generator.state},
            "nstep": list(self.nstep_buf.buf),
            "epsilon": self.epsilon, "beta": self.beta, "steps": self.steps,
            "rng": self.rng.getstate(), "torch_rng": torch.get_rng_state(), "mu_table": self.mu_table,
        }

    def load_full_state(self, s: Dict):
        """full_state()로 저장한 상태 복원"""
        self.policy_net.load_state_dict(s["policy"])
        self.target_net.load_state_dict(s["target"])
        self.optimizer.load_state_dict(s["optimizer"])
        B, b = self.buffer, s["buffer"]
        B.buf = list(b["buf"]) + [None] * (B.capacity - len(b["buf"]))
        B.priorities[:] = b["priorities"]
        B.pos, B.size, B._max_priority = b["pos"], b["size"], b["max_priority"]
        B.rng.bit_generator.state = b["rng"]
        self.nstep_buf.buf = deque(s["nstep"])
        self.epsilon, self.beta, self.steps = s["epsilon"], s["beta"], s["steps"]
        self.rng.setstate(s["rng"])
        torch.set_rng_state(s["torch_rng"])
        self.mu_table = s["mu_table"]

    def save(self, path: str, meta: Optional[Dict] = None):
        """네트워크 가중치와 학습 조건(meta: arm, λ, M, K, 이동 조건, 시나리오 등)을 체크포인트로 저장"""
        torch.save({
            "policy_state":     self.policy_net.state_dict(),
            "target_state":     self.target_net.state_dict(),
            "steps":            self.steps,
            "epsilon":          self.epsilon,
            "beta":             self.beta,
            "robot_capacity":   self.robot_capacity,
            "num_robots":       self.num_robots,
            "num_workstations": self.num_workstations,
            "sim_time":         self.sim_time,
            "mu_table":         self.mu_table,
            "meta":             meta or {},
        }, path)

    def load(self, path: str):
        """save()로 저장한 체크포인트의 가중치와 μ(λ) 표 로드"""
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_state"])
        self.target_net.load_state_dict(ckpt["target_state"])
        self.steps    = ckpt.get("steps", 0)
        self.beta     = ckpt.get("beta", PER_BETA_END)
        self.mu_table = ckpt.get("mu_table")


def read_checkpoint(path: str) -> Dict:
    """체크포인트의 구조 정보(용량, 로봇 수, WS 수, sim_time)와 학습 조건 meta (가중치 제외)"""
    ckpt = torch.load(path, map_location="cpu")
    return {k: ckpt.get(k) for k in ("robot_capacity", "num_robots", "num_workstations", "sim_time")} \
        | {"meta": ckpt.get("meta", {})}
