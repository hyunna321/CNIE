"""후보 스코어링 Double DQN 에이전트와 (토트, 후보 로봇) 특징

- 결정 단위는 토트 (BL1~BL6과 같음). 토트의 WS는 주문 도착 시 정해져 있음
- 행동: 후보 로봇 리스트 내 인덱스 (행동 공간 = 후보 로봇)
- 후보마다 같은 RobotScoreNet으로 Q값을 내므로 로봇 수와 무관하게 네트워크 크기 고정
- 특징 세트 (feature_set)
  - "base"  : 토트 상대좌표, 삽입 detour, 로봇 상태, 컨텍스트
  - "bl6"   : base + BL6 한계비용 구성요소 5
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

from environment import Robot
from constant import (
    SIM_TIME, NUM_WORKSTATIONS, DEFAULT_SEED,
    GAMMA, LR, BATCH_SIZE, REPLAY_BUFFER_SIZE, TARGET_UPDATE_STEPS,
    EPS_START, EPS_MIN, EPS_DECAY, GRAD_CLIP, N_STEP,
    MAX_CANDIDATES, MAX_CANDIDATES_MARGIN,
    PER_ALPHA, PER_BETA_START, PER_BETA_END, PER_PRIORITY_EPS,
    DELTA_NORM_CLIP, TOTAL_EPISODES, ROBOT_SPEED, SMDP_COST_DIST,
    STATE_MAX_TIME, SMDP_REWARD_SCALE,
    GLOBAL_ROBOT_DIM, GLOBAL_RESIDUAL_SCALE, GLOBAL_EPSILON, GLOBAL_BUFFER_SIZE, HUBER_DELTA,
)

FEATURE_SETS  = ("base", "bl6")
BL6_EXTRA_DIM = 5                                 # feature_set="bl6"일 때 추가되는 특징 수


def candidate_feature_dim(robot_capacity: int, feature_set: str = "base") -> int:
    """(토트, 후보 로봇) 쌍 특징벡터의 차원

    토트 상대좌표 2 + 삽입 detour 1 + 로봇 상태 (5 + capacity*2)
    + 컨텍스트 4 (큐 길이, 경과시간, 시스템 내 주문 수, 운행 로봇 비율)
    + feature_set별 추가 특징 (DQNAgent.build_candidates 참고)
    """
    extra = {"base": 0, "bl6": BL6_EXTRA_DIM}[feature_set]
    return 2 + 1 + (5 + robot_capacity * 2) + 4 + extra


class RobotScoreNet(nn.Module):
    """(토트, 후보 로봇) 쌍 하나의 Q값을 출력하는 MLP"""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128),      nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64),                          nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(..., feat_dim) -> (...) Q값"""
        return self.net(x).squeeze(-1)


class PrioritizedReplayBuffer:
    """TD 오차 기반 우선순위 경험 재생 버퍼 (PER)

    저장 항목: (feat, reward, next_feats[max_candidates, feat_dim],
               next_mask[max_candidates], done, discount)
    discount: n-step 이후 부트스트랩 항에 곱할 할인 계수
    """

    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE, alpha: float = PER_ALPHA,
                 seed: int = DEFAULT_SEED):
        self.capacity      = capacity
        self.alpha         = alpha
        self.buf           = [None] * capacity
        self.priorities    = np.zeros(capacity, dtype=np.float32)
        self.pos           = 0
        self.size          = 0
        self._max_priority = 1.0
        self.rng           = np.random.default_rng(seed)

    def push(self, feat, reward, next_feats, next_mask, done, discount):
        """전이 1건을 현재까지의 최대 우선순위로 저장"""
        self.buf[self.pos]        = (feat, reward, next_feats, next_mask, done, discount)
        self.priorities[self.pos] = self._max_priority
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = PER_BETA_START):
        """우선순위 비례 확률로 배치 샘플링

        출력:
            (feat, reward, next_feats, next_mask, done, discount, idxs, weights) numpy 배열 튜플
        """
        prios   = self.priorities[:self.size] ** self.alpha
        probs   = prios / prios.sum()
        idxs    = self.rng.choice(self.size, batch_size, replace=False, p=probs)
        weights = (self.size * probs[idxs]) ** (-beta)
        weights /= weights.max()
        batch   = [self.buf[i] for i in idxs]
        feat, r, nf, nm, d, g = zip(*batch)
        return (
            np.array(feat, dtype=np.float32),
            np.array(r,    dtype=np.float32),
            np.array(nf,   dtype=np.float32),
            np.array(nm,   dtype=np.bool_),
            np.array(d,    dtype=np.float32),
            np.array(g,    dtype=np.float32),
            idxs,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, idxs, td_errors):
        """샘플된 항목의 우선순위를 |TD 오차|로 갱신"""
        prios = np.abs(td_errors) + PER_PRIORITY_EPS
        self.priorities[idxs] = prios
        self._max_priority = max(self._max_priority, prios.max())

    def __len__(self):
        return self.size


class NStepBuffer:
    """n-step 누적 보상을 만들기 위한 임시 버퍼

    항목 형식: (feat, reward, next_feats, next_mask, done, discount)
    discount: 그 전이 하나의 할인 계수 (SMDP: e^{-βτ})
    G = r_0 + g_0·r_1 + g_0·g_1·r_2 + ..., 부트스트랩 할인 = Π g_i
    """

    def __init__(self, n: int = N_STEP):
        self.n   = n
        self.buf = deque()

    def push(self, transition):
        """1-step 전이 추가"""
        self.buf.append(transition)

    def ready(self) -> bool:
        """n개 이상 쌓여 n-step 전이를 만들 수 있으면 True"""
        return len(self.buf) >= self.n

    def _accumulate(self):
        """가장 오래된 전이 기준 n-step 전이 생성 후 그 전이를 버퍼에서 제거"""
        feat0 = self.buf[0][0]
        _, _, nf, nm, d, _ = self.buf[-1]
        G, disc = 0.0, 1.0
        for (_, r_i, _, _, d_i, g_i) in self.buf:
            G    += disc * r_i
            disc *= g_i
            if d_i:
                break
        self.buf.popleft()
        return (feat0, G, nf, nm, d, disc)

    def get(self):
        """가장 오래된 전이 기준 n-step 전이"""
        return self._accumulate()

    def flush(self):
        """에피소드 종료 시 남은 전이를 모두 n-step 전이로 변환 후 비움"""
        results = []
        while self.buf:
            results.append(self._accumulate())
        return results


class DQNAgent:
    """후보 스코어링 방식의 Double DQN 에이전트

    - 행동: 그 순간 후보 로봇 리스트 내 인덱스
    - 각 후보를 RobotScoreNet으로 스코어링해 최댓값 선택
    """

    def __init__(self, robot_capacity: int, seed: int = DEFAULT_SEED,
                 sim_time: float = SIM_TIME, num_robots: Optional[int] = None,
                 num_workstations: int = NUM_WORKSTATIONS, feature_set: str = "base"):
        """
        입력:
            robot_capacity: 로봇 용량
            seed: 네트워크 초기화 / 탐색 / 버퍼 샘플링 난수 시드
            sim_time: 경과시간 정규화 기준 (s)
            num_robots: 로봇 수 (최대 후보 수 결정)
            num_workstations: 워크스테이션 수 (기록용)
            feature_set: "base", "bl6" (모듈 설명 참고)
        """
        assert feature_set in FEATURE_SETS, feature_set
        self.feature_set      = feature_set
        self.robot_capacity   = robot_capacity
        self.sim_time         = sim_time
        self.num_robots       = num_robots
        self.num_workstations = num_workstations
        self.feat_dim         = candidate_feature_dim(robot_capacity, feature_set)

        # 후보 = 빈 슬롯이 있는 로봇 (최대 로봇 전원)
        if num_robots is not None:
            self.max_candidates = max(MAX_CANDIDATES, num_robots + MAX_CANDIDATES_MARGIN)
        else:
            self.max_candidates = MAX_CANDIDATES

        # 네트워크 초기화가 전역 torch RNG를 쓰므로 생성 전에 시드 고정
        torch.manual_seed(seed)

        self.device     = torch.device("cpu")
        self.policy_net = RobotScoreNet(self.feat_dim).to(self.device)
        self.target_net = RobotScoreNet(self.feat_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.buffer    = PrioritizedReplayBuffer(REPLAY_BUFFER_SIZE, seed=seed)
        self.nstep_buf = NStepBuffer(n=N_STEP)

        self.epsilon  = EPS_START
        self.beta     = PER_BETA_START
        self.beta_end = PER_BETA_END
        self.steps    = 0
        self.rng      = random.Random(seed)
        self.huber    = False                  # True이면 Huber 손실 (False: MSE)
        self.mu_table: Optional[Tuple[List[float], List[float]]] = None   # local 보상 고정 기준점 (λ 격자, μ)
        self.last_stats = (float("nan"), float("nan"))                     # 마지막 train_step (평균 Q, 평균 |TD|)

    def local_mu(self, lam: float) -> Optional[float]:
        """local 보상 고정 기준점 μ(λ) (mu_table 선형 보간, 범위 밖은 끝값). 없으면 None (누적 평균 사용)"""
        if self.mu_table is None:
            return None
        return float(np.interp(lam, *self.mu_table))

    def _td_loss(self, q: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """PER 가중 TD 손실 (MSE 또는 Huber), 학습 통계 기록"""
        self.last_stats = (q.mean().item(), (q - target).abs().mean().item())
        if self.huber:
            return (w * F.huber_loss(q, target, reduction="none", delta=HUBER_DELTA)).mean()
        return (w * (q - target).pow(2)).mean()

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
            "rng": self.rng.getstate(), "torch_rng": torch.get_rng_state(),
            "local_cost_sum": getattr(self, "local_cost_sum", 0.0), "local_cost_n": getattr(self, "local_cost_n", 0),
            "mu_table": self.mu_table,
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
        self.local_cost_sum, self.local_cost_n = s["local_cost_sum"], s["local_cost_n"]
        self.mu_table = s["mu_table"]

    def build_candidates(self, tote_loc: Tuple[int, int], ws_id: int, robots: List[Robot],
                         deltas: List[float], layout, ctx: Dict, now: float = 0.0) -> np.ndarray:
        """후보 로봇마다 (토트, 로봇) 특징벡터

        공통 (base):
        - 토트 위치의 상대좌표 2 (후보 로봇의 트립 마지막 WS 기준), 삽입 detour 1
        - 로봇 상태 5 + capacity*2 (Robot.state_vector)
        - 컨텍스트 4 (후보 공통 전역 상태)
          대기 큐 길이 / (M·C), 경과시간 / sim_time, 시스템 내 주문 수 / (M·C) (상한 5), 운행 로봇 수 / M
        bl6: + _bl6_features 5

        입력:
            tote_loc: 토트 피킹 셀
            ws_id: 토트가 갈 워크스테이션 (현재 특징에는 미사용)
            robots: 후보 로봇 리스트
            deltas: 후보별 삽입 추가 이동거리 (m, SimulatorBase._insertion_delta)
            layout: GridWarehouseLayout
            ctx: RLSimulator._context() 결과 (n_wait, n_in_system, n_busy, num_robots, onboard)
            now: 현재 시각 (s)
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
            row = ([(tote_loc[0] - wx) / cn, (tote_loc[1] - wy) / cn, min(delta / dn, DELTA_NORM_CLIP)]
                   + r.state_vector(STATE_MAX_TIME) + base_ctx)
            if self.feature_set == "bl6":
                row += self._bl6_features(r, delta, layout, ctx["onboard"][r.id])
            feats[i] = row
        return feats

    def _bl6_features(self, r: Robot, delta: float, layout, onboard: int) -> List[float]:
        """BL6(MarginalCostSimulator._marginal_cost)와 같은 식의 한계비용 구성요소 (정규화)

        [적재 토트 비율, 적재 토트 지연, 배정 후 남은 거리, 추가 이동거리, 한계비용 합계]
        onboard: 아직 하역하지 않은 토트 수 (capacity − idle_slots는 트립 중 하역분을 빼지 않아 과대)
        정규화 기준: 거리는 layout.dist_norm, 시간은 dist_norm / 속도
        """
        if r.is_idle:
            remaining_after, delay_onboard = delta, 0.0
        else:
            remaining_after = r.remaining_distance + delta
            delay_onboard   = onboard * delta / ROBOT_SPEED
        mc = delay_onboard + remaining_after / ROBOT_SPEED + SMDP_COST_DIST * delta
        dn, tn = layout.dist_norm, layout.dist_norm / ROBOT_SPEED
        return [onboard / self.robot_capacity, min(delay_onboard / tn, 5.0), min(remaining_after / dn, 5.0),
                min(delta / dn, 5.0), min(mc / tn, 10.0)]

    def _pad(self, feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """후보 특징을 max_candidates 길이로 0-패딩 (초과분은 경고 후 버림)

        출력:
            (padded (max_candidates, feat_dim), mask (max_candidates,))
        """
        n      = feats.shape[0]
        padded = np.zeros((self.max_candidates, self.feat_dim), dtype=np.float32)
        mask   = np.zeros(self.max_candidates, dtype=np.bool_)
        k      = min(n, self.max_candidates)
        padded[:k] = feats[:k]
        mask[:k]   = True
        if n > self.max_candidates:
            warnings.warn(
                f"[DQNAgent] candidate count {n} exceeds max_candidates="
                f"{self.max_candidates}; extra candidates are being dropped.")
        return padded, mask

    def select_index(self, cand_feats: np.ndarray) -> int:
        """ε-greedy로 후보 하나 선택 (후보가 없으면 -1)"""
        n = cand_feats.shape[0]
        if n == 0:
            return -1
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(n)
        with torch.no_grad():
            q = self.policy_net(torch.tensor(cand_feats)).numpy()
        return int(np.argmax(q))

    def push(self, feat: np.ndarray, reward: float,
             next_feats: Optional[np.ndarray], done: bool,
             discount: float = GAMMA):
        """전이를 n-step 버퍼를 거쳐 리플레이 버퍼에 저장

        입력:
            feat: 선택한 후보의 특징
            reward: 즉시 보상
            next_feats: 다음 결정 시점 후보 특징 (없으면 None)
            done: 에피소드 종료 여부
            discount: 이 전이의 할인 계수 (e^{-βτ})
        """
        if next_feats is None or len(next_feats) == 0:
            nf_pad = np.zeros((self.max_candidates, self.feat_dim), dtype=np.float32)
            nm     = np.zeros(self.max_candidates, dtype=np.bool_)
        else:
            nf_pad, nm = self._pad(np.asarray(next_feats, dtype=np.float32))
        self.nstep_buf.push((feat, reward, nf_pad, nm, done, discount))
        if self.nstep_buf.ready():
            self.buffer.push(*self.nstep_buf.get())
        if done:
            for t in self.nstep_buf.flush():
                self.buffer.push(*t)

    def train_step(self) -> Optional[float]:
        """리플레이 버퍼에서 배치 하나를 뽑아 Double DQN 업데이트 1회

        다음 상태의 행동은 policy_net으로 선택, 값은 target_net으로 평가

        출력:
            가중 MSE 손실값, 버퍼가 BATCH_SIZE보다 작으면 None
        """
        if len(self.buffer) < BATCH_SIZE:
            return None

        feat, r, nf, nm, d, g, idxs, weights = self.buffer.sample(BATCH_SIZE, beta=self.beta)
        feat_t = torch.tensor(feat)
        r_t    = torch.tensor(r)
        nf_t   = torch.tensor(nf)
        nm_t   = torch.tensor(nm)
        d_t    = torch.tensor(d)
        g_t    = torch.tensor(g)
        w_t    = torch.tensor(weights)

        q_vals = self.policy_net(feat_t)

        with torch.no_grad():
            B    = nf_t.shape[0]
            flat = nf_t.view(B * self.max_candidates, self.feat_dim)

            q_policy_next = self.policy_net(flat).view(B, self.max_candidates)
            q_policy_next = q_policy_next.masked_fill(~nm_t, float("-inf"))
            best_idx      = q_policy_next.argmax(dim=1)

            q_target_next = self.target_net(flat).view(B, self.max_candidates)
            next_q = q_target_next.gather(1, best_idx.unsqueeze(1)).squeeze(1)

            no_valid = ~nm_t.any(dim=1)
            next_q   = torch.where(no_valid, torch.zeros_like(next_q), next_q)

            target = r_t + g_t * next_q * (1 - d_t)

        td_errors = (q_vals - target).detach().cpu().numpy()
        self.buffer.update_priorities(idxs, td_errors)

        loss = self._td_loss(q_vals, target, w_t)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), GRAD_CLIP)
        self.optimizer.step()

        self.steps += 1
        if self.steps % TARGET_UPDATE_STEPS == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()

    def decay_epsilon(self, total_episodes: int = TOTAL_EPISODES):
        """에피소드 종료마다 ε 감소, PER β 선형 증가"""
        self.epsilon = max(EPS_MIN, self.epsilon * EPS_DECAY)
        self.beta    = min(self.beta_end,
                           self.beta + (self.beta_end - PER_BETA_START) / total_episodes)

    def save(self, path: str, meta: Optional[Dict] = None):
        """네트워크 가중치와 학습 상태를 체크포인트로 저장

        입력:
            path: 저장 경로
            meta: 학습 조건 기록 (λ, 이동 모드, 시나리오 등)
        """
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
            "feature_set":      self.feature_set,
            "mu_table":         self.mu_table,
            "meta":             meta or {},
        }, path)

    def load(self, path: str):
        """save()로 저장한 체크포인트 로드"""
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_state"])
        self.target_net.load_state_dict(ckpt["target_state"])
        self.steps   = ckpt.get("steps", 0)
        self.epsilon = ckpt.get("epsilon", EPS_MIN)
        self.beta    = ckpt.get("beta", self.beta_end)
        self.mu_table = ckpt.get("mu_table")


def global_ctx_dim(num_workstations: int) -> int:
    """전역 상태 W 벡터 차원: WS별 4 + 토트 WS one-hot K + 컨텍스트 4"""
    return num_workstations * 4 + num_workstations + 4


def _mlp(*dims: int) -> nn.Sequential:
    """Linear-ReLU 반복, 마지막 Linear 뒤에는 활성화 없음"""
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])


class GlobalScoreNet(nn.Module):
    """전역 상태 + BL6 잔차 후보 스코어링 네트워크 (dueling 없음)

    s   = enc([mean_i φ(R_i), max_i φ(R_i), W])        (R: 전 로봇 특징 집합, W: WS 상태 + 토트 WS + 컨텍스트)
    Q_i = α · f([후보 특징_i, s]) − k · mc_i            (mc_i: bl6 특징의 한계비용 합계, k: 보상 단위 환산)
    α는 학습 파라미터 (초기값 residual_scale)
      alpha_max 없음: α = exp(log_alpha), 상한 없음 (v1)
      alpha_max 있음: α = alpha_max · sigmoid(alpha_logit), (0, alpha_max) 범위 (v2)
    f의 마지막 층을 0으로 초기화해 학습 시작 시 argmax = BL6
    """

    def __init__(self, feat_dim: int, ctx_dim: int, mc_scale: float,
                 residual_scale: float = GLOBAL_RESIDUAL_SCALE, alpha_max: Optional[float] = None):
        super().__init__()
        self.phi = _mlp(GLOBAL_ROBOT_DIM, 64, 64)
        self.enc = nn.Sequential(_mlp(128 + ctx_dim, 128, 64), nn.ReLU())
        self.f   = _mlp(feat_dim + 64, 128, 128, 64, 1)
        nn.init.zeros_(self.f[-1].weight)
        nn.init.zeros_(self.f[-1].bias)
        self.register_buffer("mc_scale", torch.tensor(float(mc_scale)))
        self.alpha_max = alpha_max
        if alpha_max is None:
            self.log_alpha = nn.Parameter(torch.tensor(math.log(residual_scale)))
        else:
            assert 0 < residual_scale < alpha_max, (residual_scale, alpha_max)
            p = residual_scale / alpha_max
            self.alpha_logit = nn.Parameter(torch.tensor(math.log(p / (1 - p))))

    def _alpha(self) -> torch.Tensor:
        if self.alpha_max is None:
            return self.log_alpha.exp()
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


class GlobalDQNAgent(DQNAgent):
    """전역 상태 + BL6 잔차 Double DQN, dueling 없음 (feature_set="bl6" 필요)

    - 결정마다 RLSimulator가 set_global(R, W)로 전역 상태를 넘김
    - 전이의 상태 = key (후보 특징 패딩, 마스크, 선택 인덱스, R, W). 다음 상태도 key (같은 객체 공유)
    - ε는 GLOBAL_EPSILON으로 고정 (BL6 근처 데이터)
    """

    uses_global = True

    def __init__(self, robot_capacity: int, dist_norm: float, seed: int = DEFAULT_SEED,
                 sim_time: float = SIM_TIME, num_robots: Optional[int] = None,
                 num_workstations: int = NUM_WORKSTATIONS, feature_set: str = "bl6",
                 alpha_max: Optional[float] = None, huber: bool = False):
        """
        입력:
            dist_norm: layout.dist_norm (bl6 한계비용 특징의 정규화 기준, 보상 단위 환산에 사용)
            alpha_max: 잔차 계수 α 상한 (None이면 상한 없음, GlobalScoreNet 참고)
            huber: True이면 Huber 손실 (False: MSE)
            나머지: DQNAgent와 같음

        BL6 prior 척도 mc_scale = dist_norm / v / Z (Z = SMDP_REWARD_SCALE)는 local/smdp 보상 단위에 맞춘 값
        legacy 보상(결정당 O(1), arm legacy_r8_ddqn_v2)과 함께 써도 그대로 둠 → prior(≈ mc / Z)가 보상 척도보다 작고
        α f가 그 차이를 흡수해야 함 (척도 차이는 실험 설계상 의도, 원 논문 보상을 바꾸지 않기 위해)
        """
        assert feature_set == "bl6", "GlobalDQNAgent는 bl6 특징(한계비용 열)이 필요합니다"
        super().__init__(robot_capacity, seed=seed, sim_time=sim_time, num_robots=num_robots,
                         num_workstations=num_workstations, feature_set=feature_set)
        self.ctx_dim  = global_ctx_dim(num_workstations)
        mc_scale      = self.mc_scale_for(dist_norm)
        self.huber    = huber
        torch.manual_seed(seed)
        self.policy_net = GlobalScoreNet(self.feat_dim, self.ctx_dim, mc_scale, alpha_max=alpha_max).to(self.device)
        self.target_net = GlobalScoreNet(self.feat_dim, self.ctx_dim, mc_scale, alpha_max=alpha_max).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.buffer    = PrioritizedReplayBuffer(GLOBAL_BUFFER_SIZE, seed=seed)
        self.epsilon   = GLOBAL_EPSILON
        self._global   = None
        self.last_key  = None
        # local 보상의 평균 차감용 누적 (학습 전 기간, mu_table이 없을 때만 기준점으로 사용)
        self.local_cost_sum = 0.0
        self.local_cost_n   = 0

    @staticmethod
    def mc_scale_for(dist_norm: float) -> float:
        """레이아웃 dist_norm에서 BL6 prior 척도 (prior = mc_scale · mc / (dist_norm / v) = mc / Z)"""
        return dist_norm / ROBOT_SPEED / SMDP_REWARD_SCALE

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

    def set_global(self, R: np.ndarray, W: np.ndarray):
        """이번 결정의 전역 상태 (RLSimulator._global_state 결과)"""
        self._global = (R, W)

    def select_index(self, cand_feats: np.ndarray) -> int:
        """ε-greedy로 후보 하나 선택. 선택 결과를 last_key에 보관"""
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

    def push(self, key, reward: float, next_key, done: bool, discount: float = GAMMA):
        """key 전이를 n-step 버퍼를 거쳐 리플레이 버퍼에 저장 (next_key가 None이면 다음 상태 없음)"""
        self.nstep_buf.push((key, reward, next_key, None, done, discount))
        if self.nstep_buf.ready():
            self.buffer.push(*self.nstep_buf.get())
        if done:
            for t in self.nstep_buf.flush():
                self.buffer.push(*t)

    def _stack(self, keys, j):
        return torch.tensor(np.stack([k[j] for k in keys]))

    def train_step(self) -> Optional[float]:
        """PER 배치 하나로 Double DQN 업데이트 1회 (다음 상태 행동은 policy_net, 값은 target_net)"""
        B = self.buffer
        if len(B) < BATCH_SIZE:
            return None
        prios   = B.priorities[:B.size] ** B.alpha
        probs   = prios / prios.sum()
        idxs    = B.rng.choice(B.size, BATCH_SIZE, replace=False, p=probs)
        weights = (B.size * probs[idxs]) ** (-self.beta)
        weights /= weights.max()
        batch = [B.buf[i] for i in idxs]

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

        B.update_priorities(idxs, (q - target).detach().cpu().numpy())
        loss = self._td_loss(q, target, w_t)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), GRAD_CLIP)
        self.optimizer.step()

        self.steps += 1
        if self.steps % TARGET_UPDATE_STEPS == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()

    def decay_epsilon(self, total_episodes: int = TOTAL_EPISODES):
        """ε는 고정, PER β만 선형 증가"""
        self.beta = min(self.beta_end, self.beta + (self.beta_end - PER_BETA_START) / total_episodes)
