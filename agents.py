import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import warnings
from collections import deque
from typing import Dict, List, Tuple, Optional

from environment import WarehouseLayout, Robot, nearest_workstation
from constant import (
    SIM_TIME, NUM_WORKSTATIONS, DEFAULT_SEED,
    GAMMA, LR, BATCH_SIZE, REPLAY_BUFFER_SIZE, TARGET_UPDATE_STEPS,
    EPS_START, EPS_MIN, EPS_DECAY, GRAD_CLIP, N_STEP,
    MAX_CANDIDATES, MAX_CANDIDATES_MARGIN,
    PER_ALPHA, PER_BETA_START, PER_BETA_END, PER_PRIORITY_EPS,
    QUEUE_NORM_FALLBACK, DELTA_NORM_CLIP,
    TOTAL_EPISODES,
)


def candidate_feature_dim(robot_capacity: int) -> int:
    """(주문, 후보 로봇) 쌍 특징벡터의 차원.

    구성: 주문 상대좌표 2 + 삽입 detour 1 + 로봇 상태 (5 + capacity*2)
          + 컨텍스트 2 (큐 길이, 경과시간)

    입력:
        robot_capacity: 로봇 용량
    출력:
        특징 차원 수
    """
    return 2 + 1 + (5 + robot_capacity * 2) + 2


class RobotScoreNet(nn.Module):
    """(주문, 후보 로봇) 쌍 하나의 Q값을 출력하는 MLP.

    후보마다 같은 파라미터로 독립 평가하므로 로봇 수와 무관하게 크기가 고정된다.
    """

    def __init__(self, feat_dim: int):
        """
        입력:
            feat_dim: 입력 특징 차원
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128),      nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64),                          nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력:
            x: (..., feat_dim) 특징 텐서
        출력:
            (...) Q값 텐서
        """
        return self.net(x).squeeze(-1)


class PrioritizedReplayBuffer:
    """TD 오차 기반 우선순위 경험 재생 버퍼 (PER).

    저장 항목: (feat, reward, next_feats[max_candidates, feat_dim],
               next_mask[max_candidates], done, discount)
    discount는 n-step 이후 부트스트랩 항에 곱할 할인 계수다.
    """

    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE, alpha: float = PER_ALPHA,
                 seed: int = DEFAULT_SEED):
        """
        입력:
            capacity: 최대 저장 개수 (초과 시 오래된 항목부터 덮어씀)
            alpha: 우선순위 지수
            seed: 샘플링 난수 시드
        """
        self.capacity      = capacity
        self.alpha         = alpha
        self.buf           = [None] * capacity
        self.priorities    = np.zeros(capacity, dtype=np.float32)
        self.pos           = 0
        self.size          = 0
        self._max_priority = 1.0
        self.rng           = np.random.default_rng(seed)

    def push(self, feat, reward, next_feats, next_mask, done, discount):
        """전이 1건을 현재까지의 최대 우선순위로 저장한다.

        입력:
            feat: 선택한 후보의 특징
            reward: (n-step) 보상
            next_feats: 다음 결정 시점 후보 특징 (패딩됨)
            next_mask: next_feats 유효 마스크
            done: 종료 여부
            discount: 부트스트랩 할인 계수
        """
        self.buf[self.pos]        = (feat, reward, next_feats, next_mask, done, discount)
        self.priorities[self.pos] = self._max_priority
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = PER_BETA_START):
        """우선순위 비례 확률로 배치를 샘플링한다.

        입력:
            batch_size: 배치 크기
            beta: 중요도 샘플링 보정 지수
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
        """샘플된 항목의 우선순위를 |TD 오차|로 갱신한다.

        입력:
            idxs: 버퍼 인덱스 배열
            td_errors: 대응하는 TD 오차 배열
        """
        prios = np.abs(td_errors) + PER_PRIORITY_EPS
        self.priorities[idxs] = prios
        self._max_priority = max(self._max_priority, prios.max())

    def __len__(self):
        return self.size


class NStepBuffer:
    """n-step 누적 보상을 만들기 위한 임시 버퍼.

    항목 형식: (feat, reward, next_feats, next_mask, done, discount)
    discount는 그 전이 하나의 할인 계수다 (legacy: γ, SMDP: e^{-βτ}).
    G = r_0 + g_0·r_1 + g_0·g_1·r_2 + ..., 부트스트랩 할인 = Π g_i.
    """

    def __init__(self, n: int = N_STEP):
        """
        입력:
            n: 누적할 스텝 수
        """
        self.n   = n
        self.buf = deque()

    def push(self, transition):
        """1-step 전이를 추가한다.

        입력:
            transition: (feat, reward, next_feats, next_mask, done, discount)
        """
        self.buf.append(transition)

    def ready(self) -> bool:
        """n개 이상 쌓여 n-step 전이를 만들 수 있으면 True."""
        return len(self.buf) >= self.n

    def _accumulate(self):
        """가장 오래된 전이 기준 n-step 전이를 만들고 그 전이를 버퍼에서 뺀다."""
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
        """가장 오래된 전이 기준 n-step 전이를 만든다.

        출력:
            (feat0, G, next_feats_n, next_mask_n, done_n, discount_n)
        """
        return self._accumulate()

    def flush(self):
        """에피소드 종료 시 남은 전이를 모두 n-step 전이로 변환해 비운다.

        출력:
            (feat0, G, next_feats, next_mask, done, discount) 리스트
        """
        results = []
        while self.buf:
            results.append(self._accumulate())
        return results


class DQNAgent:
    """후보 스코어링 방식의 Double DQN 에이전트.

    행동은 그 순간 배정 가능한 후보 로봇 리스트 내 인덱스이며,
    각 후보를 RobotScoreNet으로 스코어링해 최댓값을 고른다.
    """

    def __init__(self, robot_capacity: int, seed: int = DEFAULT_SEED,
                sim_time: float = SIM_TIME, num_robots: Optional[int] = None,
                num_workstations: int = NUM_WORKSTATIONS):
        """
        입력:
            robot_capacity: 로봇 용량
            seed: 네트워크 초기화 / 탐색 / 버퍼 샘플링 난수 시드
            sim_time: 경과시간 정규화 기준 (s)
            num_robots: 로봇 수 (최대 후보 수 결정, 로깅용)
            num_workstations: 워크스테이션 수 (로깅용)
        """
        self.robot_capacity   = robot_capacity
        self.sim_time         = sim_time
        self.num_robots       = num_robots
        self.num_workstations = num_workstations
        self.feat_dim         = candidate_feature_dim(robot_capacity)

        if num_robots is not None:
            self.max_candidates = max(MAX_CANDIDATES, num_robots + MAX_CANDIDATES_MARGIN)
        else:
            self.max_candidates = MAX_CANDIDATES

        # 네트워크 초기화가 전역 torch RNG를 쓰므로, 생성 순서와 무관하게
        # 재현되도록 네트워크 생성 전에 시드를 고정한다.
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

    def build_candidates(self, order_loc: Tuple[float, float],
                         robots: List[Robot], n_wait: int,
                         layout: WarehouseLayout, now: float = 0.0,
                         ws_capacity: Optional[float] = None) -> np.ndarray:
        """후보 로봇마다 (주문, 로봇) 특징벡터를 만든다.

        삽입 detour는 idle 후보면 주문 위치에서 가장 가까운 워크스테이션으로
        복귀한다고 가정하고, mid-trip 후보면 주문 포함 후 최적 워크스테이션을
        재탐색해 계산한다 (실제 insert_and_reorder 결과와 같은 기준).

        입력:
            order_loc: 주문 위치
            robots: 후보 로봇 리스트
            n_wait: 현재 대기 큐 길이
            layout: WarehouseLayout
            now: 현재 시각 (s)
            ws_capacity: 큐 길이 정규화 기준 (보통 num_robots * robot_capacity)
        출력:
            (len(robots), feat_dim) float32 배열
        """
        queue_denom  = ws_capacity if (ws_capacity is not None and ws_capacity > 0) else QUEUE_NORM_FALLBACK
        queue_norm   = min(n_wait / queue_denom, 1.0)
        elapsed_norm = min(now / self.sim_time, 1.0)

        nearest_ws_id = nearest_workstation(order_loc, layout.workstations, layout.aisle_distance)
        target_coords = layout.workstations[nearest_ws_id]

        feats = np.zeros((len(robots), self.feat_dim), dtype=np.float32)
        for i, r in enumerate(robots):
            wx, wy    = r.workstation
            order_rel = [(order_loc[0] - wx) / layout.coord_norm,
                        (order_loc[1] - wy) / layout.coord_norm]
            end        = target_coords if r.is_idle else None
            delta_ins  = r.delta_distance_insert(
                order_loc, layout, target_ws_coords=end, dynamic_ws=(not r.is_idle))
            delta_norm = min(delta_ins / layout.dist_norm, DELTA_NORM_CLIP)
            robot_feat = r.state_vector(self.robot_capacity).tolist()
            feats[i]   = order_rel + [delta_norm] + robot_feat + [queue_norm, elapsed_norm]
        return feats

    def _pad(self, feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """후보 특징을 max_candidates 길이로 0-패딩한다. 초과분은 경고 후 버린다.

        입력:
            feats: (n, feat_dim) 후보 특징
        출력:
            (padded, mask)
            padded: (max_candidates, feat_dim)
            mask: (max_candidates,) 유효 후보 여부
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
                f"{self.max_candidates}; extra candidates are being dropped. "
                f"Increase num_robots/num_workstations hint at agent creation.")
        return padded, mask

    def select_index(self, cand_feats: np.ndarray) -> int:
        """ε-greedy로 후보 하나를 고른다.

        입력:
            cand_feats: (n_eligible, feat_dim) 후보 특징
        출력:
            선택한 후보 인덱스, 후보가 없으면 -1
        """
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
        """전이를 n-step 버퍼를 거쳐 리플레이 버퍼에 저장한다.

        입력:
            feat: 선택한 후보의 특징
            reward: 즉시 보상
            next_feats: 다음 결정 시점 후보 특징 (없으면 None)
            done: 에피소드 종료 여부
            discount: 이 전이의 할인 계수 (legacy: γ, SMDP: e^{-βτ})
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
        """리플레이 버퍼에서 배치 하나를 뽑아 Double DQN 업데이트를 1회 수행한다.

        다음 상태의 행동은 policy_net으로 고르고 값은 target_net으로 평가한다.

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

        loss = (w_t * (q_vals - target).pow(2)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), GRAD_CLIP)
        self.optimizer.step()

        self.steps += 1
        if self.steps % TARGET_UPDATE_STEPS == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()

    def decay_epsilon(self, total_episodes: int = TOTAL_EPISODES):
        """에피소드 종료마다 ε을 감소시키고 PER β를 선형 증가시킨다.

        입력:
            total_episodes: 전체 학습 에피소드 수 (β 증가 폭 결정)
        """
        self.epsilon = max(EPS_MIN, self.epsilon * EPS_DECAY)
        self.beta    = min(self.beta_end,
                           self.beta + (self.beta_end - PER_BETA_START) / total_episodes)

    def save(self, path: str, meta: Optional[Dict] = None):
        """네트워크 가중치와 학습 상태를 체크포인트로 저장한다.

        입력:
            path: 저장 경로
            meta: 학습 조건 기록 (λ, 레이아웃/이동 모드, 시나리오 등). 시각화 도구가 읽는다.
        """
        torch.save({
            "policy_state": self.policy_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "steps":        self.steps,
            "epsilon":      self.epsilon,
            "beta":         self.beta,
            "robot_capacity": self.robot_capacity,
            "num_robots":      self.num_robots,
            "num_workstations": self.num_workstations,
            "sim_time":        self.sim_time,
            "meta":            meta or {},
        }, path)

    def load(self, path: str):
        """save()로 저장한 체크포인트를 불러온다.

        입력:
            path: 체크포인트 경로
        """
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_state"])
        self.target_net.load_state_dict(ckpt["target_state"])
        self.steps   = ckpt.get("steps", 0)
        self.epsilon = ckpt.get("epsilon", EPS_MIN)
        self.beta    = ckpt.get("beta", self.beta_end)
