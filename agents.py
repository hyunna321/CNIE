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
    TOTAL_EPISODES, ROBOT_SPEED, SMDP_COST_DIST,
    STATE_MAX_TIME, WS_TIME_NORM,
)

FEATURE_SETS = ("base", "bl6", "smdp_w")
BL6_EXTRA_DIM = 5   # feature_set="bl6"일 때 추가되는 특징 수
WS_FEAT_DIM      = 4                          # 워크스테이션 1곳의 상태 특징 수
SMDP_W_EXTRA_DIM = 3 + WS_FEAT_DIM + 1 + 6    # 쌍 3 + 복귀 WS 4 + wait 플래그 1 + 전역 6


def candidate_feature_dim(robot_capacity: int, feature_set: str = "base") -> int:
    """(주문, 후보 로봇) 쌍 특징벡터의 차원

    구성: 주문 상대좌표 2 + 삽입 detour 1 + 로봇 상태 (5 + capacity*2)
          + 컨텍스트 2 (큐 길이, 경과시간)
          + (feature_set="bl6"이면) BL6 한계비용 구성요소 5
          + (feature_set="smdp_w"이면) SMDP_W_EXTRA_DIM (DQNAgent.build_candidates_w 참고)

    입력:
        robot_capacity: 로봇 용량
    출력:
        특징 차원 수
    """
    extra = {"base": 0, "bl6": BL6_EXTRA_DIM, "smdp_w": SMDP_W_EXTRA_DIM}[feature_set]
    return 2 + 1 + (5 + robot_capacity * 2) + 2 + extra


class RobotScoreNet(nn.Module):
    """(주문, 후보 로봇) 쌍 하나의 Q값을 출력하는 MLP.

    후보마다 같은 파라미터로 독립 평가하므로 로봇 수와 무관하게 크기 고정
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
    """TD 오차 기반 우선순위 경험 재생 버퍼 (PER)

    저장 항목: (feat, reward, next_feats[max_candidates, feat_dim],
               next_mask[max_candidates], done, discount)
    discount: n-step 이후 부트스트랩 항에 곱할 할인 계수
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
        """전이 1건을 현재까지의 최대 우선순위로 저장

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
        """우선순위 비례 확률로 배치 샘플링

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
        """샘플된 항목의 우선순위를 |TD 오차|로 갱신

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
    """n-step 누적 보상을 만들기 위한 임시 버퍼

    항목 형식: (feat, reward, next_feats, next_mask, done, discount)
    discount: 그 전이 하나의 할인 계수 (legacy: γ, SMDP: e^{-βτ})
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
        """1-step 전이 추가

        입력:
            transition: (feat, reward, next_feats, next_mask, done, discount)
        """
        self.buf.append(transition)

    def ready(self) -> bool:
        """n개 이상 쌓여 n-step 전이를 만들 수 있으면 True."""
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
        """가장 오래된 전이 기준 n-step 전이 생성

        출력:
            (feat0, G, next_feats_n, next_mask_n, done_n, discount_n)
        """
        return self._accumulate()

    def flush(self):
        """에피소드 종료 시 남은 전이를 모두 n-step 전이로 변환 후 비움

        출력:
            (feat0, G, next_feats, next_mask, done, discount) 리스트
        """
        results = []
        while self.buf:
            results.append(self._accumulate())
        return results


class DQNAgent:
    """후보 스코어링 방식의 Double DQN 에이전트

    - 행동: 그 순간 배정 가능한 후보 로봇 리스트 내 인덱스
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
            num_robots: 로봇 수 (최대 후보 수 결정, 로깅용)
            num_workstations: 워크스테이션 수 (로깅용)
            feature_set: "base", "bl6" (BL6 한계비용 구성요소 5개 추가),
                         "smdp_w" (RLWaitSimulator용, build_candidates_w 참고)
        """
        assert feature_set in FEATURE_SETS, feature_set
        self.feature_set      = feature_set
        self.robot_capacity   = robot_capacity
        self.sim_time         = sim_time
        self.num_robots       = num_robots
        self.num_workstations = num_workstations
        self.feat_dim         = candidate_feature_dim(robot_capacity, feature_set)

        if num_robots is not None:
            self.max_candidates = max(MAX_CANDIDATES, num_robots + MAX_CANDIDATES_MARGIN)
        else:
            self.max_candidates = MAX_CANDIDATES

        # 네트워크 초기화가 전역 torch RNG를 쓰므로, 생성 순서와 무관하게
        # 재현되도록 네트워크 생성 전에 시드 고정
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
        """후보 로봇마다 (주문, 로봇) 특징벡터 생성

        삽입 detour 계산 기준 (실제 insert_and_reorder 결과와 같은 기준):
        - idle 후보: 주문 위치에서 가장 가까운 워크스테이션으로 복귀 가정
        - mid-trip 후보: 주문 포함 후 최적 워크스테이션 재탐색

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
            row = order_rel + [delta_norm] + robot_feat + [queue_norm, elapsed_norm]
            if self.feature_set == "bl6":
                row += self._bl6_features(r, order_loc, layout, delta_ins, target_coords)
            feats[i]   = row
        return feats

    def _bl6_features(self, r: Robot, order_loc, layout: WarehouseLayout,
                      delta_ins: float, target_coords) -> List[float]:
        """BL6(MarginalCostSimulator._marginal_cost)와 같은 식의 한계비용 구성요소를 정규화해 반환

        [적재 주문 비율, 적재 주문 지연, 배정 후 남은 거리, 추가 이동거리, 한계비용 합계]
        정규화 기준: 거리는 layout.dist_norm, 시간은 dist_norm / 속도
        """
        onboard = self.robot_capacity - r.idle_slots
        dist    = layout.aisle_distance
        if r.is_idle:
            remaining_after = dist(r.pos, order_loc) + dist(order_loc, target_coords)
            delay_onboard   = 0.0
            added_distance  = remaining_after
        else:
            remaining_after = r.remaining_distance + delta_ins
            delay_onboard   = onboard * delta_ins / ROBOT_SPEED
            added_distance  = delta_ins
        mc = delay_onboard + remaining_after / ROBOT_SPEED + SMDP_COST_DIST * added_distance
        dn, tn = layout.dist_norm, layout.dist_norm / ROBOT_SPEED
        return [onboard / self.robot_capacity, min(delay_onboard / tn, 5.0), min(remaining_after / dn, 5.0),
                min(added_distance / dn, 5.0), min(mc / tn, 10.0)]

    def build_candidates_w(self, order_loc: Tuple[float, float], robots: List[Robot],
                           layout: WarehouseLayout, ctx: Dict, now: float = 0.0,
                           with_wait: bool = False) -> np.ndarray:
        """feature_set="smdp_w"용 후보 특징. with_wait이면 마지막 행이 wait 행동

        base 특징 뒤에 붙는 SMDP_W_EXTRA_DIM개:
        - 쌍 3: 적재 주문 지연, 새 주문 예상 완료시간, 남은 피킹 비율
        - 복귀 WS 4: 삽입 후 복귀할 워크스테이션의 상태 (RLWaitSimulator._ws_state)
        - wait 플래그 1
        - 전역 6: ctx["glob"] 5 + WS 예상 대기 평균

        입력:
            order_loc: 주문 위치
            robots: 후보 로봇 리스트
            layout: WarehouseLayout
            ctx: RLWaitSimulator._context() 결과
            now: 현재 시각 (s)
            with_wait: wait 행동 행 추가 여부
        출력:
            (len(robots) + with_wait, feat_dim) float32 배열
        """
        C, tn  = self.robot_capacity, STATE_MAX_TIME
        cn, dn = layout.coord_norm, layout.dist_norm
        M, ws  = ctx["num_robots"], ctx["ws"]
        near   = nearest_workstation(order_loc, layout.workstations, layout.aisle_distance)
        # WS별 예상 대기 = 도킹 잔여 처리 시간 + 복귀 예정 작업량
        load     = float((ws[:, 0] + ws[:, 3]).mean())
        base_ctx = [min(ctx["n_wait"] / (M * C), 1.0), min(now / self.sim_time, 1.0)]
        tail     = ctx["glob"] + [min(load / WS_TIME_NORM, 5.0)]

        feats = np.zeros((len(robots) + int(with_wait), self.feat_dim), dtype=np.float32)
        for i, r in enumerate(robots):
            wx, wy   = r.workstation
            delta, k = r.insert_preview(order_loc, layout, near)
            n_unpicked, t_left, proc = ctx["robot"].get(r.id, (0, 0.0, 0.0))
            onboard = C - r.idle_slots
            if r.is_idle:
                eta = (layout.aisle_distance(r.pos, order_loc)
                       + layout.aisle_distance(order_loc, layout.workstations[k])) / ROBOT_SPEED
            else:
                eta = t_left + delta / ROBOT_SPEED
            rem, occ, n_in, work = ws[k]
            if not r.is_idle and r.workstation_id == k:   # 후보 자신은 복귀 예정 집계에서 제외
                n_in, work = n_in - 1.0, work - proc
            feats[i] = (
                [(order_loc[0] - wx) / cn, (order_loc[1] - wy) / cn, min(delta / dn, DELTA_NORM_CLIP)]
                + r.state_vector(C).tolist() + base_ctx
                + [min(onboard * delta / ROBOT_SPEED / tn, 5.0), min(eta / tn, 3.0), n_unpicked / C]
                + [min(rem / WS_TIME_NORM, 5.0), occ, n_in / M, min(work / WS_TIME_NORM, 5.0)]
                + [0.0] + tail)
        if with_wait:
            wx, wy = layout.workstations[near]
            feats[-1] = (
                [(order_loc[0] - wx) / cn, (order_loc[1] - wy) / cn, 0.0]
                + [0.0] * (5 + C * 2) + base_ctx
                + [0.0] * (3 + WS_FEAT_DIM) + [1.0] + tail)
        return feats

    def _pad(self, feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """후보 특징을 max_candidates 길이로 0-패딩 (초과분은 경고 후 버림)

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
        """ε-greedy로 후보 하나 선택

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
        """전이를 n-step 버퍼를 거쳐 리플레이 버퍼에 저장

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
        """리플레이 버퍼에서 배치 하나를 뽑아 Double DQN 업데이트 1회 수행

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
        """에피소드 종료마다 ε 감소, PER β 선형 증가

        입력:
            total_episodes: 전체 학습 에피소드 수 (β 증가 폭 결정)
        """
        self.epsilon = max(EPS_MIN, self.epsilon * EPS_DECAY)
        self.beta    = min(self.beta_end,
                           self.beta + (self.beta_end - PER_BETA_START) / total_episodes)

    def save(self, path: str, meta: Optional[Dict] = None):
        """네트워크 가중치와 학습 상태를 체크포인트로 저장

        입력:
            path: 저장 경로
            meta: 학습 조건 기록 (λ, 레이아웃/이동 모드, 시나리오 등, 시각화 도구에서 읽음)
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
            "feature_set":     self.feature_set,
            "meta":            meta or {},
        }, path)

    def load(self, path: str):
        """save()로 저장한 체크포인트 로드

        입력:
            path: 체크포인트 경로
        """
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_state"])
        self.target_net.load_state_dict(ckpt["target_state"])
        self.steps   = ckpt.get("steps", 0)
        self.epsilon = ckpt.get("epsilon", EPS_MIN)
        self.beta    = ckpt.get("beta", self.beta_end)
