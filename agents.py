import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import List, Tuple, Optional

from environment import (
    WarehouseLayout, nearest_neighbor_route, route_total_distance,
    nearest_workstation, best_workstation_route,
    ROBOT_SPEED, SIM_TIME, NUM_WORKSTATIONS,
)

# ── 하이퍼파라미터 ─────────────────────────────────────────────────────────────
GAMMA               = 0.99
LR                  = 3e-4
BATCH_SIZE          = 128
REPLAY_BUFFER_SIZE  = 50_000
TARGET_UPDATE_STEPS = 200
EPS_START           = 1.0
EPS_MIN             = 0.02
EPS_DECAY           = 0.997
GRAD_CLIP           = 10.0
N_STEP              = 3
MAX_CANDIDATES = 8


# ─────────────────────────────────────────────────────────────────────────────
# Robot
# ─────────────────────────────────────────────────────────────────────────────

class Robot:
    """로봇은 특정 워크스테이션에 영구 고정되지 않는다. workstation /
    workstation_id 는 '이번 트립이 출발해 복귀할 워크스테이션'을 뜻하며,
    새 트립을 시작할 때마다 redock() 을 통해 동적으로 바뀔 수 있다
    (어느 워크스테이션으로 복귀할지는 그 트립의 목적지들을 기준으로
    총 이동거리가 최소가 되도록 호출부에서 결정한다 — environment.
    best_workstation_route / nearest_workstation 참고).

    state_vector() 는 위치/경로를 항상 '현재 도킹된 워크스테이션 기준
    상대좌표'로 인코딩하며, 정규화 기준(coord_norm)은 레이아웃의 실제
    치수에서 유도된다.
    """

    def __init__(self, robot_id: int, workstation: Tuple[float, float],
                 layout: WarehouseLayout, capacity: int,
                 workstation_id: int = 0):
        self.id             = robot_id
        self.workstation     = workstation        # 이번 트립의 출발/복귀 워크스테이션 좌표
        self.workstation_id  = workstation_id      # 이번 트립의 출발/복귀 워크스테이션 인덱스
        self.layout          = layout
        self.capacity        = capacity
        self.reset()

    def reset(self):
        self.pos               = self.workstation
        self.route             = []
        self.idle_slots        = self.capacity
        self.is_idle           = True
        self.current_dest      = None
        self.progress          = 0.0
        self.total_travel_time = 0.0
        self._travel_start     = None

    def start_travel(self, now: float):
        if self._travel_start is None:
            self._travel_start = now

    def end_travel(self, now: float):
        if self._travel_start is not None:
            self.total_travel_time += max(0.0, now - self._travel_start)
            self._travel_start = None

    def time_to_complete_current(self) -> float:
        return self.remaining_distance / ROBOT_SPEED

    @property
    def remaining_distance(self) -> float:
        d = 0.0
        if self.current_dest:
            d += max(0.0, self._md(self.pos, self.current_dest) - self.progress)
        if self.route:
            prev = self.current_dest if self.current_dest else self.pos
            for nxt in self.route:
                d += self._md(prev, nxt)
                prev = nxt
            d += self._md(prev, self.workstation)
        elif self.current_dest:
            d += self._md(self.current_dest, self.workstation)
        return d

    def arrive_at_node(self) -> bool:
        self.pos      = self.current_dest
        self.progress = 0.0
        if self.route:
            self.current_dest = self.route.pop(0)
            return True
        self.current_dest = None
        return False

    def redock(self, ws_id: int, layout: WarehouseLayout):
        """새 목적 워크스테이션으로 재도킹한다 (idle 상태에서만 호출 가능).

        로봇의 실제 물리적 위치(pos)는 그대로 두고, 이번에 새로 시작될
        트립이 향할(그리고 복귀할) 워크스테이션만 갱신한다.
        """
        assert self.is_idle, "redock()은 idle 로봇에서만 호출할 수 있습니다."
        self.workstation_id = ws_id
        self.workstation     = layout.workstations[ws_id]

    def assign_trip(self, destinations: List[Tuple[float, float]],
                    layout: WarehouseLayout):
        """어느 워크스테이션으로 복귀해야 총 이동거리가 최소가
        되는지를 best_workstation_route 로 매번 다시 계산해 확정한다.
        """
        # 트립은 로봇의 실제 현재 위치(pos)에서 출발한다.
        best_ws_id, route, _ = best_workstation_route(
            self.pos, destinations, layout.workstations, layout.aisle_distance)
        self.workstation_id = best_ws_id
        self.workstation     = layout.workstations[best_ws_id]
        self.route       = route
        self.idle_slots  = self.capacity - len(destinations)
        self.is_idle     = False
        self._start_move()

    def insert_and_reorder(self, loc: Tuple[float, float],
                            layout: WarehouseLayout):
        """삽입되는 주문은 기존 트립이 향하던 워크스테이션과 무관할 수 있으므로, 
        삽입 후 전체 목적지 집합에 대해 최적 복귀
        워크스테이션을 best_workstation_route 로 다시 계산해 갱신한다."""
        pending = list(self.route) + [loc]
        start   = self.current_dest if self.current_dest else self.pos
        best_ws_id, new_route, _ = best_workstation_route(
            start, pending, layout.workstations, layout.aisle_distance)
        self.workstation_id = best_ws_id
        self.workstation     = layout.workstations[best_ws_id]
        self.route       = new_route
        self.idle_slots  = max(0, self.idle_slots - 1)

    def delta_distance_insert(self, loc: Tuple[float, float],
                               layout: WarehouseLayout,
                               target_ws_coords: Optional[Tuple[float, float]] = None,
                               dynamic_ws: bool = False,
                               ) -> float:
        """세 가지 모드를 지원한다.

        1) target_ws_coords 지정: (아직 redock() 되지 않은 idle 후보 로봇을
           '이 워크스테이션으로 복귀한다면' 이라는 가정 하에 평가할 때 사용)
           해당 좌표를 트립 종료(복귀) 지점으로 고정해 delta 를 계산한다.
        2) dynamic_ws=True (target_ws_coords 미지정): mid-trip 후보 로봇의
           복귀 워크스테이션을 '이 주문까지 포함했을 때 최적'인 지점으로
           재탐색하여 delta 를 계산한다 — insert_and_reorder 가 실제로
           수행할 재계산과 동일한 기준으로 평가해야 에이전트의 스코어링이
           실제 삽입 결과와 일치한다(problem description #3).
        3) 둘 다 아니면: 현재 self.workstation(마지막으로 확정된 복귀지)을
           그대로 사용한다.
        """
        pending = list(self.route) + [loc]
        start   = self.current_dest if self.current_dest else self.pos

        if target_ws_coords is not None:
            end       = target_ws_coords
            before    = self._remaining_route_dist(layout, end)
            new_route = nearest_neighbor_route(start, pending, layout.aisle_distance)
            after     = route_total_distance(start, new_route, end, layout.aisle_distance)
        elif dynamic_ws:
            before          = self._remaining_route_dist(layout, self.workstation)
            _, _, after_ws  = best_workstation_route(
                start, pending, layout.workstations, layout.aisle_distance)
            after = after_ws
        else:
            end       = self.workstation
            before    = self._remaining_route_dist(layout, end)
            new_route = nearest_neighbor_route(start, pending, layout.aisle_distance)
            after     = route_total_distance(start, new_route, end, layout.aisle_distance)

        return after - before

    def state_vector(self, max_cap: int, max_time: float = 300.0) -> np.ndarray:
        """항상 현재 도킹된 워크스테이션 기준 상대좌표로 인코딩한다.
        정규화 기준(coord_norm)은 레이아웃 실제 치수에서 유도된다."""
        wx, wy     = self.workstation
        norm       = self.layout.coord_norm
        rem_t_norm = min(self.time_to_complete_current() / max_time, 1.0)
        vec = [
            (self.pos[0] - wx) / norm,
            (self.pos[1] - wy) / norm,
            float(self.idle_slots) / max_cap,
            float(self.is_idle),
            rem_t_norm,
        ]
        route_flat = []
        for p in self.route[:max_cap]:
            route_flat.extend([(p[0] - wx) / norm, (p[1] - wy) / norm])
        while len(route_flat) < max_cap * 2:
            route_flat.extend([0.0, 0.0])
        vec.extend(route_flat)
        return np.array(vec, dtype=np.float32)

    def _remaining_route_dist(self, layout: WarehouseLayout,
                               end: Optional[Tuple[float, float]] = None) -> float:
        end   = end if end is not None else self.workstation
        start = self.current_dest if self.current_dest else self.pos
        return route_total_distance(start, self.route, end,
                                    layout.aisle_distance)

    def _start_move(self):
        if self.route:
            self.current_dest = self.route.pop(0)

    def _md(self, p1, p2):
        return self.layout.aisle_distance(p1, p2)


# ─────────────────────────────────────────────────────────────────────────────
# 후보(candidate) 특징 차원
# ─────────────────────────────────────────────────────────────────────────────

def candidate_feature_dim(robot_capacity: int) -> int:
    """
    (주문, 후보 로봇 1명) 쌍 하나를 표현하는 특징벡터의 차원.
      order_rel   : 2   (주문 위치, 후보 로봇의 현재 워크스테이션 기준 상대좌표)
      delta_ins   : 1   (이 로봇에 배정 시 예상 삽입 detour, 정규화)
      robot_state : 5 + robot_capacity*2   (Robot.state_vector, 이미 상대좌표)
      context     : 2   (전체 대기 큐 길이 정규화, 경과시간 정규화)
    """
    return 2 + 1 + (5 + robot_capacity * 2) + 2


# ─────────────────────────────────────────────────────────────────────────────
# 후보 스코어링 네트워크
# ─────────────────────────────────────────────────────────────────────────────

class RobotScoreNet(nn.Module):
    """(주문, 후보 로봇) 쌍 하나에 대해 스칼라 Q값을 출력한다.

    후보 집합의 크기(=이 순간 가용한 로봇 수)와 무관하게 동일한 파라미터로
    각 후보를 독립 평가한다(배치 차원으로 병렬 처리). 전체 로봇 대수 M 이
    늘어도 이 네트워크의 파라미터 수/출력 차원은 그대로다.
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128),      nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64),                          nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # (batch,)


# ─────────────────────────────────────────────────────────────────────────────
# Prioritized Experience Replay
# ─────────────────────────────────────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """α=0.6, β=0.4→1.0 annealing.

    저장 항목: (feat, reward, next_feats[max_candidates, feat_dim],
               next_mask[max_candidates], done)
    """

    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE, alpha: float = 0.6,
                 seed: int = 42):
        self.capacity      = capacity
        self.alpha         = alpha
        self.buf           = [None] * capacity
        self.priorities    = np.zeros(capacity, dtype=np.float32)
        self.pos           = 0
        self.size          = 0
        self._max_priority = 1.0
        self.rng           = np.random.default_rng(seed)

    def push(self, feat, reward, next_feats, next_mask, done):
        self.buf[self.pos]        = (feat, reward, next_feats, next_mask, done)
        self.priorities[self.pos] = self._max_priority
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4):
        prios   = self.priorities[:self.size] ** self.alpha
        probs   = prios / prios.sum()
        idxs    = self.rng.choice(self.size, batch_size, replace=False, p=probs)
        weights = (self.size * probs[idxs]) ** (-beta)
        weights /= weights.max()
        batch   = [self.buf[i] for i in idxs]
        feat, r, nf, nm, d = zip(*batch)
        return (
            np.array(feat, dtype=np.float32),
            np.array(r,    dtype=np.float32),
            np.array(nf,   dtype=np.float32),
            np.array(nm,   dtype=np.bool_),
            np.array(d,    dtype=np.float32),
            idxs,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, idxs, td_errors):
        prios = np.abs(td_errors) + 1e-6
        self.priorities[idxs] = prios
        self._max_priority = max(self._max_priority, prios.max())

    def __len__(self):
        return self.size


# ─────────────────────────────────────────────────────────────────────────────
# N-step return buffer
# ─────────────────────────────────────────────────────────────────────────────

class NStepBuffer:
    """항목 형식: (feat, reward, next_feats, next_mask, done)."""

    def __init__(self, n: int = N_STEP, gamma: float = GAMMA):
        self.n     = n
        self.gamma = gamma
        self.buf   = deque()

    def push(self, transition):
        self.buf.append(transition)

    def ready(self) -> bool:
        return len(self.buf) >= self.n

    def get(self):
        feat0 = self.buf[0][0]
        _, _, nf, nm, d = self.buf[-1]
        G = 0.0
        for i, (_, r_i, _, _, d_i) in enumerate(self.buf):
            G += (self.gamma ** i) * r_i
            if d_i:
                break
        self.buf.popleft()
        return (feat0, G, nf, nm, d)

    def flush(self):
        results = []
        while self.buf:
            feat0     = self.buf[0][0]
            nf, nm, d = self.buf[-1][2], self.buf[-1][3], self.buf[-1][4]
            G = sum((self.gamma ** i) * t[1] for i, t in enumerate(self.buf))
            self.buf.popleft()
            results.append((feat0, G, nf, nm, d))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# DQNAgent (Factored / Candidate-Scoring)
# ─────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    """행동 공간을 '전체 로봇 수 M' 대신 '그 순간의 실제 가용 로봇 집합'으로
    정의하는 factored D3QN 에이전트."""

    def __init__(self, robot_capacity: int, seed: int = 42,
                sim_time: float = SIM_TIME, num_robots: Optional[int] = None,
                num_workstations: int = NUM_WORKSTATIONS):
        self.robot_capacity   = robot_capacity
        self.sim_time         = sim_time
        self.num_robots       = num_robots          # 참고/로깅용
        self.num_workstations = num_workstations    # 참고/로깅용
        self.feat_dim         = candidate_feature_dim(robot_capacity)

        if num_robots is not None:
            self.max_candidates = max(MAX_CANDIDATES, num_robots + 3)
        else:
            self.max_candidates = MAX_CANDIDATES

        # 네트워크 초기화가 전역 torch RNG를 소비하므로, 생성 순서와 무관하게
        # 재현 가능하도록 네트워크를 만들기 전에 시드를 고정한다.
        torch.manual_seed(seed)

        self.device     = torch.device("cpu")
        self.policy_net = RobotScoreNet(self.feat_dim).to(self.device)
        self.target_net = RobotScoreNet(self.feat_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.buffer    = PrioritizedReplayBuffer(REPLAY_BUFFER_SIZE, seed=seed)
        self.nstep_buf = NStepBuffer(n=N_STEP, gamma=GAMMA)

        self.epsilon  = EPS_START
        self.beta     = 0.4
        self.beta_end = 1.0
        self.steps    = 0
        self.rng      = random.Random(seed)

    # ── 후보 특징 생성 ────────────────────────────────────────────────────────
    def build_candidates(self, order_loc: Tuple[float, float],
                         robots: List[Robot], n_wait: int,
                         layout: WarehouseLayout, now: float = 0.0,
                         ws_capacity: Optional[float] = None) -> np.ndarray:
        """eligible 로봇 리스트에 대해 (len(robots), feat_dim) 특징 행렬을 만든다.

        주문이 특정 워크스테이션에 고정되지 않으므로, idle 후보 로봇의 삽입
        detour는 '주문 위치에서 가장 가까운 워크스테이션으로 복귀한다면'을
        가정해 계산한다(nearest_workstation). mid-trip 후보는 self.workstation
        (과거 확정된 복귀지)을 고정 가정하지 않고, 이 주문까지 포함했을 때의
        최적 복귀 워크스테이션을 best_workstation_route 로 다시 탐색
        (dynamic_ws=True)해 detour 를 계산한다 — 실제 insert_and_reorder() 가
        수행할 재계산과 동일한 기준이어야 에이전트의 스코어와 실제 삽입
        결과가 일치한다.

        ws_capacity: 큐 압박 정규화 기준 용량. 로봇 풀이 워크스테이션에
                     고정 분할되지 않으므로 보통 전체 로봇 풀 기준
                     (num_robots * robot_capacity)을 넘긴다.
        """
        queue_denom  = ws_capacity if (ws_capacity is not None and ws_capacity > 0) else 20.0
        queue_norm   = min(n_wait / queue_denom, 1.0)
        elapsed_norm = min(now / self.sim_time, 1.0)

        nearest_ws_id = nearest_workstation(order_loc, layout.workstations, layout.aisle_distance)
        target_coords = layout.workstations[nearest_ws_id]

        feats = np.zeros((len(robots), self.feat_dim), dtype=np.float32)
        for i, r in enumerate(robots):
            wx, wy    = r.workstation
            order_rel = [(order_loc[0] - wx) / layout.coord_norm,
                        (order_loc[1] - wy) / layout.coord_norm]
            # idle 후보: '주문 위치에서 가장 가까운 워크스테이션으로 복귀한다면'을
            # 가정(target_ws_coords)해 평가한다.
            # busy(mid-trip) 후보: 더 이상 self.workstation(과거 확정된 복귀지)을
            # 고정 가정하지 않고, 이 주문까지 포함했을 때의 최적 복귀 워크스테이션을
            # 다시 탐색(dynamic_ws=True)해 평가한다 — 실제 insert_and_reorder()가
            # 수행할 재계산과 동일한 기준이어야 스코어와 실제 결과가 일치한다.
            end        = target_coords if r.is_idle else None
            delta_ins  = r.delta_distance_insert(
                order_loc, layout, target_ws_coords=end, dynamic_ws=(not r.is_idle))
            delta_norm = min(delta_ins / layout.dist_norm, 3.0)
            robot_feat = r.state_vector(self.robot_capacity).tolist()
            feats[i]   = order_rel + [delta_norm] + robot_feat + [queue_norm, elapsed_norm]
        return feats

    def _pad(self, feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n      = feats.shape[0]
        padded = np.zeros((self.max_candidates, self.feat_dim), dtype=np.float32)
        mask   = np.zeros(self.max_candidates, dtype=np.bool_)
        k      = min(n, self.max_candidates)
        padded[:k] = feats[:k]
        mask[:k]   = True
        if n > self.max_candidates:
            import warnings
            warnings.warn(
                f"[DQNAgent] candidate count {n} exceeds max_candidates="
                f"{self.max_candidates}; extra candidates are being dropped. "
                f"Increase num_robots/num_workstations hint at agent creation.")
        return padded, mask

    # ── 행동 선택 ─────────────────────────────────────────────────────────────
    def select_index(self, cand_feats: np.ndarray) -> int:
        """cand_feats: (n_eligible, feat_dim). eligible 리스트 내 인덱스를 반환."""
        n = cand_feats.shape[0]
        if n == 0:
            return -1
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(n)
        with torch.no_grad():
            q = self.policy_net(torch.tensor(cand_feats)).numpy()
        return int(np.argmax(q))

    # ── 학습 ──────────────────────────────────────────────────────────────────
    def push(self, feat: np.ndarray, reward: float,
            next_feats: Optional[np.ndarray], done: bool):
        if next_feats is None or len(next_feats) == 0:
            nf_pad = np.zeros((self.max_candidates, self.feat_dim), dtype=np.float32)
            nm     = np.zeros(self.max_candidates, dtype=np.bool_)
        else:
            nf_pad, nm = self._pad(np.asarray(next_feats, dtype=np.float32))
        self.nstep_buf.push((feat, reward, nf_pad, nm, done))
        if self.nstep_buf.ready():
            self.buffer.push(*self.nstep_buf.get())
        if done:
            for t in self.nstep_buf.flush():
                self.buffer.push(*t)

    def train_step(self) -> Optional[float]:
        if len(self.buffer) < BATCH_SIZE:
            return None

        feat, r, nf, nm, d, idxs, weights = self.buffer.sample(BATCH_SIZE, beta=self.beta)
        feat_t = torch.tensor(feat)
        r_t    = torch.tensor(r)
        nf_t   = torch.tensor(nf)
        nm_t   = torch.tensor(nm)
        d_t    = torch.tensor(d)
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

            gamma_n = GAMMA ** N_STEP
            target  = r_t + gamma_n * next_q * (1 - d_t)

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

    def decay_epsilon(self, total_episodes: int = 1000):
        self.epsilon = max(EPS_MIN, self.epsilon * EPS_DECAY)
        self.beta    = min(self.beta_end,
                           self.beta + (self.beta_end - 0.4) / total_episodes)
        
    # ── 체크포인트 ────────────────────────────────────────────────────────────
    def save(self, path: str):
        """학습된 정책을 파일로 저장한다(경로 로그 재현/분석 시 재사용하기 위함)."""
        torch.save({
            "policy_state": self.policy_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "steps":        self.steps,
            "epsilon":      self.epsilon,
            "beta":         self.beta,
            "robot_capacity": self.robot_capacity,
            "num_robots":      self.num_robots,
            "num_workstations": self.num_workstations,
        }, path)
 
    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_state"])
        self.target_net.load_state_dict(ckpt["target_state"])
        self.steps   = ckpt.get("steps", 0)
        self.epsilon = ckpt.get("epsilon", EPS_MIN)
        self.beta    = ckpt.get("beta", self.beta_end)