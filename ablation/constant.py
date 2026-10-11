"""프로젝트 전역 상수 모음

창고 레이아웃, 시뮬레이션, DQN 하이퍼파라미터, 보상 함수, 실험 설정,
어블레이션 실험 상수를 한 곳에서 관리
"""

# 공통
DEFAULT_SEED = 42

# 창고 레이아웃
BN = 3            # 블록 수
AN = 14           # 블록당 통로 수
SN = 20           # 통로 한쪽 면의 선반 행 수

Wa = 3.0          # m, 통로 폭
Wc = 4.0          # m, 교차통로 폭
Wt = 0.6          # m, 토트 폭 (y 방향 행 간격)
Dt = 0.5          # m, 토트 깊이 (x 방향 랙 깊이, single-deep)

NUM_AISLES   = BN * AN
STORAGE_LOCS = BN * AN * 2 * SN

NUM_SKUS = 350

# 로봇 / 워크스테이션
NUM_ROBOTS       = 14
ROBOT_CAPACITY   = 5
ROBOT_SPEED      = 1.0    # m/s
NUM_WORKSTATIONS = 3      # 창고 하단(y=0)에 균등 배치

# 시뮬레이션
SIM_TIME       = 10 * 3600     # s, 10시간 근무
LAMBDA_DEFAULT = 800 / 3600    # orders/s, 단발성 실행 및 run_ablation.py 기본값
LAMBDA_GRID    = [700 / 3600, 800 / 3600, 900 / 3600, 1000 / 3600]   # orders/s

# DQN 하이퍼파라미터
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
MAX_CANDIDATES      = 8
MAX_CANDIDATES_MARGIN = 3      # max_candidates = max(MAX_CANDIDATES, num_robots + margin)

# Prioritized Experience Replay
PER_ALPHA        = 0.6
PER_BETA_START   = 0.4
PER_BETA_END     = 1.0
PER_PRIORITY_EPS = 1e-6

# 특징 정규화
STATE_MAX_TIME       = 300.0   # s, 잔여 이동시간 정규화 기준
QUEUE_NORM_FALLBACK  = 20.0    # ws_capacity 미지정 시 큐 길이 정규화 기준
DELTA_NORM_CLIP      = 3.0     # 정규화된 삽입 detour 상한

# 보상 함수
REWARD_D_MAX          = 218.0  # m, 삽입 detour 정규화 기준 (≈ 2 × 통로 그래프 최대 구간)
REWARD_GAMMA_Q        = 0.3    # 큐 압박 페널티 가중치
REWARD_ALPHA_MID      = 2.0    # mid-trip 삽입 detour 감쇠 계수
REWARD_ALPHA_IDLE     = 1.0    # idle 로봇 배정 detour 감쇠 계수
REWARD_BETA_DEPLETION = 0.2    # 마지막 빈 idle 로봇 소진 페널티

# 지표
SOLO_RT_M = 80.0               # m, 주문 1건 단독 왕복 거리 추정치

# 학습 / 평가
TOTAL_EPISODES        = 1000
VERBOSE_EVERY         = 50
CURRICULUM_LAM_MARGIN = 0.7    # 커리큘럼 λ 하한 = lam × margin
N_EVAL                = 20
EVAL_SEED_BASE        = 10000
LOAD_SWEEP_SEED_BASE  = 30000
PARTIAL_K             = 3      # BL2 Partial-Buffer 출발 기준 주문 수
P_VALUE_THRESHOLD     = 0.05
OUT_DIR               = "logs"

POLICY_LABELS = [
    "RL",
    "BL1_FullBuffer",
    f"BL2_Partial_k{PARTIAL_K}",
    "BL3_Random",
    "BL4_Nearest",
    "BL5_MinInsertion",
]

# run_ablation.py
ABLATION_OUT_DIR = "logs/ablation"
PAIRED_SUMMARY_METRICS = [
    "n_completed", "avg_wait_time", "avg_flow_time", "dist_per_order",
    "total_distance", "completion_rate", "reward",
]
