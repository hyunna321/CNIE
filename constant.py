"""프로젝트 전역 상수 모음.

창고 레이아웃, 시뮬레이션, DQN 하이퍼파라미터, 보상 함수, 실험 설정,
그래프 스타일 상수를 한 곳에서 관리한다.
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
LAMBDA_DEFAULT = 850 / 3600    # orders/s, 그리드 밖 단발성 실행용 기본값
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

# run.py robot-sweep / ws-sweep: 로봇 수 / 워크스테이션 수 스윕
EXPERIMENT_LAMBDA         = 800 / 3600
EXPERIMENT_OUT_DIR        = "logs_experiment"
EXPERIMENT_ROBOT_SWEEP    = (6, 10, 14, 18, 22)   # K=3 고정
EXPERIMENT_ROBOT_SWEEP_K  = 3
EXPERIMENT_WS_SWEEP       = (1, 3, 5)             # M=14 고정
EXPERIMENT_WS_SWEEP_M     = 14

# 그래프 공통 스타일
PLOT_FONT_FAMILY = "serif"
PLOT_FONT_SERIF  = [
    "Times New Roman", "Nimbus Roman No9 L", "Nimbus Roman",
    "Liberation Serif", "DejaVu Serif",
]
PLOT_MATHTEXT_FONTSET = "stix"
PLOT_LABEL_FONTSIZE   = 15
PLOT_TICK_FONTSIZE    = 13
PLOT_DPI              = 200
FIG_DIR               = "fig"
PLOT_LAM_VALUES       = [700, 800, 900, 1000]   # orders/hr

# visualization.py: 학습 보상 수렴 곡선
REWARD_PLOT_CSV_PATH        = "training_curves_all_lambdas.csv"
REWARD_PLOT_OUT_PDF         = "fig/reward_convergence.pdf"
REWARD_PLOT_OUT_PNG         = "fig/reward_convergence.png"
REWARD_PLOT_COLORS          = {700: "#1a2a6c", 800: "#3b5fc4", 900: "#3fae7f", 1000: "#e8b93f"}
REWARD_PLOT_SMOOTH_WINDOW   = 25
REWARD_PLOT_LEGEND_FONTSIZE = 12
REWARD_PLOT_FIGSIZE         = (8, 5.5)

# visualization.py: λ별 정책 비교 막대그래프
METRIC_PLOT_CSV_PATH        = "eval_all_lambdas.csv"
METRIC_PLOT_DISPLAY_NAMES   = [
    "RL (proposed)", "BL1 Full-Buffer", "BL2 Partial-Buffer",
    "BL3 Random", "BL4 Nearest", "BL5 Min-Insertion",
]
METRIC_PLOT_COLORS          = ["#0d1b4c", "#3b5fc4", "#8fb8f0", "#bfe8e3", "#3fae7f", "#e8b93f"]
METRIC_PLOT_BAR_WIDTH       = 0.13
METRIC_PLOT_GROUP_GAP       = 0.35
METRIC_PLOT_LEGEND_FONTSIZE = 11
METRIC_PLOT_FIGSIZE         = (9, 6)
