"""프로젝트 전역 상수: 창고 레이아웃, 그리드 이동, 시뮬레이션, 배정 정책, 평가 시드, RL"""

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

# 그리드 이동
CELL_SIZE             = 1.0     # m, 셀 한 변 = 로봇 크기
ORDER_LINES_MEAN      = 2.5     # 주문당 평균 SKU(토트) 수, 기하분포
ORDER_LINES_MAX       = 10      # 주문당 최대 SKU 수 (기하분포 절단)
PICK_TIME             = 5.2     # s/토트, 랙 앞에서 토트를 꺼내 싣는 시간
WS_PROC_TIME          = 4.9     # s/토트, 워크스테이션에서 토트를 처리하는 시간
WS_QUEUE_CELLS        = None    # 워크스테이션별 대기열 셀 수, None이면 ceil(M/K)
WS_PARK_CELLS         = None    # 워크스테이션별 주차 칸 수, None이면 M (한 WS에 전원 주차 가능)
GRID_REPLAN_WAIT      = 3       # 틱, 이만큼 막히면 점유 셀을 피하는 우회 경로 탐색
GRID_DETOUR_MAX_EXTRA = 10      # 셀, 최단거리 대비 허용하는 최대 우회 길이
GRID_DEADLOCK_TICKS   = 1800    # 틱, 이동 대기 로봇이 있는데 아무도 못 움직이면 교착으로 보고 중단
GRID_TRAFFIC          = "free"     # "free": 통로/교차통로 모든 방향 이동, "lanes": 일방통행 통로 + 차선 교차통로
GRID_MOTION           = "reserve"  # "reserve": 배정 시 시공간 경로 예약, "reactive": 막히면 대기/우회
ROUTE_MODES           = ("nnh", "fixed_dock", "fixed_return")   # 트립 경로 구성 규칙 (environment 모듈 설명)
RESERVE_HORIZON_EXTRA = 300     # 틱, 구간 경로 탐색에서 최단거리 대비 허용하는 최대 지연
RESERVE_MAX_EXPAND    = 60000   # 구간 경로 탐색 1회의 최대 상태 확장 수 (넘으면 계획 실패)

# 시뮬레이션
SIM_TIME       = 10 * 3600     # s, 10시간 근무
LAMBDA_DEFAULT = 320 / 3600    # orders/s (평균 2.5토트 → 약 800토트/hr)

# 배정 정책
PARTIAL_K = 3      # BL2 Partial-Buffer 출발 기준 토트 수
COST_DIST = 1.0    # order·s / m, BL6 한계비용에서 이동거리 1m를 흐름시간 몇 초로 환산할지 (J 지표에도 사용)
BL6_CD_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)   # BL6* 튜닝 격자 (설정마다 검증 시드로 c_d 선택)
TOTE_ORDERS = ("arrival", "bottleneck_first")   # 주문 내 토트 배정 순서 (SimulatorBase tote_order, 기본 arrival)

# 계획 완료시각 추정 (planning.py)
# 확정된 계획의 완료시각 출처: "estimate" (NNH 경로 + 이동·피킹·처리 시간, 다른 로봇 예약 무시) /
#   "sipp" (reserve 모드에서 예약 경로의 도착 틱, 없거나 맞지 않으면 estimate). 후보 가상 삽입과
#   Delta_order는 배정 전후를 같은 기준으로 비교하려고 항상 estimate
PC_CONFIRMED = "estimate"
# 같은 도킹 셀에서 내리는 토트들의 완료시각: "batch" (모두 처리 합 끝 = 시뮬레이터 동작) /
#   "sequential" (j번째 토트 = j번째까지 처리 시간 합)
PC_DOCK      = "batch"

# 평가 / 시드 구간 (구간끼리 겹치지 않음)
N_EVAL            = 20
EVAL_SEED_BASE    = 10000        # 최종 테스트 시드 10000 + i (보고용, 선택/튜닝에 쓰지 않음)
VAL_SEED_BASE     = 20000        # 검증 시드 20000 + i (체크포인트·베이스라인 선택 전용)
N_VAL             = 5
TRAIN_SEED_BASE   = 1_000_000    # 학습 에피소드 시드 = BASE + 학습 시드 × STRIDE + 에피소드 번호
TRAIN_SEED_STRIDE = 100_000
OUT_DIR           = "logs"

# RL 에이전트 (agents.DQNAgent: 전역 상태 + BL6 잔차 Double DQN, dueling 없음)
GAMMA                 = 0.99    # 결정당 할인 (arm discount="decision")
LR                    = 3e-4
BATCH_SIZE            = 128
REPLAY_BUFFER_SIZE    = 50_000
TARGET_UPDATE_STEPS   = 200
EPSILON               = 0.05    # 고정 ε (BL6 근처에서 데이터 수집)
GRAD_CLIP             = 10.0
N_STEP                = 3
HUBER_DELTA           = 1.0     # Huber 손실 전환점 (보상 단위)
MAX_CANDIDATES        = 8
MAX_CANDIDATES_MARGIN = 3       # max_candidates = max(MAX_CANDIDATES, num_robots + margin)
GLOBAL_ROBOT_DIM      = 10      # 전역 상태 R의 로봇 1대 특징 수 (rl_simulator.RLSimulator._global_state)
RESIDUAL_SCALE        = 0.1     # Q = α · f − k · mc_BL6 의 α 초기값 (BL6에서 천천히 벗어나도록)
ALPHA_MAX             = 1.0     # α 상한 (α = ALPHA_MAX · sigmoid(logit))

PER_ALPHA        = 0.6
PER_BETA_START   = 0.4
PER_BETA_END     = 1.0
PER_PRIORITY_EPS = 1e-6

STATE_MAX_TIME  = 300.0   # s, 잔여 시간 정규화 기준
DELTA_NORM_CLIP = 3.0     # 정규화된 삽입 detour 상한

# 보상 (rewards.py)
SMDP_BETA    = 1 / 300    # 1/s, 연속시간 할인율 e^(−βτ) (≈ 트립 시간 규모, arm discount="time")
REWARD_SCALE = 1000.0     # local 보상 정규화 분모, BL6 prior 척도 1/Z
REWARD_GAMMA_Q        = 0.3    # legacy: 큐 압박 페널티 가중치
REWARD_ALPHA_MID      = 2.0    # legacy: mid-trip 삽입 detour 감쇠 계수
REWARD_ALPHA_IDLE     = 1.0    # legacy: idle 로봇 배정 detour 감쇠 계수
REWARD_BETA_DEPLETION = 0.2    # legacy: 마지막 빈 idle 로봇 소진 페널티

# local 보상 고정 기준점 μ(λ): 학습 전 BL6 prior 정책으로 [λ·CURRICULUM_LAM_MARGIN, λ] 격자 롤아웃
# 셀(λ, M, K, 조건)·계수마다 한 번 계산해 local_mu*.json에 캐시, 모든 학습 시드가 같은 표 사용
LOCAL_MU_POINTS  = 5      # λ 격자 점 수 (사이 λ는 선형 보간)
LOCAL_MU_SEEDS   = 10     # 격자 점당 에피소드 수 (학습 시드 0 구간의 별도 번호, 학습/검증/테스트와 겹치지 않음)
LOCAL_MU_WORKERS = 5      # 병렬 프로세스 수 (격자 점 단위)

# RL 학습
TOTAL_EPISODES        = 1000
VERBOSE_EVERY         = 50
CURRICULUM_LAM_MARGIN = 0.7    # 앞쪽 절반 에피소드의 λ 하한 = lam × margin
UPDATE_RATIO          = 8      # 결정 N개당 train_step 1회
CKPT_EVERY            = 50     # 이어하기 체크포인트 저장 주기 (에피소드)
RL_OUT_DIR            = "logs_rl"
