"""프로젝트 전역 상수 모음

창고 레이아웃, 그리드 이동, 시뮬레이션, 배정 정책, 평가 설정을 한 곳에서 관리
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
RESERVE_HORIZON_EXTRA = 300     # 틱, 구간 경로 탐색에서 최단거리 대비 허용하는 최대 지연
RESERVE_MAX_EXPAND    = 60000   # 구간 경로 탐색 1회의 최대 상태 확장 수 (넘으면 계획 실패)

# 시뮬레이션
SIM_TIME       = 10 * 3600     # s, 10시간 근무
LAMBDA_DEFAULT = 320 / 3600    # orders/s (평균 2.5토트 → 약 800토트/hr)

# 배정 정책
PARTIAL_K = 3      # BL2 Partial-Buffer 출발 기준 주문 수
COST_DIST = 1.0    # order·s / m, BL6 한계비용에서 이동거리 1m를 흐름시간 몇 초로 환산할지

# 평가
N_EVAL         = 20
EVAL_SEED_BASE = 10000
OUT_DIR        = "logs"
