# 시뮬레이션 동작 방식

코드 기준 정리. 동작이 바뀌면 이 문서도 같이 수정

## 1. 레이아웃

| 항목 | 값 | 위치 |
|---|---|---|
| 블록 × 블록당 통로 | 3 × 14 (BN, AN) | `constant.py` |
| 통로 한쪽 선반 행 | 20 (SN), 저장 위치 3 × 14 × 2 × 20 = 1680 | |
| 통로 폭 / 교차통로 폭 | 3 m / 4 m (블록 사이 교차통로는 8 m) | |
| SKU 수 | 350 | |
| 로봇 / 용량 / 속도 | 14대 / 토트 5개 / 1 m/s | |
| 워크스테이션 | 3개, 창고 하단에 균등 배치 | |

- 셀 그리드 (`grid.GridWarehouseLayout`)
  - 1 m 셀, 로봇 1대 = 1셀
  - 거리는 유향 셀 그래프 최단거리 (`aisle_distance`, 랙 통과 불가, d(a, b) ≠ d(b, a) 가능)
  - 맨 위에도 교차통로 추가
  - 워크스테이션마다 대기열 셀 → 도킹 셀 → 다중 차선 주차장
- 이동 규칙 `--traffic`
  - `free`: 통로/교차통로 전방향
  - `lanes`: 일방통행 통로 + 차선 교차통로
- 충돌 처리 `--motion`
  - `reserve`: 시공간 경로 예약
  - `reactive`: 막히면 대기 후 우회. 비교용이고 혼잡 시 교착 가능

## 2. 주문과 토트

- 도착, 주문 크기, SKU 수요, 저장 위치, 처리 시간 분포는 `scenario.py`의 `Scenario`가 결정 (`--scenario`)
- **구조**
  - 주문 1건 = 서로 다른 SKU 여러 개
  - 저장 위치 1곳 = 단일 SKU 토트 1개. SKU마다 그 SKU의 토트 1개를 WS로 가져와야 함
  - 주문 완료 = 그 주문의 모든 토트가 WS 처리를 마친 시점
  - 토트 재고/반납은 모델링하지 않음 (가져간 위치도 계속 사용 가능, 반납 이동 없음)
- **워크스테이션 배정** (`_choose_ws`)
  - 주문 도착 시 WS 하나에 고정, 그 주문의 토트는 모두 그 WS로
  - 규칙: 처리 전 토트가 가장 적은 WS (동률이면 번호 작은 WS)
- **기본값**
  - 포아송 도착 (λ, 기본 320 주문/hr ≈ 800 토트/hr)
  - 주문당 SKU 수: 1 + 기하분포, 평균 2.5, 최대 10에서 절단 (절단 후 평균 약 2.4)
  - 주문 안의 SKU는 수요 분포에서 중복 없이 추출
  - SKU 균등 수요
  - 무작위 저장
  - 피킹 5.2 s/토트, WS 처리 4.9 s/토트 (고정)
- 대기 큐는 토트 단위 단일 FIFO 하나

## 3. 사건 흐름

- **틱 구조**
  - 1틱 = 1 s (1 m / 1 m/s)
  - 틱 사이에 들어온 주문은 도착 시각에 처리, 결정 시점의 로봇 위치는 직전 틱 기준
  - 움직일 로봇도 진행 중인 피킹/처리도 없으면 다음 주문 도착까지 건너뜀
- **주문 도착 시**
  1. WS 배정
  2. 토트마다 배정 정책 호출 (배정 단위 = 토트, 한 주문의 토트가 여러 로봇·트립에 나뉠 수 있음)
  3. 후보 로봇이 없으면 그 토트를 대기 큐에 넣음
- **후보 로봇** (`_eligible_robots`)
  - 빈 슬롯이 있는 모든 로봇 (idle + 운행 중)
  - WS 처리 중인 로봇 제외
- **트립 구성** (`environment.plan_trip`)
  - 피킹 단계: 실은 토트의 피킹 셀을 NNH 순서로
  - 하역 단계: 실은 토트들의 WS 도킹 셀을 마지막 피킹 셀부터 NNH 순서로. 실은 토트의 WS가 여러 개면 여러 WS를 들름
  - 하역할 WS는 토트가 정하므로 정책이나 NNH가 고르지 않음
  - 트립 후 마지막 WS 주차장에 주차
- **운행 중 삽입** (`Robot.reroute`)
  - 현재 목적지는 유지
  - 남은 피킹 셀 + 새 토트 위치로 NNH 다시 구성, 하역할 도킹 셀도 다시 구성
- **경로 구성 변형** (`--route`, 평가 전용, 기본 `nnh`, `Robot.route_mode`)
  - `fixed_dock`: 도킹 셀 순서(복귀 WS)를 트립 출발 시 NNH로 정하고 고정, 삽입으로 새 WS가 생기면 맨 뒤에 추가. 피킹 셀은 NNH로 다시 구성. `_insertion_delta`도 같은 규칙. RL·베이스라인 공통
  - 결과 폴더 `grid_free_reserve_route-fixeddock`, RL run 이름에도 `_route-fixeddock`
  - exact(전수 열거 최적 경로)는 closed-loop로 쓰지 않음. `--log-reconstructions PATH`로 RL 평가 중 NNH 경로 구성 호출(트립 출발 `start`, 운행 중 재구성 `reroute`)의 입력·D(ρ)·호출 시간을 JSONL로 남기고 `experiments/route_gap.py`가 같은 입력을 `environment.plan_trip_exact`로 풀어 거리 차이(%)와 계산 시간(ms) 비교. 기록해도 시뮬레이션 결과는 같음
- **도킹 셀 도착**
  - 그 WS로 갈 토트 중 이미 집은 것을 모두 내리고 처리 시간 합만큼 머묾
  - 처리가 끝난 토트로 주문의 모든 토트가 끝나면 주문 완료
- **트립 완료**
  - 마지막 도킹 셀 처리를 마치면 트립 완료
  - 대기 큐에서 최대 용량만큼 꺼내 바로 새 트립 (`_dispatch`, 정책 결정 아님)
- **흐름시간** = 주문 완료 시각 − 도착 시각, **대기시간** = 첫 토트 배정 시각 − 도착 시각

## 4. 이동과 충돌 처리

- **reserve**
  - 경로를 계획하는 시점
    - idle 로봇 배정
    - 운행 중 삽입
    - 트립 종료 후 새 트립
  - 트립 전체(피킹 셀 → 도킹 셀들 → 마지막 WS의 빈 주차 칸)를 SIPP로 계획해 예약표에 등록
  - 우선순위 = 계획 순서. 먼저 계획한 로봇의 예약은 이후 계획에서 바뀌지 않음
  - 경유지 도착만으로는 다시 계획하지 않음
  - 계획에 실패하면 이전 예약을 복원하고 다음 틱에 재시도
  - 실패한 삽입은 `pending_insert`로 미뤄 두고 도킹 셀에서 반영
- **reactive**
  - 매 틱 최단경로상 다음 셀이 비어 있으면 이동, 막히면 대기
  - `GRID_REPLAN_WAIT`틱 이상 막히면 점유 셀을 피하는 우회 경로 탐색 (최단 + `GRID_DETOUR_MAX_EXTRA`칸 이내)
- **점유 시간**
  - 피킹 셀은 피킹 시간 동안, 도킹 셀은 그 WS에 내리는 토트의 처리 시간 합 동안 점유
  - 같은 위치를 연달아 방문하면 한 번 들러 피킹 시간 합만큼 머묾
- **트립을 마친 idle 로봇**
  - 도킹 셀에서 가까운 빈 주차 칸으로 이동

## 5. 배정 정책

토트마다 로봇을 고름 (WS는 이미 정해져 있음). 정책 등록은 `evaluation.POLICIES` (키 → 결과 라벨, 시뮬레이터 생성 함수)

| 키 | 라벨 | 규칙 | 운행 중 삽입 |
|---|---|---|---|
| BL1 | `BL1_FullBuffer` | 빈 슬롯이 가장 많은 idle 로봇에 쌓고 용량만큼 차면 출발 | X |
| BL1F | `BL1F_FillBuffer` | BL1과 같지만 이미 쌓기 시작한 idle 로봇부터 채움 | X |
| BL2 | `BL2_Partial_k3` | idle 로봇에 토트가 partial_k개 쌓이면 출발 | X |
| BL3 | `BL3_Random` | 후보 중 무작위 | O |
| BL4 | `BL4_Nearest` | 토트 위치에 가장 가까운 로봇 | O |
| BL5 | `BL5_MinInsertion` | 추가 이동거리 최소 (하역 WS 추가 포함) | O |
| BL6 | `BL6_MarginalCost` | (실은 토트 지연 + 새 토트 남은 시간 + c_d · 추가 거리) 최소, c_d = 1. 토트 단위 근사, 주문 완료가 마지막 토트에 달린 점은 미반영 | O |
| BL6D0 | `BL6_MarginalCost_cd0` | BL6에서 c_d = 0 | O |
| GR (별칭 BL7) | `Greedy_R` | legacy 보상 즉시값 최대 | O |
| BL8 | `BL8_OrderPlanned` | Delta_order(k, m) 최소 (주문 계획 완료시각 증가 합, `planning.py`), 동률이면 BL6 한계비용 | O |

- 매개변수 키 (`evaluation.resolve_policy`): `BL6c<c_d>` = BL6에서 c_d만 바꿈 (예: `BL6c0.25`, 라벨 `BL6_MarginalCost_cd0.25`), `<키>@bf` = 주문 내 토트 배정 순서 bottleneck_first (라벨 뒤 `_bf`)
- BL6\* = 셀마다 검증 시드 10개로 c_d ∈ {0, 0.25, 0.5, 1, 2} 중 최선 (tote_order마다 따로, `experiments/stages.py stage0`, select.json `bl6_star`)

- 논문 표기 (`metrics.PAPER_NAMES`, 라벨은 그대로): `BL1F_FillBuffer` → BL1, `BL1_FullBuffer` → BL1-Spread, `Greedy_R` → BL7
- 새 베이스라인
  - 삽입형: `baselines.InsertionSimulator`를 상속해 `_select_robot`만 구현
  - 기존 규칙의 매개변수 변형: `functools.partial`로 등록 (예: BL6D0 = `partial(MarginalCostSimulator, cost_dist=0.0)`)
  - `POLICIES`에 한 줄 추가하면 `run.py`, `run_rl.py --baselines`, 실험 실행기, 뷰어에서 모두 사용 가능
  - 셀마다 테스트할 목록은 `experiments/design.py`의 `ALL_KEYS`, 기준 선택 후보는 `SELECT_KEYS`

## 6. RL 배정

> 보상 설계를 바꿀 예정. 바뀌면 보상 항목과 arm 표를 다시 쓸 것

- **결정 시점**: 토트 도착 (주문 도착 시 그 주문의 토트마다 1번)
  - 한 주문의 토트들은 같은 시각의 연속 결정, 그 사이 τ = 0
  - 후보가 1대 이상일 때만 결정, 없으면 토트를 대기 큐에 넣음
  - 대기 큐에서 꺼내 배정하는 경우(트립 완료 시 FIFO)는 결정 아님
- **행동**: 후보 로봇(`_eligible_robots`) 내 인덱스. 행동 공간 = 후보 로봇
- **에이전트** (`agents.DQNAgent`, 모든 arm 공통): 전역 상태 + BL6 잔차 Double DQN, dueling 없음
  - Q_i = α · f([φ_i, s]) − k · mc_i
    - φ_i: (토트, 후보 i) 특징 27차원 (용량 5)
      - 토트 상대좌표 2, 삽입 detour 1, 로봇 상태 5 + 용량×2 (`Robot.state_vector`)
      - 컨텍스트 4: 대기 큐 길이 / (M·C), 경과시간 / sim_time, 시스템 내 주문 수 / (M·C) (상한 5), 운행 로봇 수 / M
      - BL6 한계비용 구성요소 5 (마지막 = 한계비용 합계 mc_i)
    - s = enc([mean φ(R), max φ(R), W]): 전 로봇 특징 집합 R (로봇당 10) + WS 상태·토트 WS·컨텍스트 W (`RLSimulator._global_state`)
    - k = mc_scale = dist_norm / v / Z (Z = 1000) → 학습 시작 시 argmax = BL6 (f 마지막 층 0 초기화)
    - α = sigmoid 방식으로 (0, 1) 범위, 초기값 0.1
  - PER, 3-step, Huber 손실, ε = 0.05 고정, 결정 8개당 업데이트 1회
  - 앞쪽 절반 에피소드는 λ ∈ [0.7λ, λ] 커리큘럼
- **보상** (`rewards.py`)
  - local (제안): 결정 k의 국소 비용 = 자기 토트 지연 (WS 처리 완료 − 결정 시각) + w_ext · 탑재 토트 지연 추정 (n_onboard · Δd / v) + w_q · 직후 큐잉 토트 지연, 보상 = −(비용 − μ(λ)) / 1000
    - μ(λ): 학습 전 BL6 prior로 [0.7λ, λ] 5점 × 10 에피소드 롤아웃 → 셀 폴더 `local_mu*.json`에 캐시, 같은 셀의 학습 시드가 공유
  - legacy (원 논문): exp(−α · Δd / D_max) − γ_q · 큐 압박 − 마지막 빈 idle 로봇 소진 페널티
    - α_mid = 2, α_idle = 1, γ_q = 0.3, 소진 페널티 0.2, D_max = 2 × (피킹 셀·도킹 셀 사이 최대 거리)
  - 할인: discount = "time"이면 e^(−βτ) (β = 1/300, SMDP), "decision"이면 결정당 γ = 0.99
  - 기록되는 reward는 완료 주문 1건당 값
  - 새 보상: `rewards.Reward`를 상속해 훅(`on_decision`, `on_queue`, `on_interval`, `finalize`)을 구현하고 `REWARDS`에 등록한 뒤 `ARMS`에 arm 추가 (μ(λ) 기준점이 필요하면 `uses_mu = True`)
- **arm** (`rewards.ARMS`) = 보상 설정

  | arm | 보상 | 할인 | 계수 (w_ext, w_q) |
  |---|---|---|---|
  | `local_r8_ddqn_v2` (기본) | local | e^(−βτ) | (1, 1) |
  | `local_r8_ddqn_v2_gamma` | local | γ | (1, 1) |
  | `legacy_r8_ddqn_v2` | legacy | γ | - |
  | `local_r8_ddqn_v2_wext{w}`, `_wq{w}` | local | e^(−βτ) | 한 항만 w ∈ {0, 0.25, 0.5, 2, 4} |

  보상 재설계 arm (`rewards.py` 모듈 설명에 식)

  | arm | 보상 | n | 특징 | tote_order | 기타 |
  |---|---|---|---|---|---|
  | `local_noterm2` (단계 1) | local, (w_ext, w_q) = (0, 1) | 3 | base | arrival | μ(λ) |
  | `order_inc`, `order_inc_n10` (안 A) | order_increment | 5, 10 | order | bottleneck_first | ρ(λ) 중심화 |
  | `little_pbrs` (안 B) | little_pbrs | 3 | order | bottleneck_first | c_phi = 1, β = 1/300, 중심화 없음 |
  | `little_pbrs_cphi0`, `_cphi0.5`, `_n10` (단계 4) | little_pbrs | 3, 3, 10 | order | bottleneck_first | c_phi = 0, 0.5, 1 |
  | `<위 arm>_nocurr` (단계 5) | 같음 | 같음 | 같음 | 같음 | 커리큘럼 없음 |

- **다른 M·K에서 평가** (`run_rl.py --load`, 재학습 없음): BL6 prior 척도 `mc_scale`을 평가 레이아웃의 `dist_norm`으로 다시 설정 (기본, 주차장 행 수가 M·K에 따라 달라짐). `--keep-train-mc-scale`이면 체크포인트 값 유지. 두 값과 적용 여부는 `run.json`의 `mc_scale`. 경과시간 특징 정규화는 체크포인트 학습 sim_time. 전역 상태 차원이 K에 의존해 K가 다른 체크포인트는 로드 불가 (K별 네트워크 필요)
- 큐 압박/소진 페널티 조건 빈도는 `train.csv`의 `n_qpen_active`, `n_depletion_active` / `n_decisions` (보상과 무관하게 기록)
- 뷰어: 정책 "RL (checkpoint)"로 재생, 결정마다 후보 Q값/Δd/BL6 한계비용 표시. 보상 설정은 체크포인트 arm을 따름

## 7. 실행과 실험 구조

- `python run.py`: 베이스라인을 같은 테스트 시드로 돌려 비교표 출력, CSV는 `logs/grid_{traffic}_{motion}/`
- `python run_rl.py --arm {arm}`: RL 학습 후 같은 테스트 시드로 RL과 베이스라인 비교, 결과는 `logs_rl/`, `--summarize`로 요약
- `python -m viewer.server`: 한 번 실행해 재생 (README 참고)
- 실험 (`experiments/`)
  - `design.py`: 실험 블록별 셀·학습 시드·arm·정책 목록 (실행기와 표 생성이 공유)
  - `runner.py <계획>`: main, variants, sens, fleet, small, factorial, routing, status. 끝난 run은 건너뜀, 진행 기록 `{out}/progress.md`
  - `tables.py`: 논문 표 CSV → `{out}/tables/`
  - `route_gap.py`: NNH vs exact 경로 비교 (runner routing이 호출), `curve_check.py`: 학습 곡선 구간 요약, `layout_info.py`: 부록 레이아웃 정보
- 실험 구조

  | 요소 | 구현 |
  |---|---|
  | 공통 테스트 인스턴스 대응 비교 | 테스트 시드 10000 + i (N_TEST = 30)를 모든 정책·학습 시드·셀이 공유, 학습·검증 시드와 겹치지 않음 |
  | 기준 선택 | 검증 시드 20000 + i로 SELECT_KEYS 중 최고를 셀마다 고름 (`select.json`), 테스트 시드는 선택에 쓰지 않음 |
  | 학습 시드 단위 CI | `analysis.seed_level_compare`: 시드별 평균 차이, nested 부트스트랩과 t 구간, 시드 간 대응 t, Holm (표 단위) |
  | 요인 실험 | factorial: λ·M은 같은 네트워크를 전이(네트워크 내), K는 K별 네트워크(네트워크 간). 효과는 네트워크 단위 (`analysis.factorial_effects_by_unit`) |
  | 재학습 없는 일반화 | fleet·small·factorial: `run_rl.py --load`로 다른 셀에서 평가 |
  | 라우팅 오차 분리 | routing: 같은 네트워크를 경로 규칙 fixed_dock·fixed_return에서 재평가 + NNH 경로 구성 기록을 exact와 오프라인 비교 |

## 8. 설정 키 (보상 재설계·베이스라인 보강)

모든 키의 기본값 = 현행 동작 (기본 설정 회귀 테스트 `python tests/regression.py`, 단위 테스트 `python -m unittest discover tests`)

| 위치 | 키 | 기본값 | 다른 값 |
|---|---|---|---|
| `rewards.Arm` | `reward` | `local` | `order_increment` (안 A), `little_pbrs` (안 B), `legacy` |
| | `local_w` | (1, 1) | (0, 1) = 2항 제거 (단계 1) |
| | `n_step` | 3 | 5, 10 |
| | `features` | `base` | `order`: 후보 특징 5 (Ť_{k,m}, slack, is_bottleneck, 늦어지는 다른 주문 수, Delta_order) + 전역 2 (최근 5분 도착 수, 빈 idle 로봇 수) |
| | `tote_order` | `arrival` | `bottleneck_first` |
| | `curriculum` | True | False (비정상 도착) |
| | `c_phi` | 1.0 | little_pbrs shaping 계수 |
| | `beta` | None (= `SMDP_BETA` 1/300) | 0 = 할인 없음 (진단 테스트) |
| `SimulatorBase` | `tote_order` | `arrival` | `bottleneck_first` (정책 키 `@bf`, RL은 arm) |
| `constant.py` | `PC_CONFIRMED` | `estimate` | `sipp` (확정 계획의 완료시각을 예약 경로에서, Delta_order는 항상 estimate) |
| | `PC_DOCK` | `batch` (도킹 처리 합 끝에 모두 완료 = 시뮬레이터) | `sequential` (j번째 토트 = j번째까지 처리 합) |
| | `BL6_CD_GRID` | (0, 0.25, 0.5, 1, 2) | BL6\* 튜닝 격자 |
| `scenario.Scenario` | `arrival` / `step_rates` | `poisson` / () | `nhpp_steps` + 구간별 λ 배수 (평균 1, thinning) |
| `experiments/tables.py` | `--pool` | `old` (BL4–BL7) | `new` (BL4–BL8 + BL6\*, tote_order 두 가지) |

- 계획 완료시각 Ť (`planning.py`): 결정 시점 로봇 상태 틱부터 남은 트립을 걸으며 이동시간 d/v, 피킹 시간 (틱 올림), 도착 시점 WS backlog 대기, 처리 시간을 더함. 다른 로봇의 예약은 무시 (Δd와 같은 가정). backlog = 그 WS에서 처리 중인 다른 로봇의 남은 처리 + 대기열 셀에서 그 도킹을 기다리는 로봇의 처리
  - 지시서의 "τ_w × 순번"은 시뮬레이터가 도킹 셀의 토트를 한꺼번에 처리 완료로 기록하는 것과 달라, 실현값과 일치해야 하는 단위 테스트를 위해 기본을 `batch`로 둠 (`PC_DOCK = "sequential"`이 지시서 식)
- 새 지표 (eval.csv 열): `flow_p99`, `utilization` (운행 로봇 시간 / (M·T)), `idle_at_decision` (후보가 있는 토트 도착 시점의 평균 idle 로봇 수), RL 평가의 결정 분류 `agree_cd0`, `agree_bl8`, `cd_reproducible` (어떤 c_d ≥ 0의 BL6로도 재현되는 결정 비율, 1차원 구간 교집합)
- 단계별 실행: `scripts/run_stage{0..6}.sh`, `scripts/run_rl_current.sh` (`experiments/stages.py` 설명). 결과 `results/<stage>/summary.md` + CSV, 학습 로그 `runs/<stage>_<arm>/s<seed>/train.log`

## 9. 알려진 한계와 TODO

- [ ] **복귀 중 로봇에 삽입할 때의 처리**
  - 현재 동작
    - 피킹을 모두 마치고 WS로 복귀 중인 로봇도 빈 슬롯이 있으면 후보
    - 이 로봇에 삽입하면 이미 집은 토트를 실은 채 새 위치에 다녀온 뒤 WS에서 한꺼번에 내림
    - 하역 중인 도킹 셀이 목적지일 때 그 WS로 갈 토트를 삽입하면, 집은 것만 먼저 내리고 새 토트를 집은 뒤 다시 들름
  - 이동 규칙은 시뮬레이터 공통이고, 이 로봇을 고를지는 정책이 결정
  - 처리 방안 (MDP 정식화 확정 후 결정)
    - (a) WS를 지나며 이미 집은 토트를 먼저 내리고 새 주문만 들고 출발
      - 실제 동작에 가까움
      - 모든 결과 재실행 필요
    - (b) 복귀 중 로봇을 후보에서 제외
    - (c) 규칙 유지, 정책 상태에 남은 피킹 수/복귀 여부 추가 (RL 상태에는 남은 피킹 비율이 이미 있음)
- [ ] **주문-WS 배정 규칙**
  - 지금은 처리 전 토트 수 최소 WS로 고정. 주문 토트 위치, WS 대기열 길이, 열린 주문 수 제한(put wall 슬롯)은 미반영
  - 정책이 WS도 고르게 할지 결정 필요
- [ ] **reactive 모드 교착**
  - 혼잡 시나리오에서 교착 감시(`GRID_DEADLOCK_TICKS`)로 중단 가능
  - 비교용으로만 사용
