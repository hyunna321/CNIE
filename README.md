# CNIE

셀 그리드 창고 시뮬레이터 (다중 SKU 주문, 단일 SKU 토트, 로봇 이동/충돌, 피킹·WS 처리 시간) + 토트 단위 배정 정책 (휴리스틱 베이스라인, RL)

시뮬레이션 동작 방식, 정책·보상 정의, 실험 구조, TODO: [SIMULATION.md](SIMULATION.md)

| 파일 | 내용 |
|---|---|
| `constant.py` | 전역 상수 (레이아웃, 이동, 시드 구간, RL 하이퍼파라미터, 보상 계수) |
| `grid.py` | 셀 레이아웃, 유향 셀 그래프, 거리 테이블, SKU 배치 |
| `environment.py` | 로봇 상태, 트립 경로 (피킹 셀 NNH → 하역 도킹 셀 NNH), 경로 규칙 변형, exact 경로 |
| `reservation.py` | 시공간 예약표와 구간 경로 탐색 (SIPP) |
| `scenario.py` | 주문 도착/주문 크기/SKU 수요/저장/처리 시간 분포 |
| `simulator.py` | 주문/토트 관리, 주문-WS 배정, 틱 이동 엔진 (reserve / reactive) |
| `baselines.py` | 휴리스틱 배정 정책 (BL1~BL6, Greedy_R, BL8) |
| `planning.py` | 계획 완료시각 추정 Ť, 주문 계획 비용 Delta_order (BL8, 주문 특징, bottleneck_first, Φ 공통) |
| `evaluation.py` | 정책 등록표 `POLICIES`, 평가 조건, 시드 구간, 평가 함수 |
| `metrics.py` | 에피소드 지표, CSV 저장, 비교표 |
| `rewards.py` | RL 보상 (local, order_increment, little_pbrs, legacy)과 arm 등록표 `ARMS` |
| `agents.py` | 전역 상태 + BL6 잔차 Double DQN, (토트, 후보 로봇) 특징 |
| `rl_simulator.py` | RL 배정 시뮬레이터 (SMDP 전이, 보상 훅) |
| `training.py` | 학습 루프, 검증 선택, local 보상 기준점 μ(λ) |
| `analysis.py` | 통계 (학습 시드 단위 비교, Holm, 네트워크 단위 요인 효과) |
| `run.py` / `run_rl.py` | 베이스라인 평가 / RL 학습·평가 진입점 |
| `experiments/` | 실험 설계(`design.py`), 실행기(`runner.py`), 단계별 실행·보고(`stages.py`), 표(`tables.py`), NNH–exact 비교(`route_gap.py`), 학습 곡선 점검, 레이아웃 정보 |
| `scripts/` | 단계별 실행 스크립트 `run_stage{0..6}.sh`, `run_rl_current.sh` |
| `tests/` | 단위 테스트 (`python -m unittest discover tests`), 기본 설정 회귀 테스트 (`python tests/regression.py`) |
| `viewer/` | 시뮬레이션 재생 뷰어 |

## 평가

```bash
python run.py                                   # 모든 베이스라인, λ=320 주문/hr, M=14, K=3, 테스트 시드 20개
python run.py --policies BL4 BL6 --lam 280 --robots 18 --n-eval 5 --motion reactive
```

## RL (torch 필요)

```bash
python run_rl.py                                              # 기본 arm, 1000 에피소드 학습 후 RL vs BL4~BL6
python run_rl.py --arm local_r8_ddqn_v2 --episodes 200 --baselines BL6 --n-eval 5
python run_rl.py --load logs_rl/.../agent.pt --robots 18      # 체크포인트를 다른 셀에서 평가 (재학습 없음)
python run_rl.py --summarize                                  # 셀별 run/베이스라인 흐름시간, BL6 대비 차이
```

## 실험

```bash
python experiments/runner.py main --dry-run      # 실행할 명령 확인
python experiments/runner.py main                # λ 240/280/320: 베이스라인 선택·테스트 + 기본 arm 학습 (시드 1~5)
python experiments/runner.py fleet               # 이후 variants, sens, small, factorial, routing (design.py 설명)
python experiments/runner.py status              # 학습 진행 상황
python experiments/tables.py                     # 논문 표 → logs_exp/tables/
```

- 새 보상: `rewards.py`에 `Reward` 하위 클래스 + `ARMS` 항목 → `runner.py main --arm <새 arm>`, `tables.py --arm <새 arm>`
- 새 베이스라인: `evaluation.POLICIES`에 등록, 셀마다 테스트하려면 `experiments/design.py`의 `ALL_KEYS`에 추가

## 보상 재설계 단계

```bash
scripts/run_stage0.sh                 # 베이스라인 보강 (BL6* 튜닝, BL8, tote_order) → results/stage0/summary.md (STOP 1)
scripts/run_stage1.sh smoke           # 20 에피소드 1 seed 스모크 → screen (3 seed) → final (5 seed)
scripts/run_stage3.sh                 # 안 B 학습 후 STOP 2 보고 → results/stop2/summary.md
```

- 설정 키와 기본값: [SIMULATION.md](SIMULATION.md) 8절

## 시뮬레이션 뷰어

```bash
python -m viewer.server    # http://127.0.0.1:8765
```

- 배정 정책(등록된 베이스라인, RL 체크포인트), 로봇/워크스테이션 수, λ, 이동 규칙(free/lanes, reserve/reactive), 시나리오 분포를 골라 실행. 시뮬레이션을 끝까지 돌린 뒤 재생
- 시나리오 분포는 `scenario.py` 참고. 혼잡 상황은 프리셋에서 선택 가능
- `Export replay`: 실행 하나를 HTML 파일로 저장. `replay.html#t=900&robot=3`처럼 시점과 로봇을 지정해 열기 가능
