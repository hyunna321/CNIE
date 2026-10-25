# CNIE

셀 그리드 창고 시뮬레이터 (다중 SKU 주문, 단일 SKU 토트, 로봇 이동/충돌, 피킹·WS 처리 시간) + 토트 단위 휴리스틱 배정 정책 BL1~BL6

시뮬레이션 동작 방식과 TODO: [SIMULATION.md](SIMULATION.md)

| 파일 | 내용 |
|---|---|
| `grid.py` | 셀 레이아웃, 유향 셀 그래프, 거리 테이블, SKU 배치 |
| `environment.py` | 로봇 상태, 트립 경로 (피킹 셀 NNH → 하역 도킹 셀 NNH) |
| `reservation.py` | 시공간 예약표와 구간 경로 탐색 (SIPP) |
| `simulator.py` | 주문/토트 관리, 주문-WS 배정, 틱 이동 엔진 (reserve / reactive) |
| `baselines.py` | 배정 정책 BL1~BL6 |
| `scenario.py` | 주문 도착/주문 크기/SKU 수요/저장/처리 시간 분포 |
| `metrics.py` | 에피소드 지표, CSV 저장, 비교표 |
| `run.py` | 정책 평가 실행 |
| `agents.py` | DQN 에이전트, (토트, 후보 로봇) 특징 |
| `rl_simulator.py` | RL 배정 시뮬레이터 (보상 모드 legacy / smdp) |
| `training.py` | RL 학습 루프, ε = 0 평가 |
| `run_rl.py` | RL arm 학습/평가 후 베이스라인과 비교, 결과 요약 |

## 평가

```bash
python run.py                                   # BL1~BL6, λ=320 주문/hr, M=14, K=3, 평가 시드 20개
python run.py --policies BL4 BL6 --lam 280 --robots 18 --n-eval 5 --motion reactive
```

## RL

```bash
python run_rl.py                                     # smdp_r8_bl6 arm, 1000 에피소드 학습 후 RL vs BL4~BL6 평가 (torch 필요)
python run_rl.py --arm smdp_r8_bl6 --episodes 200 --baselines BL6 --n-eval 5
python run_rl.py --arm smdp_r8_bl6 --load logs_rl/.../agent.pt   # 체크포인트 평가만
python run_rl.py --summarize                         # 설정별 arm/베이스라인 흐름시간, BL6 대비 차이
```

## 시뮬레이션 뷰어

```bash
python -m viewer.server    # http://127.0.0.1:8765
```

- 배정 정책(BL1~BL6, RL 체크포인트), 로봇/워크스테이션 수, λ, 이동 규칙(free/lanes, reserve/reactive), 시나리오 분포를 골라 실행. 시뮬레이션을 끝까지 돌린 뒤 재생
- 시나리오 분포는 `scenario.py` 참고. 혼잡 상황은 프리셋에서 선택 가능
- `Export replay`: 실행 하나를 HTML 파일로 저장. `replay.html#t=900&robot=3`처럼 시점과 로봇을 지정해 열기 가능
