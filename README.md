# CNIE

셀 그리드 창고 시뮬레이터 (로봇 이동/충돌, 피킹·WS 처리 시간) + 휴리스틱 배정 정책 BL1~BL6

시뮬레이션 동작 방식과 TODO: [SIMULATION.md](SIMULATION.md)

| 파일 | 내용 |
|---|---|
| `grid.py` | 셀 레이아웃, 유향 셀 그래프, 거리 테이블, SKU 배치 |
| `environment.py` | 로봇 상태, NNH 경로, 복귀 WS 선택 |
| `reservation.py` | 시공간 예약표와 구간 경로 탐색 (SIPP) |
| `simulator.py` | 주문 관리, 틱 이동 엔진 (reserve / reactive) |
| `baselines.py` | 배정 정책 BL1~BL6 |
| `scenario.py` | 주문 도착/SKU 수요/저장/처리 시간 분포 |
| `metrics.py` | 에피소드 지표, CSV 저장, 비교표 |
| `run.py` | 정책 평가 실행 |

## 평가

```bash
python run.py                                   # BL1~BL6, λ=800/hr, M=14, K=3, 평가 시드 20개
python run.py --policies BL4 BL6 --lam 900 --robots 18 --n-eval 5 --motion reactive
```

## 시뮬레이션 뷰어

```bash
python -m viewer.server    # http://127.0.0.1:8765
```

- 배정 정책(BL1~BL6), 로봇/워크스테이션 수, λ, 이동 규칙(free/lanes, reserve/reactive), 시나리오 분포를 골라 실행. 시뮬레이션을 끝까지 돌린 뒤 재생
- 시나리오 분포는 `scenario.py` 참고. 혼잡 상황은 프리셋에서 선택 가능
- `Export replay`: 실행 하나를 HTML 파일로 저장. `replay.html#t=900&robot=3`처럼 시점과 로봇을 지정해 열기 가능
