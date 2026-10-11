# CNIE

시뮬레이션 동작 방식과 TODO: [SIMULATION.md](SIMULATION.md)

## 시뮬레이션 뷰어

```bash
python -m viewer.server    # http://127.0.0.1:8765
```

- 배정 정책(BL1~BL6, Greedy-R, RL 체크포인트), 로봇/워크스테이션 수, λ, 이동 규칙(free/lanes, reserve/reactive), 시나리오 분포를 골라 실행. 시뮬레이션을 끝까지 돌린 뒤 재생
- 시나리오 분포는 `scenario.py` 참고. 혼잡 상황은 프리셋에서 선택 가능
- RL 체크포인트로 실행 시 결정마다 후보별 Q값, Δd, 한계비용, BL6 선택 함께 표시
- `Export replay`: 실행 하나를 HTML 파일로 저장. `replay.html#t=900&robot=3`처럼 시점과 로봇을 지정해 열기 가능
- `run.py`, `run_rl_improve.py`로 학습하면 결과 폴더에 `*.pt`가 학습 조건과 함께 저장되고, 뷰어는 저장소 안의 `*.pt`를 목록에 표시
