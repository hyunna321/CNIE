# CNIE

## 시뮬레이션 뷰어

```bash
python -m viewer.server    # http://127.0.0.1:8765
```

- 배정 정책(BL1~BL6, Greedy-R, RL 체크포인트), 로봇/워크스테이션 수, λ, 이동 규칙(free/lanes, reserve/reactive), 시나리오 분포를 고르고 실행한다. 시뮬레이션을 끝까지 돌린 뒤 재생한다.
- 시나리오 분포는 `scenario.py` 참고. 혼잡 상황은 프리셋에서 고를 수 있다.
- RL 체크포인트로 돌리면 결정마다 후보별 Q값, Δd, 한계비용, BL6 선택을 같이 보여준다.
- `Export replay`: 실행 하나를 HTML 파일로 저장한다. `replay.html#t=900&robot=3`처럼 시점과 로봇을 지정해 열 수 있다.
- `run.py`, `run_rl_improve.py`로 학습하면 결과 폴더에 `*.pt`가 학습 조건과 함께 저장되고, 뷰어는 저장소 안의 `*.pt`를 목록에 띄운다.
