#!/usr/bin/env bash
# 단계 5: RL 고유 영역 시나리오 (multi: SKU 수 평균 4 / steps: 2시간 구간 λ = [200, 320, 200, 280, 200])
#   시나리오마다 BL6*·BL8 재튜닝·평가 (기준 셀) → 안 B arm 학습 (steps는 커리큘럼 없는 _nocurr)
# 사용: ARM=little_pbrs scripts/run_stage5.sh [smoke|screen|final] [--jobs N]   (ARM = 단계 4의 최선 설정)
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-screen}; shift || true
ARM=${ARM:-little_pbrs}
for SC in multi steps; do
  A=$ARM; [ "$SC" = steps ] && A=${ARM}_nocurr
  python experiments/stages.py stage0 --cells base --scenario $SC --rl-root runs/none
  python experiments/stages.py train  --stage stage5_$SC --arms $A --scenario $SC --mode "$MODE" "$@"
  python experiments/stages.py report --stage stage5_$SC --arms $A --scenario $SC --out runs/stage5_${SC}_baselines
done
