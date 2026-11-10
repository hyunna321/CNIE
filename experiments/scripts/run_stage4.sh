#!/usr/bin/env bash
# 단계 4: 안 B 민감도 c_phi ∈ {0, 0.5} (c_phi = 1은 단계 3), n = 10 → runs/stage4_<arm>, results/stage4/
#   검증 곡선(val_curves.csv)으로 학습 속도 비교
# 사용: experiments/scripts/run_stage4.sh [smoke|screen|final] [--jobs N]
set -euo pipefail
cd "$(dirname "$0")/../.."
MODE=${1:-screen}; shift || true
ARMS="little_pbrs_cphi0 little_pbrs_cphi0.5 little_pbrs_n10"
python -m experiments.run_stages train  --stage stage4 --arms $ARMS --mode "$MODE" "$@"
python -m experiments.run_stages report --stage stage4 --arms little_pbrs $ARMS
