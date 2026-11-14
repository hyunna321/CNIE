#!/usr/bin/env bash
# 단계 3: 안 B Little 비용 + PBRS (arm little_pbrs: c_phi = 1, n = 3, β = 1/300) → runs/stage3_little_pbrs
# 이어서 STOP 2 보고: 현행 / 단계 1 / 안 A / 안 B → results/stop2/summary.md
# 사용: experiments/scripts/run_stage3.sh [smoke|screen|final] [--jobs N]
set -euo pipefail
cd "$(dirname "$0")/../.."
MODE=${1:-screen}; shift || true
python -m experiments.run_stages train  --stage stage3 --arms little_pbrs --mode "$MODE" "$@"
python -m experiments.run_stages report --stage stop2 --arms local_r8_ddqn_v2 local_noterm2 order_inc little_pbrs
