#!/usr/bin/env bash
# 단계 1: 현행 보상에서 2항만 제거 (arm local_noterm2) → runs/stage1_local_noterm2, results/stage1/
# 사용: scripts/run_stage1.sh [smoke|screen|final] [--jobs N]     (기본 screen = 시드 1~3)
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-screen}; shift || true
python experiments/stages.py train  --stage stage1 --arms local_noterm2 --mode "$MODE" "$@"
python experiments/stages.py report --stage stage1 --arms local_r8_ddqn_v2 local_noterm2
