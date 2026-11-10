#!/usr/bin/env bash
# 단계 0: 학습 없는 베이스라인 보강 (BL6* c_d 튜닝, BL8, tote_order 2가지, 새 best baseline 풀) → results/stage0/
# 사용: scripts/run_stage0.sh [--cells all|main|base] [--workers N] [--dry-run]
# RL(현행) − BL6* 비교는 runs/rl_current가 있을 때만 (scripts/run_rl_current.sh)
# STOP 1: results/stage0/summary.md (G1, G2 판정)
set -euo pipefail
cd "$(dirname "$0")/.."
python experiments/stages.py stage0 --out runs/stage0 --rl-root runs/rl_current "$@"
