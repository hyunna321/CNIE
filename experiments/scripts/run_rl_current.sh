#!/usr/bin/env bash
# 현행 RL(local_r8_ddqn_v2) 결과 생성 → runs/rl_current (단계 0의 RL(현행) − BL6*, 단계 1~4 비교 기준)
#   main: λ 240/280/320 학습 (시드 1~5), fleet: 기본 네트워크 전이 (Table 11), factorial: K=2,4 학습 + 전이 (Table 12)
# 사용: experiments/scripts/run_rl_current.sh [--jobs N]     (seed당 단일 스레드 약 11시간, 시드는 병렬 프로세스)
set -euo pipefail
cd "$(dirname "$0")/../.."
O=(--out runs/rl_current "$@")
python -m experiments.run_plans main      --stage train    "${O[@]}"
python -m experiments.run_plans fleet     --stage transfer "${O[@]}"
python -m experiments.run_plans factorial --stage train    "${O[@]}"
python -m experiments.run_plans factorial --stage transfer "${O[@]}"
