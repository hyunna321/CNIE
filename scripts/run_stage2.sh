#!/usr/bin/env bash
# 단계 2: 안 A 주문 누적 증분 비용 (arm order_inc: n = 5, ARMS="order_inc order_inc_n10"이면 보조 n = 10도)
#   features = order, tote_order = bottleneck_first, ρ(λ) 중심화 → runs/stage2_<arm>, results/stage2/
# 사용: [ARMS="order_inc order_inc_n10"] scripts/run_stage2.sh [smoke|screen|final] [--jobs N]
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-screen}; shift || true
ARMS=${ARMS:-order_inc}
python experiments/stages.py train  --stage stage2 --arms $ARMS --mode "$MODE" "$@"
python experiments/stages.py report --stage stage2 --arms local_r8_ddqn_v2 $ARMS
