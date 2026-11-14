#!/usr/bin/env bash
# TD ablation: 4 arm × 시드 3개 학습과 테스트 평가, 베이스라인 평가. 저장소 최상위에서 실행
#   PY=python3 bash experiments/scripts/run_td_ablation.sh
# 다시 실행하면 resume.pt로 이어서 학습. 로그는 runs/td_ablation/_logs/
set -uo pipefail
PY=${PY:-python3}
CFG=experiments/configs/td_ablation.yaml
OUT=runs/td_ablation
LOGS=$OUT/_logs
ARMS=(local_r8_ddqn_v2 td_a1_reward td_a2_buffer td_a3_online)
SEEDS=(1 2 3)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$LOGS"

# 시드끼리 공유하는 중심화 표를 먼저 한 번만 추정
"$PY" train.py --config "$CFG" --arm local_r8_ddqn_v2 --mu-only > "$LOGS/mu_local.log" 2>&1 || exit 1
"$PY" train.py --config "$CFG" --arm td_a1_reward --mu-only > "$LOGS/mu_interval.log" 2>&1 || exit 1

run_one() {
    local arm=$1 seed=$2
    "$PY" train.py --config "$CFG" --arm "$arm" --seed "$seed" || return 1
    local ckpt
    ckpt=$(find "$OUT" -path "*/${arm}_s${seed}/agent.pt" | head -1)
    "$PY" -m experiments.run_compare --config "$CFG" --load "$ckpt" --baselines
}

for arm in "${ARMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_one "$arm" "$seed" > "$LOGS/${arm}_s${seed}.log" 2>&1 &
    done
done
wait

"$PY" -m experiments.run_compare --config "$CFG" > "$LOGS/baselines.log" 2>&1
"$PY" -m experiments.run_compare --summarize --out "$OUT" > "$LOGS/summary.log" 2>&1
echo done > "$LOGS/ALL_DONE"
