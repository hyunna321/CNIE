#!/usr/bin/env bash
# 1단계: A0 (local 보상) + ring replay buffer + online 업데이트 (arm rsv_local_h3_ro_bat)
#   비교 대상 = 같은 설정의 A0 + PER + 에피소드 후 업데이트 (rsv_local_h3_bat, runs/rsv에 이미 있음)
#   설정 고정: M14 K3 λ320, 배터리 on, 로봇별 큐(예약), β = 1/10800, c_d = 1 prior, 1000 에피소드, 시드 1 2 3
#   PY=.venv/bin/python bash experiments/scripts/run_step1_ro.sh   (저장소 최상위에서 실행, 다시 실행하면 이어서 학습)
set -uo pipefail
PY=${PY:-python3}
CFG=experiments/configs/rsv_battery.yaml
OUT=runs/rsv
LOGS=$OUT/_logs
ARM=rsv_local_h3_ro_bat
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$LOGS"

run_one() {
    local seed=$1
    "$PY" train.py --config "$CFG" --arm "$ARM" --seed "$seed" || return 1
    local ckpt
    ckpt=$(find "$OUT" -path "*/${ARM}_s${seed}/agent.pt" | head -1)
    "$PY" -m experiments.run_compare --config "$CFG" --load "$ckpt" --baselines
}

for s in 1 2 3; do
    run_one "$s" > "$LOGS/${ARM}_s$s.log" 2>&1 &
done
wait
"$PY" -m experiments.run_compare --summarize --out "$OUT" > "$LOGS/summary_step1.log" 2>&1
echo done > "$LOGS/STEP1_DONE"
