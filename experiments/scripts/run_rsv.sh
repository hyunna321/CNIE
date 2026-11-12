#!/usr/bin/env bash
# 예약 행동 RL: arm 4개 × 배터리 {off, on} × 시드 3개 학습·테스트, 같은 조건 베이스라인. 저장소 최상위에서 실행
#   PY=.venv/bin/python bash experiments/scripts/run_rsv.sh
# 다시 실행하면 resume.pt로 이어서 학습. 로그는 runs/rsv/_logs/
set -uo pipefail
PY=${PY:-python3}
OUT=runs/rsv
LOGS=$OUT/_logs
ARMS=(rsv_local rsv_local_h3 rsv_local_h3_wd1 rsv_td_h3)
MU_ARMS=(rsv_local rsv_local_h3_wd1 rsv_td_h3)          # 중심화 표가 다른 arm만 (h3는 rsv_local과 같은 표)
SEEDS=(1 2 3)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$LOGS"

cfg() { [ "$1" = on ] && echo experiments/configs/rsv_battery.yaml || echo experiments/configs/rsv.yaml; }
sfx() { [ "$1" = on ] && echo _bat || echo ""; }

for b in off on; do
    for a in "${MU_ARMS[@]}"; do
        "$PY" train.py --config "$(cfg $b)" --arm "$a$(sfx $b)" --mu-only > "$LOGS/mu_$a$(sfx $b).log" 2>&1 &
    done
done
wait

run_one() {
    local c=$1 arm=$2 seed=$3
    "$PY" train.py --config "$c" --arm "$arm" --seed "$seed" || return 1
    local ckpt
    ckpt=$(find "$OUT" -path "*/${arm}_s${seed}/agent.pt" | head -1)
    "$PY" -m experiments.run_compare --config "$c" --load "$ckpt" --baselines
}

for b in off on; do
    for a in "${ARMS[@]}"; do
        for s in "${SEEDS[@]}"; do
            run_one "$(cfg $b)" "$a$(sfx $b)" "$s" > "$LOGS/$a$(sfx $b)_s$s.log" 2>&1 &
        done
    done
done
wait

for b in off on; do
    "$PY" -m experiments.run_compare --config "$(cfg $b)" > "$LOGS/baselines_$b.log" 2>&1
done
"$PY" -m experiments.run_compare --summarize --out "$OUT" > "$LOGS/summary.log" 2>&1
echo done > "$LOGS/ALL_DONE"
