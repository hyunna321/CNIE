#!/usr/bin/env bash
# R4 가중치: A0 local 보상에서 탑재 토트 지연 계수 w_ext만 1 → 2, 3 (arm rsv_r4_wext{2,3}_bat)
#   기반 = A0 + ring + online (rsv_local_h3_ro_bat, results/step1_ro)과 local_w만 다름
#   설정 고정: M14 K3 λ320, 배터리 on, 로봇별 큐(예약), β = 1/10800, c_d = 1 prior, 1000 에피소드, 시드 1 2 3
#   PY=.venv/bin/python bash experiments/scripts/run_r4_wext.sh   (저장소 최상위에서 실행, 다시 실행하면 이어서 학습)
#   JOBS = 동시 실행 수 (기본 6). 결과 정리: results/r4_wext
set -uo pipefail
PY=${PY:-python3}
JOBS=${JOBS:-6}
CFG=experiments/configs/rsv_battery.yaml
OUT=runs/rsv
LOGS=$OUT/_logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$LOGS"

run_one() {
    local arm=$1 seed=$2
    "$PY" train.py --config "$CFG" --arm "$arm" --seed "$seed" || return 1
    local ckpt
    ckpt=$(find "$OUT" -path "*/${arm}_s${seed}/agent.pt" | head -1)
    "$PY" -m experiments.run_compare --config "$CFG" --load "$ckpt" --baselines
}

# arm마다 시드 1이 μ(λ) 표(local_mu_wext{w}_wq1_rsv.json)를 만든 뒤 시드 2·3을 띄움 (μ 작업자가 겹치면 메모리 부족)
mu_ready() { grep -qE "사용$|λ=320\.0/hr\)|\[Ep |Traceback" "$1" 2>/dev/null; }
for arm in rsv_r4_wext2_bat rsv_r4_wext3_bat; do
    for s in 1 2 3; do
        while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
        run_one "$arm" "$s" > "$LOGS/${arm}_s$s.log" 2>&1 &
        if [ "$s" = 1 ]; then until mu_ready "$LOGS/${arm}_s1.log"; do sleep 20; done; fi
    done
done
wait
"$PY" -m experiments.run_compare --summarize --out "$OUT" > "$LOGS/summary_r4_wext.log" 2>&1
echo done > "$LOGS/R4_WEXT_DONE"
