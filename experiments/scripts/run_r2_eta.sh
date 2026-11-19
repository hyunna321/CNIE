#!/usr/bin/env bash
# R2 보상: 주문 증분 + 외부비용 이전 + η 0.2 (arm rsv_r2_ord_eta0.2_bat, env.rl.OrderExtReward)
#   기반 = A0 + ring + online (rsv_local_h3_ro_bat, results/step1_ro)과 보상 설정(reward, eta)만 다름
#   설정 고정: M14 K3 λ320, 배터리 on, 로봇별 큐(예약), β = 1/10800, c_d = 1 prior, 1000 에피소드, 시드 1 2 3
#   PY=.venv/bin/python bash experiments/scripts/run_r2_eta.sh   (저장소 최상위에서 실행, 다시 실행하면 이어서 학습)
#   결과 정리: results/r2_eta0.2 (test_summary.csv, eval_s1.csv)
set -uo pipefail
PY=${PY:-python3}
CFG=experiments/configs/rsv_battery.yaml
OUT=runs/rsv
LOGS=$OUT/_logs
ARM=rsv_r2_ord_eta0.2_bat
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$LOGS"

run_one() {
    local seed=$1
    "$PY" train.py --config "$CFG" --arm "$ARM" --seed "$seed" || return 1
    local ckpt
    ckpt=$(find "$OUT" -path "*/${ARM}_s${seed}/agent.pt" | head -1)
    "$PY" -m experiments.run_compare --config "$CFG" --load "$ckpt" --baselines
}

# 시드 1이 μ(λ) 표(mu_order_ext_eta0.2_rsv.json)를 만든 뒤 시드 2·3을 띄움 (μ 작업자가 겹치면 메모리 부족)
run_one 1 > "$LOGS/${ARM}_s1.log" 2>&1 &
until grep -qE "사용$|λ=320\.0/hr\)|\[Ep |Traceback" "$LOGS/${ARM}_s1.log" 2>/dev/null; do sleep 20; done
for s in 2 3; do
    run_one "$s" > "$LOGS/${ARM}_s$s.log" 2>&1 &
done
wait
"$PY" -m experiments.run_compare --summarize --out "$OUT" > "$LOGS/summary_r2_eta.log" 2>&1
echo done > "$LOGS/R2_ETA_DONE"
