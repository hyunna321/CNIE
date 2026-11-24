#!/usr/bin/env bash
# R2 −queue 계수 민감도: 기준 rsv_r2_ord_eta0.2_noq_bat (w_ext 1, w_q 0, η 0.2, 학습 시드 1~5는 run_r2_ablation.sh)에서
#   계수 하나만 바꾼 arm 7개 × 학습 시드 1~5 (env/rl.py ARMS rsv_r2_noq_*)
#     η     ∈ {0, 0.1, 0.5, 1}  (w_ext 1)   rsv_r2_noq_eta{η}_bat
#     w_ext ∈ {0, 0.5, 2}       (η 0.2)     rsv_r2_noq_wext{w}_bat
#   설정 고정: M14 K3 λ320, 배터리 on, 로봇별 큐(예약), β = 1/10800, 1000 에피소드
#   학습 35개 동시 (9965WX 24코어/48스레드). 다시 실행하면 resume으로 이어서 학습, 끝난 테스트는 건너뜀
#   PY=.venv/bin/python bash experiments/scripts/run_r2_sens.sh   (저장소 최상위에서 실행)
#   결과 정리: results/r2_sens
set -uo pipefail
PY=${PY:-python3}
CFG=experiments/configs/rsv_battery.yaml
OUT=runs/rsv
LOGS=$OUT/_logs
ARMS=(rsv_r2_noq_eta0_bat rsv_r2_noq_eta0.1_bat rsv_r2_noq_eta0.5_bat rsv_r2_noq_eta1_bat
      rsv_r2_noq_wext0_bat rsv_r2_noq_wext0.5_bat rsv_r2_noq_wext2_bat)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$LOGS"

# μ(λ) 표: η가 다르면 다름 (w_ext만 다른 arm은 값이 같지만 파일은 따로). 동시에 만들면 메모리 부족이라 하나씩
for a in "${ARMS[@]}"; do
    echo "μ: $a"
    "$PY" train.py --config "$CFG" --arm "$a" --mu-only > "$LOGS/mu_$a.log" 2>&1 || { echo "μ 실패: $a ($LOGS/mu_$a.log)"; exit 1; }
done

run_one() {
    local arm=$1 seed=$2
    if [ -f "$LOGS/${arm}_s${seed}.done" ]; then return 0; fi
    "$PY" train.py --config "$CFG" --arm "$arm" --seed "$seed" || return 1
    local ckpt
    ckpt=$(find "$OUT" -path "*/${arm}_s${seed}/agent.pt" | head -1)
    "$PY" -m experiments.run_compare --config "$CFG" --load "$ckpt" --baselines || return 1
    touch "$LOGS/${arm}_s${seed}.done"
}

for a in "${ARMS[@]}"; do
    for s in 1 2 3 4 5; do
        run_one "$a" "$s" > "$LOGS/${a}_s$s.log" 2>&1 &
    done
done
wait
n_done=$(ls "$LOGS"/rsv_r2_noq_*_s*.done 2>/dev/null | wc -l)
echo "완료 ${n_done}/35" > "$LOGS/R2_SENS_DONE"
cat "$LOGS/R2_SENS_DONE"
