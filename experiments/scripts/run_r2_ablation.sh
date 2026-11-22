#!/usr/bin/env bash
# R2 보상 항 ablation: Full(rsv_r2_ord_eta0.2_bat)에서 보상 항을 하나씩·조합으로 뺀 arm 5개 × 학습 시드 1~5
#   + Full 학습 시드 4·5 추가 (Full 1~3은 results/r2_eta0.2). 학습 설정은 Full과 같고 보상 계수만 다름 (env/rl.py ARMS)
#     c_k = inc + w_q · 큐 inc + η · 자기 지연 + w_ext · (E − X)
#     rsv_r2_ord_bat               −η          (w_ext 1, w_q 1, η 0)
#     rsv_r2_ord_eta0.2_noext_bat  −ext        (w_ext 0, w_q 1, η 0.2)
#     rsv_r2_ord_eta0.2_noq_bat    −queue      (w_ext 1, w_q 0, η 0.2)
#     rsv_r2_ord_noext_bat         −ext −η     (w_ext 0, w_q 1, η 0)
#     rsv_r2_ord_inconly_bat       inc only    (w_ext 0, w_q 0, η 0)
#   설정 고정: M14 K3 λ320, 배터리 on, 로봇별 큐(예약), β = 1/10800, 1000 에피소드
#   학습 27개 동시 (9965WX 24코어 기준 약 14~16시간). 다시 실행하면 resume으로 이어서 학습, 끝난 테스트는 건너뜀
#   PY=.venv/bin/python bash experiments/scripts/run_r2_ablation.sh   (저장소 최상위에서 실행)
#   결과 정리: results/r2_ablation
set -uo pipefail
PY=${PY:-python3}
CFG=experiments/configs/rsv_battery.yaml
OUT=runs/rsv
LOGS=$OUT/_logs
FULL=rsv_r2_ord_eta0.2_bat
ABL=(rsv_r2_ord_bat rsv_r2_ord_eta0.2_noext_bat rsv_r2_ord_eta0.2_noq_bat rsv_r2_ord_noext_bat rsv_r2_ord_inconly_bat)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$LOGS"

# μ(λ) 표: arm마다 다름. 동시에 만들면 메모리 부족이라 하나씩 (이미 있으면 바로 끝남)
for a in "$FULL" "${ABL[@]}"; do
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

for s in 4 5; do
    run_one "$FULL" "$s" > "$LOGS/${FULL}_s$s.log" 2>&1 &
done
for a in "${ABL[@]}"; do
    for s in 1 2 3 4 5; do
        run_one "$a" "$s" > "$LOGS/${a}_s$s.log" 2>&1 &
    done
done
wait
"$PY" -m experiments.run_compare --summarize --out "$OUT" > "$LOGS/summary_r2_ablation.log" 2>&1
n_done=$(ls "$LOGS"/rsv_r2_ord*_s*.done 2>/dev/null | wc -l)
echo "완료 ${n_done}/27" > "$LOGS/R2_ABLATION_DONE"
cat "$LOGS/R2_ABLATION_DONE"
