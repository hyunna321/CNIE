#!/usr/bin/env bash
# 단계 6: 최종 보상 설정으로 5 seed 재학습 + 표·그림 재생성 (새 best baseline 풀)
# 사용: ARM=<최종 arm> experiments/scripts/run_stage6.sh [--jobs N]
#   Table 8·9 = analysis tab_main/tab_sources, Table 11 = tab_fleet, Table 12 = tab_conditions, Figure 5 = fig_curves
#   Figure 6 = eval.csv의 bl6_agree/agree_cd0/agree_bl8/cd_reproducible (run_stages report)
#   CRN 450 branch 분석 스크립트는 저장소에 없음 (archive_20261109/logs_exp/rl_vs_bl6_behavior/scripts에서 이식 필요)
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${ARM:?ARM=<최종 arm>을 지정하세요}"
O=(--out runs/final --arm "$ARM" "$@")
for P in main fleet small factorial routing; do python -m experiments.run_plans $P "${O[@]}"; done
python -m experiments.run_stages stage0 --cells all --out runs/final --rl-root runs/final
python -m experiments.analysis --out runs/final --arm "$ARM" --pool new
python -m experiments.run_stages report --stage stage6 --arms "$ARM" --out runs/final --rl-root runs/final --runs runs/final
