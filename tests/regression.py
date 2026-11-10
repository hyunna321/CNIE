"""회귀 테스트: 기본 설정(M = 14, K = 3, λ = 240)에서 현행 RL·베이스라인 결과가 기준(golden)과 같은지

python tests/regression.py                # 지금 코드로 실행해 tests/golden과 비교 (약 5분)
python tests/regression.py --make         # golden 다시 만들기 (기준 코드에서만)
python tests/regression.py --repo PATH    # 다른 코드 트리로 실행 (golden을 만든 기준 코드 등)

실행 내용 (짧은 실행, 같은 시드면 비트 단위로 같아야 함)
  - 베이스라인: 기존 정책 전체 (BL1, BL1F, BL2, BL3, BL4, BL5, BL6, BL6D0, GR) × 테스트 시드 10000, 10001 × 1800 s
  - RL 기본 arm 학습: 학습 시드 1, 3 에피소드 × 900 s, μ(λ) 추정, 검증 선택 (2 에피소드마다 검증 1개), 테스트 2개
비교: 실행 시간 열 제외, golden에 없는 새 열은 무시 (새 지표 추가는 회귀 아님)
golden: tests/golden/ (2026-11-10 보상 재설계 작업 전 코드 = cleanup 브랜치 정리본에서 생성)
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
GOLDEN = os.path.join(HERE, "golden")
TIMING = {"wall_s", "decision_ms_mean", "decision_ms_p95", "decision_full_ms_mean", "decision_full_ms_p95",
          "plan_time_s", "sim_s"}
BASELINES = ["BL1", "BL1F", "BL2", "BL3", "BL4", "BL5", "BL6", "BL6D0", "GR"]
CELL = os.path.join("rl", "grid_free_reserve", "lam240_M14_K3")
FILES = ["baselines.csv", os.path.join(CELL, "local_mu.json")] + [
    os.path.join(CELL, "local_r8_ddqn_v2_s1", f) for f in ("train.csv", "curve.csv", "val.csv", "eval.csv")]


def run(repo: str, out: str):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    code = (f"import sys; sys.path.insert(0, {repo!r})\n"
            "from evaluation import Condition, evaluate_policy\nfrom metrics import MetricsStore\n"
            "L = Condition().layout(14, 3); st = MetricsStore()\n"
            f"for k in {BASELINES!r}:\n"
            "    evaluate_policy(L, k, 240/3600, [10000, 10001], 1800, store=st, verbose=False)\n"
            f"st.save_csv({os.path.join(out, 'baselines.csv')!r})\n")
    subprocess.run([sys.executable, "-c", code], cwd=repo, env=env, check=True)
    subprocess.run([sys.executable, "run_rl.py", "--lam", "240", "--robots", "14", "--ws", "3", "--episodes", "3",
                    "--sim-time", "900", "--n-eval", "2", "--select", "val", "--val-every", "2", "--n-val", "1",
                    "--baselines", "--seed", "1", "--out", os.path.join(out, "rl")],
                   cwd=repo, env=env, check=True, stdout=subprocess.DEVNULL)


def compare(golden: str, out: str) -> int:
    bad = 0
    for rel in FILES:
        a, b = os.path.join(golden, rel), os.path.join(out, rel)
        if rel.endswith(".json"):
            same = json.load(open(a)) == json.load(open(b))
            diffs = [] if same else ["내용"]
        else:
            ra, rb = list(csv.DictReader(open(a))), list(csv.DictReader(open(b)))
            diffs = (["행 수"] if len(ra) != len(rb) else
                     sorted({k for x, y in zip(ra, rb) for k in x if k not in TIMING and x[k] != y.get(k)}))
        print(("OK  " if not diffs else "DIFF") + f" {rel}" + (f"  {diffs}" if diffs else ""))
        bad += bool(diffs)
    print("회귀 테스트", "통과" if bad == 0 else f"실패 ({bad}개 파일)")
    return bad


def main():
    p = argparse.ArgumentParser(description="회귀 테스트 (모듈 설명)")
    p.add_argument("--make", action="store_true", help="golden 다시 만들기")
    p.add_argument("--repo", default=REPO, help="실행할 코드 트리")
    args = p.parse_args()
    out = tempfile.mkdtemp(prefix="mtsr_reg_")
    try:
        run(os.path.abspath(args.repo), out)
        if args.make:
            for rel in FILES:
                os.makedirs(os.path.dirname(os.path.join(GOLDEN, rel)) or GOLDEN, exist_ok=True)
                shutil.copy(os.path.join(out, rel), os.path.join(GOLDEN, rel))
            print(f"golden 저장: {GOLDEN}")
            return
        sys.exit(compare(GOLDEN, out))
    finally:
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    main()
