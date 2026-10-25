"""리비전 코드의 기본 동작이 병합 전(v2) 코드와 비트 단위로 같은지 검증

단계 (python experiments/verify_revision.py <단계> [옵션])
    baselines : BL4, BL6, BL1F, Greedy_R를 테스트 시드 10000~10002, λ = 240, M = 14, K = 3, sim_time = SIM_TIME으로
                이 코드에서 실행해 --ref-csv (기본 logs_exp/.../lam240_M14_K3/baselines/eval.csv)의 같은 행과 비교
    rl        : lam320_M14_K3/local_r8_ddqn_v2_s1/agent.pt를 --base 코드와 이 코드에서 --sim-time 3600,
                시드 10000으로 평가해 eval.csv 비교. 이 코드는 --log-reconstructions를 켠 실행도 비교 (기록해도 결과 동일)
    train     : 기본 arm을 --episodes 3 --sim-time 1800으로 두 코드에서 학습해 curve.csv 비교
    all       : 위 세 단계

비교 제외 열: 실행 시간 측정값 (wall_s, decision_ms_*, decision_full_ms_*, plan_time_s, curve의 wall_s/sim_s)과
    리비전에서 새로 생긴 열 (기준 CSV에 없음)
결과는 --work 폴더 (기본 logs_verify/, git 제외)에 남음. 기준 코드 폴더는 git worktree add --detach ../revision_base 57e1b43
"""
import argparse
import csv
import math
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

TIMING = {"wall_s", "decision_ms_mean", "decision_ms_p95", "decision_full_ms_mean", "decision_full_ms_p95",
          "plan_time_s", "sim_s"}
CELL320 = os.path.join("grid_free_reserve", "lam320_M14_K3")
CELL240 = os.path.join("grid_free_reserve", "lam240_M14_K3")


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _same(a: str, b: str) -> bool:
    if a == b:
        return True
    try:
        x, y = float(a), float(b)
    except ValueError:
        return False
    return (math.isnan(x) and math.isnan(y)) or x == y


def compare_rows(new, ref, key_cols, label=""):
    """key_cols로 대응한 행의 공통 열(실행 시간 제외)이 모두 같은지. 출력: 불일치 수"""
    ref_by = {tuple(r[k] for k in key_cols): r for r in ref}
    bad = n = 0
    for r in new:
        k = tuple(r[c] for c in key_cols)
        if k not in ref_by:
            print(f"  [{label}] 기준에 없는 행 {k}")
            bad += 1
            continue
        o = ref_by[k]
        n += 1
        cols = [c for c in r if c in o and c not in TIMING]
        diff = [(c, r[c], o[c]) for c in cols if not _same(r[c], o[c])]
        if diff:
            bad += 1
            print(f"  [{label}] {k} 불일치: " + ", ".join(f"{c}: {a} != {b}" for c, a, b in diff[:6]))
    print(f"  [{label}] 비교 {n}행, 불일치 {bad}행")
    return bad


def run(cmd, cwd, log):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise SystemExit(f"실패 (rc={rc}): {' '.join(cmd)}  로그 {log}")


def step_baselines(args):
    from evaluation import Condition, evaluate_policy
    from constant import SIM_TIME
    print("baselines: 이 코드 vs", args.ref_csv)
    L = Condition().layout(14, 3)
    rows = []
    for key in ("BL4", "BL6", "BL1F", "GR"):
        st = evaluate_policy(L, key, 240 / 3600, [10000, 10001, 10002], SIM_TIME, verbose=False)
        out = os.path.join(args.work, f"bl_{key}.csv")
        st.save_csv(out)
        rows += _rows(out)
    ref = [r for r in _rows(args.ref_csv) if r["seed"] in ("10000", "10001", "10002")]
    for r in rows + ref:
        r["episode"] = "0"          # 캐시 CSV의 episode 번호는 시드 리스트 인덱스라 비교에서 뺌
    return compare_rows(rows, ref, ("label", "seed"), "baselines")


def step_rl(args):
    ck = os.path.join(args.logs, CELL320, "local_r8_ddqn_v2_s1", "agent.pt")
    print("rl: 체크포인트", ck)
    outs = {}
    runs = (("base", args.base, []), ("new", REPO, []),
            ("new_log", REPO, ["--log-reconstructions", os.path.join(args.work, "recon_verify.jsonl.gz")]))
    for name, code, extra in runs:
        out = os.path.join(args.work, f"rl_{name}")
        run([sys.executable, os.path.join(code, "run_rl.py"), "--load", ck, "--lam", "320", "--robots", "14",
             "--sim-time", "3600", "--n-eval", "1", "--baselines", "--out", out, "--tag", "verify"] + extra,
            code, os.path.join(args.work, f"rl_{name}.log"))
        outs[name] = os.path.join(out, CELL320, "local_r8_ddqn_v2_s1_verify", "eval.csv")
    base = _rows(outs["base"])
    return (compare_rows(_rows(outs["new"]), base, ("label", "seed"), "rl")
            + compare_rows(_rows(outs["new_log"]), base, ("label", "seed"), "rl + 기록"))


def step_train(args):
    print("train: local_r8_ddqn_v2, 3 에피소드 × 1800 s")
    outs = {}
    for name, code in (("base", args.base), ("new", REPO)):
        out = os.path.join(args.work, f"train_{name}")
        run([sys.executable, os.path.join(code, "run_rl.py"), "--arm", "local_r8_ddqn_v2", "--lam", "320",
             "--robots", "14", "--episodes", "3", "--sim-time", "1800", "--n-eval", "1", "--baselines",
             "--out", out, "--seed", "1", "--fresh"], code, os.path.join(args.work, f"train_{name}.log"))
        outs[name] = os.path.join(out, CELL320, "local_r8_ddqn_v2_s1", "curve.csv")
    bad = compare_rows(_rows(outs["new"]), _rows(outs["base"]), ("episode",), "train curve")
    ev = {k: v.replace("curve.csv", "eval.csv") for k, v in outs.items()}
    return bad + compare_rows(_rows(ev["new"]), _rows(ev["base"]), ("label", "seed"), "train eval")


def main():
    p = argparse.ArgumentParser(description="리비전 기본 동작 동일성 검증")
    p.add_argument("step", choices=["baselines", "rl", "train", "all"])
    p.add_argument("--base", default=os.path.join(os.path.dirname(REPO), "revision_base"),
                   help="병합 전 코드 폴더 (git worktree, 기본 ../revision_base)")
    p.add_argument("--logs", default=os.path.join(os.path.dirname(REPO), "main", "logs_exp"),
                   help="본실험 결과 폴더 (기준 CSV, 체크포인트)")
    p.add_argument("--ref-csv", default=None)
    p.add_argument("--work", default=None)
    args = p.parse_args()
    args.ref_csv = args.ref_csv or os.path.join(args.logs, CELL240, "baselines", "eval.csv")
    args.work = args.work or os.path.join(REPO, "logs_verify")
    os.makedirs(args.work, exist_ok=True)
    steps = ["baselines", "rl", "train"] if args.step == "all" else [args.step]
    bad = sum({"baselines": step_baselines, "rl": step_rl, "train": step_train}[s](args) for s in steps)
    print("검증 통과" if bad == 0 else f"검증 실패: 불일치 {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
