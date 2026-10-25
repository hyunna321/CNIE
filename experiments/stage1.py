"""1단계: 제안 모델(--arm, 기본 smdp_r8_bl6) vs 베이스라인 (파일럿), 기본 조건 M=14, λ=280 주문/hr (--lam), K=3

단계 (python experiments/stage1.py <단계> [옵션])
    select  : 검증 시드로 BL4·BL5·BL6·Greedy_R 평가 → 평균 흐름시간 최소를 최고 베이스라인으로 (select.json)
    train   : --arm을 학습 시드 5개로 병렬 학습, 각각 테스트 시드 30개로 평가 (run_rl.py, 시드별 로그)
              중단 후 다시 실행하면 끝난 시드는 건너뛰고, 학습 중이던 시드는 resume.pt에서 이어서 학습
              (시드 로그는 이어서 덧붙임)
    test    : 모든 베이스라인(BL1~BL6, BL1-Fill, Greedy_R)을 테스트 시드 30개로 평가 (병렬, baselines/eval.csv에 누적)
    analyze : 최고 베이스라인 기준 부트스트랩·Holm·혼합효과, 학습 시드 간 SD와 검정력 분석 (stage1_*.csv)
    all     : select → test → train → analyze (메모리 16 GB 기준: 학습 5개 동시 ≈ 2.5 GB, 베이스라인은 그 전에)

결과: {out}/grid_free_reserve/lam{λ}_M14_K3/ (2·3단계가 같은 학습 결과를 재사용)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from constant import SIM_TIME  # noqa: E402

ARM          = "smdp_r8_bl6"
TRAIN_SEEDS  = [1, 2, 3, 4, 5]
SELECT_KEYS  = ["BL4", "BL5", "BL6", "GR"]
ALL_KEYS     = ["BL1", "BL1F", "BL2", "BL3", "BL4", "BL5", "BL6", "GR"]
LAM_HR, M, K = 280, 14, 3     # λ 기본값: BL6 최대 처리량 ≈ 345 주문/hr의 약 81% (M=14)
TRAFFIC, MOTION = "free", "reserve"


def cell_dir(args) -> str:
    return os.path.join(args.out, f"grid_{TRAFFIC}_{MOTION}", f"lam{args.lam:.0f}_M{M}_K{K}")


def _layout():
    from grid import GridWarehouseLayout
    from constant import DEFAULT_SEED, ROBOT_CAPACITY
    return GridWarehouseLayout(seed=DEFAULT_SEED, num_robots=M, robot_capacity=ROBOT_CAPACITY,
                               num_workstations=K, traffic=TRAFFIC, motion=MOTION)


def _run_one(job):
    """작업자 프로세스: (정책 키, 시드, sim_time, λ (orders/hr)) 1회 실행"""
    key, seed, sim_time, lam_hr = job
    from evaluation import evaluate_policy
    return evaluate_policy(_layout(), key, lam_hr / 3600.0, [seed], sim_time, verbose=False).records[0]


def run_baselines(keys, seeds, sim_time, path, workers, lam_hr):
    """(정책, 시드)를 병렬 실행해 path CSV에 누적 (이미 있으면 건너뜀), 요청분 MetricsStore 반환"""
    from metrics import MetricsStore
    from evaluation import POLICIES, evaluate_baselines_cached
    have = set()
    if os.path.isfile(path):
        have = {(r.label, r.seed) for r in MetricsStore.load_csv(path).records
                if r.seed >= 0 and abs(r.sim_time - sim_time) < 1e-6}
    jobs = [(k, s, sim_time, lam_hr) for k in keys for s in seeds if (POLICIES[k][0], s) not in have]
    if jobs:
        print(f"베이스라인 {len(jobs)}회 실행 (병렬 {workers})", flush=True)
        t0 = time.time()
        store = MetricsStore.load_csv(path) if os.path.isfile(path) else MetricsStore()
        with ProcessPoolExecutor(workers) as ex:
            for n, r in enumerate(ex.map(_run_one, jobs), 1):
                store.add(r)
                if n % 10 == 0 or n == len(jobs):
                    print(f"  {n}/{len(jobs)}  ({time.time() - t0:.0f} s)", flush=True)
        store.save_csv(path)
    # 정렬/episode 번호 정리는 캐시 함수에 맡김 (모두 있으므로 실행 없음)
    return evaluate_baselines_cached(_layout(), keys, lam_hr / 3600.0, seeds, sim_time, path)


def step_select(args):
    from evaluation import POLICIES, val_seeds
    seeds = val_seeds(args.n_val)
    store = run_baselines(SELECT_KEYS, seeds, args.sim_time,
                          os.path.join(cell_dir(args), "baselines", "val.csv"), args.workers, args.lam)
    ranking = []
    for k in SELECT_KEYS:
        rs = [r for r in store.records if r.label == POLICIES[k][0]]
        ranking.append({"key": k, "label": POLICIES[k][0], "n": len(rs),
                        "avg_flow_time": sum(r.avg_flow_time for r in rs) / len(rs),
                        "completion_rate": sum(r.completion_rate for r in rs) / len(rs)})
    ranking.sort(key=lambda d: d["avg_flow_time"])
    print(f"\n검증 시드 {seeds[0]}~{seeds[-1]} ({len(seeds)}개) 베이스라인 순위 (평균 흐름시간)")
    for d in ranking:
        print(f"  {d['label']:20s} {d['avg_flow_time']:8.1f} s   완료율 {d['completion_rate']:.3f}")
    sel = {"best_key": ranking[0]["key"], "best_label": ranking[0]["label"], "val_seeds": seeds,
           "sim_time": args.sim_time, "ranking": ranking}
    with open(os.path.join(cell_dir(args), "select.json"), "w", encoding="utf-8") as f:
        json.dump(sel, f, indent=1, ensure_ascii=False)
    print(f"최고 베이스라인: {sel['best_label']}")
    return sel


def step_train(args):
    """학습 시드마다 run_rl.py 프로세스 1개 (베이스라인은 test 단계에서 따로, 같은 CSV 동시 쓰기 방지)"""
    os.makedirs(os.path.join(cell_dir(args), "logs"), exist_ok=True)
    procs = []
    done = lambda s: all(os.path.isfile(os.path.join(cell_dir(args), f"{args.arm}_s{s}", f))
                         for f in ("agent.pt", "eval.csv"))
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    base = [sys.executable, "-u", os.path.join(REPO, "run_rl.py"), "--arm", args.arm,
            "--lam", str(args.lam), "--robots", str(M), "--ws", str(K), "--traffic", TRAFFIC,
            "--motion", MOTION, "--episodes", str(args.episodes), "--sim-time", str(args.sim_time),
            "--n-eval", str(args.n_test), "--select", args.select, "--val-every", str(args.val_every),
            "--n-val", str(args.rl_n_val), "--baselines", "--out", args.out]
    if not all(done(s) for s in args.seeds):
        # 보상 기준점 μ(λ) 표를 셀에서 한 번 만들어 모든 시드가 공유 (arm이 mu를 안 쓰면 바로 종료)
        if subprocess.run(base + ["--mu-only"], cwd=REPO, env=env).returncode != 0:
            print("μ(λ) 추정 실패", flush=True)
            return False
    for s in args.seeds:
        if done(s):
            print(f"seed {s}: 이미 학습·평가 완료, 건너뜀", flush=True)
            continue
        log = os.path.join(cell_dir(args), "logs", f"{args.arm}_s{s}.log")
        cmd = base + ["--seed", str(s)]
        print(f"학습 시작 seed {s}: 로그 {log}", flush=True)
        procs.append((s, subprocess.Popen(cmd, cwd=REPO, env=env, stdout=open(log, "a", encoding="utf-8"),
                                          stderr=subprocess.STDOUT)))
    fails = []
    for s, p in procs:
        if p.wait() != 0:
            fails.append(s)
    print("학습 완료" + (f", 실패 시드 {fails} (로그 확인)" if fails else ""), flush=True)
    return not fails


def step_test(args):
    from evaluation import test_seeds
    run_baselines(ALL_KEYS, test_seeds(args.n_test), args.sim_time,
                  os.path.join(cell_dir(args), "baselines", "eval.csv"), args.workers, args.lam)


def step_analyze(args):
    import analysis as A
    with open(os.path.join(cell_dir(args), "select.json"), encoding="utf-8") as f:
        ref = json.load(f)["best_label"]
    rows = [r for r in A.load_results(args.out) if abs(r["lam"] - args.lam) < 0.5 and r["M"] == M and r["K"] == K
            and r["setting"] == f"grid_{TRAFFIC}_{MOTION}"]
    rows = [r for r in rows if r["policy"] == args.arm or r["run"] == r["policy"]]   # 1단계: arm + 베이스라인
    prefix = os.path.join(cell_dir(args), "stage1" if args.arm == ARM else f"stage1_{args.arm}")
    n_rl = len({r["train_seed"] for r in rows if r["policy"] == args.arm})
    print(f"기준(검증 선택) {ref}, {args.arm} 학습 시드 {n_rl}개")

    for rel in (False, True):
        res = A.compare(rows, ref, args.B, "crossed", relative=rel)
        unit = "%" if rel else " s"
        print(f"\n흐름시간 차이 (정책 − {ref}{', 상대' if rel else ''}), crossed 부트스트랩 B={args.B}, Holm {len(res)}개")
        for r in sorted(res, key=lambda r: r["diff"]):
            agree = f"  BL6일치 {r['bl6_agree']:.3f}" if r["bl6_agree"] == r["bl6_agree"] else ""
            print(f"  {r['policy']:22s} {r['diff']:+9.2f}{unit} [{r['ci_lo']:+9.2f}, {r['ci_hi']:+9.2f}]  "
                  f"p_holm={r['p_holm']:.4f}  (학습 {r['n_train']} × 테스트 {r['n_test']}){agree}")
        A._save(res, prefix + ("_bootstrap_rel.csv" if rel else "_bootstrap.csv"))

    mixed = A.fit_mixed(rows, ref)
    print(f"\n혼합효과 y ~ policy + (1|테스트 시드) + (1|run), 기준 {ref}")
    for r in mixed:
        term = r["term"].split("[T.")[-1].rstrip("]")
        print(f"  {term:22s} {r['coef']:+9.2f} ± {r['se']:.2f}  p_holm={r['p_holm']:.4f}  ({r['vc']})"
              + ("" if r["converged"] else "  [수렴 경고]"))
    A._save(mixed, prefix + "_mixed.csv")

    rep = A.power_report(rows, args.arm, ref, (2, 5, 10), range(3, 11), alpha=0.05,
                         n_tests=len(ALL_KEYS), target=0.8)
    flat = []
    for cell, r in rep.items():
        vc = r["vc"]
        print(f"\n검정력: {args.arm} − {ref}, α = 0.05 / {len(ALL_KEYS)} (Holm 첫 단계), 테스트 시드 {r['n_test']}개")
        print(f"  학습 시드별 평균 차이의 SD {vc['sd_train_means']:.2f} s  (관측 평균 차이 {vc['mean']:+.2f} s)")
        print(f"  분산성분 σ_학습 {vc['var_train'] ** .5:.2f}, σ_테스트 {vc['var_test'] ** .5:.2f}, "
              f"σ_잔차 {vc['var_resid'] ** .5:.2f}  ({ref} 평균 {r['ref_mean']:.1f} s)")
        labels = list(r["n_needed"])
        deltas = {c["label"]: c["delta"] for c in r["curve"]}
        print("  n_train" + "".join(f"  {l}(δ={deltas[l]:.1f})".rjust(18) for l in labels))
        for n in range(3, 11):
            vals = {c["label"]: c["power"] for c in r["curve"] if c["n_train"] == n}
            print(f"  {n:7d}" + "".join(f"{vals[l]:18.3f}" for l in labels))
        print("  검정력 0.8 이상 최소 학습 시드: " +
              ", ".join(f"{l} → {r['n_needed'][l] or '>10'}" for l in labels))
        for c in r["curve"]:
            flat.append({**c, "sd_train_means": vc["sd_train_means"], "var_train": vc["var_train"],
                         "var_test": vc["var_test"], "var_resid": vc["var_resid"], "ref": ref})
    A._save(flat, prefix + "_power.csv")


def main():
    p = argparse.ArgumentParser(description="1단계: 제안 모델 vs 베이스라인")
    p.add_argument("step", choices=["select", "train", "test", "analyze", "all"])
    p.add_argument("--out", default=os.path.join(REPO, "logs_exp"), help="결과 상위 폴더")
    p.add_argument("--arm", default=ARM, help="학습할 run_rl.py arm (예: smdp_r8_bl6, local_r8_ddqn)")
    p.add_argument("--lam", type=float, default=LAM_HR, help="주문 도착률 (orders/hr)")
    p.add_argument("--seeds", type=int, nargs="+", default=TRAIN_SEEDS, help="학습 시드")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--sim-time", type=float, default=SIM_TIME)
    p.add_argument("--n-val", type=int, default=10, help="베이스라인 선택용 검증 시드 수")
    p.add_argument("--n-test", type=int, default=30, help="테스트 시드 수")
    p.add_argument("--select", choices=["last", "val"], default="last",
                   help="RL 테스트 체크포인트: 마지막 가중치 또는 검증 시드 최고 시점 (run_rl.py --select)")
    p.add_argument("--val-every", type=int, default=50, help="--select val 검증 주기 (에피소드)")
    p.add_argument("--rl-n-val", type=int, default=5, help="--select val 검증 시드 수 (RL 체크포인트 선택용)")
    p.add_argument("--workers", type=int, default=3,
                   help="베이스라인 병렬 프로세스 수 (프로세스당 약 200 MB)")
    p.add_argument("--B", type=int, default=10000, help="부트스트랩 반복 수")
    args = p.parse_args()

    if args.step in ("select", "all"):
        step_select(args)
    if args.step == "all":
        step_test(args)
        if step_train(args):
            step_analyze(args)
        return
    {"train": step_train, "test": step_test, "analyze": step_analyze}.get(args.step, lambda a: None)(args)


if __name__ == "__main__":
    main()
