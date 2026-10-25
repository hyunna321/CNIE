"""라우팅/이동 조건 ablation: 고정 가중치(frozen) vs 같은 구조 재학습(retrain), 같은 학습 시드·테스트 시드

기준 조건 (stage1과 같음): M=14, K=3, λ=280 주문/hr, grid free/reserve, 기준 체크포인트는 stage1 학습 결과
변형 조건: --variants "traffic/motion" 목록 (예: lanes/reserve free/reactive), 워크스테이션 배치(K)는 그대로

단계 (python experiments/ablation.py <단계> [옵션])
    frozen  : 기준 조건 학습 시드별 agent.pt를 가중치 그대로 변형 조건에서 평가 (run_rl.py --load, tag frozen)
    retrain : 변형 조건에서 같은 arm·하이퍼파라미터·학습 시드로 다시 학습 후 평가 (tag retrain)
    test    : 변형 조건마다 베이스라인을 같은 테스트 시드로 평가 (baselines/eval.csv에 누적)
    analyze : 변형 조건마다 frozen, retrain의 기준 베이스라인 대비 차이와 retrain − frozen (정책 적응 효과),
              학습 시드 간 SD, crossed 부트스트랩 CI (ablation_*.csv)
    all     : frozen → test → retrain → analyze

차이 해석
    frozen − 기준 조건 RL : 라우팅/이동 조건 변화만의 효과 (정책 고정)
    retrain − frozen      : 같은 조건에서 정책이 적응해 얻는 추가 효과
시드: 학습 시드 --seeds (기준 체크포인트와 같은 값), 테스트 시드 evaluation.test_seeds(--n-test), 모든 조건 공통

결과: {out}/grid_{traffic}_{motion}/lam{λ}_M14_K3/{arm}_s{seed}_{frozen|retrain}/
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from constant import SIM_TIME  # noqa: E402
from experiments.stage1 import ARM, TRAIN_SEEDS, ALL_KEYS, LAM_HR, M, K, TRAFFIC, MOTION  # noqa: E402

VARIANTS = ["lanes/reserve", "free/reactive"]


def setting(tm: str) -> str:
    traffic, motion = tm.split("/")
    return f"grid_{traffic}_{motion}"


def cell_dir(args, tm: str) -> str:
    return os.path.join(args.out, setting(tm), f"lam{args.lam:.0f}_M{M}_K{K}")


def base_ckpt(args, seed: int) -> str:
    return os.path.join(cell_dir(args, f"{TRAFFIC}/{MOTION}"), f"{args.arm}_s{seed}", "agent.pt")


def _layout(tm: str):
    from grid import GridWarehouseLayout
    from constant import DEFAULT_SEED, ROBOT_CAPACITY
    traffic, motion = tm.split("/")
    return GridWarehouseLayout(seed=DEFAULT_SEED, num_robots=M, robot_capacity=ROBOT_CAPACITY,
                               num_workstations=K, traffic=traffic, motion=motion)


def _run_one(job):
    """작업자 프로세스: (조건, 정책 키, 시드, sim_time, λ) 1회 실행"""
    tm, key, seed, sim_time, lam_hr = job
    from evaluation import evaluate_policy
    return tm, evaluate_policy(_layout(tm), key, lam_hr / 3600.0, [seed], sim_time, verbose=False).records[0]


def _launch(args, cmds):
    """(로그 경로, 명령) 목록을 동시에 실행하고 실패한 로그 목록 반환"""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    procs = []
    for log, cmd in cmds:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        print(f"시작: 로그 {log}", flush=True)
        procs.append((log, subprocess.Popen(cmd, cwd=REPO, env=env, stdout=open(log, "w", encoding="utf-8"),
                                            stderr=subprocess.STDOUT)))
    return [log for log, p in procs if p.wait() != 0]


def _common(args, tm: str):
    traffic, motion = tm.split("/")
    return ["--lam", str(args.lam), "--robots", str(M), "--ws", str(K), "--traffic", traffic, "--motion", motion,
            "--sim-time", str(args.sim_time), "--n-eval", str(args.n_test), "--baselines", "--out", args.out]


def step_frozen(args):
    """기준 체크포인트를 변형 조건에서 평가 (가중치 고정)"""
    cmds = []
    for s in args.seeds:
        ck = base_ckpt(args, s)
        if not os.path.isfile(ck):
            print(f"seed {s}: 기준 체크포인트 없음 ({ck}), stage1.py train 먼저", flush=True)
            continue
        for tm in args.variants:
            if os.path.isfile(os.path.join(cell_dir(args, tm), f"{args.arm}_s{s}_frozen", "eval.csv")):
                continue
            log = os.path.join(cell_dir(args, tm), "logs", f"{args.arm}_s{s}_frozen.log")
            cmds.append((log, [sys.executable, "-u", os.path.join(REPO, "run_rl.py"), "--load", ck,
                               "--tag", "frozen"] + _common(args, tm)))
    fails = _launch(args, cmds)
    print("frozen 평가 완료" + (f", 실패 {fails}" if fails else ""), flush=True)
    return not fails


def step_retrain(args):
    """변형 조건에서 같은 arm·학습 시드로 재학습 (동시 실행 수 --jobs)"""
    cmds = []
    for tm in args.variants:
        for s in args.seeds:
            d = os.path.join(cell_dir(args, tm), f"{args.arm}_s{s}_retrain")
            if all(os.path.isfile(os.path.join(d, f)) for f in ("agent.pt", "eval.csv")):
                continue
            log = os.path.join(cell_dir(args, tm), "logs", f"{args.arm}_s{s}_retrain.log")
            cmds.append((log, [sys.executable, "-u", os.path.join(REPO, "run_rl.py"), "--arm", args.arm,
                               "--seed", str(s), "--episodes", str(args.episodes), "--select", "last",
                               "--tag", "retrain"] + _common(args, tm)))
    fails = []
    for i in range(0, len(cmds), args.jobs):
        fails += _launch(args, cmds[i:i + args.jobs])
    print("재학습 완료" + (f", 실패 {fails}" if fails else ""), flush=True)
    return not fails


def step_test(args):
    """변형 조건마다 베이스라인 평가 (병렬 실행 후 조건별 CSV에 누적)"""
    from metrics import MetricsStore
    from evaluation import POLICIES, test_seeds, evaluate_baselines_cached
    seeds = test_seeds(args.n_test)
    stores, jobs = {}, []
    for tm in args.variants:
        path = os.path.join(cell_dir(args, tm), "baselines", "eval.csv")
        stores[tm] = MetricsStore.load_csv(path) if os.path.isfile(path) else MetricsStore()
        have = {(r.label, r.seed) for r in stores[tm].records
                if r.seed >= 0 and abs(r.sim_time - args.sim_time) < 1e-6}
        jobs += [(tm, k, s, args.sim_time, args.lam) for k in ALL_KEYS for s in seeds
                 if (POLICIES[k][0], s) not in have]
    if jobs:
        print(f"베이스라인 {len(jobs)}회 실행 (병렬 {args.workers})", flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(args.workers) as ex:
            for n, (tm, r) in enumerate(ex.map(_run_one, jobs), 1):
                stores[tm].add(r)
                if n % 10 == 0 or n == len(jobs):
                    print(f"  {n}/{len(jobs)}  ({time.time() - t0:.0f} s)", flush=True)
    for tm in args.variants:
        path = os.path.join(cell_dir(args, tm), "baselines", "eval.csv")
        stores[tm].save_csv(path)
        evaluate_baselines_cached(_layout(tm), ALL_KEYS, args.lam / 3600.0, seeds, args.sim_time, path)


def _paired(rows_a, rows_b):
    """같은 (학습 시드, 테스트 시드)로 맞춘 차이 행렬 a − b (학습 시드 × 테스트 시드)"""
    a = {(r["train_seed"], r["seed"]): r["y"] for r in rows_a}
    b = {(r["train_seed"], r["seed"]): r["y"] for r in rows_b}
    common = set(a) & set(b)
    ts = sorted({t for t, _ in common})
    es = sorted(e for e in {e for _, e in common} if all((t, e) in common for t in ts))
    if not ts or not es:
        return None
    return np.array([[a[(t, e)] - b[(t, e)] for e in es] for t in ts], dtype=float)


def _summ(D, B):
    import analysis as A
    boot = A.hierarchical_bootstrap(D, B, "crossed", np.random.default_rng(0))
    sd_train = float(D.mean(axis=1).std(ddof=1)) if D.shape[0] > 1 else float("nan")
    return {"diff": float(D.mean()), "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)), "p": A.bootstrap_p(boot),
            "sd_train_means": sd_train, "n_train": D.shape[0], "n_test": D.shape[1]}


def step_analyze(args):
    import analysis as A
    with open(os.path.join(cell_dir(args, f"{TRAFFIC}/{MOTION}"), "select.json"), encoding="utf-8") as f:
        ref = json.load(f)["best_label"]
    out = []
    for metric in args.metrics:
        rows = [r for r in A.load_results(args.out, metric)
                if abs(r["lam"] - args.lam) < 0.5 and r["M"] == M and r["K"] == K]
        base_rl = [r for r in rows if r["setting"] == setting(f"{TRAFFIC}/{MOTION}") and r["policy"] == args.arm]
        print(f"\n[{metric}] 기준 {ref}, 기준 조건 RL 학습 시드 {len({r['train_seed'] for r in base_rl})}개")
        for tm in args.variants:
            cell = [r for r in rows if r["setting"] == setting(tm)]
            grp = {g: [r for r in cell if r["policy"] == f"{args.arm}_{g}"] for g in ("frozen", "retrain")}
            bl  = [r for r in cell if r["policy"] == ref]
            pairs = [(f"frozen − {ref}", grp["frozen"], None), (f"retrain − {ref}", grp["retrain"], None),
                     ("retrain − frozen", grp["retrain"], grp["frozen"]),
                     ("frozen − 기준 조건 RL", grp["frozen"], base_rl)]
            for name, a, b in pairs:
                if b is None:
                    m = A.paired_matrix(a, bl) if a and bl else None
                    D = m[0] if m else None
                else:
                    D = _paired(a, b) if a and b else None
                if D is None:
                    print(f"  {tm:14s} {name:24s} 결과 없음")
                    continue
                s = _summ(D, args.B)
                print(f"  {tm:14s} {name:24s} {s['diff']:+9.2f} [{s['ci_lo']:+9.2f}, {s['ci_hi']:+9.2f}]  "
                      f"p={s['p']:.4f}  SD(학습 시드 평균) {s['sd_train_means']:.2f}  "
                      f"(학습 {s['n_train']} × 테스트 {s['n_test']})")
                out.append({"metric": metric, "variant": tm, "comparison": name, "ref": ref, **s})
    A._save(out, os.path.join(args.out, f"ablation_{args.arm}_lam{args.lam:.0f}_M{M}_K{K}.csv"))


def main():
    p = argparse.ArgumentParser(description="라우팅/이동 조건 ablation: frozen vs retrain")
    p.add_argument("step", choices=["frozen", "retrain", "test", "analyze", "all"])
    p.add_argument("--out", default=os.path.join(REPO, "logs_exp"), help="결과 상위 폴더 (stage1과 같게)")
    p.add_argument("--arm", default=ARM)
    p.add_argument("--lam", type=float, default=LAM_HR, help="주문 도착률 (orders/hr)")
    p.add_argument("--variants", nargs="+", default=VARIANTS, help="변형 조건 traffic/motion")
    p.add_argument("--seeds", type=int, nargs="+", default=TRAIN_SEEDS, help="학습 시드 (기준 체크포인트와 같게)")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--sim-time", type=float, default=SIM_TIME)
    p.add_argument("--n-test", type=int, default=30, help="테스트 시드 수")
    p.add_argument("--jobs", type=int, default=5, help="재학습 동시 프로세스 수")
    p.add_argument("--workers", type=int, default=3, help="베이스라인 병렬 프로세스 수")
    p.add_argument("--B", type=int, default=10000, help="부트스트랩 반복 수")
    p.add_argument("--metrics", nargs="+", default=["avg_flow_time", "avg_flow_time_censored"])
    args = p.parse_args()
    for tm in args.variants:
        assert tm.split("/")[0] in ("free", "lanes") and tm.split("/")[1] in ("reserve", "reactive"), tm

    if args.step == "all":
        step_frozen(args)
        step_test(args)
        if step_retrain(args):
            step_analyze(args)
        return
    {"frozen": step_frozen, "retrain": step_retrain, "test": step_test, "analyze": step_analyze}[args.step](args)


if __name__ == "__main__":
    main()
