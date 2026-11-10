"""실험 실행기: design.py의 실험 블록을 단계별로 실행 (이미 끝난 (정책, 시드)·학습 run·전이 평가는 건너뜀)

사용법:
    python experiments/runner.py <계획> [--stage ...] [--arm ARM] [--jobs N] [--workers N] [--out 폴더] [--dry-run]
    python experiments/runner.py main --dry-run                 # 실행할 명령만 출력
    python experiments/runner.py main --stage cells             # 베이스라인 선택·테스트만 (학습 전에 다른 PC에서)
    python experiments/runner.py sens --filter _wext            # 민감도 중 w_ext arm만 (PC 분할)
    python experiments/runner.py status                         # 학습 run 진행 상황

계획 (design.py 모듈 설명)
    main, variants, sens, fleet, small, factorial, routing, status
단계 (--stage, 기본 all = 계획의 모든 단계를 아래 순서로)
    cells    : 셀마다 SELECT_KEYS(+BATCH_KEYS)를 검증 시드로 평가 → baselines/val.csv, select.json
               (best = 평균 흐름시간 최소, best_batching = 배치 규칙 최고), ALL_KEYS를 테스트 시드로 → baselines/eval.csv
    train    : (셀, arm)마다 μ(λ) 표(arm이 쓰면) → 학습 시드마다 run_rl.py 학습 + 그 셀 테스트
               (--select val --val-every 50 --n-val N_VAL_RL, 테스트 N_TEST 시드)
    transfer : 체크포인트를 다른 셀에서 run_rl.py --load로 평가 (재학습 없음, 베이스라인 없음)
    recon    : (routing) 기본 네트워크를 기준 셀에서 경로 구성 기록과 함께 재평가 (--tag recon), 원래 eval.csv와
               지표 동일 확인, route_gap.py로 NNH vs exact 요약 → {out}/tables/tab_routing_gap_*.csv
--arm: 기본 arm 대신 쓸 arm (main 학습과 fleet/small/factorial/routing 전이의 네트워크, 새 보상 실험용)
진행 기록: 단계가 끝날 때마다 {out}/progress.md에 명령, 결과 폴더, 실패 run 추가. 학습 로그는 셀의 logs/
"""
import argparse
import csv
import datetime
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constant import SIM_TIME                          # noqa: E402
import design as D                                     # noqa: E402
from design import Cell, Ckpt                          # noqa: E402

PLANS = ["main", "variants", "sens", "fleet", "small", "factorial", "routing", "status"]
STAGES = ["cells", "train", "transfer", "recon"]
TIMING = {"wall_s", "decision_ms_mean", "decision_ms_p95", "decision_full_ms_mean", "decision_full_ms_p95",
          "plan_time_s"}


# 기록 / 실행 도구

def progress(args, title: str, lines: List[str]):
    """{out}/progress.md에 단계 기록 추가 (--dry-run이면 기록 안 함)"""
    if args.dry_run:
        return
    os.makedirs(args.out, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(os.path.join(args.out, "progress.md"), "a", encoding="utf-8") as f:
        f.write(f"\n## {now} {title}\n\n- 명령: `{' '.join(sys.argv)}`\n- 코드: {REPO}\n")
        for l in lines:
            f.write(f"- {l}\n")


def _env():
    return {**os.environ, "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def run_procs(items: List[dict], jobs: int, dry: bool = False) -> List[dict]:
    """subprocess를 최대 jobs개 동시 실행. items: {"cmd", "log", "name"}. 출력: 실패한 item 목록"""
    if dry:
        for it in items:
            print("  [dry]", " ".join(it["cmd"]), ">>", it["log"])
        return []
    pending, running, fails = list(items), [], []
    while pending or running:
        while pending and len(running) < jobs:
            it = pending.pop(0)
            os.makedirs(os.path.dirname(it["log"]), exist_ok=True)
            print(f"시작 {it['name']}: 로그 {it['log']}", flush=True)
            it["proc"] = subprocess.Popen(it["cmd"], cwd=REPO, env=_env(), stderr=subprocess.STDOUT,
                                          stdout=open(it["log"], "a", encoding="utf-8"))
            running.append(it)
        time.sleep(5)
        for it in list(running):
            rc = it["proc"].poll()
            if rc is not None:
                running.remove(it)
                print(f"{'완료' if rc == 0 else f'실패 (rc={rc})'} {it['name']}", flush=True)
                if rc != 0:
                    it["rc"] = rc
                    fails.append(it)
    return fails


def _summary_lines(items: List[dict], fails: List[dict]) -> List[str]:
    return ([f"{it['name']} → {it['dir']}" for it in items if it not in fails]
            + [f"실패 {it['name']} (rc={it.get('rc')}), 로그 {it['log']}" for it in fails]) or ["모두 완료 상태"]


_LAYOUTS: Dict[tuple, object] = {}


def _run_one(job):
    """작업자 프로세스: (정책 키, 시드, sim_time, 셀) 1회 실행"""
    key, seed, sim_time, cell = job
    from evaluation import evaluate_policy
    k = (cell.M, cell.K, cell.traffic, cell.motion, cell.scenario)
    if k not in _LAYOUTS:
        _LAYOUTS[k] = cell.layout()
    st = evaluate_policy(_LAYOUTS[k], key, cell.lam / 3600.0, [seed], sim_time, verbose=False,
                         sim_kw=cell.cond.sim_kw)
    return cell, st.records[0]


def run_baselines(args, cells: List[Cell], keys: List[str], seeds: List[int], fname: str) -> Dict[Cell, object]:
    """셀 × 정책 × 시드를 한 작업자 풀에서 실행해 셀마다 baselines/{fname}에 누적 (있는 (라벨, 시드)는 건너뜀)

    출력: {셀: 요청분 MetricsStore} (--dry-run이면 빈 dict)
    """
    from metrics import MetricsStore
    from evaluation import policy_label, evaluate_baselines_cached
    stores, jobs = {}, []
    for c in cells:
        path = os.path.join(c.dir(args.out), "baselines", fname)
        st = MetricsStore.load_csv(path) if os.path.isfile(path) else MetricsStore()
        have = {(r.label, r.seed) for r in st.records if r.seed >= 0 and abs(r.sim_time - args.sim_time) < 1e-6}
        stores[c] = (path, st)
        jobs += [(k, s, args.sim_time, c) for k in keys for s in seeds if (policy_label(k), s) not in have]
    if args.dry_run:
        print(f"  [dry] 베이스라인 {len(jobs)}회 ({len(cells)}셀, {fname})")
        return {}
    if jobs:
        print(f"베이스라인 {len(jobs)}회 실행 ({len(cells)}셀, 병렬 {args.workers})", flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(args.workers) as ex:
            futs = [ex.submit(_run_one, j) for j in jobs]
            for n, f in enumerate(as_completed(futs), 1):
                c, r = f.result()
                stores[c][1].add(r)
                if n % 20 == 0 or n == len(jobs):
                    for path, st in stores.values():
                        st.save_csv(path)
                    print(f"  {n}/{len(jobs)}  ({time.time() - t0:.0f} s)", flush=True)
    return {c: evaluate_baselines_cached(c.layout(), keys, c.lam / 3600.0, seeds, args.sim_time,
                                         stores[c][0], sim_kw=c.cond.sim_kw) for c in cells}


# 단계

def step_cells(args, cells: List[Cell], select: bool = True, batching: bool = True, keys: Optional[List[str]] = None):
    """셀 베이스라인: (select) 검증 시드로 기준 선택 → select.json, 그리고 keys(기본 ALL_KEYS)를 테스트 시드로"""
    from evaluation import policy_label, test_seeds, val_seeds
    lines = []
    if select:
        seeds = val_seeds(D.N_VAL_SELECT)
        res = run_baselines(args, cells, D.SELECT_KEYS + (D.BATCH_KEYS if batching else []), seeds, "val.csv")
        for c, st in res.items():
            def rank(ks):
                out = []
                for k in ks:
                    rs = [r for r in st.records if r.label == policy_label(k)]
                    out.append({"key": k, "label": rs[0].label, "n": len(rs),
                                "avg_flow_time": sum(r.avg_flow_time for r in rs) / len(rs),
                                "completion_rate": sum(r.completion_rate for r in rs) / len(rs)})
                return sorted(out, key=lambda d: d["avg_flow_time"])
            ranking = rank(D.SELECT_KEYS)
            sel = {"best_key": ranking[0]["key"], "best_label": ranking[0]["label"], "val_seeds": seeds,
                   "sim_time": args.sim_time, "ranking": ranking, "cell": vars(c)}
            if batching:
                br = rank(D.BATCH_KEYS)
                sel.update(best_batching_key=br[0]["key"], best_batching=br[0]["label"], ranking_batching=br)
            path = os.path.join(c.dir(args.out), "select.json")
            if os.path.isfile(path):          # 다른 단계가 남긴 키(stages.py stage0의 bl6_star 등) 보존
                with open(path, encoding="utf-8") as f:
                    sel = {**json.load(f), **sel}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(sel, f, indent=1, ensure_ascii=False)
            msg = f"{c}: 최고 {sel['best_label']}" + (f", 배치 규칙 최고 {sel['best_batching']}" if batching else "")
            print(msg + "  (" + ", ".join(f"{d['label']} {d['avg_flow_time']:.1f}" for d in ranking) + ")")
            lines.append(msg + f" → {path}")
    keys = keys or D.ALL_KEYS
    res = run_baselines(args, cells, keys, test_seeds(args.n_test), "eval.csv")
    lines += [f"{c}: 테스트 {len(st.records)}행 → {c.dir(args.out)}/baselines/eval.csv" for c, st in res.items()]
    progress(args, f"cells {' '.join(keys)}", lines or ["실행 없음"])


def _run_rl(args, cell: Cell, arm: str) -> List[str]:
    return [sys.executable, "-u", os.path.join(REPO, "run_rl.py"), "--arm", arm, *cell.args(),
            "--episodes", str(args.episodes), "--sim-time", str(args.sim_time), "--n-eval", str(args.n_test),
            "--select", "val", "--val-every", str(args.val_every), "--n-val", str(D.N_VAL_RL),
            "--baselines", "--out", args.out]


def step_train(args, pairs: List[tuple], seeds: List[int], title: str) -> bool:
    """(셀, arm) 쌍마다 μ(λ) 표 추정 (최대 --mu-jobs개 동시) → 쌍 × 학습 시드를 한 작업 풀(--jobs)로 학습·테스트"""
    from rewards import ARMS
    if args.filter:
        pairs = [(c, a) for c, a in pairs if args.filter in a]
    mu_items = [{"cmd": _run_rl(args, c, a) + ["--mu-only"], "name": f"μ {c} {a}", "pair": (c, a),
                 "dir": c.dir(args.out), "log": os.path.join(c.dir(args.out), "logs", f"mu_{a}.log")}
                for c, a in pairs if ARMS[a].uses_mu]
    mu_fails = run_procs(mu_items, args.mu_jobs, args.dry_run)
    if mu_items:
        progress(args, f"μ(λ) {len(mu_items)}개", _summary_lines(mu_items, mu_fails))
    failed = {it["pair"] for it in mu_fails}
    items = []
    for c, arm in pairs:
        if (c, arm) in failed:
            continue
        for s in seeds:
            ck = Ckpt(c, arm, s)
            if all(os.path.isfile(os.path.join(ck.dir(args.out), f)) for f in ("agent.pt", "eval.csv")):
                print(f"{c} {arm} s{s}: 이미 학습·평가 완료, 건너뜀")
                continue
            items.append({"cmd": _run_rl(args, c, arm) + ["--seed", str(s)], "name": f"{c} {arm} s{s}",
                          "log": os.path.join(c.dir(args.out), "logs", f"{arm}_s{s}.log"), "dir": ck.dir(args.out)})
    fails = run_procs(items, args.jobs, args.dry_run)
    progress(args, title, _summary_lines(items, fails))
    return not (fails or mu_fails)


def step_transfer(args, ckpts: List[Ckpt], targets: List[Cell], tag: Optional[str] = None,
                  recon_dir: Optional[str] = None) -> bool:
    """체크포인트 × 대상 셀 전이 평가 (같은 셀·조건 재평가는 tag 필요)"""
    items = []
    for t in targets:
        for ck in ckpts:
            src, name = ck.path(args.out), ck.run_name(t, tag)
            d = os.path.join(t.dir(args.out), name)
            if os.path.isfile(os.path.join(d, "eval.csv")):
                print(f"{t} {name}: 이미 평가됨, 건너뜀")
                continue
            if not os.path.isfile(src) and not args.dry_run:
                print(f"체크포인트 없음: {src} (학습이 끝난 뒤 다시 실행)")
                continue
            cmd = [sys.executable, "-u", os.path.join(REPO, "run_rl.py"), "--load", src, *t.args(),
                   "--sim-time", str(args.sim_time), "--n-eval", str(args.n_test), "--baselines", "--out", args.out]
            if tag:
                cmd += ["--tag", tag]
            if recon_dir:
                cmd += ["--log-reconstructions", os.path.join(recon_dir, f"recon_s{ck.seed}.jsonl.gz")]
            items.append({"cmd": cmd, "name": f"{t} ← {name}", "dir": d,
                          "log": os.path.join(t.dir(args.out), "logs", f"{name}.log")})
    fails = run_procs(items, args.jobs, args.dry_run)
    progress(args, f"transfer {len(ckpts)} ckpt × {len(targets)} 셀" + (f" tag={tag}" if tag else ""),
             _summary_lines(items, fails))
    return not fails


def _same_eval(args, ckpts: List[Ckpt], cell: Cell, tag: str) -> List[str]:
    """tag 재평가 eval.csv가 원래 run eval.csv와 지표가 같은지 (실행 시간 열 제외)"""
    out = []
    for ck in ckpts:
        a = os.path.join(cell.dir(args.out), ck.run_name(cell), "eval.csv")
        b = os.path.join(cell.dir(args.out), ck.run_name(cell, tag), "eval.csv")
        if not (os.path.isfile(a) and os.path.isfile(b)):
            out.append(f"s{ck.seed}: 비교 불가 (파일 없음)")
            continue
        ra = {r["seed"]: r for r in csv.DictReader(open(a, newline=""))}
        bad = sum(1 for r in csv.DictReader(open(b, newline=""))
                  if (o := ra.get(r["seed"])) is None or any(r[k] != o[k] for k in r if k in o and k not in TIMING))
        out.append(f"s{ck.seed}: 원래 eval.csv와 " + ("동일" if bad == 0 else f"{bad}개 시드 불일치"))
    return out


def step_recon(args, nets: List[Ckpt]):
    """기준 셀 재평가 + 경로 구성 기록 → 원래 결과와 동일 확인 → route_gap.py"""
    cell = D.BASE_CELL
    rdir = os.path.join(cell.dir(args.out), "recon")
    ok = step_transfer(args, nets, [cell], tag="recon", recon_dir=rdir)
    lines = _same_eval(args, nets, cell, "recon") if ok and not args.dry_run else []
    for l in lines:
        print(" ", l)
    logs = [p for p in (os.path.join(rdir, f"recon_s{ck.seed}.jsonl.gz") for ck in nets) if os.path.isfile(p)]
    if logs and not args.dry_run:
        cmd = [sys.executable, os.path.join(REPO, "experiments", "route_gap.py"), *logs,
               "--M", str(cell.M), "--K", str(cell.K), "--out", os.path.join(args.out, "tables", "tab_routing_gap")]
        if args.route_gap_max_calls:
            cmd += ["--max-calls", str(args.route_gap_max_calls)]
        rc = subprocess.run(cmd, cwd=REPO, env=_env()).returncode
        lines.append(f"route_gap.py rc={rc} → {args.out}/tables/tab_routing_gap_summary.csv")
    progress(args, "recon·route_gap", lines or ["실행 없음"])


def status(args):
    """모든 계획의 학습 run 진행: 완료(eval.csv) / 학습 중 마지막 [Ep N] / 시작 전"""
    pat = re.compile(r"\[Ep\s+(\d+)\]")
    pairs = ([(c, args.arm, D.TRAIN_SEEDS) for c in D.MAIN_CELLS]
             + [(D.BASE_CELL, a, D.SUB_SEEDS) for a in D.VARIANT_ARMS]
             + [(Cell(lam, 14, 3), a, D.SUB_SEEDS) for lam, a in D.SENS_DESIGN]
             + [(c, args.arm, D.SUB_SEEDS) for c in D.FACTORIAL_TRAIN.values()])
    n_done = n_all = 0
    for c, arm, seeds in pairs:
        for s in seeds:
            ck, n_all = Ckpt(c, arm, s), n_all + 1
            log = os.path.join(c.dir(args.out), "logs", f"{arm}_s{s}.log")
            if os.path.isfile(os.path.join(ck.dir(args.out), "eval.csv")):
                st, n_done = "완료", n_done + 1
            elif os.path.isfile(log):
                eps = pat.findall(open(log, encoding="utf-8", errors="replace").read())
                age = (time.time() - os.path.getmtime(log)) / 3600
                st = f"에피소드 {eps[-1] if eps else 0}/{args.episodes} (로그 갱신 {age:.1f} h 전)"
            else:
                st = "시작 전"
            print(f"  {str(c):18s} {arm:32s} s{s}: {st}")
    print(f"완료 {n_done}/{n_all}")


# 계획

def run_plan(args, plan: str):
    want = lambda s: args.stage in ("all", s)
    arm = args.arm
    if plan == "status":
        return status(args)
    if plan == "main":
        if want("cells"):
            step_cells(args, D.MAIN_CELLS)
        if want("train"):
            step_train(args, [(c, arm) for c in D.MAIN_CELLS], D.TRAIN_SEEDS, f"train main {arm}")
    elif plan == "variants":
        if want("train"):
            step_train(args, [(D.BASE_CELL, a) for a in D.VARIANT_ARMS], D.SUB_SEEDS, "train variants")
    elif plan == "sens":
        if want("train"):
            step_train(args, [(Cell(lam, 14, 3), a) for lam, a in D.SENS_DESIGN], D.SUB_SEEDS, "train sens")
    elif plan in ("fleet", "small"):
        cells = D.FLEET_CELLS if plan == "fleet" else D.SMALL_CELLS
        if want("cells"):
            step_cells(args, cells, batching=False)
        if want("transfer"):
            step_transfer(args, D.base_nets(arm), cells)
    elif plan == "factorial":
        if want("train"):
            step_train(args, [(c, arm) for c in D.FACTORIAL_TRAIN.values()], D.SUB_SEEDS, f"train factorial {arm}")
        if want("cells"):
            step_cells(args, D.FACTORIAL_CELLS, batching=False)
        if want("transfer"):
            for K in D.FACTORIAL_TRAIN:
                step_transfer(args, D.factorial_nets(K, arm), [c for c in D.FACTORIAL_CELLS if c.K == K])
            step_transfer(args, D.factorial_ref_nets(arm), [D.BASE_CELL])
    elif plan == "routing":
        if want("cells"):
            step_cells(args, D.ROUTE_CELLS, select=False, keys=["BL6"])
        if want("transfer"):
            step_transfer(args, D.base_nets(arm), D.ROUTE_CELLS)
        if want("recon"):
            step_recon(args, D.base_nets(arm))


def main():
    from rewards import ARMS
    p = argparse.ArgumentParser(description="실험 실행기", formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("plan", choices=PLANS)
    p.add_argument("--stage", choices=["all"] + STAGES, default="all", help="계획 안에서 실행할 단계")
    p.add_argument("--arm", choices=list(ARMS), default=D.BASE_ARM, help="기본 arm 대신 쓸 arm (새 보상 실험)")
    p.add_argument("--filter", default=None, help="train 단계에서 arm 이름에 이 문자열이 있는 것만 (PC 분할)")
    p.add_argument("--out", default=os.path.join(REPO, "logs_exp"), help="결과 상위 폴더")
    p.add_argument("--workers", type=int, default=4, help="베이스라인 병렬 프로세스 수")
    p.add_argument("--jobs", type=int, default=6, help="동시 run_rl.py 프로세스 수 (학습/전이)")
    p.add_argument("--mu-jobs", type=int, default=2, help="μ(λ) 추정 동시 프로세스 수 (각 LOCAL_MU_WORKERS개 작업자)")
    p.add_argument("--sim-time", type=float, default=SIM_TIME)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=D.N_TEST)
    p.add_argument("--val-every", type=int, default=50)
    p.add_argument("--route-gap-max-calls", type=int, default=0, help="route_gap.py 파일당 표본 수 (0 = 전부)")
    p.add_argument("--dry-run", action="store_true", help="실행할 명령만 출력")
    args = p.parse_args()
    args.out = os.path.abspath(args.out)
    run_plan(args, args.plan)


if __name__ == "__main__":
    main()
