"""리비전 실험 실행기: stage1.py의 고정값(λ, M, K, traffic, motion)을 셀 단위로 일반화한 단계별 실행

셀 = (λ, M, K, route[, traffic, motion]) → {out}/grid_{traffic}_{motion}[_route-fixeddock]/lam{λ}_M{M}_K{K}/
모든 단계는 이미 끝난 (정책, 시드) / 학습 run / 전이 평가를 건너뜀. 로그는 셀의 logs/
진행 기록: 단계가 끝날 때마다 {out}/revision_progress.md에 명령, 결과 폴더, 실패 run 추가

단계 (묶음 안에서 쓰는 기본 동작)
    select    : 셀마다 BL4~BL7을 검증 시드 10개로 평가 → baselines/val.csv, select.json (최고 = 평균 흐름시간 최소)
                --batching이면 BL1, BL1F, BL2도 검증 시드로 평가해 select.json의 best_batching에 기록
    baselines : 셀마다 BL1, BL1F, BL2~BL7 (또는 지정 정책)을 테스트 시드 30개로 → baselines/eval.csv
    mu        : 학습 셀 × arm마다 μ(λ) 표 (run_rl.py --mu-only, arm이 mu를 안 쓰면 바로 끝남)
    train     : 셀 × arm × 학습 시드를 run_rl.py로 학습 후 그 셀 테스트 (--select val --val-every 50 --n-val 5, 30 시드)
    transfer  : 체크포인트를 다른 (λ, M, K, route)에서 run_rl.py --load로 평가 (30 시드, 베이스라인 없음)
                run 이름 = {arm}_s{시드}[_tr{학습 M}][_trlam{학습 λ}][_route-fixeddock][_{tag}]

묶음 (python experiments/revision.py <묶음> [--workers N] [--jobs N] [--out 폴더])
    E1 : λ = 320, M = 14, K ∈ {2, 4}, local_r8_ddqn_v2, 시드 1~3 학습 (mu → train)
    E2 : 기본 셀, local_r8_ddqn_v2_gamma, 시드 1~3 학습
    E3 : 기본 셀, legacy_r8_ddqn_v2, 시드 1~3 학습
    T1 : 기본 셀 select --batching (베이스라인 테스트 결과는 본실험 것 재사용)
    T2 : M ∈ {8, 10, 12, 16, 18, 20}, λ = 240·M/14 (137, 171, 206, 274, 309, 343), K = 3:
         select + baselines + 기본 네트워크 s1~5 전이
    T3 : λ = 240, M ∈ {6, 10}, K = 3: select + baselines + 기본 네트워크 전이
    T4 : 기본 셀 --route fixed_dock: BL6 + 기본 네트워크 전이 (setting grid_free_reserve_route-fixeddock)
    T5 : 기본 셀 기본 네트워크 재평가 + 경로 구성 기록 (--tag recon, 기록 → {셀}/recon/recon_s{시드}.jsonl.gz)
         이어서 원래 eval.csv와 지표 동일 확인, route_gap.py로 tab_routing_gap 요약
    T6 : λ ∈ {240, 320} × M ∈ {10, 18} × K ∈ {2, 4}: select + baselines + E1 네트워크(해당 K, s1~3) 전이
         기준 행: (240, 14, 3)을 λ = 320 본실험 네트워크 s1~5로 전이 ((320, 14, 3)은 본실험 결과 재사용)
    T6-cells / T2-cells / T3-cells : 해당 묶음의 select + baselines만 (다른 PC에서 셀 베이스라인 선행용)
    T6-transfer / T2-transfer / T3-transfer : 전이 평가만

    단계 이름(select, baselines, mu, train, transfer)을 직접 쓸 때는 --cells "λ,M,K[,route]" ...와
    --arm, --seeds, --ckpt-cell 등을 지정 (--help)

기본 셀 = (λ = 240, M = 14, K = 3, free/reserve), 기본 네트워크 = lam240_M14_K3/local_r8_ddqn_v2_s{1..5}/agent.pt
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from constant import SIM_TIME  # noqa: E402

BASE_ARM     = "local_r8_ddqn_v2"
SELECT_KEYS  = ["BL4", "BL5", "BL6", "BL7"]          # BL7 = Greedy_R
BATCH_KEYS   = ["BL1", "BL1F", "BL2"]
ALL_KEYS     = ["BL1", "BL1F", "BL2", "BL3", "BL4", "BL5", "BL6", "BL7"]
BASE_SEEDS   = [1, 2, 3, 4, 5]
E_SEEDS      = [1, 2, 3]
T2_M         = [8, 10, 12, 16, 18, 20]


@dataclass(frozen=True)
class Cell:
    """평가/학습 셀 (λ orders/hr, M, K, 경로 규칙, 이동 가정)"""
    lam:     float
    M:       int
    K:       int
    route:   str = "nnh"
    traffic: str = "free"
    motion:  str = "reserve"

    @property
    def cond(self):
        from evaluation import Condition
        return Condition(traffic=self.traffic, motion=self.motion, route=self.route)

    def dir(self, out: str) -> str:
        return os.path.join(out, self.cond.setting(), f"lam{self.lam:.0f}_M{self.M}_K{self.K}")

    def args(self) -> List[str]:
        """run_rl.py 셀 인자"""
        a = ["--lam", str(self.lam), "--robots", str(self.M), "--ws", str(self.K),
             "--traffic", self.traffic, "--motion", self.motion]
        return a + (["--route", self.route] if self.route != "nnh" else [])

    def __str__(self):
        return f"λ={self.lam:.0f} M={self.M} K={self.K}" + ("" if self.route == "nnh" else f" route={self.route}")


BASE_CELL = Cell(240, 14, 3)


@dataclass
class Ckpt:
    """전이 평가할 체크포인트 (학습 셀, arm, 시드)"""
    cell: Cell
    arm:  str
    seed: int

    def path(self, out: str) -> str:
        return os.path.join(self.cell.dir(out), f"{self.arm}_s{self.seed}", "agent.pt")

    def run_name(self, target: Cell, tag: Optional[str] = None) -> str:
        """run_rl.py --load가 target 셀에서 만드는 run 이름 (같은 규칙)"""
        name = f"{self.arm}_s{self.seed}"
        if self.cell.M != target.M:
            name += f"_tr{self.cell.M}"
        if abs(self.cell.lam - target.lam) > 0.5:
            name += f"_trlam{self.cell.lam:.0f}"
        return name + target.cond.tag + (f"_{tag}" if tag else "")


# 기록 / 실행 도구

def progress(args, title: str, lines: List[str]):
    """{out}/revision_progress.md에 단계 기록 추가 (--dry-run이면 기록 안 함)"""
    if args.dry_run:
        return
    path = os.path.join(args.out, "revision_progress.md")
    os.makedirs(args.out, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(path, "a", encoding="utf-8") as f:
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


_LAYOUTS: Dict[tuple, object] = {}


def _run_one(job):
    """작업자 프로세스: (정책 키, 시드, sim_time, 셀) 1회 실행"""
    key, seed, sim_time, cell = job
    from evaluation import evaluate_policy
    k = (cell.M, cell.K, cell.traffic, cell.motion)
    if k not in _LAYOUTS:
        _LAYOUTS[k] = cell.cond.layout(cell.M, cell.K)
    st = evaluate_policy(_LAYOUTS[k], key, cell.lam / 3600.0, [seed], sim_time, verbose=False,
                         sim_kw=cell.cond.sim_kw)
    return cell, st.records[0]


def run_baselines(args, cells: List[Cell], keys: List[str], seeds: List[int], fname: str) -> Dict[Cell, object]:
    """셀 × 정책 × 시드를 한 작업자 풀에서 실행해 셀마다 baselines/{fname}에 누적 (있는 (라벨, 시드)는 건너뜀)

    출력: {셀: 요청분 MetricsStore}
    """
    from metrics import MetricsStore
    from evaluation import POLICIES, policy_key, evaluate_baselines_cached
    stores, jobs = {}, []
    for c in cells:
        path = os.path.join(c.dir(args.out), "baselines", fname)
        st = MetricsStore.load_csv(path) if os.path.isfile(path) else MetricsStore()
        have = {(r.label, r.seed) for r in st.records if r.seed >= 0 and abs(r.sim_time - args.sim_time) < 1e-6}
        stores[c] = (path, st)
        jobs += [(k, s, args.sim_time, c) for k in keys for s in seeds if (POLICIES[policy_key(k)][0], s) not in have]
    if jobs and not args.dry_run:
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
    elif jobs:
        print(f"  [dry] 베이스라인 {len(jobs)}회")
        return {}
    out = {}
    for c in cells:
        path = stores[c][0]
        out[c] = evaluate_baselines_cached(c.cond.layout(c.M, c.K), keys, c.lam / 3600.0, seeds, args.sim_time,
                                           path, sim_kw=c.cond.sim_kw)
    return out


# 단계

def step_select(args, cells: List[Cell], batching: bool = False):
    from evaluation import POLICIES, policy_key, val_seeds
    seeds = val_seeds(args.n_val)
    keys  = SELECT_KEYS + (BATCH_KEYS if batching else [])
    res   = run_baselines(args, cells, keys, seeds, "val.csv")
    lines = []
    for c in cells:
        if c not in res:
            continue

        def rank(ks):
            out = []
            for k in ks:
                rs = [r for r in res[c].records if r.label == POLICIES[policy_key(k)][0]]
                out.append({"key": k, "label": rs[0].label, "n": len(rs),
                            "avg_flow_time": sum(r.avg_flow_time for r in rs) / len(rs),
                            "completion_rate": sum(r.completion_rate for r in rs) / len(rs)})
            return sorted(out, key=lambda d: d["avg_flow_time"])
        ranking = rank(SELECT_KEYS)
        sel = {"best_key": ranking[0]["key"], "best_label": ranking[0]["label"], "val_seeds": seeds,
               "sim_time": args.sim_time, "ranking": ranking, "cell": vars(c)}
        path = os.path.join(c.dir(args.out), "select.json")
        if os.path.isfile(path):          # 이전 기록의 다른 키(best_batching 등) 보존
            with open(path, encoding="utf-8") as f:
                sel = {**json.load(f), **sel}
        if batching:
            br = rank(BATCH_KEYS)
            sel.update(best_batching_key=br[0]["key"], best_batching=br[0]["label"], ranking_batching=br)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sel, f, indent=1, ensure_ascii=False)
        msg = f"{c}: 최고 {sel['best_label']}" + (f", 배치 규칙 최고 {sel['best_batching']}" if batching else "")
        print(msg + "  (" + ", ".join(f"{d['label']} {d['avg_flow_time']:.1f}" for d in ranking) + ")")
        lines.append(msg + f" → {path}")
    progress(args, "select", lines or ["실행 없음"])


def step_baselines(args, cells: List[Cell], keys: List[str]):
    from evaluation import test_seeds
    res = run_baselines(args, cells, keys, test_seeds(args.n_test), "eval.csv")
    progress(args, "baselines " + " ".join(keys),
             [f"{c}: {len(res.get(c).records) if c in res else 0}행 → {c.dir(args.out)}/baselines/eval.csv"
              for c in cells])


def _train_base(args, cell: Cell, arm: str) -> List[str]:
    return [sys.executable, "-u", os.path.join(REPO, "run_rl.py"), "--arm", arm, *cell.args(),
            "--episodes", str(args.episodes), "--sim-time", str(args.sim_time), "--n-eval", str(args.n_test),
            "--select", "val", "--val-every", str(args.val_every), "--n-val", str(args.rl_n_val),
            "--baselines", "--out", args.out]


def step_mu(args, cells: List[Cell], arm: str):
    fails, lines = [], []
    for c in cells:
        log = os.path.join(c.dir(args.out), "logs", f"mu_{arm}.log")
        f = run_procs([{"cmd": _train_base(args, c, arm) + ["--mu-only"], "log": log, "name": f"μ {c} {arm}"}],
                      1, args.dry_run)
        fails += f
        lines.append(f"{c} {arm}: " + ("실패, 로그 " + log if f else f"완료 → {c.dir(args.out)}/local_mu.json"))
    progress(args, f"mu {arm}", lines)
    return not fails


def step_train(args, cells: List[Cell], arm: str, seeds: List[int]):
    items = []
    for c in cells:
        for s in seeds:
            d = os.path.join(c.dir(args.out), f"{arm}_s{s}")
            if all(os.path.isfile(os.path.join(d, f)) for f in ("agent.pt", "eval.csv")):
                print(f"{c} {arm} s{s}: 이미 학습·평가 완료, 건너뜀")
                continue
            items.append({"cmd": _train_base(args, c, arm) + ["--seed", str(s)], "name": f"{c} {arm} s{s}",
                          "log": os.path.join(c.dir(args.out), "logs", f"{arm}_s{s}.log"), "dir": d})
    fails = run_procs(items, args.jobs, args.dry_run)
    progress(args, f"train {arm} 시드 {seeds}",
             [f"{it['name']} → {it['dir']}" for it in items if it not in fails]
             + [f"실패 {it['name']} (rc={it.get('rc')}), 로그 {it['log']}" for it in fails] or ["모두 완료 상태"])
    return not fails


def step_transfer(args, ckpts: List[Ckpt], targets: List[Cell], tag: Optional[str] = None,
                  recon_dir: Optional[str] = None, n_test: Optional[int] = None):
    """체크포인트 × 대상 셀 전이 평가 (같은 셀·조건 재평가는 tag 필요)"""
    items = []
    for t in targets:
        for ck in ckpts:
            src = ck.path(args.out)
            name = ck.run_name(t, tag)
            d = os.path.join(t.dir(args.out), name)
            if os.path.isfile(os.path.join(d, "eval.csv")):
                print(f"{t} {name}: 이미 평가됨, 건너뜀")
                continue
            if not os.path.isfile(src) and not args.dry_run:
                print(f"체크포인트 없음: {src} (학습이 끝난 뒤 다시 실행)")
                continue
            cmd = [sys.executable, "-u", os.path.join(REPO, "run_rl.py"), "--load", src, *t.args(),
                   "--sim-time", str(args.sim_time), "--n-eval", str(n_test or args.n_test), "--baselines",
                   "--out", args.out]
            if tag:
                cmd += ["--tag", tag]
            if recon_dir:
                cmd += ["--log-reconstructions", os.path.join(recon_dir, f"recon_s{ck.seed}.jsonl.gz")]
            items.append({"cmd": cmd, "name": f"{t} ← {name}", "dir": d,
                          "log": os.path.join(t.dir(args.out), "logs", f"{name}.log")})
    fails = run_procs(items, args.jobs, args.dry_run)
    progress(args, f"transfer {len(ckpts)} ckpt × {len(targets)} 셀" + (f" tag={tag}" if tag else ""),
             [f"{it['name']} → {it['dir']}" for it in items if it not in fails]
             + [f"실패 {it['name']} (rc={it.get('rc')}), 로그 {it['log']}" for it in fails] or ["모두 완료 상태"])
    return not fails


def check_same_eval(args, ckpts: List[Ckpt], cell: Cell, tag: str) -> List[str]:
    """tag 재평가 eval.csv가 원래 run eval.csv와 지표가 같은지 (실행 시간 열 제외)"""
    import csv
    timing = {"wall_s", "decision_ms_mean", "decision_ms_p95", "decision_full_ms_mean", "decision_full_ms_p95",
              "plan_time_s"}
    out = []
    for ck in ckpts:
        a = os.path.join(cell.dir(args.out), ck.run_name(cell), "eval.csv")
        b = os.path.join(cell.dir(args.out), ck.run_name(cell, tag), "eval.csv")
        if not (os.path.isfile(a) and os.path.isfile(b)):
            out.append(f"s{ck.seed}: 비교 불가 (파일 없음)")
            continue
        ra = {r["seed"]: r for r in csv.DictReader(open(a, newline=""))}
        bad = 0
        for r in csv.DictReader(open(b, newline="")):
            o = ra.get(r["seed"])
            if o is None or any(r[k] != o[k] for k in r if k in o and k not in timing):
                bad += 1
        out.append(f"s{ck.seed}: 원래 eval.csv와 " + ("동일" if bad == 0 else f"{bad}개 시드 불일치"))
    return out


# 묶음

def bundle(args, name: str):
    base_nets = [Ckpt(BASE_CELL, BASE_ARM, s) for s in BASE_SEEDS]
    t2 = [Cell(round(240 * M / 14), M, 3) for M in T2_M]
    t3 = [Cell(240, M, 3) for M in (6, 10)]
    t6 = [Cell(l, M, K) for l in (240, 320) for M in (10, 18) for K in (2, 4)]
    if name == "E1":
        cells = [Cell(320, 14, K) for K in (2, 4)]
        if step_mu(args, cells, BASE_ARM):
            step_train(args, cells, BASE_ARM, E_SEEDS)
    elif name in ("E2", "E3"):
        arm = {"E2": "local_r8_ddqn_v2_gamma", "E3": "legacy_r8_ddqn_v2"}[name]
        if step_mu(args, [BASE_CELL], arm):
            step_train(args, [BASE_CELL], arm, E_SEEDS)
    elif name == "T1":
        step_select(args, [BASE_CELL], batching=True)
    elif name in ("T2", "T3", "T2-cells", "T3-cells", "T2-transfer", "T3-transfer"):
        cells = t2 if name.startswith("T2") else t3
        if not name.endswith("transfer"):
            step_select(args, cells)
            step_baselines(args, cells, ALL_KEYS)
        if not name.endswith("cells"):
            step_transfer(args, base_nets, cells)
    elif name == "T4":
        c = Cell(240, 14, 3, route="fixed_dock")
        step_baselines(args, [c], ["BL6"])
        step_transfer(args, base_nets, [c])
    elif name == "T5":
        rdir = os.path.join(BASE_CELL.dir(args.out), "recon")
        ok = step_transfer(args, base_nets, [BASE_CELL], tag="recon", recon_dir=rdir, n_test=args.recon_n_test)
        lines = check_same_eval(args, base_nets, BASE_CELL, "recon") if ok and not args.dry_run else []
        for l in lines:
            print(" ", l)
        logs = [os.path.join(rdir, f"recon_s{s}.jsonl.gz") for s in BASE_SEEDS]
        logs = [p for p in logs if os.path.isfile(p)]
        if logs and not args.dry_run:
            cmd = [sys.executable, os.path.join(REPO, "experiments", "route_gap.py"), *logs, "--M", "14", "--K", "3",
                   "--out", os.path.join(args.out, "revision_tables", "tab_routing_gap")]
            if args.route_gap_max_calls:
                cmd += ["--max-calls", str(args.route_gap_max_calls)]
            rc = subprocess.run(cmd, cwd=REPO, env=_env()).returncode
            lines.append(f"route_gap.py rc={rc} → {args.out}/revision_tables/tab_routing_gap_summary.csv")
        progress(args, "T5 기록·route_gap", lines or ["실행 없음"])
    elif name in ("T6", "T6-cells", "T6-transfer"):
        if name != "T6-transfer":
            step_select(args, t6)
            step_baselines(args, t6, ALL_KEYS)
        if name != "T6-cells":
            for K in (2, 4):
                nets = [Ckpt(Cell(320, 14, K), BASE_ARM, s) for s in E_SEEDS]
                step_transfer(args, nets, [c for c in t6 if c.K == K])
            # 기준 행 (240, 14, 3) ← λ = 320 본실험 네트워크
            step_transfer(args, [Ckpt(Cell(320, 14, 3), BASE_ARM, s) for s in BASE_SEEDS], [BASE_CELL])
    else:
        raise SystemExit(f"알 수 없는 묶음 {name}")


def _parse_cell(s: str) -> Cell:
    p = s.split(",")
    return Cell(float(p[0]), int(p[1]), int(p[2]), *(p[3:4] or []))


def main():
    p = argparse.ArgumentParser(description="리비전 실험 실행기", formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("what", help="묶음 (E1~E3, T1~T6, T2-cells, ...) 또는 단계 (select, baselines, mu, train, transfer)")
    p.add_argument("--out", default=os.path.join(REPO, "logs_exp"), help="결과 상위 폴더 (본실험 logs_exp)")
    p.add_argument("--workers", type=int, default=4, help="베이스라인 병렬 프로세스 수")
    p.add_argument("--jobs", type=int, default=6, help="동시 run_rl.py 프로세스 수 (학습/전이)")
    p.add_argument("--sim-time", type=float, default=SIM_TIME)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=30)
    p.add_argument("--n-val", type=int, default=10, help="베이스라인 선택용 검증 시드 수")
    p.add_argument("--rl-n-val", type=int, default=5, help="RL 체크포인트 선택용 검증 시드 수")
    p.add_argument("--val-every", type=int, default=50)
    p.add_argument("--recon-n-test", type=int, default=30, help="T5 기록 재평가 테스트 시드 수 (많으면 10)")
    p.add_argument("--route-gap-max-calls", type=int, default=0, help="route_gap.py 파일당 표본 수 (0 = 전부)")
    p.add_argument("--dry-run", action="store_true", help="실행할 명령만 출력")
    # 단계 직접 실행용
    p.add_argument("--cells", nargs="*", default=[], help='셀 "λ,M,K[,route]" 목록')
    p.add_argument("--arm", default=BASE_ARM)
    p.add_argument("--seeds", type=int, nargs="+", default=E_SEEDS)
    p.add_argument("--keys", nargs="+", default=ALL_KEYS, help="baselines 단계 정책 키")
    p.add_argument("--batching", action="store_true", help="select에서 배치 규칙 최고도 기록")
    p.add_argument("--ckpt-cell", default="240,14,3", help='transfer 체크포인트 학습 셀 "λ,M,K"')
    p.add_argument("--tag", default=None, help="transfer run 이름 꼬리표")
    args = p.parse_args()
    args.out = os.path.abspath(args.out)

    cells = [_parse_cell(c) for c in args.cells]
    if args.what == "select":
        step_select(args, cells, args.batching)
    elif args.what == "baselines":
        step_baselines(args, cells, args.keys)
    elif args.what == "mu":
        step_mu(args, cells, args.arm)
    elif args.what == "train":
        if step_mu(args, cells, args.arm):
            step_train(args, cells, args.arm, args.seeds)
    elif args.what == "transfer":
        src = _parse_cell(args.ckpt_cell)
        step_transfer(args, [Ckpt(src, args.arm, s) for s in args.seeds], cells, tag=args.tag)
    else:
        bundle(args, args.what)


if __name__ == "__main__":
    main()
