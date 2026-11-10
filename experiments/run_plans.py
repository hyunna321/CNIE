"""실험 설계와 실행기: experiments/configs/design.yaml의 실험 블록을 단계별로 실행
(이미 끝난 (정책, 시드)·학습 run·전이 평가는 건너뜀)

사용법:
    python -m experiments.run_plans <계획> [--stage ...] [--arm ARM] [--jobs N] [--workers N] [--out 폴더] [--dry-run]
    python -m experiments.run_plans main --dry-run                 # 실행할 명령만 출력
    python -m experiments.run_plans main --stage cells             # 베이스라인 선택·테스트만 (학습 전에 다른 PC에서)
    python -m experiments.run_plans sens --filter _wext            # 민감도 중 w_ext arm만 (PC 분할)
    python -m experiments.run_plans status                         # 학습 run 진행 상황

계획 (아래 실험 블록)
    main, variants, sens, fleet, small, factorial, routing, status
단계 (--stage, 기본 all = 계획의 모든 단계를 아래 순서로)
    cells    : 셀마다 SELECT_KEYS(+BATCH_KEYS)를 검증 시드로 평가 → baselines/val.csv, select.json
               (best = 평균 흐름시간 최소, best_batching = 배치 규칙 최고), ALL_KEYS를 테스트 시드로 → baselines/eval.csv
    train    : (셀, arm)마다 μ(λ) 표(arm이 쓰면) → 학습 시드마다 train.py 학습 + run_compare.py 그 셀 테스트
               (--select val --val-every 50 --n-val N_VAL_RL, 테스트 N_TEST 시드)
    transfer : 체크포인트를 다른 셀에서 run_compare.py --load로 평가 (재학습 없음, 베이스라인 없음)
--arm: 기본 arm 대신 쓸 arm (main 학습과 fleet/small/factorial/routing 전이의 네트워크, 새 보상 실험용)
진행 기록: 단계가 끝날 때마다 {out}/progress.md에 명령, 결과 폴더, 실패 run 추가. 학습 로그는 셀의 logs/

셀 = (λ orders/hr, M, K, 경로 규칙[, 이동 가정]) → {out}/{setting}/lam{λ}_M{M}_K{K}/
모든 비교는 같은 테스트 인스턴스(evaluation.test_seeds) 위의 대응 비교, RL은 학습 시드 단위로 집계

실험 블록 (계획 이름)
  main      : λ ∈ MAIN_LAMS, M = 14, K = 3. 베이스라인 선택(검증 시드) + 전체 베이스라인 테스트 + 기본 arm 학습 (시드 1~5)
  variants  : 기준 셀에서 학습 변형 arm (VARIANT_ARMS) 학습 (시드 1~3)
  sens      : 보상 계수 민감도 (SENS_DESIGN) 학습 (시드 1~3, 중심점 = main의 기본 arm 같은 시드)
  fleet     : 재학습 없는 일반화 — M ∈ FLEET_M, λ = 240·M/14 셀에 기본 네트워크 전이
  small     : 재학습 없는 일반화 — λ = 240, M ∈ SMALL_M 셀에 기본 네트워크 전이
  factorial : 2³ 요인 실험 — (λ, M, K) ∈ FACTORIAL_CELLS. K마다 (320, 14, K)에서 학습한 네트워크를 전이
              (λ·M = 네트워크 내 요인, K = 네트워크 간 요인), 기준 행 (240, 14, 3) ← λ = 320 main 네트워크
  routing   : 라우팅 오차 분리 — 경로 규칙 ROUTE_VARIANTS에서 같은 네트워크 재평가 (가중치 고정)
기본 네트워크 = 기준 셀(BASE_CELL)의 기본 arm 시드 TRAIN_SEEDS
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml

from env.orders import SIM_TIME
from env.rl import DEFAULT_ARM
from evaluation import Condition

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "design.yaml"), encoding="utf-8") as _f:
    _C = yaml.safe_load(_f)

BASE_ARM     = _C.get("base_arm") or DEFAULT_ARM
TRAIN_SEEDS  = _C["train_seeds"]        # main 학습 시드 (기본 네트워크)
SUB_SEEDS    = _C["sub_seeds"]          # 변형·민감도·요인 실험 학습 시드
N_TEST       = _C["n_test"]             # 테스트 인스턴스 수
N_VAL_SELECT = _C["n_val_select"]       # 베이스라인 선택용 검증 인스턴스 수
N_VAL_RL     = _C["n_val_rl"]           # RL 체크포인트 선택용 검증 인스턴스 수

SELECT_KEYS = _C["policies"]["select"]    # 기준(최고 삽입형 규칙) 선택 후보, BL7 = Greedy_R
BATCH_KEYS  = _C["policies"]["batching"]  # 배치 규칙 (개선 출처 분해용 최고 배치 규칙 선택)
ALL_KEYS    = _C["policies"]["all"]       # 셀마다 테스트할 베이스라인

MAIN_LAMS = _C["main_lams"]
FLEET_M   = _C["fleet_M"]
SMALL_M   = _C["small_M"]
VARIANT_ARMS = _C["variant_arms"]
# 보상 계수 민감도: (λ, arm), 앞에서부터 실행 (λ = 320 먼저, 항 제거 w = 0은 뒤)
SENS_DESIGN = ([(lam, f"{BASE_ARM}_{t}") for lam in _C["sens"]["lams"] for t in _C["sens"]["terms"]]
               + [(lam, f"{BASE_ARM}_{t}") for lam in _C["sens"]["lams"] for t in _C["sens"]["removal_terms"]])
ROUTE_VARIANTS = _C["route_variants"]


@dataclass(frozen=True)
class Cell:
    """평가/학습 셀 (λ orders/hr, M, K, 경로 규칙, 이동 가정, 시나리오 JSON (빈 문자열 = 기본 시나리오))"""
    lam:      float
    M:        int
    K:        int
    route:    str = "nnh"
    traffic:  str = "free"
    motion:   str = "reserve"
    scenario: str = ""

    @property
    def sc(self):
        """env.orders.Scenario (기본 시나리오이면 None)"""
        from env.orders import Scenario
        if not self.scenario:
            return None
        return Scenario(**json.loads(self.scenario))

    def layout(self):
        return self.cond.layout(self.M, self.K, self.sc)

    @property
    def cond(self) -> Condition:
        return Condition(traffic=self.traffic, motion=self.motion, route=self.route)

    @property
    def setting(self) -> str:
        return self.cond.setting(self.sc)

    @property
    def name(self) -> str:
        return f"lam{self.lam:.0f}_M{self.M}_K{self.K}"

    def dir(self, out: str) -> str:
        return os.path.join(out, self.setting, self.name)

    def args(self) -> List[str]:
        """train.py / run_compare.py 셀 인자"""
        a = ["--lam", str(self.lam), "--robots", str(self.M), "--ws", str(self.K),
             "--traffic", self.traffic, "--motion", self.motion]
        a += ["--route", self.route] if self.route != "nnh" else []
        return a + (["--scenario", self.scenario] if self.scenario else [])

    def __str__(self):
        return (f"λ={self.lam:.0f} M={self.M} K={self.K}" + ("" if self.route == "nnh" else f" route={self.route}")
                + (f" sc={self.setting}" if self.scenario else ""))


@dataclass(frozen=True)
class Ckpt:
    """체크포인트 (학습 셀, arm, 학습 시드)"""
    cell: Cell
    arm:  str
    seed: int

    def dir(self, out: str) -> str:
        return os.path.join(self.cell.dir(out), f"{self.arm}_s{self.seed}")

    def path(self, out: str) -> str:
        return os.path.join(self.dir(out), "agent.pt")

    def _suffix(self, target: Cell, tag: Optional[str]) -> str:
        s = ""
        if self.cell.M != target.M:
            s += f"_tr{self.cell.M}"
        if abs(self.cell.lam - target.lam) > 0.5:
            s += f"_trlam{self.cell.lam:.0f}"
        return s + target.cond.tag + (f"_{tag}" if tag else "")

    def run_name(self, target: Cell, tag: Optional[str] = None) -> str:
        """run_compare.py --load가 target 셀에서 만드는 run 이름 (같은 규칙)"""
        return f"{self.arm}_s{self.seed}" + self._suffix(target, tag)

    def policy(self, target: Cell, tag: Optional[str] = None) -> str:
        """target 셀에서 이 네트워크 그룹의 정책 이름 (analysis.load_records: run 이름에서 _s{시드}를 뺀 것)"""
        return self.arm + self._suffix(target, tag)


BASE_CELL = Cell(240, 14, 3)
MAIN_CELLS = [Cell(lam, 14, 3) for lam in MAIN_LAMS]
FLEET_CELLS = [Cell(round(240 * M / 14), M, 3) for M in FLEET_M]
SMALL_CELLS = [Cell(240, M, 3) for M in SMALL_M]
FACTORIAL_CELLS = [Cell(lam, M, K) for lam in (240, 320) for M in (10, 18) for K in (2, 4)]
FACTORIAL_TRAIN = {K: Cell(320, 14, K) for K in (2, 4)}      # K마다 네트워크 학습 셀
ROUTE_CELLS = [Cell(BASE_CELL.lam, BASE_CELL.M, BASE_CELL.K, route=r) for r in ROUTE_VARIANTS]


def base_nets(arm: str = BASE_ARM) -> List[Ckpt]:
    """기본 네트워크 (기준 셀, 시드 TRAIN_SEEDS)"""
    return [Ckpt(BASE_CELL, arm, s) for s in TRAIN_SEEDS]


def factorial_nets(K: int, arm: str = BASE_ARM) -> List[Ckpt]:
    """요인 실험 네트워크 (λ = 320, M = 14, 해당 K, 시드 SUB_SEEDS)"""
    return [Ckpt(FACTORIAL_TRAIN[K], arm, s) for s in SUB_SEEDS]


def factorial_ref_nets(arm: str = BASE_ARM) -> List[Ckpt]:
    """요인 실험 기준 행 네트워크 (main λ = 320 셀, 시드 TRAIN_SEEDS) → 기준 셀로 전이"""
    return [Ckpt(Cell(320, 14, 3), arm, s) for s in TRAIN_SEEDS]


PLANS = ["main", "variants", "sens", "fleet", "small", "factorial", "routing", "status"]
STAGES = ["cells", "train", "transfer"]


def save_plan_config(args, out: str, tool: str):
    """실험 설계(design.yaml)와 이 실행의 인자를 {out}/config_{tool}.yaml로 저장 (--dry-run이면 저장 안 함)"""
    if args.dry_run:
        return
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"config_{tool}.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"design": _C, "args": vars(args)}, f, allow_unicode=True, sort_keys=False)


def results_dir(out: str) -> str:
    """실험 폴더(runs/<이름>)의 사후 분석 결과 폴더 (results/<이름>)"""
    return os.path.join(REPO, "results", os.path.basename(os.path.abspath(out)))


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
    """subprocess를 최대 jobs개 동시 실행. items: {"cmd" 또는 순차 실행 "cmds", "log", "name"}. 출력: 실패한 item 목록"""
    for it in items:
        it["todo"] = list(it["cmds"]) if "cmds" in it else [it["cmd"]]
    if dry:
        for it in items:
            for cmd in it["todo"]:
                print("  [dry]", " ".join(cmd), ">>", it["log"])
        return []

    def start(it):
        it["proc"] = subprocess.Popen(it["todo"].pop(0), cwd=REPO, env=_env(), stderr=subprocess.STDOUT,
                                      stdout=open(it["log"], "a", encoding="utf-8"))

    pending, running, fails = list(items), [], []
    while pending or running:
        while pending and len(running) < jobs:
            it = pending.pop(0)
            os.makedirs(os.path.dirname(it["log"]), exist_ok=True)
            print(f"시작 {it['name']}: 로그 {it['log']}", flush=True)
            start(it)
            running.append(it)
        time.sleep(5)
        for it in list(running):
            rc = it["proc"].poll()
            if rc == 0 and it["todo"]:           # 순차 명령의 다음 단계
                start(it)
            elif rc is not None:
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
    from evaluation import MetricsStore
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
        seeds = val_seeds(N_VAL_SELECT)
        res = run_baselines(args, cells, SELECT_KEYS + (BATCH_KEYS if batching else []), seeds, "val.csv")
        for c, st in res.items():
            def rank(ks):
                out = []
                for k in ks:
                    rs = [r for r in st.records if r.label == policy_label(k)]
                    out.append({"key": k, "label": rs[0].label, "n": len(rs),
                                "avg_flow_time": sum(r.avg_flow_time for r in rs) / len(rs),
                                "completion_rate": sum(r.completion_rate for r in rs) / len(rs)})
                return sorted(out, key=lambda d: d["avg_flow_time"])
            ranking = rank(SELECT_KEYS)
            sel = {"best_key": ranking[0]["key"], "best_label": ranking[0]["label"], "val_seeds": seeds,
                   "sim_time": args.sim_time, "ranking": ranking, "cell": vars(c)}
            if batching:
                br = rank(BATCH_KEYS)
                sel.update(best_batching_key=br[0]["key"], best_batching=br[0]["label"], ranking_batching=br)
            path = os.path.join(c.dir(args.out), "select.json")
            if os.path.isfile(path):          # 다른 단계가 남긴 키(run_stages stage0의 bl6_star 등) 보존
                with open(path, encoding="utf-8") as f:
                    sel = {**json.load(f), **sel}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(sel, f, indent=1, ensure_ascii=False)
            msg = f"{c}: 최고 {sel['best_label']}" + (f", 배치 규칙 최고 {sel['best_batching']}" if batching else "")
            print(msg + "  (" + ", ".join(f"{d['label']} {d['avg_flow_time']:.1f}" for d in ranking) + ")")
            lines.append(msg + f" → {path}")
    keys = keys or ALL_KEYS
    res = run_baselines(args, cells, keys, test_seeds(args.n_test), "eval.csv")
    lines += [f"{c}: 테스트 {len(st.records)}행 → {c.dir(args.out)}/baselines/eval.csv" for c, st in res.items()]
    progress(args, f"cells {' '.join(keys)}", lines or ["실행 없음"])


def train_cmd(cell: Cell, arm: str, out: str, episodes: int, sim_time: float, val_every: int) -> List[str]:
    """train.py 명령 (--select val)"""
    return [sys.executable, "-u", "train.py", "--arm", arm, *cell.args(), "--episodes", str(episodes),
            "--sim-time", str(sim_time), "--select", "val", "--val-every", str(val_every), "--n-val", str(N_VAL_RL),
            "--out", out]


def eval_cmd(ckpt_path: str, cell: Cell, out: str, sim_time: float, n_test: int) -> List[str]:
    """체크포인트를 cell에서 테스트하는 run_compare 명령 (베이스라인 없음)"""
    return [sys.executable, "-u", "-m", "experiments.run_compare", "--load", ckpt_path, *cell.args(),
            "--sim-time", str(sim_time), "--n-eval", str(n_test), "--baselines", "--out", out]


def train_then_eval(ck: Ckpt, train: List[str], out: str, sim_time: float, n_test: int, retrain: bool = False) -> List[List[str]]:
    """학습 → 테스트 명령 목록. 체크포인트가 이미 있으면 (테스트만 실패한 경우) 학습은 건너뜀"""
    return ([] if os.path.isfile(ck.path(out)) and not retrain else [train]) + [eval_cmd(ck.path(out), ck.cell, out, sim_time, n_test)]


def step_train(args, pairs: List[tuple], seeds: List[int], title: str) -> bool:
    """(셀, arm) 쌍마다 μ(λ) 표 추정 (최대 --mu-jobs개 동시) → 쌍 × 학습 시드를 한 작업 풀(--jobs)로 학습·테스트"""
    from env.rl import ARMS
    if args.filter:
        pairs = [(c, a) for c, a in pairs if args.filter in a]
    tr = lambda c, a: train_cmd(c, a, args.out, args.episodes, args.sim_time, args.val_every)
    mu_items = [{"cmd": tr(c, a) + ["--mu-only"], "name": f"μ {c} {a}", "pair": (c, a),
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
            items.append({"cmds": train_then_eval(ck, tr(c, arm) + ["--seed", str(s)], args.out, args.sim_time, args.n_test),
                          "name": f"{c} {arm} s{s}",
                          "log": os.path.join(c.dir(args.out), "logs", f"{arm}_s{s}.log"), "dir": ck.dir(args.out)})
    fails = run_procs(items, args.jobs, args.dry_run)
    progress(args, title, _summary_lines(items, fails))
    return not (fails or mu_fails)


def step_transfer(args, ckpts: List[Ckpt], targets: List[Cell], tag: Optional[str] = None) -> bool:
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
            cmd = eval_cmd(src, t, args.out, args.sim_time, args.n_test)
            if tag:
                cmd += ["--tag", tag]
            items.append({"cmd": cmd, "name": f"{t} ← {name}", "dir": d,
                          "log": os.path.join(t.dir(args.out), "logs", f"{name}.log")})
    fails = run_procs(items, args.jobs, args.dry_run)
    progress(args, f"transfer {len(ckpts)} ckpt × {len(targets)} 셀" + (f" tag={tag}" if tag else ""),
             _summary_lines(items, fails))
    return not fails


def status(args):
    """모든 계획의 학습 run 진행: 완료(eval.csv) / 학습 중 마지막 [Ep N] / 시작 전"""
    pat = re.compile(r"\[Ep\s+(\d+)\]")
    pairs = ([(c, args.arm, TRAIN_SEEDS) for c in MAIN_CELLS]
             + [(BASE_CELL, a, SUB_SEEDS) for a in VARIANT_ARMS]
             + [(Cell(lam, 14, 3), a, SUB_SEEDS) for lam, a in SENS_DESIGN]
             + [(c, args.arm, SUB_SEEDS) for c in FACTORIAL_TRAIN.values()])
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
            step_cells(args, MAIN_CELLS)
        if want("train"):
            step_train(args, [(c, arm) for c in MAIN_CELLS], TRAIN_SEEDS, f"train main {arm}")
    elif plan == "variants":
        if want("train"):
            step_train(args, [(BASE_CELL, a) for a in VARIANT_ARMS], SUB_SEEDS, "train variants")
    elif plan == "sens":
        if want("train"):
            step_train(args, [(Cell(lam, 14, 3), a) for lam, a in SENS_DESIGN], SUB_SEEDS, "train sens")
    elif plan in ("fleet", "small"):
        cells = FLEET_CELLS if plan == "fleet" else SMALL_CELLS
        if want("cells"):
            step_cells(args, cells, batching=False)
        if want("transfer"):
            step_transfer(args, base_nets(arm), cells)
    elif plan == "factorial":
        if want("train"):
            step_train(args, [(c, arm) for c in FACTORIAL_TRAIN.values()], SUB_SEEDS, f"train factorial {arm}")
        if want("cells"):
            step_cells(args, FACTORIAL_CELLS, batching=False)
        if want("transfer"):
            for K in FACTORIAL_TRAIN:
                step_transfer(args, factorial_nets(K, arm), [c for c in FACTORIAL_CELLS if c.K == K])
            step_transfer(args, factorial_ref_nets(arm), [BASE_CELL])
    elif plan == "routing":
        if want("cells"):
            step_cells(args, ROUTE_CELLS, select=False, keys=["BL6"])
        if want("transfer"):
            step_transfer(args, base_nets(arm), ROUTE_CELLS)


def main():
    from env.rl import ARMS
    p = argparse.ArgumentParser(description="실험 실행기", formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("plan", choices=PLANS)
    p.add_argument("--stage", choices=["all"] + STAGES, default="all", help="계획 안에서 실행할 단계")
    p.add_argument("--arm", choices=list(ARMS), default=BASE_ARM, help="기본 arm 대신 쓸 arm (새 보상 실험)")
    p.add_argument("--filter", default=None, help="train 단계에서 arm 이름에 이 문자열이 있는 것만 (PC 분할)")
    p.add_argument("--out", default=os.path.join(REPO, "runs", "exp"), help="결과 폴더 (runs/<실험 이름>)")
    p.add_argument("--workers", type=int, default=4, help="베이스라인 병렬 프로세스 수")
    p.add_argument("--jobs", type=int, default=6, help="동시 학습·평가 프로세스 수")
    p.add_argument("--mu-jobs", type=int, default=2, help="μ(λ) 추정 동시 프로세스 수 (각 LOCAL_MU_WORKERS개 작업자)")
    p.add_argument("--sim-time", type=float, default=SIM_TIME)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=N_TEST)
    p.add_argument("--val-every", type=int, default=50)
    p.add_argument("--dry-run", action="store_true", help="실행할 명령만 출력")
    args = p.parse_args()
    args.out = os.path.abspath(args.out)
    if args.plan != "status":
        save_plan_config(args, args.out, f"plan_{args.plan}")
    run_plan(args, args.plan)


if __name__ == "__main__":
    main()
