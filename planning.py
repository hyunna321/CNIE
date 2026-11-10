"""계획 완료시각 추정기 planned_completion과 주문 단위 계획 비용 Delta_order

BL8, RL 주문 특징, tote_order = bottleneck_first, little_pbrs 보상의 Φ가 모두 이 모듈의 같은 함수를 씀

토트 j의 계획 처리 완료시각 Ť_j(s) (로봇의 남은 트립을 따라 걷기, robot_timeline)
  t_base = 결정 시점 로봇 상태의 틱 시각 (sim._t0_ctx · tick, 경로 계획 시작 기준과 같음)
  + 진행 중인 피킹/WS 처리의 남은 틱
  + 방문 지점마다 이동시간 d_aisle / v
  + 피킹 셀: 그 셀에서 집는 토트마다 피킹 시간 (틱 올림, 시뮬레이터와 같은 규칙)
  + 도킹 셀: 도착 시점의 WS 잔여 backlog 대기 max(0, backlog_w − (도착 − t_base))
             + 그 WS로 내리는 토트의 처리 시간 (PC_DOCK: batch = 처리 합 끝에 모두 완료 (시뮬레이터 동작),
               sequential = j번째 토트는 j번째까지의 처리 합)
  backlog_w (ws_backlog) = w 도킹 셀에서 처리 중인 다른 로봇의 남은 처리 틱
                           + w 대기열 셀에서 w 도킹을 기다리는 다른 로봇이 내릴 토트의 처리 틱
  다른 로봇의 예약·혼잡은 무시 (후보 평가의 Δd와 같은 가정)
확정된 계획 (planned_completions): PC_CONFIRMED = "estimate"이면 위 추정, "sipp"이면 reserve 모드 예약 경로의
  도착 틱 (sipp_timeline, 맞는 경로가 없으면 추정). 가상 삽입과 Delta_order는 항상 추정
주문 계획 완료시각 Ť_o(s) = max(t, max_{j∈o, 배정·미처리} Ť_j(s)), 미배정 토트(대기 큐 포함)는 t
Delta_order(k, m) = Σ_o [Ť_o(배정 후) − Ť_o(배정 전)]: 후보 m에 토트 k를 가상 삽입(NNH 재구성, 실제 배정과 같은
  경로 규칙)한 결과. 영향받는 주문 = k의 주문과 m에 실린 토트들의 주문
모든 함수는 시뮬레이터 상태를 바꾸지 않음
"""
import math
from typing import Dict, List, Optional, Tuple

from constant import ROBOT_SPEED, PC_CONFIRMED, PC_DOCK


def _ticks(sec: float, tick: float) -> int:
    """머무는 시간(s)을 틱 수로 올림 (SimulatorBase._grid_ticks와 같음)"""
    return int(math.ceil(sec / tick - 1e-9)) if sec > 0.0 else 0


def ws_backlog(sim, exclude_id: Optional[int] = None) -> List[float]:
    """WS별 잔여 backlog (s): 도킹 중인 로봇의 남은 처리 + 대기열 셀에서 그 도킹을 기다리는 로봇의 처리 (exclude_id 제외)"""
    L, tick = sim.layout, sim.layout.tick
    b = [0.0] * sim.num_workstations
    qmap = getattr(L, "_queue_ws", None)
    if qmap is None:
        qmap = L._queue_ws = {c: k for k, cells in enumerate(L.queue_cells) for c in cells}
    for r in sim.robots:
        if r.id == exclude_id:
            continue
        if r.timer_kind == "ws" and r.pos in L.dock_ws:
            b[L.dock_ws[r.pos]] += _ticks(r.timer, tick) * tick
        elif r.pos in qmap and r.current_dest == L.workstations[qmap[r.pos]]:
            k = qmap[r.pos]
            b[k] += _ticks(sum(t.proc_s for t in sim._robot_totes[r.id] if t.picked and t.ws_id == k), tick) * tick
    return b


def _visits(sim, r, extra=None) -> Tuple[List, List]:
    """r의 남은 방문 지점과 그 트립의 미하역 토트 (배정 순서). extra가 있으면 실제 배정과 같은 규칙으로 가상 삽입

    삽입 실패로 미뤄 둔 토트 위치(pending_insert)는 다음 재계획처럼 경로에 넣어 계산
    """
    totes = list(sim._robot_totes[r.id]) + ([extra] if extra is not None else [])
    if extra is None and not r.pending_insert:
        head = [r.current_dest] if r.current_dest is not None and r.timer == 0.0 else []
        return head + r.stops, totes
    docks = sim._needed_docks(r)
    if extra is not None:
        dock = sim.layout.workstations[extra.ws_id]
        if dock not in docks:
            docks.append(dock)
    new = [extra.loc] if extra is not None else []
    if r.is_idle:
        picks, drops, _ = r._plan(r.pos, [t.loc for t in totes if not t.picked], docks, keep_docks=False)
        return picks + drops, totes
    start = r.current_dest if r.current_dest is not None else r.pos
    picks, drops, _ = r._plan(start, r.route + list(r.pending_insert) + new, docks, keep_docks=True)
    head = [r.current_dest] if r.current_dest is not None and r.timer == 0.0 else []
    return head + picks + drops, totes


def robot_timeline(sim, r, extra=None, backlog: Optional[List[float]] = None) -> Dict[object, float]:
    """r에 배정된 미처리 토트(+ 가상 삽입 extra)의 계획 처리 완료시각 {토트: Ť (s)} (모듈 설명의 추정)"""
    L, tick = sim.layout, sim.layout.tick
    t_base = sim._t0_ctx * tick
    backlog = backlog if backlog is not None else ws_backlog(sim, r.id)
    visits, totes = _visits(sim, r, extra)
    done: Dict[object, float] = {}
    t = t_base
    if r.timer > 0.0:
        t += _ticks(r.timer, tick) * tick
        if r.timer_kind == "ws":
            for x in sim._dropping[r.id]:
                done[x] = t
    held = [x for x in totes if x.picked]
    left = [x for x in totes if not x.picked]
    p = r.pos
    for c in visits:
        t += L.aisle_distance(p, c) / ROBOT_SPEED
        p = c
        if L.is_dock(c):
            k = L.dock_ws[c]
            drop = [x for x in held if x.ws_id == k]
            if not drop:
                continue
            t = max(t, t_base + backlog[k])
            if PC_DOCK == "sequential":
                acc = 0.0
                for x in drop:
                    acc += x.proc_s
                    done[x] = t + _ticks(acc, tick) * tick
            t += _ticks(sum(x.proc_s for x in drop), tick) * tick
            if PC_DOCK != "sequential":
                for x in drop:
                    done[x] = t
            held = [x for x in held if x.ws_id != k]
        else:
            x = next((y for y in left if y.loc == c), None)
            if x is not None:
                left.remove(x)
                held.append(x)
                t += _ticks(x.pick_s, tick) * tick
    for x in totes:                      # 경로에 없는 토트 (있으면 안 됨): 트립 끝 시각
        done.setdefault(x, t)
    return done


def sipp_timeline(sim, r) -> Optional[Dict[object, float]]:
    """reserve 모드의 확정 예약 경로에서 읽은 완료시각 (예약이 지금 경로와 맞지 않으면 None)"""
    if not sim.reserve or not r.plan or r.plan_version != r.route_version or r.pending_insert:
        return None
    L, tick = sim.layout, sim.layout.tick
    stay0, legs = sim._reserve_legs(r)
    t0 = sim._t0_ctx
    done: Dict[object, float] = {}
    if r.timer > 0.0 and r.timer_kind == "ws":
        for x in sim._dropping[r.id]:
            done[x] = (t0 + stay0) * tick
    plan = list(r.plan)
    held = [x for x in sim._robot_totes[r.id] if x.picked]
    left = [x for x in sim._robot_totes[r.id] if not x.picked]
    i = 0
    for c, d in legs:
        while i < len(plan) and not (plan[i][1] == c and all(plan[j][1] == c for j in range(i + 1, min(len(plan), i + 1 + d)))):
            i += 1
        if i >= len(plan):
            return None
        arrive = plan[i][0]
        if L.is_dock(c):
            k = L.dock_ws[c]
            for x in [x for x in held if x.ws_id == k]:
                done[x] = (arrive + d) * tick
            held = [x for x in held if x.ws_id != k]
        else:
            x = next((y for y in left if y.loc == c), None)
            if x is not None:
                left.remove(x)
                held.append(x)
        i += 1 + d if d else 1
    if len(done) < len(sim._robot_totes[r.id]) + (len(sim._dropping[r.id]) if r.timer_kind == "ws" else 0):
        return None
    return done


def planned_completions(sim, mode: str = None) -> Dict[object, float]:
    """모든 로봇의 배정·미처리 토트 계획 완료시각 {토트: Ť}. mode: "estimate" / "sipp" (기본 PC_CONFIRMED)"""
    mode = mode or PC_CONFIRMED
    out: Dict[object, float] = {}
    for r in sim.robots:
        tl = sipp_timeline(sim, r) if mode == "sipp" else None
        out.update(tl if tl is not None else robot_timeline(sim, r))
    return out


def order_completion(order, pc: Dict[object, float], t: float) -> float:
    """주문 계획 완료시각 Ť_o = max(t, 배정·미처리 토트의 Ť)"""
    return max([t] + [pc[x] for x in order.totes if x in pc])


def tote_completion(sim, tote, r) -> float:
    """토트 tote를 로봇 r에 가상 삽입했을 때 그 토트의 계획 완료시각 Ť_{k,m}"""
    return robot_timeline(sim, r, extra=tote)[tote]


def delta_order(sim, tote, r, pc: Optional[Dict[object, float]] = None,
                detail: Optional[dict] = None) -> Tuple[float, float, int]:
    """토트 tote를 후보 r에 가상 삽입했을 때 (Delta_order, Ť_{k,m}, 계획 완료가 늦어지는 다른 주문 수)

    pc: 배정 전 계획 완료시각 (planned_completions(sim, "estimate"), 후보마다 다시 계산하지 않도록 넘김)
    detail: dict를 넘기면 {order_id: 그 주문의 Ť_o 증가}를 채움
    """
    if pc is None:
        pc = planned_completions(sim, "estimate")
    t = sim.now
    new = robot_timeline(sim, r, extra=tote)
    after = dict(pc)
    for x in sim._robot_totes[r.id]:
        after.pop(x, None)
    after.update(new)
    total, n_late = 0.0, 0
    for o in {tote.order} | {x.order for x in new}:
        d = order_completion(o, after, t) - order_completion(o, pc, t)
        total += d
        if detail is not None:
            detail[o.order_id] = d
        if o is not tote.order and d > 1e-9:
            n_late += 1
    return total, new[tote], n_late
