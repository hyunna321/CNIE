"""planning 모듈 단위 테스트 (작업 지시 2.3)

python -m unittest tests.test_planning -v
  - planned_completion: 로봇 1대, 혼잡 없음에서 계획 완료시각 = 실현 완료시각
  - Delta_order: 형제 토트가 병목이면 0, 탑재 토트가 자기 주문의 병목이 아니면 그 주문 기여 0
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines import MarginalCostSimulator          # noqa: E402
from evaluation import Condition                     # noqa: E402
from planning import planned_completions, delta_order, robot_timeline   # noqa: E402
from simulator import Order, Tote                    # noqa: E402


class _Recording(MarginalCostSimulator):
    """배정·트립 재출발 직후마다 계획 완료시각을 기록하는 BL6"""

    def reset(self, seed):
        super().reset(seed)
        self.pred = {}           # 토트 id -> 마지막 계획 완료시각
        self.n_mid = 0

    def _snap(self):
        for x, t in planned_completions(self, "estimate").items():
            self.pred[x.tote_id] = t

    def _assign_tote(self, robot, tote):
        self.n_mid += not robot.is_idle
        super()._assign_tote(robot, tote)
        self._snap()

    def _dispatch(self, robot, totes):
        super()._dispatch(robot, totes)
        self._snap()


class TestPlannedCompletion(unittest.TestCase):

    def _run(self, lam_hr, sim_time=3600, seed=10000):
        L = Condition().layout(1, 3)
        sim = _Recording(L, lam=lam_hr / 3600, seed=seed, sim_time=sim_time, num_robots=1, robot_capacity=5)
        sim.reset(seed)
        sim.run()
        done = [t for t in sim.order_mgr.all_totes if t.done_time is not None]
        return sim, done

    def test_single_robot_matches_realized(self):
        for lam in (20, 60):            # 저부하 (삽입 거의 없음), 고부하 (운행 중 삽입·큐 방출)
            sim, done = self._run(lam)
            self.assertGreater(len(done), 20)
            for t in done:
                self.assertAlmostEqual(sim.pred[t.tote_id], t.done_time, places=6,
                                       msg=f"λ={lam} 토트 {t.tote_id}")
            if lam == 60:
                self.assertGreater(sim.n_mid, 0, "운행 중 삽입 경우가 검증되지 않음")


def _manual(sim, oid, locs, ws, t=0.0):
    """주문 하나를 직접 만들어 시뮬레이터에 등록 (토트는 미배정)"""
    o = Order(oid, t, ws_id=ws)
    for i, c in enumerate(locs):
        x = Tote(1000 * oid + i, o, sku=i, loc_idx=0, loc=c, pick_s=5.2, proc_s=4.9)
        o.totes.append(x)
        sim.order_mgr.all_totes.append(x)
    sim.order_mgr.all_orders.append(o)
    sim._ws_load[ws] += len(o.totes)
    sim._n_in_system += 1
    return o


class TestDeltaOrder(unittest.TestCase):

    def setUp(self):
        L = Condition().layout(3, 3)
        self.sim = MarginalCostSimulator(L, lam=1e-6, seed=1, sim_time=3600, num_robots=3, robot_capacity=5)
        self.sim.reset(1)
        self.sim.eq = type(self.sim.eq)()                 # 자동 주문 도착 없음
        cells = sorted(set(L.access_coords))
        r0, r1 = self.sim.robots[0], self.sim.robots[1]
        self.far  = max(cells, key=lambda c: L.aisle_distance(r0.pos, c))
        self.near = min(cells, key=lambda c: L.aisle_distance(r1.pos, c))
        self.mid  = min(cells, key=lambda c: L.aisle_distance(r0.pos, c))

    def test_sibling_bottleneck_zero(self):
        sim = self.sim
        o = _manual(sim, 1, [self.far, self.near], ws=0)
        sim._assign_tote(sim.robots[0], o.totes[0])        # 형제(먼 위치)를 로봇 0에
        pc = planned_completions(sim, "estimate")
        d, tk, n_late = delta_order(sim, o.totes[1], sim.robots[1], pc)
        self.assertLess(tk, pc[o.totes[0]], "전제: 후보에서 k가 형제보다 먼저 끝나야 함")
        self.assertAlmostEqual(d, 0.0, places=9)
        self.assertEqual(n_late, 0)

    def test_onboard_not_bottleneck_zero(self):
        sim = self.sim
        o2 = _manual(sim, 2, [self.mid, self.far], ws=1)
        sim._assign_tote(sim.robots[0], o2.totes[0])       # 탑재 토트 (가까움)
        sim._assign_tote(sim.robots[2], o2.totes[1])       # 같은 주문의 병목 (먼 위치)
        o3 = _manual(sim, 3, [self.near], ws=2)
        pc = planned_completions(sim, "estimate")
        detail = {}
        delta_order(sim, o3.totes[0], sim.robots[0], pc, detail)
        after = robot_timeline(sim, sim.robots[0], extra=o3.totes[0])
        self.assertLessEqual(after[o2.totes[0]], pc[o2.totes[1]], "전제: 탑재 토트가 지연돼도 병목보다 빨라야 함")
        self.assertGreater(after[o2.totes[0]], pc[o2.totes[0]], "전제: 삽입으로 탑재 토트가 늦어져야 함")
        self.assertAlmostEqual(detail[2], 0.0, places=9)
        self.assertGreater(detail[3], 0.0)


if __name__ == "__main__":
    unittest.main()
