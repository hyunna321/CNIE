"""로봇별 다음 트립 예약 (baselines.policies.MarginalCostReserveSimulator, BL6R) 테스트

python -m unittest experiments.tests.test_reserve -v
  - 후보가 있는 동안은 BL6와 결과가 같음 (저부하)
  - 고부하에서 예약이 실제로 쓰이고 토트가 사라지거나 중복되지 않음
  - 배정 시각 = 실제 출발 시각 (도착 이후, 예약 시각 아님)
"""
import unittest

from baselines import resolve_policy
from evaluation import Condition


def _run(key, M, lam_hr, sim_time=3600, seed=10000):
    L = Condition().layout(M, 3)
    sim = resolve_policy(key)[1](L, lam=lam_hr / 3600, seed=seed, sim_time=sim_time, num_robots=M,
                                 robot_capacity=L.robot_capacity)
    sim.reset(seed)
    sim.run()
    return sim


class TestReserve(unittest.TestCase):

    def test_same_as_bl6_when_never_saturated(self):
        a, b = _run("BL6c0.25", 20, 120), _run("BL6Rc0.25", 20, 120)
        self.assertEqual(b.n_reserved, 0, "전제: 저부하에서는 예약이 없어야 함")
        self.assertEqual(a.build_result("x", 0, a.lam).avg_flow_time, b.build_result("x", 0, b.lam).avg_flow_time)

    def test_conservation_and_assign_time(self):
        sim = _run("BL6Rc0.25", 6, 240)
        self.assertGreater(sim.n_reserved, 50)
        places = []
        for r in sim.robots:
            places += sim._robot_totes[r.id] + sim._dropping[r.id] + sim._resv[r.id]
        places += sim.order_mgr.wait_queue
        ids = [t.tote_id for t in places]
        self.assertEqual(len(ids), len(set(ids)), "토트 중복")
        done = {t.tote_id for t in sim.order_mgr.all_totes if t.done_time is not None}
        self.assertFalse(done & set(ids))
        self.assertEqual(len(done) + len(ids), len(sim.order_mgr.all_totes), "토트 유실")
        reserved_now = {t.tote_id for r in sim.robots for t in sim._resv[r.id]}
        for t in sim.order_mgr.all_totes:
            if t.assign_time is not None:
                self.assertGreaterEqual(t.assign_time, t.order.arrival_time)
            if t.tote_id in reserved_now:
                self.assertIsNone(t.assign_time, "예약만 된 토트에 배정 시각이 기록됨")


class TestReserveBattery(unittest.TestCase):

    def test_reserve_with_battery(self):
        from env.battery import BatteryConfig
        cfg = BatteryConfig(range_m=1500.0, charge_rate=3.0, reserve_m=30.0, low_soc=0.25, target_soc=1.0)
        L = Condition().layout(6, 3)
        sim = resolve_policy("BL6Rc0.25")[1](L, lam=240 / 3600, seed=10000, sim_time=10800, num_robots=6,
                                             robot_capacity=L.robot_capacity, battery=cfg)
        sim.reset(10000)
        sim.run()
        res = sim.build_result("x", 0, sim.lam)
        self.assertEqual(res.n_soc_depleted, 0)
        self.assertGreater(res.n_charge_required, 0, "충전이 검증되지 않음")
        self.assertGreater(sim.n_reserved, 50, "예약이 검증되지 않음")
        for r in sim.robots:
            if r.must_charge:
                self.assertEqual(sim._resv[r.id], [], "필수 충전 로봇에 예약이 남음")
        places = [t for r in sim.robots for t in sim._robot_totes[r.id] + sim._dropping[r.id] + sim._resv[r.id]]
        places += sim.order_mgr.wait_queue
        ids = [t.tote_id for t in places]
        done = {t.tote_id for t in sim.order_mgr.all_totes if t.done_time is not None}
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(done) + len(ids), len(sim.order_mgr.all_totes), "토트 유실")
        last = max(t.done_time for t in sim.order_mgr.all_totes if t.done_time is not None)
        self.assertGreater(last, 0.95 * sim.sim_time, "처리가 중간에 멈춤")


if __name__ == "__main__":
    unittest.main()
