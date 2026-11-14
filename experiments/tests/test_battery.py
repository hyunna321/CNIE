"""배터리 제약 (env.battery) 단위 테스트

python -m unittest experiments.tests.test_battery -v
  - 제약 없음: 기본 경로에서 Battery를 만들지 않음
  - SoC 보존: 소모량 = 실제 이동 거리, 최종 SoC = 초기 + 충전 − 소모
  - hard 제약: 모든 정책 계열에서 SoC가 음수로 가지 않고, 필수 충전과 큐 방출이 실제로 일어나며 토트가 큐에 영구히 남지 않음
  - 틱 건너뛰기: 일이 없어도 경과시간만큼 충전됨
  - 설정 검증: 단일 트립도 못 하는 용량은 거부
"""
import unittest

import numpy as np

from agents import DQNAgent
from baselines import resolve_policy
from env.battery import Battery, BatteryConfig
from env.layout import ROBOT_CAPACITY
from env.rl import ARMS, RLSimulator
from evaluation import Condition

TIGHT = BatteryConfig(range_m=900.0, charge_rate=3.0, reserve_m=30.0, low_soc=0.25, target_soc=0.7)


def _sim(key, cfg, M=6, K=3, lam_hr=200, seed=10000, sim_time=7200, motion="reserve"):
    L = Condition(motion=motion).layout(M, K)
    sim = resolve_policy(key)[1](L, lam=lam_hr / 3600, seed=seed, sim_time=sim_time, num_robots=M,
                                 robot_capacity=L.robot_capacity, battery=cfg)
    sim.reset(seed)
    return sim


class TestBattery(unittest.TestCase):

    def test_off_by_default(self):
        self.assertIsNone(_sim("BL6", None).battery)

    def test_soc_balance(self):
        sim = _sim("BL6", TIGHT)
        sim.run()
        res = sim.build_result("x", 0, sim.lam)
        total = sum(sim.battery.soc(r) for r in sim.robots)
        start = len(sim.robots) * TIGHT.initial_soc * TIGHT.range_m
        self.assertAlmostEqual(total, start + sim.battery.charged_m - sim._total_sim_distance, places=6)
        self.assertAlmostEqual(res.charge_time, sim.battery.charged_m / TIGHT.charge_rate, places=6)

    def test_hard_constraint_all_policy_families(self):
        for key, motion in (("BL6", "reserve"), ("BL5", "reactive"), ("BL8", "reserve"), ("BL1", "reserve"),
                            ("BL2", "reactive")):
            with self.subTest(key=key, motion=motion):
                sim = _sim(key, TIGHT, motion=motion)
                sim.run()
                res = sim.build_result(key, 0, sim.lam)
                self.assertEqual(res.n_soc_depleted, 0)
                self.assertGreaterEqual(res.min_soc, 0.0)
                self.assertGreater(res.n_charge_required, 0, "필수 충전이 검증되지 않음")
                self.assertGreater(res.n_completed, 0)

    def test_queue_released_after_charging(self):
        # 로봇 1대, 과부하: 큐는 트립 완료·충전 완료에서만 방출되므로 충전을 거쳐도 처리가 계속되어야 함
        sim = _sim("BL6", TIGHT, M=1, lam_hr=120, sim_time=10800)
        sim.run()
        res = sim.build_result("x", 0, sim.lam)
        self.assertGreater(res.n_charge_required, 1)
        last_done = max(t.done_time for t in sim.order_mgr.all_totes if t.done_time is not None)
        self.assertGreater(last_done, 0.9 * sim.sim_time, "충전 후 처리가 멈춤")
        self.assertEqual(res.n_soc_depleted, 0)

    def test_charge_during_time_skip(self):
        cfg = BatteryConfig(range_m=900.0, charge_rate=3.0, reserve_m=30.0, low_soc=0.1, target_soc=0.7, initial_soc=0.5)
        sim = _sim("BL6", cfg, lam_hr=1e-3, sim_time=100)
        sim.run()
        self.assertEqual(sim.order_mgr.queue_size(), 0)
        for r in sim.robots:
            self.assertAlmostEqual(sim.battery.soc(r), 450.0 + 3.0 * 100.0, places=6)

    def test_mid_trip_infeasible_filtered(self):
        sim = _sim("BL6", TIGHT, M=3)
        sim.eq = type(sim.eq)()
        r = sim.robots[0]
        from experiments.tests.test_planning import _manual
        cells = sorted(set(sim.layout.access_coords))
        far = max(cells, key=lambda c: sim.layout.distance(r.pos, c))
        o = _manual(sim, 1, [far], ws=0)
        need = sim.battery.need(r, o.totes[0])
        r.soc, r.charge_t0 = need - 1.0, None
        self.assertNotIn(r, sim._eligible_robots(o.totes[0]))
        r.soc = need
        self.assertIn(r, sim._eligible_robots(o.totes[0]))

    def test_rejects_too_small_range(self):
        L = Condition().layout(3, 3)
        with self.assertRaises(ValueError):
            resolve_policy("BL6")[1](L, lam=0.01, seed=1, sim_time=100, num_robots=3, robot_capacity=5,
                                     battery=BatteryConfig(range_m=100.0))

    def test_rl_battery_arm(self):
        L = Condition().layout(6, 3)
        arm = ARMS["local_battery"]
        agent = DQNAgent(robot_capacity=ROBOT_CAPACITY, dist_norm=L.dist_norm, seed=1, sim_time=3600, num_robots=6,
                         num_workstations=3, n_step=arm.n_step, features=arm.features)
        sim = RLSimulator(L, agent, lam=200 / 3600, seed=10000, sim_time=3600, num_robots=6,
                          robot_capacity=L.robot_capacity, arm=arm, battery=TIGHT)
        sim.reset(10000)
        trans = sim.run()
        self.assertGreater(len(trans), 0)
        pad, mask, _, R, W = trans[0][0]
        self.assertEqual(pad.shape[1], agent.feat_dim)
        self.assertEqual(W.shape[0], agent.ctx_dim)
        self.assertFalse(np.isnan(pad[mask]).any())
        self.assertEqual(sim.build_result("x", 0, sim.lam).n_soc_depleted, 0)
        with self.assertRaises(ValueError):
            RLSimulator(L, agent, lam=0.01, seed=1, sim_time=100, num_robots=6, robot_capacity=L.robot_capacity, arm=arm)


if __name__ == "__main__":
    unittest.main()
