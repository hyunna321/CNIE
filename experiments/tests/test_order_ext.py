"""R2 보상 order_ext (env.rl.OrderExtReward, arm rsv_r2_ord_eta0.2[_bat]) 테스트

python -m unittest experiments.tests.test_order_ext -v
  - arm은 A0 + ring + online(rsv_local_h3_ro_bat)과 보상 설정(reward, eta)만 다름 (로봇별 예약 큐)
  - ring 확정값 = finalize 값
  - η = 0이면 Σ 증분 = Σ_o censored 흐름시간, η = 0.2이면 c_k = c_k(η = 0) + 0.2 · 자기 지연
  - 외부비용 이전 Σ(E − X)는 이전받을 결정이 없던 몫만 남음
"""
import unittest
from dataclasses import asdict, replace

import numpy as np

from env.battery import BatteryConfig
from env.layout import ROBOT_CAPACITY
from env.rl import ARMS, mu_filename
from evaluation import Condition
from td import TDAgent, rl_sim_class

M, LAM, T, SEED = 6, 240 / 3600, 3600, 10000
MU = 150.0          # 고정 중심화 기준 (에피소드 평균 대신, 비교가 μ에 흔들리지 않게)
ARM = ARMS["rsv_r2_ord_eta0.2_bat"]
ARM0 = replace(ARM, eta=0.0)          # η = 0 (순수 주문 단위)


def _run(arm):
    L = Condition().layout(M, 3)
    agent = TDAgent(robot_capacity=ROBOT_CAPACITY, dist_norm=L.dist_norm, seed=1, sim_time=T, num_robots=M,
                    num_workstations=3, n_step=arm.n_step, features=arm.features, beta=arm.beta_value)
    agent.mu_table = ([LAM], [MU])
    sim = rl_sim_class(arm, train=True)(L, agent, lam=LAM, seed=SEED, sim_time=T, num_robots=M,
                                        robot_capacity=L.robot_capacity, arm=arm, battery=BatteryConfig())
    sim.update_every = 10 ** 9            # 업데이트 없이 수집만 (학습 전 정책 = BL6R prior)
    sim.reset(SEED)
    sim.run()
    return sim, agent


def _censored_flows(sim):
    return sum(min(o.completion_time if o.completion_time is not None else T, T) - o.arrival_time
               for o in sim.order_mgr.all_orders)


def _parts(sim):
    rw = sim.reward
    return [rw._cost(k, T) for k in range(len(rw._times))]


class TestOrderExt(unittest.TestCase):

    def test_same_queue_rule_as_a0(self):
        base = asdict(ARMS["rsv_local_h3_ro_bat"])
        arm = asdict(ARM)
        self.assertTrue(arm["reserve"])
        self.assertEqual({k for k in base if base[k] != arm[k]}, {"reward", "eta"})
        self.assertEqual(mu_filename(ARM), "mu_order_ext_eta0.2_rsv.json")

    def test_ring_equals_finalize(self):
        sim, agent = _run(ARM)
        fin = sim.reward.finalize()
        n = len(fin)
        self.assertGreater(n, 100)
        self.assertTrue(np.allclose(agent.replay.r[:n], fin, atol=1e-12))
        self.assertTrue(agent.replay.has_r[:n].all())

    def test_increments_sum_to_flow(self):
        sim, _ = _run(ARM0)
        self.assertEqual(sim.reward.unattributed, 0)
        inc = sum(p[1] for p in _parts(sim))
        self.assertAlmostEqual(inc / _censored_flows(sim), 1.0, places=9)

    def test_transfer_closed(self):
        sim, _ = _run(ARM0)
        rw = sim.reward
        E, X = sum(rw._ext), sum(rw._recv)
        self.assertGreater(sum(abs(e) for e in rw._ext), 0.0, "전제: 외부비용이 있어야 함")
        self.assertLessEqual(abs(E - X), 0.05 * sum(abs(e) for e in rw._ext))   # 이전받을 결정이 없던 몫만

    def test_eta(self):
        a, _ = _run(ARM0)
        b, _ = _run(ARM)
        pa, pb = _parts(a), _parts(b)
        self.assertEqual(len(pa), len(pb))
        self.assertTrue(np.allclose([x[0] + 0.2 * x[3] for x in pa], [y[0] for y in pb], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
