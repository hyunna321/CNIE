"""R4 가중치 arm (rsv_r4_wext{2,3}[_bat]) 테스트

python -m unittest experiments.tests.test_r4_wext -v
  - arm은 A0 + ring + online(rsv_local_h3_ro_bat)과 local_w(w_ext)만 다름, μ 표는 계수별 파일
  - 같은 궤적(학습 전 정책 = BL6R prior)에서 c_k(w_ext) = c_k(A0) + (w_ext − 1) · 2항
"""
import unittest
from dataclasses import asdict

import numpy as np

from env.battery import BatteryConfig
from env.layout import ROBOT_CAPACITY
from env.rl import ARMS, REWARD_SCALE, mu_filename
from evaluation import Condition
from td import TDAgent, rl_sim_class

M, LAM, T, SEED = 6, 240 / 3600, 3600, 10000
MU = 150.0          # 고정 중심화 기준 (비교가 μ에 흔들리지 않게)


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


class TestR4Wext(unittest.TestCase):

    def test_only_wext_differs_from_a0(self):
        a0 = asdict(ARMS["rsv_local_h3_ro_bat"])
        for w in (2.0, 3.0):
            arm = ARMS[f"rsv_r4_wext{w:g}_bat"]
            d = asdict(arm)
            self.assertEqual({k for k in a0 if a0[k] != d[k]}, {"local_w"})
            self.assertEqual(arm.local_w, (w, 1.0))
            self.assertEqual(mu_filename(arm), f"local_mu_wext{w:g}_wq1_rsv.json")

    def test_cost_scales_term2(self):
        a, _ = _run(ARMS["rsv_local_h3_ro_bat"])
        ca = -np.array(a.reward.finalize()) * REWARD_SCALE
        ext = np.array([e for _, _, e, _ in a.reward._decisions])
        self.assertGreater(ext.sum(), 0.0, "전제: 탑재 토트 지연 추정이 있어야 함")
        for w in (2.0, 3.0):
            b, agent = _run(ARMS[f"rsv_r4_wext{w:g}_bat"])
            self.assertEqual(len(a.reward._decisions), len(b.reward._decisions), "같은 궤적이어야 함")
            fin = b.reward.finalize()
            cb = -np.array(fin) * REWARD_SCALE
            self.assertTrue(np.allclose(cb, ca + (w - 1.0) * ext, atol=1e-6))
            n = len(fin)
            self.assertTrue(np.allclose(agent.replay.r[:n], fin, atol=1e-12), "ring 확정값 = finalize 값")


if __name__ == "__main__":
    unittest.main()
