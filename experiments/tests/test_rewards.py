"""보상 일관성 테스트 (작업 지시 단계 2·3)

python -m unittest experiments.tests.test_rewards -v
  - order_increment, β = 0: 에피소드 Σ c_k = Σ_o censored F_o (상대 오차 1e-6)
  - little_pbrs (a) c_phi = 0, β = 0: Σ 적분항 = Σ_o censored F_o
  - little_pbrs (b) Φ 항의 에피소드 할인합 = −Φ(s_0) (종단 Φ = 0, telescoping)
학습 전 에이전트(argmax = BL6 prior)로 짧은 에피소드 1회
"""
import math
import unittest


from agents import DQNAgent                  
from evaluation import Condition             
from env.rl import Arm                      
from env.rl import RLSimulator         

M, K, LAM, T, SEED = 14, 3, 320 / 3600, 1800, 10000


def _run(arm: Arm):
    L = Condition().layout(M, K)
    agent = DQNAgent(robot_capacity=5, dist_norm=L.dist_norm, num_robots=M, num_workstations=K,
                     n_step=arm.n_step, features=arm.features)
    sim = RLSimulator(L, agent, lam=LAM, seed=SEED, sim_time=T, num_robots=M, robot_capacity=5,
                      eval_mode=True, arm=arm)
    sim.reset(SEED)
    sim.run()
    censored = sum(min(o.completion_time if o.completion_time is not None else T, T) - o.arrival_time
                   for o in sim.order_mgr.all_orders)
    return sim, censored


class TestOrderIncrement(unittest.TestCase):

    def test_sum_equals_censored_flow(self):
        for order in ("arrival", "bottleneck_first"):
            sim, F = _run(Arm(reward="order_increment", beta=0.0, features="order", tote_order=order))
            rw = sim.reward
            self.assertEqual(rw.unattributed, 0, "첫 결정 전 큐잉 토트가 있으면 등식이 성립하지 않음")
            c, _, _ = rw.costs()
            self.assertGreater(len(c), 100)
            self.assertLess(abs(sum(c) - F) / F, 1e-6, f"{order}: Σc_k {sum(c)} vs ΣF {F}")


class TestLittlePBRS(unittest.TestCase):

    def test_integral_equals_censored_flow(self):
        sim, F = _run(Arm(reward="little_pbrs", beta=0.0, c_phi=0.0, features="order", tote_order="bottleneck_first"))
        A = sum(sim.reward._area)
        self.assertLess(abs(A - F) / F, 1e-6, f"Σ적분 {A} vs ΣF {F}")

    def test_potential_telescopes(self):
        arm = Arm(reward="little_pbrs", c_phi=1.0, features="order", tote_order="bottleneck_first")
        sim, _ = _run(arm)
        rw, b = sim.reward, arm.beta_value
        sh = rw.shaping_terms()
        disc, acc = 1.0, 0.0
        times, phi = rw._times + [T], rw._phi + [0.0]
        self.assertGreater(max(abs(p) for p in phi), 0.0)
        for k, s in enumerate(sh):
            acc += disc * s
            disc *= math.exp(-b * (times[k + 1] - times[k]))
            # 부분합: Σ_{i≤k} D_i sh_i = D_{k+1} Φ_{k+1} − Φ_0 (D_i = Π_{j<i} e^{−βτ_j})
            self.assertAlmostEqual(acc, disc * phi[k + 1] - phi[0], places=9)
        self.assertAlmostEqual(acc, -phi[0], places=9)      # 종단 Φ = 0


if __name__ == "__main__":
    unittest.main()
