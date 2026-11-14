"""RL 예약 행동 (env.rl.RLReserveSimulator, td.TDReserveSimulator)과 절단·이동거리 비용 옵션 테스트

python -m unittest experiments.tests.test_rl_reserve -v
  - 학습 전 정책(BL6 잔차 f = 0, ε = 0)이 BL6R (c_d = 1)과 똑같이 동작 (예약 후보 prior = env.reserve.reserve_cost_terms)
  - 예약 결정이 실제로 일어나고 전이 key 차원이 에이전트와 맞음 (배터리 SoC 특징 포함)
  - truncate: 마지막 결정을 버리고 terminal 없음, n-step 버퍼를 에피소드마다 비움
  - w_dist: 결정 비용에 Δd · w_dist가 더해짐
"""
import unittest

import numpy as np

from agents import DQNAgent
from baselines import resolve_policy
from env.battery import BatteryConfig
from env.layout import ROBOT_CAPACITY
from env.rl import ARMS, REWARD_SCALE
from evaluation import Condition
from td import TDAgent, rl_sim_class

M, LAM, T, SEED = 6, 240, 3600, 10000


def _agent(L, arm):
    kw = dict(robot_capacity=ROBOT_CAPACITY, dist_norm=L.dist_norm, seed=1, sim_time=T, num_robots=M,
              num_workstations=3, n_step=arm.n_step, features=arm.features)
    return TDAgent(beta=arm.beta_value, **kw) if arm.buffer == "ring" else DQNAgent(**kw)


def _rl(name, eval_mode=True, battery=None):
    arm = ARMS[name]
    L = Condition().layout(M, 3)
    agent = _agent(L, arm)
    sim = rl_sim_class(arm, train=not eval_mode)(L, agent, lam=LAM / 3600, seed=SEED, sim_time=T, num_robots=M,
                                                 robot_capacity=L.robot_capacity, arm=arm, eval_mode=eval_mode,
                                                 battery=battery)
    sim.reset(SEED)
    out = sim.run()
    return sim, agent, out


class TestRLReserve(unittest.TestCase):

    def test_untrained_equals_bl6r(self):
        sim, _, _ = _rl("rsv_local")
        L = Condition().layout(M, 3)
        ref = resolve_policy("BL6Rc1")[1](L, lam=LAM / 3600, seed=SEED, sim_time=T, num_robots=M,
                                          robot_capacity=L.robot_capacity)
        ref.reset(SEED)
        ref.run()
        self.assertGreater(sim.n_reserved, 20, "전제: 예약 결정이 있어야 함")
        self.assertEqual(sim.n_reserved, ref.n_reserved)
        a, b = sim.build_result("x", 0, sim.lam), ref.build_result("x", 0, ref.lam)
        self.assertEqual(a.avg_flow_time, b.avg_flow_time)
        self.assertEqual(a.dist_per_order, b.dist_per_order)

    def test_truncate_and_dims(self):
        sim, agent, trans = _rl("rsv_local_h3", eval_mode=False)
        self.assertEqual(len(trans), sim._n_dec - 1)
        self.assertFalse(any(d for _, _, _, d, _ in trans))
        self.assertTrue(all(nk is not None for _, _, nk, _, _ in trans))
        pad, mask, _, R, W = trans[0][0]
        self.assertEqual(pad.shape[1], agent.feat_dim)
        self.assertEqual(W.shape[0], agent.ctx_dim)
        for t in trans:
            agent.push(*t)
        agent.flush_nstep()
        self.assertEqual(len(agent.nstep_buf.buf), 0)
        self.assertEqual(len(agent.buffer), len(trans))

    def test_w_dist(self):
        a, _, ta = _rl("rsv_local_h3", eval_mode=False)
        b, _, tb = _rl("rsv_local_h3_wd1", eval_mode=False)
        d = [x[3] for x in b.reward._decisions]
        diff = [(ra - rb) * REWARD_SCALE for (_, ra, *_), (_, rb, *_) in zip(ta, tb)]
        mu_shift = (b.reward.last_mu - a.reward.last_mu)      # μ가 없으면 에피소드 평균이라 w_dist 평균만큼 이동
        self.assertTrue(np.allclose(diff, [x - mu_shift for x in d[:len(diff)]], atol=1e-6))

    def test_td_reserve_battery(self):
        cfg = BatteryConfig(range_m=1500.0, charge_rate=3.0, reserve_m=30.0, low_soc=0.25, target_soc=1.0)
        sim, agent, _ = _rl("rsv_td_h3_bat", eval_mode=False, battery=cfg)
        res = sim.build_result("x", 0, sim.lam)
        self.assertGreater(sim.n_reserved, 10)
        self.assertGreater(res.n_charge_required, 0)
        self.assertEqual(res.n_soc_depleted, 0)
        self.assertEqual(agent.replay.cand.shape[2], agent.feat_dim)
        self.assertEqual(agent.replay.W.shape[1], agent.ctx_dim)
        self.assertGreater(sim.n_transitions, 100)


if __name__ == "__main__":
    unittest.main()
