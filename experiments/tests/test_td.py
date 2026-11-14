"""관측 기반 TD 학습 (td.py, env.rl의 interval 보상) 단위 테스트

python -m unittest experiments.tests.test_td -v
  - Little 항등식: β = 0, ρ = 0이면 Σ 구간 적분 = 도착 주문의 censored 흐름시간 합
  - 중심화 불변: ρ가 바꾸는 할인 보상 합 = 궤적과 무관한 상수
  - ring buffer: n-step 값과 부트스트랩 할인이 손 계산과 같고, 에피소드 경계·미확정 칸을 넘지 않으며 wrap 후에도 연속
  - 저장 일치: 버퍼의 (s, a)가 실제 선택에 쓴 값과 같음
  - toy MDP: 값을 아는 결정론 MDP에서 Q가 수렴
  - 세 arm이 학습 경로에서 동작
"""
import math
import random
import unittest

import numpy as np

from agents import DQNAgent
from env.layout import ROBOT_CAPACITY
from env.rl import ARMS, Arm, RLSimulator
from evaluation import Condition
from td import RingReplay, TDAgent, TDSimulator


def _agent(L, M, arm, cls=DQNAgent, **kw):
    extra = {"beta": arm.beta_value} if cls is TDAgent else {}
    return cls(robot_capacity=ROBOT_CAPACITY, dist_norm=L.dist_norm, seed=1, sim_time=3600, num_robots=M,
               num_workstations=3, n_step=arm.n_step, features=arm.features, **extra, **kw)


def _sim(arm, M=6, lam=240, sim_time=3600, mu=None, eval_mode=True):
    L = Condition().layout(M, 3)
    agent = _agent(L, M, arm)
    agent.mu_table = mu
    sim = RLSimulator(L, agent, lam=lam / 3600, seed=10000, sim_time=sim_time, num_robots=M,
                      robot_capacity=L.robot_capacity, arm=arm, eval_mode=eval_mode)
    sim.reset(10000)
    sim.run()
    return sim


def _key(v, a=0):
    return (np.full((3, 2), float(v), np.float32), np.ones(3, np.bool_), a,
            np.full((2, 2), float(v), np.float32), np.full(2, float(v), np.float32))


class TestReward(unittest.TestCase):

    def test_little_identity(self):
        sim = _sim(Arm(reward="interval", beta=0.0))
        om = sim.order_mgr
        self.assertAlmostEqual(sum(sim.reward._area), om.censored_flow_time(sim.sim_time) * len(om.all_orders), places=4)

    def test_centering_invariant(self):
        arm, rho = Arm(reward="interval"), 3.7
        lam = 240 / 3600
        a, b = _sim(arm), _sim(arm, mu=([lam], [rho]))
        ts = np.array(a.reward._times)
        self.assertTrue(np.array_equal(ts, np.array(b.reward._times)), "ρ가 정책을 바꿈")
        beta, t0 = arm.beta_value, ts[0]
        diff = sum(math.exp(-beta * (t - t0)) * (x - y) for t, x, y in zip(ts, b.reward._area, a.reward._area))
        expect = -rho * (1 - math.exp(-beta * (a.sim_time - t0))) / beta
        self.assertAlmostEqual(diff, expect, places=4)


class TestRing(unittest.TestCase):

    def _fill(self, cap=10, n=2, beta=0.1, seq=((1.0, 2.0), (2.0, 1.0), (3.0, 4.0), (4.0, 0.5))):
        rp = RingReplay(cap, n, beta, seed=1)
        rp.new_segment()
        for i, (r, tau) in enumerate(seq):
            rp.open(_key(i))
            rp.close(r, tau)
        rp.open(_key(len(seq)))
        return rp

    def test_nstep_value(self):
        rp, beta = self._fill(), 0.1
        b = rp._batch(np.array([0]))
        self.assertAlmostEqual(float(b["G"][0]), 1.0 + math.exp(-beta * 2.0) * 2.0, places=5)
        self.assertAlmostEqual(float(b["boot"][0]), math.exp(-beta * 3.0), places=5)
        self.assertEqual(float(b["next_W"][0, 0]), 2.0)
        self.assertEqual(float(b["W"][0, 0]), 0.0)

    def test_validity(self):
        rp = self._fill()
        self.assertEqual(rp._valid(np.arange(5)).tolist(), [True, True, True, False, False])
        rp.new_segment()
        rp.open(_key(9)); rp.close(0.0, 1.0); rp.open(_key(10)); rp.close(0.0, 1.0); rp.open(_key(11))
        self.assertEqual(rp._valid(np.arange(8)).tolist(), [True, True, True, False, False, True, False, False])

    def test_wrap(self):
        rp = RingReplay(5, 2, 0.0, seed=1)
        rp.new_segment()
        for i in range(8):                      # 칸 0..2가 덮어쓰임: seq 3,4(칸 3,4) 5,6,7(칸 0,1,2)
            rp.open(_key(i)); rp.close(float(i), 1.0)
        rp.open(_key(8))                        # 칸 3 (seq 8)
        # 칸 0 (seq 5~7), 칸 1 (seq 6~8) 유효, 칸 2 (seq 7, 8 뒤가 끊김), 칸 3 (미확정) 무효, 칸 4 (seq 4~6) 유효
        self.assertEqual(rp._valid(np.arange(5)).tolist(), [True, True, False, False, True])
        b = rp._batch(np.array([4]))            # seq 4 → 다음 칸은 wrap되어 seq 5, 6
        self.assertAlmostEqual(float(b["G"][0]), 4.0 + 5.0, places=5)
        self.assertEqual(float(b["next_W"][0, 0]), 6.0)

    def test_stored_equals_selected(self):
        arm = ARMS["td_a2_buffer"]
        L = Condition().layout(6, 3)
        agent = _agent(L, 6, arm, TDAgent)
        seen, orig = [], agent.select_index

        def spy(feats):
            i = orig(feats)
            seen.append(tuple(x.copy() if isinstance(x, np.ndarray) else x for x in agent.last_key))
            return i
        agent.select_index = spy
        sim = TDSimulator(L, agent, lam=240 / 3600, seed=10000, sim_time=1800, num_robots=6,
                          robot_capacity=L.robot_capacity, arm=arm)
        sim.reset(10000)
        self.assertEqual(sim.run(), [])
        rp = agent.replay
        self.assertEqual(rp.count, len(seen))
        self.assertEqual(sim.n_transitions, len(seen) - 1)
        for i, k in enumerate(seen):
            self.assertTrue(np.array_equal(rp.cand[i], k[0]) and np.array_equal(rp.R[i], k[3]) and np.array_equal(rp.W[i], k[4]))
            self.assertEqual(rp.act[i], k[2])
        self.assertFalse(rp.closed[len(seen) - 1], "마지막 결정은 확정되면 안 됨")


    def test_state_roundtrip(self):
        rp = self._fill()
        s = rp.state()
        other = RingReplay(10, 2, 0.1, seed=99)
        other.load_state({**s, "arrays": {k: v.copy() for k, v in s["arrays"].items()}})
        a, b = rp.sample(2) if len(rp) > 100 else rp._batch(np.array([0, 1])), other._batch(np.array([0, 1]))
        for k in a:
            self.assertTrue(np.array_equal(a[k], b[k]), k)
        self.assertEqual((rp.head, rp.count, rp.seg_id), (other.head, other.count, other.seg_id))
        self.assertEqual(rp.rng.bit_generator.state, other.rng.bit_generator.state)


class TestLocalRing(unittest.TestCase):
    """A0 (local 보상) + ring buffer: 지연 확정한 보상 = 에피소드 후 finalize() 값"""

    def _run(self, arm, battery):
        from env.battery import BatteryConfig
        from td import rl_sim_class
        M, lam = 6, 240 / 3600
        L = Condition().layout(M, 3)
        agent = _agent(L, M, arm, cls=TDAgent)
        agent.mu_table = ([lam], [150.0])
        sim = rl_sim_class(arm)(L, agent, lam=lam, seed=10000, sim_time=3600, num_robots=M,
                                robot_capacity=L.robot_capacity, arm=arm, battery=BatteryConfig() if battery else None)
        sim.update_every = 10 ** 9            # 업데이트 없이 수집만 (같은 정책으로 비교)
        sim.reset(10000)
        sim.run()
        return sim, agent

    def test_rewards_match_finalize(self):
        for name, bat in (("rsv_local_h3_ro", False), ("rsv_local_h3_ro_bat", True)):
            sim, agent = self._run(ARMS[name], bat)
            fin = sim.reward.finalize()
            rp, n = agent.replay, len(fin)
            self.assertGreater(n, 100)
            self.assertTrue(np.allclose(rp.r[:n], fin, atol=1e-12), name)
            self.assertTrue(rp.has_r[:n].all(), name)
            self.assertTrue(rp.closed[:n - 1].all() and not rp.closed[n - 1], name)   # 마지막 결정은 τ 없음 (절단)
            self.assertEqual(sim.n_transitions, n - 1)


class TestLearning(unittest.TestCase):

    def test_toy_mdp(self):
        """A: a0 → B (r 0), a1 → A (r −0.5) / B: a0 → A (r 1), a1 → B (r 0), 할인 0.9
        V(A) = γ/(1−γ²), V(B) = 1 + γ·V(A)"""
        gamma, beta = 0.9, -math.log(0.9)
        L = Condition().layout(6, 3)
        arm = ARMS["td_a2_buffer"]
        ag = _agent(L, 6, Arm(reward="interval", buffer="ring", n_step=1), TDAgent, lr=2e-3, target_update=50,
                    buffer_size=4000)
        ag.replay.beta = beta
        F, C = ag.feat_dim, ag.max_candidates
        R0, W0 = np.zeros((6, 10), np.float32), np.zeros(ag.ctx_dim, np.float32)

        def key(s, a):
            pad = np.zeros((C, F), np.float32)
            pad[0, 0], pad[0, 1] = s, 0.0
            pad[1, 0], pad[1, 1] = s, 1.0
            mask = np.zeros(C, np.bool_)
            mask[:2] = True
            return pad, mask, a, R0, W0

        rng = random.Random(1)
        s = 0
        ag.replay.new_segment()
        for _ in range(3000):
            a = rng.randrange(2)
            ag.replay.open(key(s, a))
            r, ns = {(0, 0): (0.0, 1), (0, 1): (-0.5, 0), (1, 0): (1.0, 0), (1, 1): (0.0, 1)}[(s, a)]
            ag.replay.close(r, 1.0)
            s = ns
        ag.replay.open(key(s, 0))
        for _ in range(6000):
            ag.train_step()
        import torch
        Q = {}
        for st in (0, 1):
            pad = key(st, 0)[0]
            q = ag.policy_net(torch.tensor(pad)[None], torch.tensor(R0)[None], torch.tensor(W0)[None])[0].detach().numpy()
            Q[st] = q[:2]
        va = gamma / (1 - gamma ** 2)
        vb = 1 + gamma * va
        for got, want in ((Q[0][0], gamma * vb), (Q[0][1], -0.5 + gamma * va), (Q[1][0], 1 + gamma * va), (Q[1][1], gamma * vb)):
            self.assertAlmostEqual(float(got), want, delta=0.35)
        self.assertGreater(Q[0][0], Q[0][1])
        self.assertGreater(Q[1][0], Q[1][1])

    def test_arms_run(self):
        for name, expect_online in (("td_a2_buffer", False), ("td_a3_online", True)):
            arm = ARMS[name]
            L = Condition().layout(14, 3)
            agent = _agent(L, 14, arm, TDAgent)
            sim = TDSimulator(L, agent, lam=240 / 3600, seed=10000, sim_time=3600, num_robots=14,
                              robot_capacity=L.robot_capacity, arm=arm)
            sim.update_every = 2
            sim.reset(10000)
            sim.run()
            self.assertGreater(sim.n_transitions, 200)
            self.assertEqual(sim.n_trained > 0, expect_online)
            self.assertEqual(agent.steps > 0, expect_online)
        s = _sim(ARMS["td_a1_reward"], eval_mode=False)
        self.assertEqual(len(s._transitions), len(s.reward._area))


if __name__ == "__main__":
    unittest.main()
