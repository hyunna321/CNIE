"""R2 보상 항 ablation arm (env.rl.ARMS rsv_r2_ord*_bat) 테스트

python -m unittest experiments.tests.test_r2_ablation -v
  - 각 arm은 Full(rsv_r2_ord_eta0.2_bat)과 보상 계수(local_w = (w_ext, w_q), eta)만 다름, μ 표 파일도 서로 다름
  - 같은 시나리오·같은 정책(학습 전, 업데이트 없음)에서 c_k가 항을 뺀 만큼 정확히 줄어듦
      c_k = inc + w_q · 큐 inc + η · 자기 지연 + w_ext · (E − X)
"""
import unittest
from dataclasses import asdict

import numpy as np

from env.rl import ARMS, mu_filename
from experiments.tests.test_order_ext import T, _parts, _run

FULL = "rsv_r2_ord_eta0.2_bat"
ABL = {   # arm: (w_ext, w_q, η)
    "rsv_r2_ord_bat":              (1.0, 1.0, 0.0),
    "rsv_r2_ord_eta0.2_noext_bat": (0.0, 1.0, 0.2),
    "rsv_r2_ord_eta0.2_noq_bat":   (1.0, 0.0, 0.2),
    "rsv_r2_ord_noext_bat":        (0.0, 1.0, 0.0),
    "rsv_r2_ord_inconly_bat":      (0.0, 0.0, 0.0),
}


def _queue_inc(sim):
    """결정별 큐 토트 inc 합 (w_q 곱하기 전)"""
    rw = sim.reward
    return [sum(rw._inc_of(x, T) for x in rw._attr[k][1:]) for k in range(len(rw._times))]


class TestR2Ablation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.full, _ = _run(ARMS[FULL])
        cls.pf = _parts(cls.full)
        cls.qf = _queue_inc(cls.full)
        cls.netf = [cls.full.reward._ext[k] - cls.full.reward._recv[k] for k in range(len(cls.pf))]

    def test_only_reward_coefs_differ(self):
        full = asdict(ARMS[FULL])
        names = {mu_filename(ARMS[FULL])}
        for name, (w_ext, w_q, eta) in ABL.items():
            a = ARMS[name]
            self.assertEqual((a.local_w, a.eta), ((w_ext, w_q), eta), name)
            d = asdict(a)
            self.assertLessEqual({k for k in full if full[k] != d[k]}, {"local_w", "eta"}, name)
            names.add(mu_filename(a))
        self.assertEqual(len(names), len(ABL) + 1, "arm마다 μ 표가 달라야 함")

    def test_cost_drops_exactly_the_removed_terms(self):
        self.assertGreater(sum(self.qf), 0.0, "전제: 큐 토트 inc가 있어야 함")
        self.assertGreater(sum(abs(x) for x in self.netf), 0.0, "전제: 외부비용이 있어야 함")
        for name, (w_ext, w_q, eta) in ABL.items():
            sim, _ = _run(ARMS[name])
            p = _parts(sim)
            self.assertEqual(len(p), len(self.pf), name)
            # 같은 궤적: 자기 토트 inc + 큐 inc(계수 1), 자기 지연이 Full과 같음
            self.assertTrue(np.allclose([x[3] for x in p], [y[3] for y in self.pf]), name)
            want = [y[0] - (1 - w_q) * q - (0.2 - eta) * y[3] - (1 - w_ext) * n
                    for y, q, n in zip(self.pf, self.qf, self.netf)]
            self.assertTrue(np.allclose([x[0] for x in p], want, atol=1e-6), name)
            if w_ext == 0.0:
                self.assertEqual(sum(abs(e) for e in sim.reward._ext), 0.0, name)


if __name__ == "__main__":
    unittest.main()
