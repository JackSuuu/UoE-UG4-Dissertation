"""CPU tests for fixed-budget scale selection, abstention and pairing."""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from checkvla.reference import RefCheckVLA, candidate_repair
from experiments._paired import bootstrap_delta, paired_outcomes, pooled_paired
from adapters.openvla_policy import OpenVLAPolicy


class Predictor:
    def __init__(self, kind='normal'):
        self.kind, self.risk_calls, self.rollout_calls = kind, 0, 0

    def risk(self, ctx, actions):
        self.risk_calls += 1
        scale = actions[:, -1, 0]
        r = scale.clone()
        # Deliberate nonmonotonic risk: 0.85 is safe but 0.70 is not.
        r = torch.where((scale - 0.85).abs() < 1e-5, 0.5, r)
        r = torch.where((scale - 0.7).abs() < 1e-5, 0.9, r)
        if self.kind == 'flat':
            r = torch.ones_like(scale)
        if self.kind == 'none_safe':
            r = 0.8 + 0.2 * scale
        if self.kind == 'nan':
            r = torch.full_like(scale, float('nan'))
        risk = r[:, None, None].expand(-1, actions.shape[1], 2)
        return risk, torch.zeros_like(risk)

    def rollout(self, obs, obs_prev, a_prev, actions):
        self.rollout_calls += 1
        o = torch.zeros(2, actions.shape[0], actions.shape[1], 4)
        # Short-horizon progress peaks at 0.4; H10 progress peaks at 0.15.
        o[:, :, :, 0] = (actions[:, -1, 0] - 0.4).abs()[None, :, None]
        o[:, :, -1, 0] = (actions[:, -1, 0] - 0.15).abs()
        return o, None


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.chunk = torch.ones(2, 10, 2)
        self.ctx = {'obs': torch.zeros(2, 4), 'obs_prev': torch.zeros(2, 4),
                    'a_prev': torch.zeros(2, 2)}

    def test_discrete_finds_nonmonotonic_safe_minimal_change(self):
        p = Predictor()
        out, ok = candidate_repair(p, self.ctx, self.chunk)
        torch.testing.assert_close(out, self.chunk * 0.85)
        self.assertTrue(ok.all())
        self.assertEqual(p.risk_calls, 8)
        self.assertEqual(p.rollout_calls, 0)

    def test_progress_uses_execution_not_lookahead_horizon(self):
        p = Predictor()
        out, ok = candidate_repair(p, self.ctx, self.chunk, mode='progress', execution_horizon=5)
        torch.testing.assert_close(out, self.chunk * 0.4)
        self.assertTrue(ok.all())
        self.assertEqual((p.risk_calls, p.rollout_calls), (8, 8))

    def test_no_accepted_candidate_returns_original(self):
        for kind in ('flat', 'none_safe', 'nan'):
            with self.subTest(kind=kind):
                out, ok = candidate_repair(Predictor(kind), self.ctx, self.chunk)
                torch.testing.assert_close(out, self.chunk)
                self.assertFalse(ok.any())

    def test_hard_prefix_preserved(self):
        out, _ = candidate_repair(Predictor(), self.ctx, self.chunk, hard_prefix=True, latency=2)
        torch.testing.assert_close(out[:, :2], self.chunk[:, :2])
        torch.testing.assert_close(out[:, 2:], self.chunk[:, 2:] * 0.85)

    def test_abstention_counter(self):
        ver = RefCheckVLA(Predictor('flat'), repair_mode='discrete')
        torch.testing.assert_close(ver.repair(self.ctx, self.chunk), self.chunk)
        self.assertEqual(ver.n_abstain, 2)

    def test_invalid_configuration(self):
        for scales in ((0.5, 0.1), (1, 0), (1, 0.1, 0.2)):
            with self.assertRaises(ValueError):
                candidate_repair(Predictor(), self.ctx, self.chunk, scales=scales)
        with self.assertRaises(ValueError):
            candidate_repair(Predictor(), self.ctx, self.chunk, execution_horizon=11)
        with self.assertRaises(ValueError):
            RefCheckVLA(Predictor(), repair_mode='progress', use_grad=True)


class PairedTests(unittest.TestCase):
    def test_rescue_harm_are_episode_paired(self):
        b = {'success': [True, False, True, False], 'violation': [False, True, True, False]}
        m = {'success': [False, True, True, False], 'violation': [False, False, False, False]}
        p = paired_outcomes(b, m)
        self.assertEqual((p['safe_rescue_count'], p['safe_harm_count']), (2, 1))
        self.assertEqual(p['delta_safe_success'], 0.25)
        self.assertEqual(p['delta_CVR'], -0.5)
        pooled = pooled_paired([{'ood': True, 'episodes': {'none': b, 'repair': m}}], 'repair')
        self.assertEqual(pooled['delta_safe_success'], 0.25)

    def test_bootstrap_zero_and_fixed_cell_weight(self):
        self.assertEqual(bootstrap_delta([np.zeros(8)])['ci95'], [0, 0])
        self.assertEqual(bootstrap_delta([np.ones(2), -np.ones(8)])['estimate'], 0)
        self.assertEqual(bootstrap_delta([np.ones(8)])['ci95'], [1, 1])

    def test_bad_pairs(self):
        with self.assertRaises(ValueError):
            paired_outcomes({'success': [True], 'violation': [False]},
                            {'success': [], 'violation': []})


class MicrobatchTests(unittest.TestCase):
    def test_microbatch_preserves_actions_and_order(self):
        p = object.__new__(OpenVLAPolicy)
        p.action_mode = 'planar_head'
        p.mu, p.sd, p.H = torch.zeros(2), torch.ones(2), 10
        p.sim = type('Sim', (), {'max_vel': 1.0})()
        p.head = lambda x: x[:, None].expand(-1, 10, -1)
        p.features = lambda img: img.flatten(1)[:, :2]
        img = torch.rand(17, 3, 2, 2)
        p.inference_batch_size = 0
        full = p(torch.zeros(17, 10), img)
        p.inference_batch_size = 8
        micro = p(torch.zeros(17, 10), img)
        torch.testing.assert_close(full, micro, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
