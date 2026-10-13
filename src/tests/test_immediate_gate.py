"""Causal boundaries of the immediate-gate objective experiment."""
import os
import sys
import unittest
import torch

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, 'experiments'))
from models.immediate_gate import ImmediateGate
from diagnose_risk_controller import GateRepairVerifier
from validate_immediate_gate import RootBalancedSampler, r5_labels


class Risk:
    differentiable = False

    def __init__(self, scale):
        self.scale, self.queries = scale, []

    def risk(self, ctx, actions):
        self.queries.append(actions.clone())
        mean = actions.abs().sum(-1, keepdim=True) * self.scale
        return mean, torch.zeros_like(mean)


class ImmediateGateTests(unittest.TestCase):
    def context(self):
        return {'obs': torch.randn(4, 10), 'obs_prev': torch.randn(4, 10),
                'a_prev': torch.randn(4, 2)}

    def test_r5_suffix_and_hidden_physics_cannot_change_gate(self):
        ctx, a = self.context(), torch.randn(4, 10, 2)
        m = ImmediateGate('event', hidden=8)
        before = m.risk(ctx, a)
        changed = a.clone(); changed[:, 5:] *= 100
        ctx['mass'], ctx['friction'], ctx['state'] = torch.randn(4), torch.randn(4), torch.randn(4, 10)
        after = m.risk(ctx, changed)
        for x, y in zip(before, after):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertEqual(before[0].shape, (4, 5, 1))

    def test_regression_matches_legacy_risk_computation(self):
        m = ImmediateGate('regression', hidden=8)
        ctx, a = self.context(), torch.randn(4, 10, 2)
        _, legacy = m.rollout(ctx['obs'], ctx['obs_prev'], ctx['a_prev'], a[:, :5])
        mean, std = m.risk(ctx, a)
        torch.testing.assert_close(mean[..., 0], legacy[..., 0].mean(0))
        torch.testing.assert_close(std[..., 0], legacy[..., 0].std(0))

    def test_frozen_repair_is_independent_of_gate_score_units(self):
        ctx, a = self.context(), torch.full((4, 10, 2), .4)
        outputs = []
        for scale in (10., 100.):
            gate, repair = Risk(scale), Risk(2.)
            verifier = GateRepairVerifier(gate, repair, .5, .5)
            mask, _ = verifier.check(ctx, a)
            self.assertTrue(mask.all())
            outputs.append(verifier.repair(ctx, a))
            self.assertEqual(len(gate.queries), 1)
            self.assertEqual(len(repair.queries), 8)
            self.assertTrue(all(q.shape[1] == 10 for q in repair.queries))
        torch.testing.assert_close(*outputs, rtol=0, atol=0)

    def test_normalizers_reject_dev_and_post_violation_rows(self):
        m = ImmediateGate('event', hidden=8)
        ctx = self.context()
        for split, prior in (('dev', False), ('train', True)):
            with self.assertRaises(ValueError):
                m.fit_normalizers({**ctx, 'split': split, 'prior_violation': torch.full((4,), prior)})

    def test_event_boundary_and_shared_root_balanced_draws(self):
        risk = torch.zeros(8, 1, 10, 2)
        risk[:4, 0, 4, 0] = 1.01
        risk[4:, 0, 5:, 0] = 2  # H10-only positives are R5 negatives.
        risk[4:, 0, :5, 0] = 1  # Strict >1 event, not >=1.
        data = {'risk': risk, 'episode_id': torch.tensor([0, 0, 1, 2, 0, 1, 1, 3])}
        left, right = RootBalancedSampler(data, 1), RootBalancedSampler(data, 1)
        for _ in range(5):
            a, b = left.sample(256), right.sample(256)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(int(r5_labels(data)[a].sum()), 128)


if __name__ == '__main__':
    unittest.main()
