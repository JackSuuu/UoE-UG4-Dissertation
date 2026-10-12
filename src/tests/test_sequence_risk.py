"""Representation, data isolation and simulator-label contracts."""
import os
import sys
import unittest

import torch

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, 'experiments'))
from models.sequence_risk import SequenceRisk, matched_static_width
from validate_sequence_risk import assert_split_disjoint, candidate_labels
from sims.torch_push import PushSim
from _audit import SCALES


class SequenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)
        self.ctx = {'obs': torch.rand(3, 10), 'obs_prev': torch.rand(3, 10), 'a_prev': torch.rand(3, 2)}
        self.actions = torch.rand(3, 10, 2)

    def test_parameter_match(self):
        for h in (16, 128):
            static = SequenceRisk('static', matched_static_width(h))
            sequence = SequenceRisk('sequence', h)
            counts = [sum(p.numel() for p in m.parameters()) for m in (static, sequence)]
            self.assertLess(abs(counts[0] / counts[1] - 1), 0.05)

    def test_static_is_permutation_equivariant(self):
        m = SequenceRisk('static', 16, E=2)
        torch.testing.assert_close(m(self.ctx, self.actions).flip(2), m(self.ctx, self.actions.flip(1)))

    def test_sequence_is_causal_but_responds_to_preceding_action(self):
        m = SequenceRisk('sequence', 16, E=2)
        changed = self.actions.clone(); changed[:, 5:] += 0.4
        y = m(self.ctx, self.actions)
        torch.testing.assert_close(y[:, :, :5], m(self.ctx, changed)[:, :, :5], rtol=0, atol=0)
        changed = self.actions.clone(); changed[:, 0] += 0.4
        self.assertGreater(float((y[:, :, 1] - m(self.ctx, changed)[:, :, 1]).abs().max().detach()), 1e-5)

    def test_gradient_reaches_earlier_candidate_actions(self):
        m = SequenceRisk('sequence', 16, E=2).eval()
        actions = self.actions.clone().requires_grad_(True)
        m(self.ctx, actions)[:, :, 4].sum().backward()
        self.assertTrue(torch.isfinite(actions.grad).all())
        self.assertGreater(float(actions.grad[:, 0].abs().sum()), 0)
        self.assertEqual(float(actions.grad[:, 5:].abs().sum()), 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_cuda_training_and_eval_action_gradient(self):
        m = SequenceRisk('sequence', 128, E=5).cuda()
        ctx = {k: v.cuda() for k, v in self.ctx.items()}
        optimizer = torch.optim.AdamW(m.parameters())
        loss = m(ctx, self.actions.cuda()).square().mean()
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        m.eval(); a = self.actions.cuda().requires_grad_(True)
        m(ctx, a)[:, :, 4].sum().backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertGreater(float(a.grad[:, 0].abs().sum()), 0)

    def test_normalizer_rejects_held_out_fit(self):
        m = SequenceRisk('sequence', 16)
        with self.assertRaises(ValueError):
            m.fit_normalizers({'split': 'dev', **self.ctx})
        m.fit_normalizers({'split': 'train', **self.ctx})
        torch.testing.assert_close(m.obs_mean, self.ctx['obs'].mean(0))

    def test_split_rejects_shared_episode_despite_distinct_candidates(self):
        with self.assertRaises(ValueError):
            assert_split_disjoint([{'episode_id': torch.tensor([1, 1, 2])}, {'episode_id': torch.tensor([2, 3])}])
        assert_split_disjoint([{'episode_id': torch.tensor([1, 1, 2])}, {'episode_id': torch.tensor([3, 4])}])

    def test_vector_labels_match_serial_and_preserve_input(self):
        sim = PushSim('cpu'); params = sim.make_params(3, {'mass': 2, 'friction': 0.2})
        state = sim.init_state(3, torch.Generator().manual_seed(1)); before = state.clone()
        actions, labels = candidate_labels(sim, state, params, self.actions)
        torch.testing.assert_close(before, state, rtol=0, atol=0)
        for i, scale in enumerate(SCALES):
            s = state.clone(); risks = []
            for j in range(10):
                s, r = sim.step(s, self.actions[:, j] * scale, params); risks.append(r)
            torch.testing.assert_close(labels[:, i], torch.stack(risks, 1), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actions[:, 0], self.actions)


if __name__ == '__main__':
    unittest.main()
