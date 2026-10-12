"""CPU tests for non-invasive offline labels and decision-level diagnostics."""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._audit import SCALES, decision_summary, label_chunk, verify_replay
from sims.base import TorchEnv
from sims.torch_push import PushSim
from models.nets import OrbiSimDynamics


class OfflineLabelTests(unittest.TestCase):
    def test_current_full_chunk_risk_is_action_order_invariant(self):
        torch.manual_seed(1)
        model = OrbiSimDynamics(10, 2, E=2, hidden=8).eval()
        ctx = {'obs': torch.rand(2, 10), 'obs_prev': torch.rand(2, 10),
               'a_prev': torch.rand(2, 2)}
        actions = torch.rand(2, 10, 2)
        with torch.no_grad():
            def score(a):
                mean, std = model.risk(ctx, a)
                return (mean + std).amax((1, 2))
            torch.testing.assert_close(score(actions), score(actions.flip(1)), rtol=0, atol=0)
            torch.testing.assert_close(score(actions), score(actions.roll(5, 1)), rtol=0, atol=0)

    def test_labels_equal_functional_execution_without_mutating_state(self):
        sim = PushSim('cpu')
        env = TorchEnv(sim, 3)
        params = sim.make_params(3, {'friction': 0.2, 'mass': 2.0})
        env.reset(params, seed=12)
        original = env.get_state().clone()
        actions = torch.zeros(3, 10, 2)
        actions[:, :, 0] = 0.3
        risk, goal = label_chunk(sim, original, params, actions, 5)
        torch.testing.assert_close(env.get_state(), original, rtol=0, atol=0)
        observed = []
        for t in range(10):
            obs, r, _ = env.step(actions[:, t])
            observed.append(r.amax(1))
            if t == 4:
                torch.testing.assert_close(goal[:, 0], obs[:, :2].norm(dim=1))
        torch.testing.assert_close(risk, torch.stack(observed, -1), rtol=0, atol=0)
        torch.testing.assert_close(goal[:, 1], obs[:, :2].norm(dim=1))

    def test_equal_prefix_lookahead_horizon(self):
        sim = PushSim('cpu')
        params = sim.make_params(1, {})
        state = sim.init_state(1, torch.Generator().manual_seed(1))
        _, goals = label_chunk(sim, state, params, torch.zeros(1, 5, 2), 5)
        self.assertEqual(tuple(goals.shape), (1, 2))
        torch.testing.assert_close(goals[:, 0], goals[:, 1])


class DiagnosticTests(unittest.TestCase):
    def data(self):
        shape = (2, 2, len(SCALES), 10)
        gt = np.full(shape, 0.2)
        pred = np.full(shape, 0.2)
        gt[0, 0, 0, 2] = 1.5  # missed prefix hazard, safe scales available
        gt[0, 1, 0, 8] = 1.5  # safe prefix but unsafe ten-step lookahead
        applied = gt[:, :, 0].copy()
        applied[0, 1, 3] = 1.7  # triggering introduces a local prefix hazard
        return {'candidate_gt_risk': gt, 'candidate_pred_risk': pred,
                'candidate_gt_goal_dist': np.full((2, 2, len(SCALES), 2), 0.1),
                'candidate_pred_goal_dist': np.full((2, 2, len(SCALES), 2), 0.1),
                'applied_gt_risk': applied, 'applied_pred_risk': pred[:, :, 0],
                'trigger': np.array([[False, True], [True, True]]),
                'first_viol': np.array([2, -1]), 'decision_t': np.array([0, 5]),
                'seated_at_decision': np.array([[False, False], [False, True]])}

    def test_horizon_and_pre_violation_scope(self):
        r = decision_summary(self.data(), 5)
        self.assertEqual(r['n_pre_first_violation_unseated'], 2)
        prefix = r['horizons']['prefix']
        full = r['horizons']['lookahead']
        self.assertEqual(prefix['missed_trigger'], 1)
        self.assertEqual(prefix['missed_unsafe_with_safe_scale_available'], 1)
        self.assertEqual(prefix['safe_proposal_trigger'], 2)
        self.assertEqual(full['true_trigger'], 1)
        self.assertEqual(prefix['triggered_safe_proposal_made_unsafe_locally'], 1)
        self.assertEqual(prefix['triggered_applied_unsafe_with_safe_scale_available'], 1)

    def test_ranking_inversion_count(self):
        d = self.data()
        d['candidate_gt_risk'][0, 0, 1] = 0.7
        d['candidate_pred_risk'][0, 0, 1] = 0.1
        r = decision_summary(d, 5)['horizons']['prefix']
        self.assertGreater(r['risk_ranking_inversions'], 0)
        self.assertGreater(r['risk_ranking_comparable_pairs'], 0)

    def test_order_probe_is_offline_safety_witness(self):
        d = self.data()
        d['order_probe_pred_risk'] = np.full((2, 2, 2, 10), 0.2)
        d['order_probe_gt_risk'] = np.full((2, 2, 2, 10), 0.2)
        r = decision_summary(d, 5)['order_probe']
        self.assertEqual(r['n_pairs'], 6)
        self.assertEqual(r['score_preserved_at_1e_minus_5'], 6)
        self.assertEqual(r['GT_safety_label_changed_with_score_preserved'], 4)

    def test_replay_mismatch_is_fatal(self):
        actual = {'success': np.array([True]), 'violation': np.array([False]),
                  'final_task_cost': np.array([0.01])}
        verify_replay(actual, {'success': [True], 'violation': [False], 'final_task_cost': [0.01]})
        with self.assertRaises(RuntimeError):
            verify_replay(actual, {'success': [False]})
        with self.assertRaises(RuntimeError):
            verify_replay(actual, {'final_task_cost': [1.0]})


if __name__ == '__main__':
    unittest.main()
