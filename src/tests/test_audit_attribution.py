"""Synthetic tests for local witnesses, not claims of causal attribution."""
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments'))
from experiments.analyze_verifier_audit import episode_witness, local_counts, scope_mask
from audit_intervention_branch import run_branch, skip_plan
from checkvla.verifier import Controller


def fixture():
    shape = (2, 2, 8, 10)
    gt = np.full(shape, 0.2)
    return {
        'candidate_gt_risk': gt, 'candidate_pred_risk': np.full(shape, 0.1),
        'applied_gt_risk': np.full((2, 2, 10), 0.2),
        'candidate_pred_goal_dist': np.full((2, 2, 8, 2), 0.1),
        'candidate_gt_goal_dist': np.full((2, 2, 8, 2), 0.1),
        'applied_gt_goal_dist': np.full((2, 2, 2), 0.1),
        'trigger': np.zeros((2, 2), bool), 'first_viol': np.array([-1, 7]),
        'decision_t': np.array([0, 5]), 'seated_at_decision': np.zeros((2, 2), bool),
        'proposal': np.zeros((2, 2, 10, 2)), 'applied': np.zeros((2, 2, 10, 2)),
        'state': np.zeros((2, 2, 10)), 'success': np.array([True, True]),
        'violation': np.array([False, True]), 'final_task_cost': np.zeros(2),
    }


class AttributionTests(unittest.TestCase):
    def test_scope_includes_violating_decision_not_later(self):
        d = fixture(); d['first_viol'][1] = 2
        np.testing.assert_array_equal(scope_mask(d), [[True, True], [True, False]])

    def test_miss_and_candidate_insufficiency_are_separate(self):
        d = fixture(); d['candidate_gt_risk'][0, 0, :, 2] = 1.5
        d['candidate_gt_risk'][0, 1, 0, 2] = 1.5
        r = local_counts(d, scope_mask(d), 5, 0)
        self.assertEqual(r['missed'], 2)
        self.assertEqual(r['missed_with_safe_scale'], 1)
        self.assertEqual(r['missed_without_safe_scale'], 1)

    def test_prefix_and_counterfactual_horizon_differ(self):
        d = fixture(); d['candidate_gt_risk'][0, 0, 0, 8] = 1.5
        self.assertEqual(local_counts(d, scope_mask(d), 5, 0)['unsafe_proposals'], 0)
        self.assertEqual(local_counts(d, scope_mask(d), 10, 1)['unsafe_proposals'], 1)

    def test_harm_can_be_later_safety_loss_without_success_loss(self):
        d = fixture(); b = fixture(); b['violation'][1] = False
        d['applied'][0, 1, 0, 0] = 0.1
        d['candidate_gt_risk'][1, 1, :, 2] = 1.5
        r = episode_witness(d, b, 1)
        self.assertTrue(r['safe_harm']); self.assertTrue(r['harm_safety_loss'])
        self.assertFalse(r['harm_success_loss'])
        self.assertLess(r['first_divergence']['applied_prefix_risk'], 1)
        self.assertEqual(r['first_violation']['category'], 'missed_without_safe_scale')

    def test_changed_suffix_is_not_executed_divergence(self):
        d = fixture(); b = fixture(); d['applied'][0, 0, 8, 0] = 0.1
        self.assertIsNone(episode_witness(d, b, 0)['first_divergence'])

    def test_skip_updates_current_action_plan_and_history_only_selected_row(self):
        proposal = torch.arange(40).float().reshape(2, 10, 2)
        ctl = SimpleNamespace(plan=proposal * 0.5, a_prev=proposal[:, 0] * 0.5,
                              plan_ptr=torch.ones(2, dtype=torch.long),
                              plan_left=torch.full((2,), 4), obs_prev=torch.ones(2, 10))
        old_history = ctl.obs_prev.clone()
        action = ctl.a_prev.clone()
        skip_plan(ctl, action, proposal, torch.tensor([1]))
        torch.testing.assert_close(ctl.plan[0], proposal[0] * 0.5)
        torch.testing.assert_close(ctl.plan[1], proposal[1])
        torch.testing.assert_close(action[1], proposal[1, 0])
        torch.testing.assert_close(ctl.a_prev, action)
        torch.testing.assert_close(ctl.obs_prev, old_history)
        self.assertEqual(ctl.plan_ptr.tolist(), [1, 1])
        self.assertEqual(ctl.plan_left.tolist(), [4, 4])

    def test_whole_branch_preserves_other_rows_and_skips_once(self):
        class Policy:
            H, act_dim = 10, 2
            needs_img = needs_rgb = False

            def __call__(self, obs, img=None, instruction=None):
                return torch.full((obs.shape[0], 10, 2), 0.1)

        class Verifier:
            commit = 5
            predictor = SimpleNamespace(needs_img=False)

            def check(self, ctx, chunk):
                return torch.ones(len(chunk), dtype=torch.bool), torch.ones(len(chunk))

            def repair(self, ctx, chunk):
                return chunk * 0.5

        class Env:
            def reset(self, params=None, seed=0):
                self.state = torch.zeros(2, 10)
                return self.state

            def get_state(self):
                return self.state

            def step(self, action):
                self.state = self.state.clone()
                self.state[:, :2] += action
                return self.state, torch.zeros(2, 2), torch.ones(2, dtype=torch.bool)

        env = Env(); sim = SimpleNamespace(T=80, task_cost=lambda s: s[:, :2].square().sum(-1))
        ctl = Controller(Policy(), sim, 'checkvla', Verifier(), chunk_k=5)
        obs = env.reset(); ctl.reset(env, obs)
        reference = {k: [] for k in ('state', 'proposal', 'applied')}
        for t in range(80):
            if t % 5 == 0:
                reference['state'].append(env.state.numpy().copy())
            a, info = ctl.act(env, obs)
            if t % 5 == 0:
                reference['proposal'].append(info['chunk'].numpy().copy())
                reference['applied'].append(ctl.plan.numpy().copy())
            obs, _, _ = env.step(a)
        reference = {k: np.stack(v) for k, v in reference.items()}
        reference.update(decision_t=np.arange(0, 80, 5), success=np.ones(2, bool),
                         violation=np.zeros(2, bool), first_viol=np.full(2, -1),
                         first_trig=np.zeros(2, int), first_success=np.zeros(2, int),
                         n_interventions=np.full(2, 16), final_task_cost=sim.task_cost(env.state).numpy())
        control = run_branch(env, sim, ctl, None, 0, reference, np.full(2, -1))
        np.testing.assert_allclose(control['final_task_cost'], reference['final_task_cost'])
        branch = run_branch(env, sim, ctl, None, 0, reference, np.array([0, -1]))
        self.assertEqual(int(branch['skipped'].sum()), 1)
        self.assertEqual(branch['n_interventions'].tolist(), [15, 16])
        self.assertEqual(branch['first_trig'].tolist(), [5, 0])
        np.testing.assert_allclose(branch['applied'][:, 1], reference['applied'][:, 1])
        self.assertGreater(branch['final_task_cost'][0], reference['final_task_cost'][0])


if __name__ == '__main__':
    unittest.main()
