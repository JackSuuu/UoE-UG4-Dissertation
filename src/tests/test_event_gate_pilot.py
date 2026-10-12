"""Event coverage and collector cache/history contracts."""
import os
import sys
import unittest
import torch

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, 'experiments'))
from collect_event_gate_pilot import PulseController, coverage, event_masks, passes_pilot
from sims.base import TorchEnv
from sims.torch_push import PushSim


class Policy:
    H, act_dim = 10, 2

    def __call__(self, obs, image=None, instruction=None):
        return obs.new_full((len(obs), 10, 2), .1)


class PilotTests(unittest.TestCase):
    def test_categories_exclude_post_violation_and_separate_anticipation(self):
        r = torch.zeros(4, 8, 10, 2)
        r[0, 0, 2, 0] = 1.2
        r[1, 0, 7, 0] = 1.2
        r[2, 0, 0, 0] = .5
        r[3, 0, 0, 0] = 2.
        d = {'risk': r, 'prior_violation': torch.tensor([False, False, False, True])}
        m = event_masks(d)
        self.assertEqual(m['immediate'].tolist(), [True, False, False, False])
        self.assertEqual(m['anticipation'].tolist(), [False, True, False, False])
        self.assertEqual(m['safe_contact'].tolist(), [False, False, True, False])

    def test_paired_branches_count_as_one_root(self):
        d = {'risk': torch.ones(6, 8, 10, 2) * 1.2, 'obs': torch.zeros(6, 10),
             'prior_violation': torch.zeros(6, dtype=torch.bool),
             'episode_id': torch.tensor([1, 1, 1, 2, 2, 2]),
             'physical_mults': torch.ones(6, 2)}
        c = coverage(d)
        self.assertEqual(c['events']['immediate']['decisions'], 6)
        self.assertEqual(c['events']['immediate']['root_episodes'], 2)
        self.assertFalse(passes_pilot(c, 'train'))

    def test_pulse_updates_executed_history_and_keeps_five_step_commitment(self):
        for kind in ('drive', 'brake'):
            sim = PushSim('cpu'); env = TorchEnv(sim, 3)
            obs = env.reset(sim.make_params(3, {}), 1)
            c = PulseController(Policy(), sim, kind, 123)
            c.reset(env, obs); c.onset_gap.fill_(1.)
            a, info = c.act(env, obs)
            original = info['chunk'].clone()
            modified = c.plan.clone()
            self.assertTrue(info['collector_pulse'].all())
            torch.testing.assert_close(original, torch.full_like(original, .1))
            self.assertTrue((a - original[:, 0]).abs().any())
            torch.testing.assert_close(c.a_prev, a, rtol=0, atol=0)
            self.assertTrue((c.plan_ptr == 1).all())
            self.assertTrue((c.plan_left == 4).all())
            self.assertLessEqual(float(modified.abs().max()), sim.max_vel)
            for j in range(1, 5):
                obs, _, _ = env.step(a)
                a, info = c.act(env, obs)
                torch.testing.assert_close(a, modified[:, j])
                self.assertFalse(info['collector_pulse'].any())
            self.assertEqual(c.n_policy_calls, 1)
            obs, _, _ = env.step(a)
            _, info = c.act(env, obs)
            self.assertEqual(c.n_policy_calls, 2)
            self.assertFalse(info['collector_pulse'].any())


if __name__ == '__main__':
    unittest.main()
