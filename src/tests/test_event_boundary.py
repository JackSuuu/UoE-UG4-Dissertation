"""Boundary collector interventions must preserve proposal/history/cadence."""
import os
import sys
import unittest
import torch

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, 'experiments'))
from diagnose_event_boundary import ARMS, BoundaryController, perturb
from sims.base import TorchEnv
from sims.torch_push import PushSim


class Policy:
    H, act_dim = 10, 2

    def __call__(self, obs, image=None, instruction=None):
        return obs.new_full((len(obs), 10, 2), .1)


class BoundaryTests(unittest.TestCase):
    def test_ramps_are_bounded_and_preserve_original(self):
        original = torch.ones(2, 10, 2) * .1
        v = torch.ones(2)
        for kind in ARMS:
            out = perturb(original, kind, v * 2, v * .12, v * .02, v * .25, .5)
            self.assertLessEqual(float(out.abs().max()), .5)
            torch.testing.assert_close(original, torch.ones_like(original) * .1)
            if kind == 'ramp_060':
                torch.testing.assert_close(out[0, :5, 0], torch.tensor([.112, .124, .136, .148, .160]))
            if kind == 'brake_ramp':
                torch.testing.assert_close(out[0, :5, 0], torch.tensor([.085, .070, .055, .040, .025]))

    def test_all_arms_execute_cache_once_and_restore_policy(self):
        for kind in ARMS:
            sim = PushSim('cpu'); env = TorchEnv(sim, 3)
            obs = env.reset(sim.make_params(3, {}), 1)
            ctrl = BoundaryController(Policy(), sim, kind, 1., 4)
            ctrl.reset(env, obs)
            action, info = ctrl.act(env, obs)
            plan = ctrl.plan.clone()
            self.assertEqual(bool(info['collector_pulse'].all()), kind != 'none')
            torch.testing.assert_close(info['chunk'], torch.full_like(plan, .1))
            torch.testing.assert_close(ctrl.a_prev, action, rtol=0, atol=0)
            for step in range(1, 5):
                obs, _, _ = env.step(action)
                action, info = ctrl.act(env, obs)
                torch.testing.assert_close(action, plan[:, step], rtol=0, atol=0)
                self.assertFalse(info['collector_pulse'].any())
            self.assertEqual(ctrl.n_policy_calls, 1)
            obs, _, _ = env.step(action)
            action, info = ctrl.act(env, obs)
            self.assertEqual(ctrl.n_policy_calls, 2)
            torch.testing.assert_close(action, torch.full_like(action, .1))
            self.assertFalse(info['collector_pulse'].any())

    def test_onset_is_observation_only_and_does_not_pulse_early(self):
        sim = PushSim('cpu'); env = TorchEnv(sim, 2)
        obs = env.reset(sim.make_params(2, {}), 2)
        ctrl = BoundaryController(Policy(), sim, 'ramp_030', .06, 4)
        ctrl.reset(env, obs)
        _, info = ctrl.act(env, obs)
        self.assertFalse(info['collector_pulse'].any())


if __name__ == '__main__':
    unittest.main()
