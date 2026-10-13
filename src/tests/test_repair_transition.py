"""Repair-aware shaping is checked before execution and preserves H10/k5."""
import os
import sys
import unittest
import torch

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, 'experiments'))
from validate_repair_transition import TransitionController, transition_proposal, MODES
from checkvla.verifier import Controller
from sims.base import TorchEnv
from sims.torch_push import PushSim


class Policy:
    H, act_dim = 10, 2

    def __call__(self, obs, image=None, instruction=None):
        return obs.new_full((len(obs), 10, 2), .1)


class Verifier:
    commit, needs_img = 5, False

    def __init__(self, triggers=(0,), abstain=False):
        self.predictor = self
        self.triggers, self.abstain, self.seen = triggers, abstain, []

    def check(self, ctx, chunk):
        index = len(self.seen)
        self.seen.append(chunk.clone())
        return torch.full((len(chunk),), index in self.triggers, dtype=torch.bool, device=chunk.device), chunk.new_zeros(len(chunk))

    def repair(self, ctx, chunk):
        return chunk.clone() if self.abstain else chunk * .25


class TransitionTests(unittest.TestCase):
    def make(self, mode='ramp', verifier=None):
        sim = PushSim('cpu'); env = TorchEnv(sim, 3)
        obs = env.reset(sim.make_params(3, {}), 1)
        c = TransitionController(Policy(), sim, verifier, mode)
        c.reset(env, obs)
        return sim, env, obs, c

    def test_transform_formula_suffix_and_bounds(self):
        raw = torch.full((2, 10, 2), .1)
        previous = torch.full((2, 2), .025)
        for mode in MODES:
            out = transition_proposal(raw, previous, mode)
            torch.testing.assert_close(out[:, 5:], raw[:, 5:], rtol=0, atol=0)
            if mode == 'ramp':
                torch.testing.assert_close(out[0, :5, 0], torch.tensor([.040, .055, .070, .085, .100]))
            if mode == 'flat':
                torch.testing.assert_close(out[:, :5], torch.full_like(out[:, :5], .070))
            if mode == 'pause':
                self.assertFalse(out[:, :5].any())
        edge = torch.tensor([-.5, .5]).expand(2, 10, 2)
        for mode in MODES:
            self.assertLessEqual(float(transition_proposal(edge, -edge[:, 0], mode).abs().max()), .5)
        torch.testing.assert_close(raw, torch.full_like(raw, .1), rtol=0, atol=0)

    def test_gate_sees_transformed_chunk_once_and_bridge_does_not_rearm(self):
        for mode in MODES:
            verifier = Verifier()
            _, env, obs, c = self.make(mode, verifier)
            previous = None
            for t in range(15):
                a, info = c.act(env, obs)
                if t == 0:
                    self.assertTrue(info['repair_changed'].all())
                    self.assertFalse(info['postrepair'].any())
                if t == 5:
                    self.assertTrue(info['postrepair'].all())
                    expected = transition_proposal(info['raw_proposal'], previous, mode)
                    torch.testing.assert_close(verifier.seen[-1], expected, rtol=0, atol=0)
                    torch.testing.assert_close(a, expected[:, 0], rtol=0, atol=0)
                if t == 10:
                    self.assertFalse(info['postrepair'].any())
                    torch.testing.assert_close(a, torch.full_like(a, .1), rtol=0, atol=0)
                if t % 5:
                    self.assertFalse(info['transition_changed'].any())
                    self.assertFalse(info['repair_changed'].any())
                torch.testing.assert_close(c.a_prev, a, rtol=0, atol=0)
                previous = a.clone()
                obs, _, _ = env.step(a)
            self.assertEqual(len(verifier.seen), 3)
            self.assertEqual(c.n_policy_calls, 3)
            self.assertEqual(c.n_policy_rows, 9)

    def test_abstention_does_not_arm_but_repeated_real_repair_does(self):
        for abstain in (True, False):
            verifier = Verifier(triggers=(0, 1), abstain=abstain)
            _, env, obs, c = self.make('ramp', verifier)
            for t in range(11):
                a, info = c.act(env, obs)
                if t in (5, 10):
                    self.assertEqual(bool(info['postrepair'].all()), not abstain)
                obs, _, _ = env.step(a)

    def test_direct_is_exact_original_controller(self):
        sim, env, obs, c = self.make('direct', Verifier())
        other_env = TorchEnv(sim, 3)
        other_obs = other_env.reset(sim.make_params(3, {}), 1)
        base = Controller(Policy(), sim, 'checkvla', verifier=Verifier(), chunk_k=5)
        base.reset(other_env, other_obs)
        for _ in range(20):
            a, _ = c.act(env, obs)
            b, _ = base.act(other_env, other_obs)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            obs, _, _ = env.step(a)
            other_obs, _, _ = other_env.step(b)
            torch.testing.assert_close(obs, other_obs, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
