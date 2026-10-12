"""CPU regression tests for matched open-loop execution across all arms.

Run: cd src && python -m unittest discover -s tests -p test_chunk_protocol.py
"""
import os
import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checkvla.verifier import Controller
from experiments._verif import verifier_rollout


class Policy:
    H, act_dim = 10, 1
    needs_img = needs_rgb = False

    def __call__(self, obs, img=None, instruction=None):
        # Distinct actions at every horizon step expose stale/first-action reuse.
        return obs[:, :1, None] + torch.arange(self.H)[None, :, None] / 100


class Env:
    def __init__(self):
        self.state = torch.tensor([[0.1], [0.2]])

    def get_state(self):
        return self.state

    def shadow_rollout(self, chunk, mask=None):
        return chunk.abs()

    def reset(self, params=None, seed=0):
        self.state = torch.tensor([[0.1], [0.2]])
        return self.state

    def render(self):
        return torch.zeros(2, 3, 2, 2)

    def step(self, action):
        self.state = self.state + action / 100
        return self.state, action.abs(), torch.ones(2, dtype=torch.bool)


class Verifier:
    commit = 2  # Must not override an explicitly specified chunk_k=5.
    predictor = SimpleNamespace(needs_img=False)

    def __init__(self, trigger=False):
        self.trigger = trigger
        self.checked = []

    def check(self, ctx, chunk):
        self.checked.append(chunk.shape[0])
        trig = (ctx['obs'][:, 0] > 0.15) if self.trigger else torch.zeros(
            chunk.shape[0], dtype=torch.bool)
        return trig, torch.zeros(chunk.shape[0])

    def repair(self, ctx, chunk):
        return chunk * 0.5

    def score(self, ctx, chunk):
        return chunk.abs().amax(dim=(1, 2))


def execute(mode, verifier=None):
    env = Env()
    ctl = Controller(Policy(), None, mode, verifier=verifier, commit=2, chunk_k=5)
    ctl.reset(env, env.state)
    actions, triggers = [], []
    for t in range(15):
        # Observation changes on every step, not only at chunk boundaries.
        env.state = torch.tensor([[0.1 + t / 1000], [0.2 + t / 1000]])
        action, info = ctl.act(env, env.state)
        actions.append(action.clone())
        triggers.append(info['trig'].clone())
    return ctl, torch.stack(actions), torch.stack(triggers)


class ChunkProtocolTests(unittest.TestCase):
    def test_non_triggering_verifier_is_identical_to_none(self):
        base, expected, _ = execute('none')
        ver = Verifier()
        ctl, actual, _ = execute('checkvla', ver)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(ctl.n_policy_calls, base.n_policy_calls)
        self.assertEqual(ctl.n_policy_calls, 3)
        self.assertEqual(ver.checked, [2, 2, 2])

    def test_non_triggering_oracle_is_identical_to_none(self):
        _, expected, _ = execute('none')
        ctl, actual, _ = execute('gt_shadow')
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(ctl.n_policy_calls, 3)

    def test_mixed_trigger_rows_keep_the_same_commitment(self):
        _, expected, _ = execute('none')
        ver = Verifier(trigger=True)
        ctl, actual, triggers = execute('checkvla', ver)
        self.assertTrue(torch.equal(actual[:, 0], expected[:, 0]))
        self.assertTrue(torch.equal(actual[:, 1], expected[:, 1] * 0.5))
        self.assertEqual(ctl.n_policy_calls, 3)
        self.assertEqual(ctl.n_policy_rows, 6)
        self.assertEqual(triggers[:, 1].nonzero().flatten().tolist(), [0, 5, 10])

    def test_triggering_oracle_keeps_the_same_commitment(self):
        env = Env()
        env.state = torch.tensor([[0.1], [2.0]])
        ctl = Controller(Policy(), None, 'gt_shadow', commit=2, chunk_k=5)
        ctl.reset(env, env.state)
        triggers = []
        for _ in range(15):
            _, info = ctl.act(env, env.state)
            triggers.append(info['trig'].clone())
        self.assertEqual(ctl.n_policy_calls, 3)
        self.assertEqual(torch.stack(triggers)[:, 1].nonzero().flatten().tolist(), [0, 5, 10])

    def test_calibration_marks_only_real_chunk_decisions(self):
        class Predictor:
            def risk(self, ctx, actions):
                return actions.abs(), torch.zeros_like(actions)

        ver = Verifier()
        ver.predictor = Predictor()
        sim = SimpleNamespace(T=12, act_dim=1)
        result = verifier_rollout(Env(), sim, Policy(), None, {'orbisim': ver}, chunk_k=5)
        mask = result['decision_mask']
        self.assertEqual(mask.shape, (2, 12))
        self.assertEqual(mask[0].nonzero()[0].tolist(), [0, 5, 10])
        self.assertEqual(mask.sum(), 6)


if __name__ == '__main__':
    unittest.main()
