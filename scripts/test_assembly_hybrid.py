"""Mechanism invariants: coupling, release, uncertainty fallback and real override."""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from adapters.assembly_hybrid import candidates, decide


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.obs = np.zeros(40, dtype=np.float32)
        self.obs[:3] = [.012, .012, .1]
        self.nominal = np.array([0, 0, -.15, 0, 0, 0, 1.])
        self.bank = candidates(torch.tensor(self.nominal, dtype=torch.float32)).numpy()[:, 0]
        self.p = dict(scores=np.ones(11)*.5, objective=np.ones(11), objective_std=np.zeros(11),
            mean=np.zeros((11, 3, 40)), state_std=np.zeros((11, 3, 40)))
        self.p['scores'][0] = 2.
        self.monitor = dict(grasped=True, touching=True, seated=False, tilt_deg=0.)
        self.history = dict(used=0, stall=0, ungrasped=0, last_load=0.,
            residual=dict(xyz_m=.001, force_N=1., tilt_deg=1.))

    def decision(self, arm='hybrid80'):
        return decide(arm, self.obs, self.nominal, self.bank, self.p, self.monitor, self.history, .65)

    def test_lost_coupling_returns_actor(self):
        self.monitor['grasped'] = False
        self.history['ungrasped'] = 3
        r = self.decision()
        self.assertFalse(r['changed'])
        self.assertEqual(r['reason'], 'lost_grasp_return_actor')

    def test_untrusted_model_uses_geometry(self):
        self.history['residual']['xyz_m'] = .04
        self.p['objective'][2] = -10
        r = self.decision()
        self.assertEqual(r['selection'], 'x_minus')
        self.assertEqual(r['owner'], 'geometry')

    def test_prediction_can_actually_override_geometry(self):
        self.p['objective'][6] = .1
        r = self.decision()
        self.assertEqual(r['selection'], 'y_minus')
        self.assertEqual(r['owner'], 'prediction')
        self.assertTrue(r['changed'])

    def test_outward_direction_cannot_win(self):
        self.p['objective'][5] = -100
        r = self.decision()
        self.assertNotIn(5, r['eligible'])
        self.assertEqual(r['selection'], 'x_minus')

    def test_above_ceiling_cannot_override(self):
        self.p['objective'][6] = -100
        self.p['scores'][6] = 10
        r = self.decision()
        self.assertNotIn(6, r['accepted'])
        self.assertEqual(r['selection'], 'x_minus')

    def test_release_and_budget(self):
        self.monitor['seated'] = True
        r = self.decision()
        self.assertEqual(r['selection'], 'open_hold')
        self.history['used'] = 80
        r = self.decision()
        self.assertFalse(r['changed'])
        self.assertEqual(r['reason'], 'budget_exhausted')

    def test_premature_release_preserves_proposal_pose(self):
        self.nominal[-1] = -1
        self.bank = candidates(torch.tensor(self.nominal, dtype=torch.float32)).numpy()[:, 0]
        self.history['residual'] = None
        r = self.decision()
        self.assertEqual(r['selection'], 'retain_proposal')
        np.testing.assert_allclose(r['selected'][:6], self.nominal[:6])
        self.assertEqual(r['selected'][6], 1)


if __name__ == '__main__':
    unittest.main()
