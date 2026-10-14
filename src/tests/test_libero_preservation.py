import unittest
import numpy as np
from adapters.libero_preservation import PreservationWindow, candidates, rank_key


class PreservationTests(unittest.TestCase):
    def test_one_causal_window_no_reset_after_loss(self):
        gate = PreservationWindow()
        self.assertFalse(gate.active(0, None, None))
        self.assertFalse(gate.active(1, {'candidate': False}, None))
        self.assertTrue(gate.active(2, {'candidate': True}, None))
        for step in range(3, 22):
            self.assertTrue(gate.active(step, {'candidate': False}, None))
        self.assertFalse(gate.active(22, {'candidate': True}, None))
        self.assertEqual(gate.used, 20)
        self.assertEqual(gate.trigger, 2)

    def test_completion_forces_actor_only_resumption(self):
        gate = PreservationWindow()
        self.assertTrue(gate.active(11, {'candidate': True}, None))
        self.assertFalse(gate.active(12, {'candidate': True}, 11))
        self.assertFalse(gate.active(13, {'candidate': False}, 11))

    def test_original_is_exact_and_future_is_not_repeated_vla(self):
        action = np.array([.4, -.3, .2, -.1, .8, .6, 1.])
        copy = action.copy()
        plans = candidates(action, [0, 0, 0, 0, 0, 0, -1])
        np.testing.assert_array_equal(action, copy)
        np.testing.assert_array_equal(plans[0, 0], action)
        np.testing.assert_array_equal(plans[:, 1:, :6], 0)
        np.testing.assert_array_equal(plans[0, 1:, -1], 1)
        self.assertEqual(plans[4, 0, 2] * .05, .005000000000000001)
        self.assertEqual(plans[4, 0, -1], -1)

    def test_progress_beats_quiet_contact_and_ties_preserve_proposal(self):
        good = dict(candidate=True, conditions=dict(released=True, footprint_inside=True, bottom_seated=True))
        bad = dict(candidate=False, conditions=dict(released=True, footprint_inside=False, bottom_seated=True))
        proposal = np.ones(7)
        correction = np.zeros(7)
        self.assertLess(rank_key([good]*5, correction, proposal, 3),
                        rank_key([bad]*5, proposal, proposal, 0))
        self.assertLess(rank_key([good]*5, proposal, proposal, 0),
                        rank_key([good]*5, correction, proposal, 3))


if __name__ == '__main__':
    unittest.main()
