import unittest

import numpy as np

from adapters.libero_stress import ReleaseDelay, release_opportunity


class StressTests(unittest.TestCase):
    def test_sham_is_exact_and_does_not_mutate(self):
        a=np.array([.2,-.3,.4,0,0,0,-1.])
        original=a.copy()
        for step in range(10):
            out,trace=ReleaseDelay(3,0).apply(a,step)
            np.testing.assert_array_equal(out,original)
            self.assertFalse(trace['active'] or trace['changed'])
        np.testing.assert_array_equal(a,original)

    def test_same_physical_fault_after_different_method_actions(self):
        for command in [np.array([.5,0,0,0,0,0,-1.]),np.array([0,0,0,0,0,0,-1.])]:
            fault=ReleaseDelay(5,4)
            for step in range(11):
                effective,trace=fault.apply(command,step)
                np.testing.assert_array_equal(effective[:6],command[:6])
                self.assertEqual(effective[-1],1 if 5<=step<9 else -1)
                self.assertEqual(trace['active'],5<=step<9)
        # Already-closed commands still consume elapsed physical fault time.
        _,trace=ReleaseDelay(5,4).apply(np.array([0,0,0,0,0,0,1.]),6)
        self.assertTrue(trace['active'])
        self.assertFalse(trace['changed'])

    def test_anchor_is_a_release_opportunity_not_a_success_label(self):
        previous=dict(native_in=True,bowl_gripper=True,closed=False)
        opening=np.array([0,0,0,0,0,0,-1.])
        self.assertTrue(release_opportunity(previous,opening))
        self.assertFalse(release_opportunity(dict(previous,closed=True),opening))
        self.assertFalse(release_opportunity(dict(previous,bowl_gripper=False),opening))
        self.assertFalse(release_opportunity(previous,-opening))

    def test_invalid_commands_fail_before_injection(self):
        with self.assertRaises(ValueError): ReleaseDelay(-1,4)
        with self.assertRaises(ValueError): ReleaseDelay(0,-1)
        for bad in [np.zeros(6),np.full(7,np.nan)]:
            with self.assertRaises(ValueError): ReleaseDelay(0,4).apply(bad,0)


if __name__ == '__main__':
    unittest.main()
