"""Budget exits cannot become verified recoveries; handoff means execution."""
import unittest
from src.repair.fridge_probe import Probe


class ProbeTests(unittest.TestCase):
    def fixture(self):
        state = dict(angle=0., door_speed=0., grasp=False, extracted=False, placed=False,
            clear=False, eef=[0., 0., 0.], item=[1., 0., 0.], handle=[1., 0., 0.], floor=0.)
        return Probe('full task', {'open':'open the fridge'}, True), state, {'phase':'open'}

    def enter(self, probe, state, task):
        for i in range(30):
            decision = probe.before(i, state, task)
            probe.after(i, state, decision)
        self.assertEqual(decision['owner'], 'repair')

    def test_budget_then_actual_unverified_handoff(self):
        p, s, task = self.fixture()
        self.enter(p, s, task)
        for i in range(30, 61):
            p.after(i, s, p.before(i, s, task))
        self.assertFalse(p.repair['verified'])
        self.assertEqual(p.repair['used'], 32)
        self.assertIsNone(p.repair['handoff_step'])
        p.after(61, s, p.before(61, s, task))
        self.assertEqual(p.repair['handoff_step'], 61)
        self.assertIsNone(p.repair['stable_nominal_continuation'])
        self.assertEqual(p.before(90, s, task)['owner'], 'nominal')

    def test_verified_exit_requires_retention_after_actual_handoff(self):
        p, s, task = self.fixture()
        self.enter(p, s, task)
        s['angle'] = 2.
        for i in range(30, 33):
            p.after(i, s, p.before(i, s, task))
        self.assertTrue(p.repair['verified'])
        self.assertIsNone(p.repair['handoff_step'])
        s['angle'] = 0.
        p.after(33, s, p.before(33, s, task))
        self.assertFalse(p.repair['stable_nominal_continuation'])
        self.assertEqual(p.repair['post_handoff_loss_step'], 33)

    def test_bare_control_never_changes_language_or_owner(self):
        p, s, task = self.fixture()
        p.enabled = False
        for i in range(100):
            d = p.before(i, s, task)
            p.after(i, s, d)
            self.assertEqual((d['owner'], d['language']), ('nominal', 'full task'))
        self.assertIsNone(p.repair)


if __name__ == '__main__':
    unittest.main()
