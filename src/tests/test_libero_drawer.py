import unittest

import numpy as np

from adapters.libero_drawer import MAX_REPAIR, SubgoalTracker, TransitionGate, contact_flags, decision_trace


class DrawerGateTests(unittest.TestCase):
    def setUp(self):
        self.proposal = np.array([.2, .1, -.1, 0, 0, 0, 1.])
        self.measured = dict(retained=True, clear=False, outward_world=[0., -1., 0.])
        self.tracking = dict(entry=True, clear=False, released=False)

    def test_no_release_before_supported_event(self):
        gate = TransitionGate('phase_repair')
        for step in range(60):
            action, d = gate.decide(self.proposal, self.measured,
                                   dict(self.tracking, entry=False), step)
            np.testing.assert_array_equal(action, self.proposal)
            self.assertEqual(d['used'], 0)

    def test_release_then_clearance_then_actor_closing(self):
        gate = TransitionGate('phase_repair')
        a, _ = gate.decide(self.proposal, self.measured, self.tracking, 5)
        np.testing.assert_array_equal(a, [0, 0, 0, 0, 0, 0, -1])
        a, _ = gate.decide(self.proposal, self.measured, dict(self.tracking, released=True), 6)
        np.testing.assert_allclose(a, [0, -.1, 0, 0, 0, 0, -1])
        a, d = gate.decide(self.proposal, self.measured, dict(self.tracking, clear=True), 7)
        np.testing.assert_array_equal(a, self.proposal)
        self.assertEqual(d['handoff']['reason'], 'stable_clearance')
        a, _ = gate.decide(self.proposal, self.measured, self.tracking, 8)
        np.testing.assert_array_equal(a, self.proposal)  # One transaction, no hidden retrigger.

    def test_budget_exhaustion_and_placement_loss(self):
        gate = TransitionGate('phase_repair')
        for step in range(MAX_REPAIR):
            gate.decide(self.proposal, self.measured, self.tracking, step)
        a, d = gate.decide(self.proposal, self.measured, self.tracking, MAX_REPAIR)
        np.testing.assert_array_equal(a, self.proposal)
        self.assertEqual(d['handoff']['reason'], 'budget_exhausted')
        gate = TransitionGate('phase_repair')
        gate.decide(self.proposal, self.measured, self.tracking, 0)
        a, d = gate.decide(self.proposal, dict(self.measured, retained=False), self.tracking, 1)
        np.testing.assert_array_equal(a, self.proposal)
        self.assertEqual(d['handoff']['reason'], 'placement_lost')

    def test_controls_are_distinct_and_bounded(self):
        for arm in ['gate_hold', 'timed_repair']:
            gate = TransitionGate(arm)
            for step in range(MAX_REPAIR):
                a, _ = gate.decide(self.proposal, self.measured, self.tracking, step)
                self.assertEqual(a[-1], -1)
                self.assertAlmostEqual(np.linalg.norm(a[:3]), .1 if arm == 'timed_repair' and step >= 10 else 0.)
            a, _ = gate.decide(self.proposal, self.measured, self.tracking, MAX_REPAIR)
            np.testing.assert_array_equal(a, self.proposal)

    def test_contact_pair_identity_and_final_sample_support(self):
        floor = 'white_cabinet_1_g38'
        def c(a, b):
            return dict(geom1=a, geom2=b, distance=-.0001)
        samples = [dict(contacts=[c('akita_black_bowl_1_g1', floor),
                                 c('gripper0_finger1_collision', 'akita_black_bowl_1_g1')]),
                   dict(contacts=[c('gripper0_finger1_collision', floor)])]
        flags = contact_flags(samples, dict(floor=floor))
        self.assertTrue(flags['bowl_gripper'])
        self.assertTrue(flags['cabinet_gripper'])
        self.assertFalse(flags['bowl_supported'])

    def test_milestones_require_history_not_one_endpoint(self):
        tracker = SubgoalTracker()
        m = dict(retained=True, relative_speed=0., angular_speed=0., bowl_gripper=False,
                 clear=True, candidate=False, native_in=True, closed=False,
                 alignment_error=0., drawer_q=-.15)
        self.assertFalse(tracker.update(m, 0)['entry'])
        self.assertFalse(tracker.update(m, 1)['released'])
        t = tracker.update(m, 2)
        self.assertTrue(t['entry'] and t['released'] and t['clear'])
        self.assertFalse(t['complete'])
        self.assertEqual(t['phase'], 'close_drawer')
        t = tracker.update(dict(m, retained=False), 3)
        self.assertFalse(t['entry'] or t['released'] or t['clear'])

    def test_observable_trace_is_pre_action_and_exact(self):
        m = dict(retained=True, relative_speed=0., angular_speed=0., bowl_gripper=False,
            cabinet_gripper=False, bowl_supported=True, clear=False, candidate=False,
            native_in=True, closed=False, alignment_error=0., drawer_q=-.15,
            bowl_min=[0,0,0], bowl_max=[1,1,1], drawer_lower=[-1,-1,0],
            drawer_upper=[2,2,2], gripper_front_margin=-.02, gripper_above_margin=-.01,
            outward_world=[0,-1,0])
        tracker, gate = SubgoalTracker(), TransitionGate('phase_repair')
        for step in range(3):
            tracking = tracker.update(m, step)
        action, d = gate.decide(self.proposal, m, tracking, 3)
        trace = decision_trace(3, 'phase_repair', m, tracking, self.proposal, action, d)
        self.assertEqual(trace['evidence_through_step'], 2)
        self.assertNotIn('observed_after_execution', trace)
        self.assertEqual(trace['predicates']['detached']['consecutive'], 3)
        self.assertEqual(trace['gate']['reason'], 'retract_outward')
        np.testing.assert_array_equal(trace['executed_action'], action)
        np.testing.assert_array_equal(trace['nominal_action'], self.proposal)


if __name__ == '__main__':
    unittest.main()
