"""Evidence semantics only; synthetic cases are never robot-success episodes."""
import unittest

from adapters.fridge_stage_evidence import FridgeStageEvidence, predicates


def state(**kw):
    s = dict(angle=0., door_speed=0., grasp=False, extracted=False, shelf=True,
             placed=False, table=False, hand_contact=False, item_speed=0.,
             item_angular_speed=0., clear=False, door_contact=False)
    s.update(kw)
    return s


class TestFridgeStageEvidence(unittest.TestCase):
    def test_unknown_release_and_stale_observation_cannot_satisfy(self):
        s = state(placed=True, table=True)
        self.assertEqual(predicates(s, None)['place']['status'], 'unknown')
        self.assertEqual(predicates(s, True, evidence_age=1)['close']['status'], 'unknown')
        del s['angle']
        self.assertEqual(predicates(s, True)['open']['status'], 'unknown')
        s['angle'] = float('nan')
        self.assertEqual(predicates(s, True)['open']['status'], 'unknown')
        s['item_speed'] = .1
        self.assertEqual(predicates(s, True)['place']['status'], 'violated')

    def test_initial_complete_pose_does_not_invent_manipulation(self):
        monitor = FridgeStageEvidence('closed')
        for i in range(30):
            row = monitor.observe(i, state(placed=True, table=True, clear=True), True)
        self.assertFalse(row['full_closed_start_completion'])
        self.assertFalse(row['milestone_history']['grasp'])

    def test_full_sequence_and_release_interrupt(self):
        for initial in ('open', 'closed'):
            monitor = FridgeStageEvidence(initial)
            sequence = [state(angle=2.)]*3
            sequence += [state(angle=2., grasp=True, hand_contact=True)]*3
            sequence += [state(angle=2., grasp=True, hand_contact=True, extracted=True, shelf=False)]*3
            sequence += [state(angle=2., placed=True, table=True, shelf=False)]*10
            sequence += [state(placed=True, table=True, clear=True, shelf=False)]*22
            for i, s in enumerate(sequence):
                row = monitor.observe(i, s, not s['hand_contact'])
            self.assertTrue(row['remaining_sequence_complete'])
            self.assertEqual(row['full_closed_start_completion'], initial=='closed')
            row = monitor.observe(len(sequence), sequence[-1], None)
            self.assertEqual(row['complete_dwell'], 0)
            self.assertFalse(row['remaining_sequence_complete'])
            self.assertFalse(any(e.get('control_handoff_claim') for e in row['events']))

    def test_missing_step_does_not_count_as_continuous_dwell(self):
        monitor = FridgeStageEvidence('closed')
        monitor.observe(0, state(), True)
        with self.assertRaises(ValueError):
            monitor.observe(2, state(), True)


if __name__ == '__main__':
    unittest.main()
