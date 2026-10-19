import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import unittest
from experiment3_fridge_gate import command_checks, FridgeCandidateGate
from adapters.fridge_stage_evidence import FridgeStageEvidence, predicates


def state(support=True, native=False):
    return dict(grasp=True, local_release_support=support, native_success=native,
        item=[0, 0, 0], eef=[0, 0, 0], angle=0.)


def stages(value='violated'):
    return dict(phase='open', predicates=dict(open=dict(status=value)))


class FridgeTests(unittest.TestCase):
    def test_supported_open_allowed_before_full_native_success(self):
        self.assertTrue(command_checks([0]*6+[-1], None, state(True, False))['local_release_consistency'])

    def test_native_success_does_not_substitute_for_local_support(self):
        self.assertFalse(command_checks([0]*6+[-1], None, state(False, True))['local_release_consistency'])

    def test_unknown_support_is_not_release_permission(self):
        c = command_checks([0]*6+[-1], None, state(None))
        self.assertFalse(c['local_release_consistency'])
        self.assertEqual(c['local_release_support'], 'unknown')
        self.assertEqual(c['future_support'], 'unknown')

    def test_three_current_rechecks_and_unknown_resets(self):
        g = FridgeCandidateGate('verification')
        for step in range(41): g.begin(step, state(), stages())
        self.assertTrue(g.active)
        g.begin(41, state(), stages('satisfied'))
        g.begin(42, state(), stages('satisfied'))
        g.begin(43, state(), stages('unknown'))
        self.assertEqual(g.stable, 0)
        for step in (44, 45): g.begin(step, state(), stages('satisfied'))
        m = g.begin(46, state(), stages('satisfied'))
        self.assertFalse(g.active)
        self.assertEqual(m['events'][0]['reason'], 'three_current_stage_rechecks')

    def test_selector_returns_exact_model_action_no_motion_generation(self):
        g = FridgeCandidateGate('verification')
        for step in range(41): m = g.begin(step, state(False), stages())
        cs = [dict(action=[0]*6+[1], query=0, source='greedy'),
              dict(action=[.1, 0, 0, 0, 0, 0, -1], query=1, source='sample')]
        a, d = g.choose(cs, state(False, True), m)
        self.assertEqual(a, cs[0]['action'])
        self.assertFalse(d['candidates'][1]['checks']['local_release_consistency'])

    def test_open_initialization_not_achieved_full_closed_start(self):
        t = FridgeStageEvidence('open')
        self.assertTrue(t.milestones['open'])
        self.assertFalse(FridgeStageEvidence('closed').milestones['open'])
        self.assertEqual(predicates({}, None)['place']['status'], 'unknown')


if __name__ == '__main__': unittest.main()
