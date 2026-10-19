"""Meaningful gate/budget tests without a simulator or model."""
import unittest
from experiment3_candidate_gate import Budget, CandidateGate, checks, LOCAL_CAP, executed_events


def evidence(x=0., native=False, grasped=False):
    return dict(item_position=[x, 0, 0], eef_position=[0, 0, 0], grasped=grasped,
        native_success=native, completion=dict(candidate=False, complete=False, max_book_caddy_penetration=0.))


class GateTests(unittest.TestCase):
    def test_budget_counts_candidates_not_just_actions(self):
        b = Budget(3)
        for _ in range(3): b.query()
        b.action()
        self.assertEqual((b.queries, b.actions), (3, 1))
        with self.assertRaises(RuntimeError): b.query()

    def test_checks_cannot_verify_future_and_reject_premature_release(self):
        c = checks([0]*6+[-1], None, evidence(grasped=True))
        self.assertFalse(c['grasp_release_consistency'])
        self.assertEqual(c['future_collision'], 'unknown')
        self.assertTrue(checks([0]*6+[-1], None, evidence(native=True, grasped=True))['grasp_release_consistency'])

    def test_reverification_requires_three_and_exit_is_not_handoff(self):
        g = CandidateGate('verification')
        g.begin(40, evidence(native=True))
        self.assertTrue(g.active)
        for step in (41, 42):
            m = g.begin(step, evidence(.004, native=True))
            self.assertTrue(g.active)
            self.assertEqual(m['events'], [])
        m = g.begin(43, evidence(.004, native=True))
        self.assertFalse(g.active)
        self.assertEqual([e['type'] for e in m['events']], ['local_exit'])
        self.assertFalse(any(e['type'] == 'handoff_executed' for e in m['events']))

    def test_local_cap_exits_and_same_step_nominal_is_selected(self):
        g = CandidateGate('verification')
        g.begin(40, evidence(native=True))
        g.used = LOCAL_CAP
        m = g.begin(41, evidence(native=True))
        a = [0]*6+[1]
        selected, d = g.choose([dict(action=a, source='greedy', query=0)], evidence(), m)
        self.assertEqual(selected, a)
        self.assertEqual(d['owner'], 'nominal')
        self.assertEqual(d['events'][0]['reason'], 'local_action_cap')

    def test_selection_uses_only_passed_sample_and_preserves_exact_action(self):
        g = CandidateGate('verification')
        m = g.begin(40, evidence(native=True))
        actions = [[0]*6+[1], [.1, 0, 0, 0, 0, 0, 1], [2, 0, 0, 0, 0, 0, 1]]
        g.actions.append(actions[0])
        cs = [dict(action=a, source='greedy' if i == 0 else 'sample', query=i) for i, a in enumerate(actions)]
        chosen, d = g.choose(cs, evidence(), m)
        self.assertEqual(chosen, actions[1])
        self.assertEqual(d['chosen'], 1)

    def test_executed_handoff_requires_actual_nominal_action(self):
        a = [0]*6+[1]
        d = dict(owner='nominal', events=[dict(type='local_exit', step=20)])
        result = executed_events(d, 20, a, a, 1)
        self.assertEqual([e['type'] for e in result], ['local_exit', 'handoff_executed'])
        self.assertEqual(len(d['events']), 1)
        with self.assertRaises(AssertionError):
            executed_events(d, 20, [1]*7, a, 1)


if __name__ == '__main__':
    unittest.main()
