"""Model-free recovery-contract checks for the actual five-layer controller."""
import importlib.util
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1]/'src/legacy/microwave_controller.py'
spec = importlib.util.spec_from_file_location('framework_controller', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
NativeFramework = module.NativeFramework
LANGUAGE = 'put the yellow and white mug in the microwave and close it'
ACTION = [.01, -.02, .03, .04, -.05, .06, 1.]


def state(z=.9, grasp=True, inside=False, closed=False, table=True, contact=False,
          gripper_contact=None, x=0.):
    pairs = []
    if table:
        pairs.append(dict(geom1='table_collision', geom2='white_yellow_mug_1_g1'))
    if grasp if gripper_contact is None else gripper_contact:
        pairs.append(dict(geom1='gripper0_left', geom2='white_yellow_mug_1_g1'))
    if contact:
        pairs.append(dict(geom1='microwave_1_g4', geom2='white_yellow_mug_1_g1'))
    return dict(predicates={'["in","white_yellow_mug_1","microwave_1_heating_region"]': inside,
                           '["close","microwave_1"]': closed},
        grasped={'white_yellow_mug_1': grasp},
        object_observations={'white_yellow_mug_1_pos': [x, 0., z]},
        contacts_endpoint=pairs, native_success=inside and closed)


def execute(controller, step, before, after):
    decision = controller.before(step, before)
    action = list(ACTION)
    record = controller.after(step, after, action, decision)
    assert action == ACTION
    return decision, record


def contact_entry(controller):
    before = state()
    high = state(z=.93, table=False, contact=True)
    for step in range(13):
        decision, _ = execute(controller, step, before, high)
        assert decision['owner'] == 'nominal'
        before = high
    return 13, high


def contained_release(controller):
    before = state()
    contained = state(z=.95, table=False, inside=True)
    for step in range(3):
        execute(controller, step, before, contained)
        before = contained
    released = state(z=.95, grasp=False, table=False)
    execute(controller, 3, contained, released)
    return 4, released


class FrameworkTests(unittest.TestCase):
    def test_delayed_successful_lift_is_not_a_failure(self):
        for lift_after in (156, 158):
            c = NativeFramework('framework', LANGUAGE)
            before = state()
            for step in range(190):
                after = state(z=.93, table=False) if step >= lift_after else state()
                decision, _ = execute(c, step, before, after)
                self.assertIsNone(decision['trigger'])
                self.assertEqual(decision['owner'], 'nominal')
                before = after

    def test_loss_before_containment_still_intervenes_immediately(self):
        for contact in (False, True):
            c = NativeFramework('framework', LANGUAGE)
            high = state(z=.93, table=False)
            execute(c, 0, state(), high)
            lost = state(z=.92, grasp=False, table=False, contact=contact)
            execute(c, 1, high, lost)
            decision = c.before(2, lost)
            self.assertEqual(decision['trigger'], 'elevated_grasp_lost')
            self.assertEqual(decision['owner'], 'repair')
            self.assertEqual(decision['subgoal'], 'acquire')

    def test_contained_release_then_contact_keeps_nominal_control(self):
        c = NativeFramework('framework', LANGUAGE)
        start, before = contained_release(c)
        settled = state(z=.92, grasp=False, table=False, contact=True)
        for step in range(start, start+20):
            decision, record = execute(c, step, before, settled)
            self.assertEqual(decision['owner'], 'nominal')
            self.assertTrue(record['evidence']['raw_containment_lost'])
            self.assertTrue(record['evidence']['release_transition'])
            before = settled

    def test_release_exemption_ends_on_fall_escape_or_missing_contact(self):
        cases = [state(z=.92, grasp=False, table=True, contact=True),
                 state(z=.92, grasp=False, table=False, contact=True, x=.1),
                 state(z=.92, grasp=False, table=False)]
        for lost in cases:
            c = NativeFramework('framework', LANGUAGE)
            start, before = contained_release(c)
            for step in range(start, start+4):
                execute(c, step, before, lost)
                before = lost
            self.assertTrue(c.summary['repairs'])

    def test_one_frame_containment_does_not_qualify_release_exemption(self):
        c = NativeFramework('framework', LANGUAGE)
        high = state(z=.95, inside=True, table=False)
        execute(c, 0, state(), high)
        lost = state(z=.95, grasp=False, table=False, contact=True)
        execute(c, 1, high, lost)
        self.assertEqual(c.before(2, lost)['owner'], 'repair')

    def test_acquire_contract_accepts_containment_without_moving_to_close(self):
        c = NativeFramework('framework', LANGUAGE)
        high = state(z=.93, table=False)
        lost = state(z=.93, table=False, grasp=False, contact=True)
        execute(c, 0, state(), high)
        execute(c, 1, high, lost)
        # A false grasp flag alone does not mean detached: still request release.
        contained = state(z=.93, table=False, grasp=False, inside=True,
                          contact=True, gripper_contact=True)
        before = lost
        for step in range(2, 5):
            decision, record = execute(c, step, before, contained)
            self.assertEqual(decision['recovery_goal'], 'acquire')
            if step > 2:
                self.assertEqual(decision['subgoal'], 'contain_release')
            self.assertEqual(record['verification']['repair_verified'], step == 4)
            before = contained
        repair = c.summary['repairs'][0]
        self.assertEqual(repair['verified_by'], 'contained')
        self.assertIn('in_goal', repair['preserve'])
        self.assertIsNone(repair['handoff_step'])
        decision = c.before(5, contained)
        self.assertEqual(decision['language'], LANGUAGE)
        self.assertIsNone(c.summary['repairs'][0]['handoff_step'])
        complete = state(z=.93, table=False, grasp=False, inside=True, closed=True)
        record = c.after(5, complete, ACTION, decision)
        self.assertTrue(c.summary['repairs'][0]['completed_after_handoff'])
        self.assertIn('handoff_executed', [e['type'] for e in record['events']])

    def test_alternating_acceptance_branches_cannot_make_three_checks(self):
        c = NativeFramework('framework', LANGUAGE)
        high = state(z=.93, table=False)
        lost = state(z=.93, grasp=False, table=False)
        execute(c, 0, state(), high)
        execute(c, 1, high, lost)
        before = lost
        for step in range(2, 10):
            after = high if step % 2 == 0 else state(z=.93, grasp=False, inside=True, table=False)
            _, record = execute(c, step, before, after)
            self.assertFalse(record['verification']['repair_verified'])
            self.assertEqual(record['verification']['consecutive'], 1)
            before = after

    def test_transfer_requires_containment_not_retained_grasp(self):
        c = NativeFramework('framework', LANGUAGE)
        start, high = contact_entry(c)
        for step in range(start, start+32):
            _, record = execute(c, step, high, high)
            self.assertFalse(record['verification']['progress'])
        self.assertEqual(c.summary['repairs'][0]['exit_reason'], 'local_budget_exit_unverified')

    def test_new_containment_is_promoted_and_its_later_loss_recorded(self):
        c = NativeFramework('framework', LANGUAGE)
        before = state(closed=True)
        for step in range(180):
            execute(c, step, before, before)
        contained = state(z=.93, inside=True, table=False)
        # Losing the preserved Close predicate prevents immediate verification;
        # attaining In must nevertheless add it to preservation.
        for step in range(180, 183):
            _, record = execute(c, step, before, contained)
            before = contained
        self.assertIn('in_goal', c.summary['repairs'][0]['preserve'])
        self.assertIn('preservation_promoted', [e['type'] for e in record['events']])
        _, record = execute(c, 183, contained, state(z=.93, table=False))
        self.assertIn('in_goal', record['verification']['lost_entry_goals'])
        self.assertFalse(record['verification']['repair_verified'])

    def test_entry_goal_violation_blocks_recovery_even_if_restored(self):
        c = NativeFramework('framework', LANGUAGE)
        before = state(closed=True)
        for step in range(180):
            execute(c, step, before, before)
        after = state(z=.93, table=False, closed=False)
        _, record = execute(c, 180, before, after)
        self.assertEqual(record['verification']['lost_entry_goals'], ['close_goal'])
        before = after
        for step in range(181, 185):
            after = state(z=.93, table=False, closed=True)
            _, record = execute(c, step, before, after)
            self.assertFalse(record['verification']['repair_verified'])
            before = after

    def test_same_failed_context_is_not_retried_but_fresh_motion_can_rearm(self):
        c = NativeFramework('framework', LANGUAGE)
        start, high = contact_entry(c)
        before = high
        for step in range(start, 65):
            decision, _ = execute(c, step, before, high)
            if step >= 53:
                self.assertEqual(decision['intervention_blocked'], 'failed_repair_requires_changed_evidence')
        self.assertEqual(len(c.summary['repairs']), 1)
        changed = state(z=.93, x=.03, table=False, contact=True)
        for step in range(65, 79):
            decision, _ = execute(c, step, before, changed)
            before = changed
        self.assertEqual(len(c.summary['repairs']), 2)

    def test_budget_tie_remains_unverified_and_completion_is_separate(self):
        for complete in (False, True):
            c = NativeFramework('framework', LANGUAGE)
            start, before = contact_entry(c)
            for offset in range(32):
                after = (state(z=.93, grasp=False, inside=True, table=False,
                               closed=complete and offset == 31) if offset >= 29 else before)
                _, record = execute(c, start+offset, before, after)
                before = after
            self.assertFalse(record['verification']['repair_verified'])
            repair = c.summary['repairs'][0]
            self.assertEqual(repair['completed_during_repair'], complete)
            self.assertIsNone(repair['handoff_step'])
            self.assertEqual(repair['exit_reason'], 'native_completion_during_repair'
                if complete else 'local_budget_exit_unverified')

    def test_verified_handoff_does_not_hide_later_goal_loss(self):
        c = NativeFramework('framework', LANGUAGE)
        start, before = contact_entry(c)
        contained = state(z=.93, grasp=False, inside=True, table=False)
        for step in range(start, start+3):
            execute(c, step, before, contained)
            before = contained
        execute(c, start+3, contained, contained)
        escaped = state(grasp=False, x=.1)
        _, record = execute(c, start+4, contained, escaped)
        self.assertIn('post_handoff_goal_loss', [e['type'] for e in record['events']])
        self.assertEqual(c.before(start+5, escaped)['repair_id'], 2)

    def test_changed_contexts_cannot_exceed_total_repair_cap(self):
        c = NativeFramework('framework', LANGUAGE)
        start, before = contact_entry(c)
        for step in range(start, 400):
            # Each displacement is measured input, never a generated action.
            after = state(z=.93, x=.03*((step-start)//50), table=False, contact=True)
            execute(c, step, before, after)
            before = after
        self.assertEqual(c.summary['repair_actions'], 160)
        self.assertEqual(len(c.summary['repairs']), 5)
        self.assertEqual(c.before(400, before)['intervention_blocked'], 'total_repair_budget')

    def test_lifetime_budget_baseline_actions_and_no_silent_action_edits(self):
        c = NativeFramework('framework', LANGUAGE)
        baseline = NativeFramework('baseline', LANGUAGE)
        for step in range(520):
            decision, _ = execute(c, step, state(), state())
            shadow, _ = execute(baseline, step, state(), state())
            self.assertEqual(shadow['language'], LANGUAGE)
            self.assertEqual(shadow['owner'], 'nominal')
            self.assertFalse(any(k in decision for k in ('action', 'gripper', 'waypoints')))
        self.assertEqual(c.summary['repair_actions'], 32)
        self.assertEqual(c.summary['queries'], 520)
        self.assertEqual(c.summary['actions'], 520)
        with self.assertRaises(RuntimeError):
            c.before(520, state())

    def test_missing_evidence_and_execution_call_contract(self):
        c = NativeFramework('framework', LANGUAGE)
        missing = state()
        missing.pop('contacts_endpoint')
        decision = c.before(0, missing)
        self.assertIsNone(decision['evidence']['state']['held_clear'])
        with self.assertRaises(ValueError):
            c.before(0, missing)
        with self.assertRaises(ValueError):
            c.after(0, missing, ACTION, dict(decision, language='tampered'))
        c.after(0, missing, ACTION, decision)
        with self.assertRaises(ValueError):
            c.after(0, missing, ACTION, decision)


class ResumptionFactorialTests(unittest.TestCase):
    def enter_acquire(self, variant):
        c = NativeFramework('framework', LANGUAGE, variant=variant)
        high = state(z=.93, table=False)
        lost = state(z=.93, table=False, grasp=False, contact=True)
        execute(c, 0, state(), high)
        execute(c, 1, high, lost)
        return c, lost

    def test_factors_are_independent_and_baseline_keeps_full_language(self):
        for variant, prompt, gate in [('v2', False, False), ('prompt_only', True, False),
                                     ('gate_only', False, True), ('combined', True, True)]:
            c, before = self.enter_acquire(variant)
            inside = state(z=.93, grasp=False, table=False, contact=True,
                           inside=True, gripper_contact=True)
            execute(c, 2, before, inside)
            decision, _ = execute(c, 3, inside, inside)
            self.assertEqual(decision['language'], 'put the yellow and white mug in the microwave'
                             if prompt else 'release the yellow and white mug in the microwave')
            self.assertEqual(c.config['resumption_factor'], gate)
            baseline = NativeFramework('baseline', LANGUAGE, variant=variant)
            d, _ = execute(baseline, 0, before, inside)
            self.assertEqual(d['owner'], 'nominal')
            self.assertEqual(d['language'], LANGUAGE)

    def test_root4_like_local_containment_cannot_verify_while_withdrawing(self):
        for variant in ('gate_only', 'combined'):
            c, before = self.enter_acquire(variant)
            for step, x in enumerate((.008, .011, .004), 2):
                after = state(z=.93, x=x, inside=True, grasp=False, table=False,
                              contact=True, gripper_contact=True)
                _, record = execute(c, step, before, after)
                before = after
            self.assertTrue(record['verification']['local_condition_met'])
            self.assertFalse(record['verification']['repair_verified'])
            self.assertFalse(record['verification']['resumption_checks']['contained']['stable'])
            self.assertEqual(c.summary['repairs'][0]['local_recovery_step'], 4)
            self.assertIsNone(c.summary['repairs'][0]['handoff_step'])

    def test_stationary_one_sided_gripper_contact_is_not_held_or_released(self):
        c, before = self.enter_acquire('gate_only')
        after = state(z=.93, inside=True, grasp=False, table=False,
                      contact=True, gripper_contact=True)
        for step in range(2, 7):
            _, record = execute(c, step, before, after)
            before = after
        self.assertTrue(record['verification']['resumption_checks']['contained']['stable'])
        self.assertFalse(record['verification']['exit_condition_met'])

    def test_stable_contained_held_and_released_are_distinct_valid_routes(self):
        for grasp, expected in [(True, 'contained_held'), (False, 'contained_released_contact')]:
            c, before = self.enter_acquire('combined')
            after = state(z=.93, inside=True, grasp=grasp, table=False, contact=True)
            for step in range(2, 5):
                _, record = execute(c, step, before, after)
                before = after
            self.assertTrue(record['verification']['repair_verified'])
            self.assertEqual(c.summary['repairs'][0]['verified_by'], 'contained')
            self.assertEqual(c.summary['repairs'][0]['resumption_by'], expected)
            self.assertIsNone(c.summary['repairs'][0]['handoff_step'])
            decision, _ = execute(c, 5, after, after)
            self.assertEqual(decision['language'], LANGUAGE)
            self.assertEqual(c.summary['repairs'][0]['handoff_step'], 5)

    def test_mixed_held_released_windows_do_not_combine(self):
        c, before = self.enter_acquire('gate_only')
        for step in range(2, 9):
            after = state(z=.93, inside=True, grasp=step%2 == 0, table=False, contact=True)
            _, record = execute(c, step, before, after)
            before = after
        self.assertTrue(record['verification']['local_condition_met'])
        self.assertFalse(record['verification']['repair_verified'])

    def test_unknown_contact_position_and_fast_detached_motion_block_readiness(self):
        for missing in ('position', 'contact', 'fast'):
            c, before = self.enter_acquire('gate_only')
            for step in range(2, 6):
                after = state(z=.93, inside=True, grasp=False, table=False, contact=True,
                              x=.004*step if missing == 'fast' else 0.)
                if missing == 'position':
                    after.pop('object_observations')
                if missing == 'contact':
                    after.pop('contacts_endpoint')
                _, record = execute(c, step, before, after)
                before = after
            self.assertFalse(record['verification']['repair_verified'])

    def test_free_held_recovery_can_handoff_without_full_placement(self):
        c, before = self.enter_acquire('combined')
        after = state(z=.93, inside=False, table=False, contact=False)
        for step in range(2, 5):
            _, record = execute(c, step, before, after)
            before = after
        self.assertTrue(record['verification']['repair_verified'])
        self.assertEqual(c.summary['repairs'][0]['resumption_by'], 'free_held')

    def test_motion_window_must_be_inside_repair_and_budget_tie_stays_unverified(self):
        c = NativeFramework('framework', LANGUAGE, variant='combined')
        start, before = contact_entry(c)
        for offset in range(32):
            after = state(z=.93, inside=True, grasp=False, table=False, contact=True) if offset >= 29 else before
            _, record = execute(c, start+offset, before, after)
            if offset == 0:
                self.assertFalse(record['verification']['resumption_checks']['contained']['complete_window'])
            before = after
        self.assertTrue(record['verification']['exit_condition_met'])
        self.assertFalse(record['verification']['repair_verified'])
        self.assertEqual(c.summary['repairs'][0]['exit_reason'], 'local_budget_exit_unverified')

    def test_thirty_cells_have_one_lane_per_concurrent_job_and_balanced_roots(self):
        source = Path(__file__).resolve().parent/'run_framework_comparison.py'
        spec = importlib.util.spec_from_file_location('comparison_schedule_test', source)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        cells = runner.cohort_cells([dict(baseline=f'reference_{r}') for r in range(6)])
        waves = runner.launch_waves(cells)
        self.assertEqual([c for wave in waves for c in wave], cells)
        self.assertEqual(len({c['name'] for c in cells}), len(cells))
        for wave in waves:
            self.assertLessEqual(len(wave), 2)
            self.assertEqual(len({c['lane']['physical_uuid'] for c in wave}), len(wave))
            self.assertEqual(len({c['root'] for c in wave}), 1)
            self.assertTrue(all(c['lane']['gpu'] in (2, 3) for c in wave))
        for cohort in runner.COHORTS:
            selected = [c for c in cells if c['cohort'] == cohort]
            self.assertEqual(sum(c['lane']['gpu'] == 2 for c in selected), 3)
            self.assertEqual(sum(c['lane']['gpu'] == 3 for c in selected), 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
