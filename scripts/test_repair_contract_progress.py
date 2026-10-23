"""Objective-progress evidence must not confuse motion, contact or missing data."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.repair.experimental.contract_progress import ContractProgress
from src.legacy.microwave_controller import NativeFramework
from src.repair.receding_horizon_repair import RecedingHorizonRepair
from test_native_framework import state as diagnostics, ACTION, LANGUAGE
from test_joint_framework import evidence


def state(x=0., **kwargs):
    return dict(dict(position=[x, 0., .9], grasp=True, held_clear=False,
        in_goal=False, close_goal=False, table=True), **kwargs)


class ProgressTests(unittest.TestCase):
    def tracker(self):
        return ContractProgress('acquire', window=12, motion_m=.003)

    def test_stationary_unmet_contract_is_not_reset_by_table_contact_flicker(self):
        p = self.tracker()
        for index in range(12):
            result = p.observe(state(table=bool(index % 2)), dict(held_clear=False, contained=False), True)
            self.assertEqual(result['stop_unverified'], index == 11)
        self.assertEqual(result['diagnosis'], 'stationary_unmet_contract')

    def test_small_per_step_but_material_cumulative_motion_is_not_stagnation(self):
        p = self.tracker()
        for index in range(12):
            result = p.observe(state(x=index*.001), dict(held_clear=False), True)
        self.assertFalse(result['stop_unverified'])

    def test_missing_positions_or_semantics_do_not_prove_stagnation(self):
        for missing in (dict(position=None), dict(grasp=None)):
            p = self.tracker()
            for _ in range(12):
                result = p.observe(state(**missing), dict(held_clear=False), True)
            self.assertFalse(result['stop_unverified'])
            self.assertEqual(result['diagnosis'], 'missing_progress_measurements')

    def test_local_success_waiting_for_gate_is_not_classified_as_stalled(self):
        p = self.tracker()
        for _ in range(20):
            result = p.observe(state(held_clear=True), dict(held_clear=True), True)
        self.assertFalse(result['stop_unverified'])
        self.assertEqual(result['diagnosis'], 'local_condition_observed_not_handoff_verified')

    def test_recent_progress_and_semantic_transition_are_not_stagnation(self):
        for prior_success in (True, False):
            p = self.tracker()
            for i in range(12):
                result = p.observe(state(grasp=i != 0), dict(held_clear=prior_success and i == 0), True)
            self.assertFalse(result['stop_unverified'])

    def test_preservation_failure_stops_without_claiming_verified_recovery(self):
        result = self.tracker().observe(state(in_goal=False), dict(held_clear=True), False)
        self.assertTrue(result['stop_unverified'])
        self.assertEqual(result['diagnosis'], 'preservation_failed_or_unknown')


class ContractControlTests(unittest.TestCase):
    def enter(self, control):
        c = NativeFramework('framework', LANGUAGE, fusion_mode='gate', joint_mode=True,
            joint_admission='measured_only', repair_control=control)
        before = diagnostics()
        for i, post in enumerate((diagnostics(z=.93, table=False),
                                  diagnostics(z=.93, table=False, grasp=False))):
            d = c.before(i, before, evidence(i))
            c.after(i, post, ACTION, d)
            before = post
        return c, before

    def gate(self):
        return dict(probability=.1, threshold=.8, stage='acquire',
            feature_schema='controller_state_values12_validity12_v2', model_sha256='a'*64,
            joint_prediction_veto=False, joint_risk_veto=False)

    def test_measured_progress_changes_actual_request_not_acceptance_contract(self):
        requests = []
        for control in ('existing', 'measured_stage'):
            c, before = self.enter(control)
            d = c.before(2, before, evidence(2, 'acquire'))
            high = diagnostics(z=.93, table=False)
            c.after(2, high, ACTION, d, self.gate())
            d = c.before(3, high, evidence(3, 'acquire'))
            requests.append(RecedingHorizonRepair().request(d)[0]['language'])
            self.assertEqual(d['recovery_goal'], 'acquire')
            self.assertEqual(d['recovery_acceptance'], ['held_clear', 'contained'])
        self.assertNotEqual(*requests)
        self.assertEqual(d['subgoal'], 'transfer')
        self.assertTrue(d['request_diagnosis']['request_changed_from_joint'])

    def test_progress_stop_is_distinct_from_stage_policy_and_verified_exit(self):
        for control, expected_actions in (('measured_stage', 32), ('measured_stage_progress', 12)):
            c, before = self.enter(control)
            for t in range(2, 2+expected_actions):
                d = c.before(t, before, evidence(t))
                self.assertEqual(d['owner'], 'repair')
                c.after(t, before, ACTION, d, self.gate())
            repair = c.summary['repairs'][0]
            self.assertEqual(repair['used'], expected_actions)
            self.assertFalse(repair['verified'])
            self.assertIsNone(repair['local_recovery_step'])
            self.assertIsNotNone(repair['exit_step'])
            if control.endswith('progress'):
                self.assertEqual(repair['exit_reason'], 'contract_stationary_unmet_contract_exit_unverified')
            t = 2+expected_actions
            d = c.before(t, before, evidence(t))
            self.assertEqual(d['owner'], 'nominal')
            self.assertEqual(d['language'], LANGUAGE)
            c.after(t, before, ACTION, d)
            self.assertEqual(c.summary['repairs'][0]['handoff_step'], t)

    def test_fusion_wait_with_local_progress_is_not_aborted_as_stagnant(self):
        c, before = self.enter('measured_stage_progress')
        high = diagnostics(z=.93, table=False)
        for t in range(2, 22):
            d = c.before(t, before, evidence(t))
            result = c.after(t, high, ACTION, d, self.gate())
            before = high
        self.assertTrue(result['verification']['exit_condition_met'])
        self.assertTrue(result['verification']['fusion']['blocked_verified_exit'])
        self.assertIsNone(c.summary['repairs'][0]['exit_step'])

    def test_non_joint_and_oracle_modes_cannot_mix_with_contract_control(self):
        for kw in ({}, dict(joint_mode=True, fusion_mode='gate', recovery_options=True)):
            with self.assertRaises(ValueError):
                NativeFramework('framework', LANGUAGE, repair_control='measured_stage', **kw)


if __name__ == '__main__':
    unittest.main()
