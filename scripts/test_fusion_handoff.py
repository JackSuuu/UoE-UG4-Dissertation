"""CPU regression checks for outcome labels, masked fusion and causal handoff."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from src.legacy.microwave_controller import NativeFramework
from src.training.handoff_dataset import outcome_label, root_split
from src.training.train_fusion_mlp import select_threshold
from src.verification.fusion_mlp import (VerificationFusionMLP, extract_physics_features,
    save_model, load_model, FEATURE_SCHEMA)
from test_native_framework import state, execute, ACTION, LANGUAGE


class FusionTests(unittest.TestCase):
    def test_missing_is_not_false(self):
        f = extract_physics_features(dict(in_goal=False, grasp=None, position=[1., 2., 3.]))
        self.assertEqual(f.shape, (24,))
        self.assertEqual((f[0], f[12]), (0., 1.))
        self.assertEqual((f[2], f[14]), (0., 0.))
        with self.assertRaises(ValueError):
            extract_physics_features(dict(predicates={}, grasped={}))

    def test_class_one_is_pass_and_roundtrip(self):
        model = VerificationFusionMLP()
        for p in model.parameters():
            torch.nn.init.zeros_(p)
        model.net[-1].bias.data[1] = 4.
        vf = np.ones(384, dtype=np.float32)/np.sqrt(384)
        pf = extract_physics_features(dict(in_goal=False))
        self.assertGreater(model.predict_verify_probability(vf, pf, 'acquire'), .98)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'model.pt'
            save_model(model, p)
            restored = load_model(p)
            self.assertEqual(model.predict_verify_probability(vf, pf, 'acquire'),
                restored.predict_verify_probability(vf, pf, 'acquire'))
            torch.save(dict(state_dict={}), p)
            with self.assertRaises(ValueError):
                load_model(p)
        with self.assertRaises(ValueError):
            model.predict_verify_probability(np.zeros(384), pf, 'acquire')

    def test_labels_require_observed_continuation_not_owner(self):
        s = dict(held_clear=True, in_goal=False, close_goal=False)
        future = [dict(owner='nominal', state=s, native_success=False) for _ in range(12)]
        self.assertEqual(outcome_label(s, 'acquire', future)[0], 1)
        failed = copy.deepcopy(future)
        failed[7]['state']['held_clear'] = False
        self.assertEqual(outcome_label(s, 'acquire', failed)[0], 0)
        self.assertIsNone(outcome_label(s, 'acquire', future[:5])[0])
        censored = copy.deepcopy(future)
        censored[5]['owner'] = 'repair'
        self.assertIsNone(outcome_label(s, 'acquire', censored)[0])
        self.assertEqual(outcome_label(dict(s, held_clear=False), 'acquire', future)[0], 0)

    def test_preserved_goal_loss_is_negative(self):
        s = dict(held_clear=True, in_goal=False, close_goal=True)
        future = [dict(owner='nominal', state=dict(s, close_goal=False), native_success=False)]
        self.assertEqual(outcome_label(s, 'acquire', future)[0], 0)

    def test_threshold_can_decline_and_roots_never_overlap(self):
        self.assertIsNone(select_threshold([0, 1], [.9, .1]))
        self.assertEqual(select_threshold([0, 1], [.1, .9]), .9)
        self.assertEqual([root_split(i) for i in range(6)],
            ['evaluation', 'train', 'train', 'validation', 'validation', 'train'])

    def recovery(self, mode, probability=.1):
        c = NativeFramework('framework', LANGUAGE, fusion_mode=mode)
        high = state(z=.93, table=False)
        lost = state(z=.93, table=False, grasp=False)
        execute(c, 0, state(), high)
        execute(c, 1, high, lost)
        before = lost
        records = []
        for step in range(2, 6):
            decision = c.before(step, before)
            preview = c.preview_fusion_state(high)
            evidence = dict(probability=probability, threshold=.8, stage=decision['recovery_goal'],
                feature_schema=FEATURE_SCHEMA, model_sha256='a'*64)
            record = c.after(step, high, ACTION, decision, handoff_evidence=evidence)
            records.append(record)
            for k, v in preview.items():
                self.assertEqual(record['evidence']['state'][k], v)
            before = high
        return c, records

    def test_logging_is_noncausal_gate_blocks_and_accepts(self):
        off, off_rows = self.recovery('off')
        log, log_rows = self.recovery('log')
        self.assertTrue(off.summary['repairs'][0]['verified'])
        self.assertEqual(off.summary['repairs'], log.summary['repairs'])
        gate, gate_rows = self.recovery('gate')
        self.assertFalse(gate.summary['repairs'][0]['verified'])
        self.assertTrue(gate_rows[2]['verification']['fusion']['blocked_verified_exit'])
        accepted, _ = self.recovery('gate', probability=.9)
        self.assertTrue(accepted.summary['repairs'][0]['verified'])

    def test_missing_scores_fail_closed_but_caps_are_not_removed(self):
        c = NativeFramework('framework', LANGUAGE, fusion_mode='gate')
        high, lost = state(z=.93, table=False), state(z=.93, table=False, grasp=False)
        execute(c, 0, state(), high)
        execute(c, 1, high, lost)
        before = lost
        for step in range(2, 34):
            _, record = execute(c, step, before, high)
            before = high
        self.assertFalse(c.summary['repairs'][0]['verified'])
        self.assertEqual(c.summary['repairs'][0]['exit_reason'], 'local_budget_exit_unverified')


if __name__ == '__main__':
    unittest.main()
