"""Causal connection checks, without starting the frozen actor or a simulator."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from src.controller import NativeFramework
from src.perception.vision_subgoal_inference import VisionSubgoalInference
from src.repair.receding_horizon_repair import RecedingHorizonRepair
from src.training.train_joint_components import risk_label
from src.verification.joint_runtime import JOINT_SCHEMA, JointRuntime
from src.verification.fusion_mlp import FEATURE_SCHEMA
from test_native_framework import state, ACTION, LANGUAGE


def evidence(index, stage='acquire', trigger=None):
    return dict(schema=JOINT_SCHEMA, observation_index=index, selected_stage=stage, trigger=trigger,
        memory=dict(selected_stage=stage), visual=dict(source='real_test_features'), risk={}, prediction={})


class JointTests(unittest.TestCase):
    def test_no_silent_fallback_or_stale_evidence(self):
        c = NativeFramework('framework', LANGUAGE, fusion_mode='gate', joint_mode=True)
        for e in (None, evidence(1)):
            with self.assertRaises(ValueError):
                c.before(0, state(), joint_evidence=e)
        with self.assertRaises(ValueError):
            NativeFramework('framework', LANGUAGE, joint_mode=True)

    def test_prediction_and_risk_can_each_enter_repair(self):
        for trigger in ('joint_no_lift', 'joint_predicted_exit', 'joint_outcome_risk'):
            c = NativeFramework('framework', LANGUAGE, fusion_mode='gate', joint_mode=True)
            decision = c.before(0, state(), evidence(0, trigger=trigger))
            self.assertEqual(decision['owner'], 'repair')
            self.assertEqual(decision['trigger'], trigger)
            self.assertEqual(decision['recovery_goal'], 'acquire')

    def test_visual_memory_stage_changes_repair_request_not_contract(self):
        requests = []
        for stage in ('acquire', 'transfer'):
            c = NativeFramework('framework', LANGUAGE, fusion_mode='gate', joint_mode=True)
            decision = c.before(0, state(), evidence(0, stage, 'joint_no_lift'))
            self.assertEqual(decision['recovery_goal'], 'acquire')
            self.assertEqual(decision['recovery_acceptance'], ['held_clear', 'contained'])
            requests.append(RecedingHorizonRepair().request(decision)[0]['language'])
        self.assertNotEqual(*requests)

    def test_visual_prototypes_physically_constrained(self):
        module = VisionSubgoalInference(None)
        a, b = np.eye(384, dtype=np.float32)[:2]
        prototypes = dict(acquire=a, transfer=b, close=b)
        s = dict(in_goal=False, grasp=True)
        self.assertEqual(module.infer_features(a, s, prototypes)['subgoal'], 'acquire')
        self.assertEqual(module.infer_features(b, s, prototypes)['subgoal'], 'transfer')
        self.assertEqual(module.infer_features(b, dict(s, grasp=False), prototypes)['subgoal'], 'acquire')

    def test_joint_veto_blocks_an_accepted_mlp(self):
        c = NativeFramework('framework', LANGUAGE, fusion_mode='gate', joint_mode=True)
        before, high = state(), state(z=.93, table=False)
        for index in range(3):
            d = c.before(index, before, evidence(index, trigger='joint_no_lift'))
            f = dict(probability=.99, threshold=.8, stage='acquire', feature_schema=FEATURE_SCHEMA,
                model_sha256='a'*64, joint_prediction_veto=True, joint_risk_veto=False)
            record = c.after(index, high, ACTION, d, f)
            before = high
        self.assertTrue(record['verification']['exit_condition_met'])
        self.assertFalse(record['verification']['repair_verified'])
        self.assertTrue(record['verification']['fusion']['blocked_verified_exit'])

    def test_future_risk_labels_not_based_on_ownership(self):
        s = dict(held_clear=False, in_goal=False, close_goal=False, grasp=True, position=[0., 0., 1.])
        future = [dict(owner='nominal', state=s) for _ in range(24)]
        self.assertEqual(risk_label(s, [s]*10, future)[0], 1)
        future = copy.deepcopy(future)
        future[-1]['state']['held_clear'] = True
        self.assertEqual(risk_label(s, [s]*10, future)[0], 0)
        self.assertIsNone(risk_label(s, [s]*10, future[:5])[0])

    def test_retrieval_actually_changes_stage_and_observation_is_single_use(self):
        a, b = np.eye(384, dtype=np.float32)[:2]
        h = Mock()
        h.encoder.provenance = dict(sha256='a'*64)
        h.encoder.encode.return_value = a
        risk = Mock()
        risk.artifact = dict(encoder=h.encoder.provenance, bank_stages=['acquire', 'transfer'],
            bank_features=torch.tensor(np.stack([a, b])), bank_metadata=[dict(root=1), dict(root=2)],
            train_roots=[1, 2, 5], stage_dwell=1, risk_persistence=3, no_lift_steps=30)
        risk.score.return_value = dict(score=.1, threshold=.8, exceeds=False)
        observed = dict(in_goal=False, grasp=True, held_clear=False, table=True,
            mug_gripper=True, mug_microwave=False, gripper_microwave=False, position=[0., 0., 1.])
        choices = []
        for preferred in ('acquire', 'transfer'):
            with patch('src.verification.joint_runtime.HandoffVerifier', return_value=h), patch(
                    'src.verification.joint_runtime.OutcomeRisk', return_value=risk):
                runtime = JointRuntime('fake_fusion', 'fake_joint')
            runtime.bank.read_prototype = lambda stage: a if stage in ('acquire', 'transfer') else None
            runtime.bank.read_topk = lambda q, stage, k: [dict(similarity=.99 if stage == preferred else .1,
                stage=stage, metadata=dict(root=1))]
            result = runtime.observe(np.zeros((224, 224, 3), dtype=np.uint8), observed, 0)
            choices.append(result['selected_stage'])
            with self.assertRaises(ValueError):
                runtime.observe(np.zeros((224, 224, 3), dtype=np.uint8), observed, 0)
        self.assertEqual(choices, ['acquire', 'transfer'])


if __name__ == '__main__':
    unittest.main()
