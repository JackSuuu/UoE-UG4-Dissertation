"""Candidate scope, action provenance, score isolation and outcome censoring."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.repair.candidate_pool import CONFIG, context, requests, select
from src.training.recovery_dataset import record


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.decision = dict(owner='repair', language='pick up the mug', recovery_goal='acquire',
            repair_id=1, trigger='stagnation')
        self.context = context(self.decision, {'grasp': False}, 'a'*64, [[0]*7]*8)
        self.candidates = [dict(candidate_id=i, decode=dict(valid=True, action=[i/10]*6+[1.])) for i in range(4)]

    def test_nominal_cannot_generate_or_score_context(self):
        for fn in (lambda: requests(dict(self.decision, owner='nominal'), 0, 254),
                   lambda: context(dict(self.decision, owner='nominal'), {}, '', [])):
            with self.assertRaises(ValueError): fn()

    def test_requests_are_fixed_and_budgeted(self):
        r = requests(self.decision, 0, 254)
        self.assertEqual((r[0]['source'], r[0]['seed']), ('greedy', 0))
        self.assertEqual(len({v['seed'] for v in r}), 4)
        self.assertEqual(r, requests(self.decision, 0, 254))
        self.assertNotEqual(r[1]['seed'], requests(self.decision, 4, 254)[1]['seed'])
        self.assertEqual(CONFIG['total_query_cap'], CONFIG['action_cap']+len(r)-1)

    def test_collection_keeps_action_and_has_no_fake_score(self):
        decode, log = select(self.candidates, self.context, collection_slot=2)
        self.assertEqual(decode, self.candidates[2]['decode'])
        self.assertIsNone(log['scores'])
        decode['action'][0] = 999
        self.assertEqual(self.candidates[2]['decode']['action'][0], .2)
        self.assertEqual(len(self.context['action_history']), 6)

    def test_scorer_receives_candidates_not_future_and_cannot_edit_them(self):
        class Scorer:
            provenance = dict(artifact_sha256='b'*64, training_split='development/train')
            def score_candidates(self, vision_feat, physics_feat, candidates,
                                 evidence_list=None):
                self.vision_feat = vision_feat
                self.physics_feat = physics_feat
                scores = [c['decode']['action'][0] for c in candidates]
                candidates[3]['decode']['action'][0] = 999
                return scores
        scorer = Scorer()
        decode, log = select(self.candidates, self.context, scorer=scorer,
            vision_feat=np.zeros(384, dtype=np.float32), physics_feat=np.zeros(24, dtype=np.float32))
        self.assertEqual(log['selected_id'], 3)
        self.assertEqual(decode['action'][0], .3)
        self.assertEqual(scorer.vision_feat.shape[0], 384)
        self.assertEqual(scorer.physics_feat.shape[0], 24)
        # scorer must not edit the executed candidate's action (deepcopy isolation)
        self.assertEqual(self.candidates[3]['decode']['action'][0], .3)

    def test_reject_bad_scores_and_ambiguous_selection(self):
        class Scorer:
            provenance = dict(artifact_sha256='b'*64, training_split='train')
            def score_candidates(self, vision_feat, physics_feat, candidates,
                                 evidence_list=None): return [float('nan')]*4
        with self.assertRaises(ValueError): select(self.candidates, self.context, scorer=Scorer(),
            vision_feat=np.zeros(384, dtype=np.float32), physics_feat=np.zeros(24, dtype=np.float32))
        with self.assertRaises(ValueError): select(self.candidates, self.context, scorer=Scorer(),
            vision_feat=np.zeros(384, dtype=np.float32), physics_feat=np.zeros(24, dtype=np.float32))
        with self.assertRaises(ValueError): select(self.candidates, self.context)
        with self.assertRaises(ValueError): select(self.candidates, self.context, collection_slot=4)
        with self.assertRaises(ValueError): select(self.candidates, self.context, scorer=Scorer(),
            vision_feat=np.zeros(384, dtype=np.float32), physics_feat=np.zeros(24, dtype=np.float32),
            collection_slot=0)

    def fixture(self):
        _, selection = select(self.candidates, self.context, collection_slot=1)
        steps = [dict(step=0, candidate_selection=selection, decision=self.decision,
            action=self.candidates[1]['decode']['action'], before={'state':'same'})]
        repair = dict(repair_id=1, local_recovery_step=0, verified=True,
            handoff_step=1, verified_by='contained', post_handoff_goal_loss_step=None,
            preserve=['close_goal'], completed_during_repair=False, completed_after_handoff=False)
        status = dict(valid_episode=True, any_native_success=False, cell={'root':0},
            controller_summary={'repairs':[repair]}, plan_sha256='c'*64, actor_build={},
            actor_queries=12, steps_executed=9, candidate_extra_queries=3)
        controls = [dict(event='after', step=i, decision={'owner':'nominal'},
            recovery_dataset_state={'in_goal':True, 'close_goal':True}) for i in range(1,9)]
        return status, steps, controls

    def test_handoff_is_not_task_completion(self):
        status, steps, controls = self.fixture()
        r = record(status, steps, controls, '/example', True, True)
        self.assertTrue(r['labels']['stable_nominal_continuation'])
        self.assertFalse(r['labels']['complete_task_success'])
        self.assertEqual(r['executed_candidate_id'], 1)
        self.assertEqual(r['split_group'], 'microwave_root0')

    def test_incomplete_window_and_missing_measurements_are_unknown(self):
        status, steps, controls = self.fixture()
        self.assertIsNone(record(status, steps, controls[:3], '/example', True, True)
            ['labels']['stable_nominal_continuation'])
        controls[2]['recovery_dataset_state']['in_goal'] = None
        self.assertIsNone(record(status, steps, controls, '/example', True, True)
            ['labels']['stable_nominal_continuation'])

    def test_preservation_loss_or_reentry_fails_continuation(self):
        status, steps, controls = self.fixture()
        controls[1]['recovery_dataset_state']['close_goal'] = False
        self.assertFalse(record(status, steps, controls, '/example', True, True)
            ['labels']['stable_nominal_continuation'])
        controls[1]['decision']['owner'] = 'repair'
        self.assertFalse(record(status, steps, controls[:2], '/example', True, True)
            ['labels']['stable_nominal_continuation'])

    def test_invalid_execution_is_not_a_negative_training_label(self):
        status, steps, controls = self.fixture()
        r = record(status, steps, controls, '/example', False, True)
        self.assertFalse(r['eligible_for_training'])
        self.assertTrue(all(v is None for v in r['labels'].values()))

    def test_unverified_exit_is_not_stable_recovery(self):
        status, steps, controls = self.fixture()
        status['controller_summary']['repairs'][0]['verified'] = False
        r = record(status, steps, controls, '/example', True, True)
        self.assertFalse(r['labels']['verified_exit'])
        self.assertTrue(r['labels']['actual_handoff'])
        self.assertIsNone(r['labels']['stable_nominal_continuation'])


if __name__ == '__main__':
    unittest.main()
