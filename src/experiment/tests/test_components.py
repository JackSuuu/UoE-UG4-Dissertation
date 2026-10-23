"""Generic component wiring tests; fixtures are not production model fallbacks."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.experiment.tests.test_task_runtime import (IN, CLOSED, GRASP, PNG, task, recovery, frame,
                                                   Environment, Policy, FixturePlanner)
from src.controller import Limits
from src.memory.keyframe_bank import KeyframeBank
from src.perception.vision_subgoal_inference import VisionSubgoalInference
from src.prediction.short_term_predictor import ShortTermMotionPredictor
from src.runtime import Session
from src.training.handoff_dataset import LABEL_SCHEMA, continuation_label
from src.training.recovery_dataset import recovery_label, regression_label
from src.training.train_fusion_mlp import validate_rows, threshold_metrics
from src.verification.features import FEATURE_DIM, FEATURE_SCHEMA, encoder_signature, fusion_features
from src.verification.physical_verifier import PhysicalVerifier
from src.verification.vision_physical_monitor import VisionPhysicalMonitor
from src.journal import Journal


VISUAL = (1.,) + (0.,)*383
PROVENANCE = dict(sha256='fixture', source_sha256='fixture_source', transform='fixture_rgb', feature_dim=384)
SIGNATURE = encoder_signature(PROVENANCE)


class FixtureEncoder:
    provenance = PROVENANCE

    def __init__(self):
        self.calls = 0

    def encode_png(self, image):
        self.calls += 1
        return VISUAL


class FixtureRisk:
    provenance = {'fixture': True}

    def score(self, features):
        return dict(score=.9, threshold=.5, exceeds=True)


def physics(observation):
    return PhysicalVerifier().evidence(observation, target='item')


class MemoryTests(unittest.TestCase):
    def bank(self):
        bank = KeyframeBank(encoder_signature=SIGNATURE)
        bank.write(VISUAL, predicates=(IN,), instruction=task().instruction,
                   root=10, split='training', measured_success=True)
        return bank

    def test_training_only_strict_dimension_and_measured_success(self):
        bank = KeyframeBank()
        kwargs = dict(predicates=(IN,), instruction='place it', root=10,
                      split='training', measured_success=True)
        for change in (dict(root=6), dict(root=20), dict(split='evaluation'), dict(measured_success=False)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                bank.write(VISUAL, **dict(kwargs, **change))
        for vector in ((0.,)*384, (1.,)*383, (float('nan'),)*384):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                bank.write(vector, **kwargs)

    def test_retrieval_requires_predicate_binding_and_does_not_mutate_bank(self):
        bank = self.bank()
        matches = bank.retrieve(VISUAL, (IN,))
        self.assertEqual(matches[0]['root'], 10)
        self.assertEqual(matches[0]['similarity'], 1.)
        self.assertEqual(bank.retrieve(VISUAL, (CLOSED,)), [])
        matches[0]['predicates'].clear()
        self.assertTrue(bank.retrieve(VISUAL, (IN,)))

    def test_memory_save_load_rejects_old_schema_and_mismatched_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'memory.json'
            self.bank().save(path)
            bank = KeyframeBank.load(path, encoder_signature=SIGNATURE)
            self.assertEqual(len(bank.retrieve(VISUAL, (IN,))), 1)
            with self.assertRaises(ValueError):
                KeyframeBank.load(path, encoder_signature=dict(SIGNATURE, sha256='different'))
            path.write_text(json.dumps(dict(schema='old_stage_bank')))
            with self.assertRaises(ValueError):
                KeyframeBank.load(path, encoder_signature=SIGNATURE)

    def test_visual_similarity_never_overrides_unknown_measured_predicate(self):
        inferred = VisionSubgoalInference().infer_features(VISUAL, task(), frame(), [], self.bank())
        self.assertEqual(inferred['predicates'][0]['similarity'], 1.)
        self.assertIsNone(inferred['predicates'][0]['measured'])
        self.assertFalse(inferred['visual_is_verification'])


class PredictionTests(unittest.TestCase):
    def test_causal_constant_velocity_forecast_not_action_blended(self):
        history = [frame(i, position=(i*.01, 0., 0.)) for i in range(4)]
        report = ShortTermMotionPredictor(horizon_steps=2).predict(history[-1], history)
        self.assertAlmostEqual(report['objects']['item']['positions'][0][0], .04)
        self.assertAlmostEqual(report['objects']['item']['positions'][1][0], .05)
        self.assertFalse(report['action_conditioned'])
        self.assertFalse(report['verification'])

    def test_missing_or_nonconsecutive_history_stays_unknown(self):
        predictor = ShortTermMotionPredictor()
        for history in ([frame(0)], [frame(0), frame(3)]):
            report = predictor.predict(history[-1], history)
            self.assertFalse(report['objects']['item']['valid'])


class MonitorTests(unittest.TestCase):
    def test_shared_encoder_cache_and_persistence_count_executed_frames_only(self):
        encoder = FixtureEncoder()
        monitor = VisionPhysicalMonitor(encoder=encoder, risk=FixtureRisk(), mode='gate')
        for _ in range(4):
            report = monitor.observe(frame(0), task(), physics(frame(0)))
        self.assertEqual(encoder.calls, 1)
        self.assertFalse(report['context']['trigger'])
        monitor.observe(frame(1), task(), physics(frame(1)))
        report = monitor.observe(frame(2), task(), physics(frame(2)))
        self.assertTrue(report['context']['trigger'])
        self.assertEqual(encoder.calls, 3)
        self.assertEqual(len(report['features']), FEATURE_DIM)

    def test_log_mode_has_no_intervention_or_handoff_veto(self):
        monitor = VisionPhysicalMonitor(encoder=FixtureEncoder(), risk=FixtureRisk(), mode='log', persistence=1)
        report = monitor.observe(frame(), task(), physics(frame()))['context']
        self.assertTrue(report['would_trigger'])
        self.assertFalse(report['trigger'])
        self.assertFalse(report['handoff_veto'])

    def test_live_risk_changes_controller_and_context_reaches_planner(self):
        bank = KeyframeBank(encoder_signature=SIGNATURE)
        bank.write(VISUAL, predicates=(IN,), instruction='place it', root=10,
                   split='training', measured_success=True)
        monitor = VisionPhysicalMonitor(encoder=FixtureEncoder(), memory=bank,
                                       risk=FixtureRisk(), mode='gate')
        planner, policy = FixturePlanner(), Policy()
        env = Environment([{IN: False, CLOSED: False}]*10)
        result = Session(env, policy, planner, monitor=monitor,
                         limits=Limits(actions=6, queries=6, stagnation=100)).run('place it and close it')
        self.assertEqual(result['controller']['repairs'][0]['entry'], 2)
        event = planner.recoveries[0][1]
        self.assertEqual(event['reason'], 'persistent_learned_risk')
        self.assertEqual(event['context']['memory'][0]['root'], 10)
        self.assertTrue(event['context']['prediction']['objects']['item']['valid'])
        self.assertEqual(env.executed, [p.action for p in policy.proposals])

    def test_baseline_never_repairs_even_with_a_logged_high_risk(self):
        planner = FixturePlanner()
        result = Session(Environment([{IN: False, CLOSED: False}]*8), Policy(), planner, mode='baseline',
            monitor=VisionPhysicalMonitor(encoder=FixtureEncoder(), risk=FixtureRisk(), mode='log'),
            limits=Limits(actions=5, queries=5, stagnation=1)).run('place it and close it')
        self.assertEqual(planner.recoveries, [])
        self.assertEqual(result['controller']['budget']['repair_actions'], 0)

    def test_learned_risk_veto_rejects_real_handoff_without_execution(self):
        class ContainedRisk(FixtureRisk):
            def score(self, features):
                high = features[384] > .5
                return dict(score=.9 if high else .1, threshold=.5, exceeds=high)
        monitor = VisionPhysicalMonitor(encoder=FixtureEncoder(), risk=ContainedRisk(), mode='gate')
        env, policy, records = Environment([{IN: False, CLOSED: False}]*2+[{IN: True, CLOSED: False}]*12), Policy(), []
        result = Session(env, policy, FixturePlanner(), monitor=monitor, record=records.append,
                         limits=Limits(actions=8, queries=12, local_repair=6, stagnation=1)).run('place it and close it')
        vetoes = [r for r in records if r['type'] == 'handoff_assessment' and not r['assessment']['accept']]
        self.assertTrue(vetoes)
        for row in vetoes:
            self.assertNotIn(tuple(row['assessment']['nominal_action']), env.executed)
        self.assertFalse(result['controller']['repairs'][0]['verified'])

    def test_malformed_risk_and_incompatible_memory_rejected(self):
        class BadRisk(FixtureRisk):
            def score(self, features):
                return dict(score=float('nan'), threshold=.5, exceeds=True)
        with self.assertRaises(ValueError):
            VisionPhysicalMonitor(encoder=FixtureEncoder(), risk=BadRisk()).observe(frame(), task(), physics(frame()))
        with self.assertRaises(ValueError):
            VisionPhysicalMonitor(memory=KeyframeBank())
        with self.assertRaises(ValueError):
            VisionPhysicalMonitor(encoder=FixtureEncoder(), memory=KeyframeBank())


class FeatureAndLabelTests(unittest.TestCase):
    def test_features_consume_generated_clauses_and_exact_action(self):
        current = frame(1, {IN: False, GRASP: True})
        evidence = physics(current)
        original = deepcopy(evidence)
        action = (.1, .2, .3, .4, .5, .6, -1.)
        vector = fusion_features(VISUAL, current, task(), evidence, plan=recovery(), action=action)
        self.assertEqual(len(vector), FEATURE_DIM)
        self.assertEqual(vector[-8:], (*action, 1.))
        self.assertNotEqual(vector, fusion_features(VISUAL, current, task((GRASP,)), evidence,
                                                   plan=recovery(), action=action))
        self.assertEqual(evidence, original)
        self.assertEqual(len(evidence['values']), 4)

    def test_continuation_requires_actual_nominal_future_and_same_alternative(self):
        start = frame(3, {IN: True, GRASP: False})
        row = dict(owner='nominal', observation=frame(4, {IN: False, GRASP: True}), complete=False)
        self.assertEqual(continuation_label(recovery(((IN,), (GRASP,))), start, [row])[0], 0)
        row['owner'] = 'repair'
        self.assertIsNone(continuation_label(recovery(), start, [row])[0])
        row = dict(owner='nominal', observation=frame(4, {IN: True}), complete=True)
        self.assertEqual(continuation_label(recovery(), start, [row])[0], 1)
        row['complete'] = False
        row['observation'] = frame(4)
        self.assertIsNone(continuation_label(recovery(), start, [row])[0])

    def test_regression_label_is_future_nominal_loss_not_repair_entry_success(self):
        start = frame(3, {IN: True})
        future = [dict(owner='nominal', observation=frame(i, {IN: False}), complete=False) for i in (4, 5, 6)]
        self.assertEqual(regression_label(task(), start, future)[0], 1)
        future[1]['owner'] = 'repair'
        self.assertIsNone(regression_label(task(), start, future)[0])

    def test_completion_absorbs_local_contract_loss_but_not_unobserved_future(self):
        start = frame(3, {IN: True})
        completed = dict(owner='nominal', observation=frame(4), complete=True)
        self.assertEqual(continuation_label(recovery(), start, [completed])[0], 1)
        self.assertEqual(regression_label(task(), start, [completed])[0], 0)
        self.assertEqual(recovery_label(dict(exit_reason='task_completion_during_repair',
                                             preserve_lost=True, verified=False))[0], 1)
        for change in (dict(owner='repair'), dict(observation=frame(6))):
            row = dict(completed, **change)
            self.assertIsNone(continuation_label(recovery(), start, [row])[0])
            self.assertIsNone(regression_label(task(), start, [row])[0])

    def test_dataset_rejects_reserved_roots_mixed_encoders_and_duplicate_samples(self):
        row = dict(schema=FEATURE_SCHEMA, label_schema=LABEL_SCHEMA, role='risk', root=10,
                   split='training', label=0, sample_id='fixture', features=[0.]*FEATURE_DIM,
                   encoder_signature=SIGNATURE)
        self.assertEqual(validate_rows([row], 'risk'), SIGNATURE)
        for change in (dict(root=6), dict(root=20), dict(features=[0.]), dict(label=True)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_rows([dict(row, **change)], 'risk')
        with self.assertRaises(ValueError):
            validate_rows([row, row], 'risk')
        with self.assertRaises(ValueError):
            validate_rows([row, dict(row, sample_id='other', encoder_signature=dict(SIGNATURE, sha256='other'))], 'risk')

    def test_threshold_calibration_requires_both_classes(self):
        self.assertEqual(threshold_metrics([.1, .9], [0, 1])['fpr'], 0.)
        with self.assertRaises(ValueError):
            threshold_metrics([.1, .2], [0, 0])


class ExperimentTests(unittest.TestCase):
    def test_fixed_slot_collection_is_training_only_and_excluded_from_measured(self):
        from src.experiment.microwave import parse_args
        base = ['--out', 'new-run', '--policy-checkpoint', '/models/vla',
                '--planner-url', 'http://localhost/v1', '--planner-model', 'vlm', '--candidate-slot', '2']
        for arm, root in (('measured', 10), ('context', 0), ('context', 20)):
            with self.subTest(arm=arm, root=root), patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--arm', arm, '--init-index', str(root), *base])
        forwarded = parse_args(['--arm', 'context', '--init-index', '10', *base,
                                '--visual-checkpoint', 'dino.pt', '--visual-source', 'dino'])
        self.assertIn('--candidate-slot', forwarded)

    def test_summary_uses_native_outcome_and_rejects_mismatched_pairs(self):
        from src.experiment.summarize import summarize
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'root0'
            status = dict(status='completed', initial_states_sha256='state', bddl_sha256='scene',
                instruction='place it', benchmark_instruction='place it', policy={'sha256': 'policy'},
                limits={'actions': 520}, source_sha256={'runtime.py': 'source'},
                args=dict(init_index=0, seed=0, planner_url='http://localhost/v1', planner_model='vlm',
                          planner_calls=24, planner_timeout=60, rollout_seconds=1200))
            result = dict(native_success=False, termination='generated_task_complete',
                controller=dict(complete=True, budget=dict(actions=1, queries=1), repairs=[],
                                task=dict(goals=[IN.to_dict()], milestones=[])))
            for arm in ('baseline', 'context'):
                directory = root/arm
                directory.mkdir(parents=True)
                (directory/'status.json').write_text(json.dumps(status))
                (directory/'result.json').write_text(json.dumps(result))
            report = summarize(Path(tmp), 'context')
            self.assertEqual(report['counts']['both_failure'], 1)
            result['native_success'] = True
            (root/'context/result.json').write_text(json.dumps(result))
            self.assertEqual(summarize(Path(tmp), 'context')['counts']['rescue'], 1)
            for field, value in (('seed', 1), ('planner_model', 'different')):
                changed = deepcopy(status)
                changed['args'][field] = value
                (root/'context/status.json').write_text(json.dumps(changed))
                self.assertEqual(summarize(Path(tmp), 'context')['counts']['unpaired_or_infrastructure'], 1)

    def test_new_microwave_protocol_requires_real_full_artifacts(self):
        from src.experiment.microwave import parse_args
        base = ['--out', 'new-run', '--policy-checkpoint', '/models/vla',
                '--planner-url', 'http://localhost/v1', '--planner-model', 'vlm']
        forwarded = parse_args(['--arm', 'measured', *base])
        self.assertIn('libero_10', forwarded)
        with patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--arm', 'full', *base])
        with patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--arm', 'context', *base, '--visual-checkpoint', 'dino.pt', '--visual-source', 'dino',
                        '--risk-checkpoint', 'risk.pt'])

    def test_completed_training_episode_datasets_and_memory_pipeline(self):
        from src.experiment.dataset import collect
        from src.experiment.build_memory import build
        from src.experiment.summarize import summarize
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/'episode'
            journal = Journal(directory)
            states = [{IN: False, CLOSED: False}]*2+[{IN: True, CLOSED: False}]*4+[{IN: True, CLOSED: True}]
            result = Session(Environment(states), Policy(), FixturePlanner(),
                monitor=VisionPhysicalMonitor(encoder=FixtureEncoder()), record=journal.record,
                limits=Limits(actions=10, queries=10, stagnation=1)).run('place it and close it')
            journal.save('result.json', result)
            journal.save('status.json', dict(status='completed', args={'init_index': 10}))
            journal.close()
            for role in ('risk', 'recovery', 'handoff'):
                rows, _ = collect(directory, role)
                self.assertTrue(rows, role)
                self.assertEqual(rows[0]['root'], 10)
                self.assertEqual(len(rows[0]['features']), FEATURE_DIM)
                self.assertEqual(rows[0]['role'], role)
            bank = build([directory])
            self.assertTrue(bank.retrieve(VISUAL, (IN,)))
            (directory/'status.json').write_text(json.dumps(dict(status='failed', args={'init_index': 10})))
            with self.assertRaises(ValueError):
                collect(directory, 'risk')
            (directory/'status.json').write_text(json.dumps(dict(status='completed', args={'init_index': 20})))
            with self.assertRaises(ValueError):
                collect(directory, 'handoff')
            pair_root = Path(tmp)/'root0'
            (pair_root/'context').mkdir(parents=True)
            (pair_root/'context/status.json').write_text(json.dumps(dict(status='failed', error='endpoint down')))
            report = summarize(Path(tmp), 'context')
            self.assertEqual(report['counts']['unpaired_or_infrastructure'], 1)
            self.assertEqual(report['counts']['both_failure'], 0)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional CPU torch backend is not installed')
class LearnedHeadTests(unittest.TestCase):
    def test_cpu_train_save_load_role_schema_and_encoder_checks(self):
        from src.training.train_fusion_mlp import main
        from src.verification.fusion_mlp import FusionMLP
        from src.verification.calibrated_risk_head import CalibratedRiskHead
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            rows = []
            for root in (10, 18):
                for label in (0, 1):
                    rows.append(dict(schema=FEATURE_SCHEMA, label_schema=LABEL_SCHEMA, role='risk', root=root,
                        split='training', label=label, sample_id=f'fixture:{root}:{label}',
                        features=[float(label)]*FEATURE_DIM, encoder_signature=SIGNATURE))
            dataset = directory/'data.jsonl'
            dataset.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            main(['--dataset', str(dataset), '--role', 'risk', '--out', str(directory/'head'), '--epochs', '2'])
            checkpoint = directory/'head/model.pt'
            head = CalibratedRiskHead(checkpoint, encoder_signature=SIGNATURE)
            self.assertIsInstance(head.score(rows[0]['features'])['exceeds'], bool)
            from src.repair.recovery_scorer import RecoveryScorer
            from src.verification.handoff_verifier import HandoffVerifier
            monitor = VisionPhysicalMonitor(encoder=FixtureEncoder())
            current = frame(3, {IN: True})
            monitor.observe(current, task(), physics(current))
            for role in ('recovery', 'handoff'):
                for row in rows:
                    row['role'] = role
                dataset.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                main(['--dataset', str(dataset), '--role', role, '--out', str(directory/role), '--epochs', '2'])
                path = directory/role/'model.pt'
                if role == 'recovery':
                    scorer = RecoveryScorer(path, monitor)
                    scores = scorer.score(current, task(), recovery(), [(0.,)*7]*4, physics(current))
                    self.assertEqual(len(scores), 4)
                    self.assertTrue(all(0 <= score <= 1 for score in scores))
                else:
                    assessment = HandoffVerifier(path, monitor).assess(current, task(), recovery(),
                                                                      (0.,)*7, physics(current))
                    self.assertEqual(assessment['accept'], assessment['score'] > assessment['threshold'])
            for kwargs in (dict(role='handoff', encoder_signature=SIGNATURE),
                           dict(role='risk', encoder_signature=dict(SIGNATURE, sha256='other'))):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    FusionMLP.load(checkpoint, **kwargs)
            import torch
            torch.save(dict(risk_schema='old_microwave_stage'), directory/'old.pt')
            with self.assertRaises(ValueError):
                FusionMLP.load(directory/'old.pt', role='risk', encoder_signature=SIGNATURE)


if __name__ == '__main__':
    unittest.main()
