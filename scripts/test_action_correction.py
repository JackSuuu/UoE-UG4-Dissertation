"""CPU regression tests for the opt-in numeric-correction development protocol."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_native_long_branch import validate_action
from src.verification.physical_verifier import extract_physics_state
from src.prediction.short_term_predictor import ShortTermMotionPredictor


class PackageLayoutTests(unittest.TestCase):
    def test_runtime_imports_and_pinned_sources(self):
        from run_framework_comparison import REPO, SOURCES, _load_v5_modules, framework_class
        from src.controller import NativeFramework
        self.assertEqual(framework_class().__name__, NativeFramework.__name__)
        self.assertIn('src/controller.py', SOURCES)
        self.assertNotIn('src/adapters/native_framework.py', SOURCES)
        self.assertEqual(len(SOURCES), len(set(SOURCES)))
        self.assertTrue(all((REPO / path).is_file() for path in SOURCES))
        modules = _load_v5_modules()
        self.assertEqual(modules['make_monitor'].__module__, 'src.verification.vision_physical_monitor')
        self.assertEqual(modules['make_grounding'].__module__, 'src.perception.visual_grounding')
        self.assertEqual(modules['make_short_term_predictor'].__module__, 'src.prediction.short_term_predictor')
        self.assertEqual(modules['make_action_suffix_corrector'].__module__, 'src.repair.action_suffix_corrector')
        self.assertEqual(modules['make_keyframe_bank'].__module__, 'src.memory.keyframe_bank')


class ActionValidationTests(unittest.TestCase):
    def row(self):
        return dict(step=0, action=[0., 0., .05, 0., 0., 0., -1.],
            executed_mapped_action=[0., 0., .05, 0., 0., 0., -1.],
            proposal=[0., 0., 0., 0., 0., 0., -1.],
            native_proposal=[0., 0., 0., 0., 0., 0., 1.], tokens=list(range(7)),
            original_action=[0., 0., 0., 0., 0., 0., -1.],
            action_modified=True, action_protocol='bounded_translation_v1',
            correction_reason='no_lift_after_grasp',
            correction_delta=[0., 0., .05, 0., 0., 0., 0.])

    def test_explicit_opt_in(self):
        with self.assertRaises(ValueError):
            validate_action(self.row(), 0)
        validate_action(self.row(), 0, allow_correction=True)

    def test_unchanged_protocol(self):
        row = self.row()
        row.update(action=row['proposal'], executed_mapped_action=row['proposal'], action_modified=False)
        validate_action(row, 0)

    def test_reject_tampering(self):
        for field, value in [('action_protocol', 'unknown'), ('correction_reason', None),
                             ('original_action', [0.] * 7), ('native_proposal', [0.] * 7),
                             ('correction_delta', [0.] * 7)]:
            with self.subTest(field=field):
                row = self.row()
                row[field] = value
                with self.assertRaises((ValueError, AssertionError)):
                    validate_action(row, 0, allow_correction=True)

    def test_reject_rotation_gripper_and_large_translation(self):
        for index, value in [(3, .01), (6, 1.), (2, .101), (2, float('nan'))]:
            row = deepcopy(self.row())
            row['action'][index] = value
            row['executed_mapped_action'] = list(row['action'])
            row['correction_delta'][index] = value - row['proposal'][index]
            with self.assertRaises((ValueError, AssertionError)):
                validate_action(row, 0, allow_correction=True)


class NoLiftTests(unittest.TestCase):
    def test_wait_lift_release_and_reset(self):
        p = ShortTermMotionPredictor()
        p.update([0, 0, 1], grasp=True, step=0)
        self.assertFalse(p.predicts_no_lift(True, [0, 0, 1], step=29)['no_lift_predicted'])
        self.assertTrue(p.predicts_no_lift(True, [0, 0, 1], step=30)['no_lift_predicted'])
        p.update([0, 0, 1.03], grasp=True, step=31)
        p.update([0, 0, 1], grasp=True, step=32)
        self.assertFalse(p.predicts_no_lift(True, [0, 0, 1], step=32)['no_lift_predicted'])
        p.update([0, 0, 1], grasp=False, step=33)
        self.assertIsNone(p.predicts_no_lift(True, [0, 0, 1], step=34))
        p.reset()
        self.assertEqual(p._position_history, [])

    def test_missing_evidence_cannot_trigger(self):
        p = ShortTermMotionPredictor()
        p.update([0, 0, 1], grasp=True)
        self.assertFalse(p.predicts_no_lift(True, [0, 0, 1])['no_lift_predicted'])
        p.update([0, 0, 1], grasp=None, step=40)
        self.assertIsNone(p.predicts_no_lift(True, [0, 0, 1], step=41))

    def test_bound_target_not_first_vector(self):
        diagnostics = dict(object_observations={'other_pos': [9, 9, 9],
            'target_to_robot0_eef_pos': [8, 8, 8], 'target_pos': [1, 2, 3]},
            grasped={'other': False, 'target': True})
        self.assertIsNone(extract_physics_state(diagnostics)['object_pos'])
        state = extract_physics_state(diagnostics, target='target')
        self.assertEqual(state['object_pos'], [1, 2, 3])
        self.assertTrue(state['grasp'])
        self.assertIsNone(extract_physics_state(diagnostics, target='absent')['object_pos'])


if __name__ == '__main__':
    unittest.main()
