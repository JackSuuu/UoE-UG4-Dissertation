"""Protect split isolation and success-independent recovery collection."""
from copy import deepcopy
import unittest

import run_framework_comparison as comparison
from run_repair_data_expansion import sibling_rule
from test_repair_candidates import CandidateTests
from src.training.recovery_dataset import record


class ExpansionTests(unittest.TestCase):
    def test_training_identity_and_paired_lanes(self):
        cells = comparison.training_cells([{'baseline': '/runtime/reference'}])
        self.assertEqual(len(cells), 20)
        self.assertEqual({c['root'] for c in cells}, set(range(10, 20)))
        for a, b in zip(cells[::2], cells[1::2]):
            self.assertEqual(a['lane'], b['lane'])
            self.assertEqual(a['root'], b['root'])
            self.assertEqual((a['arm'], b['arm']), ('baseline', 'framework'))
            self.assertEqual(a['data_split'], 'training')

    def test_collection_includes_triggered_successes(self):
        for success in (False, True):
            status = dict(valid_episode=True, any_native_success=success, candidate_pools=1)
            self.assertEqual(sibling_rule(status, 0, 0), 'collect_all_three')
            self.assertEqual(sibling_rule(dict(status, candidate_pools=0), 0, 0),
                'not_applicable_no_admitted_repair')
            for replay, audit in ((1, 0), (0, 1), (None, None)):
                self.assertEqual(sibling_rule(status, replay, audit), 'blocked_invalid_control')
        self.assertEqual(sibling_rule({}, 0, 0), 'blocked_invalid_control')

    def test_export_retains_training_group(self):
        fixture = CandidateTests()
        fixture.setUp()
        status, steps, controls = fixture.fixture()
        status = deepcopy(status)
        status['cell'] = dict(root=10, data_split='training')
        result = record(status, steps, controls, '/example', True, True)
        self.assertEqual(result['split'], 'training')
        self.assertEqual(result['split_group'], 'microwave_root10')


if __name__ == '__main__':
    unittest.main()
