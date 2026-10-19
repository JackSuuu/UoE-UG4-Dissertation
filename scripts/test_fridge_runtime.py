"""Meaningful controller/endpoint guard checks; no simulated task-success claim."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from adapters.fridge_runtime import FridgeRepair, TaskMonitor, SIDE
from scipy.spatial.transform import Rotation


def sample(**changes):
    s = dict(item=np.array([-.04,.38,1.004]), eef=np.array([-.25,.38,1.012]),
        quat=Rotation.from_matrix(SIDE).as_quat(), angle=2.27, door_speed=0.,
        handle=np.array([-.12,.43,1.05]), hinge=np.array([-.1,.26,1.04]),
        fridge=np.array([0,.35,.905]), front=-.095, floor=.905, grasp=False,
        shelf=True, table=False, hand_contact=False, placed=False, extracted=False,
        lifted=False, door_contact=False, clear=True)
    s.update(changes)
    return s


class Guards(unittest.TestCase):
    def test_endpoint_requires_sequence_release_and_dwell(self):
        monitor = TaskMonitor('closed')
        terminal = sample(angle=0., placed=True, table=True, shelf=False)
        for _ in range(25):
            self.assertFalse(monitor.observe(terminal)['full_completion'])
        monitor.observe(sample())
        monitor.observe(sample(grasp=True))
        monitor.observe(sample(grasp=True, extracted=True, shelf=False))
        for _ in range(25):
            self.assertFalse(monitor.observe(terminal, interval_released=False)['full_completion'])
        for _ in range(19):
            self.assertFalse(monitor.observe(terminal)['full_completion'])
        self.assertTrue(monitor.observe(terminal)['full_completion'])

    def test_open_start_never_counts_closed_sequence(self):
        monitor = TaskMonitor('open')
        monitor.observe(sample(grasp=True, extracted=True, shelf=False))
        for _ in range(25):
            result = monitor.observe(sample(angle=0., placed=True, table=True, shelf=False))
        self.assertTrue(result['remaining_sequence_complete'])
        self.assertFalse(result['full_completion'])

    def test_release_dwell_does_not_restart_grasp(self):
        monitor = TaskMonitor('open')
        monitor.observe(sample(grasp=True, extracted=True, shelf=False))
        for _ in range(9):
            self.assertEqual(monitor.observe(sample(placed=True, table=True, shelf=False))['phase'], 'place')
        self.assertEqual(monitor.observe(sample(placed=True, table=True, shelf=False))['phase'], 'close')

    def test_failure_budget_not_successful_handoff(self):
        repair = FridgeRepair(max_total=4, max_local=2)
        events, owners = [], []
        for _ in range(100):
            _, gate = repair.decide(sample(), 'grasp', np.zeros(7))
            events.append(gate['event'])
            owners.append(gate['owner'])
        self.assertLessEqual(owners.count('repair'), 4)
        self.assertIn('repair_budget_exit_unverified', events)
        self.assertNotIn('verified_handoff', events)

    def test_successful_handoff_uses_current_proposal(self):
        repair = FridgeRepair()
        for _ in range(30):
            repair.decide(sample(), 'grasp', np.zeros(7))
        proposal = np.array([.1,.2,.3,0,0,0,1.])
        for _ in range(3):
            action, gate = repair.decide(sample(grasp=True), 'extract', proposal)
        self.assertEqual(gate['event'], 'verified_handoff')
        self.assertEqual(gate['owner'], 'vla')
        np.testing.assert_array_equal(action, proposal)

    def test_premature_release_protection_is_budgeted(self):
        repair = FridgeRepair(max_total=1)
        proposal = np.array([0,0,0,0,0,0,-1.])
        action, gate = repair.decide(sample(grasp=True), 'extract', proposal)
        self.assertEqual(action[-1], 1)
        self.assertEqual(gate['event'], 'retain_grip')
        action, gate = repair.decide(sample(grasp=True), 'extract', proposal)
        self.assertEqual(gate['owner'], 'vla')


if __name__ == '__main__':
    unittest.main()
