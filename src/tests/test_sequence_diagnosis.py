"""Diagnostic counts and exact-risk substitution contracts."""
import os
import sys
import unittest
import torch

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, 'experiments'))
from analyze_sequence_risk import ExactRisk, pair_counts, local_counts
from diagnose_risk_controller import GateRepairVerifier
from checkvla.reference import RefCheckVLA
from checkvla.reference import scale_repair
from sims.torch_push import PushSim


class DiagnosisTests(unittest.TestCase):
    def test_gate_substitution_does_not_change_repair_and_ll_is_identical(self):
        class Model:
            def __init__(self, bias):
                self.bias = bias

            def risk(self, ctx, a):
                r = self.bias + a.abs()
                return r, torch.zeros_like(r)
        model, other = Model(0), Model(2)
        original = RefCheckVLA(model, tau=1., use_grad=False, hard_prefix=False)
        same = GateRepairVerifier(model, model, 1., .5)
        swap = GateRepairVerifier(other, model, 1., .5)
        a = torch.full((3, 10, 2), .8)
        torch.testing.assert_close(original.check({}, a)[0], same.check({}, a)[0])
        self.assertFalse(bool(same.check({}, a)[0].any()))
        self.assertTrue(bool(swap.check({}, a)[0].all()))
        torch.testing.assert_close(original.repair({}, a), same.repair({}, a), rtol=0, atol=0)
        torch.testing.assert_close(same.repair({}, a), swap.repair({}, a), rtol=0, atol=0)

    def test_ranking_keeps_prediction_ties_in_fixed_denominator(self):
        c = pair_counts(torch.tensor([[.2, 1.2, .5]]), torch.ones(1, 3))
        self.assertEqual(c, dict(comparable=3, inverted=0, tied=3,
                                 boundary_pairs=2, boundary_inverted=0, boundary_tied=2))

    def test_local_harm_rescue_only_counts_triggered_rows(self):
        a = torch.ones(3, 10, 2)
        old = torch.tensor([[.5] * 10, [1.5] * 10, [.5] * 10])
        new = torch.tensor([[1.5] * 10, [.5] * 10, [1.5] * 10])
        c = local_counts(torch.tensor([True, True, False]), a, .5 * a, old, new)
        self.assertEqual(c['R5_local_harms'], 1)
        self.assertEqual(c['R5_local_rescues'], 1)
        self.assertEqual(c['execution_changes'], 2)

    def test_exact_risk_matches_serial_and_is_subset_safe(self):
        sim = PushSim('cpu')
        cell = {'friction': .2, 'mass': 2.}
        m = ExactRisk(sim, cell)
        s = sim.init_state(3, torch.Generator().manual_seed(12))
        before = s.clone()
        a = torch.full((3, 10, 2), .1)
        r, std = m.risk({'state': s}, a)
        rs, _ = m.risk({'state': s[[2, 0]]}, a[[2, 0]])
        torch.testing.assert_close(rs, r[[2, 0]])
        torch.testing.assert_close(s, before, rtol=0, atol=0)
        p = sim.make_params(3, cell)
        manual = []
        for action in a.unbind(1):
            s, risk = sim.step(s, action, p)
            manual.append(risk)
        torch.testing.assert_close(r, torch.stack(manual, 1))
        self.assertEqual(float(std.abs().sum()), 0.)

    def test_responsive_bisection_can_return_uncleared_floor(self):
        class Model:
            def risk(self, ctx, a):
                r = .9 + .1 * a.abs()
                return r, torch.zeros_like(r)
        a = torch.ones(2, 10, 2)
        m = Model()
        repaired = scale_repair(m, {}, a)
        torch.testing.assert_close(repaired, .05 * a)
        self.assertTrue((m.risk({}, repaired)[0].amax((1, 2)) > .75).all())


if __name__ == '__main__':
    unittest.main()
