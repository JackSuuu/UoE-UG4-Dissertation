import unittest
import json
import numpy as np

from adapters.libero_mining import (ReleaseGate, EvidenceMemory, apply_fault,
                                    apply_verification, candidate_plans)


def measured(**kw):
    m=dict(native_in=True,bowl_gripper=True,closed=False,clear=False,retained=False,outward_world=[0,-1,0])
    m.update(kw); return m


def tracking(**kw):
    t=dict(released=False,clear=False,phase='insert_release');t.update(kw);return t


class MiningTests(unittest.TestCase):
    def test_fault_is_external_and_pose_bias_has_registered_integral(self):
        for axis in (0,1):
            for sign in (-1,1):
                a=np.array([0,0,0,0,0,0,-1.])
                case=dict(fault='pose_bias',anchor_step=3,duration=4,axis=axis,requested_m=sign*.02)
                effective=[apply_fault(a,i,case)[0] for i in range(10)]
                self.assertAlmostEqual(.05*sum(x[axis] for x in effective),sign*.02)
                np.testing.assert_array_equal(a,[0,0,0,0,0,0,-1])
                for x in effective: self.assertEqual(x[-1],-1)

    def test_entry_requires_nominal_open_and_observed_region_contact(self):
        for m,p in [(measured(native_in=False),[0,0,0,0,0,0,-1]),
                    (measured(bowl_gripper=False),[0,0,0,0,0,0,-1]),
                    (measured(),[0,0,0,0,0,0,1])]:
            g=ReleaseGate('phase_repair'); a,d,_=g.decide(p,m,tracking(),0)
            self.assertIsNone(d['trigger']);np.testing.assert_array_equal(a,p)

    def test_early_entry_release_clearance_handoff_and_no_retrigger(self):
        p=np.array([.3,.2,0,0,0,0,-1.]);g=ReleaseGate('phase_repair')
        a,d,_=g.decide(p,measured(),tracking(),10)
        np.testing.assert_array_equal(a,[0,0,0,0,0,0,-1]);self.assertEqual(d['used'],1)
        a,d,_=g.decide(p,measured(bowl_gripper=False,retained=True),tracking(released=True),11)
        np.testing.assert_allclose(a,[0,-.1,0,0,0,0,-1]);self.assertEqual(d['forecast_event'],'released')
        a,d,_=g.decide(p,measured(clear=True),tracking(released=True,clear=True),12)
        np.testing.assert_array_equal(a,p);self.assertEqual(d['handoff']['reason'],'stable_clearance')
        a,d,_=g.decide(p,measured(),tracking(),13)
        self.assertEqual(d['used'],2);self.assertEqual(d['reason'],'actor_after_handoff')

    def test_hold_is_strong_release_control_and_timed_is_distinct(self):
        p=np.array([.3,0,0,0,0,0,-1.])
        hold,timed=ReleaseGate('gate_hold'),ReleaseGate('timed_repair')
        for gate in (hold,timed): gate.decide(p,measured(),tracking(),0)
        a,h,_=hold.decide(p,measured(),tracking(released=True),1)
        self.assertEqual(h['reason'],'stable_release');np.testing.assert_array_equal(a,p)
        for i in range(1,11): a,d,_=timed.decide(p,measured(),tracking(released=True),i)
        self.assertEqual(d['reason'],'retract_outward')

    def test_bounded_budget_and_debounced_loss(self):
        p=[.2,0,0,0,0,0,-1]
        g=ReleaseGate('phase_repair')
        for i in range(41): a,d,_=g.decide(p,measured(),tracking(),i)
        self.assertEqual(d['used'],40);self.assertEqual(d['reason'],'budget_exhausted')
        g=ReleaseGate('phase_repair');g.decide(p,measured(),tracking(),0)
        for i in range(1,4): a,d,_=g.decide(p,measured(native_in=False),tracking(),i)
        self.assertEqual(d['reason'],'release_envelope_lost');np.testing.assert_array_equal(a,p)

    def test_verification_only_changes_registered_arm_while_active(self):
        p=np.zeros(7);d=dict(mode='contact_transition',reason='release_hold')
        f=dict(first_action=[0,.1,0,0,0,0,-1],selected=1)
        a,_=apply_verification('phase_repair',p,d,f);np.testing.assert_array_equal(a,p)
        a,_=apply_verification('verified_repair',p,d,f);np.testing.assert_array_equal(a,f['first_action'])
        a,_=apply_verification('verified_repair',p,dict(mode='nominal'),f);np.testing.assert_array_equal(a,p)
        plans=candidate_plans(np.ones(7),np.array([0,-.1,0,0,0,0,-1]))
        self.assertEqual(plans.shape,(3,3,7));self.assertTrue(np.all(plans[:,1:,:6]==0))

    def test_memory_binds_selected_not_faulted_command_and_resets(self):
        mem=EvidenceMemory();p=np.array([0,0,0,0,0,0,-1.])
        for i in range(5): e=mem.observe(p,measured(),tracking(),i)
        self.assertEqual(e['contact_persists_after_open'],5)
        e=mem.observe(p,measured(bowl_gripper=False),tracking(released=True),5)
        self.assertEqual(e['contact_persists_after_open'],0);self.assertEqual(len(e['keyframes']),2)
        json.dumps(e)


if __name__=='__main__': unittest.main()
