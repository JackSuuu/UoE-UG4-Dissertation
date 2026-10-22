"""Regression tests for oracle skill provenance, stopping and failure memory."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.repair.recovery_options import RecoveryOptions, CONFIG, SCHEMA, validate_skill
from src.controller import NativeFramework
from test_native_framework import state, ACTION, LANGUAGE
from test_joint_framework import evidence


def controller():
    return dict(action_scale=[.05]*3+[.5]*3, use_delta=True,
        action_input_transform=[0.]*6, action_output_transform=[0.]*6)


def diagnostics():
    return dict(object_observations={'obj_pos': [0., 0., .9]}, eef_position=[0., 0., .92])


def decision(step=0):
    return dict(owner='repair', repair_id=1, recovery_goal='acquire', step=step,
        evidence=dict(state=dict(in_goal=False, held_clear=False, grasp=True)))


class RecoveryTests(unittest.TestCase):
    def test_live_scale_and_unknowns_fail_closed(self):
        c = controller(); c['use_delta'] = False
        with self.assertRaises(ValueError): RecoveryOptions(dict(target='obj'), c)
        r = RecoveryOptions(dict(target='obj'), controller())
        with self.assertRaises(ValueError): r.command(decision(), {}, ACTION)

    def test_action_target_not_policy_z_nudge(self):
        r = RecoveryOptions(dict(target='obj'), controller())
        proposal = [0., 0., -.3, .4, .2, .1, -1.]
        a, rec = r.command(decision(), diagnostics(), proposal)
        self.assertAlmostEqual(a[2], .008/.05)
        self.assertEqual(a[3:], [0., 0., 0., 1.])
        validate_skill(dict(step=0, action=a, decision=decision(), action_protocol=SCHEMA, recovery_action=rec))
        tampered = list(a);tampered[0]=.5
        with self.assertRaises(AssertionError):
            validate_skill(dict(step=0, action=tampered, decision=decision(), action_protocol=SCHEMA, recovery_action=rec))

    def test_skill_permission_is_distinct_from_legacy_corrections(self):
        from run_native_long_branch import validate_action
        r=RecoveryOptions(dict(target='obj'),controller())
        proposal=[0.,0.,0.,0.,0.,0.,-1.]
        a,rec=r.command(decision(),diagnostics(),proposal)
        row=dict(step=0,action=a,executed_mapped_action=a,proposal=proposal,
            native_proposal=[0.,0.,0.,0.,0.,0.,1.],original_action=proposal,
            tokens=list(range(7)),decision=decision(),action_protocol=SCHEMA,
            action_modified=True,recovery_action=rec)
        for kwargs in ({},{'allow_correction':True}):
            with self.assertRaises(ValueError):validate_action(row,0,**kwargs)
        validate_action(row,0,allow_recovery=True)

    def test_stall_switches_option_before_local_cap(self):
        r = RecoveryOptions(dict(target='obj'), controller())
        for step in range(6):
            _, record = r.command(decision(step), diagnostics(), ACTION)
            feedback = r.observe(diagnostics(), dict(held_clear=False))
        self.assertEqual(feedback['outcome'], 'lift_no_object_progress')
        _, rec = r.command(decision(6), diagnostics(), ACTION)
        self.assertEqual(rec['option'], 'regrasp')
        self.assertEqual(rec['phase'], 'release')

    def test_same_physical_context_blocks_all_trigger_names(self):
        r = RecoveryOptions(dict(target='obj'), controller())
        r.anchor=[0.,0.,.9];r.failed={'lift_probe','regrasp'}
        s=dict(position=[0.,0.,.9],in_goal=False,held_clear=False)
        for step in range(5):
            self.assertFalse(r.admission(dict(s,grasp=bool(step%2)),step)['allowed'])
        for step in range(5,8):result=r.admission(dict(s,position=[.03,0.,.9]),step)
        self.assertTrue(result['allowed'])

    def test_achieved_containment_never_released(self):
        r=RecoveryOptions(dict(target='obj'),controller());d=decision()
        d['evidence']['state']['in_goal']=True
        a,record=r.command(d,diagnostics(),ACTION)
        self.assertEqual(a,ACTION);self.assertIsNone(record)

    def test_sequence_contains_distinct_release_retreat_reclose_lift(self):
        r=RecoveryOptions(dict(target='obj'),controller());r.failed.add('lift_probe')
        diag=diagnostics();phases=[]
        for step in range(24):
            a,rec=r.command(decision(step),diag,ACTION);phases.append(rec['phase'])
            diag['eef_position']=(np.array(diag['eef_position'])+np.array(a[:3])*.05).tolist()
            f=r.observe(diag,dict(held_clear=False))
            if f['exhausted']:break
        self.assertEqual(list(dict.fromkeys(phases)),['release','retreat','approach','descend','close','lift'])
        self.assertTrue(f['exhausted'])

    def test_early_exit_never_marked_verified(self):
        c=NativeFramework('framework',LANGUAGE,fusion_mode='gate',joint_mode=True,recovery_options=True)
        e=evidence(0,trigger='joint_no_lift');e['recovery_admission']=dict(schema=SCHEMA,allowed=True)
        d=c.before(0,state(),e)
        f=dict(schema=SCHEMA,exhausted=True,outcome='lift_no_object_progress')
        result=c.after(0,state(),ACTION,d,recovery_feedback=f)
        self.assertFalse(result['verification']['repair_verified'])
        self.assertEqual(c.summary['repairs'][0]['exit_reason'],'recovery_options_exhausted_unverified')


if __name__ == '__main__':unittest.main()
