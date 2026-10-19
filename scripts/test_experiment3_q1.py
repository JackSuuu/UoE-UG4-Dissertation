import copy
import unittest
from experiment3_q1_metrics import measurements, endpoint


def fixture(angle=0.):
    s = dict(angle=angle, eef=[0,0,0], item=[1,0,0], handle=[1,0,0],
        release_footprint_margin_m=-.1, release_seated_margin_m=-.1,
        grasp=False, local_release_support=False, placed=False, max_penetration=0.)
    names=('open','grasp','extract','place','close','withdraw')
    t=dict(predicates={p:dict(status='satisfied' if p=='close' else 'violated') for p in names},
        milestone_history=dict.fromkeys(names,False), consecutive_intervals=dict.fromkeys(names,0),
        complete_dwell=0, full_closed_start_completion=False)
    return s,t


def check(before, after, phase='open', btask=None, atask=None):
    _, default=fixture()
    return endpoint(before,btask or default,measurements(before,.2),[after],[atask or default],
        [measurements(after,.2)],[dict(max_contact_force_n=0.,max_contact_penetration_m=0.)],phase,1)


class Q1Tests(unittest.TestCase):
    def test_opening_goal_gap_threshold_not_arbitrary_motion(self):
        s,t=fixture()
        after=copy.deepcopy(s)
        after['eef']=[.5,0,0]
        self.assertFalse(check(s,after)['descriptive_useful'])
        after['angle']=.02
        result=check(s,after)
        self.assertTrue(result['descriptive_useful'])
        self.assertFalse(result['stage_satisfied'])

    def test_expected_initial_close_predicate_loss_does_not_veto_opening(self):
        s,t=fixture()
        after,at=fixture(.1)
        at['predicates']['close']['status']='violated'
        result=check(s,after,btask=t,atask=at)
        self.assertTrue(result['descriptive_useful'])
        self.assertIn('close',result['current_predicate_losses'])
        self.assertEqual(result['guarded_achievement_losses'],[])

    def test_grasp_approach_proxy_is_not_useful_without_grasp(self):
        s,t=fixture()
        after=copy.deepcopy(s)
        after['eef']=[.5,0,0]
        self.assertFalse(check(s,after,phase='grasp')['descriptive_useful'])

    def test_new_penetration_proxy_blocks_useful_but_damage_stays_unknown(self):
        s,t=fixture()
        after,at=fixture(.1)
        after['max_penetration']=.002
        result=check(s,after)
        self.assertFalse(result['descriptive_useful'])
        self.assertEqual(result['actual_physical_damage'],'unknown')

    def test_previous_supported_placement_loss_is_reported(self):
        s,t=fixture()
        s['placed']=True
        after,at=fixture(.1)
        result=check(s,after)
        self.assertFalse(result['descriptive_useful'])
        self.assertIn('previous_supported_placement_lost',result['guarded_achievement_losses'])


if __name__=='__main__': unittest.main()
