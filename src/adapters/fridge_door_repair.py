"""Versioned door-contact repair development; R2 frozen controller stays unchanged."""
import numpy as np
from scipy.spatial.transform import Rotation

from adapters.fridge_runtime import SIDE, pose_action


def door_feedback(env, state):
    m,d=env.sim.model,env.sim.data
    names=['gripper0_finger1_pad_collision','gripper0_finger2_pad_collision']
    pads=[m.geom_name2id(n) for n in names]
    door={m.geom_name2id(f'short_fridge_1_g{k}') for k in range(15,20)}
    pad_contacts=[False,False]
    for c in d.contact[:d.ncon]:
        for k,p in enumerate(pads):
            if (c.geom1==p and c.geom2 in door) or (c.geom2==p and c.geom1 in door):
                pad_contacts[k]=True
    positions=np.array([d.geom_xpos[p] for p in pads])
    return dict(state,pad_contacts=pad_contacts,pad_center=positions.mean(axis=0),
                pad_span=float(np.linalg.norm(positions[0]-positions[1])))


class DoorContactRepair:
    """Move pads to handle, establish bilateral contact, then follow hinge arc."""
    def __init__(self, limit=200):
        self.stage='raise'
        self.used=0
        self.limit=limit
        self.contact_streak=0
        self.success_streak=0
        self.close_steps=0
        self.retries=0
        self.pad_offset_local=None
        self.retreat_target=None

    def act(self,s):
        self.used+=1
        bilateral=all(s['pad_contacts'])
        self.contact_streak=self.contact_streak+1 if bilateral else 0
        self.success_streak=self.success_streak+1 if s['angle']>=1.8 else 0
        if self.success_streak>=3:
            return np.array([0,0,0,0,0,0,-1.]),dict(stage='verified_open',event='verified_open',done=True)
        if self.used>self.limit:
            return np.zeros(7),dict(stage=self.stage,event='budget_exhausted',done=True)
        yaw=Rotation.from_rotvec([0,0,s['angle']]).as_matrix()
        orientation=yaw@SIDE
        # Closed-loop pad-centroid offset, not an object-specific EEF offset.
        offset=s['eef']-s['pad_center']
        if self.pad_offset_local is not None and self.stage in ('grip','pull'):
            # Keep the symmetric, open-jaw calibration: adapting this offset to
            # asymmetric closed fingers makes the hand follow slip into a joint stop.
            offset=yaw@self.pad_offset_local
        target=s['handle']+offset
        grip=-1.
        event=None
        if self.stage=='retreat':
            target=self.retreat_target
            if np.linalg.norm(target-s['eef'])<.012:
                self.stage='align'
                self.pad_offset_local=None
        elif self.stage=='raise':
            target=np.array([-.25,s['eef'][1],1.28])
            orientation=None
            if np.linalg.norm(target-s['eef'])<.012:self.stage='cross'
        elif self.stage=='cross':
            target=s['handle']+[-.12,0,1.28-s['handle'][2]]
            orientation=None
            if np.linalg.norm(target-s['eef'])<.012:self.stage='descend'
        elif self.stage=='descend':
            target=s['handle']+[-.12,0,0]
            if np.linalg.norm(target-s['eef'])<.012:self.stage='align'
        elif self.stage=='align':
            rotation_error=np.linalg.norm((Rotation.from_matrix(orientation)*Rotation.from_quat(s['quat']).inv()).as_rotvec())
            # A neighboring door lip limits the approach at ~4.1mm error in v1.
            # Six millimetres is an approach tolerance, never grasp verification.
            if np.linalg.norm(s['pad_center']-s['handle'])<.006 and rotation_error<.08:
                self.pad_offset_local=yaw.T@(s['eef']-s['pad_center'])
                self.stage='grip'
                self.close_steps=0
        elif self.stage=='grip':
            grip=1.
            self.close_steps+=1
            if self.contact_streak>=3:
                self.stage='pull'
                event='bilateral_contact_established'
            elif self.close_steps>=30:
                self.stage='retreat'
                self.retreat_target=s['handle']-yaw@np.array([.07,0,0])
                self.retries+=1
                event='no_contact_realign'
        elif self.stage=='pull':
            grip=1.
            centering_error=np.linalg.norm((s['eef']-s['pad_center'])-offset)
            if centering_error>.008:
                self.stage='retreat'
                self.retreat_target=s['handle']-yaw@np.array([.07,0,0])
                target=self.retreat_target
                grip=-1.
                self.retries+=1
                event='asymmetric_grip_recenter'
            elif not bilateral:
                self.stage='grip'
                self.close_steps=0
                event='lost_contact_stop_pull'
            else:
                delta=.06
                rotation=Rotation.from_rotvec([0,0,delta]).as_matrix()
                target=s['hinge']+rotation@(s['handle']-s['hinge'])+offset
                orientation=Rotation.from_rotvec([0,0,s['angle']+delta]).as_matrix()@SIDE
        return pose_action(s,target,grip,orientation),dict(stage=self.stage,event=event,done=False,
            bilateral_contact=bilateral,pad_handle_error=float(np.linalg.norm(s['pad_center']-s['handle'])),
            target=np.asarray(target).tolist(),used=self.used,retries=self.retries)
