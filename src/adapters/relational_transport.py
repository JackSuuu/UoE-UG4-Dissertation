"""Bounded held-object transport from object/obstacle/support geometry.

No asset names or world-coordinate waypoints in this skill. The adapter supplies
an extraction frame (x inward, z up), obstacle bounds, target support and feedback.
This conservative outside-corridor route is a development skill, not a general planner.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from adapters.fridge_runtime import pose_action


class RelationalTransport:
    def __init__(self, state, limit=240, margin=.02):
        self.frame = np.asarray(state['frame'])
        self.origin = np.asarray(state['origin'])
        self.orientation = Rotation.from_quat(state['quat']).as_matrix()
        self.offset = np.asarray(state['eef'])-np.asarray(state['item'])
        self.limit, self.used, self.stage, self.dwell = limit, 0, 0, 0
        eef = self.local(state['eef'])
        support = self.local(state['support_center'])
        object_goal = support.copy()
        object_goal[2] += state['item_half_height']
        eef_goal = object_goal+self.frame.T@self.offset
        # Include the entire moving hand + held object, not just the EEF point.
        relative_payload = (np.asarray(state['payload_points'])-state['eef'])@self.frame
        bounds = np.asarray(state['obstacle_bounds'])
        outside = min(eef[0], bounds[0, 0]-relative_payload[:, 0].max()-margin)
        transit_z = max(eef[2], eef_goal[2]+margin)
        self.waypoints = [np.array([outside, eef[1], eef[2]]),
                          np.array([outside, support[1], transit_z]),
                          np.array([eef_goal[0], eef_goal[1], transit_z]), eef_goal]
        self.release_position = None

    def local(self, point):
        return self.frame.T@(np.asarray(point)-self.origin)

    def world(self, point):
        return self.origin+self.frame@point

    def act(self, state):
        events = []
        if self.used >= self.limit:
            return None, dict(events=['budget_exhausted'], done=True, success=False, stage=self.stage)
        if self.stage < 4 and not state['grasp']:
            return None, dict(events=['lost_grasp'], done=True, success=False, stage=self.stage)
        grip = 1.
        if self.stage < 4:
            target = self.world(self.waypoints[self.stage])
            if self.stage < 3 and np.linalg.norm(target-state['eef']) < .01:
                self.stage += 1
                events.append('waypoint_reached')
            elif self.stage == 3 and state['table']:
                self.stage = 4
                self.release_position = np.asarray(state['eef']).copy()
                events.append('support_contact_before_release')
        else:
            grip = -1.
            target = self.release_position.copy()
            if not state['hand_contact']:
                target += self.frame[:, 2]*.06
            self.dwell = self.dwell+1 if state['placed'] else 0
            if self.dwell >= 10:
                return None, dict(events=['verified_placement'], done=True, success=True, stage=self.stage)
        self.used += 1
        return pose_action(state, target, grip, self.orientation), dict(events=events,
            stage=self.stage, target=target.tolist(), used=self.used, done=False, success=False)
