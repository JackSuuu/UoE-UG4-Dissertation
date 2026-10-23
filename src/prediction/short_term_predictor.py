"""Constant-velocity forecasts from consecutive measured object positions.

No normalized action/metres blending, invented container geometry or rollout.
Forecasts inform recovery planning but cannot verify a measured contract.
"""
import math


class ShortTermMotionPredictor:
    def __init__(self, horizon_steps=5, velocity_window=3):
        if any(type(x) is not int or x < 1 for x in (horizon_steps, velocity_window)):
            raise ValueError('Prediction horizon/window must be positive integers')
        self.horizon, self.window = horizon_steps, velocity_window

    def predict(self, observation, history):
        recent = list(history)[-self.window-1:]
        if not recent or recent[-1].index != observation.index:
            recent.append(observation)
        recent = recent[-self.window-1:]
        consecutive = (len(recent) >= 2 and
                       [o.index for o in recent] == list(range(recent[0].index, observation.index+1)))
        forecasts = {}
        for entity, current in observation.positions.items():
            positions = [o.positions.get(entity) for o in recent]
            if not consecutive or any(p is None for p in positions):
                forecasts[entity] = dict(valid=False, positions=None)
                continue
            velocity = tuple((b-a)/(len(positions)-1) for a, b in zip(positions[0], current))
            forecasts[entity] = dict(valid=True, displacement_per_action=velocity,
                speed_m_per_action=math.sqrt(sum(v*v for v in velocity)),
                positions=[tuple(x+h*v for x, v in zip(current, velocity))
                           for h in range(1, self.horizon+1)])
        return dict(method='observed_constant_velocity_v1', horizon_actions=self.horizon,
                    observation_index=observation.index, action_conditioned=False,
                    verification=False, objects=forecasts)
