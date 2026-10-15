"""Assembly-specific self-developed OrbiSim-role adapter, NOT official OrbiSim.

Unlike the historical push model, risk consumes the autoregressive future state.
All lookahead actions are explicit pulse/settle primitives, never VLA chunks.
"""
import torch
from models.nets import OrbiSimDynamics


class AssemblyDynamics(OrbiSimDynamics):
    def rollout(self, obs, obs_prev, a_prev, actions):
        o = obs[None].expand(self.E, -1, -1)
        op = obs_prev[None].expand_as(o)
        ap = a_prev[None].expand(self.E, -1, -1)
        states, risks = [], []
        for action in actions.unbind(1):
            action = action[None].expand(self.E, -1, -1)
            delta, risk = self._one(o, op, ap, action)
            op, o, ap = o, o+delta, action
            states.append(o)
            risks.append(risk)
        return torch.stack(states, 2), torch.stack(risks, 2)


def candidates(nominal):
    """Eight bounded pulse/settle hypotheses; only the first action is executed."""
    actions = nominal.repeat(8, 3, 1)
    actions[:, 1:, :6] = 0
    actions[1:, :, 6] = 1  # keep grasp during local correction
    actions[1:, 0, :6] = 0
    actions[2, 0, 2] = .15
    actions[3, 0, 0] = .15
    actions[4, 0, 0] = -.15
    actions[5, 0, 1] = .15
    actions[6, 0, 1] = -.15
    actions[7, 0, 2] = -.10
    return actions


NAMES = ['nominal', 'hold_grasp', 'retract', 'x_plus', 'x_minus', 'y_plus', 'y_minus', 'insert']
