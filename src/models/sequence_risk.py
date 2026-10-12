"""Matched static/causal risk heads; neither consumes predicted observations.

The static control is permutation equivariant in actions. The recurrent head
conditions each physical-risk prediction on the preceding candidate actions.
Both use exactly the same measured one-step context and normalizers.
"""
import torch
from torch import nn
from torch.nn import functional as F


class RiskMember(nn.Module):
    def __init__(self, kind, hidden, obs_dim=10, act_dim=2):
        super().__init__()
        self.kind = kind
        context_dim = 2 * obs_dim + act_dim
        if kind == 'static':
            self.net = nn.Sequential(nn.Linear(context_dim + act_dim, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))
        elif kind == 'sequence':
            self.init = nn.Linear(context_dim, hidden)
            self.gru = nn.GRU(act_dim, hidden, batch_first=True)
            self.out = nn.Linear(hidden, 2)
        else:
            raise ValueError(kind)

    def forward(self, context, actions):
        if self.kind == 'static':
            x = torch.cat([context[:, None].expand(-1, actions.shape[1], -1), actions], -1)
            y = self.net(x)
        else:
            initial = torch.tanh(self.init(context))[None]
            # cuDNN's inference RNN does not retain a backward reserve. Preserve
            # differentiability for future input-gradient audits in eval mode.
            if actions.is_cuda and not self.training and torch.is_grad_enabled():
                with torch.backends.cudnn.flags(enabled=False):
                    z, _ = self.gru(actions, initial)
            else:
                z, _ = self.gru(actions, initial)
            y = self.out(z)
        return F.softplus(y)


def matched_static_width(sequence_width=128):
    """Match trainable parameter counts, not hand-chosen widths after evaluation."""
    target = 3 * sequence_width ** 2 + 37 * sequence_width + 2
    return min(range(16, 513), key=lambda h: abs(2 * h * h + 29 * h + 2 - target))


class SequenceRisk(nn.Module):
    needs_img = False
    differentiable = True

    def __init__(self, kind, hidden, E=5, max_vel=0.5):
        super().__init__()
        self.kind, self.hidden, self.E, self.max_vel = kind, hidden, E, max_vel
        self.members = nn.ModuleList([RiskMember(kind, hidden) for _ in range(E)])
        self.register_buffer('obs_mean', torch.zeros(10))
        self.register_buffer('obs_std', torch.ones(10))
        self.register_buffer('delta_std', torch.ones(10))

    @torch.no_grad()
    def fit_normalizers(self, training_data):
        if training_data['split'] != 'train':
            raise ValueError('normalization may only fit training episodes')
        o, p = training_data['obs'], training_data['obs_prev']
        self.obs_mean.copy_(o.mean(0))
        self.obs_std.copy_(o.std(0).clamp_min(0.01))
        self.delta_std.copy_((o - p).std(0).clamp_min(0.01))

    def context(self, ctx):
        return torch.cat([(ctx['obs'] - self.obs_mean) / self.obs_std,
                          (ctx['obs'] - ctx['obs_prev']) / self.delta_std,
                          ctx['a_prev'] / self.max_vel], -1)

    def forward(self, ctx, actions):
        c = self.context(ctx)
        return torch.stack([m(c, actions / self.max_vel) for m in self.members])

    def risk(self, ctx, actions):
        r = self(ctx, actions)
        return r.mean(0), r.std(0, unbiased=self.E > 1)


class RiskWithFrozenDynamics:
    """Same state/progress rollout for both heads; only risk interface changes."""
    needs_img = False
    differentiable = True

    def __init__(self, risk_head, dynamics):
        self.risk_head, self.dynamics = risk_head, dynamics

    def risk(self, ctx, actions):
        return self.risk_head.risk(ctx, actions)

    def rollout(self, obs, obs_prev, a_prev, actions):
        return self.dynamics.rollout(obs, obs_prev, a_prev, actions)
