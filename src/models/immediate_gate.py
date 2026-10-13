"""Matched R5 regression/event objectives on the original residual MLP topology.

The gate reads measured context and the first five proposed actions only. The
frozen original predictor, not this gate, supplies every repair score.
"""
import torch
from torch.nn import functional as F

from models.nets import OrbiSimDynamics


class ImmediateGate(OrbiSimDynamics):
    def __init__(self, objective, hidden=256, E=5, max_vel=.5):
        super().__init__(10, 2, E=E, hidden=hidden, max_vel=max_vel)
        if objective not in ('regression', 'event'):
            raise ValueError(objective)
        self.objective, self.hidden = objective, hidden

    @torch.no_grad()
    def fit_normalizers(self, data):
        if data['split'] != 'train' or data['prior_violation'].any():
            raise ValueError('normalizers require eligible pre-violation training rows')
        self.obs_n.fit(data['obs'])
        self.dobs_n.fit(data['obs'] - data['obs_prev'])

    def logits(self, ctx, actions):
        if actions.shape[1] < 5:
            raise ValueError('R5 requires at least five actions')
        b = len(actions)
        o = self.obs_n(ctx['obs'])
        delta = (ctx['obs'] - ctx['obs_prev']) / self.dobs_n.std
        context = torch.cat([o, delta, ctx['a_prev'] / self.max_vel], -1)
        x = torch.cat([context[:, None].expand(-1, 5, -1),
                       actions[:, :5] / self.max_vel], -1).reshape(b * 5, -1)
        x = x[None].expand(self.E, -1, -1)
        h = F.silu(self.l1(x))
        h = F.silu(self.l2(h)) + h
        h = F.silu(self.l3(h)) + h
        # Keep the exact legacy output topology. Delta/deformation outputs are
        # unused in BOTH arms; no state auxiliary objective in this gate study.
        return self.out(h)[..., self.obs_dim].reshape(self.E, b, 5)

    def loss(self, ctx, actions, true_risk):
        z = self.logits(ctx, actions)
        target = true_risk[:, :5, 0][None].expand_as(z)
        if self.objective == 'regression':
            return F.mse_loss(F.softplus(z), target.clamp(max=3))
        return F.binary_cross_entropy_with_logits(z, (target > 1).to(z.dtype))

    def risk(self, ctx, actions):
        z = self.logits(ctx, actions)
        values = F.softplus(z) if self.objective == 'regression' else z.sigmoid()
        values = values[..., None]
        return values.mean(0), values.std(0, unbiased=self.E > 1)


class PrefixRisk:
    """Horizon-only control: frozen original model, truncated gate query."""
    needs_img = False
    differentiable = True

    def __init__(self, base):
        self.base = base

    def risk(self, ctx, actions):
        return self.base.risk(ctx, actions[:, :5])
