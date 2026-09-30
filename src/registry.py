"""
Component registry — the ONE place that decides which implementation fills
each slot. Experiment scripts select implementations with CLI flags:

    --policy        bc | openvla
    --orbisim_impl  standin | official      (fills the "orbisim" predictor role)
    --vision_impl   standin | official      (fills the "vision" baseline role)
    --verifier      ref | official          (CheckVLA trigger + repair)
    --backend       torch | genesis         (GT simulator, see sims/base.py)

To plug in a new implementation: write an adapter satisfying the Protocol in
``interfaces.py`` and add a branch below.

Predictor roles used in results files:
    orbisim       fast differentiable physics predictor (RQ1 treatment)
    vision        visual world model (RQ1 baseline, §3.7)
    vision_noact  observation-only visual WM (ablation; stand-in only)
"""
from __future__ import annotations

import os

import torch

PREDICTOR_ROLES = ("orbisim", "vision", "vision_noact")


class BCPolicyAdapter:
    """Wrap the stand-in BCPolicy (nn.Module) as a PolicyAdapter."""
    needs_img = False
    needs_rgb = False

    def __init__(self, model):
        self.model, self.H, self.act_dim = model, model.H, model.act_dim

    @torch.no_grad()
    def __call__(self, obs, img=None, instruction=None):
        return self.model(obs)


def build_policy(args, od, dev, sim):
    impl = getattr(args, "policy", "bc")
    if impl == "bc":
        from models.nets import BCPolicy, load_model
        return BCPolicyAdapter(load_model(BCPolicy, os.path.join(od, "bc_policy.pt"), dev))
    if impl == "openvla":
        from adapters.openvla_policy import OpenVLAPolicy
        return OpenVLAPolicy(sim, dev, model_id=getattr(args, "vla_model", "openvla/openvla-7b"))
    raise ValueError(f"unknown policy impl {impl}")


def build_predictor(args, od, dev, sim, role):
    from models.nets import OrbiSimDynamics, VisionWM, load_model
    if role == "orbisim":
        impl = getattr(args, "orbisim_impl", "standin")
        if impl == "standin":
            return load_model(OrbiSimDynamics, os.path.join(od, "orbisim.pt"), dev)
        if impl == "official":
            from adapters.orbisim_official import OrbiSimOfficial
            return OrbiSimOfficial(sim, dev)
    elif role in ("vision", "vision_noact"):
        impl = getattr(args, "vision_impl", "standin")
        if impl == "standin" or role == "vision_noact":
            return load_model(VisionWM, os.path.join(od, f"{role}.pt"), dev)
        if impl == "official":
            from adapters.checkvla_official import CheckVLAVisualWM
            return CheckVLAVisualWM(sim, dev)
    raise ValueError(f"unknown predictor role/impl {role}")


def build_verifier(args, predictor, sim, tau=float("inf")):
    impl = getattr(args, "verifier", "ref")
    kw = dict(tau=tau, latency=getattr(args, "latency", 1), commit=getattr(args, "commit", 5))
    if impl == "ref":
        from checkvla.reference import RefCheckVLA
        return RefCheckVLA(predictor, max_vel=sim.max_vel,
                           hard_prefix=bool(getattr(args, "hard_prefix", 0)),
                           use_grad=bool(getattr(args, "use_grad", 0)),
                           abstain=bool(getattr(args, "abstain", 1)), **kw)
    if impl == "official":
        from adapters.checkvla_official import CheckVLAOfficial
        return CheckVLAOfficial(predictor, sim=sim, **kw)
    raise ValueError(f"unknown verifier impl {impl}")


def add_component_args(p):
    g = p.add_argument_group("components (see registry.py)")
    g.add_argument("--policy", choices=["bc", "openvla"], default="bc")
    g.add_argument("--vla_model", default="openvla/openvla-7b")
    g.add_argument("--orbisim_impl", choices=["standin", "official"], default="standin")
    g.add_argument("--vision_impl", choices=["standin", "official"], default="standin")
    g.add_argument("--verifier", choices=["ref", "official"], default="ref")
    g.add_argument("--latency", type=int, default=1)
    g.add_argument("--hard_prefix", type=int, default=0,
                   help="CheckVLA's latency-aware constraint: hold the first "
                        "`latency` dispatched actions, damp only the suffix. "
                        "Off by default — measured worse here, since the "
                        "violating contact is at the first step of the chunk. "
                        "Kept for the ablation.")
    g.add_argument("--use_grad", type=int, default=0,
                   help="also consider the gradient suffix repair and keep "
                        "whichever the predictor rates safer. Off by default: "
                        "that branch holds a hard prefix, so it cannot change "
                        "the violating step, yet still scores lower on the "
                        "predictor and wins the selection. Kept for the "
                        "ablation that explains the RQ3 gradient table.")
    g.add_argument("--abstain", type=int, default=1,
                   help="return the proposed chunk unchanged when the bisection "
                        "certifies no scale, instead of falling through to the "
                        "floor. On by default: the floor is only conservative if "
                        "the predictor responds to the action, and an "
                        "action-blind one drives the search to a near-stop that "
                        "was never certified (measured on vision_noact: "
                        "magnitude x0.05, CVR 0.278 vs 0.272 baseline, SR "
                        "-0.057). Set to 0 to reproduce that defect.")
    g.add_argument("--commit", type=int, default=5)
    g.add_argument("--roles", nargs="+", default=list(PREDICTOR_ROLES),
                   help="predictor roles to evaluate")
    return p
