"""Shared helpers for experiment scripts."""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

from common import get_device, out_dir, seed_all  # noqa: E402
from registry import (PREDICTOR_ROLES, add_component_args, build_policy,  # noqa: E402
                      build_predictor, build_verifier)
from sims.base import make_env, make_sim  # noqa: E402

CHUNK_H = 10
PREDICTORS = PREDICTOR_ROLES


def base_parser(desc=""):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--task", choices=["push", "cloth"], default="push")
    p.add_argument("--backend", choices=["torch", "genesis"], default="torch",
                   help="GT simulator backend (genesis: Task A only)")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="tiny sizes for a smoke test")
    add_component_args(p)
    return p


def setup(args):
    seed_all(args.seed)
    dev = get_device(args.device)
    sim = make_sim(args.task, dev)
    od = out_dir(args.task, args.backend)
    print(f"[setup] task={args.task} backend={args.backend} device={dev} out={od} | "
          f"policy={args.policy} orbisim={args.orbisim_impl} vision={args.vision_impl} "
          f"verifier={args.verifier}")
    return dev, sim, od


def env_for_cell(args, n, dev, cell):
    """Build (env, params) for one OOD cell (dict of multipliers)."""
    env = make_env(args.task, args.backend, n, dev, cell,
                   camera=getattr(args, "policy", "bc") == "openvla")
    params = env.sim.make_params(n, cell) if args.backend == "torch" else None
    return env, params


def load_policy(args, od, dev, sim):
    return build_policy(args, od, dev, sim)


def load_predictor(args, od, role, dev, sim):
    return build_predictor(args, od, dev, sim, role)


def load_verifiers(args, od, dev, sim, taus=None):
    """{role: VerifierAdapter} for every requested predictor role."""
    out = {}
    for role in args.roles:
        pred = load_predictor(args, od, role, dev, sim)
        tau = (taus or {}).get(role, float("inf"))
        out[role] = build_verifier(args, pred, sim, tau)
    return out
