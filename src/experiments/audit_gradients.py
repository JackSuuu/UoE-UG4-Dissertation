"""
Week 1 — Gradient path audit (plan §3.6).

For every OOD cell we roll out the scripted expert in the differentiable GT and,
every ``--every`` steps, back-propagate the common chunk objective J through the
next H steps w.r.t. the action chunk. Each probed state is tagged

    valid  : finite, non-zero gradient
    zero   : gradient norm < 1e-8  (e.g. no contact in horizon, stuck friction,
             clamped actions) — the GT provides *no* signal here
    nan    : non-finite gradient

and bucketed by contact regime (sim.regime_names). The resulting map gates
what RQ1's gradient-agreement metric may claim (Fig B2).

With ``--genesis_probe`` it additionally tries to differentiate through a
Genesis rigid rollout (requires_grad=True) for each cell / horizon and records
valid / zero / nan / error(+message) — i.e. whether Genesis offers a gradient
path at all for this scene in the installed version.

Usage: python experiments/audit_gradients.py --task push [--genesis_probe]
"""
import os

import numpy as np
import torch

from _common import CHUNK_H, base_parser, setup
from _verif import GOAL_DIMS, J_gt
from common import grid_cells, save_json
from sims.base import TorchEnv


def main():
    p = base_parser(__doc__)
    p.add_argument("--n_envs", type=int, default=32)
    p.add_argument("--every", type=int, default=4)
    p.add_argument("--genesis_probe", action="store_true")
    p.add_argument("--probe_horizons", type=int, nargs="+", default=[10, 50, 100])
    args = p.parse_args()
    dev, sim, od = setup(args)
    if args.quick:
        args.n_envs, args.every = 4, 20
    H = CHUNK_H
    R = len(sim.regime_names)
    cells = grid_cells(args.task)
    results = []
    for cell in cells:
        env = TorchEnv(sim, args.n_envs)
        params = sim.make_params(args.n_envs, cell)
        env.reset(params, seed=args.seed)
        counts = np.zeros((R, 3))          # regime x (valid, zero, nan)
        norms = []
        for t in range(sim.T):
            s = env.get_state()
            a = sim.expert(s)
            if t % args.every == 0:
                chunk = a[:, None].repeat(1, H, 1).clone().requires_grad_(True)
                with torch.enable_grad():
                    J, _ = J_gt(sim, s.detach(), params, chunk, GOAL_DIMS[args.task])
                    g, = torch.autograd.grad(J.sum(), chunk)
                gn = g.flatten(1).norm(dim=1)
                cls = torch.where(~torch.isfinite(gn), 2, torch.where(gn < 1e-8, 1, 0))
                reg = sim.contact_regime(s)
                for r_, c_ in zip(reg.tolist(), cls.tolist()):
                    counts[r_, c_] += 1
                norms += gn[torch.isfinite(gn)].tolist()
            env.step(a)
        tot = counts.sum(1, keepdims=True).clip(min=1)
        entry = {"cell": cell, "counts": counts.tolist(),
                 "valid_frac_by_regime": (counts[:, 0:1] / tot)[:, 0].tolist(),
                 "valid_frac": float(counts[:, 0].sum() / counts.sum()),
                 "grad_norm_median": float(np.median(norms)) if norms else float("nan")}
        results.append(entry)
        print(f"[audit] {cell} valid={entry['valid_frac']:.3f} by-regime="
              f"{dict(zip(sim.regime_names, np.round(entry['valid_frac_by_regime'], 2)))}")

    out = {"task": args.task, "regimes": list(sim.regime_names), "torch_gt": results}

    if args.genesis_probe:
        if args.task != "push":
            out["genesis"] = [{"status": "error", "msg": "Genesis PBD cloth has no gradient path "
                               "(PBD solver not differentiable); not probed."}]
        else:
            from sims.genesis_push import genesis_grad_probe
            gres = []
            for cell in ([{}] + [c for c in cells if c != {}][:3]):
                for hz in args.probe_horizons:
                    r = genesis_grad_probe(sim, cell, horizon=hz)
                    r.update(cell=cell, horizon=hz)
                    print(f"[audit/genesis] {cell} H={hz}: {r['status']} {r['msg'][:120]}")
                    gres.append(r)
            out["genesis"] = gres
    save_json(out, os.path.join(od, "audit.json"))
    print(f"[audit] overall torch-GT valid fraction: "
          f"{np.mean([r['valid_frac'] for r in results]):.3f}")


if __name__ == "__main__":
    main()
