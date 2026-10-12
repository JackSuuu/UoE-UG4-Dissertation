"""
RQ2: closed-loop OOD evaluation grid (plan §5 Week 4, Fig A / Fig F).

Arms:
  none              uncorrected VLA proxy
  gt_shadow         Phase-0-style GT shadow-rollout correction (slow engine in loop)
  checkvla_vision   CheckVLA with the vision world-model predictor (baseline, §3.7)
  checkvla_orbisim  CheckVLA with OrbiSim-Dynamics (full pipeline)
  [checkvla_vision_noact  with --with_noact]

Metrics per cell: SR, CVR, RR, intervention rate, control latency.

Usage: python experiments/rq2_eval.py --task push
"""
import os
from copy import copy

import numpy as np

from _common import (base_parser, calibration_path, env_for_cell, experiment_metadata,
                     load_policy, load_predictor, setup)
from registry import build_verifier
from checkvla.verifier import Controller, run_episodes, summarize
from common import grid_cells, is_ood, load_json, save_json
from _paired import paired_outcomes, pooled_paired


def build_controller(arm, policy, sim, od, dev, taus, args):
    kw = dict(chunk_k=args.chunk_k)
    if arm == "none":
        return Controller(policy, sim, "none", **kw)
    if arm == "gt_shadow":
        return Controller(policy, sim, "gt_shadow", commit=args.commit, **kw)
    modes = {'checkvla_orbisim_discrete': 'discrete', 'checkvla_orbisim_progress': 'progress'}
    configured = copy(args)
    role = 'orbisim' if arm in modes or arm == 'checkvla_orbisim_legacy_tau' else arm.replace('checkvla_', '')
    if arm in modes:
        configured.repair_mode = modes[arm]
    tau = taus[role]
    if arm == 'checkvla_orbisim_legacy_tau':
        if not args.legacy_calibration_path:
            raise ValueError('legacy threshold arm requires --legacy_calibration_path')
        tau = load_json(args.legacy_calibration_path)['tau'][role]
    ver = build_verifier(configured, load_predictor(args, od, role, dev, sim), sim, tau)
    return Controller(policy, sim, "checkvla", verifier=ver, **kw)


def main():
    p = base_parser(__doc__)
    p.add_argument("--n_envs", type=int, default=64)
    p.add_argument("--with_noact", action="store_true")
    p.add_argument("--arms", nargs="+", default=None)
    p.add_argument("--chunk_k", type=int, default=5,
                    help="steps between policy calls (open-loop chunk execution)")
    p.add_argument("--rollout_seed", type=int, default=3000,
                    help="episode seed base; use disjoint seeds for validation and test")
    p.add_argument("--save_episodes", action="store_true",
                   help="save paired per-episode outcomes and bootstrap intervals")
    p.add_argument("--split", choices=["validation", "confirmation", "test"], default="validation")
    p.add_argument("--overwrite", action="store_true", help="explicitly allow replacing a result")
    p.add_argument('--legacy_calibration_path', default='', help='explicit old-tau comparison artefact')
    p.add_argument('--cell_indices', type=int, nargs='+', default=None,
                   help='optional grid subset for implementation smoke tests, not headline evaluation')
    args = p.parse_args()
    dev, sim, od = setup(args)
    tag = args.tag or ("" if args.policy == "bc" else args.policy)
    sfx = f"_{tag}" if tag else ""
    result_path = os.path.join(od, f"rq2{sfx}.json")
    partial_path = os.path.join(od, f"rq2{sfx}.partial.json")
    if not args.overwrite and (os.path.exists(result_path) or os.path.exists(partial_path)):
        raise FileExistsError(f"refusing to overwrite {result_path} or its partial checkpoint")
    if args.quick:
        args.n_envs = 8
    policy = load_policy(args, od, dev, sim)
    threshold_path = calibration_path(args, od)
    taus = load_json(threshold_path)["tau"]
    metadata = experiment_metadata(args, od, policy, threshold_path)
    arms = args.arms or (["none", "gt_shadow", "checkvla_vision", "checkvla_orbisim"]
                         + (["checkvla_vision_noact"] if args.with_noact else []))
    ctrls = {a: build_controller(a, policy, sim, od, dev, taus, args) for a in arms}
    metadata['arm_configuration'] = {
        a: {'mode': c.mode, 'tau': getattr(c.verifier, 'tau', None),
            'repair_mode': getattr(c.verifier, 'repair_mode', None),
            'candidate_scales': [1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.15, 0.05]
            if getattr(c.verifier, 'repair_mode', None) in ('discrete', 'progress') else None}
        for a, c in ctrls.items()}

    cells = grid_cells(args.task)
    cell_indices = args.cell_indices if args.cell_indices is not None else list(range(len(cells)))
    if not cell_indices or len(set(cell_indices)) != len(cell_indices) or any(
            i < 0 or i >= len(cells) for i in cell_indices):
        raise ValueError('cell_indices must be distinct valid grid indices')
    metadata['evaluated_cell_indices'] = cell_indices
    # the most stressful OOD cell (by uncorrected CVR) is traced for Fig F
    results, traces = [], {}
    for ci in cell_indices:
        cell = cells[ci]
        env, params = env_for_cell(args, args.n_envs, dev, cell)
        entry = {"cell": cell, "ood": is_ood(args.task, cell)}
        if args.save_episodes:
            entry['episodes'] = {}
            entry['episode_seed'] = args.rollout_seed + ci
        for a, c in ctrls.items():
            res = run_episodes(env, sim, c, params, seed=args.rollout_seed + ci,
                               record=a.startswith("checkvla"),
                               audit_repair=a != "none")
            entry[a] = summarize(res)
            if args.save_episodes:
                keys = ('success', 'violation', 'first_viol', 'first_trig',
                        'n_interventions', 'first_success', 'final_task_cost',
                        'stall_decisions', 'n_decisions')
                entry['episodes'][a] = {k: res[k] for k in keys if k in res}
            if "trace" in res:
                vi = int(np.argmax(res["violation"])) if res["violation"].any() else 0
                traces.setdefault(str(ci), {})[a] = {k: v[vi] for k, v in res["trace"].items()}
        results.append(entry)
        if args.save_episodes and 'none' in arms:
            entry['paired_vs_none'] = {
                a: paired_outcomes(entry['episodes']['none'], entry['episodes'][a])
                for a in arms if a != 'none'}
        # Persist every completed cell atomically, including the actual episode
        # pairs. A long run must not lose all outcomes on interruption.
        save_json({'complete': False, 'arms': arms, 'metadata': metadata,
                   'policy': args.policy, 'per_cell': results}, partial_path + '.tmp')
        os.replace(partial_path + '.tmp', partial_path)
        print(f"[rq2] {cell} " + " | ".join(
            f"{a}: SR {entry[a]['SR']:.2f} CVR {entry[a]['CVR']:.2f} "
            f"safe {entry[a]['safe_success']:.2f} {entry[a]['latency_ms_mean']:.1f}ms"
             for a in arms), flush=True)
        expected_calls = int(np.ceil(sim.T / min(args.chunk_k, policy.H)))
        for a, c in ctrls.items():
            if c.n_policy_calls != expected_calls or c.n_policy_rows != expected_calls * args.n_envs:
                raise RuntimeError(f"{a}: unmatched policy-call cadence: "
                                   f"{c.n_policy_calls} calls / {c.n_policy_rows} rows")

    # pooled over OOD cells
    pooled = {}
    for a in arms:
        ood = [r[a] for r in results if r["ood"]] or [r[a] for r in results]
        # union, not ood[0]: arms carry optional keys (the repair audit appears
        # only where the arm actually intervened), so one arm's key set is not
        # a valid template for another's cells
        keys = {k for x in ood for k in x}
        pooled[a] = {}
        for m in keys:
            values = np.asarray([x.get(m, np.nan) for x in ood], dtype=float)
            pooled[a][m] = float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else float('nan')
        # intervention counts add, they do not average
        if "n_interventions" in keys:
            pooled[a]["n_interventions"] = int(sum(x.get("n_interventions", 0) for x in ood))
    base = pooled.get("none", {}).get("CVR", np.nan)
    for a in arms:
        pooled[a]["CVR_reduction_vs_none"] = float(1 - pooled[a]["CVR"] / base) if base > 0 else float("nan")
    worst = int(np.argmax([r["none"]["CVR"] if "none" in r else 0 for r in results]))
    # The results dir also holds the trained predictors and taus, so it cannot
    # move; only the result files are renamed. A non-bc policy always gets a
    # tag, so a VLA run can never overwrite the bc headline table.
    paired = {a: pooled_paired(results, a, args.rollout_seed + 99)
              for a in arms if a != 'none'} if args.save_episodes and 'none' in arms else {}
    save_json({"complete": True, "arms": arms, "per_cell": results, "pooled_ood": pooled,
               "paired_vs_none": paired,
               "policy": args.policy, "metadata": metadata,
               "fig_f_cell": results[worst]["cell"]},
               result_path + '.tmp')
    os.replace(result_path + '.tmp', result_path)
    print(f"[rq2] wrote {result_path}", flush=True)
    worst_grid_index = cell_indices[worst]
    if str(worst_grid_index) in traces:
        np.savez(os.path.join(od, f"rq2_trace{sfx}.npz"),
                  **{f"{a}__{k}": v for a, d in traces[str(worst_grid_index)].items() for k, v in d.items()},
                 tau_orbisim=taus.get("orbisim", 1.0), tau_vision=taus.get("vision", 1.0))
    print("[rq2] pooled OOD:", {a: {m: round(v[m], 3) for m in
                                    ("SR", "CVR", "safe_success", "CVR_reduction_vs_none")}
                                for a, v in pooled.items()})
    print("[rq2] repair audit - GT risk of the chunk actually executed:")
    for a in arms:
        v = pooled[a]
        if np.isfinite(v.get("gt_risk_rel_drop", np.nan)):
            print(f"       {a:22s} GT risk {v['gt_risk_proposed']:.3f} -> "
                  f"{v['gt_risk_applied']:.3f} ({100 * v['gt_risk_rel_drop']:+.1f}%)  "
                  f"cleared {v['frac_cleared']:.2f}  no-op {v['frac_ineffective']:.2f}  "
                  f"mag x{v['mag_ratio_mean']:.2f}  abst {v.get('frac_abstain', float('nan')):.2f}"
                   f"  n={v['n_interventions']}", flush=True)
    for arm, p in paired.items():
        print(f"[paired] {arm}: safe rescue={p['safe_rescue_count']} "
              f"harm={p['safe_harm_count']} delta={p['delta_safe_success']:+.4f} "
              f"CI95={p['paired_intervals']['safe_success']['ci95']}", flush=True)


if __name__ == "__main__":
    main()
