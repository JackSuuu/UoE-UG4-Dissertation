"""
Generate Figures A, B, B2, C, D, E, F (plan §6) from results/<task>_<backend>/.

Usage: python experiments/make_figures.py --task push
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _common import base_parser  # noqa: E402
from common import OOD_GRIDS, load_json, out_dir  # noqa: E402


def savefig(fig, od, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(od, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {name}")


def fig_a(task, od):
    p = os.path.join(od, "rq2.json")
    if not os.path.exists(p):
        return
    d = load_json(p)
    g = OOD_GRIDS[task]
    a1, a2 = g["axes"]
    arms = d["arms"]
    for metric in ("SR", "CVR"):
        fig, axs = plt.subplots(1, len(arms), figsize=(3.2 * len(arms), 3), squeeze=False)
        for j, arm in enumerate(arms):
            M = np.full((len(g[a1]), len(g[a2])), np.nan)
            for r in d["per_cell"]:
                M[g[a1].index(r["cell"][a1]), g[a2].index(r["cell"][a2])] = r[arm][metric]
            ax = axs[0, j]
            im = ax.imshow(M, vmin=0, vmax=1, cmap="RdYlGn" if metric == "SR" else "RdYlGn_r",
                           origin="lower", aspect="auto")
            for (i, k), v in np.ndenumerate(M):
                ax.text(k, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
            ax.set_xticks(range(len(g[a2]))); ax.set_xticklabels(g[a2])
            ax.set_yticks(range(len(g[a1]))); ax.set_yticklabels(g[a1])
            ax.set_xlabel(f"{a2} ×"); ax.set_ylabel(f"{a1} ×")
            ax.set_title(arm, fontsize=9)
        fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.8)
        fig.suptitle(f"Fig A — {metric} over OOD grid ({task})")
        savefig(fig, od, f"figA_{metric}")


def fig_b(task, od):
    p = os.path.join(od, "rq1.json")
    if not os.path.exists(p):
        return
    d = load_json(p)
    if "fidelity_vs_horizon" in d:
        m = np.array(d["fidelity_vs_horizon"]["mean"]); s = np.array(d["fidelity_vs_horizon"]["std"])
        h = np.arange(1, len(m) + 1)
        fig, ax = plt.subplots(figsize=(4.5, 3))
        ax.plot(h, m, "o-", label="OrbiSim-Dynamics vs GT (gradient-valid states)")
        ax.fill_between(h, m - s, m + s, alpha=0.2)
        ax.set_xlabel("prediction horizon (steps)"); ax.set_ylabel("normalised state RMSE")
        ga = d.get("grad_agreement", {})
        ax.set_title(f"Fig B — fidelity decay | grad cos={ga.get('cos_mean', float('nan')):.2f}")
        ax.legend(fontsize=7)
        savefig(fig, od, "figB_fidelity")
    # trigger calibration bar chart
    S = d["summary"]
    preds = list(S)
    mets = ["auroc", "precision", "recall", "timely_recall", "false_alarm_rate"]
    fig, ax = plt.subplots(figsize=(6, 3))
    w = 0.8 / len(preds)
    for i, k in enumerate(preds):
        ax.bar(np.arange(len(mets)) + i * w, [S[k].get(m, np.nan) for m in mets], w, label=k)
    ax.set_xticks(np.arange(len(mets)) + w * (len(preds) - 1) / 2); ax.set_xticklabels(mets, fontsize=8)
    ax.set_ylim(0, 1); ax.legend(fontsize=7); ax.set_title("RQ1 — trigger calibration (pooled OOD grid)")
    savefig(fig, od, "figB_trigger_calibration")


def fig_b2(task, od):
    p = os.path.join(od, "audit.json")
    if not os.path.exists(p):
        return
    d = load_json(p)
    regs = d["regimes"]
    cells = d["torch_gt"]
    M = np.array([c["valid_frac_by_regime"] for c in cells])
    cnt = np.array([np.array(c["counts"]).sum(1) for c in cells])
    M[cnt == 0] = np.nan
    fig, ax = plt.subplots(figsize=(1.2 * len(regs) + 2, 0.35 * len(cells) + 1.5))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    for (i, k), v in np.ndenumerate(M):
        if not np.isnan(v):
            ax.text(k, i, f"{v:.2f}\n(n={cnt[i, k]})", ha="center", va="center", fontsize=6,
                    color="w" if v < 0.5 else "k")
    ax.set_xticks(range(len(regs))); ax.set_xticklabels(regs, fontsize=8)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels([", ".join(f"{k}={v}" for k, v in c["cell"].items()) for c in cells], fontsize=7)
    fig.colorbar(im, ax=ax, label="fraction gradient-valid")
    title = "Fig B2 — gradient path audit (differentiable GT)"
    if "genesis" in d:
        st = {}
        for r in d["genesis"]:
            st[r["status"]] = st.get(r["status"], 0) + 1
        title += f"\nGenesis probe: {st}"
    ax.set_title(title, fontsize=9)
    savefig(fig, od, "figB2_gradient_audit")


def fig_c(task, od):
    p = os.path.join(od, "rq3.json")
    if not os.path.exists(p):
        return
    d = load_json(p).get("mem")
    if not d:
        return
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    modes = sorted({r["mode"] for r in d["rows"]}, key=lambda m: ["naive", "ckpt_gpu", "ckpt_offload", "ckpt_offload_trunc"].index(m))
    for m in modes:
        rows = [r for r in d["rows"] if r["mode"] == m]
        h = [r["horizon"] for r in rows]
        axs[0].plot(h, [r.get("peak_mem_mb", np.nan) for r in rows], "o-", label=m)
        axs[1].plot(h, [r.get("seconds", np.nan) for r in rows], "o-", label=m)
        for r in rows:
            if r.get("oom"):
                axs[0].scatter([r["horizon"]], [axs[0].get_ylim()[1]], marker="x", color="r")
    if "genesis_naive" in d:
        g = [(r["horizon"], r["peak_mem_mb"]) for r in d["genesis_naive"] if r["status"] == "valid"]
        if g:
            axs[0].plot(*zip(*g), "k--", label="Genesis naive BPTT")
    for ax, yl in zip(axs, ["peak GPU memory (MB)", "wall-clock (s)"]):
        ys = np.concatenate([l.get_ydata() for l in ax.get_lines()] or [np.array([np.nan])]).astype(float)
        ax.set_xscale("log")
        if np.any(ys[np.isfinite(ys)] > 0):
            ax.set_yscale("log")
        else:
            ax.text(0.5, 0.5, "no data (CUDA required for memory)", transform=ax.transAxes, ha="center")
        ax.set_xlabel("horizon (control steps)")
        ax.set_ylabel(yl); ax.legend(fontsize=7)
    fig.suptitle(f"Fig C — memory / time vs horizon ({task}, seg={d['seg']}, B={d['n_envs']})")
    savefig(fig, od, "figC_memory_vs_horizon")


def fig_d(task, od):
    p = os.path.join(od, "rq3.json")
    if not os.path.exists(p):
        return
    d = load_json(p).get("sched")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(5, 3))
    names = list(d["modes"])
    ax.boxplot([d["modes"][n]["latency_ms_all"] for n in names], showfliers=False)
    ax.set_xticks(range(1, len(names) + 1)); ax.set_xticklabels(names)
    ax.set_yscale("log"); ax.set_ylabel("control-loop latency (ms/step)")
    a = d["modes"].get("async", {})
    ax.set_title(f"Fig D — fast/slow scheduling | async staleness={a.get('staleness_mean', float('nan')):.1f} steps",
                 fontsize=9)
    savefig(fig, od, "figD_latency")


def fig_e(task, od):
    p = os.path.join(od, "rq3.json")
    if not os.path.exists(p):
        return
    d = load_json(p).get("stab")
    if not d:
        return
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    axs[0].semilogy(d["modes"]["none"]["trace_raw"], label="raw (no stabilizer)", alpha=0.8)
    for m in ("clip", "relax"):
        if m in d["modes"]:
            axs[0].semilogy(d["modes"][m]["trace_used"], label=f"stabilizer: {m}", alpha=0.8)
    axs[0].set_xlabel("time step (reverse sweep)"); axs[0].set_ylabel("||dL/ds_t||")
    axs[0].legend(fontsize=7); axs[0].set_title(f"grad-norm trace, cell {d['cell']}", fontsize=9)
    for m, v in d["modes"].items():
        axs[1].plot(v["opt_curve"], label=f"{m} (logvar={v['log_grad_norm_var']:.2f})")
    axs[1].set_xlabel("optimisation iteration"); axs[1].set_ylabel("GT trajectory cost")
    axs[1].legend(fontsize=7)
    fig.suptitle("Fig E — adaptive gradient truncation")
    savefig(fig, od, "figE_gradient_stabilizer")


def fig_f(task, od):
    p = os.path.join(od, "rq2_trace.npz")
    src = "rq2"
    if not os.path.exists(p):
        p, src = os.path.join(od, "rq1_trace.npz"), "rq1"
        if not os.path.exists(p):
            return
    d = dict(np.load(p))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    if src == "rq2":
        for arm, c in (("checkvla_orbisim", "C0"), ("checkvla_vision", "C1")):
            if f"{arm}__score" not in d:
                continue
            ax.plot(d[f"{arm}__score"], color=c, label=f"{arm} score")
            ax.plot(d[f"{arm}__risk"].max(-1), color=c, ls=":", label=f"{arm} GT risk")
            tr = np.where(d[f"{arm}__trig"])[0]
            ax.scatter(tr, np.full(len(tr), 0.05), marker="v", color=c, label=f"{arm} intervention")
        ax.axhline(float(d.get("tau_orbisim", 1.0)), color="C0", ls="--", lw=0.8, label="τ orbisim")
        ax.axhline(float(d.get("tau_vision", 1.0)), color="C1", ls="--", lw=0.8, label="τ vision")
    else:
        ax.plot(d["step_risk"], "k:", label="GT step risk")
        for k in ("orbisim", "vision"):
            if f"score_{k}" in d:
                ax.plot(d[f"score_{k}"], label=f"{k} score")
    ax.axhline(1.0, color="r", lw=0.8, label="constraint limit")
    ax.set_xlabel("control step"); ax.set_ylabel("normalised risk")
    ax.legend(fontsize=6, ncol=2); ax.set_title(f"Fig F — CheckVLA trigger trace ({task})")
    savefig(fig, od, "figF_trigger_trace")


def fig_g(task, od):
    """AUROC does not predict CVR reduction (plan §5, RQ1/RQ2 join).

    Left: AUROC (can it *detect* a violating chunk) next to the CVR reduction it
    buys in RQ2, and next to the number that actually explains the ordering --
    the intervention count. The pixel model has the best AUROC and buys nothing,
    because it fires 2.7x as often as the state model. Ranked by AUROC the
    conclusion is backwards.

    Right: the mechanism. Mean predicted chunk score as the chunk is scaled
    down, which is exactly the curve repair's bisection walks. A flat or rising
    curve means the search has no signal to follow, whatever the AUROC. The
    action-blind ablation is the control: it is flat at zero.
    """
    p1, p2 = os.path.join(od, "rq1.json"), os.path.join(od, "rq2.json")
    if not (os.path.exists(p1) and os.path.exists(p2)):
        return
    d1 = load_json(p1)
    # rq1 keys predictors by bare role ("orbisim"), rq2 by arm name
    # ("checkvla_orbisim"); join on the role so the join can never silently
    # come out empty again.
    S, R = d1["summary"], load_json(p2)["pooled_ood"]
    preds = [k for k in S if f"checkvla_{k}" in R]
    if not preds:
        return
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.2))
    x = np.arange(len(preds))
    w = 0.28
    au = [S[k].get("auroc", np.nan) for k in preds]
    rd = [100 * R[f"checkvla_{k}"].get("CVR_reduction_vs_none", np.nan) for k in preds]
    # normalised to the largest, so the three bars are readable on one axis:
    # AUROC is a 0-1 score while CVR reduction is a percentage
    nint = [R[f"checkvla_{k}"].get("n_interventions", np.nan) for k in preds]
    mx = max([v for v in nint if v == v] or [1]) or 1
    ni = [100 * (v / mx if v == v else np.nan) for v in nint]
    for off, vals, lab in ((-w, au, "AUROC (RQ1)"),
                           (0.0, rd, "CVR reduction (RQ2)"),
                           (w, ni, "interventions (rel.)")):
        ax.bar(x + off, vals, w, label=lab)
        for i, v in enumerate(vals):
            if v == v:
                ax.text(i + off, v, f"{v:.0f}" if lab.endswith("rel.)") else f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(preds, fontsize=8)
    ax.set_ylim(0, 118); ax.legend(fontsize=7, loc="upper left")
    ax.set_title("AUROC does not predict the CVR reduction", fontsize=9)

    C = d1.get("ctrl_curve", {})
    if C:
        for k in preds:
            if k in C:
                ln, = bx.plot(C["scales"], np.asarray(C[k]["pred"]) * 100, "o-",
                              label=f"{k} predicted")
                bx.plot(C["scales"], np.asarray(C[k]["gt"]) * 100, "--", alpha=.5,
                        color=ln.get_color(), label=f"{k} GT")
        bx.invert_xaxis()
        bx.set_xlabel("chunk scale (1.0 = as proposed)")
        bx.set_ylabel("mean chunk score")
        bx.set_title("Controllability: what repair bisects on", fontsize=9)
        bx.legend(fontsize=6, ncol=2)
    savefig(fig, od, "figG_detection_vs_repair")


def fig_h(task, od):
    """Rate-matched CVR / safe-success curves (RQ2, the confound-removal figure).

    The headline table compares each predictor at *its own* split-conformal tau,
    so a gap there confounds "better risk model" with "looser threshold" --
    vision's calibrated tau fires 3x as often as orbisim's (rate 0.065 vs 0.022).
    Here each predictor's threshold is swept and the x-axis is the *measured*
    intervention rate, so the comparison is at equal intervention cost.

    Plotted points are the measured sweep points, not the interpolated
    at_rate() rows: the vision rate-tau curve is much steeper (fitted exponent
    -2.61 vs -2.16) and jumps 0.019 -> 0.134 between two of its own measured
    points, so interpolating inside that gap would draw a curve the data does
    not constrain. Where a predictor has no measured point, it has none drawn.

    Read-off: orbisim reaches CVR 0.156 at rate 0.029, a 32% reduction at
    unchanged SR. vision's measured points never beat the no-verifier baseline
    (0.262) at any rate that leaves the task intact -- its next point is rate
    0.134, where it has driven SR to 0.42. Its CVR=0.012 point at rate 0.193
    is not a win: SR is 0.046, so it "prevents" violations by never acting.
    """
    p = os.path.join(od, "rq2_matched_tau.json")
    p2 = os.path.join(od, "rq2.json")
    if not (os.path.exists(p) and os.path.exists(p2)):
        return
    d = load_json(p)
    base = load_json(p2)["pooled_ood"].get("none", {}).get("CVR", np.nan)
    base_safe = load_json(p2)["pooled_ood"].get("none", {}).get("safe_success", np.nan)
    arms = [a for a in d["arms"] if a in d.get("sweep", {})]

    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for ax, key, ylab, ref in ((axs[0], "CVR", "violation rate (CVR)", base),
                               (axs[1], "safe_success", "safe success", base_safe)):
        if ref == ref:
            ax.axhline(ref, color="k", ls=":", lw=1.0, label="no verifier")
        for a in arms:
            role = a.replace("checkvla_", "")
            sw = [q for q in d["sweep"][a] if q.get("pooled")]
            pts = sorted((q["pooled"]["intervention_rate"], q["pooled"][key])
                         for q in sw if q["pooled"][key] == q["pooled"][key])
            xs, ys = zip(*pts) if pts else ((), ())
            ax.plot(xs, ys, "o-", ms=3.5, label=role)
            cal = [q for q in sw if q.get("target_rate") is None]
            if cal:
                cp = cal[0]["pooled"]
                ax.scatter([cp["intervention_rate"]], [cp[key]], marker="*", s=110,
                           zorder=5, edgecolor="k", linewidth=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("measured intervention rate")
        ax.set_ylabel(ylab)
        ax.set_ylim(-0.03, 1.03)
    axs[0].set_title("CVR at matched intervention rate", fontsize=9)
    axs[1].set_title("safe success at matched rate  ($*$ = own calibrated $\\tau$)",
                     fontsize=9)
    axs[1].legend(fontsize=7, loc="lower left")
    savefig(fig, od, "figH_rate_matched")


def main():
    p = base_parser(__doc__)
    args = p.parse_args()
    od = out_dir(args.task, args.backend)
    for f in (fig_a, fig_b, fig_b2, fig_c, fig_d, fig_e, fig_f, fig_g, fig_h):
        try:
            f(args.task, od)
        except Exception as e:   # keep going: figures are independent
            print(f"[fig] {f.__name__} failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
