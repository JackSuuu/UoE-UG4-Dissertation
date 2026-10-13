# Continuation results: one released placement, two distinct task failures

Run `/home/ami/scratch/libero_continuation_20261013_a` completed on all three workers. Frozen [protocol](libero_continuation_protocol_20261013.md), pre-launch source commit `9f39e2b`. Full artifacts verified by `scripts/summarize_libero_continuation.py`: every component acceptance check/dwell recomputed, action mapping checked, all original-prefix tokens/actions identical and simulator-state max error **0** for each root, finite states,25physics samples/control interval, video/state lengths and artifact hashes. `verified_summary.json` and `index.html` contain the complete outcomes and all three videos. Additional terminal/component audit: `failure_audit.json`.

| Development root | First native success (1-based policy step) | Total steps | Released-placement completion | Best simultaneous dwell |
|---|---:|---:|---|---:|
|0|185|313|yes|20/20intervals|
|1|229|520|no|0/20|
|2|166|520|no|6/20|

Native success ever3/3; geometry-aware released placement1/3 (33.33%; descriptive Wilson95% interval6.15–79.23%). These are three fixed development cases, not a robust SR estimate, an improvement comparison or independent confirmation; the interval does not correct development-root selection. CVR and safe success remain null. No training or physical repair was applied. All runs have ended.

## What the trajectories establish

**Root0 — a task-capable example exists.** After first native success at185steps, the actor continues for another128steps. Book/gripper contact disappears from zero-based step280 onward; all registered conditions hold continuously at steps293–312. Final book footprint lies inside the rear compartment, bottom atZ≈0.01380992m versus floor0.01381m, with negligible motion. Thus local full7B can achieve the registered detached, seated1second placement, rather than merely reach a reference-point region.

The final command still closes the gripper (+1). Consequently this is **observed physical detachment**, not evidence that the actor intentionally issued an open-gripper release or completed a safe retreat. The test stops at first1second dwell; longer-term stability under continued actor commands remains unmeasured. Contact safety and robot/caddy interactions remain unclassified.

**Root1 — placement/alignment failure with persistent contact.** No control interval has the complete book footprint inside the rear compartment. After initial grasp contact begins (zero-based step73), there is no contact-free control interval. At step519, footprintX≈[−0.08703,−0.02984]m lies outside rear-compartmentX[0.00247,0.06277]m; the book bottom is≈0.05917m above the caddy origin, versus floor0.01381m. Final image shows a tilted book displaced toward the front region. Low final speed does not imply successful placement. These observations do not uniquely establish a mechanical jam, a perception error or a decoding issue.

**Root2 — brief placement followed by renewed robot contact.** The book is detached at steps187–193; all acceptance components pass at188–193 (0.3seconds), below the registered1second dwell. At194, renewed book/gripper contact and excess speed break acceptance; footprint remains valid initially, then leaves the bounds after200. This is a concrete continuation failure to investigate, not yet counterfactual proof that a particular intervention rescues it. At519the book is seated/slow but remains in gripper contact; its maximumX exceeds the footprint bound by only≈0.038mm. That tiny geometric excess must not be overinterpreted, since persistent contact and insufficient dwell independently prevent completion.

Substep solver readouts support real contact rather than merely contact-list membership: root2's maximum individual book/gripper normal force is≈9.56N at194,≈48.15N at200and≈71.17N at519; root1's terminal maximum is≈50.58N. These are raw solver diagnostics, **not validated physical force peaks or safety violations**. No numerical-convergence/force-limit claim follows from these values.

## Research implication and bounded next design

Baseline feasibility is now **partially demonstrated**: one full released-placement example, with large remaining completion variation. Neither native3/3 nor released1/3 proves reliable task performance or physical-verifier efficacy.

The next privileged mechanism diagnostic should address these observed stages: (a) alignment/placement while grasping and (b) preserving a completed placement during subsequent commands. Before learned training, compare bounded physics-guided corrections against matched simple phase-aware stopping/hold/retreat controls and the continuing actor. A simple completion/termination correction might explain a rescue without learned physics; that possibility must be measured, not hidden. Holds/retreats require their own task/contact checks and count against the same execution budget. Restoring branch states must also include controller/internal state, not just the stored qpos/qvel arrays.

Freeze the branch/control budget and observation permissions before collecting counterfactual outcomes; keep exact simulator state privileged and diagnosis-only. Validate substep contact labels and numerical sensitivity before specifying force/impact constraints. Do not train from these three diagnostic outcomes or launch another broad collector/loss sweep. No next intervention or training job is launched by this results entry.
