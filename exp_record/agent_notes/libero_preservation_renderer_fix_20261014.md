# Aborted preservation run and renderer-context fix — 14 Oct 2026

## Failure, not efficacy

Run `libero_preservation_20261014_a` was automatically aborted at root2/physics decision190. The previous decision189completed; on the next decision, branch0matched the live MuJoCo integration and controller hashes exactly but both camera hashes differed. The supervisor terminated the other two workers. No fresh root3/4/5episode had begun. There is no complete physics comparison and no framework-efficacy result from this aborted run.

The first one-decision smoke was insufficient: it checked predicted physical consequences, but did not render the live environment after branch destruction and then re-check a subsequent decision's observation. The recorded-action regression extended to consecutive decisions reproduced the same second-decision failure without loading the VLA. Reproduction log: `/home/ami/scratch/libero_preservation_renderer_repro_20261014.log`.

## Root cause and local correction

In installed robosuite1.4.1, `MjRenderContext.__init__` binds its EGL context, but `MjRenderContext.render` and `MjSim.render` do not bind the renderer before drawing. Constructing/closing forecasting environments changes thread-local current GL context. The main environment can consequently produce invalid camera observations despite identical physical state. A physics-only first-step check cannot detect this failure.

`libero_preservation.bind_renderer_context()` now wraps the render method with `self.gl_ctx.make_current()` before each draw, inside the existing simulator render lock. The change is project/process-local, applied to all preservation-study arms; installed packages and historical runtime modules are not edited. No force/geometry/dwell threshold, candidate, trigger, score, actor or budget was retuned from partial outcomes. The original protocol file remains frozen; restart run ID will be `libero_preservation_20261014_b` with this documented implementation correction.

## Regression and restart requirements

The three-consecutive-decision regression passed:18branches, full integration/controller/image identity, live physics/controller non-mutation and selected-first-step prediction/actual equality. Forecast/replay time57.7s. Saved `/home/ami/scratch/libero_preservation_smoke_20261014_b.json`. The further **eight-decision integrity probe also passed**:48branches over154.5s, selections`[0,0,0,0,2,2,2,2]`, including exact reconstruction of a modified executed prefix. Saved `/home/ami/scratch/libero_preservation_smoke_20261014_c.json` and [backed up here](../../backups/libero_preservation_smoke_20261014_c.json). Its recorded actions are test stimuli only, never counterfactual future VLA actions in an efficacy rollout. The real runner continues to query the actor on new images at every step. All seven unit tests still pass.

The [paired frame at state190](../../backups/libero_preservation_renderer_comparison_20261014.png) shows the old failure directly: physically identical baseline and post-branch states, but the post-branch camera image has severe incorrect texture/render content. This is an input-integrity bug, not a VLA or candidate-score failure. The corrected regression retains exact camera equality; no tolerance was relaxed.

The supervisor now requires a consecutive-decision smoke and records exit codes of all workers, including terminated peers. Restart all24episodes in a new directory after checks and commit/push; do not combine selected completed `_a`arms with `_b`physics results. This keeps renderer behavior matched across arms. Preserve aborted artifacts and publish them with Git LFS.

## Preserved partial observations

`aborted_verified_partial.json` independently checked eight completed episodes' action selection, contact/geometry acceptance, budgets and resumption. Root0VLA/hold completed and retained final dwell; retreat failed. Root1VLA/hold did not complete. Root2hold/retreat attained the original placement dwell but lost final dwell during actor-only resumption; VLA did not complete. These are incomplete development diagnostics, **not a completed paired study or evidence of physical-prediction benefit**. They motivate retaining both simple controls and full resumption measurements exactly as registered; no protocol adaptation was made from them.

Archive: `backups/libero_preservation_abort_20261014_a/`,100original files including raw incomplete runs, failures, supervisor logs and the reproduction log. Existing frozen baseline/recovery snapshots remain available separately.

**Backup/restart confirmed:** fix/regression/aborted archive pushed as `ec9ff87`. The new LFS archive was independently downloaded from GitHub and all100members fully restored with matching hashes ([verification](../../backups/libero_preservation_abort_remote_verified_20261014.json)). Run `_b` then launched after that push: supervisor749068; workers749073/749074/749075on healthy GPUs1/2/3, root allocation0/3,1/4,2/5. Launch manifests record `ec9ff87` and the eight-decision smoke. Outcomes remain pending; all24arms are rerun rather than reusing partial `_a`outcomes.
