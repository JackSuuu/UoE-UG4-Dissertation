# Book/caddy feasibility pilot: native success3/3; released stable placement unestablished

Run `/home/ami/scratch/libero_book_pilot_20261013_a` completed on all three workers. Protocol/source commit `bf7c6c6`, frozen [protocol](libero_baseline_protocol_20261013.md). Artifact verification passed: complete manifests, ordered action records, mapped gripper/native actions, finite states, state/video length agreement and per-file SHA256. Results/report: `verified_summary.json`, `index.html`; every episode has `rollout.mp4`, `initial.png`, `policy_input.png`, `final.png`, actions/contact JSONL, states NPZ and compiled XML.

| Development initial index | Native success | Policy steps | Rollout wall seconds | Terminal gripper command | Terminal book/gripper contact |
|---|---|---|---|---|---|
|0|yes|185|135.08|+1(closed)|yes|
|1|yes|229|169.83|+1(closed)|yes|
|2|yes|166|122.41|+1(closed)|yes|

Native SR3/3; descriptive Wilson95% interval43.85–100%. These are fixed development roots, not independent evidence for high generalization SR. CVR, safe-success and stable precision-success remain **null/unmeasured**, not0% or100%. All three stopped at the first native predicate, as preregistered; no post-success policy continuation or release/dwell was executed. No verifier gain has been tested.

## Important predicate correction discovered during terminal audit

The frozen protocol says `In` checks contact with the caddy and reference-point containment. This overstates the effective check for this task. Although `In.__call__` invokes `arg2.check_contact(arg1)`, its target is `SiteObjectState`, whose `check_contact` **always returns True** (`base_object_states.py:172–176`). Thus this task's native success effectively tests only reference-point containment in the region, including the1cm lower-Z extension. It does not require actual caddy contact, release, full book geometry containment, appropriate orientation or stability. This is an erratum to the pre-run source interpretation; the frozen protocol and native outcomes are preserved.

Final rendered frames and contact records show the robot still contacting the book at native termination in all three episodes. Closed gripper commands alone would not prove attachment, but the simultaneous gripper/book contacts confirm release has not been established. Episode2 has no caddy/book endpoint contact in its terminal log despite native success, consistent with the site predicate implementation. Episodes0/1 have caddy/book endpoint normal contact forces around92.27/55.04N respectively; these are individual logged endpoint values, not converged peak forces, total object force, or validated safety violations.

## Consequence for the next experiment

The full local VLA/simulator/action pipeline can grasp, transport and reach the target region. **Complete released, stable task execution is not yet demonstrated.** Do not proceed from3/3native success directly to learned-verifier training or claim the intended precision baseline is solved.

Next bounded protocol should continue the same actor after the native predicate (native success becomes a logged event, not immediate termination), retain the complete time budget and define geometry-supported release/stability acceptance before the rerun. Collect substep contact diagnostics and validate numerical sensitivity before fixing safety thresholds. If the policy cannot finish release/stability, identify whether termination/controller/action decoding or task capability is responsible before attributing the failure to physical prediction. Exact-simulator intervention headroom and learned replacement follow only after this task-level acceptance is meaningful.

No experiment remains running after this pilot. No learned model was trained, no thresholds were fitted to these outcomes, and these three roots remain development-only.
