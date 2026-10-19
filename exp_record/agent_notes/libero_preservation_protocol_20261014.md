# Frozen placement-preservation mechanism protocol — 14 Oct 2026

Status: to be committed/pushed before the simulator branch smoke and registered rollout. Run ID: `libero_preservation_20261014_a`. This is a **privileged H1/k1 mechanism diagnostic**, not learned-framework efficacy, the historical H10/k5 protocol, or a generalization claim.

## Question and scope

Can bounded intervention preserve a physically valid placement through the frozen one-second dwell, and does accurate physical lookahead add benefit beyond simple hold/open or retreat/open? Root2's observed post-placement re-contact motivates this first mechanism. Root1's earlier alignment failure is deliberately a coverage limitation: this intervention cannot fire before an actual valid placement. Do not loosen the trigger retrospectively to rescue it.

Keep the task checkpoint, RGB preprocessing, native7D Panda OSC controller, nominal physics, seeds/initial-state loading and frozen geometry/release/support/motion/dwell acceptance unchanged. Frozen full7B actor; batch1. Safety/CVR remain null pending justified, numerically stable force/impact constraints. Necessary support contact is not a violation.

## Roots and arms

- Development mechanism roots0/1/2: already consumed for design/audit. Include all three, including the original success, to expose harms and coverage failure.
- Additional **development-only** roots3/4/5: fresh before this registration; same protocol, no intervening retuning. Report separately from0/1/2. They become consumed development data, not confirmation/test or later training labels.
- Four arms per root: `vla`, `hold`, `retreat`, `physics` (24trajectories total). Each uses new environment state and the same immutable actor weights. Each obtains one native proposal from its own current RGB every20Hz interval, including correction intervals. Actor inference after a correction uses the resulting image, never a baseline future-action tape.

All arms construct the same candidates on the same causal phase opportunities. `vla` always executes its unchanged current proposal; `hold` always chooses open+zero pose increment; `retreat` chooses open+up; `physics` ranks the six candidates using accurate simulated consequences. Both simple controls get the same privileged current/past completion measurements as `physics`. No future physics outputs enter either simple comparator. No arm-dependent termination rules or input-permission differences between the correction comparators.

## Trigger and bounded correction family

The first previously observed interval satisfying **all six** frozen placement conditions triggers a single preservation window at the next decision. The trigger reads current/past measurements only, never root IDs, a specific historical step, future labels or privileged baseline trajectories. Window length: at most20consecutive20Hz opportunities (1s). No retrigger. Corrections switch off immediately once the first20-interval completion dwell is achieved. An invalid interval after trigger does not renew the budget.

K=6 first-action candidates, fixed before rollout:

0. The unmodified current VLA action, including gripper.
1. Half of its six pose-command components; retain proposed gripper.
2. Zero pose increments; retain last executed gripper command.
3. Zero pose increments; open gripper (`-1`).
4. World-z increment`+0.1` in controller input units, all other pose increments zero; open gripper.
5. The same upward pose increment; retain last executed gripper command.

Controller checks require fixed-impedance delta OSC_POSE, input range±1, translation scaling±0.05m, orientation scaling±0.5rad, no interpolator,20Hz control and0.002s integration. Thus the upward alternative requests+5mm world-z per executed interval, at most100mm summed requested increments across the bounded window (actual EEF displacement is not assumed equal). Zero pose increments retain robosuite's orientation-goal semantics; this is not an assertion of perfect immobility. Gripper opening is eligible only inside the common measured-detached-placement phase window. Original/half actions remain subject to the unchanged controller limits.

Each candidate **forecast** consists of its first action followed by four zero-pose-increment intervals with that candidate's gripper command. This is a0.25s pulse-and-settle **controller primitive**, not five predicted VLA actions and not genuineH10. Execute only the first interval; get a fresh actor proposal and re-evaluate at the next20Hz opportunity. Maximum forecast budget600control intervals per episode (20×6×5). Log replay overhead separately. Simple arms enumerate identical first-action candidates but do not run prediction. Accurate simulation is a privileged diagnostic resource, not a claim of equal compute to the simple control.

## Branch integrity and candidate ranking

Build each candidate branch independently from the same root and10settling intervals, then deterministically replay **every action actually executed so far**. Do not install a qpos/qvel-only state or use counterfactual future baseline actions. At the branch point require bitwise equality of MuJoCo`mjSTATE_INTEGRATION` (including warm-start/control/internal integration state), numeric controller state, gripper accumulated command, environment counters and rendered camera observations. Restore global Python/NumPy RNG state after forecast construction, and assert that the live environment fingerprint is unchanged. A mismatch aborts the episode/study for investigation; it is not a fallback success.

Lexicographically minimize these fixed quantities across each five-interval forecast:

1. Number of intervals failing any frozen placement condition.
2. Number of intervals with book/gripper re-contact.
3. Sum of footprint-invalid and bottom-unseated interval counts.
4. Euclidean change in six normalized pose-command components plus1if the gripper differs from the current VLA proposal.
5. Candidate index, for deterministic ties.

This prefers the original action when it predicts no placement loss; no fitted scalar reward, force threshold, lookahead/candidate sweep or hand-selected root-specific correction. Forecast each candidate with a fresh completion monitor initialized to the live dwell count. Only these registered candidate futures enter ranking; no full-task future label is available to the selector.

## Complete execution and resumption

All arms have the same520-policy-step budget. Primary endpoint: **ever achieving the original frozen released-placement20-interval dwell**, reported separately from native-ever success. Rather than ending immediately, disable corrections and execute40additional **real VLA** intervals after first completion, subject to the same520step cap. This is a secondary post-completion resumption stress test, not a retroactive change to the historical task acceptance definition.

Report first completion time, continuous validity across all40actor-only intervals, and validity/dwell over their final20intervals. If fewer than40fit in the520budget, retention metrics are null (incomplete stress observation), not success. Also report correction count, trigger coverage, lost/recovered dwell, VLA/forecast/replay queries and elapsed time. Completion can remain true even if later resumption breaks placement; show both outcomes plainly. No contact-based safe-success percentage until safety labels are validated.

## Verification and reporting

Before rollout: controller/action and trigger/ranking unit checks, recorded-prefix simulator replay smoke at root2's first causal opportunity, all six forecasts from identical branches, predicted-first-step agreement with actual execution, and live-environment non-mutation. Smoke labels stay development diagnostics, not training data; do not retune registered constants from the smoke. Then launch on healthy GPUs1/2/3 using `setsid nohup ... < /dev/null`. Preserve source/protocol hashes and commit-before-launch evidence.

Afterward: independently recompute every acceptance component from states/logged geometry/substep contacts, verify executed actions against candidate selection, forecast ranking and first-step predictions, exact pre-intervention agreement across arms (and the historical continuation where available), budgets, actor-only resumption, artifact hashes and video lengths. Produce 2x MP4s labeled by method and phase. Report paired rescue/harm of each correction arm versus VLA and physics versus **each** simple control; use exact paired outcome counts with root-level intervals interpreted descriptively (six development roots are not powered efficacy evidence). Show0/1/2and3/4/5separately; no pooled OOD claim.

Decision after this study: if simple control fully explains any gain, report that and target alignment/other demonstrated failures before training. If accurate lookahead adds complete-task benefit over both simple controls without material harm, establish a permitted observation/estimation interface and collect separate training-root data for a small learned predictor. Accurate simulator benefit alone neither proves learnability nor completes the neural framework claim. Formal cadence and held-out efficacy remain separate work.
