# D1 verified outcome: no observed post-event completion headroom

Run: `/home/ami/scratch/libero_drawer_target_20261014_a`. Frozen [protocol](libero_drawer_target_protocol_20261014.md), [implementation/launch](libero_drawer_target_launch_20261014.md). Supervisor status: `no_observed_post_event_headroom_verified`; all three workers exited0. No intervention workers were launched. No actor training or further root/noise sweep occurred.

## Result and attribution

**Frozen VLA completes the full registered physical task on3/3development roots**, in368,564and344executed intervals. This is not merely native point-containment success: all collision-footprint, seating, floor support, vertical fit, gripper detachment from bowl/cabinet, shallow contact and slow-motion conditions hold for20consecutive intervals. Native-ever is also3/3.

| Root | First native success | Supported event / next decision trigger | Released milestone | Clear milestone | Closed with retained bowl | Full physical completion | Executed intervals |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 240 | 175 / 176 | 175 | 179 | 240 | 367 | 368 |
| 1 | 273 | 156 / 157 | 204 | 222 | 273 | 563 | 564 |
| 2 | 251 | 270 / 271 | 167 | 167 | 253 | 343 | 344 |

Milestone values are zero-based executed interval indices. A decision trigger occurs on the next action after evidence has accumulated. No changed actions, intervention intervals or forecasts occurred. The3/3observation is a small, consumed development sample; it does not establish population reliability or generalization.

The registered prerequisite required at least one baseline to reach the event and fail full completion. All event-reaching baselines completed, so D1stopped after three episodes rather than spending another nine on a completion-gain comparison with no observed headroom. `comparisons` is empty in the verified summary. **There is no measured framework improvement, harm estimate or online repair efficacy result from D1.** The implementation and its synthetic tests are not substituted for an intervention evaluation.

## Specific findings for the fixed architecture

1. **This task/cohort does not reproduce the book/caddy post-placement failure.** From the first closed-with-retained-bowl milestone onward, all three baselines have zero observed bowl/gripper-contact intervals. These are raw post-closure measurements, not a post-repair-handoff result. `handoff` is null in every baseline; the result file's zero post-handoff count must not be interpreted as successful framework handoff.
2. **Completion delay is mainly cabinet detachment, not bowl re-contact.** After the first closed-with-retained-bowl interval, there are106,188and51cabinet/gripper-contact intervals on roots0/1/2. Multiple acceptance conditions can fail simultaneously, so these counts are not mutually exclusive stage labels. The actor eventually detaches and achieves the1sphysical dwell within the shared600step cap. This suggests a possible future efficiency question, but does not justify changing D1's primary completion objective after seeing its outcomes.
3. **Event coverage must include timing relative to the intended phase.** Root2's supported-placement trigger occurs at decision271, after release/clearance at167and closed-with-retained-bowl at253. The conjunction of3stable supported intervals is therefore late for the intended release-to-close transition on this root. Merely reporting3/3trigger coverage would conceal that mismatch. The original logs/protocol are preserved; no trigger was loosened retrospectively.
4. **Observable proofs are working, but only baseline routing is exercised online here.** All1,276real actor intervals have causal evidence, numeric predicate/dwell values, baseline-pass routing, nominal/executed actions and separately labeled observed outcomes. All are recomputed by exact original-root replay. The gate/repair branches were unit-tested and the export path exercised with synthetic fixtures; actual actor-driven intervention evidence remains pending.

## Verification and review

The verifier replayed all1,276actual actions from original task roots without loading the actor. Exact integration/controller/camera fingerprints, states,25substep contact records, reconstructed geometry, causal decision traces, milestone/dwell histories, native command mapping, query/termination budgets and video lengths passed. There are27per-episode artifact hashes. Videos and clickable traces were generated for all three roots.

Files named `root_<i>_methods_2x.mp4` contain **one VLA panel only in this stopped run**, because no intervention arm executed. They are not four-method comparisons. Each `review/root_<i>_trace.html` presents the expandable step-by-step proof and frame-linked timeline. Full raw traces are `root_<i>/vla/decision_trace.jsonl`. Collision Rate/CVR/safe success remain null; raw contact counts are not collision-safety evidence.

## Consequence and preservation

D1is closed under its predefined stop rule. Keep the five-layer architecture and explicit proof logging. For the next efficacy-targeted edit, the previously demonstrated book/caddy release/resumption failure is the stronger known failure case; a future transition-focused protocol should be built from that evidence before returning to the drawer task for transfer. Do not add disturbances, search new roots or redefine success simply to obtain a positive D1comparison. No further experiment is automatically launched by this result.

Recovery archive: `backups/libero_drawer_target_20261014_a/`, one archive with40original files plus three standalone videos;26,321,596bytes across four LFS objects. Automatic local per-file verification passed. Remote recovery verification is recorded separately after push.

**Remote recovery passed:** archive/results pushed as `81eb54e`, independently downloaded from GitHub and fully restored with all40source-file hashes and three standalone-video hashes verified. [Recovery record](../../backups/libero_drawer_target_remote_verified_20261014.json).
