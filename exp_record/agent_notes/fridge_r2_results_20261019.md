# R2 completed: local progress, no complete-task recovery

## Frozen cohort and all-case outcomes

Source freeze `c4afde8`; all four runs completed 600 actual frozen-OpenVLA queries
and 600 actions. Both arms used shared phase-prompt rules, the same initial
physics/controller/images within each pair, and the same observation permissions,
action/query/wall-time caps and stopping rules. No injected disturbances.
The wrapper used privileged simulation state/contact and scene-specific primitives.
This is development integration evidence, not a generalizable repair result.

| Initial door | Arm | Item grasp observed | Held extraction observed | Corrections | Remaining sequence completed |
|---|---|---|---|---:|---|
| Closed | VLA + shared phase prompts | No | No | 0 | No |
| Closed | VLA + wrapper | No | No | 420 | No |
| Open | VLA + shared phase prompts | No | No | 0 | No |
| Open | VLA + wrapper | Yes | Yes | 420 | No |

Neither closed-door arm opened the door. Open-start runs are not full closed-door
task trials. The open wrapper first recorded item grasp at zero-based step 242 and
held extraction at 299; placement dwell never started. The task tracker returned
from `place` to `extract` at 368: the clearance condition was lost during transport.
Final phase was `grasp`. Local progress did not become a complete retrieval episode.
Both wrapper runs used 70% of all actions for corrections, leaving 180 nominal
actions each. This high occupancy is a concern for the claimed *local* role.

Mean inference time per query: closed actor 0.7066s, closed wrapper 0.7170s,
open actor 0.7156s, open wrapper 0.7263s. Simulation pauses during inference;
20fps videos show simulation time, not real-time policy throughput.

## Replay and event-accounting correction

Three runs passed exact two-camera/physics/controller/decision replay. Open-wrapper
physics, controller, gate, task decisions and actor-facing images replayed exactly;
two wrist frames differed: step 237 has two RGB channels differing by 1, step 276
has one channel differing by 1. The original `replay_mismatch` label is retained,
so supervisor status remains `complete_with_issues`, despite all rollouts and videos
being completed. No tolerance was introduced to turn this into bit-exact success.
Replay consumes recorded proposals; it does not re-query or verify actor decode.

**Correction to the preliminary “zero verified handoffs” statement:** the frozen
controller stores a single event string. `retain_grip` overwrites `verified_handoff`
when both occur in one decision. `audit_fridge_r2_handoff.py` independently replayed
all 600 open-wrapper actions using frozen sources and verified physics/controller,
gate outputs and task tracking. It reconstructed:

| Local verified exit | Step | Actual owner on that step | First subsequent unmodified VLA action |
|---|---:|---|---:|
| Grasp | 245 | Repair: retain grip | 247 |
| Extraction | 302 | Repair: retain grip | 493 |

There were **two verified local exits**, neither an immediate unmodified-policy
handoff. The first later nominal action at 493 also follows another placement
repair and total-budget exhaustion; it is not prompt, successful extraction handoff.
Neither exit led to remaining-sequence completion. The original logs and status
fields are preserved; `handoff_audit/audit.json` is a separate post-hoc correction.
Future instrumentation must store multiple events and distinguish condition recovery,
actual command ownership, and downstream completion. The frozen R2 runner is unchanged.

## Door failure diagnosis (development branches, not new efficacy trials)

Replaying the closed-wrapper trajectory showed fingers closing in air: pad-to-handle
surface distances were approximately 32mm after closure. The object-grasp EEF offset
had been reused for the handle, and the transition to pulling checked proximity
rather than bilateral contact. The pull target also moved away before establishing
grip. This is an implementation failure, not evidence that a verifier cannot help.

Four completed post-hoc branches reproduced the same 29-action VLA prefix and exact
initial physics/controller/both-camera fingerprint. All selected this known failure;
they are development adaptation, not held-out results:

| Variant | Action cap | Bilateral door contact | Maximum door angle | Verified opening |
|---|---:|---|---:|---|
| contact_v1 | 200 | No | approximately 0 | No |
| contact_v2 | 200 | Yes | 0.20546rad (11.77deg) | No |
| contact_v2_budget420 | 420 | Yes | 0.58853rad (33.72deg) | No |
| contact_v3_budget420 | 420 | Yes | 0.97717rad (55.99deg) | No |

V2 aligns pad centers and requires contact before pulling. V3 freezes an open-jaw
centering offset instead of following asymmetric finger motion. Contact detection
in these probes accepts door geoms 15–19, not exclusively handle contact: do not
overstate it as verified handle grasp. A proposed v4 re-centering branch was
interrupted before a rollout/report; only scene construction and its log remain.
The current development source contains that unvalidated v4 change. No successful
door-opening result or independent full replay is claimed for these branches.

## User-directed generalization correction

The user correctly identified that fixed world-space approach heights, asset geom
IDs, scene-specific offsets and outcome-driven tolerance changes make this repair
scene-specific. Wiring it into five layers does not establish a reusable method.
Further per-scene coordinate tuning is paused. Preserve the failed/incomplete
branches and use them only to diagnose the control/verification interface.

The next method gate is **one frozen repair implementation across held-out layouts
and object instances**, with inputs supplying object-relative frames, hinge axis,
grasp affordances, collision geometry, robot limits and uncertainty. A shared engine
should consume violated preconditions and bounded skill proposals, then revalidate
and record actual handoff. Task-family representations (hinged rigid objects versus
cloth) may differ, but each environment must not require new routes or thresholds.
Removing world coordinates alone does not establish generality or feasibility.

Use oracle state first to isolate control transfer, then estimated state with declared
sensors. Compare the same primitives under simple feedback versus the verification
gate; isolate trigger, repair and handoff; report adaptation cost, failures, occupancy
and full-task completion. A nominal actor with no basic task skill must be improved
through declared shared adaptation or replaced by a capable shared frozen actor.
Local repair must not silently become the entire task policy. These are next-version
acceptance criteria, not implemented or demonstrated capabilities.

## Delivery

[All-case archive and paired videos](../../backups/fridge_r2_20261019/README.md).
Both MP4s have 601 frames at 20fps, common simulation clock, baseline left and
wrapper right, phase/ownership/event/budget labels, and explicit end-frame labels.
The displayed event is the original single-event log; the audit above qualifies it.
All failed trials and interrupted door diagnostics are retained.

Independent GitHub/LFS recovery from commit`afd97b0` passed: all200R2archive members,
both standalone paired MP4s, and all374R1actor-cohort members match their SHA256
manifests. Recovery used a fresh repository at
`/tmp/opencode/fridge_r2_remote_restore_20261019`, with no local LFS cache reuse.
[R2 proof](../../backups/fridge_r2_20261019/remote_verified.json) ·
[R1 proof](../../backups/fridge_r1_20261019/actor_remote_verified.json).
