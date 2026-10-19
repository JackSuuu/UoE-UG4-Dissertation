# Actor failure audit results: capability exists, refrigerator transfer is unresolved

## Bottom line

The same frozen LIBERO10 OpenVLA grasps and transports objects in native benchmark
tasks, but stalls under the custom refrigerator observations even with official
TF4.40.1 inference. Changing only the book task's camera to the refrigerator-side
pose removes native success and grasp in this one initial state. That camera also
loses important task visibility, so this is not an isolated viewpoint-equivariance
test and does not prove the refrigerator failure is entirely a camera problem.

R2's manual geometric repairs supplied its first grasp/extraction. These new trials
contain **zero repair actions** and establish no framework improvement. They diagnose
the baseline rather than turning scripted competence into an autonomy claim.

## Frozen experiment and verification

Protocol and actor source freeze: `1477011`. Five planned conditions completed with
1,997 executed model actions, each independently replayed with exact physics,
controller, image fingerprints and recorded task diagnostics. Replay does not imply
model re-query reproducibility. Each run had the same 600-action/query cap and
1,200-second episode wall cap; native `environment_done` stopped successful native
tasks early. Every executed action exactly equals its recorded model proposal.

Book current/pinned initial physics, controller and both camera images match exactly.
Book native/side initial physics and controller match exactly; images intentionally
differ. New official-stack fridge and historical R2 baseline also match initial
physics/controller/images. Both fridge tapes use the same instruction throughout
all 600 steps. The historical R2 runner additionally has phase prompts and a strict
completion stop; neither changes the realized baseline sequence or stopping outcome.
Thus this is a historical diagnostic control, not a new fully identical-runner
framework comparison. All failures are retained.

## Outcomes

First-event steps are zero-based. “Native” is LIBERO's task predicate, not our strict
released/supported/stationary placement endpoint. These are single development
initial states, not estimates of general success rates.

| Task / camera / inference | Actions executed | First grasp | First native success | Near-zero XYZ commands | Strict endpoint |
|---|---:|---:|---:|---:|---|
| Book / native / existing TF5 | 196 | 70 | 195 | 0 | Not verified; stops at native done |
| Book / native / official TF4 | 359 | 67 | 358 | 6 | Not verified; stops at native done |
| Book / robot-side / official TF4 | 600 | None | None | 410 | No complete placement |
| Cheese+butter / native / official TF4 | 242 | 63 (cheese) | 241 | 1 | Strict dwell unscored |
| Refrigerator / robot-side / official TF4 | 600 | None | None | 576 | No grasp, no full-task success |
| Refrigerator / historical TF5 baseline | 600 | None | None | 504 | No grasp, no full-task success |

Near-zero means L2 of the three normalized controller translation commands <0.01,
a descriptive post-hoc threshold; it does not measure physical immobility. Official
fridge inference has only 20 distinct seven-token sequences over 600 actions. Item
displacement stays below 8e-13 m; EEF endpoint displacement is approximately 0.040 m.
Book side-view has seven token sequences, no item motion above numerical noise,
and about 0.272 m EEF endpoint displacement. Thus a stall need not mean the robot
never moved; it can approach an unproductive pose and then repeat commands.

Both native-book trajectories move the item over 0.43 m, demonstrating substantial
learned manipulation. Neither ever produces a strict placement candidate or dwell
interval. At native termination both fail full-footprint containment, seated-bottom,
whole-interval release and slow-motion checks. The official trajectory also lacks
final solver-sample caddy support. Stopping at the native predicate censors their
subsequent behavior: do not claim they would or would not eventually stably finish.
Cheese's native predicate concerns both cheese and butter in the basket; logged grasp
is for cheese only, and no strict two-object release/dwell monitor is implemented.

## What the controls establish

1. **The actor/execution interface is not globally incapable of manipulation.** Both
   stacks grasp/transport the book, and the official stack reaches the two-object
   native basket predicate. This does not certify all simulator/control interfaces
   or complete manipulation competence.
2. **Visual conditions can cause failure without changing physical difficulty.** In
   the matched book initial state, the camera intervention alone changes successful
   native behavior to no grasp within 600 actions. The caddy and much of the gripper
   are outside the new view. Viewpoint, framing and visibility change together.
3. **Official inference is not a sufficient refrigerator fix.** It still fails to
   grasp and emits even more near-zero commands in this trial. It is therefore wrong
   to explain this failure solely by the TF5 compatibility layer.
4. **The stacks are not equivalent.** With identical initial states/images, book
   tokens first diverge at step 3 and fridge tokens at step 6. The fingerprints at
   each first divergence still match, so this is a genuine inference-path/environment
   difference rather than later state feedback. Subsequent action differences cannot
   be counted as fixed-observation disagreements because trajectories diverge. The
   six-case offline discrepancy is not harmlessly dismissed. Multiple library/runtime
   versions differ; no specific numerical kernel or library is identified as root cause.
5. **Task-distribution mismatch remains a supported hypothesis, not a separated
   causal result.** The LIBERO10 fine-tuning suite lacks refrigerator tasks, and custom
   fridge geometry/camera/occlusion differs from the native tasks. The present cohort
   does not isolate refrigerator task semantics from visual domain and physical
   arrangement. Base pretraining exposure is not established by the suite inventory.

Latency median is approximately 0.28–0.29 s per official action versus 0.71 s for
the current adapter. This includes preprocessing and IPC where applicable, under
the tested concurrent GPU workloads, and is not an isolated kernel benchmark.
Simulation pauses during inference. Video uses 20 Hz simulation time, not 20 Hz
real-world inference; these latency values cannot explain the physical simulation
stall through stale asynchronous actions because there is no asynchronous controller.

## Consequence for the research method

The previous comparison combined a struggling out-of-distribution actor with an
extra hand-written, privileged controller. That can produce local progress while
leaving the actual scientific question unanswered. No current evidence isolates a
verifier benefit, autonomous repair generation or successful downstream handoff.

The next improvement must address an identified failure without encoding the motion
answer. A fixed, declared visibility/observation correction shared by both arms is
a defensible next diagnostic; it is not itself framework efficacy. Its camera rule
must be chosen before outcome inspection, preserve physical initial conditions and
avoid choosing the best of many rollout-tuned views. If the actor still lacks the
new task skill, use declared shared learning/adaptation or a capable shared actor
before testing recovery. Keep the adapted actor frozen during the actual comparison.

Retain the five layers: high-level subgoals → frozen VLA → risk/event monitor →
dual-mode gate → bounded local repair. The unresolved part is an autonomous repair
proposal source (for example a learned policy or constrained candidate search),
not another hand-written motion sequence. Any candidate source must be shared with
simple-feedback controls and its information, search/query budget and failures
reported. Verification must separately demonstrate detection, selection, recheck,
actual handoff and complete downstream completion. These are requirements, not
implemented capabilities or established gains.

## Review and recovery

[Artifact index](../../backups/vla_failure_audit_20261019/README.md) contains all five
run tapes, frozen sources, independent replay records, metrics and four annotated
videos: book inference pair, book camera pair, fridge inference pair and the native
cheese control. These are explicitly **actor diagnostic controls**, not a fake
baseline/framework pair. Every source frame is retained at 20 fps, with common
simulation speed, zero-intervention/handoff labels, diagnostic state, timings and
explicit end-frame holds. Holds are not extra physics or verified dwell.

Primary data: `/home/ami/scratch/vla_failure_actor_20261019_a`.
Analysis/review: `/home/ami/scratch/vla_failure_actor_review_20261019_a`.
Reproducible postprocessing: `scripts/summarize_vla_failure_audit.py`.

Published artifact commit: `4d67845`. Fresh GitHub/LFS recovery verified all185archive
members and four standalone MP4s exactly. [Recovery proof](../../backups/vla_failure_audit_20261019/cohort_remote_verified.json).
User-edited `notes/Experiment_plan_1year.md` remains untouched and uncommitted.
