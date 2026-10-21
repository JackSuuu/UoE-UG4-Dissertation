# Next experiment: useful candidates first, then verification attribution

## User-approved priority (19 October 2026)

The user approves the assessment and explicitly requests that the next experiments
answer two questions, before proceeding to cloth:

1. **Does the model-generated candidate set contain actions that advance the task?**
2. **Does verification select, recheck and hand back control better than simple feedback?**

The five layers remain subgoals → frozen VLA → risk/event monitoring → dual-mode
gate → bounded local repair. Refrigerator manipulation is the primary task.
Cloth manipulation experiments are deferred until this rigid-body mechanism stage
has been evaluated; the later sequence is rectangular-cloth single fold, then a
laid-out garment. Cooking remains a later extension, not an established capability.

## Evidence at this decision point

* Completed native-book model-only qualification: three initial states, each600
  model queries/actions. All attain native success at least once; none satisfies
  the additional strict endpoint or even one strict candidate interval. All three
  independent replays pass. The additional geometric endpoint still needs physical
  feasibility/semantic validation; this does not invalidate native benchmark success.
* Completed native-book three-arm pilot: nominal600actions/600queries,
  simple-feedback474/600, verification530/600. All fail strict completion; only
  nominal attains native success. Verification samples duplicate same-observation
  greedy actions54/70times. Its two actual handoffs follow local-budget exits,
  not verified-progress exits. Retain this adverse result.
* The original/framed fridge camera cohort is still running at the status check.
  Both seed0 rollouts finish600actions without grasp. Their original replay tools
  time out after300s due to repeated full-NPZ image decompression; the separately
  versioned cached verifier reproduces both exactly in approximately8–9s. Frozen
  rollout sources and original timeout records are retained.
* Follow-on closed-start fridge candidate v1 is already running on GPU2 at
  `/home/ami/scratch/experiment3_candidates_fridge_20261019_a`.
  Its nominal arm has finished and replayed; the simple-feedback arm is running
  at this check. This is an integration pilot, not an efficacy result.
* The LIBERO90 frozen-actor fridge transfer test is already running on GPU3 at
  `/home/ami/scratch/shared_actor_fridge90_20261019_a`. Closed-start execution and
  replay have finished; open-start is running at this check. This checkpoint has
  related primitive exposure but no identified LIBERO90 fridge training task.
  No trained refrigerator skill or full-task success is inferred from launch.

These are timestamped execution snapshots, not claims that running cohorts have
finished. Existing pilots retain their frozen protocols and all failures.

### Subsequent experiment-2 completion notification

The actor lane has now returned: both LIBERO90 fridge runs finished600actions,
with no closed-door opening or open-door grasp. All1200actions replay exactly;
with the book controls this is3000exact-replayed model actions across five runs.
The fridge normalization, token decoding, RGB, contacts and clocks also pass the
artifact audit. No shared fridge actor is qualified and no adaptation was trained.

The LIBERO90 test used the original robot-side camera and constant language;
the GPU2 candidate pilot uses the frozen reframed camera and shared stage prompts.
They are not matched weight-only comparisons. No in-domain positive control was
run for the new third-party checkpoint, so the result does not isolate missing
skill from observation/domain or checkpoint quality. Relevant fridge-trained
resources exist, but inspected policies have incompatible embodiment/interfaces.
See [fridge results](shared_actor_fridge90_results_20261019.md),
[book results](shared_actor_qualification_results_20261019.md) and
[remote-resource audit](shared_actor_fridge_pretrained_audit_20261019.md).
The Q1 diagnostic remains useful; formal Q2 efficacy still needs shared capability.

### Subsequent Q1 completion notification and parent review

The GPU2 lane has completed both the frozen fridge pilot and the queued Q1 cohort.
Fridge nominal/simple/verification use600queries each and600/552/504actions;
none opens the door or attains any task-stage milestone. Its six actually executed
handoffs are all unverified cap exits. The book and fridge source cohorts have
1604/1656exact-replayed actions respectively.

Q1 evaluates all9candidates from source entries368/403/443 (six distinct state/action
pairs). Fresh-prefix reconstruction, all288branch actions and independent replays
pass exactly;279fresh queries are charged separately from original candidates and
prefix/replay overhead. At horizons1/12/32 no sampled opportunity contains a useful
candidate under the frozen greedy continuation. Maximum opening progress is
1.33e-15rad, far below the frozen.01rad threshold, and no opening stage is attained.

Conditional useful-selection rates have a zero denominator and stay undefined.
Zero offline regret on this degenerate set is not verifier optimality, equivalence
or successful recovery. This finding is restricted to the sampled development
states/candidate source/continuation horizons. Q2 remains gated, not launched;
cloth stays deferred. Next priority is verified shared actor/proposal capability,
including the unresolved observation/domain contract and genuinely learned skills.
Independent demonstrations and splits remain missing prerequisites for adaptation;
no new training or scripted motion substitute is introduced.

[Q1 frozen protocol and results](experiment3_fridge_q1_20261019/results.md)
· [Fridge integration results](experiment3_fridge_candidate_20261019/results.md)
· [Book mechanism preflight](experiment3_candidate_20261019/results.md).

## Q1: common-state candidate outcome diagnostic (next queued experiment)

Assigned to the existing GPU2 experiment lane, to start **after** its frozen fridge
v1 cohort/replays finish. The implementer must freeze its executable protocol and
source manifest before inspecting branch outcomes; its run manifest is the
authority on whether it actually launched. Assignment alone is not launch.

### Sampling and costs

Use the completed, exact-replayed fridge v1 verification arm. Select the first
three local-entry rows containing at least two saved candidates, with source step
at most550. If fewer exist, use all and disclose the reduced coverage; no search
for successful states. Evaluate every saved candidate, including duplicates, at
each selected row: at most9branches. This is a development-state diagnostic, not
independent episode generalization evidence or a population success-rate estimate.

Each branch executes the saved candidate once, then11new greedy model actions
from its actual evolving RGB observations and the shared phase prompt. Follow
with20new greedy model actions to examine persistence/downstream continuation.
Thus each branch uses at most32actions and31new model queries; total at most288
branch actions/279new queries. Count original candidate-generation queries,
prefix-replay physics steps and offline simulator cost separately. Batch1;
GPU2 only; proposed1200s per-branch and3600s whole-cohort hard bounds.

### Exact common state and information boundary

Create a fresh environment and replay the original action prefix for every branch.
Before branching, require exact source physics/controller/image fingerprints and
measured state/stage-tracker agreement. Merely restoring a MuJoCo state vector
does not establish controller or history equivalence. Stop and disclose mismatches.
Independently replay the resulting branch trajectories.

The first actions must equal the recorded model candidates. Subsequent actions
must be fresh model outputs; no manually repeated action sequences, waypoint
policies, gripper overrides or geometry-to-action fallbacks. No evaluation-data
training. The actor/checkpoint/normalization/camera remain frozen.

**Branch outcomes are privileged offline evaluation labels only.** Neither
selector may inspect them. Do not deploy simulator branch search under the name
of a learned verifier or hide its compute in an online control budget. The20-action
tail is a counterfactual nominal continuation, not an actual online framework
handoff or proof of full-task recovery.

### Progress and selection measurements

Before branch outcomes, compute the current simple-feedback and verification
choices on the same candidate set, current state and previous-action history.
This isolates their immediate selection decisions on this source-state sample.
The sample comes from verification-arm trajectories; report this distribution bias.

Report horizons1/12/32 separately. Opening directional progress is reduction of
`max(0, 1.8 - door_angle)`; a reduction of0.01rad is a descriptive opening-progress
signal, not achieved opening. Also record stage satisfaction, persistence, lost
achievements, contacts and observed harms. Any other reached stage needs its
directional metric/feasibility checks frozen before outcomes. Unavailable harm or
future-outcome evidence stays unknown. EEF displacement, arbitrary item motion and
action novelty alone never count as task progress.

Core outputs:

* Exact/numerical candidate diversity and duplicate fraction.
* Fraction of sampled states with at least one candidate yielding measured useful
  progress under this fixed continuation and horizon.
* Each selector's useful-choice fraction **conditional on useful availability**.
* Useful candidates rejected, harmful candidates selected, and reasons.
* Offline best measured progress minus selected progress, clearly labeled an
  evaluation-oracle gap, never an online achievable performance claim.
* Local predicate attainment versus persistence and full downstream completion.

If no candidate advances the task, useful-selection ability is **not identifiable
on those states**, rather than evidence that the selectors are equivalent.
Absence of progress at32steps is not proof that every possible longer policy fails.

## Q2: separate selection from rechecking and actual handoff

The existing three-arm pilot bundles candidate selection, checks, recheck duration
and realized exposure. It cannot isolate these contributions. After Q1 and shared
actor qualification, freeze a new version with the following factorial controls:

| Arm | Candidate selection | Exit evidence |
|---|---|---|
| F | Simple feedback selector | One measured local-stage check |
| S | Verification selector | One measured local-stage check |
| R | Simple feedback selector | Consecutive local-stage rechecks |
| V | Verification selector | Consecutive local-stage rechecks |

Retain an unmodified nominal actor reference. F/S isolates selection under the
same recheck rule; F/R and S/V isolate rechecking under the same selector. V/F
measures the combined policy, not either factor alone. Select any improved
progress-aware verifier only on declared development evidence, then freeze it for
new held-out cases. The present continuity/release/novelty rules are heuristics,
not validated progress prediction. Prediction remains optional and separately tested.

All arms share actor, observations/evidence access, phase prompts, triggers,
candidate-generation mechanism, caps, disturbances and stopping rules. F/S/R/V
share available measurements; nominal's use of fewer measurements is disclosed.
Common-state Q1 uses literally identical candidates. Online trajectories diverge,
so common generator/seeds do **not** imply identical later observations/candidates.
Use matched initializations and seed rules and report actual exposure.

Pre-register action/query/wall and local-repair budgets before this new cohort;
the existing600/600/1200s and12-action local caps are starting references, not a
claim that a new cohort is already frozen or running. Account every candidate
query. Compare costs as well as outcomes; equal caps do not equal realized actions.

For each repair transaction separately record trigger, useful availability where
measured, selected action, local exit, first actually executed nominal action,
post-handoff loss/recurrence, and remaining-task completion. Budget exit is not
verified recovery. Compare final task outcomes over all paired cases; conditional
post-handoff rates have different selected populations and are descriptive.

## Decision gates and delivery

1. **No useful candidates:** prioritize shared learned actor/candidate capability,
   authentic demonstrations or a compatible pretrained actor. Do not keep tuning
   selection thresholds to compensate for absent task skills.
2. **Useful candidates missed:** develop task-directed selection/verification on
   the declared development split, then confirm on untouched states/episodes.
3. **Local progress lost after exit:** test recheck and handoff factors explicitly.
4. **Complete fridge recovery established:** freeze mechanisms and evaluate held-out
   layouts/objects/disturbances; only then advance the cloth manipulation experiment.

The full fridge endpoint remains closed-start opening, grasp, extraction, supported
stable release, closure and withdrawal/dwell. Open-start trials are component
diagnoses. Validate endpoint feasibility and semantics before capability claims.

Retain all candidate branches, rejected proposals, failures and timeouts. Videos
must show subgoal, trigger, source, evidence/checks, actual owner, budget, local exit
and actual handoff, with baseline left/comparison right, matched playback and
explicit final-frame holds. Offline branch videos must instead be labeled
counterfactual diagnostics. Publish completed evidence and videos to GitHub/LFS
and independently verify recovery; no publication is claimed for queued artifacts.
