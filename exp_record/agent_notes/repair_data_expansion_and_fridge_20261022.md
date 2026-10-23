# Training-data expansion and frozen-policy fridge transfer pilot

## Motivation and registered scope

Following the single root4 candidate rescue, the user requested additional data,
more conditions/tasks, and a fresh experiment on the persistently failed fridge.
The objective remains verification → bounded model-generated repair →
reverification → actual nominal resumption → complete-task success.

Current evidence shows useful detection in some development traces and one
successful recovery alternative. It does not establish general detection recall,
automatic selection of useful alternatives, or dependable handoffs. The prior
greedy root4 repair passed local verification but lost containment at handoff.

## A. Microwave training expansion

- Official roots10–19 only, as designated by `core_research_plan.md`; roots6–9
  remain calibration and20–49 confirmation. No evaluation rollouts are inspected.
- Every root receives bare frozen-policy and greedy framework controls (20runs).
- If the valid, replay/audit-passed framework control admits its first repair,
  collect all three sampled-slot siblings, irrespective of success/failure.
  No-admission siblings are skipped, invalid controls separately block siblings.
- Maximum50episodes,26,150actual model queries;520actions and523query ceiling
  per episode. Bare nominal uses no auxiliary queries. Candidate arms all query
  one greedy plus three samples at the same first admitted repair state.
- Measured-only admission, original repair mode, real joint/fusion gate; frozen
  OpenVLA, unchanged selected proposals, no numeric correction or recovery scripts.
- All candidate siblings retain root-group identity and `split=training`.
  The outcome dataset labels only executed branches; unknowns remain unknown.
- Every valid episode receives exact replay; framework episodes additionally
  receive independent neural recomputation. Primary endpoint remains native
  complete-task success; compare rescues/harms against bare nominal and slot0.

Plan: `/home/ami/scratch/framework_repair_training_20261022_r2/plan/plan.json`  
SHA256: `52eb8c760c861077cdd997432c74d4d71374289ec1409b8c4049700787e57495`

This adds initial-state coverage in one task, not cross-task generalization.
Success cases with actual admission, if present, test intervention harm; inactive
success cases alone cannot establish intervention safety.

## B. Fridge transfer pilot

Two reused seed0 conditions, each with bare nominal plus four fixed candidate
slots (10episodes). The closed condition uses the original constant full task
instruction; the initially open condition uses the original constant grasp
instruction and is explicitly a diagnostic, not a full-task rescue test.

- Same frozen LIBERO10 checkpoint/normalization as the microwave, official H1
  greedy decoding except the selected first repair sample. The earlier LIBERO90
  failure remains a separate model-transfer result.
- Reuse R1 scene geometry/camera and20setup noops,600task actions/603queries max.
  GPU2runs closed; GPU3runs open. Maximum6,030queries for the study.
- New task adapter admits one repair on measured30-observation stagnation or
  grasp loss. A single32-action transaction freezes its phase/acceptance and
  applicable preservation predicates; three consecutive valid observations permit
  verified exit. Budget exits remain unverified. Nominal control resumes with the
  original instruction; eight executed nominal actions assess retained recovery.
- Only the existing archived R2 **measurements, endpoint tracker, and phase
  instructions** are reused. Its geometric action primitives are never called.
  Every executed action is an unchanged frozen-policy output.
- Microwave-trained neural risk/fusion thresholds are not assumed calibrated to
  this task. This is a measured-verification transfer pilot with a new adapter,
  not a claim to have transferred the complete learned verifier unchanged.
- Primary closed endpoint: native task success; historical strict full-sequence
  status is also logged. Native termination is respected: no extra actions after
  done to manufacture a dwell result. Open grasp is reported separately.
- Every valid episode gets fresh-scene replay of states, integration, both RGB
  cameras, controller fields, action mapping, decisions, contacts and endpoints.
  Same-state first pools/prefixes and paired outcomes are independently checked.
- All fridge data remain development data grouped by `fridge_seed0`, including
  both door conditions; the native numbered split does not apply to this scene.

Plan: `/home/ami/scratch/fridge_repair_candidates_20261022_r2/plan/plan.json`  
SHA256: `08e023c3a34c29b950ddba3d3b5a86bcdb6b138927777d9e5ade3f290af6aebf`

## Checks and execution status

-71unit-check executions passed, covering candidate provenance, bounded repair,
  handoff/censoring, success-independent collection, split labels and prior logic.
- Both fridge scenes/endpoints passed zero-actor-query preflight; training
  roots10/11passed exact registered state and compiled-scene preflight.
- First fridge preparation encountered missing legacy simulator helpers in the
  working tree. It failed before writing a plan or querying an actor. The runner
  now explicitly reuses hash-pinned archived helpers, preserving the current tree.
- The `_r1` plans and zero-query preflights are retained. `_r2` plans reconcile
  the fridge600/603budget metadata and freeze final sources before actor use.
- Available scratch space was37GB; prior520-step episodes used about364MB each,
  so the declared maximum expansion plus fridge tapes fits without deleting data.
- Queue script: `/home/ami/scratch/repair_expansion_queue_20261022.py`. Studies run
  sequentially (fridge first, then training), each with at most two actors on
  GPU2/3. This prevents overlapping supervisors from exceeding the resource cap.
- No automatic episode retries or efficacy-based stopping. Per-episode command,
  return status, raw failures and source snapshots are retained. Analysis/export
  follows each completed study. No commit or push.

The serialized supervisor is launched (shell `sh_165b54d3a0011sFPQTXamvTpOV`);
fridge runs first and training expansion is queued behind it. Results pending.
Durable launch status is recorded separately in
`/home/ami/scratch/repair_expansion_queue_20261022/status.json`.
