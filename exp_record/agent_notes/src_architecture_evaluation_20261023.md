# Source architecture evaluation and next repair experiment

23 October 2026. Review of the current working tree, existing literature notes,
saved candidate episodes and training artifacts. No policy rollout, model fitting,
controller modification or GPU audit was performed in this evaluation.

## Executive assessment

The project has an implemented, auditable bounded verification/repair state
machine. It does not yet have a demonstrated reliable repair selector or reliable
resumption verifier. Current-state detection, intervention benefit, candidate
utility and nominal resumability are different targets.

The candidate study evaluates four **single-action proposals from the same
frozen OpenVLA at the first admitted repair**, followed by the existing greedy
closed-loop controller. It has not evaluated four complete recovery policies or
reproduced four external papers. The architecture is worth retaining; the next
work should isolate generation, selection and handoff rather than add modules
without a matched benefit test.

## 1. Actual current execution path

Canonical runner: `scripts/run_framework_comparison.py:549–759`.

1. Policy RGB and privileged measured predicates/contacts/positions are collected.
2. `verification/joint_runtime.py:53–130` shares a strict pretrained DINOv2
   representation across visual stage inference, successful-keyframe retrieval,
   outcome risk and handoff scoring. Constant-velocity diagnostics use observed
   position history.
3. `controller.py:306–414` decides admission, freezes the recovery contract and
   preservation predicates, and selects a stage-specific language request.
4. `repair/receding_horizon_repair.py:13–26` packages that request and records
   stage revisions; the external frozen policy generates the numerical action.
5. At the first repair only, `repair/candidate_pool.py:11–24` requests one greedy
   and three sampled H1 proposals. The runner selects the declared collection
   slot (`run_framework_comparison.py:582–617`). All later actions are greedy.
6. The chosen proposal is executed unchanged. `controller.py:475–624` measures
   local recovery and preservation, applies handoff checks, and records actual
   nominal resumption separately from permission to resume.

The latest collector uses measured-only **admission**, but retains joint visual
stage routing and fusion/risk/prediction handoff vetoes. It is not a fully
measured-only controller. `variant=v2` leaves the separate three-endpoint measured
resumption gate off; the MLP and joint vetoes still operate.

## 2. What works structurally

- Sequential before/after contracts and source provenance distinguish requested
  actions, executed actions, local acceptance and actual handoff.
- The entry recovery objective stays fixed when the requested stage changes.
- Preservation, missing measurements, local/total budgets and failed-context
  retry gating are explicit.
- Candidate scoring receives pre-action inputs and copies proposals; it cannot
  silently mutate the executed action. Sampling costs are counted separately.
- Recovery records assign outcomes only to executed branches, preserve root
  grouping and censor missing/invalid evidence.
- Tape replay and independent neural recomputation have separate meanings.
- The active joint path has real frozen visual features. Random visual fallback
  and hand-coded translation corrections belong to the opt-in legacy v5 path,
  not the current candidate collector.

These are useful engineering foundations; they do not establish task-level gains.

## 3. Principal limitations, ordered by research importance

### A. Useful generation and learned selection remain incomplete

`candidate_pool.select` supports a scorer object, but no concrete trained
action-conditioned scorer is implemented in `src/`, and the canonical collector
only calls it with `collection_slot`. There is no deployed ranking experiment.

The four proposals share the same current repair instruction and image. The
sampler changes just one action in the first repair; subsequent closed-loop
actions diverge through the resulting physical states. This is a minimal coverage
probe, not a multistep recovery strategy search. `RecedingHorizonRepair` itself
does not learn, search, predict or synthesize recovery actions.

Candidate diversity is often low. The saved first pools contain:

|Context|Distinct actions among four slots|Outcome|
|---|---:|---|
|Development root0|1|All fail|
|Development root4|4|Only slot3 completes, with verified resumption|
|Training root11|2|All four fail|
|Training root12|3|All four fail|
|Training root13|2|Slots0/1/2 fail; slot3 completes, joint audit pending|
|Training root16|2|Slots0/1 actor outcomes fail; slot1 checks pending, slots2/3 absent|

Root13 slots0/1/2 have identical first actions. Slot3 changes only the mapped Z
coordinate (approximately0.0803834 to0.0870536), not an entire recovery strategy.
No recommendation to hard-code that delta or select slot3 follows from this.

### B. Verification has three different maturity levels

**Measured detection and local contract checks:** implemented and replayable,
but task-specific. `controller.py:57–76,130–170` hard-codes microwave object,
region, predicates and instructions. In is a benchmark predicate, not proof of
full-body support, stable detachment or useful future continuation.

**Intervention admission:** detection of trouble does not estimate whether a
repair improves on nominal continuation. On root13, the baseline loses grasp
at293 but eventually completes in456 actions. The framework intervenes at294;
greedy repair eventually fails. The anomaly is real, but intervention benefit
is not established. Measured-only admission removed earlier joint harms on
roots0–5 but did not prevent this new harm.

**Resumption verification:** not solved. Fusion is an instantaneous
RGB/state/contract classifier (`fusion_mlp.py:9–12,36–53`), with no candidate
action, next nominal action or explicit motion/contact history as inputs.
Its published validation rejects193/283 safe current-contract-true samples;
on actual handoffs it rejected both safe examples and accepted an unsafe one.
Its training roots contained no actual repair handoffs. This is a substantial
distribution/target mismatch, not just a tuning issue.

The current risk artifact also has TP=0,FN=146 on its validation violation
samples. The full-joint development comparison was2/6 versus baseline4/6;
measured admission restored4/6 but did not establish rescue. Risk computation
and causal wiring should not be described as validated useful detection.

### C. The predictor cannot compare proposed repair actions

`joint_runtime.py:38,80–96` explicitly uses `action_blend=0` and
`proposed_action=None`. It extrapolates observed constant velocity and checks
observed no-lift. It is not an action-conditioned neural dynamics model.
The same current state therefore supplies no candidate-specific predicted
consequence through this path. A state-risk score cannot rank identical-state
alternative actions without action conditioning.

### D. Recovery memory and training pipeline need precise scope

`memory/keyframe_bank.py` stores visual stage exemplars, not successful recovery
action sequences or failure-to-recovery mappings. `joint_runtime.py:99–113`
delays online writes for12 observed preservation endpoints, but receives no
nominal/repair ownership. These writes are stage/preservation memory, not proof
of successful nominal resumption.

`training/recovery_dataset.py` exports candidate outcomes; it does not train a
ranker or recovery generator. Existing `handoff_dataset.collect_samples` rejects
roots outside0–5 (`handoff_dataset.py:77–79`). Thus the new roots10–19 collection
does not automatically feed the existing handoff training pipeline.

The new audited training data currently contain12 executed first-probe records:
8 with local recovery,0 verified exits,0 task completions, and12 unknown
`stable_nominal_continuation` labels. There are only four distinct root contexts,
not12 independent failures. Training a useful generalizing ranker from these
records alone is not justified. Root13 slot3 is potentially a positive task
outcome after its pending audit, but remains an unverified budget-exit trajectory.

## 4. Concrete semantic findings

### Confirmed label defect: detached continuation accepts regrasp

`training/recovery_dataset.py:37–54` checks held-clear specially, native completion
specially, and otherwise only In. Consequently `verified_by=contained_detached`
is treated as plain containment. An eight-step continuation with
`in_goal=True, grasp=True, mug_gripper=True` is exported as stable, even though
`handoff_dataset.contract(state, 'contain_release')` is false at all eight steps.
A minimal in-memory reproduction confirmed this. The analogous post-handoff
loss check in `controller.py:604–607` also tracks containment rather than
detachment. No effect on the known root4 success is asserted: its accepted
criterion was containment. Fix the shared contract semantics before fitting new
models; retain/version historical exported evidence.

### Gate-conditioned labels omit observed budget-exit continuation

`recovery_dataset.py:28` computes continuation only after a verified exit.
Keeping the verified-chain label separate is correct. However, a future handoff
model also needs independently measured continuation after **unverified** exits.
Otherwise the current gate determines which observed continuations can receive
that label. Add a separate observed continuation/contract-retention target with
explicit exit contract and censoring, rather than relabel budget exits as verified.

### Fridge/microwave boundary semantics differ

`repair/fridge_probe.py:79–84` accepts a passing check on action32 before checking
the cap. `controller.py:558–575` gives the local cap priority and marks the same
tie unverified. A synthetic tie reproduced this difference. It did not change
the recorded fridge negatives (none passed recovery), but should be documented
or harmonized before claiming shared cross-task verification semantics.

## 5. New saved-trace observations about the handoff gate

Computed from saved controls, without rerunning the actor or neural models:

- Root13 greedy reaches local recovery at300. Across its complete episode,
 38 locally eligible endpoints are blocked; all38 have MLP probability below
  the frozen threshold. Prediction and risk veto counts are zero at those checks.
- Root13 slot3 reaches local recovery at302. Its10 eligible checks are likewise
  MLP-rejected, then it exits unverified at325, hands off326 and completes in447
  actions. Replay passed; independent joint audit remains pending.
- Root11 and root16 also have local acceptance followed by MLP rejection.
- These are observed control effects, **not evidence that earlier handoff would
  have completed**. Changing handoff changes future states and policy actions;
  that counterfactual requires a matched rollout.

## 6. Relationship to the reviewed literature

- CoVer-VLA / RoboMonkey: sampled proposal and scorer interface ideas adopted;
  learned scorer fitting and online selection not yet implemented/evaluated.
- CARE / FailSafe: executed recovery labels, failure cases and grouped data
  collection adopted; no corrective executor or recovery head has been trained.
- VLA-Corrector: action-conditioned residual dynamics remains a proposal; the
  current constant-velocity predictor is not its implementation. H1 has no stale
  multi-action chunk to truncate.
- Mostly Harmless: relative benefit over nominal continuation is relevant to
  root13 harm; no comparative benefit head or guarantee is implemented.

See `framework_literature_review_20261022.md` for the original inspected sources
and compatibility limits. This evaluation did not re-check external releases.

## 7. Recommended next sequence

1. **Recover evidence and correct label semantics.** Finish the missing audit
   for root13 slot3 and replay/audit for root16 slot1. Validate relocated runtime
   paths first. Preserve the frozen incomplete batch; any completion of its
   remaining jobs uses its original protocol and a real resume procedure.
   Add shared/versioned continuation contracts and separate observed budget-exit
   continuation labels before new training. These are proposals, not edits made
   by this review.
2. **Run one small handoff-isolation experiment.** On reused development cases
   root4 and root13, fix measured admission, stage routing, greedy versus sampled
   first-probe actions, budgets and all other checks. Compare current fusion
   veto versus an explicitly declared MLP-advisory mode that retains measured
   checks and other joint vetoes. Pair both with fresh bare nominal controls.
   The current joint constructor forces `fusion_mode=gate`; implement a scoped
   ablation rather than assuming `log` already works for joint mode. Because
   local greedy/sampled candidate differences also matter, use a preregistered
   2(candidate choice)×2(MLP veto) contrast per root. Fixed sampled choices are
   development diagnostics, never a deployed learned selector. Check root4's
   existing rescue, root13's harm, actual handoffs and complete task outcomes.
3. **Choose the generator/selector work from coverage.** After freezing a
   defensible handoff setting, collect independent training failure contexts and
   success controls. Evaluate unique proposals and matched executed continuations.
   If useful alternatives recur, fit a small action-conditioned scorer on the
   same pools and compare greedy, seeded random and learned selection with equal
   query budgets. Include nominal continuation/no intervention as a comparator.
   If pools remain duplicate or all fail, first test a declared short feedback-
   conditioned recovery option or collect corrective demonstrations for a separate
   learned recovery generator. H1 needs a closed-loop option, not an invented
   open-loop chunk. Keep OpenVLA frozen and label any auxiliary learned generator
   explicitly. Do not spend a larger ranking budget on identical actions.
4. **Then test the prediction contribution.** With generator and candidate pool
   fixed, compare measured/visual scoring versus scoring augmented by a trained
   action-conditioned transition model. Score action effects, task progress and
   preservation; latent prediction accuracy alone is not repair benefit.

Fridge should remain a capability/data diagnostic until bounded recovery behavior
can actually open or grasp. Its10 negative episodes do not support immediate
cross-task deployment of a microwave ranker. Calibration roots6–9 and confirmation
roots20–49 should remain outside development fitting/selection.

## 8. Checks executed during this evaluation

From the repository root:

```bash
PYTHONPATH=scripts:. /home/jack/miniconda3/bin/python -B -m unittest \
  test_native_framework test_joint_framework test_fusion_handoff \
  test_repair_candidates test_repair_contract_progress \
  test_repair_data_expansion test_fridge_repair_probe
```

**78 tests passed.** Additional in-memory probes reproduced the detached-label
defect and fridge cap tie behavior. Tests use the current local Python environment;
they are model-free checks, not revalidation of frozen GPU rollout dependencies.
The inspected outputs remain under `/home/jack/scratch/`; historical manifests
retain `/home/ami/` paths. Source/runtime behavior was not changed by this review.
