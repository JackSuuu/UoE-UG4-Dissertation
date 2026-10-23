# Repair-only candidate scoring interface and executed recovery data

## Main line and scope

User authorized implementation and experiments after the literature review,
then explicitly reaffirmed the main line:

**Verification → bounded model-generated repair → reverification → policy
resumption with continuation checks.**

Candidate selection is subordinate to an admitted repair. It does not replace
normal execution with global best-of-N control. OpenVLA weights stay frozen.
The new sampler changes decoding only at the declared repair probe; selected
seven-coordinate mapped actions are executed unchanged. Existing admission,
repair caps, acceptance/preservation, fusion and joint handoff vetoes remain active.

## Implementation

- `src/repair/candidate_pool.py`: `score(observation, context, candidates)`
  interface with action histories, recovery objective and current measured state;
  finite score and artifact/split provenance checks; deterministic argmax;
  model proposals cannot be numerically mutated by the scorer.
- `src/training/recovery_dataset.py`: executed outcome labels, including local
  recovery, verified exit, actual handoff, stable nominal continuation and complete
  task success. Missing evidence is unknown; invalid runs are excluded from
  training eligibility. Unexecuted candidates receive no copied outcome labels.
- Canonical `scripts/run_framework_comparison.py`: opt-in, plan-pinned first-repair
  pool, complete per-candidate query/token/image provenance and exact replay of
  selection, budgets, dynamics and continuation labels.
- `scripts/run_repair_candidate_study.py`: fixed development collector and audits.
- `scripts/analyze_repair_candidates.py`: same-state/pool contrasts and grouped
  recovery-data export. Candidate siblings belong to the same root split group.

The scoring interface is implemented, but **this first experiment collects
candidate outcomes; it does not deploy a trained ranker**. No external paper's
checkpoint is assumed to transfer to LIBERO. Observed outcome labels are not
fed back into the selector during collection.

## Frozen first experiment, declared before outcomes

Purpose: test whether the available frozen-policy H1 proposals contain an
alternative that helps the existing verifier–repair–resumption chain. This is a
minimal causal candidate-coverage test, not a complete test of multi-action
recovery or of learned ranking.

- Development roots0/4: four full episodes each, fixed selected slots0/1/2/3.
- Previously successful root1: slots0/1 as two inactive-path sentinels.
- Every episode starts from its original native initial state and ten noops.
- The first **admitted repair action only** queries a pool of four proposals
  from the same RGB and same current repair instruction before any execution.
- Slot0: original greedy seed0. Slots1–3: existing official sampling interface,
  temperature0.7, top-p0.95, top-k50, seed20261022+root×100000+step×4+slot.
- Fixed slot chosen only for collection; no hand-coded preference or simulator
  oracle ranking. Every later action uses the existing greedy framework.
- Original520action budget,32local/160total repair caps; maximum523actual model
  queries =520nominal executed-action requests+3auxiliary candidates. Slot0 pays
  the same candidate cost. Controller query counters count executed H1 requests;
  actor query counters and durable query logs include all proposals.
- Hold admission at`measured_only`, repair control at`existing`, frozen fusion/
  joint artifacts and handoff thresholds. No learned-policy update or new prompt.
- Stable continuation: eight nominal actions after an actual verified handoff,
  retaining the achieved recovery condition and preserved task predicates.
  Incomplete windows/unknown measurements do not pass; task completion remains
  a separate label even when it occurs before a full continuation window.
- All ten episodes, no efficacy stopping or automatic retry. Infrastructure
  failures retained separately. GPU2/3 canonical lane mapping, at most two actors.
  Roots0/4 run serially on GPU3; root1 on GPU2.
- Exact replay plus independent GPU joint-observation and handoff-score audit.
  Confirm same initial conditions, common pre-probe tapes, identical first pool,
  unchanged selected actions, and actual query/action budgets.

Primary: complete native task success, paired rescues/harms versus slot0.
Secondary: candidate diversity, local/verified repair, handoff/continuation,
costs. Positive coverage is an offline upper bound until a causal scorer selects
the candidate. Negative coverage concerns this one-action pool only. Reused
roots are not held-out evidence, and an inactive successful sentinel does not
establish safety when repair actually intervenes.

## Prelaunch checks and evidence

70targeted checks pass. A first unittest invocation used package-qualified names
without the legacy scripts import path and produced four import errors; rerunning
with`PYTHONPATH=scripts:.` passes. No production logic was changed for that issue.

All28previous controller traces remain exact (12,403actions; zero actor queries).
Compatibility report:
`/home/ami/scratch/framework_repair_candidates_20261022_r1/pre_candidate_compatibility28.json`.

Output root: `/home/ami/scratch/framework_repair_candidates_20261022_r1/`.
Plan SHA256: `3b763db8f11d3f1893d94183c19087530eeaeb34b5a31962188beff965d1bd5d`.
All ten declared episodes and audits completed. Each actor/replay/audit command
and return status is durably recorded under `study/`; full source copies are
retained in `plan/source/`.

## Results: one useful recovery candidate, ranking still untested

| Root | Selected slot | Complete task | Actions | Actual queries | Repair actions |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 /1 /2 /3 | All fail | 520 each | 523 each | 96 each |
| 1 | 0 /1 | Both succeed | 444 each | 444 each | 0 |
| 4 | 0 (greedy) | Fail | 520 | 523 | 80 |
| 4 | 1 | Fail | 520 | 523 | 96 |
| 4 | 2 | Fail | 520 | 523 | 85 |
| 4 | 3 | **Success** | **506** | **509** | **11** |

Root0's four candidates have identical tokens/actions; all four full physical
and action tapes are identical. Its first probe at254therefore provides only
one unique action, not four distinct recovery options.

Root4's first probe at148has four distinct actions, with the same pre-action
state, image, instruction and candidate pool across all four runs. Slot3is the
only successful branch, a paired rescue relative to slot0. No compared branch
introduces a complete-task harm, but the only successful-root sentinel never
enters repair and cannot establish intervention safety.

### Successful verification–repair–resumption chain

All indices below are zero-based executed-action indices:

-148: measured elevated-grasp-loss admission; execute sampled candidate3.
-153: local contained condition reaches its required streak.
-158: existing joint/fusion handoff checks permit verified repair exit.
-159: first actual nominal action after repair.
-159–505:347nominal actions; measured containment remains true throughout, no
  further repair, no recorded recurrence or post-handoff containment loss.
-505: native In AND Close completion,506total actions /509actual actor queries.

This exceeds the preregistered eight-action stable-continuation check. In
contrast, slot0verified at163but lost containment at its first handoff action164
and ultimately failed. Slots1/2left their first repair unverified at the32-action
cap. Slot1attained a local condition without verified recovery, illustrating why
local, verified, handoff and complete-task labels must stay distinct.

At148, candidate3and greedy have identical translations and closed-gripper
commands; only mapped rotation coordinates3and5differ. These differences come
from frozen-policy samples, not numerical edits or a release/regrasp script.
Later greedy requests/actions follow their diverged physical states; this is
not a claim that only one action differs across the complete trajectories.

### Validation and recovery-data artifacts

-10/10valid episodes, no infrastructure failures.
-5,034unchanged executed VLA proposals and5,058actual model queries.
-5,154exact replay frames,5,044independently recomputed joint observations and
 656handoff scores; all pass.
-Matched scene/initial state and identical first pool verified. Slot0full action
 and physical tapes also exactly match prior admission controls on roots0/1/4.
-`analysis/recovery_dataset.jsonl`:8executed candidate outcome records, grouped
 by the two development roots; no labels for unexecuted alternatives. Seven
 complete-task-negative branches and one positive branch, not eight independent
 failure contexts or a train/test split.
-`analysis/verified_recovery_root4_slot3.json`:11-action executed recovery segment
 with action/token/image provenance and verified nominal continuation evidence.
-`analysis/candidate_validation.json`: independent budget, action-provenance,
 historical-control and complete-continuation checks.
-[Compact evidence](framework_repair_candidate_evidence_20261022/manifest.json):
 46files /789,492bytes plus a SHA256manifest; full tapes/images stay in scratch.

### Interpretation and next decision

This establishes that **one useful model-generated alternative exists at one
matched development failure state, and the existing verification–repair chain
can resume nominal execution and finish after that alternative**. It is positive
candidate-coverage evidence, not learned-ranking efficacy, a generalized success
rate, or an ablation proving that the verifier caused the rescue.

Selecting slot3because this experiment found it successful would leak outcomes
into deployment. The next scoring experiment needs additional independent
failure contexts, split at the task/scene/root level, and continuation-aware
positive/negative labels. The current two contexts cannot support a credible
generalizing ranker evaluation. Root0also requires better candidate coverage;
more scoring cannot distinguish identical actions. No further batch was launched
on completion of this study.
