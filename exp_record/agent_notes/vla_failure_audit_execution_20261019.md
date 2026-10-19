# Why the actor stalls, and what actually caused R2 progress

## User requirement

The user explicitly rejects a human who observes a failed episode and writes its
motion solution as evidence of autonomous intelligent repair. Priority is now
failure diagnosis and attributable improvements. A hand-written route expressed in
relative coordinates is still a hand-written route; transfer of such a route alone
would not answer this autonomy concern. R2 and the later transport probes must be
described accordingly. No new repair-enabled actor cohort is launched here.

## Established R2 provenance

`audit_fridge_action_ownership.py` checked all four tapes. Both arms have exactly
matching initial physics/controller/images and pre-intervention executed/proposed
actions and fingerprints (29steps closed;51steps open). Baseline executes every
logged VLA proposal unchanged. This supports honest paired execution, not autonomy
of the additional controller or a causal benefit from verification itself.

Open baseline emits near-zero normalized XYZ commands (L2<0.01) in504/600steps;
it has44distinct seven-token sequences. It is repeatedly predicting small commands,
not merely showing a frozen video or a stopped process. This threshold describes
commands, not physical precision or a formal failure predicate. Closed baseline
has73/600near-zero translations and422distinct token sequences: its failure is not
the same stationary pattern.

Open wrapper executes382scene-specific geometric-primitive actions,38gripper-only
retention actions and180unmodified VLA actions. Primitive allocation is194grasp,
27extraction and161placement steps. First grasp at242and first held extraction at299
both occur under the geometric primitive. No complete task is achieved. The local
progress cannot be credited to an autonomously learned repair policy or to verifier
novelty. The38retention actions plausibly prevent release, but their causal benefit
would require matched branches rather than inference from the action log alone.

## Pinned inference audit: correction and remaining discrepancy

An isolated Python3.11venv uses transformers4.40.1, tokenizers0.19.1, timm0.9.10and
torch2.6.0. It loads the same task weights and local upstream model code. Six fixed
R2observations were exported before evaluation. All six processed-image tensors and
prompt IDs match the current adapter exactly. Five action/token sequences match;
the open-wrapper step400sequence differs, including the gripper sign (max action
difference2). Official cached and uncached predictions match each other for all six,
so merely disabling the cache does not resolve the cross-stack discrepancy.

**Correction:** an intermediate assistant update incorrectly said6/6match after
reading only part of the report. Correct result is5/6; the discrepancy is retained.
The four selected baseline samples agree, including the initial near-zero command.
This neither proves global parity nor explains the entire refrigerator failure.
The numerical/model-library root cause remains open, and multiple library versions
differ. Do not call this a proven transformers-only bug or dismiss the differing
action as harmless. Fresh pinned-stack rollouts are included in the next audit.

Data: `/home/ami/scratch/openvla_tf440_audit_20261019/cases_b/` contains fixed images,
processor tensors, original tokens/actions, cached and uncached reports. The first
export import-order failure and first reference processor-loading failure were
corrected before successful comparison; reference error logs are retained.
Pinned worker IPC was checked against the exact initial refrigerator action.

## Task and camera hypotheses

The checkpoint is fine-tuned for LIBERO10. Local suite definitions contain book/caddy
and cream-cheese/basket tasks, but no refrigerator-opening/retrieval task. This does
not establish absence from base pretraining. A custom fridge layout, occlusion and
robot-side camera add distribution changes; training-object familiarity alone does
not imply competence at opening doors or extracting through shelves.

Camera smoke tests show that applying the exact R1camera pose to the standard book
task removes most of the gripper/caddy from the actor view, while the native view
shows them. Thus that intervention includes loss of task visibility, not just a
different viewing angle. Report this explicitly; no controlled conclusion about
camera sensitivity or the refrigerator's root cause exists before rollouts finish.
Book physical/controller initial states match across cameras; the pinned fridge
initial physics/controller/both-camera fingerprint exactly matches R2baseline.

## New actor-only cohort

[Frozen protocol](vla_failure_audit_protocol_20261019.md): five600-action/query trials,
book native/current, book native/pinned, book robot-side/pinned, cheese native/pinned,
fridge robot-side/pinned. Existing R2open baseline is the fridge current-stack control.
One development initial state per condition; this is diagnosis, not a success-rate
or generalization claim. No scripted action route or intervention. All control-interval
actions are model proposals; state/contact is used only for logging/evaluation.
Each run receives independent frozen-source replay. A one-action pinned book preflight
passes exact independent physics/controller/images/diagnostic replay.

## Stopped transport attempt

Before the user's sharper clarification, an object-relative outside-corridor route
was implemented as a same-state development probe from R2step332. The first attempt
failed on an unsupported mesh-envelope assertion; the second exhausted240actions
at corridor stage1without placement. These are preserved negative engineering
probes, not framework results. Neither is used for a new claim or an actor rollout.
Further work on hand-written motion routes is paused.
