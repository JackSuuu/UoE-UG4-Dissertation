# Experiment 2 extension: LIBERO90 learned actor transfer to the fridge

Frozen before rollout; follows the user's fridge-priority update. This is an
actor-only capability test, not a framework comparison or local fine-tuning.

## Candidate and relevance

Public third-party `VQ-VLA/openvla-7b-finetuned-libero-90`, revision
`794ef81b7be928ea9270e81ca1ef5b60ffa9420f`, approximately 15.085 GB repository,
four safetensor shards. Metadata and file hashes are in
`/home/ami/scratch/shared_actor_fridge_pretrained_audit_20261019`.
Its configuration matches the current LIBERO10 actor's architecture, including
text/vision backbones. The card contains only a license; training recipe/evaluation
split is not established by the card. Statistics identify 3,919 trajectories and
569,315 transitions under `libero_90_no_noops`.

LIBERO90's 90 named tasks and BDDL files contain **no fridge task**. Relevant
learned-task overlap is opening/closing microwave doors and grasping/transporting
cream cheese, on the same fixed Panda / seven-dimensional OSC_POSE interface.
This is a wider-skill **out-of-distribution transfer candidate**, not a checkpoint
known to solve the custom fridge. Select this one checkpoint by architecture and
primitive overlap before observing results, with no checkpoint/seed search.

Actual remotely available fridge data/checkpoints (RoboCasa / Doosan / ACT) are
documented separately; their different embodiment and action interfaces prevent
silently substituting them in the current fixed-base Panda scene.

## Resource and inference contract

Download at most this revision's required config/tokenizer/preprocessor/statistics
and weight files, bounded to 900 seconds. Verify each weight's expected Hub LFS
SHA256 before any actor rollout. If unavailable, record blocked; no substitute.
Use GPU3 only (both CUDA and EGL IDs 3; UUID checked), batch 1, existing official
TF4.40.1 pinned Python, cached greedy H=1, frozen local base model source. A new
worker accepts an **explicit** normalization key; never rename LIBERO90 statistics
to LIBERO10. Use its q01/q99 action dequantization and mask, then the same
`-sign(2*g-1)` gripper conversion. No hand-coded motion, geometric action labels,
state-conditioned prompt changes, repairs or model training.

## Conditions and stopping

Exactly two fresh seed-0 environments, using the existing R1 corrected fridge
geometry: yaw -pi/2, fridge y=.35, initial open angle 130 degrees, item-forward .04,
item-lateral .035, fixed robot-side camera. Existing 20 zero-motion open-gripper
setup settling calls are retained and declared. This is a reused development
seed, not held-out evaluation. The same candidate would have to be shared by all
comparison arms if subsequently adopted; no framework-only actor upgrade.

1. **Closed-door full-task attempt:** constant native canonical full instruction
   (open, remove cream cheese, place on table, close). Report full TaskMonitor
   completion, native predicate, opening/grasp/extraction/released placement/close
   milestones. No stage prompt changes.
2. **Open-door grasp diagnostic:** constant instruction
   `pick up the cream cheese box inside the fridge`, as in the prior official
   LIBERO10 failure audit. Report grasp/extraction and remaining-sequence completion
   separately; do not count initially open door as successful opening or full task.

At most 600 model queries/actions and 1200 rollout wall seconds per condition,
1800 hard process-group seconds including loading, 360 seconds per replay.
Stop on the existing fridge TaskMonitor's 20-interval remaining completion, budget,
or explicit error; native `done` alone does not stop. Record internal timestep/time
and assert continuous physics with no reset after native done. Retain every failure.
The fridge endpoint is the existing TaskMonitor/sense endpoint (full footprint,
table support, release throughout interval, slow object, opened/grasp/extracted
history, stable closed door, clear gripper, 20 intervals); it is distinct from the
book CompletionMonitor and has no extra penetration-threshold requirement.

## Evidence and interpretation

Freeze all local imports, protocol, named-stats worker and relevant external scene
sources before launching either condition. Preserve complete action/token/state/RGB,
contact and clock tapes, preprocessing hashes, latencies, constant language and
identity-of-proposal decision trace, owner=VLA throughout. TaskMonitor is logging/
termination only and never a policy input. Independent fresh-process replay must
match states, images, fingerprints, contacts and all task observations exactly.
Annotated video labels candidate, initial door, native vs strict endpoint, control
owner and stop reasons, retaining every frame and marking end holds.

No local demonstrations or evaluation traces enter training. These two runs only
test transfer of an already learned public checkpoint; neither success nor failure
resolves its undisclosed training provenance or establishes framework efficacy.
