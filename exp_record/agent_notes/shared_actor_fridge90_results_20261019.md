# Experiment 2 fridge priority: LIBERO90 transfer did not establish capability

**Neither bounded actor-only run opened/grasped/completed the fridge task.** The
closed-door full-task attempt and open-door grasp diagnostic both exhausted 600
model actions. All 1,200 recorded actions passed independent exact replay and
artifact checks. This candidate is not qualified as the shared fridge baseline.

## Frozen candidate and protocol

Protocol: `shared_actor_fridge90_protocol_20261019.md`, frozen before either rollout.
Actor: `VQ-VLA/openvla-7b-finetuned-libero-90`, revision
`794ef81b7be928ea9270e81ca1ef5b60ffa9420f`. The four downloaded safetensor shards
passed expected Hub LFS SHA256s. Required files total **15,085,104,774 bytes**;
download and hashing took **331.43 s**, below the 900-s cap.

Physical GPU3 only, batch 1, pinned Transformers 4.40.1 / Torch 2.6.0+cu124,
official greedy H=1 prediction using cached base-code revision
`47a0ec7fc4ec123775a391911046cf33cf9ed83f`. Dataset normalization was explicitly
`libero_90_no_noops`, with its own q01/q99/mask; seven-axis gripper conversion
was preserved. No statistics were relabelled as LIBERO10. There was no local
training or framework comparison. Twenty declared setup settling calls precede
the evaluated actions in each fresh seed-0 development environment.

| Condition | Queries/actions | Door opened by actor | Grasp | Native | Strict full / remaining | Max dwell | Stop |
|---|---:|---|---|---|---|---:|---|
| Initially closed, constant full-task instruction | 600/600 | No | No | No | No / No | 0/20 | action budget |
| Initially open, constant grasp instruction | 600/600 | Not tested | No | No | No / No | 0/20 | action budget |

The initially open door was set at 130° during setup; `opened=true` in that
diagnostic is not a learned opening success. Its single grasp prompt does not
request the remaining full sequence, so the relevant capability result is **no
grasp**, not a language-conditioned failure to close the door.

No sampled task observation reported door contact or item–hand contact. The item
position changed by only ~8.1e-13 m relative to the first post-action state, and
closed-door angle stayed within ~2.2e-14 rad of zero. Closed/open runs produced
13/2 distinct seven-token sequences. In the open-door run the six arm action
components were constant near zero, with only gripper token changes. This is
recorded near-stationary behavior, not a frozen simulator: all 1,200 physics
transitions advanced 0.05 s and one timestep, with exact replay and changing
robot state. Closed-door end-effector axis ranges were .00368/.00213/.08350 m;
open-door ranges were .000088/.000746/.001477 m (post-action states).

Median end-to-end query latency was .28593/.28643 s. Rollout wall times were
183.90/185.20 s. Complete supervisor time, including replay/video, was 437.72 s.

## Independent checks and evidence

Both replays: `exact_replay_passed`, **max state gap 0, no mismatches**. They
re-execute stored actions in fresh environments and compare RGB, physics/controller
fingerprints, contacts, native predicate, sensed task conditions and all monitor
history/streak fields. They do not re-query the model.

`scripts/check_shared_actor_fridge90_artifacts.py` additionally verifies all
stored solver-contact tapes, raw/preprocessed actor RGB hashes, constant language,
VLA ownership and proposal identity, contiguous clocks, frozen local source hashes,
and **all 1,200 token sequences → named-statistics native actions → executed gripper
conversions**. Thus the failure is not an observed use of LIBERO10 normalization.
No native success occurred here, so continuation-after-native-done was exercised
by the separate book control rather than these fridge runs.

The checker decoded the video and confirmed 960×800, 20 fps, 641 frames, 32.05 s:
601 simulator-state frames followed by 40 additional held endpoint frames. The
final-state frame is also labelled as stopped. Initial/mid/final contact-sheet
images and a decoded video midpoint were visually inspected. Owner, native versus
strict endpoint, initial door condition and stop reason are visible. Displayed
time is simulation time; inference pauses physics.

## Exact artifact paths

Root: **`/home/ami/scratch/shared_actor_fridge90_20261019_a/`**

* `launch.json`, `source/`, `external/`: commands, bounds, timing, frozen protocol,
  local code and relevant external scene/step semantics.
* `closed/` and `open/`: `status.json`, `steps.jsonl`, `contacts.jsonl`, `states.npz`,
  `images.npz`, `model.xml`, `scene.bddl`, `policy_input.png`, `pinned_worker.log`.
  Root `closed.log` / `open.log` preserve the rollout logs.
* `replay_closed/verification.json`, `replay_open/verification.json`: exact replay.
* `summary.json`, **`artifact_checks.json`**, `artifact_checks.log`: outcomes/checks.
* **`fridge90_actor_only_annotated.mp4`**: paired actor-only review video.
* `fridge90_endpoint_contact_sheet.png`, `fridge90_decoded_video_midpoint.png`.

Checkpoint and verified file hashes:
`/home/ami/scratch/shared_actor_fridge90_checkpoint_20261019/download_manifest.json`.
Download log: `/home/ami/scratch/shared_actor_fridge90_checkpoint_20261019.download.log`.
Supervisor log: `/home/ami/scratch/shared_actor_fridge90_20261019_a.supervisor.log`.
Base-code hashes/runtime provenance are also in
`/home/ami/scratch/shared_actor_qualification_data_audit_20261019/checkpoint_provenance.json`.

## Implications for the parent experiment

1. Keep fridge as the primary target, but the evaluated LIBERO10 baseline and this
   LIBERO90 transfer candidate do not yet supply demonstrated learned fridge
   competence. This test is two development conditions, not a success-rate estimate
   or proof of impossibility. Camera/domain shift and undocumented third-party
   training quality remain unresolved; the new checkpoint has no in-domain
   positive-control rollout in this bounded protocol.
2. The book control reached native success in 3/3 but strict completion in 0/3,
   despite 1,087 total post-first-native actions. See
   `shared_actor_qualification_results_20261019.md`. This qualifies native book
   manipulation only, not the strict endpoint or fridge capability.
3. **Real fridge learning resources exist:** RoboCasa CloseFridge demonstrations
   and pretrained SmolVLA; an RLinf/OpenPI OpenFridge checkpoint; Doosan fridge
   demonstrations/policies. Their provenance, pinned revisions, sizes, splits,
   normalization and embodiment mismatch are in
   `shared_actor_fridge_pretrained_audit_20261019.md`. LIBERO90 itself contains no
   fridge task, confirmed against both task inventory and official 90-demo listing.
4. For the current scene, the next capability-building step is independent human
   demonstration collection and a verified seven-axis learned adaptation route
   with a declared train/development/evaluation split. A native RoboCasa test is a
   separately specified environment/skill and cannot establish current-scene full
   retrieval competence. No framework-only actor swap should be made; any adopted
   learned actor must be shared and frozen across comparison arms.

No evaluation trace was used for training; no adapted checkpoint was produced.
All failures are retained. No commit or push was made. Only experiment-2 new
scripts/notes were authored; parent integration/archive remains pending.
