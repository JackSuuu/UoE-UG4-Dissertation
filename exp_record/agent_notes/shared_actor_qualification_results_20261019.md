# Experiment 2 native-book control: 3 native successes, 0 strict completions

**The frozen official-TF4 LIBERO10 actor grasps/transports the book, but does not
qualify for released, supported, stable full-footprint completion on any of these
three initial states within 600 model actions.** Continuing beyond native `done`
does not resolve the strict endpoint in this bounded cohort. This is a control;
the fridge remains the primary rigid-body scene.

Protocol: `shared_actor_qualification_protocol_20261019.md`, frozen from main
`89002dedbf74ac49d3383cd966d014af5b024d9b` before rollout. GPU3 only, batch 1,
same checkpoint and pinned `/home/ami/scratch/openvla_tf440_audit_20261019/venv/bin/python`.
There was no adaptation, repair, motion script, or human action in the evaluated
sequence. Twenty declared zero-motion open-gripper setup calls precede each tape.

| Initial index | Queries/actions | First grasp | First native | Actions after first native | Strict complete | Max strict streak | Stop |
|---|---:|---:|---:|---:|---|---:|---|
| 0 | 600/600 | 67 | 358 | 241 | No | 0 | action budget |
| 1 | 600/600 | 83 | 193 | 406 | No | 0 | action budget |
| 2 | 600/600 | 69 | 159 | 440 | No | 0 | action budget |

Event steps are zero-based. All 1,800 actions are identity copies of returned VLA
proposals; all failures retained. There are 1,087 actions after the first native
event, of which 483 immediately follow a returned `done=True` (241/153/89).
Native success is non-monotonic: it is false at index 1's final frame. Final book
release and full-footprint containment fail in every run; index 0 also fails
bottom seating. No run has even one interval satisfying all strict conditions.
Per-run median inference was .286/.287/.288 seconds, and rollout wall time
186.8/187.7/188.6 seconds. Video time is simulation time, not inference throughput.

## Continuation and independent evidence

LIBERO returns task success as a local `done` without changing Robosuite's internal
horizon flag; its wrapper performs no autoreset. Frozen source inspection plus
all actual continuation transitions validate stepping after native success without
reset or clearing done. Internal done remains false, timestep increments exactly
once per action, and simulation time advances continuously by .05 seconds.

All three fresh-process replays pass with **zero state gap and zero mismatches**:
initialization, every RGB frame, controller/physics/image fingerprints, contact
solver-sample hashes, native predicates, every strict check and completion streak.
Replay re-executes the recorded actions; it does not establish model re-query
determinism. A separate artifact check validates stored contact tapes, all 1,800
raw/processed image hashes, seven tokens/native actions/executed actions, ownership,
constant language, continuous clocks, source hashes and decoded video frame count.
The observable decision trace is RGB+language → tokens/action → identity execution,
with no assertion of hidden model reasoning.

## Exact artifacts

Root: `/home/ami/scratch/shared_actor_qualification_20261019_a/`

* `book_init{0,1,2}/`: `status.json`, `steps.jsonl`, `contacts.jsonl`,
  `states.npz`, `images.npz`, `model.xml`, `initial_source.json`, policy input and
  pinned-worker logs; cohort logs are in the root.
* `source/`, `external_semantics/`, `launch.json`: pre-rollout source/protocol freeze,
  external step semantics hashes, exact commands, timing and GPU UUID.
* `replay_init{0,1,2}/verification.json`: independent exact replay results.
* `artifact_checks.json`, `summary.json`: machine-readable results and checks.
* **`actor_only_qualification_verified.mp4`**: recommended review video, 1152×704,
  641 frames at 20 fps, 32.05 s; includes every state frame and 40 explicitly labelled
  end-hold frames, with owner, native/strict outcomes, failed strict checks and stop
  reasons. Initial-frame annotations are aligned before the first query.
* `qualification_endpoint_contact_sheet.png`: initial / first native / final
  frames for all three cases, inspected visually.
* Original frozen-runner `actor_only_qualification_annotated.mp4` retained too;
  use the verified video above for correctly aligned initial-frame annotations.

Postprocessing: `scripts/check_shared_actor_qualification_artifacts.py`.
Its first check attempt found `ffprobe` absent **after** all evidence checks and
video rendering; the retained `artifact_checks.log` records this. Replaced that
metadata check with imageio's bundled FFmpeg and reran successfully
(`artifact_checks_v2.log`). No actor run or frozen source was modified/retried.

Supplemental checkpoint/base-code hashes and both runtimes' package versions:
`/home/ami/scratch/shared_actor_qualification_data_audit_20261019/checkpoint_provenance.json`.
Local adaptation audit: `shared_actor_adaptation_data_audit_20261019.md`.
Fridge pretrained/data extension: `shared_actor_fridge90_protocol_20261019.md`.

Limitations: index 0 was already a development state; indices 1/2 were fresh here,
but three cases are not a general success-rate estimate. Native-book manipulation
does not establish fridge completion or framework-component efficacy. No training
data were generated from these evaluation tapes, and no training was launched.
The strict release/support/dwell endpoint is stronger than LIBERO's native goal;
0/3 strict completions is not a contradiction of a published native-goal score,
nor proof the model cannot learn release with suitable independent demonstrations.

Both cohorts' complete evidence and verified videos are now
[published with independent remote recovery proof](../../backups/shared_actor_20261019/README.md).
