# Experiment 1: fixed fridge-camera cohort — completed evidence, no measured grasps

## Result and scope

**All six actor-only rollouts completed 600 queries/actions and stopped at the action
budget. Neither camera produced a measured grasp or native success on any of the
three development seeds. All six cached independent replays passed exactly.**
The full stable retrieval endpoint was **unscored**, not a measured strict-success
rate: this is an initially open fridge with the constant instruction
`pick up the cream cheese box inside the fridge`.

The camera change produced different initial images and first-action tokens, but
these data establish no grasp benefit, general skill improvement, or causal
explanation of failure. Later trajectory differences also change the observations.
The reframed initial view visibly includes more of the hand/arm; that is not a
verified occlusion solution or learned retrieval capability. Three nearby
development initializations do not support a population success-rate claim.

This follow-up performed **zero new actor queries**, four fresh replay subprocesses
on physical GPU1, and CPU video/metric finalization. The previously integrated
experiment-2 results were not rerun or edited.

## Status accounting and preserved failures

Root: `/home/ami/scratch/fridge_observation_cohort_20261019_a/`.

* Original `launch.json` remains `complete_with_issues`; all six original jobs
  remain `job_error`, with rollout `returncode: 0` followed by a **300-s independent
  replay timeout**. Supervisor shell exit 0 did not mean those original replays
  passed. All six per-run `status.json` files say `completed` and
  `termination: action_budget`.
* Original `replay_seed{0,1,2}_{robot_side,framed_side}/` folders, logs, frozen
  sources, images, states and statuses remain intact. No timeout folder was
  overwritten, renamed, or reclassified as a passing replay.
* `finalization_v1/replay_finalization.json` says
  `six_cached_exact_replays_validated`.
* `finalization_v1/review/results.json` says `complete_cached_replay_verified`.
  This is a separate finalization status, not a replacement for original history.
* The before/after SHA256 inventory covers **242 pre-existing files**, including
  original timeout evidence and the two existing cached seed-0 reports;
  `historical_file_changes` is empty after replay and rendering.

### Complete per-run outcomes and motion metrics

Every row below has **600 queries, 600 executed model actions, zero grasp-positive
steps, zero native-positive steps, zero item–hand endpoint-contact steps, zero
interventions**, and `action_budget` termination. No failed interval was removed.
Near-zero means the L2 norm of XYZ controller commands is below .01; it does not
mean the robot is physically motionless. Displacements reference the settled
pre-query observation, not the first post-action state.

| Seed | Camera | Near-zero XYZ /600 | Unique token sequences | Max item displacement (m) | EEF endpoint displacement (m) | Max EEF displacement (m) |
|---|---|---:|---:|---:|---:|---:|
| 0 | original `robot_side` | 492 | 51 | 6.59e-13 | .112123 | .119624 |
| 0 | frozen `framed_side` | 515 | 79 | 9.29e-13 | .201821 | .201821 |
| 1 | original `robot_side` | 510 | 43 | 8.26e-13 | .103225 | .109557 |
| 1 | frozen `framed_side` | 175 | 379 | 7.52e-13 | .677583 | .681207 |
| 2 | original `robot_side` | 343 | 160 | 8.47e-13 | .233268 | .245031 |
| 2 | frozen `framed_side` | 319 | 249 | 7.29e-13 | .400152 | .400152 |

The object remained numerically stationary while robot motion differed. More
motion or more distinct tokens is not evidence of useful task progress. Contact
counts here are endpoint measurements; this cohort does not contain solver-wide
contact interval tapes or a strict released-placement dwell score.

## Initial pairing, actual seed diversity and fixed camera

For **each of the three pairs**, checks passed for exact initial simulator arrays,
integration/controller fingerprints, complete initial diagnostics, wrist-image
arrays, and task BDDL. Actor-camera images differ. The saved model XML is identical
within each pair **except the agentview camera position**. Actor build, constant
language, action cap and calibration file match. First token divergence is at
zero-based action **0** in every pair.

The three seeds have **three distinct integration hashes but one controller hash**.
Initial EEF position/quaternion and door angle are identical; the open angle is
2.2689280275926285 rad (130°). Actual initial item positions are:

| Seed | Item XYZ (m, rounded) | Fridge body XYZ from saved XML (m, XML precision) |
|---|---|---|
| 0 | (-.0402656214, .3824978984, 1.0039059215) | (.00583450, .350578, .905) |
| 1 | (-.0555523703, .3853293502, 1.0039059215) | (-.00945225, .353409, .905) |
| 2 | (-.0436774455, .3825028419, 1.0039059215) | (.00242268, .350583, .905) |

Pairwise item-position distances are **15.547 mm (0–1), 3.412 mm (0–2), and
12.207 mm (1–2)**. This is real but small fixture/item translation diversity, not
three materially different robot/door configurations. Full fingerprints and
geometry are in `review/results.json`.

All six copies of `camera_config.json` have SHA256
`d3374b2fc031298999214d41b6f3275ce9f61ddb14ba932d01d49e12d1924ea1`.
Original position is `[-.55, .85, 1.55]`; the single seed-0-calibrated framed
position is `[-.7026721992411216, .7452847767147052, 1.6754719882552405]`.
Orientation is unchanged, with quaternion
`[-.35707962581567276, -.19015189672780805, .42984819422424386, .8071969567090258]`;
FOV is 45°. Frozen scene code asserts FOV compatibility, and all saved camera
attributes are identical across seeds within each camera condition. There was no
per-seed recalibration, orientation/FOV tuning, or outcome-driven camera search
during this finalization. The existing policy crop remains part of the observation
pipeline; review images show the recorded full agent view, not a new policy crop.

## Independent replay timeout cause and corrected provenance

The original frozen verifier opens `images.npz` once but evaluates
`images[key][i]` inside its frame loop. NumPy's NPZ member access decompresses the
**whole named array** each time, rather than selecting one compressed frame. With
601 states and two cameras, this entails 1,202 whole-array accesses. The corrected
verifier materializes the two arrays once before the loop. This removes redundant
decompression without changing scene creation, stored actions, measurement
functions, comparison conditions or acceptance tolerances.

Corrected tool: `scripts/verify_fridge_observation_cached.py`, byte-checked against
commit **`e4944c7`**, SHA256
`620ee3041e55608eb8dfb3879f0547de6f1d5993276d81e04637c5c5e5d616d0`.
An exact copy was captured in `finalization_v1/source/`. It imports each run's
hash-checked frozen scene/measure implementation; replay does not load or query
the actor. Every accepted report has **600 steps, `max_state_gap: 0.0`, and
`mismatches: []`**, with exact raw agent/wrist images, integration/controller
fingerprints, pre-action trace evidence, post-action measurements and done flags.
Action/proposal/executed-action identity is also checked. No tolerance was relaxed.

| Run | Cached report handling | Verifier elapsed (s) |
|---|---|---:|
| `seed0_robot_side` | Existing report validated; not rerun | 8.692 |
| `seed0_framed_side` | Existing report validated; not rerun | 8.057 |
| `seed1_robot_side` | New fresh subprocess, return code 0 | 8.930 |
| `seed1_framed_side` | New fresh subprocess, return code 0 | 8.946 |
| `seed2_robot_side` | New fresh subprocess, return code 0 | 8.928 |
| `seed2_framed_side` | New fresh subprocess, return code 0 | 9.017 |

Each corrected report is at
`cached_replay_<run>/verification.json`, separately from `replay_<run>/`.
The four new subprocess commands, PIDs, 300-s hard bounds, exit codes, timings,
report hashes and logs are recorded under `finalization_v1/`. Exact replay
establishes reproducibility of stored actions and measurements, not model
re-query determinism or successful task performance.

## Costs

| Run | Query median / p95 (s) | Total inference (s) | Recorded run wall incl. setup (s) |
|---|---:|---:|---:|
| seed0 original | .282940 / .294886 | 173.363 | 199.222 |
| seed0 framed | .284407 / .296894 | 174.511 | 202.020 |
| seed1 original | .284118 / .299560 | 174.466 | 200.431 |
| seed1 framed | .284053 / .297762 | 174.208 | 201.381 |
| seed2 original | .283779 / .292613 | 173.188 | 199.361 |
| seed2 framed | .285172 / .295337 | 174.346 | 201.342 |

Totals: **3,600 queries/actions**, 1,044.082 s recorded inference,
1,203.758 s summed per-run status wall, and 3,025.217 s original supervisor wall
(including six 300-s replay timeouts, 1,800 s of timeout budgets). The six cached
verifiers report 52.571 s combined; the four new subprocesses took 37.679 s wall
combined, within 37.992 s of replay-finalizer wall. Video/metric generation took
88.798 s on CPU. No model download, training or new inference was performed.

The frozen actor was LIBERO10 revision
`80970322773f81baa2e22fe495d0487b93a05cfa`, official TF4.40.1 greedy H=1,
Torch 2.6.0+cu124, normalization `libero_10`. Twenty declared zero-motion,
open-gripper settling calls precede each recorded sequence. Each evaluated sequence
contains 30 s of simulation, irrespective of approximately three minutes of
inference/rollout wall time.

## Videos and annotation audit

Review directory:
**`/home/ami/scratch/fridge_observation_cohort_20261019_a/finalization_v1/review/`**

All three paired movies are **original left / frozen framed right**, 1280×992,
20 fps, 641 frames and 32.05 s. They contain the initial observation plus all 600
post-action frames, then 40 explicitly labelled extra endpoint-hold frames. Held
frames add no physics, actions or dwell. Initial frame annotations precede the
first query; source frame `i>0` shows action `i-1` and its pre-action evidence
through action `i-2`. Playback is 1× simulation time, not real-time inference.

| File | SHA256 |
|---|---|
| `seed0_original_left_framed_right.mp4` | `8413ccb7afa430dfacb9e51018d32e4cb5214643403ff9cad1adce0309cf7dc7` |
| `seed1_original_left_framed_right.mp4` | `facf25699469ee699735b89cd74bdcff2307278b5a32eee0153cc05cb561ff49` |
| `seed2_original_left_framed_right.mp4` | `38f23813a8bdf41de179fcd1d9ff4dbcc8ac59b7febdf5624d726bab25a458ac` |

Each movie has a matching `.annotations.jsonl` containing every panel's source
frame/hash, action/evidence indices, hold flag and rendered text. The audit covers
**3,600 action annotations, six initial annotations and 240 held annotations**,
including action identity, timing alignment, per-frame RGB fingerprint checks,
VLA ownership, no repair/handoff, label bounds, strict-endpoint-unscored labels and
termination. Video decoding verifies frame counts, dimensions and frame rate.

`annotation_audit.json` reports `all_programmatic_annotation_checks_passed`.
`visual_review.json` records sampled decoded-image inspection: all three initial
frames, seed-0 midpoint, seed-1 endpoint, seed-2 end hold, and the complete six-run
initial/mid/final contact sheet. Sampled labels are legible and aligned; this does
not claim every decoded frame was watched. Additional decoded samples at frames
0/300/600/640 for every movie are retained. `all_runs_contact_sheet.png` includes
all failures, and `results.json` contains complete metrics, checks, source/video
hashes and replay provenance.

## New files, preservation and resource release

Only these new repository files were authored for this follow-up:

* `scripts/finalize_fridge_observation_cached_v1.py`
* `scripts/review_fridge_observation_cached_v1.py`
* `exp_record/agent_notes/fridge_observation_results_20261019.md` (this note)

Main's verifier, renderer, frozen rollout code and shared plans were not edited.
The new scripts are also copied/hash-identified under the finalization artifact
directories. This finalization does not depend on the original renderer's
`replay_<run>` path convention; it explicitly validates `cached_replay_<run>`.

GPU1 UUID: `GPU-3f4f72ef-ed7f-2ddb-e454-abcc6c0330b0`. All four new replay processes
exited; `finalization_v1/resource_release.json` records no remaining replay PIDs,
no GPU1 compute process and 5 MiB device memory at the release check. GPU1 is
released. There were no commits, pushes or archives by this follow-up; publication
and integration remain with main.
