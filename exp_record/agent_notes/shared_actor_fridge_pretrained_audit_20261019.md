# Experiment 2: learned fridge data / checkpoint search

## Decision

**Fridge demonstrations and learned fridge policies do exist remotely. None of the
inspected fridge-trained policies has the current fixed-base LIBERO Panda's
observation/action contract. LIBERO90 itself has no fridge task.** Therefore the
local “no fridge demos found” result must not be read as “no fridge data exist.”

The immediate compatible candidate is a public LIBERO90 OpenVLA checkpoint, tested
under the separately frozen `shared_actor_fridge90_protocol_20261019.md`. It is a
**primitive-overlap OOD transfer candidate**, not a demonstrated fridge expert.
There is no framework-only actor switch. Any later actor adoption must be shared
and frozen across all study arms.

Metadata-only audit script: `scripts/audit_fridge_pretrained_candidates.py`.
Evidence root: `/home/ami/scratch/shared_actor_fridge_pretrained_audit_20261019/`.
`summary.json` preserves repository revisions, public URLs, file sizes/LFS hashes,
cards/configs/statistics and their hashes; no downloaded code was executed.
`official_libero90_inventory.json` and `robocasa_task_metadata.json` strengthen
the task-overlap check. Access date: 2026-10-19. Sizes below are decimal GB/MB;
they describe repository files, not installed runtime memory.

## LIBERO90 is not an overlooked fridge-demonstration suite

* Local `libero_suite_task_map.py` has exactly 90 LIBERO90 task names, with no
  `fridge` / `refrigerator`. All 130 BDDL definitions in each local checkout were
  also scanned; none contains fridge. The custom `short_fridge` asset's existence
  does not imply benchmark demonstrations use it.
* The official public [LIBERO dataset](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets)
  revision `f13aa24a3da8c43c7225569f28c562979fa0e35a` has **90 LIBERO90 HDF5 files,
  66,658,085,995 bytes**, with no fridge filename. All five suites total ~100.443 GB.
  These are genuinely available demonstrations, but not demonstrations of the
  custom fridge task. They were not downloaded.
* [jesbu1/libero_90_rlds](https://huggingface.co/datasets/jesbu1/libero_90_rlds),
  revision `93169e35e1e6ddf6c43171bf038cb4971b60e72a`, is ~20.312 GB. Its card
  says it removes no-ops/unsuccessful trajectories and flips images upright using
  OpenVLA's processing. This is a third-party conversion, not newly collected
  fridge data. No train/held-out fridge split can be derived from it.
* Actual related LIBERO90 tasks include `KITCHEN_SCENE7_open_the_microwave`,
  `KITCHEN_SCENE6_close_the_microwave`, and cream-cheese-to-basket/tray tasks.
  They share hinged-door or grasp/transport primitives and Panda OSC_POSE control;
  door geometry, occlusion, camera and full task semantics differ from the fridge.

## Candidate inventory and interface audit

| Resource | Provenance and task overlap | Size and compatibility | Decision |
|---|---|---|---|
| [VQ-VLA/openvla-7b-finetuned-libero-90](https://huggingface.co/VQ-VLA/openvla-7b-finetuned-libero-90), `794ef81b7be928ea9270e81ca1ef5b60ffa9420f` | Third-party; card only says MIT. Statistics report 3,919 trajectories / 569,315 transitions, `libero_90_no_noops`. Exact training recipe, split and claimed performance unavailable | ~15.085 GB, four safetensor shards. Architecture/config matches existing LIBERO10 actor; same 7-D Panda OSC interface and RGB+language. Explicit new q01/q99 statistics needed | Selected once, before outcomes, for bounded actor-only fridge transfer; not called fridge-trained |
| [Stanford-ILIAD/minivla-libero90-prismatic](https://huggingface.co/Stanford-ILIAD/minivla-libero90-prismatic), `4289f87e8e00706e188c7a3a61fc6e7d72ab2564` | LIBERO90; stats report 3,924 trajectories / 569,230 transitions. Same task-overlap limitation | Repository ~11.110 GB for two 5.555-GB checkpoints; new MiniVLA loader/backbone rather than pinned TF4 OpenVLA | Not selected; another compatible-domain candidate would add an outcome-driven actor search |
| [lerobot/smolvla_robocasa](https://huggingface.co/lerobot/smolvla_robocasa), `f7af588d11e30b0069f200792fb285904b74c80b` | Card names `lerobot/robocasa_target_human_unified`; its actual task metadata includes `CloseFridge` (index 1). This is evidence of a genuinely fridge-trained policy, but not of full open/extract/place/close competence | ~906.747 MB. **16-D state, three 256² cameras, 12-D action, chunk=50**, PandaOmron mobile embodiment. State/action MEAN_STD; visual IDENTITY; released processor stats must be used | Strong native-RoboCasa close-fridge candidate; cannot truncate 12→7 or invent absent state/cameras to run it in LIBERO |
| [pepijn223/robocasa_CloseFridge](https://huggingface.co/datasets/pepijn223/robocasa_CloseFridge), `b4b32587d3fd89f7c7121e62ff55c1c8f06f5aa4` | Actual fridge-closing demonstrations: 513 episodes / 155,443 frames, 20 Hz, PandaOmron. Task metadata includes “Close the fridge doors.”, “Close the fridge door.” and `CloseFridge`; info declares two tasks (language variants) | ~410.671 MB. Three views, 16-D state, 12-D actions. Dataset exposes only train `0:513` | Real data; lacks a declared independent held-out split and is not native fixed-Panda fridge data |
| [lerobot/robocasa_target_human_unified](https://huggingface.co/datasets/lerobot/robocasa_target_human_unified), `a1c7cba1a128f5dd8d3084012ba366d10ce174d9` | 25,307 human episodes / 14,957,899 frames / 50 tasks, 20 Hz; `CloseFridge` verified in task metadata | ~188.394 GB, only train `0:25307` in metadata | Do not download entire set to obtain one skill; establish episode/layout split and provenance before adaptation |
| [aleleanza/diffusion-policy-vac-fridge](https://huggingface.co/aleleanza/diffusion-policy-vac-fridge), `7fe0d9ece0dd94acaa8962abf84b320d4a1b266f` | Card: real fridge/variable-impedance pick on **Doosan M0609 + Inspire RH56 hand**, 200 demos / 54,833 frames, 20 Hz; per-variant split metadata advertised | Whole ablation repo ~20.247 GB; one best checkpoint ~1.25–1.29 GB. 84² RGB; 7-D Cartesian **velocity**+hand or 9-D pre-admittance+compliance. Optional wrench/state | Even the seven-dimensional variant is not Panda OSC pose-delta control; dimension equality is not interface equivalence |
| [ryota-tron/test_act_open_fridge](https://huggingface.co/ryota-tron/test_act_open_fridge), `1d799a8a2098f32587bf685d7768f70762c73b63` | Name suggests fridge opening; no README available, so source/embodiment/performance unresolved | ~206.730 MB; 128² image, 7-D state, **8-D action**, chunk=100, MEAN_STD | Insufficient provenance and incompatible output; not executed |

Additional task-specific lead: [wls04/pi0.5_OpenFridge_60k](https://huggingface.co/wls04/pi0.5_OpenFridge_60k),
revision `a123f4337d3a8c288e3aeb1fdbe4534efc18831b`, 8,526,557,492-byte
`model_state_dict/full_weights.pt`. Its card reports sequential 20k-step stages
OpenMicrowave → OpenCabinet → OpenFridge, initialized from a C0 calibration
checkpoint, configuration `pi05_robocasa365_pretrain_human300`. It supplies
`norm_stats.json` / `rlinf_model.yaml`, is **native RLinf/OpenPI rather than a
Transformers export**, and explicitly makes no performance claim. The held-out
split is not stated. It remains a genuine fridge-learning lead requiring native
RoboCasa setup and provenance checks, not a drop-in current-scene actor.
Its downloaded `rlinf_model.yaml` confirms **12-D actions, three images, proprio,
five action chunks and five inference steps**. Pinned card/config/normalization
metadata and hashes are retained in the evidence root's
`wls04_pi05_OpenFridge_60k/` directory; no weights were downloaded for this lead.

## Compute and split implications

The selected 7B candidate can use the already validated GPU3 batch-1 official TF4
runtime: current measured 600-action inference+physics rollout is ~187 seconds,
but this is an estimate for the new weights, not a promised latency. Download is
hard-bounded at 900 seconds; four LFS SHA256s must pass. Two rollout processes each
have 600 queries/actions, 1200-s rollout and 1800-s process caps, plus independent
360-s replays. No training or demonstration download is needed for this test.
The new normalization key and statistics are preserved separately from LIBERO10.

The ~0.907-GB SmolVLA checkpoint is plausible on a 24-GB GPU **in its native
environment**, but no memory/latency measurement was performed. Neither `lerobot`
nor `robocasa` is installed in the current runtime. Their [documented setup](https://huggingface.co/docs/lerobot/en/robocasa)
requires dedicated newer Robosuite/RoboCasa/LeRobot dependencies and kitchen assets;
the optional extra object registry alone is ~30 GB. Minimum required asset size
was not established. Installing these into the pinned LIBERO environment would
invalidate its runtime. A dedicated environment and versioned native scene would
be needed; this audit does not silently migrate the experiment.

For an actual RoboCasa CloseFridge capability test, the feasible **future** protocol
is: pin the model/release/runtime/assets and released processor stats, use the
documented three-camera mapping and full 12-D mobile action space, batch 1/GPU3,
three predeclared fresh layout/initialization seeds, 600 executed actions and 1200 s
per episode, no scripted motion, native close plus a separately frozen stable
closed/released/clear dwell endpoint, and independent replay. Count both chunk
queries and executed actions because n_action_steps=50. Do not call this success
on the current LIBERO fridge or on the entire retrieval sequence. Confirm that
evaluation layouts/seeds were not training rollouts; published train-only data
metadata alone does not establish this.

For adapting **the current fixed-base fridge**, the route in
`shared_actor_adaptation_data_audit_20261019.md` still applies: independent full-task
human demonstrations with an episode/session/layout split, a verified seven-axis
converter/learner, and a shared frozen learned actor. Remote PandaOmron or Doosan
demonstrations need a real embodiment-adaptation method; copying labels or using
handwritten motion solutions is not such a method. All current and historical
evaluation tapes remain excluded from training.

## Completed compatible-candidate test

The selected LIBERO90 checkpoint downloaded and passed all weight hashes. Both
600-action actor-only conditions failed to grasp or complete; the closed-door
condition also never opened the door. Exact replay and token/normalization/RGB/
contact/clock checks all passed. See `shared_actor_fridge90_results_20261019.md`
and `/home/ami/scratch/shared_actor_fridge90_20261019_a/artifact_checks.json`.
The candidate is not qualified for adoption as the shared fridge baseline.
