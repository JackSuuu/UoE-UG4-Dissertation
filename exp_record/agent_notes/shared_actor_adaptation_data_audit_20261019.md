# Experiment 2: refrigerator adaptation-data audit

**Actual fridge adaptation is blocked: no valid local fridge demonstrations or
training/development/evaluation split were found. No training, relabeling, or
adapted checkpoint was produced.** Native-book qualification is a separate
model-only experiment; it cannot establish fridge competence.

**Fridge-priority update:** remote search subsequently identified genuine RoboCasa
CloseFridge data/policies and Doosan fridge demonstrations, with different robot
interfaces. See `shared_actor_fridge_pretrained_audit_20261019.md`; this local audit
does not imply that fridge data are unavailable remotely. A compatible LIBERO90
OOD-transfer candidate is covered by `shared_actor_fridge90_protocol_20261019.md`.

## Inspected evidence

Reproducible read-only inventory:
`scripts/audit_shared_actor_adaptation_data.py`; output
`/home/ami/scratch/shared_actor_qualification_data_audit_20261019/inventory.json`.
It records absolute paths, sizes, timestamps, HDF5 schemas/attributes/episode counts,
and SHA256 of inspected collection/training utilities. Dataset contents themselves
were not exhaustively hashed. Roots: project, `/home/ami/scratch`, Hugging Face
dataset cache, and `/home/ami/jack-project/data`; package/venv/git directories are
excluded. Additional home-wide file-name searches found only package sample demos
outside these roots, and no TFRecord shards. This is a local availability audit,
not a claim about all remotely obtainable robot datasets or OpenVLA pretraining.

| Available resource | Observed provenance / contents | Suitable fridge labels? |
|---|---|---|
| `/home/ami/scratch/LIBERO/datasets` | Exists, empty; submodule `third_party/LIBERO/datasets` absent | No |
| Both LIBERO checkouts' `bddl_files` | 130 task definitions each, no `fridge` text match; LIBERO10's ten tasks listed in inventory | Definitions are not demonstrations |
| LIBERO `init_files/*.pruned_init` | Benchmark simulator initial states, used for evaluation initialization | No action demonstration sequence |
| Frozen LIBERO10 checkpoint | Revision `80970322773f81baa2e22fe495d0487b93a05cfa`; weights/statistics, not a dataset | No |
| `/home/ami/scratch/openvla_push_demos.h5` | 3,000 episodes / 120,000 frames, 224x224 RGB, actions `[120000,2]`, units `m/s world-frame planar velocity`; `src/vla/collect_demos.py:69` calls `sim.expert` | No: handwritten planar pushing expert, wrong task and action space |
| `openvla_push_features{,_visfast}.h5` | 54,000 feature rows each, 3,000 episodes, 4096-D features; targets H=5/10 by 2 actions; pushing instruction in attributes | No: derived from the same pushing data |
| `dagger_{r1,r2,h10_r1,h10_r2}.h5` | 1,024 episodes / 16,384 feature rows each; 2-D targets; `src/vla/dagger.py:40-48` labels cloned states with `sim.expert` | No: scripted expert labels, not fridge demos |
| `/home/ami/scratch/fridge_*`, previous VLA audits, current qualification | Development/actor/repair evaluation tapes; include manually controlled repair trials | Excluded: evaluation traces cannot become training data; scripted repairs cannot supply labels |
| HF dataset cache / `jack-project/data` | LLaVA instruction-image resources, no action-labelled fridge dataset identified | No |

## What the local utilities can actually do

* `/home/ami/scratch/LIBERO/scripts/collect_demonstration.py` supports a human
  keyboard/SpaceMouse device, `OSC_POSE`, 20 Hz, custom `--bddl-file`, and records
  model XML/states/actions in HDF5. It requires an interactive display/device;
  this headless qualification did not establish that interface works. Stock
  collection only holds the **native** predicate for ten steps, so collection
  acceptance must be extended to the declared fridge release/support/close endpoint.
  It also removes aborted demonstrations; a new collection wrapper must retain
  an attempts ledger and failures, rather than silently count only successes.
* `scripts/create_dataset.py` can replay demonstrations and produce RGB/action
  datasets. Its stock image size is 128, it discards five startup steps, and it
  renders observations **after** stepping. Before training a new adapter, explicitly
  test and document `(observation_t, action_t, observation_t+1)` alignment against
  the raw tape; do not blindly assume its resulting arrays match the actor schema.
  It needs the same runtime compatibility fixes, custom geometry and camera as
  collection/evaluation. A state-playback mismatch warning is not an exact-replay pass.
* `src/vla/extract_features.py` is implemented for base OpenVLA, TF5 compatibility,
  a hardcoded pushing prompt, and pushing HDF5 fields. It is **not** an existing
  pinned-TF4 LIBERO10 fridge training pipeline.
* `src/vla/train_chunk_head.py` uses episode-level validation and train-only
  feature normalization, but `ChunkHead` is hardcoded to `(H,2)` planar velocity.
  Changing the output to seven dimensions requires a new actor/schema and explicit
  controller/gripper validation. It cannot directly adapt the seven-token actor.
* LIBERO `libero/lifelong/{datasets.py,main.py,algos/base.py}` and
  `models/bc_transformer_policy.py` supply genuine behavioral-cloning utilities
  with HDF5 sequence loading. They are a different policy family, not an OpenVLA
  fine-tuning implementation. Base Python has no `robomimic` or
  `tensorflow_datasets`; merely finding source does not establish runnable training.
  No local official OpenVLA fine-tuning checkout was found.

## Concrete feasible learned route — proposed, not performed

1. **Freeze collection conditions and split before collecting.** Fix the corrected
   reachable fridge geometry, camera, 20-Hz OSC_POSE scales, language and full task
   endpoint (open, extract, release/support on table, close). Reserve disjoint
   initialization/layout seeds and collection sessions: e.g. 100 independently
   teleoperated episodes allocated 70 training / 15 development / 15 held-out,
   with a separate fresh held-out closed-loop seed list. These are proposed counts,
   not available samples or a guarantee of sufficient coverage. Never allocate
   adjacent frames or alternate views of the same episode across splits. Do not
   reuse any current or historical evaluation trace as a training episode.
2. **Collect real demonstrations rather than motion-program labels.** A human
   operates the existing LIBERO keyboard/SpaceMouse interface on the reserved
   collection seeds, including release dwell and door-close behavior. Retain all
   attempts, their outcomes, operator/session ID, XML/BDDL/camera/controller hashes,
   seed, timestamp and unedited seven-dimensional actions. No scripted waypoints,
   analytic geometric expert or privileged simulator-state-to-action relabeler.
   An agent-only headless task cannot manufacture this missing human/device input.
3. **Convert and verify.** Re-render paired pre-action RGB at the exact pinned
   actor orientation/resolution/preprocessing, preserve native language and actions,
   validate action timing with exact replay and strict completion. Fit action/feature
   normalization only on training episodes; keep gripper convention explicit.
   Create immutable manifest hashes and episode split lists before feature extraction.
4. **Implement a bounded learned adaptation.** A feasible single-24-GB-GPU route
   is to cache frozen **pinned-TF4 LIBERO10** fused features at batch 1, then learn a
   new H=1 seven-dimensional readout (six continuous OSC outputs plus a gripper
   classifier) from those training demonstrations. Existing feature/head scripts
   supply reusable ideas, but need a new adapter, per-episode language, seven-axis
   targets and official-stack parity tests. Train the small readout separately from
   feature extraction so optimizer state does not compete with the 7B backbone.
   This is a **new learned shared actor**, not the current official token decoder;
   preserve that distinction in all reports. Choose checkpoint on development
   episodes only, freeze normalization/head/backbone/code, and test the independent
   held-out demos and fresh closed-loop seeds once under the same strict endpoint.
5. **Only after shared competence is demonstrated**, use that frozen actor for both
   framework and simple-feedback arms, with identical observation/action/query
   budgets and information. No online update from their evaluation trajectories.

Prerequisites still missing: valid fridge demonstrations and split, verified
teleoperation setup, aligned converter, pinned seven-axis learned adapter and
held-out competence result. No learned adaptation, refrigerator success, or
framework advantage is claimed by this audit.
