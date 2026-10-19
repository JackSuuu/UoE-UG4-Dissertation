# RoboCasa365 R0 readiness audit — 2026-10-19

## Decision

**Public native policies with published full-composite success are available. R0 artifact discovery is complete; local runtime readiness and actor competence remain unqualified.** Shortlist **Xiaomi-Robotics-1-RoboCasa365** and **Awomo-0.5I/0.5V-RoboCasa365**. An official **GR00T N1.5 composite-seen target-posttrained** checkpoint is a useful fallback under a different training/evaluation regime. No policy is selected by this note.

For a later first qualification, prefer Xiaomi provisionally for its packaged native checkpoint and comparatively straightforward observation-history interface; choose Awomo-I if its documented approximately 25 GB/server GPU requirement and additional dependencies fit the available resources. Awomo-V remains a task-dependent alternative, with a less-complete dependency/interface audit here. Neither preference establishes local feasibility. **PrepareCoffee is not empirically the easiest of the three proposed tasks.** Preserve it as the planned entry task if desired, but do not make failure on coffee alone evidence that the platform lacks composite-capable actors.

OpenVLA + native LIBERO-Long remains the immediate mechanism-development platform, following [the authoritative experiment queue](current_goal_and_experiment_queue_20261019.md). A large RoboCasa efficacy campaign remains gated by LIBERO mechanism evidence and actor-only RoboCasa qualification. CheckVLA is currently a published contextual comparison, not a verified runnable public baseline.

### Audit scope and evidence strength

- Inspected official benchmark pages/paper/source, public policy repositories, checkpoint metadata/configurations, native adapters and evaluation entry points. All availability statements are dated **2026-10-19**.
- HF metadata reports the listed repositories as public and ungated. Weight-file HEAD checks returned HTTP 200 for Awomo-I `model.pt` and Xiaomi's first shard. This verifies artifact access, not successful loading or compatibility of every dependency.
- Saved source/metadata evidence in `/home/ami/scratch/robocasa365_readiness_20261019/`; `sources_manifest.json` records URLs, byte counts and SHA256 values, and `checkpoint_inventory.json` records checkpoint revisions and file sizes. One guessed `utils/gym_utils.py` URL returned 404; the correct `wrappers/gym_wrapper.py` was subsequently identified [S3].
- No weights downloaded, installations, GPU jobs, training or rollouts were performed. Published results below are authors' results, not local reproductions. Only this new repository note was written for R0.

## 1. Native task support and published capability

All three tasks are **`composite_seen`**, included in Human300. The contemporary leaderboard evaluates the target50 **task identities** using **`split=pretrain`**; this is distinct from evaluation in target kitchens. Human300-only submissions therefore support these task identities without establishing target-kitchen generalization [S1, S2, S4, S5].

| Policy / published evaluation | PrepareCoffee | SteamInMicrowave | StoreLeftoversInBowl |
|---|---:|---:|---:|
| Awomo-0.5I; 30 trials/task | 5/30 = 16.7% | 19/30 = 63.3% | 25/30 = 83.3% |
| Awomo-0.5V; 30 trials/task | 10/30 = 33.3% | 23/30 = 76.7% | 18/30 = 60.0% |
| Xiaomi-Robotics-1; 50 trials/task | 8/50 = 16.0% | 32/50 = 64.0% | 42/50 = 84.0% |
| RoboCasa paper, best foundation-learning GR00T variant; target-data regime | 13% | 30% | 43% |

Sources: pinned Awomo per-task summaries [S4], Xiaomi per-task/per-episode summary [S5], RoboCasa paper Table 11 [S2]. The GR00T row uses Human300+MG60 pretraining followed by 100% target-data posttraining and target-kitchen evaluation. It is contextual evidence for the released model family/regime, **not an exact current-checkpoint reproduction claim**. It must not be attributed to the Human300-only GR00T leaderboard checkpoint. Trial sets, training and versions differ; this table does not establish a controlled ranking between actors.

Awomo's published 50-task aggregates are I **64.40%**, V **62.53%**; composite-seen aggregates are 58.75% and 60.42%. Their reported Human300-only SFT schedules are I 60k steps/global batch 512 and V 40k/batch 512. Xiaomi's current source/HF summary gives **1432/2500 = 57.28%**, whereas the leaderboard displays **57.4%**; preserve both provenance labels rather than silently equating them. Its submission reports Human300 training for 120k steps/batch 512 [S1, S4, S5]. These schedules describe released training, not work needed for frozen-policy qualification.

### Native completion predicates and horizons

Audited current RoboCasa source: `6f90063c0ec2b9916bbd977368b54c98d9c2d908`. RoboCasa v1.0.1 increased task horizons by 1.5× on 2026-05-12. Freeze the simulator commit and registry horizons rather than copying an older benchmark budget [S1–S3].

| Task | Native full-goal predicate, summarized from source | Current horizon / nominal simulated time at 20 Hz |
|---|---|---:|
| `PrepareCoffee` | Mug placed at coffee dispenser; machine on; gripper sufficiently far from mug and button | 1800 / 90 s |
| `SteamInMicrowave` | Vegetable in bowl; bowl inside microwave; microwave door closed and microwave on | 2100 / 105 s |
| `StoreLeftoversInBowl` | Both foods in bowl; bowl contacts a refrigerator rack; gripper sufficiently far | 2550 / 127.5 s |

Leftovers starts with the refrigerator open; **closing the refrigerator is not required by its native success predicate**. Monitor constituent attainment and loss, but report full native success separately from additional physical diagnostics. These simulated times are not wall-time estimates: inference, rendering and serialization add cost.

## 2. Public artifact inventory

Sizes are decimal GB from HF file metadata, not GPU-memory estimates. Revisions below are the inspected immutable identities; release branches and leaderboard code references can differ.

| Candidate | Exact public checkpoint / revision | Available weights and status |
|---|---|---|
| Xiaomi-Robotics-1 | `XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365` @ `3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4` | Three safetensor shards, **10,106,433,336 bytes** total; custom model/processor/tokenizer and native normalization stats included; Apache-2.0; shortlisted |
| Awomo-0.5I | `Auwomo/Awomo-0.5I-RoboCasa365` @ `d02e3560a9e5e649bb0718a97b9e4b923df59594` | `model.pt`, **8,281,033,675 bytes**; config and SHA256SUMS; additional base components required; weights CC-BY-NC-4.0; shortlisted |
| Awomo-0.5V | `Auwomo/Awomo-0.5V-RoboCasa365` @ `947121b8a356753fec60489f744d40f71b44f447` | `model.pt`, **10,069,865,739 bytes**; config and SHA256SUMS; Wan2.2-TI2V-5B base dependency; weights CC-BY-NC-4.0; alternative |
| Official GR00T N1.5 | `robocasa/robocasa365_checkpoints` @ `c484448aba1a9b60a04c9b0ca117241518ea69f3` | Native multitask and foundation/target-posttrained variants; two model shards total **7,586,072,344 bytes** per inspected variant; fallback |
| Official π0 / π0.5 | Same official bundle/revision | Native Openpi checkpoints and stats; available, weaker Human300-only composite-seen aggregate evidence |
| SmolVLA RoboCasa | `lerobot/smolvla_robocasa` @ `f7af588d11e30b0069f200792fb285904b74c80b` | **906,712,520-byte** model plus processors/stats; composite competence not established by inspected evidence |

Checkpoint pages and inventory sources: [S4–S8] and local `checkpoint_inventory.json`.

### Code and simulator pins

- **Xiaomi:** audited code `0dd7aef8dc87296246aae812a1f59ccb708e5546`; leaderboard-linked code `4da1db0a4deefa6de7ebb4ef0b8754017290f5f7`. Use the RoboCasa365 checkpoint, not the generic deployment-guide example model. Current public artifacts were verified despite an older submission's private-access wording [S5].
- **Awomo:** audited code `d22dda9aae0f9ae8096e95580a140a784a17b513`; leaderboard-linked code `1145c18afd13a3876c6740f098bd734f5e35617a`. The reproduction README pins RoboCasa `4f8a2980def75a55dff96b990745b83540425f09` (1.0.1) and robosuite `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`. Resolve the current-code/submission-code difference in the eventual manifest [S4].
- **GR00T benchmark fork:** `robocasa-benchmark/Isaac-GR00T` @ `9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10` [S6].
- **Openpi benchmark fork:** `robocasa-benchmark/openpi` @ `5a6beda9ff99da30b4e1b59320f6a32971d7c397` [S7].
- A leaderboard “open source” flag alone is insufficient evidence of usable artifacts: the inspected Phasor submission explicitly described a private repository. Shortlist decisions here require checkpoint and native-code evidence.

## 3. Observation, action and temporal contracts

### Shared simulator boundary

The robot is **PandaOmron**, a mobile manipulator. Native gym observations expose three RGB cameras as HWC `uint8`, 256×256: `video.robot0_agentview_left`, `video.robot0_agentview_right`, `video.robot0_eye_in_hand`. The gym wrapper already vertically flips simulator images. Native state fields comprise base position (3), base quaternion (4), base-relative EEF position (3), EEF quaternion (4), and gripper qpos (2). **A shared 16-dimensional field count does not imply a shared flattening order.** Use each actor's adapter [S3–S7].

`convert_action` consumes 12 channels in this order:

```text
[0:3] EEF translation; [3:6] EEF rotation; [6:7] gripper;
[7:11] base xy/yaw and torso; [11:12] control mode
```

The pinned PandaOmron controller is hybrid mobile-base control: arm `OSC_POSE`, delta input in the base frame, translation scaling ±0.05 m and rotation scaling ±0.5 rad; base joint velocity and torso joint position controllers [S3]. These are controller-scaled commands, not absolute world-frame poses. Gym conversion thresholds gripper/mode at 0.5; preserve actor-specific preprocessing before this boundary. In particular, Awomo's internal base-first 12-vector is explicitly reordered before `convert_action`.

### Xiaomi-Robotics-1 — audited native evaluator

- **State:** EEF-relative position (3), EEF quaternion converted from xyzw to axis-angle (3), gripper qpos (2), base position (3), base quaternion converted to axis-angle (3): **14D**, padded to **60D**. Do not pass the raw native 16-vector.
- **Vision/history:** three labelled camera video streams; four image/state history samples at interval two control steps, repeating initial frames when history is short. Center-crop ratio 0.95, resized back to native dimensions; use the released processor and its chat template/instruction formatting.
- **Actions:** processor selected with `robot_type=robocasa365`; released action normalization has shape **16×60**. Decode through the checkpoint processor, take the first **12 channels** in native engine order; predict/execute **16 actions per query**. Default model sampler uses five steps. Control is 20 Hz, so a full chunk spans 0.8 simulated seconds.
- **Repair implication:** branch state must include image/state history, action queue, RNG and simulator/controller state. The explicit client-side history makes it a promising first integration, but this is an engineering inference, not a measured integration result.

Sources: `eval_robocasa365/entry.py`, checkpoint `preprocessor_config.json`/`config.json`, deploy client/model implementation [S5]; saved `xiaomi_entry.py` and corresponding configs.

### Awomo-0.5I — audited stateful server contract

- **Input state16 order:** base position/quaternion, EEF-relative position/quaternion, gripper qpos. The adapter builds masked **80D** state: EEF position slots 7:10, rotation6D 10:16, gripper closedness 16, base xy 58:60, base yaw 63. Unused slots are zero.
- **Actions:** masked 80D output; EEF rotation delta uses 10:13 and control mode 76. Continuous commands are clipped to [-1,1]; gripper closes at closedness ≥0.5 and internal mode >0 selects base; torso output stays zero. The base-first intermediate vector is reordered into native 12D engine order.
- **Vision:** all three 256×256 cameras, H.264 encode/decode roundtrip using libx264 CRF23. Canvas places wrist above the two external views; resize to **height 320 × width 224**. Preserve this preprocessing for reproduction.
- **Time/history:** 20 Hz control, **16 predicted/executed actions**, **16 flow-sampling steps**. Frames arrive every four control steps (5 Hz). Replan zero must contain exactly frame zero; every subsequent replan supplies exactly the four new frames with consecutive session/replan indices. Model uses current frame plus nine history slots spaced every 40 control steps, with validity masking for missing early history.
- **Repair implication:** server stores one episode's instruction, frame history and timeline. In-chunk interruptions, alternative branch queries and later handoff require explicit timeline/history restoration or a validated extension. Simply resetting the session on each branch changes policy conditioning. The released API is not a drop-in arbitrary-time repair interface.

Sources: `eval_robocasa365/common.py`, `awomo_05i/policy.py`, checkpoint `awomo_config.json` [S4]. The exact I preprocessing/sampler description is not generalized to V; V has a separate policy module and base model.

### GR00T N1.5 fallback

Use data config **`panda_omron`** and embodiment **`new_embodiment`**, confirmed by checkpoint metadata. The native config uses all three cameras, current observation only, quaternion-to-rotation6D transforms, min/max normalization, binary gripper/mode transforms, state padded to 64 and action to 32, crop ratio 0.95 and resize224. It predicts/executes 16 actions; the benchmark server sets four denoising steps. Preserve modality transforms and normalization metadata rather than borrowing Xiaomi/Awomo packing [S6].

Exact official paths:

```text
gr00t_n1-5/multitask_learning/checkpoint-120000
gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000
```

The first is the Human300 multitask baseline (leaderboard 23.9% overall / 14.8% composite-seen); the second is the target-posttrained fallback. A whole inspected checkpoint directory is about 16.14 GB, including an approximately 8.55 GB optimizer file unnecessary for inference. Weight-only selection still needs configs, index and metadata; it does not mean downloading only one shard [S1, S6].

### Other native policies: available, lower priority

- **π0 / π0.5:** official bundle paths `pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000` and `pi05_pretrain_human300/multitask_learning/75000`. Human300-only leaderboard composite-seen results are 6.1% / 7.1%, evaluated under v1.0.0 rather than the newer 1.0.1 submissions. Native Openpi state16 is EEF position/quaternion, base position/quaternion, gripper; π0 uses left+wrist and masks right, π0.5 uses all three; outputs12. Client default `replan_steps=5` is execution cadence, not proof of predicted chunk length. π0 payload is about 12.014 GB; π0.5 whole directory about 44.722 GB includes training state and overstates inference-only storage. Entry points: `scripts/serve_policy.py` and `examples/robocasa/main.py` [S1, S7].
- **SmolVLA:** CloseFridge documentation is not evidence of full-composite competence. Conversely, the inspected checkpoint is not established as CloseFridge-only: its training config references `lerobot/robocasa_target_human_unified`, all episodes, 20k steps/batch32; dataset metadata reports 50 tasks and 25,307 episodes at 20 Hz. Inspected config: state16/action12, predicted/executed chunk50, ten flow steps, image padding512, mean/std normalization. No inspected per-task result establishes success on the three composites. **Camera mapping discrepancy:** LeRobot documentation maps left→camera1, wrist→camera2, right→camera3; checkpoint training config maps left→camera1, right→camera2, wrist→camera3. Resolve against the released preprocessing/runtime before qualification [S8].

## 4. Deployment cost and reproducible evaluation entry points

| Candidate | Source-stated requirements / known payload | Still unverified locally |
|---|---|---|
| Xiaomi | Python3.12 in deployment guide versus 3.11 in HF card; torch2.8.0, transformers **exactly4.57.1**, FlashAttention2 (guide2.8.3), bf16; ~10.11 GB model weights | Resolve Python choice and simulator pin; actual peak VRAM, load time, per-query latency and render coexistence. No numerical VRAM guarantee found |
| Awomo-I | Python3.11/CUDA12.8, torch2.8.0, torchvision0.23.0, transformers5.5.0, diffusers0.39.0; deployment says **~25 GB GPU/server**; published I evaluation on RTX5090 | Actual peak and headroom with simulator; H.264 CPU overhead and query latency; full dependency installation |
| Awomo-V | Own requirements/base model; published evaluation RTX5090 except nine tasks on RTX PRO6000 | Complete base-model download footprint, V-specific contract and per-task resource differences |
| GR00T / Openpi | Official algorithm docs give generic ≥8 GB inference recommendation; GR00T weights ~7.59 GB | Recommendation is not a measured peak or guarantee for this checkpoint/runtime |

Awomo-I also loads **Qwen/Qwen3-VL-4B-Instruct** @ `ebb281ec70b05090aa6165b016eac8ec08e71b17` (8,875,719,344 weight bytes) and the **black-forest-labs/FLUX.2-klein-base-4B VAE** @ `a3b4f4849157f664bdbc776fd7453c2783562f4d` (168,120,878 bytes). Both repositories are ungated. Selected I+Qwen+VAE weights total **17,324,873,897 bytes (~17.33 GB)** before configs, caches and environment. The full FLUX repository is unnecessary for the VAE-only dependency. Do not interpret “0.5I” as a 0.5 GB runtime [S4].

RoboCasa installation documentation estimates about 10 GB for assets; fuller object registries can add roughly 30 GB per LeRobot documentation. Reserve separate space for environments, model cache, source, logs and videos. CPU/rendering, filesystem throughput and inference all affect episode wall time. No reliable local episode-cost estimate can be derived from weight bytes alone [S3, S8, S9].

### Runnable source paths and future command patterns

These are inspected entry points, **not executed commands or a frozen launch protocol**. Use pinned local checkpoint directories, one worker for initial qualification, a new output directory per manifest and the full native horizon.

**Xiaomi:** `deploy/server.py` / `scripts/deploy.sh`, then `eval_robocasa365/entry.py`; `dynamic_eval.py` and `scripts/launch_robocasa365.sh` provide orchestration. Reference evaluation uses target50, `split=pretrain`, 50 trials/task and seed7. The single-worker entry supports `--task-name PrepareCoffee`; freeze trial count and seeds before running. The server launch script takes `<model_path> <num_ports> <num_gpus>`; the first argument must identify the **RoboCasa365** checkpoint [S5].

**Awomo:** `scripts/deploy.sh` with module `eval_robocasa365.awomo_05i.policy` and a pinned local checkpoint; separate simulator environment. After one server is available, the source-supported task filter is:

```bash
TASK_SET=composite_seen TASKS=PrepareCoffee NUM_TRIALS=<frozen-count> \
  bash scripts/launch_robocasa365.sh 1 <new-output-directory>
```

The launch/evaluation system records identity/config, resumes unfinished episodes and reports errors; preserve retries and infrastructure failures in complete-attempt accounting rather than treating resume as a new clean sample [S4].

**GR00T:** `scripts/run_eval.py` supports the following pattern for the target-posttrained fallback:

```bash
python scripts/run_eval.py --model_path <pinned-local-checkpoint> \
  --data_config panda_omron --embodiment_tag new_embodiment \
  --task_set composite_seen --split target --n_episodes <frozen-count> \
  --n_envs 1 --n_action_steps 16
```

This command selects the **whole composite_seen set**, not only the three priority tasks. The inspected CLI has task-set selection rather than a single-task flag. Separate `--server`/`--client` processes are preferable for supervision. The client catches exceptions and skips existing `stats.json`; add explicit all-attempt accounting before interpreting summary success. Match `split=pretrain` instead when reproducing the Human300-only leaderboard regime [S6].

## 5. CheckVLA availability and comparability

Primary paper: **CheckVLA**, arXiv **2607.26789**, v1 submitted 2026-07-29 [S10]. The inspected paper describes π0.5 with separately trained decoupled experts, a frozen V-JEPA2-AC encoder, a trained action-conditioned predictor/risk head and memory. It is **not merely an evaluation-time wrapper around a stock public π0.5 checkpoint**.

Paper-reported context: 36.1% success versus 27.6% periodic baseline; 10 Hz control with illustrative chunk H50; full-system latency p50/p95 286/338 ms; 23.3 GB VRAM; added 88.4M trainable parameters and 16.9 GFLOPs/step; monitor latency12.1/16.4 ms; 1.18× wall time. These numbers describe that paper's setup and cannot be transferred to the native 20 Hz policies above or treated as matched local baselines.

**Bounded release finding:** no official executable repository, checkpoint or project/release URL was identified in the inspected paper links, web results, GitHub repository search or HF model search. Saved GitHub search returned only unrelated `bhoerl/checkvlans` and `Ed-HR/CheckVlabsExcels`; HF model search returned `[]`. This is an as-of-date search result, not proof that no release can exist or appear later [S10, S11].

To become runnable, CheckVLA still needs a discoverable official code/weight release, exact expert and monitor checkpoints, package/simulator manifest, native observation/action/camera interface, task/split/training mapping and hardware identity for its timing claims. Reimplementing and training those components would be a separate project. For now, use the paper to motivate/contrast intervention timing and verification, with training/backbone/task differences stated; do not claim a reproduced CheckVLA comparison.

## 6. Concrete readiness recommendation and remaining blockers

| Readiness layer | R0 finding |
|---|---|
| Native public checkpoint and evaluation code | Established for Xiaomi, Awomo and official baseline families |
| Published success on all three proposed composites | Established for Xiaomi/Awomo; qualified regime-level GR00T evidence |
| Exact native interface sufficient to plan a port | Audited in detail for Xiaomi, Awomo-I and GR00T; V requires further audit |
| Local environment, renderer, model load, VRAM and latency | Untested |
| Actor-only full native completion on frozen local protocol | Untested |
| Repair, reverification and policy-history handoff | Requires engineering and validation |
| Runnable released CheckVLA baseline | Not verified |

Recommended next decision: freeze **one** Xiaomi or Awomo actor-only qualification manifest when resources are allocated. Specify source/checkpoint/base-model hashes, simulator/assets/controller, split/kitchens, instruction, cameras/preprocessing, normalization, control frequency, action chunk, sampler, seeds, task list, episode count, full horizons and wall/query budgets **before viewing new outcomes**. First reproduce the actor's native execution cadence; early-replanning changes belong to a later matched experiment.

Qualification should retain every attempted episode and distinguish infrastructure failure, native failure and full success. Record constituent attainment/loss, actual action provenance, query/action counts, timings and videos. Obtain full-goal successes and informative partial-progress failures under a declared eligibility rule before attempting repair efficacy. Prefer Steam/Leftovers as likely informative composition/progress-preservation cases on the published evidence, while retaining coffee results and avoiding post-hoc seed/task substitution.

Only then port the five-layer framework. Snapshot simulator/controller, policy history, action queue and RNG together; restore equivalent branch state; explicitly verify actual nominal-policy handoff and downstream completion. Report which components are reused, calibrated or retrained. Compare each actor with its own matched controls. Three tasks do not constitute a 50-task benchmark, and two different model/platform pairs do not isolate model from platform effects.

**R0 outcome:** availability blocker removed; runtime/competence/integration blockers remain. This supports later bounded RoboCasa qualification while L1–L3 OpenVLA/LIBERO mechanism work remains the immediate priority.

## Primary-source index

All links inspected for this audit on 2026-10-19; pinned links take precedence over moving branch documentation. Local evidence filenames are under the scratch directory given above.

- **[S1] Benchmark and leaderboard:** https://robocasa.ai/ ; https://robocasa.ai/leaderboard.html (including submission/version/training details).
- **[S2] RoboCasa365 paper:** https://arxiv.org/html/2603.04356v1 (training regimes and Table11).
- **[S3] Simulator/controller:** [task registry](https://github.com/robocasa/robocasa/blob/6f90063c0ec2b9916bbd977368b54c98d9c2d908/robocasa/utils/dataset_registry.py), [environment utilities](https://github.com/robocasa/robocasa/blob/6f90063c0ec2b9916bbd977368b54c98d9c2d908/robocasa/utils/env_utils.py), [correct gym wrapper at reproduction pin](https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/robocasa/wrappers/gym_wrapper.py), [PandaOmron controller](https://github.com/ARISE-Initiative/robosuite/blob/5ce6643f3092639d08f7b0f90ed1c6a84f50552c/robosuite/controllers/config/robots/default_pandaomron.json). Exact task sources and installation source are indexed in `sources_manifest.json` as `prepare_coffee.py`, `steam_in_microwave.py`, `store_leftovers_in_bowl.py`, `robocasa_installation.md`.
- **[S4] Awomo:** [pinned evaluator README](https://github.com/Awomo-WestlakeDI/Awomo-0.5/blob/d22dda9aae0f9ae8096e95580a140a784a17b513/eval_robocasa365/README.md), [I summary](https://github.com/Awomo-WestlakeDI/Awomo-0.5/blob/d22dda9aae0f9ae8096e95580a140a784a17b513/eval_robocasa365/awomo_05i/summary.json), [V summary](https://github.com/Awomo-WestlakeDI/Awomo-0.5/blob/d22dda9aae0f9ae8096e95580a140a784a17b513/eval_robocasa365/awomo_05v/summary.json), [I checkpoint](https://huggingface.co/Auwomo/Awomo-0.5I-RoboCasa365/tree/d02e3560a9e5e649bb0718a97b9e4b923df59594), [V checkpoint](https://huggingface.co/Auwomo/Awomo-0.5V-RoboCasa365/tree/947121b8a356753fec60489f744d40f71b44f447). Local `awomo_common.py`, `awomo_policy.py`, `awomo_config.json`, `awomo_deployment.md`, `awomo_requirements.txt` have pinned URLs/hashes in the manifest; base-model inventory is in `checkpoint_inventory.json`.
- **[S5] Xiaomi:** [pinned native evaluator](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/blob/0dd7aef8dc87296246aae812a1f59ccb708e5546/eval_robocasa365/entry.py), [detailed summary](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/blob/0dd7aef8dc87296246aae812a1f59ccb708e5546/eval_robocasa365/summary.json), [deployment guide](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/blob/0dd7aef8dc87296246aae812a1f59ccb708e5546/docs/DEPLOYMENT.md), [checkpoint](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365/tree/3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4). Local `xiaomi_*` evidence is indexed in the manifest.
- **[S6] Official checkpoints / GR00T:** [pinned bundle](https://huggingface.co/robocasa/robocasa365_checkpoints/tree/c484448aba1a9b60a04c9b0ca117241518ea69f3), [data config](https://github.com/robocasa-benchmark/Isaac-GR00T/blob/9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10/gr00t/experiment/data_config.py), [evaluation CLI](https://github.com/robocasa-benchmark/Isaac-GR00T/blob/9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10/scripts/run_eval.py). Local `gr00t_target_metadata.json` records the target checkpoint's modality/normalization metadata.
- **[S7] Openpi:** [native policy adapter](https://github.com/robocasa-benchmark/openpi/blob/5a6beda9ff99da30b4e1b59320f6a32971d7c397/src/openpi/policies/robocasa_policy.py), [evaluation client](https://github.com/robocasa-benchmark/openpi/blob/5a6beda9ff99da30b4e1b59320f6a32971d7c397/examples/robocasa/main.py), official checkpoints [S6].
- **[S8] SmolVLA / LeRobot:** [RoboCasa documentation](https://huggingface.co/docs/lerobot/en/robocasa), [pinned checkpoint training config](https://huggingface.co/lerobot/smolvla_robocasa/blob/f7af588d11e30b0069f200792fb285904b74c80b/train_config.json), [dataset metadata](https://huggingface.co/datasets/lerobot/robocasa_target_human_unified/blob/main/meta/info.json). The moving dataset metadata was saved as `smolvla_dataset_info.json` with SHA256 in the manifest.
- **[S9] Official algorithm resource guidance:** https://github.com/robocasa/robocasa/blob/main/docs/benchmarking/policy_learning_algorithms.md . Generic resource recommendations are distinguished from measurements above.
- **[S10] CheckVLA:** https://arxiv.org/abs/2607.26789 ; https://arxiv.org/html/2607.26789v1 ; saved `checkvla.html`.
- **[S11] Bounded release searches:** https://api.github.com/search/repositories?q=CheckVLA ; https://huggingface.co/api/models?search=CheckVLA ; saved `checkvla_github_search.json` and `checkvla_hf_search.json`.
