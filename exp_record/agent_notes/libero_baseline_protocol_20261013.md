# LIBERO book/caddy native-action feasibility pilot — registered before full rollouts

## Scope and fixed budget

Task: `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` in LIBERO-10. LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`. Full task-finetuned OpenVLA-7B checkpoint `80970322773f81baa2e22fe495d0487b93a05cfa`; all weights loaded with strict key matching, BF16, frozen. Task-specific statistics from `dataset_statistics.json`; base checkpoint supplies remote architecture code, not task weights.

Three **development/feasibility** initial-state indices:0,1,2 from the official `.pruned_init`. These roots are consumed for integration/design, not independent confirmation or future final test. One worker per healthy physical GPU1/2/3, batch1. Up to520 policy steps per episode,20Hz simulated control,10initial zero-pose/open-gripper settling steps, environment seed0 and inference seed0. Stop at first native success or environment termination. No extra attempts, noise tuning, training or verifier interventions in this pilot. Numerical/action-interface corrections require a separately labeled rerun with the earlier artifact preserved.

This is native **H1/k1** feasibility, not the existing H10/k5 efficacy experiment. Do not repeat one native action into a purported H10 proposal. Before formal verifier evaluation, proposal/controller compatibility and a separately frozen matched protocol must be resolved.

## Interface and observations

Panda `OSC_POSE`,7D pose/gripper. Policy consumes only the agent-view image plus language. Rotate image180°, official TensorFlow JPEG round-trip/Lanczos3 antialiased resize to224,90%-area center crop using `crop_and_resize`, official processor. The port matched the official helper functions pixel-for-pixel on one actual simulator frame and one fixed random image. Official gripper normalization/binarization/inversion follows action unnormalization; pose commands are passed directly as the suite-trained controller inputs, never reinterpreted as the old planar velocities.

Use the existing explicit uncached multimodal7-token greedy decoder because TF5 `generate` was previously found incompatible. Prompt terminator and token-to-bin/unnormalization follow the native head. This establishes a runnable local decoding path, **not parity with the official transformers4.40.1 inference stack**. A poor pilot cannot establish that the official checkpoint lacks capability before decoding/simulator parity is investigated.

## Outputs and limits

Persist every episode's video, initial/raw-policy-input images, action tokens, native and mapped actions, native predicate at every step, book pose, EEF position, endpoint contact wrench/distance/geom pairs, simulation states, compiled XML and result. Run manifests include source hashes, git revisions, versions and actor-build information. Privileged simulator state/contact is logging-only; it does not enter the actor.

Report native completion count/SR for all three selected roots. Three episodes cannot support an efficacy/generalization claim. `CVR`, safe-success and stable precision-success remain explicitly **null**, because thresholds/geometry acceptance have not been validated. Endpoint contacts are not substep force peaks and cannot certify safety.

The current `In` predicate checks contact with the caddy and the book's reference position inside an expanded region (lower Z extended by1cm). It does not require release, complete geometry containment, an orientation tolerance or dwell. The back-region site has half-sizes approximately27.75×62.16×60.46mm in the checked XML; this is a predicate region, not measured insertion clearance. Next metric work must use actual collision geometry, release/stability and substep contacts before freezing precision/safety acceptance.

## Integration evidence before full rollout

- Environment reset/render/ten settling steps passed in `/home/ami/scratch/libero_smoke_d`.
- Full7B strict loading and one native action passed in `/home/ami/scratch/libero_native_action_smoke_c`; this truncated action smoke is not counted among the three complete pilot episodes.
- Two robosuite1.4.1/MuJoCo3.14 Python ABI adapters: integer joint-type address lookup (enum/NumPy reverse equality broke tuple membership), and the new `mj_fullM(model,data,dst)` signature. Both are process-local and covered by address-type/mass-matrix tests; they do not establish old-versus-new simulator dynamics parity.
- Package installation added robosuite1.4.1, gym0.25.2, bddl1.0.1, easydict1.13, future1.0.0, tensorflow-cpu2.21.0 and missing dependencies. The initial constraints export encountered stale duplicate NumPy metadata and temporarily installed2.3.3; it was immediately restored to the pre-existing active2.3.4 before any successful simulation/model smoke. Active torch2.14.0, transformers5.17.0 and MuJoCo3.14.0 were preserved. Preserve install reports and before-state at `/home/ami/scratch/libero_*`.
- Failed smoke artifacts `_a/_b` and earlier env smokes remain preserved. TF5 task repo uses base-repo remote code; robosuite EGL requires numeric GPU IDs rather than UUID strings. Physical GPU identity was checked before using numeric1/2/3; physical0 remains excluded.

After nominal feasibility, inspect complete trajectories and decide the smallest physical-intervention diagnostic. No learned training or final safety threshold is authorized by this protocol.
