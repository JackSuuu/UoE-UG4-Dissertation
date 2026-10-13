# Local LIBERO/OpenVLA integration — 13 Oct 2026

## Implemented and checked

- Added `src/adapters/libero_runtime.py`: exact book/caddy task, official initial states, project-local LIBERO config, Panda OSC environment and two process-local MuJoCo3.14/robosuite1.4.1 ABI adaptations.
- Added `src/adapters/libero_openvla.py`: strict full task-checkpoint load (7.54B parameters,32LLM layers), native seven-token greedy decoding, suite statistics, official image preprocessing and gripper mapping. No planar head is used in this pilot.
- Added `scripts/smoke_libero.py`, `scripts/run_libero_baseline.py` and `scripts/summarize_libero_baseline.py`. The latter checks artifacts, token/action mapping, state/frame counts and emits a local HTML video report with all three outcomes.
- Environment smoke succeeded; full-model one-action smoke succeeded. Official preprocessing parity passed exactly on an actual simulator image and a fixed random image, using AST-extracted upstream helper functions. Reference hashes/results at `/home/ami/scratch/openvla_libero_reference/preprocess_parity.json`.
-12unittest checks passed:2new simulator ABI checks,5existing chunk-cadence checks and5R5boundary checks. Standalone `test_tf5_compat.py` and `test_action_map.py` also passed after dependency setup. These establish interface checks, not official TF4.40 decoding or old-versus-new simulator dynamics parity.
- Dependency installation evidence and the temporary NumPy metadata/version correction are recorded in the frozen [pilot protocol](libero_baseline_protocol_20261013.md). Active core package versions restored/preserved; no site-package source edits. Failed integration artifacts are retained.

## Full-episode pilot launch

Protocol and runner committed/pushed **before full-episode launch**, commit `bf7c6c6`. Run root `/home/ami/scratch/libero_book_pilot_20261013_a`, worker roots0/1/2, physical GPUs1/2/3, PIDs688724/688725/688726. Each process uses `setsid nohup`, detached stdin and its own log. At this entry, the full pilot has been launched; completion is not yet claimed. The final verifier will write `verified_summary.json` and `index.html` after all worker manifests complete. No learned-verifier/actor training or physics-intervention rollout has started.

Artifacts include compiled simulator XML and policy-visible input images, allowing subsequent geometry/perception audits. Video is rendered at20fps simulated time; it is not a real-time inference/latency demonstration. Browser preview was unavailable because this session had no connected desktop browser; artifacts remain on disk.

## Mechanical observations for metric design

Actual compiled model audit saved as `geometry_audit.json` in the pilot directory. Book collision geometry is one rigid box, full dimensions28.68×110.38×134.40mm, offset from the body's reference point; caddy uses17collision boxes, including angled wall sections. Thus checking just body position inside a site is insufficient to establish full insertion/containment. The region's half-sizes are27.75×62.16×60.46mm, but they are not physical clearance. Original timestep2ms, solver enum2,100iterations; contact `solref=[0.001,1]` appears in object geometry. Force/impact acceptance must account for actual effective solver settings and numerical convergence, not convert a raw endpoint wrench into a safety guarantee.

Next after pilot: inspect full trajectories, confirm grasp/place/release behaviour and the dominant failure, then specify geometry-aware completion/dwell and substep force/impact measurements. Preserve native predicate results alongside stricter metrics. Establish bounded accurate-physics intervention headroom under matched execution opportunities before another learned training run. The three pilot roots remain development-only.
