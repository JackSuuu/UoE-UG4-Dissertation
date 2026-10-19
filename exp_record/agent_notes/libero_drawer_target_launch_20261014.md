# D1 implementation and execution record — 14 Oct 2026

## Specific task

Implement and test the locked five-layer framework's **release → clearance → actor handoff → remaining drawer closing** mechanism. [Frozen protocol](libero_drawer_target_protocol_20261014.md). The book/caddy result motivates this mechanism; it does not establish that the new drawer task has the same failure. Therefore the supervisor first checks three registered development baselines for supported-placement coverage and at least one post-event failure before launching all matched controls on all three roots.

Task-specific measured-state rules make this a privileged **geometric physical guardrail**, not a learned-verifier or prediction-efficacy experiment. All actor calls remain trueH1/k1with canonical language. `gate_hold` means pose suspension plus open gripper; it is explicitly not same-frame replanning or a gate with no control action. `timed_repair` provides a simple same-primitive comparator; `phase_repair` provides state-conditioned release/clearance timing and handoff.

## Implemented interfaces

- `src/adapters/libero_drawer.py`: geometry monitor for40bowl collision boxes and five gripper collision geometries, milestone/history tracker, dual-mode gate, bounded hold/open and outward/open repair, causal `decision_trace` builder.
- `scripts/run_libero_drawer.py`: frozen actor integration, one proposal per actual interval, recorded evidence/action chains, raw25substep contact logs, complete-task stopping and video.
- `scripts/launch_libero_drawer.py`: detached healthy-GPU supervisor, registered target-headroom stop, full matched-arm launch, verification/video/HTML and local recovery archive.
- `scripts/summarize_libero_drawer.py`: replay **all** actual actions from their original roots without actor inference; check integration/controller/image identity, raw contacts, geometry, state/action/trace causality and budgets; shared-prefix tokens/proposals; streaming four-panel video and clickable decision timeline.
- `scripts/smoke_libero_drawer.py` and `src/tests/test_libero_drawer.py`: geometry/measurement integrity and event/gate/trace boundary checks.
- Backwards-compatible optional task selection in `libero_runtime.make_env` and optional contact keyword filter in `ContactRecorder`; historical defaults remain book/caddy.

## Geometry and acceptance findings

The original native containment predicate tests a point rather than the collision footprint. Native close is drawer q>0. Compiled bowl geometry is40boxes; drawer collision-wall inner faces, floor and shelf are measured in the moving bottom-drawer frame. A real-simulator synthetic fixture reaches the registered complete released/closed task dwell. Moving it laterally supplies a native-point-inside/full-footprint-outside counterexample that the physical metric rejects. These fixtures are diagnostic, not task outcomes or data for learning.

The gripper uses three compiled meshes and two pad boxes. All compiled collision vertices contribute to clearance. No simplistic upward primitive is used under the shelf. The geometric outward heuristic is bounded but not certified collision-free. Complete-task release requires no gripper contact with either bowl or cabinet, not only native drawer closure. Collision/CVR/safe-success metrics remain null until independently justified.

## Observable decision chain

Every interval has a structured, reproducible record:

`evidence through t−1 → subgoal/milestones → supported/released/clear predicates + numeric thresholds/dwell → gate mode/route/reason/budget → frozen proposal versus actual command → separately labeled observed result after t → handoff evidence`.

`decision_trace.jsonl` contains the complete chain. The exported video overlays phase, support/contact, entry/release/clearance dwell, clearance margin, actual gate route/reason, action-change flag, measured retention/closure and full-task dwell. `review/root_<i>_trace.html` exposes expandable raw proofs and clickable frame-linked events; no post-hoc text model invents explanations. Causal trace reconstruction is part of exact action replay, not merely decorative logging.

## Prelaunch validation

Sixteen unit tests cover the new gate/trace behavior plus existing preservation/completion/runtime behavior. Real-simulator smoke passes: three successive environments produce exact integration/controller/camera histories with and without instrumentation; measurement is read-only; positive settled full-task fixture and negative partial-footprint fixture pass. An initial12short synthetic-noop episode pipeline test exercised all four arms across three roots, exact replay, three videos and HTML timelines. The final trace-label revision also passed a four-arm fixture, full replay/trace reconstruction and streaming review export; final source/protocol hashes match the current files. These are software tests, not OpenVLA capability results. Evidence: `backups/libero_drawer_smoke_20261014.json`, `backups/libero_drawer_pipeline_smoke_20261014.json`, `backups/libero_drawer_trace_smoke_20261014.json`.

The first smoke caught an environment-wrapper attribute access (`env.env.object_states_dict` is required) and one unit assertion needed floating-point tolerance for0.005/0.05. Both were corrected before any actor experiment. Neither failure changed physical thresholds or used actor outcomes.

## Planned run

Output: `/home/ami/scratch/libero_drawer_target_20261014_a`. Protocol and source are committed/pushed before launch. Healthy physical GPUs1/2/3map to task roots0/1/2. Phase1runs three actual OpenVLA baselines. Phase2runs the remaining three methods on all roots only when registered target headroom exists. All source hashes and the protocol hash are recorded. Persistent supervisor and worker logs survive SSH termination. Automated archive path is `backups/libero_drawer_target_20261014_a` after completed verification; remote LFS recovery must then be independently verified.

No efficacy or completion counts are available in this prelaunch record. Record actual launch IDs and verified outcomes in subsequent updates rather than replacing the frozen protocol.

## Launch update

Implementation/protocol/preflight commit **`d3889e2` was pushed before launch**. Detached D1 supervisor started successfully in `/home/ami/scratch/libero_drawer_target_20261014_a`, PID779352. Baseline workers: GPU1/root0/PID779356; GPU2/root1/PID779357; GPU3/root2/PID779358. Initial status is `baseline_target_coverage`; no outcome claim yet. Supervisor log is `/home/ami/scratch/libero_drawer_target_20261014_a_supervisor.log`. Frozen protocol SHA256: `3f7e2784a70f916e11ff051eb9009bdcb4998f85194da62d27331557e60bd827`.

**Completed:** [verified results](libero_drawer_target_results_20261014.md). Baselines complete3/3under full physical acceptance; the registered no-post-event-headroom stop prevented the nine intervention episodes. All1,276actions and observable proofs replay-verified; three single-VLA review videos/trace pages and recovery archive generated. No framework-efficacy claim.
