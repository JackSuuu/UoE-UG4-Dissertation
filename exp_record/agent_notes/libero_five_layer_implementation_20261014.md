# Five-layer implementation and long-horizon handoff — 14 Oct 2026

Authority: the [user-locked five-layer architecture](libero_framework_design_20261014.md#authoritative-architecture-lock--user-decision-14-oct-2026). Purpose: determine concrete edits before launching the next task, with one architecture and a small sequential implementation plan.

## 1. Interfaces to implement

The existing planar `src/interfaces.py` remains a historical H10reference; its force/deformation channels and2Drepair must not be silently repurposed as Panda contact events. Add a robot-specific integration layer around the working native adapter:

| Component | Proposed interface | Required trace |
|---|---|---|
| Subgoal tracker | `update(observation_history) -> phase, completed_milestones, canonical_instruction` | Phase entry/exit and observed evidence; task completion separately from milestone completion |
| Frozen actor | `propose(image, canonical_instruction) -> action_or_genuine_chunk` | Checkpoint/code hashes, action units, actual H, proposal and executed action |
| Event monitor | `observe(history, phase, proposal) -> event_flags, margins, progress, uncertainty, provenance` | Contact onset/release, unexpected re-contact, clearance/alignment, stagnation trend; missing measurements remain missing |
| Dual-mode gate | `decide(monitor_output, gate_state, budget) -> mode, route, reason` | Nominal/contact-transition mode, continue/replan/repair route, hysteresis/dwell, budget remaining |
| Repair controller | `correct(phase, observation_history, proposal, budget) -> bounded_action, exit_evidence` | Primitive, pose/gripper delta, bounds, exit condition and post-handoff response |

Observation provenance is part of the contract. Simulator object poses, collision lists and joint state may support a **privileged mechanism variant**. An image/proprioception-only claim requires an actual estimator path and equal access for the replan-only and simple-control comparators. Hidden mass/friction and future outcome labels do not enter runtime features. Do not train a new estimator/actor/repair simultaneously to mask attribution.

## 2. Concrete edits within layers3–5

### Monitor beyond "already placed"

The current `PreservationWindow` only fires after all placement conditions are true. Its inability to intervene in an earlier alignment failure is structural, not a missed numerical threshold. The new monitor must track **phase-conditioned error/progress history**, rather than changing the old frozen trigger or interpreting stationary support as a hazard:

- transport/align: target-relative pose, approach direction, decreasing clearance and alignment-error trend;
- insertion/contact: expected versus unexpected contact, motion response to commands, persistent lack of task progress;
- release: loss of grasp contact plus supported/retained object state;
- clearance/handoff: EEF separation and renewed approach toward an already placed object;
- drawer closing: motion of the drawer, bowl retention, obstruction and gripper/object separation.

Use a small finite history window and interpretable state variables first. Exact numeric thresholds/window lengths are not frozen here: derive them from task geometry, controller scale and separate calibration data, then register them before the online comparison. The current root0–5audit results are consumed development evidence, not training/test samples to recycle.

### Persistent dual-mode routing

Nominal mode executes the actor and handles ordinary milestone-boundary replan requests. Contact/transition mode owns a bounded correction interval and checks return conditions. Entry is causal and logged; exit requires stable task-compatible geometry/contact evidence, not merely the first quiet timestep or expiry of a blind timer. Exhausted repair budget produces an explicit failed-recovery/handoff record under a frozen rule, not hidden extra control time.

Reaching a placement dwell should advance to clearance/next-subgoal logic before indiscriminate actor resumption. Nevertheless, all arms must share whole-task termination and milestone rules; a gain due solely to improved stopping is reported as such. The existing40-step resumption stress test remains a separate historical diagnostic. In the drawer task, success requires completing the remaining closing stage, not stopping after bowl placement.

### Bounded repair primitives

- **Retract:** move along a geometrically defined feasible separation direction with a fixed translation/rotation/gripper budget; upward motion alone is not universally safe.
- **Re-align:** correct task-relative lateral/orientation error within bounded increments, conditional on a compatible contact mode. Unload first when direct correction would worsen contact.
- **Slow approach:** decrease pose increments along the approach direction while checking progress/contact feedback; necessary task contact remains allowed.

Prioritize deterministic geometric controllers. A small residual model is optional only after the hand-designed controller's remaining failure is identified. No complex RL repair training, larger model or candidate-count sweep is queued. Keep an unchanged-proposal option and separately log interventions that do not alter the action.

## 3. Next task and prerequisite audit

Pinned source references:

- `third_party/LIBERO/libero/libero/benchmark/libero_suite_task_map.py`, `libero_10`, task index3in the unpermuted map.
- `third_party/LIBERO/libero/libero/bddl_files/libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.bddl`.
- BDDL initial predicate: `(Open white_cabinet_1_bottom_region)`.
- Terminal predicates: `(Close white_cabinet_1_bottom_region)` and `(In akita_black_bowl_1 white_cabinet_1_bottom_region)` jointly.

Use the exact task string/path in manifests rather than relying on task-index permutation. The full LIBERO-10checkpoint is already cached; its actual success on this task is **not yet established**. Audit the cabinet joint/closed predicate, bowl collision geometry versus reference-point containment, support/release and retaining the bowl while closing. This audit must precede naming native success as complete manipulation or assigning collision/safety labels. Do not transplant the rectangular-book metric directly to a bowl.

Milestones: acquire bowl → transport/align at the already open bottom drawer → insert and release → clear the gripper → close drawer → verify closed drawer and retained bowl. Detect opening/re-opening as a recovery condition if it arises, rather than imposing an unnecessary opening step.

## 4. Resolve proposal/control cadence before the main comparison

The working `LiberoOpenVLA.act` returns one7Daction. It does not offer a genuine open-loop chunk. This blocks the requested **native OpenVLA chunk** label, not the five-layer architecture.

Implementation work must explicitly establish the proposal interface before registering the long-task three-family efficacy study. If a lightweight task-specific chunk head over frozen features is introduced, call it an **adapted frozen-backbone chunk policy**, freeze it before evaluation and use exactly that actor in all arms, including actor-only. Its capability/training data and attribution must be reported; it is not native checkpoint decoding. Actual future observations cannot be used to precompute the baseline's nominally open-loop proposal. An action-repeat baseline must be named action-repeat, not genuineH10.

Preserve historical full7B H10/k5 experiments and microbatch≤8. Any robot H1protocol revision needs its own explicit cadence/query semantics. With a deterministicH1actor called on every new frame, a second "replan" on the same frame is identical and cannot serve as a meaningful scheduling ablation. Do not launch a three-row table with this degeneracy hidden.

## 5. Fair comparison and minimum ablations

Main families: actor-only execution; fixed-frequency/event-only replanning; event-gated local repair. The fixed and event replanners are separately identifiable controls, not interchangeable labels. Shared items: task roots, disturbance schedules, frozen actor/proposals, high-level milestone logic, monitor sensing/estimates, action limits, termination, execution opportunities and maximum policy/repair-query budgets. Log actual inference calls and intervention counts. If event scheduling changes observation/replan opportunities, include the matched-rate control rather than crediting the extra feedback to physical understanding.

Minimum component checks, sequentially after the complete system works:

1. Event-only versus event+repair: benefit/cost of local correction.
2. Fixed-rate versus event-only: value of event timing at matched budgets.
3. Shared simple phase/geometric repair versus the full monitor-driven controller: attribution beyond additional state or termination rules.
4. Remove measured-history/stagnation or phase-aware return **one at a time**, only to test the mechanism implicated by the failure audit. Avoid a combinatorial architecture sweep.

Whole-task SR is primary. Report stage completion, episode-level unintended-collision incidence, collision-event count per simulated second, repair/replan frequency, completion steps, inference costs and paired rescue/harm with root-clustered uncertainty. Freeze an allowed/forbidden contact-pair-by-phase table and event onset/debounce rule before evaluating collision rate. Validate raw contact/impulse numerical sensitivity; force-based CVR/safe success stay null until meaningful limits exist. Lower contact counts alone cannot establish safe manipulation.

Nominal capability first, then held-out initial states and bounded observation/actuation/dynamics perturbations with predefined amplitudes and paired schedules. Training, design, calibration and final test roots must remain separated. A long task offers more transition opportunities but does not guarantee a framework gain. One development task plus one sequential long-task study is not broad zero-shot generalization.

## 6. Thesis language and deliverables

Suggested Introduction framing:

> Robust manipulation requires recognizing when an action approaches a task-relevant physical boundary and selecting a recovery that preserves progress. This dissertation investigates a lightweight execution-time physical guardrail around a frozen vision-language-action policy. The guardrail monitors geometric affordances, contact transitions and observed progress, and gates bounded local corrections during contact-sensitive phases. We evaluate whether this operational form of physical understanding improves complete-task outcomes beyond matched replanning and simple geometric control.

Methodology should show the five-layer diagram, allowed observations, state/event definitions, dual-mode transitions, repair bounds, handoff criteria and matched budgets. Label simulator-state variants as privileged and keep exact-physics lookahead separate from deployable heuristic/learned monitoring. Results figures should combine whole-task/stage outcomes, paired rescue/harm, event/intervention timelines and method-labeled accelerated videos. Negative results and renderer/metric corrections belong in the evidence record, not hidden by a new task name.

Next implementation order: finish/archive the running diagnostic → wire the five-layer interfaces and phase-aware monitor/gate/repair on the existing integration task → audit the bowl/drawer task and frozen actor/proposal capability → freeze the concrete cadence/splits/collision definitions → launch the matched long-task comparison. This document determines edits; it does not claim the new controller, chunk adapter or bowl/drawer experiment has already run.
