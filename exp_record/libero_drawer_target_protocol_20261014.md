# Target experiment D1: release → clearance → actor handoff → drawer closing

**Status: frozen before actor rollouts. Development mechanism test, not final efficacy.**

## Research question and fixed scope

Within the locked five layers, does a measured release/clearance-conditioned local controller improve **the remaining task, including drawer closing**, compared with the same frozen actor, a hold/open gate control and a time-based repair using the same primitives?

Motivation is specific: the completed book/caddy diagnostic rescued placement but lost it after actor handoff, and accurate lookahead did not beat simple hold. D1 tests a **phase-aware handoff mechanism**, not a new world model, learned verifier, general architecture search or a prediction-specific benefit. The task is `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` (pinned `libero_10`). Its drawer starts open. All steps use the unchanged canonical instruction and frozen cached LIBERO-10 OpenVLA checkpoint.

The five interfaces are now executable: shared milestone tracker; frozen actor; privileged geometry/contact/history monitor; nominal/contact-transition gate; bounded hold/release and outward-retract primitives. D1 implements the **release-to-close transition slice** of this architecture. Alignment/stagnation are recorded for stage diagnosis but do not trigger another algorithm. Do not claim every failure phase is repaired.

## Cohort and stopping rule

- Task-specific official initial states **0,1,2**, all development. They are distinct task states from book/caddy indices0–5. No new noise/dynamics sweep.
- Run the three VLA episodes first. If **none reaches the registered supported-placement event**, stop D1 as `target_unreachable`: this actor/task does not expose the hypothesized transition under this trigger. Do not launch ineffective duplicate arms or relax thresholds post hoc.
- If the event is reached but **every event-reaching baseline already completes the full task**, stop as `no_observed_post_event_headroom`: all observed failures, if any, precede this intervention. Do not manufacture difficulty. These are development go/no-go criteria, not unbiased held-out efficacy sampling.
- Only if at least one baseline reaches the event and fails full completion, run every remaining arm on **all three roots**, including non-triggering/successful roots. Do not select just the rescuable roots. At most12full episodes.
- Further roots3–5are unassigned;6–9reserved for future calibration;10–19future training only if separately justified;20–49reserved confirmation. This D1 run consumes only0–2. Synthetic smoke fixtures are diagnostics, never training/test trajectories.
- Same maximum600actual20Hz control intervals,10initial settling intervals, one nativeH1query per interval and at most40intervention intervals. All arms stop only at the first20consecutive whole-task completion intervals or the600cap. No extra execution budget for repairs.

## Matched arms and attribution

| Arm | One supported-placement event → bounded transition control |
|---|---|
| `vla` | Track/log events but execute every frozen proposal unchanged. |
| `gate_hold` | Suspend pose motion and command open; return on stable clearance, placement loss, or40interval exhaustion. This is a **hold/open veto control with no spatial repair**, not a pure replan or an action-free gate. |
| `timed_repair` | Same event;10hold/open intervals, then30outward/open intervals, independent of release/clearance. Placement loss aborts. Hand back on40interval exhaustion. |
| `phase_repair` | Same event; hold/open until3released intervals; retract outward only while not currently clear; hold/open while accumulating3clear intervals; hand back on stable clearance, placement loss, or40interval exhaustion. |

Each actor is queried on the actual current image every interval, including intervention intervals. A requery on the same frame would be identical and is not used. **H1/k1, no genuine chunks, no replanning-efficacy claim.** All arms have the same state sensing/logging, nominal actor, observation cadence, query cap and termination. Zero simulated candidate/forecast queries; the only change is routing and bounded pose/gripper commands. Actual query/intervention counts are reported. Canonical instructions are never rewritten by milestones.

`phase_repair` versus `gate_hold` tests added spatial repair plus its handoff. Versus `timed_repair` tests the combined state-conditioned release timing/exit mechanism with the same primitives and cap; it does not isolate hysteresis alone. A gain versus VLA alone is insufficient. A hold-only gain is credited to hold/release. A phase-controller failure is recorded, not followed by a parameter sweep.

## Causal event and bounded actions

At decision t only completed intervals up to t−1are used. Supported placement requires the full collision-box footprint within drawer inner faces, bottom within5mm of the floor upper face, final-substep bowl/floor contact, relative translational speed≤.02m/s and bowl angular speed≤.1rad/s for3consecutive intervals. It does **not** require successful detachment first. The initial unmeasured contact state cannot trigger.

Release condition: retained placement with no bowl/gripper contact anywhere in each of3successive25-substep intervals. Clearance condition: retained placement, no bowl/gripper or cabinet/gripper contacts during each interval, plus **all compiled gripper collision vertices** either10mm in front of the drawer front outer face or10mm above the highest bowl collision vertex, for3intervals. Already-clear cases hand back without gratuitous retracting.

Hold/open is zero6Dpose command and gripper−1. Outward/open is a5mm requested translation along the drawer frame's negative y direction, zero rotation and gripper−1. Native OSC scaling is validated: translation5cm per unit, so vector norm0.1. It is a bounded geometric heuristic, not a collision-free planned motion; the physical contact results must be observed. No upward-motion command is introduced under the cabinet shelf. At most30outward intervals for timed repair; phase repair may use up to40total hold/retract intervals, with5mm per retract. There is one transaction, no retrigger. Placement loss or exhausted budget produces an explicit unsuccessful handoff; it is never counted as stable clearance. Subsequent motion, including drawer closing, is real frozen VLA execution.

The10+30schedule,3interval debounce,10mm clearance and40step cap are registered engineering choices, not thresholds tuned on D1actor outcomes or validated force-safety limits. Existing controller scale and collision geometry motivate these bounded values. D1may reject them.

## Physical task measurement audit

Native `In` approximates the bowl by its reference point in a site AABB; native `Close` is bottom drawer joint q>0. These are separately reported. Full bowl collision geometry comprises40boxes, not a transplanted book box. Compiled inner bounds in bottom drawer coordinates:

- lower = (−0.0992349854, −0.0648300000,0.0208300000)m;
- upper = (0.1062649854,0.0872100000,0.0781723717)m.

Side/front/back bounds are the conservative axis-aligned inner faces of the modeled drawer wall boxes; the upper z face is the lower surface of the cabinet shelf. The drawer has pure local-ytranslation. All40box corners are transformed to this moving drawer frame. The real-simulator smoke confirmed a physically settled whole-bowl placement can satisfy all conditions and that native point-inside can coexist with an invalid full footprint.

**Whole-task physical completion:** native conjunction; full footprint inside; bottom seated within5mm; final-substep floor contact; all bowl vertices between floor−1mm and shelf+1mm; no gripper contact with bowl **or cabinet** across all25substeps; maximum bowl/cabinet penetration≤1mm; bowl linear speed≤.02m/s, angular speed≤.1rad/s, drawer speed≤.01m/s; all for20consecutive intervals. This is an explicitly registered development acceptance definition, not a certified safety predicate. Required absence of cabinet/gripper contact means pushing against a closed drawer indefinitely does not count as released completion. Native success is also retained as its own metric.

## Outcomes and decision rule

Primary: full physical task completion and native-ever SR on all three paired roots. Secondary: stage milestones, supported-placement coverage, stable versus exhausted handoff, post-handoff bowl re-contact intervals, native-to-physical acceptance gaps, interventions, control steps and actual policy calls. Annotated four-method2xsim-time videos are mandatory.

Report paired rescue/harm against **both** hold and timed repair, along with descriptive uncertainty; three developer roots do not establish significance or generalization. If phase repair improves closing/completion beyond both controls without offsetting paired harm, that supports larger frozen-method development/confirmation planning. If controls tie/win, report it and use the identified transition failure to choose at most one next layer-internal edit. If failures precede the supported event, report lack of target coverage and audit the earlier phase; do not call the intervention successful based on fewer contact samples.

Contacts are raw physical diagnostics. Necessary floor support, grasp and drawer-handle contact are not automatically collisions. No validated contact-pair/phase collision metric or converged force thresholds yet: **Collision Rate/CVR/safe success remain null**. No learned model is trained, no final-test roots are used, and privileged geometry/contact are not advertised as image-derived perception.

## Integrity, execution and preservation

**Observable decision chain (user requirement, before launch):** every interval emits `decision_trace.jsonl`: measured evidence through t−1→ subgoal/milestones → numeric entry/release/clearance predicates with thresholds and consecutive counts → actual gate mode/route/reason/budget → nominal versus executed pose/gripper → separately labeled observed outcome after action t → handoff evidence. The same deterministic trace builder is checked during action replay. No post-hoc language-model justification or fictitious forecast/candidate search. Annotated video and clickable HTML event timelines expose this chain, including nominal passes, exhausted budgets, lost placement and unsuccessful interventions. Raw traces retain every interval; the visible timeline highlights changes and intervention intervals.

Healthy physical GPUs1/2/3only, numeric CUDA/EGL IDs matched. Supervisor/workers detached with persistent logs. Source/protocol/task/checkpoint fingerprints are recorded. Every episode records pre-action integration/controller/camera hashes, proposals, actual actions, modes/reasons, raw25substep contacts, milestones, states and video.

Before launch: unit checks for causal entry, release→retract→handoff, no retrigger, bounded budget/placement-loss exit, control-arm distinction and contact identity; real-simulator instrumented/uninstrumented exact replay and geometry-positive/negative fixtures. After execution: replay all actual actions from original roots without loading the actor, recompute measured geometry/contact/milestones/actions, check saved state/controller/camera equality, action decode, termination/budgets and exact common prefixes through the first changed action. Only then render comparisons and archive. Source changes require a new version/run; do not mutate the running protocol.
