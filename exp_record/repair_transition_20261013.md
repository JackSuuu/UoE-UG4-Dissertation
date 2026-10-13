# Actual-verifier repair-transition ablation — registered 13 Oct 2026

## Research purpose: return to learned-framework outcomes

Recent collector studies explained data-coverage failure but did not improve or independently validate the verifier. Their early-brake first-step failures are not automatically the mechanism behind real verifier harms. This experiment tests a concrete, deployable **repair-aware proposal-continuity algorithm** directly on the existing validation benchmark, without another collector sweep or a model-capacity change.

**Premise audit on the original learned-verifier OOD trajectories (448 episodes):** 129 violating episodes, 12 paired safe-success harms, 340 pre-violation decisions immediately after an execution-changing repair. Sixteen first violations fall in that next decision chunk, including **all 12 harms**. Only seven occur on a direct-release chunk (current repair did not change execution), including **four of the 12 harms**; **zero** direct-release first violations occur at the chunk's first step. The other eight harm incidents occur while another repair changes execution. Thus the warranted hypothesis is about **adjacent repaired decision chunks and resulting state/action continuity**, not merely a first-step release jerk. Temporal association is not proof of mechanism.

Prior skip-one diagnosis establishes that interventions can causally produce these harms, but does not tell us which alternative repair/continuation preserves the eight baseline rescues. This ablation must measure both harm reversal and rescue retention. Removing every original harm alone would yield only about +1.79 pp safe success over policy on this cohort, below the broader +3 pp transfer goal; transition handling is a possible component improvement, not a promise to solve the gate bottleneck.

## Exact algorithm and controls

Every controller retains the full frozen OpenVLA actor and original learned repair predictor/rule. At decision t record whether the previous decision's **actual repair** changed any of the first five actions by >1e−7 relative to the proposal that the gate checked. A triggered no-op/abstention does not count. At the next decision only, that flag enables a deterministic prefix transform of the new raw actor proposal u, using the **last actually executed velocity** v. No future risks, hidden physics, success/harm labels, selected episodes or contact-state oracle determine the transform.

For j=1…5, with raw suffix j=6…10 retained:

| Mode | Prefix presented to gate | Purpose |
|---|---|---|
| direct | u_j | Exact old-controller replay |
| **ramp (primary)** | (1−j/5)v + (j/5)u_j | Gradually return to proposed velocity within existing five-step commitment |
| flat | .4v + .6u_j | Constant blend with the same average interpolation weight; not claimed exactly energy/displacement matched |
| pause | 0 | Five-step pause control: tests whether mere delay/less movement explains changes |

The unchanged gate checks **the transformed H10 candidate once**; if triggered, unchanged six-iteration bisection/floor/target/responsiveness acts on that candidate. Nothing is changed after verification. A prefix transform does not itself arm another transform. A subsequent real repair does re-arm it. These are stateless frozen models plus an explicit per-episode controller memory bit, not a larger neural architecture. Interpolation remains within action bounds by convexity; pause retains the same total 80-step episode budget and can lose completion time. H10 scores remain open-loop candidate risk, not continuation value.

**Nine arms, same root episodes:**
1. none: original policy, no gate or transforms;
2–5. learned gate + original learned repair, modes direct/ramp/flat/pause;
6–9. exact GT H10-event gate + the same learned repair, same four modes.

The learned ramp is the fixed primary method; flat/pause are controls and will not silently become replacement winners. GT arms are privileged component diagnostics, never learned efficacy. Their role is to ask whether a useful transition change is masked by the known learned-gate defect. No static/GRU checkpoint is promoted or retrained in this experiment; all six previously failed new heads remain recorded, not best-seed selected.

## Matched information and budget

- Complete OpenVLA-7B + frozen H10 DAgger×2 head, inference microbatch8; 16 decisions, **H10/k5**, 80 executed steps, 32 roots/cell, same cameras/instructions and parameters.
- Original repair: six bisection iterations, floor .05, target min(.8,.75×score), responsiveness .02, abstain enabled, no gradient or hard-prefix repair. **One gate risk query per decision; eight repair risk queries per triggered row**, asserted at runtime. A deterministic transform offers one candidate, not an expanded search set. Different trigger totals are downstream outcomes, not equalized using extra feedback.
- The history bit and previous executed action are available to all transition/direct controls. GT information is used only by declared GT arms or offline labels, never learned-runtime arms.
- Offline audit labels raw/prepared/applied chunks identically in every arm; labels never select actions. Log actual state/history/action traces, scores, trigger/repair/transform flags and first-violation timing. Assert first-five-step label/execution parity, exact initial pairing and unchanged rows until first transformed prefix.
- none, learned_direct and gt_direct must reproduce the previous factorial's episode outcomes and live risk traces (rtol1e−5, atol1e−6). Do not relax tolerances if replay fails.

## Cohort, sharding and immutable sources

Use the existing **20-cell validation grid**, seeds **26000–26019**, 640 episodes/arm (448 OOD, 192 ID). These episodes have informed design and are **not independent confirmation**. No confirmation/final-test labels or episodes are accessed. The completed temporal evaluation and gate×repair factorial define frozen sources. The old audit is diagnostic provenance only and is not training data.

Total **20 cells ×9 arms ×32 episodes = 5,760 trajectories**, only 640 independent paired episode initializations. Three healthy RTX A5000 devices split **cells**, not methods: grid index modulo3 gives 7/7/6 cells and 63/63/54 arm batches. Every worker runs all nine arms for each assigned cell. UUIDs: shard0 `GPU-3f4f72ef-ed7f-2ddb-e454-abcc6c0330b0`, shard1 `GPU-99b39a1b-5fdd-1acb-398a-426672ebc1a8`, shard2 `GPU-c164f9d9-f157-daeb-d7be-5c98029c2a2b`.

Code, protocol, source model/calibration/controller/simulator artifacts and results are hashed. Source files/checkpoints remain unchanged. Each worker uses `setsid nohup ... < /dev/null`, offline loading and thread caps. Save per-arm raw data and per-cell partial summaries atomically. The last completed worker aggregates all shards exactly once via an exclusive lock; no polling service is required. Total fixed budget: **180 arm batches**. Do not add parameter sweeps or rerun failures with tuned ramps on the same results.

## Endpoints and decisions fixed before outcomes

Primary: pooled OOD **safe-success difference, learned_ramp vs learned_direct**, paired within fixed parameter cells. Always report SR, CVR, safe success, safe rescue/harm and paired 95% intervals together (4,000 bootstrap draws). Also compare every arm with policy-alone, and ramp with flat/pause. Report all ID/OOD cells and both gate strata; GT gains cannot satisfy learned-method acceptance.

Mechanism checks: baseline harm reversal, original rescue retention, first violations in post-repair chunks, command discontinuity before shaping/after shaping/after repair, and same-state raw→prepared R5 hazard removal/induction. Different arms' later states are different; within-arm conditional rates are descriptive, not independently randomized mediator effects. Pre-branch matching identifies the first treatment divergence.

- **Incremental algorithm evidence:** learned ramp has positive paired 95% lower bound vs learned direct and no >2 pp SR loss vs learned direct. Inspect every stress-cell regression and whether existing rescues are lost. This is validation evidence only.
- **Broader candidate screen:** additionally ≥+3 pp safe success vs policy, positive paired 95% lower bound vs policy and no >2 pp SR loss vs policy. Even a pass still requires method freezing, an independent cohort/confirmation and the established robustness checks; it is not transfer authorization.
- If local discontinuity decreases but safe success does not improve, the tested continuity rule has failed its task-level goal. Do not claim benefit from a smoother trace alone.
- If pause/flat matches or exceeds ramp, do not claim the gradual schedule is the mechanism; report the comparison.
- If only GT-gated versions improve, transition handling remains a conditional component result; learned detection is still a blocker.
- If none improves, stop this transition family after this fixed test and return to the gate's data/objective/observability problem. Do not continue an unbounded smoothing sweep.

## How this guides architecture and algorithm changes

1. **Algorithm tested now:** explicit repair-aware continuity before verification, with fixed models and budgets. This directly asks whether controller state/acceptance integration can recover real harms without erasing rescues.
2. **Detection algorithm next:** separately trained R5/H10 event gate, frozen original repair; compare coverage-only regression against event-aware loss on matched accepted data. Anticipation coverage remains unresolved and must not be hidden by an R5-only result.
3. **Architecture only with matched evidence:** static max-score order blindness is established on offline probes, but the tested GRU still fails actual detection. A subsequent order-sensitive/event representation or longer measured-history/dynamics-context model must isolate inputs, loss and coverage, retain all training seeds and demonstrate closed-loop benefit. This transition experiment does not prove a new neural architecture necessary.

This protocol replaces the suggested further collector-only resumption study with a direct framework ablation. Production launch, checks and later results are recorded in `week4.md`; keep this numerical protocol fixed after launch.
