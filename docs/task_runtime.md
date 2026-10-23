# Generated-task runtime

## Input and ownership

The user supplies an instruction, such as “put the yellow and white mug in the
microwave”. The system generates grounded goals and runtime recovery contracts.
There is no per-task Python class or hand-authored task JSON. `TaskSpec` and
`RecoveryPlan` are validated model-output records, not user configuration.

```text
instruction + image + scene inventory
              ↓
ChatPlanner → TaskSpec (goals + optional progress milestones)
              ↓
Session → Controller → frozen OpenVLA proposal → environment.step
              ↑                            ↓
      PhysicalVerifier ← measured endpoint
              ↓ progress loss / stagnation
ChatPlanner → RecoveryPlan → bounded repair → rechecks → nominal proposal → handoff
```

- `src/planning/schema.py`: supported predicate grammar, bound entity IDs,
  nonempty AND/OR contracts, type/contradiction checks. No executable model output.
- `src/planning/multimodal.py`: real multimodal HTTP client for a local or remote
  OpenAI-compatible `/chat/completions` vision endpoint. Image + instruction +
  current measured facts enter the request. Raw responses and request hashes are logged.
- `src/controller.py`: generic progress monitoring, repair transactions, fixed
  entry acceptance, preservation, endpoint rechecks, budgets and actual handoff.
- `src/runtime.py`: sole owner of policy queries and environment execution.
- `src/interfaces.py`: structural backend contracts and the finite H1 `Proposal`.
  Source API: `from src import Session, Limits, Proposal`. ML/simulator packages
  are not imported by this API. A Session runs once; the caller owns cleanup.
- `src/verification/physical_verifier.py`: tri-state predicate evaluation and
  four-dimensional measured evidence, always used in the main loop.
- `src/adapters/libero_environment.py`: one environment bridge across tasks.
  Builds the entity inventory from scene objects/fixtures/sites and evaluates
  requested predicates. Does not expose native BDDL goals to the planner.
- `src/adapters/openvla_policy.py`: frozen official OpenVLA `predict_action` H1,
  with the official LIBERO gripper mapping. No numerical repair edits.
- `src/journal.py`: append-only events plus exact input PNGs.
- `src/perception/`: strict pretrained DINOv2 and visual generated-predicate
  prototype context; no fixed `acquire/transfer/close` stage list.
- `src/memory/keyframe_bank.py`: read-only measured-success training keyframes.
- `src/prediction/short_term_predictor.py`: causal constant-velocity forecasts.
- `src/verification/vision_physical_monitor.py`: shared live module context.
  Vision/memory/forecasts enter recovery planning; loaded learned risk can add
  persistent triggers and exact-handoff vetoes in `gate` mode. Similarity and
  forecasts never establish measured predicate truth.
- `src/experiment/`: all active experiment entry points, collection and tests.

Model-generated semantics can be wrong. Schema validation checks structure and
bindings, not language entailment or perceptual correctness. The prompt forbids
inventing final objectives (e.g. closing a container for a placement-only task),
but real-model grounding/semantic accuracy requires separate evaluation.

## What is measured

Supported primitives: `in`, `on`, `open`, `close`, `turnon`, `turnoff`, `grasped`,
`contact`, and `stable`. Each predicate binds exact inventory IDs and an expected
boolean value. Unsupported predicates/IDs reject the generated plan. Missing or
inapplicable measurements remain **unknown**, including negated predicates.

Task goals are AND conditions. Recovery acceptance is OR-of-AND groups: one
whole alternative must hold for three consecutive executed endpoints; evidence
from different alternatives cannot be mixed. Preservation conditions must be
measured true at entry and throughout the transaction. Stability requires three
consecutive post-entry positions within 3 mm. The model cannot change thresholds.

The first implementation monitors durable generated-goal/milestone loss and
180 actions without newly verified progress. It invokes recovery generation on
an event, rather than querying an MLLM every action. Recovery can defer an
intentional transition. Malformed/unavailable recovery planning is logged and
nominal execution continues; failed initial task generation stops the run.
Repeated attempts require cooldown AND changed measured context. This generic
monitor is a new experimental mechanism, not claimed equivalent to the old
microwave contact-stall detector.

Budgets remain **520 executed actions, 520 policy queries, 32 repair actions per
transaction, 160 total repair actions**. Candidate queries and rejected handoff
queries all consume the policy budget. Planner calls have a separate explicit
limit (24 by default) and latency log. Initial environment settling uses ten
separately recorded protocol noops before task execution.

Budget exits are explicitly unverified. A verified local condition is not an
executed handoff. Handoff occurs only when the actual fresh nominal proposal is
executed. Subsequent contract loss and completion are recorded separately.
`native_success` is benchmark evaluation metadata and is separate from generated
task completion, especially when the user instruction differs from the benchmark.

## Modules A and B

`Session(..., recovery_scorer=A, handoff_scorer=B)` accepts task-independent
scorers. Neither changes an action numerically.

```python
# A: one score per unchanged proposal, using only pre-execution context.
scores = A.score(observation, task_spec, recovery_plan, actions, physics_evidence)

# B: assess the exact fresh nominal proposal to execute after acceptance.
assessment = B.assess(observation, task_spec, recovery_plan,
                      nominal_action, physics_evidence)
# assessment must contain accept: bool; additional scores/provenance are logged.
```

With A, the first repair action in an episode uses four queries at the same
observation/instruction: one greedy and three sampled. Highest score wins; ties
prefer the greedy candidate. Later actions remain greedy. If fewer than four
queries remain, only one greedy query is made. This is H1 selection, not tree
search. Without A, all actions are greedy.

B receives the **original-task nominal** proposal, not the repair proposal. If
rejected, that nominal action is never executed; the controller continues
bounded repair and subsequently queries a fresh nominal action. Without B, the
runtime records `measured_contract_only` and `predictive_resumability=unassessed`;
it does not pretend a learned handoff estimate exists.

Physical evidence keeps four values: measured containment, measured grasp, last
executed end-effector action/motion alignment, and target contact count. Validity
flags and provenance are separate metadata, not extra numeric dimensions. Missing
values are encoded as zero **with validity=false**. Alignment is retrospective;
all candidates see the same measured evidence. It is never a candidate rollout
or predicted consequence. For multi-object tasks this initial evidence view uses
the first object in the goal clauses; the full observation and all clauses remain
available to scorers/verifiers. Articulation-only tasks may have missing values.

Generic implementations now exist in `repair/recovery_scorer.py`,
`verification/handoff_verifier.py`, `verification/calibrated_risk_head.py` and
`verification/fusion_mlp.py`. Their new versioned vector includes pretrained
visual384, physical4 + mask4, generated predicate grammar/truth pooling, acceptance
summaries, causal history and the exact optional action. It is not unrestricted
language encoding, action-conditioned physical prediction or proof of entailment.
Old stage checkpoints are rejected, not padded or reinterpreted. Compatible heads
and memory require matching visual weight/source/preprocessing signatures.

Training labels and supervised fitting are implemented in `src/training/`;
collection/training commands are in `src/experiment/`. New risk predicts observed
nominal24 progress loss; B labels the actual nominal action's observed contract
continuation; A labels the executed first proposal's observed measured recovery.
Unknown evidence/future repair is censored. Real-data training and efficacy are
pending. No untrained head is used, and empirical thresholding is not claimed to
provide conformal/episode-level guarantees.

`Session(..., monitor=VisionPhysicalMonitor(...))` shares one encoding per image
across subgoal context, retrieval, risk and A/B. `--monitor-mode log` records risk
without intervention/veto; `gate` requires the supplied trained head to influence
control. Default sessions without an encoder still log observed-motion forecasts
but explicitly lack visual/learned modules. Memory is read-only during episodes;
new keyframes are built offline from measured successful roots10–19 only.

## CPU checks

```bash
python -B -m unittest discover -s src/experiment/tests
```

The core tests require no torch, NumPy, GPU or downloaded model. With torch,
an additional synthetic CPU training/save/load test runs. The suite exercises the
production planner transport through mocked HTTP replies, production controller,
runtime, verifier, journal and a simulated LIBERO API. Policy/planner fixtures
are confined to tests; they are not runtime fallbacks.

## GPU-machine launch

Follow [GPU setup](gpu_setup.md) for the candidate Python 3.10 dependency recipe,
editable LIBERO registration, noninteractive asset paths and model download.
This is **not** a GPU-validated lockfile. Do not use an old launcher or install
LIBERO's old requirements over OpenVLA's `transformers==4.40.1`.

Start a vision-capable small MLLM service separately. It must accept image data
URLs and JSON-object output requests at `/v1/chat/completions`. The repository
does not yet select/benchmark a specific MLLM checkpoint. Specify its served
model ID; optional authentication is read from `PLANNER_API_KEY`.

Example: LIBERO-10 task index **9**, task order 0, is the native microwave
placement-and-close task. Initial-state index is independent of task index.

```bash
CUDA_VISIBLE_DEVICES=0 python -B -m src.experiment.run_task \
  --suite libero_10 --task-index 9 --init-index 0 \
  --policy-checkpoint /models/openvla-libero-10 \
  --unnorm-key libero_10 --device cuda:0 --egl-device 0 \
  --planner-url http://localhost:8000/v1 \
  --planner-model YOUR_SERVED_VISION_MODEL \
  --out /path/to/new/microwave_run
```

Omit `--instruction` to use the native benchmark sentence, or pass a user's own
sentence explicitly. Use `--mode baseline` for unchanged full-instruction VLA
execution with recovery disabled. An optional `--scorer-factory module:function`
loads a Python factory returning `{'recovery': A, 'handoff': B}`; it is a model
integration option, not a task configuration. Use a fresh output directory.
Built-in generic loaders use `--visual-checkpoint`, `--visual-source`, optional
`--memory-bank`, `--risk-checkpoint`, `--recovery-checkpoint`, `--handoff-checkpoint`
and `--monitor-mode log|gate`. All experiment commands are documented in the
[microwave protocol](../src/experiment/README.md), including strict full-arm gates.

Before an episode, use the same entry point with `--preflight scene`,
`--preflight planner` or `--preflight policy`. Scene needs neither model; planner
requires only its URL/model; policy requires only its checkpoint. These checks
perform initial settling, save the actual image and never execute policy actions.
Planner check generates/measures a TaskSpec; policy check makes one H1 query.
`--preflight components` additionally checks real encoding/memory/risk at the
initial image, requiring visual paths and planner but no OpenVLA checkpoint.
Preflight completion is **not** task success. Full episodes (including baseline)
require a planner to define generated goals; baseline never requests recovery.
The wall deadline includes setup/model loading; increase `--rollout-seconds` if
needed and record it identically for paired arms.

Outputs include `status.json`, `result.json`, `events.jsonl`, `images/`, the scene
BDDL and selected official initial state. Events retain all proposed actions,
sampling seeds, selections, task/recovery outputs, measurements, unknown flags,
query counts and actual handoffs. This new runner does not yet implement the old
runner's independent simulator tape replay, so its runs are **not automatically
replay-audited experimental results**.

## Historical boundary and next validation

Obsolete fixed-task controllers/runners/training paths were removed. Required
visual/risk/memory/prediction/A/B roles are retained and adapted to generated
semantics, not left as an empty five-layer shell. No old
runner is aliased to the new Controller: their interfaces are incompatible.
Records, archives, results and checkpoints remain intact. Reproduction of an
old experiment requires its original Git revision or pinned source snapshot.
`scripts/backup_research_progress.py` is retained only for archive restore/verify,
not as an alternative experiment entry point. Archived source is not active code.

Next GPU checks: real endpoint structured-output/grounding smoke, one complete
microwave episode, exact action/input audit, then paired runs and calibrated A/B
training on actual recovery and nominal continuation traces. Preserve splits:
roots 0–5 development, 10–19 training, 6–9 calibration, 20–49 confirmation.
Packaging should expose this single implementation; it should not copy the
controller into another package tree or reintroduce per-task adapters.
