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
- `src/verification/physical_verifier.py`: tri-state predicate evaluation and
  four-dimensional measured evidence, always used in the main loop.
- `src/adapters/libero_environment.py`: one environment bridge across tasks.
  Builds the entity inventory from scene objects/fixtures/sites and evaluates
  requested predicates. Does not expose native BDDL goals to the planner.
- `src/adapters/openvla_policy.py`: frozen official OpenVLA `predict_action` H1,
  with the official LIBERO gripper mapping. No numerical repair edits.
- `src/journal.py`: append-only events plus exact input PNGs.

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

Existing `repair/recovery_scorer.py`, `repair/handoff_scorer.py`, fusion/risk/stage
models and their checkpoints belong to the historical experiment. They are not
silently converted to this schema: the old B labels were repair-entry outcomes,
not actual nominal-action-conditioned continuation labels. A generic trained
implementation must consume generated task/contract context and compatible
features, and be evaluated on root-separated data before efficacy claims.

## CPU checks

```bash
python -B scripts/test_task_runtime.py
```

No torch, NumPy, GPU or downloaded model is required. The suite exercises the
production planner transport through mocked HTTP replies, production controller,
runtime, verifier, journal and a simulated LIBERO API. Policy/planner fixtures
are confined to tests; they are not runtime fallbacks.

Historical checks (in the existing environment with NumPy/torch):

```bash
PYTHONPATH=scripts:. python -B -m unittest \
  test_native_framework test_joint_framework test_fusion_handoff \
  test_repair_candidates test_repair_contract_progress \
  test_repair_data_expansion test_fridge_repair_probe \
  test_action_correction test_recovery_options
```

## GPU-machine launch

Clone code and the pinned benchmark:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules \
  https://github.com/JackSuuu/UoE-UG4-Dissertation.git
cd UoE-UG4-Dissertation
python -B scripts/run_task.py --help
```

Prerequisites: a working LIBERO/robosuite/MuJoCo environment, official initial
states/assets, PyTorch, OpenVLA-compatible `transformers==4.40.1`, AutoProcessor
dependencies (including timm), and TensorFlow for official image preprocessing.
Use a Python environment compatible with those packages (typically Python 3.10).
Do not blindly install LIBERO's old requirements over the OpenVLA transformers
version. This refactor has **not** produced a newly validated environment lock.

Start a vision-capable small MLLM service separately. It must accept image data
URLs and JSON-object output requests at `/v1/chat/completions`. The repository
does not yet select/benchmark a specific MLLM checkpoint. Specify its served
model ID; optional authentication is read from `PLANNER_API_KEY`.

Example: LIBERO-10 task index **9**, task order 0, is the native microwave
placement-and-close task. Initial-state index is independent of task index.

```bash
CUDA_VISIBLE_DEVICES=0 python -B scripts/run_task.py \
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

Outputs include `status.json`, `result.json`, `events.jsonl`, `images/`, the scene
BDDL and selected official initial state. Events retain all proposed actions,
sampling seeds, selections, task/recovery outputs, measurements, unknown flags,
query counts and actual handoffs. This new runner does not yet implement the old
runner's independent simulator tape replay, so its runs are **not automatically
replay-audited experimental results**.

## Historical boundary and next validation

The old controller was moved (not copied into a second active framework) to
`src/legacy/microwave_controller.py`. Historical comparison/training/analysis
imports point there. `scripts/run_framework_comparison.py` remains the historical
microwave experiment driver; `--use-v5` and its numerical-correction launch path
have been removed. Old archives retain their original source/manifests and must
not be edited to pretend compatibility with new sources.

Next GPU checks: real endpoint structured-output/grounding smoke, one complete
microwave episode, exact action/input audit, then paired runs and calibrated A/B
training on actual recovery and nominal continuation traces. Preserve splits:
roots 0–5 development, 10–19 training, 6–9 calibration, 20–49 confirmation.
Packaging should expose this single implementation; it should not copy the
controller into another package tree or reintroduce per-task adapters.
