# Admission isolation, then diagnosis-conditioned recovery

User authorized this sequence after reviewing the A2 assembly feedback mechanism
and the 18-episode microwave diagnostic. The framework remains frozen OpenVLA
with measured verification, bounded repair and reverification. The first contrast
changes admission only; subsequent recovery changes are a separate experiment.

## A. Frozen admission-only contrast

- Output: `/home/ami/scratch/framework_admission_20261022_r1/`.
- Six new episodes, roots0–5, with `--joint-admission measured_only`.
- All joint observations, memory updates, stages, risk/no-lift/exit predictions
  and handoff vetoes remain active. Joint-only anomalies remain logged with
  `joint_trigger_advisory`; they cannot START repair. Measured triggers still can.
- Active repair generation, stage refeeding, fixed entry acceptance, failed
  retry logic, fusion threshold and exit budgets remain identical to joint v6.
- Native complete task is primary. Report paired rescue/harm vs retained bare
  VLA and full joint; intervention count, cost and verified/stable handoff are
  secondary. All six cases are run without efficacy early stopping. Technical
  failures are retained and excluded from task-success denominators.
- Same model/artifacts, language dictionary, initialization and seed; 520 total
  actions/queries, 32 local / 160 total repair actions, greedy H1, no numeric
  modification. GPU2/3, at most two actors, canonical worker and replay.
- Controls are the previous study's six bare-VLA and six full-joint episodes;
  measured+fusion is context, not the isolated control. A fresh canonical plan
  pins the new source. Every one of the previous 18 controller traces (7,792
  actions) matches the new code with default options exactly, including complete
  decision, after-verification and final summary. This is a logical compatibility
  check, distinct from their already-passed simulator and neural audits.
- Every new episode receives exact simulator replay and independent joint model
  recomputation. These remain reused-root, single-task development results.

Implementation: `NativeFramework.joint_admission`, canonical CLI/worker/status/
replay propagation, and supervisor `--phase admission`. Default behavior remains
trace-identical. Tests cover advisory-only entry, retained measured admission,
joint-directed stage requests, and an active joint veto after local recovery.

Plan SHA256: `e5c3bb087afeced3c4c140a36badc8f372a757d9464b1aa53c3cf48917bb78a9`.
The canonical 36-cell inventory is retained for runner integrity; the separately
frozen admission protocol schedules only its six declared joint-admission cells.
50 existing/integration tests passed before launching. Historical controller
compatibility is saved as `default_compatibility.json`.

### A results: admission isolation completed

All six episodes, simulator replays and joint audits pass. Complete success is
4/6 (roots1/2/3/5), with0rescues /0harms vs baseline, compared with full joint's
2/6 and0rescues /2harms. All six full action AND physical fingerprint tapes
equal the earlier measured+fusion arm. Joint computation and handoff vetoes were
active; additional joint-only admission was sufficient to explain the two harms
under this frozen development comparison. Individual risk/no-lift/exit admission
effects are not separated by this contrast.

Repair cost is176actions, with one verified repair and the same immediate root4
post-handoff containment loss. Advisory-only observations by root are150/158/45/
0/2/0. Audit totals:2,603replayed frames,2,537joint observations,176handoff scores;
zero mismatches. See `admission_validation.json` and `analysis24/`. No failed
baseline task was rescued; removing harm is not sufficient framework efficacy.
Compact reports, original audits and compatibility sources are copied to
`framework_admission_evidence_20261022/` beside this note with a hash manifest.

## B. Diagnosis-to-behavior experiment (separate freeze)

Question: can measured recovery objectives and observed progress change actual
policy requests or stop ineffective repetition, while keeping action provenance
and budgets explicit? Adding diagnosis text to logs alone does not meet this goal.

First inspect admission-isolated repairs. Use retained exact prefixes to compare
nominal continuation, the existing repair and a separately declared diagnostic
repair from the same state. Preserve the simulator/controller history and prior
budget; do not treat same action indices on diverged paths as matched states.

The diagnostic contract should expose objective, preservation requirements,
observed progress/regression, behavior attempted and reason to continue/change/
stop. Existing acceptance alternatives supply measured objectives; arbitrary
release/regrasp scripts and new task-specific prompt recipes are not substitutes.

Concrete initial mechanisms to evaluate separately are (1) compatibility of the
requested stage with achieved physical progress, and (2) terminating unproductive
attempts and recording failure by objective rather than only trigger name. Exact
windows/thresholds and state selection are frozen before new branch outcomes.
The runtime risk head is not an action-conditioned dynamics model. Offline
simulator branches are diagnostic experiments, not online foresight.

No success claim follows from avoiding intervention alone. Progress requires
verified task-level rescues without new harms, followed by a full-task regression
comparison once a local mechanism shows useful behavior.

### Offline preparation while admission runs

`scripts/diagnose_repair_contracts.py` annotates the previous 18 episodes using
the existing12-endpoint /3mm motion settings, without changing actors or claiming
alternative outcomes. All15joint repairs on roots0–2 contain a stationary,
unmet-contract window before their32-action cap. Root4 additionally contains
requested-stage / measured-stage disagreement. The output is retained at
`historical_contract_diagnostics/`.

A pure experimental `ContractProgress` component is implemented and six targeted
checks pass (missing data, local success awaiting verification, semantic changes,
cumulative motion, contact flicker, preservation loss). It is not connected to
the controller during phase A. It diagnoses measured stagnation, not VLA intent
or hypothetical action effects. Runtime integration and its next protocol will
be frozen only after inspecting phase A.

### B executable protocol, declared before new outcomes

Run four full episodes: roots0/4 × `contract_stage` / `contract_progress`. These
are the two failures remaining after admission isolation, selected as development
mechanism cases, not a new held-out set. Retain phase-A root0/4 as the existing-
repair control and baseline root0/4 as the nominal control. Because both roots'
canonical v2 cells use GPU3, run serially on that lane rather than changing lane
identity for convenience.

1. **contract_stage:** keep measured-only admission and all joint evidence /
   handoff checks, but choose active repair requests from the existing measured
   stage rule. Same instruction dictionary and fixed entry acceptance. Log joint
   stage, measured stage, requested stage and whether the request changed. This
   tests a concrete diagnostic-to-request mapping, not a new generic planner.
2. **contract_progress:** same as contract_stage, plus unverified early exit when
   a12-endpoint local window has no true acceptance alternative, no changes in
   grasp/held-clear/In/Close and object-motion diameter <=3mm. Missing measurements
   do not establish stagnation. Failure/unknown status of preservation also ends
   the attempt unverified. Local acceptance awaiting fusion is not stagnation.
   Existing native completion, cap and verified-exit priority remain in place.

The12-endpoint /3mm constants are inherited, not optimized on new outcomes.
Retry admission/memory,8-action cooldown,32local /160repair /520total action and
query caps, fusion threshold, model and language dictionary remain unchanged.
Early exit returns control to nominal; it never produces a numeric action or
claims verified recovery. Every action is still a fresh unchanged VLA proposal.

Each full rerun starts at the original initial condition. Verify exact shared
prefixes and pre-action state at the first entry and first changed request; this
is a matched continuation implemented by rerunning the prefix, not a hidden
state reset. All prefix queries count in each episode's520-query cap. Subsequent
diverged episodes do not constitute matched-state evidence for later repairs.
Report complete success first, then actual request changes, early exits, repair
cost, local recovery, verified exit, actual handoff and regression. Complete all
four cases, retain technical failures, exact replay and independent neural audit.

Before launch,60tests pass and all24retained controller traces (10,323actions)
remain exact under their recorded old modes (`pre_repair_compatibility24.json`).
The successful roots1/2/3/5 have no repair and are not new live phase-B trials;
no new full-cohort success rate will be claimed from these four diagnostic runs.

B output: `/home/ami/scratch/framework_contract_repair_20261022_r1/`.
Plan SHA256: `8707d206fbd4d25fcdc6800bc13b17af44dfdc73a5161467ef49b9f5c4df17f7`.
All four jobs completed through the canonical worker on GPU3.

### B results, 22 October

| Root | Mode | Complete success | Total actions | Repair actions | Verified repairs |
| --- | --- | --- | --- | --- | --- |
| 0 | contract_stage | No | 520 | 96 | 0 |
| 0 | contract_progress | No | 520 | 67 | 0 |
| 4 | contract_stage | No | 520 | 64 | 0 |
| 4 | contract_progress | No | 520 | 54 | 0 |

Stage-only uses160repair actions across the two roots; adding progress exit
uses121, a39-action reduction (24.4%) in repair ownership, not total execution.
Both modes have zero task rescues and zero verified repairs. These are selected
failed development roots, not a new six-root success-rate estimate or a test of
harms on successful roots.

`repair_validation.json` passes:2,080unchanged mapped VLA actions,
2,128replayed frames,2,084joint observations and281handoff scores. Source hashes,
initial-condition pairing and request/diagnosis consistency pass. Combined
results and first-change evidence are in `analysis28/` and `contrasts/` under
the B output directory above.

On root0, stage-only equals the admission action tape. On root4, stage-only
changes the request and action at index149 from the same pre-action state:
`pick up ...` becomes `put ... in the microwave`; the mapped gripper changes
from+1(closed) to−1(open). This establishes a real behavioral effect, not useful
recovery: the run still fails and has no verified repairs. Later states diverge
and cannot establish matched-state causal effects. Progress versus stage-only
first changes the executed action at267(root0) and241(root4).

The next architectural decision is informed by the
[recent-paper and source review](framework_literature_review_20261022.md).
No new experimental batch was launched for that review.
