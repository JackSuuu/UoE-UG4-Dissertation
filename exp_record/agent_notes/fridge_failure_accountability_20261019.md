# Parent failure audit: late stagnation detection and non-goal-directed selection

User asks why newer experiments fail and what was actually changed. This audit
reads the existing completed tapes; no new rollout, threshold adjustment or model
training is performed. Source: `experiment3_candidates_fridge_20261019_a`, published
in artifact commit`0cb42b8`; all distances below use the existing logged EEF position
and `short_fridge_1_g17` handle-geometry center, not a new action target/controller.

## Newly checked evidence

| Arm | Initial EEF–handle distance (m) | Minimum over initial+post-action states | Final | First local entry |
|---|---:|---:|---:|---|
| Nominal |0.4601463901|0.4599418788|1.0128130234|none|
| Simple feedback |0.4601463901|0.4599418788|0.7682426055|step368, distance1.0191651777m|
| Verification |0.4601463901|0.4599418788|0.9803930565|step368, distance1.0191651777m|

No logged endpoint reports the existing gripper–door contact predicate. Door angles
remain numerical-scale. The substantive failure precedes contact establishment;
it is not merely a stricter endpoint/dwell rejection of otherwise completed opening.
Endpoint contact observations do not by themselves exclude every substep contact.

Verification's further local entries are403/443/475 at distances1.0186096828,
0.9719199953,0.9814625980m. Q1's three selected entries368/403/443 therefore all
occur after substantial departure from the handle. Its no-useful-candidate finding
is restricted to these late visited states and fixed greedy continuation, not
early-state candidate availability or every possible model-generated action sequence.

## Implementation deficiencies versus unresolved causes

1. `experiment3_fridge_gate.FridgeCandidateGate.begin` detects same-phase stagnation
   from small net EEF/item/door motion over20intervals. Continuous wrong-direction
   movement can evade it. First intervention at368is therefore a late stagnation
   trigger, not a validated detector of loss of task progress.
2. `choose` ranks candidates by command novelty subject to current contract,
   continuity and release checks. Novelty is not task-directed progress. No validated
   prediction of opening, approach usefulness or future feasibility is available.
3. Candidates are greedy plus two samples from the same H=1 frozen actor, not an
   established recovery-skill policy or verified multi-step action proposal. Q1 uses
   one alternative action followed by greedy continuation and cannot establish
   useful recovery-sequence coverage. Diversity alone would not establish competence.
4. Repair sampling spends the common600query budget, leaving552/504executed actions
   in simple/verification versus600nominal. This is a declared cost of the method,
   not evidence that all arms had equal realized action exposure.
5. Actor/observation/task mismatch remains unresolved. The custom fridge is absent
   from the identified LIBERO10/90 task suites; the LIBERO90 candidate also lacks an
   in-domain positive-control evaluation. Framing only fits the scene into the crop;
   it does not establish handle visibility, learned visual familiarity or affordance
   grounding. Exact replay checks implementation consistency, not task compatibility.

The parent should not have treated unqualified full-task actor capability plus a
stagnation/novelty prototype as a mature framework-efficacy experiment. These runs
are useful diagnoses, but they have not produced the requested task improvement.
The weak trigger/selection design is a framework implementation limitation, not
solely a missing-skill explanation.

## Why historical results looked better

Historical fridge R2's grasp/extraction relied on handwritten primitives, with an
initially open door; it never completed the whole task. Removing those action routes
removes the source of that progress. The new closed-start autonomous pilot is harder
and not a same-condition regression comparison. Book native-success3/3versus
additional strict-completion0/3also uses different endpoints, not proof of declining
native capability. In the matched book candidate pilot, however, only nominal attains
the native goal: adverse intervention evidence must remain visible.

## Corrective ordering

Pause repeated full-fridge efficacy sweeps with this prototype and defer Q2/cloth.
First establish a bounded positive control and the shared task observation/action
contract, including calibrated handle/hand visibility and physically meaningful
task endpoints. If compatible basic skill is absent, obtain independent genuine
demonstrations and declare adaptation/splits; evaluation tapes are not training data.
Do not replace those prerequisites with manual task-motion code.

Then evaluate earlier failure/event detection on held-out labeled evidence and
task-directed candidate-sequence coverage before online gains. Progress criteria must
allow necessary indirect motion; a universal rule to always decrease EEF–handle
distance would itself be an unvalidated handcrafted policy. Freeze any redesign as
a new development version and subsequently separate selection/recheck/handoff factors.
No new successful actor, verifier or experiment is claimed by this audit.
