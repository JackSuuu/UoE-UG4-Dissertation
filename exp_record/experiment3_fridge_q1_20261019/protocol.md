# Q1 v1: do saved model candidates advance the fridge task?

OFFLINE evaluation oracle only. Branch outcomes never enter a selector, live repair,
motion generator, subsequent candidate search, or training. The completed frozen
fridge v1 and failed book preflight remain unchanged. Q2 causal isolation of
selection/recheck/actual handoff remains a separate gated future experiment.

## Opportunity sampling and pre-outcome choices

Source: completed exact-replayed `verification` arm in
`/home/ami/scratch/experiment3_candidates_fridge_20261019_a`.
Take the first THREE chronological rows with an actual local_entry event, >=2 saved
candidates and step<=550; if fewer, take all. No outcome/novelty-based filtering.
Evaluate every saved candidate including duplicates, maximum9 branches. Record
source-arm distribution bias: these are verification-policy states, not an unbiased
task-state sample. This is a single seed/scene/camera/actor and no transfer estimate.

Before any branch rollout, precompute both existing simple-feedback and verification
choices on identical candidate set and state, with identical preceding five executed
source actions, active entry phase, and current constraint checks. Freeze indices,
actions, all checks/novelty, source hashes, candidate duplicate groups and opportunity
list in plan/manifest. Verify the recomputed verification choice equals its original
entry-row choice. These choices cannot change in response to branch outcomes.

## Exact reconstruction and branch resource contract

For every candidate independently: new environment, original closed-start seed0
and fixed camera, standard20-step setup, then deterministic action-by-action replay
of the source prefix. Reconstruct the shared stage tracker by observing every
replayed interval, preserving milestone/streak/history semantics. Assert every
prefix before/after fingerprint and final source-row before physics/controller/all
camera fingerprints, raw state, agent image, read_state and full stage evidence
exactly. No approximate state fork and no tolerance relaxation. Any mismatch stops
the cohort and is retained.

The saved candidate is action1 (zero new queries). Actions2..12 are11 freshly
queried same-frozen official TF4.40.1 greedy actions on actual branch RGB and shared
stage prompt. Then20 fresh greedy continuation actions13..32, with identical
observer and prompt rules. No scripted repeated action, geometric primitive or
action modification. Query seeds use absolute source step plus branch offset;
all duplicate branches share the same deterministic greedy seed schedule. All32
actions run unless technical/budget failure, including after an intermediate stage
or native milestone, to measure retention. No online gate runs after candidate1.

Caps:32 newly executed branch actions/31 fresh model calls per branch; at most
288 actions/279 fresh calls total; GPU2 only, batch1. Each branch plus its independent
replay is bounded by1200s; cohort execution including reviews bounded by3600s.
Count original source candidate generation separately (one charged source call per
saved candidate, including duplicates); original entire source-arm600 calls remain
reported. Prefix replay, standard settling and independent replay physics steps
are separately counted overhead, not hidden within the288 new branch-action cap.
Horizon640 accommodates20 setup+entry<=550+32 branch actions. No source step can
exceed550. Simulation pauses during inference.

## Frozen phase-progress definitions

Report at action horizons1,12(primary),32(secondary), using BEFORE source evidence
as reference. Positive margin means reduction in the following active-entry-phase
goal gap, never arbitrary EEF/item motion or command novelty:

* open: max(0,1.8-angle), rad; descriptive directional signal >=.01rad.
* grasp: EEF-to-item distance, m, **approach proxy only**. A useful grasp signal
  requires the current measured grasp predicate satisfied; distance reduction alone
  never qualifies.
* extract: max(0,item collision-box max world-x-(fridge front-.015)), m;
  directional threshold.005m; useful signal also requires current grasp. Full extracted/grasp/no-shelf predicate reported
  separately; geometry is measurement only and never an action route.
* place: target-footprint violation gap plus bottom-seating violation gap, m,
  derived from existing measured footprint margin and5mm floor envelope;
  directional threshold.005m. Release/support/slow-motion satisfaction separately.
* close: max(0,abs(angle)-5degrees), rad; directional threshold.01rad.
* withdraw: max(0,.16-distance(EEF,handle)), m; threshold.005m; useful signal also requires no current grasp or hand-item contact. Empty hand,
  noncontact and closure status reported separately.

A descriptive useful signal is the declared directional threshold OR newly
satisfied active-stage predicate (grasp always requires satisfaction), with no
detected guarded achievement loss and no newly worsened item-penetration proxy.
This is a measured local progress signal, not full phase success or validated repair.
Stage satisfied/violated/unknown, consecutive satisfaction, entire last20-interval
retention at32, ordered milestones and full completion dwell are reported separately.

Guarded losses over each prefix: previous grasp lost without measured table support
or placement; previous supported placement lost; previously opened door becomes
<1.5rad before placement during grasp/extract/place; previously achieved closure
lost during withdraw. Also report every initially satisfied predicate that becomes
unsatisfied, explicitly including expected transitions (initial close predicate
loss while opening is not guarded failure; grasp/extract predicate loss after a
supported release can be expected). Item penetration proxy is new maximum beyond
max(1mm,initial+1mm). Record solver-sample contact-force and penetration maxima;
physical damage and validated contact safety remain UNKNOWN, because these proxies
have no damage/safety calibration. No unsupported safe/harm-free assertion.

## Q1 reporting and Q2 limit

For every horizon report useful-candidate availability per opportunity, duplicate
groups, both frozen selectors' useful rate conditional on availability, candidates
rejected by verification that nevertheless show measured useful progress, and
offline oracle-best measured directional margin minus each selected margin. The
oracle is post-run evaluation only. Report both raw directional signal and guarded
descriptive useful classification, so thresholds/losses remain inspectable.
Zero available useful candidates means **inability to assess useful selection**,
not verifier/simple-feedback equivalence. Duplicates are not independent samples.

The20-action tail is **counterfactual nominal continuation**, not an actual online
successful repair or handoff. Original v1 exits can be audited but its bundled
three-arm comparison does not isolate Q2 recheck/handoff effects.

Save states and pre-action images plus final, fresh query/action/token/cost tapes,
all exact reconstruction/replay evidence, protocol/source/plan hashes and negative
outcomes. Independently replay every branch from a fresh source-prefix environment.
Videos compare all candidate branches at common20fps, annotate real synchronous
pre-action evidence, frozen selector choices and counterfactual ownership; explicit
held final frames never add physics/dwell. No outcome-fabricated narration. No
changes to v1, main-owned stage observer/docs/core, or commits/push.
