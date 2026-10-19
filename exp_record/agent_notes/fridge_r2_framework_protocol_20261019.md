# R2 first actual five-layer refrigerator comparison (development)

User requests direct frozen VLA versus VLA+framework comparison, rather than another
script-only feasibility video. This protocol is frozen before R2 actor outcomes.
R1 geometry and frozen OpenVLA checkpoint remain unchanged. This is a first integrated
development test, not confirmation or a guarantee of successful opening/retrieval.

## Executed architecture

1. Shared task tracker: open → grasp → extract → place/release → close → withdraw.
   Both arms receive identical phase-prompt rules and permitted state access.
2. Frozen OpenVLA queried once each control interval, including intervention steps.
3. Monitor: phase progress, actual pad grasp, support, clearance, premature release.
4. Gate: nominal VLA versus bounded local correction; no-progress window30steps,
   range<0.005 in phase metric; premature release independently retains grip.
5. Feedback repair: phase-local approach/regrasp, extraction, placement or door
   approach/arc motion; geometry targets updated from current state. Check successful
   exit for3consecutive observations, then hand back a fresh VLA proposal. Unverified
   budget exits are separately labeled, never counted as successful handoffs.

Repair has at most200actions per activation and420actions per episode; verified
handoff cooldown20steps, budget-exit cooldown30steps. No object teleportation,
attachment, post-reset state installation, task-specific actor training or hidden
script prefix. Repair can still dominate this first version: report occupancy,
VLA actions, verified handoffs and remaining completion after handoff.

The wrapper uses privileged simulation poses/contact. This is an explicitly
oracle-feedback development integration. The actor uses RGB, and shared phase
prompts also depend on the common task tracker. No visual pose estimation, learned
physics prediction or asynchronous execution is claimed. A simple-feedback arm is
required later to attribute improvements to the structured verification mechanism.

## Cohort and budgets

Seed0, closed-door full-sequence start and open-door extraction diagnostic;
each paired across baseline and framework:4runs total. Both arms start empty-handed
from identical initial physics/controller/images. Same600action and600query caps,
20Hz simulated control,1200s episode wall cap excluding model load; external worker
timeout1800s. Success termination uses the same monitor. Query counts may differ
only with common success/done/time termination. No injected disturbances in this batch.
Open-door cases do not count as full closed-door task success.

Use healthy GPUs1/3, one actor per GPU, batch1, detached supervisor. Source hashes
are checked before every launch. Each run logs nominal proposals AND actual actions,
ownership/events, predicates, task transitions, contact-informed completion, timings,
physics/controller fingerprints and raw agent/wrist images. Independently replay
physics, gate decisions and completion from frozen sources and recorded proposals;
this does not re-query the actor or establish decode parity.

## Development endpoint

Observed opening≥1.8rad, observed grasp and held extraction; item fully outside the
front plane before transport. Placement requires its full collision-box footprint
inside the BDDL table region x[-0.15,0.10],y[-0.25,-0.05], table support, bottom within
5mm of table height, no gripper contact throughout the interval, speed<0.02m/s and
angular speed<0.1rad/s. Final closure≤5degrees and door speed<0.05rad/s; empty hand
clear of handle by>0.16m without hand/item or hand/door contact;20consecutive control
intervals. Endpoint checks are development predicates, not physical safety guarantees.
Full success additionally requires a closed initial door. Retain all failed/incomplete
cases and native labels separately. Qualify positive/negative endpoint fixtures
before interpreting complete-task rates as validated efficacy.

## Video and decision

Automatically render each completed pair: baseline VLA left, actual VLA+framework
right, same simulation clock and view, phase/ownership/trigger/handoff labels, actual
step budgets and explicit end-frame holds. Failed jobs are listed rather than omitted.
No prompt/threshold changes within this cohort. The first result may be failure:
localize missing nominal skill, faulty trigger, blocked repair or failed handoff
before creating a new version. This comparison tests the integrated loop now;
it does not retroactively turn R1 servo clips into framework evidence.
