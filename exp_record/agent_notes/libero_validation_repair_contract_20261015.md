# Separate behavioral validation from repair efficacy

## Scope and timing

User steering on 15 Oct: define incorrect behavior before searching for favorable repairs; use parameterized failure discovery and strong comparisons. This is a **post-launch analysis/design addendum**, not a preregistration of outcomes already observed. M1's frozen controller, case grid, acceptance rule and source hashes remain the experiment definition. The five-layer architecture remains authoritative. No held-out roots are opened by this addendum.

Two questions share case identities but have different denominators:

* **A — behavioral validation:** under which declared operating conditions does the unmodified actor violate a task/phase/safety/timing specification? Also audit violations introduced by interventions. Publish failures independent of whether our controller rescues them.
* **B — repair efficacy:** on the same registered cases, does intervention improve complete-task outcomes relative to the actor and capable controls? Publish unconditional paired differences, rescues, harms and conditional rescue rates separately.

Finite simulations can check a finite trace against predicates. They do not prove every possible behavior correct, nor physical-world safety. An exact replay proves reproducibility, not correctness of the specification or simulator. Universal claims require explicit environment assumptions, reachable-state reasoning and validated dynamics/measurement bounds.

## Executable M1 completion contract (existing, not tightened retrospectively)

Let C(t) be the conjunction of the eight recorded `DrawerMonitor.conditions` after control interval t:

| Predicate | Existing executable definition |
|---|---|
| native | Original task success: bowl in bottom drawer and drawer closed. |
| full_footprint | All vertices of the bowl's 40 collision boxes fit within inner drawer x/y faces. |
| seated | Bowl lowest point is within 5 mm of the drawer floor inner face. |
| supported | Bowl/floor contact exists at the last physics substep. This is a support proxy. |
| vertical_fit | Bowl vertical extent fits drawer bounds with 1 mm slack. |
| released | Neither bowl/gripper nor cabinet/gripper contact during the interval. |
| shallow | Maximum observed bowl/cabinet penetration no greater than 1 mm. |
| slow | Bowl linear speed ≤0.02 m/s, angular speed ≤0.1 rad/s, drawer joint speed ≤0.01 m/s. |

For an episode of at most 600 control intervals (including the replayed prefix), success is:

`exists t <= 599: all(C(j) for j in [t-19, ..., t])`.

This is a bounded eventual 20-interval dwell requirement (1 simulated second at 20 Hz). Failure means no such window before the cap. Milestones and native success alone cannot terminate. Failure to satisfy a terminal predicate during acquisition is expected, not an instantaneous task failure. The earliest successful window terminates; continued stability beyond that window is not measured by M1.

**Precision:** report each geometric margin and its unit. Drawer centering is not required by the instruction; `alignment_error` is diagnostic. Do not introduce an arbitrary 1 mm centering requirement merely to make VLA fail. A new insertion/assembly tolerance must follow actual geometry, task semantics and measured numerical/estimation accuracy, and be frozen before new outcomes.

**Phase observations:** three retained/no-bowl-contact intervals establish release; three retained/clear intervals establish clearance. Geometry clearance uses a 10 mm front OR above margin, plus no bowl/gripper or cabinet/gripper contact. The monitor uses 25 physics substeps per control interval. A milestone is historical evidence; the current condition can subsequently become false. Record regression and return reason rather than inferring permanent correctness from the milestone name.

**Safety:** terminal shallow penetration and contact-free release do not establish an all-time safety invariant. Intended floor support and drawer-handle contact must not be mislabeled collision. A future invariant must enumerate phase-allowed contact pairs, prohibited contacts, admissible force/impulse/penetration limits, workspace/joint bounds, and sampling/solver convergence. These thresholds are not validated here. `Collision Rate`, safety constraint compliance and safe success stay unknown, rather than being inferred from C(t).

**Timing:** simulated 20 Hz is not wall-clock 20 Hz. A synchronous 20 Hz controller would require observation-to-effective-command age ≤50 ms, including sensing, transfer, actor, monitor, candidate prediction, gate and dispatch. M1 records actor and forecast wall time, not the full path. Report their measured distributions as partial lower bounds, separate shadow forecasts from active algorithm cost, and never turn observed maximum into a guaranteed worst-case bound. Full-prefix replay is an offline diagnostic forecast implementation, not demonstrated real-time deployment. A future realtime test must advance the plant while computation takes time and record deadline misses, stale observations and defined fallback behavior. No silent simulator pause can validate that claim.

## A — failure discovery and monitor evaluation

1. Generate identities before outcomes using task × physically motivated disturbance × severity × observation quality × deadline class × root. Start with single-factor strata and sham; cross factors only after interpreting single-factor failures. Check feasibility without using method success as a filter.
2. M1 currently implements only one task, two actuator-fault families, fixed observations and no wall-clock deadline enforcement. It is not a perception-degradation, dynamics-randomization, precision-assembly or human-interaction benchmark.
3. Classify **full-task timeout**, phase regression, measured constraint violation and computation deadline miss separately; execution infrastructure failure is a fifth category, not robot task failure. Preserve recoverability-unknown cases. “Safe, observable, repairable” are properties to assess, not guarantees conferred by a fault label.
4. Build baseline severity curves from all registered development cases. A baseline failure fraction of 30–70% is a useful descriptive development signal, not a necessary condition for usefulness or a final-test filter. With only three roots, 1/3 or 2/3 failures cannot locate a robust interval. Do not independently count 39 correlated variants as 39 initial conditions.
5. Fix the next cohort's sampling distribution from task relevance and baseline-only development diagnostics before inspecting new method effects. Preserve easy/no-harm, transition and severe strata. Report the full sweep and any explicitly conditional challenge stratum separately. Retain existing calibration 6–9, reference/training 10–19 (only if justified), confirmation 20–49 reservations; no reallocation from inspecting final outcomes.
6. A gate trigger is not a ground-truth failure. M1 triggers on a release opportunity, including clean successful VLA traces. It cannot currently yield a defensible failure-detector precision/recall figure. Future onset labels need separately fixed phase obligations and an independently audited reference, not the same threshold as the detector. Measure detector lead/delay on unmodified common trajectories; repaired trajectories censor later failure events.

The read-only `scripts/audit_libero_mining_contract.py` recomputes the existing completion dwell and checks artifact hashes. It reports final unmet predicates as observations, **not causal failure diagnoses**. It retains incomplete, missing and unverified runs explicitly, and creates a baseline-only configuration map independent of repair success. This post-launch audit does not add an independent physical safety oracle.

## B — repair contribution and stronger comparisons

For every matched case, let Y_m and Y_b denote complete-task success under method and comparator. Primary effect: `mean(Y_m - Y_b)` over all registered, completed pairs, with all missing/unverified cases disclosed. Rescues count `(1,0)` and harms `(0,1)`. Conditional rescue is rescues / comparator failures, undefined when none fail. Do not report conditional rescue alone: it omits harms. Estimate uncertainty over independent roots (variants clustered), not steps; M1's three roots support descriptive development conclusions only.

M1 already includes native H1 VLA closed-loop requery, release-conditioned hold and timed geometry controls. These are relevant mechanism controls, not a replacement for a strong planning baseline. The actor already replans its next action from each new image; repeated deterministic queries on one image are not a new replanning baseline.

**Next strong comparator to implement after the current integrity issue is resolved:** simulator-state receding-horizon shooting controller using the same state access and bounded action primitives as the privileged repair. At each registered decision opportunity it evaluates candidate action sequences under a frozen physical progress cost, executes the first action, and replans on the next available observation. State cloning/replay must include controller state. Horizon, candidate count, opportunity schedule, deadlines, objective and control cap must be fixed before new outcomes. It is a privileged planning comparator, not a learned policy or certified safety controller. A three-candidate lookup at only two events should not be relabeled full MPC.

Two complementary comparisons are necessary: (i) common trigger/backend/opportunity budget to attribute the phase or candidate-ranking component; (ii) capable planner under a common episode compute/deadline budget, allowed to allocate it sensibly, to compare complete systems. Restricting a planner to our two events can handicap it and cannot be the only comparison. Include gate-only/detector substitutions with a shared repair backend if claiming detector contribution. Learned recovery requires compatible data/checkpoint, training-budget accounting and independent splits; it is not an automatic requirement to launch new RL training now.

Identical allowable information, physical disturbances, success/stopping rules and actor/query limits are required. Report actual forecast counts, wall time and active versus shadow computation. A non-privileged deployment claim will additionally require image-derived state estimates and matched uncertainty, rather than giving only our repair ground truth.

## Immediate evidence and decision

At the first inspection following this steering, M1 had stopped automatically: 9 completed episodes, 8 with successful independent replay. Root2/sham/phase_repair failed the pre-action fingerprint assertion in the verifier; two other workers were terminated by the supervisor. No faulted case had yet run. Three verified VLA sham episodes succeeded; all three verified hold sham episodes failed. The two verified timed shams gave one success and one failure. These are development harms, not a baseline-failure sweet spot. The unverified phase episode and interrupted episodes are excluded from efficacy denominators and kept visible.

This corrects the earlier live status: the batch is not currently progressing toward 195. Preserve raw data, diagnose replay integrity, and version any technical fix. Do not change the controller because the first shams are unfavorable, discard them, or increase fault severity to hide them. Finish/restore the registered comparison only with explicit integrity handling. The immediate methodological lesson is already concrete: release opportunity alone is insufficient evidence that intervening is beneficial. Whether an evidence-conditioned repair fixes that problem is a new hypothesis, not an established result.

### Follow-up audit in this session

The originally rejected root2/sham/phase trajectory passes the unchanged full verifier in a new isolated process, including forecast and physical replay. There are now **9 replay-verified complete episodes** (VLA3/3, hold0/3, timed1/2, phase0/1, verified-repair not run). This is a successful independent retry, **not resolution of the original in-process mismatch**. The launch remains failed; interrupted episodes remain incomplete. The original failure log and retry evidence are preserved. No controller/source/protocol used by M1 was edited.

The five fully executed unsuccessful intervention episodes all reach native task success. At the terminal interval each has retained bowl placement, closed drawer, no bowl/gripper contact, but cabinet/gripper contact; `released` is the only false terminal completion check. They fail the unchanged full-completion contract, not a newly tightened geometric threshold. This observation does not prove collision danger or the causal reason for the prolonged contact. Root2 phase handed back at stable clearance but ultimately failed full completion, directly demonstrating why a successful local handoff is not sufficient evidence.

Across the nine completed episodes, per-episode median live actor inference times are approximately **0.706–0.726 s**. Every recorded live actor query exceeds 0.05 s. Thus the current synchronous implementation does not meet a 20 Hz observation-to-command deadline even before monitor/active prediction overhead. This conclusion applies to this implementation/hardware/run; it is not an impossibility claim about optimized or asynchronous VLA systems. M1's frozen 600-step completion endpoint remains unchanged; real-time feasibility is a separate, explicitly post-launch analysis.

Audit output: `backups/libero_mining_contract_audit_20261015.json`. Reproduce on a fresh output path:

```bash
python scripts/audit_libero_mining_contract.py \
  /home/ami/scratch/libero_failure_mining_20261015_a \
  --output /tmp/opencode/m1_contract_audit.json
```
