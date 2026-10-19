# Capability-first reliability study: improve a task the frozen VLA already performs

## User clarification and scope correction

The user explicitly confirms that the intended contribution is making an already
capable VLA more reliable, not teaching an incapable VLA a new manipulation skill.
This supersedes suggestions to make new fridge demonstrations/actor training the
next main method-development project. Existing negative experiments remain intact.

Retain the five-layer architecture. Suspend full-task efficacy trials with the
unqualified custom LIBERO fridge pairing, Q2 factor trials on empty useful-candidate
sets, and cloth manipulation. Actor/task qualification comes first. No new actor
training or manual task-motion policy is part of this next stage.

## Immediate deliverable: a reproduced capable actor–task pairing

Fridge remains the preferred domain. Select an existing learned policy together
with its native robot, observation processor, action space, simulator/assets and
task definition. A task name or training-data match is a lead, not a competence
certificate. Require compatible provenance and reproduce capability before adopting
the pairing. Do not silently transplant a12-D mobile policy into7-D fixed-Panda
control or change the framework actor alone.

The existing metadata audit identifies native RoboCasa fridge-related resources.
The first bounded readiness candidate is `lerobot/smolvla_robocasa`, revision
`f7af588d11e30b0069f200792fb285904b74c80b`, in its native RoboCasa/LeRobot contract:
three image views,16-D state,12-D action, released normalization and chunk handling.
Its linked training dataset contains CloseFridge, but its pinned model card does
not provide task-specific success results and contains generic example commands.
Those generic hardware examples are not the deployment contract. Its actual
fridge competence must be tested, never inferred from the model/dataset name.

Current runtime preflight: `robosuite` is import-discoverable; `robocasa`, `lerobot`,
`openpi`, `rlinf` are not. Scratch free space is approximately173.8GiB at this check.
No new native environment, weights or rollout was launched by this planning update.
Use a dedicated pinned environment, retain current LIBERO installations, establish
minimal required assets and evaluation entry point before a bounded smoke run.
The native setup may require substantial assets; download dependencies selectively,
not the188GB training dataset. Check GPU availability at actual launch; batch<=8,
prefer1 for qualification, detached bounded processes.

**Scope guard:** CloseFridge is a component qualification task, not opening,
retrieval, placement and closure. It must not replace the user's full-fridge target
under the same name. A capable full-retrieval actor/task pairing is still needed
before making a complete fridge-recovery claim. Opening/closing capabilities alone
do not establish that composition. Record a blocked full task if no capable pairing
is available; do not compensate by installing a handwritten motion solver.

## Capability admission protocol to freeze before rollout

1. Pin model/runtime/assets, input/output contract and original task goal. Check
   configuration/normalization/camera/action-chunk execution and stop semantics.
2. Validate any additional release/stability endpoint with suitable independent
   fixtures or existing genuine demonstrations before interpreting failure under
   it. Report native and additional endpoints separately.
3. Freeze10development initializations, nominal budget and success rule before
   seeing outcomes. Suggested operational admission is at least8/10complete
   outcomes for the **named task**, with coherent approach/contact/manipulation;
   freeze the final admission rule in the executable protocol. This small cohort
   is an engineering qualification, not a general success-rate estimate.
4. Keep all attempts. No retrying difficult seeds, checkpoint sweep until lucky,
   episode selection for videos or online training. If admission fails, investigate
   interface/competence before framework trials; do not lower the task criterion
   after observing outcomes. Record selection history and training overlap.
5. Select and freeze the admitted actor and task interface before fresh held-out
   reliability tests. Development admission and held-out evaluation are distinct.

## Then establish a recoverable reliability problem

On the admitted pairing, preregister a physically meaningful bounded disturbance
family (for example a small displacement or transient grasp slip supported by the
native simulator). Disturbances are evaluation conditions, not robot action routes.
Develop severity on separate cases and freeze it before confirmation. Retain clean
cases to measure whether the framework harms normal operation. Do not tune the
disturbance to force a preferred arm to win or select only baseline failures.

Require evidence that the nominal policy performs the task without disturbance
and encounters a localized recoverable failure under the registered perturbation.
Avoid intentionally unrecoverable states or failure from the very first skill.
Apply matched exogenous schedules/conditions and report phase exposure where
trajectories diverge. Report natural local failures too, without retroactive filtering.

## Finally evaluate the framework on these conditions

* Nominal: the frozen capable VLA.
* Simple feedback: the same candidate source and information available to the
  verification arm, with a basic fixed retry/requery policy.
* Verification/recheck: the same actor/candidate source, with task-directed checks,
  bounded correction, measured rechecking and actual nominal handoff.

Validate early event detection and candidate usefulness on relevant early failure
states before full comparisons. Keep the selection × recheck factorial controls
for attributing any combined gain. Do not equate novelty, continuous motion or
local-budget expiry with progress/recovery. No privileged future simulator branch
outcomes may enter the online gate unless an explicitly separate oracle baseline.

Match observations/evidence, disturbances and query/action/time caps; record actual
costs. Count all episodes in full-task completion and paired rescue/harm outcomes,
then separately report detection timing, local recovery, actual handoff and later
completion. Normal-case preservation matters as much as rescue. Videos remain
baseline-left/framework-right with synchronous evidence and retained failures.

## Success criterion for this research stage

The deliverable is a demonstrated improvement in reliability of an already learned
task, under a fixed actor and matched conditions. It is not a newly trained skill,
an attractive scripted video or a reproducible failure archive. After this rigid-body
mechanism result and frozen transfer checks, apply the same admission discipline
to an existing capable cloth-folding actor/task pairing.
