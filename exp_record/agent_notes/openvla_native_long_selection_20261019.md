# Fixed OpenVLA + native LIBERO-Long: task selection and next experiment

## Latest user decision

Keep the existing OpenVLA model and its native LIBERO tasks. Choose complex,
long-horizon tasks with imperfect execution to test framework benefit. The book
task is not the main evaluation. Cancel RoboCasa and alternate-model preparation.
This supersedes the earlier custom-fridge-first and native-RoboCasa qualification
proposals. The five-layer verification/repair architecture remains unchanged.

**Current status: source research and cleanup completed; no new actor rollout was
launched in this task-selection update.** The user requested the plan before
execution; the proposed screening below is not a completed or running experiment.

## Existing actor already targets the long-horizon suite

* Actor stays `openvla/openvla-7b-finetuned-libero-10`, revision
  `80970322773f81baa2e22fe495d0487b93a05cfa`.
* The [pinned official model card](https://huggingface.co/openvla/openvla-7b-finetuned-libero-10/blob/80970322773f81baa2e22fe495d0487b93a05cfa/README.md)
  explicitly calls this **LIBERO-10 (LIBERO-Long)**, fine-tuned with LoRA rank32.
  No new weights, model family, fine-tuning or LIBERO90 checkpoint is needed.
* [OpenVLA paper, Appendix E, Table12](https://arxiv.org/html/2406.09246v3#A5)
  reports **53.7% ± 1.3% standard error** on LIBERO-Long, across three random seeds
  with500trials per seed. This is a published suite aggregate, not an established
  success rate for each task or the local installed runtime.
* Official evaluation assigns LIBERO-Long **520model actions**, after10initial
  settle/no-op actions. The code comments that its longest training demonstration
  has505steps, compared with193/254/270for Spatial/Object/Goal. These are dataset
  maxima, not per-task lengths or measured local rollout times. Long-horizon here
  means the benchmark's multi-stage tabletop manipulation, not a full cooking task.
* All10local native BDDL files match the runtime checkout byte-for-byte. The
  official initial-state files exist and their hashes are recorded. No initial
  state was executed for this audit. The existing LIBERO10/90 task inventory has
  no refrigerator task; a refrigerator asset alone was never a task-support claim.

Read-only audit: `scripts/audit_openvla_long_tasks.py`.
Evidence: `backups/openvla_long_selection_20261019/` (inventory, pinned card,
retrieved official evaluator and paper, source hashes). No downloaded source was
executed by this audit.

## Recommended four-task cohort, selected before new outcomes

These are structural candidates for useful recovery opportunities, **not claims
that all four are known to fail locally**. Proposed priority is microwave, two
pots, two mugs, then drawer. Retain results for every selected task, including
ceiling performance and unqualified failures.

| Candidate / native task name | Actual initial state and native goal | Why relevant to verification and recovery |
|---|---|---|
| Microwave: `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` | Microwave starts **open**, target mug on table; goal `In(mug, heating_region) AND Close(microwave)` | Placement followed by articulated closure; detect lost grasp, inadequate insertion, closure with obstruction and regression after handoff. Opening is not an additional native subgoal. |
| Two pots: `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | Two pots on table; stove already **on**; goal both pots `On(cook_region)` AND stove remains on | Two grasp/transport/place operations, preservation of the first object's progress while manipulating the second, crowded support region. Do not add a stove-turn-on requirement. |
| Two mugs: `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` | Two target mugs plus a red distractor and two plates; goal each specified mug on its designated plate | Multi-object grounding, two placements and retention of the first result; monitor wrong target, misplacement and later displacement. Native goal imposes no mandatory left-first/right-first order. |
| Bowl/drawer: `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` | Bottom drawer starts **open**; goal `Close(bottom_region) AND In(bowl, bottom_region)` | Placement-to-release-to-closing transition and preservation during closure. Existing3/3historical development successes make this a useful competence/preservation control, not an already demonstrated weakness. |

The four full names, initial predicates and goal expressions are retained in the
machine-readable inventory. Do not change native initialization, geometry, camera
or instruction to make the tasks longer or force baseline failure.

### Historical evidence is useful but limited

* [D1 bowl/drawer](libero_drawer_target_results_20261014.md) completed the registered
  stronger physical endpoint on3/3development roots, with368/564/344actions. The
  native predicate was reached earlier. That historical runtime used the previous
  actor adapter, not the subsequently pinned TF4.40.1 inference path, and used a
  different600-action/physical-stop protocol. Reuse it as evidence of attainable
  native-task skill, not as the current official-protocol success rate.
* [Official-stack cheese+butter diagnostic](vla_failure_audit_results_20261019.md)
  reaches the two-object native goal in242actions on one initial state. It confirms
  this frozen actor is not limited to manipulating one book; it does not certify
  the new four-task cohort.
* Book remains an existing interface sanity check; its strict release failures
  must not be used as the main substitute for multi-stage framework evaluation.

### Other native long tasks, kept visible rather than silently searched

The suite also contains three two-object basket tasks (soup+tomato sauce,
cheese+butter, soup+cheese), stove-on+single-moka-pot, mug-on-plate+pudding-right-of-
plate, and the book/caddy task. The entire10-task inventory is published. The four
above cover articulated closure, repeated placement and different target matching;
any expansion beyond them must be declared as additional development selection.

## Proposed next run: native actor screening, then mechanism evaluation

### A. Fixed screening cohort: four tasks × six development roots =24episodes

Use official initial-state indices0–5for each task. Preserve the project's existing
reservations:6–9calibration,10–19reserved component-training data (unused; no
training planned),20–49confirmation. Do not repurpose those reserved roots simply
to satisfy the earlier informal10-episode proposal. Historical consumption of
drawer0–2is explicitly disclosed. This screening is development, not held-out
benchmark evidence; no OpenVLA-training-disjoint claim is made for these roots.

Before actual launch, freeze the executable runner/source manifest and checks:

* Same existing weights and official TF4.40.1 worker; BF16, no quantization, native
  H=1 greedy decoding, batch1, healthy physical GPU and matching CUDA/EGL IDs.
* Native agentview, original instruction throughout, official orientation/resize/
  center-crop and gripper/normalization mapping. No phase prompt replacement during
  this screening. Verify interfaces before interpreting any all-failure result.
*520model queries/actions plus10explicitly labelled initialization no-ops. The
  no-ops are simulator settling, not task-solving or repair actions. Native
  success stops an episode; preserve all setup failures, caps and incomplete runs.
  Proposed episode wall cap1200s excludes model loading, outer process cap1800s.
* Primary endpoint is the unchanged native full-goal conjunction. Separately log
  each constituent predicate, first attainment/loss, grasps/releases, joint state
  and actual durations. Native success does not automatically establish released,
  supported or stable success. Additional physical checks are diagnostics until
  their semantics/feasibility have been validated and their criteria frozen.
* Every task action equals its model output after the official action mapping.
  Save RGB/action/state/token tapes, actual timing, all videos and independent
  replay. No handwritten task-motion candidates, no manually solved prefix.

**What selects the mechanism-development task:** look for repeated full-task
successes together with naturally occurring partial-progress failures. For this
small development cohort, a proposed eligibility rule is at least2full successes
and2failures after meaningful manipulation progress (e.g. first object placed or
target grasped/transported). Among eligible tasks choose the predeclared priority
order above, not the largest observed benefit of our method. This operational rule
must be frozen with the runner before outcomes; it is not a population reliability
estimate. All0/6tasks remain unqualified for efficacy claims;6/6tasks remain useful
normal-case preservation controls. Do not relax rules after seeing results.

This supersedes the earlier **8/10mandatory admission** suggestion: the user now
explicitly wants an already learned but imperfect long task, with natural failure
headroom. It also removes the assumption that artificial disturbances must be the
first source of failures. Controlled perturbations are a later separately frozen
condition if needed, not a way to turn a ceiling task into a favorable headline.

### B. Diagnose recoverability and redesign the weak mechanisms

For the selected development task, locate early failure states and test whether
model-generated candidates can restore progress while preserving achieved goals.
The actor remains fixed. Candidate proposals must be shared with simple-feedback
controls. Do not rank by action novelty or wait until gross departure becomes
stagnation. Multi-object progress is a set of achieved/preserved goals, not an
invented compulsory object order. State-branch outcomes are offline labels only;
they do not feed the online selector. No repair benefit is assumed in advance.

### C. Freeze the framework, then compare on reserved initial states

Compare original OpenVLA, same-source simple feedback, and verification/repair/
recheck/handoff with matched information and resource caps. Separate selection
and rechecking factors once useful proposals exist. Keep all clean successes,
failures, rescues and harms in denominators, including selected tasks that later
show no improvement. Headline outcome is **complete native task success**, with
validated physical completion and costs separately reported. Preserve the
baseline-left/framework-right video/evidence contract. Holdout tasks test reuse
within this fixed actor/suite; cross-model or cross-platform generality remains a
later empirical question, not implied by one positive result.

## RoboCasa cancellation / deletion

User cancelled the alternate setup before any native model-control rollout. The
dedicated scratch environment/source/assets/cache tree and new alternate-policy
repositories were removed. Exclusively referenced shared Hub blobs were also
removed after checking their references. The untracked setup script was deleted.
Original OpenVLA/LIBERO environments/checkpoints and historical evidence were
preserved. [Deletion record](robocasa_cancelled_20261019.json) retains metadata and
the distinction between completed downloads, terminated installation and zero
actor queries/actions. This is not an experimental failure score.
