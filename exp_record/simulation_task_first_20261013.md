# Decision: validate complete target-simulation tasks first — 13 Oct 2026

The user explicitly agreed that continued verifier optimization is meaningless if the final manipulation task cannot be completed. This decision supersedes proposals to make another box-push loss, initialization or targeted-fine-tuning experiment the immediate next action. Historical negative results remain valid. The active deliverable is a working **single-task robot-simulation baseline**, followed by a bounded physics-intervention feasibility check on that same task. Learned efficacy is not inferred from either milestone.

## Acceptance sequence

1. **Task and actor feasibility:** environment reset/render/action execution, task-matched full-OpenVLA checkpoint, actual complete nominal task rollouts, machine-readable success/failure and video. Validate controller scaling, coordinate conventions, gripper sign, preprocessing and episode termination. A scripted controller or one appealing video cannot replace measured policy completion.
2. **Physical intervention feasibility:** define mechanically meaningful completion/stability and force/impact measurements, then compare the same actor/control opportunities with a privileged finite-budget physics diagnostic. The diagnostic must improve end-to-end safe completion, not merely lower local force. A failed diagnostic redirects attention to action/control/task formulation rather than automatically launching another verifier training run. Exact simulation is a privileged diagnostic, not a learned method or a global controllability bound.
3. **Learned contribution:** replace the privileged estimator with an allowed-input learned predictor/repair design, matched against the policy and simple control baselines. Preserve SR/CVR/safe, paired rescue/harm and uncertainty. Existing≥+3pp safe-success target, bounded SR loss, seed/cohort robustness and fresh independent testing remain efficacy requirements; no transfer claim precedes measured learned performance.

Only one task proceeds through these gates. Task feasibility may now precede proof of stronger box-push learned efficacy; it does not erase that failure or turn a change of benchmark into improvement evidence.

## Initial feasibility audit performed

Local environment: PyTorch2.14.0, transformers5.17.0, MuJoCo3.14.0 are installed. `robosuite`, `libero`, `robocasa`, `gym` and `gymnasium` are absent. Only the base `openvla/openvla-7b` HF model cache was found; the task-specific LIBERO checkpoint is not cached. No simulator installation, model-weight download, training or rollout was started by this audit.

Official OpenVLA documentation lists publicly available LIBERO-finetuned full7B checkpoints. Confirmed the HF API for `openvla/openvla-7b-finetuned-libero-goal`: public, ungated, revision `fa5ae1e7509348889295bba8e08621d8b55e9baf`, about15.08GB BF16 weights,4shards. Published benchmark performance is feasibility context only, not our experimental result.

**First integration target:** LIBERO-Goal `open_the_middle_drawer_of_the_cabinet`, present in the official task map and BDDL. The BDDL instruction is “Open the middle layer of the drawer”; the goal is `(Open wooden_cabinet_1_middle_region)`. It provides a real articulated household contact task and a documented task-suite checkpoint. Select it by exact task name, not an assumed runtime integer index. This is an initial integration target, **not** a claim that the existing predicate certifies gentle force, fine tolerance or stable completion. Inspect the joint predicate and mechanics before defining additional precision/safety metrics. The earlier illustrative “gently close a drawer” is not silently claimed to be this task.

Official wrapper defaults: Panda robot, `OSC_POSE`,20Hz environment control,7D pose/gripper action. Official OpenVLA evaluation uses one native action per image,10settling steps,300policy steps for LIBERO-Goal, fixed initial states and env seed0. Image preprocessing rotates the agent-view image180°, applies the documented resize/JPEG pipeline and90%-area center crop. Native action unnormalization uses the task-suite statistics (possibly `libero_goal_no_noops`), followed by gripper rescaling/binarization and sign inversion. These conventions must be reproduced/tested, not guessed from the existing planar velocity adapter.

## Compatibility blockers to resolve before execution

- Local `adapters/openvla_policy.py` is a frozen full7B feature backbone plus a task-trained2D planar H10 velocity head. Its `native_head` branch maps Bridge EE deltas to selected planar axes and repeats one action over H; that is not a valid Panda/LIBERO integration or a genuine H10 proposal model.
- The official native-action baseline is H1, so a first official-cadence reproduction is **baseline feasibility only**, not a matched H10/k5 verifier evaluation. Box-push H10/k5 results remain intact. A subsequent formal study needs a separately frozen compatible proposal/control protocol; repeating a native action10times does not satisfy the existing H10 requirement. Do not mix H1baseline and H10verifier numbers as a causal comparison.
- OpenVLA's official evaluated stack uses Python3.10/transformers4.40.1; the shared base currently uses Python3.13/transformers5.17.0. Existing local TF5 decoding compatibility needs parity verification for the task checkpoint; working planar hidden features do not establish native-action decoding parity.
- Official OpenVLA LIBERO requirements use robosuite1.4.1. LIBERO's broader requirements pin several old packages, including transformers4.21.1/numpy1.22.4/robosuite1.4.0. Running those full requirement files against shared base would conflict with the additive-only dependency constraint. Audit the minimal runtime requirements and MuJoCo compatibility before additive installation; do not downgrade/replace shared libraries to force a reproduction.

The next implementation deliverable is therefore a task-specific environment/actor adapter and a bounded nominal baseline protocol, with these compatibility checks explicit. Neither the original trained verifier nor a new loss is being promoted. A precise rollout budget and mechanically valid risk thresholds are not yet registered; no efficacy rollout is authorized by this feasibility note alone.

## Sources checked

- https://github.com/openvla/openvla — README, LIBERO setup/evaluation and public checkpoint links.
- https://raw.githubusercontent.com/openvla/openvla/main/experiments/robot/libero/run_libero_eval.py
- https://raw.githubusercontent.com/openvla/openvla/main/experiments/robot/libero/libero_utils.py
- https://raw.githubusercontent.com/openvla/openvla/main/experiments/robot/libero/libero_requirements.txt
- https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/master/libero/libero/benchmark/libero_suite_task_map.py
- https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/master/libero/libero/bddl_files/libero_goal/open_the_middle_drawer_of_the_cabinet.bddl
- https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/master/libero/libero/envs/env_wrapper.py
- https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/master/requirements.txt
- https://huggingface.co/api/models/openvla/openvla-7b-finetuned-libero-goal
