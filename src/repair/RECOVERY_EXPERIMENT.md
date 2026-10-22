# Oracle recovery-options experiment (v7)

This is a controlled **recovery-capability baseline**, not a learned physics
engine, model-generated suffix optimizer or visual-only repair policy.
It addresses the observed repeated acquire/no-lift failure without adding a new
encoder, RL, a trajectory optimizer, or changing fusion/risk thresholds.

## Execution and causal differences

`recovery_options.py` adds two bounded acquisition options:

1. `lift_probe`: maintain closure and servo to a target 4 cm above the entry EEF.
   Six observed endpoints with less than 3 mm vertical object progress terminate
   the option; its hard limit is eight actions. Object/EEF motion mismatch is
   recorded, so contact grasp is not presented as dynamically verified grasp.
2. `regrasp`: release, retract, approach above the current object using the entry
   relative grasp offset, descend, close, and lift. Its total limit is 24 actions.
   Lack of EEF progress in motion phases also terminates it early. It keeps the
   measured approach orientation and offset; it does **not** synthesize a new
   grasp pose or guarantee recovery of an unsuitable entry pose.

Both options operate inside the original 32-action transaction / 160 total
repair / 520 action-and-query caps. Failed options are remembered across trigger
names and transactions. Three consecutive observations at least 2 cm from the
context anchor reset that memory; elapsed time and contact/grasp flicker do not.
If no supported untried option remains, repair entry is suppressed. Exhaustion
inside a transaction causes an explicitly **unverified** early exit. Original
three-endpoint measured recovery contracts, fusion/risk/prediction vetoes, native
completion, and bounded unverified exits remain unchanged.

Acquisition is the new skill's explicit scope. When containment is already true
or the entry contract is another stage, the frozen VLA proposal remains the
action source. A raised acquisition recovery is held for the existing handoff
checks; it is not immediately declared safe. No generic collision safety
guarantee or new transfer/placement recovery capability is claimed.

Cartesian errors are capped at 8 mm per action and converted through the live,
positive zero-offset delta OSC scale. Missing positions or unsupported controller
semantics fail closed. Rotations are held at zero delta. Skill actions may change
translation and gripper commands: this is **not** the historical unchanged-VLA
protocol, nor the old bounded-translation correction protocol.

Each executed step still makes one frozen VLA query, including skill-controlled
steps. The unused proposal, tokens and model input are retained and explicitly
labelled; query matching is an experimental control, not an efficiency claim.
`oracle_skill_actions` is separate from old numeric-correction counts.

## Matched full-task development study

`scripts/run_recovery_comparison.py` runs:

- `baseline`: unchanged frozen OpenVLA;
- `current`: joint v6 with stage-prompt, frozen-policy repair;
- `options`: same joint gates plus the oracle acquisition options, progress stop
  and cross-trigger failure memory.

The new arm bundles execution capability and failure-memory changes. Its contrast
does not independently identify each change's effect. This is intentional for
the initial repair-capability check; no component-specific efficacy is claimed.

Only the six already used development roots (0..5) are scheduled. Root0's new
arm executes first for implementation checking, then its controls; the method
remains frozen for the remaining roots. These are reused conditions (including
training/calibration roots), **not** held-out generalization. No reserved state
or new task is opened. One actor at a time uses the established GPU2/3 lanes.

Each valid episode gets fresh exact tape replay. The replay reruns skill actions,
admission/failure memory and feedback from measured states instead of copying
them. Joint neural/memory/predictor evidence is independently recomputed by the
existing audit. All arm outcomes, infrastructure failures, unverified exits and
regressions are retained. Full-task native In AND Close is the primary endpoint;
paired rescue/harm, action/query and repair costs are reported. Six-root Wilson
intervals are descriptive, not independent held-out-task evidence.
