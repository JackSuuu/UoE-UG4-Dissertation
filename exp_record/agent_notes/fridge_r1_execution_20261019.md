# R1 refrigerator capability diagnosis — execution record

Protocol: [closed-loop recovery and R1](closed_loop_recovery_r1_20261019.md).
Engineering output: `/home/ami/scratch/fridge_r1_20261019_a`.
Actor cohort target: `/home/ami/scratch/fridge_r1_actor_20261019_a`.

## Engineering observations before actor trials

* R0 opening is world +x while robot base is x=-0.66 m: the robot approaches the
  back wall. The original downward gripper also conflicts with the upper shelf.
* Enabled the existing hinge limits in a process-local asset construction patch;
  closed initialization is zero. Upstream asset and R0 code/protocol remain unchanged.
* A UUID-valued CUDA visibility exposed robosuite's numeric EGL validation on one
  launch; retained its failed status. Subsequent runs verify GPU UUID identities but
  use matching numeric CUDA/EGL device IDs. GPU0 is excluded.
* Robot-facing y=0.15/0.25 layouts with a widely open door can contact the reset hand.
  Retained these geometries. The chosen y=0.35 layout and 130-degree opening avoid
  that initial interaction and expose the opening with a robot-side oblique camera.
* Servo v1 closes the door accidentally. v2/v3 fail to reach the requested sideways
  orientation/position. Offline IK shows exact pose solutions can intersect the table
  or fridge, so low IK error is not collision-free reachability. Opposite wrist roll
  gives reachable outside approach poses. v4 reaches the object but palm/door contact
  prevents stable pad grasp.
* Item moved toward the lip (+0.04 m local x) and away from hinge (+0.035 m local z).
  v5 achieves bilateral pad grasp, actual lift clear of the shelf, and outward
  transport by **0.173744 m**, retaining grasp at the final step. Door angle remains
  unchanged. This qualifies one contact extraction diagnostic, not nominal VLA
  capability or complete refrigerator-task success. All five servo versions retained.

Initial independent replay of v4/v5 has **exact physics/controller and diagnostics**,
but a few image fingerprints differ. This is an unresolved visual determinism issue,
not an exact full replay pass. Added raw agent/wrist image arrays to subsequent runs
to measure camera/channel differences rather than silently relaxing tolerances.
The first verifier import also exposed a missing transitive dependency in the early
source freeze; it is explicitly hashed as a supplemental dependency for those runs,
and all new runs freeze `libero_completion.py` directly.

## Actor experiment

The scene/prompt/seed/query-budget contract is frozen before actor outcomes. Twelve
fresh-reset runs: seeds0–2 × closed/open × canonical/skill instruction, 160 queries
each. No scripted prefix, repair or task training. Full-completion fields stay null;
record door progress, object displacement, grasp/contact and latency. A detached
supervisor runs one worker per selected healthy GPU and independently replays each tape.
GPU1/3 are selected for the initial launch while the existing pixel-replay diagnostic
continues independently on GPU2; seeds0/2 are serialized on GPU1 and seed1 on GPU3.
Worker timeouts are900s, replay180s. Completion/replay issues remain distinct from
manipulation failures. This record will append actual launch and outcomes.
