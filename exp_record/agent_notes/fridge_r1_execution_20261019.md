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

### Actual launch

Source/protocol freeze `080c9a4` pushed before launch. Detached supervisor PID3644620;
initial actor workers3644629/GPU1 (seed0 closed canonical) and3644630/GPU3 (seed1
closed skill; authoritative PID is in `launch.json`). Total planned12,160queries each.
No actor outcome is claimed at launch. The pending pixel-replay job remains separate.

[Engineering archive and labeled video](../../backups/fridge_r1_20261019/README.md)
retain205files, all failed geometry versions and exact-physics/nonexact-image replay
reports. v5's grasp predicate remains true for150consecutive control endpoints
(335–484); the final object has no shelf contact. This supports contact extraction
feasibility only. The actor cohort and pending replay are excluded from this snapshot.

### Pixel-replay completion

The background replay of `side_grasp_v5_pixels` completed with exact physics,
controller, diagnostics and every agentview image. Four wrist frames (steps92,104,
266,268) differ in6/3/9/12RGBchannel values, each by at most1. Total30channels.
The active actor uses agentview only. The source of this tiny wrist rasterization
variation remains unproven; do not call it a fully bit-exact replay or change the
acceptance tolerance. The archive was refreshed to include this completed report.
Replay image arrays are now decompressed once rather than once per frame; the slow
initial replay was backgrounded at the user's request. No running actor source changed.

### Remote recovery verification

Engineering/video archive pushed as`3c716fb`; a fresh repository under
`/tmp/opencode/fridge_r1_remote_restore_20261019` fetched GitHub/LFS bytes and verified
all208archive members and the standalone labeled video by SHA256. Proof:
[`remote_verified.json`](../../backups/fridge_r1_20261019/remote_verified.json).
User-edited `notes/Experiment_plan_1year.md` still hashes to
`d4617d1c65e2ef2ecbad3cb11c19753e0e2b9bbbafafeebeb5d7ccb57d55c83b`.
