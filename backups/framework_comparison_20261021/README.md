# Framework v2: complete native microwave comparison

Frozen execution source: `ab48b63f0fc44ed4ecebc6c625045586e4c56688`.
Plan SHA256: `7fac280cd77cf1b2cd57e6594e7f923937860672bdf84f81928351f4c1d23c5e`.
Six reused development initial states; fresh bare-VLA/framework pairs;520queries
and520actions maximum per episode plus10initialization noops. Same frozen OpenVLA,
native inputs, initial states and endpoint. Verification uses privileged current/
past measurements; no learned predictor. Every task/repair numerical action is a
saved model output. Scoped prompts and the bundled controller revision are the
treatment, not an isolated verification ablation.

**12/12valid, exact-replayed episodes. Bare VLA4/6; framework4/6. Zero rescues and
zero harms.** The old root3harm disappears, but the old root4rescue is lost.
One repair meets the frozen local acceptance contract and actually hands back;
its recovered containment is lost on the first nominal action. It does not complete.

| Root | Bare VLA | Framework v1 | Framework v2 | v2actions per arm | v2repair actions |
|---|---|---|---|---:|---:|
|0|Failure|Failure|Failure|520|96|
|1|Success|Success|Success|444|0|
|2|Success|Success|Success|397|0|
|3|Success|Failure|Success|352|0|
|4|Failure|Success|Failure|520|80|
|5|Success|Success|Success|298|0|

Each v2arm uses2531queries/actions,5062combined plus120initialization actions.
Framework repair actions fall from352in v1to176in v2; this is not a50%reduction
in total VLA queries. There are6repair transactions:1verified,5unverified budget
exits,6actual nominal handoffs,0completed episodes after any repair handoff.

## Full paired videos

Baseline left, framework right; full endpoint tapes at native20fps, synchronized
simulation time with explicit end-frame holds. Overlays show active prompt goal,
fixed recovery contract, causal evidence, ownership, budgets, checks and handoffs.

| Root | Video |
|---|---|
|0|[Full comparison](videos/root0_baseline_left_framework_right.mp4)|
|1|[Full comparison](videos/root1_baseline_left_framework_right.mp4)|
|2|[Full comparison](videos/root2_baseline_left_framework_right.mp4)|
|3|[Former harm removed](videos/root3_baseline_left_framework_right.mp4)|
|4|[Verified local recovery, immediate handoff regression](videos/root4_baseline_left_framework_right.mp4)|
|5|[Full comparison](videos/root5_baseline_left_framework_right.mp4)|

Use **Download raw file** if GitHub does not play the video inline.

## What the trajectories establish

**Root3:** no intervention in v2. Its entire scientific trajectory matches the
fresh baseline, which completes at step351(zero-based). Release-transition
classification is active from198through351. This preserves nominal self-recovery;
it is not an example of successful model-generated repair.

**Root4:** early intervention is retained at148. V1/v2scientific trajectories
remain identical through155, then diverge at156: v1uses a close prompt while v2
uses a release prompt. V2attains In at161–163and accepts its fixed acquisition
contract via the contained alternative at163. Mug–gripper contact is still true
at all three endpoints, and the object moves6.571mm from162to163. The contract
does not test detachment, settling, independent support or nominal continuation
compatibility. At164the first actual full-instruction nominal action loses In;
165begins another acquisition repair. There is no later In attainment or full
completion in the520-action trajectory.

This is a correct implementation of the frozen **local containment** acceptance
rule, but evidence that this rule is insufficient to establish resumability. The
prompt change precedes handoff; its causal effect cannot be separated from exit
timing/acceptance using this bundled comparison alone. It would be unjustified to
claim that merely delaying handoff is proven to restore the old rescue.

**Root0:**3transactions use96repair actions without verified recovery or completion.
Failed-context gating suppresses some repeats but does not make the repair effective.

## Integrity and recovery

- [Original runner report](report.md) and [complete report data](report.json).
- [Post-run association/pairing audit](review.json): all5062model-query/action/input
  associations checked. All six fresh baseline trajectories equal their v1
  scientific trajectories. V2paired roots1/2/3/5are fully identical; roots0/4first
  diverge exactly at interventions254/148. Runtime timing/controller metadata are
  excluded from scientific-trajectory equality, not physical states or observations.
- Seven archives preserve the complete run and freeze, including model inputs,
  outputs/actions, physics/image tapes, archived sources, exact-replay proofs,
  supervisor logs and reports. Six videos are also separately available through LFS.
- `manifest.json` pins archive bytes and every member, plus the full videos.
  Remote recovery status is recorded separately when verification completes.

These are development results. They establish neither a net full-task gain nor
learned-prediction benefit nor verified-repair→handoff→full-completion success.
