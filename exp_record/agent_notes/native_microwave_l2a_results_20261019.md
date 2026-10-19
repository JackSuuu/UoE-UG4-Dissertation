# L2A v1 result: valid candidate bank, no failure-state local recovery

**Decision: stop v1; do not launch L2B.** The precommitted failure-root gain gate
is false. All18branches and fresh action replays pass; all six fresh greedy
controls reproduce their original suffix.552queries/actions were used of576.
The unused budget comes from native early completion, not missing/censored cells.

## Frozen horizon results

At12actions all18useful-progress labels are false. At32actions:

| Root | Original native outcome | C0greedy U32 | C1sample U32 | C2sample U32 |
|---|---|---|---|---|
|0|failure|false|false|false|
|1|success|true|true|true|
|2|success|true|true|true|
|3|success|false|true|true|
|4|failure|false|false|false|
|5|success|true|true|true|

Root3sampled branches finish the native full task after29/31local actions; original
nominal needs35, beyond the32-action window. This is faster completion of an
already-successful development case, not recovery of a failed task. Root5finishes
at25/26/25actions acrossC0/C1/C2. No snapshot-true native goal predicate is lost
within any observed local branch; this is not a downstream-harm certification.

## Source diversity and interpretation

Each root's first query uses byte-identical policy PNGs across candidates. Unique
first action/token counts acrossC0/C1/C2are1,1,1,3,3,2for roots0–5. Duplicates
remain charged and retained. Root0's first candidate collapses to the same action;
root4has three distinct actions but no useful32-action progress. Full-trajectory
analysis is needed before attributing failure solely to source collapse or timing.

This is a bounded negative reachability result for the frozen first-grasp+180
states, checkpoint, sampling distribution and32-action intervention. It does not
rule out earlier prevention, longer model execution or another model-generated
source. Any next experiment must freeze a new version and change one mechanism;
there are no new seeds/timestamps or automatic longer branches inside v1.

No online selector, learned prediction, verified repair, or actual greedy handoff
ran here. Future simulator labels remain offline-only. The result provides no
framework efficacy or predictive-verification increment.

## Evidence

Run: `/home/ami/scratch/native_microwave_l2a_20261019_v1`.
Independent offline audit: `/home/ami/scratch/native_microwave_l2a_20261019_v1_audit`.
Auditor SHA256`89c85d3266927ceadc8f49a10dd61dae4ca79913dc2aa1b24ba05cbce4a6ade1`.
`summary.json` retains all36horizon labels, source diversity, query provenance,
exact original/fresh controls, reconstruction/replay checks and the falseL2Bgate.
Full run/audit archives and paired candidate videos are being prepared, including
the negative failure-root cases.

Full bank now locally archived:18,132files in11bounded archives. The
[failure analysis](native_microwave_l2a_failure_analysis_20261019.md) distinguishes
root0's restricted translational support from root4's disengaged late-state
diversity. A separately frozen v2is being prepared with only snapshot timing
changed from first-grasp+180to+120; v1remains closed and its L2Bgate remains false.
