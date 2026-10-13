# Preregistered R5 gate-objective comparison — 13 Oct 2026

## Question and scope

Does event supervision improve intervention selection and closed-loop safe completion over matched risk regression, with original repair frozen? This is the sole active experiment under [the mainline decision](research_mainline_20261013.md). It is a **validation study**, not independent confirmation, and does not establish H10 anticipation or transfer to a new precision task. Commit this protocol and implementation before production collection. Smoke fixtures are explicitly excluded.

## Fixed components and information

- Complete OpenVLA-7B `llm` actor, frozen DAgger×2 deterministic H10 world-frame planar head `openvla_chunk_head_llm_h10_d2.pt`, inference microbatch8. Same `matched_open_loop_v2`, k5 execution,80 steps/episode,16 policy calls and512 policy rows per32-episode batch. No intermediate replanning for any arm.
- Original `orbisim.pt` remains the **repair** predictor for every learned arm. Six bisection iterations, floor.05, target min(.8,.75×original repair score), responsiveness.02, abstention returns original; no gradient/hard-prefix. One gate risk query per decision and eight repair risk queries per triggered row, asserted. Total triggered rows may differ as the gate changes; report them, rather than pretending they are identical.
- Gate inputs: measured obs(10), previous-step obs(10), actual previous action(2), candidate action(2). No hidden mass/friction, exact state, GT risk, future observations or oracle labels enter runtime. Offline labels use exact simulator physics, with live-state immutability and executed-prefix parity checks. Old audit/branch/validation labels are not training data.
- Frozen source metadata/artifact hashes come from `sequence_risk_20261012_a/status.json`. Fresh metadata hashes this protocol, implementation and helpers; stage transitions fail on changes. Legacy models and calibrations are never overwritten.

## Bounded new support collection

Fresh training-only-domain roots: train76000–76003 (128roots), dev77000–77001 (64), calibration78000–78001 (64),32 envs per batch. Physics sampled once/root using unchanged `PREDICTOR_TRAIN_RANGES['push']`; same initial states/physics across the three collectors. Total **24 batches /768 trajectories**. No second collection round, threshold relaxation or physics-range expansion under this protocol.

Three collectors, fixed from prior diagnosis before new outcomes:

1. `none`: original policy alone.
2. `frozen`: original learned H10 gate with frozen decision tau and original repair.
3. `brake_ramp`: once per episode, at first decision with0<observed goal gap≤.20m, multiply proposal steps1–5 by(.85,.70,.55,.40,.25); steps6–10 also×.25 in the collector's cached proposal, but only the first5 execute before normal replanning. No subsequent perturbation. This is a **handcrafted data collector**, not a proposed deployed repair; no GT-based action selection. It targets the previously observed immediate resumption failures rather than another anticipation sweep.

At each actual decision record the original unmodified H10 proposal, measured context, previous-violation flag, root ID, collector, physical strata for offline composition only, and labels. The existing eight-scale label helper is retained unchanged for audit parity; **only original(scale1) labels enter either training objective**, never scale selection. Applied-plan labels are checked against the next five executed steps. Keep all branches of a root in one split; normalizers use eligible training rows only. Exclude every row after any executed violation; include the original decision immediately before a violation. Persist excluded rows for audit, never fit on them.

Support must pass separately for train/dev/calibration:

| Criterion (distinct roots, not decision count) | Train | Dev | Calibration |
|---|---:|---:|---:|
| Original R5 event: max risk in steps1–5>1 |64|32|32|
| Safe contact: .1≤max H10 risk≤1 |64|32|32|
| Each above class, each of4 fixed physics bins |8|4|4|
| Policy-alone roots with an R5-safe decision |64|32|32|

Bins use friction multiplier split1.0 and mass multiplier split1.1; bin counts are label-composition checks, never model inputs. Report each collector separately and H10-only events; **no H10-only minimum is asserted or reinterpreted as met**. A root may contain both safe and unsafe decisions; counts are support checks, not independent positive/negative samples. If any criterion fails: save `blocked_insufficient_support`, stop all training/evaluation, report the exact deficit and revisit scope explicitly. CPU smoke may bypass support only with an explicit smoke flag and separate output; it never authorizes production.

## Matched objectives and training budget

Both arms use the original residual-MLP topology:5-member ensemble, hidden256,3 residual/SiLU layers,24 input dimensions,12 output dimensions. Retain the legacy output shape; only the contact-force output(index10) is trained/read in both arms. State-delta and deformation outputs are unused. Assert deformation risk is zero in collected task labels; no silently unmodelled second constraint. A static context is reused across the first5 proposed actions; this deliberately retains the known order-blind architecture for the loss comparison.

- **Regression:** per-step MSE(softplus(logit), min(true risk,3)).
- **Event:** per-step unweighted BCEWithLogits(logit, true risk>1).
- Seeds0/1/2, paired identical random initialization and exactly shared minibatches. Neither warm-starts from the original predictor. This isolates the objective within the two new arms; new-vs-frozen is a broader training-procedure control, not a pure data-only causal contrast (normalization, sampling and auxiliary objectives also differ from legacy training).
- Each minibatch256 original decisions:128 R5-event decisions and128 safe decisions. In each class draw a root uniformly, then an eligible decision within it uniformly; replacement allowed. Both objectives receive this same sampler. This changes the training prior; event outputs are ranking scores, not calibrated physical probabilities. No candidate augmentation or per-seed best-model selection.
- Exactly3000 updates, AdamW lr1e−3×(.5+.5cos(πstep/3000)), weight_decay1e−5, gradient clip10, no early stopping or dev-based hyperparameter/epoch choice. All3 final checkpoints are evaluated.
- Normalizers fit only eligible training obs and measured obs differences; original input conventions retained. Final score=max over the5 steps of ensemble mean+1×std (sample std), in the arm's own risk/probability units.
- Per-arm threshold: `calibrate_tau(alpha=.05)` on **R5-safe original policy-alone calibration decisions**, pooled over eligible roots. Use no unsafe labels to choose threshold. These dependent trajectory decisions and shifted deployment inputs do **not** establish a conformal OOD safety guarantee. Report calibration counts, thresholds, dev recall/safe-trigger rate with4000 root-bootstrap draws, and collector-stratified/H10-only diagnostics. Dev results do not determine whether an arm receives closed-loop evaluation.

## Closed-loop comparison and accounting

9 arms: policy-alone, original frozen H10 gate/tau, frozen original model restricted toR5 and freshly calibrated by the same R5 rule, regression s0/1/2, event s0/1/2. All gates use the original frozen H10 repair. The frozen-R5 arm exposes horizon/calibration changes; it does not isolate horizon from threshold changes. Previously completed ramp is an external component reference, not combined with the new gate.

20 original validation cells×32 episodes×9 arms=**180 batches /5760 trajectories**, seeds26000–26019 (already consumed validation),14 OOD cells/448 episodes and6 ID/192 per arm. Shard whole cells round-robin across the3 healthy A5000 UUIDs. Full OpenVLA is loaded separately on each GPU; never use physicalGPU0. Verify none/frozen episode parity against prior evaluation and live-risk replay against completed factorial results. The confirmation16000 cohort and unused final-test46000 cohort remain untouched.

All arms retain16 policy calls and512 policy rows per cell. Gate scores one query/decision (R5 or legacyH10 as declared); repair gets8 H10 queries/triggered row. No GT runtime feedback. Persist episode outcomes and live-risk/trigger/score traces; verify hashes, violation and first-violation reconstruction before aggregation. Report per-cell and pooled SR/CVR/safe, paired rescue/harm, task cost, stall and trigger statistics. Paired4000-draw within-cell bootstrap gives SR/CVR/safe deltas vs policy and event vs matched regression/frozen, conditional on this grid; it is not task-level generalization or multiplicity-corrected evidence. Never average the3 seeds as independent episodes.

## Predefined decisions

- Objective point-direction support: event−regression safe delta>0 for **all3 seeds**. Stronger within-validation support: each seed's paired95% lower>0 and each SR delta≥−2pp. An offline detection improvement alone fails task-level support; evaluate all seeds regardless.
- Framework candidate screen: each event seed vs policy has safe delta≥+3pp, paired95% lower>0 and SR delta≥−2pp; inspect all per-cell safe losses and>2pp SR losses. Stress regressions require review. Even passing is only permission to propose a frozen independent confirmation; second-cohort/training-seed robustness and actual everyday precision transfer remain required, not inferred.
- If event adds no benefit over matched regression, reject this tested objective as a demonstrated solution; do not automatically enlarge the network. If offline detection improves but closed-loop completion does not, local risk detection is insufficient under this repair/control formulation. Specify that limitation before any new study.
- No automatic threshold sweeps, collector reruns, GRU/SEAL launches, ramp combinations or independent-test consumption follow this study. No runtime/kernel work is included.

## Execution

`scripts/run_immediate_gate.py` coordinates collection → support →3 paired training jobs →3 evaluation shards → integrity/aggregation. Each stage waits for worker exits; failed support stops cleanly before training. Launch using `setsid nohup … < /dev/null`, with logs under scratch and artifacts in the ignored result directory. Register launch PID/command separately after the protocol commit. Tests exercise R5 isolation, frozen-repair separation, prior/split restrictions and matched sampling; CPU smoke tests the entire pipeline without using full OpenVLA or experimental roots. Smoke training uses an explicitly marked in-memory synthetic two-class fixture, because tiny BC rollouts need not contain any unsafe decisions; persisted collection labels stay intact and no smoke result counts as efficacy evidence.
