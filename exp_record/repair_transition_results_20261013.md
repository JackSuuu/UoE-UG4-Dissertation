# Repair-transition completion and SEAL adoption assessment — 13 Oct 2026

## Completion and integrity

Run `repair_transition_20261013_a` completed **180/180 arm batches, 5,760 trajectories**, all20 validation cells ×9 arms ×32 episodes. These are640 paired roots, not5,760 independent samples; OOD n448, ID n192 per arm. All three detached launchers/workers exited. No experimental job from this run remains active.

The aggregate and all three shard summaries have `complete=true`. Completion review independently verified **180 NPZ SHA-256 hashes**, all saved episode vectors, violation flags reconstructed from live risk, executed-prefix/offline-label parity, and three shard-summary hashes. Production assertions also enforce direct-arm source replay, initial-root and pre-treatment matching, full actor cadence and gate/repair-query accounting. No protocol parameter changed after launch.

Aggregate: `src/results/push_torch/repair_transition_20261013_a/summary.json`, SHA-256 `596b4ed81e5a800efc04864bdbc1f937aaf7a19459cc39268145bbd5c456c9ea`.

## All-arm results

Pooled OOD n448/arm. SR=task success; CVR=any constraint violation; safe=success without violation. Percentages; safe rescue/harm is paired **against policy alone**.

| Arm | SR | CVR | Safe | Rescue/harm | Δsafe vs policy, pp [95% CI] |
|---|---:|---:|---:|---:|---:|
| Policy alone |87.05|29.24|65.63|—|—|
| Learned direct |84.60|28.79|64.73|8/12|−0.89 [−2.90,+0.89]|
| **Learned ramp (primary)** |84.82|26.79|66.74|10/5|+1.12 [−0.67,+2.68]|
| Learned flat |85.04|26.56|66.52|8/4|+0.89 [−0.67,+2.46]|
| Learned pause |85.04|29.24|64.51|9/14|−1.12 [−3.13,+0.89]|
| GT direct |80.80|8.04|77.46|53/0|+11.83 [+9.38,+14.51]|
| GT ramp |82.14|8.04|77.90|55/0|+12.28 [+9.60,+14.96]|
| GT flat |81.92|7.81|77.23|52/0|+11.61 [+9.15,+14.29]|
| GT pause |81.03|9.15|75.45|44/0|+9.82 [+7.59,+12.28]|

GT arms use privileged simulator risk and a physical event threshold. They are diagnostics, not deployable learned-method evidence, and do not isolate estimator quality from threshold semantics.

Registered paired comparisons (pp,4000 within-cell episode-pair bootstrap draws):

| Comparison | ΔSR [95% CI] | ΔCVR [95% CI] | Δsafe [95% CI] | Safe rescue/harm |
|---|---:|---:|---:|---:|
| Ramp vs learned direct |+0.22 [−0.67,+1.12]|−2.01 [−3.35,−0.89]|**+2.01 [+0.89,+3.35]**|9/0|
| Ramp vs policy |−2.23 [−4.24,−0.22]|−2.46 [−4.24,−0.67]|+1.12 [−0.67,+2.68]|10/5|
| Ramp vs learned flat |−0.22 [−1.12,+0.67]|+0.22 [0,+0.67]|+0.22 [−0.45,+0.89]|2/1|
| Ramp vs learned pause |−0.22 [−1.34,+0.89]|−2.46 [−3.79,−1.12]|+2.23 [+0.89,+3.57]|10/0|
| GT ramp vs GT direct |+1.34 [−0.45,+3.13]|0 [−1.34,+1.34]|+0.45 [−1.12,+2.01]|8/6|

ID n192: policy SR/CVR/safe94.79/2.60/92.71%; all four learned arms95.31/2.08/93.23%; GT direct/ramp/flat94.79/0/94.79%; GT pause94.79/2.08/92.71%. No learned transition difference in aggregate ID outcomes.

## What the ablation resolves

- Ramp restores **7/12 original safe harms** while retaining **8/8 original rescues**. Flat restores8/12 but retains7/8. Pause restores0/12 and retains7/8. The primary therefore passes the registered **incremental improvement over the original verifier** screen on this reused validation cohort.
- It **fails the broader framework screen**: +1.12pp vs policy rather than ≥3pp, the safe-success interval crosses zero, and SR loses2.23pp. No independent confirmation, training-seed robustness or simulated-arm transfer claim is justified.
- Ramp and flat are indistinguishable at this sample size. The experiment supports the tested repair-aware blending intervention; it does not establish that the gradual schedule is uniquely beneficial.
- Learned direct/ramp/flat/pause have16/7/6/7 first violations in pre-violation post-repair chunks. Their mean executed action jump is0.05563/0.01431/0.03860/0.01519 m/s. **Pause smooths and reduces local post-repair incidents, yet fails overall safe success.** Local smoothness or R5 hazard removal alone cannot substitute for task completion or later safety.
- Same-state raw→prepared R5 hazard removals on each arm's own pre-violation post-repair states are19/22/19 for ramp/flat/pause, with zero induced hazards in those scopes. Different trajectories and scope sizes (287/309/293 decisions) prevent causal interpretation of cross-arm conditional rates.
- The earlier release-only audit had zero first-step incidents; the broader all-post-repair scope has8 under learned direct and0 under ramp. These are different scopes, not conflicting results. Repeated repair still belongs in the diagnosis.
- GT ramp adds only+0.45pp safe over GT direct with an interval crossing zero and loses6 of53 GT-direct rescues while adding8. Thus continuity is not a general solution to residual repair/continuation failures under a stronger gate.
- Stress review flags cells1/2/3/7/19. At friction0.2, mass1/1.5/2, ramp still loses12.5/12.5/6.25pp SR vs policy; cell7 (friction0.6,mass2) loses3.125pp safe; cell19 (friction1.8,mass2) recovers much of the direct-arm safety loss but still loses3.125pp SR and safe vs policy. Each cell has32 episodes; these are review signals, not independent population claims.

**Decision:** retain frozen ramp as a promising component/control, preserve original direct as baseline, and stop broad smoothing sweeps. Detection and completion-aware candidate/recovery choice remain unresolved. Confirmation seeds16000–16019 are already consumed and cannot become fresh tuning data; final-test base46000 remains unused.

## Can we use the previously discussed SEAL approach?

**Yes to a targeted SEAL-inspired candidate/verification investigation; a full official SEAL deployment is not currently plug-and-play.** This is a feasibility/design conclusion, not a new experiment launch.

### Verified upstream status

Checked the [paper](https://arxiv.org/html/2510.16281v1), [project](https://yilin-wu98.github.io/steering-reasoning-vla/), [official policy README](https://github.com/NVlabs/actalign), and [benchmark README](https://github.com/NVlabs/Libero-10-r) on13Oct2026:

- Official policy and LIBERO-10-R simulation code are public. The project and policy README still list pretrained reasoning-policy checkpoints as **coming soon/TBD**; no ready checkpoint is supplied by those documented links.
- SEAL uses a reasoning π0 policy with text-plan-conditioned stochastic action sequences, interleaved autoregressive simulation up to the next think token, then VLM semantic outcome verification. The paper's simulation Predict stage uses parallel true environments. Its default K=10 and variable plan horizons/control loop differ from our H10/k5 protocol.
- Official instructions list full fine-tuning at>70GB single-GPU memory, with multi-GPU FSDP possible. Our three healthy24GB A5000s do not establish that this setup will fit or perform acceptably without a separate compatibility/memory test. This is not a claim that multi-GPU training is impossible.
- Local base currently has MuJoCo, but no LIBERO, robosuite or RoboCasa. Our policy is a deterministic H10 planar head on full OpenVLA; repeating it does not produce diverse hypotheses, and it has no trained text-plan/action switching interface. Installing official requirements wholesale would conflict with the standing additive shared-base setup; compatibility must be audited first. Latency is deferred and is not the adoption veto.

### What was already tried versus what remains new

The previous K=8 scale/progress experiment tested only collinear amplitude changes. It gave safe success64.73/64.73/65.18% for bisection/discrete/progress versus65.63% policy, with no established net benefit. It did **not** test genuine steering/re-contact alternatives or plan-conditioned stochastic SEAL proposals. Conversely, merely increasing K or adding a VLM does not correct a learned physics gate that misses force hazards.

The useful hypothesis now is: **a recoverable alternative may require a different short trajectory, not another smaller forward command; acceptance must preserve the ability to resume the policy and complete the task.** Current results motivate that question but do not prove the alternative exists or is learnable.

### Recommended next bounded study (design sketch, not frozen registration)

1. Keep the full actor/H10/k5 and freeze a **K=8** candidate budget. Contrast the old eight-scale set with an equally sized, explicitly specified non-collinear recovery set (original, braking/hold, lateral correction or retreat/re-contact). Numerical schedules and fresh training-only roots must be registered before collecting labels. Handcrafted residual candidates must be named as such, not advertised as samples from a stochastic OpenVLA.
2. First audit **candidate availability and policy resumption** on paired training roots: exact simulator rollout may diagnose whether any candidate avoids R5/H10 hazards and still allows actual policy continuation to complete. These branch labels remain diagnosis-only. A short safe chunk or smaller predicted goal distance is not a continuation-value label.
3. If alternatives exist, compare learned versus privileged selectors on the **identical frozen candidate set**, with matched total risk/state queries, continuation/control opportunities, fallback and tie-breaking. Separate candidate-family effects from selection effects; report minimum-change, risk-filtered progress and any proposed recovery criterion separately. Raw original and ramp baselines should both remain explicit controls.
4. If the oracle candidate/continuation witness fails, revise the action family or control formulation rather than train a larger verifier. If witnesses exist but learned ranking fails, target risk/progress/observable-history learning on separately collected, permitted training targets; do not silently train on the diagnostic branch labels. Event-gate R5 work remains scoped separately from the unresolved H10 anticipation coverage.
5. For everyday manipulation, official LIBERO-10-R environments can support an independent baseline-feasibility track. A later hierarchy could use semantic plan verification for task progress and our physics verifier for contact constraints. Official semantic verification neither measures hidden contact forces nor establishes physical safety; formal verifier transfer still requires learned efficacy.

This makes the proposed contribution **physics- and continuation-aware local verification/repair under a fixed budget**, rather than reproducing SEAL's best-of-N scaling or substituting semantic checks for physical detection. No checkpoint training, dependency installation, or new GPU job was launched during this completion review.
