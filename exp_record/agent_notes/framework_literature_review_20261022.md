# Frozen-VLA verification and recovery: literature and code review

Date:22October2026. Main first-submission window:22October2025–22October2026.
Older, especially relevant implementations are labeled separately. This is a
source inspection, not a reproduction of any external paper's reported gains.

## 1. Current result and architectural gap

The six-root native microwave baseline succeeds4/6; full joint succeeds2/6.
Measured-only admission with joint computation and handoff checks retained
returns to4/6, with0rescues /0harms versus baseline. It removes the two observed
joint harms but does not establish effective repair.

Four subsequent full diagnostic episodes on failed development roots0/4 all
fail: stage routing uses160repair actions and progress exit uses121, with
zero verified repairs in either mode. Every episode consumes520actions.
See [results and validation](framework_admission_and_repair_20261022.md).

The missing interfaces are:

1. A generator that offers genuinely useful recovery actions from failure states.
2. A verifier that distinguishes the consequences of different proposed actions.
3. Handoff evidence that predicts preservation and useful nominal continuation,
   beyond instantaneous local acceptance.

The existing observed-motion risk head is not an action-conditioned evaluator.
Repeating the same deterministic H1 query at an identical state does not create
a diverse candidate set. Stage routing can change actions but has not produced
task-level rescue here.

## 2. Recent work and concrete reuse opportunities

Dates below are arXiv first submissions. Venue labels are included only as
reported by the linked official sources; publication timing and code release
timing are distinct.

### A. CoVer-VLA —12February2026

- [Paper](https://arxiv.org/abs/2602.12281),
  [code](https://github.com/cover-vla/cover-vla).
- Mechanism: score image/instruction/action-history-plus-future-chunk alignment,
  then select among sampled action and instruction candidates. This is action
  conditioning, not a calibrated physical-success probability or a dynamics model.
- Inspected `bridge_verifier/ensemble_eval/model.py`,
  `efficient_ensemble_merged.py` and the SIMPLER OpenPI evaluator.
  `predict(image, instruction, possible_action_histories)` encodes candidates,
  computes similarity scores and selects `argmax`.
- Reuse priority: high for the candidate-scoring interface, action encoder and
  shared visual-feature batching. The README provides a Bridge verifier download
  (`cover-vla/cover-vla-bridge`); weights were not downloaded or evaluated here.
- Compatibility: released example uses π0/Bridge/SIMPLER; our LIBERO/Panda action
  normalization, camera distribution, H1 and failure-state data need adaptation.
  Its example budget is8rephrases×5samples, not our one greedy query per action.
  We should borrow action scoring without making rephrasing the contribution.
- Important implementation attribution: evaluator lines372–390 apply gripper
  voting and overwrite the selected gripper command. Copying this executor would
  not preserve our unchanged-proposal action contract. Its similarity threshold
  also should not be imported as a calibrated threshold for our domain.

### B. CARE —21September2026

- [Paper](https://arxiv.org/abs/2609.24118),
  [code](https://github.com/xiaojunlan/care).
- Mechanism: fit stage-conditioned deviations from actual failed rollouts,
  synthesize representative physical failure states, collect short corrective
  demonstrations, train the executor, then invoke corrections through3D monitoring.
- Inspected `envs/put_object_cabinet_regrasp.py` and
  `policy/pi0/eval_vla_put_object_cabinet_pointcloud.py`. The former uses sampled
  grasp offsets and motion-planning correction; the latter uses a task-specific
  `RulePlanner`, point-cloud checks and remote policy queries. Stage changes can
  interrupt the queued chunk and cause a fresh request.
- Reuse priority: high for failure-driven recovery-data collection and stage/
  preservation labels. It addresses our absent recovery supervision directly.
- Compatibility: CARE trains the VLA executor on corrective data; it does not
  demonstrate that our frozen OpenVLA can recover merely by receiving its
  instructions. For our framing, a separately trained recovery generator would
  be an adaptation, not a reproduction of CARE. Task-specific regrasp routines
  are data-generation oracles, not evidence of general learned online repair.
- Release limitations: README explicitly says code-only, without assets/data/
  checkpoints; distribution-fitting export/command and the complete fixed FSR
  initialization manifest are absent. Evaluation expects an HTTP policy server
  different from the bundled WebSocket server. Source available is not equivalent
  to ready-to-run reproduction.

### C. VLA-Corrector —2July2026

- [Paper](https://arxiv.org/abs/2607.01804),
  [code](https://github.com/ZJU-OmniAI/vla-corrector); README reports NeurIPS2026.
- Mechanism: external action-conditioned latent residual model, predicted versus
  observed latent-motion discrepancy, stale-chunk truncation and online gradient
  guidance for the recovery replan; backbone weights remain frozen.
- Inspected `src/siglip_dynamics/MLP.py` and inference
  `circuit_breaker.py`, `guidance_injector.py`, `safety_module.py`.
  `SiglipResidualMLP` takes visual latents and actions and predicts latent change.
  The inspected breaker uses cosine discrepancy, median+MAD and cooldown;
  the guidance helper exposes a cosine loss, not by itself a complete OpenVLA
  recovery generator.
- Reuse priority: high for a compact `f(z_t,a_t) -> delta_z` design and transition
  logging; moderate for monitoring. Predictive mismatch does not establish
  goal failure, and accurate latent prediction alone does not rank useful repair.
- Compatibility: our H1 already reobserves/requeries every step, so stale-chunk
  truncation supplies no analogous horizon reduction. Flow-policy guidance is
  not a drop-in interface for discrete autoregressive OpenVLA. Frozen weights
  also do not imply unchanged actions under guided inference.
- README explicitly excludes corrector and fine-tuned-policy checkpoints and
  datasets. No external model was trained or run in this review.

### D. Mostly Harmless VLA Steering —10June2026

- [Paper](https://arxiv.org/abs/2606.12299),
  [project](https://hyunjoe.xyz/LanguagePolicy); revised16October2026.
- Mechanism: search useful language-feedback sequences, distill a feedback
  policy, learn an improvement head and conformalize the decision to intervene
  versus retain the original instruction. Backbone remains frozen.
- Reuse priority: high conceptually for **benefit relative to nominal execution**
  as the admission target. A detector of current trouble cannot tell whether the
  proposed intervention will help more than continuing nominally.
- Project still says `Code (coming soon)` at inspection. Calibration assumptions
  and data requirements must be checked before interpreting any harm guarantee.
  This review does not claim universal protection under arbitrary distribution shift.
- Its language-feedback generator is not the recommended main repair mechanism
  for our project; the comparative improvement target is the transferable idea.

### E. ViFailback —2December2025

- [Paper](https://arxiv.org/abs/2512.02787),
  [code](https://github.com/x1nyuzhou/ViFailback); CVPR2026 official repository.
- Diagnoses failures and produces textual and visual-symbol corrective guidance.
  Inspected `vifailback_infer.py` and README: released utilities perform VLM
  inference, parse symbol commands and render overlays. Model/data links are
  provided for ViFailback-8B and the failure dataset.
- Useful for structured failure labels and inspectable correction specifications.
  Diagnosis quality and useful robot execution must still be evaluated separately.
- The paper's visual-symbol-following method fine-tunes π0.5 with symbol-following
  trajectories and task demonstrations, including image masking. Overlaying arrows
  on our frozen OpenVLA inputs is not an established substitute for that training.
  The inspected release is not a complete frozen-OpenVLA recovery loop.

### F. Foresight —22June2026

- [Paper](https://arxiv.org/abs/2606.23085),
  [project](https://haoranzhangumich.github.io/Forsight_web).
- Uses action-conditioned world-model latents and final task success/failure
  labels for long-horizon failure detection, with functional conformal calibration.
- Relevant to replacing our action-independent risk representation and learning
  from endpoint labels. Candidate ranking and corrective action generation are
  additional requirements, not supplied merely by having a failure detector.
- Project says `Code (TBD)`; method reference, not currently an inspected reusable
  implementation. Models mentioned include V-JEPA2-AC; no local compatibility
  or throughput result was obtained.

### G. AgentChord —12May2026

- [Paper](https://arxiv.org/abs/2605.11951),
  [code](https://github.com/EDEM-AI/AgentChord); README reports RSS2026.
- Inspected hierarchical recovery/compile agents, monitoring functions and the
  `SinglePourWater/agent_recovery_spec.json` example. Recovery bindings associate
  a task-graph edge, a failure monitor, recovery operations and `merge: source`
  or `merge: target`; examples include `repeat_until_success`.
- Useful representation of which stage to resume and what must remain valid.
  The inspected example uses concrete `regrasp`/relative-motion skills. It does
  not supply a general frozen-VLA numerical recovery generator; copying those
  skills would return us to task-specific scripted execution. Bounded retries
  and our measured handoff/continuation tests remain necessary.

## 3. Older work with unusually relevant code

### RoboMonkey —21June2025, outside the main one-year window

- [Paper](https://arxiv.org/abs/2506.17811),
  [code](https://github.com/robomonkey-vla/RoboMonkey).
- Particularly close engineering precedent: OpenVLA serving, test-time sampling
  and a separate learned action verifier. README includes OpenVLA/SGLang and
  SIMPLER evaluation commands and model links.
- Inspected `monkey-verifier/src/infer_server.py` and `action_processing.py`:
  `get_rewards(instruction,image_path,actions)` explicitly evaluates candidate
  actions with a LLaVA-based reward model. The tokenizer discretizes numeric
  actions with its own binning; its action/camera assumptions are not our Panda
  action mapper. No claim of checkpoint transfer or local runtime success.
- Strongest direct reference for the sampling→score→select interface around
  frozen OpenVLA. A7Bverifier is relatively expensive; adapt the interface before
  choosing model size. Positive-temperature candidate generation would change
  our current greedy decoding protocol, despite preserving policy weights.

### FailSafe —2October2025, slightly outside; revised7July2026

- [Paper](https://arxiv.org/abs/2510.01642),
  [code](https://github.com/Jimntu/FailSafe_code),
  [project](https://jimntu.github.io/FailSafe/).
- Inspected `examples/ex_failgen_data_collection.py` and `data/sanity_check.py`.
  The latter invokes task-specific sanity solvers and saves correction examples
  only when the returned rollout `success` is true. Labels couple task/stage,
  failure type, image history and corrective end-effector displacement.
- Useful principle: label **executably verified correction**, rather than only
  plausible diagnosis text. Its sanity solvers and ground-truth trajectories
  are privileged offline supervision, not learned online deployment capability.
- The code contains unresolved `TBD` path placeholders and task-specific solver
  mappings for pick/push/stack. Adaptation to LIBERO is substantive. Successful
  solver continuation is also not proof of successful continuation by our frozen VLA.

## 4. Recommended smallest next contrast

This is a proposal, not a launched experiment or frozen protocol.

**First establish recoverable candidate coverage, then learned selection.**

1. Use development-only failure checkpoints and successful-control checkpoints.
   Restore exactly matched state and policy history. Keep held-out task/scene/
   root groups separate from fitting, calibration and model selection.
2. Keep the OpenVLA weights frozen. Explicitly test whether a small fixed set of
   policy-sampled actions under the same task instruction offers useful recovery.
   Include the greedy proposal. Sampling must be an opt-in experimental decoding
   change; repeating identical greedy queries is not a candidate generator.
   Count all proposals, verifier inference and rollout costs. Freeze a matched
   total resource budget before comparing any online task success.
3. In an offline diagnostic only, execute matched short branches and then resume
   the nominal VLA. Label local acceptance, preservation, verified exit, actual
   handoff, stable continuation and complete native success separately. Branch
   rollouts provide labels / an oracle candidate upper bound, never an online
   selection mechanism or deployable performance claim. Recoverability must
   follow from executed trajectories, not near-zero action error alone.
4. If useful candidates exist, compare the **same candidate pool** under fixed/
   random selection versus a learned action-conditioned scorer. Borrow CoVer's
   input/batching interface and RoboMonkey's policy/verifier separation. Train
   against our recovery/continuation outcomes; include nominal continuation as
   a comparator instead of treating current-state risk as intervention benefit.
5. If the pool lacks useful recoveries, a better scorer cannot fix it. Borrow
   CARE/FailSafe's verified-data design to train a separate bounded recovery head.
   This retains frozen OpenVLA but is a new learned numerical-action generator,
   explicitly distinct from unchanged-VLA execution. Data-generation oracles
   must not become hand-coded runtime release/regrasp scripts.

An action-conditioned physical predictor can then contribute to the verifier:
compare measured-only scoring against measured+predicted consequences with the
same generator and candidate set. This tests the dissertation's physical-
prediction component instead of attributing generator or geometry-controller
gains to prediction. Handoff checks should measure retained task predicates under
actual nominal continuation, not equate a local pass or budget exit with recovery.

After offline feasibility, paired full episodes must include nominally successful
roots to detect harms. Report complete-task success, paired rescues/harms, query/
action/latency costs and infrastructure failures separately. The current two
failed roots are mechanism-development evidence only.

## 5. Inspection provenance

Read-only download of25source files from7repositories, with exact commit and
SHA256 per file, is retained at:

`/home/ami/scratch/framework_literature_20261022_r1/manifest.json`

| Repository | Inspected commit |
| --- | --- |
| cover-vla/cover-vla | `21a9960639a8340c1e8511fb04776fb11271f246` |
| xiaojunlan/care | `a0a5b90632f9e06143afb033b2894ceb79fbc9bb` |
| ZJU-OmniAI/vla-corrector | `e02ab51f030d1e25c1ca1b5642667730d2e495d3` |
| EDEM-AI/AgentChord | `74fe7dd756964b737952d26e9e65d4d5e2e29ec3` |
| x1nyuzhou/ViFailback | `3be62c639b1a5a59bea201d646309d225eb00db9` |
| Jimntu/FailSafe_code | `b863aaecc50407a769c191f67d79ddfaab9f6880` |
| robomonkey-vla/RoboMonkey | `db9f8d314ea78e986f515617ac43afc58f984a53` |

Primary arXiv pages were checked for titles and submission dates; project and
paper pages were used to distinguish released code from promised code. Repository
existence, inspected source, advertised checkpoints and reproduced results are
different evidence levels. No external repository was installed or executed,
and no claimed external benchmark improvement was independently reproduced.
