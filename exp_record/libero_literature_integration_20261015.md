# SEAL / CheckVLA integration audit — 15 Oct 2026

This audit accompanies [M1 batch failure discovery](libero_failure_mining_protocol_20261015.md). User requests established reasoning/verification mechanisms rather than another isolated ad hoc experiment. Their published success motivates a design; it is not inherited evidence for our H1actor/controller.

## Verified SEAL source

Official repository `NVlabs/actalign`, observed main revision **ed65d42e1466d9eb85ce8097f5034817a7e0ddc1**.

- [Pinned README](https://github.com/NVlabs/actalign/blob/ed65d42e1466d9eb85ce8097f5034817a7e0ddc1/README.md): reasoningπ0training/evaluation code, reasoning LIBERO datasets, pretrained reasoning checkpoint repositories still TBD.
- [Pinned live evaluator](https://github.com/NVlabs/actalign/blob/ed65d42e1466d9eb85ce8097f5034817a7e0ddc1/examples/libero/eval_libero_reason_sample.py): multiple model-sampled continuations in simulator environments; textual subgoal extraction; verification on initial/final agent/wrist images; selection of an accepted continuation and execution in the separate live environment.
- [Pinned verifier prompt](https://github.com/NVlabs/actalign/blob/ed65d42e1466d9eb85ce8097f5034817a7e0ddc1/examples/libero/prompt_template_verifier.txt): pick/place/close checks against the stated subtask, output semantic success/failure. The official caller uses `gpt-4o-2024-08-06`.

We examined this code, but do not import it, call external VLM APIs or install its π0environment into the shared runtime. Our frozen OpenVLA lacks its text-plan-conditioned stochastic action sequences. **M1 transfers the operational propose→predict→verify→execute pattern:** three bounded native controller candidates; replay-identical simulator forecasts; deterministic physical subgoal checks; actual selected-command execution; predicted versus observed outcome audit. Only the verified arm selects from candidates; others run the same forecasts in shadow under common caps. This is privileged, SEAL-inspired local verification, not an official SEAL reproduction and not a learned physics model.

## Verified CheckVLA source

[Paper2607.26789v1](https://arxiv.org/html/2607.26789v1) and [author homepage](https://scpsyl.github.io/) checked. Author homepage links the paper; official executable code/model release was not located. This is a search result limitation, not a claim that no release can exist elsewhere.

Paper components:

1. Separately trained/frozen action-conditioned short-horizon feature world model, repeatedly re-anchored to arriving observations.
2. Causal temporal discrepancy risk, with first-intervention functional conformal calibration on independent nominal-success episodes.
3. Same policy's flow-based suffix rewrite constrained by already-dispatched/latency-infeasible prefix, with calibrated-threshold exceedance controlling reference retention.
4. Event-driven pooled-feature keyframe bank read by the policy and summarized to the monitor; changed policy inputs are trained accordingly.

**M1 implemented:** selected-command→observed physical effect audit; contact persistence after commanded release; event-indexed phase/retention/release/clearance memory; causal transition-history gating; separate pre-decision evidence and post-execution outcomes. **M1 diagnostic only:** the new open/contact residual and16-entry memory bank do not directly trigger or condition the frozen actor. **Not implemented:** learned feature world model/risk head, conformal threshold guarantee, flow-rewriter, learned keyframe reader or official code. Existing `checkvla_official.py` remains an unused explicit skeleton, not evidence of integration.

The current actor is nativeH1and synchronously queried each continuation interval. It has no multi-action unexecuted suffix to rewrite and cannot receive a new trained cross-attention bank without changing the actor. Preserve historical genuineH10experiments separately. Do not introduce a weaker action-repeat baseline and then attribute its failure to missing CheckVLA.

## Attribution to test now

- Phase repair versus same-event hold/timed repair: usefulness of measured release/clearance-conditioned correction.
- Verified repair versus phase repair: added value of the transferred candidate-prediction-verification mechanism, under the same allowed forecast/query caps and actual cost reporting.
- All versus frozen VLA, especially sham: rescues and induced failures, ending at full drawer completion.
- Command/effect/event memory: inspect recorded proofs to diagnose trigger timing, prediction mismatch, exhausted repair and post-handoff regressions. A useful explanation is not itself task-efficacy evidence.

Controls deliberately compute candidate predictions in shadow to match available forecast evidence/budgets in this development comparison. This is experimental instrumentation overhead, not a requirement of vanilla OpenVLA. Runtime reports must separate actor inference, active algorithm computation, shadow computation, replay verification and rendering; raw worker elapsed time is not a deployment latency comparison. All3nominal roots are development examples, not enough to claim CheckVLA's5%episode-level conformal bound. A future conformal calibration protocol needs enough independent episode-level units and a separately fitted fixed score; repeated frames or variants do not create independent calibration episodes.

If useful physical state-based headroom is established, the next actual learned observation/prediction integration can be evaluated against this reference. Do not start more model training solely because the cited architecture worked in another setting.
