# Placement-preservation implementation and launch — 14 Oct 2026

The user authorized proceeding on this scene with an active framework diagnostic. [Frozen protocol](libero_preservation_protocol_20261014.md) and the runner/selector/verifier were committed and pushed as `9336782` **before** simulator branch smoke or registered rollout. This is accurate-physics selection in the action loop, not a learned predictor or historical H10/k5 efficacy study.

## Checks before launch

-7unit tests passed: four causal window/candidate/ranking checks and three existing completion tests.
-`smoke_libero_preservation.py` passed on consumed development root2. It independently replayed all190states up to the first causal opportunity (decision189) with exact equality to the original continuation. All six branches matched full MuJoCo integration state, controller/gripper/counters and camera hashes; the live environment was unchanged by forecasting. The selected first forecast equaled the actual next interval exactly. Controller scaling/axes assumptions were checked. Six5-interval forecasts plus complete prefix reconstruction took19.1s; this is an offline privileged diagnostic, not a real-time implementation claim.
-Smoke outputs: `/home/ami/scratch/libero_preservation_smoke_20261014_a.json` and `.log`. Candidate0was selected at this first opportunity; no correction constants or trigger were tuned from the outcome.

## Registered run

Run directory `/home/ami/scratch/libero_preservation_20261014_a`. Supervisor `scripts/launch_libero_preservation.py` starts three independent-session workers on identity-checked healthy physical GPUs1/2/3. Root allocation is0/3,1/4,2/5. Each runs VLA, hold-open, retreat-open and accurate-physics selection in that fixed order. Six roots×four arms=24trajectories. Root0/1/2are mechanism development;3/4/5are an additional frozen development cohort, not final tests. Source/protocol hashes, commands/PIDs and status live in run manifests.

**Launched after push of `b9d7a31`:** supervisor PID743321; GPU1/2/3worker PIDs743325/743326/743327. `launch.json` confirms all three workers were started after UUID/idle-memory checks. Supervisor stdout/stderr: `/home/ami/scratch/libero_preservation_20261014_a_supervisor.log`. Outcomes are pending; worker creation is not a completion or efficacy result.

All methods use nativeH1/k1 and the same520step budget, current-state trigger and six candidates. At most20correction opportunities occur after the first measured valid placement. Actual actor inference is preserved at every step. After first complete placement, correction switches off and up to40additional actor-only steps stress-test resumption. Original released-placement completion and secondary preservation metrics are reported separately.

On successful worker completion the supervisor automatically verifies recorded acceptance, candidate choices/ranking, forecast-first-step equality, cross-arm and historical pre-intervention prefixes, budgets, resumption, states/video lengths and hashes; it emits per-root four-method2x videos and an HTML report. A worker failure stops the remaining workers and is recorded, rather than silently omitting an arm.

The selector targets placement preservation, not early alignment. Root1may never trigger; that is an explicit coverage result. Only a benefit beyond **both** simple controls can support physics-specific headroom. Force safety and learned efficacy remain unmeasured.
