# Local VLA scope: precision tasks, noisy observations and physical feedback

## User-confirmed scope

The intended long-term framework can sit between a task-generating model and embodied execution. **The present study is local VLA execution only.** Remote LLM planning, network/cloud scheduling and a model-agnostic planner demonstration are future work. A successful VLA instance would support the physical-feedback principle at the execution layer; it would not by itself prove LLM or cross-platform generality.

The implementation and optimization method may change as evidence requires. The fixed objective is improved **complete, precise and safe task execution**, with matched information/control budgets and independent evaluation. Neither fine-tuning itself nor a changed action is evidence of improvement. Start with a frozen task-capable VLA and train/adapt the physical estimator/verifier; actor fine-tuning, if later needed, must have its own actor-only control so that contribution attribution remains possible.

## Task shortlist, checked against public definitions

Select one primary task for the first closed loop; the remaining entries are subsequent coverage candidates, not parallel launches. This refines the earlier drawer-opening integration target into a precision-oriented choice. A simple drawer-opening rollout remains an optional interface smoke, not the main precision contribution.

| Role | Concrete candidate | Physical mechanism to evaluate | Existing definition and remaining qualification |
|---|---|---|---|
| First everyday-task candidate | LIBERO-10 `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` | Placement alignment, edge contacts, release and stable placement in the correct compartment | BDDL confirmed: `(In black_book_1 desk_caddy_1_back_contain_region)`. Actual clearance/collision geometry and orientation/stability criteria still need inspection; “In” alone is not a millimetre insertion metric. |
| Second everyday-task candidate | LIBERO-10 `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` | Object placement followed by articulated contact and closure; completion must include both stages | BDDL confirmed: drawer initially open; goal is `(Close ...) AND (In bowl ...)`. Gentle closure, force limits, pinching/jamming and retained containment require explicit mechanically supported measurements. |
| Later controlled mechanism task | robosuite `NutAssemblySquare` | Translational/orientation alignment and contact while placing a square nut around its peg | Class confirmed in upstream source, but default `on_peg` uses3cm XY thresholds and a height check plus gripper separation in `_check_success`; not a strict precision/safety certificate. Need collision/pose/depth/stability validation and a task-capable actor; no task-matched OpenVLA checkpoint verified here. |

Do not promote `put_the_wine_bottle_on_the_rack` as a narrow-slot insertion merely from its name: the checked LIBERO-Goal BDDL uses `(On wine_bottle_1 wine_rack_1_top_region)`. Similarly, do not claim a connector, snap-fit, compliant seal or real electrical operation without supported mechanics. Any modified clearance or precision-success requirement is an explicit custom variant, not an unchanged official benchmark. Do not manufacture verifier gains with arbitrary force thresholds or perturbations selected after outcomes.

The official `openvla/openvla-7b-finetuned-libero-10` checkpoint is public and ungated, revision `80970322773f81baa2e22fe495d0487b93a05cfa`, about15.08GB BF16 weights. Its existence supports integration feasibility only; no local success rate has been measured. The prior audit's additive-dependency, native-action/preprocessing and H1-versus-genuine-H10 compatibility issues remain open.

## Framework and noise treatment

Proposed local flow (not an implemented/validated new architecture):

`measured observations/history → estimated contact state + uncertainty`

`frozen VLA proposals → short-horizon physical prediction → task-aware accept/local repair → execution → new observations`

Task geometry/goal constraints enter prediction and acceptance. Ground-truth simulation state and hidden mass/friction do not enter the learned runtime path. A privileged exact-simulator diagnostic remains explicitly separate. The predictor should distinguish alignment/progress and likely force/contact evolution; an event score alone is insufficient to assess whether resuming the actor completes the task. Differentiability is a possible repair mechanism and still requires usable-gradient evidence near contact, not an assumption from the existence of autograd.

Noise cannot simply be removed. Separate mechanisms before combining them:

- **Observation error:** object pose/image noise, occlusion and measurement latency. Define the perception path; adding noise to privileged object state is a controlled sensor proxy, not proof of visual sim-to-real perception. Use measured history/filtering and uncertainty where supported, with the same input access available to controls.
- **Action execution error:** pose-command bias, control lag and tracking error. Apply at the actuator/controller interface, and measure the actual executed response.
- **Dynamics uncertainty:** friction, mass and contact compliance/model mismatch. Keep true values hidden from learned runtime inputs; infer a bounded belief from observable motion/contact responses if the chosen sensors support it. An estimated belief is not identification of the exact parameters.
- **Simulator numerical error:** contact solver/discretization effects. Check contact labels and conclusions against a bounded numerical-convergence check before using force peaks as reliable labels. This is not physical environmental noise.

First verify nominal task completion. Then preregister bounded, physically motivated disturbances, test each source separately and a specified combined case, and use shared exogenous noise schedules/paired initial states across methods. Noise amplitudes, solver settings, success tolerances, force/impact definitions and termination rules are not yet numerically registered. A no-op/hold is not inherently safe during every contact mode; a fallback needs task-specific evidence and must count against success/time budgets if it prevents completion. Uncertainty-aware scores, ensembles and finite-horizon simulation do not provide unconditional real-world safety guarantees.

## What data adaptation means

Collect new training trajectories from the **target task's simulator training split**, including ordinary successful executions, natural actor failures and bounded interventions representative of the actual contact problem. Root-group initial states and related branches; maintain separate train/dev/calibration/test roots and held-out physical/noise conditions. Freeze methods before independent testing. Existing validation/audit/branch labels remain diagnosis-only. Do not train on the final evaluation outcomes and relabel the resulting gain as generalization.

Prior R5 evidence is directly relevant:120/128positive training decisions were from a single handcrafted brake-resumption collector, and the old gate already detected all62dev positives. Future support checks need relevance to the target failure mechanism and negative/safe-contact coverage, not just enough positive counts. No automatic broad collector sweep follows this observation.

Record predicted vs observed state/contact/progress and **eventual complete-task outcome**. This permits testing whether physical predictions are calibrated and useful for action choice, rather than simply memorizing success labels. New counterfactual/continuation supervision would require a fresh registered training-label protocol; old diagnostic labels do not become training data by this scope update.

## Evidence required for the contribution

On the same local actor and task, compare policy alone, matched simple feedback/slowing control and the physical verifier/repair. Match observations, sensing rate, control opportunities, candidate/query budget and calibration/data access; any changed low-level controller needs a matched control. Decompose state-estimation and physics-prediction effects when testing them instead of changing both silently. Also distinguish oracle-state from estimated/noisy-state performance.

Report SR, CVR, safe success, task-native precision/stable-completion measurements, paired rescue/harm, uncertainty and stress cases. Generalization claims require held-out roots/conditions and eventually a second task or actor; no current LLM result is claimed. The earlier≥+3pp safe-success, bounded SR loss and independent-confirmation requirements remain project efficacy targets, with the final task protocol frozen before testing. All declared primary noise strata matter; do not select the noise level producing the best gain.

Immediate deliverable: one task-capable local VLA rollout pipeline, videos and machine-readable completion/contact evidence. Then establish physical-intervention feasibility and implement the smallest learned change that addresses an observed task failure. No LLM/edge-cloud implementation or new training/rollout is launched by this scope decision.

## Sources checked

- https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/master/libero/libero/bddl_files/libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.bddl
- https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/master/libero/libero/bddl_files/libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.bddl
- https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/master/libero/libero/bddl_files/libero_goal/put_the_wine_bottle_on_the_rack.bddl
- https://raw.githubusercontent.com/ARISE-Initiative/robosuite/master/robosuite/environments/manipulation/nut_assembly.py (upstream master inspected, not an installed/runtime-pinned version)
- https://huggingface.co/api/models/openvla/openvla-7b-finetuned-libero-10
