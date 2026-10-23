# Framework diagnostic study

Development evidence; same microwave task, reused initial states. All indices are zero based.

| Root | Arm | Native success | Actions | Repair actions | Verified repairs |
|---|---|---|---|---|---|
| 0 | baseline | False | 520 | 0 | 0 |
| 0 | joint | False | 520 | 160 | 0 |
| 0 | joint_admission | False | 520 | 96 | 0 |
| 0 | measured_fusion | False | 520 | 96 | 0 |
| 1 | baseline | True | 444 | 0 | 0 |
| 1 | joint | False | 520 | 160 | 0 |
| 1 | joint_admission | True | 444 | 0 | 0 |
| 1 | measured_fusion | True | 444 | 0 | 0 |
| 2 | baseline | True | 397 | 0 | 0 |
| 2 | joint | False | 520 | 160 | 0 |
| 2 | joint_admission | True | 397 | 0 | 0 |
| 2 | measured_fusion | True | 397 | 0 | 0 |
| 3 | baseline | True | 352 | 0 | 0 |
| 3 | joint | True | 352 | 0 | 0 |
| 3 | joint_admission | True | 352 | 0 | 0 |
| 3 | measured_fusion | True | 352 | 0 | 0 |
| 4 | baseline | False | 520 | 0 | 0 |
| 4 | joint | False | 520 | 160 | 0 |
| 4 | joint_admission | False | 520 | 80 | 1 |
| 4 | measured_fusion | False | 520 | 80 | 1 |
| 5 | baseline | True | 298 | 0 | 0 |
| 5 | joint | True | 298 | 0 | 0 |
| 5 | joint_admission | True | 298 | 0 | 0 |
| 5 | measured_fusion | True | 298 | 0 | 0 |

## First-intervention evidence

- Root 0 joint: first entry 104; first action divergence 104; shared before-observation prefix 105; nominal24 label {'label': 1, 'reason': 'observed_sustained_grasp_without_lift', 'semantics': 'Training-defined short-horizon label, NOT complete-task failure or intervention benefit'}; rescue=False, harm=False.
- Root 0 joint_admission: first entry 254; first action divergence 254; shared before-observation prefix 255; nominal24 label {'label': 1, 'reason': 'observed_sustained_grasp_without_lift', 'semantics': 'Training-defined short-horizon label, NOT complete-task failure or intervention benefit'}; rescue=False, harm=False.
- Root 0 measured_fusion: first entry 254; first action divergence 254; shared before-observation prefix 255; nominal24 label {'label': 1, 'reason': 'observed_sustained_grasp_without_lift', 'semantics': 'Training-defined short-horizon label, NOT complete-task failure or intervention benefit'}; rescue=False, harm=False.
- Root 1 joint: first entry 89; first action divergence 89; shared before-observation prefix 90; nominal24 label {'label': 0, 'reason': 'observed_nominal_window_without_defined_violation', 'semantics': 'Training-defined short-horizon label, NOT complete-task failure or intervention benefit'}; rescue=False, harm=True.
- Root 1 joint_admission: first entry None; first action divergence None; shared before-observation prefix 444; nominal24 label None; rescue=False, harm=False.
- Root 1 measured_fusion: first entry None; first action divergence None; shared before-observation prefix 444; nominal24 label None; rescue=False, harm=False.
- Root 2 joint: first entry 145; first action divergence 145; shared before-observation prefix 146; nominal24 label {'label': 0, 'reason': 'observed_nominal_window_without_defined_violation', 'semantics': 'Training-defined short-horizon label, NOT complete-task failure or intervention benefit'}; rescue=False, harm=True.
- Root 2 joint_admission: first entry None; first action divergence None; shared before-observation prefix 397; nominal24 label None; rescue=False, harm=False.
- Root 2 measured_fusion: first entry None; first action divergence None; shared before-observation prefix 397; nominal24 label None; rescue=False, harm=False.
- Root 3 joint: first entry None; first action divergence None; shared before-observation prefix 352; nominal24 label None; rescue=False, harm=False.
- Root 3 joint_admission: first entry None; first action divergence None; shared before-observation prefix 352; nominal24 label None; rescue=False, harm=False.
- Root 3 measured_fusion: first entry None; first action divergence None; shared before-observation prefix 352; nominal24 label None; rescue=False, harm=False.
- Root 4 joint: first entry 98; first action divergence 98; shared before-observation prefix 99; nominal24 label {'label': 0, 'reason': 'observed_nominal_window_without_defined_violation', 'semantics': 'Training-defined short-horizon label, NOT complete-task failure or intervention benefit'}; rescue=False, harm=False.
- Root 4 joint_admission: first entry 148; first action divergence 148; shared before-observation prefix 149; nominal24 label {'label': 0, 'reason': 'observed_nominal_window_without_defined_violation', 'semantics': 'Training-defined short-horizon label, NOT complete-task failure or intervention benefit'}; rescue=False, harm=False.
- Root 4 measured_fusion: first entry 148; first action divergence 148; shared before-observation prefix 149; nominal24 label {'label': 0, 'reason': 'observed_nominal_window_without_defined_violation', 'semantics': 'Training-defined short-horizon label, NOT complete-task failure or intervention benefit'}; rescue=False, harm=False.
- Root 5 joint: first entry None; first action divergence None; shared before-observation prefix 298; nominal24 label None; rescue=False, harm=False.
- Root 5 joint_admission: first entry None; first action divergence None; shared before-observation prefix 298; nominal24 label None; rescue=False, harm=False.
- Root 5 measured_fusion: first entry None; first action divergence None; shared before-observation prefix 298; nominal24 label None; rescue=False, harm=False.

## Interpretation boundaries

- Full joint versus measured_fusion is a bundled ablation of joint stage/memory/prediction/risk, not risk-only.
- Risk labels describe 24-action measured events, not eventual task failure or the benefit of intervention.
- Later repairs occur after trajectory divergence; same-index baseline windows are not same-state counterfactuals.
- A budget exit and actual nominal handoff do not imply verified recovery.
- Replay/model recomputation reports must be consulted independently of actor completion.
- Gripper signs are mapped Panda commands; action coordinates are not measured Cartesian displacement.
