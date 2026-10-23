# Framework source and experiments

The five-layer architecture is preserved. Task semantics are generated, not
encoded in a microwave controller. One `Session` owns every policy query/step.

```text
src/
├── controller.py                 # Generic transactions, rechecks, budgets, handoff
├── runtime.py                    # Sole query/step owner and live module integration
├── interfaces.py                 # Backend contracts and unchanged H1 Proposal
├── observation.py                # Measured environment-neutral endpoints
├── journal.py                    # Events and exact input PNGs
├── planning/                     # Vision MLLM TaskSpec / RecoveryPlan and grammar
├── adapters/                     # Shared LIBERO environment/image + frozen OpenVLA
├── perception/
│   ├── frozen_visual_encoder.py  # Strict pretrained DINOv2; no random fallback
│   ├── visual_grounding.py       # Train-prototype similarity, not verification
│   └── vision_subgoal_inference.py # Generated predicates, no fixed task stages
├── verification/
│   ├── physical_verifier.py      # Tri-state contracts and four measured features
│   ├── features.py               # Versioned generic task/contract/action features
│   ├── fusion_mlp.py             # Learned head and strict artifact loader
│   ├── calibrated_risk_head.py   # Empirically calibrated observed-regression risk
│   ├── handoff_verifier.py       # B: exact nominal-action continuation assessment
│   └── vision_physical_monitor.py # Shared live encoder/memory/prediction/risk context
├── prediction/short_term_predictor.py # Causal motion extrapolation, no world model
├── memory/keyframe_bank.py       # Read-only training-predicate episodic retrieval
├── repair/                       # A: unchanged proposal ranking; B public name
├── training/                     # Offline labels and generic learned-head fitting
├── experiment/                   # ALL active experiment commands and tests
├── checkpoints/                  # Preserved historical weights, not auto-loaded
└── results/                      # Preserved historical evidence
```

`legacy/microwave_controller.py`, old runners, fixed-stage/random fallback paths,
and handwritten numerical action correction are not restored. Repair still uses
fresh unchanged frozen-VLA proposals. Old code is recoverable from Git/archives;
old artifacts are not silently reinterpreted as generic models.

- Source API: `from src import Session, Limits, Proposal` (no ML imports).
- Experiments: [experiment guide](experiment/README.md).
- Tests: `python -B -m unittest discover -s src/experiment/tests`.
- [Runtime and integration](../docs/task_runtime.md) · [GPU setup](../docs/gpu_setup.md).

Source-importable, not yet an installable distribution. Real MLLM/OpenVLA/DINO
rollouts and useful learned heads remain to be validated on real episodes.
