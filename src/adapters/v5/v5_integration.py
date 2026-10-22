"""Integration layer: plugs v5 vision+physical verification into the runner.

This shows how to augment the existing NativeFramework without modifying its
frozen source.  The runner calls `active.before(step, measured)` and
`active.after(...)`.  We wrap those calls with:

  1. VisionPhysicalMonitor.evaluate()  -> vision + physical evidence
  2. VerificationFusionMLP             -> learned fusion probability
  3. VisionSubgoalInference            -> vision-based stage inference

The fusion probability is exposed in the decision's `verification_extra`
field and can be used by the runner to gate repair entry / handoff.
"""
from typing import Optional, Dict, Any, List
from copy import deepcopy

from .physical_verifier import PhysicalVerifier, make_verifier
from .visual_grounding import VisualGroundingModule, make_grounding
from .vision_physical_monitor import VisionPhysicalMonitor, make_monitor
from .vision_subgoal_inference import VisionSubgoalInference, make_subgoal_inference
from .fusion_mlp import (
    VerificationFusionMLP, extract_physics_features, STAGE_TO_IDX, STAGES,
)


class V5VerificationWrapper:
    """Wraps a NativeFramework instance with v5 vision+physical verification.

    Usage in the runner loop:

        wrapper = V5VerificationWrapper(framework, run_dir, device='cuda:2')
        # load or train fusion MLP
        wrapper.load_or_train_fusion_mlp()

        # inside the per-step loop:
        measured = tape.last['diagnostics']
        v5_evidence = wrapper.evaluate_v5(
            current_image=obs['agentview_image'],
            measured=measured,
            proposed_action=None,       # filled after VLA query
            step=step,
        )
        decision = active.before(step, deepcopy(measured))
        decision['v5_evidence'] = v5_evidence   # expose to runner/logging
        # ... query VLA, execute ...
        # after execution:
        v5_evidence['proposed_action'] = executed_action
        wrapper.update_v5_after(v5_evidence, measured, executed_action, step)
        verification = active.after(step, deepcopy(row['diagnostics']),
                                    list(row['action']), deepcopy(decision))
        verification['v5'] = wrapper.v5_verification_summary()
    """

    def __init__(
        self,
        framework,
        run_dir=None,
        device='cuda:2',
        vision_threshold=0.75,
        mlp_path=None,
        mlp_gate_threshold=0.5,
    ):
        self.framework = framework
        self.device = device
        self.mlp_gate_threshold = mlp_gate_threshold

        # v5 components
        self.monitor = make_monitor(
            target=framework._config.get('target', 'white_yellow_mug_1'),
            receptacle=framework._config.get('receptacle', 'microwave_1'),
            region=framework._config.get('region', 'microwave_1_heating_region'),
            vision_threshold=vision_threshold,
        )
        self.subgoal_infer = make_subgoal_inference(
            target=framework._config.get('target', 'white_yellow_mug_1'),
            receptacle=framework._config.get('receptacle', 'microwave_1'),
        )

        # Fusion MLP (optional; trains on physics-only if vision encoder missing)
        self.mlp: Optional[VerificationFusionMLP] = None
        self.mlp_path = mlp_path
        if mlp_path:
            self._load_or_init_mlp()

        # Running state
        self._prev_object_pos: Optional[List[float]] = None
        self._current_v5_evidence: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # MLP lifecycle
    # ------------------------------------------------------------------
    def _load_or_init_mlp(self):
        from .fusion_mlp import load_model, VerificationFusionMLP
        if self.mlp_path and __import__('os').path.exists(self.mlp_path):
            self.mlp = load_model(self.mlp_path, device=self.device)
        else:
            self.mlp = VerificationFusionMLP()

    def train_mlp(self, pairs, epochs=50, lr=1e-3, batch_size=32):
        from .fusion_mlp import train_on_pairs
        losses = train_on_pairs(
            self.mlp, pairs, epochs=epochs, lr=lr,
            batch_size=batch_size, device=self.device,
        )
        if self.mlp_path:
            from .fusion_mlp import save_model
            save_model(self.mlp, self.mlp_path)
        return losses

    # ------------------------------------------------------------------
    # Per-step evaluation (call before active.before)
    # ------------------------------------------------------------------
    def evaluate_v5(
        self,
        current_image,
        measured: Dict[str, Any],
        proposed_action: Optional[List[float]] = None,
        curr_object_pos: Optional[List[float]] = None,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run vision + physical monitor and (optionally) fusion MLP."""
        evidence = self.monitor.evaluate(
            current_image=current_image,
            diagnostics=measured,
            proposed_action=proposed_action,
            curr_object_pos=curr_object_pos,
            step=step,
        )

        # Physics features for MLP
        physics_feat = extract_physics_features(measured)

        # Vision feature: from monitor's visual grounding score (placeholder)
        # If a real encoder is wired, replace with encoder output.
        vision_score = evidence['vision_evidence'].get('score') or 0.0
        vision_feat = np_array = None  # placeholder; encoder not yet wired
        # We pass a scalar vision score as a 1-dim feature for now;
        # full 256-dim encoder features require the encoder to be loaded.
        vision_feat = np.array([vision_score], dtype=np.float32)

        # Stage one-hot from framework's shadow subgoal
        stage = measured.get('_shadow_subgoal', 'transfer')
        stage_idx = STAGE_TO_IDX.get(stage, 1)
        stage_onehot = np.zeros(len(STAGES), dtype=np.float32)
        stage_onehot[stage_idx] = 1.0

        # MLP fusion probability
        mlp_prob = None
        if self.mlp is not None:
            try:
                import torch
                with torch.no_grad():
                    vf = torch.from_numpy(vision_feat.reshape(1, -1)).to(self.device)
                    pf = torch.from_numpy(physics_feat.reshape(1, -1)).to(self.device)
                    soh = torch.from_numpy(stage_onehot.reshape(1, -1)).to(self.device)
                    logits = self.mlp.forward(vf, pf, soh)
                    probs = torch.softmax(logits, dim=-1)
                    mlp_prob = float(probs[0, 0].item())
            except Exception:
                mlp_prob = None

        evidence['mlp_verify_probability'] = mlp_prob
        evidence['mlp_gate_pass'] = (
            mlp_prob is not None and mlp_prob >= self.mlp_gate_threshold)
        evidence['physics_features'] = physics_feat.tolist()
        evidence['stage_onehot'] = stage_onehot.tolist()

        self._current_v5_evidence = evidence
        return evidence

    def update_v5_after(
        self,
        v5_evidence: Dict[str, Any],
        measured: Dict[str, Any],
        executed_action: List[float],
        step: int,
    ) -> None:
        """Update position history after action execution (for action-physics consistency)."""
        pos = measured.get('position')
        if pos is not None:
            self._prev_object_pos = [float(v) for v in pos]

    # ------------------------------------------------------------------
    # Summary for logging
    # ------------------------------------------------------------------
    def v5_verification_summary(self) -> Dict[str, Any]:
        ev = self._current_v5_evidence or {}
        return {
            'trigger': ev.get('trigger'),
            'vision_in_goal': ev.get('vision_evidence', {}).get('in_goal_vision'),
            'vision_score': ev.get('vision_evidence', {}).get('score'),
            'physical_in_goal': ev.get('physical_evidence', {}).get('in_goal'),
            'physical_score': ev.get('physical_evidence', {}).get('score'),
            'action_physics_consistent': ev.get('action_physics_consistent'),
            'action_physics_score': ev.get('action_physics_score'),
            'mlp_verify_probability': ev.get('mlp_verify_probability'),
            'mlp_gate_pass': ev.get('mlp_gate_pass'),
            'vision_physical_consistent': ev.get('consistent'),
        }


def patch_runner_step(
    runner_module,
    wrapper: V5VerificationWrapper,
):
    """Monkey-patch helper (optional).  Preferred: manually insert calls in
    run_framework_comparison.py's actor loop as shown in v5_integration docstring.

    This is provided for quick experimentation; for a frozen, reproducible
    run, edit the runner source directly.
    """
    raise NotImplementedError(
        'For reproducibility, edit run_framework_comparison.py directly: '
        'call wrapper.evaluate_v5() before active.before() and '
        'wrapper.update_v5_after() after runner.step_once().')
