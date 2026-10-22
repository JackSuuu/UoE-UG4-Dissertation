"""Vision + physical joint monitor with CheckVLA-inspired extensions.

Integrates:
- L3 trigger: vision anomaly AND physical verifier agree (suppresses false interventions)
- Short-term motion prediction (CheckVLA Module 1): pre-emptive exit warning
- Calibrated risk head (CheckVLA Module 2): risk score r_t with conformal threshold
- Action suffix corrector (CheckVLA Module 3): lightweight action safety nudge
- Keyframe bank (CheckVLA Module 4): episodic context for stage inference

Does NOT generate actions itself — the corrector returns a corrected action
candidate that the runner may choose to use or ignore.
"""
from typing import Optional, Dict, Any, List
import numpy as np
from .physical_verifier import PhysicalVerifier, extract_physics_state
from ..perception.visual_grounding import VisualGroundingModule
from ..prediction.short_term_predictor import ShortTermMotionPredictor, make_short_term_predictor
from .calibrated_risk_head import CalibratedRiskHead, extract_trajectory_stats, make_risk_head
from ..repair.action_suffix_corrector import ActionSuffixCorrector, make_action_suffix_corrector
from ..memory.keyframe_bank import KeyframeBank, make_keyframe_bank


class VisionPhysicalMonitor:
    def __init__(
        self,
        verifier: PhysicalVerifier,
        grounding: VisualGroundingModule,
        predictor: ShortTermMotionPredictor,
        risk_head: CalibratedRiskHead,
        corrector: ActionSuffixCorrector,
        keyframe_bank: KeyframeBank,
        vision_threshold: float = 0.75,
        risk_alpha: float = 0.05,
        target: Optional[str] = None,
    ):
        self.verifier = verifier
        self.grounding = grounding
        self.predictor = predictor
        self.risk_head = risk_head
        self.corrector = corrector
        self.keyframe_bank = keyframe_bank
        self.vision_threshold = vision_threshold
        self.risk_alpha = risk_alpha
        self.target = target

        # History for trajectory stats + predictor
        self._history: List[Dict[str, Any]] = []
        self._prev_object_pos: Optional[List[float]] = None

    # ------------------------------------------------------------------
    # Keyframe bank API (called by runner on successful stage completion)
    # ------------------------------------------------------------------
    def write_keyframe(
        self,
        visual_feature: np.ndarray,
        stage: str,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.keyframe_bank.write(visual_feature, stage, success, metadata)

    def read_stage_prototype(self, stage: str) -> Optional[np.ndarray]:
        return self.keyframe_bank.read_prototype(stage)

    # ------------------------------------------------------------------
    # Action correction API (CheckVLA Module 3 lightweight)
    # ------------------------------------------------------------------
    def correct_action(
        self,
        proposed_action: List[float],
        action_physics_consistent: Optional[bool] = None,
        action_physics_score: Optional[float] = None,
        intent_vs_observed_cosine: Optional[float] = None,
    ) -> List[float]:
        """Return corrected action candidate (runner decides whether to use)."""
        return self.corrector.correct(
            proposed_action,
            action_physics_consistent,
            action_physics_score,
            intent_vs_observed_cosine,
        )

    def correct_action_with_physics(
        self,
        proposed_action: List[float],
        prev_pos: Optional[List[float]],
        curr_pos: Optional[List[float]],
        receptacle_bounds: Optional[Dict[str, List[float]]],
    ) -> List[float]:
        """Correct using observed motion vs action intent."""
        return self.corrector.correct_with_physics_state(
            proposed_action, prev_pos, curr_pos, receptacle_bounds,
        )

    # ------------------------------------------------------------------
    # Main evaluation (per step)
    # ------------------------------------------------------------------
    def evaluate(
        self,
        current_image,
        diagnostics: Dict[str, Any],
        proposed_action: Optional[List[float]] = None,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return joint vision + physical + prediction + risk evidence bundle.

        New keys vs. earlier version:
          - 'short_term_prediction': dict with exit_predicted, exit_step, etc.
          - 'risk_score': float r_t ∈ [0,1]
          - 'risk_threshold': float δ_t (None if not calibrated)
          - 'risk_trigger': bool (r_t > δ_t)
          - 'trigger': combined trigger name (None / containment_lost / action_physics_regression
                        / preemptive_exit / high_risk)
        """
        # --- Append current step to history (before this step's diagnostics) ---
        measured_for_stats = dict(diagnostics)
        measured_for_stats['step'] = step if step is not None else len(self._history)
        self._history.append(measured_for_stats)

        # --- Vision ---
        vision = self.grounding.infer_in_goal(
            current_image, goal_prototype_key='contained',
            threshold=self.vision_threshold)
        in_goal_vision = vision.get('in_goal_vision')
        vision_score = vision.get('score')

        # --- Physical ---
        phys_state = extract_physics_state(diagnostics, target=self.target)
        phys_in_goal = self.verifier.verify_in_goal(
            phys_state['object_pos'],
            phys_state['receptacle_bounds'],
            phys_state['contact_info'],
        )

        # --- Update predictor history (with grasp state for no-lift tracking) ---
        # Grasp boolean: from diagnostics['grasped'][target] if available,
        # else from phys_state['grasp'] (which may be None).
        grasp = phys_state.get('grasp')
        self.predictor.update(
            phys_state['object_pos'],
            grasp=grasp,
            step=step,
        )

        # --- Action-physics consistency (post-execution, uses prev_pos) ---
        action_phys = None
        if (proposed_action is not None and
                self._prev_object_pos is not None and
                phys_state['object_pos'] is not None):
            action_phys = self.verifier.verify_action_physics_consistency(
                proposed_action, self._prev_object_pos, phys_state['object_pos'])

        if phys_state['object_pos'] is not None:
            self._prev_object_pos = phys_state['object_pos']

        # --- Short-term motion prediction: containment exit (contain_release phase) ---
        short_term_exit = self.predictor.predicts_exit(
            phys_state['object_pos'],
            proposed_action,
            phys_state['receptacle_bounds'],
        )

        # --- Short-term motion prediction: no-lift after grasp (acquire phase) ---
        short_term_lift = self.predictor.predicts_no_lift(
            grasp=grasp,
            current_pos=phys_state['object_pos'],
            step=step,
        )

        # --- Risk head ---
        traj_stats = extract_trajectory_stats(self._history)
        vision_feat = None
        risk_score = None
        risk_trigger = False
        if self.risk_head is not None:
            # Use a placeholder vision feature (zeros) until encoder is wired;
            # the risk head still uses physics + stats which are informative.
            vision_feat = np.zeros(self.grounding.feature_dim, dtype=np.float32)
            risk_score = self.risk_head.compute_risk(
                vision_feat, phys_state_to_feat(phys_state), traj_stats,
            )
            risk_trigger = self.risk_head.should_trigger(risk_score)

        # --- Joint trigger (priority order) ---
        trigger = None
        if risk_trigger:
            trigger = 'high_risk'
        elif short_term_lift is not None and short_term_lift['no_lift_predicted']:
            trigger = 'no_lift_after_grasp'
        elif short_term_exit is not None and short_term_exit['exit_predicted']:
            trigger = 'preemptive_exit'
        elif (in_goal_vision is False and not phys_in_goal.consistent):
            trigger = 'containment_lost'
        elif (in_goal_vision is True and phys_in_goal.consistent
              and action_phys is not None and not action_phys.consistent):
            trigger = 'action_physics_regression'

        return {
            'trigger': trigger,
            'vision_evidence': vision,
            'physical_evidence': {
                'in_goal': phys_in_goal.consistent,
                'score': phys_in_goal.score,
                'details': phys_in_goal.details,
            },
            'action_physics_consistent': (
                action_phys.consistent if action_phys else None),
            'action_physics_score': (
                action_phys.score if action_phys else None),
            'short_term_exit': short_term_exit,
            'short_term_lift': short_term_lift,
            'risk_score': risk_score,
            'risk_threshold': self.risk_head.threshold if self.risk_head else None,
            'risk_trigger': risk_trigger,
            'consistent': (
                (in_goal_vision == phys_in_goal.consistent)
                if in_goal_vision is not None else None),
            'physics_state': phys_state,
            'traj_stats': traj_stats.tolist(),
        }

    def verify_grasp(self, gripper_force: Optional[float]):
        return self.verifier.verify_grasp(gripper_force)

    def reset(self) -> None:
        self._prev_object_pos = None
        self._history = []
        self.predictor.reset()

    def size(self) -> int:
        return self.keyframe_bank.size()


# ------------------------------------------------------------------
# Helper: convert physics state dict to feature vector for risk head
# ------------------------------------------------------------------
def phys_state_to_feat(phys_state: Dict[str, Any]) -> np.ndarray:
    def b(val):
        return 1.0 if val is True else 0.0

    pos = phys_state.get('object_pos') or [0.0, 0.0, 0.0]
    return np.array([
        b(phys_state.get('in_goal')) if phys_state.get('in_goal') is not None else 0.0,
        b(phys_state.get('close_goal')) if phys_state.get('close_goal') is not None else 0.0,
        b(phys_state.get('grasp')) if phys_state.get('grasp') is not None else 0.0,
        b(phys_state.get('held_clear')) if phys_state.get('held_clear') is not None else 0.0,
        b(phys_state.get('table')) if phys_state.get('table') is not None else 0.0,
        b(phys_state.get('mug_microwave')) if phys_state.get('mug_microwave') is not None else 0.0,
        b(phys_state.get('gripper_microwave')) if phys_state.get('gripper_microwave') is not None else 0.0,
        b(phys_state.get('mug_gripper')) if phys_state.get('mug_gripper') is not None else 0.0,
        float(np.clip(pos[0], -2, 2)),
        float(np.clip(pos[1], -2, 2)),
        float(np.clip(pos[2], -2, 2)),
        0.0,  # rise_m placeholder (not in phys_state)
    ], dtype=np.float32)


def make_monitor(
    vision_threshold: float = 0.75,
    offline: bool = True,
    device: str = 'cuda:0',
    risk_alpha: float = 0.05,
    lift_threshold_m: float = 0.02,
    target: Optional[str] = None,
) -> VisionPhysicalMonitor:
    verifier = PhysicalVerifier()
    grounding = VisualGroundingModule(offline=offline, device=device)
    predictor = make_short_term_predictor(lift_threshold_m=lift_threshold_m)
    risk_head = make_risk_head(alpha=risk_alpha)
    corrector = make_action_suffix_corrector()
    keyframe_bank = make_keyframe_bank()
    return VisionPhysicalMonitor(
        verifier=verifier, grounding=grounding,
        predictor=predictor, risk_head=risk_head,
        corrector=corrector, keyframe_bank=keyframe_bank,
        vision_threshold=vision_threshold, risk_alpha=risk_alpha, target=target,
    )
