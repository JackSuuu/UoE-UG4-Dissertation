"""Single causal visual/physical/predictive/memory/risk/handoff runtime.

Every observation is processed once. Real features are shared across all learned
paths; no random fallback, zero-vision risk head or empty memory in this mode.
"""
from collections import deque
from copy import deepcopy
import hashlib

import numpy as np

from ..memory.keyframe_bank import KeyframeBank
from ..perception.vision_subgoal_inference import VisionSubgoalInference
from ..prediction.short_term_predictor import ShortTermMotionPredictor
from ..training.handoff_dataset import contract
from .handoff_verifier import HandoffVerifier
from .outcome_risk import OutcomeRisk

JOINT_SCHEMA = 'joint_visual_physical_prediction_memory_risk_h1_v1'


class JointRuntime:
    def __init__(self, fusion_checkpoint, joint_checkpoint, device='cpu'):
        self.handoff = HandoffVerifier(fusion_checkpoint, device=device)
        self.risk = OutcomeRisk(joint_checkpoint, device=device)
        a = self.risk.artifact
        if a['encoder'] != self.handoff.encoder.provenance:
            raise ValueError('Joint heads must share identical pretrained visual provenance')
        if (tuple(a['bank_features'].shape) != (len(a['bank_stages']), 384) or
                len(a['bank_metadata']) != len(a['bank_stages']) or not a['bank_stages']):
            raise ValueError('Incomplete or incompatible episodic artifact')
        self.bank = KeyframeBank(capacity=len(a['bank_stages'])+200, feature_dim=384)
        for feature, stage, metadata in zip(a['bank_features'].numpy(), a['bank_stages'], a['bank_metadata']):
            if metadata['root'] not in a['train_roots']:
                raise ValueError('Non-training root found in episodic memory')
            self.bank.write(feature, stage, success=True, metadata=metadata)
        self.inference = VisionSubgoalInference(grounding=None)
        self.predictor = ShortTermMotionPredictor(action_blend=0.)
        self.history = deque(maxlen=181)
        self.pending_memory = deque()
        self.last_index = -1
        self.last_visual = self.last_rgb = self.last_evidence = None
        self.high_risk_streak = 0
        self.stage_candidate = self.stage_selected = None
        self.stage_streak = 0
        self.stage_dwell = a['stage_dwell']
        self.risk_persistence = a['risk_persistence']
        self.no_lift_steps = a['no_lift_steps']
        self.counters = dict(observations=0, visual_stage_evaluations=0, memory_retrievals=0,
            risk_evaluations=0, prediction_evaluations=0, online_memory_writes=0,
            risk_trigger_observations=0, prediction_trigger_observations=0, handoff_scores=0)

    def observe(self, policy_rgb, state, index, region=None):
        if index != self.last_index+1:
            raise ValueError('Joint observations must be sequential and processed exactly once')
        rgb = np.ascontiguousarray(policy_rgb)
        visual = self.handoff.encoder.encode(rgb)
        self.history.append(deepcopy(state))
        if state.get('position') is None:
            self.predictor.reset()  # Do not bridge unknown observation intervals.
        self.predictor.update(state.get('position'), state.get('grasp'), index)
        prototypes = {stage: self.bank.read_prototype(stage) for stage in self.inference.STAGES}
        inferred = self.inference.infer_features(visual, state, prototypes)
        retrieved, combined = {}, {}
        for stage in inferred['admissible']:
            entries = self.bank.read_topk(visual, stage, k=3)
            retrieved[stage] = entries
            if entries and stage in inferred['all_scores']:
                combined[stage] = .5*inferred['all_scores'][stage]+.5*float(np.mean([e['similarity'] for e in entries]))
        if not combined:
            raise ValueError('No episodic support for current stage; do not silently disable memory')
        candidate = max(combined, key=combined.get)
        self.stage_streak = self.stage_streak+1 if candidate == self.stage_candidate else 1
        self.stage_candidate = candidate
        if self.stage_selected not in combined or self.stage_streak >= self.stage_dwell:
            self.stage_selected = candidate
        stage = self.stage_selected
        risk = self.risk.score(visual, state, list(self.history), stage)
        self.high_risk_streak = self.high_risk_streak+1 if risk['exceeds'] else 0
        lift = self.predictor.predicts_no_lift(state.get('grasp'), state.get('position'),
            index, max_steps_since_grasp=self.no_lift_steps)
        positions = self.predictor.predict(state.get('position'), proposed_action=None)
        prediction = dict(method='observed_constant_velocity_world_coordinates',
            predicted_positions=positions, region_available=region is not None,
            exit_predicted=False, minimum_margin_m=None, no_lift=lift)
        # Use the live oriented official region, not a guessed world AABB.
        if region is not None and positions is not None:
            center = np.asarray(region['center'])
            rotation = np.asarray(region['rotation']).reshape(3, 3)
            size = np.asarray(region['half_size'])
            margins = [float(np.min(size-np.abs(rotation.T@(np.asarray(p)-center)))) for p in positions]
            prediction['minimum_margin_m'] = min(margins)
            prediction['exit_predicted'] = state.get('in_goal') is True and min(margins) < 0.
        prediction_trigger = ('joint_predicted_exit' if prediction['exit_predicted'] else
            'joint_no_lift' if lift is not None and lift['no_lift_predicted'] else None)
        trigger = prediction_trigger or ('joint_outcome_risk' if self.high_risk_streak >= self.risk_persistence else None)
        # A delayed successful memory write uses only evidence already observed
        # by NOW; it is never immediately marked safe merely by local recovery.
        for old in self.pending_memory:
            if contract(state, old['contract']) is not True or any(
                    old['state'].get(k) is True and state.get(k) is not True for k in ('in_goal', 'close_goal')):
                old['valid'] = False
        while self.pending_memory and index-self.pending_memory[0]['index'] >= 12:
            old = self.pending_memory.popleft()
            if old['valid']:
                self.bank.write(old['visual'], old['stage'], True,
                    dict(source='online_12_observation_preservation', index=old['index'], validated_at=index))
                self.counters['online_memory_writes'] += 1
        # Only current-true recovery contracts are eligible for delayed memory.
        recovered = next((s for s in ('contain_release', 'transfer', 'acquire') if contract(state, s) is True), None)
        if recovered is not None and index % 8 == 0:
            self.pending_memory.append(dict(index=index, visual=visual.copy(), stage=stage,
                state=deepcopy(state), contract=recovered, valid=True))
        for key in ('observations', 'visual_stage_evaluations', 'risk_evaluations', 'prediction_evaluations'):
            self.counters[key] += 1
        self.counters['memory_retrievals'] += sum(bool(v) for v in retrieved.values())
        self.counters['risk_trigger_observations'] += int(trigger == 'joint_outcome_risk')
        self.counters['prediction_trigger_observations'] += int(prediction_trigger is not None)
        evidence = dict(schema=JOINT_SCHEMA, observation_index=index, selected_stage=stage,
            trigger=trigger, visual=inferred, memory=dict(retrieved=retrieved, combined_scores=combined,
                selected_stage=stage, bank_counts=self.bank.stage_counts()),
            risk=dict(risk, consecutive=self.high_risk_streak, required=self.risk_persistence),
            prediction=prediction,
            physical=dict(native_in_goal=state.get('in_goal'), grasp=state.get('grasp'),
                held_clear=state.get('held_clear'), contact_target_bound=True, official_region=region),
            policy_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            encoder_sha256=self.handoff.encoder.provenance['sha256'])
        self.last_index, self.last_visual, self.last_rgb = index, visual, rgb
        self.last_evidence = evidence
        return deepcopy(evidence)

    def score_handoff(self, state, recovery_goal):
        self.counters['handoff_scores'] += 1
        evidence = self.handoff.score_features(self.last_visual, state, recovery_goal, self.last_rgb)
        # An observed/predicted imminent failure vetoes an otherwise accepted
        # fusion score. Record the original score instead of editing it.
        evidence['joint_prediction_veto'] = bool(self.last_evidence['prediction']['exit_predicted'] or
            (self.last_evidence['prediction']['no_lift'] or {}).get('no_lift_predicted'))
        evidence['joint_risk_veto'] = self.high_risk_streak >= self.risk_persistence
        return evidence
