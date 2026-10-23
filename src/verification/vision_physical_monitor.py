"""Shared live visual/memory/prediction/risk context consumed by the controller.

Encode each image once. Retrieval and visual subgoal estimates inform the MLLM's
recovery request, never establish measured success. Only a loaded learned risk
head can add a persistent intervention signal or a handoff veto in gate mode.
"""
from collections import deque
from copy import deepcopy
import hashlib
import math

from ..memory.keyframe_bank import KeyframeBank, unit
from ..perception.vision_subgoal_inference import VisionSubgoalInference
from ..prediction.short_term_predictor import ShortTermMotionPredictor
from .features import encoder_signature, fusion_features


class VisionPhysicalMonitor:
    def __init__(self, *, encoder=None, memory=None, predictor=None, risk=None, mode='log', persistence=3):
        if mode not in ('log', 'gate') or type(persistence) is not int or persistence < 1:
            raise ValueError('Invalid monitor mode/persistence')
        if (memory is not None or risk is not None) and encoder is None:
            raise ValueError('Memory/risk requires a real visual encoder')
        self.encoder, self.risk, self.mode, self.persistence = encoder, risk, mode, persistence
        self.signature = encoder_signature(encoder.provenance) if encoder else None
        self.memory = memory or KeyframeBank(encoder_signature=self.signature)
        if self.memory.encoder_signature != self.signature or self.memory.feature_dim != 384:
            raise ValueError('Memory/encoder provenance mismatch')
        self.predictor = predictor or ShortTermMotionPredictor()
        self.subgoals = VisionSubgoalInference()
        self._image_key, self._visual = None, None
        self._risks = deque(maxlen=persistence)
        self.history = deque(maxlen=12)
        self.provenance = dict(mode=mode, encoder=encoder.provenance if encoder else None,
                               risk=getattr(risk, 'provenance', None), memory_entries=len(self.memory._entries),
                               memory_policy='read_only_training_roots10_19', predictor='observed_constant_velocity_v1')

    def observe(self, observation, task, physics):
        if self.history and observation.index < self.history[-1].index:
            raise ValueError('Monitor endpoints must be sequential')
        if self.history and observation.index == self.history[-1].index:
            self.history[-1] = deepcopy(observation)
        else:
            self.history.append(deepcopy(observation))
        prediction = self.predictor.predict(observation, self.history)
        report = dict(observation_index=observation.index, prediction=prediction,
                      visual=None, memory=[], risk=None, trigger=False, handoff_veto=False,
                      mode=self.mode)
        features = None
        if self.encoder:
            key = (observation.index, hashlib.sha256(observation.image_png).hexdigest())
            if self._image_key is not None and key[0] == self._image_key[0] and key != self._image_key:
                raise ValueError('Image changed at a monitored endpoint')
            if key != self._image_key:
                self._visual = unit(self.encoder.encode_png(observation.image_png), 384)
                self._image_key = key
            report['visual'] = self.subgoals.infer_features(self._visual, task, observation,
                                                           self.history, self.memory)
            report['memory'] = self.memory.retrieve(self._visual, (*task.goals, *task.milestones))
            features = fusion_features(self._visual, observation, task, physics, history=self.history)
            if self.risk:
                risk = self.risk.score(features)
                if (not isinstance(risk, dict) or type(risk.get('exceeds')) is not bool
                        or any(type(risk.get(k)) not in (int, float) or not math.isfinite(risk[k])
                               or not 0 <= risk[k] <= 1 for k in ('score', 'threshold'))
                        or risk['exceeds'] != (risk['score'] > risk['threshold'])):
                    raise ValueError('Risk head must return a bounded score/threshold and consistent exceeds')
                entry = (observation.index, risk['exceeds'])
                if self._risks and self._risks[-1][0] == observation.index:
                    self._risks[-1] = entry
                else:
                    self._risks.append(entry)
                persistent = (len(self._risks) == self.persistence and all(v for _, v in self._risks)
                    and [i for i, _ in self._risks] == list(range(observation.index-self.persistence+1,
                                                               observation.index+1)))
                report.update(risk=risk, would_trigger=persistent,
                              trigger=self.mode == 'gate' and persistent,
                              handoff_veto=self.mode == 'gate' and risk['exceeds'])
        return dict(context=report, features=features, visual=self._visual,
                    encoder_signature=self.signature)

    def action_features(self, observation, task, plan, action, physics):
        if self._image_key != (observation.index, hashlib.sha256(observation.image_png).hexdigest()):
            raise ValueError('Action scorer needs the current encoded endpoint')
        return fusion_features(self._visual, observation, task, physics, plan=plan,
                               action=action, history=self.history)
