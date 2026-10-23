"""Repair-only H1 proposals. Collection selects a declared slot, never an oracle.

Frozen policy weights are independent of decoding: slots 1..3 use sampling.
One pool at first repair admission isolates a single action and leaves subsequent
repair verification and nominal resumption to the existing controller.
"""
from copy import deepcopy
import math


CONFIG = dict(schema='repair_candidate_probe_v1', size=4, max_pools=1,
    schedule='first_executed_repair_action', base_seed=20261022,
    action_cap=520, nominal_query_cap=520, additional_query_cap=3,
    total_query_cap=523, modification='select_unchanged_model_proposal',
    collection='fixed_slot_not_learned_ranking', continuation='existing_controller')


def requests(decision, root, step):
    if decision['owner'] != 'repair':
        raise ValueError('Candidates are only generated inside an admitted repair')
    return [dict(candidate_id=i, language=decision['language'],
        source='greedy' if i == 0 else 'sample',
        seed=0 if i == 0 else CONFIG['base_seed'] + root*100000 + step*4 + i)
        for i in range(CONFIG['size'])]


def context(decision, measured_state, image_sha256, history):
    """Only pre-action information; no outcome, simulator branch or future frame."""
    if decision['owner'] != 'repair':
        raise ValueError('Scoring is repair-scoped')
    return deepcopy(dict(schema='repair_scoring_context_v1',
        language=decision['language'], recovery_goal=decision['recovery_goal'],
        repair_id=decision['repair_id'], trigger=decision['trigger'],
        state=measured_state, policy_png_sha256=image_sha256,
        action_history=history[-6:]))


def validate_candidates(candidates):
    if not candidates or [c['candidate_id'] for c in candidates] != list(range(len(candidates))):
        raise ValueError('Candidates require ordered unique IDs')
    for c in candidates:
        a = c['decode']['action']
        if not c['decode']['valid'] or len(a) != 7 or not all(math.isfinite(x) for x in a):
            raise ValueError('Invalid model-generated candidate')


def select(candidates, scoring_context, *, observation=None, scorer=None, collection_slot=None):
    """score(observation, context, candidates) -> per-action scores.

    A supplied scorer must depend on proposed actions and have external training
    provenance. The coverage collector deliberately has no fabricated scores.
    Exact argmax ties prefer the first candidate (greedy). No action is edited.
    """
    validate_candidates(candidates)
    if (scorer is None) == (collection_slot is None):
        raise ValueError('Choose either a scorer or a collection slot')
    if collection_slot is not None:
        if type(collection_slot) is not int or not 0 <= collection_slot < len(candidates):
            raise ValueError('Collection slot out of range')
        selected, scores, method = collection_slot, None, 'fixed_slot_collection'
        provenance = None
    else:
        provenance = deepcopy(scorer.provenance)
        digest = provenance.get('artifact_sha256', '')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest) or not provenance.get('training_split'):
            raise ValueError('Learned scorer needs artifact and split provenance')
        if observation is None:
            raise ValueError('Scoring requires the current observation, not only an image hash')
        scores = list(scorer.score(observation=deepcopy(observation),
            context=deepcopy(scoring_context), candidates=deepcopy(candidates)))
        if len(scores) != len(candidates) or not all(math.isfinite(float(v)) for v in scores):
            raise ValueError('Expected one finite score per candidate')
        scores = [float(v) for v in scores]
        selected = max(range(len(scores)), key=scores.__getitem__)
        method = 'learned_action_conditioned_ranking'
    return deepcopy(candidates[selected]['decode']), dict(
        schema=CONFIG['schema'], method=method, selected_id=selected, scores=scores,
        scorer=provenance, context=deepcopy(scoring_context), candidates=deepcopy(candidates))
