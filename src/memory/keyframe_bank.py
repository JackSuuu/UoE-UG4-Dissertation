"""Strict visual keyframes labelled by generated predicates, not fixed stages.

Only measured, successful training entries can be loaded/written. Evaluation
never writes online, so current or held-out outcomes cannot leak into retrieval.
"""
from copy import deepcopy
import json
import math
from pathlib import Path


def unit(feature, dimension):
    result = tuple(float(x) for x in feature)
    if len(result) != dimension or not all(math.isfinite(x) for x in result):
        raise ValueError('Visual feature dimension/values do not match the encoder')
    norm = math.sqrt(sum(x*x for x in result))
    if norm < 1e-8:
        raise ValueError('Missing/zero visual feature cannot be retrieved')
    return tuple(x/norm for x in result)


class KeyframeBank:
    schema = 'generated_predicate_keyframes_v1'

    def __init__(self, capacity=200, feature_dim=384, *, encoder_signature=None):
        if type(capacity) is not int or capacity < 1 or type(feature_dim) is not int or feature_dim < 1:
            raise ValueError('Positive memory capacity and feature dimension required')
        self.capacity, self.feature_dim = capacity, feature_dim
        self.encoder_signature = encoder_signature
        self._entries = []

    def write(self, feature, *, predicates, instruction, root, split, measured_success):
        if split != 'training' or type(root) is not int or root not in range(10, 20):
            raise ValueError('Memory accepts training roots 10–19 only')
        if measured_success is not True or not instruction or not predicates:
            raise ValueError('Memory requires measured successful predicate context')
        from ..planning.schema import Predicate, ARITY
        for p in predicates:
            Predicate.parse(p.to_dict(), set(p.args), ARITY)
        self._entries.append(dict(feature=unit(feature, self.feature_dim), root=root, split=split,
            measured_success=True, instruction=instruction, predicates=[p.to_dict() for p in predicates]))
        self._entries = self._entries[-self.capacity:]

    def retrieve(self, feature, predicates, k=3):
        q = unit(feature, self.feature_dim)
        # Exact bindings are required: visually similar unrelated objects must
        # not be treated as successful examples of this generated clause.
        keys = {(p.op, p.args, p.expected) for p in predicates}
        candidates = []
        for entry in self._entries:
            labels = {(p['op'], tuple(p['args']), p.get('expected', True)) for p in entry['predicates']}
            if keys & labels:
                similarity = sum(a*b for a, b in zip(q, entry['feature']))
                candidates.append(dict(similarity=similarity,
                    **{key: deepcopy(value) for key, value in entry.items() if key != 'feature'}))
        return sorted(candidates, key=lambda e: -e['similarity'])[:k]

    def prototype(self, predicate):
        features = [e['feature'] for e in self._entries if predicate.to_dict() in e['predicates']]
        if not features:
            return None
        mean = tuple(sum(column)/len(features) for column in zip(*features))
        if sum(x*x for x in mean) < 1e-12:
            return None
        return unit(mean, self.feature_dim)

    def save(self, path):
        with Path(path).open('x') as stream:
            json.dump(dict(schema=self.schema, feature_dim=self.feature_dim,
                encoder_signature=self.encoder_signature, entries=self._entries), stream, allow_nan=False)

    @classmethod
    def load(cls, path, *, encoder_signature):
        data = json.loads(Path(path).read_text())
        if data.get('schema') != cls.schema or data.get('encoder_signature') != encoder_signature:
            raise ValueError('Memory schema/encoder provenance mismatch; old stage banks are incompatible')
        from ..planning.schema import Predicate, ARITY
        bank = cls(capacity=max(200, len(data['entries'])), feature_dim=data['feature_dim'],
                   encoder_signature=encoder_signature)
        for e in data['entries']:
            ids = {arg for p in e['predicates'] for arg in p['args']}
            predicates = tuple(Predicate.parse(p, ids, ARITY) for p in e['predicates'])
            bank.write(e['feature'], predicates=predicates, instruction=e['instruction'], root=e['root'],
                       split=e['split'], measured_success=e['measured_success'])
        if not bank._entries:
            raise ValueError('An explicitly loaded memory bank must contain training keyframes')
        return bank
