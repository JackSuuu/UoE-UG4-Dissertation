"""Environment-neutral endpoint observations. Indices count executed actions."""
from dataclasses import dataclass, field
import hashlib
import json
import math


@dataclass(frozen=True)
class Entity:
    id: str
    label: str
    kind: str

    def to_dict(self):
        return dict(id=self.id, label=self.label, kind=self.kind)


@dataclass
class Observation:
    index: int
    image_png: bytes
    entities: tuple[Entity, ...]
    facts: dict[str, bool | None] = field(default_factory=dict)
    positions: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    contacts: tuple[tuple[str, str], ...] | None = None
    eef_position: tuple[float, float, float] | None = None
    native_success: bool | None = None

    def __post_init__(self):
        ids = [e.id for e in self.entities]
        if type(self.index) is not int or self.index < 0 or len(ids) != len(set(ids)):
            raise ValueError('Invalid observation index or duplicate entity IDs')
        if not isinstance(self.image_png, bytes):
            raise ValueError('Observation image must be encoded PNG bytes')
        if any(type(v) is not bool and v is not None for v in self.facts.values()):
            raise ValueError('Measured facts must be true, false or unknown')
        for p in [*self.positions.values(), self.eef_position]:
            if p is not None and (len(p) != 3 or not all(math.isfinite(v) for v in p)):
                raise ValueError('Positions must be finite three-vectors')

    def to_dict(self):
        return dict(index=self.index, image_sha256=hashlib.sha256(self.image_png).hexdigest(),
                    entities=[e.to_dict() for e in self.entities], facts=self.facts,
                    positions=self.positions, contacts=self.contacts, eef_position=self.eef_position,
                    native_success=self.native_success)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
                                        allow_nan=False).encode()).hexdigest()
