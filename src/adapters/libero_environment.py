"""LIBERO sensor/scene bridge shared by tasks, with no task-specific classes.

Scene metadata grounds names to IDs. Native goal clauses are used only by
LIBERO's evaluation; they are never supplied as the generated TaskSpec.
"""
from io import BytesIO

from ..observation import Entity, Observation
from ..planning.schema import Predicate


class LiberoEnvironment:
    capabilities = frozenset(('in', 'on', 'open', 'close', 'turnon', 'turnoff',
                              'grasped', 'contact', 'stable'))

    def __init__(self, env, observation, *, image_transform=None):
        if image_transform is None:
            from .libero_image import policy_image
            image_transform = policy_image
        self.env, self.raw, self.image_transform = env, observation, image_transform
        self.index, self.done = 0, False
        inner = env.env
        states = inner.object_states_dict
        objects = set(inner.objects_dict)
        self.entities = tuple(Entity(name, name.replace('_', ' '),
                             'object' if name in objects else 'region'
                             if getattr(state, 'object_state_type', '') == 'site' else 'fixture')
                              for name, state in sorted(states.items()))
        if 'gripper' in states:
            raise ValueError('Scene entity collides with the reserved gripper ID')
        self.entities += (Entity('gripper', 'robot gripper', 'gripper'),)
        self._cached = None

    def _contact_geoms(self, entity):
        inner = self.env.env
        if entity == 'gripper':
            return tuple(inner.robots[0].gripper.contact_geoms)
        objects = dict(inner.fixtures_dict, **inner.objects_dict)
        obj = objects.get(entity)
        return None if obj is None else tuple(obj.contact_geoms)

    def _measure(self, predicate):
        inner = self.env.env
        try:
            if predicate.op == 'stable':
                return None  # Computed causally from endpoint history by the verifier.
            if predicate.op == 'grasped':
                obj = inner.objects_dict.get(predicate.args[0])
                return None if obj is None else bool(inner._check_grasp(inner.robots[0].gripper, obj.contact_geoms))
            if predicate.op == 'contact':
                a, b = (self._contact_geoms(x) for x in predicate.args)
                return None if a is None or b is None else bool(inner.check_contact(a, b))
            return bool(inner._eval_predicate([predicate.op, *predicate.args]))
        except (KeyError, AttributeError, ValueError, TypeError, AssertionError):
            return None

    def observe(self, predicates=()):
        if self._cached is None:
            inner, sim = self.env.env, self.env.sim
            positions = {}
            for entity in self.entities:
                value = self.raw.get(entity.id+'_pos')
                if value is not None:
                    positions[entity.id] = tuple(float(v) for v in value)
            # Contacts are explicitly bound to entity IDs, not substring matches.
            mapping = {}
            for entity in self.entities:
                for geom in self._contact_geoms(entity.id) or ():
                    mapping[geom] = entity.id
            contacts = []
            for contact in sim.data.contact[:sim.data.ncon]:
                a, b = (mapping.get(sim.model.geom_id2name(g)) for g in (contact.geom1, contact.geom2))
                if a is not None and b is not None:
                    contacts.append((a, b))
            buffer = BytesIO()
            self.image_transform(self.raw['agentview_image']).save(buffer, format='PNG')
            facts = {}
            for entity in self.entities:
                if entity.kind == 'object':
                    p = Predicate('grasped', (entity.id,))
                    facts[p.key] = self._measure(p)
            eef = self.raw.get('robot0_eef_pos')
            self._cached = Observation(self.index, buffer.getvalue(), self.entities, facts, positions,
                tuple(contacts), None if eef is None else tuple(float(v) for v in eef),
                bool(self.env.check_success()))
        # Enrichment only; callers receive copies so past observations cannot change.
        from copy import deepcopy
        for predicate in predicates:
            if predicate.key not in self._cached.facts and predicate.op != 'stable':
                self._cached.facts[predicate.key] = self._measure(predicate)
        return deepcopy(self._cached)

    def step(self, action):
        self.raw, _, done, _ = self.env.step(list(action))
        self.index += 1
        self.done = bool(done)
        self._cached = None

    def close(self):
        self.env.close()
