"""Training-only fixed-slot siblings, not a learned recovery scorer.

All four unchanged H1 proposals are queried; exactly the requested slot executes.
Only its observed outcome can become a label. No unexecuted sibling is labelled.
"""


class FixedCandidateSlot:
    def __init__(self, slot):
        if type(slot) is not int or slot not in range(4):
            raise ValueError('Candidate slot must be 0..3')
        self.slot = slot
        self.provenance = dict(method='training_fixed_candidate_slot', slot=slot, learned=False)

    def score(self, observation, task, plan, actions, physics):
        if len(actions) != 4:
            raise ValueError('Collection requires the unchanged four-proposal pool')
        return [float(i == self.slot) for i in range(4)]
