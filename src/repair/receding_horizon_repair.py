"""Frozen-policy H1 repair of the remaining execution, not numeric offsets.

OpenVLA's installed actor emits ONE action, not a multi-action chunk. We revise
the remaining stage plan on measured/predicted failure, then request each repair
action from the same frozen actor on its fresh image. No cached numeric chunk is
claimed to exist and no extra hidden model query is made.
"""
class RecedingHorizonRepair:
    def __init__(self):
        self.generation = 0
        self.last_repair = self.last_stage = None

    def request(self, decision):
        changed = decision['owner'] == 'repair' and (
            decision['repair_id'] != self.last_repair or decision['subgoal'] != self.last_stage)
        if changed:
            self.generation += 1
        event = dict(protocol='frozen_policy_h1_remaining_execution_v1',
            owner=decision['owner'], generation=self.generation,
            stage_plan_revised=changed, numeric_chunk_invalidated=False,
            model_generated_repair=decision['owner'] == 'repair',
            selected_stage=decision['subgoal'], repair_id=decision['repair_id'],
            requested_language=decision['language'], additional_queries=0)
        self.last_repair = decision['repair_id']
        self.last_stage = decision['subgoal']
        return dict(language=decision['language'], source='greedy', seed=0), event
