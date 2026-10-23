"""One owner of model queries and environment execution for every task.

An environment supplies observe(predicates), step(action), capabilities, done.
A policy supplies propose(image_png, instruction, sample=False, seed=0).
Optional A scores unchanged proposals; optional B assesses the exact fresh
nominal proposal that will execute on acceptance. Old stage-specific scorer
checkpoints do not implement these interfaces and are not auto-loaded.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import math

from .controller import Controller, Limits
from .planning.schema import TaskSpec


@dataclass(frozen=True)
class Proposal:
    action: tuple[float, ...]
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.action, tuple) or len(self.action) != 7 or not all(
                math.isfinite(float(v)) for v in self.action):
            raise ValueError('A frozen policy must return one finite seven-coordinate proposal')


class Session:
    def __init__(self, environment, policy, planner, *, limits=None, mode='framework',
                 recovery_scorer=None, handoff_scorer=None, record=None):
        self.env, self.policy, self.planner = environment, policy, planner
        self.limits, self.mode = limits or Limits(), mode
        self.recovery_scorer, self.handoff_scorer = recovery_scorer, handoff_scorer
        self.record = record or (lambda event: None)
        self.controller = None
        self._requested = set()
        self._pooled = False
        self._physics = None
        self._previous = self._last_action = None

    def _observe(self, predicates=()):
        self._requested.update(predicates)
        return self.env.observe(tuple(sorted(self._requested, key=lambda p: (p.key, p.expected))))

    def _query(self, observation, instruction, *, sample=False, seed=0):
        self.controller.charge_queries()
        request = dict(type='policy_query', query=self.controller.queries,
                       observation_index=observation.index, image_sha256=observation.to_dict()['image_sha256'],
                       instruction=instruction, sample=sample, seed=seed)
        self.record(request)
        proposal = self.policy.propose(observation.image_png, instruction, sample=sample, seed=seed)
        if not isinstance(proposal, Proposal):
            raise ValueError('Policy must return a Proposal')
        self.record(dict(type='policy_response', query=self.controller.queries,
                         action=proposal.action, provenance=proposal.provenance))
        return proposal

    def _target(self, task, observation):
        objects = {e.id for e in observation.entities if e.kind == 'object'}
        return next((a for p in task.goals for a in p.args if a in objects), None)

    def _select(self, observation, decision):
        c = self.controller
        pool = (decision['owner'] == 'repair' and not self._pooled and self.recovery_scorer is not None
                and c.budget['remaining_queries'] >= 4)
        proposals = [self._query(observation, decision['language'])]
        if not pool:
            return proposals[0]
        for i in range(1, 4):
            proposals.append(self._query(observation, decision['language'], sample=True,
                                         seed=20261022 + observation.index*4 + i))
        self._pooled = True
        actions = tuple(p.action for p in proposals)
        scores = self.recovery_scorer.score(deepcopy(observation), c.task, c.active['plan'],
                                             actions, deepcopy(self._physics))
        if len(scores) != 4 or not all(math.isfinite(float(s)) for s in scores):
            raise ValueError('Module A must return one finite score per unchanged proposal')
        selected = max(range(4), key=lambda i: float(scores[i]))
        self.record(dict(type='candidate_selection', observation_index=observation.index,
                         actions=actions, scores=[float(s) for s in scores], selected=selected,
                         physics=deepcopy(self._physics), schedule='first_repair_action_only'))
        return proposals[selected]

    def run(self, instruction):
        if self.controller is not None:
            raise RuntimeError('A Session runs exactly one episode')
        initial = self._observe()
        self.record(dict(type='observation', observation=initial))
        generated = self.planner.task(instruction, deepcopy(initial), self.env.capabilities)
        # Validate even when a different planner backend is injected.
        data = generated.to_dict()
        data.pop('instruction')
        task = TaskSpec.parse(data, instruction, {e.id for e in initial.entities}, self.env.capabilities)
        self.record(dict(type='task_generated', task=task.to_dict()))
        c = self.controller = Controller(task, self.planner, self.env.capabilities,
                                         limits=self.limits, mode=self.mode)
        observation = self._observe(c.predicates)
        target = self._target(task, observation)
        self._physics = c.verifier.evidence(observation, target=target)
        termination = 'episode_budget'
        while c.actions < self.limits.actions and c.queries < self.limits.queries:
            if c.verifier.check_all(task.goals, observation, [*c.history]) is True:
                c.complete = True
                termination = 'generated_task_complete'
                break
            if self.env.done:
                termination = 'environment_done'
                break
            decision = c.before(observation)
            refreshed = self._observe(c.predicates)
            if not c.refresh(refreshed):
                self.record(dict(type='repair_rejected', repair=c.summary['repairs'][-1]))
                observation = refreshed
                continue
            observation = refreshed
            self._physics = c.verifier.evidence(observation, target=target,
                                                previous=self._previous, executed_action=self._last_action)
            self.record(dict(type='decision', decision=decision, observation=observation,
                             physics=deepcopy(self._physics),
                             repair=c.active['plan'].to_dict() if c.active else None))
            proposal = self._select(observation, decision)
            assessment = None
            if decision['handoff']:
                if self.handoff_scorer is None:
                    assessment = dict(accept=True, method='measured_contract_only',
                                      predictive_resumability='unassessed')
                else:
                    assessment = self.handoff_scorer.assess(deepcopy(observation), task,
                        c.active['plan'], proposal.action, deepcopy(self._physics))
                    if not isinstance(assessment, dict) or type(assessment.get('accept')) is not bool:
                        raise ValueError('Module B must return an explicit boolean acceptance')
                assessment = dict(assessment, observation_index=observation.index,
                                  nominal_action=list(proposal.action))
                self.record(dict(type='handoff_assessment', assessment=assessment))
                if assessment['accept'] is not True:
                    c.reject_handoff(decision, assessment)
                    continue
            # Exactly one unchanged proposal is executed. No candidate lookahead
            # steps, branch rollouts, gripper overrides or action blending.
            self.env.step(proposal.action)
            endpoint = self._observe(c.predicates)
            result = c.after(endpoint, proposal.action, decision, handoff_assessment=assessment)
            self._physics = c.verifier.evidence(endpoint, target=target, previous=observation,
                                                executed_action=proposal.action)
            self.record(dict(type='execution', action=proposal.action, provenance=proposal.provenance,
                             decision=decision, observation=endpoint, result=result,
                             physics=deepcopy(self._physics)))
            self._previous, self._last_action = observation, proposal.action
            observation = endpoint
            if c.complete:
                termination = 'generated_task_complete'
                break
        c.finish(termination)
        result = dict(termination=termination, controller=c.summary,
                      native_success=observation.native_success,
                      planner_calls=getattr(self.planner, 'calls', None))
        self.record(dict(type='episode_end', **result))
        return result
