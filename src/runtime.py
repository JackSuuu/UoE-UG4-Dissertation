"""One owner of model queries and environment execution for every task.

An environment supplies observe(predicates), step(action), capabilities, done.
A policy supplies propose(image_png, instruction, sample=False, seed=0).
Optional A scores unchanged proposals; optional B assesses the exact fresh
nominal proposal that will execute on acceptance. Old stage-specific scorer
checkpoints do not implement these interfaces and are not auto-loaded.
"""
from copy import deepcopy
import math

from .controller import Controller, Limits
from .interfaces import Environment, HandoffScorer, Planner, Policy, Proposal, RecoveryScorer
from .observation import Observation
from .planning.schema import ARITY, TaskSpec, text
from .verification.vision_physical_monitor import VisionPhysicalMonitor


class Session:
    def __init__(self, environment: Environment, policy: Policy, planner: Planner, *,
                 limits: Limits | None = None, mode='framework',
                  recovery_scorer: RecoveryScorer | None = None,
                  handoff_scorer: HandoffScorer | None = None, monitor=None, record=None):
        if mode not in ('framework', 'baseline'):
            raise ValueError('Expected framework or baseline')
        if limits is not None and not isinstance(limits, Limits):
            raise TypeError('limits must be a Limits record')
        for backend, methods in ((environment, ('observe', 'step')), (policy, ('propose',)),
                                 (planner, ('task', 'recover'))):
            if any(not callable(getattr(backend, method, None)) for method in methods):
                raise TypeError(f'Backend must implement {methods}')
        if (not isinstance(environment.capabilities, frozenset)
                or not environment.capabilities <= ARITY.keys() or type(environment.done) is not bool):
            raise TypeError('Environment must expose supported capabilities and a boolean done flag')
        for scorer, method in ((recovery_scorer, 'score'), (handoff_scorer, 'assess')):
            if scorer is not None and not callable(getattr(scorer, method, None)):
                raise TypeError(f'Scorer must implement {method}')
        self.env, self.policy, self.planner = environment, policy, planner
        self.limits, self.mode = limits or Limits(), mode
        self.recovery_scorer, self.handoff_scorer = recovery_scorer, handoff_scorer
        self.monitor = monitor if monitor is not None else VisionPhysicalMonitor()
        if not callable(getattr(self.monitor, 'observe', None)):
            raise TypeError('Monitor must implement observe')
        self.record = record or (lambda event: None)
        self.controller = None
        self._started = False
        self._requested = set()
        self._pooled = False
        self._physics = None
        self._previous = self._last_action = None

    def _context(self, observation, task):
        context = self.monitor.observe(deepcopy(observation), task, deepcopy(self._physics))
        self.record(dict(type='monitor', observation_index=observation.index, **context))
        return context['context']

    def _observe(self, predicates=()):
        self._requested.update(predicates)
        observation = self.env.observe(tuple(sorted(self._requested, key=lambda p: (p.key, p.expected))))
        if not isinstance(observation, Observation):
            raise TypeError('Environment.observe must return an Observation')
        return observation

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
        if (not isinstance(scores, (list, tuple)) or len(scores) != 4
                or any(type(s) not in (int, float) or not math.isfinite(s) for s in scores)):
            raise ValueError('Module A must return one finite score per unchanged proposal')
        selected = max(range(4), key=lambda i: float(scores[i]))
        self.record(dict(type='candidate_selection', observation_index=observation.index,
                          actions=actions, scores=[float(s) for s in scores], selected=selected,
                          features=[self.monitor.action_features(observation, c.task, c.active['plan'], a,
                                    self._physics) for a in actions] if self.monitor.encoder else None,
                         physics=deepcopy(self._physics), schedule='first_repair_action_only'))
        return proposals[selected]

    def run(self, instruction):
        if self._started:
            raise RuntimeError('A Session runs exactly one episode')
        text(instruction)
        self._started = True
        initial = self._observe()
        if initial.index != 0:
            raise ValueError('A Session must start at endpoint zero')
        self.record(dict(type='observation', observation=initial))
        generated = self.planner.task(instruction, deepcopy(initial), self.env.capabilities)
        if not isinstance(generated, TaskSpec):
            raise TypeError('Planner.task must return a TaskSpec')
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
            self._physics = c.verifier.evidence(observation, target=target,
                                                previous=self._previous, executed_action=self._last_action)
            context = self._context(observation, task)
            decision = c.before(observation, context=context)
            refreshed = self._observe(c.predicates)
            if not c.refresh(refreshed):
                self.record(dict(type='repair_rejected', repair=c.summary['repairs'][-1]))
                observation = refreshed
                continue
            observation = refreshed
            self._physics = c.verifier.evidence(observation, target=target,
                                                 previous=self._previous, executed_action=self._last_action)
            # Generated clauses may enrich this unchanged frame. Encoding is
            # cached, and risk persistence never increments on repeated reads.
            context = self._context(observation, task)
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
                if context.get('handoff_veto') is True:
                    assessment.update(accept=False, risk_veto=context['risk'])
                self.record(dict(type='handoff_assessment', assessment=assessment))
                if assessment['accept'] is not True:
                    c.reject_handoff(decision, assessment)
                    continue
            # Exactly one unchanged proposal is executed. No candidate lookahead
            # steps, branch rollouts, gripper overrides or action blending.
            model_features = (self.monitor.action_features(observation, task, c.active['plan'] if c.active else None,
                              proposal.action, self._physics) if self.monitor.encoder else None)
            self.env.step(proposal.action)
            endpoint = self._observe(c.predicates)
            result = c.after(endpoint, proposal.action, decision, handoff_assessment=assessment)
            self._physics = c.verifier.evidence(endpoint, target=target, previous=observation,
                                                executed_action=proposal.action)
            self.record(dict(type='execution', action=proposal.action, provenance=proposal.provenance,
                              decision=decision, observation=endpoint, result=result,
                              physics=deepcopy(self._physics), model_features=model_features))
            self._previous, self._last_action = observation, proposal.action
            observation = endpoint
            if c.complete:
                termination = 'generated_task_complete'
                break
        self._context(observation, task)
        c.finish(termination)
        result = dict(termination=termination, controller=c.summary,
                       native_success=observation.native_success,
                       components=self.monitor.provenance,
                      planner_calls=getattr(self.planner, 'calls', None))
        self.record(dict(type='episode_end', **result))
        return result
