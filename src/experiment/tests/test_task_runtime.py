"""CPU integration checks for generated semantics and execution ownership."""
import base64
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.controller import Controller, Limits
from src.journal import Journal
from src.observation import Entity, Observation
from src.planning.multimodal import ChatPlanner
from src.planning.schema import ARITY, PlanningError, Predicate, RecoveryPlan, TaskSpec
from src.runtime import Proposal, Session
from src.verification.physical_verifier import PhysicalVerifier
from src.experiment.run_task import load_scorers, parse_args, preflight


PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aWQAAAABJRU5ErkJggg==')
ENTITIES = (Entity('item', 'yellow and white mug', 'object'),
            Entity('box', 'microwave', 'fixture'), Entity('region', 'inside microwave', 'region'),
            Entity('gripper', 'robot gripper', 'gripper'))
IDS = {e.id for e in ENTITIES}
CAPABILITIES = frozenset(ARITY)
IN = Predicate('in', ('item', 'region'))
CLOSED = Predicate('close', ('box',))
GRASP = Predicate('grasped', ('item',))
CONTACT = Predicate('contact', ('item', 'gripper'))
STABLE = Predicate('stable', ('item',))


def task(goals=(IN,), milestones=()):
    return TaskSpec('put the yellow and white mug in the microwave', tuple(goals), tuple(milestones))


def recovery(alternatives=((IN,),), preserve=()):
    return RecoveryPlan('finish placing the mug', tuple(tuple(g) for g in alternatives), tuple(preserve),
                        'progress stopped')


def frame(index=0, facts=None, *, position=(0., 0., 0.), entities=ENTITIES):
    return Observation(index, PNG, entities, {p.key: v for p, v in (facts or {}).items()},
                       {'item': position}, (), (float(index)*.01, 0., 0.), False)


class FixturePlanner:
    """Explicit test fixture, never used as a production language parser."""
    def __init__(self, spec=None, plan=None):
        self.spec, self.plan = spec or task((IN, CLOSED)), plan or recovery()
        self.calls, self.recoveries = 0, []

    def task(self, instruction, observation, capabilities):
        self.calls += 1
        return replace(self.spec, instruction=instruction)

    def recover(self, task, observation, capabilities, event, history):
        self.calls += 1
        self.recoveries.append((deepcopy(observation), deepcopy(event), deepcopy(history)))
        return self.plan


class Environment:
    capabilities = CAPABILITIES
    done = False

    def __init__(self, states, entities=ENTITIES):
        self.states, self.entities = states, entities
        self.index, self.executed = 0, []
        self.requested = set()

    def observe(self, predicates=()):
        self.requested.update(predicates)
        data = self.states[min(self.index, len(self.states)-1)]
        values = {p: data.get(p) for p in self.requested if p.op != 'stable'}
        return frame(self.index, values, entities=self.entities)

    def step(self, action):
        self.executed.append(tuple(action))
        self.index += 1


class Policy:
    def __init__(self):
        self.requests, self.proposals = [], []

    def propose(self, image_png, instruction, *, sample=False, seed=0):
        self.requests.append((image_png, instruction, sample, seed))
        proposal = Proposal((len(self.requests)*.01, 0., 0., 0., 0., 0., -1.), {'query': len(self.requests)})
        self.proposals.append(proposal)
        return proposal


class SchemaTests(unittest.TestCase):
    def test_no_implicit_close_goal(self):
        spec = TaskSpec.parse(dict(goals=[IN.to_dict()], milestones=[]), task().instruction, IDS, CAPABILITIES)
        self.assertEqual(spec.goals, (IN,))

    def test_multi_object_goals_without_target_receptacle_fields(self):
        ids = {'a', 'b', 'plate'}
        spec = TaskSpec.parse(dict(goals=[dict(op='on', args=[x, 'plate']) for x in ('a', 'b')], milestones=[]),
                              'put both items on the plate', ids, CAPABILITIES)
        self.assertEqual(len(spec.goals), 2)

    def test_unbound_unsupported_malformed_and_code_rejected(self):
        cases = [dict(op='fly', args=['item']), dict(op='in', args=['missing', 'region']),
                 dict(op='grasped', args=['item'], expected='false'),
                 dict(op='grasped', args=['item'], code='arbitrary Python')]
        for value in cases:
            with self.subTest(value=value), self.assertRaises(PlanningError):
                Predicate.parse(value, IDS, CAPABILITIES)
        with self.assertRaises(PlanningError):
            RecoveryPlan.parse(dict(instruction='x', acceptance=[[]], preserve=[], reason='x'), IDS, CAPABILITIES)
        with self.assertRaises(PlanningError):
            RecoveryPlan.parse(dict(instruction='x', acceptance=[[IN.to_dict()]], preserve=[], reason='x',
                                    local_cap=10000), IDS, CAPABILITIES)

    def test_contradictory_clauses_rejected(self):
        data = dict(goals=[IN.to_dict(), replace(IN, expected=False).to_dict()], milestones=[])
        with self.assertRaises(PlanningError):
            TaskSpec.parse(data, 'place it', IDS, CAPABILITIES)
        data = recovery(preserve=(replace(IN, expected=False),)).to_dict()
        with self.assertRaises(PlanningError):
            RecoveryPlan.parse(data, IDS, CAPABILITIES)


class PlannerTests(unittest.TestCase):
    def reply(self, data, finish='stop'):
        return io.BytesIO(json.dumps(dict(choices=[dict(finish_reason=finish,
                             message=dict(content=json.dumps(data)))])).encode())

    def test_multimodal_request_and_audit(self):
        records = []
        planner = ChatPlanner('http://localhost:8000/v1', 'small-vlm', record=records.append)
        with patch('src.planning.multimodal.urlopen', return_value=self.reply(
                dict(goals=[IN.to_dict()], milestones=[]))) as send:
            result = planner.task(task().instruction, frame(), CAPABILITIES)
        payload = json.loads(send.call_args.args[0].data)
        content = payload['messages'][1]['content']
        self.assertEqual(content[1]['type'], 'image_url')
        self.assertEqual(base64.b64decode(content[1]['image_url']['url'].split(',')[1]), PNG)
        self.assertNotIn('native_success', json.loads(content[0]['text'])['observation'])
        self.assertEqual(result.goals, (IN,))
        self.assertEqual(records[0]['model'], 'small-vlm')
        self.assertIn('response', records[0])

    def test_no_text_fallback_timeout_and_query_cap(self):
        planner = ChatPlanner('http://localhost/v1', 'vlm', max_calls=1)
        with patch('src.planning.multimodal.urlopen', side_effect=TimeoutError('timeout')) as send:
            with self.assertRaises(PlanningError):
                planner.task('place it', frame(), CAPABILITIES)
            with self.assertRaises(PlanningError):
                planner.task('place it', frame(), CAPABILITIES)
        self.assertEqual(send.call_count, 1)
        with self.assertRaises(PlanningError):
            ChatPlanner('http://localhost/v1', 'vlm').task('place it', replace(frame(), image_png=b''), CAPABILITIES)

    def test_truncated_output_and_unsupported_grounding(self):
        for data, finish in [(dict(goals=[IN.to_dict()], milestones=[]), 'length'),
                             (dict(unsupported='ambiguous mug'), 'stop')]:
            planner = ChatPlanner('http://localhost/v1', 'vlm')
            with patch('src.planning.multimodal.urlopen', return_value=self.reply(data, finish)):
                with self.assertRaises(PlanningError):
                    planner.task('place it', frame(), CAPABILITIES)

    def test_recovery_generated_from_fresh_observation_can_defer(self):
        planner = ChatPlanner('http://localhost/v1', 'vlm')
        with patch('src.planning.multimodal.urlopen', return_value=self.reply(dict(defer='intentional release'))):
            self.assertIsNone(planner.recover(task(), frame(9), CAPABILITIES,
                                             {'reason': 'progress_lost'}, []))


class PhysicalTests(unittest.TestCase):
    def test_unknown_negated_predicate_does_not_pass(self):
        verifier = PhysicalVerifier()
        self.assertIsNone(verifier.evaluate(replace(GRASP, expected=False), frame()))
        self.assertIs(verifier.evaluate(replace(GRASP, expected=False), frame(facts={GRASP: False})), True)

    def test_stability_uses_executed_consecutive_endpoints(self):
        verifier = PhysicalVerifier()
        for indices in ((0, 1, 2), (1, 2), (1, 3, 4)):
            history = [frame(i) for i in indices]
            self.assertIsNone(verifier.evaluate(STABLE, history[-1], history))
        history = [frame(i) for i in (1, 2, 3)]
        self.assertIs(verifier.evaluate(STABLE, history[-1], history), True)
        history[-1].positions['item'] = (.02, 0., 0.)
        self.assertIs(verifier.evaluate(STABLE, history[-1], history), False)

    def test_four_dimensions_validity_and_causal_alignment(self):
        verifier = PhysicalVerifier()
        before, after = frame(0), frame(1, {IN: True, GRASP: False})
        evidence = verifier.evidence(after, target='item')
        self.assertEqual(len(evidence['values']), 4)
        self.assertEqual(evidence['valid'], [True, True, False, True])
        self.assertFalse(evidence['prospective'])
        executed = verifier.evidence(after, target='item', previous=before,
                                     executed_action=[1., 0., 0., 0., 0., 0., 0.])
        self.assertEqual(executed['values'][2], 1.)
        stale = verifier.evidence(frame(5), target='item', previous=before,
                                  executed_action=[1., 0., 0., 0., 0., 0., 0.])
        self.assertFalse(stale['valid'][2])


class SessionTests(unittest.TestCase):
    def run_session(self, states, *, plan=None, spec=None, limits=None, **kwargs):
        env, policy = Environment(states), Policy()
        planner, records = FixturePlanner(spec, plan), []
        session = Session(env, policy, planner, limits=limits or Limits(actions=16, queries=24, stagnation=1),
                          record=records.append, **kwargs)
        result = session.run((spec or task((IN, CLOSED))).instruction)
        return result, env, policy, planner, records

    def test_placement_without_close_finishes(self):
        result, env, policy, _, _ = self.run_session([{IN: False}, {IN: True}], spec=task())
        self.assertTrue(result['controller']['complete'])
        self.assertEqual(len(env.executed), 1)
        self.assertFalse(result['native_success'])  # Distinct from generated task completion.
        self.assertEqual(policy.requests[0][1], task().instruction)

    def test_same_runtime_on_articulated_task_with_different_entities(self):
        opened = Predicate('open', ('drawer',))
        spec = TaskSpec('open the drawer', (opened,), ())
        env = Environment([{opened: False}, {opened: True}],
                          (Entity('drawer', 'desk drawer', 'fixture'),))
        result = Session(env, Policy(), FixturePlanner(spec)).run('open the drawer')
        self.assertTrue(result['controller']['complete'])

    def test_fixed_contract_three_checks_then_actual_nominal_handoff(self):
        states = [{IN: False, CLOSED: False}]*2 + [{IN: True, CLOSED: False}]*4 + [{IN: True, CLOSED: True}]
        result, env, policy, planner, records = self.run_session(states)
        repair = result['controller']['repairs'][0]
        self.assertEqual(repair['used'], 3)
        self.assertEqual(repair['handoff_step'], 4)
        self.assertEqual(repair['plan']['acceptance'], [[IN.to_dict()]])
        self.assertTrue(repair['verified'])
        self.assertEqual(policy.requests[4][1], task().instruction)
        self.assertEqual(planner.recoveries[0][0].index, 1)
        self.assertEqual(env.executed, [p.action for p in policy.proposals])
        exits = [e for r in records if r['type'] == 'execution' for e in r['result']['events']]
        self.assertEqual(sum(e['type'] == 'handoff_executed' for e in exits), 1)

    def test_regrasp_and_missing_contact_prevent_release_verification(self):
        release = recovery(((IN, replace(GRASP, expected=False), replace(CONTACT, expected=False), STABLE),))
        for state in ({IN: True, CLOSED: False, GRASP: True, CONTACT: False},
                      {IN: True, CLOSED: False, GRASP: False}):
            result, _, _, _, _ = self.run_session([{IN: False, CLOSED: False}]*2+[state]*12,
                plan=release, limits=Limits(actions=10, queries=12, local_repair=6, stagnation=1))
            self.assertFalse(result['controller']['repairs'][0]['verified'])
            self.assertEqual(result['controller']['repairs'][0]['exit_reason'], 'local_budget_exit_unverified')

    def test_alternative_streaks_cannot_mix(self):
        plan = recovery(((IN,), (GRASP,)))
        states = [{IN: False, CLOSED: False, GRASP: False}]*2
        states += [{IN: i % 2 == 0, GRASP: i % 2 != 0, CLOSED: False} for i in range(10)]
        result, _, _, _, _ = self.run_session(states, plan=plan,
            limits=Limits(actions=9, queries=10, local_repair=6, stagnation=1))
        self.assertFalse(result['controller']['repairs'][0]['verified'])

    def test_budget_tie_is_unverified_and_handoff_only_after_execution(self):
        states = [{IN: False, CLOSED: False}]*2 + [{IN: True, CLOSED: False}]*5
        result, _, _, _, _ = self.run_session(states,
            limits=Limits(actions=6, queries=6, local_repair=3, stagnation=1))
        repair = result['controller']['repairs'][0]
        self.assertEqual(repair['exit_reason'], 'local_budget_exit_unverified')
        self.assertFalse(repair['verified'])
        self.assertEqual(repair['handoff_step'], 4)

    def test_preservation_unknown_is_not_a_verified_exit(self):
        plan = recovery(preserve=(GRASP,))
        states = [{IN: False, CLOSED: False, GRASP: True}]*2
        states += [{IN: True, CLOSED: False, GRASP: None}]*8
        result, _, _, _, _ = self.run_session(states, plan=plan,
            limits=Limits(actions=7, queries=7, local_repair=5, stagnation=1))
        self.assertFalse(result['controller']['repairs'][0]['verified'])
        self.assertTrue(result['controller']['repairs'][0]['preserve_lost'])

    def test_invalid_entry_preservation_does_not_execute_repair(self):
        result, env, policy, planner, _ = self.run_session([{IN: False, CLOSED: False, GRASP: False}]*8,
            plan=recovery(preserve=(GRASP,)), limits=Limits(actions=5, queries=5, stagnation=1))
        self.assertEqual(result['controller']['budget']['repair_actions'], 0)
        self.assertEqual(len(planner.recoveries), 1)
        self.assertTrue(all(r[1] == task().instruction for r in policy.requests))

    def test_module_a_selects_unchanged_candidate_and_counts_all_queries(self):
        class Ranker:
            def score(self, observation, task, plan, actions, physics):
                self.context = (observation.index, actions, physics)
                return [0., 1., 3., 2.]
        ranker = Ranker()
        result, env, policy, _, _ = self.run_session([{IN: False, CLOSED: False}]*8,
            recovery_scorer=ranker, limits=Limits(actions=4, queries=7, stagnation=1))
        self.assertEqual(len(policy.proposals), 7)
        self.assertEqual(len(env.executed), 4)
        self.assertEqual(env.executed[1], policy.proposals[3].action)
        self.assertEqual(ranker.context[0], 1)
        self.assertEqual([r[2] for r in policy.requests[:5]], [False, False, True, True, True])
        self.assertEqual(len({r[0] for r in policy.requests[1:5]}), 1)
        self.assertFalse(ranker.context[2]['prospective'])
        self.assertEqual(result['controller']['budget']['queries'], 7)

    def test_module_b_rejects_nominal_query_then_executes_fresh_accepted_proposal(self):
        class Gate:
            def __init__(self):
                self.calls = []
            def assess(self, observation, task, plan, action, physics):
                self.calls.append((observation.index, action))
                return dict(accept=len(self.calls) > 1)
        gate = Gate()
        states = [{IN: False, CLOSED: False}]*2+[{IN: True, CLOSED: False}]*7
        result, env, policy, _, _ = self.run_session(states, handoff_scorer=gate,
            limits=Limits(actions=7, queries=8, stagnation=1))
        self.assertEqual(len(gate.calls), 2)
        self.assertNotIn(gate.calls[0][1], env.executed)
        self.assertIn(gate.calls[1][1], env.executed)
        self.assertEqual(result['controller']['budget']['queries'], len(env.executed)+1)
        self.assertEqual(result['controller']['repairs'][0]['handoff_step'], 5)

    def test_query_failure_is_counted_without_execution(self):
        class BrokenPolicy(Policy):
            def propose(self, *args, **kwargs):
                raise RuntimeError('actor failed')
        env = Environment([{IN: False}])
        session = Session(env, BrokenPolicy(), FixturePlanner(task()))
        with self.assertRaises(RuntimeError):
            session.run('place it')
        self.assertEqual(session.controller.queries, 1)
        self.assertEqual(session.controller.actions, 0)
        self.assertEqual(env.executed, [])

    def test_handoff_rejection_on_last_query_has_no_phantom_execution(self):
        class Reject:
            def assess(self, *args):
                return dict(accept=False)
        states = [{IN: False, CLOSED: False}]*2+[{IN: True, CLOSED: False}]*6
        result, env, _, _, _ = self.run_session(states, handoff_scorer=Reject(),
            limits=Limits(actions=10, queries=5, stagnation=1))
        repair = result['controller']['repairs'][0]
        self.assertEqual(len(env.executed), 4)
        self.assertEqual(repair['exit_reason'], 'episode_budget_exit_unverified')
        self.assertIsNone(repair['handoff_step'])
        self.assertFalse(repair['verified'])

    def test_post_handoff_regression_is_recorded_separately(self):
        states = [{IN: False, CLOSED: False}]*2+[{IN: True, CLOSED: False}]*3
        states += [{IN: False, CLOSED: False}]*4
        result, _, _, _, _ = self.run_session(states,
            limits=Limits(actions=6, queries=6, stagnation=1))
        repair = result['controller']['repairs'][0]
        self.assertTrue(repair['verified'])
        self.assertEqual(repair['post_handoff_loss_step'], repair['handoff_step'])
        self.assertFalse(repair['completed_after_handoff'])

    def test_total_repair_budget_and_completion_priority(self):
        states = [{IN: False, CLOSED: False}]*2+[{IN: True, CLOSED: False}]*5
        result, _, _, _, _ = self.run_session(states,
            limits=Limits(actions=6, queries=6, local_repair=5, total_repair=3, stagnation=1))
        self.assertEqual(result['controller']['repairs'][0]['exit_reason'], 'total_repair_budget_exit_unverified')
        self.assertEqual(result['controller']['budget']['repair_actions'], 3)
        result, _, _, _, _ = self.run_session(
            [{IN: False, CLOSED: False}]*2+[{IN: True, CLOSED: True}]*5,
            limits=Limits(actions=6, queries=6, local_repair=1, stagnation=1))
        repair = result['controller']['repairs'][0]
        self.assertEqual(repair['exit_reason'], 'task_completion_during_repair')
        self.assertIsNone(repair['handoff_step'])

    def test_invalid_generated_recovery_is_blocked_without_execution(self):
        invalid = recovery(((Predicate('fly', ('item',)),),))
        result, _, policy, planner, records = self.run_session([{IN: False, CLOSED: False}]*8,
            plan=invalid, limits=Limits(actions=5, queries=5, stagnation=1))
        self.assertEqual(result['controller']['repairs'], [])
        self.assertEqual(len(planner.recoveries), 1)
        self.assertTrue(any('planning_unavailable' in (r['decision']['blocked'] or '')
                            for r in records if r['type'] == 'decision'))
        self.assertTrue(all(r[1] == task().instruction for r in policy.requests))

    def test_baseline_never_requests_recovery(self):
        result, _, policy, planner, _ = self.run_session([{IN: False, CLOSED: False}]*8, mode='baseline',
            limits=Limits(actions=5, queries=5, stagnation=1))
        self.assertEqual(planner.recoveries, [])
        self.assertEqual(result['controller']['budget']['repair_actions'], 0)
        self.assertTrue(all(r[1] == task().instruction for r in policy.requests))

    def test_rejects_stale_endpoint_and_modified_decision(self):
        c = Controller(task(), FixturePlanner(), CAPABILITIES)
        decision = c.before(frame(0, {IN: False}))
        with self.assertRaises(ValueError):
            c.after(frame(0, {IN: False}), [0.]*7, decision)
        with self.assertRaises(ValueError):
            c.after(frame(1, {IN: False}), [0.]*7, dict(decision, language='edited'))
        c.charge_queries()
        c.after(frame(1, {IN: False}), [0.]*7, decision)
        with self.assertRaises(ValueError):
            c.before(frame(1, {IN: True}))


class JournalTests(unittest.TestCase):
    def test_observation_inputs_and_events_are_durable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'episode'
            journal = Journal(path)
            journal.record(dict(type='observation', observation=frame()))
            journal.save('result.json', dict(success=False))
            journal.close()
            event = json.loads((path/'events.jsonl').read_text())
            self.assertEqual((path/event['observation']['image']).read_bytes(), PNG)
            self.assertEqual(json.loads((path/'result.json').read_text()), dict(success=False))


class InterfaceTests(unittest.TestCase):
    def test_public_api_has_no_ml_imports(self):
        import subprocess
        process = subprocess.run([sys.executable, '-B', '-c',
            'from src import Session, Limits, Proposal; import sys; '
            'assert not {"torch", "numpy", "transformers", "tensorflow"} & sys.modules.keys()'],
            cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_proposal_rejects_non_numeric_nonfinite_and_wrong_horizon(self):
        for action in ([0.]*7, (0.,)*6, ((0.,)*7,), ('0',)*7, (True,)*7,
                       (float('nan'),)*7, (float('inf'),)*7):
            with self.subTest(action=action), self.assertRaises(ValueError):
                Proposal(action)
        with self.assertRaises(ValueError):
            Proposal((0.,)*7, [])

    def test_invalid_interfaces_fail_before_any_query(self):
        env, policy, planner = Environment([{}]), Policy(), FixturePlanner()
        for kwargs in (dict(mode='old'), dict(limits={}), dict(recovery_scorer=object()),
                       dict(handoff_scorer=object())):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                Session(env, policy, planner, **kwargs)
        with self.assertRaises(TypeError):
            Session(env, object(), planner)
        self.assertEqual(policy.requests, [])
        self.assertEqual(planner.calls, 0)

    def test_bad_task_output_and_nonzero_initial_endpoint(self):
        planner = FixturePlanner()
        planner.task = lambda *args: {}
        session = Session(Environment([{}]), Policy(), planner)
        with self.assertRaises(TypeError):
            session.run('place it')
        with self.assertRaises(RuntimeError):
            session.run('place it')
        env = Environment([{}])
        env.index = 1
        with self.assertRaises(ValueError):
            Session(env, Policy(), FixturePlanner()).run('place it')

    def test_malformed_recovery_output_is_blocked(self):
        planner = FixturePlanner()
        planner.recover = lambda *args: {}
        env = Environment([{IN: False}]*5)
        result = Session(env, Policy(), planner,
                         limits=Limits(actions=3, queries=3, stagnation=1)).run('place it')
        self.assertEqual(result['controller']['repairs'], [])
        self.assertEqual(result['controller']['budget']['repair_actions'], 0)

    def test_bad_scorer_outputs_never_execute_a_candidate(self):
        class BadRanker:
            def score(self, *args):
                return ['1']*4
        env = Environment([{IN: False, CLOSED: False}]*8)
        session = Session(env, Policy(), FixturePlanner(), recovery_scorer=BadRanker(),
                          limits=Limits(actions=5, queries=10, stagnation=1))
        with self.assertRaises(ValueError):
            session.run('place it and close it')
        self.assertEqual(len(env.executed), 1)
        self.assertEqual(session.controller.queries, 5)

    def test_handoff_must_match_assessed_action_and_endpoint(self):
        c = Controller(task((IN, CLOSED)), FixturePlanner(), CAPABILITIES,
                       limits=Limits(stagnation=1))
        # Drive the real controller to a measured ready contract.
        for index in range(4):
            observation = frame(index, {IN: index >= 2, CLOSED: False})
            decision = c.before(observation)
            c.refresh(observation)
            c.charge_queries()
            c.after(frame(index+1, {IN: index >= 1, CLOSED: False}), (0.,)*7, decision)
        decision = c.before(frame(4, {IN: True, CLOSED: False}))
        self.assertTrue(decision['handoff'])
        c.charge_queries()
        for assessment in (dict(accept=True),
                           dict(accept=True, nominal_action=[1.]*7, observation_index=4),
                           dict(accept=True, nominal_action=[0.]*7, observation_index=3)):
            with self.subTest(assessment=assessment), self.assertRaises(ValueError):
                c.after(frame(5, {IN: True, CLOSED: False}), (0.,)*7, decision,
                        handoff_assessment=assessment)

    def test_planner_bounds_reject_nonfinite_and_fractional_limits(self):
        for kwargs in (dict(timeout=float('nan')), dict(timeout=float('inf')),
                       dict(max_calls=1.5), dict(max_calls=True)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ChatPlanner('http://localhost/v1', 'vlm', **kwargs)


class RunnerTests(unittest.TestCase):
    def test_cli_stages_require_only_the_backend_under_test(self):
        base = ['--task-index', '9', '--out', 'new-run']
        self.assertEqual(parse_args(base+['--preflight', 'scene']).preflight, 'scene')
        self.assertEqual(parse_args(base+['--preflight', 'planner', '--planner-url',
                         'http://localhost/v1', '--planner-model', 'vlm']).preflight, 'planner')
        self.assertEqual(parse_args(base+['--preflight', 'policy', '--policy-checkpoint',
                         '/models/openvla']).preflight, 'policy')
        valid = base+['--policy-checkpoint', '/models/openvla', '--planner-url',
                     'http://localhost/v1', '--planner-model', 'vlm']
        self.assertIsNone(parse_args(valid).preflight)
        for extra in (['--planner-timeout', 'nan'], ['--planner-calls', '0'],
                      ['--egl-device', '-1'], ['--scorer-factory', 'bad'],
                      ['--mode', 'baseline', '--scorer-factory', 'a:b']):
            with self.subTest(extra=extra), patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                parse_args(valid+extra)
        with patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
            parse_args(base)

    def test_preflights_do_not_step_the_policy_action(self):
        for stage in ('scene', 'planner', 'policy'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                journal = Journal(Path(tmp)/'run')
                env, policy, planner = Environment([{IN: False}]), Policy(), FixturePlanner(task())
                try:
                    result = preflight(stage, env, 'place it', journal, planner=planner, policy=policy)
                finally:
                    journal.close()
                self.assertEqual(env.executed, [])
                self.assertEqual(result['executed_actions'], 0)
                self.assertEqual(len(policy.requests), int(stage == 'policy'))
                self.assertEqual(planner.calls, int(stage == 'planner'))
                if stage == 'planner':
                    self.assertIs(result['facts'][IN.key], False)

    def test_factory_rejects_unknown_and_incomplete_scorer_interfaces(self):
        self.assertEqual(load_scorers(None), {})
        for value in ([], {'old': object()}, {'recovery': object()}, {'handoff': None}):
            module = SimpleNamespace(make=lambda: value)
            with self.subTest(value=value), patch('src.experiment.run_task.importlib.import_module', return_value=module):
                with self.assertRaises((TypeError, ValueError)):
                    load_scorers('plugin:make')


class LiberoBridgeTests(unittest.TestCase):
    def test_scene_inventory_and_measurements_not_native_goal_config(self):
        from src.adapters.libero_environment import LiberoEnvironment
        class Image:
            def save(self, stream, format):
                stream.write(PNG)
        class Inner:
            objects_dict = {'item': SimpleNamespace(contact_geoms=('item_geom',))}
            fixtures_dict = {'box': SimpleNamespace(contact_geoms=('box_geom',))}
            object_states_dict = {'item': SimpleNamespace(object_state_type='object'),
                                  'box': SimpleNamespace(object_state_type='object'),
                                  'region': SimpleNamespace(object_state_type='site')}
            robots = [SimpleNamespace(gripper=SimpleNamespace(contact_geoms=('gripper_geom',)))]
            def _eval_predicate(self, clause):
                if clause == ['in', 'item', 'region']:
                    return True
                raise ValueError('unmeasurable binding')
            def _check_grasp(self, gripper, geoms):
                return False
            def check_contact(self, a, b):
                return False
        raw = dict(agentview_image='raw', item_pos=[0., 0., 0.], robot0_eef_pos=[0., 0., 0.])
        names = {0: 'item_geom', 1: 'box_geom'}
        env = SimpleNamespace(env=Inner(), check_success=lambda: False,
            sim=SimpleNamespace(model=SimpleNamespace(geom_id2name=names.get),
                data=SimpleNamespace(ncon=1, contact=[SimpleNamespace(geom1=0, geom2=1)])),
            step=lambda action: (raw, 0., False, {}))
        bridge = LiberoEnvironment(env, raw, image_transform=lambda raw: Image())
        before = bridge.observe()
        self.assertNotIn(IN.key, before.facts)
        measured = bridge.observe((IN, CLOSED, CONTACT))
        self.assertTrue(measured.facts[IN.key])
        self.assertIsNone(measured.facts[CLOSED.key])
        self.assertFalse(measured.facts[CONTACT.key])
        self.assertEqual(measured.contacts, (('item', 'box'),))
        self.assertEqual(measured.entities[2].kind, 'region')
        self.assertNotIn(IN.key, before.facts)  # Previous observations are not mutated.
        bridge.step((0.,)*7)
        self.assertEqual(bridge.observe((IN,)).index, 1)


if __name__ == '__main__':
    unittest.main()
