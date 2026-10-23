"""Executed recovery labels with explicit censoring, provenance and group splits.

Inspired by CARE/FailSafe data design: a proposed action is not a demonstrated
recovery. Full continuation labels belong to the executed policy branch, not to
unexecuted candidates or every intermediate action as independent examples.
"""
from copy import deepcopy


SCHEMA = 'executed_repair_candidate_outcomes_v1'
CONTINUATION_WINDOW = 8  # inherited nominal cooldown; declared before outcomes


def record(status, steps, controls, source, replay_passed, audit_passed):
    selections = [r for r in steps if 'candidate_selection' in r]
    if len(selections) != 1:
        raise ValueError('Expected one first-repair candidate probe')
    chosen = selections[0]
    selection = chosen['candidate_selection']
    rid = chosen['decision']['repair_id']
    repair = next(r for r in status['controller_summary']['repairs'] if r['repair_id'] == rid)
    handoff = repair['handoff_step']
    valid = bool(status.get('valid_episode') and replay_passed and audit_passed)
    checks = {r['step']: r for r in controls if r['event'] == 'after'}
    # All endpoints and nominal ownership are required. Missing/truncated windows
    # are unknown, not successes. Preserve the criterion achieved at verified exit.
    stable = None
    if valid and repair['verified'] and handoff is not None:
        window = [checks.get(i) for i in range(handoff, handoff+CONTINUATION_WINDOW)]
        observed = [r for r in window if r is not None]
        bad = any(r['decision']['owner'] != 'nominal' for r in observed)
        loss = repair.get('post_handoff_goal_loss_step')
        bad |= loss is not None and handoff <= loss < handoff+CONTINUATION_WINDOW
        if bad:
            stable = False
        elif len(observed) == CONTINUATION_WINDOW:
            criterion = repair.get('verified_by')
            def retained(r):
                state = r.get('recovery_dataset_state', {})
                inside = state.get('in_goal')
                if criterion == 'held_clear':
                    held = state.get('held_clear')
                    goal = True if inside is True or held is True else (
                        False if inside is False and held is False else None)
                elif criterion == 'native_complete':
                    goal = state.get('native_complete')
                else:
                    goal = inside
                values = [goal] + [state.get(k) for k in repair['preserve']]
                return False if any(v is False for v in values) else (
                    True if all(v is True for v in values) else None)
            retained_values = [retained(r) for r in observed]
            stable = False if any(v is False for v in retained_values) else (
                True if all(v is True for v in retained_values) else None)
    labels = dict(complete_task_success=status.get('any_native_success') if valid else None,
        local_recovery=repair['local_recovery_step'] is not None if valid else None,
        verified_exit=repair['verified'] if valid else None,
        actual_handoff=handoff is not None if valid else None,
        stable_nominal_continuation=stable,
        completed_during_repair=repair.get('completed_during_repair') if valid else None,
        completed_after_handoff=repair.get('completed_after_handoff') if valid else None)
    return dict(schema=SCHEMA, source=source, eligible_for_training=valid,
        split_group=f"microwave_root{status['cell']['root']}", split=status['cell'].get('data_split', 'development_unassigned'),
        root=status['cell']['root'], step=chosen['step'], repair_id=rid,
        context=deepcopy(selection['context']), candidates=deepcopy(selection['candidates']),
        executed_candidate_id=selection['selected_id'], executed_action=chosen['action'],
        unexecuted_outcomes='unknown; never label unexecuted proposals from the selected outcome',
        target_scope='one_H1_candidate_then_recorded_controller_continuation',
        labels=labels, repair=deepcopy(repair),
        continuation_window=CONTINUATION_WINDOW,
        validation=dict(valid_episode=status.get('valid_episode'), replay_passed=replay_passed,
            joint_audit_passed=audit_passed),
        provenance=dict(plan_sha256=status['plan_sha256'], checkpoint=status['actor_build'],
            physical_fingerprint=chosen['before'], scene_source=source+'/initial_source.json',
            steps=source+'/steps.jsonl', controls=source+'/controller.jsonl'),
        costs=dict(actor_queries=status['actor_queries'], actions=status['steps_executed'],
            extra_candidate_queries=status.get('candidate_extra_queries', 0),
            episode_wall_s=status.get('episode_wall_s')))
