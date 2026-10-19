"""Frozen offline-only phase progress definitions; never a candidate selector."""
import math
import numpy as np

THRESHOLDS = dict(open=.01, grasp=None, extract=.005, place=.005, close=.01, withdraw=.005)


def measurements(state, extraction_gap):
    return dict(gaps=dict(open=max(0., 1.8-state['angle']),
        grasp=float(np.linalg.norm(np.asarray(state['eef'])-state['item'])),
        extract=max(0., extraction_gap),
        place=max(0., -state['release_footprint_margin_m'])+max(0., -state['release_seated_margin_m']),
        close=max(0., abs(state['angle'])-math.radians(5)),
        withdraw=max(0., .16-float(np.linalg.norm(np.asarray(state['eef'])-state['handle'])))),
        gap_units=dict(open='rad', grasp='m_approach_proxy_only', extract='m', place='m', close='rad', withdraw='m'))


def endpoint(before_state, before_stages, before_measure, states, stages, measures, contact_metrics, phase, horizon):
    """Candidate+nominal continuation measurements at preregistered horizons."""
    assert horizon in (1, 12, 32) and len(states) >= horizon
    s, task, metric = states[horizon-1], stages[horizon-1], measures[horizon-1]
    margin = before_measure['gaps'][phase]-metric['gaps'][phase]
    achieved = task['predicates'][phase]['status']=='satisfied'
    before_achieved = before_stages['predicates'][phase]['status']=='satisfied'
    losses = set()
    for current, current_task in zip(states[:horizon], stages[:horizon]):
        if before_state['grasp'] and not current['grasp'] and not current['local_release_support'] and not current['placed']:
            losses.add('previous_grasp_lost_without_measured_table_support')
        if (before_state['placed'] or before_stages['milestone_history']['place']) and not current['placed']:
            losses.add('previous_supported_placement_lost')
        if (before_stages['milestone_history']['open'] and phase in ('grasp', 'extract', 'place')
                and not current_task['milestone_history']['place'] and current['angle'] < 1.5):
            losses.add('previous_door_access_lost_before_placement')
        if before_stages['milestone_history']['close'] and phase=='withdraw' and current_task['predicates']['close']['status']!='satisfied':
            losses.add('previous_door_closure_lost_during_withdraw')
    max_item_pen = max(v['max_penetration'] for v in states[:horizon])
    new_penetration = max_item_pen > max(.001, before_state['max_penetration']+.001)
    directional = margin >= THRESHOLDS[phase] if THRESHOLDS[phase] is not None else False
    progress = directional or (achieved and not before_achieved)
    if phase=='grasp': progress = achieved  # approach motion alone is not a useful grasp
    if phase=='extract': progress = progress and s['grasp']
    if phase=='withdraw': progress = progress and not s['grasp'] and not s['hand_contact']
    useful = bool(progress and not losses and not new_penetration)
    predicate_losses = [p for p,v in before_stages['predicates'].items()
        if v['status']=='satisfied' and task['predicates'][p]['status']!='satisfied']
    return dict(horizon=horizon, phase=phase, goal_gap_before=before_measure['gaps'][phase],
        goal_gap_after=metric['gaps'][phase], directional_margin=margin,
        units=metric['gap_units'][phase], useful_direction_threshold=THRESHOLDS[phase],
        descriptive_directional_signal=bool(directional), descriptive_useful=useful,
        stage_satisfied=achieved, stage_consecutive_intervals=task['consecutive_intervals'][phase],
        stage_satisfied_entire_last20=(all(t['predicates'][phase]['status']=='satisfied' for t in stages[horizon-20:horizon])
            if horizon>=20 else None),
        stage_status=task['predicates'][phase]['status'], current_predicate_losses=predicate_losses,
        current_predicate_losses_include_expected_transitions=True,
        guarded_achievement_losses=sorted(losses), new_item_penetration_proxy=new_penetration,
        max_item_penetration_m=max_item_pen,
        max_recorded_contact_force_n=max(c['max_contact_force_n'] for c in contact_metrics[:horizon]),
        max_recorded_contact_penetration_m=max(c['max_contact_penetration_m'] for c in contact_metrics[:horizon]),
        actual_physical_damage='unknown', validated_contact_safety='unknown',
        historical_milestones=task['milestone_history'], strict_dwell=task['complete_dwell'],
        full_closed_start_completion=task['full_closed_start_completion'],
        interpretation='offline measured signal; not a validated repair, safety certificate or actual online handoff')
