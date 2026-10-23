"""Export causal per-action logs, repair transactions and paired divergences.

All indices are zero based. Before action t is observation t; after action t is
observation t+1. Baseline outcomes after a shared prefix are diagnostic evidence,
not counterfactual outcomes for later, already-diverged repair entries.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.train_joint_components import risk_label
from src.controller import NativeFramework


def load(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def delta(a, b):
    return [float(y-x) for x, y in zip(a, b)] if a is not None and b is not None else None


def norm(v):
    return math.sqrt(sum(x*x for x in v)) if v is not None else None


def episode(source, arm, root, out):
    status = load(source/'status.json')
    controls = list(rows(source/'controller.jsonl'))
    before = {r['step']: r for r in controls if r['event'] == 'before'}
    after = {r['step']: r for r in controls if r['event'] == 'after'}
    target = status['controller_summary']['config']['target']
    trace = []
    for raw in rows(source/'steps.jsonl'):
        t = raw['step']
        decision = before[t]['decision']
        evidence = decision['evidence']
        verification = after[t]['verification']
        checks = verification['verification']
        joint = evidence.get('joint', {})
        fusion = checks.get('fusion', {})
        pre, post = evidence['state'], verification['evidence']['state']
        diag_pre, diag_post = before[t]['diagnostics'], raw['diagnostics']
        position_delta = delta(pre['position'], post['position'])
        eef_delta = delta(diag_pre.get('eef_position'), diag_post.get('eef_position'))
        action = raw['action']
        record = dict(action_index=t, observation_before=t, observation_after=t+1,
            owner=decision['owner'], repair_id=decision['repair_id'],
            stage=decision['subgoal'], recovery_goal=decision['recovery_goal'],
            measured_stage_reference=NativeFramework._subgoal(pre),
            language=decision['language'], trigger=decision['trigger'],
            native_trigger=evidence.get('native_trigger', evidence.get('trigger')),
            flags=evidence['flags'], intervention_blocked=decision['intervention_blocked'],
            pre=pre, post=post, joint_risk=joint.get('risk'),
            joint_prediction=joint.get('prediction'), visual=joint.get('visual'),
            memory=joint.get('memory'), acceptance=checks.get('acceptance_checks'),
            acceptance_streaks=checks.get('acceptance_streaks'),
            local_condition_met=checks.get('local_condition_met', False),
            exit_condition_met=checks.get('exit_condition_met', False),
            preserved=checks.get('preserved'), fusion=fusion,
            events=verification['events'], budget=verification['budget'],
            proposal=raw['proposal'], native_proposal=raw['native_proposal'],
            executed_action=action, action_modified=raw['action_modified'],
            mapped_gripper_command=action[-1],
            gripper_command_semantics='close' if action[-1] > 0 else 'open' if action[-1] < 0 else 'hold',
            gripper_qpos_before=diag_pre.get('gripper_qpos'), gripper_qpos_after=diag_post.get('gripper_qpos'),
            object_delta_m=position_delta, eef_delta_m=eef_delta,
            target_contacts_after=[c for c in diag_post['contacts_endpoint'] if any(
                c[k].startswith(target+'_') for k in ('geom1', 'geom2'))],
            pre_fingerprint=raw['before'], post_fingerprint=raw['after'])
        trace.append(record)
    name = f'{arm}_root{root}'
    with (out/f'{name}.steps.jsonl').open('x') as stream:
        for row in trace:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    flat = []
    for row in trace:
        risk = row['joint_risk'] or {}
        pred = row['joint_prediction'] or {}
        fusion = row['fusion']
        f = {k: row[k] for k in ('action_index', 'observation_before', 'observation_after', 'owner',
            'repair_id', 'stage', 'recovery_goal', 'trigger', 'native_trigger', 'intervention_blocked',
            'mapped_gripper_command', 'gripper_command_semantics', 'local_condition_met', 'exit_condition_met')}
        f.update({f'post_{k}': row['post'].get(k) for k in ('grasp', 'held_clear', 'in_goal', 'close_goal', 'table', 'rise_m')})
        f.update(risk_score=risk.get('score'), risk_threshold=risk.get('threshold'),
            risk_streak=risk.get('consecutive'), no_lift=(pred.get('no_lift') or {}).get('no_lift_predicted'),
            fusion_probability=fusion.get('evidence', {}).get('probability'),
            fusion_blocks_exit=fusion.get('blocked_verified_exit'),
            object_dz_m=(row['object_delta_m'] or [None]*3)[2],
            eef_dz_m=(row['eef_delta_m'] or [None]*3)[2])
        flat.append(f)
    if flat:
        with (out/f'{name}.steps.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    transactions = []
    for repair in status['controller_summary']['repairs']:
        start, end = repair['entry_step'], repair['exit_step']
        selected = trace[start:end+1]
        first, last = selected[0], selected[-1]
        pre, post = first['pre'], last['post']
        # Reconstruct retry admission from exactly the controller's predicate /
        # three-endpoint / 20-mm logic, not a generic "fluctuation" explanation.
        previous = [p for p in status['controller_summary']['repairs']
            if p['trigger'] == repair['trigger'] and not p['verified'] and
            p['exit_step'] is not None and p['exit_step'] < start]
        retry = dict(previous_failed_same_trigger=None, changed_stable_predicates=[], displaced_20mm=False)
        if previous:
            old = previous[-1]
            retry['previous_failed_same_trigger'] = old['repair_id']
            old_state = trace[old['exit_step']]['post']
            recent = [r['post'] for r in trace[max(0, start-3):start]]
            if len(recent) == 3 and recent[0]['step'] > old['exit_step']:
                retry['changed_stable_predicates'] = [k for k in
                    ('grasp', 'held_clear', 'in_goal', 'close_goal', 'table', 'mug_microwave')
                    if recent[0][k] is not None and old_state[k] is not None and recent[0][k] != old_state[k]
                    and all(s[k] == recent[0][k] for s in recent)]
                retry['displaced_20mm'] = old_state['position'] is not None and all(
                    s['position'] is not None and math.dist(s['position'], old_state['position']) >= .02 for s in recent)
        transactions.append(dict(repair, entry_risk=first['joint_risk'], entry_prediction=first['joint_prediction'],
            entry_native_flags=first['flags'], entry_state=pre, exit_state=post,
            object_delta_m=delta(pre['position'], post['position']),
            eef_delta_m=[sum((r['eef_delta_m'] or [0, 0, 0])[i] for r in selected) for i in range(3)],
            stage_counts=dict(Counter(r['stage'] for r in selected)),
            stage_differs_from_measured_reference_steps=sum(
                r['stage'] != r['measured_stage_reference'] for r in selected),
            gripper_command_counts=dict(Counter(r['gripper_command_semantics'] for r in selected)),
            local_condition_steps=sum(r['local_condition_met'] for r in selected),
            fusion_blocked_eligible_steps=sum(bool(r['fusion'].get('blocked_verified_exit')) for r in selected),
            fusion_blocked_before_local_total_caps=sum(
                bool(r['fusion'].get('blocked_verified_exit')) and i+1 < 32 and
                r['budget']['repair_actions'] < 160 and not r['post']['native_complete']
                for i, r in enumerate(selected)),
            eligible_fusion_reasons=dict(Counter(reason for r in selected if r['exit_condition_met']
                for reason, value in (
                    ('score_below_threshold', r['fusion'].get('evidence', {}).get('probability', 1.) <
                        r['fusion'].get('evidence', {}).get('threshold', 0.)),
                    ('risk_veto', r['fusion'].get('evidence', {}).get('joint_risk_veto', False)),
                    ('prediction_veto', r['fusion'].get('evidence', {}).get('joint_prediction_veto', False))) if value)),
            fusion_probability_range=(
                [min(r['fusion']['evidence']['probability'] for r in selected),
                 max(r['fusion']['evidence']['probability'] for r in selected)]
                if all(r['fusion'].get('evidence') for r in selected) else None),
            any_acceptance_steps=sum(any((r['acceptance'] or {}).values()) for r in selected),
            retry_admission=retry))
    summary = dict(name=name, arm=arm, root=root, source=str(source),
        status=status['status'], valid_episode=status.get('valid_episode', False),
        success=status.get('any_native_success'), termination=status.get('termination'),
        actions=status['steps_executed'], queries=status['actor_queries'],
        action_modifications=sum(r['action_modified'] for r in trace),
        repair_actions=status['controller_summary']['repair_actions'],
        milestones=status['controller_summary']['milestones'], repairs=transactions,
        blocked_entries=dict(Counter(r['intervention_blocked'] for r in trace if r['intervention_blocked'])),
        source_hashes={f: digest(source/f) for f in ('status.json', 'controller.jsonl', 'steps.jsonl',
            'frames.jsonl', 'actions.npz', 'selected_initial_state.npy', 'model.xml')})
    save(out/f'{name}.summary.json', summary)
    return summary, trace


def paired(base, other):
    bs, bt = base
    os, ot = other
    shared = 0
    for b, o in zip(bt, ot):
        if b['pre_fingerprint'] != o['pre_fingerprint']:
            break
        shared += 1
    divergence = next((i for i, (b, o) in enumerate(zip(bt, ot))
        if b['executed_action'] != o['executed_action']), None)
    first_entry = next((r['action_index'] for r in ot if r['owner'] == 'repair'), None)
    comparison = dict(root=bs['root'], treatment=os['arm'], baseline_success=bs['success'],
        treatment_success=os['success'], valid_pair=bs['valid_episode'] and os['valid_episode'],
        same_initial_state=bs['source_hashes']['selected_initial_state.npy'] == os['source_hashes']['selected_initial_state.npy'],
        same_scene=bs['source_hashes']['model.xml'] == os['source_hashes']['model.xml'],
        shared_before_observation_prefix=shared, first_action_divergence=divergence,
        action_tapes_identical=len(bt) == len(ot) and divergence is None,
        first_repair_entry=first_entry,
        rescue=not bs['success'] and os['success'], harm=bs['success'] and not os['success'])
    if first_entry is not None:
        t = first_entry
        matched = t < shared
        comparison['first_entry_shared_state'] = matched
        if matched:
            current = bt[t]['pre']
            history = [r['pre'] for r in bt[:t+1]]
            future = [dict(owner=r['owner'], state=r['post']) for r in bt[t:t+24]]
            label, reason = risk_label(current, history, future)
            comparison['baseline_nominal24_risk_label_at_entry'] = dict(label=label, reason=reason,
                semantics='Training-defined short-horizon label, NOT complete-task failure or intervention benefit')
            comparison['baseline_first_future_attainment'] = {
                key: next((r['action_index'] for r in bt[t:] if r['post'][key] is True), None)
                for key in ('held_clear', 'in_goal', 'native_complete')}
        # Later comparisons are wall/action-index reference windows only.
        comparison['first_entry_window'] = {
            label: [r for r in tr[max(0, t-12):min(len(tr), t+45)]]
            for label, tr in (('baseline', bt), (os['arm'], ot))}
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--existing', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sources = {}
    for root in range(6):
        for arm, prefix in (('baseline', 'baseline'), ('joint', 'v2')):
            source = args.existing/f'{prefix}_root{root}'
            if (source/'status.json').exists():
                sources[(root, arm)] = source
    for phase in ('controls', 'measured'):
        path = args.study/phase/'results.json'
        if path.exists():
            for job in load(path):
                sources[(job['root'], job['arm'])] = Path(job['source'])
    analyzed, skipped = {}, []
    for key, source in sorted(sources.items()):
        status = load(source/'status.json')
        if status.get('status') != 'completed' or not status.get('valid_episode'):
            skipped.append(dict(root=key[0], arm=key[1], source=str(source), status=status))
            continue
        analyzed[key] = episode(source, key[1], key[0], args.out)
    pairs = []
    for (root, arm), value in analyzed.items():
        if arm != 'baseline' and (root, 'baseline') in analyzed:
            item = paired(analyzed[(root, 'baseline')], value)
            save(args.out/f'{arm}_root{root}.paired.json', item)
            pairs.append({k: v for k, v in item.items() if k != 'first_entry_window'})
    summaries = [s for s, _ in analyzed.values()]
    report = dict(indexing='zero-based action index; after action t is observation t+1',
        development_only=True, invalid_episodes=skipped, episodes=summaries, pairs=pairs,
        analysis_source_sha256=digest(Path(__file__)),
        risk_label_source_sha256=digest(Path(__file__).resolve().parents[1]/'src/training/train_joint_components.py'),
        gripper_semantics='Mapped robosuite PandaGripper command: +1 closes, -1 opens. Native proposal is a different representation.',
        arms={arm: dict(successes=sum(s['success'] is True for s in summaries if s['arm'] == arm),
            valid_episodes=sum(s['arm'] == arm for s in summaries),
            repair_actions=sum(s['repair_actions'] for s in summaries if s['arm'] == arm),
            verified_repairs=sum(r['verified'] for s in summaries if s['arm'] == arm for r in s['repairs']))
            for arm in sorted({s['arm'] for s in summaries})})
    save(args.out/'report.json', report)
    lines = ['# Framework diagnostic study', '',
        'Development evidence; same microwave task, reused initial states. All indices are zero based.', '',
        '| Root | Arm | Native success | Actions | Repair actions | Verified repairs |',
        '|---|---|---|---|---|---|']
    for s in summaries:
        lines.append(f"| {s['root']} | {s['arm']} | {s['success']} | {s['actions']} | {s['repair_actions']} | {sum(r['verified'] for r in s['repairs'])} |")
    lines += ['', '## First-intervention evidence', '']
    for p in pairs:
        lines.append(f"- Root {p['root']} {p['treatment']}: first entry {p['first_repair_entry']}; "
            f"first action divergence {p['first_action_divergence']}; shared before-observation prefix "
            f"{p['shared_before_observation_prefix']}; nominal24 label "
            f"{p.get('baseline_nominal24_risk_label_at_entry')}; rescue={p['rescue']}, harm={p['harm']}.")
    lines += ['', '## Interpretation boundaries', '',
        '- Full joint versus measured_fusion is a bundled ablation of joint stage/memory/prediction/risk, not risk-only.',
        '- Risk labels describe 24-action measured events, not eventual task failure or the benefit of intervention.',
        '- Later repairs occur after trajectory divergence; same-index baseline windows are not same-state counterfactuals.',
        '- A budget exit and actual nominal handoff do not imply verified recovery.',
        '- Replay/model recomputation reports must be consulted independently of actor completion.',
        '- Gripper signs are mapped Panda commands; action coordinates are not measured Cartesian displacement.']
    (args.out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report['arms'], indent=2))


if __name__ == '__main__':
    main()
