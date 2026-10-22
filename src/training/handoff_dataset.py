"""Observed-continuation labels with whole-initial-condition split isolation.

Future observations are used ONLY as targets. A sample contains current policy
RGB features, current measured state and the requested recovery contract.
No nominal-owner success labels, no counterfactual successful repair labels.
"""
import hashlib
import json
from pathlib import Path

from ..verification.fusion_mlp import STAGES

LABEL_SPEC = 'current_contract_and_preserved_goals_survive_12_nominal_actions_v1'


def contract(state, stage):
    if state.get('native_complete') is True:
        return True
    if stage == 'acquire':
        values = [state.get('held_clear'), state.get('in_goal')]
        return True if True in values else False if all(v is False for v in values) else None
    if stage == 'transfer':
        return state.get('in_goal')
    if stage == 'contain_release':
        values = [state.get('in_goal'), not state['grasp'] if state.get('grasp') is not None else None,
                  not state['mug_gripper'] if state.get('mug_gripper') is not None else None]
        return False if False in values else True if all(v is True for v in values) else None
    if stage == 'close':
        return state.get('native_complete')
    raise ValueError('Unknown recovery contract')


def outcome_label(state, stage, future, horizon=12):
    """future rows contain executed owner, post-action state, and terminal flag.

    A return to repair censors an otherwise safe window (no nominal rollout was
    observed), unless a measured failure already establishes a negative label.
    Episode truncation/missing evidence is censored, not silently negative.
    """
    current = contract(state, stage)
    if current is False:
        return 0, 'current_contract_false'
    if current is None:
        return None, 'current_contract_unknown'
    preserve = [key for key in ('in_goal', 'close_goal') if state.get(key) is True]
    for index, row in enumerate(future[:horizon]):
        if row['owner'] != 'nominal':
            return None, 'nominal_continuation_censored_by_repair'
        observed = row['state']
        check = contract(observed, stage)
        if check is False or any(observed.get(key) is False for key in preserve):
            return 0, 'post_handoff_contract_or_goal_loss'
        if check is None or any(observed.get(key) is None for key in preserve):
            return None, 'future_evidence_missing'
        if row.get('native_success') is True:
            return 1, 'native_completion_without_regression'
        if index+1 == horizon:
            return 1, 'observed_nominal_window_preserved'
    return None, 'right_censored_by_episode_end'


def root_split(root):
    # All development cohorts sharing a root stay together. Root0 is never fit
    # or used to choose a threshold; this is reused-development evaluation,
    # not a claim that root0 is a previously unseen research task.
    return 'train' if root in (1, 2, 5) else 'validation' if root in (3, 4) else 'evaluation'


def collect_samples(run_dirs, stride=8, horizon=12):
    samples, manifest, seen, censored = [], [], set(), {}
    for run in run_dirs:
        for path in sorted(Path(run).glob('*/status.json')):
            episode = path.parent
            status = json.loads(path.read_text())
            if status.get('status') != 'completed' or not (episode/'controller.jsonl').exists():
                continue
            root = status.get('cell', {}).get('root', status.get('args', {}).get('init_index'))
            if root not in range(6):
                raise ValueError('Only explicitly approved development roots 0..5 are permitted')
            # Initial-state content + complete executed actions identify repeated
            # development trajectories, regardless of cohort/run directory name.
            digest = hashlib.sha256((episode/'selected_initial_state.npy').read_bytes())
            with (episode/'steps.jsonl').open() as stream:
                for line in stream:
                    row = json.loads(line)
                    digest.update(json.dumps(row['action'], separators=(',', ':')).encode())
            identity = digest.hexdigest()
            if identity in seen:
                continue
            seen.add(identity)
            before, after = {}, {}
            with (episode/'controller.jsonl').open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if row['event'] == 'before':
                        d = row['decision']
                        before[row['step']] = dict(state=d['evidence']['state'], owner=d['owner'],
                            subgoal=d['subgoal'], repair_id=d.get('handoff_repair_id'))
                    elif row['event'] == 'after':
                        v = row['verification']
                        after[row['step']] = dict(state=v['evidence']['state'],
                            owner=row['decision']['owner'], native_success=v.get('native_complete') is True)
            repairs = {r['repair_id']: r for r in status.get('controller_summary', {}).get('repairs', [])}
            episode_samples = 0
            for step, row in before.items():
                actual_handoff = row['repair_id'] is not None
                if row['owner'] != 'nominal' or (step % stride and not actual_handoff):
                    continue
                image = episode/'model_inputs'/f'{step:06d}.png'
                if not image.exists():
                    raise ValueError(f'Missing causal policy image {image}')
                future = [after[j] for j in range(step, min(step+horizon, len(after))) if j in after]
                recovery = repairs.get(row['repair_id'], {}).get('recovery_goal')
                for stage in STAGES[:3]:
                    label, reason = outcome_label(row['state'], stage, future, horizon)
                    if label is None:
                        censored[reason] = censored.get(reason, 0)+1
                        continue
                    samples.append(dict(episode=str(episode), trajectory_sha256=identity, root=root,
                        split=root_split(root), step=step, stage=stage, state=row['state'],
                        label=label, reason=reason, image=str(image),
                        actual_handoff=actual_handoff and stage == recovery))
                    episode_samples += 1
            manifest.append(dict(episode=str(episode), trajectory_sha256=identity, root=root,
                split=root_split(root), samples=episode_samples,
                initial_sha256=hashlib.sha256((episode/'selected_initial_state.npy').read_bytes()).hexdigest(),
                status_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                controller_sha256=hashlib.sha256((episode/'controller.jsonl').read_bytes()).hexdigest()))
    initial_splits = {}
    for episode in manifest:
        previous = initial_splits.setdefault(episode['initial_sha256'], episode['split'])
        if previous != episode['split']:
            raise ValueError('Same initial state occurs across split boundaries')
    return samples, dict(episodes=manifest, label_spec=LABEL_SPEC, horizon=horizon,
        stride=stride, censored=censored, group='initial-state root shared across all cohorts',
        roots=dict(train=[1, 2, 5], validation=[3, 4], evaluation=[0]))
