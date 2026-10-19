"""Post-run evidence consistency/candidate-quality audit; never selects actions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

ARMS = ('nominal', 'simple_feedback', 'verification')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main(root):
    freeze = json.loads((root/'freeze.json').read_text())
    result = dict(scope='post_run_consistency_and_candidate_quality_not_efficacy', arms={}, videos={})
    for relative, expected in freeze['sources'].items():
        assert digest(root/'source'/relative) == expected, relative
    result['frozen_source_hashes_verified'] = len(freeze['sources'])
    result['freeze_sha256'] = digest(root/'freeze.json')
    initials = []
    for arm in ARMS:
        run = root/arm
        rows, queries = records(run/'steps.jsonl'), records(run/'queries.jsonl')
        status = json.loads((run/'status.json').read_text())
        assert status['status'] == 'completed'
        assert status['freeze_sha256'] == result['freeze_sha256']
        assert len(rows) == status['actions'] <= 600
        assert len(queries) == status['queries'] <= 600
        assert [q['query'] for q in queries] == list(range(len(queries)))
        initials.append(status['initial'])
        used_queries = []
        active, samples, sample_equal_nominal, sampled_selected = 0, 0, 0, 0
        rejection = Counter()
        unknowns = Counter()
        unique_counts = []
        exits = Counter()
        entry_steps, handoff_steps = [], []
        checks_passed_chosen = 0
        for i, row in enumerate(rows):
            assert row['step'] == i and row['actions_used'] == i+1
            cs = row['candidates']
            g = row['decision_trace']['gate']
            assert cs[0]['source'] == 'greedy'
            assert row['action'] == cs[g['chosen']]['action']
            assert row['decision_trace']['executed_action'] == row['action']
            assert row['decision_trace']['evidence'] == row['before_evidence']
            assert row['decision_trace']['evidence_through_step'] == i-1
            assert not row['simulator_done_internal']
            for j, c in enumerate(cs):
                assert c == queries[c['query']]
                assert c['step'] == i and c['candidate_index'] == j and c['seed'] == i*10+j
                used_queries.append(c['query'])
            assert row['queries_used'] == cs[-1]['query']+1
            if arm == 'nominal':
                assert len(cs) == 1 and row['action'] == cs[0]['action'] and row['owner'] == 'nominal'
            if row['decision_trace']['predicates']['active']:
                active += 1
                unique_counts.append(len({tuple(c['action']) for c in cs}))
                sampled_selected += cs[g['chosen']]['source'] == 'sample'
                for c in cs[1:]:
                    samples += 1
                    sample_equal_nominal += c['action'] == cs[0]['action']
                for c in g['candidates']:
                    for key in ('command_contract', 'command_continuity', 'grasp_release_consistency'):
                        rejection[key] += not c['checks'][key]
                    for key in ('future_collision', 'future_support', 'future_task_progress'):
                        assert c['checks'][key] == 'unknown'
                        unknowns[key] += 1
                chosen = g['candidates'][g['chosen']]['checks']
                checks_passed_chosen += all(chosen[k] for k in ('command_contract', 'command_continuity', 'grasp_release_consistency'))
            for event in row['events']:
                if event['type'] == 'local_entry': entry_steps.append(i)
                if event['type'] == 'local_exit': exits[event['reason']] += 1
                if event['type'] == 'handoff_executed':
                    assert event['actual_action'] == cs[0]['action'] == row['action']
                    assert event['step'] == i and row['owner'] == 'nominal'
                    assert any(e['type'] == 'local_exit' for e in row['events'])
                    handoff_steps.append(i)
        assert len(used_queries) == len(set(used_queries))
        assert len(used_queries) == len(queries), 'unexecuted queries require separate accounting'
        first_native = next((i for i,r in enumerate(rows) if r['after_evidence']['native_success']), None)
        replay = json.loads((root/f'replay_{arm}'/'verification.json').read_text())
        assert replay['status'] == 'exact_replay_passed'
        states, images = np.load(run/'states.npz')['states'], np.load(run/'images.npz')['agent']
        assert len(states) == len(images) == len(rows)+1
        result['arms'][arm] = dict(actions=len(rows), queries=len(queries), samples=samples,
            active_actions=active, samples_equal_same_observation_greedy=sample_equal_nominal,
            sample_greedy_duplicate_fraction=sample_equal_nominal/samples if samples else None,
            candidates_unique_per_active_step_histogram=dict(Counter(unique_counts)),
            sampled_candidate_selected=sampled_selected, active_selected_passed_all_checks=checks_passed_chosen,
            current_check_rejection_counts=dict(rejection), unknown_future_check_counts=dict(unknowns),
            local_entry_steps=entry_steps, actual_handoff_steps=handoff_steps, local_exit_reasons=dict(exits),
            first_grasp=next((i for i,r in enumerate(rows) if r['after_evidence']['grasped']), None),
            first_native=first_native, actions_after_first_native=len(rows)-first_native-1 if first_native is not None else None,
            max_dwell=status['max_dwell'], strict_complete=status['strict_complete'],
            first_strict_candidate=next((i for i,r in enumerate(rows) if r['after_evidence']['completion']['candidate']), None),
            termination=status['termination'], inference_s=sum(q['model_inference_s'] for q in queries),
            query_wall_s=sum(q['query_wall_s'] for q in queries),
            query_wall_median_s=float(np.median([q['query_wall_s'] for q in queries])),
            episode_wall_s=status['episode_wall_s'], exact_replay_steps=replay['steps'],
            active_transaction_censored_at_end=bool(rows[-1]['decision_trace']['predicates']['active']))
    assert all(i == initials[0] for i in initials)
    result['equal_initial_physics_controller_images'] = True
    for right in ('verification', 'simple_feedback'):
        path = root/f'nominal_left_{right}_right.mp4'
        import imageio.v2 as imageio
        with imageio.get_reader(path) as reader:
            meta = reader.get_meta_data()
            info = dict(width=meta['size'][0], height=meta['size'][1],
                avg_frame_rate=f"{int(meta['fps'])}/1", nb_frames=reader.count_frames(), duration=meta['duration'])
        manifest = json.loads((root/f'{right}_video_manifest.json').read_text())
        assert info['avg_frame_rate'] == '20/1' and int(info['nb_frames']) == manifest['frames']
        result['videos'][right] = dict(path=str(path), sha256=digest(path), **info)
    result['status'] = 'passed'
    result['audit_script_sha256'] = digest(Path(__file__))
    (root/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    (root/'postrun_audit_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    main(p.parse_args().root.resolve())
