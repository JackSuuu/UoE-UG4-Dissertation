"""Freeze D2 development case identities from verified real-actor clean prefixes.

No perturbed rollouts are run here. No success/failure outcome enters anchor choice.
All roots are retained, including unavailable-event cases. Final roots are untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from adapters.libero_stress import release_opportunity


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(run, output):
    report = json.loads((run/'verified_summary.json').read_text())
    assert report['status'] == 'verified' and report['full_action_replay']
    assert report['causal_decision_trace_verified']
    cases = []
    for index in range(3):
        directory = run/f'root_{index}'/'vla'
        path = directory/'steps.jsonl'
        assert report['artifact_sha256'][str(path.relative_to(run))] == digest(path)
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        anchor = next((t for t in range(1,len(rows)) if release_opportunity(
            rows[t-1]['measured'],rows[t]['proposal'])), None)
        for duration in [0,4,8]:
            cases.append(dict(case_id=f'drawer_root{index}_release_delay{duration}',
                root=index, split='development', fault='release_delay', duration_intervals=duration,
                duration_sim_seconds=duration/20, anchor_step=anchor,
                eligible=anchor is not None,
                prefix_before_fingerprint=rows[anchor]['before'] if anchor is not None else None,
                prefix_actions_sha256=hashlib.sha256(json.dumps(
                    [r['action'] for r in rows[:anchor]],separators=(',',':')).encode()).hexdigest()
                    if anchor is not None else None,
                source_steps_sha256=digest(path)))
    result = dict(schema=1, status='prepared_not_executed', development_only=True,
        source_run=str(run.resolve()), source_verified_summary_sha256=digest(run/'verified_summary.json'),
        task=json.loads((run/'root_0/vla/manifest.json').read_text())['task'],
        methods_receive_fault_metadata=False, external_fault_after_repair=True,
        no_outcome_based_case_filter=True, replay_prefix_required=True,
        total_episode_cap_including_prefix=600, actual_final_test_roots_used=[], cases=cases)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:
        f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(cases=len(cases),anchors={i:cases[3*i]['anchor_step'] for i in range(3)},
                         status=result['status'])))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    prepare(args.run,args.output)
