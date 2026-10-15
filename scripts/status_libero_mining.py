"""Read-only live status: distinguish launched, changed actions, and verified benefit."""
import argparse
from collections import Counter
import json
from pathlib import Path


def status(run):
    launch_path=run/'launch.json'
    launch=json.loads(launch_path.read_text()) if launch_path.exists() else {}
    cases_path=run/'cases.json'
    cases=json.loads(cases_path.read_text())['cases'] if cases_path.exists() else []
    completed,verified,active=[],[],[]
    results={}
    for path in sorted(run.glob('*/*/manifest.json')):
        meta=json.loads(path.read_text());directory=path.parent
        result_path=directory/'result.json'
        if meta['status']=='complete' and result_path.exists():
            result=json.loads(result_path.read_text());completed.append(result)
            proof=directory/'verification.json'
            if proof.exists() and json.loads(proof.read_text())['status']=='verified':
                verified.append(result);results[result['case_id'],result['arm']]=result
        else:
            log=directory/'steps.jsonl';last=None;changed=0
            if log.exists():
                with log.open() as handle:
                    for line in handle:
                        try: row=json.loads(line)
                        except json.JSONDecodeError: break  # Writer may be midway through the last record.
                        last=row;changed+=int(row['correction'])
            active.append(dict(case=meta['case']['case_id'],arm=meta['arm'],status=meta['status'],
                last_step=None if last is None else last['step'],changed_actions_so_far=changed,
                last_reason=None if last is None else last['decision']['reason'],error=meta.get('error')))
    paired=[]
    for arm in ['phase_repair','verified_repair']:
        for control in ['vla','gate_hold','timed_repair']:
            counts=Counter()
            for c in cases:
                a=results.get((c['case_id'],arm));b=results.get((c['case_id'],control))
                if a is None or b is None: continue
                x,y=a['complete'],b['complete']
                counts['both_success' if x and y else 'rescue' if x else 'harm' if y else 'both_failure']+=1
            paired.append(dict(method=arm,control=control,verified_pairs=sum(counts.values()),outcomes=dict(counts)))
    by_arm={}
    for r in verified:
        counts=by_arm.setdefault(r['arm'],dict(verified=0,complete=0,changed_actions=0))
        counts['verified']+=1;counts['complete']+=int(r['complete']);counts['changed_actions']+=r['changed_actions']
    return dict(supervisor_status=launch.get('status','not_launched'),planned=launch.get('episodes_planned'),
        completed=len(completed),verified=len(verified),by_arm=by_arm,active_or_failed=active,
        partial_paired_results=paired,interpretation='Partial development discovery, not final efficacy',
        videos=[p.name for p in sorted((run/'review').glob('*.mp4'))])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    print(json.dumps(status(a.run),indent=2))
