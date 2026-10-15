"""Physical pose-bias fixture: two axes, actual state change, no live efficacy."""
import argparse
import json
from pathlib import Path

from smoke_libero_mining import RecordedActor
from run_libero_mining import episode, source_hashes
from verify_libero_mining import verify, read_rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--root',default='/home/ami/scratch/LIBERO')
    args=p.parse_args();args.out.mkdir(exist_ok=False)
    grid=json.loads(args.cases.read_text())
    source=read_rows(Path(grid['source_run'])/'root_0/vla/steps.jsonl')
    outcomes=[]
    for label in ['root_0_pose_x_pos_20mm','root_0_pose_y_neg_20mm']:
        case=next(c for c in grid['cases'] if c['case_id']==label)
        path=episode(RecordedActor(source,case['anchor_step']),args,case,'vla',grid,fixture=True)
        verify(path,args.root)
        rows=read_rows(path/'steps.jsonl');anchor=case['anchor_step']
        assert rows[anchor]['before']==source[anchor]['before']
        assert rows[anchor+1]['before']['integration']!=source[anchor+1]['before']['integration']
        assert sum(r['fault']['changed'] for r in rows)==4
        assert all(r['selected'][-1]==r['action'][-1] for r in rows)
        assert sum(r['correction'] for r in rows)==0
        outcomes.append(dict(case=label,physical_state_changed=True,framework_changed_actions=0,
                             external_changed_actions=4,full_replay_verified=True))
    report=dict(status='passed',fixture_only=True,online_actor_efficacy=False,
                outcomes=outcomes,source_sha256=source_hashes())
    (args.out/'smoke.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
