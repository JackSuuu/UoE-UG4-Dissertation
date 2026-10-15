"""Recorded-command pipeline fixture; not online VLA efficacy evidence."""
import argparse
import json
from pathlib import Path
from PIL import Image

from run_libero_mining import episode, source_hashes, np, ARMS
from verify_libero_mining import verify, render_case


class RecordedActor:
    build={'fixture':'recorded nominal commands, not live actor'}
    def __init__(self,rows,anchor): self.rows,self.step=rows,anchor
    def act(self,image,language):
        row=self.rows[self.step];self.step+=1
        return np.asarray(row['proposal']),{k:row[k] for k in ['tokens','native_action']},Image.fromarray(image)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--root',default='/home/ami/scratch/LIBERO')
    a=p.parse_args();a.out.mkdir(exist_ok=False)
    grid=json.loads(a.cases.read_text());case=next(c for c in grid['cases'] if c['case_id']=='root_0_delay_8')
    rows=[json.loads(s) for s in (Path(grid['source_run'])/'root_0/vla/steps.jsonl').read_text().splitlines()]
    for arm in ARMS:
        actor=RecordedActor(rows,case['anchor_step'])
        directory=episode(actor,a,case,arm,grid,fixture=True)
        verify(directory,a.root)
    render_case(a.out,case)
    sham=next(c for c in grid['cases'] if c['case_id']=='root_0_delay_0')
    directory=episode(RecordedActor(rows,sham['anchor_step']),a,sham,'vla',grid,fixture=True)
    verify(directory,a.root)
    actual=[json.loads(s) for s in (directory/'steps.jsonl').read_text().splitlines()]
    assert all(r['measured']==rows[i]['measured'] and r['before']==rows[i]['before'] for i,r in enumerate(actual))
    report=dict(status='passed',fixture_only=True,online_actor_efficacy=False,arms=list(ARMS),
        exact_sham=True,full_action_and_forecast_replay=True,source_sha256=source_hashes())
    (a.out/'smoke.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
