"""Branch from identical R2 pre-trigger history; no VLA/task-efficacy claim."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from run_fridge_framework import create,sense,measure,fingerprint,save,REPO
from adapters.fridge_door_repair import DoorContactRepair,door_feedback


def main(run,out,budget):
    out.mkdir(parents=True,exist_ok=False)
    args=argparse.Namespace(**json.loads((run/'status.json').read_text())['args'])
    args.gpu=2
    args.steps=max(args.steps,budget+40)
    rows=[json.loads(l) for l in (run/'steps.jsonl').read_text().splitlines()]
    trigger=next(r['step'] for r in rows if r['event']=='stall_trigger')
    env,obs=create(args,out)
    report=dict(scope='same_trigger_door_repair_diagnostic_not_full_framework',source=str(run),
        prefix_actions=trigger,budget=budget,status='started',any_bilateral=False,max_angle=0.,
        sources={p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in
            ['scripts/probe_fridge_door_repair.py','src/adapters/fridge_door_repair.py']})
    for p in report['sources']:
        dest=out/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((REPO/p).read_bytes())
    states=[]
    try:
        for _ in range(20):obs,*_=env.step([0,0,0,0,0,0,-1])
        for r in rows[:trigger]:obs,*_=env.step(r['action'])
        report['branch_fingerprint']=fingerprint(env,obs)
        assert report['branch_fingerprint']==rows[trigger]['before']
        ctrl=DoorContactRepair(budget)
        with imageio.get_writer(out/'raw.mp4',fps=20) as video,(out/'steps.jsonl').open('w') as log:
            for i in range(budget):
                video.append_data(obs['agentview_image'][::-1,::-1]);states.append(env.get_sim_state().copy())
                s=door_feedback(env,sense(env,obs))
                action,event=ctrl.act(s)
                before=fingerprint(env,obs)
                obs,*_=env.step(action.tolist())
                post=door_feedback(env,sense(env,obs))
                r=dict(step=i,action=action.tolist(),before=before,after=fingerprint(env,obs),
                    **event,**measure(env,obs),pad_contacts=post['pad_contacts'],pad_center=post['pad_center'].tolist())
                log.write(json.dumps(r)+'\n');log.flush()
                report['any_bilateral'] |= all(post['pad_contacts'])
                report['max_angle']=max(report['max_angle'],post['angle'])
                if i%25==0:print(json.dumps(r),flush=True)
                if event['done']:break
            video.append_data(obs['agentview_image'][::-1,::-1]);states.append(env.get_sim_state().copy())
        report.update(status='completed',steps=i+1,final_angle=post['angle'],final_stage=event['stage'],
            verified_open=event['event']=='verified_open')
        imageio.imwrite(out/'final.png',obs['agentview_image'][::-1,::-1])
    finally:
        np.savez_compressed(out/'states.npz',states=np.asarray(states));save(out/'report.json',report);env.close()
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--budget',type=int,default=200);a=p.parse_args();main(a.run.resolve(),a.out.resolve(),a.budget)
