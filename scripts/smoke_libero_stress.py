"""Replay a real clean prefix and validate external fault/sham commands.

Recorded actions are regression stimuli, NOT online OpenVLA recovery results.
"""
import argparse
import hashlib
import json
from pathlib import Path

from run_libero_drawer import reset_seeds
from adapters.libero_completion import ContactRecorder
from adapters.libero_drawer import DrawerMonitor, settled_drawer
from adapters.libero_preservation import fingerprint
from adapters.libero_stress import ReleaseDelay


def main(args):
    cases=json.loads(args.cases.read_text())
    assert cases['status']=='prepared_not_executed' and cases['development_only']
    source=Path(cases['source_run'])
    rows=[json.loads(x) for x in (source/'root_0/vla/steps.jsonl').read_text().splitlines()]
    reports=[]
    for case in cases['cases']:
        if case['root']!=0:
            continue
        assert case['eligible']
        reset_seeds()
        env,obs=settled_drawer(args.root,0)
        recorder=ContactRecorder(env.sim,keywords=('akita_black_bowl','white_cabinet'))
        monitor=DrawerMonitor(env)
        try:
            anchor=case['anchor_step']
            for row in rows[:anchor]:
                assert fingerprint(env,obs)==row['before']
                recorder.reset()
                obs,_,_,_=env.step(row['action'])
            before=fingerprint(env,obs)
            assert before==case['prefix_before_fingerprint']==rows[anchor]['before']
            fault=ReleaseDelay(anchor,case['duration_intervals'])
            trace=[]
            for step in range(anchor,anchor+8):
                selected=rows[step]['action']
                effective,log=fault.apply(selected,step)
                recorder.reset()
                obs,_,_,_=env.step(effective.tolist())
                measured=monitor.measure(recorder.samples)
                assert len(recorder.samples)==25
                after=fingerprint(env,obs)
                if case['duration_intervals']==0:
                    assert measured==rows[step]['measured']
                    assert after==rows[step+1]['before']
                log['observed_bowl_gripper']=measured['bowl_gripper']
                log['observed_retained']=measured['retained']
                log['after_fingerprint']=after
                trace.append(log)
            if case['duration_intervals']:
                assert after['integration']!=rows[anchor+8]['before']['integration']
            reports.append(dict(case_id=case['case_id'],prefix_exact=True,
                zero_fault_exact_reproduction=case['duration_intervals']==0,
                physical_state_changed_from_sham=case['duration_intervals']>0,trace=trace))
        finally:
            recorder.close()
            env.close()
    assert len(reports)==3
    report=dict(status='passed',fixture_only=True,online_actor_recovery_evaluated=False,
        cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path(__file__).resolve().parents[1]/'src/adapters/libero_stress.py']},
        branches=reports)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='passed',matched_prefix_branches=3,online_actor_recovery_evaluated=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--root',default='/home/ami/scratch/LIBERO')
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
