"""Frozen three-arm full-task development study, one actor at a time on GPUs2/3.

Runs root0 first then the other existing development roots without changing the
method. Does not open new/reserved initial conditions. Each arm uses the same
520 action/query caps; skill steps still query the VLA and retain unused output.
"""
import argparse
import math
import os
from pathlib import Path
import subprocess
import time
import traceback

import run_framework_comparison as c

ARMS = ('baseline', 'current', 'options')


def summarize(out, roots):
    results = []
    for root in roots:
        for arm in ARMS:
            name=f'{arm}_root{root}';p=out/name
            status=c.load(p/'status.json') if (p/'status.json').exists() else {}
            replay=c.load(out/f'replay_{name}'/'verification.json') if (out/f'replay_{name}'/'verification.json').exists() else {}
            audit=c.load(p/'joint_recomputation.json') if (p/'joint_recomputation.json').exists() else {}
            repairs=status.get('controller_summary',{}).get('repairs',[])
            valid=bool(status.get('valid_episode') and replay.get('status')=='exact_replay_passed' and
                (arm=='baseline' or audit.get('status')=='joint_recomputation_passed'))
            initial=p/'selected_initial_state.npy'
            results.append(dict(root=root,arm=arm,valid=valid,status=status.get('status','not_run'),
                native_success=status.get('any_native_success'),termination=status.get('termination'),
                actions=status.get('steps_executed'),queries=status.get('actor_queries'),
                repair_actions=status.get('controller_summary',{}).get('repair_actions'),
                repairs=len(repairs),verified=sum(r['verified'] for r in repairs),
                unverified_exits=sum(not r['verified'] for r in repairs),
                post_handoff_regressions=sum(r['post_handoff_goal_loss_step'] is not None for r in repairs),
                skill_actions=status.get('oracle_skill_actions',0),
                initial_state_sha256=c.sha(initial) if initial.exists() else None,
                joint_counters=status.get('joint_counters'),
                replay=replay.get('status'),audit=audit.get('status')))
    aggregates={}
    for arm in ARMS:
        samples=[r for r in results if r['arm']==arm and r['valid']]
        n=len(samples);success=sum(r['native_success'] is True for r in samples)
        if n:
            p=success/n;z=1.959963984540054;den=1+z*z/n
            mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
            interval=[mid-half,mid+half]
        else:interval=None
        aggregates[arm]=dict(valid=n,planned=len(roots),successes=success,
            success_rate=success/n if n else None,wilson95=interval,
            interval_scope='descriptive_development_roots_not_unseen_task_inference')
    pairs=[]
    for root in roots:
        group={r['arm']:r for r in results if r['root']==root}
        hashes={r['initial_state_sha256'] for r in group.values() if r['valid']}
        for arm in ARMS[1:]:
            b,r=group['baseline'],group[arm];valid=b['valid'] and r['valid'] and len(hashes)==1
            pairs.append(dict(root=root,arm=arm,valid=valid,
                rescue=bool(valid and not b['native_success'] and r['native_success']),
                harm=bool(valid and b['native_success'] and not r['native_success'])))
    report=dict(results=results,aggregates=aggregates,baseline_pairs=pairs,
        scope='Reused development roots; root0 debugging condition, not untouched evaluation. No cross-task claim.',
        primary_endpoint='original native In AND Close',invalid_policy='retain infrastructure failures separately')
    c.save(out/'comparison.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--roots',type=int,nargs='+',default=list(range(6)))
    args=p.parse_args()
    c.require(len(set(args.roots))==len(args.roots) and all(r in range(6) for r in args.roots),'Development roots 0..5 only')
    plan=c.load(args.plan);c.check_plan(plan)
    c.require(plan.get('joint') and plan.get('fusion'),'Pinned joint models required')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    c.save(out/'plan.json',plan)
    status=dict(status='running',started=time.time(),pid=os.getpid(),actors=1,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.REPO,text=True).strip(),
        plan_sha256=c.sha(args.plan),roots=args.roots,arms=ARMS,commands=[])
    c.save(out/'design.json',dict(roots=args.roots,arms=ARMS,actions=520,queries=520,repair_cap=160,local_cap=32,
        baseline='frozen OpenVLA unchanged',current='joint v6 phase-prompt repair',
        options='same joint gates plus oracle acquisition skills, progress stop and cross-trigger failure memory',
        query_matching='one VLA call per executed step even when skill replaces proposal',
        thresholds='unchanged frozen risk/fusion artifacts',reserved_conditions='not used',
        scope='development comparison; not held-out generalization',
        root0_policy='implementation check only; method remains frozen for subsequent roots; retain all outcomes'))

    def save():c.save(out/'supervisor_status.json',status)

    def run(command,name,lane,timeout):
        with (out/f'{name}.log').open('xb') as log:
            process=subprocess.Popen(command,cwd=c.REPO,env=c.environment(lane),stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status.update(child_pid=process.pid,phase=name)
            status['commands'].append(dict(command=command,pid=process.pid,log=f'{name}.log'))
            save()
            try:return process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                c.terminate_group(process);return -15

    save()
    try:
        for root in args.roots:
            # Run the new execution path first on the already reused debug
            # condition, then matched controls. No result-dependent tuning.
            for arm in (('options', 'baseline', 'current') if root==0 else ARMS):
                c.check_plan(plan)
                cell=next(x for x in plan['cells'] if x['name']==f'{"baseline" if arm=="baseline" else "v2"}_root{root}')
                name=f'{arm}_root{root}';status.update(root=root,arm=arm)
                command=[c.PYTHON,'-B',str(c.SCRIPT),'--worker','--plan',str(args.plan.resolve()),
                    '--cell',cell['name'],'--out',str(out/name)]
                if arm!='baseline':command+=['--fusion-mode','gate','--mlp-path',plan['fusion']['model'],
                    '--joint-components',plan['joint']['artifact']]
                if arm=='options':command+=['--recovery-options']
                code=run(command,name,cell['lane'],c.LIMITS['outer_seconds'])
                status[name+'_returncode']=code
                if code==0:
                    status['replay_'+name+'_returncode']=run([c.PYTHON,'-B',str(c.SCRIPT),'--replay',str(out/name),
                        '--out',str(out/f'replay_{name}')],f'replay_{name}',cell['lane'],360)
                    if arm!='baseline':
                        status['audit_'+name+'_returncode']=run([c.PYTHON,'-B',str(c.REPO/'scripts/run_joint_framework.py'),
                            '--audit',str(out/name)],f'audit_{name}',cell['lane'],600)
                summarize(out,args.roots);save()
        status['status']='completed' if all(v==0 for k,v in status.items() if k.endswith('_returncode')) else 'completed_with_failures'
    except BaseException:
        status.update(status='supervisor_failed',error=traceback.format_exc())
    finally:
        status.update(ended=time.time());summarize(out,args.roots);save()
    return 0 if status['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
