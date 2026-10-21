"""Freeze Q1 choices before branch outcomes, then launch bounded GPU2-only oracle."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
if (ROOT/'base_source/scripts').exists(): sys.path.insert(0,str(ROOT/'base_source/scripts'))
from experiment3_candidate_launch import bounded,sha,save
REPO=Path(__file__).resolve().parents[1]
SOURCE=Path('/home/ami/scratch/experiment3_candidates_fridge_20261019_a')


def supervise(out):
    plan=json.loads((out/'plan.json').read_text())
    freeze=json.loads((out/'freeze.json').read_text())
    state=dict(status='running',pid=os.getpid(),started=time.time(),jobs=[],
        resources=dict(gpu=2,max_new_actions=288,max_new_queries=279,cohort_wall_s=3600),
        offline_oracle_never_used_by_selector=True)
    start=time.monotonic()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',MUJOCO_EGL_DEVICE_ID='2',MUJOCO_GL='egl',
        PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
        TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='2',TOKENIZERS_PARALLELISM='false')
    try:
        for rel,h in freeze['sources'].items(): assert sha(out/'source'/rel)==h
        for rel,h in freeze['base_sources'].items(): assert sha(out/'base_source'/rel)==h
        for path,h in freeze['source_tapes'].items(): assert sha(Path(path))==h
        assert sha(out/'plan.json')==freeze['plan_sha256']
        for b in plan['branches']:
            remaining=3600-(time.monotonic()-start)
            if remaining<=0:
                state['status']='cohort_wall_cap'
                break
            branch_start=time.monotonic()
            job=dict(id=b['id'],started=time.time(),source_step=b['opportunity']['step'],candidate_index=b['candidate_index'])
            state['jobs'].append(job)
            save(out/'launch.json',state)
            runner=out/'source/scripts/experiment3_q1_run.py'
            job['returncode']=bounded([sys.executable,'-u',str(runner),'--branch',b['id'],'--out',str(out/b['id'])],
                out/f"{b['id']}.log",min(1200,remaining),env)
            if job['returncode']!=0:
                state['status']='stopped_on_branch_error_no_tolerance_change'
                break
            replay_budget=min(1200-(time.monotonic()-branch_start),3600-(time.monotonic()-start))
            if replay_budget<=0:
                state['status']='branch_or_cohort_wall_cap_before_replay'
                break
            job['replay_returncode']=bounded([sys.executable,'-u',str(runner),'--branch',b['id'],
                '--out',str(out/f"replay_{b['id']}"),'--replay',str(out/b['id'])],
                out/f"replay_{b['id']}.log",replay_budget,env)
            job['ended']=time.time()
            save(out/'launch.json',state)
            if job['replay_returncode']!=0:
                state['status']='stopped_on_replay_error_no_tolerance_change'
                break
        if len(state['jobs'])==len(plan['branches']) and all(j.get('returncode')==j.get('replay_returncode')==0 for j in state['jobs']):
            remaining=3600-(time.monotonic()-start)
            if remaining>0:
                state['review_returncode']=bounded([sys.executable,str(out/'source/scripts/experiment3_q1_review.py'),str(out)],
                    out/'review.log',min(300,remaining),env)
                state['status']='completed' if state['review_returncode']==0 else 'review_failed'
        if state['status']=='running': state['status']='completed_with_issues'
    except BaseException:
        state.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        state.update(ended=time.time(),cohort_wall_s=time.monotonic()-start,gpu2_released=True)
        save(out/'launch.json',state)


def launch(out):
    out.mkdir(parents=True,exist_ok=False)
    source_freeze=json.loads((SOURCE/'freeze.json').read_text())
    source_status=json.loads((SOURCE/'verification/status.json').read_text())
    replay=json.loads((SOURCE/'replay_verification/verification.json').read_text())
    assert source_status['status']=='completed' and replay['status']=='exact_replay_passed'
    for rel,h in source_freeze['sources'].items(): assert sha(SOURCE/'source'/rel)==h
    shutil.copytree(SOURCE/'source',out/'base_source')
    (out/'camera_config.json').write_bytes((SOURCE/'camera_config.json').read_bytes())
    sys.path.insert(0,str(out/'base_source/src'))
    sys.path.insert(0,str(out/'base_source/scripts'))
    from experiment3_fridge_gate import FridgeCandidateGate
    rows=[json.loads(l) for l in (SOURCE/'verification/steps.jsonl').read_text().splitlines()]
    eligible=[r for r in rows if r['step']<=550 and len(r['candidates'])>=2 and any(e['type']=='local_entry' for e in r['events'])][:3]
    assert eligible,'No eligible source entries; do not silently choose another distribution'
    opportunities=[]
    branches=[]
    for row in eligible:
        step=row['step']
        previous=[r['action'] for r in rows[max(0,step-5):step]]
        selectors={}
        for arm in ('simple_feedback','verification'):
            g=FridgeCandidateGate(arm)
            g.active_phase=row['phase']
            g.actions.extend(previous)
            action,d=g.choose(row['candidates'],row['before_state'],dict(events=[]))
            selectors[arm]=dict(chosen=d['chosen'],action=action,reason=d['reason'],candidates=d['candidates'],
                computed_before_branch_outcomes=True)
        assert selectors['verification']['chosen']==row['decision_trace']['gate']['chosen']
        assert selectors['verification']['action']==row['action']
        groups={}
        for i,c in enumerate(row['candidates']): groups.setdefault(tuple(c['action']),[]).append(i)
        o=dict(id=f'step{step}',step=step,source_run=str(SOURCE/'verification'),phase=row['phase'],
            before=row['before'],before_state=row['before_state'],before_stages=row['before_stages'],
            candidates=row['candidates'],common_previous_five_actions=previous,selectors=selectors,
            duplicate_action_groups=list(groups.values()),selection_rule='first_three_eligible_entries_not_outcomes')
        opportunities.append(o)
        for i in range(len(row['candidates'])):
            branches.append(dict(id=f'step{step}_candidate{i}',candidate_index=i,opportunity=o))
    assert len(branches)<=9
    plan=dict(created_before_branch_outcomes=time.time(),source=str(SOURCE),source_arm='verification',
        source_distribution_bias='states visited by v1 verification policy, not random or baseline states',
        original_source_arm_queries=source_status['queries'],original_selected_candidate_generation_calls=len(branches),
        original_candidate_generation_seconds=sum(c['model_inference_s'] for o in opportunities for c in o['candidates']),
        max_new_branch_actions=len(branches)*32,max_fresh_queries=len(branches)*31,
        planned_prefix_replay_actions=sum(b['opportunity']['step'] for b in branches),
        planned_independent_replay_prefix_actions=sum(b['opportunity']['step'] for b in branches),
        planned_independent_replay_branch_actions=len(branches)*32,
        planned_setup_actions=2*len(branches)*20,opportunities=opportunities,branches=branches)
    save(out/'plan.json',plan)
    files=[p.relative_to(REPO) for p in (REPO/'scripts').glob('experiment3_q1_*.py')]
    files += [Path('scripts/test_experiment3_q1.py'),Path('exp_record/agent_notes/experiment3_fridge_q1_20261019/protocol.md')]
    sources={}
    for rel in files:
        dest=out/'source'/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((REPO/rel).read_bytes())
        sources[str(rel)]=sha(dest)
    tapes=[SOURCE/'verification'/name for name in ('status.json','steps.jsonl','queries.jsonl','states.npz','images.npz')]
    tapes += [SOURCE/'freeze.json',SOURCE/'replay_verification/verification.json']
    for path,h in source_freeze['model_files'].items(): assert sha(Path(path))==h
    for path,h in source_freeze['model_code'].items(): assert sha(Path(path))==h
    for path,h in source_freeze['external_sources'].items(): assert sha(Path(path))==h
    save(out/'freeze.json',dict(frozen_at=time.time(),sources=sources,base_sources=source_freeze['sources'],
        source_tapes={str(p):sha(p) for p in tapes},plan_sha256=sha(out/'plan.json'),
        camera_sha256=sha(out/'camera_config.json'),source_freeze_sha256=sha(SOURCE/'freeze.json'),
        model_files_verified=source_freeze['model_files'],model_code_verified=source_freeze['model_code'],
        external_sources_verified=source_freeze['external_sources'],gpu=2,batch=1,
        offline_oracle_only=True,never_used_for_live_selection_or_training=True))
    with (out/'supervisor.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,'-u',str(out/'source/scripts/experiment3_q1_launch.py'),
            str(out),'--supervise'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(out/'detached.json',dict(pid=proc.pid,start_new_session=True,gpu=2,freeze_sha256=sha(out/'freeze.json')))
        print(json.dumps(dict(detached_pid=proc.pid,out=str(out),branches=len(branches),steps=[o['step'] for o in opportunities])),flush=True)
        return proc.wait()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('out',type=Path)
    p.add_argument('--supervise',action='store_true')
    a=p.parse_args()
    if a.supervise: supervise(a.out.resolve())
    else: raise SystemExit(launch(a.out.resolve()))
