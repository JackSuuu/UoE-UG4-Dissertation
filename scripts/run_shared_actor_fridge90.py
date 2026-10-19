"""Versioned LIBERO90-to-fridge actor-only transfer test; no repair controller."""
import argparse
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
from run_shared_actor_qualification import (SOURCES as QUAL_SOURCES, PINNED, REPO,
    gpu_check, clock, transition, digest, save)
from diagnose_vla_failure import PinnedActor, make_scene, fingerprint, ContactRecorder
from adapters.fridge_runtime import sense, TaskMonitor

SELF='scripts/run_shared_actor_fridge90.py'
PROTOCOL='exp_record/agent_notes/shared_actor_fridge90_protocol_20261019.md'
WORKER='scripts/openvla_tf440_named_stats_worker.py'
SNAPSHOT='/home/ami/scratch/shared_actor_fridge90_checkpoint_20261019'
KEY='libero_90_no_noops'
SOURCES=list(dict.fromkeys(QUAL_SOURCES+[SELF,PROTOCOL,WORKER,
    'scripts/fetch_shared_actor_fridge90.py','src/adapters/fridge_runtime.py']))


class Actor(PinnedActor):
    def __init__(self,args,out):
        from adapters.tf5_compat import snapshot_dir
        self.log=(out/'pinned_worker.log').open('w')
        self.process=subprocess.Popen([PINNED,'-u',str(REPO/WORKER),SNAPSHOT,snapshot_dir(),KEY],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1,
            env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
        self.build=self.receive()
        assert self.build['ready'] and self.build['unnorm_key']==KEY


def measure(env,obs,monitor,samples):
    sensed=sense(env,obs)
    released=not any('cream_cheese' in str(c) and 'gripper' in str(c)
                     for sample in samples for c in sample['contacts'])
    task=monitor.observe(sensed,interval_released=released)
    return dict(sensed={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in sensed.items()},
                task=task,interval_released=released,native_success=bool(env.check_success()))


def episode(out,door,manifest):
    import torch
    gpu_check()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    out.mkdir(parents=True,exist_ok=False)
    args=argparse.Namespace(scene='fridge',camera='robot_side',root='/home/ami/scratch/LIBERO',
        steps=600,gpu=3,seed=0,door=door,yaw=-np.pi/2,fridge_y=.35,
        open_angle=float(np.deg2rad(130)),item_forward=.04,item_lateral=.035,
        snapshot=SNAPSHOT,pinned_python=PINNED)
    status=dict(status='started',started=time.time(),args=vars(args),sources=manifest['sources'],
        scope='learned_LIBERO90_actor_fridge_OOD_transfer_not_framework',queries=0,steps_executed=0,
        intervention_actions=0,setup_settling_calls=20,normalization_key=KEY)
    save(out/'status.json',status)
    env=actor=recorder=None
    states,images,rows=[],[],[]
    try:
        env,obs=make_scene(args,out)
        language=env.language_instruction if door=='closed' else 'pick up the cream cheese box inside the fridge'
        status.update(language=language,initial=fingerprint(env,obs),initial_clock=clock(env))
        (out/'model.xml').write_text(env.sim.model.get_xml())
        actor=Actor(args,out)
        status['actor_build']=actor.build
        recorder=ContactRecorder(env.sim,keywords=('cream_cheese','short_fridge','gripper','table_collision'))
        monitor=TaskMonitor(door)
        def forbidden(*a,**kw):
            raise RuntimeError('Post-setup reset forbidden')
        env.reset=env.set_init_state=env.env.reset=forbidden
        start=time.monotonic()
        previous_done=False
        with (out/'steps.jsonl').open('w') as log,(out/'contacts.jsonl').open('w') as contacts:
            for step in range(600):
                if time.monotonic()-start>1200:
                    status['termination']='wall_budget'
                    break
                states.append(env.get_sim_state().copy())
                images.append(obs['agentview_image'].copy())
                before,tick=fingerprint(env,obs),clock(env)
                assert not tick['internal_done']
                t=time.monotonic()
                action,decode,image=actor.act(obs['agentview_image'],language)
                latency=time.monotonic()-t
                status['queries']+=1
                assert action.shape==(7,) and np.isfinite(action).all()
                if step==0: image.save(out/'policy_input.png')
                recorder.reset()
                t=time.monotonic()
                obs,reward,done,_=env.step(action.tolist())
                simulation=time.monotonic()-t
                diagnostic=measure(env,obs,monitor,recorder.samples)
                row=dict(step=step,owner='vla',action=action.tolist(),proposal=action.tolist(),
                    before=before,after=fingerprint(env,obs),clock_before=tick,clock_after=transition(env,tick),
                    returned_done=bool(done),action_after_native_done=previous_done,reward=float(reward),
                    inference_s=latency,simulation_s=simulation,elapsed_s=time.monotonic()-start,
                    contact_samples_sha256=digest(json.dumps(recorder.samples,sort_keys=True).encode()),
                    decision_trace=dict(input_scope='RGB_and_constant_language_only',input_language=language,
                        input_rgb_sha256=digest(images[-1].tobytes()),processed_rgb_sha256=digest(np.asarray(image).tobytes()),
                        selection='identity_of_model_proposal',hidden_reasoning_available=False,
                        monitor_role='logging_and_termination_only'),**decode,**diagnostic)
                assert bool(done)==row['native_success']
                previous_done=bool(done)
                rows.append(row)
                log.write(json.dumps(row)+'\n'); log.flush()
                contacts.write(json.dumps(dict(step=step,samples=recorder.samples))+'\n'); contacts.flush()
                status['steps_executed']=step+1
                if step%40==0:
                    save(out/'status.json',status)
                    print(json.dumps(dict(step=step,angle=row['sensed']['angle'],grasp=row['sensed']['grasp'],
                        task=row['task'],native=row['native_success'])),flush=True)
                if row['task']['remaining_sequence_complete']:
                    status['termination']='strict_remaining_completion'
                    break
            states.append(env.get_sim_state().copy())
            images.append(obs['agentview_image'].copy())
        status.update(status='completed',termination=status.get('termination','action_budget'),
            rollout_wall_s=time.monotonic()-start,final_task=rows[-1]['task'],final_sensed=rows[-1]['sensed'],
            any_grasp=any(r['sensed']['grasp'] for r in rows),
            any_native=any(r['native_success'] for r in rows),
            first_grasp=next((r['step'] for r in rows if r['sensed']['grasp']),None),
            first_native=next((r['step'] for r in rows if r['native_success']),None),
            full_completion=any(r['task']['full_completion'] for r in rows),
            remaining_completion=any(r['task']['remaining_sequence_complete'] for r in rows))
    except BaseException:
        status.update(status='failed',termination='error',error=traceback.format_exc())
        raise
    finally:
        if states:
            np.savez_compressed(out/'states.npz',states=np.asarray(states))
            np.savez_compressed(out/'images.npz',agent=np.asarray(images))
        status['ended']=time.time()
        save(out/'status.json',status)
        if actor: actor.close()
        if recorder: recorder.close()
        if env: env.close()


def replay(run,out):
    gpu_check()
    status=json.loads((run/'status.json').read_text())
    assert status['status']=='completed'
    assert all(digest((REPO/p).read_bytes())==h for p,h in status['sources'].items())
    out.mkdir(exist_ok=False)
    env,obs=make_scene(argparse.Namespace(**status['args']),out)
    recorder=ContactRecorder(env.sim,keywords=('cream_cheese','short_fridge','gripper','table_collision'))
    monitor=TaskMonitor(status['args']['door'])
    report=dict(status='started',mismatches=[],max_state_gap=0.,actor_requeried=False)
    try:
        rows=[json.loads(s) for s in (run/'steps.jsonl').read_text().splitlines()]
        states=np.load(run/'states.npz')['states']; images=np.load(run/'images.npz')['agent']
        assert len(states)==len(images)==len(rows)+1
        previous_done=False
        for i,row in enumerate(rows):
            report['max_state_gap']=max(report['max_state_gap'],float(np.max(np.abs(env.get_sim_state()-states[i]))))
            assert np.array_equal(obs['agentview_image'],images[i]),(i,'image')
            assert row['action']==row['proposal'] and row['owner']=='vla'
            expected=dict(before=fingerprint(env,obs),clock_before=clock(env))
            recorder.reset()
            obs,reward,done,_=env.step(row['action'])
            expected.update(after=fingerprint(env,obs),clock_after=transition(env,expected['clock_before']),
                returned_done=bool(done),action_after_native_done=previous_done,reward=float(reward),
                contact_samples_sha256=digest(json.dumps(recorder.samples,sort_keys=True).encode()),
                **measure(env,obs,monitor,recorder.samples))
            previous_done=bool(done)
            for key,value in expected.items():
                if value!=row[key]: report['mismatches'].append([i,key])
        assert np.array_equal(obs['agentview_image'],images[-1])
        report['max_state_gap']=max(report['max_state_gap'],float(np.max(np.abs(env.get_sim_state()-states[-1]))))
        report.update(status='exact_replay_passed' if not report['mismatches'] and report['max_state_gap']==0
            else 'replay_mismatch',steps=len(rows),final_task=rows[-1]['task'])
    finally:
        save(out/'verification.json',report)
        recorder.close(); env.close()
    assert report['status']=='exact_replay_passed',report


def review(out):
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    tapes=[]
    summary=[]
    for door in ('closed','open'):
        run=out/door
        status=json.loads((run/'status.json').read_text())
        rows=[json.loads(s) for s in (run/'steps.jsonl').read_text().splitlines()]
        images=np.load(run/'images.npz')['agent']
        tapes.append((status,rows,images))
        summary.append(dict(door=door,**{k:status[k] for k in ('queries','steps_executed','first_grasp',
            'first_native','any_grasp','full_completion','remaining_completion','termination','final_task')},
            median_inference_s=float(np.median([r['inference_s'] for r in rows])),
            max_door_angle=max(r['sensed']['angle'] for r in rows),
            max_item_displacement=float(max(np.linalg.norm(np.asarray(r['sensed']['item'])-
                np.asarray(rows[0]['sensed']['item'])) for r in rows))))
    with imageio.get_writer(out/'fridge90_actor_only_annotated.mp4',fps=20,codec='libx264',quality=8,
                           macro_block_size=16) as writer:
        for frame in range(max(len(t[2]) for t in tapes)+40):
            canvas=Image.new('RGB',(960,800),'#121721'); draw=ImageDraw.Draw(canvas)
            draw.text((12,8),'EXPERIMENT 2 | LIBERO90 learned actor -> fridge OOD transfer | zero repair actions',font=font)
            draw.text((12,30),'Actor-only candidate test, no local adaptation | 20 Hz simulation, inference pauses physics',font=font)
            for index,(status,rows,images) in enumerate(tapes):
                f=min(frame,len(images)-1); x=index*480; ended=frame>=len(images)-1
                canvas.paste(Image.fromarray(images[f][::-1,::-1]).resize((480,480)),(x,64))
                lines=[f'Initial door {status["args"]["door"]} | owner VLA | actions {f}/600']
                if f:
                    row=rows[f-1]; task=row['task']
                    lines += [f'Native {row["native_success"]} | full strict {task["full_completion"]}',
                        f'Remaining complete {task["remaining_sequence_complete"]} | dwell {task["completion_streak"]}/20',
                        f'Grasp {row["sensed"]["grasp"]} | extracted ever {task["extracted_seen"]}',
                        f'Door angle {row["sensed"]["angle"]:.3f} rad | placed {row["sensed"]["placed"]}',
                        f'Inference {row["inference_s"]:.3f}s | identity model proposal']
                else: lines += ['INITIAL: before first model query']
                lines += [f'STOP: {status["termination"]}' if ended else 'Constant language; no state-conditioned action edits',
                          'END HOLD: no physics / no extra dwell' if ended else 'Monitor is not an actor input']
                for j,line in enumerate(lines): draw.text((x+8,554+j*26),line,font=font,fill='white')
            writer.append_data(np.asarray(canvas))
    save(out/'summary.json',summary)


def supervise(out):
    gpu_check()
    out.mkdir(parents=True,exist_ok=False)
    checkpoint=json.loads((Path(SNAPSHOT)/'download_manifest.json').read_text())
    assert checkpoint['status']=='hash_verified'
    frozen=out/'source'; hashes={}
    for name in SOURCES:
        data=(REPO/name).read_bytes(); dest=frozen/name
        dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(data)
        hashes[name]=digest(data)
    external={}
    import robosuite
    paths=[Path('/home/ami/scratch/LIBERO/libero/libero')/n for n in
        ('envs/bddl_base_domain.py','envs/env_wrapper.py','assets/articulated_objects/short_fridge.xml')]
    paths += [Path(robosuite.__file__).parent/'environments/base.py']
    for i,path in enumerate(paths):
        data=path.read_bytes(); dest=out/'external'/f'{i}_{path.name}'
        dest.parent.mkdir(exist_ok=True); dest.write_bytes(data); external[str(path)]=digest(data)
    manifest=dict(status='running',started=time.time(),pid=os.getpid(),sources=hashes,external_sources=external,
                  checkpoint=checkpoint,gpu=3,jobs=[])
    save(out/'launch.json',manifest)
    def bounded(cmd,log,timeout):
        with log.open('w') as f:
            p=subprocess.Popen(cmd,cwd=frozen,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                               start_new_session=True)
            try: return p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL); p.wait(); raise
    for door in ('closed','open'):
        job=dict(door=door,status='running',started=time.time()); manifest['jobs'].append(job)
        save(out/'launch.json',manifest)
        try:
            assert all(digest((frozen/p).read_bytes())==h for p,h in hashes.items())
            assert all(digest(Path(p).read_bytes())==h for p,h in external.items())
            cmd=[sys.executable,'-u',str(frozen/SELF),'episode',str(out/door),'--door',door,
                 '--manifest',str(out/'launch.json')]
            job['command']=cmd; job['actor_returncode']=bounded(cmd,out/f'{door}.log',1800)
            assert job['actor_returncode']==0
            cmd=[sys.executable,'-u',str(frozen/SELF),'replay',str(out/door),'--replay-out',str(out/f'replay_{door}')]
            job['replay_returncode']=bounded(cmd,out/f'replay_{door}.log',360)
            assert job['replay_returncode']==0
            job['status']='exact_verified'
        except Exception: job.update(status='failed',error=traceback.format_exc())
        finally: job['ended']=time.time(); save(out/'launch.json',manifest)
    if all(j['status']=='exact_verified' for j in manifest['jobs']):
        manifest['review_returncode']=bounded([sys.executable,str(frozen/SELF),'review',str(out)],out/'review.log',360)
    manifest.update(status='complete_verified' if all(j['status']=='exact_verified' for j in manifest['jobs'])
        and manifest.get('review_returncode')==0 else 'complete_with_issues',ended=time.time())
    save(out/'launch.json',manifest)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['supervise','episode','replay','review'])
    p.add_argument('out',type=Path); p.add_argument('--door',choices=['closed','open'])
    p.add_argument('--manifest',type=Path); p.add_argument('--replay-out',type=Path)
    a=p.parse_args()
    if a.mode=='supervise': supervise(a.out.resolve())
    elif a.mode=='episode': episode(a.out.resolve(),a.door,json.loads(a.manifest.read_text()))
    elif a.mode=='replay': replay(a.out.resolve(),a.replay_out.resolve())
    else: review(a.out.resolve())
