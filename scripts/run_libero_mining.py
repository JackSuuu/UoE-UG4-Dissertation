"""Live frozen OpenVLA continuations on a fixed development fault grid."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from run_libero_drawer import reset_seeds, write_json, torch, imageio, np
from adapters.libero_completion import ContactRecorder
from adapters.libero_drawer import DrawerMonitor, SubgoalTracker, settled_drawer, MAX_STEPS
from adapters.libero_mining import (ARMS, ReleaseGate, EvidenceMemory, apply_fault,
    forecast_candidates, apply_verification, trace_record, observed_outcome, prefix_decision)
from adapters.libero_openvla import LiberoOpenVLA
from adapters.libero_preservation import fingerprint

PROTOCOL = 'exp_record/libero_failure_mining_protocol_20261015.md'


def source_hashes():
    repo=Path(__file__).resolve().parents[1]
    paths=[Path(__file__),repo/'scripts/run_libero_drawer.py',repo/'scripts/run_libero_baseline.py',
           repo/'scripts/verify_libero_mining.py']
    paths += [repo/'src/adapters'/name for name in ['libero_mining.py','libero_stress.py',
        'libero_drawer.py','libero_preservation.py','libero_completion.py',
        'libero_runtime.py','libero_openvla.py','tf5_compat.py']]
    return {str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def episode(actor, args, case, arm, manifest, fixture=False):
    out=args.out/case['case_id']/arm
    out.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[1]
    source=Path(manifest['source_run'])/f"root_{case['root']}/vla/steps.jsonl"
    assert hashlib.sha256(source.read_bytes()).hexdigest()==case['source_steps_sha256']
    old=[json.loads(s) for s in source.read_text().splitlines()]
    previous_manifest=json.loads((source.parent/'manifest.json').read_text())
    task_base=Path(args.root)/'libero/libero'
    for path,sha in previous_manifest['task_sha256'].items():
        assert hashlib.sha256((task_base/path).read_bytes()).hexdigest()==sha
    if not fixture:
        assert actor.build==previous_manifest['actor'], 'Frozen actor provenance changed'
    meta=dict(status='started',start=time.time(),case=case,arm=arm,task=manifest['task'],
        actor=actor.build,H=1,privileged=True,fixture_only=fixture,source_sha256=source_hashes(),
        protocol_sha256=hashlib.sha256((repo/PROTOCOL).read_bytes()).hexdigest(),
        grid_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        task_sha256=previous_manifest['task_sha256'],source_run=manifest['source_run'],
        git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        max_steps=case['anchor_step']+12 if fixture else MAX_STEPS,
        GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),methods_receive_fault_metadata=False)
    write_json(out/'manifest.json',meta)
    reset_seeds()
    env=rec=None
    try:
        env,obs=settled_drawer(args.root,case['root'])
        monitor,tracker,gate,memory=DrawerMonitor(env),SubgoalTracker(),ReleaseGate(arm),EvidenceMemory()
        rec=ContactRecorder(env.sim,keywords=('akita_black_bowl','white_cabinet'))
        previous=monitor.measure([])
        tracking=tracker.update(previous,-1)
        previous_selected=None
        states,rows,actions=[],[],[]
        (out/'model.xml').write_text(env.sim.model.get_xml())
        with imageio.get_writer(out/'rollout.mp4',fps=20,codec='libx264',
                ffmpeg_params=['-movflags','+faststart']) as video, \
                (out/'steps.jsonl').open('w') as log, (out/'substeps.jsonl').open('w') as raw, \
                (out/'decision_trace.jsonl').open('w') as trace_log:
            for step in range(meta['max_steps']):
                states.append(env.get_sim_state().copy())
                before=fingerprint(env,obs)
                video.append_data(obs['agentview_image'][::-1,::-1])
                memory_record=memory.observe(previous_selected,previous,tracking,step)
                forecast=None
                actor_seconds=0.
                if step<case['anchor_step']:
                    assert before==old[step]['before']
                    proposal=np.asarray(old[step]['proposal'])
                    selected=proposal.copy()
                    decode={k:old[step][k] for k in ['tokens','native_action']}
                    decision=prefix_decision()
                    live_query=False
                else:
                    if step==case['anchor_step']:
                        assert before==case['prefix_before_fingerprint']
                    if not fixture:
                        torch.cuda.synchronize()
                    start=time.monotonic()
                    proposal,decode,image=actor.act(obs['agentview_image'],env.language_instruction)
                    if not fixture:
                        torch.cuda.synchronize()
                    actor_seconds=time.monotonic()-start
                    if step==case['anchor_step']:
                        image.save(out/'policy_input.png')
                        if not fixture:
                            assert decode['tokens']==old[step]['tokens'], 'Anchor actor decode changed'
                            np.testing.assert_array_equal(proposal,old[step]['proposal'])
                    selected,decision,phase_action=gate.decide(proposal,previous,tracking,step)
                    if decision['forecast_event'] is not None:
                        forecast=forecast_candidates(args.root,case['root'],actions,before,proposal,phase_action)
                    selected,decision=apply_verification(arm,selected,decision,forecast)
                    live_query=not fixture
                trace=trace_record(step,arm,previous,tracking,proposal,selected,decision,memory_record,forecast)
                assert fingerprint(env,obs)==before, 'Monitor/branch mutated live physics or camera'
                effective,fault=apply_fault(selected,step,case)
                assert np.isfinite(effective).all()
                rec.reset()
                obs,_,_,_=env.step(effective.tolist())
                assert len(rec.samples)==25
                after=fingerprint(env,obs)
                measured=monitor.measure(rec.samples)
                assert fingerprint(env,obs)==after
                post=tracker.update(measured,step)
                if step<case['anchor_step']:
                    assert measured==old[step]['measured'] and post==old[step]['tracking']
                changed=not np.array_equal(selected,proposal)
                trace['effective_action']=effective.tolist()
                trace['external_fault_evaluator_only']=fault
                trace['observed_after_execution']=observed_outcome(measured,post)
                if forecast is not None:
                    trace['forecast_first_step_matches_observed']=[p['rows'][0]==measured for p in forecast['candidates']]
                    # Exact first-step identity is required only if no actuator fault changed the command.
                    for k,p in enumerate(forecast['candidates']):
                        if p['plan'][0]==effective.tolist():
                            assert p['rows'][0]==measured, ('first forecast step',step,k)
                row=dict(step=step,before=before,proposal=proposal.tolist(),selected=selected.tolist(),
                    action=effective.tolist(),**decode,decision=decision,tracking_before=tracking,
                    tracking=post,measured=measured,correction=changed,live_query=live_query,
                    actor_seconds=actor_seconds,fault=fault,forecast=forecast)
                log.write(json.dumps(row)+'\n'); log.flush()
                raw.write(json.dumps(dict(step=step,substeps=rec.samples))+'\n'); raw.flush()
                trace_log.write(json.dumps(trace)+'\n'); trace_log.flush()
                rows.append(row); actions.append(effective.tolist())
                previous,tracking,previous_selected=measured,post,selected.copy()
                if step==case['anchor_step'] or step%40==0 or decision['mode']=='contact_transition' or post['complete']:
                    print(json.dumps(dict(case=case['case_id'],arm=arm,step=step,reason=decision['reason'],
                        changed=changed,dwell=post['final_streak'])),flush=True)
                if post['complete']:
                    break
            states.append(env.get_sim_state().copy())
            video.append_data(obs['agentview_image'][::-1,::-1])
        np.savez_compressed(out/'states.npz',states=np.asarray(states))
        result=dict(case_id=case['case_id'],index=case['root'],arm=arm,steps=len(rows),
            complete=tracking['complete'],native_ever=any(r['measured']['native_success'] for r in rows),
            milestone_steps=tracker.milestones,trigger=gate.trigger,handoff=gate.handoff,
            intervention_steps=gate.used,changed_actions=sum(r['correction'] for r in rows),
            fault_changed_actions=sum(r['fault']['changed'] for r in rows),
            VLA_queries=sum(r['live_query'] for r in rows),
            replayed_prefix_steps=case['anchor_step'],forecast_queries=sum(r['forecast'] is not None for r in rows),
            forecast_steps=sum(r['forecast']['forecast_steps'] for r in rows if r['forecast']),
            final=previous,CVR=None,safe_success=None,fixture_only=fixture,
            post_handoff_bowl_contact_intervals=None if gate.handoff is None else sum(
                r['measured']['bowl_gripper'] for r in rows if r['step']>=gate.handoff['step']))
        write_json(out/'result.json',result)
        meta['status']='complete'
    except BaseException:
        meta.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        if rec is not None: rec.close()
        if env is not None: env.close()
        meta['end']=time.time()
        write_json(out/'manifest.json',meta)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--root',default='/home/ami/scratch/LIBERO')
    p.add_argument('--snapshot',required=True)
    p.add_argument('--index',type=int,choices=range(3),required=True)
    args=p.parse_args()
    torch.set_num_threads(4)
    reset_seeds()
    manifest=json.loads(args.cases.read_text())
    actor=LiberoOpenVLA(args.snapshot)
    from verify_libero_mining import verify, render_case
    for case in manifest['cases']:
        if case['root']!=args.index: continue
        for arm in ARMS:
            directory=episode(actor,args,case,arm,manifest)
            verify(directory,args.root)
        render_case(args.out,case)
