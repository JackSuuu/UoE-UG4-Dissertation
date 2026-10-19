"""Exact-prefix offline candidate branch; its outcomes never enter online selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'base_source/scripts'))
from experiment3_fridge_run import (make_scene, read_state, initial_stages, interval_release,
    FridgeStageEvidence, ContactRecorder, CandidateActor, PROMPTS, fingerprint, save, box_corners)
from experiment3_q1_metrics import measurements, endpoint


def metric(env, state):
    m,d=env.sim.model,env.sim.data
    g=m.geom_name2id('cream_cheese_1_g1')
    corners=box_corners(d.geom_xpos[g],d.geom_xmat[g].reshape(3,3),m.geom_size[g])
    return measurements(state,float(corners[:,0].max()-(state['front']-.015)))


def contact_metric(samples):
    contacts=[c for sample in samples for c in sample['contacts']]
    return dict(max_contact_force_n=max((float(np.linalg.norm(c['wrench'][:3])) for c in contacts),default=0.),
        max_contact_penetration_m=max((max(0.,-c['distance']) for c in contacts),default=0.),
        actual_damage='unknown', validated_safety='unknown')


def reconstruct(opportunity,out):
    source=Path(opportunity['source_run'])
    source_status=json.loads((source/'status.json').read_text())
    args=argparse.Namespace(**source_status['args'])
    args.camera_config=str(ROOT/'camera_config.json')
    env,obs=make_scene(args,out)
    recorder=ContactRecorder(env.sim,keywords=('cream_cheese','short_fridge','gripper','table_collision'))
    tracker=FridgeStageEvidence('closed')
    s=read_state(env,obs)
    stages=initial_stages(tracker,s)
    rows=[json.loads(l) for l in (source/'steps.jsonl').read_text().splitlines()]
    target=opportunity['step']
    report=dict(status='started',source_step=target,prefix_actions_replayed=0,mismatches=[],
        method='fresh environment and exact action-prefix replay, no approximate state fork')
    try:
        for i in range(target):
            row=rows[i]
            assert fingerprint(env,obs)==row['before'], (i,'prefix_before_fingerprint')
            assert s==row['before_state'] and stages==row['before_stages'], (i,'prefix_before_evidence')
            recorder.reset()
            obs,*_=env.step(row['action'])
            s=read_state(env,obs)
            stages=tracker.observe(i,s,interval_release(recorder.samples))
            assert fingerprint(env,obs)==row['after'], (i,'prefix_after_fingerprint')
            assert s==row['after_state'] and stages==row['after_stages'], (i,'prefix_after_evidence')
            report['prefix_actions_replayed']=i+1
        row=rows[target]
        assert fingerprint(env,obs)==row['before']==opportunity['before'], 'branch_before_fingerprint'
        assert s==row['before_state']==opportunity['before_state'], 'branch_before_read_state'
        assert stages==row['before_stages']==opportunity['before_stages'], 'branch_before_stages'
        with np.load(source/'states.npz') as tape:
            assert np.array_equal(tape['states'][target],env.get_sim_state()), 'branch_before_raw_state'
        with np.load(source/'images.npz') as tape:
            assert np.array_equal(tape['agent'][target],obs['agentview_image']), 'branch_before_raw_image'
        assert env.env.horizon==640 and env.env.timestep+32<640 and not env.env.done
        report.update(status='exact_prefix_passed',before=fingerprint(env,obs),
            tracker_last_step=tracker.last_step, internal_timestep=env.env.timestep,
            all_prefix_fingerprints_and_evidence_equal=True, raw_state_and_image_equal=True)
    except BaseException as exc:
        report.update(status='prefix_mismatch_stop',error=repr(exc))
        recorder.close()
        env.close()
        raise
    finally:
        save(out/'prefix_identity.json',report)
    return env,obs,recorder,tracker,s,stages,args


def run(branch,out):
    import torch
    import imageio.v2 as imageio
    assert os.environ['CUDA_VISIBLE_DEVICES']==os.environ['MUJOCO_EGL_DEVICE_ID']=='2'
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    out.mkdir(parents=True,exist_ok=False)
    opportunity=branch['opportunity']
    status=dict(status='started',branch=branch['id'],source_step=opportunity['step'],
        candidate_index=branch['candidate_index'],started=time.time(),
        freeze_sha256=hashlib.sha256((ROOT/'freeze.json').read_bytes()).hexdigest(),
        scope='OFFLINE counterfactual evaluation only; no online handoff or repair claim')
    env=actor=recorder=None
    states,images,rows,queries=[],[],[],[]
    started=time.monotonic()
    try:
        env,obs,recorder,tracker,s,stages,args=reconstruct(opportunity,out)
        before_state,before_stages,before_measure=s,stages,metric(env,s)
        imageio.imwrite(out/'before.png',obs['agentview_image'][::-1,::-1])
        actor=CandidateActor(args,out)
        status['actor_build']=actor.build
        save(out/'status.json',status)
        with (out/'steps.jsonl').open('w') as log,(out/'queries.jsonl').open('w') as qlog:
            for offset in range(32):
                assert time.monotonic()-started<1200,'branch_wall_cap'
                absolute=opportunity['step']+offset
                before=fingerprint(env,obs)
                old_s,old_stages=s,stages
                old_metric=metric(env,s)
                language=PROMPTS[stages['phase']]
                if offset==0:
                    proposal=opportunity['candidates'][branch['candidate_index']]
                    owner='saved_frozen_model_candidate'
                    assert proposal['language']==language
                else:
                    t=time.monotonic()
                    proposal,_=actor.query(obs['agentview_image'],language,'greedy',absolute*10)
                    proposal.update(query=len(queries),branch_offset=offset,absolute_step=absolute,
                        language=language,query_wall_s=time.monotonic()-t)
                    queries.append(proposal)
                    qlog.write(json.dumps(proposal)+'\n')
                    qlog.flush()
                    owner='counterfactual_nominal_local_prefix' if offset<12 else 'counterfactual_nominal_tail_NOT_online_handoff'
                action=proposal['action']
                assert np.asarray(action).shape==(7,) and np.isfinite(action).all()
                states.append(env.get_sim_state().copy())
                images.append(obs['agentview_image'].copy())
                recorder.reset()
                obs,_,done,_=env.step(action)
                assert not env.env.done
                s=read_state(env,obs)
                released=interval_release(recorder.samples)
                stages=tracker.observe(absolute,s,released)
                row=dict(offset=offset,absolute_step=absolute,before=before,after=fingerprint(env,obs),
                    before_state=old_s,after_state=s,before_stages=old_stages,after_stages=stages,
                    before_measure=old_metric,after_measure=metric(env,s),contact=contact_metric(recorder.samples),
                    interval_released=released,language=language,action=action,proposal=proposal,owner=owner,
                    native_done_returned=bool(done),fresh_queries_used=len(queries),
                    frozen_selectors=opportunity['selectors'],offline_oracle_only=True,
                    counterfactual_tail_start=offset==12,actual_online_handoff=False)
                log.write(json.dumps(row,allow_nan=False)+'\n')
                log.flush()
                rows.append(row)
            states.append(env.get_sim_state().copy())
            images.append(obs['agentview_image'].copy())
        outcomes={str(h):endpoint(before_state,before_stages,before_measure,
            [r['after_state'] for r in rows],[r['after_stages'] for r in rows],
            [r['after_measure'] for r in rows],[r['contact'] for r in rows],opportunity['phase'],h)
            for h in (1,12,32)}
        status.update(status='completed',actions=len(rows),fresh_queries=len(queries),
            reused_original_candidate_queries=1,prefix_replay_actions=opportunity['step'],
            measured_at=outcomes,wall_s=time.monotonic()-started,
            model_inference_s=sum(q['model_inference_s'] for q in queries),
            query_wall_s=sum(q['query_wall_s'] for q in queries),
            termination='preregistered_32_action_counterfactual_horizon')
        imageio.imwrite(out/'after.png',obs['agentview_image'][::-1,::-1])
    except BaseException:
        status.update(status='failed',error=traceback.format_exc(),actions=len(rows),fresh_queries=len(queries))
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


def replay(branch,run_path,out):
    out.mkdir(parents=True,exist_ok=False)
    env,obs,recorder,tracker,s,stages,_=reconstruct(branch['opportunity'],out)
    rows=[json.loads(l) for l in (run_path/'steps.jsonl').read_text().splitlines()]
    states,images=np.load(run_path/'states.npz')['states'],np.load(run_path/'images.npz')['agent']
    report=dict(status='started',steps=len(rows),mismatches=[],max_state_gap=0.,
        prefix_replay_actions=branch['opportunity']['step'],fresh_queries=0)
    try:
        assert len(states)==len(images)==len(rows)+1==33
        for i,r in enumerate(rows):
            report['max_state_gap']=max(report['max_state_gap'],float(np.abs(states[i]-env.get_sim_state()).max()))
            assert np.array_equal(images[i],obs['agentview_image']),(i,'image')
            assert fingerprint(env,obs)==r['before'] and s==r['before_state'] and stages==r['before_stages'],(i,'before')
            assert metric(env,s)==r['before_measure'] and PROMPTS[stages['phase']]==r['language'],(i,'measurement_prompt')
            assert r['action']==r['proposal']['action']
            recorder.reset()
            obs,_,done,_=env.step(r['action'])
            s=read_state(env,obs)
            stages=tracker.observe(r['absolute_step'],s,interval_release(recorder.samples))
            assert s==r['after_state'] and stages==r['after_stages'] and fingerprint(env,obs)==r['after'],(i,'after')
            assert metric(env,s)==r['after_measure'] and contact_metric(recorder.samples)==r['contact'],(i,'contact_metric')
            assert bool(done)==r['native_done_returned'] and not env.env.done
        report['max_state_gap']=max(report['max_state_gap'],float(np.abs(states[-1]-env.get_sim_state()).max()))
        assert np.array_equal(images[-1],obs['agentview_image']) and report['max_state_gap']==0.
        report['status']='exact_replay_passed'
    except BaseException as exc:
        report.update(status='replay_mismatch_stop',error=repr(exc))
        raise
    finally:
        save(out/'verification.json',report)
        recorder.close()
        env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--branch',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--replay',type=Path)
    a=p.parse_args()
    plan=json.loads((ROOT/'plan.json').read_text())
    b=next(b for b in plan['branches'] if b['id']==a.branch)
    replay(b,a.replay,a.out) if a.replay else run(b,a.out)
