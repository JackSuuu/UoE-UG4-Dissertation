"""Replay every effective action, reconstruct causal proof, publish all case outcomes."""
import argparse
import hashlib
import html
import itertools
import json
from pathlib import Path

from run_libero_mining import (PROTOCOL, reset_seeds, np, imageio, DrawerMonitor, SubgoalTracker,
    ReleaseGate, EvidenceMemory, ContactRecorder, settled_drawer, fingerprint, apply_fault,
    apply_verification, trace_record, observed_outcome, prefix_decision, forecast_candidates, ARMS)
from adapters.libero_mining import candidate_plans, forecast_key
from PIL import Image, ImageDraw, ImageFont


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def verify(directory, root):
    meta=json.loads((directory/'manifest.json').read_text())
    result=json.loads((directory/'result.json').read_text())
    assert meta['status']=='complete'
    repo=Path(__file__).resolve().parents[1]
    for path,sha in meta['source_sha256'].items():
        assert hashlib.sha256((repo/path).read_bytes()).hexdigest()==sha,path
    assert hashlib.sha256((repo/PROTOCOL).read_bytes()).hexdigest()==meta['protocol_sha256']
    for path,sha in meta['task_sha256'].items():
        assert hashlib.sha256((Path(root)/'libero/libero'/path).read_bytes()).hexdigest()==sha
    rows,traces=read_rows(directory/'steps.jsonl'),read_rows(directory/'decision_trace.jsonl')
    case,arm=meta['case'],meta['arm']
    states=np.load(directory/'states.npz')['states']
    assert len(rows)==len(traces)==result['steps']<=meta['max_steps']
    assert len(states)==len(rows)+1
    source=Path(meta['source_run'])/f"root_{case['root']}/vla/steps.jsonl"
    assert hashlib.sha256(source.read_bytes()).hexdigest()==case['source_steps_sha256']
    old=read_rows(source)
    reset_seeds()
    env,obs=settled_drawer(root,case['root'])
    rec=ContactRecorder(env.sim,keywords=('akita_black_bowl','white_cabinet'))
    mon,tr,gate,memory=DrawerMonitor(env),SubgoalTracker(),ReleaseGate(arm),EvidenceMemory()
    previous=mon.measure([])
    tracking=tr.update(previous,-1)
    previous_selected=None
    actions=[]
    try:
        with (directory/'substeps.jsonl').open() as raw:
            for step,row in enumerate(rows):
                assert row['step']==step and row['before']==fingerprint(env,obs)
                np.testing.assert_array_equal(states[step],env.get_sim_state())
                assert row['tracking_before']==tracking
                proposal=np.asarray(row['native_action']).copy()
                proposal[-1]=-np.sign(2*proposal[-1]-1)
                np.testing.assert_array_equal(proposal,row['proposal'])
                mem=memory.observe(previous_selected,previous,tracking,step)
                forecast=row['forecast']
                if step<case['anchor_step']:
                    assert row['before']==old[step]['before']
                    assert row['proposal']==old[step]['proposal'] and row['tokens']==old[step]['tokens']
                    assert not row['live_query'] and forecast is None
                    selected,decision=proposal.copy(),prefix_decision()
                else:
                    assert row['live_query']==(not meta['fixture_only'])
                    if step==case['anchor_step']:
                        assert row['before']==case['prefix_before_fingerprint']
                        if not meta['fixture_only']:
                            assert row['proposal']==old[step]['proposal'] and row['tokens']==old[step]['tokens']
                    selected,decision,phase_action=gate.decide(proposal,previous,tracking,step)
                    assert (forecast is not None)==(decision['forecast_event'] is not None)
                    if forecast:
                        regenerated=forecast_candidates(root,case['root'],actions,row['before'],proposal,phase_action)
                        assert {k:v for k,v in regenerated.items() if k!='wall_seconds'}=={
                            k:v for k,v in forecast.items() if k!='wall_seconds'},(directory,step,'forecast')
                        assert fingerprint(env,obs)==row['before']
                        np.testing.assert_array_equal(candidate_plans(proposal,phase_action),
                            [p['plan'] for p in forecast['candidates']])
                        for k,pred in enumerate(forecast['candidates']):
                            assert pred['key']==forecast_key(pred['rows'],np.asarray(pred['plan'][0]),proposal,k)
                    selected,decision=apply_verification(arm,selected,decision,forecast)
                assert decision==row['decision']
                np.testing.assert_array_equal(selected,row['selected'])
                effective,fault=apply_fault(selected,step,case)
                np.testing.assert_array_equal(effective,row['action'])
                assert fault==row['fault']
                trace=trace_record(step,arm,previous,tracking,proposal,selected,decision,mem,forecast)
                rec.reset()
                obs,_,_,_=env.step(effective.tolist())
                assert json.loads(next(raw))==dict(step=step,substeps=rec.samples)
                assert len(rec.samples)==25
                measured=mon.measure(rec.samples)
                tracking=tr.update(measured,step)
                assert measured==row['measured'] and tracking==row['tracking']
                trace['effective_action']=effective.tolist()
                trace['external_fault_evaluator_only']=fault
                trace['observed_after_execution']=observed_outcome(measured,tracking)
                if forecast:
                    trace['forecast_first_step_matches_observed']=[p['rows'][0]==measured for p in forecast['candidates']]
                    for pred in forecast['candidates']:
                        if pred['plan'][0]==effective.tolist(): assert pred['rows'][0]==measured
                assert trace==traces[step],(directory,step,'causal trace')
                assert row['correction']==(not np.array_equal(selected,proposal))
                if tracking['complete']: assert step==len(rows)-1
                previous,previous_selected=measured,selected.copy()
                actions.append(effective.tolist())
            assert raw.readline()==''
        np.testing.assert_array_equal(states[-1],env.get_sim_state())
        assert tracking['complete'] or len(rows)==meta['max_steps']
        assert result['complete']==tracking['complete']
        assert result['native_ever']==any(r['measured']['native_success'] for r in rows)
        assert result['milestone_steps']==tr.milestones
        assert result['trigger']==gate.trigger and result['handoff']==gate.handoff
        assert result['intervention_steps']==gate.used<=40
        assert result['changed_actions']==sum(r['correction'] for r in rows)
        assert result['fault_changed_actions']==sum(r['fault']['changed'] for r in rows)
        assert result['VLA_queries']==sum(r['live_query'] for r in rows)
        assert result['forecast_queries']==sum(r['forecast'] is not None for r in rows)<=2
        assert result['forecast_steps']==9*result['forecast_queries']
        assert result['final']==previous
        expected_contact=None if gate.handoff is None else sum(
            r['measured']['bowl_gripper'] for r in rows if r['step']>=gate.handoff['step'])
        assert result['post_handoff_bowl_contact_intervals']==expected_contact
        assert result['CVR'] is None and result['safe_success'] is None
        with imageio.get_reader(directory/'rollout.mp4') as reader:
            assert reader.count_frames()==len(states) and reader.get_meta_data()['fps']==20
        report=dict(status='verified',fixture_only=meta['fixture_only'],full_action_replay=True,
            causal_trace=True,forecast_replay=True,artifact_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in directory.iterdir() if p.is_file() and p.name!='verification.json'})
        (directory/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
        return result
    finally:
        rec.close(); env.close()


def render_case(run, case):
    review=run/'review'; review.mkdir(exist_ok=True)
    rows={a:read_rows(run/case['case_id']/a/'steps.jsonl') for a in ARMS}
    traces={a:read_rows(run/case['case_id']/a/'decision_trace.jsonl') for a in ARMS}
    manifests=[json.loads((run/case['case_id']/a/'manifest.json').read_text()) for a in ARMS]
    for m in manifests[1:]:
        for key in ['case','actor','source_sha256','protocol_sha256','grid_sha256','task_sha256','fixture_only']:
            assert m[key]==manifests[0][key],key
    anchor=case['anchor_step']
    for arm in ARMS[1:]:
        for i in range(anchor+1):
            assert rows[arm][i]['before']==rows['vla'][i]['before']
            assert rows[arm][i]['proposal']==rows['vla'][i]['proposal']
    readers={a:imageio.get_reader(run/case['case_id']/a/'rollout.mp4') for a in ARMS}
    name=case['case_id']+'_methods_2x.mp4'
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    last={}; count=max(len(r)+1 for r in rows.values())
    try:
        with imageio.get_writer(review/name,fps=40,codec='libx264',ffmpeg_params=['-movflags','+faststart']) as writer:
            for frame in range(count):
                panels=[]
                for arm in ARMS:
                    if frame<=len(rows[arm]): last[arm]=readers[arm].get_next_data()
                    canvas=Image.new('RGB',(320,560),'#101722'); draw=ImageDraw.Draw(canvas)
                    draw.text((6,6),arm+' | privileged H1 | 2x',font=font,fill='white')
                    draw.text((6,24),case['case_id'],font=font,fill='#9db1ce')
                    canvas.paste(Image.fromarray(last[arm]).resize((320,320)),(0,44))
                    if frame:
                        r=rows[arm][min(frame,len(rows[arm]))-1]; t=traces[arm][min(frame,len(rows[arm]))-1]
                        d=r['decision']
                        lines=[f"t={r['step']} phase={t['subgoal']}",f"{d['mode']} / {d['route']}",
                            f"Reason: {d['reason']}",f"Framework change={r['correction']} used={d['used']}/40",
                            f"External fault active={r['fault']['active']}",
                            f"Retained={r['measured']['retained']} grip={r['measured']['bowl_gripper']}",
                            f"Open/contact history={t['action_consequence_memory']['contact_persists_after_open']}",
                            f"Full task dwell={r['tracking']['final_streak']}/20",
                            'ENDED' if frame>=len(rows[arm]) else '']
                        for j,line in enumerate(lines): draw.text((6,374+18*j),line,font=font,fill='white')
                    panels.append(np.asarray(canvas))
                writer.append_data(np.concatenate(panels,axis=1))
    finally:
        for r in readers.values(): r.close()
    with imageio.get_reader(review/name) as r: assert r.count_frames()==count
    events=[]
    for arm in ARMS:
        last_signature=None
        for t in traces[arm]:
            g=t['gate']; signature=(t['subgoal'],g['mode'],g['reason'])
            if signature!=last_signature or g['mode']=='contact_transition':
                label=html.escape(f"{arm} | {t['step']} | {g['reason']}")
                payload=html.escape(json.dumps(t,indent=2))
                events.append((t['step'],f'<details><summary><button onclick="jump({(t["step"]+1)/40})">▶</button>{label}</summary><pre>{payload}</pre></details>'))
            last_signature=signature
    page=f'''<!doctype html><meta charset="utf-8"><title>{case['case_id']}</title>
<style>body{{background:#101722;color:white;font:16px system-ui}}video{{width:100%;position:sticky;top:0}}pre{{white-space:pre-wrap}}</style>
<h1>{case['case_id']}: all five methods</h1><p>Development discovery; privileged physical proof, not official SEAL/CheckVLA. Fault labels are evaluator-only. Ended panels freeze.</p>
<video id="v" controls src="{name}"></video><script>function jump(t){{v.currentTime=t;v.pause()}}</script>'''
    (review/(case['case_id']+'_trace.html')).write_text(page+''.join(s for _,s in sorted(events)))


def summarize(run, cases_path):
    grid=json.loads(cases_path.read_text()); cases=grid['cases']
    results={}; hashes={}
    for c in cases:
        for arm in ARMS:
            d=run/c['case_id']/arm
            proof=json.loads((d/'verification.json').read_text())
            assert proof['status']=='verified' and not proof['fixture_only']
            for path,sha in proof['artifact_sha256'].items():
                assert hashlib.sha256((d/path).read_bytes()).hexdigest()==sha
                hashes[str((d/path).relative_to(run))]=sha
            results[c['case_id'],arm]=json.loads((d/'result.json').read_text())
    comparisons=[]
    for method,control in [('phase_repair','vla'),('verified_repair','vla'),
        ('phase_repair','gate_hold'),('phase_repair','timed_repair'),
        ('verified_repair','gate_hold'),('verified_repair','timed_repair'),('verified_repair','phase_repair')]:
        for family in ['all','sham','release_delay','pose_bias']:
            subset=[c for c in cases if family=='all' or (family=='sham' and c['duration']==0)
                or (family==c['fault'] and c['duration']>0)]
            bins={'both_success':[],'rescue':[],'harm':[],'both_failure':[]}
            differences={i:[] for i in range(3)}
            for c in subset:
                a=results[c['case_id'],method]['complete']; b=results[c['case_id'],control]['complete']
                key='both_success' if a and b else 'rescue' if a else 'harm' if b else 'both_failure'
                bins[key].append(c['case_id']); differences[c['root']].append(int(a)-int(b))
            means=[float(np.mean(differences[i])) for i in range(3)]
            bootstrap=[float(np.mean([means[i] for i in sample])) for sample in itertools.product(range(3),repeat=3)]
            comparisons.append(dict(method=method,control=control,family=family,n=len(subset),cases=bins,
                delta=float(np.mean(means)),root_cluster_percentile95=np.percentile(bootstrap,[2.5,97.5]).tolist(),
                uncertainty_note='Only three consumed development roots; descriptive, not confirmation'))
    report=dict(status='verified',development_only=True,privileged=True,official_reproductions=False,
        full_action_replay=True,causal_decision_trace_verified=True,forecast_replay_verified=True,
        episodes=list(results.values()),comparisons=comparisons,artifact_sha256=hashes,
        Collision_Rate=None,CVR=None,safe_success=None)
    (run/'verified_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    table=['# Development discovery: all cases','', '| Case | VLA | Hold | Timed | Phase | Verified |','|---|---|---|---|---|---|']
    links=[]
    for c in cases:
        cells=['PASS' if results[c['case_id'],a]['complete'] else 'FAIL' for a in ARMS]
        table.append('| '+c['case_id']+' | '+' | '.join(cells)+' |')
        links.append(f'<li><a href="review/{c["case_id"]}_trace.html">{c["case_id"]}: {" / ".join(cells)}</a></li>')
    (run/'discovery_map.md').write_text('\n'.join(table)+'\n')
    (run/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Development fault discovery</h1><p>All cases, including harms and failures. Columns VLA / hold / timed / phase / verified. Not final-test evidence.</p><ul>'+''.join(links)+'</ul>')
    print(json.dumps(dict(status='verified',episodes=len(results))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('run',type=Path); p.add_argument('--cases',type=Path,required=True)
    a=p.parse_args(); summarize(a.run,a.cases)
