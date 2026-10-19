"""Offline-only Q1 oracle report and synchronous counterfactual branch videos."""
import argparse
import hashlib
import json
from pathlib import Path
import textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def panel(opportunity,index,data,frame):
    held=frame>=32
    canvas=Image.new('RGB',(512,850),'#111827')
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype(FONT,15)
    title=ImageFont.truetype(FONT,18)
    c=opportunity['candidates'][index]
    chosen=[name for name,d in opportunity['selectors'].items() if d['chosen']==index]
    draw.text((10,8),f"SOURCE STEP {opportunity['step']} / CANDIDATE {index}",font=title,fill='#e8ecff')
    draw.text((10,34),f"{c['source']} | selectors: {', '.join(chosen) or 'neither'}",font=font,fill='#ffd88b')
    canvas.paste(Image.fromarray(data['images'][min(frame,32)][::-1,::-1]).resize((384,384)),(64,64))
    if held:
        e=data['status']['measured_at']['32']
        lines=['HELD FINAL FRAME: no extra physics / dwell',
            '32 actions = candidate +11 greedy +20 greedy tail',
            f"Final active-phase gap reduction={e['directional_margin']:+.7g} {e['units']}",
            f"Useful signal={e['descriptive_useful']} | stage={e['stage_status']}",
            f"Stage entire last20={e['stage_satisfied_entire_last20']}",
            f"Guarded achievement losses={e['guarded_achievement_losses']}",
            f"New item-penetration proxy={e['new_item_penetration_proxy']}",
            'Physical damage / validated contact safety: UNKNOWN',
            'Tail is COUNTERFACTUAL nominal continuation.',
            'No actual online successful repair or handoff claim.']
    else:
        r=data['rows'][frame]
        before_gap=data['rows'][0]['before_measure']['gaps'][opportunity['phase']]
        reduction=before_gap-r['before_measure']['gaps'][opportunity['phase']]
        phase_status=r['before_stages']['predicates'][opportunity['phase']]['status']
        lines=[f"BEFORE branch action {frame+1}/32 | simulated t={frame/20:.2f}s",
            f"Absolute source-clock step={r['absolute_step']}",
            f"Owner: {r['owner']}",
            f"Shared prompt: {r['language']}",
            f"Observed active-phase gap reduction so far={reduction:+.7g}",
            f"Active-entry stage={phase_status}; current prompt phase={r['before_stages']['phase']}",
            f"Door angle before={r['before_state']['angle']:.7g} rad",
            f"Proposed/executed model action: {np.round(r['action'],4).tolist()}",
            f"Fresh calls used={r['fresh_queries_used']}/31; original candidate separately charged",
            'Both selector choices were frozen before ANY branch outcome.',
            'Future outcome UNKNOWN to selectors; offline evaluation only.']
    y=462
    for line in lines:
        for part in textwrap.wrap(line,width=57):
            draw.text((10,y),part,font=font,fill='#ffe1a6' if held else '#e1e7f4')
            y+=20
    draw.text((10,826),'OFFLINE ORACLE: never fed into live repair or training.',font=font,fill='#aebed5')
    return np.asarray(canvas)


def main(root):
    plan=json.loads((root/'plan.json').read_text())
    freeze=json.loads((root/'freeze.json').read_text())
    for rel,h in freeze['sources'].items(): assert hashlib.sha256((root/'source'/rel).read_bytes()).hexdigest()==h
    assert hashlib.sha256((root/'plan.json').read_bytes()).hexdigest()==freeze['plan_sha256']
    result=dict(scope='Q1 OFFLINE oracle only, not causal Q2 or online handoff',
        opportunities=[],branches={},horizons={},videos={},resource_accounting={},
        source_distribution_bias=plan['source_distribution_bias'],harm='physical damage and validated contact safety unknown')
    data={}
    for b in plan['branches']:
        run=root/b['id']
        status=json.loads((run/'status.json').read_text())
        rows=[json.loads(l) for l in (run/'steps.jsonl').read_text().splitlines()]
        queries=[json.loads(l) for l in (run/'queries.jsonl').read_text().splitlines()]
        replay=json.loads((root/f"replay_{b['id']}"/'verification.json').read_text())
        prefix=json.loads((run/'prefix_identity.json').read_text())
        assert status['status']=='completed' and replay['status']=='exact_replay_passed' and prefix['status']=='exact_prefix_passed'
        assert len(rows)==status['actions']==32 and len(queries)==status['fresh_queries']==31
        assert rows[0]['action']==b['opportunity']['candidates'][b['candidate_index']]['action']
        for i,r in enumerate(rows):
            assert r['offset']==i and r['action']==r['proposal']['action'] and r['actual_online_handoff'] is False
            assert r['frozen_selectors']==b['opportunity']['selectors']
            if i: assert r['proposal']==queries[i-1] and r['proposal']['source']=='greedy'
        result['branches'][b['id']]=dict(actions=32,fresh_queries=31,
            source_step=b['opportunity']['step'],candidate_index=b['candidate_index'],
            measured_at=status['measured_at'],model_inference_s=status['model_inference_s'],
            query_wall_s=status['query_wall_s'],wall_s=status['wall_s'],
            prefix_exact=True,independent_replay=replay)
        data[b['id']]=dict(status=status,rows=rows,images=np.load(run/'images.npz')['agent'])
    for o in plan['opportunities']:
        report=dict(step=o['step'],phase=o['phase'],duplicate_action_groups=o['duplicate_action_groups'],
            selectors={k:d['chosen'] for k,d in o['selectors'].items()},at={})
        for h in ('1','12','32'):
            outcomes=[data[f"step{o['step']}_candidate{i}"]['status']['measured_at'][h] for i in range(len(o['candidates']))]
            useful=[i for i,e in enumerate(outcomes) if e['descriptive_useful']]
            best=max(e['directional_margin'] for e in outcomes)
            rejected=[i for i in useful if not all(o['selectors']['verification']['candidates'][i]['checks'][k]
                for k in ('command_contract','command_continuity','local_release_consistency'))]
            report['at'][h]=dict(useful_candidate_indices=useful,useful_available=bool(useful),
                directional_signals=[i for i,e in enumerate(outcomes) if e['descriptive_directional_signal']],
                stage_satisfied_indices=[i for i,e in enumerate(outcomes) if e['stage_satisfied']],
                rejected_useful_candidates=rejected,oracle_best_directional_margin=best,
                selected={name:dict(index=d['chosen'],useful=outcomes[d['chosen']]['descriptive_useful'],
                    measured_margin=outcomes[d['chosen']]['directional_margin'],
                    offline_only_margin_regret=best-outcomes[d['chosen']]['directional_margin'])
                    for name,d in o['selectors'].items()})
        result['opportunities'].append(report)
    for h in ('1','12','32'):
        available=[o['at'][h] for o in result['opportunities'] if o['at'][h]['useful_available']]
        result['horizons'][h]=dict(opportunities=len(plan['opportunities']),useful_available_opportunities=len(available),
            useful_candidate_count=sum(len(o['at'][h]['useful_candidate_indices']) for o in result['opportunities']),
            stage_satisfied_candidate_count=sum(len(o['at'][h]['stage_satisfied_indices']) for o in result['opportunities']),
            selected_useful_rate_conditional_on_available={name:sum(o['selected'][name]['useful'] for o in available)/len(available)
                if available else None for name in ('simple_feedback','verification')},
            interpretation='cannot assess useful selection; no useful candidates available, not equivalence evidence' if not available else
                'descriptive conditional selection comparison on source-verification states; not causal Q2')
    n=len(plan['branches'])
    result['resource_accounting']=dict(new_branch_actions=n*32,fresh_model_queries=n*31,
        original_source_arm_queries=plan['original_source_arm_queries'],
        original_selected_candidate_generation_calls=plan['original_selected_candidate_generation_calls'],
        original_selected_candidate_generation_seconds=plan['original_candidate_generation_seconds'],
        prefix_replay_actions=plan['planned_prefix_replay_actions'],
        independent_replay_prefix_actions=plan['planned_independent_replay_prefix_actions'],
        independent_replay_branch_actions=n*32,standard_setup_actions=plan['planned_setup_actions'],
        fresh_model_inference_s=sum(b['model_inference_s'] for b in result['branches'].values()),
        fresh_query_wall_s=sum(b['query_wall_s'] for b in result['branches'].values()),
        exact_branch_action_replays=n*32,source_candidate_calls_are_not_new_queries=True)
    assert n*32<=288 and n*31<=279
    with imageio.get_writer(root/'all_opportunities_candidates_offline.mp4',fps=20,codec='libx264',macro_block_size=2,quality=8) as combined:
        for o in plan['opportunities']:
            path=root/f"step{o['step']}_candidates_offline.mp4"
            with imageio.get_writer(path,fps=20,codec='libx264',macro_block_size=2,quality=8) as video:
                for frame in range(72):
                    image=np.concatenate([panel(o,i,data[f"step{o['step']}_candidate{i}"],frame)
                        for i in range(len(o['candidates']))],axis=1)
                    video.append_data(image)
                    combined.append_data(image)
                    if frame in (0,12,71): imageio.imwrite(root/f"step{o['step']}_frame{frame}.png",image)
            with imageio.get_reader(path) as reader:
                assert reader.count_frames()==72 and reader.get_meta_data()['fps']==20
            result['videos'][o['id']]=dict(path=str(path),fps=20,frames=72,held_final_frames=40,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    result['status']='completed_exact_verified'
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    main(p.parse_args().root)
