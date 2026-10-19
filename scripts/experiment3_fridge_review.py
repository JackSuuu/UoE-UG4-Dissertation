"""Recorded fridge stage/decision overlays and post-run provenance accounting."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

ARMS = ('nominal', 'simple_feedback', 'verification')
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def panel(d, step):
    held = step >= len(d['rows'])
    image_index = min(step, len(d['rows']))
    canvas = Image.new('RGB', (640, 1000), '#101724')
    draw = ImageDraw.Draw(canvas)
    font, title = ImageFont.truetype(FONT, 16), ImageFont.truetype(FONT, 21)
    draw.text((12, 8), {'nominal':'SHARED-PHASE FROZEN NOMINAL', 'simple_feedback':'MODEL CANDIDATES + SIMPLE FEEDBACK',
        'verification':'MODEL CANDIDATES + CURRENT CHECKS'}[d['arm']], font=title, fill='#e8ecff')
    draw.text((12, 40), f'Closed-start fridge | simulation {step/20:.2f}s | 20fps', font=font, fill='white')
    canvas.paste(Image.fromarray(d['images'][image_index][::-1, ::-1]).resize((512, 512)), (64, 72))
    if held:
        s = d['status']
        lines = ['HELD FINAL FRAME - NO ADDITIONAL PHYSICS OR DWELL',
            f"Ended: {s['termination']} | actions={s['actions']} calls={s['queries']}",
            f"Full closed-start complete={s['full_closed_start_completion']} max dwell={s['max_complete_dwell']}/20",
            'Historical milestones: '+str(s['final_stages']['milestone_history']),
            'No new evidence, decision or handoff during display hold.']
    else:
        r = d['rows'][step]
        t = r['decision_trace']
        m, g, stages, s = t['predicates'], t['gate'], r['before_stages'], r['before_state']
        statuses = ' '.join(f"{k}:{v['status']}" for k,v in stages['predicates'].items())
        cs = g['candidates'][g['chosen']]['checks']
        lines = [f'BEFORE action {step} | evidence through action {step-1}',
            f"Phase={r['phase']} owner={r['owner']} chosen={g['chosen']} changed={t['correction']['changed']}",
            f"Prompt: {r['language']}",
            statuses,
            f"Trigger={m['trigger']} margins item={m['item_motion_margin_m']:+.4f}m eef={m['eef_motion_margin_m']:+.4f}m door={m['door_motion_margin_rad']:+.4f}rad",
            f"Source: {', '.join(str(c['index'])+':'+c['source'] for c in g['candidates'])} | calls={r['queries_used']}/600",
            f"Chosen checks: contract={cs['command_contract']} continuity={cs['command_continuity']} release={cs['local_release_consistency']}",
            f"Local release support={cs['local_release_support']} | strict dwell={stages['complete_dwell']}/20",
            f"Choice: {g['reason']}",
            f"Local={g['used']}/12 rechecks={m['stage_recheck_streak']}/3 events="+(','.join(e['type'] for e in g['events']) or 'none'),
            'Future collision / support / progress: UNKNOWN.']
        if any(e['type']=='handoff_executed' for e in r['events']):
            lines.append('THIS TRANSITION: same-step nominal handoff actually executed.')
    y = 596
    for line in lines:
        for part in textwrap.wrap(line, width=72):
            draw.text((12, y), part, font=font, fill='#ffe1a6' if held else '#e1e7f4')
            y += 21
    draw.text((12, 974), 'Exploratory integration; actor readiness / formal efficacy unresolved.', font=font, fill='#abb6cc')
    return np.asarray(canvas)


def main(root):
    data, metrics = {}, {}
    freeze = json.loads((root/'freeze.json').read_text())
    for rel, h in freeze['sources'].items():
        assert hashlib.sha256((root/'source'/rel).read_bytes()).hexdigest() == h
    for arm in ARMS:
        run = root/arm
        rows = [json.loads(l) for l in (run/'steps.jsonl').read_text().splitlines()]
        queries = [json.loads(l) for l in (run/'queries.jsonl').read_text().splitlines()]
        s = json.loads((run/'status.json').read_text())
        replay = json.loads((root/f'replay_{arm}'/'verification.json').read_text())
        assert s['status']=='completed' and replay['status']=='exact_replay_passed'
        assert len(queries) == s['queries'] <= 600 and len(rows) == s['actions'] <= 600
        assert [q['query'] for q in queries] == list(range(len(queries)))
        referenced, samples, duplicate, selected_samples = [], 0, 0, 0
        for i,r in enumerate(rows):
            assert r['step'] == i and not r['simulator_done_internal']
            g, cs = r['decision_trace']['gate'], r['candidates']
            assert r['action'] == cs[g['chosen']]['action']
            assert r['decision_trace']['evidence_through_step'] == i-1
            selected_samples += cs[g['chosen']]['source']=='sample'
            for j,c in enumerate(cs):
                assert c == queries[c['query']] and c['seed'] == i*10+j and c['language']==r['language']
                referenced.append(c['query'])
                if j:
                    samples += 1
                    duplicate += c['action']==cs[0]['action']
            if arm=='nominal': assert len(cs)==1 and r['action']==cs[0]['action']
            for e in r['events']:
                if e['type']=='handoff_executed': assert e['actual_action']==r['action']==cs[0]['action']
        assert len(referenced)==len(set(referenced))==len(queries)
        metrics[arm] = dict(**{k:s[k] for k in ('actions', 'queries', 'termination', 'any_grasp', 'any_native',
            'full_closed_start_completion', 'remaining_sequence_complete', 'max_complete_dwell',
            'interventions', 'local_entries', 'actual_handoffs', 'episode_wall_s')},
            milestones=s['final_stages']['milestone_history'], phase_actions=dict(Counter(r['phase'] for r in rows)),
            final_angle_rad=s['final_state']['angle'], max_angle_rad=max(r['after_state']['angle'] for r in rows),
            samples=samples, samples_equal_greedy=duplicate, selected_sample_actions=selected_samples,
            sampled_greedy_duplicate_fraction=duplicate/samples if samples else None,
            inference_s=sum(q['model_inference_s'] for q in queries),
            query_wall_s=sum(q['query_wall_s'] for q in queries),
            query_wall_median_s=float(np.median([q['query_wall_s'] for q in queries])),
            local_exit_reasons=dict(Counter(e['reason'] for r in rows for e in r['events'] if e['type']=='local_exit')),
            active_transaction_censored=rows[-1]['decision_trace']['predicates']['active'],
            actual_handoff_steps=[r['step'] for r in rows if any(e['type']=='handoff_executed' for e in r['events'])],
            stage_events=[e for r in rows for e in r['stage_events']], replay=replay)
        data[arm] = dict(arm=arm, rows=rows, status=s, images=np.load(run/'images.npz')['agent'])
    assert all(d['status']['initial']==data['nominal']['status']['initial'] for d in data.values())
    metrics['equal_initial_physics_controller_images'] = True
    metrics['scope'] = 'closed-start integration exploration, not formal efficacy or actor readiness qualification'
    metrics['videos'] = {}
    for right in ('verification', 'simple_feedback'):
        frames = max(len(data['nominal']['rows']), len(data[right]['rows']))+40
        path = root/f'nominal_left_{right}_right.mp4'
        with imageio.get_writer(path, fps=20, codec='libx264', macro_block_size=2, quality=8) as video:
            for step in range(frames):
                video.append_data(np.concatenate([panel(data['nominal'], step), panel(data[right], step)], axis=1))
        with imageio.get_reader(path) as reader:
            assert reader.count_frames()==frames and reader.get_meta_data()['fps']==20
        metrics['videos'][right] = dict(path=str(path), frames=frames, fps=20,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), held_final_extra_frames=40,
            semantics='all pre-action images + same-step saved trace; explicit shorter-run final holds')
    (root/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    main(p.parse_args().root)
