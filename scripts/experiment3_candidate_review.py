"""Synchronous overlays sourced solely from actual recorded pre-decision evidence."""
import argparse
import json
from pathlib import Path
import textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

ARMS = ('nominal', 'simple_feedback', 'verification')


def load(root, arm):
    run = root/arm
    return dict(arm=arm, rows=[json.loads(l) for l in (run/'steps.jsonl').read_text().splitlines()],
        images=np.load(run/'images.npz')['agent'], status=json.loads((run/'status.json').read_text()))


FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def panel(data, step):
    rows = data['rows']
    held = step >= len(rows)
    index = min(step, len(rows))
    frame = Image.new('RGB', (640, 930), '#101724')
    draw = ImageDraw.Draw(frame)
    font = ImageFont.truetype(FONT, 16)
    title_font = ImageFont.truetype(FONT, 22)
    title = {'nominal':'FROZEN NOMINAL GREEDY', 'simple_feedback':'MODEL SAMPLES + SIMPLE FEEDBACK',
             'verification':'MODEL SAMPLES + COMMAND CHECKS'}[data['arm']]
    draw.text((12, 8), title, font=title_font, fill='#e8ecff')
    draw.text((12, 42), f'Common simulation clock {step/20:.2f}s | 20 fps', font=font, fill='white')
    image = Image.fromarray(data['images'][index][::-1, ::-1]).resize((512, 512))
    frame.paste(image, (64, 75))
    if held:
        lines = ['HELD FINAL FRAME - NO ADDITIONAL PHYSICS / DWELL',
            f"Ended after {len(rows)} actions: {data['status']['termination']}",
            f"Queries {data['status']['queries']}/600; actions {len(rows)}/600",
            f"Strict complete: {data['status']['strict_complete']}; max dwell {data['status']['max_dwell']}/20",
            'No new decision or handoff during this display hold.']
    else:
        row = rows[step]
        t = row['decision_trace']
        e, m, g = t['evidence'], t['predicates'], t['gate']
        c = e.get('completion', {})
        margins = f"item={m['item_motion_margin_m']:+.4f}m eef={m['eef_motion_margin_m']:+.4f}m"
        choices = ', '.join(f"{v['index']}:{v['source']}" for v in g['candidates'])
        events = ', '.join(v['type'] for v in g['events']) or 'none'
        lines = [f"BEFORE action {step} | evidence through action {step-1}",
            f"Owner: {g['owner']} | chosen: {g['chosen']} | changed: {t['correction']['changed']}",
            f"Trigger={m['trigger']} stalled={m['stalled']} native_unfinished={m['unfinished_native']}",
            f"20-step motion threshold margins: {margins}",
            f"Before: grasp={e['grasped']} native={e['native_success']} strict dwell={c.get('streak', 0)}/20",
            f"Sources: {choices} | queries used={row['queries_used']}/600",
            f"Choice: {g['reason']}",
            f"Local used={g['used']}/12; measured rechecks={m['progress_streak']}/3; events: {events}",
            'Future collision/support/progress: UNKNOWN; command heuristics only.']
        if any(v['type'] == 'handoff_executed' for v in row['events']):
            # Explicitly labels the following transition, not a pre-execution observation.
            lines.append('THIS TRANSITION: actual same-step nominal handoff executed (event log).')
    y = 601
    for line in lines:
        for part in textwrap.wrap(line, width=73):
            draw.text((12, y), part, font=font, fill='#ffe1a6' if held else '#e1e7f4')
            y += 22
    draw.text((12, 905), 'Observable rule trace; not internal reasoning or causal proof.', font=font, fill='#abb6cc')
    return np.asarray(frame)


def main(root):
    data = {arm:load(root, arm) for arm in ARMS}
    metrics = {}
    for arm, d in data.items():
        q = [json.loads(l) for l in (root/arm/'queries.jsonl').read_text().splitlines()]
        metrics[arm] = dict(**{k:d['status'][k] for k in ('actions', 'queries', 'termination', 'any_grasp',
            'any_native', 'strict_complete', 'max_dwell', 'first_native', 'interventions', 'local_entries', 'actual_handoffs', 'episode_wall_s')},
            inference_s_total=sum(x['model_inference_s'] for x in q),
            query_wall_s_total=sum(x['query_wall_s'] for x in q),
            query_wall_s_median=float(np.median([x['query_wall_s'] for x in q])),
            sampled_queries=sum(x['source']=='sample' for x in q),
            replay=json.loads((root/f'replay_{arm}'/'verification.json').read_text()))
    metrics['initial_fingerprints_equal'] = len({json.dumps(d['status']['initial'], sort_keys=True) for d in data.values()}) == 1
    (root/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    for right in ('verification', 'simple_feedback'):
        count = max(len(data['nominal']['rows']), len(data[right]['rows']))+40
        name = root/f'nominal_left_{right}_right.mp4'
        with imageio.get_writer(name, fps=20, codec='libx264', macro_block_size=2, quality=8) as video:
            for step in range(count):
                video.append_data(np.concatenate([panel(data['nominal'], step), panel(data[right], step)], axis=1))
        (root/f'{right}_video_manifest.json').write_text(json.dumps(dict(path=str(name), fps=20,
            frames=count, panel_width=640, height=930, left='nominal', right=right,
            frame_rule='pre-action image and same-step saved pre-decision trace; final image explicitly held',
            final_hold_frames=40, inference_clock='simulation paused during inference'), indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    main(p.parse_args().root)
