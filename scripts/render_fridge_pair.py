"""Same-clock actor-left / intervention-right review with explicit evidence scope."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load(run):
    status = json.loads((run/'status.json').read_text())
    assert status['status'] == 'completed', run
    rows = [json.loads(line) for line in (run/'steps.jsonl').read_text().splitlines()]
    assert rows and len(rows) == status['steps_executed']
    return status, rows


def main(left, right, out, scope):
    ls, lr = load(left)
    rs, rr = load(right)
    assert ls['args']['mode'] == 'actor' and not ls.get('hybrid'), 'Left must be baseline VLA'
    if scope == 'framework':
        assert rs.get('hybrid') is True, 'Right must be an actual framework run'
        for key in ('steps', 'snapshot', 'prompt'):
            assert ls['args'][key] == rs['args'][key], key
        assert ls.get('comparison_contract') and ls['comparison_contract'] == rs.get('comparison_contract'), \
            'Formal comparisons require a shared frozen observation/budget/termination contract'
    else:
        assert rs['args']['mode'] == 'servo', 'Engineering scope requires explicitly scripted servo'
    scene_keys = ('seed', 'door', 'yaw', 'fridge_y', 'camera', 'open_angle', 'item_forward', 'item_lateral')
    for key in scene_keys:
        assert ls['args'][key] == rs['args'][key], key
    assert (left/'scene.bddl').read_bytes() == (right/'scene.bddl').read_bytes()
    assert lr[0]['before'] == rr[0]['before'], 'Initial controller/physics/images differ'
    with np.load(left/'states.npz') as a, np.load(right/'states.npz') as b:
        assert np.array_equal(a['states'][0], b['states'][0])
    # This schema comes from the 20Hz R1 runner; do not infer wall-time playback.
    fps = 20
    out.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    title = 'ENGINEERING COMPARISON: different controllers and budgets; not framework efficacy' if scope == 'engineering' else 'MATCHED VLA / FRAMEWORK - development, privileged feedback'
    labels = ['BASELINE: FROZEN VLA' + (' + shared phase prompts' if scope == 'framework' else ''), 'OUR GEOMETRIC SERVO (scripted; no VLA)' if scope == 'engineering' else 'OUR VLA + VERIFICATION / REPAIR']
    runs = [(ls, lr), (rs, rr)]
    with imageio.get_reader(left/'raw.mp4') as lvideo, imageio.get_reader(right/'raw.mp4') as rvideo, \
            imageio.get_writer(out, fps=fps) as writer:
        readers = [iter(lvideo), iter(rvideo)]
        frames = [None, None]
        for i in range(max(len(lr), len(rr))+1):
            canvas = Image.new('RGB', (1280, 800), (20, 24, 30))
            draw = ImageDraw.Draw(canvas)
            draw.text((16, 8), title, font=font, fill='#ffce73')
            draw.text((16, 36), f'Same initial physics / controller / images | simulation 1x | common t={i/fps:.2f}s', font=small, fill='white')
            for side, (status, rows) in enumerate(runs):
                x = side*640
                if i <= len(rows):
                    frames[side] = next(readers[side])
                canvas.paste(Image.fromarray(frames[side]).resize((576, 576)), (x+32, 96))
                draw.text((x+16, 70), labels[side], font=font, fill='white')
                ended = i >= len(rows)
                row = rows[min(max(i-1, 0), len(rows)-1)]
                phase = 'initial' if i == 0 else row['phase']
                text = f"{min(i,len(rows))}/{len(rows)} steps | phase: {phase}"
                draw.text((x+16, 682), text, font=small, fill='white')
                grasp = 'initial' if i == 0 else str(row['grasped'])
                draw.text((x+16, 706), f'Grasp: {grasp} | source sim t={min(i,len(rows))/fps:.2f}s', font=small, fill='white')
                if ended:
                    draw.rectangle((x+40, 104, x+590, 134), fill=(90, 25, 25))
                    draw.text((x+48, 108), f'END FRAME HELD - run ended at {len(rows)/fps:.2f}s', font=small, fill='white')
                wall = status['ended']-status['started']
                endpoint = 'full task unscored' if scope == 'engineering' else f"dev full-task endpoint: {status.get('full_completion', False)}"
                draw.text((x+16, 730), f'Wall incl. setup: {wall:.1f}s | {endpoint}', font=small, fill='#ffce73')
                if scope == 'framework':
                    draw.text((x+16, 754), f"Owner: {row.get('owner')} | event: {row.get('event') or '-'}", font=small, fill='white')
                    draw.text((x+16, 778), f"Corrections: {row.get('applied_corrections', 0)} | next phase: {row.get('next_phase')}", font=small, fill='white')
            writer.append_data(np.asarray(canvas))
        for reader in readers:
            assert next(reader, None) is None, 'Unexpected extra source frames'
    manifest = dict(scope=scope, left=str(left), right=str(right), fps=fps,
        same_initial_fingerprint=True, same_initial_state=True,
        left_steps=len(lr), right_steps=len(rr), equal_step_budget=ls['args']['steps']==rs['args']['steps'],
        playback='common simulation clock 1x; shorter run holds explicitly labeled end frame',
        full_task_success_scored=scope == 'framework', endpoint_status='development_only',
        source_hashes={str(run/name): hashlib.sha256((run/name).read_bytes()).hexdigest()
                      for run in (left, right) for name in ('status.json', 'steps.jsonl', 'raw.mp4')},
        video_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    out.with_suffix('.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('left', type=Path)
    p.add_argument('right', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--scope', choices=['engineering', 'framework'], required=True)
    a = p.parse_args()
    main(a.left.resolve(), a.right.resolve(), a.out.resolve(), a.scope)
