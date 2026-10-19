"""Render recorded actor-only evidence with explicit ownership and native goals."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render(run, out):
    status = json.loads((run / 'status.json').read_text())
    rows = [json.loads(line) for line in (run / 'steps.jsonl').read_text().splitlines()]
    init = [json.loads(line) for line in (run / 'initialization/steps.jsonl').read_text().splitlines()]
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    frames = []
    for phase, directory, steps in [('initialization', run / 'initialization', init),
                                    ('actor', run, rows)]:
        records = [json.loads(line) for line in (directory / 'frames.jsonl').read_text().splitlines()]
        with np.load(directory / 'images.npz') as images:
            for i, record in enumerate(records):
                # Skip duplicated post-initialization state; retain every action endpoint.
                if phase == 'actor' and i == 0:
                    continue
                canvas = Image.new('RGB', (1024, 800), '#111827')
                draw = ImageDraw.Draw(canvas)
                for camera, x in [('agentview_image', 0), ('robot0_eye_in_hand_image', 512)]:
                    image = Image.fromarray(images[camera][i][::-1, ::-1]).resize((512, 512))
                    canvas.paste(image, (x, 44))
                draw.text((12, 10), f'{run.name} | agent view', font=font, fill='white')
                draw.text((524, 10), 'Wrist view (offline diagnostic only)', font=font, fill='white')
                d = record['diagnostics']
                count = i if phase == 'actor' else 0
                lines = [f'OWNER: {"frozen OpenVLA" if phase == "actor" else "initialization noop"} | '
                         f'model actions {count}/520 | init {10 if phase == "actor" else i}/10',
                         'TRIGGER / CANDIDATE / REPAIR / HANDOFF: none (actor-only baseline)',
                         f'NATIVE FULL GOAL: {d["native_success"]} | privileged evidence below, not actor input']
                lines += [f'{key}: {value}' for key, value in d['predicates'].items()]
                lines.append('GRASP: ' + ', '.join(f'{key}={value}' for key, value in d['grasped'].items()))
                if phase == 'actor' and i == len(records) - 1:
                    lines.append(f'EXIT: {status.get("termination", "see status.json")} | '
                                 f'recorded native success={status["final"]["native_success"]}')
                for line, text in enumerate(lines):
                    draw.text((12, 564 + line * 26), text, font=font, fill='white')
                frames.append(np.asarray(canvas))
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    with imageio.get_writer(out, fps=20, codec='libx264', quality=7) as writer:
        for frame in frames:
            writer.append_data(frame)
    return dict(source=str(run), video=str(out), frames=len(frames), fps=20,
                alignment='one recorded simulator action per frame, plus reset; no holds or discarded failures',
                display='both raw images rotated180degrees; actor crop remains in original policy_input evidence')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = []
    launch = json.loads((args.run / 'launch.json').read_text())
    for job in launch['jobs']:
        run = Path(job['episode_dir'])
        if (run / 'status.json').exists():
            manifest.append(render(run, args.out / (job['name'] + '.mp4')))
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))
