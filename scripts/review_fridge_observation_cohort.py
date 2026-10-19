"""Replay-aligned camera-control evidence panels; no inferred model rationale."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from summarize_vla_failure_audit import load, pair_checks, save

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def panel(run, frame, label):
    rows = run['rows']
    i = min(frame, len(rows))
    canvas = Image.new('RGB', (640, 896), '#121c27')
    d = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 18)
    small = ImageFont.truetype(FONT, 15)
    d.text((16, 8), label, font=font, fill='white')
    d.text((16, 36), f'ACTOR ONLY | model actions {i}/{run["status"]["args"]["steps"]} | source sim {i/20:.2f}s', font=small, fill='#ffd07c')
    canvas.paste(Image.fromarray(run['images'][i][::-1, ::-1]).resize((576, 576)), (32, 64))
    if frame >= len(rows):
        d.rectangle((40, 72, 600, 102), fill='#7b2929')
        d.text((48, 77), 'END FRAME HELD - not added simulation or dwell', font=small, fill='white')
    if i:
        row = rows[i-1]
        trace = row['trace']
        evidence = trace['evidence']
        assert row['action'] == trace['executed_action'] == trace['nominal_action']
        lines = [
            f'Decision for action {row["step"]}; evidence through {trace["evidence_through_step"]}',
            f'1. OpenVLA tokens: {row["tokens"]}',
            f'2. Before: grasp={evidence["grasped"]}; native={evidence["native_success"]}',
            '3. Gate: direct actor control | verification/repair inactive',
            f'4. Execute unmodified model action | XYZ norm={np.linalg.norm(row["action"][:3]):.4f}',
            f'5. AFTER action: grasp={row["grasped"]}; native={row["native_success"]}',
            'Owner: VLA | interventions=0 | actual handoff=N/A',
            f'Inference {row["inference_s"]:.3f}s | cumulative {run["inference_cumulative"][i]:.2f}s',
            'Trace is execution evidence; camera benefit needs paired outcomes.']
    else:
        lines = ['Initial settled observation; no model action executed.',
                 'Shared physics/controller; only fixed camera pose differs.',
                 'No privileged geometry is passed into actor inference.']
    for j, text in enumerate(lines):
        d.text((16, 650+j*24), text, font=small, fill='white' if j<8 else '#ffd07c')
    return canvas


def main(root, out):
    out.mkdir(parents=True, exist_ok=False)
    launch = json.loads((root/'launch.json').read_text())
    assert launch['status'] in ('complete_verified','complete_with_issues')
    report = dict(scope='experiment1_camera_framing_controls_not_framework_efficacy',
        source=str(root), status=launch['status'], pairs=[], videos=[], runs={})
    initial = []
    for seed in range(3):
        names = [f'seed{seed}_{camera}' for camera in ('robot_side','framed_side')]
        runs = [load(root/name) for name in names]
        check = pair_checks(*runs, camera_change=True)
        for name, run in zip(names,runs):
            verification = json.loads((root/f'replay_{name}'/'verification.json').read_text())
            report['runs'][name] = dict(**run['metric'], replay=verification)
        initial.append(dict(seed=seed, fingerprint=runs[0]['rows'][0]['before'],
                            item_position=runs[0]['status']['initial_diagnostics']['item_position']))
        report['pairs'].append(dict(seed=seed, **check))
        path = out/f'seed{seed}_original_left_framed_right.mp4'
        frames = max(len(r['rows']) for r in runs)+1+40
        with imageio.get_writer(path, fps=20, codec='libx264', ffmpeg_params=['-movflags','+faststart']) as writer:
            for frame in range(frames):
                canvas = Image.new('RGB',(1280,944),'#121c27')
                d = ImageDraw.Draw(canvas)
                d.text((16,8),f'EXPERIMENT 1 | seed {seed} | paired camera diagnosis | 1x simulation time, not real-time inference',
                    font=ImageFont.truetype(FONT,18),fill='#ffd07c')
                for side,run in enumerate(runs):
                    canvas.paste(panel(run,frame,['LEFT: original robot-side camera','RIGHT: frozen reframed camera'][side]),(side*640,48))
                writer.append_data(np.asarray(canvas))
        with imageio.get_reader(path) as reader:
            assert reader.count_frames()==frames and reader.get_meta_data()['fps']==20
        report['videos'].append(dict(file=path.name, frames=frames, fps=20,
            terminal_hold_frames=40, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    report['initial_states'] = initial
    report['distinct_initial_physics_hashes'] = len({s['fingerprint']['integration'] for s in initial})
    save(out/'results.json',report)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('out',type=Path)
    a = p.parse_args()
    main(a.root.resolve(),a.out.resolve())
