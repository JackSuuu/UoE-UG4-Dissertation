"""Causal diagnostic overlays + accelerated MP4s of ALL continuation episodes.

Each original20fps frame is retained;2x playback uses40fps. Videos illustrate
different initial states under the SAME policy, never a method comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from adapters.libero_diagnostics import ExecutionDiagnostics

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def panel(frame, index, frame_index, rows, diagnostics, speed, ended=False):
    canvas = Image.new('RGB', (512, 736), '#101722')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 17)
    small = ImageFont.truetype(FONT, 15)
    draw.text((12, 8), f'Root {index} | frozen full7B VLA | {speed:g}x', font=font, fill='white')
    draw.text((12, 34), f'Sim t={frame_index/20:.2f}s | executed steps={frame_index}'
              + (' | ENDED' if ended else ''), font=small, fill='#9db1ce')
    canvas.paste(Image.fromarray(frame).resize((512, 512), Image.Resampling.NEAREST), (0, 64))
    if frame_index:
        row = rows[frame_index - 1]
        cue = diagnostics[frame_index - 1]
        # Source frame j is after action j-1; never label it with future row j.
        draw.text((12, 584), cue['phase'], font=small, fill='#7bdcb5' if row['complete'] else '#ffd08a')
        c = row['conditions']
        draw.text((12, 609), f"Footprint {'OK' if c['footprint_inside'] else 'OUT'} | "
                  f"seated {c['bottom_seated']} | detached {c['released']}", font=small, fill='white')
        draw.text((12, 634), f"Dwell {row['streak']}/20 | margin {cue['signed_footprint_margin_mm']:+.2f}mm | "
                  f"floor {cue['bottom_floor_error_mm']:+.2f}mm", font=small, fill='white')
        draw.text((12, 659), f"Native {row['native_success']} | geometry completion {row['complete']}",
                  font=small, fill='#9db1ce')
    else:
        draw.text((12, 600), 'Initial settled frame; no policy action executed.', font=small, fill='white')
    draw.text((12, 694), 'Diagnostic simulator data | contact safety unmeasured', font=small, fill='#a9b3c2')
    return np.asarray(canvas)


def render(run, speed=2.):
    run = Path(run)
    output = run / 'review'
    output.mkdir(exist_ok=True)
    episodes, audit = [], []
    for i in range(3):
        w = run / f'worker_{i}'
        manifest = json.loads((w / 'manifest.json').read_text())
        assert manifest['status'] == 'complete'
        rows = [json.loads(line) for line in (w / 'steps.jsonl').read_text().splitlines()]
        result = json.loads((w / 'result.json').read_text())
        detector = ExecutionDiagnostics(manifest['bounds'], manifest['floor'])
        cues = [detector.update(row) for row in rows]
        with imageio.get_reader(w / 'rollout.mp4') as source:
            assert abs(source.get_meta_data()['fps'] - 20) < 1e-6
            frames = [frame for frame in source]
        assert len(frames) == len(rows) + 1
        name = f'root_{i}_{speed:g}x.mp4'
        with imageio.get_writer(output / name, fps=20 * speed, codec='libx264',
                               ffmpeg_params=['-movflags', '+faststart']) as writer:
            for j, frame in enumerate(frames):
                writer.append_data(panel(frame, i, j, rows, cues, speed, j == len(rows)))
        timeline = [dict(step=row['step'], **cue) for row, cue in zip(rows, cues)]
        (output / f'root_{i}_diagnostics.json').write_text(json.dumps(timeline, indent=2))
        events = {}
        for key in ['placement_preservation_opportunity', 'recontact_after_placement', 'contact_stall']:
            found = [row['step'] for row in timeline if row[key]]
            events[key] = dict(first_step=found[0] if found else None, intervals=len(found))
        audit.append(dict(root=i, released_placement=result['released_placement'], events=events,
                          final=cues[-1], frames=len(frames), source_fps=20,
                          output_fps=20*speed, video=name))
        episodes.append((frames, rows, cues))
    combined = output / f'all_roots_{speed:g}x.mp4'
    with imageio.get_writer(combined, fps=20*speed, codec='libx264',
                           ffmpeg_params=['-movflags', '+faststart']) as writer:
        for j in range(max(len(e[0]) for e in episodes)):
            panels = []
            for i, (frames, rows, cues) in enumerate(episodes):
                t = min(j, len(frames)-1)
                panels.append(panel(frames[t], i, t, rows, cues, speed, j >= len(frames)-1))
            # Shorter completed runs freeze with ENDED; not extra simulated dwell.
            writer.append_data(np.concatenate(panels, axis=1))
    videos = []
    for path in sorted(output.glob('*.mp4')):
        with imageio.get_reader(path) as reader:
            count = reader.count_frames()
            fps = reader.get_meta_data()['fps']
        expected = (max(len(e[0]) for e in episodes) if path == combined else
                    len(episodes[int(path.name.split('_')[1])][0]))
        assert count == expected and abs(fps-20*speed) < 1e-6
        videos.append(dict(file=path.name, frames=count, fps=fps, seconds=count/fps,
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    report = dict(speed_relative_to_simulation=speed, safety_classified=False,
                  actor_comparison=False, privileged_diagnostics=True, episodes=audit, videos=videos)
    (output / 'review.json').write_text(json.dumps(report, indent=2))
    (output / 'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>OpenVLA execution review</title><style>body{background:#101722;color:white;'
        'font:18px system-ui;margin:24px}video{max-width:100%;width:1400px}a{color:#9bd3ff}</style>'
        '<h1>Three initial states, the same frozen OpenVLA</h1>'
        f'<p>{speed:g}x simulated-time playback. No frames dropped. Not real-time inference or a method comparison.</p>'
        '<p>Geometry/contact overlays use privileged diagnostic logs. Contact safety remains unmeasured.</p>'
        f'<video controls src="{combined.name}"></video>'
        + ''.join(f'<p><a href="root_{i}_{speed:g}x.mp4">Root{i} individual MP4</a></p>' for i in range(3))
        + '<p><a href="review.json">Timing, hashes and diagnostic events</a></p></html>')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--speed', type=float, default=2.)
    args = parser.parse_args()
    if not (0 < args.speed <= 4):
        parser.error('speed must be in (0,4]')
    render(args.run, args.speed)
