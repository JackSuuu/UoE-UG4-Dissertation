"""Default paired MP4 deliverable: same-case VLA left, hybrid right.

One pre-action frame per decision; align by decision/simulated time, never by
independently stretched task phases. Finished panels hold a labeled end frame.
Stream frames, validate source proofs, and list every case before picking highlights.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(folder):
    status = json.loads((folder/'status.json').read_text())
    proof = json.loads((folder/'replay.json').read_text())
    assert proof['status'] == 'exact_replay_passed'
    assert digest(folder/'trace.jsonl') == status['trace_sha256'] == proof['trace_sha256']
    assert not status.get('fixture_only', False)
    return status


def main(root, out, hybrid, publish, control_hz, speed):
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    for folder in sorted(root.glob(f'live_s*_c*_{hybrid}')):
        right = load(folder)
        left_folder = root/f"live_s{right['seed']}_c{right['case']}_vla"
        left = load(left_folder)
        assert left['seed'] == right['seed'] and left['case'] == right['case']
        first = [json.loads((p/'trace.jsonl').open().readline()) for p in (left_folder, folder)]
        assert first[0]['before'] == first[1]['before'], 'Unmatched initial observations'
        readers = [imageio.get_reader(p/'annotated_2x.mp4') for p in (left_folder, folder)]
        for reader in readers:
            assert reader.get_meta_data()['fps'] == control_hz*speed
        statuses = [left, right]
        counts = [x['steps'] for x in statuses]
        label = ('both_success' if left['complete'] else 'rescue') if right['complete'] else ('harm' if left['complete'] else 'both_failure')
        name = f"s{right['seed']}_c{right['case']}_vla_vs_{hybrid}_{speed:g}x.mp4"
        last, iterators = [None, None], [iter(x) for x in readers]
        written = 0
        try:
            with imageio.get_writer(out/name, fps=control_hz*speed) as writer:
                for step in range(max(counts)):
                    panels = []
                    for index, iterator in enumerate(iterators):
                        held = step >= counts[index]
                        if not held:
                            last[index] = next(iterator)
                        source = Image.fromarray(last[index]).convert('RGB').resize((512, 512))
                        panel = Image.new('RGB', (512, 608), '#10151d')
                        panel.paste(source, (0, 64))
                        draw = ImageDraw.Draw(panel)
                        arm = 'Frozen VLA baseline' if index == 0 else 'Hybrid framework'
                        end = 'PASS' if statuses[index]['complete'] else 'FAIL'
                        draw.text((10, 8), f"{arm} | seed {right['seed']} case {right['case']}", fill='white')
                        draw.text((10, 27), f'decision {step} | sim {step/control_hz:.2f}s | {speed:g}x simulated-time playback', fill='white')
                        draw.text((10, 44), 'END FRAME HELD' if held else 'Running', fill='#ffc857' if held else '#a3d9ff')
                        draw.text((10, 586), f'Final: {end} | steps={counts[index]} | repairs={statuses[index]["changed"]}', fill='white')
                        panels.append(np.asarray(panel))
                    frame = np.concatenate(panels, axis=1)
                    writer.append_data(frame)
                    if step == 0: Image.fromarray(frame).save(out/name.replace('.mp4', '.jpg'))
                    written += 1
            for iterator in iterators:
                assert next(iterator, None) is None, 'Source has unaccounted frames'
            with imageio.get_reader(out/name) as check:
                assert sum(1 for _ in check) == written
        finally:
            for reader in readers: reader.close()
        cases.append(dict(seed=right['seed'], case=right['case'], outcome=label, file=name,
            frames=written, sha256=digest(out/name), vla=left, hybrid=right,
            sources=[str(left_folder/'annotated_2x.mp4'), str(folder/'annotated_2x.mp4')]))
    assert cases, 'No verified matched pairs'
    # First sorted case of each outcome, plus first rescue on prospective A2 roots.
    highlights = []
    for label in ('both_success', 'rescue', 'harm', 'both_failure'):
        entry = next((x for x in cases if x['outcome'] == label), None)
        if entry: highlights.append(entry['file'])
    prospective = next((x for x in cases if x['seed'] >= 3100 and x['outcome'] == 'rescue'), None)
    if prospective and prospective['file'] not in highlights: highlights.append(prospective['file'])
    manifest = dict(source=str(root), hybrid=hybrid, alignment='decision index / simulated time, not wall time',
        control_hz=control_hz, speed=speed, images='pre-action; original endpoint/load labels are post-action',
        disclaimer='Accelerated playback is not evidence of real-time control. Held frames are labeled.',
        highlight_rule='first sorted both-success/rescue/harm/both-failure, plus first prospective A2 rescue',
        highlights=highlights, cases=cases)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    cards = ''.join(f'<h2>{x["seed"]}/c{x["case"]}: {x["outcome"]}</h2><video controls preload="none" width="1024" poster="{x["file"].replace(".mp4", ".jpg")}" src="{x["file"]}"></video>' for x in cases)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>VLA vs hybrid</title><h1>VLA left / hybrid right</h1><p>'+html.escape(manifest['disclaimer'])+'</p>'+cards)
    if publish:
        publish.mkdir(parents=True, exist_ok=False)
        for name in highlights: shutil.copy2(out/name, publish/name)
        shutil.copy2(out/'manifest.json', publish/'manifest.json')
        text = '# A2 paired videos: VLA baseline vs hybrid framework\n\n'
        text += 'Left: frozen VLA. Right: hybrid80. 2x simulated-time playback, **not wall-clock real-time**. Same initial state and decision-index alignment; shorter runs hold a labeled end frame. Original image is pre-action and load/endpoint annotation post-action.\n\n'
        text += 'All20 paired cases are indexed in `manifest.json`. Highlights include every observed outcome category, including harm and both-failure; no favorable-only selection. Hybrid prediction-ranking overrides were zero in A2; geometry and hybrid trajectories were identical.\n\n'
        for name in highlights:
            entry = next(x for x in cases if x['file'] == name)
            text += f'- **{entry["outcome"]}**, {entry["seed"]}/c{entry["case"]}: [{name}]({name})\n'
        text += '\nMP4 files use Git LFS. On GitHub use **Download raw file** if no inline playback appears.\n'
        (publish/'README.md').write_text(text)
    print(json.dumps(dict(pairs=len(cases), index=str(out/'index.html'), highlights=highlights)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--hybrid', default='hybrid80')
    p.add_argument('--publish', type=Path)
    p.add_argument('--control-hz', type=float, default=20)
    p.add_argument('--speed', type=float, default=2)
    a = p.parse_args()
    main(a.root, a.out, a.hybrid, a.publish, a.control_hz, a.speed)
