"""User-facing A1 video index and paired failure triptychs; stream video frames."""
import argparse
import html
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def main(root, out):
    out.mkdir(exist_ok=True)
    failures = [(3000, 1), (3000, 3), (3001, 2), (3001, 3)]
    arms = ['vla', 'framework', 'feedback']
    sections = []
    for seed in (3000, 3001):
        for case in range(5):
            cards = []
            for arm in arms:
                folder = root/f'live_s{seed}_c{case}_{arm}'
                status = json.loads((folder/'status.json').read_text())
                relative = '../'+folder.name+'/annotated_2x.mp4'
                cards.append(f'<article><h3>{arm}: {"PASS" if status["complete"] else "FAIL"}</h3>'
                    f'<p>{status["steps"]} steps; {status["changed"]} changed actions</p>'
                    f'<video controls preload="metadata" src="{relative}"></video>'
                    f'<p><a href="{relative}">Open MP4</a></p></article>')
            sections.append(f'<section><h2>seed {seed}, case {case}</h2><div class="row">'+''.join(cards)+'</div></section>')
            if (seed, case) not in failures:
                continue
            readers = [imageio.get_reader(root/f'live_s{seed}_c{case}_{a}'/'annotated_2x.mp4') for a in arms]
            counts = [json.loads((root/f'live_s{seed}_c{case}_{a}'/'status.json').read_text())['steps'] for a in arms]
            statuses = [json.loads((root/f'live_s{seed}_c{case}_{a}'/'status.json').read_text()) for a in arms]
            iterators = [iter(r) for r in readers]
            last = [None]*3
            samples = []
            try:
                with imageio.get_writer(out/f's{seed}_c{case}_comparison_2x.mp4', fps=40) as writer:
                    for step in range(max(counts)):
                        panels = []
                        for index, iterator in enumerate(iterators):
                            if step < counts[index]:
                                last[index] = next(iterator)
                            image = Image.fromarray(last[index]).copy()
                            draw = ImageDraw.Draw(image)
                            draw.rectangle((0, 486, 512, 512), fill='black')
                            label = f'{arms[index]} | '+('COMPLETE' if statuses[index]['complete'] else 'TIMEOUT')
                            if step >= counts[index]:
                                label += ' | END FRAME HELD'
                            draw.text((6, 492), label, fill='white')
                            panels.append(np.asarray(image))
                        frame = np.concatenate(panels, axis=1)
                        writer.append_data(frame)
                        if step in (0, 9, 18, 59, 159):
                            samples.append(frame)
                Image.fromarray(np.concatenate(samples, axis=0)).save(out/f's{seed}_c{case}_contact_sheet.jpg')
            finally:
                for reader in readers:
                    reader.close()
    comparisons = ''.join(f'<p><a href="s{s}_c{c}_comparison_2x.mp4">Failure {s}/c{c}: three-arm comparison (2x)</a></p>' for s, c in failures)
    page = '<!doctype html><meta charset="utf-8"><title>A1 assembly videos</title><style>body{font:16px system-ui;margin:24px;background:#111;color:#eee}a{color:#88c9ff}.row{display:flex;gap:16px}article{width:32%}video{width:100%}section{border-top:1px solid #555;margin-top:32px}</style>'
    page += '<h1>A1 assembly: all 30 videos</h1><p>2x simulated-time playback. Order: VLA / framework / shared-trigger feedback ranking. Images are pre-action; original load/endpoint labels are post-action. Completed shorter clips hold their labeled end frame in comparisons.</p>'
    page += comparisons+''.join(sections)
    (out/'index.html').write_text(page)
    print(out/'index.html')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('out', type=Path)
    a = p.parse_args()
    main(a.root, a.out)
