"""Validate a three-root pilot and make a local HTML video report."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    rows = []
    hashes = {}
    for i in range(3):
        worker = args.run / f'worker_{i}'
        manifest = json.loads((worker / 'manifest.json').read_text())
        assert manifest['status'] == 'complete', manifest
        assert manifest['args']['indices'] == [i]
        assert manifest['args']['steps'] == 520
        episode = worker / f'episode_{i:03d}'
        result = json.loads((episode / 'result.json').read_text())
        steps = [json.loads(line) for line in (episode / 'steps.jsonl').read_text().splitlines()]
        assert len(steps) == result['policy_steps']
        assert [s['step'] for s in steps] == list(range(len(steps)))
        assert all(len(s['tokens']) == 7 and np.isfinite(s['action']).all() for s in steps)
        assert steps[-1]['success'] == result['native_success']
        assert not any(s['success'] for s in steps[:-1])
        for step in steps:
            native = np.asarray(step['native_action'])
            expected = native.copy()
            expected[-1] = -np.sign(2 * native[-1] - 1)
            np.testing.assert_array_equal(expected, step['action'])
        with np.load(episode / 'states.npz') as archive:
            assert len(archive['states']) == len(steps) + 1
            assert np.isfinite(archive['states']).all()
        with imageio.get_reader(episode / 'rollout.mp4') as video:
            assert video.count_frames() == len(steps) + 1
        for path in [worker / 'manifest.json', episode / 'result.json', episode / 'steps.jsonl',
                     episode / 'states.npz', episode / 'rollout.mp4', episode / 'model.xml']:
            hashes[str(path.relative_to(args.run))] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(result)
    successes = sum(row['native_success'] for row in rows)
    # Wilson interval is descriptive only for these three fixed development roots.
    n = len(rows)
    p = successes / n
    z = 1.96
    center = (p + z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
    summary = dict(episodes=rows, n=n, native_successes=successes, native_SR=p,
                   descriptive_wilson95=[center-half, center+half],
                   CVR=None, safe_success=None, precision_stable_success=None,
                   feasibility_only=True, artifact_sha256=hashes)
    (args.run / 'verified_summary.json').write_text(json.dumps(summary, indent=2))
    cards = []
    for row in rows:
        i = row['index']
        video = f'worker_{i}/episode_{i:03d}/rollout.mp4'
        cards.append(f'<section><h2>Initial state {i}: {"native success" if row["native_success"] else "not completed"}</h2>'
                     f'<p>{row["policy_steps"]} policy steps</p>'
                     f'<video controls preload="metadata" src="{html.escape(video)}"></video></section>')
    (args.run / 'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>LIBERO book/caddy feasibility</title><style>body{max-width:1000px;margin:40px auto;'
        'font:18px system-ui;background:#101722;color:#edf3ff}section{display:inline-block;margin:12px;'
        'vertical-align:top}video{width:300px}h2{font-size:18px}</style>'
        f'<h1>Local OpenVLA: book into caddy</h1><p>Native completion: {successes}/{n} fixed development roots.</p>'
        '<p>Full 7B · H1/k1 · no verifier · no injected noise. '
        'Native success does not certify release, stable precision or safe contact.</p>'
        + ''.join(cards) + '<p><a href="verified_summary.json">Verified machine-readable results</a></p></html>')
    print(json.dumps({k: v for k, v in summary.items() if k != 'artifact_sha256'}, indent=2))


if __name__ == '__main__':
    main()
