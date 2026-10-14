"""Check continuation artifacts, prefix identity, dwell accounting and videos."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('--baseline', type=Path,
                   default=Path('/home/ami/scratch/libero_book_pilot_20261013_a'))
    a = p.parse_args()
    results, hashes, cards = [], {}, []
    for index in range(3):
        w = a.run / f'worker_{index}'
        manifest = json.loads((w / 'manifest.json').read_text())
        assert manifest['status'] == 'complete', manifest
        assert manifest['args']['index'] == index
        np.testing.assert_allclose(manifest['bounds'], [[0.00247, -0.06184], [0.06277, 0.06699]], atol=1e-12, rtol=0)
        assert abs(manifest['floor'] - .01381) < 1e-12
        result = json.loads((w / 'result.json').read_text())
        steps = [json.loads(line) for line in (w / 'steps.jsonl').read_text().splitlines()]
        assert len(steps) == result['policy_steps'] <= 520
        streak = maximum = 0
        first_native = first_complete = None
        counts = {k: 0 for k in steps[0]['conditions']}
        with (w / 'substeps.jsonl').open() as physics:
            for i, row in enumerate(steps):
                sub = json.loads(next(physics))
                assert sub['step'] == row['step'] == i
                assert len(sub['substeps']) == 25
                corners = np.asarray(row['book_corners_caddy'])
                assert np.isfinite(corners).all()
                book = [c for s in sub['substeps'] for c in s['contacts'] if 'black_book' in str(c)]
                penetration = max([max(0., -c['distance']) for c in book if 'desk_caddy' in str(c)] or [0.])
                condition = dict(
                    footprint_inside=bool(np.all(corners[:, :2] >= manifest['bounds'][0]) and
                                          np.all(corners[:, :2] <= manifest['bounds'][1])),
                    bottom_seated=bool(abs(corners[:, 2].min() - manifest['floor']) <= .005),
                    released=not any('gripper' in str(c) for c in book),
                    supported=any('black_book' in str(c) and 'desk_caddy' in str(c)
                                  for c in sub['substeps'][-1]['contacts']),
                    shallow_contact=penetration <= .001,
                    slow=row['linear_speed'] <= .02 and row['angular_speed'] <= .1)
                assert condition == row['conditions']
                for k, v in condition.items():
                    counts[k] += int(v)
                candidate = all(condition.values())
                streak = streak + 1 if candidate else 0
                maximum = max(maximum, streak)
                assert (candidate, streak, streak >= 20) == (row['candidate'], row['streak'], row['complete'])
                if row['native_success'] and first_native is None:
                    first_native = i
                if row['complete'] and first_complete is None:
                    first_complete = i
                native = np.asarray(row['native_action'])
                native[-1] = -np.sign(2 * native[-1] - 1)
                np.testing.assert_array_equal(native, row['action'])
            assert physics.readline() == ''
        assert first_native == result['first_native_step']
        assert first_complete == result['first_complete_step']
        assert maximum == result['max_streak']
        assert (first_complete is not None) == result['released_placement']
        assert len(steps) == 520 if first_complete is None else first_complete == len(steps)-1
        old = a.baseline / f'worker_{index}' / f'episode_{index:03d}'
        previous = [json.loads(line) for line in (old / 'steps.jsonl').read_text().splitlines()]
        assert len(steps) >= len(previous)
        for oldrow, newrow in zip(previous, steps):
            assert oldrow['tokens'] == newrow['tokens']
            np.testing.assert_array_equal(oldrow['action'], newrow['action'])
        with np.load(old / 'states.npz') as archive:
            oldstates = archive['states']
        with np.load(w / 'states.npz') as archive:
            states = archive['states']
        assert len(states) == len(steps)+1 and np.isfinite(states).all()
        error = float(np.max(np.abs(states[:len(oldstates)] - oldstates)))
        assert error < 1e-10, error
        with imageio.get_reader(w / 'rollout.mp4') as video:
            assert video.count_frames() == len(states)
        result.update(prefix_max_state_error=error, condition_pass_counts=counts)
        results.append(result)
        for name in ['manifest.json', 'result.json', 'steps.jsonl', 'substeps.jsonl', 'states.npz', 'model.xml', 'rollout.mp4']:
            path = w / name
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(chunk)
            hashes[str(path.relative_to(a.run))] = digest.hexdigest()
        cards.append(f'<section><h2>Root {index}: {"released placement" if result["released_placement"] else "not completed"}</h2>'
                     f'<p>{len(steps)} steps; best dwell {maximum}/20.</p>'
                     f'<video controls preload="metadata" src="worker_{index}/rollout.mp4"></video></section>')
    summary = dict(episodes=results, n=3, native_successes=sum(r['native_ever'] for r in results),
                   released_placements=sum(r['released_placement'] for r in results),
                   CVR=None, safe_success=None, development_only=True, artifact_sha256=hashes)
    (a.run / 'verified_summary.json').write_text(json.dumps(summary, indent=2))
    (a.run / 'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>OpenVLA released-placement continuation</title><style>body{max-width:1100px;margin:30px auto;'
        'font:18px system-ui;background:#101722;color:#edf3ff}section{display:inline-block;margin:10px;'
        'vertical-align:top}video{width:330px}h2{font-size:17px}</style>'
        f'<h1>Released placement: {summary["released_placements"]}/3</h1>'
        '<p>Frozen local full7B VLA · same development roots · H1/k1 · 520step total budget.</p>'
        '<p>Geometry/seating, detachment, slow motion and1s dwell. Contact safety remains unmeasured.</p>'
        + ''.join(cards) + '<p><a href="verified_summary.json">Verified results</a></p></html>')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['episodes','artifact_sha256']},indent=2))
    for row in results:
        print(json.dumps({k:v for k,v in row.items() if k != 'final'},indent=2))
    # Produce an accelerated, annotated MP4 after every verified complete run.
    from render_libero_review import render
    render(a.run, speed=2.)


if __name__ == '__main__':
    main()
