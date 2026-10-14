"""Independent acceptance, action/forecast/prefix checks and method-labeled video."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ARMS = ('vla', 'hold', 'retreat', 'physics')
NAMES = ('original', 'half_pose', 'hold_grip', 'hold_open', 'up_open', 'up_grip')


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    result = json.loads((directory/'result.json').read_text())
    assert manifest['status'] == 'complete'
    assert result['CVR'] is None and result['safe_success'] is None
    rows = read_rows(directory/'steps.jsonl')
    decisions = {r['step']: r for r in read_rows(directory/'decisions.jsonl')}
    states = np.load(directory/'states.npz')['states']
    assert len(rows) == result['policy_steps'] <= 520
    assert len(states) == len(rows)+1 and np.isfinite(states).all()
    streak = maximum = corrections = opportunities = forecast_steps = replay_steps = 0
    first = first_native = trigger = None
    arm = result['arm']
    with (directory/'substeps.jsonl').open() as f:
        for i, row in enumerate(rows):
            sub = json.loads(next(f))
            assert sub['step'] == row['step'] == i and len(sub['substeps']) == 25
            c = np.asarray(row['book_corners_caddy'])
            contacts = [c for s in sub['substeps'] for c in s['contacts'] if 'black_book' in str(c)]
            penetration = max([max(0., -c['distance']) for c in contacts if 'desk_caddy' in str(c)] or [0.])
            checks = dict(footprint_inside=bool(np.all(c[:, :2] >= manifest['bounds'][0]) and
                                               np.all(c[:, :2] <= manifest['bounds'][1])),
                bottom_seated=bool(abs(c[:, 2].min()-manifest['floor']) <= .005),
                released=not any('gripper' in str(c) for c in contacts),
                supported=any('black_book' in str(c) and 'desk_caddy' in str(c)
                              for c in sub['substeps'][-1]['contacts']),
                shallow_contact=penetration <= .001,
                slow=row['linear_speed'] <= .02 and row['angular_speed'] <= .1)
            assert checks == row['conditions']
            if trigger is None and i and rows[i-1]['candidate']:
                trigger = i
            active = trigger is not None and i < trigger+20 and first is None
            assert row['active'] == active
            native = np.asarray(row['native_action']).copy()
            native[-1] = -np.sign(2*native[-1]-1)
            np.testing.assert_array_equal(native, row['proposal'])
            if active:
                opportunities += 1
                decision = decisions[i]
                expected = np.zeros((6, 7))
                expected[0] = expected[1] = native
                expected[1, :6] *= .5
                previous_grip = rows[i-1]['action'][-1]
                expected[2, -1] = expected[5, -1] = previous_grip
                expected[3, -1] = expected[4, -1] = -1
                expected[4, 2] = expected[5, 2] = .1
                np.testing.assert_array_equal(decision['first_actions'], expected)
                chosen = decision['selected']
                if arm != 'physics':
                    assert chosen == {'vla': 0, 'hold': 3, 'retreat': 4}[arm]
                    assert decision['prediction'] is None
                else:
                    prediction = decision['prediction']
                    assert len(prediction['candidates']) == 6
                    keys = []
                    for k, candidate in enumerate(prediction['candidates']):
                        assert candidate['replay_identity'] and candidate['name'] == NAMES[k]
                        future = candidate['rows']
                        assert len(future) == 5
                        key = [sum(not r['candidate'] for r in future),
                               sum(not r['conditions']['released'] for r in future),
                               sum(not r['conditions']['footprint_inside'] for r in future) +
                               sum(not r['conditions']['bottom_seated'] for r in future),
                               float(np.linalg.norm(expected[k, :6]-native[:6])) +
                               float(expected[k, -1] != native[-1]), k]
                        assert candidate['key'] == key
                        keys.append(key)
                    assert chosen == prediction['selected'] == min(range(6), key=lambda k: keys[k])
                    for key, value in prediction['candidates'][chosen]['rows'][0].items():
                        assert row[key] == value, (i, key)
                    assert prediction['forecast_control_steps'] == 30
                    assert prediction['replay_control_steps'] == 6*(10+i)
                    forecast_steps += 30
                    replay_steps += 6*(10+i)
                np.testing.assert_array_equal(row['action'], expected[chosen])
                assert row['selected'] == NAMES[chosen]
            else:
                assert i not in decisions and row['selected'] == 'original'
                np.testing.assert_array_equal(row['action'], native)
            changed = not np.array_equal(row['action'], native)
            assert row['correction'] == changed
            corrections += int(changed)
            valid = all(checks.values())
            streak = streak+1 if valid else 0
            maximum = max(maximum, streak)
            assert (row['candidate'], row['streak'], row['complete']) == (valid, streak, streak >= 20)
            if first is None and streak >= 20:
                first = i
            if first_native is None and row['native_success']:
                first_native = i
        assert f.readline() == ''
    assert set(decisions) == {r['step'] for r in rows if r['active']}
    assert result['trigger'] == trigger and result['opportunities'] == opportunities <= 20
    assert result['corrections'] == corrections and result['max_streak'] == maximum
    assert result['first_complete_step'] == first and result['released_placement'] == (first is not None)
    assert result['first_native_step'] == first_native and result['native_ever'] == (first_native is not None)
    assert result['VLA_queries'] == len(rows)
    assert result['simulator_forecast_steps'] == forecast_steps <= 600
    assert result['simulator_replay_steps'] == replay_steps
    assert len(rows) == (520 if first is None else min(520, first+41))
    tail = rows[first+1:] if first is not None else []
    assert result['tail_steps'] == len(tail)
    assert result['tail_final_dwell'] == (all(r['candidate'] for r in tail[-20:]) if len(tail) == 40 else None)
    assert result['tail_continuous'] == (all(r['candidate'] for r in tail) if len(tail) == 40 else None)
    assert not any(r['active'] or r['correction'] for r in tail)
    with imageio.get_reader(directory/'rollout.mp4') as reader:
        assert reader.count_frames() == len(states)
        assert reader.get_meta_data()['fps'] == 20
    return result, rows, states


def paired(results, arm, comparator, indices, metric):
    pairs = [(results[i, comparator][metric], results[i, arm][metric]) for i in indices]
    rescue = sum(a is False and b is True for a, b in pairs)
    harm = sum(a is True and b is False for a, b in pairs)
    # Wilson intervals on root-level rescue/harm proportions, descriptive only.
    def wilson(k):
        n, z = len(indices), 1.959963984540054
        p = k/n
        mid = (p+z*z/(2*n))/(1+z*z/n)
        radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        return [mid-radius, mid+radius]
    return dict(arm=arm, comparator=comparator, roots=indices, metric=metric,
                rescue=rescue, harm=harm, paired_delta=(rescue-harm)/len(indices),
                rescue_wilson95=wilson(rescue), harm_wilson95=wilson(harm))


def render(run, index, rows_by_arm):
    output = run/'review'
    output.mkdir(exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    frames = {}
    for arm in ARMS:
        with imageio.get_reader(run/f'root_{index}'/arm/'rollout.mp4') as reader:
            # FFmpeg readers may advertise an effectively unbounded __len__;
            # list(reader) uses that hint to preallocate and can MemoryError.
            frames[arm] = [frame for frame in reader]
    name = f'root_{index}_methods_2x.mp4'
    with imageio.get_writer(output/name, fps=40, codec='libx264',
                           ffmpeg_params=['-movflags','+faststart']) as writer:
        for j in range(max(map(len, frames.values()))):
            panels = []
            for arm in ARMS:
                t = min(j, len(frames[arm])-1)
                canvas = Image.new('RGB', (384, 528), '#101722')
                draw = ImageDraw.Draw(canvas)
                draw.text((8, 8), f'Root {index} | {arm} | 2x sim time', font=font, fill='white')
                draw.text((8, 31), f'Steps {t}' + (' | ENDED' if j >= len(frames[arm])-1 else ''), font=font, fill='#9db1ce')
                canvas.paste(Image.fromarray(frames[arm][t]).resize((384,384)), (0,56))
                if t:
                    row = rows_by_arm[arm][t-1]
                    draw.text((8, 447), f"Dwell {row['streak']}/20 | {row['selected']}", font=font, fill='#ffd08a')
                    draw.text((8, 470), f"Correction {row['correction']} | detached {row['conditions']['released']}", font=font, fill='white')
                draw.text((8, 501), 'Privileged diagnostic | safety unmeasured', font=font, fill='#9db1ce')
                panels.append(np.asarray(canvas))
            writer.append_data(np.concatenate(panels, axis=1))
    with imageio.get_reader(output/name) as reader:
        assert reader.count_frames() == max(map(len, frames.values()))
    return name


def main(run):
    results, hashes, videos = {}, {}, []
    for index in range(6):
        episode_rows, episode_states = {}, {}
        for arm in ARMS:
            directory = run/f'root_{index}'/arm
            result, rows, states = verify(directory)
            assert result['index'] == index and result['arm'] == arm
            results[index, arm], episode_rows[arm], episode_states[arm] = result, rows, states
            for path in directory.iterdir():
                if path.is_file():
                    hashes[str(path.relative_to(run))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for arm in ARMS[1:]:
            prefix = next((i for i, r in enumerate(episode_rows[arm]) if r['correction']), len(episode_rows[arm]))
            reference = episode_rows['vla']
            assert prefix <= len(reference)
            for i in range(prefix):
                assert episode_rows[arm][i]['tokens'] == reference[i]['tokens']
                assert episode_rows[arm][i]['action'] == reference[i]['action']
            np.testing.assert_array_equal(episode_states[arm][:prefix+1], episode_states['vla'][:prefix+1])
        if index < 3:
            historical = Path('/home/ami/scratch/libero_continuation_20261013_a')/f'worker_{index}'
            oldrows = read_rows(historical/'steps.jsonl')
            oldstates = np.load(historical/'states.npz')['states']
            for old, new in zip(oldrows, episode_rows['vla']):
                assert old['tokens'] == new['tokens'] and old['action'] == new['action']
            n = min(len(oldstates), len(episode_states['vla']))
            np.testing.assert_array_equal(oldstates[:n], episode_states['vla'][:n])
        videos.append(render(run, index, episode_rows))
    comparisons = [paired(results, arm, comparator, roots, 'released_placement')
                   for roots in [list(range(3)), list(range(3,6)), list(range(6))]
                   for arm, comparator in [('hold','vla'),('retreat','vla'),('physics','vla'),
                                           ('physics','hold'),('physics','retreat')]]
    report = dict(development_only=True, privileged=True, learned=False, CVR=None, safe_success=None,
        episodes=list(results.values()), comparisons=comparisons, artifact_sha256=hashes, videos=videos)
    (run/'verified_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    (run/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Placement preservation</title>'
        '<style>body{background:#101722;color:white;font:18px system-ui}video{width:100%}a{color:#9bd3ff}</style>'
        '<h1>Placement preservation: privileged mechanism diagnostic</h1>'
        '<p>Frozen VLA / hold-open / up-open / accurate physics. All videos 2x simulated time. No learned efficacy claim.</p>'
        + ''.join(f'<h2>Root {i}</h2><video controls src="review/{name}"></video>' for i,name in enumerate(videos))
        + '<a href="verified_summary.json">Verified outcomes, paired rescue/harm and hashes</a>')
    print(json.dumps(dict(comparisons=comparisons, videos=videos), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    main(p.parse_args().run)
