"""Post-hoc descriptive audit and actor-control videos; no policy changes."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def load(path):
    status = json.loads((path/'status.json').read_text())
    rows = [json.loads(line) for line in (path/'steps.jsonl').read_text().splitlines()]
    assert status['status'] == 'completed' and len(rows) == status['steps_executed']
    assert all(r['owner'] == 'vla' and r['action'] == r['proposal'] for r in rows)
    with np.load(path/'images.npz') as data:
        images = data['agent']
    assert len(images) == len(rows)+1
    initial = status.get('initial_diagnostics')
    if initial is None:  # Historical R2 baseline has no pre-action Cartesian log.
        initial = dict(eef_position=rows[0]['eef_position'], item_position=rows[0]['item_position'])
    item = np.array([initial['item_position']]+[r['item_position'] for r in rows])
    eef = np.array([initial['eef_position']]+[r['eef_position'] for r in rows])
    actions = np.array([r['proposal'] for r in rows])
    infer = np.array([r['inference_s'] for r in rows])
    metric = dict(steps=len(rows), termination=status['termination'],
        intervention_actions=0, first_grasp_step=next((r['step'] for r in rows if r['grasped']), None),
        first_native_step=next((r['step'] for r in rows if r['native_success']), None),
        near_zero_xyz_commands=int((np.linalg.norm(actions[:, :3], axis=1)<.01).sum()),
        unique_token_sequences=len({tuple(r['tokens']) for r in rows}),
        max_item_displacement_m=float(np.linalg.norm(item-item[0], axis=1).max()),
        eef_endpoint_displacement_m=float(np.linalg.norm(eef[-1]-eef[0])),
        displacement_reference='initial observation' if 'initial_diagnostics' in status else 'after first action',
        inference_median_s=float(np.median(infer)), inference_p95_s=float(np.percentile(infer, 95)),
        inference_total_s=float(infer.sum()), wall_including_setup_s=status['ended']-status['started'],
        final_native=rows[-1]['native_success'],
        strict_book_complete=any(r['completion']['complete'] for r in rows) if 'completion' in rows[0] else None,
        max_book_dwell_intervals=max(r['completion']['streak'] for r in rows) if 'completion' in rows[0] else None,
        final_book_conditions=rows[-1]['completion']['conditions'] if 'completion' in rows[-1] else None)
    return dict(path=path, status=status, rows=rows, images=images, metric=metric,
                inference_cumulative=np.r_[0., np.cumsum(infer)])


def pair_checks(left, right, camera_change=False):
    a, b = left['rows'], right['rows']
    fa, fb = a[0]['before'], b[0]['before']
    assert fa['integration'] == fb['integration'] and fa['controller'] == fb['controller']
    with np.load(left['path']/'states.npz') as x, np.load(right['path']/'states.npz') as y:
        assert np.array_equal(x['states'][0], y['states'][0])
    assert left['status']['args']['steps'] == right['status']['args']['steps'] == 600
    assert (left['path']/'scene.bddl').read_bytes() == (right['path']/'scene.bddl').read_bytes()
    if not camera_change:
        assert fa == fb and np.array_equal(left['images'][0], right['images'][0])
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x['tokens'] != y['tokens']), None)
    return dict(same_initial_physics_controller=True, same_initial_images=fa['images']==fb['images'],
                same_600_query_action_cap=True, camera_intervention=camera_change,
                first_token_divergence_step=first,
                same_observation_at_first_token_divergence=a[first]['before']==b[first]['before'] if first is not None else None,
                note='After divergence, later token differences also reflect different states.')


FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def panel(run, index, label):
    rows = run['rows']
    i = min(index, len(rows))
    row = rows[i-1] if i else None
    canvas = Image.new('RGB', (640, 800), '#141a22')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 18)
    small = ImageFont.truetype(FONT, 16)
    draw.text((16, 10), label, font=font, fill='white')
    draw.text((16, 38), 'ACTOR ONLY | interventions=0 | handoff=N/A', font=small, fill='#ffce73')
    canvas.paste(Image.fromarray(run['images'][i][::-1, ::-1]).resize((576, 576)), (32, 68))
    if index >= len(rows):
        draw.rectangle((40, 76, 600, 106), fill='#702828')
        draw.text((48, 80), f'END FRAME HELD | stopped at {len(rows)/20:.2f}s', font=small, fill='white')
    state = 'initial' if row is None else 'native predicate reached' if row['native_success'] else 'grasp detected' if row['grasped'] else 'no grasp detected'
    draw.text((16, 654), f'Diagnostic state: {state}', font=small, fill='white')
    draw.text((16, 678), f'Steps {i}/600 cap | source simulated time {i/20:.2f}s', font=small, fill='white')
    draw.text((16, 702), f'Inference accumulated {run["inference_cumulative"][i]:.2f}s | sim playback 1x', font=small, fill='white')
    strict = 'unscored' if 'completion' not in rows[0] else 'not verified'
    draw.text((16, 726), f'Strict stable endpoint: {strict} | native != stable release', font=small, fill='#ffce73')
    draw.text((16, 750), f'Run wall incl. setup: {run["metric"]["wall_including_setup_s"]:.1f}s', font=small, fill='white')
    draw.text((16, 774), f'Stop rule outcome: {run["metric"]["termination"]}', font=small, fill='white')
    return canvas


def render(out, name, runs, labels, note):
    path = out/(name+'.mp4')
    frames = max(len(r['rows']) for r in runs)+1+40
    with imageio.get_writer(path, fps=20, codec='libx264', ffmpeg_params=['-movflags', '+faststart']) as writer:
        for i in range(frames):
            canvas = Image.new('RGB', (640*len(runs), 848), '#141a22')
            draw = ImageDraw.Draw(canvas)
            draw.text((16, 4), 'ACTOR DIAGNOSTICS - no framework arm, no repair efficacy claim',
                      font=ImageFont.truetype(FONT, 16), fill='#ffce73')
            draw.text((16, 26), note, font=ImageFont.truetype(FONT, 15), fill='white')
            for j, (run, label) in enumerate(zip(runs, labels)):
                canvas.paste(panel(run, i, label), (640*j, 48))
            writer.append_data(np.asarray(canvas))
    with imageio.get_reader(path) as reader:
        assert reader.count_frames() == frames and reader.get_meta_data()['fps'] == 20
    result = dict(file=path.name, frames=frames, fps=20, common_simulation_speed=1,
        terminal_hold_extra_frames=40, hold_is_not_simulated_dwell=True,
        labels=labels, runs=[str(r['path']) for r in runs], note=note,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        sources={str(r['path']/f):hashlib.sha256((r['path']/f).read_bytes()).hexdigest()
                 for r in runs for f in ('steps.jsonl', 'status.json', 'images.npz')})
    save(path.with_suffix('.json'), result)
    return result


def main(root, historical, out):
    out.mkdir(parents=True, exist_ok=False)
    launch = json.loads((root/'launch.json').read_text())
    assert launch['status'] == 'complete_verified'
    runs = {}
    for job in launch['jobs']:
        name = job['name']
        replay = json.loads((root/f'replay_{name}'/'verification.json').read_text())
        assert replay['status'] == 'exact_replay_passed' and not replay['mismatches'] and replay['max_state_gap']==0
        runs[name] = load(root/name)
    runs['fridge_current_historical'] = load(historical)
    pairs = [
        ('book_decode', 'book_native_current', 'book_native_pinned', False,
         ['LEFT: Book / native camera / current TF5', 'RIGHT: Book / native camera / official TF4'],
         'Same initial state and images; change inference stack. Equal 600-action cap; stop on native done.'),
        ('book_camera', 'book_native_pinned', 'book_robot_side_pinned', True,
         ['LEFT: Book / native camera / official TF4', 'RIGHT: Book / robot-side camera / official TF4'],
         'Same initial physics; camera intervention includes task visibility loss. Equal 600-action cap.'),
        ('fridge_decode', 'fridge_current_historical', 'fridge_robot_side_pinned', False,
         ['LEFT: Fridge / historical current TF5', 'RIGHT: Fridge / official TF4'],
         'Same initial physics/images and observed instruction; both exhaust 600 actions. No repairs.')]
    report = dict(scope='single_initial_state_actor_diagnosis_not_framework_efficacy',
        near_zero_definition='L2 of XYZ normalized controller commands <0.01; not physical immobility',
        step_indexing='zero-based first-event steps; executed action counts are one-based',
        strict_endpoint_note='Book native termination censors subsequent release/dwell; neither book run verifies stable placement.',
        fridge_comparison_note='Historical R2 also has a full-completion stop and phase prompts; neither activates in the baseline tape. All 600 instructions match.',
        runs={name:r['metric'] for name,r in runs.items()}, pairs={}, videos=[])
    fridge = runs['fridge_robot_side_pinned']
    assert all(r['language']==fridge['status']['language'] for r in runs['fridge_current_historical']['rows'])
    for name, left, right, camera, labels, note in pairs:
        report['pairs'][name] = pair_checks(runs[left], runs[right], camera)
        report['videos'].append(render(out, name, [runs[left], runs[right]], labels, note))
    report['videos'].append(render(out, 'cheese_native', [runs['cheese_native_pinned']],
        ['Cheese + butter / native camera / official TF4'], 'Native task control; no intervention. Stable dwell unscored.'))
    sheet = Image.new('RGB', (256*4, 288*len(runs)), '#141a22')
    draw = ImageDraw.Draw(sheet)
    for i, (name, run) in enumerate(runs.items()):
        n = len(run['rows'])
        for j, index in enumerate([0, n//3, 2*n//3, n]):
            sheet.paste(Image.fromarray(run['images'][index][::-1, ::-1]), (j*256, i*288+32))
            draw.text((j*256+4, i*288+4), f'{name[:24]} | {index}', fill='white')
    sheet.save(out/'contact_sheet.png')
    save(out/'results.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('videos',)}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('historical', type=Path)
    p.add_argument('out', type=Path)
    a = p.parse_args()
    main(a.root.resolve(), a.historical.resolve(), a.out.resolve())
