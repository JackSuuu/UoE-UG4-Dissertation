"""Versioned cached-replay review: all six failures, paired movies and annotation audit.

Read-only with respect to original rollout/replay folders. No inference or physics.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def image_hash(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def load(root, name, camera_hash, verifier_hash):
    path = root/name
    status = json.loads((path/'status.json').read_text())
    rows = [json.loads(line) for line in (path/'steps.jsonl').read_text().splitlines()]
    verification_path = root/f'cached_replay_{name}/verification.json'
    replay = json.loads(verification_path.read_text())
    assert replay['status'] == 'exact_replay_passed' and replay['steps'] == 600
    assert replay['max_state_gap'] == 0 and not replay['mismatches']
    assert replay['source'] == str(path) and replay['verifier_sha256'] == verifier_hash
    assert status['status'] == 'completed' and status['steps_executed'] == len(rows) == 600
    assert status['args']['gpu'] == 1 and status['intervention_actions'] == 0
    assert status['camera_config_sha256'] == digest(path/'camera_config.json') == camera_hash
    with np.load(path/'images.npz') as data:
        images, wrists = data['agent'], data['wrist']
    with np.load(path/'states.npz') as data:
        states = data['states']
    assert len(states) == len(images) == len(wrists) == 601
    assert status['initial'] == rows[0]['before']
    assert status['initial_diagnostics'] == rows[0]['trace']['evidence']
    for i, row in enumerate(rows):
        trace = row['trace']
        assert row['step'] == trace['step'] == i and trace['evidence_through_step'] == i-1
        assert row['owner'] == 'vla' and row['language'] == status['language']
        assert row['action'] == row['proposal'] == trace['nominal_action'] == trace['executed_action']
        assert trace['gate'] == dict(mode='actor_only', reason='execute_unmodified_model_proposal',
                                    repair_active=False, actual_handoff=False)
        assert trace['evidence_use'] == 'logging_only_not_action_selection'
        assert not trace['events'] and trace['verification'] == 'not_applied_actor_control'
        assert len(row['tokens']) == len(row['action']) == 7
        for key, tape in [('agentview_image', images), ('robot0_eye_in_hand_image', wrists)]:
            assert row['before']['images'][key] == image_hash(tape[i])
            assert row['after']['images'][key] == image_hash(tape[i+1])
        if i:
            assert row['before'] == rows[i-1]['after']
            assert trace['evidence'] == {key: rows[i-1][key] for key in trace['evidence']}
    actions = np.array([r['action'] for r in rows])
    inference = np.array([r['inference_s'] for r in rows])
    item = np.array([status['initial_diagnostics']['item_position']]+[r['item_position'] for r in rows])
    eef = np.array([status['initial_diagnostics']['eef_position']]+[r['eef_position'] for r in rows])
    metric = dict(status=status['status'], queries=len(rows), actions=len(rows),
        termination=status['termination'], intervention_actions=0,
        grasp_steps=sum(r['grasped'] for r in rows), any_grasp=any(r['grasped'] for r in rows),
        first_grasp_step=next((r['step'] for r in rows if r['grasped']), None),
        native_steps=sum(r['native_success'] for r in rows), any_native=any(r['native_success'] for r in rows),
        first_native_step=next((r['step'] for r in rows if r['native_success']), None),
        final_native=rows[-1]['native_success'], strict_full_completion='unscored',
        near_zero_xyz_commands=int((np.linalg.norm(actions[:, :3], axis=1)<.01).sum()),
        unique_token_sequences=len({tuple(r['tokens']) for r in rows}),
        max_item_displacement_m=float(np.linalg.norm(item-item[0], axis=1).max()),
        eef_endpoint_displacement_m=float(np.linalg.norm(eef[-1]-eef[0])),
        max_eef_displacement_m=float(np.linalg.norm(eef-eef[0], axis=1).max()),
        item_hand_endpoint_contact_steps=sum(any('cream_cheese' in str(c) and 'gripper' in str(c)
            for c in r['contacts_endpoint']) for r in rows),
        action_min=actions.min(axis=0).tolist(), action_max=actions.max(axis=0).tolist(),
        inference_median_s=float(np.median(inference)), inference_p95_s=float(np.percentile(inference, 95)),
        inference_total_s=float(inference.sum()), wall_including_setup_s=status['ended']-status['started'],
        rollout_last_logged_elapsed_s=rows[-1]['episode_elapsed_s'], simulation_action_seconds=30.,
        replay_report_path=str(verification_path), replay_report_sha256=digest(verification_path), replay=replay)
    assert metric['any_grasp'] == status['any_grasp'] and metric['any_native'] == status['any_native_success']
    xml = ET.parse(path/'model.xml')
    camera = next(e for e in xml.iter('camera') if e.get('name') == 'agentview')
    camera_attributes = dict(camera.attrib)
    fridge = next(e for e in xml.iter('body') if e.get('name') == 'short_fridge_1_main')
    camera.attrib.pop('pos')
    return dict(path=path, status=status, rows=rows, images=images, initial_wrist=wrists[0],
        initial_state=states[0], metric=metric, inference_cumulative=np.r_[0., np.cumsum(inference)],
        model_without_camera_position=ET.tostring(xml.getroot()), camera=camera_attributes,
        fridge_body=dict(fridge.attrib))


def panel(run, frame, side, font):
    i = min(frame, 600)
    row = run['rows'][i-1] if i else None
    canvas = Image.new('RGB', (640, 944), '#121c27')
    draw = ImageDraw.Draw(canvas)
    lines = [f'{side}: '+('original robot-side camera' if side == 'LEFT' else 'frozen reframed camera'),
        f'ACTOR ONLY | executed {i}/600 | source simulation {i/20:.2f}s']
    canvas.paste(Image.fromarray(run['images'][i][::-1, ::-1]).resize((576, 576)), (32, 64))
    if frame > 600:
        draw.rectangle((40, 72, 600, 102), fill='#7b2929')
        draw.text((48, 77), 'END HOLD: no additional physics or dwell', font=font, fill='white')
    body = [f'Frame {i}: after zero-based action {i-1}' if i else 'INITIAL: before first actor query']
    if row:
        evidence = row['trace']['evidence']
        body += [f'Pre-action evidence through {i-2}: grasp={evidence["grasped"]}; native={evidence["native_success"]}',
            f'OpenVLA tokens: {row["tokens"]}',
            'Gate: direct model proposal | repair inactive | handoff N/A',
            f'AFTER: grasp={row["grasped"]}; native={row["native_success"]}',
            f'XYZ command norm={np.linalg.norm(row["action"][:3]):.4f} | inference {row["inference_s"]:.3f}s']
    else:
        body += ['Initial settled state; no post-action outcome yet.',
            'Only fixed camera position differs within this seed pair.',
            'RGB + constant language enter the frozen actor.',
            'Logged geometry is not an actor input.', 'Twenty setup settling calls precede frame zero.']
    body += [f'Owner VLA | interventions 0 | cumulative inference {run["inference_cumulative"][i]:.2f}s',
        'Strict full retrieval endpoint: UNSCORED in this grasp diagnostic',
        f'STOP: {run["metric"]["termination"]}' if i == 600 else 'Budget: 600 actions; native done or wall cap can stop early',
        'Cached exact replay: 600 steps | gap 0 | mismatches 0',
        'Execution evidence, not hidden reasoning or a causal skill claim.']
    for j, text in enumerate(lines):
        assert draw.textbbox((0, 0), text, font=font)[2] <= 608, text
        draw.text((16, 8+j*28), text, font=font, fill='#ffd07c')
    for j, text in enumerate(body):
        assert draw.textbbox((0, 0), text, font=font)[2] <= 608, text
        assert 650+j*24+20 < 944
        draw.text((16, 650+j*24), text, font=font, fill='white')
    annotation = dict(frame=frame, source_frame=i, after_action=i-1 if i else None,
        evidence_through_action=i-2 if i else None, end_hold=frame>600,
        source_rgb_sha256=image_hash(run['images'][i]), header=lines, text=body)
    return canvas, annotation


def main(root, finalization, out):
    started = time.time()
    out.mkdir(parents=True, exist_ok=False)
    frozen = out/'source'
    frozen.mkdir()
    (frozen/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    replay_manifest = json.loads((finalization/'replay_finalization.json').read_text())
    assert replay_manifest['status'] == 'six_cached_exact_replays_validated'
    camera_hash = digest(root/'camera_config.json')
    camera_config = json.loads((root/'camera_config.json').read_text())
    launch = json.loads((root/'launch.json').read_text())
    assert camera_hash == launch['camera_config_sha256']
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    report = dict(status='started', started=started, scope='experiment1_fixed_camera_actor_only_diagnosis',
        original_launch_status=launch['status'], original_timeout_count=6,
        replay_finalization=str(finalization/'replay_finalization.json'),
        replay_finalization_sha256=digest(finalization/'replay_finalization.json'),
        renderer_sha256=digest(Path(__file__)), camera_config=camera_config, camera_config_sha256=camera_hash,
        near_zero_definition='L2 XYZ controller command <0.01; not physical displacement',
        strict_endpoint='unscored; no full retrieval monitor in this cohort',
        displacement_reference='pre-query settled initial observation', runs={}, pairs=[], videos=[], initial_states=[])
    annotation_audit = dict(status='started', total_action_annotations=0, total_initial_annotations=0,
        total_held_annotations=0, checked='all frame mappings, trace timing, ownership, actions, RGB hashes and text bounds',
        replay='cached verifier e4944c7; original timeouts preserved', samples=[])
    sheet = Image.new('RGB', (1152, 400*6), '#121c27')
    sheet_draw = ImageDraw.Draw(sheet)
    actor_build = None
    for seed in range(3):
        names = [f'seed{seed}_{c}' for c in ('robot_side', 'framed_side')]
        runs = [load(root, name, camera_hash, replay_manifest['verifier_sha256']) for name in names]
        a, b = runs
        assert np.array_equal(a['initial_state'], b['initial_state'])
        assert np.array_equal(a['initial_wrist'], b['initial_wrist'])
        assert not np.array_equal(a['images'][0], b['images'][0])
        assert a['status']['initial_diagnostics'] == b['status']['initial_diagnostics']
        for key in ('integration', 'controller'):
            assert a['status']['initial'][key] == b['status']['initial'][key]
        assert a['model_without_camera_position'] == b['model_without_camera_position']
        assert a['camera']['quat'] == b['camera']['quat']
        assert a['camera'].get('fovy') == b['camera'].get('fovy')
        assert (a['path']/'scene.bddl').read_bytes() == (b['path']/'scene.bddl').read_bytes()
        first = next((i for i, (x, y) in enumerate(zip(a['rows'], b['rows'])) if x['tokens'] != y['tokens']), None)
        report['pairs'].append(dict(seed=seed, initial_state_exact=True, initial_controller_exact=True,
            initial_diagnostics_exact=True, initial_wrist_exact=True, initial_actor_image_different=True,
            model_XML_identical_except_agentview_position=True, cameras=[r['camera'] for r in runs],
            first_token_divergence_step=first, later_differences_confounded_by_trajectory=True))
        report['initial_states'].append(dict(seed=seed, fingerprint=a['status']['initial'],
            item_position=a['status']['initial_diagnostics']['item_position'],
            eef_position=a['status']['initial_diagnostics']['eef_position'],
            door_angle_rad=a['status']['initial_diagnostics']['door_angle_rad'], fridge_body=a['fridge_body']))
        for side, (name, run) in enumerate(zip(names, runs)):
            if actor_build is None:
                actor_build = run['status']['actor_build']
            assert actor_build == run['status']['actor_build']
            report['runs'][name] = run['metric']
            for col, frame in enumerate((0, 300, 600)):
                x, y = col*384, (seed*2+side)*400
                sheet.paste(Image.fromarray(run['images'][frame][::-1, ::-1]).resize((352,352)), (x+16, y+24))
                sheet_draw.text((x+6,y+4), f'{name} | actions {frame}', font=font, fill='white')
                label = 'INITIAL before query' if not frame else f'Grasp {run["rows"][frame-1]["grasped"]}; native {run["rows"][frame-1]["native_success"]}'
                sheet_draw.text((x+6,y+380), label, font=font, fill='white')
        video = out/f'seed{seed}_original_left_framed_right.mp4'
        annotations = video.with_suffix('.annotations.jsonl')
        with imageio.get_writer(video, fps=20, codec='libx264', quality=8,
                ffmpeg_params=['-movflags', '+faststart']) as writer, annotations.open('x') as audit:
            for frame in range(641):
                canvas = Image.new('RGB', (1280, 992), '#121c27')
                draw = ImageDraw.Draw(canvas)
                draw.text((16,8), f'EXPERIMENT 1 | seed {seed} | camera-only diagnosis | 20 fps simulation; inference pauses physics', font=font, fill='#ffd07c')
                for side, run in enumerate(runs):
                    image, annotation = panel(run, frame, ('LEFT', 'RIGHT')[side], font)
                    canvas.paste(image, (640*side,48))
                    audit.write(json.dumps(dict(run=names[side], **annotation))+'\n')
                    key = 'total_initial_annotations' if frame == 0 else 'total_action_annotations' if frame <= 600 else 'total_held_annotations'
                    annotation_audit[key] += 1
                writer.append_data(np.asarray(canvas))
        with imageio.get_reader(video) as reader:
            count, meta = reader.count_frames(), reader.get_meta_data()
            assert count == 641 and meta['fps'] == 20 and meta['size'] == (1280,992)
            for frame in (0, 300, 600, 640):
                sample = out/f'seed{seed}_decoded_frame{frame}.png'
                imageio.imwrite(sample, reader.get_data(frame))
                annotation_audit['samples'].append(dict(file=sample.name, frame=frame, sha256=digest(sample)))
        report['videos'].append(dict(file=str(video), sha256=digest(video), frames=count, fps=20,
            size=list(meta['size']), duration=meta['duration'], source_frames=601, additional_hold_frames=40,
            annotations=str(annotations), annotations_sha256=digest(annotations),
            sources={str(run['path']/file):digest(run['path']/file) for run in runs
                     for file in ('status.json', 'steps.jsonl', 'images.npz', 'states.npz', 'model.xml')}))
    initial = report['initial_states']
    report['seed_diversity'] = dict(distinct_initial_physics_hashes=len({s['fingerprint']['integration'] for s in initial}),
        distinct_controller_hashes=len({s['fingerprint']['controller'] for s in initial}),
        eef_identical=all(s['eef_position'] == initial[0]['eef_position'] for s in initial),
        door_angle_identical=all(s['door_angle_rad'] == initial[0]['door_angle_rad'] for s in initial),
        item_pairwise_distances_m={f'{a["seed"]}-{b["seed"]}':float(np.linalg.norm(np.array(a['item_position'])-b['item_position']))
                                  for a,b in itertools.combinations(initial,2)},
        geometry_note='Small fixture/item translations; same robot initial pose, door angle and fixed camera configuration.')
    for side in (0,1):
        assert all(p['cameras'][side] == report['pairs'][0]['cameras'][side] for p in report['pairs'])
    report['actor_build'] = actor_build
    metrics = list(report['runs'].values())
    report['totals'] = dict(actions=sum(m['actions'] for m in metrics), queries=sum(m['queries'] for m in metrics),
        grasp_runs=sum(m['any_grasp'] for m in metrics), native_runs=sum(m['any_native'] for m in metrics),
        inference_s=sum(m['inference_total_s'] for m in metrics),
        rollout_status_wall_s=sum(m['wall_including_setup_s'] for m in metrics),
        original_supervisor_wall_s=launch['ended']-launch['started'], original_replay_timeout_budget_s=6*300,
        cached_replay_elapsed_s=sum(m['replay']['elapsed_s'] for m in metrics))
    assert annotation_audit['total_action_annotations'] == 3600
    assert annotation_audit['total_initial_annotations'] == 6 and annotation_audit['total_held_annotations'] == 240
    original = json.loads((finalization/'historical_files_before.json').read_text())
    report['historical_file_changes'] = [name for name,h in original.items() if digest(root/name) != h]
    assert not report['historical_file_changes']
    annotation_audit['status'] = 'all_programmatic_annotation_checks_passed'
    report.update(status='complete_cached_replay_verified', ended=time.time())
    sheet.save(out/'all_runs_contact_sheet.png')
    save(out/'annotation_audit.json', annotation_audit)
    save(out/'results.json', report)
    print(json.dumps(dict(status=report['status'], totals=report['totals'], seed_diversity=report['seed_diversity']), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('finalization', type=Path)
    p.add_argument('out', type=Path)
    a = p.parse_args()
    main(a.root.resolve(), a.finalization.resolve(), a.out.resolve())
