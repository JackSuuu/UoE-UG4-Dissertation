"""Read-only frozen-runner replay diagnostics; every invocation has a new output.

The frozen replay function and its exact comparisons remain authoritative. Probe
mode subclasses ReplayTape only to save evidence after the original comparison.
Cross-device runs require an explicit flag and cannot certify original replay.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
COHORT = Path('/home/ami/scratch/native_long_l1_20261019_a')
OUTPUT = Path('/home/ami/scratch/native_long_replay_diagnosis_20261019')
GPU_UUID = 'GPU-3f4f72ef-ed7f-2ddb-e454-abcc6c0330b0'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def leaves(value, prefix=''):
    if isinstance(value, dict):
        return {p: v for key, child in value.items()
                for p, v in leaves(child, prefix + ('.' if prefix else '') + key).items()}
    return {prefix: value}


def pixel_delta(expected, actual):
    import numpy as np
    delta = actual.astype(np.int16) - expected.astype(np.int16)
    absolute = np.abs(delta)
    changed = np.any(delta != 0, axis=-1)
    coords = np.argwhere(changed)
    values, counts = np.unique(delta[delta != 0], return_counts=True)
    return dict(shape=list(actual.shape), dtype=str(actual.dtype),
        changed_pixels=int(changed.sum()), changed_channels=int(np.count_nonzero(delta)),
        max_absolute_channel_delta=int(absolute.max()), mean_absolute_channel_delta=float(absolute.mean()),
        rmse=float(np.sqrt(np.mean(delta.astype(float)**2))),
        signed_nonzero_delta_histogram={str(int(v)): int(n) for v, n in zip(values, counts)},
        bbox_yx_inclusive=[coords.min(0).tolist(), coords.max(0).tolist()] if len(coords) else None,
        changed_pixel_details=[dict(y=int(y), x=int(x), expected=expected[y, x].tolist(),
            actual=actual[y, x].tolist(), delta=delta[y, x].tolist()) for y, x in coords])


def cameras(env):
    m, d = env.sim.model, env.sim.data
    return {name: dict(id=int(m.camera_name2id(name)),
        body_id=int(m.cam_bodyid[m.camera_name2id(name)]),
        local_position=m.cam_pos[m.camera_name2id(name)].tolist(),
        local_quaternion=m.cam_quat[m.camera_name2id(name)].tolist(),
        fovy=float(m.cam_fovy[m.camera_name2id(name)]),
        world_position=d.cam_xpos[m.camera_name2id(name)].tolist(),
        world_rotation=d.cam_xmat[m.camera_name2id(name)].tolist())
        for name in env.env.camera_names}


def run(args):
    import numpy as np
    from PIL import Image
    episode = args.episode.resolve()
    out = args.out.resolve()
    diagnostic_source = Path(__file__).read_bytes()
    if not out.is_relative_to(OUTPUT):
        raise ValueError(f'Output must be below {OUTPUT}')
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frozen = episode/'source'
    sys.path[:0] = [str(frozen/'scripts'), str(frozen/'src')]
    runner = importlib.import_module('run_native_long_episode')
    assert Path(runner.__file__).resolve() == frozen/'scripts/run_native_long_episode.py'
    assert runner.REPO == frozen
    manifest = json.loads((episode/'source_manifest.json').read_text())
    for name, entry in manifest['repository'].items():
        assert sha(frozen/name) == entry['sha256'], name
    status = json.loads((episode/'status.json').read_text())
    original_gpu = status['args']['gpu']
    if original_gpu != 1 and not args.cross_device:
        raise ValueError('Only GPU1 is authorized; other recorded lanes require --cross-device')
    os.environ.update(CUDA_VISIBLE_DEVICES=GPU_UUID, MUJOCO_EGL_DEVICE_ID='2',
                      PYTHONDONTWRITEBYTECODE='1')
    proof = dict(actor_constructor_attempts=0, actor_query_attempts=0,
        worker_spawn_attempts=0, model_weight_open_attempts=0, subprocesses=[])

    class ForbiddenActor:
        def __new__(cls, *a, **kw):
            proof['actor_constructor_attempts'] += 1
            raise RuntimeError('Actor construction prohibited in this diagnostic')

        def act(self, *a, **kw):
            proof['actor_query_attempts'] += 1
            raise RuntimeError('Model queries prohibited in this diagnostic')

    runner.PinnedActor = ForbiddenActor
    import diagnose_vla_failure
    diagnose_vla_failure.PinnedActor = ForbiddenActor

    def audit(event, values):
        if event == 'subprocess.Popen':
            command = str(values[1])
            proof['subprocesses'].append(command)
            if 'openvla_tf440_worker' in command:
                proof['worker_spawn_attempts'] += 1
                raise RuntimeError('Model worker prohibited')
        if event == 'open' and isinstance(values[0], (str, bytes)):
            path = os.fsdecode(values[0])
            if path.endswith('.safetensors') or ('huggingface' in path and path.endswith('.bin')):
                proof['model_weight_open_attempts'] += 1
                raise RuntimeError('Model weight access prohibited')

    sys.addaudithook(audit)
    modifications = ['fail-closed actor/weight/worker guards']
    if original_gpu != 1:
        original_environment = runner.local_environment

        def local_environment(original, destination):
            original.gpu, original.egl_device = 1, 2
            return original_environment(original, destination)

        runner.local_environment = local_environment
        modifications.append('explicit cross-device local_environment lane override to GPU1/EGL2')
    prior = json.loads((COHORT/('replay_' + episode.name)/'verification.json').read_text())
    targets = {(row['phase'], row['frame']) for row in prior['mismatches'] if 'frame' in row}
    original_targets = targets.copy()
    sensor_records = []
    cursor = [None, None]
    if args.sensor_rerenders:
        for verification in OUTPUT.glob('*/verification.json'):
            previous_report = json.loads(verification.read_text())
            if previous_report['source'] == str(episode):
                targets.update((row['phase'], row['frame']) for row in previous_report['mismatches'] if 'frame' in row)
        original_make_scene = runner.make_scene

        def make_scene(original, destination, base):
            env, obs = original_make_scene(original, destination, base)
            original_render = env.sim.render
            from robosuite import macros
            from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
            convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]

            def render(*a, **kw):
                result = original_render(*a, **kw)
                phase, last_index = cursor
                if phase is None or (phase, last_index+1) not in targets:
                    return result
                index = last_index+1
                camera = kw.get('camera_name')
                key = camera+'_image'
                directory = out/'sensor_time'/f'{phase}_{index:06d}'/camera
                directory.mkdir(parents=True, exist_ok=False)
                first = result[::convention].copy()
                Image.fromarray(first).save(directory/'first_raw.png')
                original_directory = episode/'initialization' if phase == 'initialization' else episode
                with np.load(original_directory/'frames'/f'{index:06d}.npz', allow_pickle=False) as frame:
                    expected_image = frame[key]
                Image.fromarray(expected_image).save(directory/'original_tape_raw.png')
                before = runner.fingerprint(env, {})
                before_cameras = cameras(env)
                render_arrays = {}
                for label, obj, names in (
                    ('model', env.sim.model._model, ('geom_rgba', 'site_rgba', 'mat_rgba',
                        'cam_pos', 'cam_quat', 'cam_fovy', 'light_pos', 'light_dir', 'light_diffuse')),
                    ('data', env.sim.data._data, ('geom_xpos', 'geom_xmat', 'site_xpos',
                        'site_xmat', 'cam_xpos', 'cam_xmat', 'light_xpos', 'light_xdir'))):
                    for name in names:
                        render_arrays[label+'_'+name] = np.asarray(getattr(obj, name)).copy()
                np.savez_compressed(directory/'render_state.npz', **render_arrays,
                    integration=runner.integration_state(env), state=env.get_sim_state())
                record = dict(phase=phase, frame=index, camera=key,
                    sim_time=float(env.sim.data.time), fingerprint=before, cameras=before_cameras,
                    first_hash=runner.array_sha(first), first_against_original=pixel_delta(expected_image, first),
                    repetitions=[])
                for repetition in range(args.sensor_rerenders):
                    rgb = original_render(*a, **kw)[::convention].copy()
                    Image.fromarray(rgb).save(directory/f'repeat_{repetition:03d}_raw.png')
                    record['repetitions'].append(dict(sha256=runner.array_sha(rgb),
                        against_first=pixel_delta(first, rgb), against_original=pixel_delta(expected_image, rgb)))
                record['fingerprint_unchanged'] = runner.fingerprint(env, {}) == before
                record['cameras_unchanged'] = cameras(env) == before_cameras
                record['render_arrays_unchanged'] = {key: bool(np.array_equal(value,
                    getattr(env.sim.model._model if key.startswith('model_') else env.sim.data._data,
                            key.split('_', 1)[1]))) for key, value in render_arrays.items()}
                assert record['fingerprint_unchanged'] and record['cameras_unchanged'] and all(record['render_arrays_unchanged'].values())
                save(directory/'details.json', record)
                sensor_records.append(record)
                # Return the first original draw, never a replacement or averaged image.
                return result

            env.sim.render = render
            return env, obs

        runner.make_scene = make_scene
        modifications.append(f'{args.sensor_rerenders} repeated direct renders at original sensor sampling instant; return original first draw')
    details, frame_checks = [], []
    original_tape = runner.ReplayTape

    class DiagnosticTape(original_tape):
        def compare(self, index, env, obs, diagnostic, report, phase):
            super().compare(index, env, obs, diagnostic, report, phase)
            cursor[:] = [phase, index]
            expected = self.frames[index]
            fp = runner.fingerprint(env, obs)
            left, right = leaves(expected['fingerprint']), leaves(fp)
            checks = {key: left.get(key) == right.get(key) for key in left.keys() | right.keys()}
            arrays = dict(state=env.get_sim_state(), integration=runner.integration_state(env),
                          **{k: v for k, v in obs.items() if k.endswith('_image')})
            equal = {k: bool(np.array_equal(v[index], arrays[k])) for k, v in self.arrays.items()}
            frame_checks.append(dict(phase=phase, frame=index, arrays_equal=equal,
                fingerprint_equal=checks, diagnostics_equal=diagnostic == expected['diagnostics']))
            mismatch = not all(equal.values()) or not all(checks.values())
            if not mismatch and (phase, index) not in targets:
                return
            directory = out/'evidence'/f'{phase}_{index:06d}'
            directory.mkdir(parents=True)
            np.savez_compressed(directory/'actual.npz', **arrays)
            np.savez_compressed(directory/'original.npz', **{k: v[index] for k, v in self.arrays.items()})
            numeric = {}
            for key, value in vars(env.robots[0].controller).items():
                if isinstance(value, np.ndarray) and value.dtype.kind in 'biuf':
                    numeric[key] = value.tolist()
                elif isinstance(value, (bool, int, float, str)) or value is None:
                    numeric[key] = value
            numeric.update(gripper_current_action=env.robots[0].gripper.current_action.tolist(),
                timestep=env.env.timestep, cur_time=env.env.cur_time, done=bool(env.env.done))
            assert hashlib.sha256(json.dumps(numeric, sort_keys=True).encode()).hexdigest() == fp['controller']
            save(directory/'controller_fingerprint_fields.json', numeric)
            render_state = {}
            for label, obj, names in (
                ('model', env.sim.model._model, ('geom_rgba', 'site_rgba', 'mat_rgba',
                    'cam_pos', 'cam_quat', 'cam_fovy', 'light_pos', 'light_dir', 'light_diffuse')),
                ('data', env.sim.data._data, ('geom_xpos', 'geom_xmat', 'site_xpos',
                    'site_xmat', 'cam_xpos', 'cam_xmat', 'light_xpos', 'light_xdir'))):
                for name in names:
                    render_state[label+'_'+name] = np.asarray(getattr(obj, name)).copy()
            np.savez_compressed(directory/'render_state.npz', **render_state)
            entry = dict(phase=phase, frame=index, mismatch=mismatch,
                originally_reported_mismatch=(phase, index) in original_targets,
                expected_fingerprint=expected['fingerprint'], actual_fingerprint=fp,
                fingerprint_fields_equal=checks, arrays_equal=equal,
                diagnostics_equal=diagnostic == expected['diagnostics'], cameras=cameras(env), images={})
            for key in fp['images']:
                original, actual = self.arrays[key][index], arrays[key]
                Image.fromarray(original).save(directory/(key+'_original_raw.png'))
                Image.fromarray(actual).save(directory/(key+'_actual_raw.png'))
                delta = np.abs(actual.astype(np.int16)-original.astype(np.int16))
                Image.fromarray(np.minimum(delta*32, 255).astype(np.uint8)).save(directory/(key+'_absdiff_x32.png'))
                entry['images'][key] = pixel_delta(original, actual)
            if args.rerenders:
                from robosuite import macros
                from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
                convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
                entry['image_convention'] = macros.IMAGE_CONVENTION
                entry['rerenders'] = {}
                before = runner.integration_state(env).copy()
                for key in fp['images']:
                    camera = key.removesuffix('_image')
                    results = []
                    for repetition in range(args.rerenders):
                        rgb = env.sim.render(camera_name=camera, width=256, height=256)[::convention].copy()
                        Image.fromarray(rgb).save(directory/f'{key}_rerender_{repetition:02d}_raw.png')
                        results.append(dict(sha256=runner.array_sha(rgb),
                            against_original=pixel_delta(self.arrays[key][index], rgb),
                            against_observation=pixel_delta(arrays[key], rgb)))
                    entry['rerenders'][key] = results
                entry['rerender_integration_unchanged'] = bool(np.array_equal(before, runner.integration_state(env)))
                entry['rerender_fingerprint_unchanged'] = runner.fingerprint(env, obs) == fp
                entry['rerender_cameras_unchanged'] = cameras(env) == entry['cameras']
                assert entry['rerender_integration_unchanged'] and entry['rerender_fingerprint_unchanged']
            context = env.sim._render_context_offscreen
            from OpenGL import GL
            entry['renderer'] = dict(vendor=GL.glGetString(GL.GL_VENDOR).decode(),
                renderer=GL.glGetString(GL.GL_RENDERER).decode(), version=GL.glGetString(GL.GL_VERSION).decode(),
                offsamples=int(context.con.offSamples), scene_flags=context.scn.flags.tolist())
            save(directory/'details.json', entry)
            details.append(entry)

    if args.mode == 'probe':
        runner.ReplayTape = DiagnosticTape
        modifications.append('ReplayTape subclass: original compare first, then evidence capture')
        if args.rerenders:
            modifications.append(f'{args.rerenders} additional direct renders per camera at evidence frames; no forward/step/state install')
    code_hashes = dict(replay=hashlib.sha256(runner.replay.__code__.co_code).hexdigest(),
                       compare=hashlib.sha256(original_tape.compare.__code__.co_code).hexdigest())
    result = runner.replay(argparse.Namespace(replay=episode, out=out,
                          gpu=None if args.cross_device else 1,
                          egl_device=None if args.cross_device else 2))
    imported = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if path and Path(path).is_relative_to(frozen):
            imported[name] = dict(path=path, sha256=sha(path))
    report = json.loads((out/'verification.json').read_text())
    save(out/'zero_query_proof.json', dict(**proof, actor_queries=report['actor_queries'],
        model_calls=0, guard='actor constructors/query entry points, worker spawn and model weights fail closed',
        query_files_created=list(map(str, out.glob('**/queries.jsonl')))))
    (out/'diagnostic_source.py').write_bytes(diagnostic_source)
    save(out/'provenance.json', dict(episode=str(episode), mode=args.mode,
        frozen_import_root=str(frozen), imported_frozen_modules=imported,
        frozen_runner_sha256=sha(runner.__file__), diagnostic_sha256=hashlib.sha256(diagnostic_source).hexdigest(),
        frozen_function_bytecode_sha256=code_hashes, process_local_instrumentation=modifications,
        original_gpu=original_gpu, actual_gpu=1, actual_egl=2, actual_uuid=GPU_UUID,
        strict_original_replay_eligible=original_gpu == 1 and not args.rerenders and not args.sensor_rerenders,
        cross_device=original_gpu != 1, argv=sys.argv, executable=sys.executable,
        previous_verification=prior))
    if args.mode == 'probe':
        save(out/'frame_checks.json', frame_checks)
        save(out/'pixel_diagnosis.json', details)
    if args.sensor_rerenders:
        save(out/'sensor_time_diagnosis.json', sensor_records)
    print(json.dumps(dict(out=str(out), result=result, status=report['status'],
                         evidence_frames=len(details))), flush=True)
    return result


def suite(args):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    jobs = [('two_pots_root0', 'baseline', 0, 'strict_repeat1'),
            ('two_pots_root0', 'baseline', 0, 'strict_repeat2'),
            ('two_pots_root0', 'probe', 0, 'pixels'),
            ('two_pots_root1', 'probe', 0, 'pixels_cross_gpu1'),
            ('two_pots_root2', 'probe', 0, 'pixels_cross_gpu1'),
            ('two_pots_root0', 'probe', 8, 'rerender'),
            ('two_pots_root1', 'probe', 8, 'rerender_cross_gpu1'),
            ('two_pots_root2', 'probe', 8, 'rerender_cross_gpu1')]
    summary = []
    for episode, mode, rerenders, tag in jobs:
        out = OUTPUT/(episode+'_'+tag)
        command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()),
            '--episode', str(COHORT/episode), '--out', str(out), '--mode', mode,
            '--rerenders', str(rerenders)]
        if episode != 'two_pots_root0':
            command.append('--cross-device')
        with (OUTPUT/(out.name+'.log')).open('x') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES=GPU_UUID,
                         MUJOCO_EGL_DEVICE_ID='2'))
        summary.append(dict(episode=episode, mode=mode, rerenders=rerenders,
            output=str(out), returncode=completed.returncode, command=command))
        save(OUTPUT/'suite_progress.json', summary)
        print(json.dumps(summary[-1]), flush=True)
    return 0


def audit_inputs():
    """Verify stored PNG-independent RGB arrays and their recorded fingerprints."""
    import numpy as np
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    for root in range(3):
        episode = COHORT/f'two_pots_root{root}'
        result = dict(episode=str(episode), frames_checked=0, errors=[], files_sha256={})
        for relative in ('status.json', 'source_manifest.json', 'model.xml', 'initial_source.json',
                         'actions.npz', 'frames.jsonl', 'steps.jsonl', 'states.npz', 'images.npz',
                         'initialization/frames.jsonl', 'initialization/steps.jsonl',
                         'initialization/states.npz', 'initialization/images.npz'):
            result['files_sha256'][relative] = sha(episode/relative)
        for phase, directory in (('initialization', episode/'initialization'), ('actor', episode)):
            frames = [json.loads(line) for line in (directory/'frames.jsonl').read_text().splitlines()]
            arrays = {}
            for archive in ('states.npz', 'images.npz'):
                with np.load(directory/archive, allow_pickle=False) as data:
                    arrays.update({key: data[key] for key in data.files})
            for index, row in enumerate(frames):
                with np.load(directory/'frames'/f'{index:06d}.npz', allow_pickle=False) as spool:
                    for key, array in arrays.items():
                        if not np.array_equal(array[index], spool[key]):
                            result['errors'].append([phase, index, key, 'spool_archive_difference'])
                        digest = hashlib.sha256(np.ascontiguousarray(array[index]).tobytes()).hexdigest()
                        expected = (row['fingerprint']['images'][key] if key.endswith('_image')
                                    else row['fingerprint']['integration'] if key == 'integration' else None)
                        if expected is not None and digest != expected:
                            result['errors'].append([phase, index, key, 'fingerprint_difference'])
                result['frames_checked'] += 1
        results.append(result)
    save(OUTPUT/'original_tape_integrity.json', results)
    print(json.dumps([dict(episode=r['episode'], frames=r['frames_checked'], errors=r['errors']) for r in results]))
    return int(any(r['errors'] for r in results))


def summarize():
    runs = []
    for directory in sorted(OUTPUT.iterdir()):
        if not directory.is_dir() or not (directory/'verification.json').is_file():
            continue
        verification = json.loads((directory/'verification.json').read_text())
        provenance = json.loads((directory/'provenance.json').read_text()) if (directory/'provenance.json').is_file() else {}
        entry = dict(name=directory.name, status=verification['status'],
            frames_checked=verification['frames_checked'], actor_queries=verification['actor_queries'],
            strict_original_replay_eligible=provenance.get('strict_original_replay_eligible'),
            cross_device=provenance.get('cross_device'),
            state_gap=verification['max_state_gap'], integration_gap=verification['max_integration_gap'],
            mismatches=verification['mismatches'], images=[])
        if 'error' in verification:
            entry['error'] = verification['error']
        if (directory/'pixel_diagnosis.json').is_file():
            details = json.loads((directory/'pixel_diagnosis.json').read_text())
            for detail in details:
                for key, delta in detail['images'].items():
                    image = dict(phase=detail['phase'], frame=detail['frame'], camera=key,
                        changed_pixels=delta['changed_pixels'], changed_channels=delta['changed_channels'],
                        max_delta=delta['max_absolute_channel_delta'], rmse=delta['rmse'],
                        changed_pixel_details=delta['changed_pixel_details'])
                    if 'rerenders' in detail:
                        repetitions = detail['rerenders'][key]
                        image['rerender_unique_hashes'] = len({r['sha256'] for r in repetitions})
                        image['rerender_original_exact'] = sum(r['against_original']['changed_pixels'] == 0 for r in repetitions)
                        image['rerender_observation_exact'] = sum(r['against_observation']['changed_pixels'] == 0 for r in repetitions)
                        image['rerender_max_delta_against_observation'] = max(r['against_observation']['max_absolute_channel_delta'] for r in repetitions)
                    entry['images'].append(image)
                    if delta['changed_pixels']:
                        from PIL import Image, ImageDraw
                        evidence = directory/'evidence'/f"{detail['phase']}_{detail['frame']:06d}"
                        board = Image.new('RGB', (768, 580), 'white')
                        draw = ImageDraw.Draw(board)
                        first = delta['changed_pixel_details'][0]
                        x, y = first['x'], first['y']
                        x0, y0 = max(0, min(240, x-8)), max(0, min(240, y-8))
                        for column, suffix in enumerate(('original_raw', 'actual_raw', 'absdiff_x32')):
                            rgb = Image.open(evidence/(key+'_'+suffix+'.png'))
                            board.paste(rgb, (column*256, 24))
                            draw.text((column*256+4, 6), suffix, fill='black')
                            crop = rgb.crop((x0, y0, x0+16, y0+16)).resize((256, 256), Image.Resampling.NEAREST)
                            board.paste(crop, (column*256, 302))
                        draw.text((4, 284), f"Raw y={y}, x={x}; RGB {first['expected']} -> {first['actual']}; 16x nearest zoom", fill='black')
                        draw.text((4, 564), f"{directory.name}: {detail['phase']} frame {detail['frame']}; {key}", fill='black')
                        board.save(evidence/(key+'_contact_sheet.png'))
        if (directory/'frame_checks.json').is_file():
            frames = json.loads((directory/'frame_checks.json').read_text())
            entry['nonimage_failures'] = [row for row in frames if not row['diagnostics_equal'] or
                not row['arrays_equal']['state'] or not row['arrays_equal']['integration'] or
                not row['fingerprint_equal']['controller'] or not row['fingerprint_equal']['integration']]
            entry['agentview_mismatch_frames'] = [row['frame'] for row in frames if not row['arrays_equal']['agentview_image']]
        if (directory/'sensor_time_diagnosis.json').is_file():
            entry['sensor_time'] = []
            for record in json.loads((directory/'sensor_time_diagnosis.json').read_text()):
                repeats = record['repetitions']
                entry['sensor_time'].append(dict(frame=record['frame'], camera=record['camera'],
                    first_changed_pixels=record['first_against_original']['changed_pixels'],
                    repeats=len(repeats), unique_hashes=len({record['first_hash'], *[r['sha256'] for r in repeats]}),
                    exact_to_original=sum(r['against_original']['changed_pixels'] == 0 for r in repeats),
                    exact_to_first=sum(r['against_first']['changed_pixels'] == 0 for r in repeats),
                    max_delta_from_first=max(r['against_first']['max_absolute_channel_delta'] for r in repeats),
                    max_pixels_from_first=max(r['against_first']['changed_pixels'] for r in repeats),
                    fingerprint_unchanged=record['fingerprint_unchanged'],
                    render_arrays_unchanged=all(record['render_arrays_unchanged'].values())))
        runs.append(entry)
    save(OUTPUT/'summary.json', runs)
    for run in runs:
        print(json.dumps(dict(name=run['name'], status=run['status'],
            mismatch_frames=[m['frame'] for m in run['mismatches'] if 'frame' in m],
            images=[i for i in run['images'] if i['changed_pixels'] or i.get('rerender_unique_hashes', 1) > 1],
            sensor_time=run.get('sensor_time'))))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--mode', choices=['baseline', 'probe'], default='probe')
    parser.add_argument('--rerenders', type=int, default=0)
    parser.add_argument('--sensor-rerenders', type=int, default=0)
    parser.add_argument('--cross-device', action='store_true')
    parser.add_argument('--suite', action='store_true')
    parser.add_argument('--audit-inputs', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    arguments = parser.parse_args()
    if arguments.sensor_rerenders and (arguments.rerenders or arguments.mode != 'probe'):
        parser.error('--sensor-rerenders requires --mode probe and --rerenders 0')
    if arguments.audit_inputs:
        raise SystemExit(audit_inputs())
    if arguments.summarize:
        raise SystemExit(summarize())
    if not arguments.suite and (arguments.episode is None or arguments.out is None):
        parser.error('--episode and --out are required unless --suite')
    raise SystemExit(suite(arguments) if arguments.suite else run(arguments))
