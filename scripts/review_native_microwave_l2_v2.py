"""Render all twelve L2A v2 (g+120) pairs from recorded images and offline labels.

No actor, simulator, or GPU imports. Each video contains the shared initial
boundary and 32 action endpoints at 20 fps. All negative outcomes are retained.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROTOCOL = 'native_microwave_l2_v2_20261019'
SNAPSHOT_DELAY = 120
REFERENCES = [0, 0, 0, 0, 1, 2]
FPS = 20
HORIZON = 32
SIZE = (2048, 1440)
OBJ = 'white_yellow_mug_1'
IN = '["in","white_yellow_mug_1","microwave_1_heating_region"]'
CLOSE = '["close","microwave_1"]'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BG = '#101923'
FG = '#eef4fa'
MUTED = '#b9c8d7'
ACCENT = '#81c9ff'
YELLOW = '#ffcf73'


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def raw_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def inventory(directories):
    return {str(p): [p.stat().st_size, p.stat().st_mtime_ns]
            for directory in directories for p in sorted(directory.rglob('*'))
            if p.is_file()}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def png_bytes(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format='PNG')
    return stream.getvalue()


def label_text(cell, h, endpoint):
    label = cell['audit']['labels'][str(h)]
    if endpoint < h:
        return f'U{h}: pending (offline readout at a{h})', False
    require(label['observed_branch_actions'] == h, 'Label measurement count')
    value = str(label['useful_proxy']).lower()
    text = (f'U{h}={value}; p={label["terminal_level"]}, b={label["snapshot_level"]}'
            f' | OFFLINE @ a{h}')
    if h == 32 and cell['id'] == 'C2_root4':
        text += ' | held-clear retained'
    elif h == 32 and cell['id'] == 'C2_root3':
        text += ' | containment'
    return text, True


def render(snap, left, right, tick):
    image = Image.new('RGB', SIZE, BG)
    draw = ImageDraw.Draw(image)
    overlay = []

    def text(x, y, value, size=24, color=FG, width=1984):
        font = ImageFont.truetype(FONT, size)
        box = draw.textbbox((0, 0), value, font=font)
        require(box[2] <= width, f'Overlay exceeds panel: {value}')
        draw.text((x, y), value, font=font, fill=color)
        overlay.append(value)

    text(32, 15, 'L2A v2 | NOMINAL vs MODEL CANDIDATE | not a full-framework comparison', 34)
    text(32, 62, 'Task: put the yellow and white mug in the microwave and close it', 26)
    context = 'original-success root' if left['original_native_success'] else 'original-failure root'
    text(32, 103, f'ROOT {snap["root"]} | boundary: post-action t={snap["post_action_step"]} (zero-based)'
         f' | state / next action s={snap["state_index"]} | {context}', 27, ACCENT)
    text(32, 143, f'Causal collection: latch first target grasp g={snap["first_grasp_step"]};'
         f' collect at t=g+{SNAPSHOT_DELAY} if not natively terminated.', 24)
    text(32, 177, 'Regardless of current grasp / height / goals / eventual outcome; no timer reset.'
         ' Collection timer, not a failure alarm.', 24, MUTED)
    text(32, 211, 'Selector: NONE | verified repair: NONE | handoff: NONE | all 6 roots / all 12 pairs retained', 24, YELLOW)
    draw.line((1024, 256, 1024, 1380), fill='#4c6070', width=2)
    mapping = []
    for side, cell in enumerate((left, right)):
        x = side * 1024 + 32
        diagnostic = cell['frames'][tick]['diagnostics']
        role = 'NOMINAL greedy control' if side == 0 else 'MODEL sampled candidate'
        text(x, 260, f'{cell["id"]} | {role}', 29, ACCENT, 960)
        if side == 0:
            source = 'Source: greedy | fixed seed 0 | fresh closed-loop calls'
        else:
            source = f'Source: sampled | fixed seed schedule {cell["plan"]["seed"]}+k'
        text(x, 302, source, 24, FG, 960)
        text(x, 337, 'Owner: local_candidate (including greedy control); actor: VLA', 24, FG, 960)
        text(x, 373, f'Budget: {tick}/32 actions | aligned clock +{tick/FPS:.2f}s'
             f' | actual +{tick/FPS:.2f}s', 25, FG, 960)
        post = snap['post_action_step'] + tick
        text(x, 411, f'Post-action t={post} | recorded sim time {diagnostic["sim_time"]:.2f}s'
             f' | {"INITIAL SHARED FRAME" if tick == 0 else "endpoint a" + str(tick)}', 23, MUTED, 960)
        text(x, 449, f'Subgoals: In={int(diagnostic["predicates"][IN])}'
             f'  Close={int(diagnostic["predicates"][CLOSE])}'
             f'  grasp={int(diagnostic["grasped"][OBJ])}'
             f' | native={int(diagnostic["native_success"])}', 28, FG, 960)
        camera = Image.fromarray(cell['images'][tick]).transpose(Image.Transpose.ROTATE_180)
        camera = camera.resize((768, 768), Image.Resampling.NEAREST)
        image.paste(camera, (side*1024 + 128, 492))
        if tick == HORIZON:
            exit_text = 'HORIZON EXIT / 32 actions; native success NOT reached'
        else:
            exit_text = 'LOCAL CANDIDATE ACTIVE / native success not yet reached'
        text(x, 1274, exit_text, 25, YELLOW, 960)
        visible_labels = []
        for y, h in ((1312, 12), (1348, 32)):
            value, visible = label_text(cell, h, tick)
            text(x, y, value, 24, FG if visible else MUTED, 960)
            if visible:
                visible_labels.append(h)
        mapping.append(dict(source_id=cell['id'], recorded_frame_index=tick,
                            actual_actions=tick, budget=32, held=False,
                            source_frame_rgb_sha256=raw_sha(cell['images'][tick]),
                            sim_time=diagnostic['sim_time'], actual_elapsed_sim_s=tick/FPS,
                            visible_offline_labels=visible_labels, exit_overlay=exit_text))
    text(32, 1395, '20 fps = 1 action / 0.05s simulator time, NOT inference wall time | Main camera: rotate180, uncropped'
         ' | U: offline, not online verification', 22, MUTED)
    return image, dict(protocol=PROTOCOL, video_frame=tick, pts_s=tick/FPS, alignment_sim_s=tick/FPS,
                       sides=mapping, overlay=overlay)


def check_video(path, expected_frames):
    reader = imageio.get_reader(path, 'ffmpeg')
    meta = reader.get_meta_data()
    if meta.get('nframes') == float('inf'):
        meta['nframes'] = None
    count = reader.count_frames()
    decoded = sum(1 for _ in reader)
    reader.close()
    require(count == decoded == expected_frames, f'Frame count: {path}')
    require(meta['fps'] == FPS and tuple(meta['size']) == SIZE, 'Video dimensions/fps')
    require(abs(meta['duration'] - expected_frames/FPS) < .011, 'Video duration')
    result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(path),
                             '-map', '0:v:0', '-f', 'framemd5', '-'],
                            capture_output=True, text=True, check=True)
    frame_lines = [line for line in result.stdout.splitlines() if line and not line.startswith('#')]
    require(len(frame_lines) == expected_frames, 'Independent ffmpeg decode count')
    require([int(line.split(',')[2]) for line in frame_lines] == list(range(expected_frames)),
            'Noncontiguous encoded PTS')
    require('#tb 0: 1/20' in result.stdout, 'Encoded time base')
    return dict(imageio_metadata=meta, imageio_count_frames=count, imageio_decoded_frames=decoded,
                ffmpeg_decoded_frames=len(frame_lines), ffmpeg_pts_contiguous=True,
                ffmpeg_time_base='1/20', width=SIZE[0], height=SIZE[1], fps=FPS,
                frames=expected_frames, duration_s=expected_frames/FPS,
                last_endpoint_sim_s=HORIZON/FPS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--snapshots', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run, audit_dir, snapshot_path, out = (p.resolve() for p in
                                         (args.run, args.audit, args.snapshots, args.out))
    require(not out.exists(), 'Output directory must be new')
    protected = [run, audit_dir, snapshot_path.parent]
    before = inventory(protected)
    consumed = {}

    def track(path):
        consumed[str(path)] = sha(path)
        return path

    # Read-only pin proves the prior renderer was preserved; no code is imported.
    old_renderer = Path(__file__).with_name('review_native_microwave_l2.py').resolve()
    track(old_renderer)
    plan = load(track(run/'plan.json'))
    audit = load(track(audit_dir/'summary.json'))
    design = load(track(snapshot_path))
    freeze = load(track(snapshot_path.parent/'design_freeze.json'))
    require(plan['protocol'] == audit['protocol'] == design['protocol'] == PROTOCOL, 'v2 protocol identity')
    require(design['snapshot_rule'] == 'first target grasp post-step g; first nonterminal post-step g+120',
            'v2 causal collection rule')
    require(design['label_contract']['reference_levels'] == REFERENCES and
            design['local_horizon'] == 32 and design['nested_measurements'] == [12, 32], 'v2 label contract')
    require(sha(snapshot_path) == plan['snapshot_manifest_sha256'] ==
            freeze['files'][str(snapshot_path)], 'Frozen snapshot manifest hash')
    require(sha(snapshot_path.parent/'design_freeze.json') == plan['design_freeze_sha256'],
            'Frozen design anchor')
    require(plan['snapshots'] == design['snapshots'], 'Plan/design snapshots')
    require(Path(audit['run']) == run and Path(audit['snapshots']) == snapshot_path, 'Audit source identity')
    require(audit['valid_candidate_count'] == 18 and all(c['passed'] for c in audit['checks'].values()),
            'Complete valid source audit required')
    require(audit['budget']['requested_calls_conservative'] == audit['budget']['actions'] == 576 and
            audit['budget']['accounting_exact'] and audit['gate']['pass_L2B'] is False,
            'Completed v2 budget and negative continuation gate')
    cells = {}
    boundaries = []
    for snap in design['snapshots']:
        root = snap['root']
        require(snap['post_action_step'] == snap['first_grasp_step'] + SNAPSHOT_DELAY and
                snap['state_index'] == snap['post_action_step'] + 1 and
                snap['reason'] == 'first_grasp_plus_120', 'v2 causal boundary')
        for name, expected in snap['source_sha256'].items():
            require(sha(track(Path(snap['source'])/name)) == expected, 'Baseline source pin')
        original_status = load(Path(snap['source'])/'status.json')
        baseline_frame = Path(snap['source'])/'frames'/f'{snap["state_index"]:06d}.npz'
        with np.load(track(baseline_frame), allow_pickle=False) as archive:
            original = archive['agentview_image']
        png_hashes, initial_hashes, initial_pngs = [], [], []
        for candidate in ('C0', 'C1', 'C2'):
            name = f'{candidate}_root{root}'
            directory = run/name
            entry = next(c for c in audit['cells'] if c['root'] == root and c['candidate'] == candidate)
            planned = next(c for c in plan['cells'] if c['name'] == name)
            require(entry['valid'] and entry['directory'] == str(directory), 'Audited source ID')
            status = load(track(directory/'status.json'))
            track(directory/'source_manifest.json')
            frames = list(rows(track(directory/'frames.jsonl')))
            steps = []
            for row in rows(track(directory/'steps.jsonl')):
                row.pop('contacts_substeps', None)
                steps.append(row)
            with np.load(track(directory/'images.npz'), allow_pickle=False) as archive:
                images = archive['agentview_image']
            n = len(steps)
            require(n == entry['actions'] == status['steps_executed'] == status['actor_queries'] == 32 and
                    len(images) == len(frames) == n+1, 'v2 source action/query/frame count')
            require(status['candidate_id'] == name and status['handoff'] is None and
                    not status['repair_verified'], 'Source owner/handoff')
            require(status['termination'] == entry['termination'] == 'action_budget', 'v2 horizon exits')
            require(Path(status['args']['baseline']) == Path(snap['source']) and
                    status['args']['snapshot_step'] == snap['post_action_step'], 'v2 branch boundary identity')
            require(frames[0]['fingerprint'] == snap['fingerprint'] and
                    np.array_equal(images[0], original), 'Original boundary camera/fingerprint differs')
            for k, frame in enumerate(frames):
                require(frame['index'] == k and raw_sha(images[k]) ==
                        frame['fingerprint']['images']['agentview_image'], 'Image fingerprint')
                require(not frame['diagnostics']['native_success'], 'Unexpected early native success in v2')
                require(abs(frame['diagnostics']['sim_time'] - frames[0]['diagnostics']['sim_time'] - k/FPS)
                        < 1e-7, 'Recorded simulator action endpoint timing')
                if k:
                    step = steps[k-1]
                    require(step['branch_step'] == k-1 and step['step'] == snap['state_index']+k-1 and
                            step['before'] == frames[k-1]['fingerprint'] and step['after'] == frame['fingerprint'] and
                            step['diagnostics'] == frame['diagnostics'], 'Action endpoint alignment')
                    require(step['candidate_owner'] == 'local_candidate' and step['owner'] == 'vla' and
                            not step['repair_verified'] and not step['forced_diagnostic_handoff'], 'Local owner')
                    require(step['seed'] == planned['seeds'][k-1] and step['source'] == planned['source'],
                            'Fixed source/seed schedule')
            for h in ('12', '32'):
                require(entry['labels'][h]['snapshot_level'] == REFERENCES[root] and
                        not entry['labels'][h]['censored'], 'v2 frozen reference/measurement')
            png = track(directory/'policy_inputs/000000.png')
            require(sha(png) == entry['boundary']['policy_png_sha256'], 'Audit first PNG')
            png_hashes.append(sha(png))
            initial_hashes.append(raw_sha(images[0]))
            initial_pngs.append(png_bytes(images[0]))
            cells[name] = dict(id=name, directory=str(directory), status=status, frames=frames,
                               original_native_success=original_status['any_native_success'],
                               inference_wall_sum_s=sum(s['inference_s'] for s in steps),
                               images=images, count=n, audit=entry, plan=planned)
        require(len(set(png_hashes)) == len(set(initial_hashes)) == 1 and
                initial_pngs[0] == initial_pngs[1] == initial_pngs[2], 'First same PNG assertion')
        expected_png = next(b['policy_png_sha256'] for b in plan['boundary_inputs'] if b['root'] == root)
        require(png_hashes[0] == expected_png, 'Launch first PNG pin')
        boundaries.append(dict(root=root, policy_png_sha256=png_hashes[0],
                               initial_main_rgb_sha256=initial_hashes[0],
                               initial_main_png_sha256=hashlib.sha256(initial_pngs[0]).hexdigest(),
                               first_same_policy_png=True, first_same_camera_png=True,
                               exact_original_boundary=True, snapshot=snap))
    require(set(cells) == {f'C{c}_root{r}' for r in range(6) for c in range(3)}, 'All 18 identities')
    root4 = cells['C2_root4']['audit']['labels']['32']
    root3 = cells['C2_root3']['audit']['labels']['32']
    require(root4['terminal_level'] == root4['snapshot_level'] == 1 and
            root4['held_clear_dwell_8'] and root4['useful_proxy'] is False,
            'Root4/C2 retained held-clear is not useful progress or recovery')
    require(root3['terminal_level'] == 2 and root3['snapshot_level'] == 0 and
            root3['in_dwell_8'] and root3['useful_proxy'] is True and
            cells['C2_root3']['original_native_success'], 'Root3/C2 containment on original-success root')
    out.mkdir(parents=True)
    (out/'samples').mkdir()
    manifest = dict(protocol=PROTOCOL, version='v2', snapshot_delay_actions=SNAPSHOT_DELAY,
                    scope='L2A v2 nominal versus model candidate; not full framework',
                    renderer=str(Path(__file__).resolve()), renderer_sha256=sha(Path(__file__).resolve()),
                    preserved_v1_renderer=str(old_renderer), preserved_v1_renderer_sha256=consumed[str(old_renderer)],
                    run=str(run), audit=str(audit_dir), design_snapshot_manifest=str(snapshot_path),
                    model_calls=0, simulator_actions=0, recorded_arrays_only=True,
                    recorded_experiment_budget=audit['budget'], continuation_gate=audit['gate'],
                    interpretation={'root4_C2': 'Retained held-clear p32=1 equals reference b=1; U32=false; not recovery.',
                                    'root3_C2': 'Useful containment p32=2, b=0, U32=true on an original-success root; no native success.'},
                    camera=dict(name='agentview_image', source_size=[256, 256], rotate_degrees=180,
                                cropped=False, display_size=[768, 768], resize='nearest'),
                    timing=dict(fps=20, frames=33, initial_frame=True, action_endpoint_dt_s=.05,
                                final_alignment_sim_s=1.6, encoded_duration_s=1.65, early_exits=0, held_frames=0,
                                explanation='PTS k/20 is endpoint k. Last frame has normal 0.05s display duration. '
                                'All v2 branches reached the 32-action horizon. Inference wall time is not used.'),
                    labels=dict(source='frozen v2 offline audit', online_verification=False,
                                reveal='Only after measured horizon; no future-label leakage',
                                contract=design['label_contract']),
                    selector=None, verified_repair=False, handoff=None, boundaries=boundaries,
                    checkpoint_revision=plan['checkpoint_revision'],
                    base_code_revision=plan['base_code_revision'], sampling=plan['sampling'],
                    sources=[dict(source_id=c['id'], directory=c['directory'],
                                  source=c['plan']['source'], seed_schedule=c['plan']['seeds'],
                                  original_native_success=c['original_native_success'],
                                  actual_actions=c['count'], termination=c['status']['termination'],
                                  initial_sim_s=c['frames'][0]['diagnostics']['sim_time'],
                                  final_sim_s=c['frames'][-1]['diagnostics']['sim_time'],
                                  actual_elapsed_sim_s=c['count']/FPS,
                                  recorded_inference_wall_sum_s=c['inference_wall_sum_s'],
                                  inference_wall_used_for_alignment=False,
                                  offline_labels=c['audit']['labels']) for c in cells.values()],
                    videos=[], samples=[])
    for snap in design['snapshots']:
        root = snap['root']
        for candidate in ('C1', 'C2'):
            left, right = cells[f'C0_root{root}'], cells[f'{candidate}_root{root}']
            stem = f'root{root}_C0_vs_{candidate}'
            path = out/f'{stem}.mp4'
            frame_map = []
            with imageio.get_writer(path, fps=FPS, codec='libx264', quality=None, macro_block_size=16,
                                    ffmpeg_params=['-crf', '18', '-preset', 'medium', '-threads', '2',
                                                   '-movflags', '+faststart', '-metadata', f'comment={PROTOCOL}; g+120']) as writer:
                for tick in range(33):
                    frame, mapping = render(snap, left, right, tick)
                    writer.append_data(np.asarray(frame))
                    frame_map.append(mapping)
            write_json(out/f'{stem}.frames.json', frame_map)
            checks = check_video(path, 33)
            reader = imageio.get_reader(path, 'ffmpeg')
            for tick in (0, 12, 32):
                sample = out/'samples'/f'{stem}_f{tick:02d}.png'
                Image.fromarray(reader.get_data(tick)).save(sample)
                manifest['samples'].append(dict(path=str(sample), sha256=sha(sample),
                                                 decoded_video_frame=tick, source_video=str(path)))
            reader.close()
            manifest['videos'].append(dict(path=str(path), sha256=sha(path), root=root, protocol=PROTOCOL,
                source_ids=[left['id'], right['id']], source_directories=[left['directory'], right['directory']],
                counts=[left['count'], right['count']],
                terminations=[left['status']['termination'], right['status']['termination']],
                frame_map=str(out/f'{stem}.frames.json'), frame_map_sha256=sha(out/f'{stem}.frames.json'),
                checks=checks))
            print(json.dumps(dict(video=str(path), checks=checks)), flush=True)
    after = inventory(protected)
    require(before == after, 'Source tree file sizes/mtimes changed during rendering')
    require(all(sha(Path(path)) == digest for path, digest in consumed.items()), 'Consumed source changed')
    manifest['source_sha256_before_and_after'] = consumed
    manifest['source_tree_inventory'] = before
    manifest['checks'] = dict(protocol_v2=True, g_plus_120=True, v2_reference_levels=True,
        all_12_pairs=True, all_6_roots=True, all_18_source_ids=True, original_v1_renderer_unchanged=True,
        first_same_png_all_roots=True, original_boundary_exact=True, fixed_seeds_checked=True,
        all_18_horizon_exits=True, no_synthetic_or_held_frames=True, offline_labels_time_gated=True,
        root4_C2_reference_equal_U_false=True, root3_C2_containment_original_success=True,
        readable_overlay_bounds_checked=True, minimum_font_px=22,
        imageio_and_ffmpeg_counts_pass=True, source_hashes_unchanged=True,
        source_tree_sizes_mtimes_unchanged=True, source_inventory_files=len(before))
    require(len(manifest['videos']) == 12, 'All 12 videos required')
    write_json(out/'manifest.json', manifest)
    write_json(out/'checks.json', manifest['checks'])
    print(json.dumps(dict(out=str(out), videos=12, checks=manifest['checks'])), flush=True)


if __name__ == '__main__':
    main()
