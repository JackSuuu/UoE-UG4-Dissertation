"""Post-run evidence consistency checks; no actor queries or simulator actions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main(root):
    import imageio.v2 as imageio
    manifest = json.loads((root/'launch.json').read_text())
    assert manifest['status'] == 'complete_verified'
    frozen = root/'source'
    for name, expected in manifest['sources'].items():
        assert sha((frozen/name).read_bytes()) == expected, name
    sys.path.insert(0, str(frozen/'src'))
    from adapters.libero_openvla import policy_image
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    sheet = Image.new('RGB', (1152, 1056), '#17202c')
    draw = ImageDraw.Draw(sheet)
    results = []
    tapes = []
    for index in range(3):
        path = root/f'book_init{index}'
        status = json.loads((path/'status.json').read_text())
        replay = json.loads((root/f'replay_init{index}/verification.json').read_text())
        rows = [json.loads(s) for s in (path/'steps.jsonl').read_text().splitlines()]
        images = np.load(path/'images.npz')['agent']
        tapes.append((status, rows, images))
        assert status['queries'] == status['steps_executed'] == len(rows) <= 600
        assert status['args']['init_index'] == index and status['args']['gpu'] == 3
        assert status['intervention_actions'] == 0
        assert replay['status'] == 'exact_replay_passed'
        assert replay['max_state_gap'] == 0 and not replay['mismatches']
        with (path/'contacts.jsonl').open() as contacts:
            for step, row in enumerate(rows):
                contact = json.loads(next(contacts))
                assert contact['step'] == step
                assert sha(json.dumps(contact['samples'], sort_keys=True).encode()) == row['contact_samples_sha256']
                assert row['owner'] == 'vla' and row['action'] == row['proposal']
                assert len(row['tokens']) == len(row['native_action']) == len(row['action']) == 7
                assert sha(images[step].tobytes()) == row['decision_trace']['input_rgb_sha256']
                processed = policy_image(images[step])
                assert sha(np.asarray(processed).tobytes()) == row['decision_trace']['processed_rgb_sha256']
                assert row['decision_trace']['input_language'] == status['language']
                assert row['decision_trace']['hidden_reasoning_available'] is False
                assert row['clock_before']['internal_done'] is False
                if step:
                    assert row['clock_before'] == rows[step-1]['clock_after']
                    assert row['before'] == rows[step-1]['after']
                    assert row['action_after_native_done'] == rows[step-1]['returned_done']
            assert contacts.read() == ''
        native = status['first_native']
        post_native = len(rows)-native-1 if native is not None else 0
        result = dict(index=index, actions=len(rows), queries=status['queries'],
            first_native=native, first_complete=status['first_complete'],
            all_actions_after_first_native=post_native,
            actions_immediately_after_native_done=status['native_done_continuations'],
            any_grasp=status['any_grasp'], first_grasp=status['first_grasp'],
            max_strict_streak=status['max_strict_streak'], termination=status['termination'],
            rollout_wall_s=status['rollout_wall_s'],
            median_inference_s=float(np.median([r['inference_s'] for r in rows])),
            native_at_final=rows[-1]['native_success'],
            condition_true_counts={key:sum(r['completion']['conditions'][key] for r in rows)
                                   for key in rows[0]['completion']['conditions']},
            initial_source=json.loads((path/'initial_source.json').read_text()),
            checks='source_hashes_contact_tapes_RGB_preprocessing_action_identity_contiguous_clocks_exact_replay_passed')
        results.append(result)
        events = [(0, 'INITIAL: before actor query'),
                  (native+1 if native is not None else len(rows),
                   f'FIRST NATIVE after action {native+1}' if native is not None else 'NO NATIVE: final state'),
                  (len(rows), f'FINAL: {status["termination"]}')]
        for col, (frame, label) in enumerate(events):
            x, y = col*384, index*352
            sheet.paste(Image.fromarray(images[frame][::-1, ::-1]).resize((288,288)), (x+48,y+28))
            draw.text((x+5,y+5), f'Init {index} | {label}', font=font, fill='white')
            text = 'owner VLA | setup complete' if frame == 0 else (
                f'owner VLA | native {rows[frame-1]["native_success"]} | strict '
                f'{rows[frame-1]["completion"]["complete"]}')
            draw.text((x+5,y+324), text, font=font, fill='white')
    sheet.save(root/'qualification_endpoint_contact_sheet.png')
    # This postprocessing-only video aligns frame t with the outcome of action t-1;
    # frame zero explicitly precedes the first model query. Original video retained.
    frames = max(len(tape[2]) for tape in tapes)
    with imageio.get_writer(root/'actor_only_qualification_verified.mp4', fps=20,
                           codec='libx264', quality=8, macro_block_size=16) as video:
        for frame in range(frames+40):
            canvas = Image.new('RGB', (1152, 704), '#121721')
            draw = ImageDraw.Draw(canvas)
            draw.text((12,8), 'EXPERIMENT 2: frozen official-TF4 OpenVLA | native-book qualification | no adaptation', font=font)
            draw.text((12,30), 'All source frames at 20 Hz simulation time; inference pauses physics | zero repair / human motion actions', font=font)
            for index, (status, rows, images) in enumerate(tapes):
                f = min(frame, len(images)-1)
                ended = frame >= len(images)-1
                x = index*384
                canvas.paste(Image.fromarray(images[f][::-1, ::-1]).resize((384,384)), (x,64))
                row = rows[f-1] if f else None
                lines = [f'Init {index} | owner VLA | {f}/600 actions']
                if row:
                    checks = row['completion']
                    lines += [f'Native now {row["native_success"]} | strict {checks["complete"]}',
                        f'First native step {status["first_native"]} | first strict {status["first_complete"]}',
                        f'Strict dwell {checks["streak"]}/20 | grasp {row["grasped"]}',
                        f'Inference {row["inference_s"]:.3f}s | sim {row["simulation_s"]:.3f}s',
                        'Failed: '+','.join({'footprint_inside':'foot','bottom_seated':'seat',
                            'released':'release','supported':'support','shallow_contact':'depth',
                            'slow':'slow'}[k] for k,v in checks['conditions'].items() if not v),
                        f'Immediately after done: {row["action_after_native_done"]}']
                else:
                    lines += ['INITIAL state, before first actor query',
                              'No post-action outcome shown for this frame']
                lines += [f'STOP: {status["termination"]}' if ended else 'Model RGB+language -> tokens -> identity action',
                          'END HOLD: no physics or additional dwell' if ended else 'Strict monitor observes; no corrective control']
                for j, line in enumerate(lines):
                    draw.text((x+6,456+j*25), line, fill='#83e1ae' if j==0 else 'white', font=font)
            video.append_data(np.asarray(canvas))
    with imageio.get_reader(root/'actor_only_qualification_verified.mp4') as reader:
        metadata = reader.get_meta_data()
        video = dict(fps=metadata['fps'], size=list(metadata['size']),
                     frames=reader.count_frames(), duration=metadata['duration'],
                     reader='imageio_bundled_ffmpeg')
    assert video['fps'] == 20
    assert video['frames'] == frames+40
    report = dict(status='all_artifact_checks_passed', source_commit=manifest['source_commit'],
        gpu=manifest['gpu'], runs=results, video=video,
        notes=['Replay re-executes model actions, not model inference.',
               'Benchmark init 0 was already used in earlier development.'])
    (root/'artifact_checks.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    main(p.parse_args().root.resolve())
