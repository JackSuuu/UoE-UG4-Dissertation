"""Read-only rollout checks plus contact sheet; no inference or physics steps."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main(root):
    launch = json.loads((root/'launch.json').read_text())
    assert launch['status'] == 'complete_verified'
    for name, expected in launch['sources'].items():
        assert sha((root/'source'/name).read_bytes()) == expected, name
    checkpoint = Path(launch['checkpoint']['path'])
    for name in ('config.json', 'dataset_statistics.json'):
        assert sha((checkpoint/name).read_bytes()) == launch['checkpoint']['files'][name]['sha256']
    config = json.loads((checkpoint/'config.json').read_text())
    stats = json.loads((checkpoint/'dataset_statistics.json').read_text())['libero_90_no_noops']['action']
    bins = np.linspace(-1, 1, config['n_action_bins'])
    centers = (bins[:-1]+bins[1:])/2
    vocab = config['text_config']['vocab_size']-config['pad_to_multiple_of']
    low, high, mask = np.array(stats['q01']), np.array(stats['q99']), np.array(stats['mask'])
    sys.path.insert(0, str(root/'source/src'))
    from adapters.libero_openvla import policy_image
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    sheet = Image.new('RGB', (1152, 800), '#17202c')
    draw = ImageDraw.Draw(sheet)
    results = []
    for index, door in enumerate(('closed', 'open')):
        run = root/door
        status = json.loads((run/'status.json').read_text())
        replay = json.loads((root/f'replay_{door}/verification.json').read_text())
        rows = [json.loads(line) for line in (run/'steps.jsonl').read_text().splitlines()]
        images = np.load(run/'images.npz')['agent']
        states = np.load(run/'states.npz')['states']
        assert status['queries'] == status['steps_executed'] == len(rows) <= 600
        assert len(images) == len(states) == len(rows)+1
        assert status['intervention_actions'] == 0 and status['args']['gpu'] == 3
        assert status['normalization_key'] == status['actor_build']['unnorm_key'] == 'libero_90_no_noops'
        assert replay['status'] == 'exact_replay_passed' and replay['max_state_gap'] == 0
        assert not replay['mismatches']
        with (run/'contacts.jsonl').open() as contacts:
            for step, row in enumerate(rows):
                tape = json.loads(next(contacts))
                assert tape['step'] == row['step'] == step
                assert sha(json.dumps(tape['samples'], sort_keys=True).encode()) == row['contact_samples_sha256']
                assert row['owner'] == 'vla' and row['action'] == row['proposal']
                assert len(row['tokens']) == len(row['action']) == 7
                normalized = centers[np.clip(vocab-np.array(row['tokens'])-1, 0, len(centers)-1)]
                native = np.where(mask, .5*(normalized+1)*(high-low)+low, normalized)
                assert np.array_equal(native, row['native_action']), (door, step, 'native decode')
                native[-1] = -np.sign(2*native[-1]-1)
                assert np.array_equal(native, row['action']), (door, step, 'gripper conversion')
                trace = row['decision_trace']
                assert trace['input_language'] == status['language']
                assert trace['hidden_reasoning_available'] is False
                assert sha(images[step].tobytes()) == trace['input_rgb_sha256']
                assert sha(np.asarray(policy_image(images[step])).tobytes()) == trace['processed_rgb_sha256']
                before, after = row['clock_before'], row['clock_after']
                assert not before['internal_done'] and not after['internal_done']
                assert after['timestep'] == before['timestep']+1
                assert abs(after['sim_time']-before['sim_time']-.05) < 1e-9
                if step:
                    assert before == rows[step-1]['clock_after']
                    assert row['before'] == rows[step-1]['after']
                    assert row['action_after_native_done'] == rows[step-1]['returned_done']
            assert contacts.read() == ''
        actions = np.array([r['action'] for r in rows])
        eef = np.array([r['sensed']['eef'] for r in rows])
        results.append(dict(door=door, actions=len(rows), queries=status['queries'],
            full_completion=status['full_completion'], remaining_completion=status['remaining_completion'],
            any_native=status['any_native'], any_grasp=status['any_grasp'],
            any_door_contact=any(r['sensed']['door_contact'] for r in rows),
            any_item_hand_contact=any(r['sensed']['hand_contact'] for r in rows),
            max_strict_streak=max(r['task']['completion_streak'] for r in rows),
            unique_token_sequences=len({tuple(r['tokens']) for r in rows}),
            action_min=actions.min(axis=0).tolist(), action_max=actions.max(axis=0).tolist(),
            eef_axis_range=np.ptp(eef, axis=0).tolist(),
            rollout_wall_s=status['rollout_wall_s'], termination=status['termination'],
            checks='source_contacts_tokens_named_stats_RGB_action_identity_clocks_exact_replay_passed'))
        for col, frame in enumerate((0, len(rows)//2, len(rows))):
            x, y = col*384, index*400
            sheet.paste(Image.fromarray(images[frame][::-1, ::-1]).resize((352, 352)), (x+16, y+24))
            draw.text((x+6, y+4), f'{door} | {frame} actions | owner VLA', font=font, fill='white')
            label = 'INITIAL: before first query' if not frame else (
                f'grasp {rows[frame-1]["sensed"]["grasp"]} | full {rows[frame-1]["task"]["full_completion"]}')
            draw.text((x+6, y+380), label, font=font, fill='white')
    sheet.save(root/'fridge90_endpoint_contact_sheet.png')
    with imageio.get_reader(root/'fridge90_actor_only_annotated.mp4') as reader:
        metadata = reader.get_meta_data()
        video = dict(fps=metadata['fps'], size=list(metadata['size']),
                     frames=reader.count_frames(), duration=metadata['duration'])
        assert video['fps'] == 20 and video['frames'] == 641
        imageio.imwrite(root/'fridge90_decoded_video_midpoint.png', reader.get_data(300))
    report = dict(status='all_artifact_checks_passed', runs=results, video=video,
                  checker_sha256=sha(Path(__file__).read_bytes()),
                  replay_scope='recorded action replay; no model re-query')
    (root/'artifact_checks.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    main(p.parse_args().root.resolve())
