"""Exact action replay, observable decision-chain verification and 2x review."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from run_libero_drawer import reset_seeds
from adapters.libero_completion import ContactRecorder
from adapters.libero_drawer import (ARMS, MAX_STEPS, MAX_REPAIR, DrawerMonitor,
    SubgoalTracker, TransitionGate, decision_trace, settled_drawer)
from adapters.libero_preservation import fingerprint
from summarize_libero_preservation import paired, read_rows


def verify(directory, root):
    manifest = json.loads((directory/'manifest.json').read_text())
    result = json.loads((directory/'result.json').read_text())
    assert manifest['status'] == 'complete'
    repo = Path(__file__).resolve().parents[1]
    for source, sha in manifest['source_sha256'].items():
        assert hashlib.sha256((repo/source).read_bytes()).hexdigest() == sha, source
    protocol = repo/'exp_record/libero_drawer_target_protocol_20261014.md'
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == manifest['protocol_sha256']
    rows = read_rows(directory/'steps.jsonl')
    traces = read_rows(directory/'decision_trace.jsonl')
    assert len(rows) == len(traces) == result['steps'] <= MAX_STEPS
    states = np.load(directory/'states.npz')['states']
    assert len(states) == len(rows)+1
    reset_seeds()
    env, obs = settled_drawer(root, result['index'])
    rec = ContactRecorder(env.sim, keywords=('akita_black_bowl', 'white_cabinet'))
    monitor, tracker, gate = DrawerMonitor(env), SubgoalTracker(), TransitionGate(result['arm'])
    try:
        previous = monitor.measure([])
        tracking = tracker.update(previous, -1)
        changes = 0
        with (directory/'substeps.jsonl').open() as raw:
            for i, row in enumerate(rows):
                assert row['step'] == i
                np.testing.assert_array_equal(states[i], env.get_sim_state())
                assert row['before'] == fingerprint(env, obs), (directory, i, 'replay fingerprint')
                assert row['tracking_before'] == tracking
                # Independent native-command mapping check.
                proposal = np.asarray(row['native_action']).copy()
                proposal[-1] = -np.sign(2*proposal[-1]-1)
                np.testing.assert_array_equal(proposal, row['proposal'])
                action, decision = gate.decide(proposal, previous, tracking, i)
                assert decision == row['decision']
                np.testing.assert_array_equal(action, row['action'])
                trace = decision_trace(i, result['arm'], previous, tracking, proposal, action, decision)
                # Compare the pre-action proof before using any future measurement.
                assert trace == {k:v for k,v in traces[i].items() if k != 'observed_after_execution'}
                rec.reset()
                obs, _, _, _ = env.step(action.tolist())
                sample = json.loads(next(raw))
                assert sample == dict(step=i, substeps=rec.samples), (directory, i, 'raw substeps')
                assert len(rec.samples) == 25
                measured = monitor.measure(rec.samples)
                assert measured == row['measured'], (directory, i, 'geometry/contact')
                tracking = tracker.update(measured, i)
                assert tracking == row['tracking']
                after = dict(retained=measured['retained'], bowl_gripper=measured['bowl_gripper'],
                    cabinet_gripper=measured['cabinet_gripper'], clear=measured['clear'],
                    drawer_q=measured['drawer_q'], native=measured['native_success'],
                    full_completion=tracking['complete'], phase=tracking['phase'])
                assert after == traces[i]['observed_after_execution']
                changed = not np.array_equal(action, proposal)
                assert row['correction'] == changed
                changes += int(changed)
                if tracking['complete']:
                    assert i == len(rows)-1
                previous = measured
            assert raw.readline() == ''
        np.testing.assert_array_equal(states[-1], env.get_sim_state())
        assert tracking['complete'] or len(rows) == MAX_STEPS
        assert result['complete'] == tracking['complete']
        assert result['native_ever'] == any(r['measured']['native_success'] for r in rows)
        assert result['milestone_steps'] == tracker.milestones
        assert result['trigger'] == gate.trigger
        assert result['intervention_steps'] == gate.used <= MAX_REPAIR
        assert result['changed_actions'] == changes
        assert result['handoff'] == gate.handoff
        assert result['VLA_queries'] == len(rows) and result['forecast_queries'] == 0
        assert result['final'] == previous
        assert result['post_handoff_bowl_contact_intervals'] == sum(r['measured']['bowl_gripper'] for r in rows
            if gate.handoff is not None and r['step'] >= gate.handoff['step'])
        assert result['CVR'] is None and result['safe_success'] is None
        with imageio.get_reader(directory/'rollout.mp4') as video:
            assert video.count_frames() == len(states) and video.get_meta_data()['fps'] == 20
        return result, rows, traces
    finally:
        rec.close()
        env.close()


def render(run, index, rows, traces, arms):
    review = run/'review'
    review.mkdir(exist_ok=True)
    name = f'root_{index}_methods_2x.mp4'
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    # Stream frames; retain only the last frame after an episode ends.
    readers = {a:imageio.get_reader(run/f'root_{index}'/a/'rollout.mp4') for a in arms}
    last = {}
    count = max(len(rows[a])+1 for a in arms)
    try:
        with imageio.get_writer(review/name, fps=40, codec='libx264',
                               ffmpeg_params=['-movflags','+faststart']) as writer:
            for frame in range(count):
                panels = []
                for arm in arms:
                    if frame <= len(rows[arm]):
                        last[arm] = readers[arm].get_next_data()
                    canvas = Image.new('RGB', (384, 672), '#101722')
                    draw = ImageDraw.Draw(canvas)
                    draw.text((8, 7), f'Root {index} | {arm} | 2x simulated time', font=font, fill='white')
                    draw.text((8, 28), f'Executed intervals: {min(frame,len(rows[arm]))}'
                        + (' | ENDED' if frame >= len(rows[arm]) else ''), font=font, fill='#9db1ce')
                    canvas.paste(Image.fromarray(last[arm]).resize((384,384)), (0, 50))
                    if frame:
                        k = min(frame, len(rows[arm]))-1
                        row, trace = rows[arm][k], traces[arm][k]
                        ev, pred, gate = trace['evidence'], trace['predicates'], trace['gate']
                        lines = [f"Phase: {trace['subgoal']} -> {row['tracking']['phase']}",
                            f"Evidence: support={ev['bowl_supported']} grip={ev['bowl_gripper']}",
                            f"Entry {pred['supported_entry']['consecutive']}/3 | release {pred['detached']['consecutive']}/3",
                            f"Clear {pred['stable_clearance']['consecutive']}/3 | margin front {ev['gripper_front_margin']*1000:.1f}mm",
                            f"Gate: {gate['mode']} -> {gate['route']}",
                            f"Reason: {gate['reason']} | used {gate['used']}/40",
                            f"Action changed={trace['correction']['changed']} | handoff={gate['handoff'] is not None}",
                            f"Observed: retained={row['measured']['retained']} closed={row['measured']['closed']}",
                            f"Whole-task dwell: {row['tracking']['final_streak']}/20"]
                        for j, line in enumerate(lines):
                            draw.text((8, 443+21*j), line, font=font,
                                      fill='#ffd08a' if j in [4,5] else 'white')
                    draw.text((8, 646), 'Privileged rule trace | safety not validated', font=font, fill='#9db1ce')
                    panels.append(np.asarray(canvas))
                writer.append_data(np.concatenate(panels, axis=1))
    finally:
        for reader in readers.values():
            reader.close()
    with imageio.get_reader(review/name) as reader:
        assert reader.count_frames() == count and reader.get_meta_data()['fps'] == 40
    # Observable full JSON proof is attached to each selected timeline entry.
    timeline = []
    for arm in arms:
        last_signature = None
        for t in traces[arm]:
            g = t['gate']
            signature = (t['subgoal'], g['mode'], g['reason'], json.dumps(g['handoff']))
            if signature != last_signature or g['mode'] == 'contact_transition':
                payload = html.escape(json.dumps(t, indent=2))
                label = html.escape(f"{arm} | step {t['step']} | {t['subgoal']} → {g['mode']} → {g['reason']}")
                timeline.append((t['step'], f'<details><summary><button onclick="jump({(t["step"]+1)/40})">▶</button> '
                    f'{label}</summary><pre>{payload}</pre></details>'))
            last_signature = signature
    page = f'''<!doctype html><meta charset="utf-8"><title>D1 root {index} trace</title>
<style>body{{background:#101722;color:white;font:16px system-ui}}video{{width:100%;position:sticky;top:0}}
pre{{white-space:pre-wrap}}details{{padding:6px;border-bottom:1px solid #456}}a{{color:#9bd3ff}}</style>
<h1>Root {index}: evidence → phase → event → gate → action → observed outcome</h1>
<p>Privileged deterministic rule trace. Pre-action evidence ends at t−1; outcome is separately labeled.
Video preserves all frames at 2x simulated time. No learned reasoning or validated safety claim.</p>
<video id="v" controls src="{name}"></video>
<script>function jump(t){{let v=document.getElementById('v');v.currentTime=t;v.pause();}}</script>
''' + ''.join(x[1] for x in sorted(timeline, key=lambda x:x[0]))
    (review/f'root_{index}_trace.html').write_text(page)
    return name


def main(args):
    results, hashes, videos = {}, {}, []
    for index in range(3):
        rows_by_arm, traces_by_arm, manifests = {}, {}, []
        for arm in args.arms:
            directory = args.run/f'root_{index}'/arm
            result, rows, traces = verify(directory, args.root)
            assert result['index'] == index and result['arm'] == arm
            results[index, arm], rows_by_arm[arm], traces_by_arm[arm] = result, rows, traces
            manifest = json.loads((directory/'manifest.json').read_text())
            manifests.append(manifest)
            for p in directory.iterdir():
                if p.is_file():
                    hashes[str(p.relative_to(args.run))] = hashlib.sha256(p.read_bytes()).hexdigest()
        for manifest in manifests[1:]:
            for key in ['source_sha256','protocol_sha256','task_sha256','actor']:
                assert manifest[key] == manifests[0][key]
        reference = rows_by_arm['vla']
        for arm in args.arms[1:]:
            rows = rows_by_arm[arm]
            first_changed = next((i for i,r in enumerate(rows) if r['correction']), None)
            prefix = len(rows) if first_changed is None else first_changed+1
            assert prefix <= len(reference)
            for i in range(prefix):
                assert rows[i]['before'] == reference[i]['before'], (index, arm, i, 'common prefix')
                assert rows[i]['tokens'] == reference[i]['tokens']
                assert rows[i]['proposal'] == reference[i]['proposal']
            if first_changed is None:
                assert len(rows) == len(reference)
        videos.append(render(args.run, index, rows_by_arm, traces_by_arm, args.arms))
    comparisons = [paired(results, a, b, list(range(3)), metric)
        for a, b in [('gate_hold','vla'), ('timed_repair','vla'), ('phase_repair','vla'),
                     ('phase_repair','gate_hold'), ('phase_repair','timed_repair')]
        if a in args.arms and b in args.arms for metric in ['complete','native_ever']]
    report = dict(status='verified', development_only=True, privileged=True, learned=False,
        full_action_replay=True, causal_decision_trace_verified=True,
        episodes=list(results.values()), comparisons=comparisons,
        artifact_sha256=hashes, videos=videos, Collision_Rate=None, CVR=None, safe_success=None)
    (args.run/'verified_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.run/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>D1 release-to-close</h1>'
        '<p>Frozen native H1; privileged measured-state guardrail. All evidence/action chains replay-verified.</p>'
        + ''.join(f'<h2>Root {i}</h2><a href="review/root_{i}_trace.html">Observable decision chain + video</a>'
                  f'<video controls width="100%" src="review/{name}"></video>' for i,name in enumerate(videos))
        + '<p><a href="verified_summary.json">Verified paired outcomes</a></p>')
    print(json.dumps(dict(status='verified', comparisons=comparisons, videos=videos)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('--root', default='/home/ami/scratch/LIBERO')
    p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    main(p.parse_args())
