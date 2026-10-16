"""A2 mixed-controller development: actual-state audit, live rollout, replay."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from run_assembly_development import OFFSETS, SNAPSHOT, save, sha, fingerprint
from adapters.assembly_runtime import make_assembly, prepare_insertion, features, step_measured, endpoint
from adapters.assembly_predictor import AssemblyDynamics
from adapters.assembly_hybrid import ARMS, NAMES, CAP, candidates, forecast, measured, residual, decide
from checkvla.reference import RefCheckVLA


def load_model(source):
    training = json.loads((source/'training.json').read_text())
    assert sha(source/'assembly_dynamics.pt') == training['checkpoint_sha256']
    model = AssemblyDynamics(40, 7, E=3, hidden=128).cuda()
    model.load_state_dict(torch.load(source/'assembly_dynamics.pt', weights_only=True))
    model.eval().requires_grad_(False)
    verifier = RefCheckVLA(model, tau=training['tau'], beta=1., latency=0, commit=1)
    return model, verifier, training


def predict(model, verifier, current, previous, ap, nominal):
    chunks = candidates(torch.tensor(nominal, dtype=torch.float32, device='cuda'))
    p = forecast(model, verifier, torch.tensor(current[None], device='cuda'),
        torch.tensor(previous[None], device='cuda'), torch.tensor(ap[None], device='cuda'), chunks)
    return chunks.cpu().numpy(), p


def context(env, current, last_forecast, history):
    monitor = measured(env, current)
    history['ungrasped'] = 0 if monitor['grasped'] else history['ungrasped']+1
    history['residual'] = residual(last_forecast, current, history['last_load'], monitor['tilt_deg'])
    return monitor


def initial_history():
    return dict(used=0, stall=0, ungrasped=0, last_load=0., residual=None)


def advance(history, current, after, action, load, changed):
    history['used'] += int(changed)
    stalled = action[2] < -.03 and load > .1 and current[2]-after[2] < .0005
    history['stall'] = history['stall']+1 if stalled else 0
    history['last_load'] = load


def audit(source, out):
    """All30 A1 actual trajectories. Shadow choices are NOT executed outcomes."""
    model, verifier, training = load_model(source)
    reports = []
    for seed in (3000, 3001):
        for case in range(5):
            for arm in ('vla', 'framework', 'feedback'):
                folder = source/f'live_s{seed}_c{case}_{arm}'
                status = json.loads((folder/'status.json').read_text())
                proof = json.loads((folder/'replay.json').read_text())
                assert sha(folder/'trace.jsonl') == status['trace_sha256'] == proof['trace_sha256']
                rows = [json.loads(line) for line in (folder/'trace.jsonl').read_text().splitlines()]
                env = make_assembly(seed, False)
                counters = Counter()
                errors = []
                details = []
                try:
                    _, setup = prepare_insertion(env, OFFSETS[case])
                    assert setup == json.loads((folder/'setup.json').read_text())
                    previous = features(env)
                    ap = np.array(setup[-1], dtype=np.float32)
                    history, last = initial_history(), None
                    for row in rows:
                        current = features(env)
                        np.testing.assert_array_equal(current, np.array(row['obs'], dtype=np.float32))
                        monitor = context(env, current, last, history)
                        nominal = np.array(row['nominal'])
                        bank, p = predict(model, verifier, current, previous, ap, nominal)
                        np.testing.assert_allclose(p['mean'][:8], row['forecast_mean'], atol=2e-5, rtol=2e-5)
                        # State-by-state shadow: no fictional hybrid continuation/budget.
                        history['used'] = 0
                        route = decide('hybrid80', current, nominal, bank[:, 0], p, monitor, history, verifier.tau)
                        counters['states'] += 1
                        counters['reliable'] += int(route['prediction_reliable'])
                        counters['owner_'+route['owner']] += 1
                        counters['reason_'+route['reason']] += 1
                        if history['residual'] is not None:
                            errors.append(history['residual'])
                        details.append(dict(step=row['step'], monitor=monitor, route=route,
                            observed_action=row['effective'], actual_endpoint=row['endpoint']))
                        _, load = step_measured(env, row['effective'])
                        after = features(env)
                        actual_ep = endpoint(env)
                        np.testing.assert_array_equal(env.sim.get_state().flatten(), row['after_state'])
                        assert load == row['peak_nut_peg_load_N'] and actual_ep == row['endpoint']
                        actual_index = ['nominal', 'hold_grasp', 'retract', 'x_plus', 'x_minus', 'y_plus', 'y_minus', 'insert'].index(row['selection'])
                        last = dict(state=p['mean'][actual_index, 0], risk=p['risk_mean'][actual_index, 0])
                        advance(history, current, after, np.array(row['effective']), load, False)
                        previous, ap = current, np.array(row['effective'], dtype=np.float32)
                    entry = dict(seed=seed, case=case, source_arm=arm, trace_sha256=status['trace_sha256'],
                        counters=dict(counters), residual_mean={k: float(np.mean([r[k] for r in errors])) for k in ('xyz_m', 'force_N', 'tilt_deg')},
                        physics_prefix_exact=True)
                    save(out/f'audit_s{seed}_c{case}_{arm}.json', dict(summary=entry, rows=details))
                    reports.append(entry)
                    print(json.dumps(entry), flush=True)
                finally:
                    env.close()
    save(out/'actual_state_audit.json', dict(status='complete', trajectories=len(reports),
        scope='all30 actual A1 trajectories, no outcome filtering; shadow decisions only, not hybrid success',
        checkpoint_sha256=training['checkpoint_sha256'], thresholds_frozen_before_audit=True, episodes=reports))


def live(source, out, seed, case, arm, steps, fixture=None):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    from adapters.libero_openvla import LiberoOpenVLA
    dest = out/f'live_s{seed}_c{case}_{arm}'
    dest.mkdir(exist_ok=False)
    save(dest/'status.json', dict(status='started', seed=seed, case=case, arm=arm, fixture_only=fixture is not None))
    model, verifier, training = load_model(source)
    fixture_rows = [json.loads(x) for x in fixture.read_text().splitlines()] if fixture else None
    actor = None if fixture else LiberoOpenVLA(SNAPSHOT)
    env = make_assembly(seed, True)
    writer = imageio.get_writer(dest/'annotated_2x.mp4', fps=40)
    start = time.monotonic()
    try:
        obs, setup = prepare_insertion(env, OFFSETS[case])
        save(dest/'setup.json', setup)
        previous = features(env)
        ap = np.array(setup[-1], dtype=np.float32)
        history, last = initial_history(), None
        counts = Counter()
        dwell = max_dwell = 0
        peak = 0.
        with (dest/'trace.jsonl').open('x') as log:
            for step in range(steps):
                current = features(env)
                before = fingerprint(env, obs)
                monitor = context(env, current, last, history)
                t0 = time.monotonic()
                if fixture:
                    nominal = np.array(fixture_rows[step]['nominal'])
                    tokens = dict(fixture_action=True)
                else:
                    nominal, tokens, actor_image = actor.act(obs['agentview_image'][:, ::-1],
                        'put the square nut onto the square peg and release it')
                    if step == 0: actor_image.save(dest/'actor_input.png')
                actor_s = time.monotonic()-t0
                t0 = time.monotonic()
                bank, p = predict(model, verifier, current, previous, ap, nominal)
                pre_history = dict(history)
                route = decide(arm, current, nominal, bank[:, 0], p, monitor, pre_history, verifier.tau)
                action = np.array(route['selected'])
                torch.cuda.synchronize()
                decision_s = time.monotonic()-t0
                next_obs, load = step_measured(env, action)
                after = features(env)
                ep = endpoint(env)
                advance(history, current, after, action, load, route['changed'])
                dwell = dwell+1 if ep['complete'] else 0
                max_dwell, peak = max(max_dwell, dwell), max(peak, load)
                counts['owner_'+route['owner']] += 1
                counts['reason_'+route['reason']] += 1
                counts['prediction_changed'] += int(route['owner'] == 'prediction' and route['changed'])
                idx = route['index']
                last = dict(state=p['mean'][idx, 0], risk=p['risk_mean'][idx, 0])
                row = dict(step=step, before=before, obs=current.tolist(), after=after.tolist(),
                    nominal=nominal.tolist(), selected=action.tolist(), effective=action.tolist(), tokens=tokens,
                    actor_s=actor_s, decision_s=decision_s, monitor=monitor, history=pre_history, decision=route,
                    changed=route['changed'], selection=route['selection'], corrections_used=history['used'],
                    forecast={k: v.tolist() for k, v in p.items()}, candidates=bank.tolist(),
                    peak_nut_peg_load_N=load, endpoint=ep, dwell=dwell,
                    handoff='fresh actual image actor query next interval',
                    after_hash=fingerprint(env, next_obs), after_state=env.sim.get_state().flatten().tolist())
                log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush()
                image = Image.fromarray(obs['agentview_image'][::-1]).resize((512, 512))
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 512, 82), fill='black')
                draw.text((6, 5), f'A2 {arm} s{seed}/c{case} t={step} 2x\n'
                    f'{route["subgoal"]} {route["owner"]}:{route["selection"]}\n'
                    f'{route["reason"]}\nload={load:.2f}N budget={history["used"]}/{CAP} dwell={dwell}/5', fill='white')
                writer.append_data(np.asarray(image))
                previous, ap, obs = current, action.astype(np.float32), next_obs
                if dwell >= 5: break
        save(dest/'status.json', dict(status='completed_pending_replay', seed=seed, case=case, arm=arm,
            fixture_only=fixture is not None, complete=dwell >= 5, max_dwell=max_dwell, steps=step+1,
            changed=history['used'], counters=dict(counts), peak_nut_peg_load_N=peak,
            elapsed_s=time.monotonic()-start, trace_sha256=sha(dest/'trace.jsonl'),
            checkpoint_sha256=training['checkpoint_sha256'], actor_build=None if fixture else actor.build))
    finally:
        writer.close(); env.close()


def verify(source, out, seed, case, arm):
    dest = out/f'live_s{seed}_c{case}_{arm}'
    status = json.loads((dest/'status.json').read_text())
    assert status['status'] == 'completed_pending_replay'
    assert sha(dest/'trace.jsonl') == status['trace_sha256']
    model, verifier, _ = load_model(source)
    env = make_assembly(seed, True)
    try:
        obs, setup = prepare_insertion(env, OFFSETS[case])
        assert setup == json.loads((dest/'setup.json').read_text())
        previous, ap = features(env), np.array(setup[-1], dtype=np.float32)
        history, last = initial_history(), None
        dwell = max_dwell = 0
        for step, text in enumerate((dest/'trace.jsonl').read_text().splitlines()):
            row = json.loads(text)
            assert step == row['step'] and fingerprint(env, obs) == row['before'], ('before', step)
            current = features(env)
            monitor = context(env, current, last, history)
            assert monitor == row['monitor'] and history == row['history'], ('monitor/history', step)
            nominal = np.array(row['nominal'])
            bank, p = predict(model, verifier, current, previous, ap, nominal)
            for key in p: np.testing.assert_array_equal(p[key], row['forecast'][key])
            route = decide(arm, current, nominal, bank[:, 0], p, monitor, history, verifier.tau)
            assert route == row['decision'], ('decision', step)
            np.testing.assert_array_equal(route['selected'], row['effective'])
            action = np.array(row['effective'])
            obs, load = step_measured(env, action)
            after = features(env)
            ep = endpoint(env)
            advance(history, current, after, action, load, route['changed'])
            assert fingerprint(env, obs) == row['after_hash'], ('after', step)
            assert ep == row['endpoint'] and load == row['peak_nut_peg_load_N'], ('outcome', step)
            dwell = dwell+1 if ep['complete'] else 0
            max_dwell = max(max_dwell, dwell)
            idx = route['index']
            last = dict(state=p['mean'][idx, 0], risk=p['risk_mean'][idx, 0])
            previous, ap = current, action.astype(np.float32)
        assert step+1 == status['steps'] and (dwell >= 5) == status['complete'] and max_dwell == status['max_dwell']
        assert history['used'] == status['changed']
        save(dest/'replay.json', dict(status='exact_replay_passed', steps=step+1,
            trace_sha256=status['trace_sha256'], checks='physics/images/forecasts/monitor/decision/actions/outcome'))
    finally:
        env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['audit', 'live', 'verify'])
    p.add_argument('--source', type=Path, default=Path('/home/ami/scratch/assembly_a1_20261015_a'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=3000)
    p.add_argument('--case', type=int, default=0)
    p.add_argument('--arm', choices=ARMS, default='hybrid80')
    p.add_argument('--steps', type=int, default=160)
    p.add_argument('--fixture', type=Path)
    a = p.parse_args()
    torch.set_num_threads(4)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    if a.stage == 'audit': audit(a.source, a.out)
    elif a.stage == 'live': live(a.source, a.out, a.seed, a.case, a.arm, a.steps, a.fixture)
    else: verify(a.source, a.out, a.seed, a.case, a.arm)
