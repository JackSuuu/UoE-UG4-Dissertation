"""A1 development: train self-developed dynamics, freeze, then paired live VLA.

See exp_record/assembly_a1_protocol_20261015.md. Never touches LIBERO M1 roots.
"""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from adapters.assembly_runtime import make_assembly, prepare_insertion, features, step_measured, endpoint, positions
from adapters.assembly_predictor import AssemblyDynamics, candidates, NAMES
from checkvla.reference import RefCheckVLA

OFFSETS = [(0., 0.), (.012, 0.), (-.012, 0.), (0., .012), (0., -.012)]
SNAPSHOT = '/home/ami/.cache/huggingface/hub/models--openvla--openvla-7b-finetuned-libero-10/snapshots/80970322773f81baa2e22fe495d0487b93a05cfa'


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(env, obs):
    h = hashlib.sha256()
    h.update(env.sim.get_state().flatten().tobytes())
    h.update(np.ascontiguousarray(obs['agentview_image']).tobytes())
    return h.hexdigest()


def collect(out):
    rows = []
    for split, seeds in [('train', range(1000, 1024)), ('calibration', range(2000, 2008)), ('validation', range(2100, 2108))]:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            offset = rng.uniform(-.018, .018, 2)
            env = make_assembly(seed, False)
            try:
                _, setup = prepare_insertion(env, offset)
                obs = features(env)
                prev, ap = obs.copy(), np.array(setup[-1], dtype=np.float32)
                episode = []
                for t in range(160):
                    action = np.zeros(7, dtype=np.float32)
                    action[-1] = 1
                    if t < 45:
                        action[2] = -.15
                    else:
                        # Contact-rich exploration includes alignment, release and retreat.
                        if t % 5 == 0:
                            choice = int(rng.integers(8))
                        if choice == 0:
                            action[:2] = np.clip(-obs[:2]/.05, -.25, .25)
                            action[2] = -.15
                        elif choice == 1:
                            action[2] = .15
                        elif choice in (2, 3, 4, 5):
                            action[(choice-2)//2] = .15 if choice % 2 == 0 else -.15
                        elif choice == 6:
                            action[2] = -.15
                        if t >= 130:
                            action[-1] = -1
                            action[2] = .15
                    _, load = step_measured(env, action)
                    nxt = features(env)
                    tilt = np.arccos(np.clip(nxt[14], -1, 1))
                    risk = [load/10., tilt/np.deg2rad(15)]
                    episode.append(dict(obs=obs.tolist(), prev=prev.tolist(), ap=ap.tolist(),
                        action=action.tolist(), nxt=nxt.tolist(), risk=risk))
                    prev, obs, ap = obs, nxt, action
                record = dict(seed=seed, split=split, offset=offset.tolist(), setup=setup, rows=episode)
                save(out/f'data_{seed}.json', record)
                rows.append(dict(seed=seed, split=split, transitions=len(episode), sha256=sha(out/f'data_{seed}.json')))
                print(f'collected {split} {seed}', flush=True)
            finally:
                env.close()
    save(out/'data_manifest.json', rows)


def windows(out, split, device):
    items = []
    for entry in json.loads((out/'data_manifest.json').read_text()):
        if entry['split'] != split:
            continue
        path = out/f"data_{entry['seed']}.json"
        assert sha(path) == entry['sha256']
        rows = json.loads(path.read_text())['rows']
        for i in range(len(rows)-2):
            items.append([rows[i]['obs'], rows[i]['prev'], rows[i]['ap'],
                [r['action'] for r in rows[i:i+3]], [r['nxt'] for r in rows[i:i+3]],
                [r['risk'] for r in rows[i:i+3]]])
    return [torch.tensor(np.array([r[k] for r in items]), dtype=torch.float32, device=device) for k in range(6)]


def train(out):
    torch.manual_seed(17)
    device = 'cuda:0'
    data = windows(out, 'train', device)
    model = AssemblyDynamics(40, 7, E=3, hidden=128).to(device)
    model.obs_n.fit(data[0])
    model.dobs_n.fit(data[4][:, 0]-data[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    history = []
    for iteration in range(1800):
        # Maximum minibatch is eight, including validation below.
        idx = torch.randint(len(data[0]), (8,), device=device)
        o, op, ap, actions, target, risks = [x[idx] for x in data]
        predicted, pred_risk = model.rollout(o, op, ap, actions)
        loss_state = torch.nn.functional.smooth_l1_loss(
            predicted/model.dobs_n.std, target[None].expand_as(predicted)/model.dobs_n.std)
        loss_risk = torch.nn.functional.smooth_l1_loss(pred_risk, risks[None].expand_as(pred_risk))
        loss = loss_state + loss_risk
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if iteration % 100 == 0:
            history.append(dict(iteration=iteration, state=float(loss_state.detach()), risk=float(loss_risk.detach())))
            print(history[-1], flush=True)
    model.eval().requires_grad_(False)
    torch.save(model.state_dict(), out/'assembly_dynamics.pt')
    metrics = {}
    safe_scores = []
    for split in ('calibration', 'validation'):
        data = windows(out, split, device)
        errors, risk_errors, scores, actuals = [], [], [], []
        with torch.no_grad():
            for start in range(0, len(data[0]), 8):
                o, op, ap, actions, target, risks = [x[start:start+8] for x in data]
                predicted, pred_risk = model.rollout(o, op, ap, actions)
                score = (pred_risk.mean(0)+pred_risk.std(0)).amax((1, 2))
                actual = risks.amax((1, 2))
                errors.extend((predicted.mean(0)[..., :3]-target[..., :3]).abs().cpu().tolist())
                risk_errors.extend((pred_risk.mean(0)-risks).abs().cpu().tolist())
                scores.extend(score.cpu().tolist()); actuals.extend(actual.cpu().tolist())
                if split == 'calibration':
                    safe_scores.extend(score[actual <= 1].cpu().tolist())
        metrics[split] = dict(windows=len(scores), xyz_mae_m=np.mean(errors, axis=0).tolist(),
            risk_mae=np.mean(risk_errors, axis=0).tolist(), scores=scores, actual_max_risk=actuals)
    assert len(safe_scores) >= 20, 'Insufficient low-load calibration windows'
    k = min(len(safe_scores)-1, int(np.ceil((len(safe_scores)+1)*.95))-1)
    tau = sorted(safe_scores)[k]
    report = dict(status='trained_and_frozen', checkpoint_sha256=sha(out/'assembly_dynamics.pt'),
        implementation='self-developed OrbiSim-role autoregressive MLP ensemble; not official',
        tau=tau, safe_calibration_windows=len(safe_scores),
        calibration_claim='empirical development quantile; overlapping windows are NOT independent conformal units',
        history=history, metrics=metrics)
    save(out/'training.json', report)


def live(out, seed, case, arm, steps):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    from adapters.libero_openvla import LiberoOpenVLA
    dest = out/f'live_s{seed}_c{case}_{arm}'
    dest.mkdir(exist_ok=False)
    save(dest/'status.json', dict(status='started', seed=seed, case=case, arm=arm, max_steps=steps))
    training = json.loads((out/'training.json').read_text())
    assert sha(out/'assembly_dynamics.pt') == training['checkpoint_sha256']
    model = AssemblyDynamics(40, 7, E=3, hidden=128).cuda()
    model.load_state_dict(torch.load(out/'assembly_dynamics.pt', weights_only=True))
    model.eval().requires_grad_(False)
    verifier = RefCheckVLA(model, tau=training['tau'], beta=1., latency=0, commit=1)
    actor = LiberoOpenVLA(SNAPSHOT)
    env = make_assembly(seed, True)
    writer = imageio.get_writer(dest/'annotated_2x.mp4', fps=40)
    start = time.monotonic()
    try:
        obs, setup = prepare_insertion(env, OFFSETS[case])
        save(dest/'setup.json', setup)
        prev = features(env)
        ap = np.asarray(setup[-1], dtype=np.float32)
        memory = deque(maxlen=16)
        changed, stall, dwell, max_dwell, peak = 0, 0, 0, 0, 0.
        prediction = None
        with (dest/'trace.jsonl').open('x') as log:
            for step in range(steps):
                current = features(env)
                residual = None if prediction is None else float(np.linalg.norm(current[:3]-prediction[:3]))
                before = fingerprint(env, obs)
                t0 = time.monotonic()
                # Stock robosuite camera needs vertical flip. Libero preprocessing
                # rotates 180 degrees; compensate horizontal flip explicitly here.
                nominal, tokens, actor_image = actor.act(obs['agentview_image'][:, ::-1],
                    'put the square nut onto the square peg and release it')
                actor_s = time.monotonic()-t0
                if step == 0:
                    actor_image.save(dest/'actor_input.png')
                t0 = time.monotonic()
                ctx = dict(obs=torch.tensor(current[None], device='cuda'),
                    obs_prev=torch.tensor(prev[None], device='cuda'), a_prev=torch.tensor(ap[None], device='cuda'))
                chunks = candidates(torch.tensor(nominal, dtype=torch.float32, device='cuda'))
                expanded = {k: v.expand(8, -1) for k, v in ctx.items()}
                with torch.no_grad():
                    states, risks = model.rollout(expanded['obs'], expanded['obs_prev'], expanded['a_prev'], chunks)
                    scores = verifier.score(expanded, chunks)
                    fired, _ = verifier.check(ctx, chunks[:1])
                    pred = states.mean(0)
                    # Predicted distance to seated/aligned goal + peak load/tilt penalty.
                    objective = (pred[:, -1, :2].norm(dim=-1)/.02 +
                        (pred[:, -1, 2]+.02).abs()/.12 + .5*scores)
                    choice = int(objective.argmin())
                contact_mode = bool(current[2] < .13 or current[-1] > .1)
                subgoal = 'release_and_clear' if current[2] < -.01 else ('align_and_insert' if contact_mode else 'approach')
                trigger = contact_mode and (bool(fired[0]) or stall >= 3)
                reason = 'persistent_contact_stall' if stall >= 3 else 'predicted_load_or_tilt'
                selected = nominal.copy()
                selection = 0
                if trigger and changed < 40:
                    if arm == 'framework' and choice != 0 and float(objective[choice]) < float(objective[0])-.01:
                        selection = choice
                        selected = chunks[choice, 0].cpu().numpy().copy()
                    elif arm == 'feedback':
                        # Same trigger/candidates; measured geometry, no forecast ranking.
                        if current[-1] > 10:
                            selection = 2
                        elif np.max(np.abs(current[:2])) > .004:
                            axis = int(np.argmax(np.abs(current[:2])))
                            selection = (4 if current[0] > 0 else 3) if axis == 0 else (6 if current[1] > 0 else 5)
                        else:
                            selection = 7
                        selected = chunks[selection, 0].cpu().numpy().copy()
                is_changed = bool(np.max(np.abs(selected-nominal)) > 1e-8)
                changed += int(is_changed)
                torch.cuda.synchronize()
                decision_s = time.monotonic()-t0
                next_obs, load = step_measured(env, selected)
                after = features(env)
                stalled = selected[2] < -.03 and load > .1 and current[2]-after[2] < .0005
                stall = stall+1 if stalled else 0
                ep = endpoint(env)
                dwell = dwell+1 if ep['complete'] else 0
                max_dwell = max(max_dwell, dwell)
                peak = max(peak, load)
                prediction = pred[selection, 0].cpu().numpy()
                memory.append(dict(step=step, residual_m=residual, load_N=load, stall=stall))
                row = dict(step=step, before=before, obs=current.tolist(), after=after.tolist(),
                    nominal=nominal.tolist(), selected=selected.tolist(), effective=selected.tolist(),
                    tokens=tokens, actor_s=actor_s, decision_s=decision_s,
                    subgoal=subgoal,
                    mode='contact_transition' if contact_mode else 'nominal_tracking',
                    trigger=trigger, gate_reason=reason if trigger else 'continue',
                    route=('hold' if selection == 1 else 'repair') if is_changed else 'continue',
                    selection=NAMES[selection], changed=is_changed, corrections_used=changed,
                    forecast_scores=scores.cpu().tolist(), objective=objective.cpu().tolist(),
                    forecast_mean=pred.cpu().tolist(), risk_mean=risks.mean(0).cpu().tolist(),
                    risk_std=risks.std(0).cpu().tolist(), candidates=chunks.cpu().tolist(),
                    prediction_residual_m=residual, memory=list(memory),
                    peak_nut_peg_load_N=load, endpoint=ep, dwell=dwell,
                    handoff='fresh observed OpenVLA query next interval', after_hash=fingerprint(env, next_obs),
                    after_state=env.sim.get_state().flatten().tolist(),
                    after_image_sha256=hashlib.sha256(np.ascontiguousarray(next_obs['agentview_image']).tobytes()).hexdigest())
                log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush()
                image = Image.fromarray(obs['agentview_image'][::-1]).resize((512, 512))
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 512, 66), fill='black')
                draw.text((6, 5), f'A1 DEV {arm} seed={seed} offset={OFFSETS[case]} 2x\n'
                    f't={step} {row["mode"]} {row["route"]}:{row["selection"]}\n'
                    f'load={load:.2f}N changes={changed}/40 dwell={dwell}/5', fill='white')
                writer.append_data(np.asarray(image))
                prev, ap, obs = current, selected.astype(np.float32), next_obs
                if dwell >= 5:
                    break
        save(dest/'status.json', dict(status='completed_pending_replay', seed=seed, case=case, arm=arm,
            steps=step+1, complete=dwell >= 5, max_dwell=max_dwell, changed=changed,
            peak_nut_peg_load_N=peak, elapsed_s=time.monotonic()-start,
            trace_sha256=sha(dest/'trace.jsonl'), checkpoint_sha256=training['checkpoint_sha256'],
            actor_build=actor.build))
    finally:
        writer.close()
        env.close()


def verify(out, seed, case, arm):
    dest = out/f'live_s{seed}_c{case}_{arm}'
    status = json.loads((dest/'status.json').read_text())
    assert status['status'] == 'completed_pending_replay'
    assert sha(dest/'trace.jsonl') == status['trace_sha256']
    env = make_assembly(seed, True)
    try:
        obs, setup = prepare_insertion(env, OFFSETS[case])
        assert setup == json.loads((dest/'setup.json').read_text())
        dwell = max_dwell = 0
        for step, line in enumerate((dest/'trace.jsonl').read_text().splitlines()):
            row = json.loads(line)
            assert row['step'] == step and fingerprint(env, obs) == row['before'], f'pre-action mismatch {step}'
            obs, load = step_measured(env, row['effective'])
            ep = endpoint(env)
            assert load == row['peak_nut_peg_load_N'] and ep == row['endpoint'], f'metric mismatch {step}'
            if fingerprint(env, obs) != row['after_hash']:
                actual = env.sim.get_state().flatten()
                detail = dict(step=step, actual_state=actual.tolist(),
                    expected_state=row.get('after_state'),
                    state_max_error=None if 'after_state' not in row else float(np.max(np.abs(actual-row['after_state']))),
                    expected_image_sha256=row.get('after_image_sha256'),
                    actual_image_sha256=hashlib.sha256(np.ascontiguousarray(obs['agentview_image']).tobytes()).hexdigest())
                save(dest/'replay_mismatch.json', detail)
                np.save(dest/'replay_mismatch_image.npy', obs['agentview_image'])
                raise AssertionError(f'post-action mismatch {step}; see replay_mismatch.json')
            dwell = dwell+1 if ep['complete'] else 0
            max_dwell = max(max_dwell, dwell)
        assert step+1 == status['steps'] and (dwell >= 5) == status['complete'] and max_dwell == status['max_dwell']
        save(dest/'replay.json', dict(status='exact_replay_passed', steps=step+1, trace_sha256=status['trace_sha256']))
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['collect', 'train', 'live', 'verify'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=3000)
    p.add_argument('--case', type=int, default=0)
    p.add_argument('--arm', choices=['vla', 'framework', 'feedback'], default='framework')
    p.add_argument('--steps', type=int, default=160)
    a = p.parse_args()
    torch.set_num_threads(4)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    if a.stage == 'collect': collect(a.out)
    elif a.stage == 'train': train(a.out)
    elif a.stage == 'live': live(a.out, a.seed, a.case, a.arm, a.steps)
    else: verify(a.out, a.seed, a.case, a.arm)


if __name__ == '__main__':
    main()
