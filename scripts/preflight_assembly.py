"""Physical-contact/instrumentation/exact-replay fixture; no efficacy claim."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from adapters.assembly_runtime import make_assembly, prepare_insertion, features, step_measured, endpoint
from adapters.assembly_predictor import AssemblyDynamics, candidates
from checkvla.reference import RefCheckVLA
from run_assembly_development import fingerprint, verify, save, sha

SOURCES = ['src/adapters/assembly_runtime.py', 'src/adapters/assembly_predictor.py',
    'src/models/nets.py', 'src/checkvla/reference.py', 'src/adapters/libero_openvla.py',
    'src/adapters/tf5_compat.py', 'src/adapters/libero_runtime.py',
    'src/adapters/libero_preservation.py', 'scripts/run_assembly_development.py',
    'scripts/launch_assembly_development.py', 'scripts/preflight_assembly.py',
    'exp_record/assembly_a1_protocol_20261015.md']


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    results = []
    for case in (0, 1):
        from run_assembly_development import OFFSETS
        dest = out/f'live_s0_c{case}_vla'
        dest.mkdir()
        env = make_assembly(0, True)
        rows = []
        states = []
        frames = []
        try:
            obs, setup = prepare_insertion(env, OFFSETS[case])
            save(dest/'setup.json', setup)
            dwell = max_dwell = 0
            for step in range(130):
                action = np.array([0, 0, -.15 if step < 100 else .15, 0, 0, 0, 1 if step < 100 else -1.])
                before = fingerprint(env, obs)
                obs, load = step_measured(env, action)
                ep = endpoint(env)
                dwell = dwell+1 if ep['complete'] else 0
                max_dwell = max(max_dwell, dwell)
                rows.append(dict(step=step, before=before, after_hash=fingerprint(env, obs),
                    effective=action.tolist(), peak_nut_peg_load_N=load, endpoint=ep,
                    after_state=env.sim.get_state().flatten().tolist(),
                    after_image_sha256=hashlib.sha256(np.ascontiguousarray(obs['agentview_image']).tobytes()).hexdigest()))
                states.append(env.sim.get_state().flatten().copy())
                frames.append(obs['agentview_image'].copy())
            assert features(env).shape == (40,) and np.isfinite(features(env)).all()
        finally:
            env.close()
        (dest/'trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        np.savez_compressed(dest/'fixture_frames.npz', frames=np.array(frames))
        save(dest/'status.json', dict(status='completed_pending_replay', steps=130,
            complete=dwell >= 5, max_dwell=max_dwell, trace_sha256=sha(dest/'trace.jsonl')))
        verify(out, 0, case, 'vla')
        # Instrumentation must not modify physics or rendered observations.
        env = make_assembly(0, True)
        try:
            obs, _ = prepare_insertion(env, OFFSETS[case])
            for i, row in enumerate(rows):
                obs, _, _, _ = env.step(np.array(row['effective']))
                endpoint(env)
                np.testing.assert_array_equal(env.sim.get_state().flatten(), states[i])
                assert fingerprint(env, obs) == row['after_hash']
        finally:
            env.close()
        results.append(dict(case=case, peak_load_N=max(r['peak_nut_peg_load_N'] for r in rows),
            max_dwell=max_dwell, native_ever=any(r['endpoint']['native_success'] for r in rows),
            exact_replay=True, instrumentation_invariant=True))
    assert results[0]['max_dwell'] >= 5, results
    assert results[1]['peak_load_N'] > 10, results
    torch.manual_seed(17)
    model = AssemblyDynamics(40, 7, E=3, hidden=128).cuda()
    actions = candidates(torch.zeros(7, device='cuda')).requires_grad_(True)
    ctx = dict(obs=torch.zeros(8, 40, device='cuda'), obs_prev=torch.zeros(8, 40, device='cuda'),
               a_prev=torch.zeros(8, 7, device='cuda'))
    verifier = RefCheckVLA(model, tau=0, latency=0, commit=1)
    predicted, risks = model.rollout(ctx['obs'], ctx['obs_prev'], ctx['a_prev'], actions)
    trigger, scores = verifier.check(ctx, actions)
    assert predicted.shape == (3, 8, 3, 40) and risks.shape == (3, 8, 3, 2)
    assert trigger.all() and scores.std() > 0
    scores.sum().backward()
    assert torch.isfinite(actions.grad).all() and actions.grad.abs().sum() > 0
    report = dict(status='passed', fixture_only=True, live_actor_loaded=False,
        physics=results, action_conditioning_and_autograd=True,
        sources={rel: hashlib.sha256((repo/rel).read_bytes()).hexdigest() for rel in SOURCES})
    save(out/'report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args().out)
