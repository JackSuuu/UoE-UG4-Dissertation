"""Branch integrity/controller smoke on consumed root2; no actor/model training."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '4')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from adapters.libero_completion import CompletionMonitor, ContactRecorder
from adapters.libero_preservation import candidates, fingerprint, forecast, settled_env


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/home/ami/scratch/LIBERO')
    p.add_argument('--reference', type=Path, default=Path('/home/ami/scratch/libero_continuation_20261013_a/worker_2'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--decisions', type=int, default=3)
    a = p.parse_args()
    rows = [json.loads(line) for line in (a.reference/'steps.jsonl').read_text().splitlines()]
    reference_states = np.load(a.reference/'states.npz')['states']
    branch = next(i+1 for i, row in enumerate(rows) if row['candidate'])
    env, obs = settled_env(a.root, 2)
    recorder = ContactRecorder(env.sim)
    monitor = CompletionMonitor(env)
    try:
        np.testing.assert_array_equal(env.get_sim_state(), reference_states[0])
        actions = []
        for i in range(branch):
            recorder.reset()
            obs, _, _, _ = env.step(rows[i]['action'])
            measured = monitor.measure(i, recorder.samples)
            assert measured == {k: rows[i][k] for k in measured}
            np.testing.assert_array_equal(env.get_sim_state(), reference_states[i+1])
            actions.append(np.asarray(rows[i]['action']))
        reports = []
        for decision in range(a.decisions):
            step = branch + decision
            before = fingerprint(env, obs)
            # This is an integrity test using a recorded action as a probe,
            # never an efficacy rollout or a counterfactual future VLA tape.
            plans = candidates(rows[step]['action'], actions[-1])
            chosen, report = forecast(a.root, 2, actions, plans, before, measured)
            assert before == fingerprint(env, obs)
            recorder.reset()
            obs, _, _, _ = env.step(plans[chosen, 0].tolist())
            actual = monitor.measure(step, recorder.samples)
            assert actual == report['candidates'][chosen]['rows'][0]
            actions.append(plans[chosen, 0].copy())
            measured = actual
            reports.append(report)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(status='passed', branch_step=branch,
            prefix_states_checked=branch+1, controller_validated=True, branches=6,
            full_integration_controller_image_identity=True, live_nonmutation=True,
            selected_first_prediction_equal_actual=True, consecutive_decisions=a.decisions,
            reports=reports), indent=2)+'\n')
        print(json.dumps(dict(status='passed', branch_step=branch, consecutive_decisions=a.decisions,
                             seconds=sum(r['wall_seconds'] for r in reports))))
    finally:
        recorder.close()
        env.close()
