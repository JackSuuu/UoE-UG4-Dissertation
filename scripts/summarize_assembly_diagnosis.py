"""Combine finished local counterfactuals and replay-qualified budget probes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(root, diagnosis, budget):
    descriptive = json.loads((diagnosis/'descriptive.json').read_text())
    assert json.loads((diagnosis/'launch.json').read_text())['status'] == 'completed'
    assert json.loads((budget/'launch.json').read_text())['status'] == 'complete_verified'
    tau = json.loads((root/'training.json').read_text())['tau']
    names = ['nominal', 'hold_grasp', 'retract', 'x_plus', 'x_minus', 'y_plus', 'y_minus', 'insert']
    summary = dict(post_hoc_failure_selected=True, original_results_unchanged=True, cases=[])
    fig, axes = plt.subplots(4, 4, figsize=(17, 12), sharex='col', constrained_layout=True)
    for row_index, failure in enumerate(descriptive['failures']):
        seed, case = failure['seed'], failure['case']
        folder = f'live_s{seed}_c{case}_framework'
        points = json.loads((diagnosis/f'points_s{seed}_c{case}_framework.json').read_text())
        trace = [json.loads(line) for line in (root/folder/'trace.jsonl').read_text().splitlines()]
        b = json.loads((budget/folder/'status.json').read_text())
        replay = json.loads((budget/folder/'replay.json').read_text())
        prefix = json.loads((budget/folder/'matched_prefix.json').read_text())
        digest = hashlib.sha256((budget/folder/'trace.jsonl').read_bytes()).hexdigest()
        assert digest == b['trace_sha256'] == replay['trace_sha256'] == prefix['diagnostic_trace_sha256']
        assert prefix['original_trace_sha256'] == failure['trace_sha256'] and prefix['status'] == 'exact_matched'
        branches = [json.loads((diagnosis/f'branch_s{seed}_c{case}_t{step}.json').read_text()) for step in failure['events']]
        e = dict(seed=seed, case=case, original_success=False, original_budget_step=failure['budget_exhausted_step'],
            budget80_success=b['complete'], budget80_steps=b['steps'], budget80_changed=b['changed'],
            budget80_prefix_through=prefix['through_step'], budget80_replay=True,
            repair_scores_still_above_trigger=sum(r['forecast_scores'][names.index(r['selection'])] > tau for r in trace if r['changed']),
            all_candidates_above_trigger=sum(min(r['forecast_scores']) > tau for r in trace if r['changed']),
            local_branches=[{k: v for k, v in r.items() if k != 'values'} for r in branches])
        summary['cases'].append(e)
        steps = [r['step'] for r in points]
        ax = axes[row_index]
        ax[0].plot(steps, [r['post_xy_error_mm'] for r in points], label='actual lateral error')
        ax[0].set_ylabel(f'{seed}/c{case}\nmm')
        ax[1].plot(steps, [r['xyz_error_mm'] for r in points], label='executed-action prediction error')
        ax[1].set_ylabel('mm')
        ax[2].plot(steps, [r['actual_load_N'] for r in points], label='actual interval peak')
        ax[2].plot(steps, [r['predicted_load_N'] for r in points], label='predicted first interval')
        ax[2].set_ylabel('N')
        ax[3].plot(steps, [r['budget_used'] for r in points], label='used corrections')
        ax[3].fill_between(steps, 0, 40, where=[r['detached'] for r in points], alpha=.15, label='post-action detached')
        for a in ax:
            a.axvline(failure['budget_exhausted_step'], color='red', ls='--', alpha=.6)
            a.grid(alpha=.2)
        ax[3].set_title(f'budget80: {"PASS" if b["complete"] else "FAIL"}, {b["changed"]} changes')
    for a, title in zip(axes[0, :3], ['Lateral alignment', 'One-step xyz prediction error', 'Nut/peg contact load']):
        a.set_title(title)
    for a in axes[-1]: a.set_xlabel('Original A1 decision index (zero-based)')
    for a in axes[0]: a.legend(fontsize=7)
    fig.suptitle('A1 four failures: original trajectories; red line = 40th correction; budget80 outcome independently replayed')
    fig.savefig(diagnosis/'failure_timelines.png', dpi=140)
    plt.close(fig)
    with (diagnosis/'mechanism_summary.json').open('x') as f:
        json.dump(summary, f, indent=2)
        f.write('\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('diagnosis', type=Path)
    p.add_argument('budget', type=Path)
    a = p.parse_args()
    main(a.root, a.diagnosis, a.budget)
