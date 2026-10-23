"""Plot recorded progress, local repair occupancy and joint risk by root."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = json.loads((args.analysis/'report.json').read_text())
    colors = dict(baseline='#333333', joint='#cb4b16', measured_fusion='#268bd2',
        joint_admission='#859900', contract_stage='#6c71c4', contract_progress='#d33682')
    for root in sorted({e['root'] for e in report['episodes']}):
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
            gridspec_kw=dict(height_ratios=[2, 1.5, 1.5, 1.5]))
        selected = [e for e in report['episodes'] if e['root'] == root]
        for index, ep in enumerate(selected):
            trace = [json.loads(line) for line in
                (args.analysis/f"{ep['name']}.steps.jsonl").read_text().splitlines()]
            color = colors[ep['arm']]
            x = [r['action_index'] for r in trace]
            height = [float('nan') if r['post']['rise_m'] is None else 1000*r['post']['rise_m'] for r in trace]
            axes[0].plot(x, height, color=color,
                label=f"{ep['arm']}: {'success' if ep['success'] else ep['termination']}, {ep['actions']} actions")
            for y, key in ((index*2, 'held_clear'), (index*2+.7, 'in_goal')):
                hits = [r['action_index'] for r in trace if r['post'][key] is True]
                axes[1].scatter(hits, [y]*len(hits), marker='|', s=10, color=color)
                axes[1].text(522, y, f"{ep['arm']} {key}", fontsize=7, va='center', color=color)
            for repair in ep['repairs']:
                a, b = repair['entry_step'], repair['exit_step']
                axes[2].broken_barh([(a, b-a+1)], (index-.3, .6), facecolors=color)
                axes[2].text(a, index+.32, f"R{repair['repair_id']}", fontsize=7, color=color)
            if ep['arm'] == 'joint':
                risk = [r['joint_risk']['score'] for r in trace]
                axes[3].plot(x, risk, color=color, label='joint risk score (before action)')
                axes[3].axhline(trace[0]['joint_risk']['threshold'], color=color, linestyle=':', label='fixed risk threshold')
        axes[0].axhline(20, linestyle=':', color='#999999', label='20 mm lift threshold; grasp/contact also required')
        axes[0].set_ylabel('Cup rise from first grasp (mm)')
        axes[0].legend(fontsize=8, loc='upper left')
        axes[1].set_yticks([])
        axes[1].set_ylim(-.5, len(selected)*2)
        axes[1].set_ylabel('Observed progress')
        axes[2].set_yticks(range(len(selected)), [e['arm'] for e in selected], fontsize=8)
        axes[2].set_ylim(-.7, len(selected)-.3)
        axes[2].set_ylabel('Repair actions')
        axes[3].set_ylabel('Risk score')
        axes[3].set_xlabel('Zero-based action index; progress measured AFTER action, risk BEFORE action')
        axes[3].legend(fontsize=8, loc='upper left')
        for ax in axes:
            ax.set_xlim(0, 520)
            ax.grid(alpha=.15)
        fig.suptitle(f'Root {root}: recorded trajectories (same-index states diverge after first intervention)', fontsize=12)
        fig.tight_layout(rect=[0, 0, .85, .97])
        fig.savefig(args.out/f'root{root}.png', dpi=160, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()
