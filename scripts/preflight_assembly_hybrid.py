"""Four-arm real-simulator fixture and independent full decision replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main(source, out):
    repo = Path(__file__).resolve().parents[1]
    out.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, str(repo/'scripts/test_assembly_hybrid.py')], cwd=repo, check=True)
    fixture = source/'live_s3000_c3_vla/trace.jsonl'
    reports = []
    for arm in ('vla', 'feedback80', 'phase_geometry80', 'hybrid80'):
        for stage in ('live', 'verify'):
            command = [sys.executable, '-u', str(repo/'scripts/run_assembly_hybrid.py'), stage,
                '--source', str(source), '--out', str(out), '--seed', '3000', '--case', '3', '--arm', arm]
            if stage == 'live': command += ['--steps', '32', '--fixture', str(fixture)]
            with (out/f'{stage}_{arm}.log').open('x') as log:
                subprocess.run(command, cwd=repo, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, check=True)
        folder = out/f'live_s3000_c3_{arm}'
        status = json.loads((folder/'status.json').read_text())
        proof = json.loads((folder/'replay.json').read_text())
        assert status['fixture_only'] and proof['status'] == 'exact_replay_passed'
        reports.append(dict(arm=arm, changed=status['changed'], counters=status['counters'],
            trace_sha256=status['trace_sha256'], independent_full_replay=True))
    old_sources = json.loads((source/'launch.json').read_text())['sources']
    for path, digest in old_sources.items():
        assert hashlib.sha256((repo/path).read_bytes()).hexdigest() == digest
    files = list(old_sources)+['src/adapters/assembly_hybrid.py', 'scripts/run_assembly_hybrid.py',
        'scripts/launch_assembly_hybrid.py', 'scripts/preflight_assembly_hybrid.py',
        'scripts/test_assembly_hybrid.py', 'scripts/archive_assembly_run.py',
        'exp_record/assembly_a2_protocol_20261016.md']
    report = dict(status='passed', fixture_only=True, actor_loaded=False,
        invariants_passed=7, episodes=reports, candidate_count=11, max_microbatch=8,
        sources={path: hashlib.sha256((repo/path).read_bytes()).hexdigest() for path in files})
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=Path('/home/ami/scratch/assembly_a1_20261015_a'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.source, a.out)
