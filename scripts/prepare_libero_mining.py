"""Outcome-independent 39-case discovery grid. Never touches final roots."""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(source, output):
    original = json.loads(source.read_text())
    assert original['development_only'] and original['status'] == 'prepared_not_executed'
    cases = []
    for index in range(3):
        base = next(c for c in original['cases'] if c['root'] == index)
        assert base['eligible']
        shared = {k:base[k] for k in ['root','anchor_step','prefix_before_fingerprint','source_steps_sha256']}
        configurations = [dict(fault='release_delay', duration=d, label=f'delay_{d}') for d in (0,4,8,12,16)]
        configurations += [dict(fault='pose_bias', duration=4, axis=a, requested_m=sign*mag,
            label=f'pose_{"xy"[a]}_{"pos" if sign>0 else "neg"}_{round(mag*1000)}mm')
            for mag in (.01,.02) for a in (0,1) for sign in (-1,1)]
        for config in configurations:
            cases.append(dict(**shared, **config, case_id=f'root_{index}_{config["label"]}', split='development'))
    manifest = dict(schema=1, status='prepared_not_executed', development_only=True,
        source_run=original['source_run'], source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        task=original['task'], cases=cases, final_roots_used=[], case_selection='fixed factorial, no method outcomes',
        arms=['vla','gate_hold','timed_repair','phase_repair','verified_repair'],
        total_episodes=len(cases)*5, max_steps_including_prefix=600)
    with output.open('x') as f:
        f.write(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(cases=len(cases), episodes=manifest['total_episodes'])))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('source',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    prepare(a.source,a.output)
