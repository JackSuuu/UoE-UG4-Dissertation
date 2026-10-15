"""Failure-only post-hoc budget intervention: 40 -> 80, all other rules frozen.

Generate and hash an explicitly labeled function variant without editing the A1
source. This is a causal diagnostic on selected failures, not a new benchmark.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import run_assembly_development as original


def main(root, out, seed, case):
    expected = json.loads((root/'launch.json').read_text())['sources']['scripts/run_assembly_development.py']
    source_path = Path(original.__file__)
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == expected
    out.mkdir(parents=True, exist_ok=True)
    for name in ('training.json', 'assembly_dynamics.pt'):
        target = out/name
        if not target.exists():
            # Launch lanes share only already-frozen read-only files.
            try: target.symlink_to((root/name).resolve())
            except FileExistsError: pass
        assert original.sha(target) == original.sha(root/name)
    function = inspect.getsource(original.live)
    old = 'if trigger and changed < 40:'
    new = 'if trigger and changed < 80:'
    assert function.count(old) == 1 and function.count('changes={changed}/40') == 1
    derived = function.replace(old, new).replace('changes={changed}/40', 'changes={changed}/80')
    spec = dict(type='post-hoc failure-only budget diagnostic', seed=seed, case=case,
        change='maximum changed intervals 40 -> 80; display corrected to /80',
        original_source_sha256=expected, derived_live_sha256=hashlib.sha256(derived.encode()).hexdigest(),
        checkpoint_sha256=original.sha(root/'assembly_dynamics.pt'),
        unchanged='actor/checkpoint, input, gate, score, candidates, horizon160, endpoint',
        full_task_success_claim_scope='diagnostic on the four selected original failures only')
    (out/f'derived_live_s{seed}_c{case}.py').write_text(derived)
    original.save(out/f'variant_s{seed}_c{case}.json', spec)
    namespace = dict(original.__dict__)
    exec(compile(derived, str(out/f'derived_live_s{seed}_c{case}.py'), 'exec'), namespace)
    original.torch.set_num_threads(4)
    original.random.seed(0); original.np.random.seed(0); original.torch.manual_seed(0)
    namespace['live'](out, seed, case, 'framework', 160)
    # Exact-match the original causal prefix before extra budget becomes available.
    folder = f'live_s{seed}_c{case}_framework'
    old_rows = [json.loads(x) for x in (root/folder/'trace.jsonl').read_text().splitlines()]
    new_rows = [json.loads(x) for x in (out/folder/'trace.jsonl').read_text().splitlines()]
    boundary = next(r['step'] for r in old_rows if r['corrections_used'] == 40)
    for before, after in zip(old_rows[:boundary+1], new_rows[:boundary+1]):
        for field in ('before', 'nominal', 'selected', 'effective', 'after_hash', 'objective', 'forecast_mean'):
            assert before[field] == after[field], (before['step'], field)
    assert len(new_rows) > boundary
    original.save(out/folder/'matched_prefix.json', dict(status='exact_matched', through_step=boundary,
        original_trace_sha256=original.sha(root/folder/'trace.jsonl'),
        diagnostic_trace_sha256=original.sha(out/folder/'trace.jsonl')))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--case', type=int, required=True)
    a = p.parse_args()
    main(a.root, a.out, a.seed, a.case)
