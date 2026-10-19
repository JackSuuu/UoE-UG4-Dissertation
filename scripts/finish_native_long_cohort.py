"""One-shot GPU1 continuation of the two untouched native L1 root3 cells.

--validate-only --evidence NEW_DIR freezes a manifest and exercises reporting
against the actual 22 external episodes, without GPU access or actor queries.
--launch --manifest FILE accepts that frozen manifest, rechecking all pins.
Each fresh actor is followed by the unchanged strict replay in a fresh process;
any infrastructure/replay failure blocks the remaining cell. No retries.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from run_native_long_cohort import REPO, GPUS, EGL_DEVICES, TASK_ARGS, argument, require, sha
from run_fridge_r1 import SNAPSHOT
from resume_native_long_cohort import (IDENTITIES, absent, check_episode, check_replay,
                                       dead, read_json, untouched)


ORIGINAL = Path('/home/ami/scratch/native_long_l1_20261019_a')
PRIOR = Path('/home/ami/scratch/native_long_l1_20261019_b')
OUT = Path('/home/ami/scratch/native_long_l1_20261019_c')
SCRIPT = Path(__file__).resolve()
REPORT = 'scripts/summarize_native_long_cohort.py'
REPORT_SHA = '22dc112ec87b4010bc040f13875f9e1f602ba5527a6fd2c26f6df3d553027a47'
NAMES = ['two_mugs_root3', 'drawer_root3']
ISSUES = [f'two_pots_root{root}' for root in range(4)]
GPU_UUID = 'GPU-3f4f72ef-ed7f-2ddb-e454-abcc6c0330b0'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def check_untouched():
    for run in (ORIGINAL, PRIOR):
        for name in NAMES:
            untouched(run, dict(name=name, episode_dir=str(run / name),
                                replay_dir=str(run / ('replay_' + name))))


def check_pins(manifest):
    for path, expected in manifest['frozen_files_sha256'].items():
        require(sha(Path(path)) == expected, f'Frozen file drift: {path}')
    require(sha(SCRIPT) == manifest['finish_script_sha256'], 'Finish script drift')
    check_untouched()


def prepare():
    require(absent(OUT), 'Continuation output already exists')
    old = read_json(PRIOR / 'launch.json')
    prior_manifest = read_json(PRIOR / 'manifest.json')
    plan = read_json(ORIGINAL / 'plan.json')
    original = read_json(ORIGINAL / 'launch.json')
    for launch in (original, old):
        require(launch['status'] == 'complete_with_issues' and launch.get('ended'),
                'Previous supervisor incomplete')
        require(dead(launch['pid']), 'Previous supervisor still alive')
    require(old['prior_plan_sha256'] == original['plan_sha256'] == sha(ORIGINAL / 'plan.json'),
            'Original plan changed')
    require(old['prior_launch_sha256'] == sha(ORIGINAL / 'launch.json'), 'Original launch changed')
    require(old['prior_source_sha256'] == plan['source_sha256'] == original['source_sha256'],
            'Original source manifests disagree')
    require(all(old.get(key) == value for key, value in prior_manifest.items() if key != 'jobs'),
            'Previous manifest/launch mismatch')
    require(old['report_source_sha256'] == REPORT_SHA, 'Reporter pin changed')
    require(GPUS[1] == GPU_UUID and EGL_DEVICES[1] == 2, 'GPU1/EGL identity changed')
    require(old['actor_snapshot'] == SNAPSHOT and Path(SNAPSHOT).is_dir(), 'Checkpoint changed')
    frozen = {}

    def pin(path, expected=None):
        actual = sha(path)
        require(expected is None or actual == expected, f'Source/evidence mismatch: {path}')
        frozen[str(path)] = actual

    for run, names in ((ORIGINAL, ('plan.json', 'launch.json')),
                       (PRIOR, ('manifest.json', 'launch.json', 'decision.json'))):
        for name in names:
            pin(run / name)
    live_sources = dict(plan['source_sha256'])
    live_sources[REPORT] = REPORT_SHA
    live_sources['scripts/resume_native_long_cohort.py'] = old['continuation_script_sha256']
    for name, expected in plan['source_sha256'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe source path')
        pin(ORIGINAL / 'source' / name, expected)
    for name, expected in live_sources.items():
        pin(PRIOR / 'source' / name, expected)
        pin(REPO / name, expected)
    live_sources['scripts/finish_native_long_cohort.py'] = sha(SCRIPT)
    pin(SCRIPT)
    for path, expected in old['original_evidence_sha256'].items():
        pin(Path(path), expected)

    require(len(plan['jobs']) == len(old['jobs']) == len(prior_manifest['jobs']) == 24,
            'Expected24jobs')
    require({(j['task'], j['init_index']) for j in old['jobs']} == IDENTITIES,
            'Cohort identities changed')
    combined, observed = [], []
    for base, previous_plan, before in zip(plan['jobs'], prior_manifest['jobs'], old['jobs']):
        name = base['name']
        require(before['name'] == previous_plan['name'] == name, 'Prior order changed')
        for key in base:
            require(before[key] == previous_plan[key], f'Previous frozen job changed: {name}/{key}')
        for key in ('task', 'init_index', 'gpu', 'egl_device'):
            require(before[key] == base[key], f'Original identity changed: {name}/{key}')
        origin = ORIGINAL if name in {f'{t}_root{r}' for t in ('microwave', 'two_pots')
                                     for r in range(3)} else PRIOR
        episode, replay = str(origin / name), str(origin / ('replay_' + name))
        require(before['episode_dir'] == episode and before['replay_dir'] == replay,
                'Previous output identity mismatch')
        prefix = ['/home/ami/miniconda3/bin/python', '-B', '-u',
                  str(REPO / 'scripts/run_native_long_episode.py')]
        require(before['command'] == prefix + ['--task', TASK_ARGS[base['task']],
                '--init-index', str(base['init_index']), '--seed', '0', '--gpu', str(base['gpu']),
                '--egl-device', str(base['egl_device']), '--steps', '520', '--snapshot', SNAPSHOT,
                '--out', episode], 'Actor command changed')
        require(before['replay_command'] == prefix + ['--replay', episode, '--out', replay,
                '--gpu', str(base['gpu']), '--egl-device', str(base['egl_device'])],
                'Replay command changed')
        for key in ('command', 'replay_command'):
            require(before[key] == [episode if v == base['episode_dir'] else
                    replay if v == base['replay_dir'] else v for v in base[key]],
                    'Command differs from original beyond outputs')
        if name in NAMES:
            require(before['status'] == 'blocked_after_infrastructure_issue' and
                    before['blocked_by'] == 'two_pots_root3', 'Unexpected continuation state')
            require(not set(before).intersection({'started', 'episode_pid', 'replay_pid',
                    'actor_returncode', 'replay_returncode', 'episode_status'}), 'Cell already attempted')
            require(before['gpu'] == 1 and before['egl_device'] == 2 and before['init_index'] == 3,
                    'Continuation must use GPU1/EGL2/root3')
            new = deepcopy(base)
            new.update(episode_dir=str(OUT / name), replay_dir=str(OUT / ('replay_' + name)),
                       status='planned', prior_status=before['status'], prior_blocked_by=before['blocked_by'])
            for key in ('command', 'replay_command'):
                new[key] = [new['episode_dir'] if v == episode else
                            new['replay_dir'] if v == replay else v for v in before[key]]
            combined.append(new)
        else:
            require(before['status'] == ('infrastructure_or_replay_issue' if name in ISSUES
                                        else 'exact_verified'), 'Previous result reclassified')
            require(before['actor_returncode'] == 0 and before.get('ended'), 'Previous actor incomplete')
            for key in ('episode_pid', 'replay_pid'):
                require(dead(before[key]), f'Previous worker alive: {name}/{key}')
            status = read_json(Path(episode) / 'status.json')
            require(status == before['episode_status'], 'Previous episode evidence changed')
            check_episode(before, status)
            proof = read_json(Path(replay) / 'verification.json')
            require(proof['source'] == episode, 'Previous replay source mismatch')
            if name in ISSUES:
                require(proof['status'] == 'replay_mismatch' and proof.get('mismatches') and
                        before['replay_returncode'] != 0, 'Previous issue evidence missing')
            else:
                require(before['replay_returncode'] == 0, 'Previous exact replay returncode changed')
                check_replay(before, status, proof)
            for path in (Path(episode) / 'status.json', Path(episode) / 'steps.jsonl',
                         Path(replay) / 'verification.json'):
                pin(path)
            combined.append(deepcopy(before))
            observed.append(name)
    require(len(observed) == 22 and [j['name'] for j in combined if j['name'] in NAMES] == NAMES,
            'Expected exactly22references and2untouched cells')
    check_untouched()
    report = list(old['report_command'])
    require(argument(report, '--run') == str(PRIOR) and argument(report, '--out') == str(PRIOR / 'report')
            and str(REPO / REPORT) in report, 'Unexpected report command')
    report[report.index('--run') + 1] = str(OUT)
    report[report.index('--out') + 1] = str(OUT / 'report')
    return dict(schema_version=1, scope='L1_native_actor_final_two_untouched', out=str(OUT),
        original_run=str(ORIGINAL), prior_run=str(PRIOR), planned=24, continuation_planned=2,
        referenced_original=22, continuation_names=NAMES, observed_names=observed, jobs=combined,
        original_issues_retained=ISSUES, source_sha256=live_sources, frozen_files_sha256=frozen,
        finish_script_sha256=sha(SCRIPT), report_source_sha256=REPORT_SHA, actor_snapshot=SNAPSHOT,
        gpu_uuids={'1': GPU_UUID}, gpu_to_egl={'1': 2}, budgets=deepcopy(old['budgets']),
        report_command=report, reviewed_decision=dict(decision='continue_untouched_2',
            reviewed_by='coordinator explicit instruction in session ses_eaa6c9047ffeDUHigUkcKa6L8s',
            reviewed_reason='Complete only two untouched root3 cells after the retained wrist-only '
                'two_pots_root3 rendering replay issue; retain all four old strict replay issues. '
                'Same frozen actor, fresh strict replay per cell, failure stops remaining cell.'))


def check_report(manifest, report, preflight=False):
    cases = report['episodes']
    require(len(cases) == 24 and {(c['task'], c['root']) for c in cases} == IDENTITIES,
            'Report omitted/duplicated cohort identities')
    by_name = {f"{c['task']}_root{c['root']}": c for c in cases}
    for job in manifest['jobs']:
        case = by_name[job['name']]
        require(case['artifacts']['directory'] == job['episode_dir'], 'Report artifact path mismatch')
        if job['name'] in manifest['observed_names']:
            require(case['lifecycle'] == 'completed' and case['supervisor']['status'] == job['status'],
                    'Report failed to retain previous completed episode')
        elif preflight:
            require(case['lifecycle'] != 'completed', 'Untouched episode unexpectedly completed')
        if job['name'] in ISSUES:
            require(case['replay']['state'] == 'issues' and not case['selection_evidence_usable'],
                    'Report discarded previous strict replay issue')


def validate(evidence):
    manifest = prepare()
    require(json.loads(json.dumps(manifest)) == manifest, 'Manifest JSON roundtrip changed values')
    require(evidence not in (OUT, ORIGINAL, PRIOR) and
            not any(run in evidence.parents for run in (OUT, ORIGINAL, PRIOR)), 'Unsafe evidence directory')
    evidence.mkdir(parents=True, exist_ok=False)
    write_json(evidence / 'launch.json', dict(manifest, status='preflight_only'))
    command = list(manifest['report_command'])
    command[command.index('--run') + 1] = str(evidence)
    command[command.index('--out') + 1] = str(evidence / 'report')
    with (evidence / 'report.log').open('x') as log:
        subprocess.run(command, cwd=REPO, stdin=subprocess.DEVNULL, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=180)
    check_report(manifest, read_json(evidence / 'report' / 'summary.json'), preflight=True)
    check_pins(manifest)
    require(absent(OUT), 'Preflight touched continuation output')
    manifest['preflight'] = dict(evidence=str(evidence), completed_references=22,
        untouched_cells=NAMES, model_queries=0, robot_actions=0, gpu_access=False,
        summary_sha256=sha(evidence / 'report' / 'summary.json'))
    write_json(evidence / 'manifest.json', manifest)
    (evidence / 'manifest.sha256').write_text(sha(evidence / 'manifest.json') + '\n')
    print(json.dumps(dict(status='validated', manifest=str(evidence / 'manifest.json'),
        manifest_sha256=sha(evidence / 'manifest.json'), script_sha256=sha(SCRIPT),
        referenced_original=22, untouched=NAMES, retained_issues=ISSUES)), flush=True)


def launch(manifest_path):
    manifest_hash = sha(manifest_path)
    require(manifest_path.with_suffix('.sha256').read_text().strip() == manifest_hash,
            'Preparation manifest changed')
    manifest = read_json(manifest_path)
    current = prepare()
    require({k: v for k, v in manifest.items() if k != 'preflight'} == current,
            'Preparation manifest stale')
    evidence = Path(manifest['preflight']['evidence'])
    require(sha(evidence / 'report' / 'summary.json') == manifest['preflight']['summary_sha256'],
            'Preflight report drift')
    check_report(manifest, read_json(evidence / 'report' / 'summary.json'), preflight=True)
    raw = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid',
                                   '--format=csv,noheader'], text=True, timeout=30).strip()
    require(raw == GPU_UUID, 'PhysicalGPU1 identity mismatch')
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'manifest.json').write_bytes(manifest_path.read_bytes())
    for name, expected in manifest['source_sha256'].items():
        dest = OUT / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((REPO / name).read_bytes())
        require(sha(dest) == expected, f'Archived source drift: {name}')
    status = dict(manifest, jobs=deepcopy(manifest['jobs']), status='running', pid=os.getpid(),
        started=time.time(), manifest_sha256=manifest_hash,
        git=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip())
    active = None

    def save():
        write_json(OUT / 'launch.tmp', status)
        (OUT / 'launch.tmp').replace(OUT / 'launch.json')

    def change(job, **fields):
        job.update(fields)
        save()

    def cleanup():
        if active is not None:
            try:
                os.killpg(active.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def terminate(signum, frame):
        cleanup()
        change(status, status='interrupted', signal=signum, ended=time.time())
        raise SystemExit(128 + signum)

    def run(command, logname, timeout, job, phase, env):
        nonlocal active
        with (OUT / logname).open('x') as log:
            active = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                change(job, **{phase + '_pid': active.pid})
                return active.wait(timeout=timeout)
            finally:
                cleanup()
                active.wait()
                active = None

    def report(phase, preflight=False):
        check_pins(manifest)
        code = run(manifest['report_command'], phase + '.log', 180, status, phase, dict(os.environ))
        change(status, **{phase + '_returncode': code})
        require(code == 0, f'{phase} failed: {code}')
        check_report(manifest, read_json(OUT / 'report' / 'summary.json'), preflight=preflight)

    handlers = {sig: signal.signal(sig, terminate) for sig in (signal.SIGTERM, signal.SIGINT)}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU_UUID, MUJOCO_EGL_DEVICE_ID='2',
        MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false', TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2',
        PYTHONOPTIMIZE='0')
    try:
        save()
        report('report_preflight', preflight=True)
        blocked = None
        for job in status['jobs']:
            if job['name'] not in NAMES:
                continue
            if blocked:
                change(job, status='blocked_after_infrastructure_issue', blocked_by=blocked)
                continue
            try:
                check_pins(manifest)
                untouched(OUT, job)
                change(job, status='running', started=time.time())
                code = run(job['command'], job['name'] + '.log', 1800, job, 'episode', env)
                change(job, actor_returncode=code)
                require(code == 0, f'Actor worker returned {code}')
                episode = read_json(Path(job['episode_dir']) / 'status.json')
                check_episode(job, episode)
                change(job, status='replaying', episode_status=episode)
                check_pins(manifest)
                code = run(job['replay_command'], 'replay_' + job['name'] + '.log',
                           900, job, 'replay', env)
                change(job, replay_returncode=code)
                require(code == 0, f'Replay worker returned {code}')
                check_replay(job, episode, read_json(Path(job['replay_dir']) / 'verification.json'))
                change(job, status='exact_verified', ended=time.time())
            except Exception:
                blocked = job['name']
                change(job, status='infrastructure_or_replay_issue', error=traceback.format_exc(),
                       ended=time.time())
        change(status, status='reporting', rollouts_ended=time.time(), rollouts_verified=False,
            continuation_verified=all(j['status'] == 'exact_verified' for j in status['jobs']
                                      if j['name'] in NAMES))
        report('report')
    except Exception:
        change(status, supervisor_error=traceback.format_exc())
    finally:
        cleanup()
        change(status, status='interrupted' if status['status'] == 'interrupted'
               else 'complete_with_issues', ended=time.time())
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    print(json.dumps(dict(status=status['status'], continuation_verified=status.get('continuation_verified'),
                          run=str(OUT))), flush=True)
    # Four previous strict replay issues remain even if both new cells pass.
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--validate-only', action='store_true')
    mode.add_argument('--launch', action='store_true')
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    if args.validate_only:
        require(args.evidence is not None and args.manifest is None, 'Validation needs --evidence only')
        validate(args.evidence.resolve())
        return 0
    require(args.manifest is not None and args.evidence is None, 'Launch needs --manifest only')
    return launch(args.manifest.resolve())


if __name__ == '__main__':
    raise SystemExit(main())
