"""Explicit, one-shot continuation of the 18 untouched native L1 cells.

Dry inspection: --out NEW_DIRECTORY --validate-only (no GPU access or writes).
Launch requires --launch --decision-note NOTE.json. The coordinator's new JSON
note must contain decision="continue_untouched_18", reviewed_by, reviewed_reason,
out, prior_plan_sha256, prior_launch_sha256, continuation_script_sha256 and
report_source_sha256. Dry validation prints the hashes needed to freeze the note.
Only the reporting source may differ from the original live source hashes; its
reviewed hash is pinned separately. Original archived sources are all checked.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import traceback

from run_native_long_cohort import (REPO, GPUS, EGL_DEVICES, TASK_ARGS, TASK_NAMES,
                                    argument, require, sha)
from run_fridge_r1 import SNAPSHOT


PRIOR = Path('/home/ami/scratch/native_long_l1_20261019_a')
SCRIPT = Path(__file__).resolve()
REPORT = 'scripts/summarize_native_long_cohort.py'
OBSERVED = {(task, root) for task in ('microwave', 'two_pots') for root in range(3)}
IDENTITIES = {(task, root) for task in TASK_ARGS for root in range(6)}


def read_json(path):
    return json.loads(path.read_text())


def absent(path):
    return not path.exists() and not path.is_symlink()


def dead(pid):
    require(type(pid) is int and pid > 1, 'Invalid prior PID')
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def check_sources(prior, hashes, report_hash):
    for name, expected in hashes.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,
                'Unsafe source path')
        require(sha(prior / 'source' / name) == expected, f'Archived source drift: {name}')
        require(sha(REPO / name) == (report_hash if name == REPORT else expected),
                f'Live source drift: {name}')


def untouched(prior, job):
    for path in (Path(job['episode_dir']), Path(job['replay_dir']),
                 prior / (job['name'] + '.log'), prior / ('replay_' + job['name'] + '.log')):
        require(absent(path), f'Previously touched cell: {path}')


def prepare(prior, out, note_path=None):
    """Read-only validation and manifest construction; never imports the actor."""
    require(prior.is_absolute() and out.is_absolute(), 'Absolute run paths required')
    require(absent(out) and prior not in out.parents, 'Output must be new and outside prior run')
    plan, old = read_json(prior / 'plan.json'), read_json(prior / 'launch.json')
    require(old['status'] == 'complete_with_issues' and old.get('ended'),
            'Prior supervisor has not terminated with issues')
    require(dead(old['pid']), 'Prior supervisor PID is still alive')
    require(sha(prior / 'plan.json') == old['plan_sha256'], 'Prior plan hash mismatch')
    require(plan['source_sha256'] == old['source_sha256'], 'Prior source manifests disagree')
    hashes = plan['source_sha256']
    require({'scripts/run_native_long_episode.py', 'scripts/run_native_long_cohort.py',
             'scripts/run_fridge_r1.py', REPORT}.issubset(hashes), 'Missing frozen sources')
    report_hash = sha(REPO / REPORT)
    check_sources(prior, hashes, report_hash)
    require(plan['actor_snapshot'] == SNAPSHOT, 'Checkpoint changed')
    require(Path(SNAPSHOT).is_dir(), 'Frozen checkpoint missing')
    jobs, previous = plan['jobs'], old['jobs']
    require(len(jobs) == len(previous) == 24, 'Expected original 24 jobs')
    require({(j['task'], j['init_index']) for j in jobs} == IDENTITIES,
            'Wrong original cohort identities')
    combined, resumed, evidence = [], [], {}
    for job, before in zip(jobs, previous):
        require(all(before.get(key) == value for key, value in job.items()),
                'Prior launch differs from frozen plan')
        task, root, gpu = job['task'], job['init_index'], job['gpu']
        name = f'{task}_root{root}'
        require(job['name'] == name and gpu == 1 + root % 3 and
                job['egl_device'] == EGL_DEVICES[gpu], 'Wrong job/lane identity')
        episode, replay = str(prior / name), str(prior / ('replay_' + name))
        require(job['episode_dir'] == episode and job['replay_dir'] == replay,
                'Prior output identity mismatch')
        prefix = ['/home/ami/miniconda3/bin/python', '-B', '-u',
                  str(REPO / 'scripts/run_native_long_episode.py')]
        require(job['command'] == prefix + ['--task', TASK_ARGS[task], '--init-index', str(root),
                '--seed', '0', '--gpu', str(gpu), '--egl-device', str(EGL_DEVICES[gpu]),
                '--steps', '520', '--snapshot', SNAPSHOT, '--out', episode],
                'Frozen actor arguments changed')
        require(job['replay_command'] == prefix + ['--replay', episode, '--out', replay,
                '--gpu', str(gpu), '--egl-device', str(EGL_DEVICES[gpu])],
                'Frozen replay arguments changed')
        if (task, root) in OBSERVED:
            expected = 'exact_verified' if task == 'microwave' else 'infrastructure_or_replay_issue'
            require(before['status'] == expected, 'Original result reclassified')
            require(before['actor_returncode'] == 0 and before.get('ended'), 'Original actor incomplete')
            for key in ('episode_pid', 'replay_pid'):
                require(dead(before[key]), f'Prior worker still alive: {name}/{key}')
            episode_status = read_json(Path(episode) / 'status.json')
            require(episode_status == before['episode_status'] and
                    episode_status['status'] == 'completed', 'Original episode evidence changed')
            proof = read_json(Path(replay) / 'verification.json')
            require(proof['source'] == episode, 'Original replay source mismatch')
            if task == 'two_pots':
                require(proof['status'] == 'replay_mismatch' and proof.get('mismatches') and
                        before['replay_returncode'] != 0, 'Original mismatch evidence missing')
            else:
                require(proof['status'] == 'exact_replay_passed' and not proof.get('mismatches')
                        and before['replay_returncode'] == 0, 'Original exact replay changed')
            for path in (Path(episode) / 'status.json', Path(episode) / 'steps.jsonl',
                         Path(replay) / 'verification.json'):
                evidence[str(path)] = sha(path)
            combined.append(deepcopy(before))
        else:
            require(before['status'] in ('planned', 'unstarted', 'blocked_after_infrastructure_issue'),
                    'Continuation contains a started cell')
            require(not set(before).intersection({'started', 'episode_pid', 'replay_pid',
                    'actor_returncode', 'replay_returncode', 'episode_status'}), 'Cell already attempted')
            untouched(prior, job)
            new = deepcopy(job)
            new.update(episode_dir=str(out / name), replay_dir=str(out / ('replay_' + name)),
                       status='planned', prior_status=before['status'], prior_blocked_by=before.get('blocked_by'))
            for key in ('command', 'replay_command'):
                new[key] = [new['episode_dir'] if value == episode else
                            new['replay_dir'] if value == replay else value for value in job[key]]
            combined.append(new)
            resumed.append(name)
    require(len(resumed) == 18 and len(set(resumed)) == 18, 'Expected exactly 18 untouched cells')
    pins = dict(prior_plan_sha256=sha(prior / 'plan.json'),
                prior_launch_sha256=sha(prior / 'launch.json'),
                continuation_script_sha256=sha(SCRIPT), report_source_sha256=report_hash)
    note = None
    if note_path is not None:
        require(prior not in note_path.parents, 'Decision must be a new external note')
        note = read_json(note_path)
        require(note.get('decision') == 'continue_untouched_18', 'No explicit coordinator decision')
        require(all(isinstance(note.get(k), str) and note[k].strip()
                    for k in ('reviewed_by', 'reviewed_reason')), 'Missing reviewed reason/reviewer')
        require(note.get('out') == str(out), 'Decision output directory mismatch')
        require(all(note.get(k) == v for k, v in pins.items()), 'Decision hashes are stale')
    report_command = list(plan['report_command'])
    require(str(REPO / REPORT) in report_command and
            argument(report_command, '--run') == str(prior) and
            argument(report_command, '--out') == str(prior / 'report'), 'Wrong original report command')
    report_command[report_command.index('--run') + 1] = str(out)
    report_command[report_command.index('--out') + 1] = str(out / 'report')
    return dict(schema_version=1, scope='L1_native_actor_explicit_untouched_continuation',
        prior_run=str(prior), out=str(out), planned=24, continuation_planned=18,
        referenced_original=6, continuation_names=resumed, jobs=combined,
        prior_source_sha256=hashes, original_evidence_sha256=evidence,
        gpu_uuids=GPUS, gpu_to_egl=EGL_DEVICES, actor_snapshot=SNAPSHOT,
        budgets=dict(initialization_actions=10, actor_queries=520, model_actions=520,
                     scene_episode_wall_seconds=1200, actor_process_seconds=1800,
                     replay_process_seconds=900, report_seconds=180),
        original_issues_retained=[j['name'] for j in combined
                                  if j['status'] == 'infrastructure_or_replay_issue'],
        decision_note=str(note_path) if note_path else None,
        decision_note_sha256=sha(note_path) if note_path else None,
        reviewed_decision=note, report_command=report_command, **pins)


def check_pins(manifest):
    prior = Path(manifest['prior_run'])
    for name, key in (('plan.json', 'prior_plan_sha256'), ('launch.json', 'prior_launch_sha256')):
        require(sha(prior / name) == manifest[key], f'Prior {name} changed')
    check_sources(prior, manifest['prior_source_sha256'], manifest['report_source_sha256'])
    require(sha(SCRIPT) == manifest['continuation_script_sha256'], 'Continuation source drift')
    require(sha(Path(manifest['decision_note'])) == manifest['decision_note_sha256'], 'Decision drift')
    for path, expected in manifest['original_evidence_sha256'].items():
        require(sha(Path(path)) == expected, f'Original evidence changed: {path}')


def check_episode(job, episode):
    require(episode['status'] == 'completed', 'Actor incomplete')
    args = episode['args']
    require(args['task'] == TASK_NAMES[job['task']] and args['init_index'] == job['init_index']
            and args['gpu'] == job['gpu'] and args['egl_device'] == job['egl_device'],
            'Episode task/root/GPU identity mismatch')
    require(args['steps'] == 520 and args['seed'] == 0 and args['snapshot'] == SNAPSHOT
            and not args['smoke'] and args['replay'] is None, 'Episode protocol mismatch')
    require(Path(args['out']).resolve() == Path(job['episode_dir']), 'Episode output mismatch')
    require(episode['initialization_actions'] == 10 and
            0 <= episode['steps_executed'] <= episode['actor_queries'] <= 520,
            'Episode action/query budget mismatch')
    require(episode['protocol']['wall_seconds'] == 1200, 'Episode wall budget mismatch')


def check_replay(job, episode, replay):
    require(replay['status'] == 'exact_replay_passed', 'No exact replay proof')
    require(not replay.get('mismatches') and replay.get('max_state_gap') == 0,
            'Replay mismatches or missing state evidence')
    require(replay.get('max_integration_gap') == 0 and replay['steps'] == episode['steps_executed'],
            'Replay integration/step mismatch')
    require(Path(replay['source']).resolve() == Path(job['episode_dir']), 'Replay source mismatch')
    require(replay.get('actor_queries') == 0, 'Replay queried the actor')


def check_report(manifest, report):
    cases = report['episodes']
    require(len(cases) == 24 and {(c['task'], c['root']) for c in cases} == IDENTITIES,
            'Report omitted or duplicated cohort identities')
    by_name = {f"{c['task']}_root{c['root']}": c for c in cases}
    for job in manifest['jobs']:
        case = by_name[job['name']]
        require(case['artifacts']['directory'] == job['episode_dir'], 'Report artifact path mismatch')
        if (job['task'], job['init_index']) in OBSERVED:
            require(case['lifecycle'] == 'completed', 'Report failed to read original episode')
            require(case['supervisor']['status'] == job['status'], 'Report reclassified original status')
        if job['name'] in manifest['original_issues_retained']:
            require(case['replay']['state'] == 'issues' and not case['selection_evidence_usable'],
                    'Report discarded original replay issue')


def launch(manifest):
    require(manifest['reviewed_decision'] is not None, 'Launch requires coordinator decision note')
    check_pins(manifest)
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                   '--format=csv,noheader'], text=True)
    actual = {int(line.split(',')[0]): line.split(',')[1].strip() for line in raw.splitlines()}
    require(all(actual.get(g) == GPUS[g] for g in (1, 2, 3)), 'Physical GPU identity mismatch')
    out = Path(manifest['out'])
    out.mkdir(parents=True, exist_ok=False)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'decision.json').write_bytes(Path(manifest['decision_note']).read_bytes())
    for name in set(manifest['prior_source_sha256']) | {'scripts/resume_native_long_cohort.py'}:
        dest = out / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((REPO / name).read_bytes())
    status = dict(manifest, jobs=deepcopy(manifest['jobs']), status='running', pid=os.getpid(),
                  started=time.time(), git=subprocess.check_output(
                      ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip())
    lock, active_lock = threading.RLock(), threading.Lock()
    active, stopping = set(), threading.Event()

    def save():
        tmp = out / 'launch.tmp'
        tmp.write_text(json.dumps(status, indent=2) + '\n')
        tmp.replace(out / 'launch.json')

    def change(job, **fields):
        with lock:
            job.update(fields)
            save()

    def cleanup(pid, sig=signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            pass

    def terminate(signum, frame):
        stopping.set()
        with lock:
            status.update(status='interrupted', signal=signum, interrupted=time.time())
            save()
        with active_lock:
            for pid in active:
                cleanup(pid)
        raise SystemExit(128 + signum)

    def run_command(command, env, log_path, timeout, job, phase):
        with log_path.open('x') as log:
            with active_lock:
                require(not stopping.is_set(), 'Supervisor stopping')
                child = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active.add(child.pid)
            try:
                change(job, **{phase + '_pid': child.pid})
                return child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                cleanup(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    cleanup(child.pid)
                raise
            finally:
                cleanup(child.pid)
                child.wait()
                with active_lock:
                    active.discard(child.pid)

    def lane(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS[gpu],
            MUJOCO_EGL_DEVICE_ID=str(EGL_DEVICES[gpu]), MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
            OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
            TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2', PYTHONOPTIMIZE='0')
        blocked = None
        for job in status['jobs']:
            if job['gpu'] != gpu or job['name'] not in manifest['continuation_names']:
                continue
            if blocked or stopping.is_set():
                change(job, status='blocked_after_infrastructure_issue', blocked_by=blocked)
                continue
            try:
                check_pins(manifest)
                original = dict(job, episode_dir=str(Path(manifest['prior_run']) / job['name']),
                                replay_dir=str(Path(manifest['prior_run']) / ('replay_' + job['name'])))
                untouched(Path(manifest['prior_run']), original)
                require(absent(Path(job['episode_dir'])) and absent(Path(job['replay_dir'])),
                        'Continuation output already touched')
                change(job, status='running', started=time.time())
                code = run_command(job['command'], env, out / (job['name'] + '.log'), 1800, job, 'episode')
                change(job, actor_returncode=code)
                require(code == 0, f'Actor worker returned {code}')
                episode = read_json(Path(job['episode_dir']) / 'status.json')
                check_episode(job, episode)
                change(job, status='replaying', episode_status=episode)
                check_pins(manifest)
                code = run_command(job['replay_command'], env, out / ('replay_' + job['name'] + '.log'),
                                   900, job, 'replay')
                change(job, replay_returncode=code)
                require(code == 0, f'Replay worker returned {code}')
                check_replay(job, episode, read_json(Path(job['replay_dir']) / 'verification.json'))
                change(job, status='exact_verified', ended=time.time())
            except Exception:
                blocked = job['name']
                change(job, status='infrastructure_or_replay_issue', error=traceback.format_exc(), ended=time.time())

    previous_handlers = {sig: signal.signal(sig, terminate) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with lock:
            save()
        # Confirm the reporter can read all six absolute external episode paths
        # before spending any actor queries. The same report is refreshed at end.
        try:
            code = run_command(manifest['report_command'], dict(os.environ),
                               out / 'report_preflight.log', 180, status, 'report_preflight')
            change(status, report_preflight_returncode=code)
            require(code == 0, f'Report preflight returned {code}')
            check_report(manifest, read_json(out / 'report' / 'summary.json'))
        except Exception:
            change(status, status='complete_with_issues', report_error=traceback.format_exc(),
                   ended=time.time(), continuation_verified=False)
            return 1
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(lane, gpu) for gpu in (1, 2, 3)]
            for future in futures:
                future.result()
        with lock:
            status.update(status='reporting', rollouts_ended=time.time(), rollouts_verified=False,
                continuation_verified=all(j['status'] == 'exact_verified' for j in status['jobs']
                                          if j['name'] in manifest['continuation_names']))
            save()
        try:
            check_pins(manifest)
            code = run_command(manifest['report_command'], dict(os.environ), out / 'report.log',
                               180, status, 'report')
            change(status, report_returncode=code)
            require(code == 0, f'Report returned {code}')
            check_report(manifest, read_json(out / 'report' / 'summary.json'))
        except Exception:
            change(status, report_error=traceback.format_exc())
        change(status, status='complete_with_issues', ended=time.time())
        print(json.dumps(dict(status=status['status'], continuation_verified=status['continuation_verified'],
                              run=str(out))), flush=True)
        # Original replay issues remain cohort issues even if all new cells pass.
        return 1
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--decision-note', type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--validate-only', action='store_true')
    mode.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    manifest = prepare(PRIOR, args.out.resolve(), args.decision_note.resolve() if args.decision_note else None)
    if args.validate_only:
        print(json.dumps(dict(status='continuation_validated', launch_authorized=bool(manifest['reviewed_decision']),
            model_queries=0, robot_actions=0, gpu_access=False, manifest=manifest), indent=2))
        return 0
    return launch(manifest)


if __name__ == '__main__':
    raise SystemExit(main())
