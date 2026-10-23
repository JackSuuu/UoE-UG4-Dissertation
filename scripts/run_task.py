"""Run a natural-language task with frozen OpenVLA and a vision planner.

Scene selection is benchmark setup. Goals, recovery contracts and instructions
are generated at runtime; no per-task adapter or task config is required.
"""
import argparse
import hashlib
import importlib
import os
from pathlib import Path
import signal
import sys
import traceback

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instruction', help='User task; defaults to the selected benchmark instruction')
    parser.add_argument('--suite', default='libero_10')
    parser.add_argument('--task-index', type=int, required=True)
    parser.add_argument('--init-index', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--policy-checkpoint', required=True)
    parser.add_argument('--policy-revision')
    parser.add_argument('--unnorm-key', default='libero_10')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--egl-device', type=int, default=0)
    parser.add_argument('--planner-url', required=True, help='Vision chat API base, e.g. http://localhost:8000/v1')
    parser.add_argument('--planner-model', required=True)
    parser.add_argument('--planner-timeout', type=float, default=60)
    parser.add_argument('--planner-calls', type=int, default=24)
    parser.add_argument('--rollout-seconds', type=int, default=1200)
    parser.add_argument('--mode', choices=('framework', 'baseline'), default='framework')
    parser.add_argument('--scorer-factory', help='Optional module:function returning recovery/handoff scorers')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.task_index, args.init_index) < 0 or args.rollout_seconds <= 0:
        parser.error('Indices must be nonnegative and rollout duration positive')
    if args.instruction is not None and not args.instruction.strip():
        parser.error('Instruction must be nonempty')
    return args


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    from src.journal import Journal
    from src.planning.multimodal import ChatPlanner
    from src.runtime import Session
    journal = Journal(args.out)
    manifest = dict(status='started', args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    source_sha256={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [Path(__file__), *sorted((REPO/'src').rglob('*.py'))]},
                    initialization_noops=10, limits=dict(actions=520, queries=520, local_repair=32, total_repair=160))
    for relative in manifest['source_sha256']:
        target = args.out/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO/relative).read_bytes())
    journal.save('status.json', manifest)
    env = session = None
    try:
        import numpy as np
        import torch
        from libero.libero import get_libero_path
        from libero.libero.benchmark import get_benchmark
        from libero.libero.envs import OffScreenRenderEnv
        from src.adapters.libero_environment import LiberoEnvironment
        from src.adapters.openvla_policy import OpenVLAPolicy
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        suite = get_benchmark(args.suite)(task_order_index=0)
        if args.task_index >= suite.get_num_tasks():
            raise ValueError('Task index outside selected suite')
        task = suite.get_task(args.task_index)
        bddl = Path(suite.get_task_bddl_file_path(args.task_index))
        state_file = Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file
        states = torch.load(state_file, weights_only=False, map_location='cpu')
        if args.init_index >= len(states):
            raise ValueError('Initial-state index outside selected task')
        chosen = np.asarray(states[args.init_index])
        np.save(args.out/'selected_initial_state.npy', chosen, allow_pickle=False)
        (args.out/'scene.bddl').write_bytes(bddl.read_bytes())
        manifest.update(task=task.name, bddl_sha256=hashlib.sha256(bddl.read_bytes()).hexdigest(),
                        initial_states_sha256=hashlib.sha256(state_file.read_bytes()).hexdigest())
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                 horizon=531, render_gpu_device_id=args.egl_device)
        env.seed(args.seed)
        # Avoid the wrapper's unlimited reset retry loop.
        env.env.reset()
        raw = env.set_init_state(chosen)
        for _ in range(10):
            raw, _, _, _ = env.step([0., 0., 0., 0., 0., 0., -1.])
        instruction = args.instruction or env.language_instruction
        manifest.update(instruction=instruction, benchmark_instruction=env.language_instruction)
        bridge = LiberoEnvironment(env, raw)
        planner = ChatPlanner(args.planner_url, args.planner_model,
            api_key=os.environ.get('PLANNER_API_KEY'), timeout=args.planner_timeout,
            max_calls=args.planner_calls, record=lambda event: journal.record(dict(type='planner_call', **event)))
        policy = OpenVLAPolicy(args.policy_checkpoint, unnorm_key=args.unnorm_key,
                               device=args.device, revision=args.policy_revision)
        manifest['policy'] = policy.provenance
        scorers = {}
        if args.scorer_factory:
            module, factory = args.scorer_factory.split(':', 1)
            scorers = getattr(importlib.import_module(module), factory)()
            if not isinstance(scorers, dict) or set(scorers) - {'recovery', 'handoff'}:
                raise ValueError('Scorer factory must return a recovery/handoff mapping')
        session = Session(bridge, policy, planner, mode=args.mode, record=journal.record,
                          recovery_scorer=scorers.get('recovery'), handoff_scorer=scorers.get('handoff'))
        manifest['status'] = 'running'
        journal.save('status.json', manifest)

        def timeout(signum, frame):
            # Distinct from the recoverable HTTP timeout in the planner.
            raise RuntimeError('rollout_wall_budget_exhausted')

        previous_handler = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(args.rollout_seconds)
        try:
            result = session.run(instruction)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
        journal.save('result.json', result)
        manifest.update(status='completed', termination=result['termination'])
    except BaseException:
        manifest.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        if session is not None and session.controller is not None:
            manifest['controller'] = session.controller.summary
        journal.save('status.json', manifest)
        journal.close()
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
