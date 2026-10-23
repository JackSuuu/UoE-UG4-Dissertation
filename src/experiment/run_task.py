"""Run a natural-language task with frozen OpenVLA and a vision planner.

Scene selection is benchmark setup. Goals, recovery contracts and instructions
are generated at runtime; no per-task adapter or task config is required.
"""
import argparse
import hashlib
import importlib
import math
import os
from pathlib import Path
import signal
import sys
import traceback

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instruction', help='User task; defaults to the selected benchmark instruction')
    parser.add_argument('--suite', default='libero_10')
    parser.add_argument('--task-index', type=int, required=True)
    parser.add_argument('--init-index', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--policy-checkpoint')
    parser.add_argument('--policy-revision')
    parser.add_argument('--unnorm-key', default='libero_10')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--egl-device', type=int, default=0)
    parser.add_argument('--planner-url', help='Vision chat API base, e.g. http://localhost:8000/v1')
    parser.add_argument('--planner-model')
    parser.add_argument('--planner-timeout', type=float, default=60)
    parser.add_argument('--planner-calls', type=int, default=24)
    parser.add_argument('--rollout-seconds', type=int, default=1200)
    parser.add_argument('--mode', choices=('framework', 'baseline'), default='framework')
    parser.add_argument('--preflight', choices=('scene', 'planner', 'policy', 'components'),
                        help='Check one backend at the initial frame; no policy actions are executed')
    parser.add_argument('--visual-checkpoint', type=Path, help='Pretrained DINOv2 ViT-S/14 weights')
    parser.add_argument('--visual-source', type=Path, help='Explicit DINOv2 source checkout')
    parser.add_argument('--visual-sha256')
    parser.add_argument('--feature-device', default='cpu')
    parser.add_argument('--memory-bank', type=Path, help='Generated-predicate training keyframes JSON')
    parser.add_argument('--risk-checkpoint', type=Path, help='New-schema trained risk head')
    parser.add_argument('--recovery-checkpoint', type=Path, help='New-schema trained Module A')
    parser.add_argument('--handoff-checkpoint', type=Path, help='New-schema trained Module B')
    parser.add_argument('--monitor-mode', choices=('log', 'gate'), default='log')
    parser.add_argument('--candidate-slot', type=int, choices=range(4),
                        help='Training roots 10–19 only: select a fixed first-repair candidate for data collection')
    parser.add_argument('--scorer-factory', help='Optional module:function returning recovery/handoff scorers')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.task_index, args.init_index) < 0 or args.rollout_seconds <= 0:
        parser.error('Indices must be nonnegative and rollout duration positive')
    if args.instruction is not None and not args.instruction.strip():
        parser.error('Instruction must be nonempty')
    if (not math.isfinite(args.planner_timeout) or args.planner_timeout <= 0
            or args.planner_calls < 1):
        parser.error('Planner timeout and call limit must be positive and finite')
    if args.egl_device < 0:
        parser.error('EGL device must be nonnegative')
    if args.preflight in (None, 'policy') and not args.policy_checkpoint:
        parser.error('--policy-checkpoint is required for policy checks and episodes')
    if args.preflight in (None, 'planner', 'components') and not (args.planner_url and args.planner_model):
        parser.error('--planner-url and --planner-model are required for planner checks and episodes')
    if args.scorer_factory and (args.preflight or args.mode == 'baseline'):
        parser.error('Scorers apply only to framework episodes')
    if args.scorer_factory and (args.scorer_factory.count(':') != 1
                               or not all(args.scorer_factory.split(':'))):
        parser.error('--scorer-factory must be module:function')
    if bool(args.visual_checkpoint) != bool(args.visual_source):
        parser.error('--visual-checkpoint and --visual-source must be supplied together')
    if (args.memory_bank or args.risk_checkpoint or args.recovery_checkpoint or args.handoff_checkpoint
            or args.preflight == 'components') and not args.visual_checkpoint:
        parser.error('Memory/learned heads/component checks require a pretrained visual encoder')
    if args.visual_sha256 and not args.visual_checkpoint:
        parser.error('--visual-sha256 needs --visual-checkpoint')
    if args.scorer_factory and (args.recovery_checkpoint or args.handoff_checkpoint):
        parser.error('Choose a scorer factory OR built-in A/B checkpoint loaders')
    if args.mode == 'baseline' and (args.monitor_mode == 'gate' or args.recovery_checkpoint or args.handoff_checkpoint):
        parser.error('Baseline cannot gate or load recovery/handoff scorers')
    if args.preflight in ('scene', 'planner', 'policy') and args.visual_checkpoint:
        parser.error('Use --preflight components to check vision/memory/risk without policy execution')
    if args.candidate_slot is not None and (args.init_index not in range(10, 20) or args.preflight
            or args.mode != 'framework' or args.scorer_factory or args.recovery_checkpoint):
        parser.error('Fixed-slot collection requires a framework training episode (roots 10–19), without another A')
    return args


def load_scorers(factory_path):
    if not factory_path:
        return {}
    module, factory = factory_path.split(':')
    scorers = getattr(importlib.import_module(module), factory)()
    if not isinstance(scorers, dict) or set(scorers) - {'recovery', 'handoff'}:
        raise ValueError('Scorer factory must return a recovery/handoff mapping')
    for name, method in (('recovery', 'score'), ('handoff', 'assess')):
        if name in scorers and not callable(getattr(scorers[name], method, None)):
            raise TypeError(f'{name} scorer must implement {method}')
    return scorers


def load_components(args):
    from src.verification.vision_physical_monitor import VisionPhysicalMonitor
    encoder = memory = risk = None
    if args.visual_checkpoint:
        from src.perception.frozen_visual_encoder import FrozenVisualEncoder
        from src.verification.features import encoder_signature
        encoder = FrozenVisualEncoder(args.visual_checkpoint, args.visual_source,
                                      device=args.feature_device, expected_sha=args.visual_sha256)
        signature = encoder_signature(encoder.provenance)
        if args.memory_bank:
            from src.memory.keyframe_bank import KeyframeBank
            memory = KeyframeBank.load(args.memory_bank, encoder_signature=signature)
        if args.risk_checkpoint:
            from src.verification.calibrated_risk_head import CalibratedRiskHead
            risk = CalibratedRiskHead(args.risk_checkpoint, encoder_signature=signature, device=args.feature_device)
    monitor = VisionPhysicalMonitor(encoder=encoder, memory=memory, risk=risk, mode=args.monitor_mode)
    if args.memory_bank:
        monitor.provenance['memory_file'] = dict(path=str(args.memory_bank),
            sha256=hashlib.sha256(args.memory_bank.read_bytes()).hexdigest())
    scorers = load_scorers(args.scorer_factory)
    if args.candidate_slot is not None:
        from src.experiment.candidate_collection import FixedCandidateSlot
        scorers['recovery'] = FixedCandidateSlot(args.candidate_slot)
    if args.recovery_checkpoint:
        from src.repair.recovery_scorer import RecoveryScorer
        scorers['recovery'] = RecoveryScorer(args.recovery_checkpoint, monitor, device=args.feature_device)
    if args.handoff_checkpoint:
        from src.verification.handoff_verifier import HandoffVerifier
        scorers['handoff'] = HandoffVerifier(args.handoff_checkpoint, monitor, device=args.feature_device)
    return monitor, scorers


def preflight(stage, bridge, instruction, journal, *, planner=None, policy=None, monitor=None):
    """Real backend smoke checks, never a second execution loop."""
    observation = bridge.observe()
    journal.record(dict(type='observation', observation=observation))
    result = dict(preflight=stage, executed_actions=0, policy_queries=0)
    if stage in ('planner', 'components'):
        spec = planner.task(instruction, observation, bridge.capabilities)
        observation = bridge.observe((*spec.goals, *spec.milestones))
        journal.record(dict(type='task_generated', task=spec.to_dict(), observation=observation))
        result.update(task=spec.to_dict(), planner_calls=planner.calls,
                      facts=observation.facts)
        if stage == 'components':
            from src.verification.physical_verifier import PhysicalVerifier
            objects = {e.id for e in observation.entities if e.kind == 'object'}
            target = next((arg for p in spec.goals for arg in p.args if arg in objects), None)
            context = monitor.observe(observation, spec, PhysicalVerifier().evidence(observation, target=target))
            journal.record(dict(type='monitor', observation_index=observation.index, **context))
            result.update(context=context['context'], components=monitor.provenance)
    elif stage == 'policy':
        # Same public policy interface as Session; proposal is not executed.
        from src.interfaces import Proposal
        journal.record(dict(type='policy_query', query=1, observation_index=observation.index,
                            instruction=instruction, image_sha256=observation.to_dict()['image_sha256'],
                            sample=False, seed=0))
        proposal = policy.propose(observation.image_png, instruction)
        if not isinstance(proposal, Proposal):
            raise TypeError('Policy must return a Proposal')
        journal.record(dict(type='policy_response', query=1, action=proposal.action,
                            provenance=proposal.provenance))
        result.update(policy_queries=1, action=proposal.action, policy=proposal.provenance)
    elif stage != 'scene':
        raise ValueError('Unknown preflight stage')
    journal.record(dict(type='preflight_end', **result))
    return result


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
                    initialization_noops=10, limits=dict(actions=520, queries=520, local_repair=32, total_repair=160))
    env = session = None

    def timeout(signum, frame):
        raise RuntimeError('run_wall_budget_exhausted')

    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.rollout_seconds)
    try:
        manifest['source_sha256'] = {}
        for source in sorted((REPO/'src').rglob('*.py')):
            relative, content = str(source.relative_to(REPO)), source.read_bytes()
            manifest['source_sha256'][relative] = hashlib.sha256(content).hexdigest()
            target = args.out/'source'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        journal.save('status.json', manifest)
        monitor, scorers = load_components(args)
        manifest['components'] = monitor.provenance
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
        journal.save('status.json', manifest)
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
        # Force image preprocessing before loading a large policy checkpoint.
        initial = bridge.observe()
        journal.record(dict(type='observation', observation=initial))
        planner = policy = None
        if args.preflight in (None, 'planner', 'components'):
            planner = ChatPlanner(args.planner_url, args.planner_model,
                api_key=os.environ.get('PLANNER_API_KEY'), timeout=args.planner_timeout,
                max_calls=args.planner_calls, record=lambda event: journal.record(dict(type='planner_call', **event)))
        if args.preflight in (None, 'policy'):
            policy = OpenVLAPolicy(args.policy_checkpoint, unnorm_key=args.unnorm_key,
                                   device=args.device, revision=args.policy_revision)
            manifest['policy'] = policy.provenance
        manifest['scorers'] = {name: dict(implementation=type(scorer).__module__ + '.' + type(scorer).__qualname__,
                                         provenance=getattr(scorer, 'provenance', None))
                               for name, scorer in scorers.items()}
        manifest['status'] = 'running'
        journal.save('status.json', manifest)
        if args.preflight:
            result = preflight(args.preflight, bridge, instruction, journal, planner=planner, policy=policy,
                               monitor=monitor)
        else:
            session = Session(bridge, policy, planner, mode=args.mode, record=journal.record,
                              monitor=monitor,
                              recovery_scorer=scorers.get('recovery'), handoff_scorer=scorers.get('handoff'))
            result = session.run(instruction)
        journal.save('result.json', result)
        manifest.update(status='preflight_completed' if args.preflight else 'completed',
                        termination=result.get('termination'))
    except BaseException:
        manifest.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        if session is not None and session.controller is not None:
            manifest['controller'] = session.controller.summary
        cleanup_failed = False
        try:
            if env is not None:
                env.close()
        except Exception:
            cleanup_failed = manifest['status'] != 'failed'
            manifest.update(status='failed', cleanup_error=traceback.format_exc())
        finally:
            try:
                journal.save('status.json', manifest)
            finally:
                journal.close()
        if cleanup_failed:
            raise RuntimeError('Environment cleanup failed; see status.json')


if __name__ == '__main__':
    main()
