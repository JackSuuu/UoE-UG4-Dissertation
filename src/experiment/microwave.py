"""Microwave development protocol; a thin wrapper around the single runtime.

Scene selection is experimental setup, not a task adapter. The planner still
generates all task/recovery semantics from the native instruction and images.
"""
import argparse

from .run_task import main as run_task, parse_args as task_args


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__ + '\nShared arguments: python -m src.experiment.run_task --help')
    parser.add_argument('--arm', choices=('baseline', 'measured', 'context', 'full'), required=True)
    args, rest = parser.parse_known_args(argv)
    if any(arg.split('=')[0] in ('--suite', '--task-index', '--mode', '--monitor-mode') for arg in rest):
        parser.error('Microwave protocol owns suite/task/mode/monitor-mode; choose --arm instead')
    mode = 'baseline' if args.arm == 'baseline' else 'framework'
    forwarded = ['--suite', 'libero_10', '--task-index', '9', '--mode', mode,
                 '--monitor-mode', 'gate' if args.arm == 'full' else 'log', *rest]
    runtime = task_args(forwarded)
    if args.arm in ('context', 'full') and not runtime.visual_checkpoint:
        parser.error('context/full arms require --visual-checkpoint and --visual-source')
    if args.arm == 'measured' and (runtime.visual_checkpoint or runtime.scorer_factory
                                   or runtime.candidate_slot is not None):
        parser.error('measured arm excludes visual/learned/retrieval modules')
    if args.arm == 'context' and (runtime.risk_checkpoint or runtime.recovery_checkpoint
                                 or runtime.handoff_checkpoint or runtime.scorer_factory):
        parser.error('context arm isolates visual/retrieval/prediction without learned risk/A/B')
    if args.arm == 'full' and not all((runtime.risk_checkpoint, runtime.memory_bank,
                                      runtime.recovery_checkpoint, runtime.handoff_checkpoint)):
        parser.error('full requires compatible trained risk/A/B and a training memory bank; no fallback')
    return forwarded


def main(argv=None):
    run_task(parse_args(argv))


if __name__ == '__main__':
    main()
