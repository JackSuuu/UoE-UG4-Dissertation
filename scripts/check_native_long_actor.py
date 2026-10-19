"""One pinned model query on a saved native observation; zero robot actions."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from run_native_long_episode import PinnedActor, cleanup_actor, sha, SNAPSHOT
from run_fridge_r1 import GPUS


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES=GPUS[args.gpu],
                      HF_MODULES_CACHE=str(args.out / 'hf_modules'),
                      TOKENIZERS_PARALLELISM='false')
    source = json.loads((args.scene / 'status.json').read_text())
    with np.load(args.scene / 'images.npz') as images:
        raw = images['agentview_image'][0]
    actor = PinnedActor.__new__(PinnedActor)
    start = time.monotonic()
    report = dict(scope='saved_native_observation_model_interface_preflight',
                  scene=str(args.scene), source_status_sha256=sha(args.scene / 'status.json'),
                  robot_actions=0, model_queries=0, status='started')
    try:
        PinnedActor.__init__(actor, args, args.out)
        report['model_load_s'] = time.monotonic() - start
        report['actor_build'] = actor.build
        report['model_queries'] = 1
        start = time.monotonic()
        action, decode, image = actor.act(raw, source['language'])
        report['query_wall_s'] = time.monotonic() - start
        native = np.asarray(decode['native_action'])
        expected = native.copy()
        expected[-1] = -np.sign(2 * expected[-1] - 1)
        if action.shape != (7,) or not np.isfinite(action).all() or len(decode['tokens']) != 7:
            raise ValueError('Invalid action/token contract')
        np.testing.assert_array_equal(action, expected)
        image.save(args.out / 'policy_input.png')
        report.update(status='passed', mapped_action=action.tolist(), decode=decode)
    finally:
        cleanup_actor(actor)
        (args.out / 'check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=[1, 2, 3], default=1)
    parser.add_argument('--snapshot', default=str(SNAPSHOT))
    parser.add_argument('--pinned-python', default='/home/ami/scratch/openvla_tf440_audit_20261019/venv/bin/python')
    main(parser.parse_args())
