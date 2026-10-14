"""Real-simulator geometry fixtures and read-only/replay regression, no actor."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from adapters.libero_completion import ContactRecorder
from adapters.libero_drawer import DrawerMonitor, SubgoalTracker, settled_drawer
from adapters.libero_preservation import SETTLE, fingerprint
from run_libero_drawer import reset_seeds


def main(args):
    passes = []
    reference = None
    # Consecutive create/destroy and instrumented vs uninstrumented real physics.
    for instrumented in [False, True, True]:
        reset_seeds()
        e, obs = settled_drawer(args.root, 0)
        monitor = DrawerMonitor(e)
        rec = ContactRecorder(e.sim, keywords=('akita_black_bowl', 'white_cabinet')) if instrumented else None
        frames = []
        try:
            for _ in range(8):
                if rec:
                    rec.reset()
                obs, _, _, _ = e.step(SETTLE)
                before = fingerprint(e, obs)
                m = monitor.measure(rec.samples if rec else [])
                assert fingerprint(e, obs) == before
                assert not m['candidate']
                frames.append(before)
            if reference is None:
                reference = frames
            else:
                assert frames == reference, 'Measurement or EGL lifecycle changed live state/image'
        finally:
            if rec:
                rec.close()
            e.close()
    passes.append('exact state/controller/image replay with and without monitoring, 3 environments')

    reset_seeds()
    e, obs = settled_drawer(args.root, 0)
    rec = ContactRecorder(e.sim, keywords=('akita_black_bowl', 'white_cabinet'))
    monitor, tracker = DrawerMonitor(e), SubgoalTracker()
    try:
        # Geometry-only synthetic fixture. These teleports NEVER enter evaluation.
        d = e.sim.data
        d.qpos[monitor.joint] = .003
        d.qvel[:] = 0
        e.sim.forward()
        r = d.body_xmat[monitor.drawer].reshape(3, 3).copy()
        origin = d.body_xpos[monitor.drawer].copy()
        bowl_local = np.concatenate([monitor.vertices[g] @ e.sim.data.geom_xmat[g].reshape(3, 3).T
            + d.geom_xpos[g] - d.body_xpos[monitor.bowl] for g in monitor.bowl_geoms])
        minz = float(bowl_local[:, 2].min())
        target = np.array([*(.5*(monitor.lower[:2]+monitor.upper[:2])), monitor.lower[2]-minz+.002])
        q = d.get_joint_qpos('akita_black_bowl_1_joint0').copy()
        q[:3] = origin + r @ target
        # Keep the root's original bowl rotation used in the minimum-height calculation.
        d.set_joint_qpos('akita_black_bowl_1_joint0', q)
        e.sim.forward()
        rows = []
        for step in range(60):
            rec.reset()
            obs, _, _, _ = e.step(SETTLE)
            m = monitor.measure(rec.samples)
            t = tracker.update(m, step)
            rows.append(m)
        assert t['complete'], rows[-1]
        passes.append('physically settled closed+contained+supported+released bowl reaches 1s dwell')
        # Reference-point containment must NOT pass full-footprint containment.
        q = d.get_joint_qpos('akita_black_bowl_1_joint0').copy()
        q[:3] += r[:, 0] * .07
        d.set_joint_qpos('akita_black_bowl_1_joint0', q)
        e.sim.forward()
        outside = monitor.measure([])
        assert outside['native_in'] and not outside['conditions']['full_footprint'], outside
        passes.append('native point-inside / collision footprint-outside counterexample rejected')
        report = dict(status='passed', checks=passes, fixture_only=True,
            lower=monitor.lower.tolist(), upper=monitor.upper.tolist(), final_fixture=rows[-1],
            source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [Path(__file__), Path(__file__).resolve().parents[1]/'src/adapters/libero_drawer.py']})
        args.out.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report))
    finally:
        rec.close()
        e.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/home/ami/scratch/LIBERO')
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
