"""Same-state transport diagnosis; no model query or full-framework efficacy claim."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation

from run_fridge_framework import create, sense, measure, save, fingerprint, REPO
from adapters.relational_transport import RelationalTransport
from adapters.libero_completion import box_corners


def feedback(env, obs):
    state = sense(env, obs)
    m, d = env.sim.model, env.sim.data
    body = m.body_name2id('short_fridge_1_main')
    frame = d.body_xmat[body].reshape(3, 3)@Rotation.from_euler('z', np.pi/2).as_matrix()
    origin = d.body_xpos[body].copy()
    obstacles, payload = [], []
    for g in range(m.ngeom):
        name = m.geom_id2name(g) or ''
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        if 'short_fridge' in name or 'gripper' in name or 'cream_cheese' in name:
            if int(m.geom_type[g]) == 6:
                points = box_corners(d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g])
            elif int(m.geom_type[g]) == 7:
                mesh = m.geom_dataid[g]
                start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
                points = m.mesh_vert[start:start+count]@d.geom_xmat[g].reshape(3, 3).T+d.geom_xpos[g]
            else:
                raise ValueError((name, int(m.geom_type[g])))
            (obstacles if 'short_fridge' in name else payload).extend(points)
    local = (np.asarray(obstacles)-origin)@frame
    region = env.env.parsed_problem['regions']['kitchen_table_placement_region']['ranges'][0]
    target = np.array([(region[0]+region[2])/2, (region[1]+region[3])/2, state['floor']])
    g = m.geom_name2id('cream_cheese_1_g1')
    half = (np.abs(d.geom_xmat[g].reshape(3, 3))@m.geom_size[g])[2]
    return dict(state, frame=frame, origin=origin, support_center=target,
                obstacle_bounds=np.array([local.min(axis=0), local.max(axis=0)]),
                payload_points=np.asarray(payload), item_half_height=float(half))


def main(run, out, branch, budget):
    out.mkdir(parents=True, exist_ok=False)
    args = argparse.Namespace(**json.loads((run/'status.json').read_text())['args'])
    args.gpu, args.steps = 2, branch+budget+40
    rows = [json.loads(l) for l in (run/'steps.jsonl').read_text().splitlines()]
    env, obs = create(args, out)
    report = dict(scope='same_state_relational_transport_development_not_framework',
                  branch=branch, budget=budget, source=str(run), status='started')
    sources = ['scripts/probe_relational_transport.py', 'src/adapters/relational_transport.py']
    report['sources'] = {}
    for p in sources:
        data = (REPO/p).read_bytes()
        dest = out/'source'/p
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        report['sources'][p] = hashlib.sha256(data).hexdigest()
    states = []
    try:
        for _ in range(20):
            obs, *_ = env.step([0, 0, 0, 0, 0, 0, -1])
        for row in rows[:branch]:
            obs, *_ = env.step(row['action'])
        assert fingerprint(env, obs) == rows[branch]['before']
        report['branch_fingerprint'] = fingerprint(env, obs)
        state = feedback(env, obs)
        skill = RelationalTransport(state, limit=budget)
        report['world_waypoints'] = [skill.world(p).tolist() for p in skill.waypoints]
        with imageio.get_writer(out/'raw.mp4', fps=20) as video, (out/'steps.jsonl').open('w') as log:
            for step in range(budget+1):
                action, event = skill.act(state)
                if action is None:
                    report['exit'] = event
                    break
                video.append_data(obs['agentview_image'][::-1, ::-1])
                states.append(env.get_sim_state().copy())
                before = fingerprint(env, obs)
                obs, *_ = env.step(action.tolist())
                state = feedback(env, obs)
                row = dict(step=step, action=action.tolist(), **event, **measure(env, obs),
                    before=before, after=fingerprint(env, obs), placed=state['placed'])
                log.write(json.dumps(row)+'\n')
                if step%25 == 0:
                    print(json.dumps(row), flush=True)
            video.append_data(obs['agentview_image'][::-1, ::-1])
            states.append(env.get_sim_state().copy())
        report.update(status='completed', final=measure(env, obs), placed=state['placed'])
    finally:
        np.savez_compressed(out/'states.npz', states=np.asarray(states))
        save(out/'report.json', report)
        env.close()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--branch', type=int, default=332)
    p.add_argument('--budget', type=int, default=240)
    a = p.parse_args()
    main(a.run.resolve(), a.out.resolve(), a.branch, a.budget)
