"""Offline kinematic reachability diagnostic, not contact manipulation or a policy."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def main(run, output, flip=False, target_x=-.025):
    model = mujoco.MjModel.from_xml_path(str(run/'model.xml'))
    data = mujoco.MjData(model)
    state = np.load(run/'states.npz')['states'][0]
    data.qpos[:] = state[1:1+model.nq]
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'robot0_joint{i}') for i in range(1,8)]
    assert min(ids) >= 0
    addresses = model.jnt_qposadr[ids]
    lower, upper = model.jnt_range[ids].T
    initial = data.qpos[addresses].copy()
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'gripper0_grip_site')
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'robot0_right_hand')
    assert body >= 0 and site >= 0
    desired = Rotation.from_matrix([[0,0,1],[0,1,0],[-1,0,0]] if flip else [[0,0,1],[0,-1,0],[1,0,0]])
    rng = np.random.default_rng(0)
    report = dict(scope='kinematic_only_not_collision_free_plan', initial=initial.tolist(),
                  orientation=desired.as_matrix().tolist(), targets=[])
    for target in [[-.25,.30,1.012],[target_x,.2475,1.012]]:
        def residual(q):
            data.qpos[addresses] = q
            mujoco.mj_forward(model, data)
            rotation = Rotation.from_matrix(data.xmat[body].reshape(3,3))
            return np.r_[data.site_xpos[site]-target, .1*(desired*rotation.inv()).as_rotvec()]
        solutions = []
        for q in [initial]+[rng.uniform(lower+.01, upper-.01) for _ in range(7)]:
            result = least_squares(residual, q, bounds=(lower+.0001,upper-.0001), max_nfev=300,
                                   ftol=1e-10, xtol=1e-10, gtol=1e-10)
            error = residual(result.x)
            contacts = [[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),
                         mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom2),float(c.dist)]
                        for c in data.contact[:data.ncon]]
            solutions.append(dict(q=result.x.tolist(), position_error=float(np.linalg.norm(error[:3])),
                rotation_error_rad=float(np.linalg.norm(error[3:])*10), contacts=contacts))
        report['targets'].append(dict(target=target, solutions=solutions))
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--flip',action='store_true')
    p.add_argument('--target-x',type=float,default=-.025)
    a=p.parse_args()
    main(a.run,a.output,a.flip,a.target_x)
