"""Positive/negative evaluator fixtures ONLY; state installation is not task execution."""
import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from run_fridge_framework import create, sense, save, TaskMonitor


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    args=argparse.Namespace(root='/home/ami/scratch/LIBERO',open_angle=np.deg2rad(130),
        item_forward=.04,item_lateral=.035,yaw=-np.pi/2,fridge_y=.35,door='closed',
        steps=600,gpu=2,seed=0,camera='robot_side')
    env,obs=create(args,out)
    try:
        m,d=env.sim.model,env.sim.data
        item=env.env.objects_dict['cream_cheese_1']
        address=m.get_joint_qpos_addr(item.joints[0])[0]
        results=[]
        for name,position,settle,expected in [
            ('supported_released_inside',[-.025,-.15,.97],100,True),
            ('supported_outside_region',[.25,-.15,.97],100,False),
            ('unsupported_above_region',[-.025,-.15,1.15],0,False)]:
            d.qpos[address:address+3]=position
            d.qvel[:]=0
            env.sim.forward()
            obs=env.env._get_observations(force_update=True)
            for _ in range(settle):
                obs,*_=env.step([0,0,0,0,0,0,-1])
            s=sense(env,obs)
            assert s['placed'] == expected,(name,s)
            tracker=TaskMonitor('closed')
            for _ in range(25):
                task=tracker.observe(s)
            assert not task['full_completion'], 'A final pose without observed sequence is not full success'
            imageio.imwrite(out/f'{name}.png',obs['agentview_image'][::-1,::-1])
            results.append(dict(name=name,expected_placed=expected,observed_placed=s['placed'],
                table_contact=s['table'],gripper_contact=s['hand_contact'],item_position=s['item'].tolist(),
                full_completion_without_sequence=task['full_completion']))
        save(out/'qualification.json',dict(status='fixtures_passed',scope='evaluator_only_state_installed',
            no_task_success_claim=True,cases=results))
    finally:
        env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('out',type=Path)
    args=p.parse_args()
    main(args.out.resolve())
