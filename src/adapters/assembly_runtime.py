"""Single-arm square-nut insertion diagnostic; actual contacts, no welded grasp.

The scripted pre-positioning is evaluator setup, not the evaluated policy.
"""
import os
import random

import numpy as np


def make_assembly(seed, camera=True):
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    from adapters.libero_runtime import patch_joint_addresses, patch_mass_matrix
    from adapters.libero_preservation import bind_renderer_context
    patch_joint_addresses()
    patch_mass_matrix()
    bind_renderer_context()
    import robosuite as suite
    from robosuite.controllers import load_controller_config
    from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
    np.random.seed(seed)
    random.seed(seed)
    sampler = SequentialCompositeSampler(name='AssemblyInitial')
    for name, y in [('SquareNut', .15), ('RoundNut', -.15)]:
        sampler.append_sampler(UniformRandomSampler(name=name+'Sampler',
            x_range=[-.115, -.11], y_range=[y-.005, y+.005], rotation=np.pi,
            ensure_object_boundary_in_range=False, ensure_valid_placement=True,
            reference_pos=[0, 0, .82], z_offset=.02))
    return suite.make('NutAssemblySquare', robots='Panda',
        controller_configs=load_controller_config(default_controller='OSC_POSE'),
        initialization_noise=None, placement_initializer=sampler,
        has_renderer=False, has_offscreen_renderer=camera, use_camera_obs=camera,
        camera_names='agentview', camera_heights=256, camera_widths=256,
        control_freq=20, horizon=10000, ignore_done=True)


def positions(env):
    return (env.sim.data.site_xpos[env.robots[0].eef_site_id].copy(),
            env.sim.data.body_xpos[env.obj_body_id['SquareNut']].copy(),
            env.sim.data.body_xpos[env.peg1_body_id].copy())


def prepare_insertion(env, offset=(0., 0.)):
    """Physically grasp handle, lift and approach peg; retain every setup action."""
    obs = env.reset()
    if env.has_offscreen_renderer:
        # Freeze A1 appearance before actor trials; avoid driver-sensitive shadow
        # and reflection passes. This does not resolve historical M1 rendering.
        import mujoco
        from OpenGL import GL
        context = env.sim._render_context_offscreen
        context.gl_ctx.make_current()
        context.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        context.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        GL.glDisable(GL.GL_DITHER)
    actions = []
    for _ in range(20):
        action = [0., 0., 0., 0., 0., 0., -1.]
        obs, _, _, _ = env.step(action)
        actions.append(action)
    def move(target, grip, count):
        nonlocal obs
        for _ in range(count):
            eef, _, _ = positions(env)
            action = np.r_[np.clip((target-eef)/.05, -.35, .35), [0., 0., 0.], grip]
            obs, _, _, _ = env.step(action)
            actions.append(action.tolist())
    handle = env.sim.data.site_xpos[env.object_site_ids[0]].copy()
    move(handle+np.array([0, 0, .10]), -1, 50)
    move(handle+np.array([0, 0, .005]), -1, 40)
    move(handle+np.array([0, 0, .005]), 1, 20)
    move(handle+np.array([0, 0, .18]), 1, 50)
    _, nut, peg = positions(env)
    if nut[2] < .93:
        raise RuntimeError(f'Physical setup grasp failed: nut={nut.tolist()}')
    for _ in range(140):
        eef, nut, peg = positions(env)
        target_nut = np.r_[peg[:2]+offset, 1.01]
        move(eef+target_nut-nut, 1, 1)
    eef, nut, peg = positions(env)
    if np.linalg.norm(nut[:2]-peg[:2]-offset) > .008 or nut[2] < .98:
        raise RuntimeError(f'Physical preposition failed: nut={nut.tolist()} peg={peg.tolist()}')
    if not env._check_grasp(env.robots[0].gripper, env.nuts[0].contact_geoms):
        raise RuntimeError('Setup lacks bilateral finger contact')
    return obs, actions


def contact_load(env):
    """Sum of nut/peg contact-force magnitudes at the current physics substep."""
    import mujoco
    nut = env.obj_body_id['SquareNut']
    peg = env.peg1_body_id
    load = 0.
    for i in range(env.sim.data.ncon):
        c = env.sim.data.contact[i]
        if {int(env.sim.model.geom_bodyid[c.geom1]), int(env.sim.model.geom_bodyid[c.geom2])} == {nut, peg}:
            force = np.zeros(6)
            mujoco.mj_contactForce(env.sim.model._model, env.sim.data._data, i, force)
            load += np.linalg.norm(force[:3])
    return float(load)


def step_measured(env, action):
    """Capture peak load across *all* internal 2ms physics steps."""
    original = env.sim.step
    peak = [0.]
    def measured():
        original()
        peak[0] = max(peak[0], contact_load(env))
    env.sim.step = measured
    try:
        obs, _, _, _ = env.step(np.asarray(action))
    finally:
        env.sim.step = original
    return obs, peak[0]


def features(env):
    """Privileged 40D state: relative poses, velocities, joints and contact load."""
    eef, nut, peg = positions(env)
    body = env.obj_body_id['SquareNut']
    rotation = env.sim.data.body_xmat[body].reshape(3, 3)
    return np.r_[nut-peg, eef-nut, rotation.ravel(),
                 env.sim.data.qpos[:7], env.sim.data.qvel[:7],
                 env.sim.data.qpos[7:9], env.sim.data.qvel[7:9],
                 env.sim.data.get_body_xvelp('SquareNut_main'),
                 env.sim.data.get_body_xvelr('SquareNut_main'), contact_load(env)].astype(np.float32)


def endpoint(env):
    """Seated square ring, detached gripper, low object speed; five-step dwell."""
    eef, nut, peg = positions(env)
    rotation = env.sim.data.body_xmat[env.obj_body_id['SquareNut']].reshape(3, 3)
    tilt = float(np.arccos(np.clip(rotation[2, 2], -1, 1)))
    # Square opening half-width .02275, peg half-width .016; local containment.
    local = rotation.T @ (peg-nut)
    extent = .016 * np.abs(rotation[:2, :2]).sum(axis=0)
    contained = bool(np.all(np.abs(local[:2])+extent < .02275))
    speed = float(np.linalg.norm(env.sim.data.get_body_xvelp('SquareNut_main')))
    detached = not env.check_contact(env.robots[0].gripper.contact_geoms, env.nuts[0].contact_geoms)
    seated = contained and nut[2] < .84 and tilt < np.deg2rad(10)
    return dict(complete=bool(seated and detached and speed < .01),
                seated=bool(seated), contained=contained, detached=bool(detached),
                speed=speed, tilt_rad=tilt, native_success=bool(env._check_success()))
