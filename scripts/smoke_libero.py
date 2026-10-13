"""Reset/render/step smoke, explicitly not a policy success result."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from adapters.libero_runtime import make_env

p = argparse.ArgumentParser()
p.add_argument('--root', default='/home/ami/scratch/LIBERO')
p.add_argument('--out', required=True)
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=False)
env, obs = make_env(a.root)
try:
    import imageio.v2 as imageio
    frames = [obs['agentview_image'][::-1, ::-1]]
    for _ in range(10):
        obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])
        frames.append(obs['agentview_image'][::-1, ::-1])
    imageio.imwrite(out / 'initial.png', frames[0])
    imageio.mimwrite(out / 'settling.mp4', frames, fps=20)
    (out / 'smoke.json').write_text(json.dumps(dict(
        observation_keys=sorted(obs), action_dim=env.env.action_dim,
        success=bool(env.check_success()), language=env.language_instruction,
        physics_dt=env.sim.model.opt.timestep), indent=2))
    print((out / 'smoke.json').read_text())
finally:
    env.close()
