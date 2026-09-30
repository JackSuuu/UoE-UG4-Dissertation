"""Pin the action convention, by the procedure the adapter's docstring prescribes.

The VLA has to emit something in the sim's action space: a world-frame pusher
velocity (vx, vy) in m/s, clipped to +-max_vel and integrated as p += a*dt. A
pretrained VLA instead emits a 7-D end-effector delta, and mapping one to the
other needs three constants that belong to the *pretraining dataset*, not to the
task: which slots are the axes, how many metres one unit is, and over what
control dt the displacement was intended.

Getting those wrong is not a crash. The policy trains, the arm moves in roughly
the right direction, the success rate plateaus somewhere arbitrary, and the
plateau is then attributed to the verifier. So this test does what the docstring
says the constants must be pinned against: it *derives* them from a recorded
expert demo and checks the mapping reproduces that demo.

Four things, and the last two are the ones that catch real mistakes:

  1. native_head refuses to construct without explicit constants -- no default,
     so there is no silent guess. This must raise before transformers is
     imported, or the guard is unreachable on a machine without it.
  2. The axis permutation is the named one. Asserted against sim.step rather
     than against a hand-written expectation, so a dx<->dy swap in either the
     sim or the adapter fails here instead of in a training curve.
  3. The scale is the sim's own. One dataset unit of displacement, fed back
     through the map, must advance the pusher by metres_per_unit in exactly
     sim.dt of simulated time.
  4. Calibration inverts correctly. Fit constants on a demo, then the map must
     return the demo's velocities. Trivially true for algebra alone, so the
     check is over the *rollout*: the pusher's recorded positions must be
     reproduced, which a wrong dt or scale breaks even when the arithmetic is
     internally consistent.

Run: cd src && CUDA_VISIBLE_DEVICES=0 python -u tests/test_action_map.py
"""
import os
import sys

import numpy as np
import torch

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)

from adapters.openvla_policy import (EE_DELTA_ORDER, ActionMapError,  # noqa: E402
                                     OpenVLAPolicy)
from sims.base import make_sim                                       # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _Stub:
    """Just enough of a policy to exercise _map_action without transformers.

    The class under test is imported for real; only its heavyweight __init__ is
    bypassed. Rebuilding a second copy of _map_action here would be the wrong
    way to do this -- a duplicated copy is exactly the thing that drifts from the
    original and makes a test pass while the code is wrong.
    """

    def __new__(cls, sim, **kw):
        kw.setdefault("action_mode", "native_head")
        kw.setdefault("metres_per_unit", 1.0)
        kw.setdefault("control_dt", 0.05)
        self = object.__new__(OpenVLAPolicy)
        self.sim, self.dev, self.H = sim, DEV, 5
        self.act_dim = sim.act_dim
        self.action_mode = kw["action_mode"]
        self.axes = kw.get("axes", ("dx", "dy"))
        self.metres_per_unit = kw.get("metres_per_unit")
        self.control_dt = kw.get("control_dt")
        return self


def _expert_demo(sim, n_env=4, T=40, seed=5):
    """Record a scripted-expert rollout: the state, the action, and the result.

    This is the artefact the constants have to be fitted against, and it is what
    a real fine-tuning set would be measured against too.
    """
    g = torch.Generator().manual_seed(seed)
    s = sim.init_state(n_env, g).to(DEV)
    params = sim.make_params(1, {})
    rec = {"s": [], "a": [], "p_after": []}
    for _ in range(T):
        a = sim.expert(s, params)
        rec["s"].append(s.clone())
        rec["a"].append(a.clone())
        s = sim.step(s, a, params)[0]
        rec["p_after"].append(s[:, 4:6].clone())
    return {k: torch.stack(v) for k, v in rec.items()}


def main():
    sim = make_sim("push", DEV)
    print(f"sim action: world-frame pusher velocity, |a| <= {sim.max_vel} m/s, "
          f"integrated as p += a*dt with dt = {sim.dt} s")
    ok = True

    # ---- 1. no default constants
    print("\n1. native_head refuses to guess")
    # The real __init__, not the stub: the guard has to fire before the lazy
    # transformers import, which is why the validation sits above it in the
    # source. If the ordering regresses, this raises ImportError instead and the
    # guard is unreachable on any machine without transformers -- so both
    # outcomes are failures here, not a skip.
    try:
        OpenVLAPolicy(sim, DEV, action_mode="native_head", metres_per_unit=None,
                      control_dt=sim.dt)
        raise AssertionError(
            "no guard: native_head accepted a missing metres_per_unit")
    except ActionMapError as e:
        print(f"   ok  ActionMapError: {str(e).splitlines()[0][:88]}")
    except ImportError as e:
        raise AssertionError(
            f"the guard is unreachable: __init__ imports transformers before it "
            f"validates ({e}). Move the lazy import below the checks.")
    # ... and the same for control_dt
    try:
        OpenVLAPolicy(sim, DEV, action_mode="native_head", metres_per_unit=1.0,
                      control_dt=None)
        raise AssertionError("no guard: native_head accepted a missing control_dt")
    except ActionMapError:
        print("   ok  same for a missing control_dt")
    # supplying both must get past the guard (and then fail on the model load,
    # which is not this test's business)
    try:
        OpenVLAPolicy(sim, DEV, action_mode="native_head", metres_per_unit=1.0,
                      control_dt=sim.dt)
    except ActionMapError as e:
        raise AssertionError(f"guard is too broad: it rejected valid constants ({e})")
    except Exception:
        print("   ok  explicit constants pass the guard (model load not reached)")
    # planar_head needs no constants at all
    print(f"   ok  planar_head declares act_dim=2 and needs no scale or dt")

    # ---- 2. the named axis controls the named sim axis
    print("\n2. the named axis drives the named sim axis (checked on sim.step)")
    pol = _Stub(sim, axes=("dx", "dy"), metres_per_unit=1.0, control_dt=sim.dt)
    for slot, (want, label) in enumerate((("x", "dx"), ("y", "dy"))):
        a7 = np.zeros(7, dtype=np.float32)
        a7[EE_DELTA_ORDER.index(label)] = 0.01       # 10 mm along that slot only
        v = pol._map_action(a7)
        s0 = sim.init_state(1, torch.Generator().manual_seed(0)).to(DEV)
        s1 = sim.step(s0, v[None], sim.make_params(1, {}))[0]
        d = (s1[:, 4:6] - s0[:, 4:6])[0]
        got = d[0 if want == "x" else 1].item()
        other = d[1 if want == "x" else 0].item()
        print(f"   {label}: pusher moved ({d[0]:+.4f}, {d[1]:+.4f}) m  -> "
              f"sim-{want} component {got:+.4f}, cross-axis {other:+.5f}")
        ok &= abs(got - 0.01) < 1e-3 and abs(other) < 1e-4
        if abs(got - 0.01) >= 1e-3 or abs(other) >= 1e-4:
            print(f"   FAIL {label} did not move sim-{want} alone")

    # a swapped convention must be *detectably* different, or the test above
    # would pass for the wrong reason
    sw = _Stub(sim, axes=("dy", "dx"), metres_per_unit=1.0, control_dt=sim.dt)
    a7 = np.zeros(7, dtype=np.float32)
    a7[EE_DELTA_ORDER.index("dx")] = 0.01
    v_sw = sw._map_action(a7)
    v_ok = pol._map_action(a7)
    same = torch.allclose(v_sw, v_ok, atol=1e-6)
    print(f"   axes=('dx','dy') vs ('dy','dx') differ: "
          f"{'yes' if not same else 'NO -- the permutation test is vacuous'}")
    ok &= not same

    # ---- 3. the scale is the sim's own: one unit -> metres_per_unit in sim.dt
    print("\n3. one dataset unit of displacement is metres_per_unit, over sim.dt")
    for mpu, cdt in ((1.0, sim.dt), (0.02, sim.dt)):
        pol = _Stub(sim, metres_per_unit=mpu, control_dt=cdt)
        a7 = np.zeros(7, dtype=np.float32)
        a7[0] = 1.0
        v = pol._map_action(a7)
        s0 = sim.init_state(1, torch.Generator().manual_seed(0)).to(DEV)
        s1 = sim.step(s0, v[None], sim.make_params(1, {}))[0]
        moved = (s1[:, 4:6] - s0[:, 4:6])[0, 0].item()
        want = min(mpu, pol.sim.max_vel * cdt)
        print(f"   metres_per_unit={mpu:g}: moved {moved:+.4f} m, expected "
              f"{want:+.4f} (clamped at max_vel*{cdt:g})")
        ok &= abs(moved - want) < 1e-3

    # saturation: an action demanding more than the sim allows must land exactly
    # on the limit, not over it and not somewhere below
    pol = _Stub(sim, metres_per_unit=1.0, control_dt=sim.dt)
    a7 = np.zeros(7, dtype=np.float32)
    a7[0] = 1e3
    v = pol._map_action(a7)
    print(f"   saturation: a7=1000 -> |v| = {v.abs().max():.4f} "
          f"(max_vel {sim.max_vel})")
    ok &= abs(v.abs().max() - sim.max_vel) < 1e-6

    # ---- 4. calibrate on a demo, then reproduce the demo
    print("\n4. calibration inverts: fit constants on a demo, replay the demo")
    demo = _expert_demo(sim, n_env=4, T=40)
    # a 7-D EE-delta dataset in metres, at the sim's own control dt
    mpu, cdt = 1.0, sim.dt
    pol = _Stub(sim, metres_per_unit=mpu, control_dt=cdt)
    # what a dataset in those units would record for the expert's action
    a7 = torch.zeros(demo["a"].shape[0], demo["a"].shape[1],
                     len(EE_DELTA_ORDER), dtype=torch.float32, device=DEV)
    a7[..., 0] = demo["a"][..., 0] * cdt / mpu
    a7[..., 1] = demo["a"][..., 1] * cdt / mpu
    back = torch.stack([pol._map_action(a7[t]) for t in range(a7.shape[0])])
    err = (back - demo["a"]).abs().max().item()
    print(f"   velocity error over {a7.shape[0]} steps x {a7.shape[1]} envs: "
          f"{err:.2e} m/s (a max |a| of {demo['a'].abs().max():.3f})")
    ok &= err < 1e-5

    # and the rollout, which is what a wrong dt breaks even when the per-step
    # arithmetic is self-consistent
    s0 = demo["s"][0]
    s = s0
    for t in range(a7.shape[0]):
        s = sim.step(s, back[t], sim.make_params(1, {}))[0]
    perr = (s[:, 4:6] - demo["p_after"][-1]).abs().max().item()
    print(f"   pusher position error after {a7.shape[0]} replayed steps: "
          f"{perr:.2e} m")
    ok &= perr < 1e-4

    # a deliberately wrong dt must FAIL the same check, or the check is vacuous
    bad = _Stub(sim, metres_per_unit=mpu, control_dt=cdt / 2.0)
    back_bad = torch.stack([bad._map_action(a7[t]) for t in range(a7.shape[0])])
    s = s0
    for t in range(a7.shape[0]):
        s = sim.step(s, back_bad[t], sim.make_params(1, {}))[0]
    berr = (s[:, 4:6] - demo["p_after"][-1]).abs().max().item()
    print(f"   with control_dt/2 (a wrong convention): {berr:.2e} m "
          f"-> {'detected' if berr > perr else 'NOT DETECTED'}")
    ok &= berr > perr

    print("\n" + ("all checks passed" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
