"""A2: shared predictive gate, measured-state geometry, optional trusted ranking.

Self-developed assembly guardrail, not an official OrbiSim/CheckVLA implementation.
The policy and A1 sources stay frozen. No simulator candidate rollouts occur here.
"""
import numpy as np
import torch
from adapters.assembly_predictor import candidates as old_candidates, NAMES as OLD_NAMES

NAMES = OLD_NAMES + ['open_hold', 'open_retract', 'retain_proposal']
ARMS = ['vla', 'feedback80', 'phase_geometry80', 'hybrid80']
CAP = 80
POSITION_TOL = .00675/3  # one third of nominal square-ring clearance
FORCE_TOL = 5.
TILT_TOL_DEG = 5.


def candidates(nominal):
    old = old_candidates(nominal)
    extra = nominal.repeat(3, 3, 1)
    extra[:2, :, :6] = 0
    extra[:2, :, 6] = -1
    extra[1, 0, 2] = .15
    extra[2, 1:, :6] = 0
    extra[2, :, 6] = 1
    return torch.cat([old, extra])


def forecast(model, verifier, current, previous, previous_action, chunks):
    """Eleven candidates in batches <=8, shared across all arms."""
    states, risks, checked = [], [], []
    with torch.no_grad():
        for start in range(0, len(chunks), 8):
            actions = chunks[start:start+8]
            n = len(actions)
            ss, rr = model.rollout(current.expand(n, -1), previous.expand(n, -1),
                                   previous_action.expand(n, -1), actions)
            states.append(ss); risks.append(rr)
            checked.append(verifier.score(dict(obs=current.expand(n, -1),
                obs_prev=previous.expand(n, -1), a_prev=previous_action.expand(n, -1)), actions))
        states, risks = torch.cat(states, 1), torch.cat(risks, 1)
        mean, std = risks.mean(0), risks.std(0)
        scores = torch.cat(checked)
        objectives = (states[..., -1, :2].norm(dim=-1)/.02 +
                      (states[..., -1, 2]+.02).abs()/.12 + .5*risks.amax((2, 3)))
        pred = states.mean(0)
        objective = pred[:, -1, :2].norm(dim=-1)/.02 + (pred[:, -1, 2]+.02).abs()/.12 + .5*scores
    return dict(mean=pred.cpu().numpy(), state_std=states.std(0).cpu().numpy(),
                risk_mean=mean.cpu().numpy(), risk_std=std.cpu().numpy(),
                scores=scores.cpu().numpy(), objective=objective.cpu().numpy(),
                objective_std=objectives.std(0).cpu().numpy())


def measured(env, obs):
    from adapters.assembly_runtime import endpoint
    ep = endpoint(env)
    grasped = bool(env._check_grasp(env.robots[0].gripper, env.nuts[0].contact_geoms))
    return dict(grasped=grasped, touching=not ep['detached'], seated=ep['seated'],
                tilt_deg=float(np.rad2deg(ep['tilt_rad'])), endpoint=ep)


def residual(last_forecast, current, last_load, tilt_deg):
    if last_forecast is None:
        return None
    return dict(xyz_m=float(np.linalg.norm(last_forecast['state'][:3]-current[:3])),
                force_N=float(abs(last_forecast['risk'][0]*10-last_load)),
                tilt_deg=float(abs(last_forecast['risk'][1]*15-tilt_deg)))


def decide(arm, current, nominal, bank, prediction, monitor, history, tau):
    """Return an observable routing decision; pure function with no future access."""
    score, objective = prediction['scores'], prediction['objective']
    used, stall = history['used'], history['stall']
    contact = bool(current[2] < .13 or current[-1] > .1)
    legacy_gate = contact and (score[0] > tau or stall >= 3)
    premature = bool(monitor['grasped'] and not monitor['seated'] and nominal[6] < 0)
    lost = history['ungrasped'] >= 3 and not monitor['seated']
    release = monitor['seated']
    phase_gate = bool(legacy_gate or premature or release or (contact and history['last_load'] > 10))
    phase = 'release_and_clear' if release else ('reacquire' if lost else ('align_and_insert' if contact else 'approach'))
    trigger = bool(legacy_gate if arm in ('vla', 'feedback80') else phase_gate)
    chosen, geometry = 0, 0
    owner, reason = 'actor', 'no_trigger'
    eligible, accepted = [], []
    reliability = history['residual']
    reliable = bool(reliability is not None and reliability['xyz_m'] <= POSITION_TOL
        and reliability['force_N'] <= FORCE_TOL and reliability['tilt_deg'] <= TILT_TOL_DEG)
    if arm == 'feedback80':
        if current[-1] > 10:
            geometry = 2
        elif np.max(np.abs(current[:2])) > .004:
            axis = int(np.argmax(np.abs(current[:2])))
            geometry = (4 if current[0] > 0 else 3) if axis == 0 else (6 if current[1] > 0 else 5)
        else:
            geometry = 7
    else:
        if release:
            geometry = 8 if monitor['touching'] else 9
        elif lost:
            geometry = 0
        elif max(float(current[-1]), history['last_load']) > 10:
            geometry = 2
        elif premature:
            geometry = 10
        elif np.max(np.abs(current[:2])) > .004:
            axis = int(np.argmax(np.abs(current[:2])))
            geometry = (4 if current[0] > 0 else 3) if axis == 0 else (6 if current[1] > 0 else 5)
        else:
            geometry = 7
    if arm != 'vla' and trigger:
        if used >= CAP:
            reason = 'budget_exhausted'
        elif lost and arm in ('phase_geometry80', 'hybrid80'):
            reason = 'lost_grasp_return_actor'
        else:
            chosen = geometry
            owner, reason = 'geometry', 'shared_gate_feedback' if arm == 'feedback80' else 'phase_geometry'
            if arm == 'hybrid80' and not release and monitor['grasped']:
                # The measured geometry bounds the model's action choices.
                eligible = [1, 2, geometry]
                if not premature and np.dot(nominal[:2], current[:2]) <= 0:
                    eligible.append(0)
                for axis, positive, negative in [(0, 3, 4), (1, 5, 6)]:
                    if abs(current[axis]) > .004:
                        eligible.append(negative if current[axis] > 0 else positive)
                if np.max(np.abs(current[:2])) <= .004:
                    eligible.append(7)
                eligible = sorted(set(eligible))
                ceiling = max(float(tau), float(current[-1])/10, monitor['tilt_deg']/15)
                for candidate in eligible:
                    spread = np.linalg.norm(prediction['state_std'][candidate, :, :3], axis=-1).max()
                    progress = np.linalg.norm(prediction['mean'][candidate, -1, :2])
                    reference = np.linalg.norm(prediction['mean'][geometry, -1, :2])
                    # Engineering acceptance constraints, not safety certificates.
                    if (reliable and spread <= POSITION_TOL and score[candidate] <= ceiling
                            and progress <= reference+.00675/4):
                        accepted.append(candidate)
                best = min(accepted, key=lambda i: objective[i]) if accepted else None
                if best is not None and best != geometry:
                    margin = .05+2*max(prediction['objective_std'][best], prediction['objective_std'][geometry])
                    if objective[best] < objective[geometry]-margin:
                        chosen, owner, reason = best, 'prediction', 'trusted_prediction_override'
                    else:
                        reason = 'geometry_prediction_advantage_insufficient'
                else:
                    reason = 'geometry_prediction_untrusted' if not reliable else 'geometry_no_accepted_better_candidate'
    action = bank[chosen].copy()
    # Retain actor's original float64 values when no substitution is selected.
    if chosen == 0:
        action = nominal.copy()
        if owner != 'prediction':
            owner = 'actor'
    changed = bool(np.max(np.abs(action-nominal)) > 1e-8)
    return dict(index=chosen, selection=NAMES[chosen], selected=action.tolist(),
        changed=changed, owner=owner, reason=reason, trigger=trigger,
        legacy_gate=bool(legacy_gate), phase_gate=phase_gate, premature_release=premature,
        lost_grasp=lost, subgoal=phase, mode='contact_transition' if (contact or premature or release) else 'nominal_tracking',
        geometry_index=geometry, geometry_selection=NAMES[geometry],
        prediction_reliable=reliable, residual=reliability, eligible=eligible, accepted=accepted,
        route=('hold' if chosen in (1, 8) else 'repair') if changed else 'continue')
