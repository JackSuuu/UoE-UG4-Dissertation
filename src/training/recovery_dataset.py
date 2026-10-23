"""Executed repair-action and causal nominal-regression labels."""
from ..verification.physical_verifier import PhysicalVerifier


def regression_label(task, observation, future, *, horizon=24, history=()):
    """Risk target: loss of currently measured progress in observed nominal24.

    This is NOT proof that intervention was necessary or would improve success.
    No future repair actions can be treated as normal nominal continuation.
    """
    verifier = PhysicalVerifier()
    history = list(history)
    if not history or history[-1].index != observation.index:
        history.append(observation)
    predicates = tuple(dict.fromkeys((*task.goals, *task.milestones)))
    values = [verifier.evaluate(p, observation, history) for p in predicates]
    if any(v is None for v in values):
        return None, 'missing_entry_evidence'
    achieved = [p for p, value in zip(predicates, values) if value is True]
    streaks = {p: 0 for p in achieved}
    for n, row in enumerate(future[:horizon], 1):
        if row['owner'] != 'nominal':
            return None, 'repair_censored'
        endpoint = row['observation']
        if endpoint.index != history[-1].index+1:
            return None, 'nonconsecutive_future'
        history.append(endpoint)
        if row['complete']:
            return 0, 'task_completed_in_observed_nominal_continuation'
        for p in achieved:
            value = verifier.evaluate(p, endpoint, history)
            if value is None:
                return None, 'missing_future_evidence'
            streaks[p] = streaks[p]+1 if value is False else 0
        if any(n >= 3 for n in streaks.values()):
            return 1, 'observed_durable_progress_loss'
        if n == horizon:
            return 0, 'no_observed_progress_loss'
    return None, 'right_censored'


def recovery_label(repair):
    """Only executed first-repair proposals get an observed-transaction label."""
    if repair['exit_reason'] == 'task_completion_during_repair':
        return 1, 'task_completed_during_executed_recovery'
    if repair['preserve_lost']:
        return None, 'preservation_failed_or_unknown'
    if repair['verified']:
        return 1, 'observed_measured_recovery'
    if repair['exit_reason'] in ('local_budget_exit_unverified', 'total_repair_budget_exit_unverified'):
        return 0, 'observed_unverified_budget_exit'
    return None, 'incomplete_or_unexecuted_recovery'
