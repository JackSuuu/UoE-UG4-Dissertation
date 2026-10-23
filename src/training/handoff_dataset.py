"""Labels for an actually executed nominal action, never repair-entry outcomes."""
from ..verification.physical_verifier import PhysicalVerifier


LABEL_SCHEMA = 'observed_generated_contract_nominal24_v1'


def continuation_label(plan, observation, future, *, horizon=24, history=()):
    """future rows contain owner, observation, complete; unknown/repair is censored.

    One *same* acceptance alternative must survive the observed continuation.
    A current measured contract is necessary; future success is never an input.
    """
    verifier = PhysicalVerifier()
    history = list(history)
    if not history or history[-1].index != observation.index:
        history.append(observation)
    survivors = {i for i, group in enumerate(plan.alternatives)
                 if verifier.check_all(group, observation, history) is True}
    if not survivors or verifier.check_all(plan.preserve, observation, history) is not True:
        return None, 'entry_contract_unverified'
    for n, row in enumerate(future[:horizon], 1):
        if row['owner'] != 'nominal':
            return None, 'repair_censored'
        endpoint = row['observation']
        if endpoint.index != history[-1].index+1:
            return None, 'nonconsecutive_future'
        history.append(endpoint)
        if row['complete']:
            return 1, 'task_completed_after_actual_nominal_action'
        preserved = verifier.check_all(plan.preserve, endpoint, history)
        checks = {i: verifier.check_all(plan.alternatives[i], endpoint, history) for i in survivors}
        if preserved is None or any(value is None for value in checks.values()):
            return None, 'missing_evidence'
        survivors = {i for i, value in checks.items() if value is True}
        if preserved is False or not survivors:
            return 0, 'observed_nominal_contract_loss'
        if n == horizon:
            return 1, 'observed_nominal_contract_survival'
    return None, 'right_censored'
