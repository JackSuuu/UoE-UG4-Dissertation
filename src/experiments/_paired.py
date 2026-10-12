"""Paired episode outcomes and fixed-grid, cell-stratified bootstrap intervals."""
import numpy as np


def paired_outcomes(base, method):
    """Both inputs describe the same ordered initial states, not aggregate rates."""
    bs = np.asarray(base['success'], dtype=bool)
    bv = np.asarray(base['violation'], dtype=bool)
    ms = np.asarray(method['success'], dtype=bool)
    mv = np.asarray(method['violation'], dtype=bool)
    if not (bs.shape == bv.shape == ms.shape == mv.shape) or bs.ndim != 1 or not bs.size:
        raise ValueError('paired outcomes require equal nonempty episode vectors')
    safe_b, safe_m = bs & ~bv, ms & ~mv
    return {
        'n_episodes': int(bs.size),
        'success_rescue_count': int((~bs & ms).sum()),
        'success_harm_count': int((bs & ~ms).sum()),
        'safe_rescue_count': int((~safe_b & safe_m).sum()),
        'safe_harm_count': int((safe_b & ~safe_m).sum()),
        'safe_rescue_rate': float((~safe_b & safe_m).mean()),
        'safe_harm_rate': float((safe_b & ~safe_m).mean()),
        'delta_SR': float(ms.mean() - bs.mean()),
        'delta_CVR': float(mv.mean() - bv.mean()),
        'delta_safe_success': float(safe_m.mean() - safe_b.mean()),
    }


def bootstrap_delta(differences, seed=0, n_boot=4000):
    """Resample episode PAIRS within each fixed cell, preserving equal cell weight.

    This interval describes episode uncertainty on this grid. It does not
    estimate generalization to new tasks or physical-parameter cells.
    """
    rows = [np.asarray(x, dtype=float) for x in differences]
    if not rows or any(x.ndim != 1 or not x.size for x in rows):
        raise ValueError('bootstrap requires nonempty per-cell difference vectors')
    rng = np.random.default_rng(seed)
    means = np.zeros(n_boot)
    for start in range(0, n_boot, 256):
        end = min(start + 256, n_boot)
        for row in rows:
            idx = rng.integers(0, row.size, size=(end - start, row.size))
            means[start:end] += row[idx].mean(1) / len(rows)
    return {'estimate': float(np.mean([x.mean() for x in rows])),
            'ci95': np.quantile(means, [0.025, 0.975]).tolist(),
            'n_bootstrap': n_boot,
            'interval_scope': 'paired_episodes_within_fixed_parameter_cells'}


def pooled_paired(entries, arm, seed=0):
    selected = [r for r in entries if r['ood']] or entries
    rows = []
    for r in selected:
        episodes = r['episodes']
        rows.append(paired_outcomes(episodes['none'], episodes[arm]))
    result = {'n_cells': len(rows), 'n_episodes': sum(r['n_episodes'] for r in rows)}
    for key in ('success_rescue_count', 'success_harm_count', 'safe_rescue_count', 'safe_harm_count'):
        result[key] = sum(r[key] for r in rows)
    for key in ('safe_rescue_rate', 'safe_harm_rate', 'delta_SR', 'delta_CVR', 'delta_safe_success'):
        result[key] = float(np.mean([r[key] for r in rows]))
    result['paired_intervals'] = {}
    for mi, metric in enumerate(('SR', 'CVR', 'safe_success')):
        differences = []
        for r in selected:
            b, m = r['episodes']['none'], r['episodes'][arm]
            def value(e):
                s, v = np.asarray(e['success'], dtype=bool), np.asarray(e['violation'], dtype=bool)
                return s if metric == 'SR' else v if metric == 'CVR' else s & ~v
            differences.append(value(m).astype(float) - value(b).astype(float))
        result['paired_intervals'][metric] = bootstrap_delta(differences, seed + mi)
    return result
