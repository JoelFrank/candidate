"""Reproducible negative sampling for the revised temporal-inductive evaluation.

The original ``uniform`` policy is retained for forensic comparisons.  The
``matched_synthetic`` policy matches the positive count *within each* of the
four old/new endpoint strata, preventing label-stratum composition from
becoming an unintended ranking shortcut in the BiSBM benchmark.

Only available positive labels in the requested snapshot are used to set the
sampling quotas.  The eligible negative pool excludes positives from the
provided history (which normally includes the current evaluation snapshot);
future labels must NEVER be supplied merely to exclude future positives.
"""
import numpy as np
import torch
from revised.data import remap_context, module as load_project_module


def synthetic_bisbm_boundaries():
    """Read OLD-node boundaries from the same generator used by prepare().

    Do not `import data.synthetic.generator`: legacy imports can register an
    unrelated top-level `data` module (not a package) in sys.modules.
    """
    generator = load_project_module('data/synthetic/generator.py')
    return int(generator.N_U_OLD), int(generator.N_V_OLD)


def _known_edges(snapshot, history):
    ei, _ = remap_context(history, snapshot)
    return set(map(tuple, ei.T.tolist()))


def negatives(s, history, n, seed):
    """Legacy uniformly sampled negatives (kept unchanged for comparison)."""
    known = _known_edges(s, history)
    capacity = s['num_u'] * s['num_v'] - len(known)
    if capacity < n:
        raise ValueError(f'Only {capacity} eligible negatives; requested {n}')
    rng = np.random.default_rng(seed)
    result = []
    seen = set(known)
    for _ in range(max(1000, 100 * n)):
        pair = (int(rng.integers(s['num_u'])),
                int(rng.integers(s['num_v'])) + s['num_u'])
        if pair not in seen:
            seen.add(pair)
            result.append(pair)
        if len(result) == n:
            break
    if len(result) < n:
        for u in range(s['num_u']):
            for v in range(s['num_u'], s['num_u'] + s['num_v']):
                if (u, v) not in seen:
                    result.append((u, v))
                    seen.add((u, v))
                if len(result) == n:
                    break
            if len(result) == n:
                break
    return torch.tensor(result, dtype=torch.long).reshape(-1, 2).T


def _endpoint_stratum(u_is_new, v_is_new):
    return int(u_is_new) * 2 + int(v_is_new)


def synthetic_stratum_counts(s, ei, num_u_old, num_v_old):
    """Counts in OO, old-U/new-V, new-U/old-V, new/new (BiSBM raw IDs)."""
    u_keys = np.asarray(s['keys_u'])
    v_keys = np.asarray(s['keys_v'])
    nu = s['num_u']
    counts = [0, 0, 0, 0]
    for u, v in ei.T.tolist():
        i = _endpoint_stratum(u_keys[u] >= num_u_old,
                             v_keys[v - nu] >= num_v_old)
        counts[i] += 1
    return dict(zip(('OO', 'ON', 'NO', 'NN'), counts))


def matched_synthetic_negatives(s, history, seed, num_u_old, num_v_old):
    """One negative per positive within each old/new endpoint category.

    This additionally matches the O-U mixture of old-U/new-V versus
    new-U/old-V, so collapsing these two strata reproduces exactly the
    manuscript's O-O/O-U/U-U target proportions.
    """
    positive_edges = s['edges']
    quotas = synthetic_stratum_counts(s, positive_edges, num_u_old, num_v_old)
    n = positive_edges.shape[1]
    if sum(quotas.values()) != n:
        raise AssertionError('Inconsistent positive stratum counts')

    old_u = np.asarray(s['keys_u']) < num_u_old
    old_v = np.asarray(s['keys_v']) < num_v_old
    u_groups = (np.flatnonzero(old_u), np.flatnonzero(~old_u))
    v_groups = (np.flatnonzero(old_v) + s['num_u'],
                np.flatnonzero(~old_v) + s['num_u'])
    known = _known_edges(s, history)
    rng = np.random.default_rng(seed)
    sampled = []
    for tag, (i, j) in zip(('OO', 'ON', 'NO', 'NN'),
                            ((0, 0), (0, 1), (1, 0), (1, 1))):
        needed = quotas[tag]
        if not needed:
            continue
        us, vs = u_groups[i], v_groups[j]
        known_count = sum(int(u in us and v in vs) for u, v in known)
        capacity = len(us) * len(vs) - known_count
        if capacity < needed:
            raise ValueError(
                f'Insufficient {tag} negatives: {capacity} available, '
                f'{needed} required; no silent redistribution allowed'
            )
        seen = set(known)
        chosen = []
        for _ in range(max(1000, 100 * needed)):
            pair = (int(us[rng.integers(len(us))]),
                    int(vs[rng.integers(len(vs))]))
            if pair not in seen:
                seen.add(pair)
                chosen.append(pair)
            if len(chosen) == needed:
                break
        if len(chosen) < needed:
            # Deterministic fallback for dense, small categories.
            for u in us:
                for v in vs:
                    pair = (int(u), int(v))
                    if pair not in seen:
                        chosen.append(pair)
                        seen.add(pair)
                    if len(chosen) == needed:
                        break
                if len(chosen) == needed:
                    break
        if len(chosen) != needed:
            raise AssertionError(f'Sampling shortfall in {tag}')
        sampled.extend(chosen)
    if len(sampled) != n or len(set(sampled)) != n:
        raise AssertionError('Invalid matched candidate count or duplication')
    if any(p in known for p in sampled):
        raise AssertionError('Negative overlaps with a known positive edge')
    rng.shuffle(sampled)
    return torch.tensor(sampled, dtype=torch.long).reshape(-1, 2).T
