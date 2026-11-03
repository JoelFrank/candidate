"""Type-matched synthetic negative sampling for BiSBM-EASY.

OO/ON/NO/NN are defined relative to each evaluation stage's previously
observed node roster. Negative edges exclude only observed target/history
positives. The hidden block labels are NOT used for sampling or training.
"""
from __future__ import annotations
import numpy as np
import torch

# This is the early/late arrival design of generator_easy_project5.py.
OLD_CUTOFFS = {
    'train': (320, 192),
    'validation': (320, 192),
    'test': (384, 256),
}
STRATA = ('OO','ON','NO','NN')


def strata_counts(snapshot, edge_index, cutoff_u: int, cutoff_v: int):
    keys_u = np.asarray(snapshot['keys_u'])
    keys_v = np.asarray(snapshot['keys_v'])
    nu = int(snapshot['num_u'])
    counts = dict.fromkeys(STRATA, 0)
    for u, v in edge_index.T.tolist():
        new_u = int(keys_u[u] >= cutoff_u)
        new_v = int(keys_v[v-nu] >= cutoff_v)
        counts[STRATA[new_u*2 + new_v]] += 1
    return counts


def sample_matched(snapshot, history, seed: int, cutoff_u: int, cutoff_v: int, remap_context):
    pos = snapshot['edges']
    quotas = strata_counts(snapshot, pos, cutoff_u, cutoff_v)
    nu = snapshot['num_u']
    keys_u = np.asarray(snapshot['keys_u'])
    keys_v = np.asarray(snapshot['keys_v'])
    ugroups = [np.flatnonzero(keys_u < cutoff_u), np.flatnonzero(keys_u >= cutoff_u)]
    vgroups = [np.flatnonzero(keys_v < cutoff_v) + nu, np.flatnonzero(keys_v >= cutoff_v) + nu]
    known, _ = remap_context(history, snapshot)
    known_set = set(map(tuple, known.T.tolist()))
    rng = np.random.default_rng(seed)
    chosen = []
    for tag, ui, vi in (('OO',0,0),('ON',0,1),('NO',1,0),('NN',1,1)):
        amount = quotas[tag]
        if amount == 0:
            continue
        us, vs = ugroups[ui], vgroups[vi]
        if len(us) == 0 or len(vs) == 0:
            raise ValueError(f'No endpoints to sample {tag} negatives')
        # Complement fallback avoids rejection bottlenecks in dense small strata.
        known_count = sum((u in us and v in vs) for u,v in known_set)
        if len(us)*len(vs)-known_count < amount:
            raise ValueError(f'Insufficient {tag} negative capacity for {amount} samples')
        seen = set(known_set)
        current=[]
        for _ in range(max(1000, 100 * amount)):
            candidate=(int(us[rng.integers(len(us))]),int(vs[rng.integers(len(vs))]))
            if candidate not in seen:
                current.append(candidate);seen.add(candidate)
            if len(current) == amount:
                break
        if len(current) < amount:
            for u in us:
                for v in vs:
                    candidate = (int(u),int(v))
                    if candidate not in seen:
                        current.append(candidate);seen.add(candidate)
                    if len(current) == amount:
                        break
                if len(current) == amount:
                    break
        if len(current) != amount:
            raise AssertionError(f'Sampled {len(current)} negatives for {tag}; expected {amount}')
        chosen.extend(current)
    if len(chosen) != pos.shape[1] or len(set(chosen)) != len(chosen):
        raise AssertionError('Invalid negative candidate count / duplicates')
    if any(x in known_set for x in chosen):
        raise AssertionError('Negative candidate overlaps observed positives')
    rng.shuffle(chosen)
    return torch.tensor(chosen, dtype=torch.long).reshape(-1,2).T
