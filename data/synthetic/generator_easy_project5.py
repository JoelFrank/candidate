"""BiSBM-EASY for project5: easy separable bipartite benchmark.

This synthetic control keeps weights orthogonal (iid lognormal) and makes the
positive link rule easy to recover from node attributes: a pair can be positive
only when the latent class of the U-node matches the latent class of the V-node.

Purpose: positive control / separability check. It is NOT intended to replace
BiSBM v2 and must be reported separately.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

NU_OLD, NV_OLD = 320, 192
COHORT_U, COHORT_V = 64, 64
GROUPS = 16
COUNTS = {
    'train': (900,),
    'validation': (140, 90, 90, 40),
    'test': (140, 90, 90, 40),
}


def _labels(n: int, groups: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.arange(n, dtype=np.int64) % groups
    rng.shuffle(y)
    return y


def _attributes(labels: np.ndarray, groups: int, seed: int, noise: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.eye(groups, dtype=np.float32)[labels]
    x += rng.normal(0.0, noise, x.shape).astype(np.float32)
    return x


def _eligible(us, vs, yu, yv, forbidden: set[tuple[int, int]]):
    a = np.repeat(np.asarray(us, dtype=np.int64), len(vs))
    b = np.tile(np.asarray(vs, dtype=np.int64), len(us))
    ok = yu[a] == yv[b]
    pairs = np.column_stack([a[ok], b[ok]])
    if forbidden:
        keep = np.fromiter((tuple(map(int, p)) not in forbidden for p in pairs), dtype=bool, count=len(pairs))
        pairs = pairs[keep]
    return pairs


def _choose(us, vs, yu, yv, count: int, rng, forbidden: set[tuple[int, int]]):
    cand = _eligible(us, vs, yu, yv, forbidden)
    if len(cand) < count:
        raise RuntimeError(f'Not enough easy-compatible pairs ({len(cand)} < {count})')
    idx = np.sort(rng.choice(len(cand), count, replace=False))
    chosen = cand[idx].astype(np.int64)
    forbidden.update((int(u), int(v)) for u, v in chosen)
    return chosen


def generate(graph_seed: int, output: Path, groups: int = GROUPS, feature_noise: float = 0.02, allow_overwrite: bool = False) -> Path:
    if groups < 2 or NU_OLD % groups or NV_OLD % groups or COHORT_U % groups or COHORT_V % groups:
        raise ValueError('groups must divide 320, 192, 64 and 64; recommended 16')
    out = Path(output) / f'graph_{graph_seed}' / 'stable' / 'synthetic_easy.npz'
    if out.exists() and not allow_overwrite:
        with np.load(out, allow_pickle=False) as saved:
            expected = (int(saved['graph_seed'][0]), int(saved['easy_groups'][0]),
                        float(saved['easy_feature_noise'][0]))
            requested = (int(graph_seed), int(groups), float(np.float32(feature_noise)))
            if expected != requested:
                raise ValueError(f'Existing BiSBM-EASY file has different parameters: {expected} != {requested}; '
                                 'use a separate destination instead of overwriting')
        return out
    nu = NU_OLD + 2 * COHORT_U
    nv = NV_OLD + 2 * COHORT_V
    yu = _labels(nu, groups, graph_seed * 31 + 1)
    yv = _labels(nv, groups, graph_seed * 31 + 2)
    xu = _attributes(yu, groups, graph_seed * 31 + 3, feature_noise)
    xv = _attributes(yv, groups, graph_seed * 31 + 4, feature_noise)
    rng = np.random.default_rng(graph_seed * 92821 + 17)
    wrng = np.random.default_rng(graph_seed * 92821 + 19)
    used: set[tuple[int, int]] = set()

    # Train roster uses only old nodes.
    train_u = np.arange(NU_OLD, dtype=np.int64)
    train_v = np.arange(NV_OLD, dtype=np.int64)
    train_edges = _choose(train_u, train_v, yu, yv, COUNTS['train'][0], rng, used)

    # Validation adds the first new cohort.
    val_new_u = np.arange(NU_OLD, NU_OLD + COHORT_U, dtype=np.int64)
    val_new_v = np.arange(NV_OLD, NV_OLD + COHORT_V, dtype=np.int64)
    val_edges = np.vstack([
        _choose(train_u, train_v, yu, yv, COUNTS['validation'][0], rng, used),
        _choose(train_u, val_new_v, yu, yv, COUNTS['validation'][1], rng, used),
        _choose(val_new_u, train_v, yu, yv, COUNTS['validation'][2], rng, used),
        _choose(val_new_u, val_new_v, yu, yv, COUNTS['validation'][3], rng, used),
    ])
    val_roster_u = np.concatenate([train_u, val_new_u])
    val_roster_v = np.concatenate([train_v, val_new_v])

    # Test adds the second new cohort; the first cohort remains in roster.
    test_new_u = np.arange(NU_OLD + COHORT_U, NU_OLD + 2 * COHORT_U, dtype=np.int64)
    test_new_v = np.arange(NV_OLD + COHORT_V, NV_OLD + 2 * COHORT_V, dtype=np.int64)
    test_edges = np.vstack([
        _choose(val_roster_u, val_roster_v, yu, yv, COUNTS['test'][0], rng, used),
        _choose(val_roster_u, test_new_v, yu, yv, COUNTS['test'][1], rng, used),
        _choose(test_new_u, val_roster_v, yu, yv, COUNTS['test'][2], rng, used),
        _choose(test_new_u, test_new_v, yu, yv, COUNTS['test'][3], rng, used),
    ])
    test_roster_u = np.concatenate([val_roster_u, test_new_u])
    test_roster_v = np.concatenate([val_roster_v, test_new_v])

    def weights(n: int) -> np.ndarray:
        return np.clip(wrng.lognormal(mean=0.6, sigma=0.7, size=n), 0.3, 20.0).astype(np.float32)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        graph_seed=np.array([graph_seed], dtype=np.int64),
        regime=np.array(['stable']),
        condition=np.array(['orthogonal']),
        easy_groups=np.array([groups], dtype=np.int64),
        easy_feature_noise=np.array([feature_noise], dtype=np.float32),
        positive_rule=np.array(['same_latent_class']),
        weight_policy=np.array(['iid_lognormal_orthogonal']),
        node_u_attributes=xu,
        node_v_attributes=xv,
        node_u_blocks_diagnostic_only=yu,
        node_v_blocks_diagnostic_only=yv,
        train_pairs_global=train_edges,
        validation_pairs_global=val_edges,
        test_pairs_global=test_edges,
        train_weights=weights(len(train_edges)),
        validation_weights=weights(len(val_edges)),
        test_weights=weights(len(test_edges)),
        train_candidate_u=train_u,
        train_candidate_v=train_v,
        validation_candidate_u=val_roster_u,
        validation_candidate_v=val_roster_v,
        test_candidate_u=test_roster_u,
        test_candidate_v=test_roster_v,
    )
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph-seeds', type=int, nargs='+', default=[111, 112, 113])
    p.add_argument('--output', type=Path, default=Path('data/synthetic/bisbm_easy_v1'))
    p.add_argument('--groups', type=int, default=GROUPS)
    p.add_argument('--feature-noise', type=float, default=0.02)
    p.add_argument('--allow-overwrite', action='store_true')
    a = p.parse_args(argv)
    for seed in a.graph_seeds:
        path = generate(seed, a.output, a.groups, a.feature_noise, a.allow_overwrite)
        print(f'GENERATED {path}', flush=True)


if __name__ == '__main__':
    main()
