"""
BiSBM (Bipartite Stochastic Block Model) Synthetic Graph Generator
for ablation study of WBT-BGRL weight-sensitivity variants.

Graph structure
---------------
  N_U = 420 nodes type U  (320 "old" / train-visible, 100 "new" / inductive)
  N_V = 288 nodes type V  (192 "old" / train-visible, 96  "new" / inductive)
  K_U =  4 blocks for U,  K_V = 2 blocks for V
  Block connection matrix B[k_u, k_v]:
    "Strong" pairs: (0,0), (1,0), (2,1), (3,1)  → p_in  = 0.05
    "Weak"   pairs: all others                   → p_out = 0.005

  Balanced split design
  ---------------------
  | Split   | O-O          | O-U / U-O      | U-U            | Total |
  |---------|--------------|----------------|----------------|-------|
  | TRAIN   | 960 (80%)    | 0              | 0              |  960  |
  | VAL     | 120 (10%)    | 100 (50%)      | 100 (50%)      |  320  |
  | TEST P+ | 120 (10%)    | 100 (50%)      | 100 (50%)      |  320  |
  | TEST P- | 120 (1:1)    | 100 (1:1)      | 100 (1:1)      |  320  |
  | TOTAL   | 1200         | 200            | 200            | 1600  |

Three weight conditions (one .npz file per condition)
------------------------------------------------------
  informative     : same-community edges get Pareto(1.2) * 5.0 weights
                    cross-community edges get Pareto(1.2) * 1.0 weights
  anti_correlated : opposite — cross-community edges are heavier
  orthogonal      : all edges get Pareto(1.5) weights, independent of community

Node features (numerical, inductive)
-------------------------------------
  U nodes (4 dims): [deg/max_deg, cos(2π*block/4), sin(2π*block/4), N(0,0.1)]
  V nodes (3 dims): [deg/max_deg, cos(2π*block/2), sin(2π*block/2)]
  New nodes in val/test → zero vector (same convention as real datasets)

Inductive temporal splits
--------------------------
  Old-old edges  : sorted by timestamp, split 80 / 10 / 10 (train / val / test)
  Inductive edges: (old-new, new-old, new-new) → split 50 / 50 (val / test)
  Pair types stored per edge: 'O-O', 'O-U', 'U-U'

Timestamps on train edges are stored for the temporal-window variant.

Usage
-----
  python data/synthetic/generator.py            # seed 42, all 3 conditions
  python data/synthetic/generator.py --seed 99
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# ── constants ────────────────────────────────────────────────────────────────

N_U_TOTAL   = 420
N_V_TOTAL   = 288   # N_V_OLD=192, N_V_NEW=96
N_U_OLD     = 320   # train-visible U nodes  (indices 0 .. N_U_OLD-1)
N_V_OLD     = 192   # train-visible V nodes  (indices 0 .. N_V_OLD-1)
K_U         = 4
K_V         = 2
P_IN        = 0.05
P_OUT       = 0.005
TARGET_OO_EDGES = 1200  # O-O: 960 train / 120 val / 120 test
TARGET_IND_OU   = 200   # O-U pool → 100 val + 100 test  (1:1 typed negatives in eval)
TARGET_IND_UU   = 200   # U-U pool → 100 val + 100 test  (raw E[U-U]≈264 with 100 new-U)
PARETO_ALPHA        = 1.2
WEIGHT_SCALE_STRONG = 5.0
WEIGHT_SCALE_WEAK   = 1.0

CONDITIONS = ['informative', 'anti_correlated', 'orthogonal']


# ── block connection matrix ───────────────────────────────────────────────────

def _build_B() -> np.ndarray:
    """B[k_u, k_v] = P_IN for 'strong' pairs, P_OUT otherwise."""
    B = np.full((K_U, K_V), P_OUT)
    B[0, 0] = P_IN
    B[1, 0] = P_IN
    B[2, 1] = P_IN
    B[3, 1] = P_IN
    return B


# ── graph generation ──────────────────────────────────────────────────────────

def generate_bisbn(seed: int = 42):
    """
    Generate the full BiSBM graph (structure + timestamps + features).

    Returns
    -------
    graph : dict
        All structural / feature / split data (condition-independent).
    B : ndarray [K_U, K_V]
        Block connection probability matrix.
    block_u, block_v : ndarray
        Block assignment for each U / V node.
    rng : numpy Generator
        Seeded RNG (so the caller can use it for condition-specific weights).
    """
    rng = np.random.default_rng(seed)

    print(f"\n{'='*70}")
    print(f"  BiSBM GENERATOR  (seed={seed})")
    print(f"  N_U={N_U_TOTAL} ({N_U_OLD} old + {N_U_TOTAL-N_U_OLD} new), N_V={N_V_TOTAL} ({N_V_OLD} old + {N_V_TOTAL-N_V_OLD} new)")  # noqa: E501
    print(f"  K_U={K_U}, K_V={K_V},  p_in={P_IN}, p_out={P_OUT}")
    print(f"{'='*70}")

    B = _build_B()

    # Block assignments: evenly distributed (so every block is represented)
    block_u = np.array([i % K_U for i in range(N_U_TOTAL)], dtype=np.int32)
    block_v = np.array([i % K_V for i in range(N_V_TOTAL)], dtype=np.int32)

    old_u = np.arange(N_U_OLD, dtype=np.int32)
    old_v = np.arange(N_V_OLD, dtype=np.int32)
    new_u = np.arange(N_U_OLD, N_U_TOTAL, dtype=np.int32)
    new_v = np.arange(N_V_OLD, N_V_TOTAL, dtype=np.int32)

    # ── sample edges ───────────────────────────────────────────────────────
    def _sample(u_ids, v_ids):
        rows, cols = [], []
        for u in u_ids:
            for v in v_ids:
                if rng.random() < B[block_u[u], block_v[v]]:
                    rows.append(u)
                    cols.append(v)
        if not rows:
            return np.empty((0, 2), dtype=np.int32)
        return np.column_stack([rows, cols]).astype(np.int32)

    oo = _sample(old_u, old_v)   # old-old
    ou = _sample(old_u, new_v)   # old-U / new-V  → inductive
    uo = _sample(new_u, old_v)   # new-U / old-V  → inductive
    uu = _sample(new_u, new_v)   # both new        → inductive

    n_oo, n_ou, n_uo, n_uu = len(oo), len(ou), len(uo), len(uu)
    print(f"\n[EDGES raw]  O-O={n_oo}  O-U={n_ou+n_uo}  U-U={n_uu}  total={n_oo+n_ou+n_uo+n_uu}")

    if n_oo < 10:
        raise RuntimeError("Too few old-old edges — adjust p_in/p_out or graph size.")

    # ── subsample to exact target counts ──────────────────────────────────
    if len(oo) > TARGET_OO_EDGES:
        idx = rng.choice(len(oo), TARGET_OO_EDGES, replace=False)
        idx.sort()
        oo  = oo[idx]

    # Merge O-U pool (both directions share label 'O-U') then subsample
    ou_parts = [x for x in [ou, uo] if len(x)]
    ou_pool  = np.vstack(ou_parts) if ou_parts else np.empty((0, 2), dtype=np.int32)
    if len(ou_pool) > TARGET_IND_OU:
        idx     = rng.choice(len(ou_pool), TARGET_IND_OU, replace=False)
        ou_pool = ou_pool[idx]

    if len(uu) > TARGET_IND_UU:
        idx = rng.choice(len(uu), TARGET_IND_UU, replace=False)
        uu  = uu[idx]

    n_oo = len(oo)   # update after subsampling
    print(f"[EDGES sub]  O-O={n_oo}  O-U={len(ou_pool)}  U-U={len(uu)}  "
          f"total={n_oo+len(ou_pool)+len(uu)}")

    # ── timestamp-based train/val/test split for O-O edges ────────────────
    oo_ts = rng.uniform(0.0, 1.0, size=n_oo).astype(np.float32)
    order = np.argsort(oo_ts)
    oo    = oo[order]
    oo_ts = oo_ts[order]

    n_train = int(0.80 * n_oo)
    n_val   = int(0.10 * n_oo)
    # n_test  = n_oo - n_train - n_val  (remainder)

    oo_train = oo[:n_train]
    oo_val   = oo[n_train : n_train + n_val]
    oo_test  = oo[n_train + n_val :]

    train_ts = oo_ts[:n_train]       # sorted timestamps for train edges

    # ── inductive edges → 50/50 val/test ──────────────────────────────────
    # Stratified 50/50 split: each type (O-U, U-U) is split independently
    # so that val and test have equal counts per type.
    ind_val_parts, ind_val_labels = [], []
    ind_test_parts, ind_test_labels = [], []
    for arr, label in [(ou_pool, 'O-U'), (uu, 'U-U')]:
        if len(arr) == 0:
            continue
        perm = rng.permutation(len(arr))
        arr  = arr[perm]
        half = len(arr) // 2
        ind_val_parts.append(arr[:half])
        ind_val_labels.extend([label] * half)
        ind_test_parts.append(arr[half:])
        ind_test_labels.extend([label] * (len(arr) - half))

    if ind_val_parts:
        ind_val   = np.vstack(ind_val_parts)
        ind_val_l = np.array(ind_val_labels)
        ind_test  = np.vstack(ind_test_parts)
        ind_test_l = np.array(ind_test_labels)
    else:
        ind_val = ind_test = np.empty((0, 2), dtype=np.int32)
        ind_val_l = ind_test_l = np.array([], dtype=str)

    # Merge O-O and inductive portions for val / test
    def _vstack_safe(*arrs):
        arrs = [a for a in arrs if len(a)]
        return np.vstack(arrs) if arrs else np.empty((0, 2), dtype=np.int32)

    def _concat_safe(*arrs):
        arrs = [a for a in arrs if len(a)]
        return np.concatenate(arrs) if arrs else np.array([], dtype=str)

    val_edges  = _vstack_safe(oo_val,  ind_val)
    test_edges = _vstack_safe(oo_test, ind_test)

    val_types  = _concat_safe(np.full(len(oo_val),  'O-O'), ind_val_l)
    test_types = _concat_safe(np.full(len(oo_test), 'O-O'), ind_test_l)

    print(f"\n[SPLITS]")
    _ptsummary("TRAIN", oo_train, np.full(len(oo_train), 'O-O'))
    _ptsummary("VAL",   val_edges,  val_types)
    _ptsummary("TEST",  test_edges, test_types)

    # ── node features ──────────────────────────────────────────────────────
    # Mirrors the real-dataset protocol:
    #   TRAIN features  : degree from TRAIN only
    #   VAL   features  : degree from TRAIN only          (no leakage of val edges)
    #   TEST  features  : degree from TRAIN + VAL         (val is now "history")
    # Normalization always uses TRAIN-fitted max (same as MinMaxScaler on real datasets).

    # TRAIN-only degree (used for TRAIN and VAL features)
    deg_u = np.zeros(N_U_TOTAL, dtype=np.float32)
    deg_v = np.zeros(N_V_TOTAL, dtype=np.float32)
    for u, v in oo_train:
        deg_u[u] += 1
        deg_v[v] += 1
    max_du = max(deg_u.max(), 1.0)   # normalisation anchor — always from TRAIN
    max_dv = max(deg_v.max(), 1.0)

    # TRAIN + VAL degree (used for TEST features only)
    deg_u_tv = deg_u.copy()
    deg_v_tv = deg_v.copy()
    for u, v in val_edges:          # val_edges contains O-O val + inductive val
        if u < N_U_TOTAL:
            deg_u_tv[u] += 1
        if v < N_V_TOTAL:
            deg_v_tv[v] += 1

    # Feature builders — accept an explicit degree array and noise seed
    noise_u = rng.normal(0, 0.1, N_U_TOTAL).astype(np.float32)  # fixed per node

    def _u_feat(ids, deg=None):
        """Build U-node features. deg defaults to TRAIN-only degree."""
        if deg is None:
            deg = deg_u
        feat = np.zeros((len(ids), 4), dtype=np.float32)
        for i, u in enumerate(ids):
            if u < N_U_OLD:            # old node has real features
                k = int(block_u[u])
                feat[i, 0] = deg[u] / max_du   # normalised with TRAIN max
                feat[i, 1] = float(np.cos(2 * np.pi * k / K_U))
                feat[i, 2] = float(np.sin(2 * np.pi * k / K_U))
                feat[i, 3] = noise_u[u]        # same noise regardless of split
            # else: new node → zero vector (inductive protocol)
        return feat

    def _v_feat(ids, deg=None):
        """Build V-node features. deg defaults to TRAIN-only degree."""
        if deg is None:
            deg = deg_v
        feat = np.zeros((len(ids), 3), dtype=np.float32)
        for i, v in enumerate(ids):
            if v < N_V_OLD:
                k = int(block_v[v])
                feat[i, 0] = deg[v] / max_dv   # normalised with TRAIN max
                feat[i, 1] = float(np.cos(2 * np.pi * k / K_V))
                feat[i, 2] = float(np.sin(2 * np.pi * k / K_V))
        return feat

    # Unique node IDs per split
    def _uid(edges_list):
        """Union of unique U-indices from a list of edge arrays."""
        cols = [e[:, 0] for e in edges_list if len(e)]
        return np.unique(np.concatenate(cols)) if cols else np.array([], dtype=np.int32)

    def _vid(edges_list):
        cols = [e[:, 1] for e in edges_list if len(e)]
        return np.unique(np.concatenate(cols)) if cols else np.array([], dtype=np.int32)

    train_u_ids = _uid([oo_train])
    train_v_ids = _vid([oo_train])
    val_u_ids   = _uid([val_edges])
    val_v_ids   = _vid([val_edges])
    test_u_ids  = _uid([test_edges])
    test_v_ids  = _vid([test_edges])

    graph = {
        'oo_train'   : oo_train,
        'val_edges'  : val_edges,
        'test_edges' : test_edges,
        'val_types'  : val_types,
        'test_types' : test_types,
        'train_ts'   : train_ts,
        # node feature arrays — VAL uses PAST-only degree, TEST uses PAST+PRESENT degree
        'train_u_feat': _u_feat(train_u_ids),              # TRAIN degree
        'train_v_feat': _v_feat(train_v_ids),              # TRAIN degree
        'val_u_feat'  : _u_feat(val_u_ids),                # TRAIN degree (no leakage)
        'val_v_feat'  : _v_feat(val_v_ids),                # TRAIN degree (no leakage)
        'test_u_feat' : _u_feat(test_u_ids, deg=deg_u_tv), # TRAIN+VAL degree
        'test_v_feat' : _v_feat(test_v_ids, deg=deg_v_tv), # TRAIN+VAL degree
        # node ID sets
        'train_u_ids' : train_u_ids,
        'train_v_ids' : train_v_ids,
        'val_u_ids'   : val_u_ids,
        'val_v_ids'   : val_v_ids,
        'test_u_ids'  : test_u_ids,
        'test_v_ids'  : test_v_ids,
        # block info
        'block_u': block_u,
        'block_v': block_v,
    }

    return graph, B, block_u, block_v, rng


def _ptsummary(tag, edges, types):
    n = len(edges)
    oo = int((types == 'O-O').sum())
    ou = int((types == 'O-U').sum())
    uu = int((types == 'U-U').sum())
    print(f"  {tag:6s}: {n:4d} edges  (O-O={oo}, O-U={ou}, U-U={uu})")


# ── weight computation ────────────────────────────────────────────────────────

def compute_weights(
    edges: np.ndarray,
    condition: str,
    B: np.ndarray,
    block_u: np.ndarray,
    block_v: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Assign edge weights according to *condition*.

    informative    : strong-pair edges → Pareto * WEIGHT_SCALE_STRONG
                     weak-pair  edges → Pareto * WEIGHT_SCALE_WEAK
    anti_correlated: inverted multipliers
    orthogonal     : uniform Pareto, independent of community

    Weights are always >= 1.0 (Pareto minimum).
    """
    if len(edges) == 0:
        return np.array([], dtype=np.float32)

    n = len(edges)
    # Pareto(alpha) + 1  → minimum value = 1.0
    alpha = PARETO_ALPHA if condition != 'orthogonal' else PARETO_ALPHA + 0.3
    base = (rng.pareto(alpha, n) + 1.0).astype(np.float32)

    if condition == 'orthogonal':
        return base

    # Is each edge a "strong community" pair?
    strong = np.array(
        [B[int(block_u[u]), int(block_v[v])] == P_IN for u, v in edges],
        dtype=bool,
    )

    if condition == 'informative':
        mult = np.where(strong, WEIGHT_SCALE_STRONG, WEIGHT_SCALE_WEAK).astype(np.float32)
    elif condition == 'anti_correlated':
        mult = np.where(strong, WEIGHT_SCALE_WEAK, WEIGHT_SCALE_STRONG).astype(np.float32)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    return base * mult


# ── local re-indexing ─────────────────────────────────────────────────────────

def remap_edges(edges: np.ndarray, u_ids: np.ndarray, v_ids: np.ndarray) -> np.ndarray:
    """
    Remap global node IDs to local contiguous indices.
    V-indices are offset by len(u_ids) (same convention as real datasets).

    Returns: int64 array [2, E]
    """
    u_map = {int(og): i         for i, og in enumerate(u_ids)}
    v_map = {int(og): i + len(u_ids) for i, og in enumerate(v_ids)}
    lu = np.array([u_map[int(u)] for u in edges[:, 0]], dtype=np.int64)
    lv = np.array([v_map[int(v)] for v in edges[:, 1]], dtype=np.int64)
    return np.vstack([lu, lv])


# ── save one condition ────────────────────────────────────────────────────────

def save_condition(
    out_dir: Path,
    graph: dict,
    B: np.ndarray,
    block_u: np.ndarray,
    block_v: np.ndarray,
    rng: np.random.Generator,
    condition: str,
    seed: int,
):
    """Compute weights for *condition* and write synthetic_<condition>.npz."""
    g = graph  # shorthand

    train_w = compute_weights(g['oo_train'],   condition, B, block_u, block_v, rng)
    val_w   = compute_weights(g['val_edges'],  condition, B, block_u, block_v, rng)
    test_w  = compute_weights(g['test_edges'], condition, B, block_u, block_v, rng)

    # Vocab indices: old node i → vocab index i+1 (0 = UNK for new nodes)
    def _vocab_ids(ids):
        return np.array([
            int(i) + 1 if int(i) < N_U_OLD or int(i) < N_V_OLD else 0
            for i in ids
        ], dtype=np.int64)

    def _vocab_ids_u(ids):
        return np.array(
            [int(i) + 1 if int(i) < N_U_OLD else 0 for i in ids], dtype=np.int64
        )

    def _vocab_ids_v(ids):
        return np.array(
            [int(i) + 1 if int(i) < N_V_OLD else 0 for i in ids], dtype=np.int64
        )

    out_file = out_dir / f'synthetic_{condition}.npz'
    np.savez(
        out_file,
        # TRAIN
        train_edge_index   = remap_edges(g['oo_train'],   g['train_u_ids'], g['train_v_ids']),
        train_edge_weights = train_w,
        train_u_features   = g['train_u_feat'],
        train_v_features   = g['train_v_feat'],
        train_u_ids        = _vocab_ids_u(g['train_u_ids']),
        train_v_ids        = _vocab_ids_v(g['train_v_ids']),
        train_u_ids_global = g['train_u_ids'],
        train_v_ids_global = g['train_v_ids'],
        train_timestamps   = g['train_ts'],
        # VAL
        val_edge_index     = remap_edges(g['val_edges'],   g['val_u_ids'], g['val_v_ids']),
        val_edge_weights   = val_w,
        val_u_features     = g['val_u_feat'],
        val_v_features     = g['val_v_feat'],
        val_u_ids_global   = g['val_u_ids'],
        val_v_ids_global   = g['val_v_ids'],
        val_u_ids          = _vocab_ids_u(g['val_u_ids']),
        val_v_ids          = _vocab_ids_v(g['val_v_ids']),
        val_pair_types     = g['val_types'],
        # TEST
        test_edge_index    = remap_edges(g['test_edges'],  g['test_u_ids'], g['test_v_ids']),
        test_edge_weights  = test_w,
        test_u_features    = g['test_u_feat'],
        test_v_features    = g['test_v_feat'],
        test_u_ids         = _vocab_ids_u(g['test_u_ids']),
        test_v_ids         = _vocab_ids_v(g['test_v_ids']),
        test_pair_types    = g['test_types'],
        # metadata
        block_u            = block_u,
        block_v            = block_v,
        condition          = np.array([condition]),
        seed               = np.array([seed]),
        n_u_total          = np.array([N_U_TOTAL]),
        n_v_total          = np.array([N_V_TOTAL]),
        n_u_old            = np.array([N_U_OLD]),
        n_v_old            = np.array([N_V_OLD]),
    )

    print(f"\n[SAVED] {out_file.name}  condition={condition}")
    if len(train_w):
        print(f"  weights train  range=[{train_w.min():.2f}, {train_w.max():.2f}]"
              f"  mean={train_w.mean():.2f} std={train_w.std():.2f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic BiSBM dataset')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for graph structure (default: 42)')
    args = parser.parse_args()

    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    graph, B, block_u, block_v, rng = generate_bisbn(args.seed)

    for cond in CONDITIONS:
        # Each condition gets its own sub-RNG state so weights are independent
        save_condition(out_dir, graph, B, block_u, block_v, rng, cond, args.seed)

    print(f"\n{'='*70}")
    print(f"[DONE]  {len(CONDITIONS)} files saved in  {out_dir.resolve()}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
