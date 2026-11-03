"""
Export synthetic BiSBM graphs to GEXF format for Gephi visualization.
Produces one .gexf file per condition inside data/synthetic/gephi/.

Node attributes  : node_type (U/V), block, is_new, feat_0..3
Edge attributes  : weight_value, split (train/val/test), pair_type (O-O/O-U/U-U)
Visual encoding  : node colour by block, grey = new node; edge colour by split

Usage
-----
  python data/synthetic/export_gephi.py            # seed 42
  python data/synthetic/export_gephi.py --seed 99
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.synthetic.generator import (   # noqa: E402
    generate_bisbn,
    compute_weights,
    CONDITIONS,
    N_U_OLD,
    N_V_OLD,
    N_U_TOTAL,
    N_V_TOTAL,
)

# ── colour palettes ───────────────────────────────────────────────────────────
# U blocks (blue family, 4 blocks)
_U_COLORS = [
    ( 31, 119, 180),   # block 0 – steel blue
    ( 23, 190, 207),   # block 1 – cyan
    ( 44, 160,  44),   # block 2 – green
    (188, 189,  34),   # block 3 – olive
]
# V blocks (warm family, 2 blocks)
_V_COLORS = [
    (214,  39,  40),   # block 0 – red
    (255, 127,  14),   # block 1 – orange
]
_NEW_COLOR   = (160, 160, 160)   # grey for inductive (new) nodes
_EDGE_COLORS = {
    'train': (  0, 153,  76),    # green
    'val'  : (255, 165,   0),    # amber
    'test' : (220,  20,  60),    # crimson
}

GEXF_NS = 'http://gexf.net/1.3'
VIZ_NS  = 'http://gexf.net/1.3/viz'


def _rgb(t, alpha='1.0'):
    return {'r': str(t[0]), 'g': str(t[1]), 'b': str(t[2]), 'a': alpha}


# ── GEXF builder ──────────────────────────────────────────────────────────────

def build_gexf(
    graph: dict,
    block_u: np.ndarray,
    block_v: np.ndarray,
    train_w: np.ndarray,
    val_w: np.ndarray,
    test_w: np.ndarray,
    condition: str,
    seed: int,
) -> ET.ElementTree:
    """Return an ElementTree representing the full graph for *condition*."""
    ET.register_namespace('',    GEXF_NS)
    ET.register_namespace('viz', VIZ_NS)

    root = ET.Element('gexf', {
        'xmlns':   GEXF_NS,
        'version': '1.3',
    })
    meta = ET.SubElement(root, 'meta', {'lastmodifieddate': '2026-03-05'})
    ET.SubElement(meta, 'creator').text    = 'BiSBM export_gephi.py'
    ET.SubElement(meta, 'description').text = (
        f'Synthetic BiSBM  condition={condition}  seed={seed}'
    )

    g_el = ET.SubElement(root, 'graph', {
        'defaultedgetype': 'undirected',
        'mode':            'static',
    })

    # ── attribute declarations ─────────────────────────────────────────────
    na = ET.SubElement(g_el, 'attributes', {'class': 'node'})
    for aid, title, atype in [
        ('ntype',  'node_type', 'string'),
        ('block',  'block',     'integer'),
        ('is_new', 'is_new',    'boolean'),
        ('f0',     'feat_0',    'float'),
        ('f1',     'feat_1',    'float'),
        ('f2',     'feat_2',    'float'),
        ('f3',     'feat_3',    'float'),
    ]:
        ET.SubElement(na, 'attribute', {'id': aid, 'title': title, 'type': atype})

    ea = ET.SubElement(g_el, 'attributes', {'class': 'edge'})
    for aid, title, atype in [
        ('wv',  'weight_value', 'float'),
        ('spl', 'split',        'string'),
        ('pt',  'pair_type',    'string'),
    ]:
        ET.SubElement(ea, 'attribute', {'id': aid, 'title': title, 'type': atype})

    # ── build per-node feature lookup (global_id → feature vector) ────────
    u_feat: dict[int, np.ndarray] = {}
    v_feat: dict[int, np.ndarray] = {}

    for ids_arr, feat_arr in [
        (graph['train_u_ids'], graph['train_u_feat']),
        (graph['val_u_ids'],   graph['val_u_feat']),
        (graph['test_u_ids'],  graph['test_u_feat']),
    ]:
        for i, uid in enumerate(ids_arr):
            if int(uid) not in u_feat:
                u_feat[int(uid)] = feat_arr[i]

    for ids_arr, feat_arr in [
        (graph['train_v_ids'], graph['train_v_feat']),
        (graph['val_v_ids'],   graph['val_v_feat']),
        (graph['test_v_ids'],  graph['test_v_feat']),
    ]:
        for i, vid in enumerate(ids_arr):
            if int(vid) not in v_feat:
                v_feat[int(vid)] = feat_arr[i]

    # ── collect ALL node IDs (including isolated nodes) ──────────────────
    used_u = np.arange(N_U_TOTAL, dtype=np.int64)
    used_v = np.arange(N_V_TOTAL, dtype=np.int64)

    oo_train   = graph['oo_train']
    val_edges  = graph['val_edges']
    test_edges = graph['test_edges']

    # ── nodes ─────────────────────────────────────────────────────────────
    nodes_el = ET.SubElement(g_el, 'nodes')

    def _node(nid_str, label, ntype, blk_idx, is_new, feat_vec, palette):
        n = ET.SubElement(nodes_el, 'node', {'id': nid_str, 'label': label})
        avs = ET.SubElement(n, 'attvalues')
        ET.SubElement(avs, 'attvalue', {'for': 'ntype',  'value': ntype})
        ET.SubElement(avs, 'attvalue', {'for': 'block',  'value': str(blk_idx)})
        ET.SubElement(avs, 'attvalue', {'for': 'is_new', 'value': 'true' if is_new else 'false'})
        feat_ids = ['f0', 'f1', 'f2', 'f3']
        for fi, val in enumerate(feat_vec):
            ET.SubElement(avs, 'attvalue', {'for': feat_ids[fi], 'value': f'{float(val):.6f}'})
        col = _NEW_COLOR if is_new else palette[blk_idx % len(palette)]
        ET.SubElement(n, f'{{{VIZ_NS}}}color', _rgb(col))
        ET.SubElement(n, f'{{{VIZ_NS}}}size',  {'value': '5' if is_new else '10'})
        shape = 'disc' if ntype == 'U' else 'square'
        ET.SubElement(n, f'{{{VIZ_NS}}}shape', {'value': shape})

    for uid in used_u:
        uid_i  = int(uid)
        is_new = uid_i >= N_U_OLD
        feat   = u_feat.get(uid_i, np.zeros(4, np.float32))
        _node(f'U_{uid_i}', f'U{uid_i}', 'U', int(block_u[uid_i]), is_new, feat, _U_COLORS)

    for vid in used_v:
        vid_i  = int(vid)
        is_new = vid_i >= N_V_OLD
        feat   = v_feat.get(vid_i, np.zeros(3, np.float32))
        _node(f'V_{vid_i}', f'V{vid_i}', 'V', int(block_v[vid_i]), is_new, feat, _V_COLORS)

    # ── edges ─────────────────────────────────────────────────────────────
    edges_el = ET.SubElement(g_el, 'edges')
    eid = 0

    def _add_edges(arr, weights, split, types):
        nonlocal eid
        for i, (u, v) in enumerate(arr):
            w  = float(weights[i]) if len(weights) > i else 1.0
            pt = str(types[i]) if types is not None and len(types) > i else 'O-O'
            e  = ET.SubElement(edges_el, 'edge', {
                'id': str(eid),
                'source': f'U_{int(u)}',
                'target': f'V_{int(v)}',
                'weight': f'{w:.4f}',
            })
            avs = ET.SubElement(e, 'attvalues')
            ET.SubElement(avs, 'attvalue', {'for': 'wv',  'value': f'{w:.4f}'})
            ET.SubElement(avs, 'attvalue', {'for': 'spl', 'value': split})
            ET.SubElement(avs, 'attvalue', {'for': 'pt',  'value': pt})
            ET.SubElement(e, f'{{{VIZ_NS}}}color',
                          _rgb(_EDGE_COLORS[split], alpha='0.7'))
            eid += 1

    _add_edges(oo_train,   train_w, 'train', np.full(len(oo_train),  'O-O'))
    _add_edges(val_edges,  val_w,   'val',   graph['val_types'])
    _add_edges(test_edges, test_w,  'test',  graph['test_types'])

    ET.indent(root, space='  ')
    return ET.ElementTree(root)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Export BiSBM synthetic graphs to .gexf for Gephi'
    )
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed used when generating (must match generator, default=42)')
    args = parser.parse_args()

    out_dir = Path(__file__).parent / 'gephi'
    out_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  BiSBM -> GEXF  (seed={args.seed})")
    print(f"{'='*60}")

    graph, B, block_u, block_v, rng = generate_bisbn(args.seed)

    for cond in CONDITIONS:
        train_w = compute_weights(graph['oo_train'],   cond, B, block_u, block_v, rng)
        val_w   = compute_weights(graph['val_edges'],  cond, B, block_u, block_v, rng)
        test_w  = compute_weights(graph['test_edges'], cond, B, block_u, block_v, rng)

        tree = build_gexf(
            graph, block_u, block_v,
            train_w, val_w, test_w,
            cond, args.seed,
        )

        out_file = out_dir / f'synthetic_{cond}.gexf'
        tree.write(str(out_file), encoding='utf-8', xml_declaration=True)

        n_e = len(graph['oo_train']) + len(graph['val_edges']) + len(graph['test_edges'])
        print(f"  [OK] {out_file.name}  ({n_e} edges)")

    print(f"\n  Files saved in: {out_dir.resolve()}")
    print("  -> Gephi: File > Open  (or drag-and-drop each .gexf)")
    print()
    print("  Suggested Gephi workflow:")
    print("    1. Layout      : ForceAtlas2 (prevent overlap ON)")
    print("    2. Node colour : Partition  -> node_type  (U vs V)")
    print("       or          : Partition  -> block      (community)")
    print("       or          : Partition  -> is_new     (old vs inductive)")
    print("    3. Edge colour : Partition  -> split      (train/val/test)")
    print("       or          : Ranking    -> weight_value")
    print("    4. Node size   : Ranking    -> Degree")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
