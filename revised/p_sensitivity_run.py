"""Controlled P-only sensitivity run: transform edge weights AT GCN input only.

The original WBTBGRL, generator, decoder, negative sampler and revised/run.py
remain untouched. In this separate executable A=S=0, decoder=NWB.
No validation/test weights are used to fit clipping thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from revised import run as base
from src.models.wbtbgrl import WeightedGCNEncoder

ROOT = Path(__file__).resolve().parents[1]
RELEVANT = [
    'revised/run.py', 'revised/data.py', 'revised/negative_sampling.py',
    'src/models/wbtbgrl.py', 'data/synthetic/generator.py',
    'revised/p_sensitivity_run.py',
]
MODES = ('raw', 'log1p_anchor1', 'clip_train_p99')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_weight_report(weights, edges, num_u, num_v, *, quantile=0.99):
    w = weights.detach().float().cpu().numpy().astype('float64')
    if w.size == 0 or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError('Training weights must be finite, strictly positive and nonempty')
    edge = edges.detach().cpu().numpy()
    if edge.shape != (2, len(w)):
        raise ValueError('Edge/weight shape mismatch')
    u, v = edge[0], edge[1] - num_u
    if not ((u >= 0).all() and (u < num_u).all() and (v >= 0).all() and (v < num_v).all()):
        raise ValueError('Expected U->V bipartite training edges')
    d_u = np.bincount(u, weights=w, minlength=num_u)
    d_v = np.bincount(v, weights=w, minlength=num_v)
    n_heavy = max(1, math.ceil(len(w) * .01))
    return {
        'train_edge_count': len(w), 'minimum': float(w.min()),
        'median': float(np.median(w)), 'p95': float(np.quantile(w, .95)),
        'p99': float(np.quantile(w, .99)), 'maximum': float(w.max()),
        'total_weight': float(w.sum()),
        'top_1pct_edges_weight_fraction': float(np.sort(w)[-n_heavy:].sum() / w.sum()),
        'weighted_degree_u_max': float(d_u.max()), 'weighted_degree_v_max': float(d_v.max()),
        'weighted_degree_u_p99': float(np.quantile(d_u, .99)),
        'weighted_degree_v_p99': float(np.quantile(d_v, .99)),
        'clip_quantile': float(quantile),
        'clip_threshold_train_only': float(np.quantile(w, quantile)),
    }


def transform_weight(w, mode: str, cap: float | None = None):
    if w is None or mode == 'raw':
        return w
    if mode == 'log1p_anchor1':
        # Preserves the original weight=1 for corrupted edges and self-loop scale.
        return torch.log1p(w) / math.log(2)
    if mode == 'clip_train_p99':
        if cap is None or cap <= 0:
            raise ValueError('Missing TRAIN-derived clipping threshold')
        return w.clamp(max=cap)
    raise ValueError(f'Unknown mode: {mode}')


def check_isolated_p_only(args):
    if args.dataset != 'synthetic' or args.model != 'WBTBGRL':
        raise ValueError('This diagnostic is restricted to synthetic WBTBGRL')
    if args.variant != 'WP_NWB' or args.propagation is not True or args.augmentation is not False or args.loss_weights is not False:
        raise ValueError('Controlled experiment requires P=1, A=0, S=0, NWB decoder')
    if args.negative_strategy != 'matched_synthetic':
        raise ValueError('Requires matched_synthetic negatives')
    if not 0 < args.clip_quantile < 1:
        raise ValueError('clip_quantile must be strictly between 0 and 1')
    if args.propagation_mode not in MODES:
        raise ValueError('Unknown propagation mode')


def execute(args):
    check_isolated_p_only(args)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Nonempty output directory: {out}. Use separate outputs, never overwrite.')

    original_prepare = base.prepare
    original_forward = WeightedGCNEncoder.forward
    captured = {}

    def prepare_with_train_audit(a):
        ss, embed = original_prepare(a)
        captured.update(train_weight_report(ss[0]['weights'], ss[0]['edges'], ss[0]['num_u'], ss[0]['num_v'], quantile=args.clip_quantile))
        return ss, embed

    def forward_transform(self, x_u, x_v, edge_index, edge_weight, num_nodes_u):
        # Both online and EMA target encoders, and historical inference.
        # A/S receive the untransformed data; only GCN convolution is affected.
        adjusted = transform_weight(edge_weight, args.propagation_mode, captured.get('clip_threshold_train_only'))
        return original_forward(self, x_u, x_v, edge_index, adjusted, num_nodes_u)

    # Scoped, reversible interception. No changes to original project source.
    base.prepare = prepare_with_train_audit
    WeightedGCNEncoder.forward = forward_transform
    try:
        base.run(args)
    finally:
        WeightedGCNEncoder.forward = original_forward
        base.prepare = original_prepare

    provenance = {
        'mode': args.propagation_mode,
        'controlled_ablation': 'WBT_100 (P=1,A=0,S=0; NWB decoder)',
        'transform_applied_only_to': 'WeightedGCNEncoder.forward edge_weight',
        'log1p_definition': 'log(1+w)/log(2) keeps w=1 equal to 1',
        'clipping_threshold_source': 'train edges only',
        'source_sha256': {name: sha(ROOT/name) for name in RELEVANT},
        'train_weight_report': captured,
        'status': 'sensitivity-only; not a replacement for submitted method',
    }
    (out/'propagation_sensitivity.json').write_text(json.dumps(provenance, indent=2))
    # Set metadata in result.json, but preserve all metrics and training artifacts.
    path = out/'result.json'
    r = json.loads(path.read_text())
    r['propagation_sensitivity'] = provenance
    path.write_text(json.dumps(r,indent=2))
    print('P-only mode', args.propagation_mode, 'train p99', captured['p99'], 'AUC', r['metrics']['ROC-AUC'],flush=True)


def parser():
    p=base.parser()
    # Explicit flags in the saved config and command; never infer WP defaults.
    p.set_defaults(variant='WP_NWB', propagation=True, augmentation=False,
                   loss_weights=False, negative_strategy='matched_synthetic', dataset='synthetic')
    p.add_argument('--propagation-mode', choices=MODES, default='raw')
    p.add_argument('--clip-quantile', type=float, default=.99)
    return p


if __name__ == '__main__':
    execute(parser().parse_args())
