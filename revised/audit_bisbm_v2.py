"""BiSBM v2 candidate + train-relative identity audit without torch_geometric.

Run: python -m revised.audit_bisbm_v2 --output results_v2/audit_bisbm_v2.json
"""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from revised.data import module, snapshot, synthetic_train_observed_id_tokens
from revised.inductive_protocol import annotate_snapshots, matched_static_bisbm_negatives, audit_pairs, pair_group


def evaluate_seed(seed: int, candidate_seed: int = 2026):
    gen = module('data/synthetic/generator.py')
    with redirect_stdout(io.StringIO()):
        graph, _, _, _, _ = gen.generate_bisbn(int(seed))
    snaps = []
    for horizon in ('train', 'val', 'test'):
        uu = graph[f'{horizon}_u_ids']
        vv = graph[f'{horizon}_v_ids']
        raw = graph['oo_train' if horizon == 'train' else f'{horizon}_edges']
        edge_index = torch.as_tensor(gen.remap_edges(raw, uu, vv), dtype=torch.long)
        snaps.append(snapshot(edge_index, torch.ones(edge_index.shape[1]), [], uu, vv))
    annotate_snapshots(snaps)
    seen_u, seen_v = snaps[0]['train_seen_u'], snaps[0]['train_seen_v']
    assert len(seen_u) == len(graph['train_u_ids']) and len(seen_v) == len(graph['train_v_ids'])
    summary = {
        'graph_seed': int(seed),
        'generative_old_u': int(gen.N_U_OLD), 'generative_old_v': int(gen.N_V_OLD),
        'actually_observed_train_u': len(seen_u), 'actually_observed_train_v': len(seen_v),
        'old_but_never_train_u': int(gen.N_U_OLD - len(seen_u)),
        'old_but_never_train_v': int(gen.N_V_OLD - len(seen_v)),
        'horizons': {},
    }
    for i, (horizon, snap) in enumerate(zip(('train', 'val', 'test'), snaps)):
        negatives = matched_static_bisbm_negatives(snap, snaps, candidate_seed + i)
        report = audit_pairs(snap, negatives)
        tokens_u = synthetic_train_observed_id_tokens(snap['keys_u'], seen_u, gen.N_U_OLD).numpy()
        tokens_v = synthetic_train_observed_id_tokens(snap['keys_v'], seen_v, gen.N_V_OLD).numpy()
        for keys, tokens, known in ((snap['keys_u'], tokens_u, seen_u),
                                    (snap['keys_v'], tokens_v, seen_v)):
            for key, token in zip(keys, tokens):
                if (int(key) in known) != (int(token) != 0):
                    raise AssertionError(f'ID token does not agree with TRAIN membership: {key}')
        for edge in negatives.T.tolist():
            if pair_group(snap, *edge)[0] not in ('OO','OU','UO','UU'):
                raise AssertionError('Unexpected inductive category')
        report.update({
            'candidate_sha256': hashlib.sha256(snap['edges'].numpy().tobytes()
                + negatives.numpy().tobytes()).hexdigest(),
            'unknown_identity_tokens_u': int(np.count_nonzero(tokens_u == 0)),
            'unknown_identity_tokens_v': int(np.count_nonzero(tokens_v == 0)),
            'old_unseen_in_horizon_u': int(sum(int(k) < gen.N_U_OLD and int(k) not in seen_u for k in snap['keys_u'])),
            'old_unseen_in_horizon_v': int(sum(int(k) < gen.N_V_OLD and int(k) not in seen_v for k in snap['keys_v'])),
        })
        summary['horizons'][horizon] = report
    return summary


def audit(seeds=(42,43,44,45,46), candidate_seed=2026):
    return {'dataset':'synthetic_bisbm', 'protocol':'train-relative-OU-v2',
            'negative_policy':'matched_history_static_graph_exclusion_v2', 'candidate_seed':int(candidate_seed),
            'seeds': {str(k):evaluate_seed(k,candidate_seed) for k in seeds}}


def main():
    p=argparse.ArgumentParser(description='BiSBM train-relative ID and candidate audit')
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46])
    p.add_argument('--candidate-seed',type=int,default=2026)
    p.add_argument('--output',default='results_v2/audit_bisbm_v2.json')
    a=p.parse_args()
    obj=audit(a.seeds,a.candidate_seed)
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(obj,indent=2),encoding='utf-8')
    print('Audit saved to',target)
    for seed,record in obj['seeds'].items():
        t=record['horizons']['test']
        print(f'seed={seed}: {t["positive_by_stratum"]}  matched={t["positive_by_history"]==t["negative_by_history"]};'
              f' old-train-unseen={record["old_but_never_train_u"]}/{record["old_but_never_train_v"]}')


if __name__=='__main__':main()
