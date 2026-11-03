"""Standalone runner for BiSBM-EASY on project5.

Produces the same core artifacts as revised/run.py but uses the dedicated
BiSBM-EASY generator and a small feature embedder.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys, argparse, random, json, time, platform, hashlib, copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.wbtbgrl import WBTBGRL
from src.models.bgrl import BGRL
from src.models.tbgrl import TBGRL
from src.models.gbt import GBT
from src.models.cca_ssg import CCASSG
from src.evaluation.decoder import LinkPredictor
from revised.easy_generator_loader import generate_easy
from revised.easy_negatives import OLD_CUTOFFS, strata_counts, sample_matched


class EasyFeatureEmbedder(nn.Module):
    def __init__(self, in_u: int, in_v: int, hidden: int):
        super().__init__()
        self.u = nn.Sequential(nn.Linear(in_u, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.v = nn.Sequential(nn.Linear(in_v, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def forward(self, xu: torch.Tensor, xv: torch.Tensor):
        return self.u(xu.float()), self.v(xv.float())


def snapshot(edges, weights, inputs, keys_u, keys_v):
    return dict(edges=edges, weights=weights, inputs=inputs, keys_u=np.asarray(keys_u), keys_v=np.asarray(keys_v), num_u=len(keys_u), num_v=len(keys_v))


def remap_context(history, target):
    um = {x: i for i, x in enumerate(target['keys_u'])}
    vm = {x: i + target['num_u'] for i, x in enumerate(target['keys_v'])}
    counts = {}
    for h in history:
        for (u, v), w in zip(h['edges'].T.tolist(), h['weights'].tolist()):
            a = h['keys_u'][u]
            b = h['keys_v'][v - h['num_u']]
            if a in um and b in vm:
                pair = (um[a], vm[b])
                counts[pair] = counts.get(pair, 0.0) + w
    pairs = sorted(counts)
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32)
    return torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).T, torch.tensor([counts[p] for p in pairs], dtype=torch.float32)


def negatives(s, history, n, seed):
    ei, _ = remap_context(history, s)
    known = set(map(tuple, ei.T.tolist()))
    capacity = s['num_u'] * s['num_v'] - len(known)
    if capacity < n:
        raise ValueError(f'Only {capacity} eligible negatives; requested {n}')
    rng = np.random.default_rng(seed)
    result = []
    seen = set(known)
    for _ in range(max(1000, 100 * n)):
        pair = (int(rng.integers(s['num_u'])), int(rng.integers(s['num_v'])) + s['num_u'])
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
    return torch.tensor(result, dtype=torch.long).T


def metrics(y, p):
    b = p >= 0.5
    r = {
        'ROC-AUC': roc_auc_score(y, p), 'AP': average_precision_score(y, p),
        'F1': f1_score(y, b, zero_division=0), 'precision': precision_score(y, b, zero_division=0),
        'recall': recall_score(y, b, zero_division=0),
    }
    for k in [10, 20, 50, 100]:
        neg = p[y == 0]
        r[f'Hits@{k}'] = float(np.mean(p[y == 1] > np.partition(neg, -k)[-k])) if len(neg) >= k else None
    return r


def prepare_easy(args):
    path = generate_easy(args.graph_seed, ROOT / 'data/synthetic/bisbm_easy_v1', groups=args.easy_groups, feature_noise=args.easy_noise)
    data = np.load(path, allow_pickle=True)
    xu = data['node_u_attributes']
    xv = data['node_v_attributes']
    snapshots = []
    for stage in ['train', 'validation', 'test']:
        us = data[f'{stage}_candidate_u']
        vs = data[f'{stage}_candidate_v']
        u_map = {int(x): i for i, x in enumerate(us.tolist())}
        v_map = {int(x): i + len(us) for i, x in enumerate(vs.tolist())}
        pairs = data[f'{stage}_pairs_global']
        local = np.array([[u_map[int(u)], v_map[int(v)]] for u, v in pairs], dtype=np.int64).reshape(-1, 2).T
        w = data[f'{stage}_weights'].astype(np.float32)
        inp = [torch.tensor(xu[us], dtype=torch.float32), torch.tensor(xv[vs], dtype=torch.float32)]
        snapshots.append(snapshot(torch.tensor(local), torch.tensor(w), inp, us, vs))
    emb = EasyFeatureEmbedder(xu.shape[1], xv.shape[1], args.hidden)
    return snapshots, emb, str(path)


def run(args):
    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    ss, embed, dataset_path = prepare_easy(args)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    audit = {'dataset_path': dataset_path}
    for i, s in enumerate(ss):
        hist = ss[:max(1, i)]
        ei, ew = remap_context(hist, s)
        audit[['train', 'val', 'test'][i]] = {
            'nodes_u': s['num_u'], 'nodes_v': s['num_v'], 'positive_edges': int(s['edges'].shape[1]),
            'context_edges': int(ei.shape[1]), 'weight_min': float(s['weights'].min()), 'weight_max': float(s['weights'].max()),
            'context_sha256': hashlib.sha256(ei.numpy().tobytes() + ew.numpy().tobytes()).hexdigest(),
        }
    (out / 'data_audit.json').write_text(json.dumps(audit, indent=2))

    embed = embed.to(device)
    wp = args.variant.startswith('WP_'); wb = args.variant.endswith('_WB')
    config = dict(input_dim_u=args.hidden, input_dim_v=args.hidden, hidden_dim=args.hidden, output_dim=args.dim, dropout=args.dropout)
    if args.model == 'WBTBGRL':
        model = WBTBGRL(**config, lambda_param=args.lam, use_weights_pretraining=wp, use_weights_link_prediction=wb,
                        weight_propagation=args.propagation, weight_augmentation=args.augmentation, weight_loss=args.loss_weights,
                        edge_drop_rate=args.edge_drop, feat_drop_rate=args.feature_drop).to(device)
    else:
        model = {'BGRL': BGRL, 'TBGRL': TBGRL, 'GBT': GBT, 'CCASSG': CCASSG}[args.model](**config).to(device)
        wp = False; wb = False
    for s in ss:
        s['inputs'] = [x.to(device) for x in s['inputs']]
    opt = torch.optim.AdamW(list(model.parameters()) + list(embed.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    losses = []
    t0 = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    for _ in range(args.encoder_epochs):
        model.train(); embed.train(); opt.zero_grad()
        u, v = embed(*ss[0]['inputs']); ei = ss[0]['edges'].to(device); w = ss[0]['weights'].to(device)
        if args.model == 'WBTBGRL':
            loss = model.compute_loss(model(u, v, ei, ss[0]['num_u'], w), ss[0]['num_u'])
        else:
            loss = model.compute_loss(model(u, v, ei, ss[0]['num_u']))
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite SSL loss')
        loss.backward(); opt.step()
        if hasattr(model, '_update_target_encoder'):
            model._update_target_encoder()
        losses.append(float(loss.detach()))
    if device.type == 'cuda':
        torch.cuda.synchronize()
    encoder_seconds = time.perf_counter() - t0

    model.eval(); embed.eval()
    for p in list(model.parameters()) + list(embed.parameters()):
        p.requires_grad_(False)
    def representations(i):
        s = ss[i]; ei, w = remap_context(ss[:max(1, i)], s)
        with torch.no_grad():
            u, v = embed(*s['inputs'])
            if args.model == 'WBTBGRL':
                return torch.cat(model.get_embeddings(u, v, ei.to(device), s['num_u'], w.to(device)))
            return torch.cat(model.get_embeddings(u, v, ei.to(device), s['num_u']))
    reps = [representations(i) for i in range(2)]
    negative_strata = {}
    def candidates(i):
        s = ss[i]
        stage = ['train','validation','test'][i]
        cutoff_u, cutoff_v = OLD_CUTOFFS[stage]
        neg = sample_matched(s, ss[:i+1], args.candidate_seed+i,
                             cutoff_u, cutoff_v, remap_context)
        negative_strata[stage] = {
            'positive': strata_counts(s, s['edges'], cutoff_u, cutoff_v),
            'negative': strata_counts(s, neg, cutoff_u, cutoff_v),
        }
        e = torch.cat([s['edges'], neg], dim=1).to(device)
        y = torch.cat([torch.ones(s['edges'].shape[1]), torch.zeros(neg.shape[1])]).to(device)
        return e, y
    es = [candidates(i) for i in range(2)]
    decoder = LinkPredictor(args.dim * 2, args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    def score(z, e): return decoder(torch.cat([z[e[0]], z[e[1]]], dim=1)).reshape(-1)
    weights = torch.cat([torch.log1p(ss[0]['weights']) if wb else torch.ones_like(ss[0]['weights']), torch.ones(ss[0]['edges'].shape[1])]).to(device)
    best = -float('inf'); state = None; wait = 0; history = []; t0 = time.perf_counter()
    for epoch in range(args.decoder_epochs):
        decoder.train(); opt.zero_grad(); pred = score(reps[0], es[0][0]); loss = (F.binary_cross_entropy_with_logits(pred, es[0][1], reduction='none') * weights).mean(); loss.backward(); opt.step()
        decoder.eval()
        with torch.no_grad():
            vp = torch.sigmoid(score(reps[1], es[1][0])).cpu().numpy()
        auc = roc_auc_score(es[1][1].cpu().numpy(), vp)
        history.append({'epoch': epoch + 1, 'train_loss': float(loss.detach()), 'val_auc': auc})
        if auc > best:
            best = auc; state = copy.deepcopy(decoder.state_dict()); wait = 0
        else:
            wait += 1
        if wait >= args.patience:
            break
    decoder.load_state_dict(state); decoder.eval()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    decoder_seconds = time.perf_counter() - t0; t0 = time.perf_counter()
    z = representations(2); e, y = candidates(2)
    with torch.no_grad():
        p = torch.sigmoid(score(z, e)).cpu().numpy()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - t0; y = y.cpu().numpy(); result = metrics(y, p)
    pairs = e.cpu().numpy().T
    scores = pd.DataFrame({'u': ss[2]['keys_u'][pairs[:, 0]], 'v': ss[2]['keys_v'][pairs[:, 1] - ss[2]['num_u']], 'label': y, 'score': p})
    scores.to_csv(out / 'test_scores.csv', index=False)
    import torch_geometric
    payload = {
        'config': vars(args),
        'protocol': 'BiSBM-EASY; decoder=train; selection=val; test=once',
        'metrics': result, 'val_auc': best, 'history': history, 'ssl_losses': losses,
        'timing': {'encoder_s': encoder_seconds, 'decoder_s': decoder_seconds, 'inference_s': inference_seconds, 'epoch_s': encoder_seconds / args.encoder_epochs},
        'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'pyg': torch_geometric.__version__, 'device': str(device), 'hardware': torch.cuda.get_device_name() if device.type == 'cuda' else platform.processor()},
        'cuda_peak_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
        'candidate_sha256': hashlib.sha256(pairs.tobytes() + y.tobytes()).hexdigest(),
        'dataset_path': dataset_path,
        'negative_sampling': {'policy': 'matched_easy_stage', 'strata': negative_strata, 'candidate_seed': args.candidate_seed},
    }
    (out / 'result.json').write_text(json.dumps(payload, indent=2))
    torch.save({'model': model.state_dict(), 'embedder': embed.state_dict(), 'decoder': decoder.state_dict()}, out / 'checkpoint.pt')
    print(json.dumps(result))
    return result


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--graph-seed', type=int, default=111); p.add_argument('--seed', type=int, default=42); p.add_argument('--candidate-seed', type=int, default=2026)
    p.add_argument('--variant', choices=['WP_WB','WP_NWB','NWP_WB','NWP_NWB'], default='WP_NWB'); p.add_argument('--model', choices=['WBTBGRL','BGRL','TBGRL','GBT','CCASSG'], default='WBTBGRL')
    p.add_argument('--easy-groups', type=int, default=16); p.add_argument('--easy-noise', type=float, default=0.02)
    for k, d in [('hidden',128), ('dim',64), ('encoder-epochs',200), ('decoder-epochs',100), ('patience',20), ('threads',2)]: p.add_argument('--'+k, type=int, default=d)
    for k, d in [('lr',.001), ('weight-decay',1e-5), ('dropout',.1), ('lam',.5), ('edge-drop',.1), ('feature-drop',.1)]: p.add_argument('--'+k, type=float, default=d)
    for k in ['propagation', 'augmentation', 'loss-weights']: p.add_argument('--'+k, type=lambda x: bool(int(x)), default=None)
    p.add_argument('--device', default='cpu'); p.add_argument('--output', default='results/revised_easy')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
