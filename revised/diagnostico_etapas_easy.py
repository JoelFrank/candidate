"""Diagnostic-only: attribute -> bridge -> GCN -> cosine vs original MLP.

Trains ONE small BiSBM-EASY experiment per selected variant via the UNMODIFIED
revised/run_easy.py; reloads the actual checkpoints and reconstructs exactly the
same validation and test pairs. Does not write or alter model/training code.

Scores are *diagnostic*. DO NOT select model/hyperparameters on test AUC.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from revised.easy_negatives import OLD_CUTOFFS, sample_matched
from revised.run_easy import EasyFeatureEmbedder, prepare_easy, remap_context
from src.models.wbtbgrl import WBTBGRL
from src.evaluation.decoder import LinkPredictor

BITS = {'000': (0, 0, 0), '010': (0, 1, 0)}
STAGES = (('Atributos originales (coseno)', 'raw_cosine'),
          ('Feature embedder (coseno)', 'feature_cosine'),
          ('Proyecciones iniciales GCN (coseno)', 'gcn_input_cosine'),
          ('Embeddings GNN (coseno)', 'gnn_cosine'),
          ('Decoder MLP original (probabilidad)', 'mlp_probability'))
SOURCE_NAMES = ('src/models/wbtbgrl.py', 'revised/run_easy.py',
                'revised/easy_negatives.py', 'src/evaluation/decoder.py',
                'data/synthetic/generator_easy_project5.py')


def source_hashes():
    missing = [name for name in SOURCE_NAMES if not (ROOT/name).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing project5 files: {missing}')
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_NAMES}


def candidate_pairs(snapshots, index, candidate_seed):
    ss = snapshots
    stage = ('train', 'validation', 'test')[index]
    cutoff_u, cutoff_v = OLD_CUTOFFS[stage]
    negative = sample_matched(ss[index], ss[:index+1], candidate_seed + index,
                              cutoff_u, cutoff_v, remap_context)
    positives = ss[index]['edges']
    pairs = torch.cat([positives, negative], dim=1).long()
    y = np.concatenate([np.ones(positives.shape[1]), np.zeros(negative.shape[1])]).astype('float32')
    return pairs, y


def score_cosine(z_u, z_v, pairs, n_u):
    left = z_u[pairs[0]]
    right = z_v[pairs[1]-n_u]
    if left.shape[1] != right.shape[1]:
        raise ValueError('Cross-partition cosine needs matching feature dimensions')
    return F.cosine_similarity(left, right, dim=1).detach().cpu().numpy()


def prepare_model(args, saved, embedding, device):
    config = saved['config']
    bits = (int(bool(config['propagation'])), int(bool(config['augmentation'])), int(bool(config['loss_weights'])))
    model = WBTBGRL(input_dim_u=config['hidden'], input_dim_v=config['hidden'],
                    hidden_dim=config['hidden'], output_dim=config['dim'], dropout=config['dropout'],
                    lambda_param=config['lam'], use_weights_pretraining=bits != (0, 0, 0),
                    use_weights_link_prediction=False,
                    weight_propagation=bool(bits[0]), weight_augmentation=bool(bits[1]),
                    weight_loss=bool(bits[2]), edge_drop_rate=config['edge_drop'],
                    feat_drop_rate=config['feature_drop']).to(device)
    bridge = EasyFeatureEmbedder(embedding['num_in_u'], embedding['num_in_v'], config['hidden']).to(device)
    decoder = LinkPredictor(config['dim']*2, config['hidden'], dropout=config['dropout']).to(device)
    model.load_state_dict(embedding['checkpoint']['model'], strict=True)
    bridge.load_state_dict(embedding['checkpoint']['embedder'], strict=True)
    decoder.load_state_dict(embedding['checkpoint']['decoder'], strict=True)
    model.eval(); bridge.eval(); decoder.eval()
    return model, bridge, decoder


def evaluate_stages(ss, model, bridge, decoder, pairs, index, device):
    s = ss[index]
    local = pairs.to(device)
    nu = s['num_u']
    xu = s['inputs'][0].to(device)
    xv = s['inputs'][1].to(device)
    with torch.no_grad():
        raw = score_cosine(xu, xv, local, nu)
        u, v = bridge(xu, xv)
        feat = score_cosine(u, v, local, nu)
        # This is the initial learnable projection INSIDE WeightedGCNEncoder,
        # evaluated before any message passing (no code modification).
        proj_u = model.online_encoder.proj_u(u)
        proj_v = model.online_encoder.proj_v(v)
        projection = score_cosine(proj_u, proj_v, local, nu)
        ei, w = remap_context(ss[:max(1, index)], s)
        zu, zv = model.get_embeddings(u, v, ei.to(device), nu, w.to(device))
        gnn = score_cosine(zu, zv, local, nu)
        z = torch.cat([zu, zv], dim=0)
        logits = decoder(torch.cat([z[local[0]], z[local[1]]], dim=1)).reshape(-1)
        mlp = torch.sigmoid(logits).cpu().numpy()
    return dict(raw_cosine=raw, feature_cosine=feat, gcn_input_cosine=projection,
                gnn_cosine=gnn, mlp_probability=mlp)


def metrics(y, values):
    return {'roc_auc': float(roc_auc_score(y, values)),
            'ap': float(average_precision_score(y, values)),
            'positive_mean': float(np.mean(values[y == 1])),
            'negative_mean': float(np.mean(values[y == 0]))}


def draw_histograms(rows, outfile, heading):
    import math
    columns = 3
    nr = math.ceil(len(STAGES)/columns)
    fig, axes = plt.subplots(nr, columns, figsize=(15, 8), squeeze=False)
    for i, (name, key) in enumerate(STAGES):
        ax = axes[i//columns, i%columns]
        values = rows[key]
        y = rows['label']
        vmin = float(min(values.min(), 0))
        vmax = float(max(values.max(), 1))
        if key == 'mlp_probability':
            bins = np.linspace(0, 1, 41)
        else:
            bins = np.linspace(max(-1., vmin), min(1., vmax), 41)
        ax.hist(values[y == 1], bins=bins, density=True, alpha=0.58,
                color='#1769e0', label='Positivos (azul)')
        ax.hist(values[y == 0], bins=bins, density=True, alpha=0.58,
                color='#db3030', label='Negativos (rojo)')
        ax.set_title(f'{name}\nROC-AUC {roc_auc_score(y, values):.4f}', fontsize=11)
        ax.set_xlabel('Puntuación')
        ax.set_ylabel('Densidad')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    for j in range(len(STAGES), nr*columns):
        axes[j//columns, j%columns].axis('off')
    fig.suptitle(heading, fontsize=13)
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=180)
    plt.close(fig)


def create_training_command(args, bits, target):
    p, a, s = BITS[bits]
    variant = 'NWP_NWB' if bits == '000' else 'WP_NWB'
    return [sys.executable, str(ROOT/'revised/run_easy.py'), '--graph-seed', str(args.graph),
            '--seed', str(args.seed), '--candidate-seed', str(args.candidate_seed),
            '--variant', variant, '--propagation', str(p), '--augmentation', str(a),
            '--loss-weights', str(s), '--encoder-epochs', str(args.encoder_epochs),
            '--decoder-epochs', str(args.decoder_epochs), '--device', args.device,
            '--threads', str(args.threads), '--output', str(target)]


def inspect_one(args, root, bits, hashes):
    target = root/f'g{args.graph}_s{args.seed}_WBT_{bits}'
    artifact = target/'result.json'
    checkpoint = target/'checkpoint.pt'
    if not artifact.exists() or not checkpoint.exists():
        if target.exists():
            raise RuntimeError(f'Incomplete output folder: {target}; use a NEW --output directory')
        command = create_training_command(args, bits, target)
        root.mkdir(parents=True, exist_ok=True)
        logs = root/'_launcher_logs'; logs.mkdir(exist_ok=True)
        logpath = logs/f'g{args.graph}_s{args.seed}_WBT_{bits}.log'
        print(f'ENTRENANDO WBT_{bits}: grafo {args.graph}, semilla {args.seed}', flush=True)
        env = os.environ.copy(); env.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        with logpath.open('w') as fh:
            p = subprocess.run(command, cwd=str(ROOT), env=env, stdout=fh,
                               stderr=subprocess.STDOUT, check=False)
        if p.returncode:
            raise RuntimeError(f'Failed WBT_{bits} ({p.returncode}); see {logpath}')
        (target/'run.log').write_bytes(logpath.read_bytes()); logpath.unlink()
    saved = json.loads(artifact.read_text())
    cfg = saved['config']
    if cfg['graph_seed'] != args.graph or cfg['seed'] != args.seed:
        raise RuntimeError('Saved graph/seed differs from requested experiment')
    if cfg.get('candidate_seed') != args.candidate_seed:
        raise RuntimeError('Saved candidate seed differs')
    if cfg['encoder_epochs'] != args.encoder_epochs or cfg['decoder_epochs'] != args.decoder_epochs:
        raise RuntimeError('Saved epochs differ; use a NEW output directory')
    for flag, expected in zip(('propagation','augmentation','loss_weights'), BITS[bits]):
        if bool(cfg[flag]) != bool(expected):
            raise RuntimeError(f'Saved {flag} does not match WBT_{bits}')
    if saved.get('environment',{}).get('device') != args.device:
        raise RuntimeError('Saved device differs; use a new output directory')

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    from types import SimpleNamespace
    param = SimpleNamespace(graph_seed=args.graph, seed=args.seed,
                            easy_groups=cfg['easy_groups'], easy_noise=cfg['easy_noise'],
                            hidden=cfg['hidden'])
    ss, emb_template, _ = prepare_easy(param)
    ckp = torch.load(checkpoint, map_location='cpu', weights_only=True)
    conf = {'num_in_u': ss[0]['inputs'][0].shape[1],
            'num_in_v': ss[0]['inputs'][1].shape[1], 'checkpoint': ckp}
    model, bridge, decoder = prepare_model(args, saved, conf, device)
    metrics_by_split = {}
    stage_scores = None
    for idx, name in ((1, 'validation'), (2, 'test')):
        pairs, y = candidate_pairs(ss, idx, args.candidate_seed)
        vals = evaluate_stages(ss, model, bridge, decoder, pairs, idx, device)
        stage_metrics = {key: metrics(y, vals[key]) for _,key in STAGES}
        metrics_by_split[name] = stage_metrics
        if idx == 2:
            s = ss[2]
            local = pairs.numpy().T
            u_key = s['keys_u'][local[:,0]]
            v_key = s['keys_v'][local[:,1]-s['num_u']]
            stored = pd.read_csv(target/'test_scores.csv')
            if len(stored) != len(y):
                raise RuntimeError('Saved candidate count differs from reconstructed test candidates')
            if not (np.array_equal(stored['u'].to_numpy(),u_key)
                    and np.array_equal(stored['v'].to_numpy(),v_key)
                    and np.array_equal(stored['label'].to_numpy(),y)):
                raise RuntimeError('Reconstructed test candidates differ from original saved candidates')
            err = float(np.max(np.abs(stored['score'].to_numpy() - vals['mlp_probability'])))
            if err > 1e-5:
                raise RuntimeError(f'Original decoder reproduction mismatch: max absolute error {err}')
            if abs(stage_metrics['mlp_probability']['roc_auc'] - float(saved['metrics']['ROC-AUC'])) > 1e-8:
                raise RuntimeError('Original ROC-AUC could not be reproduced')
            stage_scores = pd.DataFrame({'u':u_key, 'v':v_key, 'label':y, **vals})
            metrics_by_split['test_max_decoder_score_error'] = err
    stage_scores.to_csv(root/f'WBT_{bits}_etapas_scores.csv', index=False)
    draw_histograms({c:stage_scores[c].to_numpy() for c in stage_scores.columns},
                    root/f'WBT_{bits}_etapas_histogramas.png',
                    f'BiSBM-EASY g{args.graph}, seed {args.seed}: WBT_{bits} | mismo test')
    return metrics_by_split


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph', type=int, default=111)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--variants', choices=('010','000,010','000'), default='000,010')
    p.add_argument('--encoder-epochs', type=int, default=200)
    p.add_argument('--decoder-epochs', type=int, default=100)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--candidate-seed', type=int, default=2026)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', default='results/diagnostico_etapas_easy')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.encoder_epochs < 1 or args.decoder_epochs < 1:
        raise ValueError('Epoch counts must be positive')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available')
    root = Path(args.output)
    hashes = source_hashes()
    fingerprint = dict(graph=args.graph, seed=args.seed, variants=args.variants,
                       encoder_epochs=args.encoder_epochs, decoder_epochs=args.decoder_epochs,
                       candidate_seed=args.candidate_seed, device=args.device, source_sha256=hashes)
    fp = root/'diagnostico_plan.json'
    if fp.exists() and json.loads(fp.read_text()) != fingerprint:
        raise RuntimeError('Plan / source changed: use a NEW output directory')
    root.mkdir(parents=True, exist_ok=True)
    if not fp.exists():
        fp.write_text(json.dumps(fingerprint, indent=2))
    full = {}
    for bits in args.variants.split(','):
        full[f'WBT_{bits}'] = inspect_one(args, root, bits, hashes)
    (root/'diagnostico_resumen.json').write_text(json.dumps(full, indent=2))
    rows = []
    for variant, stages in full.items():
        for split in ('validation','test'):
            for _, key in STAGES:
                rows.append(dict(variant=variant, split=split, etapa=key, **stages[split][key]))
    pd.DataFrame(rows).to_csv(root/'diagnostico_comparacion.csv', index=False)
    print('\nROC-AUC por etapa (mismo test):', flush=True)
    print(pd.DataFrame(rows).query('split == "test"').pivot(index='etapa', columns='variant', values='roc_auc').to_string(), flush=True)
    print(f'Generados CSV, JSON y figuras azul/rojo en {root}', flush=True)


if __name__ == '__main__':
    main()
