"""Scientific diagnostic: frozen EASY WBT representations vs original MLP decoder.

DOES NOT modify WBT-BGRL, its SSL loss, augmentations, pretrained encoders,
synthetic data, or the submitted-manuscript model. Reuses the TWO existing
checkpoints from diagnostico_etapas_easy. New MLP probes train exclusively on
train links, select epochs on validation AUC, and evaluate test once after
selection. Probes are exploratory; they are not manuscript benchmark results.

Run from project5 root: python revised/auditoria_decoders_easy.py --help
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import random

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load the project's model classes only in diagnostic execution.  The numeric
# unit tests deliberately do not require PyTorch Geometric on the build host.
BITS = {'000':(0,0,0), '010':(0,1,0)}


def decoder_type():
    """Load the EXACT project LinkPredictor implementation from its source."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('stage_audit_original_decoder', ROOT/'src/evaluation/decoder.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LinkPredictor

FEATURE_KEYS = ('raw', 'feature', 'projection', 'gnn')
SPLITS = ('train', 'validation', 'test')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataframe_hash(frame: pd.DataFrame) -> str:
    cols = frame[['u', 'v', 'label']].copy()
    return hashlib.sha256(cols.to_csv(index=False).encode('utf-8')).hexdigest()


def auc_ap(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def balanced_score_info(y: np.ndarray, p: np.ndarray) -> dict:
    auc, ap = auc_ap(y, p)
    return dict(roc_auc=auc, ap=ap,
                positive_mean=float(np.mean(p[y == 1])),
                negative_mean=float(np.mean(p[y == 0])))


def pair_features(u: torch.Tensor, v: torch.Tensor, pairs: torch.Tensor, n_u: int) -> torch.Tensor:
    """Exact concatenation used in revised/run_easy.py + LinkPredictor."""
    if pairs.ndim != 2 or pairs.shape[0] != 2:
        raise ValueError('Expected edge pairs with shape [2, E]')
    if n_u != len(u):
        raise ValueError('U-offset does not match current snapshot')
    if u.device != v.device:
        raise ValueError('U and V embeddings on different devices')
    pairs = pairs.to(u.device)
    uidx = pairs[0]
    vidx = pairs[1] - n_u
    if uidx.numel() and (uidx.min() < 0 or uidx.max() >= len(u)):
        raise ValueError('U index out of range')
    if vidx.numel() and (vidx.min() < 0 or vidx.max() >= len(v)):
        raise ValueError('V index out of range')
    return torch.cat([u[uidx], v[vidx]], dim=1).detach()



def representation_geometry(values: torch.Tensor) -> dict:
    """Node-level collapse diagnostics; does not use link labels."""
    x = values.detach().float().cpu()
    x = x - x.mean(dim=0, keepdim=True)
    feature_sd = x.std(dim=0, unbiased=False)
    # Effective rank of node-by-feature matrix; zero indicates total collapse.
    singular = torch.linalg.svdvals(x)
    spectrum = singular.square()
    if float(spectrum.sum()) > 0:
        probs = (spectrum / spectrum.sum()).clamp_min(1e-30)
        effective_rank = float(torch.exp(-(probs * probs.log()).sum()))
    else:
        effective_rank = 0.
    return dict(dimensions=int(values.shape[1]), nodes=int(values.shape[0]),
                across_node_std_mean=float(feature_sd.mean()),
                across_node_std_max=float(feature_sd.max()),
                effective_rank=effective_rank)


def numeric_probe(u: torch.Tensor, v: torch.Tensor, pairs: torch.Tensor, n_u: int) -> np.ndarray:
    """Cosine diagnostic ONLY when U and V have equal feature dimension."""
    if u.size(1) != v.size(1):
        raise ValueError('Cannot compute cross-partition cosine for unequal dims')
    e = pairs.to(u.device)
    return F.cosine_similarity(u[e[0]], v[e[1] - n_u], dim=1).detach().cpu().numpy()


def reconstruct_frozen_features(ss, model, bridge, device: torch.device):
    """Four frozen representations for EACH chronological snapshot.

    Historical context only: train->train, validation->train,
    test->train+validation (never includes target/test edges).
    """
    from revised.run_easy import remap_context
    all_feat = {}
    model.eval(); bridge.eval()
    with torch.no_grad():
        for idx, s in enumerate(ss):
            xu, xv = [x.to(device) for x in s['inputs']]
            bu, bv = bridge(xu, xv)
            pu, pv = model.online_encoder.proj_u(bu), model.online_encoder.proj_v(bv)
            ei, w = remap_context(ss[:max(1, idx)], s)
            zu, zv = model.get_embeddings(bu, bv, ei.to(device), s['num_u'], w.to(device))
            all_feat[SPLITS[idx]] = {
                'raw': (xu.detach(), xv.detach()),
                'feature': (bu.detach(), bv.detach()),
                'projection': (pu.detach(), pv.detach()),
                'gnn': (zu.detach(), zv.detach()),
            }
    return all_feat


def candidate_frame(s, pairs, y):
    loc = pairs.detach().cpu().numpy().T
    return pd.DataFrame(dict(u=s['keys_u'][loc[:, 0]],
                             v=s['keys_v'][loc[:, 1] - s['num_u']],
                             label=y.astype(int)))


def verify_same_test(saved_csv: Path, reconstructed: pd.DataFrame) -> str:
    old = pd.read_csv(saved_csv)
    for key in ('u', 'v', 'label'):
        if key not in old.columns:
            raise ValueError(f'Missing {key} in saved test scores')
        if not np.array_equal(old[key].to_numpy(), reconstructed[key].to_numpy()):
            raise RuntimeError(f'Mismatch in candidate {key}: current vs saved original test')
    return dataframe_hash(reconstructed)


def get_training_dataset(ss, seed):
    all_pairs = {}; all_y = {}; all_frames = {}
    for i, name in enumerate(SPLITS):
        from revised.diagnostico_etapas_easy import candidate_pairs
        pairs, y = candidate_pairs(ss, i, seed)
        frame = candidate_frame(ss[i], pairs, y)
        assert int(sum(y == 1)) == int(sum(y == 0)), name
        assert len(frame) == len(frame[['u','v']].drop_duplicates()), f'Duplicate {name} candidate pairs'
        all_pairs[name] = pairs
        all_y[name] = y.astype(np.float32)
        all_frames[name] = frame
    # No future labels can enter train/validation decoder labels.
    return all_pairs, all_y, all_frames


def build_pair_features(all_feat, all_pairs, ss):
    packed = {}
    for name, s in zip(SPLITS, ss):
        packed[name] = {
            key: pair_features(*all_feat[name][key], all_pairs[name], s['num_u'])
            for key in FEATURE_KEYS
        }
    return packed


def frozen_original_check(ss, device, model, bridge, decoder, all_feat, pairs, test_csv):
    x = pair_features(*all_feat['test']['gnn'], pairs['test'], ss[2]['num_u'])
    with torch.no_grad():
        decoder.eval()
        p = torch.sigmoid(decoder(x).reshape(-1)).cpu().numpy()
    stored = pd.read_csv(test_csv)
    error = float(np.max(np.abs(p - stored['score'].to_numpy(float))))
    if error > 1e-5:
        raise RuntimeError(f'Original MLP checkpoint NOT reproduced: max error {error}')
    return error, p


def fit_identical_mlp(xtrain, ytrain, xval, yval, hidden, dropout, lr, wd,
                      max_epochs, patience, seed, device, permute_labels=False):
    """Same LinkPredictor/AdamW as project5, only feature input changes.

    Original training decoder is unweighted (NWB). Best epoch is selected
    on validation ROC-AUC, no test references or test predictions in loop.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = decoder_type()(int(xtrain.size(1)), int(hidden), dropout=float(dropout)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(wd))
    train_y = torch.from_numpy(ytrain).to(device)
    if permute_labels:
        perm = torch.randperm(len(train_y), device=device)
        train_y = train_y[perm]
    val_y = yval
    best_auc = float('-inf')
    best_weights = None
    best_epoch = None
    wait = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(xtrain).reshape(-1)
        loss = F.binary_cross_entropy_with_logits(logits, train_y)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite probe loss')
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(model(xval).reshape(-1)).cpu().numpy()
        val_auc = float(roc_auc_score(val_y, vp))
        history.append(dict(epoch=epoch, loss=float(loss.item()), val_auc=val_auc))
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            best_weights = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    if best_weights is None:
        raise RuntimeError('No valid MLP checkpoint')
    model.load_state_dict(best_weights)
    model.eval()
    return model, best_epoch, best_auc, history


def score_mlp(model, x):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x).reshape(-1)).detach().cpu().numpy()


def leakage_diagnostics(ss, all_frames, graph_npz):
    """Report ambiguity: unobserved same-class pairs are not provably absent."""
    data = np.load(graph_npz, allow_pickle=True)
    yu = data['node_u_blocks_diagnostic_only']
    yv = data['node_v_blocks_diagnostic_only']
    info = {}
    seen_pos = set()
    for name in SPLITS:
        frame = all_frames[name]
        pos_pairs = set(map(tuple, frame.loc[frame.label == 1, ['u','v']].to_numpy().tolist()))
        neg_pairs = set(map(tuple, frame.loc[frame.label == 0, ['u','v']].to_numpy().tolist()))
        if pos_pairs & neg_pairs:
            raise RuntimeError(f'Positive and negative overlap in {name}')
        if seen_pos & pos_pairs:
            raise RuntimeError('Same positive appears in multiple chronological stages')
        seen_pos |= pos_pairs
        negatives = frame.loc[frame.label == 0]
        uids = negatives['u'].to_numpy(dtype=int)
        vids = negatives['v'].to_numpy(dtype=int)
        same_group = yu[uids] == yv[vids]
        info[name] = dict(
            positives=int(len(pos_pairs)), negatives=int(len(neg_pairs)),
            same_latent_class_among_negatives=int(same_group.sum()),
            negative_same_class_fraction=float(same_group.mean()),
        )
    future_pos = set(map(tuple, all_frames['test'].loc[all_frames['test'].label == 1, ['u','v']].to_numpy().tolist()))
    past_neg = set(map(tuple, all_frames['train'].loc[all_frames['train'].label == 0, ['u','v']].to_numpy().tolist()))
    past_neg |= set(map(tuple, all_frames['validation'].loc[all_frames['validation'].label == 0, ['u','v']].to_numpy().tolist()))
    info['past_negative_is_future_test_positive'] = int(len(past_neg & future_pos))
    return info


def plot_stage_hist(rows: pd.DataFrame, out: Path, tag: str, variant: str, seed: int):
    """Exact score histogram; BLUE real positives, RED sampled negatives."""
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    bins = np.linspace(0., 1., 41)
    pos = rows.loc[rows.label == 1, 'score'].to_numpy()
    neg = rows.loc[rows.label == 0, 'score'].to_numpy()
    ax.hist(pos, bins=bins, density=True, alpha=.6, color='#1769E0', label='Positivos (azul)')
    ax.hist(neg, bins=bins, density=True, alpha=.6, color='#DB3030', label='Negativos (rojo)')
    ax.set(title=f'{variant} | {tag} | decoder seed {seed}', xlabel='Puntuación del decoder', ylabel='Densidad', xlim=(0,1))
    ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180); plt.close(fig)


def load_variant(args, bits, device, ss, root):
    run_dir = root / f'g{args.graph}_s{args.seed}_WBT_{bits}'
    jpath = run_dir / 'result.json'
    ckpath = run_dir / 'checkpoint.pt'
    if not (jpath.is_file() and ckpath.is_file() and (run_dir/'test_scores.csv').is_file()):
        raise FileNotFoundError(f'Need existing checkpoint/result/test_scores: {run_dir}')
    info = json.loads(jpath.read_text())
    cfg = info['config']
    expected = {'graph_seed':args.graph, 'seed':args.seed, 'candidate_seed':args.candidate_seed,
                'propagation': bool(BITS[bits][0]), 'augmentation':bool(BITS[bits][1]),
                'loss_weights':bool(BITS[bits][2])}
    for k, v in expected.items():
        if cfg.get(k) != v:
            raise RuntimeError(f'{bits}: {k} mismatches: {cfg.get(k)} vs {v}')
    if cfg.get('variant') != ('NWP_NWB' if bits == '000' else 'WP_NWB'):
        raise RuntimeError('Expected NWB decoder, unexpected saved variant')
    checkpoint = torch.load(ckpath, map_location='cpu', weights_only=True)
    setup = {'num_in_u': ss[0]['inputs'][0].shape[1],
             'num_in_v': ss[0]['inputs'][1].shape[1],
             'checkpoint':checkpoint}
    from revised.diagnostico_etapas_easy import prepare_model
    model, bridge, decoder = prepare_model(args, info, setup, device)
    return info, model, bridge, decoder, run_dir, sha256(ckpath)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='results/diagnostico_etapas_easy')
    p.add_argument('--output', default='results/auditoria_decoders_easy')
    p.add_argument('--graph', type=int, default=111)
    p.add_argument('--seed', type=int, default=42, help='Pretrained WBT seed')
    p.add_argument('--variants', choices=['000','010','000,010'], default='000,010')
    p.add_argument('--candidate-seed', type=int, default=2026)
    p.add_argument('--probe-seeds', nargs='+', type=int, default=[42,43,44])
    p.add_argument('--probe-epochs', type=int, default=250)
    p.add_argument('--probe-patience', type=int, default=30)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--label-control', action='store_true', help='Extra label-permutation control (only raw/000)')
    args = p.parse_args(argv)
    if args.probe_epochs < 1 or args.probe_patience < 1 or len(set(args.probe_seeds)) != len(args.probe_seeds):
        p.error('Probe epochs/patience must be positive; probe seeds unique')
    torch.set_num_threads(args.threads)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    device = torch.device(args.device)
    source = Path(args.source)
    out = Path(args.output)
    if source.resolve() == out.resolve():
        raise ValueError('--source and --output must differ (original artifacts are read-only)')
    existing = sorted(source.glob('g*_s*_WBT_*/checkpoint.pt'))
    if not existing:
        raise FileNotFoundError(f'No checkpoints found in {source}. Use the stage diagnostic source containing two checkpoints.')

    from revised.run_easy import prepare_easy
    from revised.diagnostico_etapas_easy import source_hashes
    from types import SimpleNamespace
    bits0 = args.variants.split(',')[0]
    setup_json = json.loads((source / f'g{args.graph}_s{args.seed}_WBT_{bits0}' / 'result.json').read_text())
    cfg = setup_json['config']
    genargs = SimpleNamespace(graph_seed=args.graph, seed=args.seed, hidden=cfg['hidden'], easy_groups=cfg['easy_groups'], easy_noise=cfg['easy_noise'])
    # prepare_easy reads the original synthetic EASY graph without changing its parameters.
    ss, _, npz_path = prepare_easy(genargs)
    all_pairs, all_y, all_frames = get_training_dataset(ss, args.candidate_seed)
    proof = leakage_diagnostics(ss, all_frames, npz_path)
    original_test_hash = {}
    for bits in args.variants.split(','):
        src = source / f'g{args.graph}_s{args.seed}_WBT_{bits}' / 'test_scores.csv'
        original_test_hash[bits] = verify_same_test(src, all_frames['test'])
    if len(set(original_test_hash.values())) != 1:
        raise RuntimeError('Candidate test pool differs between variants')

    hashes = source_hashes()
    hashes['src/evaluation/decoder.py'] = sha256(ROOT/'src/evaluation/decoder.py')
    hashes['revised/auditoria_decoders_easy.py'] = sha256(Path(__file__))
    source_ckps = {}
    for bits in args.variants.split(','):
        folder = source / f'g{args.graph}_s{args.seed}_WBT_{bits}'
        source_ckps[f'WBT_{bits}'] = {'checkpoint_sha256': sha256(folder/'checkpoint.pt'),
                                     'result_sha256':sha256(folder/'result.json'),
                                     'test_scores_sha256':sha256(folder/'test_scores.csv')}
    plan = dict(source=str(source.resolve()), graph=args.graph, seed=args.seed, variants=args.variants,
                candidate_seed=args.candidate_seed, probe_seeds=args.probe_seeds, probe_epochs=args.probe_epochs,
                probe_patience=args.probe_patience, device=args.device,
                source_hashes=hashes, source_artifacts_sha256=source_ckps, dataset_sha256=sha256(Path(npz_path)),
                original_test_hash=original_test_hash, control=args.label_control)
    out.mkdir(parents=True, exist_ok=True)
    planfile = out/'audit_plan.json'
    if planfile.exists() and json.loads(planfile.read_text()) != plan:
        raise RuntimeError('Existing audit output has different plan. Choose NEW --output.')
    if not planfile.exists():
        planfile.write_text(json.dumps(plan, indent=2))
    (out/'candidate_checks.json').write_text(json.dumps(proof, indent=2))
    csv_rows = []
    result_rows = []
    geometry_rows = []
    for bits in args.variants.split(','):
        variant = f'WBT_{bits}'
        info, model, bridge, decoder, run_dir, ckhash = load_variant(args, bits, device, ss, source)
        all_feat = reconstruct_frozen_features(ss, model, bridge, device)
        packed = build_pair_features(all_feat, all_pairs, ss)
        for split, feats in all_feat.items():
            for stage, (u, v) in feats.items():
                for part, tensor in [('U',u), ('V',v)]:
                    geometry_rows.append(dict(variant=variant, split=split, stage=stage, part=part,
                                              **representation_geometry(tensor)))
        maxerr, prior_scores = frozen_original_check(ss, device, model, bridge, decoder, all_feat, all_pairs, run_dir/'test_scores.csv')
        original_report = balanced_score_info(all_y['test'], prior_scores)
        if abs(original_report['roc_auc'] - info['metrics']['ROC-AUC']) > 1e-8:
            raise RuntimeError(f'Original MLP AUC does not reproduce for {variant}')
        print(f'{variant}: reproducibility PASS, max error={maxerr:.3g}; test AUC={original_report["roc_auc"]:.4f}', flush=True)
        # Save original checkpoint predictions as historical baseline (no retraining).
        frame = all_frames['test'].copy(); frame['score'] = prior_scores
        frame.to_csv(out/f'{variant}_mlp_original_test_scores.csv', index=False)
        original_row = dict(variant=variant, stage='original_mlp', probe_seed='original', selected_epoch=int(max(x['epoch'] for x in info['history'] if x['val_auc']==info['val_auc'])),
                            validation_auc=float(info['val_auc']), test_auc=original_report['roc_auc'], test_ap=original_report['ap'],
                            train_auc=float('nan'), input_dim=int(packed['train']['gnn'].shape[1]), checkpoint_sha256=ckhash)
        result_rows.append(original_row)
        # Read-only cosine baselines by representation.
        for stage in FEATURE_KEYS:
            for name, s in zip(SPLITS, ss):
                val = numeric_probe(*all_feat[name][stage], all_pairs[name], s['num_u'])
                auc, ap = auc_ap(all_y[name], val)
                csv_rows.append(dict(variant=variant, stage=stage, split=name, metric='cosine_auc', value=auc))
                csv_rows.append(dict(variant=variant, stage=stage, split=name, metric='cosine_ap', value=ap))

        cfg = info['config']
        for stage in FEATURE_KEYS:
            train_x, val_x, test_x = (packed[k][stage] for k in SPLITS)
            # Exact SAME features and candidate labels across all probe seeds.
            for probe_seed in args.probe_seeds:
                model_new, epoch, vauc, history = fit_identical_mlp(
                    train_x, all_y['train'], val_x, all_y['validation'],
                    hidden=cfg['hidden'], dropout=cfg['dropout'], lr=cfg['lr'], wd=cfg['weight_decay'],
                    max_epochs=args.probe_epochs, patience=args.probe_patience, seed=probe_seed, device=device)
                data = {split: score_mlp(model_new, packed[split][stage]) for split in SPLITS}
                desc = {split: balanced_score_info(all_y[split], data[split]) for split in SPLITS}
                result_rows.append(dict(variant=variant, stage=stage, probe_seed=probe_seed,
                                        selected_epoch=epoch, validation_auc=vauc,
                                        train_auc=desc['train']['roc_auc'], test_auc=desc['test']['roc_auc'], test_ap=desc['test']['ap'],
                                        test_positive_mean=desc['test']['positive_mean'], test_negative_mean=desc['test']['negative_mean'],
                                        input_dim=int(train_x.shape[1]), checkpoint_sha256=ckhash))
                # Record traces to know if decoder trained and why it stops.
                tdir = out/'training_curves'/variant/stage
                tdir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(history).to_csv(tdir/f'seed_{probe_seed}.csv', index=False)
                # Use FIRST prespecified probe seed only for figures, NEVER choose on test.
                if probe_seed == args.probe_seeds[0]:
                    scores = all_frames['test'].copy(); scores['score'] = data['test']
                    scores.to_csv(out/'scores'/f'{variant}_{stage}_mlp_test_scores.csv', index=False)
                    plot_stage_hist(scores, out/'figures'/f'{variant}_{stage}_decoder_seed{probe_seed}.png',stage,variant,probe_seed)
        # Optional placebo with shuffled labels, never selected as benchmark candidate.
        if args.label_control and bits == '000':
            stage='raw'; train_x, val_x, test_x = (packed[k][stage] for k in SPLITS)
            seed=args.probe_seeds[0]
            model_ctl, epoch, vauc, history = fit_identical_mlp(
                train_x, all_y['train'], val_x, all_y['validation'], hidden=cfg['hidden'], dropout=cfg['dropout'],
                lr=cfg['lr'], wd=cfg['weight_decay'], max_epochs=args.probe_epochs,
                patience=args.probe_patience, seed=seed, device=device, permute_labels=True)
            ctl_auc, ctl_ap = auc_ap(all_y['test'], score_mlp(model_ctl, test_x))
            result_rows.append(dict(variant=variant, stage='raw_labels_permuted', probe_seed=seed,
                                    selected_epoch=epoch, validation_auc=vauc, train_auc=float('nan'),
                                    test_auc=ctl_auc, test_ap=ctl_ap, input_dim=int(train_x.shape[1]), checkpoint_sha256=ckhash))

    data = pd.DataFrame(result_rows)
    data.to_csv(out/'decoder_by_seed.csv', index=False)
    cos = pd.DataFrame(csv_rows)
    cos.to_csv(out/'cosine_reference.csv', index=False)
    pd.DataFrame(geometry_rows).to_csv(out/'representation_geometry.csv', index=False)
    aggregate = data[data.stage!='original_mlp'].groupby(['variant','stage'], as_index=False).agg(
        test_auc_mean=('test_auc','mean'), test_auc_sd=('test_auc','std'), test_ap_mean=('test_ap','mean'),
        train_auc_mean=('train_auc','mean'), validation_auc_mean=('validation_auc','mean'),
        selected_epoch_mean=('selected_epoch','mean'))
    aggregate.to_csv(out/'decoder_summary.csv', index=False)
    print('\nMLP test ROC-AUC (probe mean):',flush=True)
    print(aggregate.pivot(index='stage', columns='variant', values='test_auc_mean').to_string(float_format=lambda v: f'{v:.4f}'), flush=True)
    print(f'All artifacts written to {out}',flush=True)


if __name__ == '__main__':
    main()
