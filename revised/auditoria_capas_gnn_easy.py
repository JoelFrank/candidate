"""Read-only layer-by-layer inspection of the project5 WBT-BGRL BiSBM-EASY pipeline.

The submitted model, saved SSL weights, original decoder, data, and train/val/test
candidate rosters are NEVER changed. Probe MLPs are diagnostic-only, trained on
train candidates with validation-only epoch selection. Test is scored after.

Requires existing revised/auditoria_decoders_easy.py, revised/diagnostico_etapas_easy.py,
revised/run_easy.py, the two saved checkpoints, torch and PyTorch Geometric.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reuse the earlier verified protocol and exactly the same decoder class.
from revised.auditoria_decoders_easy import (
    get_training_dataset, load_variant, pair_features, fit_identical_mlp,
    score_mlp, balanced_score_info, verify_same_test, frozen_original_check,
    representation_geometry, numeric_probe, leakage_diagnostics, sha256,
)

SPLITS = ('train', 'validation', 'test')
BITS = ('000', '010')
STAGES = (
    'raw', 'feature', 'input_projection', 'conv1_linear',
    'conv1_relu', 'conv2_linear', 'output_projection',
    'ssl_predictor', 'no_edges_output', 'random_encoder_output',
)


def _split(t: torch.Tensor, n_u: int):
    return t[:n_u].detach(), t[n_u:].detach()


def positive_negative_hist(y, p, dest: Path, title: str):
    fig, ax = plt.subplots(figsize=(8, 4.7))
    bins = np.linspace(0, 1, 41)
    ax.hist(p[y == 1], bins=bins, color='#1769E0', alpha=.58,
            label='Positivos (azul)', density=True)
    ax.hist(p[y == 0], bins=bins, color='#DB3030', alpha=.58,
            label='Negativos (rojo)', density=True)
    ax.set(xlabel='Probabilidad del decoder', ylabel='Densidad', title=title, xlim=(0, 1))
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=180)
    plt.close(fig)


def _sha_labels(df):
    return hashlib.sha256(df[['u', 'v', 'label']].to_csv(index=False).encode()).hexdigest()


def capture_encoder_layers(encoder, call, n_u: int):
    """Capture true module outputs through hooks, never a rewritten GCN forward.

    Requires exactly 2 GCNConv modules. get_embeddings() output is independently
    checked against the captured output projection, proving exact equivalence.
    """
    if len(encoder.convs) != 2:
        raise RuntimeError(f'Expected exactly 2 GCNConv layers; found {len(encoder.convs)}')
    captured = {'output_projection': []}
    handles = []
    def save(name):
        def _hook(_module, _inputs, output):
            if not torch.is_tensor(output):
                raise RuntimeError(f'Unexpected non-tensor output in {name}')
            if name == 'output_projection':
                captured[name].append(output.detach())
            else:
                if name in captured:
                    raise RuntimeError(f'Hook {name} was called more than once')
                captured[name] = output.detach()
        return _hook
    try:
        for name, module in (
            ('u_proj', encoder.proj_u), ('v_proj', encoder.proj_v),
            ('conv1_linear', encoder.convs[0]), ('conv2_linear', encoder.convs[1]),
            ('output_projection', encoder.output_proj),
        ):
            handles.append(module.register_forward_hook(save(name)))
        with torch.no_grad():
            output = call()
    finally:
        for handle in handles:
            handle.remove()
    needed = ('u_proj', 'v_proj', 'conv1_linear', 'conv2_linear', 'output_projection')
    if any(k not in captured for k in needed) or len(captured['output_projection']) != 2:
        raise RuntimeError(f'Incomplete layer trace: {list(captured)}')
    out_u, out_v = output
    hook_u, hook_v = captured['output_projection']
    maxerr = max(float((out_u - hook_u).abs().max()), float((out_v - hook_v).abs().max()))
    if maxerr > 1e-6:
        raise RuntimeError(f'Hook trace disagrees with model forward: error={maxerr:.3g}')
    return OrderedDict([
        ('input_projection', (captured['u_proj'], captured['v_proj'])),
        ('conv1_linear', _split(captured['conv1_linear'], n_u)),
        # In eval mode dropout is the identity; conv1 ReLU is applied in original forward.
        ('conv1_relu', _split(F.relu(captured['conv1_linear']), n_u)),
        ('conv2_linear', _split(captured['conv2_linear'], n_u)),
        ('output_projection', (hook_u, hook_v)),
    ]), maxerr


def capture_one(ss, index: int, model, bridge, device, random_encoder=None):
    """Trace one chronological snapshot without reading future target links."""
    from revised.run_easy import remap_context
    s = ss[index]
    nu = s['num_u']
    xu, xv = [x.to(device) for x in s['inputs']]
    with torch.no_grad():
        bu, bv = bridge(xu, xv)
    ei, w = remap_context(ss[:max(1, index)], s)
    ei, w = ei.to(device), w.to(device)
    model.eval(); bridge.eval()
    trace, error = capture_encoder_layers(
        model.online_encoder,
        lambda: model.get_embeddings(bu, bv, ei, nu, w), nu)
    final = trace['output_projection']
    with torch.no_grad():
        pred_u = model.predictor_u(final[0])
        pred_v = model.predictor_v(final[1])
        empty = torch.empty((2,0), device=device, dtype=torch.long)
        empty_w = torch.empty((0,), device=device, dtype=torch.float32)
        no_u, no_v = model.get_embeddings(bu, bv, empty, nu, empty_w)
    values = OrderedDict([
        ('raw', (xu.detach(), xv.detach())),
        ('feature', (bu.detach(), bv.detach())),
    ])
    values.update(trace)
    values['ssl_predictor'] = (pred_u.detach(), pred_v.detach())
    values['no_edges_output'] = (no_u.detach(), no_v.detach())
    if random_encoder is not None:
        random_encoder.eval()
        with torch.no_grad():
            ru, rv = random_encoder(bu, bv, ei, w if model.weight_propagation else None, nu)
        values['random_encoder_output'] = (ru.detach(), rv.detach())
    return values, error


def score_cosine_all(values, pair, nu):
    return {key: numeric_probe(u, v, pair, nu) for key, (u, v) in values.items()}


def render_auc_figure(rows, output: Path):
    # One row per stage and variant; plot the average of separately selected decoders.
    agg = rows.groupby(['variant', 'stage'], as_index=False)['test_auc'].mean()
    ordered = [x for x in STAGES if x in set(rows['stage'])]
    fig, ax = plt.subplots(figsize=(max(10, len(ordered)*1.35), 5))
    xs = np.arange(len(ordered))
    width = .35
    for pos, variant in enumerate(BITS):
        part = agg[agg.variant == f'WBT_{variant}'].set_index('stage')
        heights = [float(part.loc[s, 'test_auc']) if s in part.index else np.nan for s in ordered]
        ax.bar(xs + (pos-.5)*width, heights, width, label=f'WBT_{variant}')
    ax.axhline(.5, linestyle='--', color='gray', linewidth=1, label='Azar')
    ax.set_xticks(xs); ax.set_xticklabels(ordered, rotation=40, ha='right')
    ax.set_ylim(0, 1.0); ax.set_ylabel('ROC-AUC en test')
    ax.set_title('BiSBM-EASY | decoders nuevos sobre capas congeladas')
    ax.legend(); ax.grid(axis='y', alpha=.2); fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='results/diagnostico_etapas_easy')
    p.add_argument('--graph', type=int, default=111)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--variants', choices=['000','010','000,010'], default='000,010')
    p.add_argument('--candidate-seed', type=int, default=2026)
    p.add_argument('--probe-seeds', nargs='+', type=int, default=[42,43,44])
    p.add_argument('--probe-epochs', type=int, default=250)
    p.add_argument('--probe-patience', type=int, default=30)
    p.add_argument('--random-seed', type=int, default=777)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', default='results/auditoria_capas_gnn_easy')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.probe_epochs < 1 or args.probe_patience < 1 or not args.probe_seeds:
        raise ValueError('Positive probe epochs, patience, and at least one probe seed required')
    if len(args.probe_seeds) != len(set(args.probe_seeds)):
        raise ValueError('Probe seeds must be distinct')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    source = Path(args.source).resolve(); out = Path(args.output).resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError('--source and --output must be separate (no nesting or overwriting)')
    requested = args.variants.split(',')
    for bits in requested:
        for fname in ['checkpoint.pt', 'result.json', 'test_scores.csv']:
            if not (source/f'g{args.graph}_s{args.seed}_WBT_{bits}'/fname).is_file():
                raise FileNotFoundError(f'Missing {bits}: {fname} in source {source}')

    from revised.run_easy import prepare_easy
    from revised.diagnostico_etapas_easy import source_hashes
    from types import SimpleNamespace
    j = json.loads((source/f'g{args.graph}_s{args.seed}_WBT_{requested[0]}'/'result.json').read_text())
    cfg = j['config']
    if cfg['candidate_seed'] != args.candidate_seed:
        raise RuntimeError('Candidate seed differs from saved run')
    ss, _, npz = prepare_easy(SimpleNamespace(graph_seed=args.graph, seed=args.seed,
                                easy_groups=cfg['easy_groups'], easy_noise=cfg['easy_noise'], hidden=cfg['hidden']))
    pairs, y, frames = get_training_dataset(ss, args.candidate_seed)
    hashes = {_s: _sha_labels(frames[_s]) for _s in SPLITS}
    orig_sha = {}
    checkpoints = {}
    for bits in requested:
        folder = source/f'g{args.graph}_s{args.seed}_WBT_{bits}'
        orig_sha[bits] = verify_same_test(folder/'test_scores.csv', frames['test'])
        checkpoints[bits] = {fn: sha256(folder/fn) for fn in ['checkpoint.pt','result.json','test_scores.csv']}
    if len(set(orig_sha.values())) != 1:
        raise RuntimeError('Test candidate mismatch between variants')
    from revised.diagnostico_etapas_easy import source_hashes
    plan = dict(input=str(source), graph=args.graph, seed=args.seed,
                variants=requested, candidate_seed=args.candidate_seed,
                probe_seeds=args.probe_seeds, probe_epochs=args.probe_epochs,
                probe_patience=args.probe_patience, random_seed=args.random_seed,
                device=args.device, candidate_sha256=hashes,
                checkpoint_hashes=checkpoints,
                generator_sha256=sha256(Path(npz)),
                source_hashes=source_hashes(),
                audit_helper_sha256=sha256(ROOT/'revised/auditoria_decoders_easy.py'),
                this_script_sha256=sha256(Path(__file__)))
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError('Output folder belongs to another plan or source version; choose a NEW --output')
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan, indent=2))
    (out/'candidate_checks.json').write_text(json.dumps(leakage_diagnostics(ss, frames, npz), indent=2))
    (out/'candidate_hashes.json').write_text(json.dumps(hashes, indent=2))
    rows=[]; geometry=[]; cosines=[]; controls=[]
    for bits in requested:
        label = f'WBT_{bits}'
        j, model, bridge, decoder, folder, ckhash = load_variant(args, bits, device, ss, source)
        # Fresh GCN, no SSL, with the trained bridge held fixed. Do NOT reset a copy:
        # WeightedGCNEncoder.reset_parameters() does not reset all biases.
        torch.manual_seed(args.random_seed)
        cls = type(model.online_encoder)
        original_encoder = model.online_encoder
        random_encoder = cls(input_dim_u=original_encoder.input_dim_u,
                             input_dim_v=original_encoder.input_dim_v,
                             hidden_dim=original_encoder.hidden_dim,
                             output_dim=original_encoder.output_dim,
                             num_layers=original_encoder.num_layers,
                             dropout=original_encoder.dropout).to(device).eval()
        features={}; hook_errors={}
        for i, split in enumerate(SPLITS):
            stages, e = capture_one(ss, i, model, bridge, device, random_encoder)
            if tuple(stages) != STAGES:
                raise RuntimeError(f'Unexpected stages {tuple(stages)}')
            features[split] = stages; hook_errors[split] = e
            for name,(u,v) in stages.items():
                for side, x in [('U',u),('V',v)]:
                    geometry.append(dict(variant=label,split=split,stage=name,partition=side,
                                         **representation_geometry(x)))
                scores = numeric_probe(u, v, pairs[split], ss[i]['num_u'])
                score = balanced_score_info(y[split], scores)
                cosines.append(dict(variant=label,split=split,stage=name,auc=score['roc_auc'],
                                    ap=score['ap'],positive_mean=score['positive_mean'],
                                    negative_mean=score['negative_mean']))
        # Check exact original embeddings + exact original decoder probabilities.
        original_x = pair_features(*features['test']['output_projection'], pairs['test'], ss[2]['num_u'])
        with torch.no_grad():
            p = score_mlp(decoder, original_x)
        old = pd.read_csv(folder/'test_scores.csv')
        maxerr=float(np.max(np.abs(p-old['score'].to_numpy())))
        if maxerr > 1e-5:
            raise RuntimeError(f'{label}: original decoder score mismatch, max error={maxerr}')
        old_auc=balanced_score_info(y['test'], p)['roc_auc']
        if abs(old_auc - j['metrics']['ROC-AUC']) > 1e-8:
            raise RuntimeError(f'{label}: original AUC mismatch')
        print(f'{label}: original decoder reproduced: AUC={old_auc:.4f}, maxerr={maxerr:.2g}',flush=True)
        controls.append(dict(variant=label, original_auc=old_auc,decoder_abs_error=maxerr,
                             **{f'{split}_forward_hook_error': hook_errors[split] for split in SPLITS}))
        saved_original = frames['test'].copy()
        saved_original['score'] = p
        (out/'scores').mkdir(parents=True, exist_ok=True)
        saved_original.to_csv(out/'scores'/f'{label}_original_decoder.csv', index=False)

        for stage in STAGES:
            packed={split: pair_features(*features[split][stage],pairs[split],ss[i]['num_u'])
                    for i,split in enumerate(SPLITS)}
            for probe_seed in args.probe_seeds:
                probe, epoch, val_auc, curve = fit_identical_mlp(
                    packed['train'],y['train'],packed['validation'],y['validation'],
                    hidden=j['config']['hidden'],dropout=j['config']['dropout'],
                    lr=j['config']['lr'],wd=j['config']['weight_decay'],
                    max_epochs=args.probe_epochs,patience=args.probe_patience,
                    seed=probe_seed, device=device)
                scores={split:score_mlp(probe,packed[split]) for split in SPLITS}
                perf={split:balanced_score_info(y[split],scores[split]) for split in SPLITS}
                rows.append(dict(variant=label,stage=stage,probe_seed=probe_seed,
                                 selected_epoch=epoch, validation_auc=val_auc,
                                 train_auc=perf['train']['roc_auc'],
                                 test_auc=perf['test']['roc_auc'],
                                 test_ap=perf['test']['ap'],
                                 input_dim=int(packed['train'].shape[1])))
                path=out/'curves'/label/stage
                path.mkdir(parents=True,exist_ok=True)
                pd.DataFrame(curve).to_csv(path/f'seed_{probe_seed}.csv', index=False)
                if probe_seed == args.probe_seeds[0]:
                    one=frames['test'].copy()
                    one['score']=scores['test']
                    one.to_csv(out/'scores'/f'{label}_{stage}_seed{probe_seed}.csv',index=False)
                    positive_negative_hist(y['test'],scores['test'],
                        out/'figures'/f'{label}_{stage}_seed{probe_seed}.png',
                        f'{label} | {stage} | MLP seed {probe_seed} | AUC {perf["test"]["roc_auc"]:.4f}')
            print(f'{label}: {stage}: completed {len(args.probe_seeds)} probe(s)',flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(out/'mlp_by_stage_seed.csv',index=False)
    summary=df.groupby(['variant','stage'],as_index=False).agg(
        test_auc_mean=('test_auc','mean'),test_auc_sd=('test_auc','std'),
        test_ap_mean=('test_ap','mean'),train_auc_mean=('train_auc','mean'),
        val_auc_mean=('validation_auc','mean'),selected_epoch_mean=('selected_epoch','mean'))
    summary.to_csv(out/'mlp_stage_summary.csv',index=False)
    pd.DataFrame(geometry).to_csv(out/'layer_geometry.csv',index=False)
    pd.DataFrame(cosines).to_csv(out/'layer_cosine.csv',index=False)
    pd.DataFrame(controls).to_csv(out/'reproduction_checks.csv',index=False)
    render_auc_figure(df,out/'figures'/'comparacion_auc_capas.png')
    print('\nROC-AUC test (MLP probed, means):', flush=True)
    print(summary.pivot(index='stage',columns='variant',values='test_auc_mean').to_string(float_format=lambda t:f'{t:.4f}'),flush=True)
    print(f'ALL COMPLETE: {out}',flush=True)


if __name__ == '__main__':
    main()
