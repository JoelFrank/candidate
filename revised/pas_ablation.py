"""Independent P/A/S (propagation/augmentation/SSL weighting) 2^3 audit.

This is an additive orchestration script. It invokes the existing revised/run.py
without changing the manuscript's SSL, the decoder, or the BiSBM generator.
No results are selected using test performance.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    'revised/run.py', 'revised/data.py', 'revised/negative_sampling.py',
    'src/models/wbtbgrl.py', 'data/synthetic/generator.py',
)
BITS = tuple(''.join(map(str, bits)) for bits in itertools.product((0, 1), repeat=3))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_bits(value: str) -> list[str]:
    values = list(BITS) if value == 'all' else value.split(',')
    if not values or any(v not in BITS for v in values) or len(set(values)) != len(values):
        raise ValueError(f'--variants must be all or comma-delimited subset of {BITS}')
    return values


def create_command(a, graph: int, seed: int, condition: str, bits: str, output: Path) -> list[str]:
    p, aug, ssl = (int(c) for c in bits)
    # NWB decoder is fixed. Explicit P/A/S flags override the WP defaults.
    variant = 'WP_NWB' if bits != '000' else 'NWP_NWB'
    return [
        sys.executable, str(ROOT / 'revised/run.py'),
        '--dataset', 'synthetic', '--model', 'WBTBGRL',
        '--graph-seed', str(graph), '--seed', str(seed),
        '--condition', condition, '--variant', variant,
        '--propagation', str(p), '--augmentation', str(aug),
        '--loss-weights', str(ssl), '--negative-strategy', 'matched_synthetic',
        '--candidate-seed', str(a.candidate_seed),
        '--encoder-epochs', str(a.encoder_epochs), '--decoder-epochs', str(a.decoder_epochs),
        '--hidden', str(a.hidden), '--dim', str(a.dim),
        '--patience', str(a.patience), '--threads', str(a.threads),
        '--lr', str(a.lr), '--weight-decay', str(a.weight_decay),
        '--dropout', str(a.dropout), '--lam', str(a.lam),
        '--edge-drop', str(a.edge_drop), '--feature-drop', str(a.feature_drop),
        '--device', a.device, '--output', str(output),
    ]


def validate_completion(path: Path, graph: int, seed: int, condition: str, bits: str, candidate_seed: int):
    j = path / 'result.json'
    csv = path / 'test_scores.csv'
    if not j.exists() and not csv.exists():
        if path.exists() and any(path.iterdir()):
            raise RuntimeError(f'Incomplete output folder (do not overwrite): {path}')
        return False
    if not j.exists() or not csv.exists():
        raise RuntimeError(f'Partial output (do not overwrite): {path}')
    d = json.loads(j.read_text())
    c = d['config']
    expected = {
        'dataset': 'synthetic', 'model': 'WBTBGRL', 'graph_seed': graph,
        'seed': seed, 'condition': condition, 'candidate_seed': candidate_seed,
        'negative_strategy': 'matched_synthetic',
        'variant': 'NWP_NWB' if bits == '000' else 'WP_NWB',
        'propagation': bool(int(bits[0])), 'augmentation': bool(int(bits[1])),
        'loss_weights': bool(int(bits[2])),
    }
    for k, v in expected.items():
        if c.get(k) != v:
            raise RuntimeError(f'Result config mismatch {path}: {k}: {c.get(k)!r} != {v!r}')
    if d.get('negative_sampling', {}).get('policy') != 'matched_synthetic':
        raise RuntimeError(f'Unexpected negative sampling policy: {path}')
    if not d.get('candidate_sha256'):
        raise RuntimeError(f'Missing candidate hash: {path}')
    strata = d.get('negative_sampling', {}).get('strata', {})
    if not all(t in strata and strata[t]['positive'] == strata[t]['negative'] for t in ['train', 'val', 'test']):
        raise RuntimeError(f'Strata were not matched: {path}')
    if 'ROC-AUC' not in d.get('metrics', {}):
        raise RuntimeError(f'Missing ROC-AUC: {path}')
    return True


def plan(a):
    paths = {name: sha256(ROOT / name) for name in SOURCES}
    if not all((ROOT / name).is_file() for name in SOURCES):
        raise FileNotFoundError('Missing required current project5 sources')
    return dict(
        title='WBT-BGRL P/A/S full factorial; NWB decoder fixed',
        scheme='2x2x2; P=GCNConv(edge_weight), A=weighted edge dropout, S=raw weight SSL edge attraction',
        conditions=list(a.conditions), graphs=list(a.graphs), seeds=list(a.seeds),
        variants=selected_bits(a.variants),
        negative_strategy='matched_synthetic', candidate_seed=a.candidate_seed,
        encoder_epochs=a.encoder_epochs, decoder_epochs=a.decoder_epochs,
        hidden=a.hidden, dim=a.dim, patience=a.patience, threads=a.threads,
        lr=a.lr, weight_decay=a.weight_decay, dropout=a.dropout,
        lam=a.lam, edge_drop=a.edge_drop, feature_drop=a.feature_drop,
        device=a.device, delete_checkpoints=a.delete_checkpoints, source_sha256=paths,
    )


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graphs', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    p.add_argument('--conditions', nargs='+', choices=['informative', 'orthogonal', 'anti_correlated'], default=['informative'])
    p.add_argument('--variants', default='all', help='all or e.g. 000,100,111 (bits in P,A,S order)')
    p.add_argument('--output', default='results/ablacion_PAS_BiSBM')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--execute', action='store_true', help='Required to start training; otherwise dry run')
    p.add_argument('--delete-checkpoints', action='store_true', help='Optional: remove checkpoint.pt ONLY after a validated successful run (saves disk)')
    p.add_argument('--candidate-seed', type=int, default=2026)
    for flag, default in [('encoder-epochs', 200), ('decoder-epochs', 100), ('hidden',128), ('dim',64), ('patience',20), ('threads',2)]:
        p.add_argument('--'+flag, type=int, default=default)
    for flag, default in [('lr',.001),('weight-decay',1e-5),('dropout',.1),('lam',.5),('edge-drop',.1),('feature-drop',.1)]:
        p.add_argument('--'+flag, type=float, default=default)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if len(set(a.graphs)) != len(a.graphs) or len(set(a.seeds)) != len(a.seeds) or len(set(a.conditions)) != len(a.conditions):
        raise ValueError('Duplicate graph, seed or condition not allowed')
    cfg = plan(a)
    outroot = Path(a.output)
    cfgfile = outroot / 'pas_plan.json'
    jobs = list(itertools.product(a.graphs, a.seeds, a.conditions, selected_bits(a.variants)))
    print(f'{len(jobs)} runs, {len(a.graphs)} graph(s), {len(a.seeds)} training seed(s), {len(a.conditions)} condition(s), {len(selected_bits(a.variants))} P/A/S combinations',flush=True)
    if cfgfile.exists() and json.loads(cfgfile.read_text()) != cfg:
        raise RuntimeError('Plan/source mismatch in existing output. Use a new --output directory; do not mix experiments.')
    if a.execute:
        outroot.mkdir(parents=True, exist_ok=True)
        if not cfgfile.exists():
            cfgfile.write_text(json.dumps(cfg, indent=2))
    successes = 0
    for i, (g, s, c, bits) in enumerate(jobs, 1):
        path = outroot / f'g{g}_s{s}_{c}_WBT_{bits}'
        done = validate_completion(path, g, s, c, bits, a.candidate_seed)
        if done:
            if a.execute and a.delete_checkpoints and (path/'checkpoint.pt').exists():
                (path/'checkpoint.pt').unlink()
            successes += 1
            print(f'[{i}/{len(jobs)}] SKIP validated: {path}',flush=True)
            continue
        cmd = create_command(a, g, s, c, bits, path)
        print(f'[{i}/{len(jobs)}] {"RUN" if a.execute else "DRY"}: {path}',flush=True)
        if not a.execute:
            print('  ' + ' '.join(cmd),flush=True)
            continue
        path.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy(); env.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        with open(path / 'run.log', 'w') as log:
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        if proc.returncode:
            raise RuntimeError(f'Execution failed with exit code {proc.returncode}: {path}/run.log')
        validate_completion(path, g, s, c, bits, a.candidate_seed)
        if a.delete_checkpoints and (path/'checkpoint.pt').exists():
            (path/'checkpoint.pt').unlink()
        successes += 1
    print(f'Validated complete: {successes}/{len(jobs)}; results: {outroot}',flush=True)


if __name__ == '__main__':
    main()
