"""Run BiSBM-EASY on selected WBT configurations and create standard artifacts."""
from __future__ import annotations
import argparse, hashlib, itertools, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    'revised/run_easy.py', 'revised/easy_sweep.py', 'revised/easy_analysis.py',
    'revised/easy_generator_loader.py', 'revised/easy_negatives.py',
    'data/synthetic/generator_easy_project5.py', 'src/models/wbtbgrl.py',
    'src/evaluation/decoder.py',
)
LABELS = {'000':'WBT_000','010':'WBT_010','100':'WBT_100','111':'WBT_111'}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_cmd(a, graph: int, seed: int, bits: str, out: Path) -> list[str]:
    p, aug, ssl = (int(c) for c in bits)
    variant = 'WP_NWB' if bits != '000' else 'NWP_NWB'
    return [
        sys.executable, str(ROOT / 'revised/run_easy.py'),
        '--graph-seed', str(graph), '--seed', str(seed), '--variant', variant,
        '--propagation', str(p), '--augmentation', str(aug), '--loss-weights', str(ssl),
        '--candidate-seed', str(a.candidate_seed), '--easy-groups', str(a.easy_groups), '--easy-noise', str(a.easy_noise),
        '--encoder-epochs', str(a.encoder_epochs), '--decoder-epochs', str(a.decoder_epochs),
        '--hidden', str(a.hidden), '--dim', str(a.dim), '--patience', str(a.patience), '--threads', str(a.threads),
        '--lr', str(a.lr), '--weight-decay', str(a.weight_decay), '--dropout', str(a.dropout), '--lam', str(a.lam),
        '--edge-drop', str(a.edge_drop), '--feature-drop', str(a.feature_drop),
        '--device', a.device, '--output', str(out),
    ]


def validate(path: Path, graph: int, seed: int, bits: str, candidate_seed: int):
    j = path / 'result.json'; csv = path / 'test_scores.csv'
    if not j.exists() and not csv.exists():
        if path.exists() and any(path.iterdir()):
            raise RuntimeError(f'Incomplete output folder: {path}')
        return False
    if not j.exists() or not csv.exists():
        raise RuntimeError(f'Partial output: {path}')
    d = json.loads(j.read_text()); c = d['config']
    exp = {
        'graph_seed': graph, 'seed': seed, 'candidate_seed': candidate_seed,
        'variant': 'WP_NWB' if bits != '000' else 'NWP_NWB',
        'propagation': bool(int(bits[0])), 'augmentation': bool(int(bits[1])), 'loss_weights': bool(int(bits[2])),
    }
    for k, v in exp.items():
        if c.get(k) != v:
            raise RuntimeError(f'Config mismatch at {path}: {k}: {c.get(k)!r} != {v!r}')
    sampling = d.get('negative_sampling', {})
    if sampling.get('policy') != 'matched_easy_stage':
        raise RuntimeError(f'Unexpected negative sampling policy: {path}')
    strata = sampling.get('strata', {})
    if any(stage not in strata or strata[stage]['positive'] != strata[stage]['negative']
           for stage in ('train','validation','test')):
        raise RuntimeError(f'Invalid negative strata matching: {path}')
    if 'ROC-AUC' not in d.get('metrics', {}):
        raise RuntimeError(f'Missing metrics at {path}')
    return True


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graphs', nargs='+', type=int, default=[111,112,113])
    p.add_argument('--seeds', nargs='+', type=int, default=[42,43,44,45,46])
    p.add_argument('--variants', default='000,010,100,111')
    p.add_argument('--output', default='results/bisbm_easy_project5')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--delete-checkpoints', action='store_true')
    p.add_argument('--candidate-seed', type=int, default=2026)
    p.add_argument('--easy-groups', type=int, default=16)
    p.add_argument('--easy-noise', type=float, default=0.02)
    for flag, default in [('encoder-epochs',200),('decoder-epochs',100),('hidden',128),('dim',64),('patience',20),('threads',2)]:
        p.add_argument('--'+flag, type=int, default=default)
    for flag, default in [('lr',.001),('weight-decay',1e-5),('dropout',.1),('lam',.5),('edge-drop',.1),('feature-drop',.1)]:
        p.add_argument('--'+flag, type=float, default=default)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    variants = a.variants.split(',')
    if any(v not in LABELS for v in variants) or len(set(variants)) != len(variants):
        raise ValueError('Allowed variants: 000,010,100,111')
    if len(set(a.graphs)) != len(a.graphs) or len(set(a.seeds)) != len(a.seeds):
        raise ValueError('Duplicate graph or seed not allowed')
    outroot = Path(a.output)
    plan = dict(graphs=a.graphs, seeds=a.seeds, variants=variants, candidate_seed=a.candidate_seed, easy_groups=a.easy_groups,
                easy_noise=a.easy_noise, encoder_epochs=a.encoder_epochs, decoder_epochs=a.decoder_epochs,
                hidden=a.hidden, dim=a.dim, patience=a.patience, threads=a.threads, lr=a.lr, weight_decay=a.weight_decay,
                dropout=a.dropout, lam=a.lam, edge_drop=a.edge_drop, feature_drop=a.feature_drop,
                device=a.device, source_sha256={s: sha256(ROOT/s) for s in SOURCES})
    cfg = outroot / 'easy_plan.json'
    jobs = list(itertools.product(a.graphs, a.seeds, variants))
    print(f'{len(jobs)} EASY runs planned', flush=True)
    if a.execute:
        outroot.mkdir(parents=True, exist_ok=True)
        if cfg.exists() and json.loads(cfg.read_text()) != plan:
            raise RuntimeError('Existing output has a different plan. Use a new --output directory.')
        if not cfg.exists():
            cfg.write_text(json.dumps(plan, indent=2))
    done = 0
    logs = outroot / '_launcher_logs'
    for i, (g, s, bits) in enumerate(jobs, 1):
        path = outroot / f'g{g}_s{s}_{LABELS[bits]}'
        if validate(path, g, s, bits, a.candidate_seed):
            if a.delete_checkpoints and (path/'checkpoint.pt').exists():
                (path/'checkpoint.pt').unlink()
            done += 1
            print(f'[{i}/{len(jobs)}] SKIP validated: {path}', flush=True)
            continue
        cmd = create_cmd(a, g, s, bits, path)
        print(f'[{i}/{len(jobs)}] {"RUN" if a.execute else "DRY"}: {path}', flush=True)
        if not a.execute:
            print('  ' + ' '.join(cmd), flush=True)
            continue
        # Launcher logs are OUTSIDE the experiment folder until the run ends.
        # run_easy.py creates its own output folder with exist_ok=False.
        if path.exists():
            raise RuntimeError(f'Partial experiment folder exists: {path}; review before retrying')
        logs.mkdir(parents=True, exist_ok=True)
        logpath = logs / f'{path.name}.log'
        if logpath.exists():
            raise RuntimeError(f'Previous launcher failure log exists: {logpath}; '
                               'inspect and archive it before retrying')
        env = os.environ.copy(); env.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        with logpath.open('w') as fh:
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT, check=False)
        if proc.returncode:
            raise RuntimeError(f'Execution failed ({proc.returncode}); inspect {logpath}')
        validate(path, g, s, bits, a.candidate_seed)
        (path/'run.log').write_bytes(logpath.read_bytes())
        logpath.unlink()
        if a.delete_checkpoints and (path/'checkpoint.pt').exists():
            (path/'checkpoint.pt').unlink()
        done += 1
    print(f'Completed and verified: {done}/{len(jobs)}', flush=True)


if __name__ == '__main__':
    main()
