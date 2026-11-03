"""BiSBM-original controlled SSL comparison, without modifying manuscript model.

Factors: BiSBM weight condition (3), SSL objective (original, symmetric),
augmentation P/A/S (000, 010). Decoder always NWB. Each model uses the
EXACT project5 revised/run.py training/selection/evaluation implementation.
The symmetric objective is the previously tested *diagnostic* variant from
revised/ssl_factorial_easy.py; corrupted branch remains ONLINE in both arms.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONDITIONS = ('informative', 'orthogonal', 'anti_correlated')
MODES = ('original', 'symmetric')
BITS = ('000', '010')
SOURCE_FILES = (
    'src/models/wbtbgrl.py',
    'revised/run.py', 'revised/data.py', 'revised/negative_sampling.py',
    'data/synthetic/generator.py', 'revised/ssl_factorial_easy.py',
    'revised/ssl_bisbm_original.py', 'revised/ssl_bisbm_analisis.py',
)


def hashes():
    missing = [k for k in SOURCE_FILES if not (ROOT / k).is_file()]
    if missing:
        raise FileNotFoundError('Missing project5 dependency: ' + ', '.join(missing))
    return {k: hashlib.sha256((ROOT/k).read_bytes()).hexdigest() for k in SOURCE_FILES}


def selected(value, options):
    parts = value.split(',')
    if not parts or not all(p in options for p in parts) or len(set(parts)) != len(parts):
        raise ValueError(f'Invalid/duplicated option {value!r}. Allowed: {options}')
    return parts


def jobs(a):
    if not a.graphs or len(set(a.graphs)) != len(a.graphs):
        raise ValueError('Graphs missing or duplicated')
    if not a.seeds or len(set(a.seeds)) != len(a.seeds):
        raise ValueError('Seeds missing or duplicated')
    return list(itertools.product(a.graphs, selected(a.conditions, CONDITIONS), a.seeds,
                                  selected(a.variants, BITS), selected(a.modes, MODES)))


def expected_run(a, graph, condition, seed, bits):
    return dict(dataset='synthetic', model='WBTBGRL', graph_seed=graph, condition=condition,
                seed=seed, candidate_seed=a.candidate_seed,
                variant='NWP_NWB' if bits == '000' else 'WP_NWB',
                propagation=bool(int(bits[0])), augmentation=bool(int(bits[1])),
                loss_weights=bool(int(bits[2])),
                negative_strategy='matched_synthetic',
                encoder_epochs=a.encoder_epochs, decoder_epochs=a.decoder_epochs,
                threads=a.threads, hidden=a.hidden, dim=a.dim, patience=a.patience,
                lr=a.lr, dropout=a.dropout, weight_decay=a.weight_decay,
                lam=a.lam, edge_drop=a.edge_drop, feature_drop=a.feature_drop,
                device=a.device)


def validate_complete(path, a, graph, condition, seed, bits, mode, source=None):
    if not path.exists():
        return False
    required = [path / n for n in ('result.json', 'test_scores.csv', 'ssl_bisbm_mode.json', 'data_audit.json')]
    if any(not f.is_file() for f in required):
        raise RuntimeError(f'Incomplete experiment: {path}. Do not overwrite automatically.')
    report = json.loads(required[0].read_text())
    metadata = json.loads(required[2].read_text())
    if (metadata.get('mode'), metadata.get('bits')) != (mode, bits):
        raise RuntimeError(f'Diagnostic-mode mismatch in {path}')
    if metadata.get('source_sha256') != (source if source is not None else hashes()):
        raise RuntimeError(f'Source files changed since {path} was trained')
    for name, value in expected_run(a, graph, condition, seed, bits).items():
        if report.get('config', {}).get(name) != value:
            raise RuntimeError(f'Config mismatch {name} in {path}: {report.get("config",{}).get(name)!r} != {value!r}')
    neg = report.get('negative_sampling', {})
    if neg.get('policy') != 'matched_synthetic':
        raise RuntimeError(f'Wrong negative policy: {path}')
    for stage in ('train','val','test'):
        strata = neg.get('strata', {}).get(stage, {})
        if not strata or any(strata['positive'].get(k)!=strata['negative'].get(k) for k in ('OO','ON','NO','NN')):
            raise RuntimeError(f'Mismatched {stage} negative strata in {path}')
    if not report.get('candidate_sha256') or 'ROC-AUC' not in report.get('metrics', {}):
        raise RuntimeError(f'Missing candidate fingerprint or ROC-AUC: {path}')
    return True


def execute_one(a, source):
    # Use the same runner as all prior BiSBM P/A/S studies. Only the SSL loss is
    # temporarily replaced when --mode=symmetric, and only for this subprocess.
    from revised import run as runner
    from revised.ssl_factorial_easy import create_model_type
    if a.mode == 'symmetric':
        runner.WBTBGRL = create_model_type(runner.WBTBGRL, a.mode)
    cfg = runner.parser().parse_args([])
    for k,v in expected_run(a, a.graph, a.condition, a.seed, a.bits).items():
        setattr(cfg, k, v)
    cfg.output = a.output
    result_dir = Path(a.output)
    if result_dir.exists():
        raise RuntimeError(f'Destination is already present: {result_dir}')
    runner.run(cfg)
    # Write provenance only after the full training completed successfully.
    (result_dir/'ssl_bisbm_mode.json').write_text(json.dumps({
        'mode':a.mode, 'bits':a.bits, 'online_corruption_for_both_modes': True,
        'method':'project5 original (unilateral) vs diagnostic U/V symmetric objective',
        'decoder':'NWB (unweighted BCE)',
        'source_sha256':source,
    }, indent=2))


def child_cmd(a, graph, condition, seed, bits, mode, folder):
    cmd = [sys.executable, str(Path(__file__).resolve()), '--single',
           '--graph',str(graph),'--condition',condition,'--seed',str(seed),
           '--bits',bits,'--mode',mode,'--output',str(folder),
           '--device',a.device,'--candidate-seed',str(a.candidate_seed)]
    for key in ('encoder_epochs','decoder_epochs','hidden','dim','patience','threads',
                'lr','weight_decay','dropout','lam','edge_drop','feature_drop'):
        cmd += ['--'+key.replace('_','-'),str(getattr(a,key))]
    return cmd


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graphs',nargs='+',type=int,default=[117,118,119],
                   help='New BiSBM realizations; default 117 118 119 were checked for valid splits')
    p.add_argument('--conditions',default=','.join(CONDITIONS))
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46])
    p.add_argument('--variants',default=','.join(BITS))
    p.add_argument('--modes',default=','.join(MODES))
    p.add_argument('--output',default='results/ssl_bisbm_original_multicondicion')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--execute',action='store_true',help='Required to actually train')
    p.add_argument('--delete-checkpoints',action='store_true',help='Delete checkpoints only for verified completed runs')
    p.add_argument('--single',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--graph',type=int,default=117,help=argparse.SUPPRESS)
    p.add_argument('--condition',choices=CONDITIONS,default='informative',help=argparse.SUPPRESS)
    p.add_argument('--seed',type=int,default=42,help=argparse.SUPPRESS)
    p.add_argument('--bits',choices=BITS,default='000',help=argparse.SUPPRESS)
    p.add_argument('--mode',choices=MODES,default='original',help=argparse.SUPPRESS)
    p.add_argument('--candidate-seed',type=int,default=2026)
    for k,v in [('encoder-epochs',200),('decoder-epochs',100),('hidden',128),('dim',64),('patience',20),('threads',2)]:
        p.add_argument('--'+k,type=int,default=v)
    for k,v in [('lr',.001),('weight-decay',1e-5),('dropout',.1),('lam',.5),('edge-drop',.1),('feature-drop',.1)]:
        p.add_argument('--'+k,type=float,default=v)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    provenance = hashes()
    if a.single:
        execute_one(a, provenance)
        return
    work = jobs(a)
    root = Path(a.output)
    params = {k:v for k,v in vars(a).items() if k not in ('execute','single','graph','condition','seed','bits','mode','delete_checkpoints')}
    plan = {'study':'Original BiSBM, 3 conditions, original vs symmetric SSL, NWB decoder',
            'protocol':'project5 revised/run.py, historical context, matched_synthetic negatives, val decoder selection',
            'params':params,'source_sha256':provenance,'planned_runs':len(work)}
    plan_path = root/'plan.json'
    if plan_path.is_file() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError(f'Different plan or sources at {root}; use a different --output')
    print(f'{len(work)} BiSBM trainings: {len(a.graphs)} graphs x {len(selected(a.conditions,CONDITIONS))} conditions '
          f'x {len(a.seeds)} training seeds x {len(selected(a.variants,BITS))} variants '
          f'x {len(selected(a.modes,MODES))} SSL objectives. NWB fixed.',flush=True)
    if a.execute:
        root.mkdir(parents=True,exist_ok=True)
        if not plan_path.is_file():
            plan_path.write_text(json.dumps(plan,indent=2,sort_keys=True))
        (root/'logs').mkdir(exist_ok=True)
    count = 0
    for i,(graph,condition,seed,bits,mode) in enumerate(work,1):
        dest = root/f'g{graph}_{condition}_s{seed}_WBT_{bits}_{mode}'
        if validate_complete(dest,a,graph,condition,seed,bits,mode,provenance):
            if a.delete_checkpoints and a.execute:
                (dest/'checkpoint.pt').unlink(missing_ok=True)
            count += 1
            print(f'[{i}/{len(work)}] SKIP verified: {dest}',flush=True)
            continue
        print(f'[{i}/{len(work)}] {"RUN" if a.execute else "DRY"}: {dest}',flush=True)
        if not a.execute:
            continue
        env = os.environ.copy()
        env.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        log = root/'logs'/f'g{graph}_{condition}_s{seed}_WBT_{bits}_{mode}.log'
        with log.open('w') as f:
            process = subprocess.run(child_cmd(a,graph,condition,seed,bits,mode,dest),
                                     cwd=str(ROOT),env=env,stdout=f,stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(f'Execution failed ({process.returncode}): {log}; data left intact at {dest}')
        validate_complete(dest,a,graph,condition,seed,bits,mode,provenance)
        if a.delete_checkpoints:
            (dest/'checkpoint.pt').unlink(missing_ok=True)
        count += 1
        print(f'[{i}/{len(work)}] VERIFIED',flush=True)
    if a.execute:
        print(f'ALL COMPLETE: {count}/{len(work)} verified trainings.',flush=True)

if __name__=='__main__':
    main()
