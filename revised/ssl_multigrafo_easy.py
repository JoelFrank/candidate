"""Reproducible 3-graph x 5-seed x 2-variant x 4-mode SSL study.

Additive orchestration over revised/ssl_factorial_easy.py already validated on the
user's GPU. NEVER modifies src/models/wbtbgrl.py or the EASY generator.
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

ROOT = Path(__file__).resolve().parents[1]
ORIG = ROOT / 'revised/ssl_factorial_easy.py'
MODES = ('original','symmetric','target_corr','symmetric_target_corr')
VARIANTS = ('000','010')
SOURCE_FILES = (
    'revised/ssl_factorial_easy.py', 'revised/ssl_multigrafo_easy.py',
    'revised/ssl_multigrafo_analisis.py', 'src/models/wbtbgrl.py',
    'revised/run_easy.py', 'revised/easy_negatives.py',
    'data/synthetic/generator_easy_project5.py',
)


def source_hashes():
    missing = [p for p in SOURCE_FILES if not (ROOT / p).is_file()]
    if missing:
        raise FileNotFoundError('Missing project5 file(s): ' + ', '.join(missing))
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCE_FILES}


def unique_ints(value):
    if len(value) == 0 or len(set(value)) != len(value):
        raise ValueError('Empty/duplicate graph or seed values')
    return value


def selection(text, options):
    chosen = tuple(text.split(','))
    if not chosen or len(chosen) != len(set(chosen)) or set(chosen) - set(options):
        raise ValueError(f'Invalid selection: {text}. Options: {options}')
    return chosen


def jobs(a):
    return list(itertools.product(unique_ints(a.graphs), unique_ints(a.seeds),
                                  selection(a.variants,VARIANTS),selection(a.modes,MODES)))


def plan(a):
    v = vars(a).copy()
    for key in ('execute',):
        v.pop(key, None)
    return {'study':'BiSBM-EASY SSL factorial across independent graph realizations',
            'protocol':'predefined factorial; decoder val selection; report test only after training',
            'param':v,'source_sha256':source_hashes(), 'number_of_runs':len(jobs(a))}


def run_command(a, graph, seed, path):
    cmd = [sys.executable, str(ORIG), '--graph',str(graph),'--seed',str(seed),
           '--variants',a.variants,'--modes',a.modes,
           '--encoder-epochs',str(a.encoder_epochs),'--decoder-epochs',str(a.decoder_epochs),
           '--candidate-seed',str(a.candidate_seed), '--threads',str(a.threads),
           '--hidden',str(a.hidden),'--dim',str(a.dim),'--patience',str(a.patience),
           '--lr',str(a.lr),'--weight-decay',str(a.weight_decay),
           '--dropout',str(a.dropout),'--lam',str(a.lam),'--edge-drop',str(a.edge_drop),
           '--feature-drop',str(a.feature_drop),'--easy-groups',str(a.easy_groups),
           '--easy-noise',str(a.easy_noise),'--device',a.device,'--output',str(path),'--execute']
    return cmd


def check_completed(group_dir:Path, a, graph:int, seed:int):
    """True only when all 8 run folders exist and are consistent."""
    import json
    candidates=[]; source = None
    origin_files=('src/models/wbtbgrl.py','revised/run_easy.py','revised/easy_negatives.py',
                  'data/synthetic/generator_easy_project5.py')
    expected_source={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in origin_files}
    for bits, mode in itertools.product(selection(a.variants,VARIANTS), selection(a.modes,MODES)):
        p=group_dir/f'g{graph}_s{seed}_WBT_{bits}_{mode}'
        jf=p/'result.json'; cf=p/'test_scores.csv'; mf=p/'ssl_diagnostic_mode.json'
        if not all(f.is_file() for f in (jf,cf,mf)):
            return False
        j=json.loads(jf.read_text()); meta=json.loads(mf.read_text()); cfg=j['config']
        if cfg.get('graph_seed') != graph or cfg.get('seed')!=seed or meta.get('mode') != mode or meta.get('bits') != bits:
            raise RuntimeError(f'Inconsistent metadata for {p}')
        expected = {'variant':'NWP_NWB' if bits=='000' else 'WP_NWB',
                    'propagation':bool(int(bits[0])), 'augmentation':bool(int(bits[1])), 'loss_weights':bool(int(bits[2])),
                    'encoder_epochs':a.encoder_epochs,'decoder_epochs':a.decoder_epochs,
                    'candidate_seed':a.candidate_seed,'hidden':a.hidden,'dim':a.dim,'device':a.device}
        for k,v in expected.items():
            if cfg.get(k)!=v:
                raise RuntimeError(f'Configuration mismatch for {p}: {k}')
        if j.get('negative_sampling',{}).get('policy')!='matched_easy_stage':
            raise RuntimeError(f'Unexpected negatives in {p}')
        s=meta.get('source_sha256')
        if s!=expected_source:
            raise RuntimeError(f'Source model or dataset changed since run: {p}')
        if source is None: source = s
        if s!=source:
            raise RuntimeError(f'Source differs inside group {group_dir}')
        if not j.get('candidate_sha256'):
            raise RuntimeError(f'Missing candidate fingerprint: {p}')
        candidates.append(j['candidate_sha256'])
    if len(set(candidates)) != 1:
        raise RuntimeError(f'Different test candidates within graph/seed: {group_dir}')
    return True


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graphs',nargs='+',type=int,default=[111,112,113])
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46])
    p.add_argument('--variants',default='000,010')
    p.add_argument('--modes',default=','.join(MODES))
    p.add_argument('--output',default='results/comparacion_ssl_multigrafo_easy')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--execute',action='store_true',help='Required to run (otherwise previews)')
    p.add_argument('--candidate-seed',type=int,default=2026)
    for k,v in [('encoder-epochs',200),('decoder-epochs',100),('threads',2),('hidden',128),('dim',64),('patience',20),('easy-groups',16)]:
        p.add_argument('--'+k,type=int,default=v)
    for k,v in [('lr',.001),('dropout',.1),('weight-decay',1e-5),('lam',.5),('edge-drop',.1),('feature-drop',.1),('easy-noise',.02)]:
        p.add_argument('--'+k,type=float,default=v)
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    opts=jobs(a)
    config=plan(a)
    out=Path(a.output)
    planfile=out/'multigraph_plan.json'
    if planfile.is_file() and json.loads(planfile.read_text()) != config:
        raise RuntimeError(f'Plan/source mismatch in {out}; use a new output folder')
    print(f'{len(opts)} planned trainings = {len(a.graphs)} independent graphs x {len(a.seeds)} training seeds x '
          f'{len(selection(a.variants,VARIANTS))} variants x {len(selection(a.modes,MODES))} SSL modes',flush=True)
    if a.execute:
        out.mkdir(parents=True,exist_ok=True)
        if not planfile.exists():planfile.write_text(json.dumps(config,indent=2,sort_keys=True))
    total=len(a.graphs)*len(a.seeds); completed=0
    for i,(graph,seed) in enumerate(itertools.product(a.graphs,a.seeds),1):
        dest=out/f'g{graph}_s{seed}'
        if check_completed(dest,a,graph,seed):
            completed += 1
            print(f'[{i}/{total}] SKIP verified: graph={graph} seed={seed} (8/8)',flush=True)
            continue
        cmd=run_command(a,graph,seed,dest)
        print(f'[{i}/{total}] {"RUN" if a.execute else "DRY"}: graph={graph} seed={seed}',flush=True)
        if not a.execute:
            print('  '+' '.join(cmd),flush=True)
            continue
        out_logs=out/'launch_logs';out_logs.mkdir(exist_ok=True)
        log=out_logs/f'g{graph}_s{seed}.log'
        env=os.environ.copy();env.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        # Guard log outside group's result directory; original runner manages subgroup logs.
        with log.open('w') as fp:
            proc=subprocess.run(cmd,cwd=str(ROOT),env=env,stdout=fp,stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError(f'Stopped graph={graph} seed={seed}, exit={proc.returncode}; see {log} and {dest}/logs/')
        if not check_completed(dest,a,graph,seed):
            raise RuntimeError(f'Group not complete after successful execution: {dest}')
        completed += 1
        print(f'[{i}/{total}] VERIFIED: graph={graph} seed={seed} (8/8)',flush=True)
    if a.execute:
        print(f'Finished: {completed}/{total} complete groups, {completed*len(selection(a.variants,VARIANTS))*len(selection(a.modes,MODES))}/{len(opts)} verified runs',flush=True)


if __name__=='__main__':main()
