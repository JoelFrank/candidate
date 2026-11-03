"""Resumable, source-locked batch for P-only GCN weight sensitivity (45 runs).

All modes use original WBT-BGRL with P=1, A=0, S=0, NWB decoder.
This is diagnosis of extreme weights, not model selection or a new paper method.
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from revised.p_sensitivity_run import MODES, RELEVANT


def make_command(a, graph, seed, mode, output):
    return [sys.executable,str(ROOT/'revised/p_sensitivity_run.py'),
        '--dataset','synthetic','--model','WBTBGRL','--graph-seed',str(graph),
        '--seed',str(seed),'--condition','informative','--candidate-seed',str(a.candidate_seed),
        '--variant','WP_NWB','--propagation','1','--augmentation','0','--loss-weights','0',
        '--negative-strategy','matched_synthetic','--propagation-mode',mode,
        '--clip-quantile',str(a.clip_quantile),'--encoder-epochs',str(a.encoder_epochs),
        '--decoder-epochs',str(a.decoder_epochs),'--patience',str(a.patience),
        '--hidden',str(a.hidden),'--dim',str(a.dim),'--threads',str(a.threads),
        '--lr',str(a.lr),'--weight-decay',str(a.weight_decay),
        '--dropout',str(a.dropout),'--lam',str(a.lam),
        '--edge-drop',str(a.edge_drop),'--feature-drop',str(a.feature_drop),
        '--device',a.device,'--output',str(output)]


def validate(directory, graph, seed, mode, a):
    result=directory/'result.json';scores=directory/'test_scores.csv';meta=directory/'propagation_sensitivity.json'
    if not result.exists() and not scores.exists() and not meta.exists():
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(f'Incomplete run with log/partials: {directory}; inspect before resuming')
        return False
    if not all(p.is_file() for p in (result,scores,meta)):
        raise RuntimeError(f'Partial output: {directory}')
    r=json.loads(result.read_text()); m=json.loads(meta.read_text());c=r['config']
    expected={'dataset':'synthetic','model':'WBTBGRL','graph_seed':graph,'seed':seed,'condition':'informative',
        'candidate_seed':a.candidate_seed,'variant':'WP_NWB','propagation':True,'augmentation':False,'loss_weights':False,
        'negative_strategy':'matched_synthetic','propagation_mode':mode}
    for k,v in expected.items():
        if c.get(k)!=v:raise RuntimeError(f'Mismatched configuration {directory} {k}: {c.get(k)} != {v}')
    if m.get('mode')!=mode or r.get('propagation_sensitivity')!=m:
        raise RuntimeError('Metadata mismatch')
    if m.get('source_sha256')!=dict((name,hashlib.sha256((ROOT/name).read_bytes()).hexdigest()) for name in RELEVANT):
        raise RuntimeError('Different code revision: do not merge results')
    for k in ['encoder_epochs','decoder_epochs','patience','hidden','dim','threads','lr','weight_decay','dropout','lam','edge_drop','feature_drop','clip_quantile']:
        if c[k]!=getattr(a,k):raise RuntimeError(f'Parameter mismatch: {k}')
    if r.get('negative_sampling',{}).get('policy')!='matched_synthetic':
        raise RuntimeError('Incorrect negative sampler')
    if not all(z['positive']==z['negative'] for z in r['negative_sampling']['strata'].values()):
        raise RuntimeError('Negative strata are not matched')
    if not r.get('candidate_sha256') or not (0 <=r['metrics']['ROC-AUC']<=1):
        raise RuntimeError('Missing candidate hash or invalid result')
    return True



def archive_known_launcher_failure(target: Path, batch: Path) -> bool:
    """Preserve and recover only the *known* log-only failure from launcher v1.

    The old launcher created ``target/run.log`` before starting its child;
    the child's output-safety guard then rejected the nonempty directory.
    Never discard a partial training run or an unrelated failure log.
    """
    if not target.is_dir():
        return False
    entries=list(target.iterdir())
    if len(entries)!=1 or entries[0].name!='run.log' or not entries[0].is_file():
        return False
    log=entries[0]
    if 'Nonempty output directory:' not in log.read_text(errors='replace'):
        return False
    old_logs=batch/'_archived_launcher_errors'
    old_logs.mkdir(parents=True,exist_ok=True)
    destination=old_logs/f'{target.name}.run.log'
    if destination.exists():
        raise FileExistsError(f'Cannot overwrite previous failure log: {destination}')
    log.rename(destination)
    target.rmdir()
    print(f'ARCHIVED prior log-only launcher failure: {destination}',flush=True)
    return True


def launch_with_external_log(cmd, root: Path, batch: Path, target: Path) -> int:
    """Start subprocess with a log OUTSIDE the model output directory.

    The original execution code requires the run output directory to be
    absent or empty. On success attach the log after the child terminates;
    on failure preserve it outside the run folder without discarding data.
    """
    log_dir=batch/'_launcher_logs'
    log_dir.mkdir(parents=True,exist_ok=True)
    logfile=log_dir/f'{target.name}.log'
    if logfile.exists():
        raise FileExistsError(f'Existing launcher log: {logfile}; inspect before retrying')
    env=os.environ.copy();env.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    with logfile.open('w') as handle:
        rc=subprocess.run(cmd,cwd=str(root),env=env,stdout=handle,stderr=subprocess.STDOUT,check=False).returncode
    if rc==0:
        if not target.is_dir():
            raise RuntimeError(f'Child reported success but produced no output directory: {target}; log: {logfile}')
        final_log=target/'run.log'
        if final_log.exists():
            raise FileExistsError(f'Unexpected existing child log: {final_log}')
        logfile.rename(final_log)
    return rc

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graphs',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46])
    p.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--output',default='results/sensibilidad_P_BiSBM')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--delete-checkpoints',action='store_true')
    p.add_argument('--candidate-seed',type=int,default=2026)
    p.add_argument('--clip-quantile',type=float,default=.99)
    for name,d in [('encoder-epochs',200),('decoder-epochs',100),('patience',20),('hidden',128),('dim',64),('threads',2)]:p.add_argument('--'+name,type=int,default=d)
    for name,d in [('lr',.001),('weight-decay',1e-5),('dropout',.1),('lam',.5),('edge-drop',.1),('feature-drop',.1)]:p.add_argument('--'+name,type=float,default=d)
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    if len(set(a.graphs))!=len(a.graphs) or len(set(a.seeds))!=len(a.seeds) or len(set(a.modes))!=len(a.modes):raise ValueError('Duplicate graphs/seeds/modes')
    if not 0<a.clip_quantile<1:raise ValueError('Invalid quantile')
    if any(not (ROOT/name).is_file() for name in RELEVANT):raise FileNotFoundError('Required source code missing')
    plan={'graphs':a.graphs,'seeds':a.seeds,'modes':a.modes,'condition':'informative','candidate_seed':a.candidate_seed,
        'clip_quantile':a.clip_quantile,'device':a.device,'experiment':'WBT_100, P-only',
        'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in RELEVANT}}
    plan.update({k:getattr(a,k) for k in ['encoder_epochs','decoder_epochs','patience','hidden','dim','threads','lr','weight_decay','dropout','lam','edge_drop','feature_drop']})
    out=Path(a.output); pfile=out/'sensitivity_plan.json'
    if pfile.exists() and json.loads(pfile.read_text())!=plan:
        raise RuntimeError(f'Existing plan differs: choose another --output folder ({pfile})')
    jobs=list(itertools.product(a.graphs,a.seeds,a.modes));print(f'{len(jobs)} P-only runs planned',flush=True)
    if a.execute:
        out.mkdir(parents=True,exist_ok=True)
        if not pfile.exists():pfile.write_text(json.dumps(plan,indent=2))
    completed=0
    for i,(g,s,m) in enumerate(jobs,1):
        target=out/f'g{g}_s{s}_informative_P_{m}'
        if a.execute:
            archive_known_launcher_failure(target,out)
        if validate(target,g,s,m,a):
            completed+=1
            if a.execute and a.delete_checkpoints and (target/'checkpoint.pt').exists():
                (target/'checkpoint.pt').unlink()
            print(f'[{i}/{len(jobs)}] SKIP verified {target}',flush=True)
            continue
        cmd=make_command(a,g,s,m,target)
        print(f'[{i}/{len(jobs)}] {"RUN" if a.execute else "DRY"} {target}',flush=True)
        if not a.execute:
            print(' '.join(cmd));continue
        exitcode=launch_with_external_log(cmd,ROOT,out,target)
        if exitcode!=0:
            raise RuntimeError(f'Run failed ({exitcode}), see {out}/_launcher_logs/{target.name}.log')
        validate(target,g,s,m,a)
        if a.delete_checkpoints and (target/'checkpoint.pt').exists():
            (target/'checkpoint.pt').unlink()
        completed+=1
    print(f'Completed and verified: {completed}/{len(jobs)}',flush=True)


if __name__=='__main__':main()
