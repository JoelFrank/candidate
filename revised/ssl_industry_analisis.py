"""Aggregate-only private INDUSTRY SSL analysis; red positive/blue etc figures."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS = ('roc_auc','ap','val_auc','ks','overlap','pos_mean','neg_mean')


def ks_score(pos: np.ndarray, neg: np.ndarray) -> float:
    xs=np.sort(np.concatenate([pos,neg]))
    return float(np.max(np.abs(np.searchsorted(np.sort(pos),xs,side='right')/len(pos) -
                                np.searchsorted(np.sort(neg),xs,side='right')/len(neg))))


def hist_overlap(pos: np.ndarray, neg: np.ndarray) -> float:
    bins=np.linspace(0,1,41)
    a,_=np.histogram(pos,bins=bins,density=True)
    b,_=np.histogram(neg,bins=bins,density=True)
    return float((np.minimum(a,b)*np.diff(bins)).sum())


def collect(root:Path,expected_count:int|None=None):
    rows=[]; candidate_hashes={}
    for folder in sorted(root.glob('industry_s*_WBT_*')):
        if not folder.is_dir(): continue
        required=('result.json','ssl_industry_mode.json','test_scores.csv')
        if not all((folder/p).is_file() for p in required):
            raise RuntimeError(f'Experimento incompleto: {folder}')
        res=json.loads((folder/'result.json').read_text())
        meta=json.loads((folder/'ssl_industry_mode.json').read_text())
        cfg=res['config']; m=res['metrics']; scores=pd.read_csv(folder/'test_scores.csv')
        if list(scores.columns)!=['label','score']:
            raise RuntimeError('Detected identifying columns in test_scores; do not export private IDs')
        y=scores.label.to_numpy(dtype=int); p=scores.score.to_numpy(float)
        if set(np.unique(y))!={0,1} or not np.all(np.isfinite(p)) or np.any((p<0)|(p>1)):
            raise RuntimeError(f'Invalid labels/scores {folder}')
        auc=float(roc_auc_score(y,p)); ap=float(average_precision_score(y,p))
        if abs(auc-float(m['ROC-AUC']))>1e-8 or abs(ap-float(m['AP']))>1e-8:
            raise RuntimeError(f'Metric recomputation failed: {folder}')
        if cfg.get('dataset')!='industry' or cfg.get('negative_strategy')!='uniform' or not cfg['variant'].endswith('NWB'):
            raise RuntimeError(f'Not the intended INDUSTRY/NWB run: {folder}')
        bits=meta['bits']; mode=meta['mode']; seed=meta['seed']
        if cfg.get('seed')!=seed or cfg.get('propagation')!=bool(int(bits[0])) or cfg.get('augmentation')!=bool(int(bits[1])) or cfg.get('loss_weights')!=bool(int(bits[2])):
            raise RuntimeError(f'Variant flags mismatch: {folder}')
        fp=res.get('candidate_sha256')
        if not fp: raise RuntimeError(f'Missing candidate hash: {folder}')
        candidate_hashes.setdefault(seed,set()).add(fp)
        pos=p[y==1]; neg=p[y==0]
        rows.append(dict(seed=seed,variant='WBT_'+bits,mode=mode,roc_auc=auc,ap=ap,
                         val_auc=float(res['val_auc']),ks=ks_score(pos,neg),overlap=hist_overlap(pos,neg),
                         pos_mean=float(pos.mean()),neg_mean=float(neg.mean()),
                         positives=int(len(pos)),negatives=int(len(neg)),candidate_sha256=fp,
                         workbook_sha256=meta['workbook_sha256'],folder=str(folder)))
    if not rows: raise RuntimeError('No se encontraron resultados completos')
    df=pd.DataFrame(rows).sort_values(['variant','mode','seed']).reset_index(drop=True)
    if expected_count is not None and len(df)!=expected_count:
        raise RuntimeError(f'Esperadas {expected_count} ejecuciones, encontradas {len(df)}; no generar tablas parciales')
    for seed,values in candidate_hashes.items():
        if len(values)!=1:
            raise RuntimeError(f'Candidatos diferentes entre modos/variantes de la semilla {seed}: {len(values)} hashes')
    if df.groupby(['seed','variant','mode']).size().max()!=1:
        raise RuntimeError('Experiment duplicated')
    if df.workbook_sha256.nunique()!=1:
        raise RuntimeError('Workbook hash mismatch')
    return df


def summarize(df:pd.DataFrame):
    by=df.groupby(['variant','mode'],as_index=False)[list(METRICS)].agg('mean')
    # Each seed is one paired training initialization on the SAME temporal split.
    table=df.pivot_table(index=['seed','variant'],columns='mode',values=['roc_auc','ap','ks','overlap'],aggfunc='first')
    if not {'original','symmetric'}.issubset(df['mode'].unique()):
        raise RuntimeError('Missing original and/or symmetric comparisons')
    delta=table.copy()
    delta=pd.DataFrame({key+'_delta_symmetric_minus_original':table[(key,'symmetric')]-table[(key,'original')]
                        for key in ('roc_auc','ap','ks','overlap')}).reset_index()
    return by,delta


def draw_hist(d:pd.DataFrame,output:Path,title:str):
    y=d.label.to_numpy(int); p=d.score.to_numpy(float)
    bins=np.linspace(0,1,31)
    fig,ax=plt.subplots(figsize=(8.3,4.6))
    ax.hist(p[y==1],bins=bins,color='royalblue',alpha=.62,density=True,label='Positive links')
    ax.hist(p[y==0],bins=bins,color='firebrick',alpha=.62,density=True,label='Negative links')
    ax.set(xlim=(0,1),xlabel='Predicted link score',ylabel='Density',title=title)
    ax.grid(axis='y',alpha=.20);ax.legend();fig.tight_layout()
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180);plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True); p.add_argument('--output',required=True)
    p.add_argument('--expected-count',type=int,default=20)
    a=p.parse_args(argv)
    inp=Path(a.input); out=Path(a.output)
    df=collect(inp,a.expected_count)
    out.mkdir(parents=True,exist_ok=True)
    view=df.drop(columns=['folder','workbook_sha256'])
    view.to_csv(out/'industry_by_seed.csv',index=False)
    overview,paired=summarize(df)
    overview.to_csv(out/'industry_overview.csv',index=False)
    paired.to_csv(out/'industry_paired_ssl.csv',index=False)
    # Compare WBT_010 and WBT_000 for each seed and SSL objective.
    pivot=df.pivot_table(index=['seed','mode'],columns='variant',values=['roc_auc','ap'],aggfunc='first')
    if {'WBT_000','WBT_010'}.issubset(df.variant.unique()):
        aug=pd.DataFrame({metric+'_delta_010_minus_000':pivot[(metric,'WBT_010')]-pivot[(metric,'WBT_000')]
                          for metric in ('roc_auc','ap')}).reset_index()
        aug.to_csv(out/'industry_paired_augmentation.csv',index=False)
    selections=[]
    for (variant,mode),subset in df.groupby(['variant','mode']):
        best=subset.sort_values(['val_auc','seed'],ascending=[False,True]).iloc[0]
        scores=pd.read_csv(Path(best.folder)/'test_scores.csv')
        draw_hist(scores,out/'figures'/f'{variant}_{mode}_selected_validation.png',
                  f'INDUSTRY {variant} {mode} | seed {int(best.seed)} selected by val AUC')
        selections.append(dict(variant=variant,mode=mode,seed=int(best.seed),val_auc=float(best.val_auc),test_auc=float(best.roc_auc)))
    (out/'selected_by_validation.json').write_text(json.dumps(selections,indent=2))
    integrity={'runs':int(len(df)),'candidate_sha256_per_seed':dict(df.groupby('seed')['candidate_sha256'].first()),
               'privacy':'scores exported contain labels and scores only; raw identifiers excluded',
               'test_usage':'diagnostic; validation for decoder checkpoint selection and figure selection; test not used to tune model',
               'inference_limits':'one private dataset temporal split; training seeds are not independent datasets; uniform negatives may include future positives'}
    (out/'integrity.json').write_text(json.dumps(integrity,indent=2))
    print('Analysis complete. Mean test AUC/AP and paired deltas:',flush=True)
    print(overview[['variant','mode','roc_auc','ap','ks','overlap']].to_string(index=False),flush=True)
    print(f'Files: {out}',flush=True)

if __name__=='__main__':
    main()
