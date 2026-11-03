"""Integrity-audited analysis for 180 original-BiSBM SSL comparisons.

Three independent graph realizations. Conditions and training seeds are
repeated measures within graph; DO NOT count 45 as independent datasets.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONDITIONS=('informative','orthogonal','anti_correlated')
MODES=('original','symmetric')
BITS=('000','010')


def ks_stat(pos, neg):
    x = np.sort(np.r_[pos, neg])
    return float(np.max(np.abs(np.searchsorted(np.sort(pos),x,side='right')/len(pos)-
                               np.searchsorted(np.sort(neg),x,side='right')/len(neg))))


def overlap(pos, neg):
    edges = np.linspace(0,1,31)
    a,_ = np.histogram(pos,bins=edges,density=True)
    b,_ = np.histogram(neg,bins=edges,density=True)
    return float(np.sum(np.minimum(a,b) * np.diff(edges)))


def sign_test_p(values):
    vals = np.asarray(values,dtype=float)
    vals = vals[np.isfinite(vals) & (np.abs(vals)>1e-12)]
    n = len(vals)
    if not n:
        return 1.0
    k = int((vals > 0).sum())
    return min(1.0, 2.0 * sum(math.comb(n,i) for i in range(min(k,n-k)+1)) / (2**n))


def score_signature(s):
    return hashlib.sha256(s[['u','v','label']].to_csv(index=False).encode()).hexdigest()


def read_study(folder):
    plan = json.loads((folder/'plan.json').read_text())
    p = plan['params']
    from itertools import product
    graphs = p['graphs']; conditions=p['conditions'].split(',');seeds=p['seeds']
    bits=p['variants'].split(',');modes=p['modes'].split(',')
    required=list(product(graphs,conditions,seeds,bits,modes))
    if not required or len(required)!=plan['planned_runs']:
        raise RuntimeError('Plan has inconsistent run count')
    rows=[]
    fingerprints={}
    for g,c,s,b,m in required:
        path=folder/f'g{g}_{c}_s{s}_WBT_{b}_{m}'
        if not all((path/x).is_file() for x in ('result.json','test_scores.csv','ssl_bisbm_mode.json','data_audit.json')):
            raise RuntimeError(f'Missing completed run: {path}')
        j=json.loads((path/'result.json').read_text())
        meta=json.loads((path/'ssl_bisbm_mode.json').read_text())
        cfg=j['config']
        checks={
            'dataset':'synthetic','model':'WBTBGRL','graph_seed':g,'condition':c,'seed':s,
            'variant':'NWP_NWB' if b=='000' else 'WP_NWB',
            'propagation':bool(int(b[0])),'augmentation':bool(int(b[1])),
            'loss_weights':bool(int(b[2])),'negative_strategy':'matched_synthetic',
            'encoder_epochs':p['encoder_epochs'],'decoder_epochs':p['decoder_epochs'],
            'candidate_seed':p['candidate_seed'],
        }
        for key,val in checks.items():
            if cfg.get(key)!=val:
                raise RuntimeError(f'Configuration mismatch for {key}: {path}')
        if meta.get('mode')!=m or meta.get('bits')!=b or meta.get('source_sha256')!=plan['source_sha256']:
            raise RuntimeError(f'Mismatched mode or source sha: {path}')
        if j['negative_sampling']['policy']!='matched_synthetic':
            raise RuntimeError(f'Wrong negative sampling policy: {path}')
        for stage in ('train','val','test'):
            strata=j['negative_sampling']['strata'][stage]
            if strata['positive']!=strata['negative']:
                raise RuntimeError(f'Unmatched {stage} strata: {path}')
        bce_weight=j.get('decoder_diagnostics',{}).get('mean_positive_bce_weight')
        if bce_weight is None or abs(float(bce_weight)-1)>1e-6:
            raise RuntimeError(f'Non-NWB decoder: {path}')
        scores=pd.read_csv(path/'test_scores.csv',dtype={'u':'str','v':'str'})
        if not set(('u','v','label','score')).issubset(scores.columns):
            raise RuntimeError(f'Missing test score columns in {path}')
        if not scores['label'].isin([0,1]).all() or not np.isfinite(scores['score']).all():
            raise RuntimeError(f'Invalid scores or labels in {path}')
        if len(scores)!=640 or (scores.label==1).sum()!=320 or (scores.label==0).sum()!=320:
            raise RuntimeError(f'Wrong 320+320 test balance: {path}')
        fp=(j.get('candidate_sha256'),score_signature(scores))
        if not fp[0]:
            raise RuntimeError(f'Missing test candidate hash {path}')
        if g in fingerprints and fingerprints[g]!=fp:
            raise RuntimeError(f'Test candidate mismatch across arms / conditions / seeds for graph {g}: {path}')
        fingerprints[g]=fp
        pos=scores.loc[scores.label==1,'score'].to_numpy(float)
        neg=scores.loc[scores.label==0,'score'].to_numpy(float)
        rows.append({'graph':g,'condition':c,'seed':s,'bits':b,'mode':m,
                     'auc':float(j['metrics']['ROC-AUC']),'ap':float(j['metrics']['AP']),
                     'val_auc':float(j['val_auc']),'f1':float(j['metrics']['F1']),
                     'ks':ks_stat(pos,neg),'overlap':overlap(pos,neg),
                     'positive_score_mean':float(pos.mean()),'negative_score_mean':float(neg.mean()),
                     'last_ssl_loss':float(j['ssl_losses'][-1]),'candidate_sha256':fp[0],
                     'path':str(path)})
    df=pd.DataFrame(rows).sort_values(['graph','condition','seed','bits','mode']).reset_index(drop=True)
    return df,plan


def tables(df):
    keys=['graph','condition','seed','bits']
    base=df[df['mode']=='original'][keys+['auc','ap','ks','overlap']].rename(columns={x:x+'_original' for x in ('auc','ap','ks','overlap')})
    paired=df.merge(base,on=keys,validate='many_to_one')
    for metric in ('auc','ap','ks','overlap'):
        paired['delta_'+metric]=paired[metric]-paired[metric+'_original']
    graph_avg=df.groupby(['graph','condition','bits','mode'],as_index=False)[['auc','ap','val_auc','ks','overlap','f1']].mean()
    graph_delta=paired[paired['mode']=='symmetric'].groupby(['graph','condition','bits'],as_index=False)[['delta_auc','delta_ap','delta_ks','delta_overlap']].mean()
    summary=graph_avg.groupby(['condition','bits','mode'],as_index=False)[['auc','ap','val_auc','ks','overlap','f1']].agg(['mean','std'])
    summary.columns=['condition','bits','mode'] + [f'{x}_{stat}' for x in ('auc','ap','val_auc','ks','overlap','f1') for stat in ('mean','std')]
    contrasts=[]
    for (cond,b),grp in graph_delta.groupby(['condition','bits']):
        for metric in ('auc','ap','ks','overlap'):
            arr=grp.sort_values('graph')['delta_'+metric].to_numpy(float)
            if len(arr)!=3:
                raise RuntimeError('Required three independent graph realizations')
            contrasts.append({'condition':cond,'bits':b,'metric':metric,'independent_graphs':len(arr),
                              'delta_graph_mean':float(arr.mean()),
                              'delta_graph_sd':float(arr.std(ddof=1)),
                              'graph_deltas':';'.join(f'{x:+.6f}' for x in arr),
                              'sign_test_p_2sided':sign_test_p(arr),
                              'improved_graphs':int(np.sum(arr<0)) if metric=='overlap' else int(np.sum(arr>0))})
    # WBT_010 versus WBT_000 for identical condition, graph, seed and SSL mode.
    vk=['graph','condition','seed','mode']
    comparison=df[df.bits=='010'].merge(df[df.bits=='000'],on=vk,suffixes=('_010','_000'),validate='one_to_one')
    for metric in ('auc','ap','ks','overlap'):
        comparison['delta_'+metric+'_010_minus_000']=comparison[metric+'_010']-comparison[metric+'_000']
    variant_graph=comparison.groupby(['graph','condition','mode'],as_index=False)[[x for x in comparison.columns if x.startswith('delta_')]].mean()
    return paired,graph_avg,graph_delta,summary,pd.DataFrame(contrasts),comparison,variant_graph


def figures(df,root,selected_seed=42):
    root.mkdir(parents=True,exist_ok=True)
    subset=df[df.seed==selected_seed]
    for (g,c,b),grp in subset.groupby(['graph','condition','bits']):
        if set(grp['mode']) != set(MODES):
            continue
        fig,ax=plt.subplots(1,2,figsize=(12,4.5),sharex=True,sharey=True)
        for ix,mode in enumerate(MODES):
            row=grp.loc[grp['mode']==mode].iloc[0]
            scores=pd.read_csv(Path(row['path'])/'test_scores.csv')
            pos=scores.loc[scores.label==1,'score'].to_numpy(float)
            neg=scores.loc[scores.label==0,'score'].to_numpy(float)
            bins=np.linspace(0,1,31)
            ax[ix].hist(pos,bins=bins,color='tab:blue',alpha=.58,density=True,label='Positivos (320)')
            ax[ix].hist(neg,bins=bins,color='tab:red',alpha=.58,density=True,label='Negativos (320)')
            ax[ix].set(title=f'{mode} | ROC-AUC {row["auc"]:.3f} | AP {row["ap"]:.3f}',xlabel='Score',xlim=(0,1))
            ax[ix].grid(alpha=.15)
            ax[ix].legend(fontsize=8)
        ax[0].set_ylabel('Densidad')
        fig.suptitle(f'BiSBM ORIGINAL | graph {g} | {c} | WBT_{b} | seed {selected_seed}')
        fig.tight_layout()
        fig.savefig(root/f'g{g}_{c}_WBT_{b}_seed{selected_seed}_original_vs_simetrica.png',dpi=170)
        plt.close(fig)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='results/ssl_bisbm_original_multicondicion')
    p.add_argument('--output',default='results/analisis_ssl_bisbm_original')
    p.add_argument('--figure-seed',type=int,default=42)
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    folder=Path(a.input)
    df,plan=read_study(folder)
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    paired,graph_avg,graph_delta,summary,contrasts,variant,variant_graph=tables(df)
    for fname,frame in [('ejecuciones_completas.csv',df),('diferencias_pareadas.csv',paired),
                        ('medias_por_grafo.csv',graph_avg),('deltas_por_grafo.csv',graph_delta),
                        ('resumen_por_grafos.csv',summary),('contrastes_simetria.csv',contrasts),
                        ('WBT010_vs_WBT000.csv',variant),('WBT010_vs_WBT000_por_grafo.csv',variant_graph)]:
        frame.to_csv(out/fname,index=False)
    figures(df,out/'figuras',a.figure_seed)
    meta={'complete':len(df)==plan['planned_runs'],'runs':len(df),'graphs':list(plan['params']['graphs']),
          'conditions':plan['params']['conditions'].split(','),'seeds':list(plan['params']['seeds']),
          'all_candidate_hashes_equal_within_graph':True,
          'candidate_policy':'matched_synthetic, positives and negatives equal within OO/ON/NO/NN',
          'decoder':'NWB; training loss weights = 1',
          'independent_statistical_units':'3 graph realizations, NOT 15 graph-seed pairs or 45 condition-seed pairs',
          'selection':'validation only, by the original project5 runner',
          'warning':'Study is exploratory; three graphs give a minimum two-sided sign-test p=0.25. '
                    'Original BiSBM may be intrinsically hard in inductive groups; score overlap alone is not the objective.',
          'source_sha256':plan['source_sha256']}
    (out/'verificacion_integridad.json').write_text(json.dumps(meta,indent=2))
    print(f'COMPLETE: {len(df)}/{plan["planned_runs"]} verified runs; identical paired test candidates per graph.',flush=True)
    print(summary[['condition','bits','mode','auc_mean','ap_mean','overlap_mean']].to_string(index=False),flush=True)
    print(f'Reports and figures: {out}',flush=True)

if __name__=='__main__':
    main()
