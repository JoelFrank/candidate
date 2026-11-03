"""Integrity-checked multigraph SSL evaluation; graph (not training seed) is the independent unit.

120 runs. Test is for reporting comparisons, never for hyperparameter selection.
Decoders choose checkpoints by validation inside existing revised/run_easy.py.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
MODES = ('original','symmetric','target_corr','symmetric_target_corr')
BITS = ('000','010')
METRICS = ('auc','ap','ks','overlap','erank_U','erank_V','val_auc')


def overlap(pos, neg):
    bins=np.linspace(0,1,31)
    h1,_=np.histogram(pos,bins=bins,density=True)
    h0,_=np.histogram(neg,bins=bins,density=True)
    return float(np.sum(np.minimum(h1,h0)*np.diff(bins)))


def ks(pos, neg):
    xs=np.sort(np.concatenate([pos,neg]))
    return float(np.max(np.abs(np.searchsorted(np.sort(pos),xs,side='right')/len(pos)-
                               np.searchsorted(np.sort(neg),xs,side='right')/len(neg))))


def graph_sign_p_two_sided(deltas):
    """Exact two-sided sign test over nonzero *independent graph* differences."""
    values=np.asarray(deltas,dtype=float)
    values=values[np.isfinite(values) & (np.abs(values)>1e-12)]
    n=len(values)
    if n==0:return float('nan')
    smaller=min(int(np.sum(values>0)),int(np.sum(values<0)))
    return float(min(1.0,2.0*sum(math.comb(n,k) for k in range(smaller+1))/2**n))


def read_runs(root:Path, expected_graphs=(111,112,113), expected_seeds=(42,43,44,45,46), rank=True):
    plan_file=root/'multigraph_plan.json'
    if not plan_file.is_file():raise FileNotFoundError('Missing multigraph_plan.json')
    plan=json.loads(plan_file.read_text())
    expected=len(expected_graphs)*len(expected_seeds)*len(BITS)*len(MODES)
    rows=[]; raw={}; sigs={}; configs={}
    calc_ranks=None
    if rank:
        from revised.ssl_factorial_analysis import embedding_ranks
        calc_ranks=embedding_ranks
    for g,s,b,m in itertools.product(expected_graphs,expected_seeds,BITS,MODES):
        path=root/f'g{g}_s{s}'/f'g{g}_s{s}_WBT_{b}_{m}'
        jfile=path/'result.json';csvfile=path/'test_scores.csv';meta=path/'ssl_diagnostic_mode.json'
        if not all(p.is_file() for p in (jfile,csvfile,meta)):
            raise RuntimeError(f'Missing/partial expected run: {path}')
        j=json.loads(jfile.read_text()); v=j['config']; tag=json.loads(meta.read_text())
        if any((v.get('graph_seed')!=g,v.get('seed')!=s,tag.get('bits')!=b,tag.get('mode')!=m)):
            raise RuntimeError(f'Mismatched run identity: {path}')
        # Verify the model settings; do not silently mix different protocols.
        p=plan['param']
        for k in ('encoder_epochs','decoder_epochs','candidate_seed','hidden','dim','patience','easy_groups','threads',
                  'lr','dropout','weight_decay','lam','edge_drop','feature_drop','easy_noise','device'):
            if v.get(k)!=p.get(k):
                raise RuntimeError(f'{path} has {k}={v.get(k)!r}; plan has {p.get(k)!r}')
        expected_sources={x:plan['source_sha256'][x] for x in ('src/models/wbtbgrl.py','revised/run_easy.py','revised/easy_negatives.py','data/synthetic/generator_easy_project5.py')}
        if tag.get('source_sha256') != expected_sources:
            raise RuntimeError(f'Changed source code during study: {path}')
        if (v.get('propagation'),v.get('augmentation'),v.get('loss_weights')) != tuple(bool(int(x)) for x in b):
            raise RuntimeError(f'Wrong P/A/S bits: {path}')
        if v.get('variant') != ('NWP_NWB' if b=='000' else 'WP_NWB'):
            raise RuntimeError(f'Wrong variant switch: {path}')
        neg=j.get('negative_sampling',{})
        if neg.get('policy') != 'matched_easy_stage':
            raise RuntimeError(f'Wrong negative sampler: {path}')
        for stage in ('train','validation','test'):
            strata=neg.get('strata',{}).get(stage,{})
            if strata.get('positive')!=strata.get('negative'):
                raise RuntimeError(f'Unmatched candidates in {stage}: {path}')
        # Exact fingerprint equality across all 8 runs of same graph and training seed.
        candidate=j.get('candidate_sha256','')
        if not candidate:raise RuntimeError(f'Missing candidate hash: {path}')
        sc=pd.read_csv(csvfile)
        if not {'u','v','label','score'}.issubset(sc.columns):raise RuntimeError(f'Missing scores: {path}')
        if len(sc)!=720 or int((sc.label==1).sum())!=360 or int((sc.label==0).sum())!=360:
            raise RuntimeError(f'Unexpected test candidate balance: {path}')
        if sc[['u','v','label']].duplicated().any():
            raise RuntimeError(f'Duplicate exact candidate(s): {path}')
        if not np.isfinite(sc.score.to_numpy(float)).all() or ((sc.score < 0)|(sc.score>1)).any():
            raise RuntimeError(f'Invalid score(s): {path}')
        key=(g,s)
        signature=hashlib.sha256(sc[['u','v','label']].to_csv(index=False).encode()).hexdigest()
        if key in sigs and sigs[key]!=(candidate,signature):
            raise RuntimeError(f'Candidate mismatch within graph {g}/seed {s}, invalid paired comparison: {path}')
        sigs[key]=(candidate,signature)
        src=json.dumps(tag.get('source_sha256',{}),sort_keys=True)
        if key in configs and configs[key]!=src:
            raise RuntimeError(f'Mixed model source hash: {path}')
        configs[key]=src
        positives=sc.loc[sc.label==1,'score'].to_numpy(float)
        negatives=sc.loc[sc.label==0,'score'].to_numpy(float)
        cp=path/'checkpoint.pt'
        u_rank=v_rank=float('nan')
        if calc_ranks and cp.is_file():
            jj=dict(j);jj['checkpoint_path']=str(cp)
            u_rank,v_rank=calc_ranks(jj)
        rows.append(dict(graph=g,seed=s,variant=f'WBT_{b}',bits=b,mode=m,
            auc=float(j['metrics']['ROC-AUC']),ap=float(j['metrics']['AP']),
            val_auc=float(j['val_auc']),f1=float(j['metrics']['F1']),
            ks=ks(positives,negatives),overlap=overlap(positives,negatives),
            erank_U=u_rank,erank_V=v_rank,pos_mean=positives.mean(),neg_mean=negatives.mean(),
            last_ssl_loss=float(j['ssl_losses'][-1]),
            candidate_hash=candidate,path=str(path)))
        if s==expected_seeds[0]:raw[(g,b,m)]=(positives,negatives)
    df=pd.DataFrame(rows)
    if any(grp.candidate_hash.nunique()!=1 for _,grp in df.groupby('graph')):
        raise RuntimeError('Different test candidate fingerprints between training seeds of same graph')
    if len(df)!=expected or df[['graph','seed','bits','mode']].duplicated().any():
        raise RuntimeError(f'Expected {expected} distinct runs, got {len(df)}')
    return df,raw,plan


def calculate_tables(df):
    metric_cols=['auc','ap','val_auc','ks','overlap','erank_U','erank_V','pos_mean','neg_mean']
    base=df.loc[df['mode']=='original',['graph','seed','bits']+metric_cols]
    base=base.rename(columns={c:c+'_original' for c in metric_cols})
    pairs=df.merge(base,on=['graph','seed','bits'],how='left',validate='many_to_one')
    for c in metric_cols:
        pairs['delta_'+c]=pairs[c]-pairs[c+'_original']
    graph=df.groupby(['graph','bits','mode'],as_index=False)[metric_cols].mean()
    graph_deltas=pairs.groupby(['graph','bits','mode'],as_index=False)[['delta_'+c for c in metric_cols]].mean()
    summary=graph.groupby(['bits','mode'],as_index=False)[metric_cols].agg(['mean','std'])
    summary.columns=['bits','mode'] + [f'{metric}_{stat}' for metric in metric_cols for stat in ('mean','std')]
    # Prespecified paired original vs symmetric within each graph and each variant.
    compare=[]
    for bits in BITS:
        for mode in ('symmetric','target_corr','symmetric_target_corr'):
            sub=graph_deltas[(graph_deltas.bits==bits)&(graph_deltas['mode']==mode)].sort_values('graph')
            if len(sub)!=3:raise RuntimeError('Expected three independent graph differences')
            for metric in ('auc','ap','ks','overlap','erank_U','erank_V'):
                vals=sub['delta_'+metric].to_numpy(float)
                valid=vals[np.isfinite(vals)]
                compare.append(dict(bits=bits,mode=mode,metric=metric,
                    independent_graphs=len(valid),delta_graph_mean=float(np.mean(valid)) if len(valid) else float('nan'),
                    delta_graph_std=float(np.std(valid,ddof=1)) if len(valid)>1 else float('nan'),
                    graph_deltas=';'.join(f'{x:+.6f}' for x in vals),
                    improved_graphs=int((valid>0).sum()) if metric not in ('overlap',) else int((valid<0).sum()),
                    sign_p_two_sided=graph_sign_p_two_sided(valid)))
    contrasts=[]
    for (graph_id,bits),sub in graph.groupby(['graph','bits']):
        values=sub.set_index('mode')
        for metric in ('auc','ap','ks','overlap','erank_U','erank_V'):
            orig=values.loc['original',metric]
            sym=values.loc['symmetric',metric]
            target=values.loc['target_corr',metric]
            both=values.loc['symmetric_target_corr',metric]
            contrasts.append(dict(graph=graph_id,bits=bits,metric=metric,
                effect_symmetry=((sym-orig)+(both-target))/2,
                effect_target=((target-orig)+(both-sym))/2,
                interaction=(both-sym-target+orig)))
    interactions=pd.DataFrame(contrasts)
    # WBT_010 vs WBT_000 within same graph, seed and SSL mode.
    by_000=df[df.bits=='000'][['graph','seed','mode']+metric_cols]
    by_010=df[df.bits=='010'][['graph','seed','mode']+metric_cols]
    combined=by_010.merge(by_000,on=['graph','seed','mode'],suffixes=('_010','_000'),validate='one_to_one')
    for c in metric_cols:
        combined['delta_'+c+'_010_minus_000']=combined[c+'_010']-combined[c+'_000']
    return pairs,graph,graph_deltas,summary,pd.DataFrame(compare),interactions,combined


def plot_hist(pos,neg,path,title):
    fig,ax=plt.subplots(figsize=(8.5,4.6)); bins=np.linspace(0,1,31)
    ax.hist(pos,bins=bins,alpha=.58,color='tab:blue',label=f'Positive links ({len(pos)})',density=True)
    ax.hist(neg,bins=bins,alpha=.58,color='tab:red',label=f'Negative links ({len(neg)})',density=True)
    ax.set(xlabel='Decoder score',ylabel='Density',title=title,xlim=(0,1))
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(path,dpi=170);plt.close(fig)


def figures(raw,output):
    output.mkdir(parents=True,exist_ok=True)
    for (g,b,m),(pos,neg) in raw.items():
        from sklearn.metrics import roc_auc_score
        yy=np.r_[np.ones(len(pos)),np.zeros(len(neg))];score=np.r_[pos,neg]
        auc=roc_auc_score(yy,score)
        plot_hist(pos,neg,output/f'g{g}_seed42_WBT_{b}_{m}.png',
                  f'BiSBM-EASY graph {g} | WBT_{b} | {m} | AUC={auc:.4f}')
    for g in (111,112,113):
        fig,axes=plt.subplots(2,4,figsize=(16.4,7.8),sharex=True)
        for i,b in enumerate(BITS):
            for j,m in enumerate(MODES):
                pos,neg=raw[(g,b,m)];ax=axes[i,j]
                ax.hist(pos,bins=np.linspace(0,1,31),color='tab:blue',alpha=.55,density=True)
                ax.hist(neg,bins=np.linspace(0,1,31),color='tab:red',alpha=.55,density=True)
                ax.set(xlim=(0,1),title=f'WBT_{b} {m}',xlabel='Score')
                if j==0:ax.set_ylabel('Density')
        fig.suptitle(f'Graph {g}, training seed 42 — blue positive / red negative',fontsize=14)
        fig.tight_layout();fig.savefig(output/f'g{g}_panel_8_configuraciones.png',dpi=170);plt.close(fig)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='results/comparacion_ssl_multigrafo_easy')
    p.add_argument('--output',default='results/analisis_ssl_multigrafo_easy')
    p.add_argument('--skip-ranks',action='store_true',help='Avoid loading checkpoints; rank metrics will be NaN')
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    inputdir=Path(a.input);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    plans=json.loads((inputdir/'multigraph_plan.json').read_text())
    params=plans['param']
    if list(params['graphs'])!=[111,112,113] or list(params['seeds'])!=[42,43,44,45,46] or params['variants']!='000,010' or params['modes']!=','.join(MODES):
        raise RuntimeError('This analysis expects the prescribed full 120-run design; do not mix pilots')
    df,raw,plan=read_runs(inputdir,rank=not a.skip_ranks)
    paired,graph,gd,summary,contrast,interactions,variant=calculate_tables(df)
    for name,data in (
        ('runs_120.csv',df),('diferencias_pareadas_120.csv',paired),
        ('medias_por_grafo.csv',graph),('deltas_por_grafo.csv',gd),
        ('resumen_por_grafos.csv',summary),('contrastes_vs_original.csv',contrast),
        ('efectos_factoriales_por_grafo.csv',interactions),('WBT010_vs_WBT000.csv',variant)):
        data.to_csv(out/name,index=False)
    figures(raw,out/'figuras')
    complete={'runs':len(df),'graphs':sorted(df.graph.unique().tolist()),
              'seeds':sorted(df.seed.unique().tolist()),'train_seeds_are_not_independent_graphs':True,
              'same_test_candidates_verified_within_each_graph_seed':True,
              'negative_policy':'matched_easy_stage',
              'ranks_available':int(df.erank_U.notna().sum()),
              'primary_comparisons':'original vs symmetric, per WBT variant, paired by graph and training seed',
              'sign_test':'exact 2-sided, 3 graph means (minimum attainable p-value=0.25)',
              'warning':'Exploratory benchmark; test already observed in prior 111/42 pilot; no claims of unseen confirmatory holdout',
              'plan_source_sha256':plan['source_sha256']}
    (out/'verificacion_e_integridad.json').write_text(json.dumps(complete,indent=2))
    (out/'INFORME_LEER_PRIMERO.txt').write_text(
        'Auditoría factorial 120 ejecuciones BiSBM-EASY.\n'
        'Usar medias_por_grafo.csv y contrastes_vs_original.csv para conclusiones: n=3 grafos independientes.\n'
        'Los 5 training seeds describen estabilidad dentro de grafo, NO n=15 datasets.\n'
        'Control principal predefinido: original vs symmetric para cada variante.\n'
        'Con 3 grafos, incluso todas las mejoras en la misma dirección dan p=0.25 (bilateral).\n'
        'El dataset EASY es control artificial fácil, no realismo empresarial.\n'
        'El grafo111/semilla42 ya se usó exploratoriamente; no es un holdout confirmatorio.\n'
        'El decoder usa selección en validación del runner original. Nunca se eligieron modos mirando test.\n')
    print('All 120 runs and paired test candidates verified; 3 independent graph realizations.',flush=True)
    print(summary[['bits','mode','auc_mean','ap_mean','overlap_mean','erank_U_mean','erank_V_mean']].to_string(index=False),flush=True)
    print(f'Saved analysis to {out}',flush=True)


if __name__=='__main__':main()
