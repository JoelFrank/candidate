"""Analysis: P-only propagation weight sensitivity on *independent graphs*.

Reads recorded model scores, training-only outlier audits and optional PAS
WBT_100 baseline. The test set is diagnostic; no winner is selected here.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from scipy.stats import ks_2samp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from revised.p_sensitivity import parser as batch_parser,validate


def read_metrics(folder):
    scores=pd.read_csv(folder/'test_scores.csv')
    if not {'u','v','label','score'}.issubset(scores.columns):
        raise RuntimeError(f'Missing score fields {folder}')
    y=scores.label.to_numpy();s=scores.score.to_numpy()
    if not np.isfinite(s).all() or set(np.unique(y))!={0,1}:
        raise RuntimeError(f'Invalid scores {folder}')
    pos=s[y==1];neg=s[y==0]
    return scores, {'roc_auc':float(roc_auc_score(y,s)),
          'ap':float(average_precision_score(y,s)),
          'ks':float(ks_2samp(pos,neg).statistic),
          'fraction_predicted_positive':float(np.mean(s>=.5))}


def analyze(src, dest, pas=None):
    plan=json.loads((src/'sensitivity_plan.json').read_text())
    a=batch_parser().parse_args([])
    for k,v in plan.items():
        if hasattr(a,k):setattr(a,k,v)
    values=[]; allscores=defaultdict(list); misses=[]
    for g in plan['graphs']:
      for seed in plan['seeds']:
        for mode in plan['modes']:
          folder=src/f'g{g}_s{seed}_informative_P_{mode}'
          if not folder.exists() or not (folder/'result.json').is_file():
              misses.append(str(folder));continue
          validate(folder,g,seed,mode,a)
          r=json.loads((folder/'result.json').read_text())
          s,meas=read_metrics(folder)
          if abs(r['metrics']['ROC-AUC']-meas['roc_auc'])>1e-6 or abs(r['metrics']['AP']-meas['ap'])>1e-6:
              raise RuntimeError('Stored metrics mismatch')
          allscores[(g,seed)].append((mode,s,r['candidate_sha256']))
          weight=r['propagation_sensitivity']['train_weight_report']
          values.append({'graph':g,'seed':seed,'mode':mode,'candidate_sha256':r['candidate_sha256'],
            'validation_auc':float(r['val_auc']),'max_train_weight':weight['maximum'],
            'p99_train_weight':weight['p99'],
            'clip_train_threshold':weight['clip_threshold_train_only'],
            'top1pct_mass':weight['top_1pct_edges_weight_fraction'],
            **meas})
    if not values:raise RuntimeError('No completed sensitivity runs')
    df=pd.DataFrame(values)
    for key,items in allscores.items():
        if len(set(h for _,_,h in items))>1:
            raise RuntimeError(f'Candidate hash mismatch {key}')
        ref=items[0][1][['u','v','label']].reset_index(drop=True)
        if any(not x[['u','v','label']].reset_index(drop=True).equals(ref) for _,x,_ in items[1:]):
            raise RuntimeError(f'Candidate table mismatch {key}')
    # Ensure tests identical ACROSS training seeds too, within a graph.
    for g in plan['graphs']:
        arr=[items[0][1] for key,items in allscores.items() if key[0]==g]
        if arr and any(not x[['u','v','label']].reset_index(drop=True).equals(arr[0][['u','v','label']].reset_index(drop=True)) for x in arr[1:]):
            raise RuntimeError(f'Candidate mismatch across seeds: graph {g}')
    baseline_results=[]
    if pas is not None:
        for _,row in df[df['mode']=='raw'].iterrows():
            old=pas/f'g{int(row.graph)}_s{int(row.seed)}_informative_WBT_100'
            if not (old/'result.json').exists():continue
            pr=json.loads((old/'result.json').read_text())
            oldsc,oldmeas=read_metrics(old)
            if not oldsc[['u','v','label']].reset_index(drop=True).equals(
                next(v for mode,v,_ in allscores[(int(row.graph),int(row.seed))] if mode=='raw')[['u','v','label']].reset_index(drop=True)):
                raise RuntimeError('PAS baseline candidates differ')
            delta=float(row.roc_auc-oldmeas['roc_auc'])
            baseline_results.append(dict(graph=int(row.graph),seed=int(row.seed),raw_auc=row.roc_auc,
                pas100_auc=oldmeas['roc_auc'],auc_delta=delta,match=abs(delta)<=1e-6))
        if baseline_results and not all(x['match'] for x in baseline_results):
            raise RuntimeError('RAW sensitivity run does not reproduce existing PAS WBT_100! Investigate before interpretation.')
    df=df.sort_values(['graph','seed','mode'])
    graphmean=df.groupby(['graph','mode'],as_index=False)[['roc_auc','ap','ks','validation_auc']].mean()
    across=graphmean.groupby('mode',as_index=False)[['roc_auc','ap','ks','validation_auc']].mean()
    contrasts=[]
    for g,group in graphmean.groupby('graph'):
        sg=group.set_index('mode')
        if 'raw' not in sg.index:continue
        for mode in ('log1p_anchor1','clip_train_p99'):
            if mode in sg.index:
                contrasts.append(dict(graph=g,contrast=f'{mode} - raw',auc_delta=float(sg.loc[mode,'roc_auc']-sg.loc['raw','roc_auc']),
                    ap_delta=float(sg.loc[mode,'ap']-sg.loc['raw','ap']),ks_delta=float(sg.loc[mode,'ks']-sg.loc['raw','ks'])))
    comparison=pd.DataFrame(contrasts)
    dest.mkdir(parents=True,exist_ok=True)
    df.to_csv(dest/'p_runs.csv',index=False)
    graphmean.to_csv(dest/'p_by_graph.csv',index=False)
    across.to_csv(dest/'p_summary.csv',index=False)
    comparison.to_csv(dest/'p_paired_graphs.csv',index=False)
    pd.DataFrame(baseline_results).to_csv(dest/'p_raw_vs_pas100.csv',index=False)
    report={'expected':len(plan['graphs'])*len(plan['seeds'])*len(plan['modes']),
        'completed':len(df),'independent_graphs':int(df.graph.nunique()),
        'incomplete':misses,'candidate_check':'PASS','raw_matches_PAS100':
        all(r['match'] for r in baseline_results) if baseline_results else None,
        'note':'Exploratory test diagnostics; no hyperparameter selection on test. Threshold fitted on TRAIN ONLY.'}
    (dest/'integrity.json').write_text(json.dumps(report,indent=2))
    lines=['# Propagation P-only sensitivity — BiSBM informative','',
        f"Complete: {report['completed']}/{report['expected']}; independent graphs: {report['independent_graphs']}",
        '', 'This experiment changes ONLY edge weights presented to `GCNConv` in WBT_100.',
        'Augmentation weights A=0; SSL weights S=0; decoder NWB. No model source or generation modifications.',
        'Modes: raw; log1p(w)/log(2), which anchors w=1; clip at training-weight 99th percentile.',
        'Do NOT select a transformation from test scores; this experiment only diagnoses sensitivity.', '',
        '| Mode | ROC-AUC | AP | KS | Val ROC-AUC |','|---|---:|---:|---:|---:|']
    for _,r in across.sort_values('mode').iterrows():
        lines.append(f"| {r['mode']} | {r['roc_auc']:.4f} | {r['ap']:.4f} | {r['ks']:.4f} | {r['validation_auc']:.4f} |")
    lines+=['','## Paired differences averaged within independent graphs','','| Contrast | Mean ΔAUC | n graphs |','|---|---:|---:|']
    if not comparison.empty:
        for key,group in comparison.groupby('contrast'):
            lines.append(f"| {key} | {group.auc_delta.mean():+.4f} | {len(group)} |")
    lines+=['','## Limits','Only a sensitivity analysis. Transformation is NOT part of submitted WBT-BGRL.',
            'Three graphs are not enough for strong statistical claims. Run held-out independent graphs.',
            'Compare to BiSBM-EASY separately as a positive control for structural separability.']
    (dest/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report,indent=2));print('Report',dest/'report.md')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='results/sensibilidad_P_BiSBM')
    p.add_argument('--output',default='results/analisis_sensibilidad_P')
    p.add_argument('--pas',default=None,help='Optional existing results/ablacion_PAS_BiSBM baseline WBT_100')
    a=p.parse_args()
    analyze(Path(a.input),Path(a.output),Path(a.pas) if a.pas else None)

if __name__=='__main__': main()
