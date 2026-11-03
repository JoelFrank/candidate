"""Report P/A/S effects at independent graph level, not independent training seeds.

Test scores are diagnostic only. No hyperparameter or variant is chosen on test.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from revised.pas_ablation import BITS, validate_completion

METRICS = ['roc_auc', 'ap', 'ks', 'overlap_hist50', 'auc_oo', 'auc_ou', 'auc_uu']
CONTRASTS = [('100', '000'), ('101','100'), ('110','100'), ('111','100'), ('111','110'), ('111','101'), ('001','000'), ('010','000'), ('111','000')]


def calc_auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float('nan')


def score_metrics(scores: pd.DataFrame, old_u: int, old_v: int):
    labels = scores['label'].to_numpy()
    pred = scores['score'].to_numpy()
    if not len(scores) or not np.isfinite(pred).all() or set(np.unique(labels)) != {0,1}:
        raise ValueError('Candidate scores must be finite and contain both labels')
    pos = pred[labels == 1]; neg = pred[labels == 0]
    hp, _ = np.histogram(pos, bins=50, range=(0, 1), density=False)
    hn, _ = np.histogram(neg, bins=50, range=(0, 1), density=False)
    r = dict(roc_auc=calc_auc(labels, pred), ap=float(average_precision_score(labels, pred)),
             ks=float(ks_2samp(pos,neg).statistic),
             overlap_hist50=float(np.minimum(hp / len(pos), hn / len(neg)).sum()),
             positive_mean=float(pos.mean()), negative_mean=float(neg.mean()),
             predicted_positive_fraction=float((pred >= .5).mean()),
             pos_count=len(pos), neg_count=len(neg))
    # Original CSV contains the raw synthetic node IDs (not local remapped indices).
    u_new=scores['u'].astype(int).to_numpy() >= old_u
    v_new=scores['v'].astype(int).to_numpy() >= old_v
    masks=dict(oo=(~u_new & ~v_new), ou=(u_new ^ v_new), uu=(u_new & v_new))
    for name, mask in masks.items():
        r[f'auc_{name}']=calc_auc(labels[mask],pred[mask])
        r[f'n_{name}']=int(mask.sum())
    return r


def _same_candidates(group):
    base = None
    for scores in group:
        # IDs and label, not scores, must be shared by every variant.
        cand = scores[['u','v','label']].reset_index(drop=True)
        if base is None: base = cand
        elif not base.equals(cand): return False
    return True


def summarize_graph_independent(paired_by_graph: pd.DataFrame, col='delta'):
    if paired_by_graph.empty:
        return pd.DataFrame()
    groupcols = [c for c in ['condition','contrast','metric','mechanism'] if c in paired_by_graph.columns]
    grouped = paired_by_graph.groupby(groupcols,dropna=False)[col].agg(['mean','std','min','max','count']).reset_index()
    grouped = grouped.rename(columns={'mean':'mean_delta','std':'sd_between_graphs','count':'n_independent_graphs'})
    grouped['note'] = 'Exploratory only; graph is the independent unit. No significance claims.'
    return grouped


def analyze(inputdir: Path, outputdir: Path, old_u: int, old_v: int):
    planfile = inputdir / 'pas_plan.json'
    if not planfile.exists(): raise FileNotFoundError(f'Missing {planfile}')
    plan = json.loads(planfile.read_text())
    rows=[]; all_scores={}; checks=[]; incomplete=[]; hashgroups=defaultdict(set)
    planned=set()
    for g in plan['graphs']:
        for s in plan['seeds']:
            for cond in plan['conditions']:
                for bits in plan['variants']:
                    planned.add((g,s,cond,bits))
                    path=inputdir / f'g{g}_s{s}_{cond}_WBT_{bits}'
                    if not path.is_dir() or not (path/'result.json').exists():
                        incomplete.append(str(path));continue
                    validate_completion(path, g, s, cond, bits, plan['candidate_seed'])
                    payload=json.loads((path/'result.json').read_text())
                    if any(payload['config'][key] != plan[key] for key in ['encoder_epochs','decoder_epochs','hidden','dim','patience','threads','lr','weight_decay','dropout','lam','edge_drop','feature_drop']):
                        raise RuntimeError(f'Hyperparameter mismatch with plan in {path}')
                    scores=pd.read_csv(path/'test_scores.csv')
                    if len(scores) != 2*payload['negative_sampling']['strata']['test']['positive']['OO'] + 2*sum(payload['negative_sampling']['strata']['test']['positive'][k] for k in ['ON','NO','NN']):
                        raise RuntimeError(f'Unexpected candidate row count: {path}')
                    results=score_metrics(scores,old_u,old_v)
                    if abs(results['roc_auc'] - payload['metrics']['ROC-AUC'])>1e-6 or abs(results['ap']-payload['metrics']['AP'])>1e-6:
                        raise RuntimeError(f'Score/metrics mismatch: {path}')
                    row=dict(graph_seed=g,seed=s,condition=cond,bits=bits,variant='WBT_'+bits,
                             candidate_sha256=payload['candidate_sha256'],
                             val_auc=float(payload['val_auc']),
                             decoder_seconds=float(payload['timing']['decoder_s']),
                             encoder_seconds=float(payload['timing']['encoder_s']),
                             cuda_peak_bytes=payload['cuda_peak_bytes'],**results)
                    rows.append(row)
                    key=(g,s,cond)
                    all_scores[(g,s,cond,bits)]=scores
                    hashgroups[key].add(payload['candidate_sha256'])
    if not rows:raise RuntimeError('No completed factorial runs were found')
    r=pd.DataFrame(rows).sort_values(['condition','graph_seed','seed','bits']).reset_index(drop=True)
    duplicate=r.duplicated(['graph_seed','seed','condition','bits']).any()
    if duplicate:raise RuntimeError('Duplicate runs')
    mismatched_hash={str(k):sorted(v) for k,v in hashgroups.items() if len(v)>1}
    if mismatched_hash:raise RuntimeError(f'Test candidates differ within a matched pair: {mismatched_hash}')
    # A hash can be wrong or stale: verify candidate rows directly as well.
    for k in hashgroups:
        frames = [all_scores[k+(bits,)] for bits in plan['variants'] if k+(bits,) in all_scores]
        if not _same_candidates(frames): raise RuntimeError(f'Different candidate rows within same (graph,seed,condition): {k}')
    # Test dataset MUST also be identical across training seeds within a graph.
    for g in plan['graphs']:
        for cond in plan['conditions']:
            frames=[all_scores[(g,s,cond,bits)] for s in plan['seeds'] for bits in plan['variants'] if (g,s,cond,bits) in all_scores]
            if len(frames)>1 and not _same_candidates(frames):
                raise RuntimeError(f'Candidate mismatch between training seeds at graph {g}, {cond}')
    means=r.groupby(['condition','bits'],as_index=False)[METRICS+['predicted_positive_fraction','positive_mean','negative_mean','encoder_seconds','decoder_seconds']].agg('mean')
    graphmeans=r.groupby(['condition','graph_seed','bits'],as_index=False)[METRICS].mean()
    pairs=[]; factor=[]
    for (cond,g),df in graphmeans.groupby(['condition','graph_seed']):
        sub=df.set_index('bits')
        for left,right in CONTRASTS:
            if left in sub.index and right in sub.index:
                for metric in METRICS:
                    pairs.append(dict(condition=cond,graph_seed=g,contrast=f'WBT_{left} - WBT_{right}',metric=metric,
                                      delta=float(sub.loc[left,metric]-sub.loc[right,metric])))
        # Main factorial effects average across the four matching combinations of the other factors.
        for pos,mech in enumerate(['P','A','S']):
            match=[]
            for bits in BITS:
                if bits[pos]=='0':
                    other=bits[:pos]+'1'+bits[pos+1:]
                    if bits in sub.index and other in sub.index:match.append((other,bits))
            if len(match)==4:
                for metric in METRICS:
                    changes=[sub.loc[a,metric]-sub.loc[b,metric] for a,b in match]
                    factor.append(dict(condition=cond,graph_seed=g,mechanism=mech,metric=metric,
                                       delta=float(np.mean(changes))))
    p=pd.DataFrame(pairs);f=pd.DataFrame(factor)
    ps=summarize_graph_independent(p);fs=summarize_graph_independent(f)
    outputdir.mkdir(parents=True,exist_ok=True)
    r.to_csv(outputdir/'pas_runs.csv',index=False)
    means.to_csv(outputdir/'pas_means.csv',index=False)
    p.to_csv(outputdir/'pas_pairwise_by_graph.csv',index=False)
    ps.to_csv(outputdir/'pas_pairwise_summary.csv',index=False)
    f.to_csv(outputdir/'pas_factorial_by_graph.csv',index=False)
    fs.to_csv(outputdir/'pas_factorial_summary.csv',index=False)
    expected=len(planned)
    diag=dict(expected_runs=expected,completed_runs=len(r),missing_runs=incomplete,
              full_factorial_requested=(set(plan['variants'])==set(BITS)),
              number_independent_graphs=len(r.graph_seed.unique()),
              number_training_seeds=len(r.seed.unique()),candidate_hash_mismatches=mismatched_hash,
              negative_policy='matched_synthetic',
              summary_is_complete=len(r)==expected,
              scientific_limitations=['Three independent graphs cannot support strong significance claims.',
                 'No model-selection or hyperparameter tuning on test scores.',
                 'The decoder WB is disabled; S uses raw edge weights in the current code.',
                 'The hist50 overlap is bin-dependent; KS and ROC-AUC are preferred.'])
    (outputdir/'pas_integrity.json').write_text(json.dumps(diag,indent=2))
    lines=[f'# P/A/S factorial — {len(r)}/{expected} runs',
           '', 'This is a descriptive test-set audit, **not** hyperparameter selection.',
           'The **graph**, not a seed on that graph, is the independent unit.',
           'P: weighted `GCNConv`; A: weight-sensitive AUG1 edge dropout; S: original RAW-weight edge SSL.',
           'Decoder NWB stays unchanged; positives/negatives use matched BiSBM strata.', '',
           f'Independent graphs: {diag["number_independent_graphs"]}; seeds per plan: {len(plan["seeds"])}.',
           f'Completed: {len(r)}/{expected}; candidate checks: PASS.', '']
    if incomplete:lines+=['## Incomplete experiments',f'{len(incomplete)} missing runs; treat comparisons as provisional.', '']
    if set(plan['variants']) != set(BITS):lines+=['WARNING: subset of 8 P/A/S variants was requested; this is NOT a full factorial analysis.', '']
    for cond in plan['conditions']:
        lines+=['## '+cond, '', '| P/A/S | ROC-AUC | AP | KS | Histogram overlap ↓ |', '|---|---:|---:|---:|---:|']
        for _,row in means[means.condition==cond].sort_values('bits').iterrows():
            lines.append(f'| WBT_{row.bits} | {row.roc_auc:.4f} | {row.ap:.4f} | {row.ks:.4f} | {row.overlap_hist50:.4f} |')
        lines+=['', '### Paired contrasts (mean of per-graph differences)', '', '| Contrast | Δ ROC-AUC | Graphs |','|---|---:|---:|']
        x=ps[(ps.condition==cond)&(ps.metric=='roc_auc')] if not ps.empty else pd.DataFrame()
        for _,row in x.iterrows():lines.append(f'| {row.contrast} | {row.mean_delta:+.4f} | {int(row.n_independent_graphs)} |')
        lines+=['', '### Factorial main effects (averaged across remaining mechanisms)', '', '| Effect | Δ ROC-AUC | Graphs |','|---|---:|---:|']
        x=fs[(fs.condition==cond)&(fs.metric=='roc_auc')] if not fs.empty else pd.DataFrame()
        for _,row in x.iterrows(): lines.append(f'| {row.mechanism} | {row.mean_delta:+.4f} | {int(row.n_independent_graphs)} |')
        lines+=['']
    lines+=['## Interpretation constraints',
        'These effects are observational comparisons under a controlled factorial configuration, not causality proofs.',
        'No statistical significance is claimed with three independent graphs. Repeat with more independently generated graphs before journal claims.',
        'Do not select the best variant using test ROC-AUC.',
        'AUG1 weighted dropout changes the expected **number** of kept edges, not only which edges survive.',
        'WBT_111 and WBT_000 should match previous WP_NWB/NWP_NWB runs only when all settings and project source versions match.']
    (outputdir/'pas_summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:diag[k] for k in ['expected_runs','completed_runs','summary_is_complete','number_independent_graphs']},indent=2))
    print('Report:',outputdir/'pas_summary.md')
    return diag


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='results/ablacion_PAS_BiSBM')
    p.add_argument('--output',default='results/analisis_PAS_BiSBM')
    p.add_argument('--old-u',type=int,default=None,help='Override generator N_U_OLD')
    p.add_argument('--old-v',type=int,default=None,help='Override generator N_V_OLD')
    a=p.parse_args(argv)
    if a.old_u is None or a.old_v is None:
        # Match the boundaries to the generator used by the benchmark.
        from revised.negative_sampling import synthetic_bisbm_boundaries
        u,v=synthetic_bisbm_boundaries()
    else:u,v=a.old_u,a.old_v
    analyze(Path(a.input),Path(a.output),u,v)

if __name__=='__main__':main()
