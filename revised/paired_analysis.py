"""Pair within graph/seed; aggregate seeds BEFORE graph-level inference."""
import argparse,json,itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import permutation_test
p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--output',default='paired_statistics.csv');a=p.parse_args()
rows=[]
for f in Path(a.directory).rglob('result.json'):
 d=json.loads(f.read_text());c=d['config']
 if c['model']!='WBTBGRL' or c['encoder_epochs']<200:continue
 rows.append(dict(graph=c['graph_seed'],seed=c['seed'],condition=c['condition'],variant=c['variant'],candidate=d['candidate_sha256'],auc=d['metrics']['ROC-AUC'],ap=d['metrics']['AP']))
df=pd.DataFrame(rows);out=[]
if df.empty:raise SystemExit('No full-schedule results available')
if (df.groupby(['graph','seed','condition']).candidate.nunique()>1).any():raise ValueError('Candidate mismatch')
for cond,sub in df.groupby('condition'):
 for metric in ['auc','ap']:
  wide=sub.pivot(index=['graph','seed'],columns='variant',values=metric)
  for a1,b1 in itertools.combinations(sorted(wide.columns),2):
   pair=wide[[a1,b1]].dropna();delta=(pair[a1]-pair[b1]).groupby('graph').mean().to_numpy();n=len(delta)
   if n<2:continue
   rng=np.random.default_rng(2026);boot=rng.choice(delta,(10000,n),replace=True).mean(1)
   # Paired sign-flip null at independent graph level; descriptive with few graphs.
   pval=permutation_test((delta,),lambda x:np.mean(x),permutation_type='samples',alternative='two-sided',n_resamples=99999,random_state=2026).pvalue
   out.append(dict(condition=cond,metric=metric,A=a1,B=b1,n_graphs=n,n_pairs=len(pair),mean_delta=delta.mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),paired_dz=delta.mean()/delta.std(ddof=1) if delta.std(ddof=1)>0 else None,p=pval,status='PILOT: descriptive only' if n<10 else 'planned graph-level analysis'))
# Holm within each condition/metric family.
res=pd.DataFrame(out)
for _,ix in res.groupby(['condition','metric']).groups.items():
 order=res.loc[ix].sort_values('p').index;adj=np.maximum.accumulate([min(1,res.loc[k,'p']*(len(order)-j)) for j,k in enumerate(order)]);res.loc[order,'p_holm']=adj
res.to_csv(a.output,index=False);print(res.to_string(index=False))
