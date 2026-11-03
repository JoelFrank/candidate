"""Generate auditable comparison and positive-blue/negative-red score plots."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def erank(x):
    import torch
    centered=x-x.mean(dim=0,keepdim=True)
    s=torch.linalg.svdvals(centered.float()).detach().cpu().numpy()
    s=s[s>1e-10]
    if not len(s):return 0.0
    prob=s/s.sum()
    return float(np.exp(-np.sum(prob*np.log(prob))))


def embedding_ranks(d):
    import sys,torch
    from revised import run_easy
    from src.models.wbtbgrl import WBTBGRL
    cfg=d['config'];device=torch.device('cpu')
    a=run_easy.parser().parse_args([])
    a.graph_seed=cfg['graph_seed'];a.hidden=cfg['hidden'];a.easy_groups=cfg['easy_groups'];a.easy_noise=cfg['easy_noise']
    ss,embed,_=run_easy.prepare_easy(a)
    checkpoint=torch.load(d['checkpoint_path'],map_location=device,weights_only=True)
    model=WBTBGRL(input_dim_u=a.hidden,input_dim_v=a.hidden,
         hidden_dim=a.hidden,output_dim=cfg['dim'],dropout=cfg['dropout'],
         weight_propagation=cfg['propagation'],weight_augmentation=cfg['augmentation'],
         weight_loss=cfg['loss_weights'])
    model.load_state_dict(checkpoint['model']);embed.load_state_dict(checkpoint['embedder'])
    model.eval();embed.eval()
    ei,w=run_easy.remap_context(ss[:2],ss[2])
    with torch.no_grad():
        x,y=embed(*ss[2]['inputs'])
        u,v=model.get_embeddings(x,y,ei,ss[2]['num_u'],w)
    return erank(u),erank(v)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.input);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    rows=[];pairs=[]
    expected=[f for f in root.glob('g*_WBT_*') if f.is_dir()]
    if not expected:raise RuntimeError(f'No WBT result directories in {root}')
    for path in sorted(expected):
        jf=path/'result.json'; sf=path/'test_scores.csv';modefile=path/'ssl_diagnostic_mode.json'
        if not jf.exists() or not sf.exists() or not modefile.exists():
            raise RuntimeError(f'Incomplete result: {path}')
        j=json.loads(jf.read_text()); mode=json.loads(modefile.read_text())['mode']; bits=json.loads(modefile.read_text())['bits']
        sc=pd.read_csv(sf); pos=sc.loc[sc.label==1,'score'].to_numpy(); neg=sc.loc[sc.label==0,'score'].to_numpy()
        ed=np.linspace(0,1,31)
        hp,_=np.histogram(pos,bins=ed,density=True);hn,_=np.histogram(neg,bins=ed,density=True)
        overlap=float(np.sum(np.minimum(hp,hn)*np.diff(ed)))
        cp=path/'checkpoint.pt'
        j['checkpoint_path']=str(cp)
        ru,rv=embedding_ranks(j) if cp.is_file() else (np.nan,np.nan)
        rows.append(dict(bits=bits,mode=mode,graph_seed=j['config']['graph_seed'],seed=j['config']['seed'],
                         auc=float(j['metrics']['ROC-AUC']),ap=float(j['metrics']['AP']),val_auc=float(j['val_auc']),
                         erank_U=ru,erank_V=rv,overlap=overlap,candidate_sha256=j['candidate_sha256'],
                         loss_last=float(j['ssl_losses'][-1]),path=str(path)))
        pairs.append((bits,mode,sc))
        fig,ax=plt.subplots(figsize=(8.5,4.7))
        ax.hist(pos,bins=ed,density=True,alpha=.58,color='tab:blue',label=f'Positive links ({len(pos)})')
        ax.hist(neg,bins=ed,density=True,alpha=.58,color='tab:red',label=f'Negative links ({len(neg)})')
        ax.set(xlabel='Decoder score',ylabel='Density',title=f'WBT_{bits} / {mode} | test ROC-AUC={j["metrics"]["ROC-AUC"]:.4f}',xlim=(0,1))
        ax.grid(alpha=.2);ax.legend();fig.tight_layout()
        fig.savefig(out/f'hist_WBT_{bits}_{mode}.png',dpi=170);plt.close(fig)
    df=pd.DataFrame(rows).sort_values(['bits','mode'])
    for bits,sub in df.groupby('bits'):
        if sub.candidate_sha256.nunique()!=1:
            raise RuntimeError(f'Candidate mismatch: WBT_{bits}; comparisons INVALID')
    if df.candidate_sha256.nunique()!=1:
        raise RuntimeError('Different candidates across modes/variants; comparisons INVALID')
    df.to_csv(out/'per_run.csv',index=False)
    reference=df[df['mode']=='original'][['bits','auc','ap','overlap','erank_U','erank_V']].rename(columns={k:k+'_original' for k in ('auc','ap','overlap','erank_U','erank_V')})
    compare=df.merge(reference,on='bits',how='left',validate='many_to_one')
    compare['delta_auc_vs_original']=compare['auc']-compare['auc_original']
    compare['delta_ap_vs_original']=compare['ap']-compare['ap_original']
    compare['delta_erank_U_vs_original']=compare['erank_U']-compare['erank_U_original']
    compare.to_csv(out/'paired_vs_original.csv',index=False)
    (out/'summary.json').write_text(json.dumps({'complete_runs':len(df),'same_test_candidates':True,
        'graph_seed':int(df.graph_seed.iloc[0]),'seed':int(df.seed.iloc[0]),
        'warning':'Exploratory diagnostic; 1 graph does not establish statistical benefit; check validation before deciding'},indent=2))
    print(compare[['bits','mode','auc','ap','overlap','erank_U','erank_V','delta_auc_vs_original']].to_string(index=False),flush=True)
    print(f'Wrote {out}',flush=True)


if __name__=='__main__':main()
