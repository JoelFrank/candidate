"""Audited runner; original scripts are retained for forensic comparison only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys, argparse, random, json, time, platform, hashlib, copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,average_precision_score,precision_score,recall_score,f1_score
from revised.data import prepare,remap_context
from revised.negative_sampling import negatives,matched_synthetic_negatives,synthetic_stratum_counts,synthetic_bisbm_boundaries
from revised.faa_sampling import matched_endpoint_negatives,stratum_counts,candidate_frame
from revised.inductive_protocol import (annotate_snapshots,matched_history_negatives,matched_static_bisbm_negatives,
                                        group_counts,candidate_records,audit_pairs,pair_group)
from src.models.wbtbgrl import WBTBGRL
from src.models.bgrl import BGRL
from src.models.tbgrl import TBGRL
from src.models.gbt import GBT
from src.models.cca_ssg import CCASSG
from src.evaluation.decoder import LinkPredictor

def metrics(y,p):
    b=p>=.5
    r={'ROC-AUC':roc_auc_score(y,p),'AP':average_precision_score(y,p),'F1':f1_score(y,b,zero_division=0),'precision':precision_score(y,b,zero_division=0),'recall':recall_score(y,b,zero_division=0)}
    for k in [10,20,50,100]:
        neg=p[y==0];r[f'Hits@{k}']=float(np.mean(p[y==1]>np.partition(neg,-k)[-k])) if len(neg)>=k else None
    return r

def run(a):
    torch.set_num_threads(a.threads);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.use_deterministic_algorithms(True)
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    ss,embed=prepare(a)
    annotate_snapshots(ss)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    audit={}
    for i,s in enumerate(ss):
        hist=ss[:max(1,i)];ei,ew=remap_context(hist,s)
        audit[['train','val','test'][i]]={'nodes_u':s['num_u'],'nodes_v':s['num_v'],'positive_edges':s['edges'].shape[1],'context_edges':ei.shape[1],'weight_min':float(s['weights'].min()),'weight_max':float(s['weights'].max()),'context_sha256':hashlib.sha256(ei.numpy().tobytes()+ew.numpy().tobytes()).hexdigest()}
    (out/'data_audit.json').write_text(json.dumps(audit,indent=2))
    if a.audit_only:return audit
    embed=embed.to(device)
    wp=a.variant.startswith('WP_');wb=a.variant.endswith('_WB')
    config=dict(input_dim_u=a.hidden,input_dim_v=a.hidden,hidden_dim=a.hidden,output_dim=a.dim,dropout=a.dropout)
    if a.model=='WBTBGRL':
        from revised.ssl_factorial_easy import create_model_type
        WBTClass=create_model_type(WBTBGRL,a.ssl_mode)
        model=WBTClass(**config,lambda_param=a.lam,use_weights_pretraining=wp,use_weights_link_prediction=wb,
                       weight_propagation=a.propagation,weight_augmentation=a.augmentation,weight_loss=a.loss_weights,
                       edge_drop_rate=a.edge_drop,feat_drop_rate=a.feature_drop).to(device)
    else:
        model={'BGRL':BGRL,'TBGRL':TBGRL,'GBT':GBT,'CCASSG':CCASSG}[a.model](**config).to(device)
        wp=False;wb=False
    # All models share the same existing domain feature bridge for controlled comparison.
    for s in ss:s['inputs']=[x.to(device) for x in s['inputs']]
    opt=torch.optim.AdamW(list(model.parameters())+list(embed.parameters()),lr=a.lr,weight_decay=a.weight_decay)
    t0=time.perf_counter()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    losses=[]
    for epoch in range(a.encoder_epochs):
        model.train();embed.train();opt.zero_grad();u,v=embed(*ss[0]['inputs']);ei=ss[0]['edges'].to(device);w=ss[0]['weights'].to(device)
        if a.model=='WBTBGRL':loss=model.compute_loss(model(u,v,ei,ss[0]['num_u'],w),ss[0]['num_u'])
        else:loss=model.compute_loss(model(u,v,ei,ss[0]['num_u']))
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite SSL loss')
        loss.backward();opt.step()
        if hasattr(model,'_update_target_encoder'):model._update_target_encoder()
        losses.append(float(loss.detach()))
    if device.type=='cuda':torch.cuda.synchronize()
    encoder_seconds=time.perf_counter()-t0
    model.eval();embed.eval()
    for p in list(model.parameters())+list(embed.parameters()):p.requires_grad_(False)
    def representations(i):
        s=ss[i];ei,w=remap_context(ss[:max(1,i)],s)
        with torch.no_grad():
            u,v=embed(*s['inputs'])
            if a.model=='WBTBGRL':return torch.cat(model.get_embeddings(u,v,ei.to(device),s['num_u'],w.to(device)))
            return torch.cat(model.get_embeddings(u,v,ei.to(device),s['num_u']))
    reps=[representations(i) for i in range(2)]
    sampling_policy='matched_history' if a.negative_strategy=='auto' else a.negative_strategy
    if sampling_policy=='matched_synthetic' and a.dataset!='synthetic':
        raise ValueError('matched_synthetic negative policy is only defined for BiSBM raw node IDs')
    if sampling_policy=='matched_endpoint' and a.dataset!='faa_sdr_ap_3y':
        raise ValueError('matched_endpoint is for FAA prepared benchmark')
    old_u_v = synthetic_bisbm_boundaries() if sampling_policy == 'matched_synthetic' else None
    negative_strata={}
    def candidates(i):
        s=ss[i]
        if sampling_policy=='matched_history':
            neg=(matched_static_bisbm_negatives(s,ss,a.candidate_seed+i)
                 if a.dataset=="synthetic" else matched_history_negatives(s,a.candidate_seed+i))
            negative_strata[['train','val','test'][i]]=audit_pairs(s,neg)
            pd.DataFrame(candidate_records(s,neg)).to_csv(
                out/f'{["train","val","test"][i]}_candidates.csv',index=False)
        elif sampling_policy=='matched_synthetic':
            num_u_old, num_v_old = old_u_v
            neg=matched_synthetic_negatives(s,ss[:i+1],a.candidate_seed+i,num_u_old,num_v_old)
            negative_strata[['train','val','test'][i]]={
                'positive':synthetic_stratum_counts(s,s['edges'],num_u_old,num_v_old),
                'negative':synthetic_stratum_counts(s,neg,num_u_old,num_v_old)}
        elif sampling_policy=='matched_endpoint':
            neg=matched_endpoint_negatives(s,a.candidate_seed+i)
            negative_strata[['train','val','test'][i]]={
                'positive':stratum_counts(s,s['edges']), 'negative':stratum_counts(s,neg)}
            # Save reproducible candidate rosters for cross-model comparisons.
            candidate_df=pd.DataFrame(candidate_frame(s,neg),columns=['u','v','label','stratum'])
            candidate_df.to_csv(out/f'{["train","val","test"][i]}_candidates.csv',index=False)
        else:
            neg=negatives(s,ss[:i+1],s['edges'].shape[1],a.candidate_seed+i)
        e=torch.cat([s['edges'],neg],dim=1).to(device);y=torch.cat([torch.ones(s['edges'].shape[1]),torch.zeros(neg.shape[1])]).to(device)
        return e,y
    es=[candidates(i) for i in range(2)]
    decoder=LinkPredictor(a.dim*2,a.hidden,dropout=a.dropout).to(device)
    opt=torch.optim.AdamW(decoder.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    def score(z,e):return decoder(torch.cat([z[e[0]],z[e[1]]],dim=1)).reshape(-1)
    weights=torch.cat([torch.log1p(ss[0]['weights']) if wb else torch.ones_like(ss[0]['weights']),torch.ones(ss[0]['edges'].shape[1])]).to(device)
    best=-float('inf');state=None;wait=0;history=[];t0=time.perf_counter()
    for epoch in range(a.decoder_epochs):
        decoder.train();opt.zero_grad();pred=score(reps[0],es[0][0]);loss=(F.binary_cross_entropy_with_logits(pred,es[0][1],reduction='none')*weights).mean();loss.backward();opt.step()
        decoder.eval()
        with torch.no_grad():vp=torch.sigmoid(score(reps[1],es[1][0])).cpu().numpy()
        auc=roc_auc_score(es[1][1].cpu().numpy(),vp);history.append({'epoch':epoch+1,'train_loss':float(loss.detach()),'val_auc':auc})
        if auc>best:best=auc;state=copy.deepcopy(decoder.state_dict());wait=0
        else:wait+=1
        if wait>=a.patience:break
    decoder.load_state_dict(state);decoder.eval()
    if device.type=='cuda':torch.cuda.synchronize()
    decoder_seconds=time.perf_counter()-t0;t0=time.perf_counter()
    z=representations(2);e,y=candidates(2)
    with torch.no_grad():p=torch.sigmoid(score(z,e)).cpu().numpy()
    if device.type=='cuda':torch.cuda.synchronize()
    inference_seconds=time.perf_counter()-t0;y=y.cpu().numpy();result=metrics(y,p)
    pairs=e.cpu().numpy().T
    scores=pd.DataFrame({'u':ss[2]['keys_u'][pairs[:,0]],'v':ss[2]['keys_v'][pairs[:,1]-ss[2]['num_u']],'label':y,'score':p})
    strata_and_context=[pair_group(ss[2],u,v) for u,v in pairs]
    scores['stratum']=[tag for tag,ctx in strata_and_context]
    scores['history_status']=[ctx for tag,ctx in strata_and_context]
    result['by_stratum']={}
    result['by_stratum_history']={}
    for tag in ('OO','OU','UO','UU'):
        mask=(scores.stratum==tag).to_numpy()
        pos=int(np.sum((y==1)&mask));neg=int(np.sum((y==0)&mask))
        result['by_stratum'][tag]={'positives':pos,'negatives':neg,
                                   'metrics':metrics(y[mask],p[mask]) if pos and neg else None}
        for ctx in ('SS','SN','NS','NN'):
            submask=mask & (scores.history_status.to_numpy()==ctx)
            np_=int(np.sum((y==1)&submask));nn_=int(np.sum((y==0)&submask))
            if np_ or nn_:
                result['by_stratum_history'][f'{tag}:{ctx}']={
                    'positives':np_,'negatives':nn_,
                    'metrics':metrics(y[submask],p[submask]) if np_ and nn_ else None}
    result['cold_start_note']='Train-relative U, with pre-horizon history controls. All-unknown nodes may have identical features; candidate roster is retrospective.'
    scores.to_csv(out/'test_scores.csv',index=False)
    import torch_geometric
    payload={'config':vars(a),'protocol':('history-only-v2; static-BiSBM-global-positive-exclusion; matched-history; decoder=train; selection=val; test=once'
                            if a.dataset=='synthetic' else 'history-only-v2; whole-day-splits; matched-history; decoder=train; selection=val; test=once')
                       if sampling_policy=='matched_history' else 'history-only-v1-legacy; decoder=train; selection=val; test=once','metrics':result,'val_auc':best,'history':history,'ssl_losses':losses,'timing':{'encoder_s':encoder_seconds,'decoder_s':decoder_seconds,'inference_s':inference_seconds,'epoch_s':encoder_seconds/a.encoder_epochs},'environment':{'python':platform.python_version(),'torch':torch.__version__,'pyg':torch_geometric.__version__,'device':str(device),'hardware':torch.cuda.get_device_name() if device.type=='cuda' else platform.processor()},'cuda_peak_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,'candidate_sha256':hashlib.sha256(pairs.tobytes()+y.tobytes()).hexdigest(),
             'negative_sampling':{'policy':sampling_policy,'seed':a.candidate_seed,'strata':negative_strata},
             'decoder_diagnostics':{'mean_positive_bce_weight':float(weights[:ss[0]['edges'].shape[1]].mean().item()),
                                    'test_predicted_positive_fraction_at_0_5':float(np.mean(p>=.5)),
                                    'note':'WB log1p positive weights are specified by the paper; a 0.5 threshold is not calibrated automatically.'}}
    (out/'result.json').write_text(json.dumps(payload,indent=2));torch.save({'model':model.state_dict(),'embedder':embed.state_dict(),'decoder':decoder.state_dict()},out/'checkpoint.pt')
    # Opportunistic consolidation. Raw results remain authoritative and unchanged.
    if not getattr(a, 'no_auto_summary', False):
        try:
            from revised.results import summarize, DEFAULT_SEEDS
            root=out.parent.parent if __import__('re').fullmatch(r'seed\d+',out.name) else out.parent
            aggregation=summarize(root, DEFAULT_SEEDS, strict=False)
            for issue in aggregation['errors']:
                print(f"[results warning] {issue['group']}: {issue['error']}",file=sys.stderr)
        except Exception as exc:
            print(f"[results warning] Consolidation failed without changing the run: {exc}",file=sys.stderr)
    print(json.dumps(result));return result

def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['synthetic','ecommerce','eurocontrol','industry','faa_sdr_ap_3y'],default='synthetic');p.add_argument('--data')
    p.add_argument('--condition',choices=['informative','anti_correlated','orthogonal'],default='informative');p.add_argument('--graph-seed',type=int,default=42);p.add_argument('--seed',type=int,default=42);p.add_argument('--candidate-seed',type=int,default=2026)
    p.add_argument('--variant',choices=['WP_WB','WP_NWB','NWP_WB','NWP_NWB'],default='WP_WB');p.add_argument('--model',choices=['WBTBGRL','BGRL','TBGRL','GBT','CCASSG'],default='WBTBGRL')
    for k,d in [('hidden',128),('dim',64),('encoder-epochs',200),('decoder-epochs',100),('patience',20),('threads',2)]:p.add_argument('--'+k,type=int,default=d)
    for k,d in [('lr',.001),('weight-decay',1e-5),('dropout',.1),('lam',.5),('edge-drop',.1),('feature-drop',.1),('train-ratio',.8),('val-ratio',.1)]:p.add_argument('--'+k,type=float,default=d)
    for k in ['propagation','augmentation','loss-weights']:p.add_argument('--'+k,type=lambda x:bool(int(x)),default=None)
    p.add_argument('--ssl-mode',choices=['original','symmetric'],default='original')
    p.add_argument('--negative-strategy',choices=['auto','matched_history','uniform','matched_synthetic','matched_endpoint'],default='auto',help='auto: 1:1 matched by train-relative and pre-horizon node status in every dataset; legacy policies explicitly selectable')
    p.add_argument('--device',default='cpu');p.add_argument('--output',default='results/revised');p.add_argument('--audit-only',action='store_true');p.add_argument('--no-auto-summary',action='store_true');return p
if __name__=='__main__':run(parser().parse_args())
