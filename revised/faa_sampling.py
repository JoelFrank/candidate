"""Category-matched reproducible negative sampling against horizon positives ONLY.
Past pairs that do not recur in this horizon remain eligible negative candidates.
Retrospective roster is a benchmarking assumption, not operational deployment knowledge.
"""
import numpy as np
import torch
TAGS=('OO','OU','UO','UU')

def pair_stratum(u,v,seen_u,seen_v):
    return ('O' if u in seen_u else 'U')+('O' if v in seen_v else 'U')

def stratum_counts(snapshot,ei,seen_u=None,seen_v=None):
    su=seen_u if seen_u is not None else snapshot['train_seen_u']
    sv=seen_v if seen_v is not None else snapshot['train_seen_v']
    keys_u=snapshot['keys_u'];keys_v=snapshot['keys_v'];nu=snapshot['num_u']
    a={key:0 for key in TAGS}
    for i,j in ei.T.tolist():a[pair_stratum(keys_u[i],keys_v[j-nu],su,sv)]+=1
    return a

def matched_endpoint_negatives(snapshot,seed):
    """1:1 in each category; no future labels queried; absent edges in current horizon only."""
    ei=snapshot['edges'];nu=snapshot['num_u'];us=snapshot['keys_u'];vs=snapshot['keys_v']
    su=snapshot['train_seen_u'];sv=snapshot['train_seen_v']
    quotas=stratum_counts(snapshot,ei)
    old_u=np.array([i for i,k in enumerate(us) if k in su],dtype=int)
    new_u=np.array([i for i,k in enumerate(us) if k not in su],dtype=int)
    old_v=np.array([i+nu for i,k in enumerate(vs) if k in sv],dtype=int)
    new_v=np.array([i+nu for i,k in enumerate(vs) if k not in sv],dtype=int)
    groups={'O':(old_u,old_v),'U':(new_u,new_v)}
    positives=set(map(tuple,ei.T.tolist()))
    result=[];rng=np.random.default_rng(seed)
    for tag in TAGS:
        count=quotas[tag]
        if not count:continue
        aa=groups[tag[0]][0];bb=groups[tag[1]][1]
        blocked=sum(1 for x,y in positives if x in aa and y in bb)
        capacity=int(len(aa))*int(len(bb))-blocked
        if capacity<count:
            raise ValueError(f'Cannot sample {count} matched negatives in {tag}: capacity={capacity}')
        chosen=set(); tries=0
        while len(chosen)<count and tries < max(2000,30*count):
            # vectorized draw for large quotas; avoid global N_u*N_v matrix
            needed=count-len(chosen)
            m=min(max(64,needed*2),100000)
            ii=rng.choice(aa,m);jj=rng.choice(bb,m)
            for x,y in zip(ii,jj):
                p=(int(x),int(y))
                if p not in positives and p not in chosen:chosen.add(p)
                if len(chosen)==count:break
            tries+=m
        if len(chosen)!=count:
            for i in aa:
                for j in bb:
                    p=(int(i),int(j))
                    if p not in positives and p not in chosen:chosen.add(p)
                    if len(chosen)==count:break
                if len(chosen)==count:break
        if len(chosen)!=count:raise AssertionError(f'Incomplete {tag} negative sampling')
        result.extend(sorted(chosen))
    if len(result)!=ei.shape[1]:raise AssertionError('Negative count mismatch')
    rng.shuffle(result)
    return torch.tensor(result,dtype=torch.long).reshape(-1,2).T

def candidate_frame(snapshot,neg):
    ei=torch.cat([snapshot['edges'],neg],dim=1)
    pairs=ei.T.tolist();nu=snapshot['num_u']
    y=[1]*snapshot['edges'].shape[1]+[0]*neg.shape[1]
    return [(str(snapshot['keys_u'][u]),str(snapshot['keys_v'][v-nu]),label,
             pair_stratum(snapshot['keys_u'][u],snapshot['keys_v'][v-nu],snapshot['train_seen_u'],snapshot['train_seen_v']))
            for (u,v),label in zip(pairs,y)]
