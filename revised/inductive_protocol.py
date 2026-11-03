"""Causal, train-relative OO/OU/UO/UU evaluation for all datasets.

Evaluation labels are horizon-specific: absence of a report in the current
horizon is a negative, including historical pairs that do not recur. Positive
labels in the target horizon are used ONLY by the offline candidate generator.
The evaluation roster is retrospective, not a deployment-time node oracle.
"""
from collections import Counter
import hashlib
import json
import numpy as np
import torch

TAGS = ('OO','OU','UO','UU')
HISTORIES = ('SS','SN','NS','NN')


def _observed_keys(s):
    """Entities that actually participate in positive edges, not idle roster IDs."""
    e=s['edges'].detach().cpu().numpy()
    return (set(s['keys_u'][e[0]].tolist()),
            set(s['keys_v'][e[1]-s['num_u']].tolist()))


def annotate_snapshots(snapshots):
    if len(snapshots)!=3:
        raise ValueError('Expected exactly train/val/test')
    train_u,train_v = _observed_keys(snapshots[0])
    val_u,val_v = _observed_keys(snapshots[1])
    for i,s in enumerate(snapshots):
        s['train_seen_u'] = frozenset(train_u)
        s['train_seen_v'] = frozenset(train_v)
        s['pre_horizon_seen_u'] = frozenset(train_u | (val_u if i==2 else set()))
        s['pre_horizon_seen_v'] = frozenset(train_v | (val_v if i==2 else set()))
        s['horizon_name'] = ('train','val','test')[i]
    return snapshots


def pair_group(s,u,v):
    keys_u=s['keys_u'];keys_v=s['keys_v'];nu=s['num_u']
    a=keys_u[int(u)]; b=keys_v[int(v)-nu]
    tag=('O' if a in s['train_seen_u'] else 'U')+('O' if b in s['train_seen_v'] else 'U')
    ctx=('S' if a in s['pre_horizon_seen_u'] else 'N')+('S' if b in s['pre_horizon_seen_v'] else 'N')
    return tag,ctx


def group_counts(s,edges,subgroup=False):
    c=Counter(pair_group(s,u,v) for u,v in edges.T.tolist())
    if subgroup:
        return {f'{tag}:{ctx}':int(n) for (tag,ctx),n in sorted(c.items())}
    return {k:int(sum(n for (t,h),n in c.items() if t==k)) for k in TAGS}


def _categories(s):
    us=s['keys_u'];vs=s['keys_v'];nu=s['num_u']
    def flags(a,train,pre):
        return ('O' if a in train else 'U','S' if a in pre else 'N')
    ug={};vg={}
    for i,a in enumerate(us):ug.setdefault(flags(a,s['train_seen_u'],s['pre_horizon_seen_u']),[]).append(i)
    for i,b in enumerate(vs):vg.setdefault(flags(b,s['train_seen_v'],s['pre_horizon_seen_v']),[]).append(i+nu)
    return {x:np.asarray(ids,dtype=np.int64) for x,ids in ug.items()}, {x:np.asarray(ids,dtype=np.int64) for x,ids in vg.items()}


def _draw(aa,bb,quota,blocked,rng,tag):
    set_aa=set(aa.tolist()); set_bb=set(bb.tolist())
    capacity=int(len(aa))*int(len(bb))-sum(1 for (u,v) in blocked if u in set_aa and v in set_bb)
    if quota>capacity:
        raise ValueError(f'Insufficient matched negatives {tag}: requested {quota}, capacity {capacity}; do not rebalance silently')
    seen=set();trials=0
    while len(seen)<quota and trials<max(1500,40*quota):
        n=min(max(128,2*(quota-len(seen))),100000)
        us=rng.choice(aa,size=n); vs=rng.choice(bb,size=n)
        for u,v in zip(us,vs):
            pair=(int(u),int(v))
            if pair not in blocked and pair not in seen:seen.add(pair)
            if len(seen)==quota:break
        trials+=n
    if len(seen)<quota:
        for u in aa:
            for v in bb:
                pair=int(u),int(v)
                if pair not in blocked and pair not in seen:seen.add(pair)
                if len(seen)==quota:break
            if len(seen)==quota:break
    if len(seen)!=quota:raise RuntimeError(f'Sampling exhausted for {tag}')
    return sorted(seen)


def matched_history_negatives(s,seed,additional_blocked=None):
    """1:1 per (train-relative OO/OU/UO/UU, pre-horizon status SS/SN/NS/NN).

    By default, only current-horizon positives are forbidden; historical
    pairs that do not recur may be sampled as horizon-specific negatives.
    Optional additional_blocked is used only for static, globally labeled
    synthetic graphs (BiSBM), never for real temporal event datasets.
    """
    pos=s['edges']; counts=Counter(pair_group(s,u,v) for u,v in pos.T.tolist())
    blocked=set(map(tuple,pos.T.tolist()))
    if additional_blocked is not None:
        blocked.update((int(u), int(v)) for u, v in additional_blocked)
    groups_u,groups_v=_categories(s)
    rng=np.random.default_rng(seed)
    selected=[]
    for (stratum,ctx),quota in sorted(counts.items()):
        u_group=(stratum[0],ctx[0]);v_group=(stratum[1],ctx[1]);
        aa=groups_u.get(u_group,np.array([],dtype=np.int64));bb=groups_v.get(v_group,np.array([],dtype=np.int64))
        selected.extend(_draw(aa,bb,quota,blocked,rng,f'{stratum}:{ctx}'))
    if len(selected)!=pos.shape[1] or len(set(selected))!=len(selected):raise AssertionError('Negative count/uniqueness failure')
    rng.shuffle(selected)
    return torch.tensor(selected,dtype=torch.long).reshape(-1,2).T



def matched_static_bisbm_negatives(s, all_snapshots, seed):
    """Matched 1:1 negatives for the *static* BiSBM graph.

    Positive edges in any of the three held-out splits are true static edges,
    and cannot be negatives in any split. Their identity is used ONLY to
    exclude impossible negatives during offline benchmark construction, not as
    input features, training edges, or graph context. Real event datasets
    retain horizon-specific label semantics.
    """
    global_pos = set()
    for h in all_snapshots:
        for u, v in h['edges'].T.tolist():
            global_pos.add((h['keys_u'][int(u)].item() if hasattr(h['keys_u'][int(u)], 'item') else h['keys_u'][int(u)],
                            h['keys_v'][int(v)-h['num_u']].item() if hasattr(h['keys_v'][int(v)-h['num_u']], 'item') else h['keys_v'][int(v)-h['num_u']]))
    umap = {key: i for i,key in enumerate(s['keys_u'])}
    vmap = {key: i+s['num_u'] for i,key in enumerate(s['keys_v'])}
    blocked = ((umap[u], vmap[v]) for u,v in global_pos if u in umap and v in vmap)
    return matched_history_negatives(s, seed, additional_blocked=blocked)

def candidate_records(s,neg):
    p=s['edges'].T.tolist();n=neg.T.tolist()
    records=[]
    for label,pairs in ((1,p),(0,n)):
        for u,v in pairs:
            tag,context=pair_group(s,u,v)
            records.append({'u':str(s['keys_u'][u]),'v':str(s['keys_v'][v-s['num_u']]),
                            'label':label,'stratum':tag,'history_status':context})
    return records


def audit_pairs(s,neg):
    pos=set(map(tuple,s['edges'].T.tolist()));ns=list(map(tuple,neg.T.tolist()))
    if len(ns)!=len(set(ns)) or bool(pos.intersection(ns)):
        raise AssertionError('Invalid or positive-overlapping negatives')
    p_groups=group_counts(s,s['edges'],True); n_groups=group_counts(s,neg,True)
    if p_groups!=n_groups:raise AssertionError(f'Historical subgroup imbalance: {p_groups} != {n_groups}')
    return {'positive_by_stratum':group_counts(s,s['edges']),
            'negative_by_stratum':group_counts(s,neg),
            'positive_by_history':p_groups,'negative_by_history':n_groups,
            'positive_count':len(pos),'negative_count':len(ns)}
