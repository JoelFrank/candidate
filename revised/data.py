"""Causal snapshots. Graph identities are raw keys; UNK is only a feature token.
Evaluation conditions on a supplied candidate entity roster (retrospective benchmark).
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
ROOT = Path(__file__).resolve().parents[1]

def module(path):
    spec=importlib.util.spec_from_file_location('loaded_'+Path(path).stem,ROOT/path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def temporal_parts(df, column, train_ratio=.8, val_ratio=.1):
    """Chronological 80/10/10 at whole calendar-DAY boundaries.

    Do not split the same calendar day between train, val, test, even when
    underlying records have different timestamps (e.g., InvoiceDate).
    """
    if not 0 < train_ratio < train_ratio+val_ratio < 1:
        raise ValueError('Invalid ratios')
    df=df.dropna(subset=[column]).sort_values(column,kind='stable').reset_index(drop=True)
    if df.empty:raise ValueError('No valid timestamps')
    days=pd.to_datetime(df[column],errors='raise').dt.normalize()
    sizes=days.value_counts(sort=False).sort_index()
    if len(sizes)<3:raise ValueError('Need at least three distinct calendar days')
    cum=sizes.to_numpy(dtype=np.int64).cumsum()
    # Nearest valid pair of cut days; independently chosen cuts could overlap.
    n=len(df); d=len(sizes)
    targets=(train_ratio*n,(train_ratio+val_ratio)*n)
    def neighbors(t):
        k=int(np.searchsorted(cum,t,side='left'))
        return range(max(1,k-12),min(d,k+13))  # boundary after days[0..i-1]
    choices=[(abs(cum[i-1]-targets[0])+abs(cum[j-1]-targets[1]),i,j)
             for i in neighbors(targets[0]) for j in neighbors(targets[1]) if 1<=i<j<d]
    if not choices:
        choices=[(abs(cum[i-1]-targets[0])+abs(cum[j-1]-targets[1]),i,j)
                 for i in range(1,d-1) for j in range(i+1,d)]
    _,i,j=min(choices)
    cuts=sizes.index[[i,j]]
    parts=[df[days<cuts[0]],df[(days>=cuts[0])&(days<cuts[1])],df[days>=cuts[1]]]
    if any(len(p)==0 for p in parts):raise ValueError('Empty temporal horizon')
    if not (parts[0][column].dt.normalize().max()<parts[1][column].dt.normalize().min()
            and parts[1][column].dt.normalize().max()<parts[2][column].dt.normalize().min()):
        raise AssertionError('Whole-day chronological boundary violated')
    return parts

def edges(df,uc,vc,us,vs):
    counts=df.groupby([uc,vc],sort=True).size()
    um={x:i for i,x in enumerate(us)};vm={x:i+len(us) for i,x in enumerate(vs)}
    ei=np.array([[um[u],vm[v]] for u,v in counts.index],dtype=np.int64).reshape(-1,2).T
    return torch.tensor(ei),torch.tensor(counts.to_numpy(),dtype=torch.float32)

def snapshot(ei,w,inputs,keys_u,keys_v):
    return dict(edges=ei,weights=w,inputs=inputs,keys_u=np.asarray(keys_u),keys_v=np.asarray(keys_v),num_u=len(keys_u),num_v=len(keys_v))

def remap_context(history,target):
    um={x:i for i,x in enumerate(target['keys_u'])};vm={x:i+target['num_u'] for i,x in enumerate(target['keys_v'])}
    counts={}
    for h in history:
        for (u,v),w in zip(h['edges'].T.tolist(),h['weights'].tolist()):
            a=h['keys_u'][u];b=h['keys_v'][v-h['num_u']]
            if a in um and b in vm:
                pair=(um[a],vm[b]);counts[pair]=counts.get(pair,0.)+w
    pairs=sorted(counts)
    return torch.tensor(pairs,dtype=torch.long).reshape(-1,2).T,torch.tensor([counts[p] for p in pairs],dtype=torch.float32)

def synthetic_train_observed_id_tokens(node_ids, train_observed, old_limit):
    """Train-relative ID tokens for BiSBM (0 denotes unknown identity).

    The generator's `N_*_OLD` is a generative cohort boundary, not proof
    that each ID appears in TRAIN.  An old-cohort node that never appeared
    in TRAIN must not receive an identity embedding in validation/test.
    Side attributes can still exist independently of this ID mapping.
    """
    ids = np.asarray(node_ids)
    if ids.ndim != 1:
        raise ValueError('BiSBM node IDs must be one-dimensional')
    if np.any(ids < 0):
        raise ValueError('BiSBM node IDs must be nonnegative')
    observed = np.isin(ids, list(train_observed))
    return torch.as_tensor(np.where(observed & (ids < old_limit), ids + 1, 0), dtype=torch.long)


def prepare(args):
    if args.dataset=='faa_sdr_ap_3y':
        from revised.faa_data import prepare_faa
        return prepare_faa(args)
    if args.dataset=='synthetic':
        gen=module('data/synthetic/generator.py');g,B,bu,bv,rng=gen.generate_bisbn(args.graph_seed)
        # Consume conditions in original order, preserving supplied seed-42 weight draws.
        weights={}
        for cond in gen.CONDITIONS:
            weights[cond]=[gen.compute_weights(g[k],cond,B,bu,bv,rng) for k in ['oo_train','val_edges','test_edges']]
        from experiments.synthetic.train_wbtbgrl import SyntheticFeatureEmbedder
        emb=SyntheticFeatureEmbedder(gen.N_U_OLD+1,gen.N_V_OLD+1,output_dim=args.hidden,id_embed_dim=32)
        # ID-embeddings are trained only for identities actually present in TRAIN.
        # Old-cohort nodes absent from TRAIN are train-relative U, hence ID 0.
        train_seen_u = set(map(int, g['train_u_ids']))
        train_seen_v = set(map(int, g['train_v_ids']))
        out=[]
        for i,s in enumerate(['train','val','test']):
            us=g[s+'_u_ids'];vs=g[s+'_v_ids'];e=g['oo_train' if s=='train' else s+'_edges']
            inp=[torch.tensor(g[s+'_u_feat']),torch.tensor(g[s+'_v_feat']),
                 synthetic_train_observed_id_tokens(us,train_seen_u,gen.N_U_OLD),
                 synthetic_train_observed_id_tokens(vs,train_seen_v,gen.N_V_OLD)]
            out.append(snapshot(torch.tensor(gen.remap_edges(e,us,vs)),torch.tensor(weights[args.condition][i]),inp,us,vs))
        return out,emb
    if args.dataset=='ecommerce':
        from experiments.ecommerce.train_wbtbgrl import FeatureEmbedder
        p=args.data or ROOT/'data/ecommerce/cleaned_data.csv';df=pd.read_csv(p)
        df=df.dropna(subset=['CustomerID','StockCode','Country']);df['StockCode']=df.StockCode.astype(str)
        df['InvoiceDate']=pd.to_datetime(df.InvoiceDate,format='%m/%d/%Y %H:%M');df['TotalPrice']=df.Quantity*df.UnitPrice;df=df[df.TotalPrice>0]
        parts=temporal_parts(df,'InvoiceDate',args.train_ratio,args.val_ratio);tr=parts[0]
        country={x:i+1 for i,x in enumerate(sorted(tr.Country.unique()))}
        pops={x:i+1 for i,x in enumerate(sorted(tr.groupby('StockCode').CustomerID.nunique().unique()))}
        emb=FeatureEmbedder(len(country)+1,len(pops)+1,8,8,args.hidden)
        uc,vc='CustomerID','StockCode'
        def raw(us,vs,h,history_index):
            a=h.groupby(uc).agg(spend=('TotalPrice','mean'),country=('Country',lambda x:x.mode().iloc[0]))
            b=h.groupby(vc).agg(price=('UnitPrice','mean'),pop=('CustomerID','nunique'))
            u=np.array([[np.log1p(a.loc[k,'spend']),country.get(a.loc[k,'country'],0)] if k in a.index else [0,0] for k in us],dtype=np.float32)
            v=np.array([[np.log1p(b.loc[k,'price']),pops.get(b.loc[k,'pop'],0)] if k in b.index else [0,0] for k in vs],dtype=np.float32)
            return u,v
    elif args.dataset=='eurocontrol':
        from experiments.eurocontrol.train_wbtbgrl import EuroControlFeatureEmbedder
        p=args.data or ROOT/'data/eurocontrol/EuroControl_flight_list_2025(jan-mar)_clean_for_graph.csv'
        df=pd.read_csv(p).dropna();df['dof']=pd.to_datetime(df.dof)
        for c in ['adep','ades','icao_operator','icao24','typecode']:df[c]=df[c].astype(str).str.strip().str.upper()
        df=df[(df.adep!='')&(df.ades!='')&(df.icao_operator!='')]
        flights=temporal_parts(df,'dof',args.train_ratio,args.val_ratio)
        def pairs(h):return pd.concat([h.rename(columns={'adep':'airport'}),h.rename(columns={'ades':'airport'})],ignore_index=True)
        parts=[pairs(h) for h in flights];tr=parts[0];uc,vc='airport','icao_operator'
        au={x:i+1 for i,x in enumerate(sorted(tr[uc].unique()))};av={x:i+1 for i,x in enumerate(sorted(tr[vc].unique()))}
        emb=EuroControlFeatureEmbedder(len(au)+1,len(av)+1,output_dim=args.hidden)
        def raw(us,vs,h,history_index):
            # Each original flight contributes one departure and one arrival incidence.
            # Preserve duplicate flights: use the explicitly retained flight history instead below.
            fh=flights[0] if history_index<2 else pd.concat(flights[:2],ignore_index=True)
            dep=fh.groupby('adep').size();arr=fh.groupby('ades').size();ug=h.groupby(uc);vg=fh.groupby(vc)
            unal=ug[vc].nunique();unac=ug.icao24.nunique();vn=vg.size();vap=h.groupby(vc)[uc].nunique();vac=vg.icao24.nunique();vt=vg.typecode.nunique()
            u=np.array([[dep.get(k,0)+arr.get(k,0),dep.get(k,0),arr.get(k,0),unal.get(k,0),unac.get(k,0)] for k in us],np.float32)
            v=np.array([[vn.get(k,0),vap.get(k,0),vac.get(k,0),vt.get(k,0)] for k in vs],np.float32)
            return np.log1p(u),np.log1p(v)
    else:
        from experiments.industry.train_wbtbgrl import IndustryFeatureEmbedder
        if not args.data:raise ValueError('Industry requires --data path (private data not supplied)')
        df=pd.read_csv(args.data) if str(args.data).endswith('.csv') else pd.read_excel(args.data)
        uc,vc='LOCALIZAÇÃO','OBSERVAÇÃO PATRIMÔNIO';tc='TIPO DO EQUIPAMENTO';dc='DATA DE ABERTURA'
        df[dc]=pd.to_datetime(df[dc],errors='coerce');df=df.dropna(subset=[uc,vc,tc,dc]);parts=temporal_parts(df,dc,args.train_ratio,args.val_ratio);tr=parts[0]
        au={x:i+1 for i,x in enumerate(sorted(tr[uc].unique()))};types={x:i+1 for i,x in enumerate(sorted(tr[tc].unique()))}
        emb=IndustryFeatureEmbedder(len(au)+1,len(types)+1,output_dim=args.hidden)
    out=[];scalers=None
    for i,part in enumerate(parts):
        h=parts[0] if i<2 else pd.concat(parts[:2],ignore_index=True)
        # Include history plus target roster; roster is never a trainable ID vocabulary.
        us=sorted(set(h[uc])|set(part[uc]));vs=sorted(set(h[vc])|set(part[vc]))
        ei,w=edges(part,uc,vc,us,vs)
        if args.dataset=='industry':
            typ=h.drop_duplicates(vc,keep='last').set_index(vc)[tc].to_dict()
            inp=[torch.tensor([au.get(k,0) for k in us]),torch.tensor([types.get(typ.get(k),0) for k in vs])]
        else:
            u,v=raw(us,vs,h,i);dims=slice(0,1) if args.dataset=='ecommerce' else slice(None)
            if scalers is None:scalers=[MinMaxScaler().fit(u[:,dims]),MinMaxScaler().fit(v[:,dims])]
            u[:,dims]=scalers[0].transform(u[:,dims]);v[:,dims]=scalers[1].transform(v[:,dims])
            inp=[torch.tensor(u),torch.tensor(v)]
            if args.dataset=='eurocontrol':inp += [torch.tensor([au.get(k,0) for k in us]),torch.tensor([av.get(k,0) for k in vs])]
        out.append(snapshot(ei,w,inp,us,vs))
    return out,emb
