"""Audit fully normalized train/validation/test candidate pools without PyG/GPU.

Examples:
 python -m revised.audit_protocol_v2 --dataset ecommerce
 python -m revised.audit_protocol_v2 --dataset eurocontrol
 python -m revised.audit_protocol_v2 --dataset industry --data data/industry_synthetic/industry_synthetic.csv
 python -m revised.audit_protocol_v2 --dataset faa_sdr_ap_3y --data data/faa_sdr/processed/faa_sdr_ap_3y
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from revised.data import temporal_parts,edges,snapshot
from revised.inductive_protocol import annotate_snapshots,matched_history_negatives,audit_pairs
ROOT=Path(__file__).resolve().parents[1]


def read_frames(dataset,source=None):
    if dataset=='faa_sdr_ap_3y':
        if not source:raise ValueError('--data required for FAA')
        base=Path(source)
        dfs=[pd.read_csv(base/f'{s}_interactions.csv',dtype=str,keep_default_na=False) for s in ('train','val','test')]
        return dfs,('aircraft_id','part_id'),None,'faa_sdr_prepared'
    if dataset=='industry':
        if not source:raise ValueError('--data required for Industry')
        p=Path(source)
        df=pd.read_csv(p) if p.suffix.lower()=='.csv' else pd.read_excel(p)
        uc,vc='LOCALIZAÇÃO','OBSERVAÇÃO PATRIMÔNIO';t='DATA DE ABERTURA'
        df[t]=pd.to_datetime(df[t],errors='coerce')
        df=df.dropna(subset=[uc,vc,'TIPO DO EQUIPAMENTO',t])
    elif dataset=='ecommerce':
        p=Path(source) if source else ROOT/'data/ecommerce/cleaned_data.csv'
        df=pd.read_csv(p)
        df=df.dropna(subset=['CustomerID','StockCode','Country'])
        df['StockCode']=df.StockCode.astype(str)
        df['InvoiceDate']=pd.to_datetime(df.InvoiceDate,format='%m/%d/%Y %H:%M')
        df['TotalPrice']=df.Quantity*df.UnitPrice
        df=df[df.TotalPrice>0]
        uc,vc='CustomerID','StockCode';t='InvoiceDate'
    elif dataset=='eurocontrol':
        p=Path(source) if source else ROOT/'data/eurocontrol/EuroControl_flight_list_2025(jan-mar)_clean_for_graph.csv'
        df=pd.read_csv(p).dropna()
        df['dof']=pd.to_datetime(df.dof)
        for col in ['adep','ades','icao_operator','icao24','typecode']:
            df[col]=df[col].astype(str).str.strip().str.upper()
        df=df[(df.adep!='')&(df.ades!='')&(df.icao_operator!='')]
        uc,vc='airport','icao_operator';t='dof'
    else:raise ValueError(f'Unsupported dataset: {dataset}')
    cuts=temporal_parts(df,t)
    if dataset=='eurocontrol':
        cuts=[pd.concat([part.rename(columns={'adep':'airport'}),part.rename(columns={'ades':'airport'})],ignore_index=True) for part in cuts]
    return cuts,(uc,vc),t,'whole_day_80_10_10'


def audit(dataset,source=None,seed=2026):
    parts,pair,t,policy=read_frames(dataset,source)
    snapshots=[];uc,vc=pair
    for i,part in enumerate(parts):
        hist=parts[0] if i<2 else pd.concat(parts[:2],ignore_index=True)
        us=sorted(set(hist[uc])|set(part[uc]));vs=sorted(set(hist[vc])|set(part[vc]))
        ei,w=edges(part,uc,vc,us,vs)
        snapshots.append(snapshot(ei,w,[],us,vs))
    annotate_snapshots(snapshots)
    summary={'dataset':dataset,'date_policy':policy,'negative_policy':'matched_history_v2',
             'seed':seed,'horizons':{},'known_nodes_from_train':{
                 'u':len(snapshots[0]['train_seen_u']),'v':len(snapshots[0]['train_seen_v'])}}
    for i,(s,part) in enumerate(zip(snapshots,parts)):
        neg=matched_history_negatives(s,seed+i)
        d=audit_pairs(s,neg)
        d['raw_interactions']=len(part)
        d['node_roster_u']=s['num_u'];d['node_roster_v']=s['num_v']
        d['candidate_sha256']=hashlib.sha256(s['edges'].numpy().tobytes()+neg.numpy().tobytes()).hexdigest()
        if t:
            d['start']=str(part[t].min());d['end']=str(part[t].max());d['min_day']=str(part[t].dt.date.min());d['max_day']=str(part[t].dt.date.max())
        summary['horizons'][('train','val','test')[i]]=d
    return summary


def parser():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=['industry','ecommerce','eurocontrol','faa_sdr_ap_3y'])
    p.add_argument('--data');p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--output',default='results/audit_protocol_v2.json');return p

if __name__=='__main__':
    a=parser().parse_args(); report=audit(a.dataset,a.data,a.seed)
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'dataset':a.dataset,'counts':{k:v['positive_by_stratum'] for k,v in report['horizons'].items()},
                      'report':str(dest)},indent=2))
