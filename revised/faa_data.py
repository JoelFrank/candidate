"""FAA prepared benchmark adapter for the shared history-only runner."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from revised.faa_features import FAAFeatureEmbedder

def prepare_faa(args):
    from revised.data import snapshot, edges
    if not args.data:raise ValueError('FAA requires --data pointing to processed FAA directory')
    base=Path(args.data)
    manifest=base/'split_manifest.json'
    if not manifest.is_file():raise FileNotFoundError(f'Prepare FAA files first; absent {manifest}')
    meta=json.loads(manifest.read_text())
    parts=[]
    for name in ['train','val','test']:
        p=base/f'{name}_interactions.csv'
        if not p.is_file():raise FileNotFoundError(str(p))
        d=pd.read_csv(p,dtype=str,keep_default_na=False)
        assert len(d)==meta['split_summary'][name]['interactions']
        parts.append(d)
    train=parts[0]
    au={k:i+1 for i,k in enumerate(sorted(train.aircraft_id.unique()))}
    av={k:i+1 for i,k in enumerate(sorted(train.part_id.unique()))}
    emb=FAAFeatureEmbedder(len(au),len(av),args.hidden)
    out=[]
    # Include target roster (retrospective evaluation only); never use its labels as context.
    for i,part in enumerate(parts):
        hist=parts[0] if i<2 else pd.concat(parts[:2],ignore_index=True)
        us=sorted(set(hist.aircraft_id)|set(part.aircraft_id));vs=sorted(set(hist.part_id)|set(part.part_id))
        ei,w=edges(part,'aircraft_id','part_id',us,vs)
        inp=[torch.tensor([au.get(k,0) for k in us],dtype=torch.long),
             torch.tensor([av.get(k,0) for k in vs],dtype=torch.long)]
        s=snapshot(ei,w,inp,us,vs)
        s['train_seen_u']=frozenset(au);s['train_seen_v']=frozenset(av)
        s['faa_horizon']=['train','val','test'][i]
        out.append(s)
    return out,emb
