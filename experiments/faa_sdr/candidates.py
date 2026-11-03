"""Generate fixed per-seed FAA candidate sets without requiring torch_geometric.

Candidate rosters are retrospective; their novel entities are not a deployment-time
forecast of which specific aircraft or parts will appear in future reporting.
"""
import argparse, json, hashlib
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import torch
from revised.faa_data import prepare_faa
from revised.faa_sampling import TAGS, stratum_counts, matched_endpoint_negatives, candidate_frame


def generate(processed, destination, candidate_seed=2026, seeds=(42,43,44,45,46)):
    processed=Path(processed)
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    snapshots,_=prepare_faa(SimpleNamespace(data=str(processed),hidden=8))
    outputs=[]
    for seed in seeds:
        for i,name in enumerate(('train','val','test')):
            snapshot=snapshots[i]
            # The same candidate_seed as revised.run; replicate across training seeds
            # by default to enable paired model comparisons.
            neg=matched_endpoint_negatives(snapshot,candidate_seed+i)
            frame=pd.DataFrame(candidate_frame(snapshot,neg),columns=('u','v','label','stratum'))
            assert not frame.duplicated(['u','v','label']).any()
            assert frame.loc[frame.label.eq(1),['u','v']].merge(frame.loc[frame.label.eq(0),['u','v']],on=['u','v']).empty
            observed=frame.groupby(['stratum','label']).size()
            for tag in TAGS:
                assert observed.get((tag,0),0)==observed.get((tag,1),0),tag
            path=destination/f'{name}_candidates_seed{seed}.csv'
            frame.to_csv(path,index=False)
            checksum=hashlib.sha256(path.read_bytes()).hexdigest()
            outputs.append({'seed':seed,'horizon':name,'positive':int(frame.label.sum()),'negative':int((frame.label==0).sum()),'sha256':checksum,'file':path.name,
                            'strata':{tag:int(((frame.stratum==tag)&(frame.label==1)).sum()) for tag in TAGS}})
    obj={'candidate_seed':candidate_seed,'training_seeds':list(seeds),'note':'Identical candidate pairs for all models and seeds; independent training initialization only','files':outputs}
    (destination/'candidate_manifest.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
    return obj

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--processed',required=True);p.add_argument('--output',required=True)
    p.add_argument('--candidate-seed',type=int,default=2026);a=p.parse_args()
    obj=generate(a.processed,a.output,a.candidate_seed)
    print(json.dumps({'count':len(obj['files']),'example':obj['files'][-1]},indent=2))
