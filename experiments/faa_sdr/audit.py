"""Offline integrity audit for FAA prepared graph; does not require PyG or training."""
import argparse,json,hashlib
from pathlib import Path
import pandas as pd
import numpy as np

def check(base):
    base=Path(base);meta=json.loads((base/'split_manifest.json').read_text())
    p=[pd.read_csv(base/f'{n}_interactions.csv',dtype=str) for n in ('train','val','test')]
    x=[pd.read_csv(base/f'{n}_positive_pairs.csv',dtype=str) for n in ('train','val','test')]
    for name,d,e in zip(('train','val','test'),p,x):
        assert not d.isna().any().any(),name
        assert not d.duplicated(['report_id','aircraft_id','part_id']).any(),name
        assert not e.duplicated(['aircraft_id','part_id']).any(),name
        assert len(e)==len(d[['aircraft_id','part_id']].drop_duplicates()),name
        assert len(d)==meta['split_summary'][name]['interactions'],name
        assert len(e)==meta['split_summary'][name]['positive_pairs'],name
        assert set(e.stratum)==set(meta['split_summary'][name]['pair_strata'])-set(k for k,v in meta['split_summary'][name]['pair_strata'].items() if v==0)
    assert p[0].submission_day.max()<p[1].submission_day.min()<p[1].submission_day.max()<p[2].submission_day.min()
    assert len(set(p[0].report_id)&set(p[1].report_id))==0
    assert len(set(p[1].report_id)&set(p[2].report_id))==0
    tr_u=set(p[0].aircraft_id);tr_v=set(p[0].part_id)
    for d in x:
        calc=np.select([d.aircraft_id.isin(tr_u)&d.part_id.isin(tr_v),d.aircraft_id.isin(tr_u)&~d.part_id.isin(tr_v),~d.aircraft_id.isin(tr_u)&d.part_id.isin(tr_v),~d.aircraft_id.isin(tr_u)&~d.part_id.isin(tr_v)],['OO','OU','UO','UU'],default='INVALID')
        assert list(calc)==list(d.stratum),'O/U mismatch'
    weight=pd.read_csv(base/'train_weighted_edges.csv')
    calc=p[0].groupby(['aircraft_id','part_id']).report_id.nunique().rename('weight').reset_index()
    pd.testing.assert_frame_equal(weight,calc,check_dtype=False)
    assert weight.weight.gt(0).all()
    return {'result':'PASS','train':len(p[0]),'val':len(p[1]),'test':len(p[2]),
            'pairs':{n:len(d) for n,d in zip(('train','val','test'),x)},
            'strata':{n:d.stratum.value_counts().to_dict() for n,d in zip(('train','val','test'),x)},
            'dates':meta['cut_start_days'],'hashes':meta['input_sha256']}

if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--processed',required=True);a=q.parse_args()
    print(json.dumps(check(a.processed),indent=2))
