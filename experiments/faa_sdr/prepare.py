"""FAA-SDR-AP-3Y deterministic benchmark builder. No learned component; no future labels in features.

Input: unmodified annual FAA CSV files; retain SHA256 for reproducibility.
Frozen optional as-of day avoids silently including subsequent 2026 updates.
Data availability is approximated by SubmissionDate, not guaranteed FAA publication date.
"""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = ['AircraftMake','AircraftSerialNumber','PartNumber','OperatorControlNumber','DifficultyDate','SubmissionDate']
UNKNOWN = {'', 'UNKNOWN','UNK','UNKOWN','NOTREPORTED','NOTREPORTE','NOT REPORTED',
           'N/A','NA','NULL','NONE','NOT AVAILABLE','UNAVAILABLE','N/R','NOT PROVIDED',
           'NOT KNOWN','UNKN','UNKNOW','NIL','-','--','---','?',
           # Exact variants observed in these FAA exports; do not use fuzzy matching,
           # which could discard legitimate short manufacturer part codes.
           'UKNKNOWN','UNNOWN','UKNOWN','UNNKOWN','UNKNWON','UNKNKOWN',
           'UNKNNOWN','UNKJNOWN','UKNKOWN','UNKONWN','UNNKNOWN',
           'KUNKNOWN','UUNKNOWN','UNKNONWN','UNKNO','UNKNOWM',
           'NOTAVAILABLE','NOTLISTED','NPN'}
STRATA = ('OO','OU','UO','UU')

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(1024*1024), b''):h.update(c)
    return h.hexdigest()

def sha256_frame(df, cols):
    s=df[cols].astype(str).to_csv(index=False,lineterminator='\n')
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def canon(x):
    if pd.isna(x):return ''
    value=re.sub(r'\s+',' ',str(x).strip().upper())
    # Preserve internal punctuation; do not remove leading zeros or merge distinct serials.
    return '' if value in UNKNOWN else value

def read_annual(file):
    err=None
    for encoding in ['utf-8-sig','cp1252']:
        try:return pd.read_csv(file, encoding=encoding, dtype=str, keep_default_na=False, low_memory=False)
        except UnicodeDecodeError as e:err=e
    raise err

def parse_dates(series, kind):
    """Parse FAA calendar dates deterministically without mixing offset-aware and naive timestamps.

    SubmissionDate contains an ISO timestamp with an explicit timezone offset. We deliberately
    use its LOCAL calendar day (first ten characters), avoiding UTC conversion across midnight.
    DifficultyDate is exported as U.S. MM/DD/YYYY.
    """
    vals=series.astype(str).str.strip()
    if kind=='submission':
        valid=vals.str.match(r'^\d{4}-\d{2}-\d{2}(?:T|$)')
        return pd.to_datetime(vals.where(valid).str[:10], format='%Y-%m-%d', errors='coerce')
    if kind=='difficulty':
        return pd.to_datetime(vals,format='%m/%d/%Y',errors='coerce')
    raise ValueError(kind)

def nearest_daily_splits(df, train_ratio=.8, val_ratio=.1):
    if not 0 < train_ratio < train_ratio+val_ratio < 1:raise ValueError('Invalid ratios')
    daily=df.groupby('submission_day',sort=True).size()
    c=daily.cumsum().to_numpy(); days=daily.index.to_numpy()
    # boundary is the FIRST day of the following horizon; require all horizons nonempty
    options=np.arange(1,len(days))
    if len(options)<2:raise ValueError('At least 3 distinct submission days are required')
    ideal1=len(df)*train_ratio;ideal2=len(df)*(train_ratio+val_ratio)
    best=None
    for a in options[:-1]:
        # Find the closest feasible b after a (not necessarily same day/event count)
        feasible=options[options>a]
        b=feasible[np.argmin(np.abs(c[feasible-1]-ideal2))]
        obj=(abs(int(c[a-1])-ideal1)+abs(int(c[b-1])-ideal2),abs(int(c[a-1])-ideal1),int(a),int(b))
        if best is None or obj<best[0]:best=(obj,a,b)
    _,a,b=best
    cut1=pd.Timestamp(days[a]);cut2=pd.Timestamp(days[b])
    p=[df[df.submission_day<cut1].copy(),df[(df.submission_day>=cut1)&(df.submission_day<cut2)].copy(),df[df.submission_day>=cut2].copy()]
    if any(x.empty for x in p):raise AssertionError('Empty horizon')
    return p, (cut1,cut2)

def pair_strata(frame, train_u, train_v):
    ou=frame.aircraft_id.isin(train_u);ov=frame.part_id.isin(train_v)
    return np.select([ou&ov,ou&~ov,~ou&ov,~ou&~ov],STRATA,default='INVALID')

def prepare(files, output, as_of='2026-10-26', part_make=False):
    paths=[Path(x) for x in files]
    for p in paths:
        if not p.is_file():raise FileNotFoundError(f'FAA original CSV absent: {p}')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    original=[];raw_hashes={};column_errors=[];row_sizes={}
    for path in paths:
        d=read_annual(path);row_sizes[path.name]=len(d);raw_hashes[path.name]=sha256_file(path)
        miss=set(REQUIRED)-set(d.columns)
        if part_make and 'PartMake' not in d.columns:miss.add('PartMake')
        if miss:column_errors.append((path.name,sorted(miss)))
        d['source_file']=path.name;d['source_row']=np.arange(2,len(d)+2)
        original.append(d)
    if column_errors:raise ValueError(f'FAA missing columns: {column_errors}')
    df=pd.concat(original,ignore_index=True)
    report={'input_rows':row_sizes,'input_sha256':raw_hashes,'input_total':len(df),'filters':{},
            'normalization':'strip/uppercase/collapse whitespace; preserve punctuation and leading zeros',
            'date_policy':'SubmissionDate is an imperfect proxy for data availability; same-day boundaries',
            'as_of':as_of,'part_definition':'PartMake::PartNumber' if part_make else 'PartNumber',
            'duplicates_policy':'Exclude report IDs with conflicting aircraft IDs or submission days; one report per aircraft-part',
            'placeholder_set':sorted(UNKNOWN)}
    # Retain original column values for audits, never as model inputs.
    keys=['AircraftMake','AircraftSerialNumber','PartNumber','OperatorControlNumber']+(['PartMake'] if part_make else [])
    for k in keys:df['_'+k]=df[k].map(canon)
    df['aircraft_id']=df['_AircraftMake']+'::'+df['_AircraftSerialNumber']
    df['part_id']=(df['_PartMake']+'::' if part_make else '')+df['_PartNumber']
    missing=np.zeros(len(df),dtype=bool)
    for k in keys:missing|=df['_'+k].eq('').to_numpy()
    audit_cols=['source_file','source_row','AircraftMake','AircraftSerialNumber','PartNumber',
                'OperatorControlNumber','DifficultyDate','SubmissionDate']
    dropped=[]
    def record_excluded(rows, reason):
        if len(rows):
            audit=rows[audit_cols].copy()
            audit.insert(0,'exclusion_reason',reason)
            dropped.append(audit)
    record_excluded(df.loc[missing], 'unidentifiable_identifiers')
    report['filters']['unidentifiable_identifiers']=int(missing.sum());df=df.loc[~missing].copy()
    df['submission_day']=parse_dates(df.SubmissionDate, 'submission')
    df['difficulty_day']=parse_dates(df.DifficultyDate, 'difficulty')
    invalid=df.submission_day.isna()|df.difficulty_day.isna()
    record_excluded(df.loc[invalid], 'invalid_dates')
    report['filters']['invalid_dates']=int(invalid.sum());df=df.loc[~invalid].copy()
    reversed_dates=df.difficulty_day>df.submission_day
    record_excluded(df.loc[reversed_dates], 'difficulty_after_submission')
    report['filters']['difficulty_after_submission']=int(reversed_dates.sum());df=df.loc[~reversed_dates].copy()
    if as_of:
        after=df.submission_day>pd.Timestamp(as_of)
        record_excluded(df.loc[after], 'submission_after_as_of')
        report['filters']['submission_after_as_of']=int(after.sum());df=df.loc[~after].copy()
    # Different dates/aircraft for same control number indicate non-unambiguous report identity.
    by=df.groupby('_OperatorControlNumber',sort=False)
    inconsistent=by.aircraft_id.transform('nunique').gt(1)|by.submission_day.transform('nunique').gt(1)
    report['filters']['inconsistent_report_id_rows']=int(inconsistent.sum())
    report['filters']['inconsistent_report_ids']=int(df.loc[inconsistent,'_OperatorControlNumber'].nunique())
    record_excluded(df.loc[inconsistent], 'inconsistent_report_id')
    df=df.loc[~inconsistent].copy()
    df=df.sort_values(['submission_day','_OperatorControlNumber','aircraft_id','part_id','source_file','source_row'],kind='stable')
    repeated=df.duplicated(['_OperatorControlNumber','aircraft_id','part_id'],keep='first')
    record_excluded(df.loc[repeated], 'duplicate_report_pair')
    report['filters']['duplicate_report_pair_rows']=int(repeated.sum());df=df.loc[~repeated].copy()
    if dropped:pd.concat(dropped,ignore_index=True).to_csv(output/'excluded_rows_audit.csv',index=False)
    df=df.rename(columns={'_OperatorControlNumber':'report_id'})
    if df.empty:raise ValueError('FAA contains no valid report-pair interactions')
    parts,cuts=nearest_daily_splits(df)
    train_u=set(parts[0].aircraft_id);train_v=set(parts[0].part_id)
    summary={};pairs_by_split={}
    for i,(name,p) in enumerate(zip(['train','val','test'],parts)):
        p=p[['report_id','aircraft_id','part_id','submission_day','difficulty_day','source_file','source_row']].copy()
        p['stratum']=pair_strata(p,train_u,train_v)
        p['submission_day']=p.submission_day.dt.strftime('%Y-%m-%d');p['difficulty_day']=p.difficulty_day.dt.strftime('%Y-%m-%d')
        p.to_csv(output/f'{name}_interactions.csv',index=False)
        x=p.drop_duplicates(['aircraft_id','part_id']).copy()
        x[['aircraft_id','part_id','stratum']].to_csv(output/f'{name}_positive_pairs.csv',index=False)
        pairs_by_split[name]=x
        summary[name]={'interactions':len(p),'positive_pairs':len(x),'first_day':p.submission_day.min(),'last_day':p.submission_day.max(),
                       'pair_strata':{s:int((x.stratum==s).sum()) for s in STRATA},
                       'unique_aircraft':int(p.aircraft_id.nunique()),'unique_parts':int(p.part_id.nunique()),
                       'sha256_interactions':sha256_frame(p,['report_id','aircraft_id','part_id','submission_day']),
                       'sha256_positive_pairs':sha256_frame(x,['aircraft_id','part_id','stratum'])}
    w=parts[0].groupby(['aircraft_id','part_id'],sort=True).report_id.nunique().rename('weight').reset_index()
    w.to_csv(output/'train_weighted_edges.csv',index=False)
    q=w.weight
    weight_stats={'count':len(w),'mean':float(q.mean()),'median':float(q.median()),'p90':float(q.quantile(.9)),
                  'p99':float(q.quantile(.99)),'maximum':int(q.max()),'fraction_weight_1':float((q==1).mean()),
                  'sum_weights':int(q.sum()),'source':'distinct OperatorControlNumber exclusively in train'}
    (output/'audit_weights.json').write_text(json.dumps(weight_stats,indent=2))
    valtest=set(pd.concat(parts[1:]).aircraft_id);vvaltest=set(pd.concat(parts[1:]).part_id)
    test=parts[2];uv=set(parts[1].aircraft_id)|train_u;vv=set(parts[1].part_id)|train_v
    uu=test.drop_duplicates(['aircraft_id','part_id']);uu=uu[(~uu.aircraft_id.isin(uv))&(~uu.part_id.isin(vv))]
    stats={'train_aircraft_seen':len(train_u),'train_parts_seen':len(train_v),
           'test_aircraft_new':len(set(test.aircraft_id)-train_u),
           'test_parts_new':len(set(test.part_id)-train_v),
           'test_uu_both_new_vs_train_and_val':len(uu),
           'test_uu_vs_train':summary['test']['pair_strata']['UU']}
    report['reference_counts']={'train_events_expected':52146,'train_pairs_expected':43293,'val_events_expected':6530,
                 'val_UU_expected':410,'test_events_expected':6488,'test_UU_expected':478}
    observed={'train_events_expected':summary['train']['interactions'],'train_pairs_expected':summary['train']['positive_pairs'],
       'val_events_expected':summary['val']['interactions'],'val_UU_expected':summary['val']['pair_strata']['UU'],
       'test_events_expected':summary['test']['interactions'],'test_UU_expected':summary['test']['pair_strata']['UU']}
    report['reference_differences']={key:observed[key]-value for key,value in report['reference_counts'].items()}
    report['cut_start_days']={'val':cuts[0].strftime('%Y-%m-%d'),'test':cuts[1].strftime('%Y-%m-%d')}
    report['split_summary']=summary;report['node_counts']=stats
    report['remaining_interactions']=len(df)
    report['combined_event_hash']=sha256_frame(df,['report_id','aircraft_id','part_id','submission_day'])
    (output/'split_manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    (output/'audit_inductive.csv').write_text('split,stratum,positive_pairs\n'+''.join(f'{n},{s},{summary[n]["pair_strata"][s]}\n' for n in ['train','val','test'] for s in STRATA))
    # Mapping is transparent; not a trainable vocabulary for new identities.
    roster={'train_aircraft':sorted(train_u),'train_parts':sorted(train_v),
            'all_aircraft':sorted(set(df.aircraft_id)),'all_parts':sorted(set(df.part_id))}
    (output/'node_mapping.json').write_text(json.dumps(roster,ensure_ascii=False))
    return report

def main():
    p=argparse.ArgumentParser(description='Build FAA SDR Aircraft–Part benchmark (no training)')
    p.add_argument('--input',nargs='+',required=True,help='3 raw FAA SDR-2024/25/26 CSV')
    p.add_argument('--output',default='data/faa_sdr/processed/faa_sdr_ap_3y')
    p.add_argument('--as-of',default='2026-10-26',help='snapshot submission cutoff, or blank to disable')
    p.add_argument('--part-make',action='store_true',help='Sensitivity only: FAA-SDR-APM-3Y')
    a=p.parse_args();r=prepare(a.input,a.output,a.as_of,a.part_make)
    print(json.dumps({'input_total':r['input_total'],'remaining_interactions':r['remaining_interactions'],
                      'split_summary':r['split_summary'],'reference_differences':r['reference_differences']},indent=2))
if __name__=='__main__':main()
