"""Generate planned (NOT EXECUTED) reproducible training command matrix for FAA."""
import argparse,json,shlex
from pathlib import Path
SEEDS=[42,43,44,45,46]
WBT_VARIANTS=['NWP_NWB','NWP_WB','WP_NWB','WP_WB']
BASELINES=['BGRL','TBGRL','GBT','CCASSG']

def jobs(data='data/faa_sdr/processed/faa_sdr_ap_3y',out='results/faa_sdr_ap_3y',device='cuda'):
    res=[]
    for seed in SEEDS:
        for ssl in ('original','symmetric'):
            for var in WBT_VARIANTS:
                res.append({'model':'WBTBGRL','variant':var,'ssl_mode':ssl,'seed':seed})
        for model in BASELINES:
            res.append({'model':model,'variant':'NWP_NWB','ssl_mode':'original','seed':seed})
    for j in res:
        params=['python','-m','revised.run','--dataset','faa_sdr_ap_3y','--data',data,
                '--model',j['model'],'--variant',j['variant'],'--ssl-mode',j['ssl_mode'],
                '--seed',str(j['seed']),'--candidate-seed','2026','--negative-strategy','matched_endpoint',
                '--device',device,'--output',str(Path(out)/f"{j['model']}_{j['variant']}_{j['ssl_mode']}_seed{j['seed']}")]
        j['command']=' '.join(map(shlex.quote,params))
        j['status']='NOT_EXECUTED'
    return res

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/faa_sdr/processed/faa_sdr_ap_3y')
    p.add_argument('--output',default='experiments/faa_sdr/experiment_plan.json');p.add_argument('--device',default='cuda')
    args=p.parse_args();z=jobs(args.data,device=args.device)
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(z,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Wrote {len(z)} planned runs to {dest} (none executed)')
