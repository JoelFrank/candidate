"""Fixed planned grid. Never select a configuration using test scores."""
import argparse,itertools,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',default='results/revision_grid');p.add_argument('--execute',action='store_true');p.add_argument('--pilot',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[1]
graphs=[42,43,44] if a.pilot else list(range(42,52));seeds=[42,43] if a.pilot else [42,43,44,45,46]
conditions=['informative'] if a.pilot else ['informative','anti_correlated','orthogonal']
for g,s,c,v in itertools.product(graphs,seeds,conditions,['WP_WB','WP_NWB','NWP_WB','NWP_NWB']):
 out=Path(a.output)/f'g{g}_s{s}_{c}_{v}'
 cmd=[sys.executable,str(root/'revised/run.py'),'--graph-seed',str(g),'--seed',str(s),'--condition',c,'--variant',v,'--device',a.device,'--output',str(out)]
 print(' '.join(cmd),flush=True)
 if a.execute:
  out.mkdir(parents=True,exist_ok=True)
  with open(out/'run.log','w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
