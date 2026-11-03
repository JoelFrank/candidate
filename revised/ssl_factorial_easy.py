"""Controlled, reversible SSL objective audit for WBT-BGRL project5 BiSBM-EASY.

No source model files are altered. Base model remains identical in original mode.
Factor A: unilateral vs symmetric U-V edge loss.
Factor B: corrupt ONLINE + predictor (project5) vs corrupt EMA TARGET (project4 inspired).
This is NOT a literal project4 reproduction (other implementation details differ).
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Resolve top-level revised imports in child processes launched via python revised/<script>.py.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODES = ('original', 'symmetric', 'target_corr', 'symmetric_target_corr')
VARIANTS = ('000','010')


def create_model_type(base_class, mode):
    """Return a temporary subclass; instantiate it through the EASY runner only."""
    if mode not in MODES:
        raise ValueError(mode)
    if mode == 'original':
        return base_class
    import torch
    import torch.nn.functional as F

    class AuditWBTBGRL(base_class):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._diagnostic_mode = mode
            self._last_corrupt = None

        def _apply_corruption(self, x_u, x_v, edge_index, num_nodes_u, num_nodes_v):
            vals = super()._apply_corruption(x_u, x_v, edge_index, num_nodes_u, num_nodes_v)
            if mode in ('target_corr', 'symmetric_target_corr'):
                self._last_corrupt = vals
            return vals

        def forward(self, x_u, x_v, edge_index, num_nodes_u=None, edge_weight=None, **kwargs):
            out = super().forward(x_u, x_v, edge_index, num_nodes_u, edge_weight, **kwargs)
            if mode in ('target_corr', 'symmetric_target_corr'):
                if self._last_corrupt is None:
                    raise RuntimeError('Corruption was not recorded')
                xuc, xvc, ec, wc = self._last_corrupt
                n = x_u.size(0) if num_nodes_u is None else num_nodes_u
                was_training = self.target_encoder.training
                # Project4 corrects EMA target mode to evaluation, but this change
                # is isolated to this additional corrupted-branch forward pass.
                self.target_encoder.eval()
                try:
                    with torch.no_grad():
                        cu, cv = self.target_encoder(xuc, xvc, ec,
                            wc if self.weight_propagation else None, n)
                finally:
                    self.target_encoder.train(was_training)
                out['diagnostic_h_u_corr_target'] = cu
                out['diagnostic_h_v_corr_target'] = cv
                self._last_corrupt = None
            return out

        def _reverse_cos(self, z_v, h_u, edges, weights, num_u, attractive):
            if edges.shape[1] == 0:
                return (z_v.sum() + h_u.sum()) * 0.0
            u = edges[0]; v = edges[1] - num_u
            if not bool(((u >= 0) & (u < h_u.shape[0]) & (v >= 0) & (v < z_v.shape[0])).all()):
                raise ValueError('Invalid bipartite canonical edge indices')
            cos = F.cosine_similarity(z_v[v], h_u[u], dim=-1)
            if self.weight_loss and weights is not None:
                sw = weights.sum()
                if bool(sw > 0):
                    val = (weights * cos).sum() / sw
                else:
                    val = cos.mean()
            else:
                val = cos.mean()
            return -val if attractive else val

        def compute_loss(self, outputs, num_nodes_u=None):
            n = len(outputs['z_u']) if num_nodes_u is None else num_nodes_u
            symmetric = mode in ('symmetric', 'symmetric_target_corr')
            target = mode in ('target_corr', 'symmetric_target_corr')
            att = self._compute_weighted_edge_loss(outputs['z_u'], outputs['h_v_target'],
                outputs['edge_index_aug1'], outputs['edge_weight_aug1'], n, attractive=True)
            cv = outputs['diagnostic_h_v_corr_target'] if target else outputs['z_v_corr']
            rep = self._compute_weighted_edge_loss(outputs['z_u'], cv,
                outputs['edge_index_corr'], outputs['edge_weight_corr'], n, attractive=False)
            if symmetric:
                att2 = self._reverse_cos(outputs['z_v'], outputs['h_u_target'],
                    outputs['edge_index_aug1'], outputs['edge_weight_aug1'], n, attractive=True)
                cu = outputs['diagnostic_h_u_corr_target'] if target else outputs['z_u_corr']
                rep2 = self._reverse_cos(outputs['z_v'], cu,
                    outputs['edge_index_corr'], outputs['edge_weight_corr'], n, attractive=False)
                att = (att + att2) / 2.0
                rep = (rep + rep2) / 2.0
            return (1-self.lambda_param)*att + self.lambda_param*rep

    AuditWBTBGRL.__name__ = 'AuditWBTBGRL_' + mode
    return AuditWBTBGRL


def execute_single(a):
    from revised import run_easy
    if a.mode != 'original':
        run_easy.WBTBGRL = create_model_type(run_easy.WBTBGRL, a.mode)
    # Use the EXACT existing EASY driver: same feature embedder, candidates,
    # training, validation, decoder, test scoring, and checkpoint writing.
    cfg = run_easy.parser().parse_args([])
    bits = a.bits
    cfg.graph_seed = a.graph; cfg.seed = a.seed; cfg.candidate_seed = a.candidate_seed
    cfg.variant = 'NWP_NWB' if bits == '000' else 'WP_NWB'
    cfg.propagation = bool(int(bits[0])); cfg.augmentation = bool(int(bits[1]))
    cfg.loss_weights = bool(int(bits[2]))
    cfg.encoder_epochs = a.encoder_epochs; cfg.decoder_epochs = a.decoder_epochs
    cfg.device = a.device; cfg.output = a.output; cfg.threads = a.threads
    cfg.hidden = a.hidden; cfg.dim = a.dim; cfg.patience = a.patience
    cfg.lr = a.lr; cfg.dropout = a.dropout; cfg.weight_decay = a.weight_decay
    cfg.lam = a.lam; cfg.edge_drop = a.edge_drop; cfg.feature_drop = a.feature_drop
    cfg.easy_groups = a.easy_groups; cfg.easy_noise = a.easy_noise
    run_easy.run(cfg)
    # Provenance only, not used for model selection or decisions.
    out = Path(a.output)
    (out / 'ssl_diagnostic_mode.json').write_text(json.dumps({
        'mode':a.mode, 'bits':a.bits, 'meaning':'controlled diagnostic, not a manuscript change',
        'source_sha256':hash_sources(),
    }, indent=2))


def hash_sources():
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
            ('src/models/wbtbgrl.py','revised/run_easy.py','revised/easy_negatives.py',
             'data/synthetic/generator_easy_project5.py')}


def make_command(a, mode, bits, folder):
    cmd = [sys.executable, str(Path(__file__).resolve()), '--single',
           '--graph',str(a.graph),'--seed',str(a.seed),'--mode',mode,'--bits',bits,
           '--output',str(folder),'--device',a.device,
           '--candidate-seed',str(a.candidate_seed)]
    for opt in ('encoder_epochs','decoder_epochs','threads','hidden','dim','patience',
                'lr','dropout','weight_decay','lam','edge_drop','feature_drop','easy_groups','easy_noise'):
        cmd.extend(['--'+opt.replace('_','-'),str(getattr(a,opt))])
    return cmd


def validate_completed(path, mode, bits, a):
    report = path/'result.json'; scores = path/'test_scores.csv'; modefile = path/'ssl_diagnostic_mode.json'
    if not path.exists():
        return False
    if not all(f.is_file() for f in (report,scores,modefile)):
        raise RuntimeError(f'Incomplete result at {path}. Do not overwrite automatically.')
    d = json.loads(report.read_text()); mc = json.loads(modefile.read_text())
    c = d['config']; expected_variant = 'NWP_NWB' if bits=='000' else 'WP_NWB'
    expected = dict(graph_seed=a.graph,seed=a.seed,candidate_seed=a.candidate_seed,
                    variant=expected_variant,propagation=bool(int(bits[0])),
                    augmentation=bool(int(bits[1])),loss_weights=bool(int(bits[2])),
                    encoder_epochs=a.encoder_epochs,decoder_epochs=a.decoder_epochs)
    for k,v in expected.items():
        if c.get(k)!=v:
            raise RuntimeError(f'Config mismatch {k} at {path}: {c.get(k)} != {v}')
    if mc.get('mode')!=mode or mc.get('bits')!=bits:
        raise RuntimeError(f'Mode mismatch at {path}')
    if mc.get('source_sha256') != hash_sources():
        raise RuntimeError(f'Source changed since {path} was run. Use a new output directory.')
    if d.get('negative_sampling',{}).get('policy')!='matched_easy_stage':
        raise RuntimeError('Expected corrected EASY matched negatives')
    return True


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--single',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--execute',action='store_true',help='Without this, show the plan only')
    p.add_argument('--graph',type=int,default=111)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--modes',default=','.join(MODES))
    p.add_argument('--variants',default=','.join(VARIANTS))
    p.add_argument('--mode',choices=MODES,default='original',help=argparse.SUPPRESS)
    p.add_argument('--bits',choices=VARIANTS,default='010',help=argparse.SUPPRESS)
    p.add_argument('--output',default='results/comparacion_ssl_p4_p5_easy')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--candidate-seed',type=int,default=2026)
    for k,v in [('encoder-epochs',200),('decoder-epochs',100),('threads',2),('hidden',128),('dim',64),('patience',20),('easy-groups',16)]:
        p.add_argument('--'+k,type=int,default=v)
    for k,v in [('lr',.001),('dropout',.1),('weight-decay',1e-5),('lam',.5),('edge-drop',.1),('feature-drop',.1),('easy-noise',.02)]:
        p.add_argument('--'+k,type=float,default=v)
    return p


def main():
    a=parser().parse_args()
    if a.single:
        execute_single(a)
        return
    modes = a.modes.split(','); variants=a.variants.split(',')
    if not modes or not variants or set(modes)-set(MODES) or set(variants)-set(VARIANTS) or len(modes)!=len(set(modes)) or len(variants)!=len(set(variants)):
        raise ValueError('Invalid or duplicate --modes / --variants')
    root=Path(a.output)
    source = hash_sources()
    plan={'graph':a.graph,'seed':a.seed,'modes':modes,'variants':variants,
          'source_sha256':source,'settings':{k:v for k,v in vars(a).items() if k not in ('single','execute')}}
    planfile=root/'plan.json'
    if planfile.exists() and json.loads(planfile.read_text())!=plan:
        raise RuntimeError('Existing plan differs. Use a new --output directory.')
    jobs=list(itertools.product(variants,modes))
    print(f'{len(jobs)} SSL diagnostic trainings planned. Originals unchanged.',flush=True)
    if a.execute:
        root.mkdir(parents=True,exist_ok=True)
        if not planfile.exists():planfile.write_text(json.dumps(plan,indent=2))
        (root/'logs').mkdir(exist_ok=True)
    for i,(bits,mode) in enumerate(jobs,1):
        dest=root/f'g{a.graph}_s{a.seed}_WBT_{bits}_{mode}'
        if validate_completed(dest,mode,bits,a):
            print(f'[{i}/{len(jobs)}] SKIP validated: {dest}',flush=True)
            continue
        cmd=make_command(a,mode,bits,dest)
        print(f'[{i}/{len(jobs)}] {"RUN" if a.execute else "DRY"}: {dest}',flush=True)
        if not a.execute:
            print(' '.join(cmd),flush=True)
            continue
        env=os.environ.copy();env.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        log=root/'logs'/f'WBT_{bits}_{mode}.log'
        with log.open('w') as fp:
            rc=subprocess.run(cmd,cwd=str(ROOT),env=env,stdout=fp,stderr=subprocess.STDOUT).returncode
        if rc:
            raise RuntimeError(f'Experiment stopped (exit={rc}). See {log}; incomplete result: {dest}')
        validate_completed(dest,mode,bits,a)
    if a.execute:
        print('All runs complete; run revised/ssl_factorial_analysis.py for metrics and red/blue figures.',flush=True)


if __name__=='__main__':main()
