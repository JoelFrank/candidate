"""Private INDUSTRY project5: original vs symmetric SSL, NWB fixed.

Additive only: original src/models/wbtbgrl.py, revised/run.py and data loader
are unchanged. Do not publish individual raw maintenance records.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DATA = '/data/joel/proyecto5/data/industry/INDUSTRY-Exportacao-SS-2021-2024_P1(OK PATRIMONIO).xlsx'
BITS = ('000', '010', '111')
MODES = ('original', 'symmetric')
SOURCE_FILES = (
    'revised/run.py', 'revised/data.py', 'revised/negative_sampling.py',
    'revised/ssl_factorial_easy.py', 'src/models/wbtbgrl.py',
    'experiments/industry/train_wbtbgrl.py',
    'revised/ssl_industry_p5.py', 'revised/ssl_industry_analisis.py',
)
REQUIRED_COLUMNS = ('LOCALIZAÇÃO', 'OBSERVAÇÃO PATRIMÔNIO', 'TIPO DO EQUIPAMENTO', 'DATA DE ABERTURA')


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dependencies() -> dict[str, str]:
    missing = [x for x in SOURCE_FILES if not (ROOT/x).is_file()]
    if missing:
        raise FileNotFoundError('Faltan archivos en proyecto5: ' + ', '.join(missing))
    return {p: hash_file(ROOT/p) for p in SOURCE_FILES}


def parse_options(text: str, choices: tuple[str, ...]) -> list[str]:
    opts = text.split(',')
    if not opts or len(set(opts)) != len(opts) or any(x not in choices for x in opts):
        raise ValueError(f'Opciones invalidas/repetidas: {text!r}; validas: {choices}')
    return opts


def resolve_data(path: str) -> Path:
    p = Path(path).expanduser().resolve(strict=True)
    if not p.is_file() or p.suffix.lower() not in ('.xlsx', '.xls', '.csv'):
        raise ValueError('Dataset privado inexistente o formato no soportado (xlsx/xls/csv)')
    return p


def expected(a, bits: str, seed: int) -> dict:
    p, aug, ssl = map(int, bits)
    return dict(dataset='industry', data=str(resolve_data(a.data)), model='WBTBGRL',
                seed=seed, candidate_seed=a.candidate_seed,
                variant='NWP_NWB' if bits=='000' else 'WP_NWB',
                propagation=bool(p), augmentation=bool(aug), loss_weights=bool(ssl),
                negative_strategy='uniform', train_ratio=a.train_ratio, val_ratio=a.val_ratio,
                encoder_epochs=a.encoder_epochs, decoder_epochs=a.decoder_epochs,
                patience=a.patience, threads=a.threads, hidden=a.hidden, dim=a.dim,
                lr=a.lr, weight_decay=a.weight_decay, dropout=a.dropout, lam=a.lam,
                edge_drop=a.edge_drop, feature_drop=a.feature_drop, device=a.device)


def run_preflight(a, store=False):
    """Checks original loader semantics, WITHOUT altering or exporting workbook rows."""
    import pandas as pd
    from revised.data import temporal_parts
    path = resolve_data(a.data)
    df = pd.read_csv(path) if path.suffix.lower()=='.csv' else pd.read_excel(path, engine='openpyxl' if path.suffix.lower()=='.xlsx' else None)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f'Faltan columnas del cargador actual: {missing}. Disponible: {list(df.columns)}')
    total = len(df)
    df = df.copy()
    df['DATA DE ABERTURA'] = pd.to_datetime(df['DATA DE ABERTURA'], errors='coerce')
    clean = df.dropna(subset=REQUIRED_COLUMNS)
    removed = total-len(clean)
    if clean.empty:
        raise ValueError('No hay filas utilizables tras descartar fechas/atributos vacios')
    tr, va, te = temporal_parts(clean, 'DATA DE ABERTURA', a.train_ratio, a.val_ratio)
    if len(tr)<10 or len(va)<5 or len(te)<5:
        raise ValueError('Particiones insuficientes para evaluar ROC-AUC')
    if not (tr['DATA DE ABERTURA'].max()<va['DATA DE ABERTURA'].min() and va['DATA DE ABERTURA'].max()<te['DATA DE ABERTURA'].min()):
        raise ValueError('La division temporal no es estricta')
    keyu, keyv = REQUIRED_COLUMNS[:2]
    pairs = [{(str(u),str(v)) for u,v in zip(part[keyu],part[keyv])} for part in (tr,va,te)]
    def detail(part):
        return {'rows':len(part),'unique_positive_pairs':len(set(zip(part[keyu],part[keyv]))),
                'unique_locations':int(part[keyu].nunique()),'unique_patrimonies':int(part[keyv].nunique()),
                'first_date':str(part['DATA DE ABERTURA'].min()),'last_date':str(part['DATA DE ABERTURA'].max())}
    data = {'privacy':'aggregate_only; no records included', 'data_sha256':hash_file(path),
            'filename':path.name, 'rows_raw':total, 'rows_dropped_missing':removed,
            'rows_usable':len(clean),'train':detail(tr),'val':detail(va),'test':detail(te),
            'repeated_pairs':{'train_and_val':len(pairs[0]&pairs[1]),
                              'val_and_test':len(pairs[1]&pairs[2]),
                              'train_and_test':len(pairs[0]&pairs[2])},
            'split':'same temporal_parts function as revised/data.py, strict timestamp boundaries',
            'important_limitations':["Un mismo par puede reaparecer en distintas fechas (tarea de reaparicion temporal, no necesariamente primer enlace).",
                                     "El muestreo de negativos uniforme original puede elegir un par que sera positivo en una fecha futura: etiqueta ambigua, no usar futuro para filtrar.",
                                     "Una sola particion temporal; las semillas no son datasets independientes."]}
    print(json.dumps(data,indent=2,ensure_ascii=False),flush=True)
    if store:
        output = Path(a.output)
        output.mkdir(parents=True,exist_ok=True)
        (output/'industry_preflight.json').write_text(json.dumps(data,indent=2,ensure_ascii=False))
    return data


def sanitize_private_scores(folder: Path):
    """Delete raw pair identifiers from per-run export, keep labels/scores."""
    import pandas as pd
    fp=folder/'test_scores.csv'
    scores=pd.read_csv(fp)
    if not {'label','score'}.issubset(scores.columns):
        raise RuntimeError('Faltan label/score en archivo test_scores')
    # Same row ordering: matched candidates remain exactly comparable.
    scores[['label','score']].to_csv(fp, index=False)
    return int(len(scores))


def validate(folder: Path, a, bits: str, mode: str, seed: int, source: dict, workbook_hash: str):
    if not folder.exists():
        return False
    needed = ('result.json','test_scores.csv','data_audit.json','ssl_industry_mode.json')
    if any(not (folder/x).is_file() for x in needed):
        raise RuntimeError(f'Resultado incompleto en {folder}; no se sobrescribe automaticamente')
    result=json.loads((folder/'result.json').read_text())
    meta=json.loads((folder/'ssl_industry_mode.json').read_text())
    cfg=result.get('config',{})
    for k,v in expected(a,bits,seed).items():
        if cfg.get(k)!=v:
            raise RuntimeError(f'Configuracion distinta en {folder}: {k}={cfg.get(k)!r} != {v!r}')
    if meta.get('mode')!=mode or meta.get('bits')!=bits or meta.get('seed')!=seed:
        raise RuntimeError(f'Modo/semilla no coinciden en {folder}')
    if meta.get('sources')!=source or meta.get('workbook_sha256')!=workbook_hash:
        raise RuntimeError('Codigo o dataset modificados; use otra carpeta --output')
    if result.get('negative_sampling',{}).get('policy')!='uniform':
        raise RuntimeError('El muestreo no es el original uniforme')
    if not result.get('candidate_sha256') or 'ROC-AUC' not in result.get('metrics',{}):
        raise RuntimeError('Faltan metricas o hash de candidatos')
    import pandas as pd
    scores=pd.read_csv(folder/'test_scores.csv')
    if list(scores.columns)!=['label','score'] or len(scores)==0 or set(scores.label.unique())!={0,1}:
        raise RuntimeError('Archivo de puntuaciones no anonimizado o sin ambas clases')
    return True


def execute_one(a, sources: dict, workbook_hash: str):
    from revised import run as runner
    from revised.ssl_factorial_easy import create_model_type
    if a.mode=='symmetric':
        runner.WBTBGRL=create_model_type(runner.WBTBGRL,'symmetric')
    cfg=runner.parser().parse_args([])
    for k,v in expected(a,a.bits,a.seed).items():
        setattr(cfg,k,v)
    cfg.output=a.output
    folder=Path(a.output)
    if folder.exists():
        raise RuntimeError('Salida ya existente; no se sobrescribe: '+str(folder))
    runner.run(cfg)  # Exact project5 train/validation/test protocol and decoder NWB.
    n=sanitize_private_scores(folder)
    (folder/'ssl_industry_mode.json').write_text(json.dumps({
        'mode':a.mode,'bits':a.bits,'seed':a.seed,
        'workbook_sha256':workbook_hash,'sources':sources,'scores_private_ids_removed':True,
        'test_n':n,'privacy':'identifiers removed from test_scores.csv',
        'encoder_method':'project5 original vs experimental symmetric SSL',
        'corruption':'online for both','decoder':'NWB fixed',
    },indent=2))


def child_cmd(a,bits,mode,seed,output,workbook_hash):
    cmd=[sys.executable,str(Path(__file__).resolve()),'--single',
         '--data',str(resolve_data(a.data)), '--output',str(output),
         '--bits',bits,'--mode',mode,'--seed',str(seed),
         '--workbook-hash',workbook_hash,'--device',a.device,
         '--candidate-seed',str(a.candidate_seed)]
    for k in ('encoder_epochs','decoder_epochs','patience','threads','hidden','dim',
              'lr','weight_decay','dropout','lam','edge_drop','feature_drop','train_ratio','val_ratio'):
        cmd += ['--'+k.replace('_','-'), str(getattr(a,k))]
    return cmd


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default=DEFAULT_DATA)
    p.add_argument('--variants', default='000,010', help='000,010; optional 111 (pretraining WP, decoder remains NWB)')
    p.add_argument('--modes',default='original,symmetric')
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46])
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--output',default='results/ssl_industry_p5')
    p.add_argument('--candidate-seed',type=int,default=2026)
    p.add_argument('--preflight',action='store_true',help='Audita workbook y cortes temporales, no entrena')
    p.add_argument('--execute',action='store_true',help='Obligatorio para entrenar')
    p.add_argument('--delete-checkpoints',action='store_true')
    p.add_argument('--single',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--seed',type=int,default=42,help=argparse.SUPPRESS)
    p.add_argument('--bits',choices=BITS,default='000',help=argparse.SUPPRESS)
    p.add_argument('--mode',choices=MODES,default='original',help=argparse.SUPPRESS)
    p.add_argument('--workbook-hash',default='',help=argparse.SUPPRESS)
    for k,v in [('encoder-epochs',200),('decoder-epochs',100),('patience',20),('threads',2),('hidden',128),('dim',64)]:
        p.add_argument('--'+k,type=int,default=v)
    for k,v in [('lr',.001),('weight-decay',1e-5),('dropout',.1),('lam',.5),('edge-drop',.1),('feature-drop',.1),('train-ratio',.8),('val-ratio',.1)]:
        p.add_argument('--'+k,type=float,default=v)
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    if a.preflight:
        run_preflight(a,store=a.execute)
        return
    seeds=a.seeds
    variants=parse_options(a.variants,BITS)
    modes=parse_options(a.modes,MODES)
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError('Semillas faltantes/duplicadas')
    src=dependencies()
    data_sha=hash_file(resolve_data(a.data))
    if a.single:
        if data_sha!=a.workbook_hash:
            raise RuntimeError('Dataset cambiando entre tareas: hash SHA256 distinto')
        execute_one(a,src,data_sha)
        return
    tasks=list(itertools.product(seeds,variants,modes))
    root=Path(a.output)
    plan={'study':'project5 private INDUSTRY; same historical protocol; original vs symmetric SSL; NWB fixed',
          'workbook_sha256':data_sha,'source_sha256':src,
          'params':{k:v for k,v in vars(a).items() if k not in ('execute','single','seed','bits','mode','workbook_hash','delete_checkpoints','preflight')},
          'total':len(tasks)}
    pf=root/'plan.json'
    if pf.is_file() and json.loads(pf.read_text())!=plan:
        raise RuntimeError('Configuracion/dataset cambiados; use otra carpeta --output')
    print(f'INDUSTRY privado: {len(tasks)} entrenamientos; SSL original/simetrica; NWB fijo.',flush=True)
    if a.execute:
        # Fail early, before any costly run; retain separate preflight report.
        run_preflight(a,store=True)
        root.mkdir(parents=True,exist_ok=True)
        if not pf.is_file(): pf.write_text(json.dumps(plan,indent=2,ensure_ascii=False))
        (root/'logs').mkdir(exist_ok=True)
    completed=0
    for i,(seed,bits,mode) in enumerate(tasks,1):
        dest=root/f'industry_s{seed}_WBT_{bits}_{mode}'
        if validate(dest,a,bits,mode,seed,src,data_sha):
            if a.execute and a.delete_checkpoints:
                (dest/'checkpoint.pt').unlink(missing_ok=True)
            completed+=1
            print(f'[{i}/{len(tasks)}] SKIP verificado: {dest}',flush=True)
            continue
        print(f'[{i}/{len(tasks)}] {"RUN" if a.execute else "DRY"}: {dest}',flush=True)
        if not a.execute: continue
        env=os.environ.copy(); env.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        log=root/'logs'/f'industry_s{seed}_WBT_{bits}_{mode}.log'
        with log.open('w') as fp:
            proc=subprocess.run(child_cmd(a,bits,mode,seed,dest,data_sha),
                                cwd=str(ROOT),env=env,stdout=fp,stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError(f'Fallo entrenando (exit={proc.returncode}). Ver {log} y no borrar {dest}')
        validate(dest,a,bits,mode,seed,src,data_sha)
        if a.delete_checkpoints:
            (dest/'checkpoint.pt').unlink(missing_ok=True)
        completed+=1
        print(f'[{i}/{len(tasks)}] VERIFICADO',flush=True)
    if a.execute:
        print(f'COMPLETO {completed}/{len(tasks)}; ejecute revised/ssl_industry_analisis.py',flush=True)

if __name__=='__main__':
    main()
