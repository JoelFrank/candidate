"""Analyze BiSBM-EASY results and create red/blue score figures."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = {'000':'WBT_000','010':'WBT_010','100':'WBT_100','111':'WBT_111'}


def ks_stat(pos: np.ndarray, neg: np.ndarray) -> float:
    values = np.sort(np.concatenate([pos, neg]))
    if len(values) == 0:
        return float('nan')
    cdf_pos = np.searchsorted(np.sort(pos), values, side='right') / max(1, len(pos))
    cdf_neg = np.searchsorted(np.sort(neg), values, side='right') / max(1, len(neg))
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def overlap_hist(pos: np.ndarray, neg: np.ndarray, bins=40) -> float:
    hp, edges = np.histogram(pos, bins=bins, range=(0,1), density=True)
    hn, _ = np.histogram(neg, bins=edges, density=True)
    widths = np.diff(edges)
    return float(np.sum(np.minimum(hp, hn) * widths))


def collect(input_dir: Path):
    rows = []
    for path in sorted(input_dir.glob('g*_s*_WBT_*')):
        res = path / 'result.json'; scores = path / 'test_scores.csv'
        if not res.exists() or not scores.exists():
            continue
        d = json.loads(res.read_text())
        cfg = d['config']; m = d['metrics']
        bits = ''.join(['1' if cfg.get('propagation') else '0', '1' if cfg.get('augmentation') else '0', '1' if cfg.get('loss_weights') else '0'])
        sc = pd.read_csv(scores)
        pos = sc.loc[sc.label == 1, 'score'].to_numpy(float)
        neg = sc.loc[sc.label == 0, 'score'].to_numpy(float)
        rows.append(dict(
            path=str(path), graph_seed=int(cfg['graph_seed']), seed=int(cfg['seed']), bits=bits, variant=LABELS[bits],
            val_auc=float(d['val_auc']), roc_auc=float(m['ROC-AUC']), ap=float(m['AP']), f1=float(m['F1']),
            precision=float(m['precision']), recall=float(m['recall']), ks=ks_stat(pos, neg), overlap=overlap_hist(pos, neg),
            pos_mean=float(pos.mean()), neg_mean=float(neg.mean()), candidate_sha256=d.get('candidate_sha256',''),
        ))
    if not rows:
        raise RuntimeError('No complete EASY runs found')
    df = pd.DataFrame(rows).sort_values(['variant','graph_seed','seed']).reset_index(drop=True)
    return df


def draw_figure(scores_csv: Path, out_png: Path, title: str):
    df = pd.read_csv(scores_csv)
    pos = df.loc[df.label == 1, 'score'].to_numpy(float)
    neg = df.loc[df.label == 0, 'score'].to_numpy(float)
    bins = np.linspace(0.0, 1.0, 31)
    fig = plt.figure(figsize=(8.5, 4.8))
    ax = fig.add_subplot(111)
    ax.hist(pos, bins=bins, alpha=0.58, color='#2563eb', label='Positive links (blue)', density=True)
    ax.hist(neg, bins=bins, alpha=0.58, color='#dc2626', label='Negative links (red)', density=True)
    ax.set_xlabel('Predicted score')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    root = Path(a.input); out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    df = collect(root)
    df.to_csv(out/'easy_by_seed.csv', index=False)
    overview = df.groupby('variant', as_index=False)[['roc_auc','ap','f1','precision','recall','ks','overlap','val_auc']].mean().sort_values('roc_auc', ascending=False)
    overview.to_csv(out/'easy_overview.csv', index=False)
    base = df[df.variant == 'WBT_000'][['graph_seed','seed','roc_auc','ap','ks']].rename(columns={'roc_auc':'roc_auc_000','ap':'ap_000','ks':'ks_000'})
    paired = df.merge(base, on=['graph_seed','seed'], how='left')
    paired['delta_auc_vs_000'] = paired['roc_auc'] - paired['roc_auc_000']
    paired['delta_ap_vs_000'] = paired['ap'] - paired['ap_000']
    paired['delta_ks_vs_000'] = paired['ks'] - paired['ks_000']
    paired.to_csv(out/'easy_paired_vs_000.csv', index=False)

    selected = {}
    figs = out/'figures'
    for variant, sub in df.groupby('variant'):
        # Selection must use validation only; never break ties using test ROC-AUC.
        best = sub.sort_values(['val_auc','graph_seed','seed'], ascending=[False,True,True]).iloc[0]
        selected[variant] = {'path': best['path'], 'graph_seed': int(best['graph_seed']), 'seed': int(best['seed']), 'val_auc': float(best['val_auc']), 'roc_auc': float(best['roc_auc'])}
        draw_figure(Path(best['path'])/'test_scores.csv', figs/f'{variant}_best.png', f'{variant} best-on-validation | graph {int(best.graph_seed)} seed {int(best.seed)}')
        # One PNG for EACH graph/seed, useful for assessing consistency.
        for _, row in sub.iterrows():
            draw_figure(Path(row['path'])/'test_scores.csv',
                        figs/f'{variant}_g{int(row.graph_seed)}_s{int(row.seed)}.png',
                        f'{variant} | graph {int(row.graph_seed)} | seed {int(row.seed)}')
        # Descriptive pooled plot. Runs are not independent graphs.
        parts = [pd.read_csv(Path(p)/'test_scores.csv') for p in sub['path']]
        pool = pd.concat(parts, ignore_index=True)
        tmp = figs/f'_{variant}_pool.csv'; pool.to_csv(tmp, index=False)
        draw_figure(tmp, figs/f'{variant}_all.png', f'{variant} pooled over all graphs/seeds')
        tmp.unlink(missing_ok=True)
    (out/'easy_selected_on_validation.json').write_text(json.dumps(selected, indent=2))
    print(f'WROTE {out}', flush=True)


if __name__ == '__main__':
    main()
