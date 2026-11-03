# -*- coding: utf-8 -*-
"""
Analyze and aggregate results from all synthetic WBT-BGRL experiments.

Reads all JSON files from:
  results/synthetic_wbtbgrl/
  results/synthetic_wbtbgrl_temporal/

Builds a DataFrame with one row per (model_type, condition, variant, seed),
aggregates mean ± std across seeds, and exports:
  results/synthetic_analysis/comparison_table.csv    — one row per (model, condition, scope, metric)
  results/synthetic_analysis/summary_wide.csv        — wide format, mean ± std

Usage:
  python experiments/synthetic/analyze_results.py
  python experiments/synthetic/analyze_results.py --metric "ROC-AUC" "Hits@50"
  python experiments/synthetic/analyze_results.py --print_only
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIRS = {
    'basic':    PROJECT_ROOT / 'results' / 'synthetic_wbtbgrl',
    'temporal': PROJECT_ROOT / 'results' / 'synthetic_wbtbgrl_temporal',
}
OUTPUT_DIR   = PROJECT_ROOT / 'results' / 'synthetic_analysis'

PRIMARY_METRICS = [
    'Hits@10', 'Hits@20', 'Hits@50', 'Hits@100',
    'ROC-AUC', 'AP',
    'F1-Score (t=0.5)', 'Recall (t=0.5)', 'Precision (t=0.5)',
]
CONDITIONS      = ['informative', 'anti_correlated', 'orthogonal']
VARIANTS        = ['WP_WB', 'WP_NWB', 'NWP_WB', 'NWP_NWB']
SCOPES          = ['global', 'OO', 'OU', 'UU']
SCOPE_LABELS    = {'global': 'GLOBAL', 'OO': 'Old–Old (O-O)', 'OU': 'Old–New (O-U)', 'UU': 'New–New (U-U)'}


# ── edge counts from .npz files ──────────────────────────────────────────────────

# Per-type evaluation uses 1:1 (negatives == positives), same as global.
_MIN_NEG_PER_TYPE = 0  # sentinel — actual value is n_positives (1:1 ratio)


def load_test_counts() -> dict:
    """
    Read test_pair_types from each synthetic_<condition>.npz and return
    per-condition counts of (positives, negatives) per pair type.

    Both global and per-type evaluation use 1:1 sampling (negatives == positives).
    Balanced design: O-O=120, O-U=100, U-U=100 per split (TEST).
    """
    data_dir = PROJECT_ROOT / 'data' / 'synthetic'
    counts = {}
    for cond in CONDITIONS:
        npz_path = data_dir / f'synthetic_{cond}.npz'
        if not npz_path.exists():
            continue
        try:
            d = np.load(npz_path, allow_pickle=True)
            types = d['test_pair_types']
            n_oo = int((types == 'O-O').sum())
            n_ou = int((types == 'O-U').sum())
            n_uu = int((types == 'U-U').sum())
            total = n_oo + n_ou + n_uu
            counts[cond] = {
                'O-O':    {'pos': n_oo,  'neg': n_oo},
                'O-U':    {'pos': n_ou,  'neg': n_ou},
                'U-U':    {'pos': n_uu,  'neg': n_uu},
                'global': {'pos': total, 'neg': total},
            }
        except Exception as e:
            print(f"[WARN] Could not load counts from {npz_path.name}: {e}")
    return counts


# ── loading ────────────────────────────────────────────────────────────────────

def load_all_results() -> list[dict]:
    rows = []
    for model_type, results_dir in RESULTS_DIRS.items():
        if not results_dir.exists():
            print(f"[WARN] Directory not found: {results_dir}")
            continue
        json_files = list(results_dir.glob('*.json'))
        print(f"[{model_type.upper()}] Found {len(json_files)} JSON files in {results_dir}")
        for jf in json_files:
            try:
                with open(jf, encoding='utf-8') as f:
                    d = json.load(f)
                row = {
                    'model_type': model_type,
                    'condition':  d.get('condition', 'unknown'),
                    'variant':    d.get('variant',   'unknown'),
                    'seed':       d.get('seed',       -1),
                    'model_name': d.get('model',      ''),
                    'file':       jf.name,
                }
                # Global metrics (TEST split — 10% future)
                res = d.get('results', {})
                for m in PRIMARY_METRICS:
                    row[f'global_{m}'] = res.get(m, np.nan)
                # Per-type metrics (O-O / O-U / U-U)
                for prefix in ['OO', 'OU', 'UU']:
                    for m in PRIMARY_METRICS:
                        row[f'{prefix}_{m}'] = res.get(f'{prefix}_{m}', np.nan)

                rows.append(row)
            except Exception as e:
                print(f"[ERROR] Could not parse {jf.name}: {e}")
    return rows


# ── aggregation ────────────────────────────────────────────────────────────────

def aggregate(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """
    Aggregate mean ± std across seeds for each (model_type, condition, variant).
    Returns a wide DataFrame.
    """
    group_cols = ['model_type', 'condition', 'variant']
    # Build list of value columns for all scopes
    val_cols = [
        f'{scope}_{m}'
        for scope in ['global', 'OO', 'OU', 'UU']
        for m in metrics
        if f'{scope}_{m}' in df.columns
    ]

    agg_mean  = df.groupby(group_cols)[val_cols].mean().add_suffix('_mean')
    agg_std   = df.groupby(group_cols)[val_cols].std(ddof=1).add_suffix('_std')
    agg_n     = df.groupby(group_cols)[val_cols[0]].count().rename('n_seeds')

    # Interleave mean/std columns scope-by-scope for readability
    wide = pd.concat([agg_mean, agg_std, agg_n], axis=1)
    ordered_cols = []
    for col in val_cols:
        ordered_cols.append(f'{col}_mean')
        ordered_cols.append(f'{col}_std')
    ordered_cols.append('n_seeds')
    wide = wide[[c for c in ordered_cols if c in wide.columns]]

    result = wide.reset_index()
    return result


def build_long_table(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """
    Build a long-format table with columns:
      model_type, condition, variant, scope, metric, mean, std
    scope is one of: 'global', 'OO', 'OU', 'UU'
    """
    group_cols = ['model_type', 'condition', 'variant']
    scopes = [('global', 'global'), ('OO', 'OO'), ('OU', 'OU'), ('UU', 'UU')]
    rows = []
    for _, grp in df.groupby(group_cols):
        key = {c: grp[c].iloc[0] for c in group_cols}
        for scope_name, col_prefix in scopes:
            for m in metrics:
                col = f'{col_prefix}_{m}'
                if col not in df.columns:
                    continue
                vals = grp[col].dropna()
                if len(vals) == 0:
                    continue
                row = dict(key)
                row['scope']  = scope_name
                row['metric'] = m
                row['mean']   = vals.mean()
                row['std']    = vals.std(ddof=1) if len(vals) > 1 else 0.0
                row['n']      = len(vals)
                rows.append(row)
    return pd.DataFrame(rows)


# ── printing ───────────────────────────────────────────────────────────────────

def _pivot_scope(long_df, cond, scope, metrics):
    """Return a pivot table for a given scope, or None if empty."""
    sub = long_df[
        (long_df['condition'] == cond) &
        (long_df['metric'].isin(metrics)) &
        (long_df['scope'] == scope)
    ]
    if sub.empty:
        return None
    pivot = sub.pivot_table(
        index   = ['model_type', 'variant'],
        columns = 'metric',
        values  = ['mean', 'std'],
        aggfunc = 'first',
    )
    # reorder columns so metrics appear in requested order
    ordered = []
    for m in metrics:
        for stat in ('mean', 'std'):
            col = (stat, m)
            if col in pivot.columns:
                ordered.append(col)
    pivot = pivot[ordered]
    pivot.columns = [f'{m}_{s}' for s, m in pivot.columns]
    return pivot


def _format_pivot_rows(pivot: pd.DataFrame) -> str:
    """Format pivot rows as mean±std per cell, one row per (model_type, variant)."""
    # Collect (metric, mean_col, std_col) triples
    mean_cols = [c for c in pivot.columns if c.endswith('_mean')]
    metric_names = [c[: -len('_mean')] for c in mean_cols]

    header_idx = 28  # width of the index part
    col_w = 14       # width per metric column

    # Header
    lines = []
    idx_header = f"{'model_type/variant':<{header_idx}}"
    col_header = ''.join(f"{m:>{col_w}}" for m in metric_names)
    lines.append(idx_header + col_header)
    lines.append('-' * (header_idx + col_w * len(metric_names)))

    for idx, row in pivot.iterrows():
        idx_str = '/'.join(str(i) for i in (idx if isinstance(idx, tuple) else [idx]))
        values = []
        for m in metric_names:
            mean_v = row.get(f'{m}_mean', float('nan'))
            std_v  = row.get(f'{m}_std',  float('nan'))
            if np.isnan(mean_v):
                cell = '   —   '
            elif np.isnan(std_v) or std_v == 0.0:
                cell = f'{mean_v:.4f}'
            else:
                cell = f'{mean_v:.4f}±{std_v:.4f}'
            values.append(f"{cell:>{col_w}}")
        lines.append(f"{idx_str:<{header_idx}}" + ''.join(values))

    return '\n'.join(lines)


def print_summary(long_df: pd.DataFrame, metrics: list[str], counts: dict):
    for cond in CONDITIONS:
        cond_df = long_df[long_df['condition'] == cond]
        if cond_df.empty:
            continue

        print(f"\n{'='*100}")
        print(f"CONDITION: {cond.upper()}")

        # Edge counts header
        cnt = counts.get(cond, {})
        if cnt:
            parts = []
            for label, key in [('global', 'global'), ('O-O', 'O-O'), ('O-U', 'O-U'), ('U-U', 'U-U')]:
                c = cnt.get(key, {})
                if c:
                    tag = '1:1' if key == 'global' else f'pos:{c["pos"]} neg:{c["neg"]}'
                    parts.append(f"{label} ({tag})")
            print("  TEST edges  " + '  |  '.join(parts))

        for scope in SCOPES:
            label = SCOPE_LABELS[scope]
            pivot = _pivot_scope(cond_df, cond, scope, metrics)
            if pivot is None:
                continue
            print(f"\n  [{label}]")
            print(_format_pivot_rows(pivot))


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Analyze synthetic WBT-BGRL experiment results'
    )
    parser.add_argument('--metrics', nargs='+', default=PRIMARY_METRICS,
                        help='Metrics to display (default: all primary metrics)')
    parser.add_argument('--print_only', action='store_true',
                        help='Print summary to stdout without saving CSVs')
    args = parser.parse_args()

    rows = load_all_results()
    if not rows:
        print("[ERROR] No result JSON files found. Run run_experiments.py first.")
        return

    df = pd.DataFrame(rows)
    print(f"\n[LOADED] {len(df)} total runs")
    print(f"  conditions : {sorted(df['condition'].unique())}")
    print(f"  variants   : {sorted(df['variant'].unique())}")
    print(f"  model_types: {sorted(df['model_type'].unique())}")
    print(f"  seeds      : {sorted(df['seed'].unique())}")

    # Always build tables with ALL primary metrics; args.metrics only filters printing
    counts  = load_test_counts()
    long_df = build_long_table(df, PRIMARY_METRICS)
    wide_df = aggregate(df, PRIMARY_METRICS)

    print_summary(long_df, metrics=args.metrics, counts=counts)

    if not args.print_only:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        long_path = OUTPUT_DIR / 'comparison_table.csv'
        wide_path = OUTPUT_DIR / 'summary_wide.csv'
        raw_path  = OUTPUT_DIR / 'raw_results.csv'

        long_df.to_csv(long_path, index=False)
        wide_df.to_csv(wide_path, index=False)
        df.to_csv(raw_path, index=False)

        print(f"\n[SAVED]")
        print(f"  {long_path}")
        print(f"  {wide_path}")
        print(f"  {raw_path}")


if __name__ == '__main__':
    main()
