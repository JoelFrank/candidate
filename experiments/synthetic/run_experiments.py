# -*- coding: utf-8 -*-
"""
Run all WBT-BGRL experiments on the synthetic BiSBM dataset.

Covers:
  - 3 weight conditions  : informative | anti_correlated | orthogonal
  - 4 model variants     : WP_WB | WP_NWB | NWP_WB | NWP_NWB
  - 2 model types        : standard (train_wbtbgrl.py) + temporal (train_wbtbgrl_temporal.py)
  - 5 seeds              : 42–46

Total runs: 3 × 4 × 2 × 5 = 120

Usage:
  python experiments/synthetic/run_experiments.py
  python experiments/synthetic/run_experiments.py --dry_run          # print cmds, do not run
  python experiments/synthetic/run_experiments.py --condition informative
  python experiments/synthetic/run_experiments.py --model_type temporal
  python experiments/synthetic/run_experiments.py --variant WP_WB --seeds 42 43
"""

import subprocess
import sys
import argparse
import time
from pathlib import Path
from itertools import product

# ── configuration ──────────────────────────────────────────────────────────────

CONDITIONS = ['informative', 'anti_correlated', 'orthogonal']
VARIANTS   = ['WP_WB', 'WP_NWB', 'NWP_WB', 'NWP_NWB']
SEEDS      = [42, 43, 44, 45, 46]

PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAIN_BASIC    = PROJECT_ROOT / 'experiments' / 'synthetic' / 'train_wbtbgrl.py'
TRAIN_TEMPORAL = PROJECT_ROOT / 'experiments' / 'synthetic' / 'train_wbtbgrl_temporal.py'

# Hyper-parameters (edit here to override defaults in the individual train scripts)
EPOCHS_ENCODER = 200
EPOCHS_DECODER = 100
HIDDEN_DIM     = 128
OUTPUT_DIM     = 64


def build_cmd(script: Path, condition: str, variant: str, seed: int) -> list:
    return [
        sys.executable, str(script),
        '--condition',      condition,
        '--variant',        variant,
        '--seed',           str(seed),
        '--epochs_encoder', str(EPOCHS_ENCODER),
        '--epochs_decoder', str(EPOCHS_DECODER),
        '--hidden_dim',     str(HIDDEN_DIM),
        '--output_dim',     str(OUTPUT_DIM),
    ]


def run_cmd(cmd: list, label: str) -> bool:
    print(f"\n{'─'*70}")
    print(f"[RUN] {label}")
    print(f"      {' '.join(cmd)}")
    t0 = time.time()
    ret = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    ok = (ret.returncode == 0)
    status = "OK" if ok else f"FAILED (rc={ret.returncode})"
    print(f"[{status}] {label}  ({elapsed:.1f}s)")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description='Run all synthetic WBT-BGRL experiments'
    )
    parser.add_argument('--conditions',  nargs='+', default=CONDITIONS,
                        choices=CONDITIONS)
    parser.add_argument('--variants',    nargs='+', default=VARIANTS,
                        choices=VARIANTS)
    parser.add_argument('--seeds',       nargs='+', type=int, default=SEEDS)
    parser.add_argument('--model_type',  nargs='+', default=['basic', 'temporal'],
                        choices=['basic', 'temporal'])
    parser.add_argument('--dry_run',     action='store_true',
                        help='Print commands without executing them')
    args = parser.parse_args()

    scripts = {
        'basic':    TRAIN_BASIC,
        'temporal': TRAIN_TEMPORAL,
    }

    # Build run list
    runs = list(product(args.conditions, args.variants, args.seeds, args.model_type))
    total = len(runs)
    print(f"\n[PLAN] {total} runs × conditions={args.conditions} "
          f"variants={args.variants} seeds={args.seeds} "
          f"model_types={args.model_type}")

    if args.dry_run:
        for i, (cond, var, seed, mtype) in enumerate(runs, 1):
            cmd = build_cmd(scripts[mtype], cond, var, seed)
            print(f"  [{i:3d}] {' '.join(cmd)}")
        return

    failed = []
    for i, (cond, var, seed, mtype) in enumerate(runs, 1):
        label = f"[{i}/{total}] {mtype.upper()} | {cond} | {var} | seed={seed}"
        cmd   = build_cmd(scripts[mtype], cond, var, seed)
        ok    = run_cmd(cmd, label)
        if not ok:
            failed.append(label)

    print(f"\n{'='*70}")
    print(f"[DONE] {total - len(failed)}/{total} runs succeeded")
    if failed:
        print("[FAILED RUNS]")
        for f in failed:
            print(f"  {f}")
    else:
        print("[ALL RUNS SUCCEEDED]")


if __name__ == '__main__':
    main()
