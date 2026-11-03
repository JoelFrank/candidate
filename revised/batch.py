"""Reusable multi-seed launcher, grouped by model, SSL mode and variant.

Usage (Linux):
 python -m revised.batch --dataset faa_sdr_ap_3y --data data/faa_sdr/processed/faa_sdr_ap_3y \\
   --models BGRL GBT CCASSG TBGRL WBTBGRL --variants NWP_NWB WP_NWB WP_WB NWP_WB \\
   --ssl-modes original symmetric --device cuda --resume
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
from revised.results import DEFAULT_SEEDS, group_name, summarize, AggregationError

MODELS = ("WBTBGRL", "BGRL", "TBGRL", "GBT", "CCASSG")
VARIANTS = ("NWP_NWB", "NWP_WB", "WP_NWB", "WP_WB")


def build_jobs(a):
    for model in a.models:
        variants = a.variants if model == "WBTBGRL" else ["NWP_NWB"]
        modes = a.ssl_modes if model == "WBTBGRL" else ["original"]
        for variant in variants:
            for mode in modes:
                group = group_name(dict(dataset=a.dataset, model=model, variant=variant, ssl_mode=mode,
                                        condition=a.condition, graph_seed=a.graph_seed))
                for seed in a.seeds:
                    output = a.results_root / a.dataset / group / f"seed{seed}"
                    command = [sys.executable, "-m", "revised.run", "--dataset", a.dataset,
                               "--model", model, "--variant", variant, "--ssl-mode", mode,
                               "--seed", str(seed), "--candidate-seed", str(a.candidate_seed),
                               "--negative-strategy", a.negative_strategy, "--device", a.device,
                               "--graph-seed", str(a.graph_seed), "--condition", a.condition,
                               "--output", str(output)]
                    if a.data:
                        command += ["--data", str(a.data)]
                    for arg in ('hidden', 'dim', 'encoder_epochs', 'decoder_epochs', 'patience', 'threads',
                                'lr', 'weight_decay', 'dropout', 'lam', 'edge_drop', 'feature_drop'):
                        command += ['--' + arg.replace('_', '-'), str(getattr(a, arg))]
                    yield output, command


def run_batch(a):
    if len(set(a.seeds)) != len(a.seeds):
        raise AggregationError("Duplicate seed in --seeds")
    root = a.results_root / a.dataset
    for output, command in build_jobs(a):
        root = a.results_root / a.dataset
        result_path = output / "result.json"
        # Resume also recognizes the previous flat layout, without moving old files.
        old_path = root / f"{output.parent.name}_seed{output.name[4:]}" / "result.json"
        found = [p for p in (result_path, old_path) if p.is_file()]
        if len(found) > 1:
            raise AggregationError(f"Duplicate old/new run for same seed: {found}")
        if found:
            result_path = found[0]
            if not a.resume:
                raise AggregationError(f"Existing result, refusing overwrite: {result_path} (pass --resume)")
            content = json.loads(result_path.read_text())
            cfg = content["config"]
            if content.get('protocol','').split(';')[0] != 'history-only-v2':
                raise AggregationError(
                    f'Cannot resume legacy v1 result with v2 protocol: {result_path}. '
                    'Use a separate --results-root (default results_v2).')
            match = {"dataset": a.dataset, "model": command[command.index("--model")+1],
                     "variant": command[command.index("--variant")+1],
                     "ssl_mode": command[command.index("--ssl-mode")+1],
                     "seed": int(command[command.index("--seed")+1]),
                     "candidate_seed": a.candidate_seed,
                     "negative_strategy": a.negative_strategy}
            # With --negative-strategy auto, the config stays auto even when the resolved policy differs.
            for key in ('hidden', 'dim', 'encoder_epochs', 'decoder_epochs', 'patience', 'threads',
                        'lr', 'weight_decay', 'dropout', 'lam', 'edge_drop', 'feature_drop'):
                match[key] = getattr(a, key)
            match['graph_seed'] = a.graph_seed
            match['condition'] = a.condition
            if a.data is not None:
                match['data'] = str(a.data)
            if any(cfg.get(k) != v for k, v in match.items()):
                raise AggregationError(f"Cannot resume: incompatible saved config at {result_path}")
            print(f"SKIP complete: {output}", flush=True)
            continue
        if a.dry_run:
            print("DRY RUN:", " ".join(command), flush=True)
            continue
        print(f"RUN: {output}", flush=True)
        subprocess.run(command, check=True)
    if not a.dry_run and root.is_dir():
        outcome = summarize(root, a.seeds, strict=True)
        print(f"Summary: {root / 'summary_index.csv'} ({len(outcome['groups'])} groups)")


def parser():
    p = argparse.ArgumentParser(description="Experiment launcher and verified multi-seed aggregation")
    p.add_argument("--dataset", required=True, choices=["synthetic", "ecommerce", "eurocontrol", "industry", "faa_sdr_ap_3y"])
    p.add_argument("--data", type=Path)
    p.add_argument("--results-root", type=Path, default=Path("results_v2"))
    p.add_argument("--models", nargs="+", choices=MODELS, default=["WBTBGRL"])
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=["NWP_NWB"])
    p.add_argument("--ssl-modes", nargs="+", choices=["original", "symmetric"], default=["original"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--candidate-seed", type=int, default=2026)
    p.add_argument("--negative-strategy", choices=["auto", "matched_history", "uniform", "matched_synthetic", "matched_endpoint"], default="auto")
    p.add_argument("--device", default="cpu")
    p.add_argument("--condition", default="informative", choices=["informative", "anti_correlated", "orthogonal"])
    p.add_argument("--graph-seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", help="Skip runs with matching existing result.json; never overwrite them")
    p.add_argument("--dry-run", action="store_true")
    for flag, default in [('hidden',128), ('dim',64), ('encoder_epochs',200), ('decoder_epochs',100),
                          ('patience',20), ('threads',2)]:
        p.add_argument('--' + flag.replace('_','-'), type=int, default=default)
    for flag, default in [('lr',.001), ('weight_decay',1e-5), ('dropout',.1), ('lam',.5),
                          ('edge_drop',.1), ('feature_drop',.1)]:
        p.add_argument('--' + flag.replace('_','-'), type=float, default=default)
    return p


if __name__ == "__main__":
    run_batch(parser().parse_args())
