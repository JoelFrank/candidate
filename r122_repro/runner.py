"""R1-22: public reproducibility audit, no model training."""

import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import re
import shlex
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
YEARS = (2024, 2025, 2026)
SEEDS = (42, 43, 44, 45, 46)
VARIANTS = ("NWP_NWB", "NWP_WB", "WP_NWB", "WP_WB")
MODES = ("original", "symmetric")
BASELINES = ("BGRL", "TBGRL", "GBT", "CCASSG")
DATASETS = ("faa_sdr_ap_3y",)

# SHA-256 recorded in the previously audited R1-19/R1-24 plan.
# These identify the exact processed FAA benchmark.
FAA_REFERENCE = {
    "split_manifest.json":
        "c215400920fb3950d55ee5abba8f6351740c0460eec5e5d06a2f7f7733692e54",
    "train_interactions.csv":
        "ca730146ae6c639ef000b719ba7afdf4b67b7af5f53873ccbcbbd559e2a944e7",
    "val_interactions.csv":
        "06f783021f160582060b7b550a71ce5df7da97784d5b4827ad279d6a44739fd8",
    "test_interactions.csv":
        "f5b812c8d00ade4257f72415e9182c0e7cb3b93714175c2a90b6ecb7eefd8355",
}

SOURCE_REFERENCE = {
    "experiments/faa_sdr/prepare.py":
        "69a7e94996908d9806594ae9c130905a4f49343ee72a862832fdbb12c3ef2f8e",
    "revised/run.py":
        "678cec9db978286ef2faa969602fe7f7b16dc291c4f3740d1ab804f0765717c1",
    "revised/data.py":
        "a06cd260c1f5f7299bcf7663a05f695e2c631ed484eb2cb555d07a033d9d626e",
    "revised/faa_data.py":
        "0177230045af9a7e5d413f655a3e4ab9525fbab3dd373efd4c1dab070dc28041",
}

REQUIRED_SOURCE = (
    "revised/run.py",
    "revised/data.py",
    "revised/faa_data.py",
    "revised/faa_features.py",
    "revised/faa_sampling.py",
    "revised/inductive_protocol.py",
    "revised/negative_sampling.py",
    "revised/ssl_factorial_easy.py",
    "src/evaluation/decoder.py",
    "data/synthetic/generator.py",
    "experiments/synthetic/train_wbtbgrl.py",
    "experiments/faa_sdr/prepare.py",
)

NUMERIC_FLAGS = (
    "hidden", "dim", "encoder_epochs", "decoder_epochs",
    "patience", "threads", "lr", "weight_decay",
    "dropout", "lam", "edge_drop", "feature_drop",
    "train_ratio", "val_ratio"
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def profiles():
    items = []
    for variant in VARIANTS:
        for mode in MODES:
            items.append((
                f"WBTBGRL_{variant}_{mode}",
                "WBTBGRL", variant, mode
            ))
    for model in BASELINES:
        items.append((model, model, "NWP_NWB", "original"))
    return items


def source_inventory():
    for name in REQUIRED_SOURCE:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(
                f"Required source missing: {name}"
            )

    allowed = (
        "src",
        "revised",
        "data/synthetic",
        "experiments/synthetic",
        "experiments/faa_sdr",
    )

    files = set()
    for dirname in allowed:
        directory = ROOT / dirname
        for path in directory.rglob("*.py"):
            if path.is_symlink():
                raise RuntimeError(
                    f"Source symlink not allowed: {path}"
                )
            relative = path.relative_to(ROOT)
            if any(
                part.startswith(".") or part == "__pycache__"
                for part in relative.parts
            ):
                continue
            files.add(path)

    files.update(ROOT / f for f in REQUIRED_SOURCE)

    inventory = {}
    warnings = []
    for path in sorted(files):
        relative = path.relative_to(ROOT).as_posix()
        contents = path.read_text(encoding="utf-8")
        compile(contents, str(path), "exec")

        if re.search(
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
            contents
        ):
            raise RuntimeError(
                f"Potential private key in {relative}"
            )

        if re.search(
            r"(?i)(?:api_key|secret_key|access_token)"
            r"\s*=\s*['\"][^'\"]{12,}",
            contents
        ):
            raise RuntimeError(
                f"Potential hardcoded credential in {relative}"
            )

        current = sha(path)
        inventory[relative] = current

        if (
            relative in SOURCE_REFERENCE
            and current != SOURCE_REFERENCE[relative]
        ):
            warnings.append(
                "Source differs from previous audited version: "
                + relative
            )

    return sorted(files), inventory, warnings


def check_faa(raw_dir, processed_dir):
    raw_hashes = {}
    for year in YEARS:
        path = raw_dir / f"SDR-{year}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.is_symlink() or path.stat().st_size == 0:
            raise RuntimeError(
                f"Invalid raw FAA file: {path}"
            )
        raw_hashes[path.name] = sha(path)

    processed_hashes = {}
    for name, expected in FAA_REFERENCE.items():
        path = processed_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha(path)
        if actual != expected:
            raise RuntimeError(
                f"FAA fingerprint mismatch: {name}\n"
                f"Expected: {expected}\n"
                f"Found:    {actual}\n"
                "Do not mix another processed version into "
                "the published benchmark."
            )
        processed_hashes[name] = actual

    manifest = json.loads(
        (processed_dir / "split_manifest.json")
        .read_text(encoding="utf-8")
    )

    serialized = json.dumps(manifest).lower()
    raw_matches = {
        name: digest.lower() in serialized
        for name, digest in raw_hashes.items()
    }

    # Inspect only aggregated information, not raw identifiers.
    # The SHA-256 checks above establish exact correspondence
    # with the previously audited processed benchmark.
    # Do not impose row counts from a different cleaning stage.
    counts = {}
    date_ranges = {}

    for split in ("train", "val", "test"):
        path = processed_dir / f"{split}_interactions.csv"
        df = pd.read_csv(
            path, dtype=str, keep_default_na=False
        )

        if df.empty:
            raise RuntimeError(
                f"Empty processed split: {split}"
            )

        counts[split] = int(len(df))

    print("FAA processed splits verified:", counts)

    for split in ("train", "val", "test"):
        path = processed_dir / f"{split}_interactions.csv"
        df = pd.read_csv(
            path, dtype=str, keep_default_na=False
        )

        for column in df.columns:
            normalized = re.sub(
                r"[^a-z0-9]", "", column.lower()
            )
            if normalized == "submissiondate":
                dates = pd.to_datetime(
                    df[column], utc=True, errors="coerce"
                )
                if dates.isna().any():
                    raise RuntimeError(
                        f"Invalid dates in {split}"
                    )
                days = dates.dt.date
                date_ranges[split] = [
                    str(days.min()), str(days.max())
                ]
                break

    if len(date_ranges) == 3:
        if not (
            date_ranges["train"][1] < date_ranges["val"][0]
            and date_ranges["val"][1] < date_ranges["test"][0]
        ):
            raise RuntimeError(
                "Overlapping or unordered temporal partitions"
            )

    return {
        "raw_directory": "data/faa_sdr/raw",
        "original_years": list(YEARS),
        "original_sha256": raw_hashes,
        "raw_hashes_found_in_manifest": raw_matches,
        "processed_sha256": processed_hashes,
        "processed_sha256_matches_reference": True,
        "rows": counts,
        "date_ranges_if_columns_available": date_ranges,
        "chronology_independently_checked":
            len(date_ranges) == 3,
        "raw_replay_performed": False,
    }


def check_runs(original_root):
    rows = []
    configurations = []
    checks = {}

    for dataset in DATASETS:
        candidate_hashes = set()
        sizes = set()

        for label, model, variant, mode in profiles():
            for seed in SEEDS:
                path = (
                    original_root / dataset /
                    label / f"seed{seed}"
                )
                result_path = path / "result.json"
                scores_path = path / "test_scores.csv"

                if not result_path.is_file():
                    raise FileNotFoundError(result_path)
                if not scores_path.is_file():
                    raise FileNotFoundError(scores_path)

                result = json.loads(
                    result_path.read_text(encoding="utf-8")
                )
                cfg = result.get("config", {})

                expected = {
                    "dataset": dataset,
                    "model": model,
                    "variant": variant,
                    "ssl_mode": mode,
                    "seed": seed,
                    "candidate_seed": 2026,
                }
                for key, value in expected.items():
                    if cfg.get(key) != value:
                        raise RuntimeError(
                            f"{path}: inconsistent {key}"
                        )

                policy = result.get(
                    "negative_sampling", {}
                ).get("policy")
                if policy != "matched_history":
                    raise RuntimeError(
                        f"{path}: unexpected negative policy: {policy}"
                    )

                if not result.get(
                    "protocol", ""
                ).startswith("history-only-v2;"):
                    raise RuntimeError(
                        f"{path}: unexpected temporal protocol"
                    )

                scores = pd.read_csv(
                    scores_path,
                    usecols=["label", "score"]
                )
                y = scores["label"].to_numpy()
                p = scores["score"].to_numpy()

                if not np.isin(y, [0, 1]).all():
                    raise RuntimeError(
                        f"{path}: invalid labels"
                    )
                if (
                    not np.isfinite(p).all()
                    or not ((p >= 0) & (p <= 1)).all()
                ):
                    raise RuntimeError(
                        f"{path}: invalid scores"
                    )

                npos = int(np.sum(y == 1))
                nneg = int(np.sum(y == 0))
                if not npos or npos != nneg:
                    raise RuntimeError(
                        f"{path}: invalid 1:1 candidate set"
                    )

                auc = float(roc_auc_score(y, p))
                ap = float(average_precision_score(y, p))
                saved = result["metrics"]

                if (
                    abs(auc - saved["ROC-AUC"]) > 1e-7
                    or abs(ap - saved["AP"]) > 1e-7
                ):
                    raise RuntimeError(
                        f"{path}: metrics mismatch"
                    )

                fingerprint = result.get("candidate_sha256")
                if not fingerprint:
                    raise RuntimeError(
                        f"{path}: missing candidate SHA"
                    )

                candidate_hashes.add(fingerprint)
                sizes.add((len(scores), npos, nneg))

                rows.append({
                    "dataset": dataset,
                    "profile": label,
                    "seed": seed,
                    "roc_auc": auc,
                    "ap": ap,
                    "candidates": len(scores)
                })

                # Only portable, reproduction-relevant options.
                clean = {
                    k: cfg[k]
                    for k in NUMERIC_FLAGS
                    if k in cfg
                }
                for k in (
                    "graph_seed", "condition",
                    "propagation", "augmentation",
                    "loss_weights"
                ):
                    if k in cfg and cfg[k] is not None:
                        clean[k] = cfg[k]

                configurations.append({
                    "dataset": dataset,
                    "profile": label,
                    "model": model,
                    "variant": variant,
                    "ssl_mode": mode,
                    "seed": seed,
                    "candidate_seed": 2026,
                    "options": clean,
                })

        if len(candidate_hashes) != 1 or len(sizes) != 1:
            raise RuntimeError(
                f"{dataset}: candidates differ "
                "between models or seeds"
            )

        checks[dataset] = {
            "runs": len(profiles()) * len(SEEDS),
            "identical_candidates": True,
            "candidate_sha256": next(iter(candidate_hashes)),
            "candidate_counts": list(next(iter(sizes)))
        }

    expected_runs = len(DATASETS) * len(profiles()) * len(SEEDS)
    if len(rows) != expected_runs:
        raise RuntimeError(
            f"Expected {expected_runs} runs, found {len(rows)}"
        )

    return rows, configurations, checks


def summarize(rows):
    df = pd.DataFrame(rows)
    summary = (
        df.groupby(
            ["dataset", "profile"], as_index=False
        )
        .agg(
            seeds=("seed", "nunique"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            ap_mean=("ap", "mean"),
            ap_std=("ap", "std"),
        )
    )
    if not (summary["seeds"] == 5).all():
        raise RuntimeError(
            "Incomplete five-seed summary"
        )
    return summary


def shell_commands(configurations):
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'cd "$(dirname "$0")"',
        'PYTHON="${PYTHON:-python}"',
        'DEVICE="${DEVICE:-cpu}"',
        "mkdir -p reproduction_outputs",
        "",
        "# WARNING: Running this file retrains 120 models.",
        "# The R1-22 audit itself never runs this script.",
        "",
    ]

    for item in configurations:
        dataset = item["dataset"]
        profile = item["profile"]
        seed = item["seed"]
        output = (
            f"reproduction_outputs/{dataset}/"
            f"{profile}/seed{seed}"
        )

        args = [
            "-m", "revised.run",
            "--dataset", dataset,
            "--model", item["model"],
            "--variant", item["variant"],
            "--ssl-mode", item["ssl_mode"],
            "--seed", str(seed),
            "--candidate-seed", "2026",
            "--negative-strategy", "matched_history",
        ]

        opts = item["options"]
        for key in NUMERIC_FLAGS:
            if key in opts:
                args.extend([
                    "--" + key.replace("_", "-"),
                    str(opts[key])
                ])

        if dataset == "faa_sdr_ap_3y":
            args.extend([
                "--data",
                "data/faa_sdr/processed/faa_sdr_ap_3y"
            ])
        else:
            args.extend([
                "--graph-seed",
                str(opts.get("graph_seed", 42)),
                "--condition",
                str(opts.get("condition", "informative"))
            ])

        for key in (
            "propagation", "augmentation", "loss_weights"
        ):
            if key in opts:
                args.extend([
                    "--" + key.replace("_", "-"),
                    str(int(bool(opts[key])))
                ])

        args.extend([
            "--output", output,
            "--no-auto-summary"
        ])

        parts = (
            ['"${PYTHON}"']
            + [shlex.quote(str(a)) for a in args]
        )
        index = parts.index("--no-auto-summary")
        parts[index:index] = [
            "--device", '"${DEVICE}"'
        ]

        lines.append(
            f"# {dataset} / {profile} / seed {seed}"
        )
        lines.append(
            f"if [ -e {shlex.quote(output)} ]; then "
            f"echo 'Output exists: {output}'; exit 1; fi"
        )
        lines.append(" ".join(parts))
        lines.append("")

    return "\n".join(lines) + "\n"


def docs():
    return """# R1-22 — Reproducibility package

## Objective
Audit the public FAA-SDR-AP-3Y benchmark and distribute
the original BiSBM synthetic generator.

Only historical FAA evaluation results are audited.
Synthetic experiment results are not certified here.

This archive is a candidate for publication. It requires
manual inspection before being uploaded to a public repository.

## FAA source
https://www.faa.gov/av-info/download_SDR

Download exactly these files:
- SDR-2024.csv
- SDR-2025.csv
- SDR-2026.csv

Place them inside:
    data/faa_sdr/raw/

The 2026 source was incomplete when the original benchmark
was constructed. Do not substitute a later updated download
without checking its SHA-256 against the provenance record.

## Building FAA-SDR-AP-3Y
Use the included ORIGINAL preprocessing implementation:

    python experiments/faa_sdr/prepare.py --help

Follow the arguments supported by that actual script.
Do not assume that the --start-year or --end-year options exist.

The three years are concatenated and cleaned BEFORE the
chronological split. The original benchmark uses whole-day
TRAIN/VAL/TEST boundaries.

Processed data directory:
    data/faa_sdr/processed/faa_sdr_ap_3y/

Expected files:
- split_manifest.json
- train_interactions.csv
- val_interactions.csv
- test_interactions.csv

Check the expected SHA-256 values in r122_audit.json.
Byte-identical processed files provide the strongest
verification of the frozen benchmark.

## Synthetic BiSBM
Source:
    data/synthetic/generator.py

The original generator is included unchanged.
Its source is supplied for independent reproduction.
Synthetic evaluation results are not part of this audit.

## Reproducing results
The included reproduce_all.sh contains 60 FAA commands:
1 dataset x 12 model configurations x 5 seeds.

The BiSBM generator is provided separately as source code.

This script DOES TRAIN MODELS. Execute it only deliberately.

For CPU:
    bash reproduce_all.sh

For GPU:
    DEVICE=cuda bash reproduce_all.sh

It writes into reproduction_outputs/ and does not overwrite
the saved original experiment.

Exact bitwise agreement may depend on library versions,
GPU behavior and other environmental factors.

## Experimental protocol
- Chronological TRAIN / VAL / TEST.
- SSL encoder trained on TRAIN.
- Decoder trained on TRAIN, selected through VAL.
- TEST evaluated after model selection.
- Same negative candidate sets across compared models.
- Five random seeds: 42, 43, 44, 45, 46.
- O/U identities are defined relative to TRAIN.
- A candidate roster is specified retrospectively.

## Important limitations
- Industry is private and has not been distributed.
- E-commerce and Eurocontrol are not redistributed here.
- FAA-SDR reports are associations, not verified failures.
- A sampled negative is not a confirmed non-failure.
- The original submission date is not guaranteed to equal
  the date a report became publicly available.
- Some inductive subgroups may have low support.
- An original generator is provided, but full
  end-to-end reproduction still requires the same inputs,
  preprocessing rules and compatible environment.

## Data redistribution
Raw FAA files are NOT included in this package.

Processed FAA files are included only when the package
was built with --include-processed-faa.

Inspect these files and confirm redistribution and
privacy requirements before public release. Processed
tables may contain aircraft/part identifiers.

Do not publish private Industry data, checkpoints,
record-level predictions or user-identifying information.

## Evidence
See:
- r122_audit.json
- r122_results_summary.csv
- r122_results_per_run.csv
- r122_run_configurations.json
- r122_environment.json

The audit does not re-run training or automatically
prove the absence of all kinds of data leakage.
"""


def environment():
    packages = [
        "numpy", "pandas", "scipy",
        "scikit-learn", "torch", "torch-geometric"
    ]
    result = {
        "python": platform.python_version(),
        "package_versions": {}
    }
    for name in packages:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "not-installed"
        result["package_versions"][name] = version
    return result


def generate(args):
    sources, source_hashes, source_warnings = source_inventory()
    faa = check_faa(args.raw_dir, args.processed_dir)
    runs, configurations, results_check = check_runs(
        args.original_root
    )

    summary = summarize(runs)
    warnings = list(source_warnings)

    if not faa["chronology_independently_checked"]:
        warnings.append(
            "Processed CSV dates were not independently verified; "
            "inspect split_manifest.json and prior audit."
        )

    if not all(
        faa["raw_hashes_found_in_manifest"].values()
    ):
        warnings.append(
            "Not all raw-file hashes were found in the manifest. "
            "The raw-to-processed correspondence was NOT replayed "
            "in this patch."
        )

    audit = {
        "reviewer_comment": "R1-22",
        "status": "PUBLICATION_CANDIDATE_NOT_YET_PUBLIC",
        "training_performed": False,
        "benchmark": "FAA-SDR-AP-3Y",
        "synthetic_generator_included": True,
        "synthetic_results_audited": False,
        "faa": faa,
        "test_evidence": results_check,
        "source_sha256": source_hashes,
        "processed_faa_included":
            bool(args.include_processed_faa),
        "industry_included": False,
        "warnings": warnings,
    }

    print("R1-22 PREFLIGHT PASSED")
    print("Historical runs checked:", len(runs))
    print("Model configurations:", len(profiles()))
    print("Seeds:", list(SEEDS))
    print("Processed FAA reference SHA: MATCH")
    print("FAA raw years:", list(YEARS))
    for warning in warnings:
        print("WARNING:", warning)

    if args.action == "preflight":
        return

    if args.output.exists():
        raise RuntimeError(
            f"Refusing overwrite: {args.output}"
        )

    csv_run = pd.DataFrame(runs).to_csv(index=False)
    csv_summary = summary.to_csv(index=False)

    with zipfile.ZipFile(
        args.output, "x",
        compression=zipfile.ZIP_DEFLATED
    ) as z:
        for path in sources:
            relative = path.relative_to(ROOT).as_posix()
            z.write(path, arcname=relative)

        z.write(
            ROOT / "r122_repro/runner.py",
            arcname="r122_repro/runner.py"
        )
        z.write(
            ROOT / "r122_repro/__init__.py",
            arcname="r122_repro/__init__.py"
        )

        z.writestr("README.md", docs())
        z.writestr("reproduce_all.sh",
                   shell_commands(configurations))
        z.writestr("r122_results_per_run.csv", csv_run)
        z.writestr("r122_results_summary.csv", csv_summary)
        z.writestr(
            "r122_run_configurations.json",
            json.dumps(configurations, indent=2)
        )
        z.writestr(
            "r122_audit.json",
            json.dumps(audit, indent=2)
        )
        z.writestr(
            "r122_environment.json",
            json.dumps(environment(), indent=2)
        )

        if args.include_processed_faa:
            for path in sorted(
                args.processed_dir.rglob("*")
            ):
                if not path.is_file():
                    continue
                if path.is_symlink():
                    raise RuntimeError(
                        f"Symlink not allowed: {path}"
                    )
                if path.suffix.lower() not in (
                    ".csv", ".json"
                ):
                    raise RuntimeError(
                        f"Unexpected processed format: {path}"
                    )
                relative = path.relative_to(
                    args.processed_dir
                ).as_posix()
                z.write(
                    path,
                    arcname=(
                        "data/faa_sdr/processed/"
                        "faa_sdr_ap_3y/" + relative
                    )
                )

    with zipfile.ZipFile(args.output) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(
                f"Corrupt archive entry: {bad}"
            )
        members = z.namelist()
        forbidden = (
            "data/faa_sdr/raw/",
            "data/industry/",
            "experiments/industry/",
            "results_v2/",
        )
        if any(
            m.startswith(forbidden)
            for m in members
        ):
            raise RuntimeError(
                "Forbidden private/raw file in archive"
            )

    print("PACKAGE CREATED:", args.output)
    print("Original files unchanged.")
    print("Publication requires manual review.")


def main():
    parser = argparse.ArgumentParser(
        description="R1-22 reproducibility patch"
    )
    parser.add_argument(
        "action", choices=["preflight", "package"]
    )
    parser.add_argument(
        "--raw-dir", type=Path,
        default=ROOT / "data/faa_sdr/raw"
    )
    parser.add_argument(
        "--processed-dir", type=Path,
        default=ROOT / "data/faa_sdr/processed/faa_sdr_ap_3y"
    )
    parser.add_argument(
        "--original-root", type=Path,
        default=ROOT / "results_v2/campana_final_v2"
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "R1_22_PUBLICATION_CANDIDATE.zip"
    )
    parser.add_argument(
        "--include-processed-faa",
        action="store_true",
        help="Explicit opt-in: package FAA processed CSV/JSON"
    )
    args = parser.parse_args()

    args.raw_dir = args.raw_dir.resolve()
    args.processed_dir = args.processed_dir.resolve()
    args.original_root = args.original_root.resolve()
    args.output = args.output.resolve()

    generate(args)


if __name__ == "__main__":
    main()
