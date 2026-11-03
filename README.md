# R1-22 — Reproducibility package

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
