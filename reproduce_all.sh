#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cpu}"
mkdir -p reproduction_outputs

# WARNING: Running this file retrains 120 models.
# The R1-22 audit itself never runs this script.

# faa_sdr_ap_3y / WBTBGRL_NWP_NWB_original / seed 42
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_original/seed42 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_original/seed42'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant NWP_NWB --ssl-mode original --seed 42 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_original/seed42 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_NWP_NWB_original / seed 43
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_original/seed43 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_original/seed43'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant NWP_NWB --ssl-mode original --seed 43 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_original/seed43 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_NWP_NWB_original / seed 44
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_original/seed44 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_original/seed44'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant NWP_NWB --ssl-mode original --seed 44 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_original/seed44 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_NWP_NWB_original / seed 45
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_original/seed45 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_original/seed45'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant NWP_NWB --ssl-mode original --seed 45 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_original/seed45 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_NWP_NWB_original / seed 46
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_original/seed46 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_original/seed46'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant NWP_NWB --ssl-mode original --seed 46 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_original/seed46 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_NWP_NWB_symmetric / seed 42
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_symmetric/seed42 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_symmetric/seed42'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant NWP_NWB --ssl-mode symmetric --seed 42 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_symmetric/seed42 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_NWP_NWB_symmetric / seed 43
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_symmetric/seed43 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_symmetric/seed43'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant NWP_NWB --ssl-mode symmetric --seed 43 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_symmetric/seed43 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_NWP_NWB_symmetric / seed 44
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_symmetric/seed44 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_symmetric/seed44'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant NWP_NWB --ssl-mode symmetric --seed 44 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_symmetric/seed44 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_NWP_NWB_symmetric / seed 45
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_symmetric/seed45 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_symmetric/seed45'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant NWP_NWB --ssl-mode symmetric --seed 45 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_symmetric/seed45 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_NWP_NWB_symmetric / seed 46
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_symmetric/seed46 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_symmetric/seed46'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant NWP_NWB --ssl-mode symmetric --seed 46 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_NWB_symmetric/seed46 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_NWP_WB_original / seed 42
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_original/seed42 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_original/seed42'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant NWP_WB --ssl-mode original --seed 42 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_original/seed42 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_NWP_WB_original / seed 43
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_original/seed43 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_original/seed43'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant NWP_WB --ssl-mode original --seed 43 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_original/seed43 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_NWP_WB_original / seed 44
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_original/seed44 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_original/seed44'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant NWP_WB --ssl-mode original --seed 44 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_original/seed44 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_NWP_WB_original / seed 45
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_original/seed45 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_original/seed45'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant NWP_WB --ssl-mode original --seed 45 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_original/seed45 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_NWP_WB_original / seed 46
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_original/seed46 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_original/seed46'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant NWP_WB --ssl-mode original --seed 46 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_original/seed46 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_NWP_WB_symmetric / seed 42
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_symmetric/seed42 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_symmetric/seed42'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant NWP_WB --ssl-mode symmetric --seed 42 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_symmetric/seed42 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_NWP_WB_symmetric / seed 43
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_symmetric/seed43 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_symmetric/seed43'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant NWP_WB --ssl-mode symmetric --seed 43 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_symmetric/seed43 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_NWP_WB_symmetric / seed 44
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_symmetric/seed44 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_symmetric/seed44'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant NWP_WB --ssl-mode symmetric --seed 44 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_symmetric/seed44 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_NWP_WB_symmetric / seed 45
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_symmetric/seed45 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_symmetric/seed45'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant NWP_WB --ssl-mode symmetric --seed 45 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_symmetric/seed45 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_NWP_WB_symmetric / seed 46
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_symmetric/seed46 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_symmetric/seed46'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant NWP_WB --ssl-mode symmetric --seed 46 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_NWP_WB_symmetric/seed46 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_WP_NWB_original / seed 42
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_original/seed42 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_original/seed42'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant WP_NWB --ssl-mode original --seed 42 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_original/seed42 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_WP_NWB_original / seed 43
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_original/seed43 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_original/seed43'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant WP_NWB --ssl-mode original --seed 43 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_original/seed43 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_WP_NWB_original / seed 44
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_original/seed44 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_original/seed44'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant WP_NWB --ssl-mode original --seed 44 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_original/seed44 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_WP_NWB_original / seed 45
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_original/seed45 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_original/seed45'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant WP_NWB --ssl-mode original --seed 45 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_original/seed45 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_WP_NWB_original / seed 46
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_original/seed46 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_original/seed46'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant WP_NWB --ssl-mode original --seed 46 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_original/seed46 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_WP_NWB_symmetric / seed 42
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_symmetric/seed42 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_symmetric/seed42'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant WP_NWB --ssl-mode symmetric --seed 42 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_symmetric/seed42 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_WP_NWB_symmetric / seed 43
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_symmetric/seed43 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_symmetric/seed43'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant WP_NWB --ssl-mode symmetric --seed 43 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_symmetric/seed43 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_WP_NWB_symmetric / seed 44
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_symmetric/seed44 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_symmetric/seed44'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant WP_NWB --ssl-mode symmetric --seed 44 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_symmetric/seed44 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_WP_NWB_symmetric / seed 45
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_symmetric/seed45 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_symmetric/seed45'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant WP_NWB --ssl-mode symmetric --seed 45 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_symmetric/seed45 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_WP_NWB_symmetric / seed 46
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_symmetric/seed46 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_symmetric/seed46'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant WP_NWB --ssl-mode symmetric --seed 46 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_NWB_symmetric/seed46 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_WP_WB_original / seed 42
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_original/seed42 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_original/seed42'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant WP_WB --ssl-mode original --seed 42 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_original/seed42 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_WP_WB_original / seed 43
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_original/seed43 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_original/seed43'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant WP_WB --ssl-mode original --seed 43 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_original/seed43 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_WP_WB_original / seed 44
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_original/seed44 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_original/seed44'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant WP_WB --ssl-mode original --seed 44 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_original/seed44 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_WP_WB_original / seed 45
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_original/seed45 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_original/seed45'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant WP_WB --ssl-mode original --seed 45 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_original/seed45 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_WP_WB_original / seed 46
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_original/seed46 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_original/seed46'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant WP_WB --ssl-mode original --seed 46 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_original/seed46 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_WP_WB_symmetric / seed 42
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_symmetric/seed42 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_symmetric/seed42'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant WP_WB --ssl-mode symmetric --seed 42 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_symmetric/seed42 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_WP_WB_symmetric / seed 43
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_symmetric/seed43 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_symmetric/seed43'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant WP_WB --ssl-mode symmetric --seed 43 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_symmetric/seed43 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_WP_WB_symmetric / seed 44
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_symmetric/seed44 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_symmetric/seed44'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant WP_WB --ssl-mode symmetric --seed 44 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_symmetric/seed44 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_WP_WB_symmetric / seed 45
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_symmetric/seed45 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_symmetric/seed45'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant WP_WB --ssl-mode symmetric --seed 45 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_symmetric/seed45 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / WBTBGRL_WP_WB_symmetric / seed 46
if [ -e reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_symmetric/seed46 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_symmetric/seed46'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model WBTBGRL --variant WP_WB --ssl-mode symmetric --seed 46 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/WBTBGRL_WP_WB_symmetric/seed46 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / BGRL / seed 42
if [ -e reproduction_outputs/faa_sdr_ap_3y/BGRL/seed42 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/BGRL/seed42'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model BGRL --variant NWP_NWB --ssl-mode original --seed 42 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/BGRL/seed42 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / BGRL / seed 43
if [ -e reproduction_outputs/faa_sdr_ap_3y/BGRL/seed43 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/BGRL/seed43'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model BGRL --variant NWP_NWB --ssl-mode original --seed 43 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/BGRL/seed43 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / BGRL / seed 44
if [ -e reproduction_outputs/faa_sdr_ap_3y/BGRL/seed44 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/BGRL/seed44'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model BGRL --variant NWP_NWB --ssl-mode original --seed 44 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/BGRL/seed44 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / BGRL / seed 45
if [ -e reproduction_outputs/faa_sdr_ap_3y/BGRL/seed45 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/BGRL/seed45'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model BGRL --variant NWP_NWB --ssl-mode original --seed 45 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/BGRL/seed45 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / BGRL / seed 46
if [ -e reproduction_outputs/faa_sdr_ap_3y/BGRL/seed46 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/BGRL/seed46'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model BGRL --variant NWP_NWB --ssl-mode original --seed 46 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/BGRL/seed46 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / TBGRL / seed 42
if [ -e reproduction_outputs/faa_sdr_ap_3y/TBGRL/seed42 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/TBGRL/seed42'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model TBGRL --variant NWP_NWB --ssl-mode original --seed 42 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/TBGRL/seed42 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / TBGRL / seed 43
if [ -e reproduction_outputs/faa_sdr_ap_3y/TBGRL/seed43 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/TBGRL/seed43'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model TBGRL --variant NWP_NWB --ssl-mode original --seed 43 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/TBGRL/seed43 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / TBGRL / seed 44
if [ -e reproduction_outputs/faa_sdr_ap_3y/TBGRL/seed44 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/TBGRL/seed44'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model TBGRL --variant NWP_NWB --ssl-mode original --seed 44 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/TBGRL/seed44 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / TBGRL / seed 45
if [ -e reproduction_outputs/faa_sdr_ap_3y/TBGRL/seed45 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/TBGRL/seed45'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model TBGRL --variant NWP_NWB --ssl-mode original --seed 45 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/TBGRL/seed45 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / TBGRL / seed 46
if [ -e reproduction_outputs/faa_sdr_ap_3y/TBGRL/seed46 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/TBGRL/seed46'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model TBGRL --variant NWP_NWB --ssl-mode original --seed 46 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/TBGRL/seed46 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / GBT / seed 42
if [ -e reproduction_outputs/faa_sdr_ap_3y/GBT/seed42 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/GBT/seed42'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model GBT --variant NWP_NWB --ssl-mode original --seed 42 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/GBT/seed42 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / GBT / seed 43
if [ -e reproduction_outputs/faa_sdr_ap_3y/GBT/seed43 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/GBT/seed43'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model GBT --variant NWP_NWB --ssl-mode original --seed 43 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/GBT/seed43 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / GBT / seed 44
if [ -e reproduction_outputs/faa_sdr_ap_3y/GBT/seed44 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/GBT/seed44'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model GBT --variant NWP_NWB --ssl-mode original --seed 44 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/GBT/seed44 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / GBT / seed 45
if [ -e reproduction_outputs/faa_sdr_ap_3y/GBT/seed45 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/GBT/seed45'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model GBT --variant NWP_NWB --ssl-mode original --seed 45 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/GBT/seed45 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / GBT / seed 46
if [ -e reproduction_outputs/faa_sdr_ap_3y/GBT/seed46 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/GBT/seed46'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model GBT --variant NWP_NWB --ssl-mode original --seed 46 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/GBT/seed46 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / CCASSG / seed 42
if [ -e reproduction_outputs/faa_sdr_ap_3y/CCASSG/seed42 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/CCASSG/seed42'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model CCASSG --variant NWP_NWB --ssl-mode original --seed 42 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/CCASSG/seed42 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / CCASSG / seed 43
if [ -e reproduction_outputs/faa_sdr_ap_3y/CCASSG/seed43 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/CCASSG/seed43'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model CCASSG --variant NWP_NWB --ssl-mode original --seed 43 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/CCASSG/seed43 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / CCASSG / seed 44
if [ -e reproduction_outputs/faa_sdr_ap_3y/CCASSG/seed44 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/CCASSG/seed44'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model CCASSG --variant NWP_NWB --ssl-mode original --seed 44 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/CCASSG/seed44 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / CCASSG / seed 45
if [ -e reproduction_outputs/faa_sdr_ap_3y/CCASSG/seed45 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/CCASSG/seed45'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model CCASSG --variant NWP_NWB --ssl-mode original --seed 45 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/CCASSG/seed45 --device "${DEVICE}" --no-auto-summary

# faa_sdr_ap_3y / CCASSG / seed 46
if [ -e reproduction_outputs/faa_sdr_ap_3y/CCASSG/seed46 ]; then echo 'Output exists: reproduction_outputs/faa_sdr_ap_3y/CCASSG/seed46'; exit 1; fi
"${PYTHON}" -m revised.run --dataset faa_sdr_ap_3y --model CCASSG --variant NWP_NWB --ssl-mode original --seed 46 --candidate-seed 2026 --negative-strategy matched_history --hidden 128 --dim 64 --encoder-epochs 200 --decoder-epochs 100 --patience 20 --threads 2 --lr 0.001 --weight-decay 1e-05 --dropout 0.1 --lam 0.5 --edge-drop 0.1 --feature-drop 0.1 --train-ratio 0.8 --val-ratio 0.1 --data data/faa_sdr/processed/faa_sdr_ap_3y --output reproduction_outputs/faa_sdr_ap_3y/CCASSG/seed46 --device "${DEVICE}" --no-auto-summary

