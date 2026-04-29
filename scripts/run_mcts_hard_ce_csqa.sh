#!/usr/bin/env bash
# Train a small compositional router on MCTS-only supervision with hard
# cross-entropy, restricted to the top-k programs covering 95% of MCTS
# positive mass in the existing ``csqa_ft_unified95`` catalogue.
#
# Data derivation does NOT rerun MCTS:
#   * Reuses ``observed/commonsenseqa.jsonl`` from the canonical
#     compositional catalogue (already MCTS-only, with assign support).
#   * scripts/build_mcts_filtered_observed.py subsets those observed rows
#     to the top-k programs by positive MCTS mass (95% coverage) and
#     writes a fresh observed/ dir. Re-running with --skip_if_exists is a
#     no-op if the filtered jsonl already exists.
#
# Training:
#   * Last-token compressor, small d_latent, unary-only (pairs off).
#   * Hard CE against the observed argmax-Δ route (no soft distribution,
#     no dense supervision).
#   * Dense Δ matrix is still loaded so per-epoch downstream mean_uplift
#     and checkpoint selection can use the full held-out uplift signal;
#     training loss itself only sees the MCTS-filtered observed support.
#   * Weights&Biases online logging on the project
#     ``csqa-ft-router-mcts-hardce``.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CATALOGUE_DIR="${CATALOGUE_DIR:-compositional_runs/csqa_ft_unified95/catalog_mass095}"
FILTERED_DIR="${FILTERED_DIR:-compositional_runs/csqa_ft_unified95/catalog_mass095_mcts095}"
OUTPUT_DIR="${OUTPUT_DIR:-router_runs/csqa_ft_mcts_hardce_$(date +%Y%m%d_%H%M%S)}"
DENSE_DELTAS="${DENSE_DELTAS:-${CATALOGUE_DIR}/dense_deltas/commonsenseqa.pt}"
MASS_COVERAGE="${MASS_COVERAGE:-0.95}"
BENCH="${BENCH:-commonsenseqa}"
WANDB_PROJECT="${WANDB_PROJECT:-csqa-ft-router-mcts-hardce}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-mcts_top$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/train.log"

echo "[run_mcts_hard_ce_csqa] catalogue_dir=${CATALOGUE_DIR}"
echo "[run_mcts_hard_ce_csqa] filtered_dir =${FILTERED_DIR}"
echo "[run_mcts_hard_ce_csqa] output_dir   =${OUTPUT_DIR}"
echo "[run_mcts_hard_ce_csqa] dense_deltas =${DENSE_DELTAS}"
echo "[run_mcts_hard_ce_csqa] mass_coverage=${MASS_COVERAGE}  bench=${BENCH}"
echo "[run_mcts_hard_ce_csqa] wandb_project=${WANDB_PROJECT}  run_name=${WANDB_RUN_NAME}"
echo "[run_mcts_hard_ce_csqa] log_file     =${LOG_FILE}"

python -m scripts.build_mcts_filtered_observed \
    --catalogue_dir "${CATALOGUE_DIR}" \
    --benchmarks "${BENCH}" \
    --mass_coverage "${MASS_COVERAGE}" \
    --output_dir "${FILTERED_DIR}" \
    --skip_if_exists

exec python -m training.train_compositional_router \
    --catalogue_dir "${CATALOGUE_DIR}" \
    --benchmarks "${BENCH}" \
    --scope single \
    --output_dir "${OUTPUT_DIR}" \
    --observed_dir "${FILTERED_DIR}/observed" \
    --dense_deltas "${BENCH}=${DENSE_DELTAS}" \
    --hard_ce \
    --compressor_type last_token \
    --d_latent 128 \
    --edit_hidden_dims 128 128 \
    --unary_hidden_dims 128 128 \
    --encoder_dropout 0.15 \
    --edit_dropout 0.15 \
    --unary_dropout 0.15 \
    --weight_decay 0.05 \
    --lr 7e-4 \
    --epochs 30 \
    --batch_size 64 \
    --val_fraction 0.15 \
    --seed 42 \
    --lam 0.0 \
    --student_temperature 1.0 \
    --use_anchor_bias \
    --checkpoint_metric mean_uplift \
    --downstream_eval_every 3 \
    --downstream_eval_subset 800 \
    --early_stopping_patience 8 \
    --wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${WANDB_RUN_NAME}" \
    --wandb_tags csqa ft mcts_only hard_ce last_token small \
    2>&1 | tee "${LOG_FILE}"
