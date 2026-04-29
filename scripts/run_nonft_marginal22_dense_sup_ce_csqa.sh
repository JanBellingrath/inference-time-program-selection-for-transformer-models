#!/usr/bin/env bash
# Same as run_nonft_marginal22_mcts_ce_csqa.sh but **dense soft supervision**:
# train CE soft-matches the full [Q,22] dense Δ (all questions, all 22 programs).
# catalogue, hparams, split, W&B pattern match the MCTS-only run otherwise.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CATALOGUE_DIR="${CATALOGUE_DIR:-compositional_runs/csqa_nonft_marginal_greedy22}"
SPLIT_JSON="${SPLIT_JSON:-splits/csqa_nonft_canonical_split.json}"
MASK_DIR="${MASK_DIR:-${CATALOGUE_DIR}/dense_masks}"
BENCH="commonsenseqa"
DENSE="${CATALOGUE_DIR}/dense_deltas/commonsenseqa.pt"
OUT="${OUTPUT_DIR:-${ROOT}/router_runs/csqa_nonft_marginal22_dense_sup_ce_$(date +%Y%m%d_%H%M%S)}"
WANDB_PROJECT="${WANDB_PROJECT:-csqa-nonft-marginal22-mcts-ce}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-nonft_marginal22_dense_sup_ce_$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-42}"
VAL_FRACTION="${VAL_FRACTION:-0.15}"
RUN_LLM_EVAL="${RUN_LLM_EVAL:-1}"

mkdir -p "${OUT}"
export WANDB_MODE="${WANDB_MODE:-online}"

MASK_FLAG=()
if [[ -d "${MASK_DIR}" ]]; then
  MASK_FLAG=(--dense_keep_mask_dir "${MASK_DIR}")
fi

python -m training.train_compositional_router \
  --catalogue_dir "${CATALOGUE_DIR}" \
  --benchmarks "${BENCH}" \
  --scope single \
  --output_dir "${OUT}" \
  --split_json "${SPLIT_JSON}" \
  --dense_deltas "${BENCH}=${DENSE}" \
  --use_dense_supervision \
  "${MASK_FLAG[@]}" \
  --compressor_type last_token \
  --d_latent 128 \
  --edit_hidden_dims 128 128 \
  --unary_hidden_dims 128 128 \
  --encoder_dropout 0.15 \
  --edit_dropout 0.15 \
  --unary_dropout 0.15 \
  --weight_decay 0.05 \
  --lr 7e-4 \
  --epochs 50 \
  --batch_size 64 \
  --val_fraction "${VAL_FRACTION}" \
  --seed "${SEED}" \
  --lam 0.0 \
  --student_temperature 1.0 \
  --checkpoint_metric loss \
  --downstream_eval_every 3 \
  --downstream_eval_subset 800 \
  --early_stopping_patience 12 \
  --wandb \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  --wandb_tags "${BENCH}" nonft marginal_greedy_22 dense_soft_ce full_22deltas \
  2>&1 | tee "${OUT}/train.log"

CKPT="${OUT}/compositional_router_best_${BENCH}.pt"
if [[ "${RUN_LLM_EVAL}" == "1" && -f "${CKPT}" ]]; then
  python -m scripts.log_llm_uplift_to_wandb \
    --output_dir "${OUT}" \
    --catalogue_dir "${CATALOGUE_DIR}" \
    --checkpoint "${CKPT}" \
    --split_json "${SPLIT_JSON}" \
    --benchmarks "${BENCH}" \
    --model_name "${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}" \
    --output_json "${OUT}/llm_val_eval.json" \
    2>&1 | tee "${OUT}/llm_eval.log" || true
fi

echo "Done. Output: ${OUT}"
