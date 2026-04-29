#!/usr/bin/env bash
# Non-FT CSQA: 22 program outputs (marginal-greedy 95% per-question oracle
# on train dense), MCTS-only softmax CE (no dense supervision), W&B, LLM pp.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CATALOGUE_DIR="${CATALOGUE_DIR:-compositional_runs/csqa_nonft_marginal_greedy22}"
SPLIT_JSON="${SPLIT_JSON:-splits/csqa_nonft_canonical_split.json}"
BENCH="commonsenseqa"
DENSE="${CATALOGUE_DIR}/dense_deltas/commonsenseqa.pt"
OUT="${OUTPUT_DIR:-${ROOT}/router_runs/csqa_nonft_marginal22_mcts_ce_$(date +%Y%m%d_%H%M%S)}"
WANDB_PROJECT="${WANDB_PROJECT:-csqa-nonft-marginal22-mcts-ce}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-nonft_marginal22_mcts_ce_$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-42}"
VAL_FRACTION="${VAL_FRACTION:-0.15}"
RUN_LLM_EVAL="${RUN_LLM_EVAL:-1}"

mkdir -p "${OUT}"
export WANDB_MODE="${WANDB_MODE:-online}"

python -m training.train_compositional_router \
  --catalogue_dir "${CATALOGUE_DIR}" \
  --benchmarks "${BENCH}" \
  --scope single \
  --output_dir "${OUT}" \
  --split_json "${SPLIT_JSON}" \
  --dense_deltas "${BENCH}=${DENSE}" \
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
  --wandb_tags "${BENCH}" nonft marginal_greedy_22 mcts_only soft_ce no_dense_sup \
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
