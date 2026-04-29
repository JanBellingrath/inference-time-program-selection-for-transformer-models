#!/usr/bin/env bash
# Non-FT (base Qwen) CSQA: compositional router with MCTS-only reduced
# supervision (95% positive-mass program subset) and hard CE. Live W&B
# during training. Primary metric: LLM **accuracy uplift in pp** vs anchor
# on the validation split — logged after training via
# scripts/log_llm_uplift_to_wandb.py (experiments/eval_compositional_downstream).
#
# Dense-Δ ``mean_uplift`` in W&B is a log-prob proxy
# (primary/*/dense_utility_delta_vs_anchor_nats), not pp.
#
# Environment overrides:
#   CATALOGUE_DIR, FILTERED_DIR, OUTPUT_DIR, WANDB_PROJECT, WANDB_RUN_NAME
#   SEED, VAL_FRACTION  (must match splits/csqa_nonft_canonical_split.json
#   if you regenerate the split)
#   RUN_LLM_EVAL=0      skip post-hoc LLM eval (faster; no uplift_pp in W&B)
#   EVAL_MAX_SAMPLES    if set, passed to --max_samples_per_bench
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CATALOGUE_DIR="${CATALOGUE_DIR:-compositional_runs/csqa_nonft_unified95/catalog_mass095}"
FILTERED_DIR="${FILTERED_DIR:-compositional_runs/csqa_nonft_unified95/catalog_mass095_mcts095}"
SPLIT_JSON="${SPLIT_JSON:-splits/csqa_nonft_canonical_split.json}"
OUTPUT_DIR="${OUTPUT_DIR:-router_runs/csqa_nonft_mcts_hardce_$(date +%Y%m%d_%H%M%S)}"
DENSE_DELTAS="${DENSE_DELTAS:-${CATALOGUE_DIR}/dense_deltas/commonsenseqa.pt}"
BENCH="${BENCH:-commonsenseqa}"
WANDB_PROJECT="${WANDB_PROJECT:-csqa-nonft-mcts-hardce}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-nonft_mcts_$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-42}"
VAL_FRACTION="${VAL_FRACTION:-0.15}"
MASS_COVERAGE="${MASS_COVERAGE:-0.95}"
RUN_LLM_EVAL="${RUN_LLM_EVAL:-1}"

mkdir -p "${ROOT}/splits"
if [[ ! -f "${SPLIT_JSON}" ]]; then
  echo "[run_nonft] generating ${SPLIT_JSON}"
  python "${ROOT}/scripts/make_canonical_split.py" \
    --observed "${BENCH}=${CATALOGUE_DIR}/observed/${BENCH}.jsonl" \
    --output "${SPLIT_JSON}" \
    --seed "${SEED}" --val_fraction "${VAL_FRACTION}"
fi

echo "[run_nonft] building MCTS top-${MASS_COVERAGE} observed -> ${FILTERED_DIR}"
python -m scripts.build_mcts_filtered_observed \
  --catalogue_dir "${CATALOGUE_DIR}" \
  --benchmarks "${BENCH}" \
  --mass_coverage "${MASS_COVERAGE}" \
  --output_dir "${FILTERED_DIR}" \
  --skip_if_exists

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/train.log"
echo "[run_nonft] training  output_dir=${OUTPUT_DIR}  log=${LOG_FILE}"
python -m training.train_compositional_router \
  --catalogue_dir "${CATALOGUE_DIR}" \
  --benchmarks "${BENCH}" \
  --scope single \
  --output_dir "${OUTPUT_DIR}" \
  --split_json "${SPLIT_JSON}" \
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
  --epochs 50 \
  --batch_size 64 \
  --val_fraction "${VAL_FRACTION}" \
  --seed "${SEED}" \
  --lam 0.0 \
  --student_temperature 1.0 \
  --use_anchor_bias \
  --checkpoint_metric loss \
  --downstream_eval_every 3 \
  --downstream_eval_subset 800 \
  --early_stopping_patience 12 \
  --wandb \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  --wandb_tags "${BENCH}" nonft mcts_only hard_ce last_token \
  2>&1 | tee "${LOG_FILE}"

CKPT="${OUTPUT_DIR}/compositional_router_best_${BENCH}.pt"
if [[ "${RUN_LLM_EVAL}" == "1" && -f "${CKPT}" ]]; then
  echo "[run_nonft] LLM eval (validation) -> W&B  llm_eval/validation/unconditional_gain_pp"
  EVAL_JSON="${OUTPUT_DIR}/llm_val_eval.json"
  EVAL_LOG="${OUTPUT_DIR}/llm_eval.log"
  EVAL_MAX_FLAG=()
  if [[ -n "${EVAL_MAX_SAMPLES:-}" ]]; then
    EVAL_MAX_FLAG=(--max_samples_per_bench "${EVAL_MAX_SAMPLES}")
  fi
  set +e
  python -m scripts.log_llm_uplift_to_wandb \
    --output_dir "${OUTPUT_DIR}" \
    --catalogue_dir "${CATALOGUE_DIR}" \
    --checkpoint "${CKPT}" \
    --split_json "${SPLIT_JSON}" \
    --benchmarks "${BENCH}" \
    --model_name "${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}" \
    --output_json "${EVAL_JSON}" \
    "${EVAL_MAX_FLAG[@]}" \
    2>&1 | tee "${EVAL_LOG}"
  EVAL_ST="${PIPESTATUS[0]}"
  set -e
  if [[ "${EVAL_ST}" -ne 0 ]]; then
    echo "[run_nonft] LLM eval failed (exit ${EVAL_ST}); see ${EVAL_LOG}" >&2
  fi
else
  echo "[run_nonft] skipping LLM eval (RUN_LLM_EVAL=${RUN_LLM_EVAL} or missing ${CKPT})"
fi
