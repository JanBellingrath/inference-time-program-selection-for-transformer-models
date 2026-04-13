#!/usr/bin/env bash
# 50 hand-designed + random joint-router configs, each logged as its own W&B run
# (same group → Workspace charts / custom parallel-coords panel over hyperparameters).
#
#   GPU=1 bash experiments/run_joint_router_local50_wandb.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

EXTRA=( )
[[ -n "${WANDB_ENTITY:-}" ]] && EXTRA+=(--wandb_entity "$WANDB_ENTITY")

GROUP="${WANDB_GROUP:-joint-local-50-$(date +%Y%m%d-%H%M)}"

exec python3 -u experiments/sweep_joint_router.py \
  --data_dir fine_routing_data_boolq_mcts fine_routing_data_commonsenseqa_mcts \
  --benchmarks boolq commonsenseqa \
  --results_dir "${RESULTS_DIR:-predictions/qwen25_0.5b_v2_sdpa}" \
  --model_name "${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}" \
  --eval_questions "${EVAL_Q:-100}" \
  --gpu 0 \
  --batch_size "${BATCH_SIZE:-64}" \
  --early_stop_patience "${EARLY_STOP_PATIENCE:-25}" \
  --local_sweep 50 \
  --local_sweep_seed "${LOCAL_SWEEP_SEED:-42}" \
  --local_sweep_wandb \
  --wandb_group "$GROUP" \
  --project "${PROJECT:-joint-router-sweep}" \
  --output_json "${OUTPUT_JSON:-results/sweep_joint_local50_wandb.json}" \
  "${EXTRA[@]}"
