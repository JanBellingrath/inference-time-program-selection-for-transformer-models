#!/usr/bin/env bash
# Overnight W&B Bayesian sweep for the joint router (BoolQ + CommonsenseQA).
#
# Prerequisites: `wandb login` once.
#
# Parallel coordinates / importance: open the project in W&B → Sweeps → your sweep
# → default sweep views include parallel coordinates and parameter importance.
#
# Training curves: each trial logs train/val loss per epoch (train/val_loss, etc.).
#
# Usage:
#   GPU=1 bash experiments/run_joint_router_wandb_overnight.sh
#   COUNT=60 WANDB_ENTITY=myteam bash experiments/run_joint_router_wandb_overnight.sh
#
# Resume an existing sweep:
#   SWEEP_ID=entity/project/sweepid bash experiments/run_joint_router_wandb_overnight.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GPU="${GPU:-0}"
COUNT="${COUNT:-50}"
PROJECT="${PROJECT:-joint-router-sweep}"
EVAL_Q="${EVAL_Q:-100}"
RESULTS_DIR="${RESULTS_DIR:-predictions/qwen25_0.5b_v2_sdpa}"

export CUDA_VISIBLE_DEVICES="$GPU"

EXTRA=( )
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  EXTRA+=(--wandb_entity "$WANDB_ENTITY")
fi

CMD=(
  python3 -u experiments/sweep_joint_router.py
  --data_dir fine_routing_data_boolq_mcts fine_routing_data_commonsenseqa_mcts
  --benchmarks boolq commonsenseqa
  --results_dir "$RESULTS_DIR"
  --model_name "${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
  --eval_questions "$EVAL_Q"
  --count "$COUNT"
  --project "$PROJECT"
  --gpu 0
  --batch_size "${BATCH_SIZE:-64}"
  --early_stop_patience "${EARLY_STOP_PATIENCE:-25}"
  --large_search_space
  "${EXTRA[@]}"
)

if [[ -n "${SWEEP_ID:-}" ]]; then
  CMD+=(--sweep_id "$SWEEP_ID")
fi

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
