#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="fine_routing_data_boolq_mcts fine_routing_data_commonsenseqa_mcts"
BENCH="boolq commonsenseqa"
RES="predictions/qwen25_0.5b_v2_sdpa"
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PROJECT="joint-router-route-enc"
GROUP="target-ablation-no-stay"
GPU="${GPU:-0}"
EVAL_Q="${EVAL_Q:-100}"

for MODE in hard soft5; do
  echo ""
  echo "================================================================"
  echo "  no_stay + $MODE"
  echo "================================================================"
  python experiments/run_target_ablation.py \
    --data_dir $DATA \
    --benchmarks $BENCH \
    --results_dir "$RES" \
    --model_name "$MODEL" \
    --eval_questions "$EVAL_Q" \
    --gpu "$GPU" \
    --project "$PROJECT" \
    --wandb_group "$GROUP" \
    --run_name "no_stay_${MODE}" \
    --mode "$MODE" \
    --no_stay \
    --batch_size 64
  echo "  Done: no_stay_${MODE}"
done

echo ""
echo "Both runs complete (W&B project: $PROJECT, group: $GROUP)."
