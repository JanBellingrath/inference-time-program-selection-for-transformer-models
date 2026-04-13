#!/usr/bin/env bash
# Sequential sweep: deviation catalogue only (skip/swap/repeat, <=2 edits per anchor).
# Logs to W&B project "joint-router-route-enc".
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_DIRS="fine_routing_data_boolq_mcts fine_routing_data_commonsenseqa_mcts"
BENCHMARKS="boolq commonsenseqa"
RESULTS_DIR="predictions/qwen25_0.5b_v2_sdpa"
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PROJECT="joint-router-route-enc"
GROUP="route-enc-sweep-deviation"
GPU=0
EVAL_Q=100

CONFIGS_FILE="experiments/sweep_transformer_route_enc.json"

N_CONFIGS=$(python3 -c "import json; print(len(json.load(open('$CONFIGS_FILE'))))")
echo "Running $N_CONFIGS configurations (catalog_mode=deviation only)..."

for i in $(seq 0 $((N_CONFIGS - 1))); do
    CFG=$(python3 -c "
import json, sys, tempfile, os
cfgs = json.load(open('$CONFIGS_FILE'))
c = cfgs[$i]
name = c.pop('_name', 'trial_$i')
head = c.pop('_route_head', 'mlp')
rdim = c.pop('_route_dim', 64)
print(f'{name}|{head}|{rdim}')
fd, path = tempfile.mkstemp(suffix='.json', prefix='route_enc_')
with os.fdopen(fd, 'w') as f:
    json.dump(c, f, indent=2)
print(path)
")

    NAME=$(echo "$CFG" | head -1 | cut -d'|' -f1)
    HEAD=$(echo "$CFG" | head -1 | cut -d'|' -f2)
    RDIM=$(echo "$CFG" | head -1 | cut -d'|' -f3)
    TMPFILE=$(echo "$CFG" | tail -1)

    echo ""
    echo "================================================================"
    echo "  Trial $((i+1))/$N_CONFIGS: $NAME  (deviation catalog, head=$HEAD, route_dim=$RDIM)"
    echo "================================================================"

    python experiments/sweep_joint_router.py \
        --data_dir $DATA_DIRS \
        --benchmarks $BENCHMARKS \
        --results_dir "$RESULTS_DIR" \
        --model_name "$MODEL" \
        --eval_questions $EVAL_Q \
        --gpu $GPU \
        --project "$PROJECT" \
        --wandb_group "$GROUP" \
        --run_name "$NAME" \
        --compressor_type last_token \
        --catalog_mode deviation \
        --route_head "$HEAD" \
        --route_dim "$RDIM" \
        --fixed_config_json "$TMPFILE" \
        --batch_size 64

    rm -f "$TMPFILE"
    echo "  Done: $NAME"
done

echo ""
echo "All $N_CONFIGS trials complete."
