#!/usr/bin/env bash
#
# Joint *fine* router (train_joint_router dense supervision) with:
#   (1) data_prep/compositional/joint.py @ 0.95 mass-coverage marginal-greedy
#       on stacked dense deltas for commonsenseqa, boolq, arc_easy
#   (2) export_compositional_joint_fine_router_assets.py
#   (3) training/train_joint_router.py --wandb + external HF eval every 5 epochs
#
# Prerequisites
# -------------
# Dense tensors must align with *source* catalogue column order (dense_deltas_matrix.pt /
# payloads with delta_matrix[Q, N_legal_programs_benchmark] keyed to that manifest).
# Typical source: compositional_runs/joint5_bundle_noassign_compositional
# mined via data_prep/dense_reevaluation.py on that catalogue split.
#
# Set these before running when SKIP_JOINT_BUILD is unset or 0:
#   DENSE_COMMONSENSEQA  DENSE_BOOLQ  DENSE_ARC_EASY
# Each should be an absolute path to a dense_deltas_matrix.pt with matching shapes.
#
# To skip joint construction (reuse an existing joint manifest + dense_deltas/):
#   export SKIP_JOINT_BUILD=1
#   export OUT_JOINT_CATALOGUE=/path/to/joint_catalogue_dir
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

JOINT_SRC="${JOINT_SRC_CATALOGUE:-compositional_runs/joint5_bundle_noassign_compositional}"
OUT_JOINT="${OUT_JOINT_CATALOGUE:-compositional_runs/joint_g95dense_csqa_boolq_arceasy_$(date +%Y%m%d_%H%M%S)}"
SPLIT_JSON="${SPLIT_JSON:-splits/joint5_csqa_boolq_arc_easy_seed42_split.json}"

if [[ "${SKIP_JOINT_BUILD:-0}" != "1" ]]; then
  : "${DENSE_COMMONSENSEQA:?export DENSE_COMMONSENSEQA=/abs/path/to/dense_deltas_matrix.pt}"
  : "${DENSE_BOOLQ:?export DENSE_BOOLQ=/abs/path/to/dense_deltas_matrix.pt}"
  : "${DENSE_ARC_EASY:?export DENSE_ARC_EASY=/abs/path/to/dense_deltas_matrix.pt}"

  python data_prep/compositional/joint.py \
    --catalogue_dir "$JOINT_SRC" \
    --output_dir "$OUT_JOINT" \
    --benchmarks commonsenseqa boolq arc_easy \
    --dense_deltas "commonsenseqa=${DENSE_COMMONSENSEQA}" \
                   "boolq=${DENSE_BOOLQ}" \
                   "arc_easy=${DENSE_ARC_EASY}" \
    --mass_coverage 0.95 \
    --split_json "$SPLIT_JSON" \
    --log_level INFO
else
  : "${OUT_JOINT:?When SKIP_JOINT_BUILD=1, set OUT_JOINT_CATALOGUE to an existing joint dir}"
fi

JOINT_ASSETS="${OUT_JOINT}/joint_fine_assets"
mkdir -p "$JOINT_ASSETS"

python scripts/export_compositional_joint_fine_router_assets.py \
  --catalogue_dir "$OUT_JOINT" \
  --out_dir "$JOINT_ASSETS" \
  --benchmarks commonsenseqa boolq arc_easy \
  --dense_columns measured_only

WANDB_PROJECT="${WANDB_PROJECT:-joint-router}"
TAG="jointfine_csqa_boolq_arc_$(date +%Y%m%d_%H%M%S)"

# Canonical dirs used only for pivot_layer in config.json (HF external eval hook).
COMMON_CAN="${COMMONSENSEQA_CANON:-$HOME/generalized_transformer-2/dr-llm/fine_routing_data_commonsenseqa_mcts}"
BOOLQ_CAN="${BOOLQ_CANON:-$ROOT/compositional_runs/boolq_qwen25_instruct_dense_assign_mc20/canonical}"
ARC_CAN="${ARC_CANON:-$ROOT/compositional_runs/arc_easy_qwen25_0.5B_instruct_dense_assign_mc20/canonical}"

TRAIN_OUT="${OUT_JOINT}/${TAG}"
mkdir -p "$TRAIN_OUT"

python training/train_joint_router.py \
  --data_dir "$JOINT_ASSETS/staging" \
  --output_dir "$TRAIN_OUT" \
  --anchor_seqs_json "$JOINT_ASSETS/anchor_seqs.json" \
  --catalog_json "$JOINT_ASSETS/catalog.json" \
  --per_bench_catalog_json "$JOINT_ASSETS/per_bench_catalog.json" \
  --dense_deltas_jsonl "$JOINT_ASSETS/dense_joint.jsonl" \
  --split_json "$SPLIT_JSON" \
  --benchmarks commonsenseqa boolq arc_easy \
  --compressor_type last_token \
  --hidden_dims 1024 1024 512 \
  --epochs "${EPOCHS:-80}" \
  --batch_size "${BATCH_SIZE:-64}" \
  --wandb \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run_name "${WANDB_RUN_NAME:-$TAG}" \
  --wandb_tags joint_fine marginal_greedy095 csqa boolq arc_easy \
  --asset_export_meta_json "$JOINT_ASSETS/export_meta.json" \
  --external_eval_every 5 \
  --external_eval_per_bench 300 \
  --external_eval_model_name "${HF_MODEL_FOR_EVAL:-Qwen/Qwen2.5-0.5B-Instruct}" \
  --external_eval_canonical \
    "commonsenseqa=${COMMON_CAN}" \
    "boolq=${BOOLQ_CAN}" \
    "arc_easy=${ARC_CAN}"

echo "Artifacts: joint=$OUT_JOINT  staging=$JOINT_ASSETS/staging  train=$TRAIN_OUT"
