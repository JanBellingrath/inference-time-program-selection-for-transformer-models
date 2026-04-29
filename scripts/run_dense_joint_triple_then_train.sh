#!/usr/bin/env bash
# Mine dense Δ per benchmark (dense_reevaluation.py), build joint greedy mass catalogue,
# export joint fine assets, train joint router with dense supervision.
#
# Requirement: CUDA GPU. Set MAX_QUESTIONS unset or raise for fuller coverage (slow).
#
set -euo pipefail

FLEX_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DR_LLM="$(cd "${FLEX_ROOT}/../generalized_transformer-2/dr-llm" && pwd)"

MANIFEST="${FLEX_ROOT}/compositional_runs/joint5_bundle_noassign_compositional/manifest.json"
JOINT_SPLIT="${JOINT_SPLIT:-${FLEX_ROOT}/splits/joint5_csqa_boolq_arc_easy_seed42_split.json}"
OUT="${OUT:-${FLEX_ROOT}/compositional_runs/joint_dense_triple_pipeline_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT/catalogs" "$OUT/dense_raw" "$OUT/dense_supervision"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
GPU="${GPU:-0}"
BATCH="${BATCH:-2}"
MAX_Q="${MAX_QUESTIONS:-96}"
TRAIN_SPLITS="${TRAIN_SPLITS:-train}"
EPOCHS="${EPOCHS:-20}"

declare -A CANON=(
  [commonsenseqa]="${FLEX_ROOT}/compositional_runs/csqa_canonical"
  [boolq]="${FLEX_ROOT}/compositional_runs/boolq_noassign_sweep20260427/canonical"
  [arc_easy]="${DR_LLM}/fine_routing_data_arc_easy_mcts_qwen25_0.5B_base_canonical_noassign"
)

log() { echo "[$(date +%H:%M:%S)] $*"; }

export PYTHONPATH="${FLEX_ROOT}:${PYTHONPATH:-}"

log "OUT=$OUT  MANIFEST=$MANIFEST  MAX_QUESTIONS=$MAX_Q  SPLIT=$TRAIN_SPLITS"

# --- Dense mining per benchmark (catalog from joint5 compositional legal programmes) -----
for bench in commonsenseqa boolq arc_easy; do
  CAT="$OUT/catalogs/${bench}_dense_catalog.json"
  log "Build catalog for ${bench}"
  python "${FLEX_ROOT}/scripts/build_dense_catalog_from_legal_programs.py" \
    --manifest "$MANIFEST" \
    --benchmark "$bench" \
    --output "$CAT"

  DOUT="$OUT/dense_raw/${bench}"
  DIMP="$OUT/dense_supervision/${bench}_dense_imported.pt"
  log "dense_reevaluation ${bench} -> $DOUT (this may take a long time)"
  (
    cd "$FLEX_ROOT"
    PYTHONPATH="${FLEX_ROOT}:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES="$GPU" \
    python -m data_prep.dense_reevaluation \
      --catalog_json "$CAT" \
      --benchmarks "$bench" \
      --model_name "$MODEL_NAME" \
      --data_dir "${CANON[$bench]}" \
      --merge_source_dir "${CANON[$bench]}" \
      --output_dir "$DOUT" \
      --split "$TRAIN_SPLITS" \
      --max_questions "$MAX_Q" \
      --batch_size "$BATCH" \
      --gpu 0 \
      --save_interval 20
  )

  NJ="$(python - <<PY
import json
with open("${DOUT}/dense_deltas.jsonl") as f:
    print(sum(1 for _ in f))
PY
)"
  log "import_mined_dense_matrix ${bench} n_rows=$NJ"
  PYTHONPATH="${FLEX_ROOT}:${PYTHONPATH:-}" python -m data_prep.import_mined_dense_matrix \
    --mined_pt "${DOUT}/dense_deltas_matrix.pt" \
    --output "$DIMP" \
    --num_questions "$NJ"

done

# --- Joint merge @ 0.95 mass marginal (requires all three dense .pt paths) -----
log "joint.build_joint_catalogue (mass_coverage=0.95)"
JC_OUT="$OUT/joint_catalogue_mass095"
JOINT_DD=(
  "commonsenseqa=$OUT/dense_supervision/commonsenseqa_dense_imported.pt"
  "boolq=$OUT/dense_supervision/boolq_dense_imported.pt"
  "arc_easy=$OUT/dense_supervision/arc_easy_dense_imported.pt"
)

python "${FLEX_ROOT}/data_prep/compositional/joint.py" \
  --catalogue_dir "${FLEX_ROOT}/compositional_runs/joint5_bundle_noassign_compositional" \
  --output_dir "$JC_OUT" \
  --benchmarks commonsenseqa boolq arc_easy \
  --dense_deltas "${JOINT_DD[@]}" \
  --mass_coverage 0.95 \
  --split_json "$JOINT_SPLIT" \
  --log_level INFO

# --- Export joint fine-router pack -----
ASSETS="$JC_OUT/joint_fine_assets"
log "export_compositional_joint_fine_router_assets -> $ASSETS"
python "${FLEX_ROOT}/scripts/export_compositional_joint_fine_router_assets.py" \
  --catalogue_dir "$JC_OUT" \
  --out_dir "$ASSETS" \
  --benchmarks commonsenseqa boolq arc_easy \
  --dense_columns measured_only

# --- Train dense-supervised joint fine router -----
TAG="jointfine_dense_mass095"
TRAIN="$OUT/train_${TAG}"
log "train_joint_router -> $TRAIN"

python "${FLEX_ROOT}/training/train_joint_router.py" \
  --data_dir "$ASSETS/staging" \
  --output_dir "$TRAIN" \
  --anchor_seqs_json "$ASSETS/anchor_seqs.json" \
  --catalog_json "$ASSETS/catalog.json" \
  --per_bench_catalog_json "$ASSETS/per_bench_catalog.json" \
  --dense_deltas_jsonl "$ASSETS/dense_joint.jsonl" \
  --split_json "$JOINT_SPLIT" \
  --benchmarks commonsenseqa boolq arc_easy \
  --compressor_type last_token \
  --hidden_dims 1024 1024 512 \
  --epochs "$EPOCHS" \
  --batch_size 64 \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-joint-router}" \
  --wandb_run_name "${WANDB_RUN_NAME:-${TAG}_$(basename "$OUT")}" \
  --wandb_tags dense_joint greedy095 csqa boolq arc_easy \
  --asset_export_meta_json "$ASSETS/export_meta.json"

log "DONE joint=$JC_OUT checkpoint=$TRAIN/joint_router_best.pt summary=$OUT/pipeline_manifest.txt"

{
  echo "OUT=$OUT"
  echo "joint_catalogue=$JC_OUT"
  echo "assets=$ASSETS"
  echo "train=$TRAIN"
  echo "MAX_QUESTIONS=$MAX_Q SPLIT=$TRAIN_SPLITS MODEL=$MODEL_NAME"
} > "$OUT/pipeline_manifest.txt"
