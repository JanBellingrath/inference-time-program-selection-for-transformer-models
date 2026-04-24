#!/usr/bin/env bash
# =============================================================================
# Hellaswag (Qwen2.5-0.5B-Instruct, identity anchor):
# assign-inclusive compositional catalogue — incremental dense data generation.
#
# Reuses the existing struct-only compositional dense matrix
# (259 routes, 39905 questions) at:
#   compositional_runs/hellaswag_pipeline/dense/train/dense_deltas_matrix.pt
# and only mines the *delta* routes that the assign-extended catalogue
# introduces.
# =============================================================================
set -euo pipefail

ROOT="/home/janerik/generalized_transformer-2/dr-llm"
FLEX="/home/janerik/flexible-test-time-program-selection"
cd "$FLEX"

RAW="${ROOT}/fine_routing_data_hellaswag_mcts_identity_anchor"
ART="${FLEX}/compositional_runs/hellaswag_pipeline"
OLD_DENSE="${ART}/dense/train"

CANON_ASSIGN="${ART}/canonical_assign"
COMP_ASSIGN="${ART}/compositional_assign"
CATALOG_INCR="${ART}/catalog_assign_increment"
NEW_DENSE="${ART}/dense_assign_increment/train"
UNIFIED="${ART}/dense_compositional_unified/train"

mkdir -p "$CANON_ASSIGN" "$COMP_ASSIGN" "$CATALOG_INCR" "$NEW_DENSE" "$UNIFIED"

bash scripts/mine_assign_increment.sh \
  --raw_data_dir          "$RAW" \
  --new_canonical_dir     "$CANON_ASSIGN" \
  --new_compositional_dir "$COMP_ASSIGN" \
  --old_dense_dir         "$OLD_DENSE" \
  --increment_catalog_dir "$CATALOG_INCR" \
  --new_dense_dir         "$NEW_DENSE" \
  --unified_dense_dir     "$UNIFIED" \
  --bench                 hellaswag \
  --model_name            Qwen/Qwen2.5-0.5B-Instruct \
  --dr_llm_dir            "$ROOT" \
  --split                 train \
  --batch_size            "${BATCH_SIZE:-8}" \
  --gpu                   "${GPU:-0}" \
  --score_mode            continuous \
  --keep_kinds            "skip repeat swap assign" \
  --max_program_len       2 \
  --swap_radius           3 \
  --editable_start        17 \
  --min_count             50 \
  --min_questions         50

echo
echo "[hellaswag_assign_increment] DONE"
echo "  Unified dense matrix:  ${UNIFIED}/dense_deltas_matrix.pt"
echo "  Unified per-question:  ${UNIFIED}/dense_deltas.jsonl"
echo "  New compositional dir: ${COMP_ASSIGN}"
