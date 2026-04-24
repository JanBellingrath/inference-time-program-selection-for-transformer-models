#!/usr/bin/env bash
# =============================================================================
# CSQA (non fine-tuned Qwen2.5-0.5B-Instruct): assign-inclusive compositional
# catalogue — incremental dense data generation.
#
# Reuses the existing struct-only compositional dense matrix (179 routes,
# 9741 questions) at:
#   /home/janerik/generalized_transformer-2/dr-llm/dense_eval/csqa_compositional_179_train/dense_deltas_matrix.pt
# and only mines the *delta* routes that the assign-extended catalogue
# introduces (programs containing >= 1 assign primitive whose induced layer
# route is not yet covered by the struct-only matrix).
#
# Wraps scripts/mine_assign_increment.sh with non-FT-specific paths and no
# LoRA adapter (base model only).
#
# score_mode is ``continuous`` (log-prob deltas). The struct-only baseline at
# OLD_DENSE must also be a *continuous* dense matrix (same model, same
# canonical_assign data_dir as this pipeline). If you only have a legacy
# binary baseline, run scripts/regen_csqa_nonft_struct179_dense_continuous.sh
# once to rebuild it before merging assign increments.
# =============================================================================
set -euo pipefail

ROOT="/home/janerik/generalized_transformer-2/dr-llm"
FLEX="/home/janerik/flexible-test-time-program-selection"
cd "$FLEX"

RAW="${ROOT}/fine_routing_data_commonsenseqa_mcts"
ART="${ROOT}/dense_artifacts_csqa_nonft_2026"
OLD_DENSE="${ROOT}/dense_eval/csqa_compositional_179_train"

CANON_ASSIGN="${RAW}_canonical_assign"
COMP_ASSIGN="${RAW}_compositional_assign"
CATALOG_INCR="${ART}/catalog_assign_increment"
NEW_DENSE="${ART}/decode_assign_increment"
UNIFIED="${ART}/decode_compositional_unified"

mkdir -p "$ART" "$CATALOG_INCR" "$NEW_DENSE" "$UNIFIED"

bash scripts/mine_assign_increment.sh \
  --raw_data_dir          "$RAW" \
  --new_canonical_dir     "$CANON_ASSIGN" \
  --new_compositional_dir "$COMP_ASSIGN" \
  --old_dense_dir         "$OLD_DENSE" \
  --increment_catalog_dir "$CATALOG_INCR" \
  --new_dense_dir         "$NEW_DENSE" \
  --unified_dense_dir     "$UNIFIED" \
  --bench                 commonsenseqa \
  --model_name            Qwen/Qwen2.5-0.5B-Instruct \
  --dr_llm_dir            "$ROOT" \
  --split                 train \
  --batch_size            "${BATCH_SIZE:-4}" \
  --gpu                   "${GPU:-0}" \
  --score_mode            continuous \
  --keep_kinds            "skip repeat swap assign" \
  --max_program_len       2 \
  --swap_radius           3 \
  --editable_start        17 \
  --min_count             50 \
  --min_questions         50

echo
echo "[csqa_nonft_assign_increment] DONE"
echo "  Unified dense matrix:  ${UNIFIED}/dense_deltas_matrix.pt"
echo "  Unified per-question:  ${UNIFIED}/dense_deltas.jsonl"
echo "  New compositional dir: ${COMP_ASSIGN}"
