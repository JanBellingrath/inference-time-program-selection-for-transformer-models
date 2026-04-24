#!/usr/bin/env bash
# Merge struct (179) + assign increment dense outputs for CSQA non-FT.
# Requires: continuous dense_deltas_matrix.pt in OLD_DENSE and NEW_DENSE.
set -euo pipefail

ROOT="/home/janerik/generalized_transformer-2/dr-llm"
FLEX="/home/janerik/flexible-test-time-program-selection"
cd "$FLEX"

OLD_DENSE="${ROOT}/dense_eval/csqa_compositional_179_train"
NEW_DENSE="${ROOT}/dense_artifacts_csqa_nonft_2026/decode_assign_increment"
COMP_ASSIGN="${ROOT}/fine_routing_data_commonsenseqa_mcts_compositional_assign"
UNIFIED="${ROOT}/dense_artifacts_csqa_nonft_2026/decode_compositional_unified"

mkdir -p "$UNIFIED"

python -m data_prep.merge_dense_increment \
  --old_dense_dir         "$OLD_DENSE" \
  --new_dense_dir         "$NEW_DENSE" \
  --new_compositional_dir "$COMP_ASSIGN" \
  --bench                 commonsenseqa \
  --output_dir            "$UNIFIED"

echo "[merge_csqa_nonft_assign_increment_only] DONE -> $UNIFIED/dense_deltas_matrix.pt"
