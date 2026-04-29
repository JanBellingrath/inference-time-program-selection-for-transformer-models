#!/usr/bin/env bash
# =============================================================================
# One-time (or repeat-safe): rebuild the 179-route struct-only CSQA dense
# matrix in continuous mode so it can merge with assign increments.
#
# Moves the existing dense_eval/csqa_compositional_179_train tree to
# dense_eval/csqa_compositional_179_train_legacy_binary (if present), extracts
# route list from that legacy matrix, then runs dense_reevaluation with
# --score_mode continuous on canonical_assign (same anchors as assign pipeline).
#
# After this finishes, restore decode_assign_increment from your continuous
# backup if needed and run merge_dense_increment (or re-run
# mine_csqa_nonft_assign_increment.sh with --skip_* flags through merge only).
# =============================================================================
set -euo pipefail

ROOT="/home/janerik/generalized_transformer-2/dr-llm"
FLEX="/home/janerik/flexible-test-time-program-selection"
RAW="${ROOT}/fine_routing_data_commonsenseqa_mcts"
CANON_ASSIGN="${RAW}_canonical_assign"
LEGACY="${ROOT}/dense_eval/csqa_compositional_179_train_legacy_binary"
OUT="${ROOT}/dense_eval/csqa_compositional_179_train"
CATALOG_DIR="${ROOT}/dense_artifacts_csqa_nonft_2026/catalog_struct_179"

mkdir -p "$CATALOG_DIR"

if [[ -d "$OUT" && -f "$OUT/dense_deltas_matrix.pt" ]]; then
  ex_mode="$(python3 -c "import torch; print(torch.load('$OUT/dense_deltas_matrix.pt', map_location='cpu', weights_only=False).get('score_mode',''))" 2>/dev/null || echo "")"
  if [[ "$ex_mode" == "continuous" ]]; then
    echo "[regen_csqa_struct179] $OUT already has continuous matrix; nothing to do."
    exit 0
  fi
fi

if [[ ! -d "$LEGACY" ]]; then
  if [[ -d "$OUT" ]]; then
    echo "[regen_csqa_struct179] Archiving current $OUT -> $LEGACY"
    mv "$OUT" "$LEGACY"
  else
    echo "FATAL: no legacy dir $LEGACY and no $OUT to archive." >&2
    exit 2
  fi
fi

LEGACY_PT="$LEGACY/dense_deltas_matrix.pt"
if [[ ! -f "$LEGACY_PT" ]]; then
  LEGACY_PT="$LEGACY/dense_deltas_matrix.pt.pre_score_mode_tag.bak"
fi
if [[ ! -f "$LEGACY_PT" ]]; then
  echo "FATAL: no matrix in $LEGACY" >&2
  exit 2
fi

echo "[regen_csqa_struct179] Building selected_catalog from $LEGACY_PT"
python3 - "$LEGACY_PT" "$CATALOG_DIR/selected_catalog.json" <<'PY'
import json, sys, torch
src, dst = sys.argv[1], sys.argv[2]
d = torch.load(src, map_location="cpu", weights_only=False)
routes = d["routes"]
with open(dst, "w") as f:
    json.dump({"selected_routes": routes}, f)
print("routes", len(routes))
PY

mkdir -p "$OUT"
rm -f "$OUT/.dense_eval.lock"

echo "[regen_csqa_struct179] dense_reevaluation -> $OUT"
cd "$FLEX"
python -m data_prep.dense_reevaluation \
  --catalog_json "$CATALOG_DIR/selected_catalog.json" \
  --benchmarks commonsenseqa \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --data_dir "$CANON_ASSIGN" \
  --merge_source_dir "$RAW" \
  --output_dir "$OUT" \
  --split train \
  --batch_size "${BATCH_SIZE:-2}" \
  --save_interval 50 \
  --gpu "${GPU:-0}" \
  --score_mode continuous

echo "[regen_csqa_struct179] DONE: $OUT/dense_deltas_matrix.pt"
echo "Next (unified 179+1006 routes):"
echo "  bash $FLEX/scripts/merge_csqa_nonft_assign_increment_only.sh"
