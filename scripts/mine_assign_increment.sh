#!/usr/bin/env bash
# End-to-end driver for incremental "assign" dense-data mining.
#
# Given an existing struct-only dense matrix (skip|repeat|swap), this script:
#   1. Re-canonicalizes the raw MCTS data with --include_assign  (existing CLI)
#   2. Re-builds primitive_support / compositional manifest with assign      (existing CLI)
#   3. Builds a *delta* selected_catalog.json containing only the routes that
#      the assign-extended catalogue introduces                                     (NEW)
#   4. Calls python -m data_prep.dense_reevaluation on the delta catalog
#      (uses the same prefix-trie hidden-state caching as the original run)         (existing CLI)
#   5. Merges old + new dense outputs into a unified dense_deltas_matrix.pt /
#      dense_deltas.jsonl / selected_catalog.json, ordered to match the
#      assign-extended manifest's legal-program ordering                              (NEW)
#
# Required args
# -------------
#   --raw_data_dir            fine_routing_data/<run>/                              (raw MCTS jsonl)
#   --new_canonical_dir       fine_routing_data/<run>_canonical_assign/             (re-canonicalized, NEW)
#   --new_compositional_dir   fine_routing_data/<run>_compositional_assign/         (rebuilt, NEW)
#   --old_dense_dir           dense_artifacts/.../decode_compositional/             (struct-only dense)
#   --increment_catalog_dir   .../catalog_assign_increment/                         (delta catalog)
#   --new_dense_dir           .../decode_assign_increment/                          (delta dense outputs)
#   --unified_dense_dir       .../decode_compositional_unified/                     (final unified outputs)
#   --bench                   commonsenseqa
#   --model_name              Qwen/Qwen2.5-0.5B-Instruct
#
# Optional
# --------
#   --dr_llm_dir              deprecated, ignored (kept so old wrappers parse)

#   --adapter_path PATH       LoRA adapter dir (forwarded to dense_reevaluation)
#   --split STR               default: train (matches the original Stage A run)
#   --batch_size N            default: 4
#   --gpu N                   default: 0
#   --keep_kinds "skip repeat swap assign"
#   --max_program_len N       default: read from canonical config
#   --swap_radius N           default: read from canonical config
#   --editable_start N        default: read from canonical config
#   --dedupe_assign_with_struct      passes --dedupe_assign_with_struct to canonicalize
#   --skip_canonicalize       skip stage 1 (e.g. if already done)
#   --skip_program_support    skip stage 2a
#   --skip_compositional      skip stage 2b
#   --skip_increment          skip stage 3
#   --skip_dense              skip stage 4
#   --skip_merge              skip stage 5
#   --score_mode              default: continuous (must match old dense matrix)
#   --no_merge_mcts           passes --no_merge_mcts to dense_reevaluation
#
# Example (CSQA, FT Qwen0.5B)
# ---------------------------
#   ./scripts/mine_assign_increment.sh \
#     --raw_data_dir            /.../fine_routing_data_ft_qwen05b_250sims_continuous_commonsenseqa \
#     --new_canonical_dir       /.../fine_routing_data_ft_qwen05b_250sims_continuous_commonsenseqa_canonical_assign \
#     --new_compositional_dir   /.../fine_routing_data_ft_qwen05b_250sims_continuous_commonsenseqa_compositional_assign \
#     --old_dense_dir           /.../dense_artifacts_csqa_ft_2026/decode_compositional \
#     --increment_catalog_dir   /.../dense_artifacts_csqa_ft_2026/catalog_assign_increment \
#     --new_dense_dir           /.../dense_artifacts_csqa_ft_2026/decode_assign_increment \
#     --unified_dense_dir       /.../dense_artifacts_csqa_ft_2026/decode_compositional_unified \
#     --bench commonsenseqa \
#     --model_name Qwen/Qwen2.5-0.5B-Instruct \
#     --dr_llm_dir /home/janerik/generalized_transformer-2/dr-llm \
#     --adapter_path /.../ft_study_results_v7/commonsenseqa/seed_42/ft_only/checkpoints/final_adapter \
#     --split train --batch_size 4 --gpu 0

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ---- defaults ------------------------------------------------------------
SPLIT="train"
BATCH_SIZE=4
GPU=0
SCORE_MODE="continuous"
KEEP_KINDS="skip repeat swap assign"
MAX_PROG_LEN=""
SWAP_RADIUS=""
EDITABLE_START=""
MIN_COUNT=""
MIN_QUESTIONS=""
MIN_BENCHMARKS=""
DEDUPE_ASSIGN=0
SKIP_CANON=0
SKIP_SUPPORT=0
SKIP_COMP=0
SKIP_INCR=0
SKIP_DENSE=0
SKIP_MERGE=0
NO_MERGE_MCTS=0
ADAPTER_PATH=""

# ---- required ------------------------------------------------------------
RAW_DATA_DIR=""
NEW_CANONICAL_DIR=""
NEW_COMPOSITIONAL_DIR=""
OLD_DENSE_DIR=""
INCREMENT_CATALOG_DIR=""
NEW_DENSE_DIR=""
UNIFIED_DENSE_DIR=""
BENCH=""
MODEL_NAME=""
DR_LLM_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --raw_data_dir)            RAW_DATA_DIR="$2"; shift 2 ;;
    --new_canonical_dir)       NEW_CANONICAL_DIR="$2"; shift 2 ;;
    --new_compositional_dir)   NEW_COMPOSITIONAL_DIR="$2"; shift 2 ;;
    --old_dense_dir)           OLD_DENSE_DIR="$2"; shift 2 ;;
    --increment_catalog_dir)   INCREMENT_CATALOG_DIR="$2"; shift 2 ;;
    --new_dense_dir)           NEW_DENSE_DIR="$2"; shift 2 ;;
    --unified_dense_dir)       UNIFIED_DENSE_DIR="$2"; shift 2 ;;
    --bench)                   BENCH="$2"; shift 2 ;;
    --model_name)              MODEL_NAME="$2"; shift 2 ;;
    --dr_llm_dir)              DR_LLM_DIR="$2"; shift 2 ;;
    --adapter_path)            ADAPTER_PATH="$2"; shift 2 ;;
    --split)                   SPLIT="$2"; shift 2 ;;
    --batch_size)              BATCH_SIZE="$2"; shift 2 ;;
    --gpu)                     GPU="$2"; shift 2 ;;
    --keep_kinds)              KEEP_KINDS="$2"; shift 2 ;;
    --max_program_len)         MAX_PROG_LEN="$2"; shift 2 ;;
    --swap_radius)             SWAP_RADIUS="$2"; shift 2 ;;
    --editable_start)          EDITABLE_START="$2"; shift 2 ;;
    --min_count)               MIN_COUNT="$2"; shift 2 ;;
    --min_questions)           MIN_QUESTIONS="$2"; shift 2 ;;
    --min_benchmarks)          MIN_BENCHMARKS="$2"; shift 2 ;;
    --dedupe_assign_with_struct) DEDUPE_ASSIGN=1; shift ;;
    --skip_canonicalize)       SKIP_CANON=1; shift ;;
    --skip_program_support)    SKIP_SUPPORT=1; shift ;;
    --skip_compositional)      SKIP_COMP=1; shift ;;
    --skip_increment)          SKIP_INCR=1; shift ;;
    --skip_dense)              SKIP_DENSE=1; shift ;;
    --skip_merge)              SKIP_MERGE=1; shift ;;
    --no_merge_mcts)           NO_MERGE_MCTS=1; shift ;;
    --score_mode)              SCORE_MODE="$2"; shift 2 ;;
    -h|--help) sed -n '2,80p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -n "${DR_LLM_DIR:-}" ]]; then
  echo "[mine_assign_increment] ignoring deprecated --dr_llm_dir (${DR_LLM_DIR})" >&2
fi

for v in RAW_DATA_DIR NEW_CANONICAL_DIR NEW_COMPOSITIONAL_DIR OLD_DENSE_DIR \
         INCREMENT_CATALOG_DIR NEW_DENSE_DIR UNIFIED_DENSE_DIR BENCH MODEL_NAME; do
  if [[ -z "${!v}" ]]; then
    echo "Missing --$(echo "$v" | tr '[:upper:]' '[:lower:]')" >&2
    exit 2
  fi
done

mkdir -p "$NEW_CANONICAL_DIR" "$NEW_COMPOSITIONAL_DIR" "$INCREMENT_CATALOG_DIR" \
         "$NEW_DENSE_DIR" "$UNIFIED_DENSE_DIR"

# -------------------------------------------------------------------------
# Stage 1: re-canonicalize with --include_assign (existing tool)
# -------------------------------------------------------------------------
if [[ "$SKIP_CANON" -eq 0 ]]; then
  echo "=== Stage 1: canonicalize_programs (--include_assign) ==="
  CANON_EXTRA=()
  [[ -n "$MAX_PROG_LEN" ]] && CANON_EXTRA+=(--max_program_len "$MAX_PROG_LEN")
  [[ -n "$SWAP_RADIUS"  ]] && CANON_EXTRA+=(--swap_radius "$SWAP_RADIUS")
  [[ -n "$EDITABLE_START" ]] && CANON_EXTRA+=(--editable_start "$EDITABLE_START")
  [[ "$DEDUPE_ASSIGN" -eq 1 ]] && CANON_EXTRA+=(--dedupe_assign_with_struct)
  python -m data_prep.canonicalize_programs \
    --data_dir "$RAW_DATA_DIR" \
    --output_dir "$NEW_CANONICAL_DIR" \
    --include_assign \
    --benchmarks "$BENCH" \
    "${CANON_EXTRA[@]}"
fi

# -------------------------------------------------------------------------
# Stage 2a: rebuild primitive_support.jsonl (existing tool)
# -------------------------------------------------------------------------
if [[ "$SKIP_SUPPORT" -eq 0 ]]; then
  echo "=== Stage 2a: program_support (assign-aware) ==="
  python -m data_prep.program_support \
    --data_dir "$NEW_CANONICAL_DIR" \
    --output_dir "$NEW_CANONICAL_DIR" \
    --benchmarks "$BENCH"
fi

# -------------------------------------------------------------------------
# Stage 2b: rebuild compositional manifest (existing tool, keep assign)
# -------------------------------------------------------------------------
if [[ "$SKIP_COMP" -eq 0 ]]; then
  echo "=== Stage 2b: build_compositional_catalogues (keep assign) ==="
  COMP_EXTRA=()
  [[ -n "$MAX_PROG_LEN" ]] && COMP_EXTRA+=(--max_program_len "$MAX_PROG_LEN")
  [[ -n "$SWAP_RADIUS"  ]] && COMP_EXTRA+=(--swap_radius "$SWAP_RADIUS")
  [[ -n "$EDITABLE_START" ]] && COMP_EXTRA+=(--editable_start "$EDITABLE_START")
  [[ -n "$MIN_COUNT"     ]] && COMP_EXTRA+=(--min_count "$MIN_COUNT")
  [[ -n "$MIN_QUESTIONS" ]] && COMP_EXTRA+=(--min_questions "$MIN_QUESTIONS")
  [[ -n "$MIN_BENCHMARKS" ]] && COMP_EXTRA+=(--min_benchmarks "$MIN_BENCHMARKS")
  python -m data_prep.build_compositional_catalogues \
    --data_dir "$NEW_CANONICAL_DIR" \
    --output_dir "$NEW_COMPOSITIONAL_DIR" \
    --benchmarks "$BENCH" \
    --keep_kinds $KEEP_KINDS \
    "${COMP_EXTRA[@]}"
fi

# -------------------------------------------------------------------------
# Stage 3: build *delta* selected_catalog.json (NEW)
# -------------------------------------------------------------------------
if [[ "$SKIP_INCR" -eq 0 ]]; then
  echo "=== Stage 3: build_assign_increment_catalog (delta routes only) ==="
  python -m data_prep.build_assign_increment_catalog \
    --new_compositional_dir "$NEW_COMPOSITIONAL_DIR" \
    --bench "$BENCH" \
    --existing_dense_matrix "$OLD_DENSE_DIR/dense_deltas_matrix.pt" \
    --output_dir "$INCREMENT_CATALOG_DIR"
fi

# -------------------------------------------------------------------------
# Stage 4: dense_reevaluation on the delta routes (existing tool, prefix cache)
# -------------------------------------------------------------------------
if [[ "$SKIP_DENSE" -eq 0 ]]; then
  echo "=== Stage 4: dense_reevaluation (delta catalog only) ==="
  ADAPTER_FLAG=()
  [[ -n "$ADAPTER_PATH" ]] && ADAPTER_FLAG+=(--adapter_path "$ADAPTER_PATH")
  MERGE_FLAG=()
  [[ "$NO_MERGE_MCTS" -eq 1 ]] && MERGE_FLAG+=(--no_merge_mcts)
  ( cd "$ROOT" && \
    python -m data_prep.dense_reevaluation \
      --catalog_json "$INCREMENT_CATALOG_DIR/selected_catalog.json" \
      --benchmarks "$BENCH" \
      --model_name "$MODEL_NAME" \
      --data_dir "$NEW_CANONICAL_DIR" \
      --merge_source_dir "$RAW_DATA_DIR" \
      --output_dir "$NEW_DENSE_DIR" \
      --split "$SPLIT" \
      --batch_size "$BATCH_SIZE" \
      --save_interval 25 \
      --gpu "$GPU" \
      --score_mode "$SCORE_MODE" \
      "${ADAPTER_FLAG[@]}" \
      "${MERGE_FLAG[@]}" )
fi

# -------------------------------------------------------------------------
# Stage 5: merge old + new into the unified dense set (NEW)
# -------------------------------------------------------------------------
if [[ "$SKIP_MERGE" -eq 0 ]]; then
  echo "=== Stage 5: merge_dense_increment (unified ordering = new manifest) ==="
  python -m data_prep.merge_dense_increment \
    --old_dense_dir "$OLD_DENSE_DIR" \
    --new_dense_dir "$NEW_DENSE_DIR" \
    --new_compositional_dir "$NEW_COMPOSITIONAL_DIR" \
    --bench "$BENCH" \
    --output_dir "$UNIFIED_DENSE_DIR"
fi

echo
echo "Done."
echo "Unified dense matrix:    $UNIFIED_DENSE_DIR/dense_deltas_matrix.pt"
echo "Unified per-question:    $UNIFIED_DENSE_DIR/dense_deltas.jsonl"
echo "Unified selected_catalog $UNIFIED_DENSE_DIR/selected_catalog.json"
echo "New compositional dir:   $NEW_COMPOSITIONAL_DIR"
echo
echo "Train the compositional router with the unified data, e.g.:"
echo "  --compositional_manifest $NEW_COMPOSITIONAL_DIR/manifest.json"
echo "  --dense_deltas_paths     ${BENCH}=$UNIFIED_DENSE_DIR/dense_deltas_matrix.pt"
