#!/usr/bin/env bash
# End-to-end driver for local Mobius supervision data.
#
# Stages
# ------
#   1. data_prep.build_local_subset_catalog
#         compositional manifest -> per-bench selected_catalog.json + route_subsets.json
#   2. python -m data_prep.dense_reevaluation (per benchmark)
#         selected_catalog.json -> dense_deltas_matrix.pt
#         (reuses the prefix-trie hidden-state caching; no new decoding code)
#   3. data_prep.build_local_moebius_targets
#         route_subsets.json + dense_deltas_matrix.pt -> local_moebius_{bench}.pt
#         (consumed by CompositionalDataset._load_local_moebius)
#
# Required env / args
# -------------------
#   --manifest        path to fine_routing_data/<run>_compositional/manifest.json
#   --output_root     where to put local_subsets/, decode/, local_moebius/
#   --benchmarks      space-separated list, e.g. "commonsenseqa boolq"
#   --model_name      e.g. Qwen/Qwen2.5-0.5B-Instruct
#   [--dr_llm_dir]    deprecated, ignored (kept so old invocations parse)
#   [--include_pairs] also enumerate pair subsets (needed for pair Mobius targets)
#   [--split STR]     split passed to dense_reevaluation (default: validation)
#   [--max_questions N] forwarded to dense_reevaluation (default: unset = all)
#   [--batch_size N]  forwarded to dense_reevaluation (default: 1)
#   [--gpu N]         forwarded to dense_reevaluation (default: 0)
#
# Example
# -------
#   ./scripts/build_local_moebius_supervision.sh \
#       --manifest fine_routing_data/csqa_run_compositional/manifest.json \
#       --output_root local_moebius_runs/csqa \
#       --benchmarks "commonsenseqa" \
#       --model_name Qwen/Qwen2.5-0.5B-Instruct \
#       --dr_llm_dir /home/janerik/generalized_transformer-2/dr-llm \
#       --include_pairs

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

INCLUDE_PAIRS=0
SPLIT="validation"
BATCH_SIZE=1
GPU=0
MAX_QUESTIONS=""
MANIFEST=""
OUTPUT_ROOT=""
BENCHMARKS=""
MODEL_NAME=""
DR_LLM_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)       MANIFEST="$2"; shift 2 ;;
    --output_root)    OUTPUT_ROOT="$2"; shift 2 ;;
    --benchmarks)     BENCHMARKS="$2"; shift 2 ;;
    --model_name)     MODEL_NAME="$2"; shift 2 ;;
    --dr_llm_dir)     DR_LLM_DIR="$2"; shift 2 ;;
    --include_pairs)  INCLUDE_PAIRS=1; shift ;;
    --split)          SPLIT="$2"; shift 2 ;;
    --max_questions)  MAX_QUESTIONS="$2"; shift 2 ;;
    --batch_size)     BATCH_SIZE="$2"; shift 2 ;;
    --gpu)            GPU="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,40p' "$0"
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -n "${DR_LLM_DIR:-}" ]]; then
  echo "[build_local_moebius_supervision] ignoring deprecated --dr_llm_dir (${DR_LLM_DIR}); dense_eval runs from repo root." >&2
fi

for v in MANIFEST OUTPUT_ROOT BENCHMARKS MODEL_NAME; do
  if [[ -z "${!v}" ]]; then
    echo "Missing --$(echo "$v" | tr '[:upper:]' '[:lower:]')" >&2
    exit 2
  fi
done

CATALOG_DIR="${OUTPUT_ROOT}/local_subsets"
DECODE_DIR="${OUTPUT_ROOT}/decode"
MOEBIUS_DIR="${OUTPUT_ROOT}/local_moebius"
mkdir -p "$CATALOG_DIR" "$DECODE_DIR" "$MOEBIUS_DIR"

echo "=== Stage 1: build per-bench subset catalogs ==="
PAIRS_FLAG=""
if [[ "$INCLUDE_PAIRS" -eq 1 ]]; then
  PAIRS_FLAG="--include_pairs"
fi
python -m data_prep.build_local_subset_catalog \
  --manifest "$MANIFEST" \
  --output_dir "$CATALOG_DIR" \
  --benchmarks $BENCHMARKS \
  $PAIRS_FLAG

echo "=== Stage 2: dense_reevaluation per benchmark (this repo) ==="
MAX_Q_FLAG=""
if [[ -n "$MAX_QUESTIONS" ]]; then
  MAX_Q_FLAG="--max_questions $MAX_QUESTIONS"
fi
for bench in $BENCHMARKS; do
  cat_json="$CATALOG_DIR/$bench/selected_catalog.json"
  out_dir="$DECODE_DIR/$bench"
  if [[ ! -f "$cat_json" ]]; then
    echo "  [skip] no catalog at $cat_json"
    continue
  fi
  mkdir -p "$out_dir"
  echo "  -> $bench"
  ( cd "$ROOT" && \
    python -m data_prep.dense_reevaluation \
      --catalog_json "$cat_json" \
      --benchmarks "$bench" \
      --model_name "$MODEL_NAME" \
      --split "$SPLIT" \
      --batch_size "$BATCH_SIZE" \
      --gpu "$GPU" \
      $MAX_Q_FLAG \
      --output_dir "$out_dir" )
done

echo "=== Stage 3: materialize local_moebius_{bench}.pt ==="
python -m data_prep.build_local_moebius_targets \
  --catalog_dir "$CATALOG_DIR" \
  --decode_dir "$DECODE_DIR" \
  --output_dir "$MOEBIUS_DIR" \
  --benchmarks $BENCHMARKS

echo
echo "Done."
echo "Per-bench Mobius artifacts in: $MOEBIUS_DIR"
echo "Train with:  --local_moebius_dir $MOEBIUS_DIR --use_local_unary [--use_local_pair]"
