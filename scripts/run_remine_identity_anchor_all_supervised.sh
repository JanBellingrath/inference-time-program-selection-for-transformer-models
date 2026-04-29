#!/usr/bin/env bash
# Run full compositional supervision pipeline (Stage 0 MCTS → dense) for
# arc_easy, hellaswag, commonsenseqa with identity default anchor [0..L-1],
# assign primitives included, continuous scoring — same knobs as prior assign mines.
#
# Uses two GPUs in parallel when available: arc_easy on GPU 0, hellaswag on GPU 1,
# then commonsenseqa on GPU 0.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DR_LLM="${DR_LLM_DIR:-/home/janerik/generalized_transformer-2/dr-llm}"
BASE="${ROOT}/compositional_runs/remine_identity_anchor_20260429"
mkdir -p "$BASE"

MAIN_LOG="${BASE}/orchestrator_$(date +%Y%m%d_%H%M%S).log"
ln -sfn "$(basename "$MAIN_LOG")" "${BASE}/orchestrator.log.latest" 2>/dev/null || true

log() { printf '[%s] %s\n' "$(date -Iseconds)" "$*" | tee -a "$MAIN_LOG"; }

run_one() {
  local bench="$1" cvd="$2" bs="$3"
  local blog="${BASE}/${bench}_run.log"
  log "=== START $bench (CUDA_VISIBLE_DEVICES=$cvd batch=$bs) log=$blog ==="
  (
    cd "$ROOT"
    export CUDA_VISIBLE_DEVICES="$cvd"
    bash scripts/mine_compositional_supervision.sh \
      --benchmark "$bench" \
      --output_root "${BASE}/${bench}" \
      --model_name Qwen/Qwen2.5-0.5B-Instruct \
      --dr_llm_dir "$DR_LLM" \
      --results_dir predictions/qwen25_0.5b_v2_sdpa \
      --mcts_split train \
      --splits "train validation" \
      --mcts_num_simulations 250 \
      --max_local_edits 2 \
      --swap_radius 3 \
      --editable_start 17 \
      --pivot_layer 16 \
      --batch_size "$bs" \
      --gpu 0 \
      --save_interval 50 \
      --keep_kinds skip repeat swap assign \
      --score_mode continuous \
      --min_count 50 \
      --min_questions 50 \
      --force_mcts \
      --mcts_anchor_source default
  ) >>"$blog" 2>&1
  log "=== DONE $bench ==="
}

log "Logging to $MAIN_LOG"
log "Output roots under $BASE/{arc_easy,hellaswag,commonsenseqa}"

run_one arc_easy 0 4 &
PID_ARC=$!
run_one hellaswag 1 8 &
PID_H=$!
wait $PID_ARC
wait $PID_H

run_one commonsenseqa 0 4

log "ALL BENCHMARKS FINISHED OK"
