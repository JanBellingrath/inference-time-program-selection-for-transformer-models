#!/usr/bin/env bash
# =============================================================================
# Generic, idempotent end-to-end miner for the compositional router.
# Works for any benchmark exposed by ``core.permutation_mcts.prepare_arc_data``
# (commonsenseqa, boolq, winogrande, arc_easy, arc_challenge, hellaswag, ...).
#
# This script is a *thin orchestrator* over existing pieces:
#
#   0. (optional) dr-llm/data_prep/build_fine_routing_dataset.py --use_mcts
#                                                   (split           -> fine_routing_data_<bench>/<bench>.jsonl
#                                                                       i.e. the per-question MCTS exploration data
#                                                                       whose primitives drive the compositional
#                                                                       program catalogue.  Skip with --skip_mcts
#                                                                       if the MCTS jsonl already exists.)
#   A. data_prep.canonicalize_programs              (MCTS jsonl  -> canonical)
#   B. data_prep.program_support                    (canonical   -> primitive_support.jsonl)
#   C. data_prep.build_compositional_catalogues     (canonical   -> primitives + legal_programs + manifest)
#                                                   (legal program catalogue = all admissible programs whose
#                                                    primitives all survived ``--min_count`` / ``--min_questions``
#                                                    over the MCTS-explored canonical rows)
#   D. scripts/build_dense_catalog_from_legal_programs
#                                                   (manifest    -> selected_catalog.json
#                                                                   = the route set the dense eval will mine)
#   E. dr-llm/data_prep/dense_reevaluation.py       (catalog x split  -> dense_deltas.jsonl + dense_deltas_matrix.pt)
#                                                   (every route applied to every question; resumes from the
#                                                    longest contiguous global_question_id prefix)
#   F. data_prep.import_mined_dense_matrix          (dense matrix -> compositional layout)
#   G. (optional) scripts/build_local_moebius_supervision.sh
#                                                   (local Mobius singleton/pair supervision)
#
# Everything is idempotent:
#   * Each stage is skipped when its primary output is already present.
#   * dense_reevaluation appends to dense_deltas.jsonl and resumes from the
#     longest contiguous global_question_id prefix, so partial runs only
#     "fill in the blanks" instead of recomputing.
#   * Each stage is wrapped in run_until_ok which retries on non-zero exit
#     until success (or until you SIGINT the supervisor twice).
#
# Defaults are tuned for hellaswag, but every input can be overridden.
#
# Usage
# -----
#   scripts/mine_compositional_supervision.sh \
#       [--benchmark hellaswag]                   # default
#       [--mcts_dir  /path/to/fine_routing_data_<bench>_mcts_*]
#                                                 # if not given, auto-detected; if not found
#                                                 # AND --skip_mcts is NOT set, Stage 0 will
#                                                 # build it from scratch via build_fine_routing_dataset.py
#       [--output_root compositional_runs/<bench>_pipeline]
#       [--model_name Qwen/Qwen2.5-0.5B-Instruct]
#       [--splits "train validation"]             # space-separated, used for Stages E/F
#       [--mcts_split train]                      # split to mine MCTS data on (Stage 0)
#       [--dr_llm_dir /path/to/dr-llm]
#       [--batch_size 8] [--gpu 0] [--save_interval 50]
#       [--adapter_path /path/to/lora_adapter]    # optional FT adapter (Stages 0 & E)
#       [--results_dir predictions/<...>]         # MCTS snapshot dir for anchor lookup (Stage 0)
#       [--max_questions N]                       # cap per split (debug)
#       [--min_count 50] [--min_questions 50]     # compositional primitive-support filters
#       [--include_local_moebius]                 # also run Mobius pipeline (Stage G)
#       [--include_pairs]                         # pair Mobius targets too
#       [--no_resume]                             # nuke existing dense state
#
# Stage 0 (MCTS) controls
# -----------------------
#       [--skip_mcts]                             # never run Stage 0 (fail fast if no MCTS data)
#       [--force_mcts]                            # always run Stage 0 even if MCTS dir exists
#       [--mcts_num_simulations 250]
#       [--max_local_edits 2] [--swap_radius 3]
#       [--editable_start 17] [--pivot_layer 16]
#       [--identity_anchor]                       # use identity anchor if no MCTS snapshot exists
#
# Background / supervisor mode
# ----------------------------
#   Wrap with nohup to survive terminal close:
#       nohup scripts/mine_compositional_supervision.sh > supervisor.out 2>&1 &
#   The script itself loops on every stage with retry-on-error and full
#   logging to ${OUTPUT_ROOT}/supervisor_<ts>.log.
# =============================================================================
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ---------------------------- defaults ---------------------------------------
BENCH="hellaswag"
MCTS_DIR=""
OUTPUT_ROOT=""
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
SPLITS="train validation"
MCTS_SPLIT="train"
DR_LLM_DIR="/home/janerik/generalized_transformer-2/dr-llm"
BATCH_SIZE=8
GPU=0
SAVE_INTERVAL=50
ADAPTER_PATH=""
RESULTS_DIR="predictions/qwen25_0.5b_v2_sdpa"
MAX_QUESTIONS=""
MIN_COUNT=50
MIN_QUESTIONS=50
INCLUDE_MOEBIUS=0
INCLUDE_PAIRS=0
NO_RESUME=0
# Stage 0 (MCTS) defaults — match the existing hellaswag MCTS config in
# fine_routing_data_hellaswag_mcts_identity_anchor/config.json
SKIP_MCTS=0
FORCE_MCTS=0
MCTS_NUM_SIMULATIONS=250
MAX_LOCAL_EDITS=2
SWAP_RADIUS=3
EDITABLE_START=17
PIVOT_LAYER=16
IDENTITY_ANCHOR=0

# ---------------------------- arg parsing ------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark)             BENCH="$2"; shift 2 ;;
    --mcts_dir)              MCTS_DIR="$2"; shift 2 ;;
    --output_root)           OUTPUT_ROOT="$2"; shift 2 ;;
    --model_name)            MODEL_NAME="$2"; shift 2 ;;
    --splits)                SPLITS="$2"; shift 2 ;;
    --mcts_split)            MCTS_SPLIT="$2"; shift 2 ;;
    --dr_llm_dir)            DR_LLM_DIR="$2"; shift 2 ;;
    --batch_size)            BATCH_SIZE="$2"; shift 2 ;;
    --gpu)                   GPU="$2"; shift 2 ;;
    --save_interval)         SAVE_INTERVAL="$2"; shift 2 ;;
    --adapter_path)          ADAPTER_PATH="$2"; shift 2 ;;
    --results_dir)           RESULTS_DIR="$2"; shift 2 ;;
    --max_questions)         MAX_QUESTIONS="$2"; shift 2 ;;
    --min_count)             MIN_COUNT="$2"; shift 2 ;;
    --min_questions)         MIN_QUESTIONS="$2"; shift 2 ;;
    --include_local_moebius) INCLUDE_MOEBIUS=1; shift ;;
    --include_pairs)         INCLUDE_PAIRS=1; shift ;;
    --no_resume)             NO_RESUME=1; shift ;;
    --skip_mcts)             SKIP_MCTS=1; shift ;;
    --force_mcts)            FORCE_MCTS=1; shift ;;
    --mcts_num_simulations)  MCTS_NUM_SIMULATIONS="$2"; shift 2 ;;
    --max_local_edits)       MAX_LOCAL_EDITS="$2"; shift 2 ;;
    --swap_radius)           SWAP_RADIUS="$2"; shift 2 ;;
    --editable_start)        EDITABLE_START="$2"; shift 2 ;;
    --pivot_layer)           PIVOT_LAYER="$2"; shift 2 ;;
    --identity_anchor)       IDENTITY_ANCHOR=1; shift ;;
    -h|--help)
      sed -n '2,60p' "$0"
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ---------------------------- defaults that depend on $BENCH -----------------
if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="${ROOT}/compositional_runs/${BENCH}_pipeline"
fi

# Where Stage 0 (if it runs) will write its MCTS data. Lives under OUTPUT_ROOT
# so the pipeline is fully self-contained per benchmark.
LOCAL_MCTS_DIR="${OUTPUT_ROOT}/mcts_data"

# MCTS dir resolution:
#   1) explicit --mcts_dir wins
#   2) fall back to the canonical local one we'd build in Stage 0
#   3) auto-detect any pre-existing fine_routing_data_<bench>* under dr-llm
#   4) leave empty -> Stage 0 will build into LOCAL_MCTS_DIR (unless --skip_mcts)
if [[ -z "$MCTS_DIR" ]]; then
  if [[ -f "${LOCAL_MCTS_DIR}/${BENCH}.jsonl" ]]; then
    MCTS_DIR="$LOCAL_MCTS_DIR"
  else
    candidates=( $(ls -dt "${DR_LLM_DIR}"/fine_routing_data_${BENCH}* 2>/dev/null | head -10) )
    for c in "${candidates[@]}"; do
      if [[ -f "${c}/${BENCH}.jsonl" ]]; then
        MCTS_DIR="$c"; break
      fi
    done
  fi
fi

mkdir -p "$OUTPUT_ROOT"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${OUTPUT_ROOT}/supervisor_${TS}.log"
ln -sfn "$(basename "$LOG")" "${OUTPUT_ROOT}/supervisor.log.latest" 2>/dev/null || true

CANON_DIR="${OUTPUT_ROOT}/canonical"
COMP_DIR="${OUTPUT_ROOT}/compositional"
CATALOG_JSON="${OUTPUT_ROOT}/dense_catalog/selected_catalog.json"
DENSE_ROOT="${OUTPUT_ROOT}/dense"          # one subdir per split
SUPERVISION_DIR="${OUTPUT_ROOT}/supervision"
MOEBIUS_ROOT="${OUTPUT_ROOT}/local_moebius"

mkdir -p "$(dirname "$CATALOG_JSON")" "$DENSE_ROOT" "$SUPERVISION_DIR"

log() { printf '[%s] %s\n' "$(date -Iseconds)" "$*" | tee -a "$LOG"; }

log "=== compositional supervision miner ==="
log "BENCH=$BENCH"
log "MCTS_DIR=${MCTS_DIR:-<will-build-in-stage-0>}"
log "LOCAL_MCTS_DIR=$LOCAL_MCTS_DIR"
log "OUTPUT_ROOT=$OUTPUT_ROOT"
log "MODEL_NAME=$MODEL_NAME"
log "SPLITS=$SPLITS  MCTS_SPLIT=$MCTS_SPLIT"
log "DR_LLM_DIR=$DR_LLM_DIR"
log "BATCH_SIZE=$BATCH_SIZE  GPU=$GPU  SAVE_INTERVAL=$SAVE_INTERVAL"
log "ADAPTER_PATH=${ADAPTER_PATH:-<none>}  RESULTS_DIR=$RESULTS_DIR"
log "MIN_COUNT=$MIN_COUNT  MIN_QUESTIONS=$MIN_QUESTIONS"
log "INCLUDE_MOEBIUS=$INCLUDE_MOEBIUS  INCLUDE_PAIRS=$INCLUDE_PAIRS  NO_RESUME=$NO_RESUME"
log "MCTS: SKIP=$SKIP_MCTS FORCE=$FORCE_MCTS sims=$MCTS_NUM_SIMULATIONS K=$MAX_LOCAL_EDITS R=$SWAP_RADIUS S=$EDITABLE_START pivot=$PIVOT_LAYER identity_anchor=$IDENTITY_ANCHOR"
log "LOG=$LOG"

run_until_ok() {
  local name="$1"; shift
  local attempt=0
  while true; do
    attempt=$((attempt + 1))
    log ">>> $name (attempt $attempt): $*"
    if "$@" 2>&1 | tee -a "$LOG"; then
      log ">>> $name finished OK"
      return 0
    fi
    log ">>> $name FAILED. Sleeping 60s before retry."
    sleep 60
  done
}

# ----------------------------------------------------------------------------
# Stage 0: build per-question MCTS exploration data (only if missing or forced)
#
# This is the upstream that determines which primitives have empirical support
# (and therefore appear in the compositional program catalogue).  We use the
# same defaults as the existing hellaswag MCTS run; override on the CLI for
# a different model/regime.
# ----------------------------------------------------------------------------
NEED_MCTS=0
if [[ "$FORCE_MCTS" -eq 1 ]]; then
  NEED_MCTS=1
elif [[ -z "$MCTS_DIR" || ! -f "${MCTS_DIR}/${BENCH}.jsonl" ]]; then
  NEED_MCTS=1
fi

if [[ "$NEED_MCTS" -eq 1 && "$SKIP_MCTS" -eq 1 ]]; then
  log "FATAL: no MCTS jsonl found at ${MCTS_DIR:-<unset>} and --skip_mcts was passed."
  log "       Either drop --skip_mcts to let Stage 0 build it, or pass --mcts_dir <existing>."
  exit 2
fi

if [[ "$NEED_MCTS" -eq 1 ]]; then
  MCTS_DIR="$LOCAL_MCTS_DIR"
  mkdir -p "$MCTS_DIR"
  IDENTITY_FLAG=()
  if [[ "$IDENTITY_ANCHOR" -eq 1 ]]; then
    IDENTITY_FLAG=( --identity_anchor_benchmarks "$BENCH" )
  fi
  ADAPTER_FLAG_MCTS=()
  # build_fine_routing_dataset.py does not currently take --adapter_path, so we
  # don't forward ADAPTER_PATH here. If you need an FT-adapter MCTS run, use
  # build_ft_fine_routing_dataset.py manually and point --mcts_dir at it.
  log "Stage 0: build_fine_routing_dataset.py --use_mcts (output=$MCTS_DIR)"
  run_until_ok "build_mcts_${BENCH}" \
    bash -lc "cd '$DR_LLM_DIR' && python data_prep/build_fine_routing_dataset.py \
      --model_name '$MODEL_NAME' \
      --results_dir '$RESULTS_DIR' \
      --benchmarks '$BENCH' \
      --output_dir '$MCTS_DIR' \
      --data_split '$MCTS_SPLIT' \
      --use_mcts \
      --mcts_num_simulations '$MCTS_NUM_SIMULATIONS' \
      --max_local_edits '$MAX_LOCAL_EDITS' \
      --swap_radius '$SWAP_RADIUS' \
      --editable_start '$EDITABLE_START' \
      --pivot_layer '$PIVOT_LAYER' \
      --gpu_rank '$GPU' \
      --resume \
      ${IDENTITY_FLAG[*]:-}"
else
  log "[skip] Stage 0: MCTS jsonl already present at ${MCTS_DIR}/${BENCH}.jsonl"
fi

if [[ ! -f "${MCTS_DIR}/${BENCH}.jsonl" ]]; then
  log "FATAL: still missing ${MCTS_DIR}/${BENCH}.jsonl after Stage 0; aborting."
  exit 2
fi

# ----------------------------------------------------------------------------
# Stage A: canonicalize MCTS jsonl
# ----------------------------------------------------------------------------
if [[ -f "${CANON_DIR}/${BENCH}.jsonl" ]]; then
  log "[skip] canonical already at ${CANON_DIR}/${BENCH}.jsonl"
else
  log "Stage A: canonicalize_programs (${MCTS_DIR} -> ${CANON_DIR})"
  run_until_ok "canonicalize_programs" \
    python -m data_prep.canonicalize_programs \
      --data_dir "$MCTS_DIR" \
      --output_dir "$CANON_DIR" \
      --benchmarks "$BENCH" \
      --copy-residuals
fi

# ----------------------------------------------------------------------------
# Stage B: primitive_support.jsonl  (filter universe for compositional)
# ----------------------------------------------------------------------------
if [[ -f "${CANON_DIR}/primitive_support.jsonl" ]]; then
  log "[skip] primitive_support already at ${CANON_DIR}/primitive_support.jsonl"
else
  log "Stage B: program_support over ${CANON_DIR}"
  run_until_ok "program_support" \
    python -m data_prep.program_support \
      --data_dir "$CANON_DIR" \
      --benchmarks "$BENCH"
fi

# ----------------------------------------------------------------------------
# Stage C: compositional manifest + legal_programs + observed
# ----------------------------------------------------------------------------
if [[ -f "${COMP_DIR}/manifest.json" && -f "${COMP_DIR}/legal_programs/${BENCH}.jsonl" ]]; then
  log "[skip] compositional manifest already at ${COMP_DIR}/manifest.json"
else
  log "Stage C: build_compositional_catalogues (${CANON_DIR} -> ${COMP_DIR})"
  run_until_ok "build_compositional_catalogues" \
    python -m data_prep.build_compositional_catalogues \
      --data_dir "$CANON_DIR" \
      --output_dir "$COMP_DIR" \
      --benchmarks "$BENCH" \
      --min_count "$MIN_COUNT" \
      --min_questions "$MIN_QUESTIONS"
fi

# ----------------------------------------------------------------------------
# Stage D: selected_catalog.json (route set to dense-evaluate)
# ----------------------------------------------------------------------------
if [[ -f "$CATALOG_JSON" ]]; then
  log "[skip] dense catalog already at $CATALOG_JSON"
else
  log "Stage D: build_dense_catalog_from_legal_programs"
  run_until_ok "build_dense_catalog_from_legal_programs" \
    python scripts/build_dense_catalog_from_legal_programs.py \
      --manifest "${COMP_DIR}/manifest.json" \
      --benchmark "$BENCH" \
      --output "$CATALOG_JSON"
fi

# Pull n_legal/n_questions for logging only.
N_ROUTES="$(python -c "import json; print(len(json.load(open('${CATALOG_JSON}'))['selected_routes']))" 2>/dev/null || echo '?')"
log "Selected route catalog: ${N_ROUTES} routes"

# ----------------------------------------------------------------------------
# Stage E: dense_reevaluation per split (the workhorse; resumes natively)
# ----------------------------------------------------------------------------
ADAPTER_FLAG=()
if [[ -n "$ADAPTER_PATH" ]]; then
  ADAPTER_FLAG=( --adapter_path "$ADAPTER_PATH" )
fi
MAX_Q_FLAG=()
if [[ -n "$MAX_QUESTIONS" ]]; then
  MAX_Q_FLAG=( --max_questions "$MAX_QUESTIONS" )
fi

for SPLIT in $SPLITS; do
  SPLIT_OUT="${DENSE_ROOT}/${SPLIT}"
  mkdir -p "$SPLIT_OUT"

  if [[ "$NO_RESUME" -eq 1 ]]; then
    log "[--no_resume] removing ${SPLIT_OUT}/dense_deltas.jsonl ${SPLIT_OUT}/dense_deltas_matrix.pt"
    rm -f "${SPLIT_OUT}/dense_deltas.jsonl" "${SPLIT_OUT}/dense_deltas_matrix.pt"
  fi

  if [[ -f "${SPLIT_OUT}/dense_deltas_matrix.pt" && ! -f "${SPLIT_OUT}/.force_rerun" ]]; then
    log "[skip] dense matrix already at ${SPLIT_OUT}/dense_deltas_matrix.pt (touch .force_rerun to redo)"
  else
    log "Stage E[$SPLIT]: dense_reevaluation (catalog=$N_ROUTES routes, output=$SPLIT_OUT)"
    # data_dir gives dense_reevaluation the anchor sequence from the canonical
    # JSONL's first record; merge_source_dir attaches the original MCTS row
    # for train (no-op when split is something other than the MCTS split).
    run_until_ok "dense_reevaluation_${SPLIT}" \
      bash -lc "cd '$DR_LLM_DIR' && python data_prep/dense_reevaluation.py \
        --catalog_json '$CATALOG_JSON' \
        --benchmarks '$BENCH' \
        --model_name '$MODEL_NAME' \
        --data_dir '$CANON_DIR' \
        --merge_source_dir '$MCTS_DIR' \
        --output_dir '$SPLIT_OUT' \
        --split '$SPLIT' \
        --batch_size '$BATCH_SIZE' \
        --save_interval '$SAVE_INTERVAL' \
        --gpu '$GPU' \
        ${ADAPTER_FLAG[*]:-} \
        ${MAX_Q_FLAG[*]:-}"
  fi

  # ------------------------------------------------------------------------
  # Stage F: import dense matrix into compositional layout
  # ------------------------------------------------------------------------
  IMPORT_OUT="${SUPERVISION_DIR}/dense_${BENCH}_${SPLIT}.pt"
  if [[ -f "$IMPORT_OUT" && ! -f "${SPLIT_OUT}/.force_rerun" ]]; then
    log "[skip] supervision tensor already at $IMPORT_OUT"
  else
    # Number of rows = number of dense_deltas.jsonl rows actually mined.
    N_Q_LOCAL="$(wc -l < "${SPLIT_OUT}/dense_deltas.jsonl" 2>/dev/null | tr -d ' ' || echo 0)"
    if [[ "$N_Q_LOCAL" -eq 0 ]]; then
      log "WARN: ${SPLIT_OUT}/dense_deltas.jsonl is empty; skipping import for $SPLIT"
    else
      log "Stage F[$SPLIT]: import_mined_dense_matrix (slicing $N_Q_LOCAL questions)"
      run_until_ok "import_mined_${SPLIT}" \
        python -m data_prep.import_mined_dense_matrix \
          --mined_pt "${SPLIT_OUT}/dense_deltas_matrix.pt" \
          --output "$IMPORT_OUT" \
          --num_questions "$N_Q_LOCAL"
    fi
  fi

  rm -f "${SPLIT_OUT}/.force_rerun"
done

# ----------------------------------------------------------------------------
# Stage G (optional): local Mobius supervision pipeline
# ----------------------------------------------------------------------------
if [[ "$INCLUDE_MOEBIUS" -eq 1 ]]; then
  PAIRS_FLAG=()
  if [[ "$INCLUDE_PAIRS" -eq 1 ]]; then
    PAIRS_FLAG=( --include_pairs )
  fi
  for SPLIT in $SPLITS; do
    SUB_ROOT="${MOEBIUS_ROOT}/${SPLIT}"
    mkdir -p "$SUB_ROOT"
    if [[ -f "${SUB_ROOT}/local_moebius/local_moebius_${BENCH}.pt" ]]; then
      log "[skip] Mobius targets already at ${SUB_ROOT}/local_moebius/local_moebius_${BENCH}.pt"
      continue
    fi
    log "Stage G[$SPLIT]: build_local_moebius_supervision.sh"
    run_until_ok "local_moebius_${SPLIT}" \
      bash scripts/build_local_moebius_supervision.sh \
        --manifest "${COMP_DIR}/manifest.json" \
        --output_root "$SUB_ROOT" \
        --benchmarks "$BENCH" \
        --model_name "$MODEL_NAME" \
        --dr_llm_dir "$DR_LLM_DIR" \
        --split "$SPLIT" \
        --batch_size "$BATCH_SIZE" \
        --gpu "$GPU" \
        "${PAIRS_FLAG[@]}"
  done
fi

log "=== ALL STAGES OK ==="
log "Compositional manifest:    ${COMP_DIR}/manifest.json"
log "Selected route catalog:    ${CATALOG_JSON}  (${N_ROUTES} routes)"
for SPLIT in $SPLITS; do
  log "Dense ${SPLIT}:               ${DENSE_ROOT}/${SPLIT}/dense_deltas_matrix.pt"
  log "Supervision ${SPLIT}:         ${SUPERVISION_DIR}/dense_${BENCH}_${SPLIT}.pt"
done
if [[ "$INCLUDE_MOEBIUS" -eq 1 ]]; then
  log "Mobius targets root:       ${MOEBIUS_ROOT}/<split>/local_moebius/"
fi
log "Full log: $LOG"
