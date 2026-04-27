#!/usr/bin/env bash
# Rebuild canonical splits + mass-pruned catalogues + restart the four
# CSQA HPO sweeps (compositional FT/non-FT, joint-router FT/non-FT) with
# the train-only-mass-coverage fix applied.
#
# This script is idempotent:
#  * Splits are written to splits/csqa_{ft,nonft}_canonical_split.json
#  * Rebuilt mass-pruned compositional catalogues are written next to the
#    old ones with a ``_train_only`` suffix, leaving the originals in
#    place so old dashboards keep resolving.
#  * The joint-router reduce-dense step is invoked inside
#    supervise_csqa_mass095_online_sweep.sh (via --split_json); nothing
#    to pre-build here.
#
# Usage (dry-run: splits + catalogues only):
#   ./scripts/restart_csqa_sweeps.sh --stage prepare
#
# Usage (launch the four sweeps under nohup with default GPU placement):
#   ./scripts/restart_csqa_sweeps.sh --stage launch
#
# Usage (do both):
#   ./scripts/restart_csqa_sweeps.sh --stage all
#
# GPU assignment defaults (override via env):
#   COMP_NONFT_GPU=0 COMP_FT_GPU=1 JOINT_NONFT_GPU=2 JOINT_FT_GPU=3

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

STAGE="all"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

COMP_NONFT_DIR="${COMP_NONFT_DIR:-$ROOT/compositional_runs/csqa_nonft_unified95}"
COMP_FT_DIR="${COMP_FT_DIR:-$ROOT/compositional_runs/csqa_ft_unified95}"
COMP_NONFT_SRC_CAT="${COMP_NONFT_SRC_CAT:-/home/janerik/generalized_transformer-2/dr-llm/fine_routing_data_commonsenseqa_mcts_compositional_assign}"
COMP_FT_SRC_CAT="${COMP_FT_SRC_CAT:-/home/janerik/generalized_transformer-2/dr-llm/fine_routing_data_ft_qwen05b_250sims_continuous_commonsenseqa_compositional_assign}"

SPLITS_DIR="${SPLITS_DIR:-$ROOT/splits}"
SPLIT_NONFT="$SPLITS_DIR/csqa_nonft_canonical_split.json"
SPLIT_FT="$SPLITS_DIR/csqa_ft_canonical_split.json"
SEED="${SEED:-42}"
VAL_FRACTION="${VAL_FRACTION:-0.15}"
HOLDOUT="${HOLDOUT:-0}"

NEW_CAT_NONFT="${NEW_CAT_NONFT:-$COMP_NONFT_DIR/catalog_mass095_train_only}"
NEW_CAT_FT="${NEW_CAT_FT:-$COMP_FT_DIR/catalog_mass095_train_only}"

# ---------------------------------------------------------------------------
# Stage 1 — canonical splits
# ---------------------------------------------------------------------------
prepare_splits() {
  mkdir -p "$SPLITS_DIR"
  echo "[splits] non-FT -> $SPLIT_NONFT"
  python3 scripts/make_canonical_split.py \
    --observed "commonsenseqa=$COMP_NONFT_DIR/catalog_mass095/observed/commonsenseqa.jsonl" \
    --output "$SPLIT_NONFT" \
    --seed "$SEED" --val_fraction "$VAL_FRACTION" \
    --train_test_holdout_count "$HOLDOUT"
  echo "[splits] FT -> $SPLIT_FT"
  python3 scripts/make_canonical_split.py \
    --observed "commonsenseqa=$COMP_FT_DIR/catalog_mass095/observed/commonsenseqa.jsonl" \
    --output "$SPLIT_FT" \
    --seed "$SEED" --val_fraction "$VAL_FRACTION" \
    --train_test_holdout_count "$HOLDOUT"
}

# ---------------------------------------------------------------------------
# Stage 2 — rebuild mass-pruned catalogues using train-only mass
# ---------------------------------------------------------------------------
rebuild_catalogue() {
  local src="$1" out="$2" split="$3" bench="$4" dense="$5"
  echo "[catalogue] rebuild (train-only mass) src=$src out=$out split=$split"
  local dense_flag=()
  if [[ -n "$dense" && -f "$dense" ]]; then
    dense_flag=(--dense_deltas "${bench}=${dense}")
  fi
  python -m data_prep.build_joint_catalogue \
    --catalogue_dir "$src" \
    --output_dir "$out" \
    --benchmarks "$bench" \
    --mass_coverage 0.95 \
    --split_json "$split" \
    "${dense_flag[@]}"
}

prepare_catalogues() {
  rebuild_catalogue "$COMP_NONFT_SRC_CAT" "$NEW_CAT_NONFT" "$SPLIT_NONFT" commonsenseqa \
    "$COMP_NONFT_DIR/dense_deltas_matrix_legal_assign1384.pt"
  rebuild_catalogue "$COMP_FT_SRC_CAT" "$NEW_CAT_FT" "$SPLIT_FT" commonsenseqa \
    "$COMP_FT_DIR/dense_legal_commonsenseqa.pt"
}

# ---------------------------------------------------------------------------
# Stage 3 — launch the four sweeps
# ---------------------------------------------------------------------------
COMP_NONFT_GPU="${COMP_NONFT_GPU:-0}"
COMP_FT_GPU="${COMP_FT_GPU:-1}"
JOINT_NONFT_GPU="${JOINT_NONFT_GPU:-2}"
JOINT_FT_GPU="${JOINT_FT_GPU:-3}"

N_TRIALS="${N_TRIALS:-400}"
WB_PROJECT_COMP_NONFT="${WB_PROJECT_COMP_NONFT:-csqa-nonft-unified95-hpo-v2}"
WB_PROJECT_COMP_FT="${WB_PROJECT_COMP_FT:-csqa-ft-unified95-hpo-v2}"

launch_comp() {
  local tag="$1" out="$2" cat="$3" split="$4" gpu="$5" project="$6" adapter="$7"
  local logdir="$out/logs"
  mkdir -p "$logdir"
  local adapter_flag=()
  if [[ -n "$adapter" ]]; then
    adapter_flag=(--ft_adapter_path "$adapter")
  fi
  echo "[launch-comp] $tag GPU=$gpu wandb=$project out=$out"
  nohup python -m experiments.unified_hpo.run \
    --hpo_backend optuna \
    --router_kind compositional \
    --scope single \
    --benchmarks commonsenseqa \
    --catalogue_dir "$cat" \
    --split_json "$split" \
    --output_dir "$out" \
    --wandb_project "$project" \
    --n_trials "$N_TRIALS" \
    --seed "$SEED" \
    --gpu "$gpu" \
    --val_fraction "$VAL_FRACTION" \
    "${adapter_flag[@]}" \
    > "$logdir/launch.log" 2>&1 &
  disown
}

launch_joint() {
  local mode="$1" gpu="$2"
  local tag="sweep_csqa_${mode}_mass095_online"
  local split
  if [[ "$mode" == "ft" ]]; then split="$SPLIT_FT"; else split="$SPLIT_NONFT"; fi
  local joint_root="/home/janerik/generalized_transformer-2/dr-llm"
  local out="/home/janerik/sweep_csqa_${mode}_mass095_online_tpe_$(date +%Y%m%d)"
  echo "[launch-joint] $tag GPU=$gpu out=$out split=$split"
  (
    cd "$joint_root"
    FT_MODE="$mode" GPU="$gpu" OUT_ROOT="$out" SPLIT_JSON="$split" \
      nohup ./experiments/supervise_csqa_mass095_online_sweep.sh \
        > "$out/logs/${tag}.launch.log" 2>&1 &
    disown
  )
}

launch_sweeps() {
  launch_comp "comp-nonft" "$ROOT/hpo_results/csqa_nonft_unified95_optuna_v2" \
    "$NEW_CAT_NONFT" "$SPLIT_NONFT" "$COMP_NONFT_GPU" "$WB_PROJECT_COMP_NONFT" ""
  launch_comp "comp-ft" "$ROOT/hpo_results/csqa_ft_unified95_optuna_v2" \
    "$NEW_CAT_FT" "$SPLIT_FT" "$COMP_FT_GPU" "$WB_PROJECT_COMP_FT" \
    "${FT_ADAPTER:-$ROOT/ft_study_results_v7/commonsenseqa/seed_42/ft_only/checkpoints/final_adapter}"
  launch_joint nonft "$JOINT_NONFT_GPU"
  launch_joint ft "$JOINT_FT_GPU"
}

case "$STAGE" in
  prepare)
    prepare_splits
    prepare_catalogues
    ;;
  launch)
    launch_sweeps
    ;;
  all)
    prepare_splits
    prepare_catalogues
    launch_sweeps
    ;;
  *) echo "unknown --stage $STAGE (use prepare|launch|all)" >&2; exit 2 ;;
esac

echo "done (stage=$STAGE)"
