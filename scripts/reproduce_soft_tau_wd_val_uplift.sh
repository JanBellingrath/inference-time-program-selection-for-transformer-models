#!/usr/bin/env bash
# Reproduce the ~9 pp standalone-eval uplift from compositional_runs/csqa_hpo_sweep/logs/soft_tau_wd_val.json
#
# Root cause (verified 2026-04): the historical "soft_tau_wd" training run used a train/val split
# with --seed 43 (val anchor mean matches seed-43 split on the dense file: ~0.5118). The reported
# JSON was produced by evaluate_compositional_router with --seed 42 (val anchor ~0.4781). Training
# and evaluation used different internal_val splits — the headline number is not a clean held-out
# metric for a single split.
#
# This script reproduces that *exact protocol* (train seed 43, eval seed 42). The headline ~9 pp
# from soft_tau_wd_val.json relied on that mismatch plus an older router (~115k params). Re-running
# with the current codebase (~186k params), the same protocol gave ~0.14 pp on seed-42 val (see
# compositional_runs/repro_soft_tau_wd_split_mismatch/val_seed42.json). For a *fair* metric, set
# FAIR_SPLIT=1 to train and evaluate with seed 42.
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DENSE="${DENSE_DELTAS_PT:-/home/janerik/generalized_transformer-2/dr-llm/dense_eval/csqa_compositional_179_train/dense_deltas_matrix.pt}"
SEED_TRAIN="${SEED_TRAIN:-43}"
if [ "${FAIR_SPLIT:-0}" = "1" ] || [ "${FAIR_SPLIT:-0}" = "true" ]; then
  SEED_TRAIN=42
  echo "FAIR_SPLIT: training and eval both use seed 42 (recommended for reporting)."
fi
OUT_DIR="${OUT_DIR:-compositional_runs/repro_soft_tau_wd_split_mismatch/seed${SEED_TRAIN}_train}"
if [ "${FAIR_SPLIT:-0}" = "1" ] || [ "${FAIR_SPLIT:-0}" = "true" ]; then
  : "${EVAL_JSON:=compositional_runs/repro_soft_tau_wd_split_mismatch/val_fair_seed42.json}"
else
  : "${EVAL_JSON:=compositional_runs/repro_soft_tau_wd_split_mismatch/val_seed42.json}"
fi
EPOCHS="${EPOCHS:-100}"

echo "=== Train (seed ${SEED_TRAIN}, tau 2.45, wd 0.14) — same hyperparameters as csqa tau245_wd14 / soft_tau_wd naming ==="
mkdir -p "$OUT_DIR"
python -m training.train_compositional_router \
  --catalogue_dir compositional_runs/csqa_compositional \
  --output_dir "$OUT_DIR" \
  --scope single --benchmarks commonsenseqa \
  --use_pairs \
  --tau 2.45 --student_temperature 1.0 \
  --weight_decay 0.14 \
  --epochs "$EPOCHS" --seed "$SEED_TRAIN" \
  --downstream_eval_every 0

CKPT="$OUT_DIR/compositional_router_best_commonsenseqa.pt"
echo "=== Eval on internal_val with seed 42 (same protocol as soft_tau_wd_val.json) ==="
python -m evaluation.evaluate_compositional_router \
  --checkpoint "$CKPT" \
  --catalogue_dir compositional_runs/csqa_compositional \
  --benchmark commonsenseqa \
  --dense_deltas commonsenseqa="$DENSE" \
  --split internal_val --seed 42 --val_fraction 0.15 \
  --output_json "$EVAL_JSON"

echo "Wrote $EVAL_JSON — check mean_uplift vs ~0.09045"
