#!/usr/bin/env bash
# Train + eval: replica of soft_tau_wd (dot unary + pair) on csqa_compositional,
# with W&B live + post-hoc val metrics to the same run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export WANDB_MODE="${WANDB_MODE:-online}"

CATALOGUE="${CATALOGUE_DIR:-compositional_runs/csqa_compositional}"
# Aligned 9287 x 179 (matches manifest) for dense downstream / fair eval
DENSE="${DENSE_DELTAS_PT:-compositional_runs/cg_exp/dense/commonsenseqa_dense.pt}"
OUT_DIR="${OUTPUT_DIR:-${ROOT}/compositional_runs/dot_soft_tau_replica_$(date +%Y%m%d_%H%M%S)}"
SEED_TRAIN="${SEED_TRAIN:-43}"
# Match internal_val split to training (same seed) unless overridden for repro protocol
SEED_EVAL="${SEED_EVAL:-${SEED_TRAIN}}"

mkdir -p "$OUT_DIR"

echo "OUT_DIR=$OUT_DIR"
echo "CATALOGUE=$CATALOGUE"
echo "DENSE=$DENSE"

python -m training.train_compositional_router \
  --catalogue_dir "$CATALOGUE" \
  --output_dir "$OUT_DIR" \
  --scope single --benchmarks commonsenseqa \
  --compressor_type last_token \
  --compressor_d_compress 256 --compressor_n_heads 4 --compressor_n_latent 1 \
  --d_latent 96 \
  --encoder_dropout 0.25 \
  --use_id_embedding \
  --unary_scorer_type dot --primitive_bias \
  --use_pairs \
  --pair_hidden_dims 48 48 --pair_dropout 0.3 --pair_l2 0.005 \
  --lam 0.0 --tau 2.2 --student_temperature 1.0 \
  --lr 0.001 --weight_decay 0.14 \
  --epochs 100 --batch_size 64 --val_fraction 0.15 --seed "$SEED_TRAIN" \
  --dense_deltas "commonsenseqa=$DENSE" \
  --downstream_eval_every 1 \
  --checkpoint_metric loss \
  --wandb --wandb_project "${WANDB_PROJECT:-compositional-router}" \
  --wandb_run_name "${WANDB_RUN_NAME:-csqa_soft_tau_wd_dot_replica_$(date +%Y%m%d_%H%M%S)}" \
  --wandb_tags soft_tau_wd dot_unary pair_head csqa_compositional

CKPT="$OUT_DIR/compositional_router_best_commonsenseqa.pt"
EVAL_JSON="$OUT_DIR/eval_internal_val.json"

python -m evaluation.evaluate_compositional_router \
  --checkpoint "$CKPT" \
  --catalogue_dir "$CATALOGUE" \
  --benchmark commonsenseqa \
  --dense_deltas "commonsenseqa=$DENSE" \
  --split internal_val --seed "$SEED_EVAL" --val_fraction 0.15 \
  --output_json "$EVAL_JSON"

export RUN_OUT_DIR="$OUT_DIR"
export RUN_EVAL_JSON="$EVAL_JSON"
python <<'PY'
import json
import os
import pathlib
import wandb

out = pathlib.Path(os.environ["RUN_OUT_DIR"])
wj = out / "wandb_run_info.json"
metrics = json.loads(pathlib.Path(os.environ["RUN_EVAL_JSON"]).read_text())
if not wj.is_file():
    print("No wandb_run_info.json; skip eval W&B (train may have disabled wandb).")
    print("First metric keys:", list(metrics)[:8])
    raise SystemExit(0)
info = json.loads(wj.read_text())
run = wandb.init(
    id=info["id"],
    project=info.get("project") or "compositional-router",
    entity=info.get("entity"),
    resume="allow",
)
flat = {
    f"eval_internal_val/{k}": v
    for k, v in metrics.items()
    if k not in ("meta",) and not isinstance(v, (dict, list))
}
if "by_pred_length" in metrics:
    for ell, d in (metrics.get("by_pred_length") or {}).items():
        for kk, vv in d.items():
            flat[f"eval_internal_val/by_pred_length_{ell}/{kk}"] = float(vv)
run.log(flat)
run.finish()
print("Logged", len(flat), "eval metrics to the same W&B run.")
PY

echo "Done. Checkpoint: $CKPT  Eval: $EVAL_JSON"
