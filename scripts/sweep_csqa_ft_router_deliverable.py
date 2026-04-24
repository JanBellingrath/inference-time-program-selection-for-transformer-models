#!/usr/bin/env python3
"""CSQA FT compositional router sweep v2 — overfit-aware, cheap eval.

Fixes from the v1 sweep:
* Trial 1 / v1 got ``val mean_uplift = -0.083`` (worse than anchor) even
  though the best-fixed route gets ``+0.092``. Signal is there; the router
  simply overfit the per-question dense Δ noise after ~epoch 15.
* The v1 trials were also slow because full train+val downstream eval ran
  every epoch (per-epoch checkpoint eval on full val) and every 10 epochs
  for logging.

Changes in this sweep:
* Uses the refactored :class:`PairwiseScorer` (≈2× faster step, half the
  pair memory) that just landed in ``routers/compositional_router.py``.
* Uses the new ``--downstream_eval_subset`` flag (800 questions) so each
  per-epoch checkpoint eval runs in a few seconds, not a few minutes.
* Uses the new ``--early_stopping_patience`` flag (``8`` epochs) so we
  don't burn compute after the model has plateaued.
* Supervises only with dense Δ (``--use_dense_supervision``); **no local
  Möbius** this round, per request.
* Target: ``val downstream mean_uplift >= 0.03`` (3 percentage points
  uplift over anchor on the held-out split; logged to W&B as
  ``commonsenseqa/downstream/val/mean_uplift``).

Trial space — four diversified recipes chosen to combat overfit:
1. ``t1_unary_small`` — pairs off, small d, strong dropout+WD. Clean
   baseline; shows how much of the signal is unary.
2. ``t2_pairs_small_regularised`` — pairs on, aggressive ``pair_l2`` and
   dropout.
3. ``t3_unary_wider_highreg`` — unary-only, wider but very high dropout
   and WD, fast LR + cosine.
4. ``t4_pairs_tiny`` — very small model, pairs with small pair head and
   strong ``pair_l2`` and pair-topk filtering.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CAT = Path(
    "/home/janerik/generalized_transformer-2/dr-llm/"
    "fine_routing_data_ft_qwen05b_250sims_continuous_commonsenseqa_compositional_assign"
)
UNIFIED_DIR = Path(
    "/home/janerik/generalized_transformer-2/dr-llm/dense_artifacts_csqa_ft_2026/"
    "decode_compositional_unified"
)
DENSE_LEGAL = UNIFIED_DIR / "dense_deltas_matrix_legal.pt"

WANDB_PROJECT = "csqa-ft-router-v2-overfit-aware"
# Deliverable: at least 3 percentage points of uplift over anchor on val.
TARGET_MEAN_UPLIFT = 0.03
# Shared eval/regularisation knobs.
EVAL_SUBSET = 800
PATIENCE = 8
EPOCHS = 40


def _ensure_dense_legal() -> None:
    if DENSE_LEGAL.is_file():
        p = torch.load(DENSE_LEGAL, map_location="cpu", weights_only=False)
        if tuple(p["delta_matrix"].shape)[1] == 1597:
            return
    UNIFIED_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "materialize_dense_legal_programs.py"),
        "--compositional_dir",
        str(CAT),
        "--unified_matrix",
        str(UNIFIED_DIR / "dense_deltas_matrix.pt"),
        "--selected_catalog",
        str(UNIFIED_DIR / "selected_catalog.json"),
        "--output",
        str(DENSE_LEGAL),
        "--bench",
        "commonsenseqa",
    ]
    print(">>> materialize legal dense:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _base_args(name: str) -> list[str]:
    """Common flags every trial shares."""
    return [
        "--catalogue_dir",
        str(CAT),
        "--benchmarks",
        "commonsenseqa",
        "--scope",
        "single",
        "--compressor_type",
        "last_token",
        "--dense_deltas",
        f"commonsenseqa={DENSE_LEGAL}",
        "--use_dense_supervision",
        "--checkpoint_metric",
        "mean_uplift",
        "--downstream_eval_every",
        "3",
        "--downstream_eval_subset",
        str(EVAL_SUBSET),
        "--early_stopping_patience",
        str(PATIENCE),
        "--epochs",
        str(EPOCHS),
        "--batch_size",
        "64",
        "--wandb",
        "--wandb_project",
        WANDB_PROJECT,
        "--wandb_run_name",
        name,
        "--wandb_tags",
        "csqa",
        "ft",
        "unified_assign_dense",
        "v2_overfit_aware",
        "no_moebius",
    ]


def run_one(name: str, extra: list[str]) -> dict[str, float] | None:
    out = ROOT / "router_runs" / "csqa_ft_v2_20260424" / name
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "training.train_compositional_router",
        "--output_dir",
        str(out),
        *_base_args(name),
        *extra,
    ]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    ckpt = out / "compositional_router_best_commonsenseqa.pt"
    if not ckpt.is_file():
        print("missing checkpoint", ckpt, flush=True)
        return None
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    vd = payload.get("metrics", {}).get("val_downstream") or {}
    mu = float(vd.get("mean_uplift", float("nan")))
    ru = float(vd.get("router_acc", float("nan")))
    aa = float(vd.get("anchor_acc", float("nan")))
    bf = float(vd.get("best_fixed_acc", float("nan")))
    be = int(payload.get("best_epoch", -1))
    print(
        f"=== {name}  best_ep={be}  val mean_uplift={mu:+.5f}  router={ru:+.5f}  "
        f"anchor={aa:+.5f}  best_fixed={bf:+.5f}  vs_fixed={(ru - bf):+.5f}",
        flush=True,
    )
    return {"mean_uplift": mu, "router_acc": ru, "anchor_acc": aa,
            "best_fixed_acc": bf, "best_epoch": be}


def main() -> int:
    _ensure_dense_legal()
    print(
        f"W&B project : {WANDB_PROJECT}\n"
        f"Deliverable : val downstream mean_uplift >= {TARGET_MEAN_UPLIFT} "
        f"(+3 pp over anchor)\n"
        f"Eval subset : {EVAL_SUBSET} questions per split (seed-pinned)\n"
        f"Patience    : {PATIENCE} epochs  |  Max epochs : {EPOCHS}\n",
        flush=True,
    )

    # NOTE: all four trials use the new efficient pair scorer; trial 1 & 3
    # have pairs off which further isolates overfit sources.
    trials: list[tuple[str, list[str]]] = [
        (
            # Strong unary-only baseline. Small, heavily regularised.
            "t1_unary_small",
            [
                "--d_latent", "192",
                "--edit_hidden_dims", "192", "192",
                "--unary_hidden_dims", "192", "192",
                "--encoder_dropout", "0.25",
                "--edit_dropout", "0.25",
                "--unary_dropout", "0.25",
                "--weight_decay", "0.08",
                "--lr", "4e-4",
                "--tau", "1.2",
                "--student_temperature", "1.0",
                "--lam", "0.001",
                "--use_anchor_bias",
                "--seed", "42",
            ],
        ),
        (
            # Pairs on, aggressive regularisation on the pair head.
            "t2_pairs_small_regularised",
            [
                "--d_latent", "224",
                "--edit_hidden_dims", "224", "224",
                "--unary_hidden_dims", "224", "224",
                "--use_pairs",
                "--pair_hidden_dims", "160", "160",
                "--pair_dropout", "0.35",
                "--pair_l2", "5e-4",
                "--pair_topk_primitives", "12",
                "--encoder_dropout", "0.30",
                "--edit_dropout", "0.30",
                "--unary_dropout", "0.30",
                "--weight_decay", "0.10",
                "--lr", "3e-4",
                "--tau", "1.0",
                "--student_temperature", "1.0",
                "--lam", "0.0015",
                "--use_anchor_bias",
                "--seed", "43",
            ],
        ),
        (
            # Unary-only but wider, with very heavy dropout + WD.
            "t3_unary_wider_highreg",
            [
                "--d_latent", "288",
                "--edit_hidden_dims", "288", "288",
                "--unary_hidden_dims", "288", "288",
                "--encoder_dropout", "0.35",
                "--edit_dropout", "0.35",
                "--unary_dropout", "0.35",
                "--weight_decay", "0.15",
                "--lr", "3e-4",
                "--tau", "1.3",
                "--student_temperature", "1.0",
                "--lam", "0.0012",
                "--use_anchor_bias",
                "--seed", "44",
            ],
        ),
        (
            # Tiny model with pairs — smallest capacity, highest pair L2.
            "t4_pairs_tiny",
            [
                "--d_latent", "160",
                "--edit_hidden_dims", "160", "160",
                "--unary_hidden_dims", "160", "160",
                "--use_pairs",
                "--pair_hidden_dims", "128", "128",
                "--pair_dropout", "0.40",
                "--pair_l2", "1e-3",
                "--pair_topk_primitives", "10",
                "--encoder_dropout", "0.30",
                "--edit_dropout", "0.30",
                "--unary_dropout", "0.30",
                "--weight_decay", "0.12",
                "--lr", "5e-4",
                "--tau", "1.1",
                "--student_temperature", "1.0",
                "--lam", "0.001",
                "--use_anchor_bias",
                "--seed", "45",
            ],
        ),
    ]

    best_mu = float("-inf")
    best_name = ""
    hit_deliverable = False
    results: list[tuple[str, dict[str, float]]] = []
    for name, extra in trials:
        stats = run_one(name, extra)
        if stats is None:
            continue
        results.append((name, stats))
        mu = stats["mean_uplift"]
        if mu > best_mu:
            best_mu = mu
            best_name = name
        if mu >= TARGET_MEAN_UPLIFT:
            hit_deliverable = True
            print(
                f"\n*** Deliverable met: {name}  mean_uplift={mu:+.5f} "
                f">= {TARGET_MEAN_UPLIFT} ***\n",
                flush=True,
            )

    print("\n=== Sweep summary ===")
    for name, s in results:
        print(
            f"  {name:<32}  best_ep={s['best_epoch']:3d}  "
            f"val mean_uplift={s['mean_uplift']:+.5f}  "
            f"router={s['router_acc']:+.5f}  "
            f"anchor={s['anchor_acc']:+.5f}  "
            f"best_fixed={s['best_fixed_acc']:+.5f}",
            flush=True,
        )
    print(
        f"\nBest val mean_uplift={best_mu:+.5f} ({best_name}).\n"
        f"Deliverable (>={TARGET_MEAN_UPLIFT}): "
        f"{'YES' if hit_deliverable else 'NO — inspect W&B / iterate'}\n"
        f"W&B project URL pattern: https://wandb.ai/<entity>/{WANDB_PROJECT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
