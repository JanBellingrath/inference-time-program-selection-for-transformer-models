#!/usr/bin/env python3
"""CLI entrypoint for the unified SMAC HPO.

Loads routing data once, sets up SMAC with ``MultiFidelityFacade`` (Hyperband
**training** budgets: scaled router/gate epochs). Calibration always uses the
full routing-val split.  Optionally runs post-HPO confirmation on the top-K
configurations.

Usage::

    python -m experiments.unified_hpo.run \\
        --data_dir fine_routing_data_winogrande_mcts \\
        --benchmark winogrande \\
        --results_dir predictions \\
        --model_name Qwen/Qwen2.5-0.5B-Instruct \\
        --n_trials 100 \\
        --wandb_project unified-fine-routing-hpo \\
        --output_dir hpo_results/winogrande \\
        --seed 42

The LLM is NOT loaded at startup.  It is loaded lazily only when
``--enable_expensive_eval`` is set **and** the trial runs at the maximum
training budget.
"""

from __future__ import annotations

import sys
from pathlib import Path
# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading (mirrors sweep_fine_routing.py patterns)
# ---------------------------------------------------------------------------

def _load_data(args) -> Dict[str, Any]:
    """Load routing data once and return a dict with everything needed.

    Handles both MCTS and enumerated data formats, exactly mirroring the
    data-loading logic in ``sweep_fine_routing.main()``.
    """
    from experiments.sweep_fine_routing import (
        load_bench_data_enumerated,
        load_bench_data_mcts,
    )
    from routers.fine_routing_deviations import apply_deviation, enumerate_deviations
    from routers.fine_routing_config import FineRoutingConfig
    from training.train_benchmark_router import load_optimal_sequences_from_results

    cfg = FineRoutingConfig(model_name=args.model_name, results_dir=args.results_dir)
    data_config_path = os.path.join(args.data_dir, "config.json")
    if os.path.isfile(data_config_path):
        with open(data_config_path) as f:
            data_cfg = json.load(f)
        for key in ("max_local_edits", "swap_radius", "editable_start"):
            if key in data_cfg:
                setattr(cfg, key, data_cfg[key])
                logger.info("  %s=%s (from data config)", key, data_cfg[key])

    # Detect MCTS vs enumerated
    jsonl_path = os.path.join(args.data_dir, f"{args.benchmark}.jsonl")
    with open(jsonl_path) as f:
        first_rec = json.loads(f.readline())
    is_mcts = first_rec.get("search_mode") == "mcts"

    # Resolve anchor sequence
    anchor_seq = _resolve_anchor_seq(args, cfg, first_rec)
    logger.info("Anchor: %s", anchor_seq)

    if is_mcts:
        logger.info("Loading MCTS training data ...")
        (residuals, gate_labels, router_targets_base,
         sequence_catalog_full, mcts_seq_to_idx_full,
         sequence_catalog_reduced, mcts_seq_to_idx_reduced,
         mcts_records) = load_bench_data_mcts(
            args.data_dir, args.benchmark, anchor_seq,
        )
        sequence_catalog = sequence_catalog_full
        seq_to_idx = mcts_seq_to_idx_full
        num_classes = len(sequence_catalog)
        records = mcts_records
    else:
        logger.info("Loading enumerated training data ...")
        residuals, gate_labels, router_targets_base = load_bench_data_enumerated(
            args.data_dir, args.benchmark,
        )
        num_classes = router_targets_base[0].shape[0]
        deviations = enumerate_deviations(
            anchor_seq,
            editable_start=cfg.editable_start,
            num_layers=len(anchor_seq),
            swap_radius=cfg.swap_radius,
            max_edits=cfg.max_local_edits,
        )
        sequence_catalog = [apply_deviation(anchor_seq, d) for d in deviations]
        seq_to_idx = {tuple(s): i for i, s in enumerate(sequence_catalog)}
        records = _load_records(args.data_dir, args.benchmark)

    d_model = residuals.shape[1]

    best_deltas: List[float] = []
    if is_mcts and records:
        best_deltas = [float(r.get("best_delta", 0.0)) for r in records]
    else:
        best_deltas = [float(gl) for gl in gate_labels]

    logger.info(
        "Data loaded: %d samples, d_model=%d, num_classes=%d, gate+=%d",
        len(gate_labels), d_model, num_classes, sum(gate_labels),
    )

    return {
        "residuals": residuals,
        "gate_labels": gate_labels,
        "records": records,
        "seq_to_idx": seq_to_idx,
        "num_classes": num_classes,
        "best_deltas": best_deltas,
        "sequence_catalog": sequence_catalog,
        "d_model": d_model,
        "anchor_seq": anchor_seq,
        "is_mcts": is_mcts,
        "cfg": cfg,
    }


def _resolve_anchor_seq(args, cfg, first_rec) -> List[int]:
    """Resolve the anchor sequence from data or results."""
    # Try from JSONL data first
    seq = first_rec.get("anchor_sequence")
    if isinstance(seq, list) and seq:
        return [int(x) for x in seq]

    # Try from results dir
    try:
        from training.train_benchmark_router import load_optimal_sequences_from_results
        anchor_seqs = load_optimal_sequences_from_results(
            args.results_dir, [args.benchmark], model_name=args.model_name,
        )
        if args.benchmark in anchor_seqs:
            return [int(x) for x in anchor_seqs[args.benchmark]]
    except Exception as exc:
        logger.warning("Anchor lookup from results failed: %s", exc)

    # Fallback: identity
    jsonl_path = os.path.join(args.data_dir, f"{args.benchmark}.jsonl")
    with open(jsonl_path) as f:
        recs = [json.loads(line) for line in f]
    pt_path = os.path.join(args.data_dir, f"{args.benchmark}_pivot_residuals.pt")
    residuals = torch.load(pt_path, map_location="cpu", weights_only=True)
    # Guess num_layers from the first record's explored sequences
    for r in recs:
        for ex in r.get("explored", []):
            return list(range(len(ex["seq"])))
    # Last resort
    return list(range(24))


def _load_records(data_dir: str, benchmark: str) -> List[Dict]:
    """Load JSONL records for a benchmark."""
    jsonl_path = os.path.join(data_dir, f"{benchmark}.jsonl")
    with open(jsonl_path) as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# Expensive evaluation builder (lazy LLM loading)
# ---------------------------------------------------------------------------

def _build_expensive_eval_fn(
    args,
    data: Dict[str, Any],
) -> Optional[Callable]:
    """Build a callable for expensive LLM-based evaluation.

    The LLM is loaded lazily on first call, not at startup.
    """
    if not args.enable_expensive_eval:
        return None

    _wrapper_cache = {}

    def expensive_eval_fn(
        train_result,
        config: Dict,
        calibration,
    ) -> Dict[str, Any]:
        """Run LLM-based evaluation using the existing evaluate() function."""
        from experiments.sweep_fine_routing import evaluate
        from core.flexible_models import FlexibleModelWrapper, get_is_instruct
        from core.benchmark_mcts import seq_to_layers
        from core.permutation_mcts import prepare_arc_data

        # Lazy LLM load
        if "wrapper" not in _wrapper_cache:
            logger.info("Loading LLM for expensive evaluation: %s", args.model_name)
            _wrapper_cache["wrapper"] = FlexibleModelWrapper(args.model_name, rank=0)

        wrapper = _wrapper_cache["wrapper"]

        # Load validation samples
        if "val_samples" not in _wrapper_cache:
            is_instruct = get_is_instruct(args.model_name)
            val_samples = prepare_arc_data(
                args.benchmark, is_instruct=is_instruct, split="validation",
            )
            if args.eval_questions > 0:
                val_samples = val_samples[:args.eval_questions]
            _wrapper_cache["val_samples"] = val_samples

        val_samples = _wrapper_cache["val_samples"]

        gating_mode = config.get("gating_mode", "gate_network")
        cfg_obj = data["cfg"]

        metrics = evaluate(
            wrapper=wrapper,
            gate=train_result.gate,
            router=train_result.router,
            gamma=calibration.best_threshold if gating_mode == "gate_network" else 0.5,
            anchor_seq=data["anchor_seq"],
            sequence_catalog=data["sequence_catalog"],
            samples=val_samples,
            benchmark=args.benchmark,
            model_name=args.model_name,
            pivot_layer=cfg_obj.pivot_layer,
            gate_device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            gating_mode=gating_mode,
            confidence_threshold=calibration.best_threshold if gating_mode == "router_confidence" else 0.0,
            delta_gate=train_result.delta_gate,
            delta_margin=calibration.best_threshold if gating_mode == "delta_gate" else 0.0,
        )

        return metrics

    return expensive_eval_fn


# ---------------------------------------------------------------------------
# Post-HPO confirmation
# ---------------------------------------------------------------------------

def _run_confirmation(
    args,
    data: Dict[str, Any],
    archive_path: str,
    top_k: int = 5,
    seeds: List[int] = (42, 123, 456),
) -> None:
    """Retrain and evaluate the top-K configurations on multiple seeds."""
    from experiments.unified_hpo.threshold_prior import ThresholdPriorArchive
    from experiments.unified_hpo.trainer import train_and_summarize
    from experiments.unified_hpo.budgeted_evaluator import evaluate_configuration

    import wandb

    archive = ThresholdPriorArchive(archive_path)

    # Top-K by proxy gain: prefer max training-budget runs; include legacy archives (budget≈1)
    mb = float(getattr(args, "max_budget", 243.0))
    high_budget_records = [
        r for r in archive.records
        if r.proxy_gain > -0.5
        and (float(r.budget) >= mb * 0.9 or abs(float(r.budget) - 1.0) < 0.02)
    ]
    high_budget_records.sort(key=lambda r: r.proxy_gain, reverse=True)
    top_configs = []
    seen_hashes = set()
    for rec in high_budget_records:
        if rec.config_hash not in seen_hashes and len(top_configs) < top_k:
            top_configs.append(rec.config)
            seen_hashes.add(rec.config_hash)

    if not top_configs:
        logger.warning("No valid archive records found for confirmation.")
        return

    logger.info("Confirming top-%d configurations on seeds %s", len(top_configs), seeds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expensive_eval_fn = _build_expensive_eval_fn(args, data)

    results = []
    for ci, cfg in enumerate(top_configs):
        for s in seeds:
            logger.info("Confirmation: config %d/%d, seed %d", ci + 1, len(top_configs), s)
            train_result = train_and_summarize(
                config=cfg,
                residuals=data["residuals"],
                gate_labels=data["gate_labels"],
                records=data["records"],
                seq_to_idx=data["seq_to_idx"],
                num_classes=data["num_classes"],
                best_deltas=data["best_deltas"],
                d_model=data["d_model"],
                device=device,
                seed=s,
            )
            eval_result = evaluate_configuration(
                train_result=train_result,
                config=cfg,
                records=data["records"],
                sequence_catalog=data["sequence_catalog"],
                seq_to_idx=data["seq_to_idx"],
                enable_expensive_eval=args.enable_expensive_eval,
                expensive_eval_fn=expensive_eval_fn,
            )
            results.append({
                "config_idx": ci,
                "seed": s,
                "config": cfg,
                "proxy_gain": eval_result.proxy_gain,
                "oracle_proxy_gain": eval_result.oracle_proxy_gain,
                "best_rho": eval_result.calibration.best_rho,
                "expensive_metrics": eval_result.expensive_metrics,
            })

    # Save confirmation results
    confirm_path = os.path.join(args.output_dir, "confirmation_results.json")
    with open(confirm_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Confirmation results saved to %s", confirm_path)

    # Log to W&B
    if not args.no_wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=f"confirmation-{args.benchmark}",
            tags=[args.benchmark, "confirmation"],
        )
        columns = ["config_idx", "seed", "proxy_gain", "oracle_proxy_gain", "best_rho"]
        if any(r["expensive_metrics"] for r in results):
            columns += ["unconditional_gain", "anchor_accuracy", "routed_accuracy"]
        table_data = []
        for r in results:
            row = [r["config_idx"], r["seed"], r["proxy_gain"],
                   r["oracle_proxy_gain"], r["best_rho"]]
            if "unconditional_gain" in columns:
                em = r.get("expensive_metrics") or {}
                row += [em.get("unconditional_gain", 0),
                        em.get("anchor_accuracy", 0),
                        em.get("routed_accuracy", 0)]
            table_data.append(row)
        table = wandb.Table(columns=columns, data=table_data)
        wandb.log({"confirmation_table": table})
        run.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Unified SMAC + Hyperband (training epochs) for fine routers",
    )
    p.add_argument("--data_dir", type=str, required=True,
                   help="Dir with {bench}_pivot_residuals.pt and {bench}.jsonl")
    p.add_argument("--benchmark", type=str, required=True)
    p.add_argument("--results_dir", type=str, required=True,
                   help="Predictions dir for resolving anchor sequences")
    p.add_argument("--model_name", type=str,
                   default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--output_dir", type=str, default="hpo_results",
                   help="Output directory for SMAC state, archive, and best models")
    p.add_argument("--wandb_project", type=str, default="unified-fine-routing-hpo")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--n_trials", type=int, default=100,
                   help="Maximum number of SMAC trials")
    p.add_argument(
        "--min_budget", type=float, default=9.0,
        help="Minimum Hyperband training budget (maps to fewer router/gate epochs)",
    )
    p.add_argument(
        "--max_budget", type=float, default=243.0,
        help="Maximum Hyperband training budget (full epoch counts)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--walltime_limit", type=float, default=float("inf"),
                   help="Maximum wall time in seconds for the optimization")

    # Expensive evaluation
    p.add_argument("--enable_expensive_eval", action="store_true",
                   help="Run LLM eval only on trials at --max_budget (costly)")
    p.add_argument("--eval_questions", type=int, default=200,
                   help="Number of validation questions for expensive eval")

    # Post-HPO confirmation
    p.add_argument("--confirm_top_k", type=int, default=0,
                   help="After HPO, retrain and evaluate the top-K configs on multiple seeds")
    p.add_argument("--confirm_seeds", type=int, nargs="+", default=[42, 123, 456],
                   help="Seeds to use for confirmation runs")

    p.add_argument("--no_wandb", action="store_true",
                   help="Disable W&B logging entirely")

    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    # --- Load data ---
    data = _load_data(args)

    # --- Build expensive eval function (lazy) ---
    expensive_eval_fn = _build_expensive_eval_fn(args, data)

    # --- Run optimization ---
    from experiments.unified_hpo.smac_runner import run_smac_optimization

    archive = run_smac_optimization(
        residuals=data["residuals"],
        gate_labels=data["gate_labels"],
        records=data["records"],
        seq_to_idx=data["seq_to_idx"],
        num_classes=data["num_classes"],
        best_deltas=data["best_deltas"],
        sequence_catalog=data["sequence_catalog"],
        d_model=data["d_model"],
        device=device,
        benchmark=args.benchmark,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        n_trials=args.n_trials,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
        enable_expensive_eval=args.enable_expensive_eval,
        expensive_eval_fn=expensive_eval_fn,
        seed=args.seed,
        walltime_limit=args.walltime_limit,
    )

    # --- Post-HPO confirmation ---
    if args.confirm_top_k > 0:
        archive_path = os.path.join(args.output_dir, "threshold_prior_archive.jsonl")
        _run_confirmation(
            args, data, archive_path,
            top_k=args.confirm_top_k,
            seeds=args.confirm_seeds,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
