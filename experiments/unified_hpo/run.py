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
# Compositional-router data loading
# ---------------------------------------------------------------------------

def _parse_dense_deltas(entries: Optional[List[str]]) -> Dict[str, Path]:
    """Parse ``--dense_deltas bench=path`` repeated entries to a dict."""
    out: Dict[str, Path] = {}
    if not entries:
        return out
    for e in entries:
        if "=" not in e:
            raise SystemExit(f"--dense_deltas expects bench=path, got {e!r}")
        bench, path = e.split("=", 1)
        p = Path(path)
        if not p.is_file():
            raise SystemExit(f"dense delta file not found: {p}")
        out[bench] = p
    return out


def _load_data_compositional(args) -> Dict[str, Any]:
    """Load compositional artifacts (catalogue + residuals) once.

    Returns a dict with ``artifacts`` (a ``CompositionalArtifacts``),
    ``benchmarks``, ``dense_paths``, ``scope``, plus a ``d_model`` for
    logging.
    """
    from routers.compositional_router import CompositionalDataset, load_artifacts

    if not args.catalogue_dir:
        raise SystemExit("--catalogue_dir is required for --router_kind compositional")
    if not args.benchmarks:
        raise SystemExit("--benchmarks is required for --router_kind compositional")

    catalogue_dir = Path(args.catalogue_dir)
    artifacts = load_artifacts(catalogue_dir, benchmarks=list(args.benchmarks))
    if not artifacts.catalogues:
        raise SystemExit(
            f"no catalogues loaded for benchmarks={args.benchmarks!r} from {catalogue_dir}"
        )
    available = list(artifacts.catalogues.keys())
    benchmarks = [b for b in args.benchmarks if b in available]
    missing = [b for b in args.benchmarks if b not in available]
    if missing:
        logger.warning("compositional artifacts missing for: %s", missing)
    if not benchmarks:
        raise SystemExit("none of --benchmarks have compositional artifacts.")

    if args.scope == "single" and len(benchmarks) != 1:
        raise SystemExit(
            f"--scope single requires exactly one benchmark, got {benchmarks!r}"
        )

    dense_paths = _parse_dense_deltas(args.dense_deltas)
    if dense_paths:
        unknown = [b for b in dense_paths if b not in benchmarks]
        if unknown:
            raise SystemExit(
                f"--dense_deltas references benchmarks not in --benchmarks: {unknown!r}"
            )

    if args.objective_metric == "mean_uplift" and not dense_paths:
        raise SystemExit(
            "--objective_metric mean_uplift requires --dense_deltas bench=path "
            "for at least one of --benchmarks."
        )

    # Optional local-Möbius supervision files per benchmark. These are
    # mandatory when SMAC samples a trial with ``use_local_*`` enabled
    # (search_space_compositional exposes those knobs; without files the
    # trial silently falls back to no local supervision — see
    # ``compositional_objective.train_and_score_compositional``).
    local_moebius_paths: Dict[str, Path] = {}
    if getattr(args, "local_moebius_dir", None):
        ldir = Path(args.local_moebius_dir)
        if not ldir.is_dir():
            raise SystemExit(f"--local_moebius_dir not a directory: {ldir}")
        for bench in benchmarks:
            cand = ldir / f"{bench}.pt"
            if not cand.is_file():
                alt = ldir / f"local_moebius_{bench}.pt"
                if alt.is_file():
                    cand = alt
            if cand.is_file():
                local_moebius_paths[bench] = cand
            else:
                logger.warning(
                    "[%s] local Möbius target file missing in %s "
                    "(expected {bench}.pt or local_moebius_{bench}.pt); "
                    "SMAC trials sampling Möbius=on will disable it for "
                    "this benchmark.",
                    bench, ldir,
                )

    # Probe d_model from a tiny dataset slice (one benchmark).
    probe_bench = benchmarks[0]
    probe_dataset = CompositionalDataset(artifacts, benchmarks=[probe_bench])
    if not probe_dataset.encoder_inputs:
        raise SystemExit(
            f"compositional dataset for {probe_bench!r} is empty; cannot infer d_model."
        )
    d_model = int(probe_dataset.encoder_inputs[0].shape[-1])
    n_samples = len(probe_dataset)

    logger.info(
        "Compositional data loaded: scope=%s benchmarks=%s d_model=%d "
        "probe_samples(%s)=%d dense_paths=%s",
        args.scope, benchmarks, d_model, probe_bench, n_samples,
        {b: str(p) for b, p in dense_paths.items()},
    )

    return {
        "kind": "compositional",
        "artifacts": artifacts,
        "benchmarks": benchmarks,
        "dense_paths": dense_paths,
        "local_moebius_paths": local_moebius_paths or None,
        "scope": args.scope,
        "d_model": d_model,
        "use_full_sequence": bool(args.use_full_sequence),
        "objective_metric": args.objective_metric,
        "val_fraction": float(args.val_fraction),
        "batch_size": int(args.batch_size),
    }


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
        description="Unified SMAC + Hyperband (training epochs) for fine / compositional routers",
    )
    p.add_argument("--router_kind", choices=["fine", "compositional"], default="fine",
                   help="Which router family to optimize. Default 'fine' preserves the "
                        "existing behaviour; 'compositional' uses the legal-catalogue router.")
    # --- Fine-router data flags (required only for --router_kind fine) ---
    p.add_argument("--data_dir", type=str, default=None,
                   help="(fine) Dir with {bench}_pivot_residuals.pt and {bench}.jsonl")
    p.add_argument("--benchmark", type=str, default=None,
                   help="(fine) single benchmark to optimize")
    p.add_argument("--results_dir", type=str, default=None,
                   help="(fine) predictions dir for resolving anchor sequences")
    p.add_argument("--model_name", type=str,
                   default="Qwen/Qwen2.5-0.5B-Instruct")
    # --- Compositional-router data flags ---
    p.add_argument("--catalogue_dir", type=str, default=None,
                   help="(compositional) build_compositional_catalogues output dir")
    p.add_argument("--scope", choices=["single", "joint"], default="single",
                   help="(compositional) per-benchmark router or one shared router.")
    p.add_argument("--benchmarks", type=str, nargs="+", default=None,
                   help="(compositional) benchmarks to load from --catalogue_dir.")
    p.add_argument("--dense_deltas", type=str, nargs="+", default=None,
                   help="(compositional) bench=path entries with dense Δ matrices "
                        "(produced by data_prep.import_mined_dense_matrix or dr-llm).")
    p.add_argument("--local_moebius_dir", type=str, default=None,
                   help="(compositional) directory with local-Möbius target .pt "
                        "files (produced by data_prep.build_local_moebius_targets). "
                        "Required when the HPO search space turns on "
                        "use_local_unary/use_local_pair; without it such trials "
                        "silently disable local supervision and SMAC cannot "
                        "identify whether Möbius helps or hurts.")
    p.add_argument("--objective_metric",
                   choices=["mean_uplift", "obs_top1_acc"], default="mean_uplift",
                   help="(compositional) which val metric SMAC maximises.")
    p.add_argument("--use_full_sequence", action="store_true",
                   help="(compositional) load full-sequence residuals (not used by "
                        "the default last_token compressor).")
    p.add_argument("--val_fraction", type=float, default=0.15,
                   help="(compositional) train/val split fraction for the inner trainer.")
    p.add_argument("--batch_size", type=int, default=64,
                   help="(compositional) batch size for the inner trainer.")
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
    p.add_argument(
        "--resume",
        choices=["yes", "no"],
        default="no",
        help="yes: keep SMAC state under output_dir (overwrite=false); append to "
             "threshold_prior_archive.jsonl; set trial index from archive size; load "
             "best_routing_system_*/meta.json. Same n_trials/seed/budgets/configspace as "
             "the saved run. no: new SMAC run (wipes smac3_output for this scenario). "
             "W&B: starts a new run unless you set WANDB_RUN_ID+resume. Per-trial metrics "
             "are flat ``hpo/*`` with step = trial index (continues from archive when resuming).",
    )

    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    resume = args.resume == "yes"

    from experiments.unified_hpo.smac_runner import run_smac_optimization

    if args.router_kind == "fine":
        if not args.data_dir or not args.benchmark or not args.results_dir:
            raise SystemExit(
                "--router_kind fine requires --data_dir, --benchmark, --results_dir."
            )
        data = _load_data(args)
        expensive_eval_fn = _build_expensive_eval_fn(args, data)

        archive = run_smac_optimization(
            router_kind="fine",
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
            resume=resume,
        )

        if args.confirm_top_k > 0:
            archive_path = os.path.join(args.output_dir, "threshold_prior_archive.jsonl")
            _run_confirmation(
                args, data, archive_path,
                top_k=args.confirm_top_k,
                seeds=args.confirm_seeds,
            )
    else:
        data = _load_data_compositional(args)
        # Compositional path: use the joined benchmarks string as the
        # tracker/logging "benchmark" identity.
        bench_label = (
            data["benchmarks"][0] if data["scope"] == "single"
            else "+".join(data["benchmarks"])
        )
        archive = run_smac_optimization(
            router_kind="compositional",
            compositional_ctx=data,
            d_model=data["d_model"],
            device=device,
            benchmark=bench_label,
            output_dir=args.output_dir,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            n_trials=args.n_trials,
            min_budget=args.min_budget,
            max_budget=args.max_budget,
            enable_expensive_eval=False,
            expensive_eval_fn=None,
            seed=args.seed,
            walltime_limit=args.walltime_limit,
            resume=resume,
        )
        if args.confirm_top_k > 0:
            logger.warning(
                "--confirm_top_k is not implemented for --router_kind compositional; "
                "skipping confirmation phase."
            )

    logger.info("Done.")


if __name__ == "__main__":
    main()
