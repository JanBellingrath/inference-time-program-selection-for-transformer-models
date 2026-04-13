#!/usr/bin/env python
"""
Run ONLY bigbench_boolean_expressions for both Qwen 2.5 7B and 0.5B models.

This script ONLY runs the boolean expression subtask of BigBench.
It does NOT run winogrande, commonsenseqa, or any other benchmarks.

Usage:
    python run_bigbench_boolean_only.py
    python run_bigbench_boolean_only.py --models 7b     # only 7B
    python run_bigbench_boolean_only.py --models 0.5b   # only 0.5B
"""
import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import argparse
import json
import logging
import os
import random
import sys
import time

from core.permutation_mcts import (
    MCTSModel,
    PermutationMCTSConfig,
    prepare_arc_data,
    set_seed,
)
from core.benchmark_mcts import BenchmarkMCTS, send_signal, _get_signal_cli_path
from core.flexible_models import get_is_instruct

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# -- Configuration -----------------------------------------------------------

SEED = 42
GPU_RANK = 0

# ONLY bigbench_boolean_expressions
BENCHMARK = "bigbench_boolean_expressions"

NUM_SIMULATIONS = 500
EVAL_BATCH_SIZE = 20
NUM_SAMPLES = 100           # tier-2
EXTENDED_SAMPLES = 500      # tier-3
EXTENDED_SAMPLES_TIER4 = 1000  # tier-4
NEIGHBORHOOD_RADIUS = 2
MAX_SWAPS = 4
EXPLORATION_CONSTANT = 1.8
RANDOM_PROB = 0.1
REPORT_EVERY = 50
VALIDATE_TOP_K = 3
PROMOTE_DELTA = 0.0
SPLIT = "train"


def run_benchmark(model_name, output_dir, notify_signal):
    """Run MCTS search on bigbench_boolean_expressions for the given model."""
    logger.info("=" * 70)
    logger.info("STARTING: %s on %s", BENCHMARK, model_name)
    logger.info("=" * 70)
    t0 = time.time()

    # Load model
    logger.info("Loading model: %s (GPU %d)", model_name, GPU_RANK)
    model = MCTSModel(model_name, rank=GPU_RANK)
    logger.info(
        "Model ready: %d layers, device=%s",
        model.num_layers,
        model.wrapper.model.device,
    )

    # Load data
    is_instruct = get_is_instruct(model_name)
    all_data = prepare_arc_data(BENCHMARK, is_instruct, split=SPLIT)
    random.shuffle(all_data)

    n_tier2 = NUM_SAMPLES
    n_tier3 = max(0, EXTENDED_SAMPLES - NUM_SAMPLES) if EXTENDED_SAMPLES > NUM_SAMPLES else 0
    n_tier4 = max(0, EXTENDED_SAMPLES_TIER4 - EXTENDED_SAMPLES) if EXTENDED_SAMPLES_TIER4 > EXTENDED_SAMPLES else 0
    n_need = n_tier2 + n_tier3 + n_tier4
    
    if len(all_data) < n_need:
        logger.warning(f"Not enough data: need {n_need}, have {len(all_data)}. Reducing tier sizes.")
        scale = len(all_data) / n_need
        n_tier2 = max(10, int(n_tier2 * scale))
        n_tier3 = int(n_tier3 * scale)
        n_tier4 = int(n_tier4 * scale)
    
    # DISJUNCT slices
    samples = all_data[:n_tier2]
    extended = all_data[n_tier2:n_tier2+n_tier3] if n_tier3 > 0 else []
    extended_tier4 = all_data[n_tier2+n_tier3:n_tier2+n_tier3+n_tier4] if n_tier4 > 0 else None
    
    logger.info(
        "Data: %d tier-2, %d tier-3, %d tier-4 from %s (%s)",
        len(samples),
        len(extended),
        len(extended_tier4) if extended_tier4 else 0,
        BENCHMARK,
        SPLIT,
    )

    # Config
    cfg = PermutationMCTSConfig(
        num_simulations=NUM_SIMULATIONS,
        exploration_constant=EXPLORATION_CONSTANT,
        random_prob=RANDOM_PROB,
        neighborhood_radius=NEIGHBORHOOD_RADIUS,
        max_swaps=MAX_SWAPS,
        model_name=model_name,
        dataset=BENCHMARK,
        num_samples=NUM_SAMPLES,
    )

    # Output paths
    ts = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"benchmark_mcts_{BENCHMARK}_{ts}.json")
    out_prefix = out_path.rsplit(".", 1)[0]

    # Run MCTS
    bench = BenchmarkMCTS(
        model,
        cfg,
        samples,
        eval_batch_size=EVAL_BATCH_SIZE,
        extended_samples=extended,
        extended_samples_tier4=extended_tier4,
        promote_delta=PROMOTE_DELTA,
        notify_signal=notify_signal,
    )
    summary = bench.search(
        NUM_SIMULATIONS,
        report_every=REPORT_EVERY,
        validate_top_k=VALIDATE_TOP_K,
        out_prefix=out_prefix,
    )

    elapsed = time.time() - t0
    summary["elapsed_seconds"] = elapsed

    # Print results
    bl = summary.get("baseline_accuracy_gen") or summary.get("baseline_accuracy")
    print("\n" + "=" * 60, flush=True)
    print(f"RESULTS -- {BENCHMARK} on {model_name}", flush=True)
    print("=" * 60, flush=True)
    print(f"Model:    {model_name}", flush=True)
    print(
        f"Dataset:  {BENCHMARK} (tier-2: {len(samples)}, tier-3: {len(extended)}, "
        f"tier-4: {len(extended_tier4) if extended_tier4 else 0})",
        flush=True,
    )
    print(
        f"Baseline (tier-2): {bl:.4f} "
        f"CI [{summary['baseline_ci'][0]:.4f}, {summary['baseline_ci'][1]:.4f}]",
        flush=True,
    )
    bl_ext = summary.get("baseline_ext_accuracy_gen") or summary.get("baseline_ext_accuracy")
    if bl_ext is not None and summary.get("baseline_ext_ci"):
        print(
            f"Baseline (tier-3): {bl_ext:.4f} "
            f"CI [{summary['baseline_ext_ci'][0]:.4f}, {summary['baseline_ext_ci'][1]:.4f}]",
            flush=True,
        )
    print(
        f"Explored: {summary['unique_sequences_explored']} unique sequences",
        flush=True,
    )
    print(
        f"Tier-2: {summary['tier2_validated']} | "
        f"Tier-3: {summary['tier3_validated']} | "
        f"Tier-4: {summary.get('tier4_validated', 0)} | "
        f"Confirmed: {summary['confirmed_better']} | "
        f"Significant: {summary['statistically_significant']}",
        flush=True,
    )
    if summary["best"]:
        b = summary["best"]
        b_acc = b.get("accuracy_gen") or b.get("accuracy")
        print(
            f"  Best: acc={b_acc:.4f} (delta +{b['delta']:.4f}) "
            f"CI [{b['ci_lo']:.4f}, {b['ci_hi']:.4f}]",
            flush=True,
        )
        print(f"  Layers ({b['length']}): {b['layers']}", flush=True)
        print(f"  Skips: {b['num_skips']}, Swaps: {b['num_swaps']}", flush=True)
    else:
        print(
            "  No sequence confirmed beating baseline on rigorous eval.", flush=True
        )
    print(f"Time: {elapsed / 60:.1f} min", flush=True)
    print("=" * 60, flush=True)

    # Save
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved results to %s", out_path)

    return summary


def main():
    p = argparse.ArgumentParser(
        description="Run ONLY bigbench_boolean_expressions for Qwen models"
    )
    p.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["7b", "0.5b"],
        choices=["7b", "0.5b"],
        help="Which models to run (default: both)",
    )
    args = p.parse_args()

    set_seed(SEED)

    notify_signal = bool(_get_signal_cli_path())
    if not notify_signal:
        print(
            "WARNING: signal-cli not found. Signal notifications disabled.",
            file=sys.stderr,
            flush=True,
        )

    models_config = {
        "7b": {
            "name": "Qwen/Qwen2.5-7B-Instruct",
            "output_dir": "predictions/qwen25_7b",
        },
        "0.5b": {
            "name": "Qwen/Qwen2.5-0.5B-Instruct",
            "output_dir": "predictions/qwen25_0.5b",
        },
    }

    all_summaries = {}
    sweep_t0 = time.time()

    for model_id in args.models:
        cfg = models_config[model_id]
        model_name = cfg["name"]
        output_dir = cfg["output_dir"]

        logger.info(">>> Starting %s", model_name)
        
        if notify_signal:
            send_signal(
                f"{model_name} {BENCHMARK} started.\n"
                f"Sims: {NUM_SIMULATIONS}\n"
            )

        try:
            summary = run_benchmark(model_name, output_dir, notify_signal)
            bl = summary.get("baseline_accuracy_gen") or summary.get("baseline_accuracy")
            all_summaries[model_name] = {
                "baseline": bl,
                "best_delta": summary["best"]["delta"] if summary["best"] else 0.0,
                "best_acc": (summary["best"].get("accuracy_gen") or summary["best"].get("accuracy")) if summary["best"] else None,
                "confirmed_better": summary["confirmed_better"],
                "significant": summary["statistically_significant"],
                "elapsed_min": summary["elapsed_seconds"] / 60,
            }
            
            if notify_signal:
                s = all_summaries[model_name]
                send_signal(
                    f"{model_name} {BENCHMARK} DONE\n"
                    f"Baseline: {s['baseline']:.4f}\n"
                    f"Best delta: {s['best_delta']:+.4f}\n"
                    f"Confirmed better: {s['confirmed_better']}\n"
                    f"Elapsed: {s['elapsed_min']:.1f} min"
                )
        except Exception as e:
            logger.error("FAILED on %s: %s", model_name, e, exc_info=True)
            all_summaries[model_name] = {"error": str(e)}
            if notify_signal:
                send_signal(f"{model_name} {BENCHMARK} FAILED: {e}")

    total_elapsed = time.time() - sweep_t0
    print("\n" + "#" * 70, flush=True)
    print(f"COMPLETE -- {BENCHMARK}", flush=True)
    print("#" * 70, flush=True)
    print(f"Total time: {total_elapsed / 3600:.1f} hours", flush=True)
    print(
        f"{'Model':<30} {'Baseline':>10} {'Best D':>10} {'Confirmed':>10} {'Signif':>10}",
        flush=True,
    )
    print("-" * 70, flush=True)
    for model_name, s in all_summaries.items():
        if "error" in s:
            print(f"{model_name:<30} {'ERROR':>10}", flush=True)
        else:
            print(
                f"{model_name:<30} {s.get('baseline', 0):.4f}     "
                f"{s.get('best_delta', 0):+.4f}     "
                f"{s.get('confirmed_better', 0):>5}      "
                f"{s.get('significant', 0):>5}",
                flush=True,
            )
    print("#" * 70, flush=True)

    # Save summary
    summary_path = os.path.join(
        "predictions", f"bigbench_boolean_summary_{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    with open(summary_path, "w") as f:
        json.dump(
            {
                "benchmark": BENCHMARK,
                "num_simulations": NUM_SIMULATIONS,
                "total_elapsed_hours": total_elapsed / 3600,
                "results": all_summaries,
            },
            f,
            indent=2,
        )
    logger.info("Summary saved to %s", summary_path)

    if notify_signal:
        send_signal(
            f"{BENCHMARK} sweep COMPLETE.\n"
            f"Total: {total_elapsed / 3600:.1f}h\n"
            f"Models: {', '.join(args.models)}"
        )


if __name__ == "__main__":
    main()
