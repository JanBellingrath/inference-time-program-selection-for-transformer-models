#!/usr/bin/env python
"""
Re-run 3 benchmarks with FIXED tier nesting (disjunct samples).

The old benchmark_mcts had a bug where tier-2/3/4 samples were not disjunct,
causing double-dipping and unreliable results. This script re-runs with the
fixed version that ensures all tiers use disjunct sample sets.

Benchmarks to run:
- bigbench_boolean_expressions (boolean output subtask only)
- boolq
- mmlu_all

Models: Qwen 2.5 7B and 0.5B Instruct

Usage:
    python run_fixed_3benchmarks.py
    python run_fixed_3benchmarks.py --models 7b          # only 7B
    python run_fixed_3benchmarks.py --models 0.5b        # only 0.5B
    python run_fixed_3benchmarks.py --benchmarks boolq   # only boolq
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

# Benchmarks: original 3 + arc_easy, arc_challenge (for 0.5B fixed runs)
BENCHMARKS = [
    "bigbench_boolean_expressions",
    "boolq",
    "mmlu_all",
    "arc_easy",
    "arc_challenge",
]

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

OUTPUT_DIR_7B = "predictions/qwen25_7b_fixed"
OUTPUT_DIR_05B = "predictions/qwen25_0.5b_fixed"


def run_benchmark(model, model_name, dataset, output_dir, notify_signal):
    """Run MCTS search on a single benchmark with FIXED disjunct tier nesting."""
    logger.info("=" * 70)
    logger.info("STARTING: %s on %s", dataset, model_name)
    logger.info("=" * 70)
    t0 = time.time()

    # Load data with FIXED disjunct tier slicing
    is_instruct = get_is_instruct(model_name)
    all_data = prepare_arc_data(dataset, is_instruct, split=SPLIT)
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
    
    # DISJUNCT slices (this is the fix!)
    samples = all_data[:n_tier2]
    extended = all_data[n_tier2:n_tier2+n_tier3] if n_tier3 > 0 else []
    extended_tier4 = all_data[n_tier2+n_tier3:n_tier2+n_tier3+n_tier4] if n_tier4 > 0 else None
    
    # Verify disjunctness
    import hashlib
    def sample_hash(s):
        return hashlib.md5((s["input"] + str(s.get("correct", ""))).encode()).hexdigest()
    
    tier2_hashes = {sample_hash(s) for s in samples}
    tier3_hashes = {sample_hash(s) for s in extended} if extended else set()
    tier4_hashes = {sample_hash(s) for s in extended_tier4} if extended_tier4 else set()
    
    overlap_2_3 = tier2_hashes & tier3_hashes
    overlap_2_4 = tier2_hashes & tier4_hashes
    overlap_3_4 = tier3_hashes & tier4_hashes
    
    if overlap_2_3 or overlap_2_4 or overlap_3_4:
        raise ValueError(f"CRITICAL: Sample overlap detected! "
                        f"tier2∩tier3={len(overlap_2_3)}, "
                        f"tier2∩tier4={len(overlap_2_4)}, "
                        f"tier3∩tier4={len(overlap_3_4)}")
    
    logger.info(
        "✓ VERIFIED DISJUNCT: tier-2=%d [0:%d], tier-3=%d [%d:%d], tier-4=%d [%d:%d]",
        len(samples), n_tier2,
        len(extended), n_tier2, n_tier2+n_tier3,
        len(extended_tier4) if extended_tier4 else 0, n_tier2+n_tier3, n_tier2+n_tier3+n_tier4,
    )

    # Config
    cfg = PermutationMCTSConfig(
        num_simulations=NUM_SIMULATIONS,
        exploration_constant=EXPLORATION_CONSTANT,
        random_prob=RANDOM_PROB,
        neighborhood_radius=NEIGHBORHOOD_RADIUS,
        max_swaps=MAX_SWAPS,
        model_name=model_name,
        dataset=dataset,
        num_samples=NUM_SAMPLES,
    )

    # Output paths
    ts = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"benchmark_mcts_{dataset}_{ts}.json")
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
    summary["fixed_disjunct_tiers"] = True  # marker for fixed version

    # Print results
    bl = summary.get("baseline_accuracy_gen") or summary.get("baseline_accuracy")
    print("\n" + "=" * 60, flush=True)
    print(f"RESULTS -- {dataset} on {model_name}", flush=True)
    print("=" * 60, flush=True)
    print(f"Model:    {model_name}", flush=True)
    print(
        f"Dataset:  {dataset} (tier-2: {len(samples)}, tier-3: {len(extended)}, "
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
        description="Re-run 3 benchmarks with FIXED disjunct tier nesting"
    )
    p.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["7b", "0.5b"],
        choices=["7b", "0.5b"],
        help="Which models to run (default: both)",
    )
    p.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=None,
        help="Which benchmarks to run (default: all 3)",
    )
    args = p.parse_args()

    benchmarks_to_run = args.benchmarks if args.benchmarks else BENCHMARKS
    # Validate
    for b in benchmarks_to_run:
        if b not in BENCHMARKS:
            logger.error("Invalid benchmark: %s (must be one of %s)", b, BENCHMARKS)
            sys.exit(1)

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
            "output_dir": OUTPUT_DIR_7B,
        },
        "0.5b": {
            "name": "Qwen/Qwen2.5-0.5B-Instruct",
            "output_dir": OUTPUT_DIR_05B,
        },
    }

    # Load models once
    loaded_models = {}
    for model_id in args.models:
        cfg = models_config[model_id]
        model_name = cfg["name"]
        logger.info("Loading model: %s (GPU %d)", model_name, GPU_RANK)
        loaded_models[model_id] = MCTSModel(model_name, rank=GPU_RANK)
        logger.info(
            "Model ready: %d layers, device=%s",
            loaded_models[model_id].num_layers,
            loaded_models[model_id].wrapper.model.device,
        )

    all_results = {}
    sweep_t0 = time.time()

    total_runs = len(args.models) * len(benchmarks_to_run)
    run_idx = 0

    for model_id in args.models:
        cfg = models_config[model_id]
        model_name = cfg["name"]
        output_dir = cfg["output_dir"]
        model = loaded_models[model_id]

        for dataset in benchmarks_to_run:
            run_idx += 1
            logger.info(
                ">>> Run %d/%d: %s on %s", run_idx, total_runs, dataset, model_name
            )

            if notify_signal:
                send_signal(
                    f"FIXED RUN {run_idx}/{total_runs}\n"
                    f"{model_name}\n"
                    f"{dataset}\n"
                    f"Sims: {NUM_SIMULATIONS}"
                )

            try:
                summary = run_benchmark(
                    model, model_name, dataset, output_dir, notify_signal
                )
                bl = summary.get("baseline_accuracy_gen") or summary.get("baseline_accuracy")
                key = f"{model_name}_{dataset}"
                all_results[key] = {
                    "model": model_name,
                    "dataset": dataset,
                    "baseline": bl,
                    "best_delta": summary["best"]["delta"] if summary["best"] else 0.0,
                    "best_acc": (summary["best"].get("accuracy_gen") or summary["best"].get("accuracy")) if summary["best"] else None,
                    "confirmed_better": summary["confirmed_better"],
                    "significant": summary["statistically_significant"],
                    "elapsed_min": summary["elapsed_seconds"] / 60,
                }

                if notify_signal:
                    s = all_results[key]
                    send_signal(
                        f"✓ {model_name} {dataset} DONE ({run_idx}/{total_runs})\n"
                        f"Baseline: {s['baseline']:.4f}\n"
                        f"Best delta: {s['best_delta']:+.4f}\n"
                        f"Confirmed better: {s['confirmed_better']}\n"
                        f"Elapsed: {s['elapsed_min']:.1f} min"
                    )
            except Exception as e:
                logger.error("FAILED on %s %s: %s", model_name, dataset, e, exc_info=True)
                key = f"{model_name}_{dataset}"
                all_results[key] = {
                    "model": model_name,
                    "dataset": dataset,
                    "error": str(e)
                }
                if notify_signal:
                    send_signal(f"✗ {model_name} {dataset} FAILED: {e}")

    total_elapsed = time.time() - sweep_t0
    print("\n" + "#" * 70, flush=True)
    print("SWEEP COMPLETE -- FIXED 3 BENCHMARKS", flush=True)
    print("#" * 70, flush=True)
    print(f"Total time: {total_elapsed / 3600:.1f} hours", flush=True)
    print(
        f"{'Model':<35} {'Benchmark':<30} {'Baseline':>10} {'Best Δ':>10} {'Conf':>6} {'Sig':>6}",
        flush=True,
    )
    print("-" * 70, flush=True)
    for key in sorted(all_results.keys()):
        s = all_results[key]
        if "error" in s:
            print(f"{s['model']:<35} {s['dataset']:<30} {'ERROR':>10}", flush=True)
        else:
            print(
                f"{s['model']:<35} {s['dataset']:<30} "
                f"{s.get('baseline', 0):>10.4f} {s.get('best_delta', 0):>10.4f} "
                f"{s.get('confirmed_better', 0):>6} {s.get('significant', 0):>6}",
                flush=True,
            )
    print("#" * 70, flush=True)

    # Save summary
    summary_path = os.path.join(
        "predictions", f"fixed_3benchmarks_summary_{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    os.makedirs("predictions", exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {
                "description": "Re-run with FIXED disjunct tier nesting (no double-dipping)",
                "benchmarks": benchmarks_to_run,
                "models": [models_config[m]["name"] for m in args.models],
                "num_simulations": NUM_SIMULATIONS,
                "total_elapsed_hours": total_elapsed / 3600,
                "results": all_results,
            },
            f,
            indent=2,
        )
    logger.info("Summary saved to %s", summary_path)

    if notify_signal:
        n_success = sum(1 for s in all_results.values() if "error" not in s and s.get("confirmed_better", 0) > 0)
        send_signal(
            f"✓ FIXED 3 BENCHMARKS COMPLETE\n"
            f"Total: {total_elapsed / 3600:.1f}h\n"
            f"Success: {n_success}/{len(all_results)} had confirmed better seqs"
        )


if __name__ == "__main__":
    main()
