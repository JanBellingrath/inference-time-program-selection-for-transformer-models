#!/usr/bin/env python
"""
Benchmark-level MCTS sweep for Qwen 2.5 0.5B Instruct.

Loads the model once, then runs MCTS simulations on each benchmark
sequentially. Neighborhood radius = 5 (each position can reach ±5 layers).

5 benchmarks: winogrande, arc_challenge, boolq, commonsenseqa, mmlu_all
Default: 10 000 simulations per benchmark.

Usage:
    # Run all 5 benchmarks from scratch to 10k sims
    python run_qwen25_0.5b_benchmark_sweep.py

    # Resume all incomplete runs
    python run_qwen25_0.5b_benchmark_sweep.py --resume_incomplete
    
    # Run specific benchmarks
    python run_qwen25_0.5b_benchmark_sweep.py --benchmarks boolq commonsenseqa
    
    CUDA_VISIBLE_DEVICES=1 nohup python run_qwen25_0.5b_benchmark_sweep.py > nohup_qwen25_0.5b_r5_sweep.log 2>&1 &
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
from pathlib import Path
from typing import Optional, Tuple

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

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 42
# Use rank 0 when launched with CUDA_VISIBLE_DEVICES=1 (physical GPU 1)
GPU_RANK = 0

# 5 MC benchmarks
BENCHMARKS = [
    "winogrande",
    "arc_challenge",
    "boolq",
    "commonsenseqa",
    "mmlu_all",
]

DEFAULT_EXCLUDE = []

EVAL_BATCH_SIZE = 20
NUM_SAMPLES = 100           # tier-2
EXTENDED_SAMPLES = 500      # tier-3
EXTENDED_SAMPLES_TIER4 = 1000  # tier-4
NEIGHBORHOOD_RADIUS = 5
MAX_SWAPS = 24
EXPLORATION_CONSTANT = 1.8
RANDOM_PROB = 0.1
PW_C = 1.0
PW_ALPHA = 0.5
LEGACY_WIDEN_PROB = 0.0
LEGACY_RANDOM_SCHEDULE = False
REPORT_EVERY = 150  # was 50; less frequent validation speeds up sims significantly
VALIDATE_TOP_K = 3
PROMOTE_DELTA = 0.0
SPLIT = "train"

OUTPUT_DIR = "predictions/qwen25_0.5b_v2_sdpa_r5_pw"


def _baseline_from_summary(summary):
    """Baseline accuracy (handles both _gen and legacy keys)."""
    return summary.get("baseline_accuracy") or summary.get("baseline_accuracy_gen")


def _best_acc_from_summary(summary):
    """Best accuracy from summary (handles both _gen and legacy keys)."""
    b = summary.get("best")
    if not b:
        return None
    return b.get("accuracy") or b.get("accuracy_gen")


def find_latest_snapshot(output_dir: str, dataset: str) -> Optional[Tuple[str, int]]:
    """Find the most recent snapshot for a dataset and return (path, sim_count).
    
    Returns None if no snapshot found.
    """
    pattern = f"benchmark_mcts_{dataset}_*_snapshot.json"
    snapshots = sorted(Path(output_dir).glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not snapshots:
        return None
    
    snap_path = snapshots[0]
    try:
        with open(snap_path) as f:
            data = json.load(f)
        sim_count = data.get("sim", 0)
        return (str(snap_path), sim_count)
    except Exception as e:
        logger.warning("Failed to read snapshot %s: %s", snap_path, e)
        return None


def run_benchmark(model, dataset, num_simulations, notify_signal, resume_snapshot=None):
    """Run MCTS search on a single benchmark, return summary dict."""
    logger.info("=" * 70)
    logger.info("STARTING BENCHMARK: %s (target: %d sims)", dataset, num_simulations)
    logger.info("=" * 70)
    t0 = time.time()

    is_instruct = get_is_instruct(MODEL_NAME)
    all_data = prepare_arc_data(dataset, is_instruct, split=SPLIT)
    
    # Use deterministic per-benchmark shuffle seed for reproducibility
    benchmark_idx = BENCHMARKS.index(dataset) if dataset in BENCHMARKS else 0
    shuffle_seed = SEED + benchmark_idx
    shuffle_rng = random.Random(shuffle_seed)
    shuffle_rng.shuffle(all_data)
    logger.info("Data shuffled with seed %d (SEED + benchmark_idx)", shuffle_seed)

    n_need = max(NUM_SAMPLES, EXTENDED_SAMPLES, EXTENDED_SAMPLES_TIER4)
    pool = all_data[: min(n_need, len(all_data))]
    samples = pool[:NUM_SAMPLES]
    extended = pool[:EXTENDED_SAMPLES] if EXTENDED_SAMPLES > NUM_SAMPLES else samples
    extended_tier4 = (
        pool[:EXTENDED_SAMPLES_TIER4]
        if EXTENDED_SAMPLES_TIER4 > len(extended)
        else None
    )

    logger.info(
        "Data: %d tier-2, %d tier-3, %d tier-4 from %s (%s)",
        len(samples),
        len(extended),
        len(extended_tier4) if extended_tier4 else 0,
        dataset,
        SPLIT,
    )

    cfg = PermutationMCTSConfig(
        num_simulations=num_simulations,
        exploration_constant=EXPLORATION_CONSTANT,
        random_prob=RANDOM_PROB,
        pw_C=PW_C,
        pw_alpha=PW_ALPHA,
        legacy_widen_prob=LEGACY_WIDEN_PROB,
        legacy_random_schedule=LEGACY_RANDOM_SCHEDULE,
        neighborhood_radius=NEIGHBORHOOD_RADIUS,
        max_swaps=MAX_SWAPS,
        model_name=MODEL_NAME,
        dataset=dataset,
        num_samples=NUM_SAMPLES,
    )

    if resume_snapshot:
        if not resume_snapshot.endswith("_snapshot.json"):
            raise ValueError("--resume_snapshot must point to a *_snapshot.json file")
        out_prefix = resume_snapshot[:- len("_snapshot.json")]
        out_path = out_prefix + ".json"
    else:
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join(OUTPUT_DIR, f"benchmark_mcts_{dataset}_{ts}.json")
        out_prefix = out_path.rsplit(".", 1)[0]

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
        num_simulations,
        report_every=REPORT_EVERY,
        validate_top_k=VALIDATE_TOP_K,
        out_prefix=out_prefix,
        resume_prefix=out_prefix if resume_snapshot else None,
    )

    elapsed = time.time() - t0
    summary["elapsed_seconds"] = elapsed

    baseline_acc = _baseline_from_summary(summary)
    best_acc = _best_acc_from_summary(summary)

    print("\n" + "=" * 60, flush=True)
    print(f"BENCHMARK MCTS RESULTS -- {dataset}", flush=True)
    print("=" * 60, flush=True)
    print(f"Model:    {MODEL_NAME}", flush=True)
    print(
        f"Dataset:  {dataset} (tier-2: {len(samples)}, tier-3: {len(extended)}, "
        f"tier-4: {len(extended_tier4) if extended_tier4 else 0})",
        flush=True,
    )
    print(
        f"Baseline (tier-2): {baseline_acc:.4f} "
        f"CI [{summary['baseline_ci'][0]:.4f}, {summary['baseline_ci'][1]:.4f}]",
        flush=True,
    )
    if summary.get("baseline_ext_accuracy") is not None or summary.get("baseline_ext_accuracy_gen") is not None:
        ext_acc = summary.get("baseline_ext_accuracy") or summary.get("baseline_ext_accuracy_gen")
        ext_ci = summary.get("baseline_ext_ci")
        if ext_ci:
            print(
                f"Baseline (tier-3): {ext_acc:.4f} "
                f"CI [{ext_ci[0]:.4f}, {ext_ci[1]:.4f}]",
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
        acc = b.get("accuracy") or b.get("accuracy_gen")
        print(
            f"  Best: acc={acc:.4f} (delta +{b['delta']:.4f}) "
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

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved results to %s", out_path)

    return summary


def main():
    p = argparse.ArgumentParser(
        description="Benchmark MCTS sweep for Qwen 2.5 0.5B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks (excluding math/arc by default) to 1000 sims
  python run_qwen25_0.5b_benchmark_sweep.py --num_simulations 1000
  
  # Resume all incomplete runs to 1000 sims
  python run_qwen25_0.5b_benchmark_sweep.py --num_simulations 1000 --resume_incomplete
  
  # Run specific benchmarks
  python run_qwen25_0.5b_benchmark_sweep.py --benchmarks bigbench_all boolq
  
  # Exclude certain benchmarks
  python run_qwen25_0.5b_benchmark_sweep.py --exclude_benchmarks math500 gsm8k_hard
"""
    )
    p.add_argument(
        "--num_simulations",
        type=int,
        default=10000,
        help="Target number of simulations (default: 10000)",
    )
    p.add_argument(
        "--resume_snapshot",
        type=str,
        default=None,
        help="Resume from specific snapshot (run only that benchmark)",
    )
    p.add_argument(
        "--resume_incomplete",
        action="store_true",
        help="Auto-resume all benchmarks that haven't reached --num_simulations",
    )
    p.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=None,
        help="Run only these benchmarks (e.g. --benchmarks bigbench_all math500)",
    )
    p.add_argument(
        "--exclude_benchmarks",
        type=str,
        nargs="+",
        default=None,
        help="Exclude these benchmarks (default: math500 gsm8k_hard arc_easy arc_challenge)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    args = p.parse_args()

    set_seed(SEED)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    notify_signal = bool(_get_signal_cli_path())
    if not notify_signal:
        print(
            "WARNING: signal-cli not found. Signal notifications disabled.",
            file=sys.stderr,
            flush=True,
        )

    logger.info("Loading model: %s (GPU %d)", MODEL_NAME, GPU_RANK)
    model = MCTSModel(MODEL_NAME, rank=GPU_RANK)
    logger.info(
        "Model ready: %d layers, device=%s",
        model.num_layers,
        model.wrapper.model.device,
    )

    # Determine which benchmarks to run
    exclude = args.exclude_benchmarks if args.exclude_benchmarks is not None else DEFAULT_EXCLUDE
    
    if args.resume_snapshot:
        # Single snapshot resume
        stem = os.path.basename(args.resume_snapshot).replace("_snapshot.json", "")
        parts = stem.replace("benchmark_mcts_", "").rsplit("_", 1)
        dataset = parts[0] if len(parts) >= 2 else stem
        idx = BENCHMARKS.index(dataset) if dataset in BENCHMARKS else 0
        benchmarks_to_run = [(idx, dataset, args.resume_snapshot)]
    elif args.benchmarks:
        # Explicit benchmark list
        benchmarks_to_run = []
        for i, ds in enumerate(BENCHMARKS):
            if ds in args.benchmarks:
                resume_snap = None
                if args.resume_incomplete:
                    snap_info = find_latest_snapshot(output_dir, ds)
                    if snap_info and snap_info[1] < args.num_simulations:
                        resume_snap = snap_info[0]
                        logger.info("Will resume %s from sim %d to %d", ds, snap_info[1], args.num_simulations)
                benchmarks_to_run.append((i, ds, resume_snap))
        if not benchmarks_to_run:
            logger.error("No matching benchmarks in %s", args.benchmarks)
            sys.exit(1)
    else:
        # Run all (with exclusions)
        benchmarks_to_run = []
        for i, ds in enumerate(BENCHMARKS):
            if ds in exclude:
                logger.info("Excluding benchmark: %s", ds)
                continue
            
            resume_snap = None
            if args.resume_incomplete:
                snap_info = find_latest_snapshot(output_dir, ds)
                if snap_info:
                    if snap_info[1] < args.num_simulations:
                        resume_snap = snap_info[0]
                        logger.info("Will resume %s from sim %d to %d", ds, snap_info[1], args.num_simulations)
                    else:
                        logger.info("Skipping %s (already at %d sims >= %d target)", ds, snap_info[1], args.num_simulations)
                        continue
            
            benchmarks_to_run.append((i, ds, resume_snap))

    if not benchmarks_to_run:
        logger.info("No benchmarks to run (all complete or excluded)")
        return

    if notify_signal:
        send_signal(
            f"Qwen2.5-0.5B sweep started.\n"
            f"Benchmarks: {len(benchmarks_to_run)}\n"
            f"Sims per benchmark: {args.num_simulations}\n"
            f"Layers: {model.num_layers}"
        )

    all_summaries = {}
    sweep_t0 = time.time()

    # Cycle through benchmarks in increments of 500 sims instead of completing one at a time
    INCREMENT = 500
    current_sims = {}  # Track current sim count per benchmark
    
    # Initialize current sim counts from snapshots
    for idx, (i, dataset, resume_snap) in enumerate(benchmarks_to_run):
        if resume_snap:
            snap_info = find_latest_snapshot(output_dir, dataset)
            current_sims[dataset] = snap_info[1] if snap_info else 0
        else:
            current_sims[dataset] = 0
    
    # Cycle until all benchmarks reach target
    while any(current_sims[ds] < args.num_simulations for _, ds, _ in benchmarks_to_run):
        for idx, (i, dataset, resume_snap) in enumerate(benchmarks_to_run):
            if current_sims[dataset] >= args.num_simulations:
                continue  # Skip completed benchmarks
            
            target_sims = min(current_sims[dataset] + INCREMENT, args.num_simulations)
            logger.info(
                ">>> Benchmark %s: sims %d → %d (cycle increment)", 
                dataset, current_sims[dataset], target_sims
            )
            
            try:
                # Find latest snapshot for this benchmark
                snap_info = find_latest_snapshot(output_dir, dataset)
                resume_snap = snap_info[0] if snap_info else None
                
                summary = run_benchmark(
                    model, dataset, target_sims, notify_signal, resume_snapshot=resume_snap
                )
                baseline = _baseline_from_summary(summary)
                best = summary.get("best")
                best_delta = best["delta"] if best else 0.0
                best_acc = _best_acc_from_summary(summary)
                all_summaries[dataset] = {
                    "baseline": baseline,
                    "best_delta": best_delta,
                    "best_acc": best_acc,
                    "confirmed_better": summary["confirmed_better"],
                    "significant": summary["statistically_significant"],
                    "elapsed_min": summary["elapsed_seconds"] / 60,
                    "current_sims": target_sims,
                }
                current_sims[dataset] = target_sims
                
                if notify_signal:
                    s = all_summaries[dataset]
                    send_signal(
                        f"Qwen2.5-0.5B {dataset} → {target_sims} sims\n"
                        f"Baseline: {s['baseline']:.4f}\n"
                        f"Best delta: {s['best_delta']:+.4f}\n"
                        f"Confirmed better: {s['confirmed_better']}"
                    )
            except Exception as e:
                logger.error("FAILED on %s: %s", dataset, e, exc_info=True)
                all_summaries[dataset] = {"error": str(e), "current_sims": current_sims[dataset]}
                if notify_signal:
                    send_signal(f"Qwen2.5-0.5B sweep FAILED on {dataset}: {e}")

    total_elapsed = time.time() - sweep_t0
    print("\n" + "#" * 70, flush=True)
    print("SWEEP COMPLETE -- Qwen 2.5 0.5B Instruct", flush=True)
    print("#" * 70, flush=True)
    print(f"Total time: {total_elapsed / 3600:.1f} hours", flush=True)
    print(
        f"{'Benchmark':<20} {'Baseline':>10} {'Best D':>10} {'Confirmed':>10} {'Signif':>10}",
        flush=True,
    )
    print("-" * 70, flush=True)
    for ds in BENCHMARKS:
        s = all_summaries.get(ds, {})
        if "error" in s:
            print(f"{ds:<20} {'ERROR':>10}", flush=True)
        else:
            print(
                f"{ds:<20} {s.get('baseline', 0):.4f}     "
                f"{s.get('best_delta', 0):+.4f}     "
                f"{s.get('confirmed_better', 0):>5}      "
                f"{s.get('significant', 0):>5}",
                flush=True,
            )
    print("#" * 70, flush=True)

    sweep_path = os.path.join(
        output_dir, f"sweep_summary_{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    with open(sweep_path, "w") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "num_layers": model.num_layers,
                "benchmarks": [ds for _, ds, _ in benchmarks_to_run],
                "num_simulations": args.num_simulations,
                "total_elapsed_hours": total_elapsed / 3600,
                "results": all_summaries,
            },
            f,
            indent=2,
        )
    logger.info("Sweep summary saved to %s", sweep_path)

    if notify_signal:
        send_signal(
            f"Qwen2.5-0.5B sweep COMPLETE.\n"
            f"Total: {total_elapsed / 3600:.1f}h\n"
            f"Results: {sum(1 for s in all_summaries.values() if s.get('confirmed_better', 0) > 0)}"
            f"/{len(benchmarks_to_run)} benchmarks had confirmed-better sequences."
        )


if __name__ == "__main__":
    main()
