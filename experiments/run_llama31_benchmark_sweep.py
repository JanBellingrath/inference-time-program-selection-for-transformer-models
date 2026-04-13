#!/usr/bin/env python
"""
Benchmark-level MCTS sweep for Llama 3.1 8B Instruct.

Loads the model once, then runs 500 MCTS simulations on each benchmark
sequentially. MC benchmarks are run first (faster), followed by math
benchmarks (max_new_tokens=256).

Usage:
    python run_llama31_benchmark_sweep.py
    python run_llama31_benchmark_sweep.py --resume_snapshot predictions/llama31_8b/benchmark_mcts_math500_20260226-190516_snapshot.json
    nohup python run_llama31_benchmark_sweep.py > nohup_llama31_sweep.log 2>&1 &
"""
import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

# BigBench HF loader requires datasets.load_metric (moved to evaluate)
import argparse
import datasets
try:
    from evaluate import load as _load_metric
    datasets.load_metric = _load_metric
except Exception:
    pass

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

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_NAME = "NousResearch/Meta-Llama-3.1-8B-Instruct"
SEED = 42

# MC benchmarks first (fast), then math benchmarks (slow)
BENCHMARKS = ["bigbench_all", "math500", "gsm8k_hard"]

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

OUTPUT_DIR = "predictions/llama31_8b"


def run_benchmark(model, dataset, notify_signal, resume_snapshot=None):
    """Run MCTS search on a single benchmark, return summary dict."""
    logger.info("=" * 70)
    logger.info("STARTING BENCHMARK: %s", dataset)
    logger.info("=" * 70)
    t0 = time.time()

    is_instruct = get_is_instruct(MODEL_NAME)
    all_data = prepare_arc_data(dataset, is_instruct, split=SPLIT)
    random.shuffle(all_data)

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
        num_simulations=NUM_SIMULATIONS,
        exploration_constant=EXPLORATION_CONSTANT,
        random_prob=RANDOM_PROB,
        neighborhood_radius=NEIGHBORHOOD_RADIUS,
        max_swaps=MAX_SWAPS,
        model_name=MODEL_NAME,
        dataset=dataset,
        num_samples=NUM_SAMPLES,
    )

    if resume_snapshot:
        if not resume_snapshot.endswith("_snapshot.json"):
            raise ValueError("--resume_snapshot must point to a *_snapshot.json file")
        out_prefix = resume_snapshot[:-len("_snapshot.json")]
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
        NUM_SIMULATIONS,
        report_every=REPORT_EVERY,
        validate_top_k=VALIDATE_TOP_K,
        out_prefix=out_prefix,
        resume_prefix=out_prefix if resume_snapshot else None,
    )

    elapsed = time.time() - t0
    summary["elapsed_seconds"] = elapsed

    # ── Print results ────────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print(f"BENCHMARK MCTS RESULTS — {dataset}", flush=True)
    print("=" * 60, flush=True)
    print(f"Model:    {MODEL_NAME}", flush=True)
    print(
        f"Dataset:  {dataset} (tier-2: {len(samples)}, tier-3: {len(extended)}, "
        f"tier-4: {len(extended_tier4) if extended_tier4 else 0})",
        flush=True,
    )
    print(
        f"Baseline (tier-2): gen={summary['baseline_accuracy_gen']:.4f} "
        f"CI [{summary['baseline_ci'][0]:.4f}, {summary['baseline_ci'][1]:.4f}]",
        flush=True,
    )
    if summary.get("baseline_ext_accuracy_gen") is not None:
        print(
            f"Baseline (tier-3): gen={summary['baseline_ext_accuracy_gen']:.4f} "
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
        print(
            f"  Best: gen={b['accuracy_gen']:.4f} (delta +{b['delta']:.4f}) "
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
    p = argparse.ArgumentParser(description="Benchmark MCTS sweep for Llama 3.1 8B")
    p.add_argument("--resume_snapshot", type=str, default=None,
                   help="Resume from snapshot (run only that benchmark)")
    p.add_argument("--benchmark", type=str, default=None,
                   help="Run only this benchmark (e.g. gsm8k_hard)")
    args = p.parse_args()

    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    notify_signal = bool(_get_signal_cli_path())
    if not notify_signal:
        print(
            "WARNING: signal-cli not found. Signal notifications disabled.",
            file=sys.stderr,
            flush=True,
        )

    # ── Load model once ──────────────────────────────────────────────────
    logger.info("Loading model: %s", MODEL_NAME)
    model = MCTSModel(MODEL_NAME, rank=0)
    logger.info(
        "Model ready: %d layers, device=%s",
        model.num_layers,
        model.wrapper.model.device,
    )

    if notify_signal:
        send_signal(
            f"Llama-3.1-8B sweep started.\n"
            f"Benchmarks: {', '.join(BENCHMARKS)}\n"
            f"Sims per benchmark: {NUM_SIMULATIONS}\n"
            f"Layers: {model.num_layers}"
        )

    # ── Sweep ────────────────────────────────────────────────────────────
    all_summaries = {}
    sweep_t0 = time.time()

    if args.resume_snapshot:
        stem = os.path.basename(args.resume_snapshot).replace("_snapshot.json", "")
        parts = stem.split("_")
        dataset = "_".join(parts[2:-1]) if len(parts) > 3 else parts[2] if len(parts) > 2 else ""
        benchmarks_to_run = [(0, dataset)]
        resume = args.resume_snapshot
    elif args.benchmark:
        if args.benchmark not in BENCHMARKS:
            sys.exit(f"Unknown benchmark '{args.benchmark}'. Choices: {BENCHMARKS}")
        idx = BENCHMARKS.index(args.benchmark)
        benchmarks_to_run = [(idx, args.benchmark)]
        resume = None
    else:
        benchmarks_to_run = list(enumerate(BENCHMARKS))
        resume = None

    for i, dataset in benchmarks_to_run:
        logger.info(
            ">>> Benchmark %d/%d: %s", i + 1, len(BENCHMARKS), dataset
        )
        try:
            summary = run_benchmark(model, dataset, notify_signal, resume_snapshot=resume)
            all_summaries[dataset] = {
                "baseline": summary["baseline_accuracy_gen"],
                "best_delta": summary["best"]["delta"] if summary["best"] else 0.0,
                "best_acc": summary["best"]["accuracy_gen"] if summary["best"] else None,
                "confirmed_better": summary["confirmed_better"],
                "significant": summary["statistically_significant"],
                "elapsed_min": summary["elapsed_seconds"] / 60,
            }
        except Exception as e:
            logger.error("FAILED on %s: %s", dataset, e, exc_info=True)
            all_summaries[dataset] = {"error": str(e)}
            if notify_signal:
                send_signal(f"Llama-3.1-8B sweep FAILED on {dataset}: {e}")

        # Completion notification per benchmark
        if notify_signal and dataset in all_summaries and "error" not in all_summaries[dataset]:
            s = all_summaries[dataset]
            send_signal(
                f"Llama-3.1-8B {dataset} DONE ({i+1}/{len(BENCHMARKS)})\n"
                f"Baseline: {s['baseline']:.4f}\n"
                f"Best delta: {s['best_delta']:+.4f}\n"
                f"Confirmed better: {s['confirmed_better']}\n"
                f"Elapsed: {s['elapsed_min']:.1f} min"
            )

    # ── Final summary ────────────────────────────────────────────────────
    total_elapsed = time.time() - sweep_t0
    print("\n" + "#" * 70, flush=True)
    print("SWEEP COMPLETE — Llama 3.1 8B Instruct", flush=True)
    print("#" * 70, flush=True)
    print(f"Total time: {total_elapsed / 3600:.1f} hours", flush=True)
    print(f"{'Benchmark':<20} {'Baseline':>10} {'Best Δ':>10} {'Confirmed':>10} {'Signif':>10}", flush=True)
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

    sweep_path = os.path.join(OUTPUT_DIR, f"sweep_summary_{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(sweep_path, "w") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "num_layers": model.num_layers,
                "benchmarks": BENCHMARKS,
                "num_simulations": NUM_SIMULATIONS,
                "total_elapsed_hours": total_elapsed / 3600,
                "results": all_summaries,
            },
            f,
            indent=2,
        )
    logger.info("Sweep summary saved to %s", sweep_path)

    if notify_signal:
        send_signal(
            f"Llama-3.1-8B sweep COMPLETE.\n"
            f"Total: {total_elapsed / 3600:.1f}h\n"
            f"Results: {sum(1 for s in all_summaries.values() if s.get('confirmed_better', 0) > 0)}"
            f"/{len(BENCHMARKS)} benchmarks had confirmed-better sequences."
        )


if __name__ == "__main__":
    main()
