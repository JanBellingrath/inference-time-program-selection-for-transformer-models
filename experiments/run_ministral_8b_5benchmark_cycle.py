#!/usr/bin/env python
"""
Benchmark-level MCTS cycling sweep for Ministral 8B Instruct.

Cycles through benchmarks in 250-sim increments until each reaches the target
(default 5000 sims). Uses DISJOINT tier-2/3/4 sample pools and deterministic
per-benchmark shuffle seeds for reproducibility across resume cycles.

Usage:
    python run_ministral_8b_5benchmark_cycle.py
    python run_ministral_8b_5benchmark_cycle.py --num_simulations 5000
    python run_ministral_8b_5benchmark_cycle.py --benchmarks mmlu_all boolq arc_easy
    nohup python run_ministral_8b_5benchmark_cycle.py > nohup_ministral_8b_cycle.log 2>&1 &

``--device auto`` (default) uses CUDA when a tiny allocation succeeds; otherwise loads on CPU
(very slow). Override with ``--device cuda`` or ``--device cpu``, or env ``DR_LLM_DEVICE``.
"""
import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import hashlib
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
from pathlib import Path
from typing import Optional, Tuple, List, Dict

from core.permutation_mcts import (
    MCTSModel,
    PermutationMCTSConfig,
    prepare_arc_data,
    set_seed,
)
from core.benchmark_mcts import BenchmarkMCTS, send_signal, _get_signal_cli_path
from core.flexible_models import get_is_instruct, resolve_inference_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

MODEL_NAME = "mistralai/Ministral-8B-Instruct-2410"
SEED = 42
BENCHMARKS = ["winogrande", "arc_easy", "arc_challenge", "boolq", "mmlu_all", "commonsenseqa"]
INCREMENT = 250
NUM_SIMULATIONS = 5000
EVAL_BATCH_SIZE = 20
NUM_SAMPLES = 100
EXTENDED_SAMPLES = 500
EXTENDED_SAMPLES_TIER4 = 1000
NEIGHBORHOOD_RADIUS = 2
MAX_SWAPS = 4
EXPLORATION_CONSTANT = 1.8
RANDOM_PROB = 0.1
REPORT_EVERY = 50
VALIDATE_TOP_K = 3
PROMOTE_DELTA = 0.0
SPLIT = "train"
OUTPUT_DIR = "predictions/ministral_8b"


def _sample_hash(s: Dict) -> str:
    return hashlib.md5((s["input"] + str(s.get("correct", ""))).encode()).hexdigest()


def _make_disjoint_tiers(
    all_data: List[Dict],
    n_tier2: int = NUM_SAMPLES,
    n_tier3_total: int = EXTENDED_SAMPLES,
    n_tier4_total: int = EXTENDED_SAMPLES_TIER4,
) -> Tuple[List[Dict], List[Dict], Optional[List[Dict]]]:
    """Split data into DISJOINT tier-2, tier-3, tier-4 pools.

    n_tier3_total and n_tier4_total are the *cumulative* CLI-style sizes
    (e.g. 500, 1000), matching the --extended_samples semantics.
    Actual tier-3 gets [n_tier2 : n_tier3_total] and tier-4 gets
    [n_tier3_total : n_tier4_total], ensuring zero overlap.
    """
    n_t3 = max(0, n_tier3_total - n_tier2) if n_tier3_total > n_tier2 else 0
    n_t4 = max(0, n_tier4_total - n_tier3_total) if n_tier4_total > n_tier3_total else 0
    n_need = n_tier2 + n_t3 + n_t4

    if len(all_data) < n_need:
        scale = len(all_data) / n_need
        n_tier2 = max(10, int(n_tier2 * scale))
        n_t3 = int(n_t3 * scale)
        n_t4 = int(n_t4 * scale)
        logger.warning("Not enough data for full disjoint tiers: need %d, have %d. "
                       "Reduced to tier2=%d, tier3=%d, tier4=%d",
                       n_need, len(all_data), n_tier2, n_t3, n_t4)

    samples = all_data[:n_tier2]
    extended = all_data[n_tier2:n_tier2 + n_t3] if n_t3 > 0 else []
    extended_tier4 = all_data[n_tier2 + n_t3:n_tier2 + n_t3 + n_t4] if n_t4 > 0 else None

    t2h = {_sample_hash(s) for s in samples}
    t3h = {_sample_hash(s) for s in extended} if extended else set()
    t4h = {_sample_hash(s) for s in extended_tier4} if extended_tier4 else set()
    o23, o24, o34 = t2h & t3h, t2h & t4h, t3h & t4h
    if o23 or o24 or o34:
        raise ValueError(f"CRITICAL: tier overlap detected! "
                         f"t2∩t3={len(o23)}, t2∩t4={len(o24)}, t3∩t4={len(o34)}")

    logger.info("DISJOINT tiers: tier2=%d [0:%d], tier3=%d [%d:%d], tier4=%d [%d:%d]",
                len(samples), n_tier2,
                len(extended), n_tier2, n_tier2 + n_t3,
                len(extended_tier4) if extended_tier4 else 0,
                n_tier2 + n_t3, n_tier2 + n_t3 + n_t4)
    return samples, extended, extended_tier4


def find_latest_snapshot(output_dir: str, dataset: str) -> Optional[Tuple[str, int]]:
    """Find latest snapshot for dataset. Returns (path, sim_count) or None."""
    base = Path(output_dir)
    pattern = f"benchmark_mcts_{dataset}_*_snapshot.json"
    snapshots = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not snapshots:
        return None
    latest = snapshots[-1]
    try:
        with open(latest) as f:
            snap = json.load(f)
        sim = snap.get("sim", 0)
        return (str(latest), sim)
    except Exception:
        return (str(latest), 0)


def run_benchmark(model, dataset, target_sims, benchmark_idx, notify_signal, resume_snapshot=None):
    """Run MCTS search on a single benchmark up to target_sims."""
    logger.info("=" * 70)
    logger.info("STARTING BENCHMARK: %s (target %d sims)", dataset, target_sims)
    logger.info("=" * 70)
    t0 = time.time()

    is_instruct = get_is_instruct(MODEL_NAME)
    all_data = prepare_arc_data(dataset, is_instruct, split=SPLIT)

    shuffle_seed = SEED + benchmark_idx
    shuffle_rng = random.Random(shuffle_seed)
    shuffle_rng.shuffle(all_data)
    logger.info("Data shuffled with deterministic seed %d (SEED=%d + idx=%d)",
                shuffle_seed, SEED, benchmark_idx)

    samples, extended, extended_tier4 = _make_disjoint_tiers(all_data)

    logger.info(
        "Data: %d tier-2, %d tier-3, %d tier-4 from %s (%s)",
        len(samples), len(extended), len(extended_tier4) if extended_tier4 else 0, dataset, SPLIT
    )

    cfg = PermutationMCTSConfig(
        num_simulations=target_sims,
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
            raise ValueError("resume_snapshot must point to a *_snapshot.json file")
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
        target_sims,
        report_every=REPORT_EVERY,
        validate_top_k=VALIDATE_TOP_K,
        out_prefix=out_prefix,
        resume_prefix=out_prefix if resume_snapshot else None,
    )

    elapsed = time.time() - t0
    summary["elapsed_seconds"] = elapsed

    print("\n" + "=" * 60, flush=True)
    print(f"BENCHMARK MCTS RESULTS — {dataset} ({target_sims} sims)", flush=True)
    print("=" * 60, flush=True)
    print(f"Model:    {MODEL_NAME}", flush=True)
    print(f"Dataset:  {dataset} (tier-2: {len(samples)}, tier-3: {len(extended)}, "
          f"tier-4: {len(extended_tier4) if extended_tier4 else 0})", flush=True)
    print(f"Baseline (tier-2): gen={summary['baseline_accuracy_gen']:.4f} "
          f"CI [{summary['baseline_ci'][0]:.4f}, {summary['baseline_ci'][1]:.4f}]", flush=True)
    if summary.get("baseline_ext_accuracy_gen") is not None:
        print(f"Baseline (tier-3): gen={summary['baseline_ext_accuracy_gen']:.4f} "
              f"CI [{summary['baseline_ext_ci'][0]:.4f}, {summary['baseline_ext_ci'][1]:.4f}]", flush=True)
    print(f"Explored: {summary['unique_sequences_explored']} unique sequences", flush=True)
    print(
        f"Tier-2: {summary['tier2_validated']} | Tier-3: {summary['tier3_validated']} | "
        f"Tier-4: {summary.get('tier4_validated', 0)} | Confirmed: {summary['confirmed_better']} | "
        f"Significant: {summary['statistically_significant']}",
        flush=True,
    )
    if summary["best"]:
        b = summary["best"]
        print(f"  Best: gen={b['accuracy_gen']:.4f} (delta +{b['delta']:.4f}) "
              f"CI [{b['ci_lo']:.4f}, {b['ci_hi']:.4f}]", flush=True)
        print(f"  Layers ({b['length']}): {b['layers']}", flush=True)
        print(f"  Skips: {b['num_skips']}, Swaps: {b['num_swaps']}", flush=True)
    else:
        print("  No sequence confirmed beating baseline on rigorous eval.", flush=True)
    print(f"Time: {elapsed / 60:.1f} min", flush=True)
    print("=" * 60, flush=True)

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved results to %s", out_path)

    return summary


def main():
    p = argparse.ArgumentParser(description="Ministral 8B benchmark cycling MCTS (disjoint tiers)")
    p.add_argument("--num_simulations", type=int, default=NUM_SIMULATIONS,
                   help=f"Target simulations per benchmark (default {NUM_SIMULATIONS})")
    p.add_argument("--benchmarks", type=str, nargs="+", default=None,
                   help=f"Benchmarks to run (default: {' '.join(BENCHMARKS)})")
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="auto: CUDA if usable else CPU; cuda: fail if CUDA broken; cpu: force CPU",
    )
    args = p.parse_args()

    benchmarks = args.benchmarks if args.benchmarks else BENCHMARKS

    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    notify_signal = bool(_get_signal_cli_path())
    if not notify_signal:
        print("WARNING: signal-cli not found. Signal notifications disabled.",
              file=sys.stderr, flush=True)

    inference_device = resolve_inference_device(args.device, rank=0)
    logger.info("Loading model: %s (inference_device=%s)", MODEL_NAME, inference_device)
    model = MCTSModel(MODEL_NAME, rank=0, device=inference_device)
    logger.info("Model ready: %d layers, device=%s",
                model.num_layers, model.wrapper.model.device)

    if notify_signal:
        send_signal(
            f"Ministral-8B cycle started (disjoint tiers).\n"
            f"Benchmarks: {', '.join(benchmarks)}\n"
            f"Target sims: {args.num_simulations}, Increment: {INCREMENT}\n"
            f"Layers: {model.num_layers}"
        )

    all_summaries = {}
    sweep_t0 = time.time()

    current_sims = {}
    for dataset in benchmarks:
        snap_info = find_latest_snapshot(OUTPUT_DIR, dataset)
        current_sims[dataset] = snap_info[1] if snap_info else 0

    while any(current_sims[ds] < args.num_simulations for ds in benchmarks):
        for dataset in benchmarks:
            if current_sims[dataset] >= args.num_simulations:
                continue

            benchmark_idx = benchmarks.index(dataset)
            target_sims = min(current_sims[dataset] + INCREMENT, args.num_simulations)
            logger.info(
                ">>> Benchmark %s: sims %d → %d (cycle increment)",
                dataset, current_sims[dataset], target_sims
            )

            try:
                snap_info = find_latest_snapshot(OUTPUT_DIR, dataset)
                resume_snap = snap_info[0] if snap_info else None

                summary = run_benchmark(
                    model, dataset, target_sims, benchmark_idx, notify_signal,
                    resume_snapshot=resume_snap
                )
                baseline = summary.get("baseline_accuracy_gen") or summary.get("baseline_accuracy")
                best = summary.get("best")
                best_delta = best["delta"] if best else 0.0
                best_acc = best.get("accuracy") or best.get("accuracy_gen") if best else None
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
                        f"Ministral-8B {dataset} → {target_sims} sims\n"
                        f"Baseline: {s['baseline']:.4f}\n"
                        f"Best delta: {s['best_delta']:+.4f}\n"
                        f"Confirmed better: {s['confirmed_better']}"
                    )
            except Exception as e:
                logger.error("FAILED on %s: %s", dataset, e, exc_info=True)
                all_summaries[dataset] = {"error": str(e), "current_sims": current_sims[dataset]}
                if notify_signal:
                    send_signal(f"Ministral-8B cycle FAILED on {dataset}: {e}")

    total_elapsed = time.time() - sweep_t0
    print("\n" + "#" * 70, flush=True)
    print("CYCLE SWEEP COMPLETE — Ministral 8B Instruct (disjoint tiers)", flush=True)
    print("#" * 70, flush=True)
    print(f"Total time: {total_elapsed / 3600:.1f} hours", flush=True)
    print(f"{'Benchmark':<20} {'Baseline':>10} {'Best Δ':>10} {'Confirmed':>10} {'Signif':>10}",
          flush=True)
    print("-" * 70, flush=True)
    for ds in benchmarks:
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

    sweep_path = os.path.join(OUTPUT_DIR, f"cycle_summary_{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(sweep_path, "w") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "num_layers": model.num_layers,
                "benchmarks": benchmarks,
                "target_simulations": args.num_simulations,
                "increment": INCREMENT,
                "disjoint_tiers": True,
                "total_elapsed_hours": total_elapsed / 3600,
                "results": all_summaries,
            },
            f,
            indent=2,
        )
    logger.info("Sweep summary saved to %s", sweep_path)

    if notify_signal:
        send_signal(
            f"Ministral-8B cycle COMPLETE.\n"
            f"Total: {total_elapsed / 3600:.1f}h\n"
            f"Results: {sum(1 for s in all_summaries.values() if s.get('confirmed_better', 0) > 0)}"
            f"/{len(benchmarks)} had confirmed-better sequences."
        )


if __name__ == "__main__":
    main()
