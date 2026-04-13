#!/usr/bin/env python3
"""
Publication-quality RC vs SC experiment.

Single-pass design: generates max_K RC responses and max_K × n_seeds SC
responses per sample, then computes metrics at every K <= max_K from stored
responses.  This avoids re-running inference for each K value.

Includes:
  - Paired McNemar test (RC vs SC, per K, per benchmark)
  - Paired bootstrap CI on accuracy difference
  - Decomposition with bootstrap CIs (route quality vs aggregation effects)
  - Multi-seed SC for variance estimation
  - Full per-sample data for post-hoc analysis

Usage:
    python run_publication_rc_vs_sc.py \
        --model_name Qwen/Qwen2.5-0.5B-Instruct \
        --benchmarks winogrande boolq arc_easy commonsenseqa mmlu_all arc_challenge \
        --max_K 20 \
        --sc_seeds 42 1337 2024 7 99 \
        --sc_temperature_json predictions/temperature_sweep_K5_20260319-172442.json \
        --holdout_fraction 0.5 \
        --tier tier4 \
        --output_dir predictions/publication
"""
import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import argparse
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from core.permutation_mcts import MCTSModel, prepare_arc_data, set_seed
from core.flexible_models import get_is_instruct
from core.benchmark_mcts import grade_response
from compare_aggregation import (
    extract_answer,
    generate_one,
    majority_vote,
    weighted_vote,
    wilson_ci,
)
from evaluation.evaluate_transfer import load_sequences_from_snapshot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Snapshot registry: (model_key, benchmark) -> relative path from SCRIPT_DIR
# ---------------------------------------------------------------------------

SNAPSHOT_REGISTRY = {
    # Qwen2.5-0.5B-Instruct (r5_pw preferred, sdpa fallback)
    ("0.5B", "winogrande"): "predictions/qwen25_0.5b_v2_sdpa_r5_pw/benchmark_mcts_winogrande_20260311-104703_snapshot.json",
    ("0.5B", "boolq"): "predictions/qwen25_0.5b_v2_sdpa_r5_pw/benchmark_mcts_boolq_20260311-115713_snapshot.json",
    ("0.5B", "arc_easy"): "predictions/qwen25_0.5b_v2_sdpa/benchmark_mcts_arc_easy_20260308-232019_snapshot.json",
    ("0.5B", "commonsenseqa"): "predictions/qwen25_0.5b_v2_sdpa_r5_pw/benchmark_mcts_commonsenseqa_20260311-125244_snapshot.json",
    ("0.5B", "mmlu_all"): "predictions/qwen25_0.5b_v2_sdpa_r5_pw/benchmark_mcts_mmlu_all_20260311-132051_snapshot.json",
    ("0.5B", "arc_challenge"): "predictions/qwen25_0.5b_v2_sdpa_r5_pw/benchmark_mcts_arc_challenge_20260311-112922_snapshot.json",
    # Qwen2.5-7B-Instruct
    ("7B", "winogrande"): "predictions/qwen25_7b_v2_sdpa/benchmark_mcts_winogrande_20260308-191822_snapshot.json",
    ("7B", "boolq"): "predictions/qwen25_7b_v2_sdpa/benchmark_mcts_boolq_20260308-195558_snapshot.json",
    ("7B", "arc_easy"): "predictions/qwen25_7b_v2_sdpa/benchmark_mcts_arc_easy_20260308-221829_snapshot.json",
    ("7B", "commonsenseqa"): "predictions/qwen25_7b_v2_sdpa/benchmark_mcts_commonsenseqa_20260308-202012_snapshot.json",
    ("7B", "mmlu_all"): "predictions/qwen25_7b_v2_sdpa/benchmark_mcts_mmlu_all_20260308-211217_snapshot.json",
    ("7B", "arc_challenge"): "predictions/qwen25_7b_v2_sdpa/benchmark_mcts_arc_challenge_20260308-013818_snapshot.json",
}


def model_key_from_name(model_name: str) -> str:
    name = model_name.lower()
    if "0.5b" in name:
        return "0.5B"
    if "7b" in name:
        return "7B"
    raise ValueError(f"Cannot infer model key from {model_name}; expected '0.5b' or '7b' in name")


def resolve_snapshot(model_name: str, benchmark: str, cli_snapshot: Optional[str] = None) -> str:
    if cli_snapshot:
        return cli_snapshot
    key = (model_key_from_name(model_name), benchmark)
    rel = SNAPSHOT_REGISTRY.get(key)
    if not rel:
        raise ValueError(f"No snapshot registered for {key}. Pass --snapshot explicitly.")
    return str(SCRIPT_DIR / rel)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def mcnemar_test(rc_correct: List[int], sc_correct: List[int]) -> Dict[str, Any]:
    """McNemar's test with continuity correction for paired binary outcomes."""
    b = sum(r == 1 and s == 0 for r, s in zip(rc_correct, sc_correct))
    c = sum(r == 0 and s == 1 for r, s in zip(rc_correct, sc_correct))
    n_discord = b + c
    if n_discord == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b": b, "c": c, "n_discordant": 0}
    chi2 = (abs(b - c) - 1) ** 2 / n_discord
    from scipy.stats import chi2 as chi2_dist
    p = 1 - chi2_dist.cdf(chi2, df=1)
    return {"chi2": float(chi2), "p_value": float(p), "b": b, "c": c, "n_discordant": n_discord}


def paired_bootstrap_ci(
    rc_correct: np.ndarray,
    sc_correct: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng_seed: int = 12345,
) -> Dict[str, float]:
    """Bootstrap CI on mean(rc_correct - sc_correct)."""
    diffs = rc_correct.astype(float) - sc_correct.astype(float)
    n = len(diffs)
    rng = np.random.RandomState(rng_seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_means[i] = diffs[idx].mean()
    boot_means.sort()
    lo = float(boot_means[int(n_boot * alpha / 2)])
    hi = float(boot_means[int(n_boot * (1 - alpha / 2))])
    return {"mean_diff": float(diffs.mean()), "ci_lo": lo, "ci_hi": hi, "n": n}


def bootstrap_scalar_ci(
    values: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng_seed: int = 12345,
) -> Dict[str, float]:
    """Bootstrap CI on the mean of a 1-D array of per-sample values."""
    n = len(values)
    rng = np.random.RandomState(rng_seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_means[i] = values[idx].mean()
    boot_means.sort()
    lo = float(boot_means[int(n_boot * alpha / 2)])
    hi = float(boot_means[int(n_boot * (1 - alpha / 2))])
    return {"mean": float(values.mean()), "ci_lo": lo, "ci_hi": hi}


# ---------------------------------------------------------------------------
# Tune/holdout split (same logic as sweep_temperature.py for reproducibility)
# ---------------------------------------------------------------------------

def split_tune_holdout(
    eval_data: List[Dict],
    holdout_fraction: float,
    seed: int,
    benchmark: str,
) -> Tuple[List[Dict], List[Dict]]:
    n = len(eval_data)
    rng = random.Random(seed + sum(ord(c) for c in benchmark))
    order = list(range(n))
    rng.shuffle(order)
    n_hold = max(1, int(round(n * holdout_fraction)))
    n_hold = min(n_hold, n - 1)
    hold_idx = set(order[-n_hold:])
    tune_data = [eval_data[i] for i in sorted(set(order[:-n_hold]))]
    holdout_data = [eval_data[i] for i in sorted(hold_idx)]
    return tune_data, holdout_data


# ---------------------------------------------------------------------------
# Grade helper
# ---------------------------------------------------------------------------

def _grade(answer: Optional[str], correct: str, benchmark: str,
           model_name: str, input_text: str) -> int:
    if answer is None:
        return 0
    return int(grade_response(answer, correct, benchmark, model_name, input_text) > 0.5)


# ---------------------------------------------------------------------------
# Core: single-pass data collection
# ---------------------------------------------------------------------------

def collect_responses(
    wrapper,
    benchmark: str,
    model_name: str,
    holdout_data: List[Dict],
    default_layers: List[int],
    routes: List[List[int]],
    sc_temperature: float,
    max_K: int,
    sc_seeds: List[int],
) -> List[Dict[str, Any]]:
    """
    For each sample, generate:
      - 1 baseline response (greedy, default route)
      - max_K RC responses (greedy, one per route)
      - max_K SC responses per seed (temperature-sampled, default route)

    Returns a list of per-sample records.
    """
    is_math = "dart" in benchmark or benchmark in ("gsm8k_hard", "math500")
    K_eff = min(max_K, len(routes))
    n_seeds = len(sc_seeds)
    passes_per_sample = 1 + K_eff + n_seeds * K_eff
    total_passes = len(holdout_data) * passes_per_sample
    logger.info(
        "%s: %d samples × (%d base + %d RC + %d×%d SC) = %d fwd passes",
        benchmark, len(holdout_data), 1, K_eff, n_seeds, K_eff, total_passes,
    )

    records = []
    t0 = time.time()

    for idx, sample in enumerate(tqdm(holdout_data, desc=f"{benchmark}")):
        text = sample["input"]
        correct_answer = sample["correct"]
        sys_prompt = sample.get("system_prompt")
        max_tokens = sample.get("max_new_tokens", 10)

        # Baseline
        base_resp = generate_one(
            wrapper, text, default_layers, sys_prompt, max_tokens,
            temperature=0.0, is_math=is_math,
        )
        base_answer = extract_answer(base_resp, benchmark, model_name, text)
        base_ok = _grade(base_answer, correct_answer, benchmark, model_name, text)

        # RC: one greedy pass per route
        rc_answers = []
        rc_ok = []
        for k in range(K_eff):
            resp = generate_one(
                wrapper, text, routes[k], sys_prompt, max_tokens,
                temperature=0.0, is_math=is_math,
            )
            ans = extract_answer(resp, benchmark, model_name, text)
            rc_answers.append(ans)
            rc_ok.append(_grade(ans, correct_answer, benchmark, model_name, text))

        # SC: max_K samples per seed
        sc_answers_by_seed = {}
        sc_ok_by_seed = {}
        for seed in sc_seeds:
            torch.manual_seed(seed + idx * 31337)
            random.seed(seed + idx * 31337)
            sc_ans = []
            sc_ok_list = []
            for k in range(K_eff):
                resp = generate_one(
                    wrapper, text, default_layers, sys_prompt, max_tokens,
                    temperature=sc_temperature, is_math=is_math,
                )
                ans = extract_answer(resp, benchmark, model_name, text)
                sc_ans.append(ans)
                sc_ok_list.append(_grade(ans, correct_answer, benchmark, model_name, text))
            sc_answers_by_seed[seed] = sc_ans
            sc_ok_by_seed[seed] = sc_ok_list

        records.append({
            "idx": idx,
            "correct_answer": correct_answer,
            "base_answer": base_answer,
            "base_ok": base_ok,
            "rc_answers": rc_answers,
            "rc_ok": rc_ok,
            "sc_answers_by_seed": {str(s): v for s, v in sc_answers_by_seed.items()},
            "sc_ok_by_seed": {str(s): v for s, v in sc_ok_by_seed.items()},
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(holdout_data):
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(holdout_data) - idx - 1) / rate if rate > 0 else 0
            logger.info(
                "  [%d/%d] %.1f samples/s  ETA %.0fs",
                idx + 1, len(holdout_data), rate, eta,
            )

    elapsed = time.time() - t0
    logger.info("%s: data collection done in %.1fs", benchmark, elapsed)
    return records


# ---------------------------------------------------------------------------
# Post-hoc analysis at each K
# ---------------------------------------------------------------------------

def analyze_at_K(
    records: List[Dict],
    K: int,
    sc_seeds: List[int],
    benchmark: str,
    model_name: str,
) -> Dict[str, Any]:
    """Compute all metrics at a specific K from pre-collected responses."""
    n = len(records)
    vote_rng = random.Random(42 + K)

    baseline_ok_arr = np.array([r["base_ok"] for r in records])

    # RC at K: majority vote over first K routes
    rc_voted_ok = np.zeros(n, dtype=int)
    per_route_ok = np.zeros((n, K), dtype=int)
    for i, r in enumerate(records):
        answers_k = r["rc_answers"][:K]
        ok_k = r["rc_ok"][:K]
        per_route_ok[i, :len(ok_k)] = ok_k
        voted = majority_vote(answers_k, vote_rng)
        rc_voted_ok[i] = _grade(voted, r["correct_answer"], benchmark, model_name, "")

    # SC at K: majority vote over first K samples, per seed
    sc_voted_ok_by_seed = {}
    per_sc_ok_by_seed = {}
    for seed in sc_seeds:
        sc_voted = np.zeros(n, dtype=int)
        per_sc = np.zeros((n, K), dtype=int)
        for i, r in enumerate(records):
            answers_k = r["sc_answers_by_seed"][str(seed)][:K]
            ok_k = r["sc_ok_by_seed"][str(seed)][:K]
            per_sc[i, :len(ok_k)] = ok_k
            voted = majority_vote(answers_k, vote_rng)
            sc_voted[i] = _grade(voted, r["correct_answer"], benchmark, model_name, "")
        sc_voted_ok_by_seed[seed] = sc_voted
        per_sc_ok_by_seed[seed] = per_sc

    # Aggregate SC across seeds: per-sample mean, then threshold at 0.5
    # For paired tests, use the mean-seed SC correctness
    sc_voted_mean = np.mean(
        [sc_voted_ok_by_seed[s] for s in sc_seeds], axis=0
    )
    # Binary SC outcome: correct if majority of seeds get it right
    sc_voted_majority = (sc_voted_mean > 0.5).astype(int)

    # Accuracies
    base_acc = float(baseline_ok_arr.mean())
    rc_acc = float(rc_voted_ok.mean())
    sc_acc_per_seed = {s: float(sc_voted_ok_by_seed[s].mean()) for s in sc_seeds}
    sc_acc_mean = float(np.mean(list(sc_acc_per_seed.values())))
    sc_acc_std = float(np.std(list(sc_acc_per_seed.values())))

    # Per-route / per-SC-sample accuracies
    per_route_acc = [float(per_route_ok[:, k].mean()) for k in range(K)]
    avg_route_acc = float(np.mean(per_route_acc))
    best_route_acc = float(np.max(per_route_acc)) if per_route_acc else 0.0

    per_sc_acc_per_seed = {}
    for seed in sc_seeds:
        per_sc_acc_per_seed[seed] = [float(per_sc_ok_by_seed[seed][:, k].mean()) for k in range(K)]
    avg_sc_acc = float(np.mean([np.mean(v) for v in per_sc_acc_per_seed.values()]))

    # Decomposition
    route_quality_effect = avg_route_acc - base_acc
    rc_aggregation_effect = rc_acc - avg_route_acc
    sc_aggregation_effect = sc_acc_mean - avg_sc_acc

    # Wilson CIs
    ci_base = wilson_ci(int(baseline_ok_arr.sum()), n)
    ci_rc = wilson_ci(int(rc_voted_ok.sum()), n)

    # Paired tests: RC vs best-seed SC (use seed with median accuracy)
    sorted_seeds = sorted(sc_seeds, key=lambda s: sc_acc_per_seed[s])
    median_seed = sorted_seeds[len(sorted_seeds) // 2]
    sc_for_paired = sc_voted_ok_by_seed[median_seed]

    mcnemar = mcnemar_test(rc_voted_ok.tolist(), sc_for_paired.tolist())
    bootstrap = paired_bootstrap_ci(rc_voted_ok, sc_for_paired)

    # Bootstrap CIs on decomposition components
    per_sample_route_quality = per_route_ok.mean(axis=1) - baseline_ok_arr
    per_sample_rc_agg = rc_voted_ok - per_route_ok.mean(axis=1)
    # SC aggregation per sample (using median seed)
    per_sample_sc_individual = per_sc_ok_by_seed[median_seed].mean(axis=1)
    per_sample_sc_agg = sc_for_paired - per_sample_sc_individual

    boot_route_quality = bootstrap_scalar_ci(per_sample_route_quality)
    boot_rc_agg = bootstrap_scalar_ci(per_sample_rc_agg)
    boot_sc_agg = bootstrap_scalar_ci(per_sample_sc_agg)

    return {
        "K": K,
        "n": n,
        "baseline_acc": base_acc,
        "rc_acc": rc_acc,
        "sc_acc_mean": sc_acc_mean,
        "sc_acc_std": sc_acc_std,
        "sc_acc_per_seed": {str(s): v for s, v in sc_acc_per_seed.items()},
        "ci_baseline": ci_base,
        "ci_rc": ci_rc,
        "per_route_acc": per_route_acc,
        "avg_route_acc": avg_route_acc,
        "best_route_acc": best_route_acc,
        "avg_sc_acc": avg_sc_acc,
        "route_quality_effect": route_quality_effect,
        "rc_aggregation_effect": rc_aggregation_effect,
        "sc_aggregation_effect": sc_aggregation_effect,
        "mcnemar_rc_vs_sc": mcnemar,
        "paired_bootstrap_rc_minus_sc": bootstrap,
        "bootstrap_route_quality": boot_route_quality,
        "bootstrap_rc_aggregation": boot_rc_agg,
        "bootstrap_sc_aggregation": boot_sc_agg,
        "paired_seed": median_seed,
    }


# ---------------------------------------------------------------------------
# Load best temperatures from a prior sweep JSON
# ---------------------------------------------------------------------------

def load_best_temperatures(path: str) -> Dict[str, float]:
    """Load best SC temperature per benchmark from a prior sweep JSON."""
    with open(path) as f:
        saved = json.load(f)
    temps = {}
    for benchmark, entry in saved.items():
        if entry.get("best_temp_from_tune") is not None:
            temps[benchmark] = float(entry["best_temp_from_tune"])
        elif "tune" in entry and entry["tune"].get("temperatures"):
            t = entry["tune"]["temperatures"]
            best = max(t.keys(), key=lambda k: t[k]["accuracy"])
            temps[benchmark] = float(best)
        elif entry.get("temperatures"):
            t = entry["temperatures"]
            best = max(t.keys(), key=lambda k: t[k]["accuracy"])
            temps[benchmark] = float(best)
    return temps


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(all_results: Dict[str, Dict[str, Any]], K_values: List[int]):
    print("\n" + "=" * 100)
    print("PUBLICATION RC vs SC SUMMARY")
    print("=" * 100)

    # Main table at each K
    for K in K_values:
        print(f"\n--- K = {K} ---")
        print(
            f"{'Benchmark':<16} {'n':>5} {'Base':>7} {'SC±std':>13} "
            f"{'RC':>7} {'RC-SC':>7} {'p(McN)':>8} {'BootCI':>17} "
            f"{'RteQ':>7} {'RCag':>7} {'SCag':>7}"
        )
        print("-" * 115)
        for benchmark, br in all_results.items():
            k_key = str(K)
            if k_key not in br["analysis"]:
                continue
            a = br["analysis"][k_key]
            sc_str = f"{a['sc_acc_mean']:.3f}±{a['sc_acc_std']:.3f}"
            diff = a['rc_acc'] - a['sc_acc_mean']
            boot = a['paired_bootstrap_rc_minus_sc']
            boot_str = f"[{boot['ci_lo']:+.3f},{boot['ci_hi']:+.3f}]"
            p_val = a['mcnemar_rc_vs_sc']['p_value']
            p_str = f"{p_val:.4f}" if p_val >= 0.001 else f"{p_val:.1e}"
            print(
                f"{benchmark:<16} {a['n']:>5} {a['baseline_acc']:>7.3f} {sc_str:>13} "
                f"{a['rc_acc']:>7.3f} {diff:>+7.3f} {p_str:>8} {boot_str:>17} "
                f"{a['route_quality_effect']:>+7.3f} {a['rc_aggregation_effect']:>+7.3f} "
                f"{a['sc_aggregation_effect']:>+7.3f}"
            )

    print("\n" + "=" * 100)
    print("  RteQ  = AvgRoute − Base (MCTS routes individually better?)")
    print("  RCag  = RC vote − AvgRoute (voting over routes adds value?)")
    print("  SCag  = SC vote − AvgSC (voting over temp samples adds value?)")
    print("  p(McN) = McNemar's test p-value (RC vs median-seed SC)")
    print("  BootCI = 95% paired bootstrap CI on RC − SC accuracy")
    print("=" * 100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Publication RC vs SC: single-pass K-sweep with paired stats."
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--benchmarks", nargs="+", required=True)
    parser.add_argument("--max_K", type=int, default=20)
    parser.add_argument("--K_values", nargs="+", type=int, default=None,
                        help="K values to analyze (default: 1,3,5,10,15,max_K)")
    parser.add_argument("--sc_seeds", nargs="+", type=int,
                        default=[42, 1337, 2024, 7, 99])
    parser.add_argument("--sc_temperature", type=float, default=None,
                        help="Fixed SC temperature for all benchmarks.")
    parser.add_argument("--sc_temperature_json", type=str, default=None,
                        help="Load best temperature per benchmark from prior sweep JSON.")
    parser.add_argument("--holdout_fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tier", type=str, default="tier4",
                        choices=["tier4", "tier3", "auto"])
    parser.add_argument("--snapshot", type=str, default=None,
                        help="Override snapshot path (applies to all benchmarks).")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Cap eval samples (0 = full split).")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_benchmarks_without_enough_routes", action="store_true",
                        help="Skip benchmarks with fewer routes than max_K instead of erroring.")
    args = parser.parse_args()

    set_seed(args.seed)

    K_values = args.K_values
    if K_values is None:
        K_values = sorted(set([1, 3, 5, 10, 15, args.max_K]))
    K_values = [k for k in K_values if k <= args.max_K]

    # Load best temperatures
    best_temps: Dict[str, float] = {}
    if args.sc_temperature_json:
        p = Path(args.sc_temperature_json)
        if not p.is_file():
            p = SCRIPT_DIR / "predictions" / args.sc_temperature_json
        best_temps = load_best_temperatures(str(p))
        logger.info("Loaded best temperatures from %s: %s", p, best_temps)

    # Load model
    logger.info("Loading model: %s", args.model_name)
    model = MCTSModel(args.model_name)
    wrapper = model.wrapper
    num_layers = model.num_layers
    default_layers = list(range(num_layers))
    is_instruct = get_is_instruct(args.model_name)

    all_results: Dict[str, Any] = {}
    out_dir = args.output_dir or str(SCRIPT_DIR / "predictions" / "publication")
    os.makedirs(out_dir, exist_ok=True)

    for benchmark in args.benchmarks:
        logger.info("=" * 70)
        logger.info("BENCHMARK: %s", benchmark)
        logger.info("=" * 70)

        # Resolve temperature
        sc_temp = args.sc_temperature
        if sc_temp is None:
            sc_temp = best_temps.get(benchmark)
        if sc_temp is None:
            sc_temp = 0.7
            logger.warning(
                "%s: no temperature found in JSON or CLI; using default %.1f",
                benchmark, sc_temp,
            )

        # Load routes
        snapshot_path = resolve_snapshot(args.model_name, benchmark, args.snapshot)
        candidates = load_sequences_from_snapshot(snapshot_path, args.tier, args.max_K)
        if not candidates:
            # Try tier3 fallback
            logger.warning("%s: no routes from tier=%s, trying tier3", benchmark, args.tier)
            candidates = load_sequences_from_snapshot(snapshot_path, "tier3", args.max_K)
        if not candidates:
            logger.warning("%s: no routes from tier3 either, trying auto", benchmark)
            candidates = load_sequences_from_snapshot(snapshot_path, "auto", args.max_K)

        K_eff = min(args.max_K, len(candidates))
        if K_eff < max(K_values):
            if args.skip_benchmarks_without_enough_routes:
                logger.warning(
                    "%s: only %d routes available (need %d); skipping.",
                    benchmark, K_eff, max(K_values),
                )
                continue
            else:
                logger.warning(
                    "%s: only %d routes available (requested %d); K_values will be capped.",
                    benchmark, K_eff, args.max_K,
                )
        routes = [c["layers"] for c in candidates[:K_eff]]
        route_meta = [
            {"label": c.get("label", f"route_{i}"),
             "source_acc": c.get("source_acc"),
             "source_delta": c.get("source_delta")}
            for i, c in enumerate(candidates[:K_eff])
        ]

        logger.info(
            "%s: loaded %d routes (tier=%s), SC temp=%.2f, seeds=%s",
            benchmark, K_eff, args.tier, sc_temp, args.sc_seeds,
        )

        # Load and split data
        eval_data = prepare_arc_data(benchmark, is_instruct, split="validation")
        if args.num_samples > 0:
            eval_data = eval_data[:args.num_samples]

        tune_data, holdout_data = split_tune_holdout(
            eval_data, args.holdout_fraction, args.seed, benchmark
        )
        logger.info(
            "%s: tune=%d, holdout=%d (fraction=%.2f)",
            benchmark, len(tune_data), len(holdout_data), args.holdout_fraction,
        )

        # Collect all responses (the expensive part)
        records = collect_responses(
            wrapper=wrapper,
            benchmark=benchmark,
            model_name=args.model_name,
            holdout_data=holdout_data,
            default_layers=default_layers,
            routes=routes,
            sc_temperature=sc_temp,
            max_K=K_eff,
            sc_seeds=args.sc_seeds,
        )

        # Analyze at each K
        bench_K_values = [k for k in K_values if k <= K_eff]
        analysis = {}
        for K in bench_K_values:
            logger.info("%s: analyzing at K=%d ...", benchmark, K)
            analysis[str(K)] = analyze_at_K(
                records, K, args.sc_seeds, benchmark, args.model_name,
            )

        all_results[benchmark] = {
            "model_name": args.model_name,
            "benchmark": benchmark,
            "sc_temperature": sc_temp,
            "sc_seeds": args.sc_seeds,
            "max_K": K_eff,
            "K_values": bench_K_values,
            "holdout_fraction": args.holdout_fraction,
            "n_tune": len(tune_data),
            "n_holdout": len(holdout_data),
            "snapshot": snapshot_path,
            "tier": args.tier,
            "route_meta": route_meta,
            "analysis": analysis,
            "per_sample": records,
        }

    # Summary
    print_summary(all_results, K_values)

    # Save
    ts = time.strftime("%Y%m%d-%H%M%S")
    mk = model_key_from_name(args.model_name).replace(".", "")
    out_path = os.path.join(out_dir, f"publication_rc_vs_sc_{mk}_K{args.max_K}_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    logger.info("Full results saved to %s", out_path)

    # Also save a lightweight summary (no per-sample data)
    summary = {}
    for bench, br in all_results.items():
        summary[bench] = {k: v for k, v in br.items() if k != "per_sample"}
    summary_path = os.path.join(out_dir, f"publication_rc_vs_sc_{mk}_K{args.max_K}_{ts}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    logger.info("Summary (no per-sample) saved to %s", summary_path)


if __name__ == "__main__":
    main()
