#!/usr/bin/env python3
"""
Analyze aggregation comparison results to disentangle:
  (a) route quality effect  – individual MCTS routes are better than default
  (b) aggregation effect    – majority voting over K routes helps beyond single best

Reads the JSON output files produced by compare_aggregation.py.
"""
import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.benchmark_mcts import grade_response


def grade_extracted(answer: Optional[str], correct: str,
                    dataset: str, model_name: str, input_text: str) -> int:
    """Grade an already-extracted answer. None → 0."""
    if answer is None:
        return 0
    score = grade_response(answer, correct, dataset, model_name, input_text)
    return int(score > 0.5)


def analyze_file(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)

    dataset = d["dataset"]
    model_name = d["model_name"]
    K = d["K"]
    n = d["num_eval_samples"]
    samples = d["per_sample"]

    per_route_correct = [0] * K
    per_sc_correct = [0] * K
    baseline_correct = 0
    rc_voted_correct = 0
    sc_voted_correct = 0

    for s in samples:
        correct = s["correct_answer"]
        input_text = ""

        baseline_correct += s["baseline_ok"]
        rc_voted_correct += s["rc_ok"]
        sc_voted_correct += s["sc_ok"]

        for k in range(K):
            if k < len(s["rc_answers"]):
                per_route_correct[k] += grade_extracted(
                    s["rc_answers"][k], correct, dataset, model_name, input_text)
            if k < len(s["sc_answers"]):
                per_sc_correct[k] += grade_extracted(
                    s["sc_answers"][k], correct, dataset, model_name, input_text)

    per_route_acc = [c / n for c in per_route_correct]
    per_sc_acc = [c / n for c in per_sc_correct]
    avg_route_acc = sum(per_route_acc) / K
    best_route_acc = max(per_route_acc)
    avg_sc_acc = sum(per_sc_acc) / K
    baseline_acc = baseline_correct / n
    rc_voted_acc = rc_voted_correct / n
    sc_voted_acc = sc_voted_correct / n

    return {
        "dataset": dataset,
        "n": n,
        "K": K,
        "baseline_acc": baseline_acc,
        "per_route_acc": per_route_acc,
        "avg_route_acc": avg_route_acc,
        "best_route_acc": best_route_acc,
        "rc_voted_acc": rc_voted_acc,
        "per_sc_acc": per_sc_acc,
        "avg_sc_acc": avg_sc_acc,
        "sc_voted_acc": sc_voted_acc,
        "route_quality_effect": avg_route_acc - baseline_acc,
        "rc_aggregation_effect": rc_voted_acc - avg_route_acc,
        "rc_vs_best_single": rc_voted_acc - best_route_acc,
        "sc_aggregation_effect": sc_voted_acc - avg_sc_acc,
    }


def print_results(results: List[dict]):
    print()
    print("=" * 90)
    print("DISENTANGLING ROUTE QUALITY vs AGGREGATION EFFECT")
    print("=" * 90)

    for r in results:
        ds = r["dataset"]
        K = r["K"]
        print(f"\n--- {ds} (n={r['n']}, K={K}) ---")
        print()

        print("  Route consistency breakdown:")
        for k, acc in enumerate(r["per_route_acc"]):
            print(f"    Route {k+1} alone (greedy):     {acc:.4f}  "
                  f"delta vs baseline: {acc - r['baseline_acc']:+.4f}")
        print(f"    Avg single route:             {r['avg_route_acc']:.4f}  "
              f"delta vs baseline: {r['route_quality_effect']:+.4f}")
        print(f"    Best single route:            {r['best_route_acc']:.4f}  "
              f"delta vs baseline: {r['best_route_acc'] - r['baseline_acc']:+.4f}")
        print(f"    Voted (majority, K={K}):       {r['rc_voted_acc']:.4f}  "
              f"delta vs baseline: {r['rc_voted_acc'] - r['baseline_acc']:+.4f}")
        print()
        print(f"    Route quality effect (avg_route - baseline):  {r['route_quality_effect']:+.4f}")
        print(f"    Aggregation effect  (voted - avg_route):      {r['rc_aggregation_effect']:+.4f}")
        print(f"    Voted vs best single route:                   {r['rc_vs_best_single']:+.4f}")

        print()
        print("  Self-consistency breakdown:")
        for k, acc in enumerate(r["per_sc_acc"]):
            print(f"    SC sample {k+1} alone:           {acc:.4f}  "
                  f"delta vs baseline: {acc - r['baseline_acc']:+.4f}")
        print(f"    Avg single SC sample:         {r['avg_sc_acc']:.4f}  "
              f"delta vs baseline: {r['avg_sc_acc'] - r['baseline_acc']:+.4f}")
        print(f"    Voted (majority, K={K}):       {r['sc_voted_acc']:.4f}  "
              f"delta vs baseline: {r['sc_voted_acc'] - r['baseline_acc']:+.4f}")
        print(f"    SC aggregation effect (voted - avg_sample):   {r['sc_aggregation_effect']:+.4f}")

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    header = (f"{'Dataset':<16} {'Baseline':>8} {'AvgRoute':>8} {'BestRte':>8} "
              f"{'RC Vote':>8} {'RteQual':>8} {'RCAggr':>8} {'AvgSC':>8} "
              f"{'SC Vote':>8} {'SCAggr':>8}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['dataset']:<16} "
              f"{r['baseline_acc']:>8.4f} "
              f"{r['avg_route_acc']:>8.4f} "
              f"{r['best_route_acc']:>8.4f} "
              f"{r['rc_voted_acc']:>8.4f} "
              f"{r['route_quality_effect']:>+8.4f} "
              f"{r['rc_aggregation_effect']:>+8.4f} "
              f"{r['avg_sc_acc']:>8.4f} "
              f"{r['sc_voted_acc']:>8.4f} "
              f"{r['sc_aggregation_effect']:>+8.4f}")
    print()
    print("  RteQual = AvgRoute - Baseline  (are MCTS routes individually better?)")
    print("  RCAggr  = RC Vote - AvgRoute   (does voting over routes help further?)")
    print("  SCAggr  = SC Vote - AvgSC      (does voting over temp samples help?)")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="predictions",
                        help="Directory containing aggregation_compare_*.json files")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Filter to specific datasets")
    args = parser.parse_args()

    pattern = os.path.join(args.results_dir, "aggregation_compare_*_K*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No result files found matching {pattern}")
        sys.exit(1)

    seen = {}
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        ds = d["dataset"]
        if args.datasets and ds not in args.datasets:
            continue
        seen[ds] = f

    results = []
    for ds in ["winogrande", "boolq", "arc_easy", "commonsenseqa", "mmlu_all"]:
        if ds in seen:
            results.append(analyze_file(seen[ds]))

    for ds in sorted(seen.keys()):
        if ds not in ["winogrande", "boolq", "arc_easy", "commonsenseqa", "mmlu_all"]:
            results.append(analyze_file(seen[ds]))

    print_results(results)


if __name__ == "__main__":
    main()
