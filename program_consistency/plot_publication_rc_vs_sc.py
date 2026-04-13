#!/usr/bin/env python3
"""
Generate publication figures and LaTeX tables from RC vs SC experiment results.

Reads the JSON output from run_publication_rc_vs_sc.py and produces:
  1. Accuracy vs K curves (RC, SC, baseline) per benchmark
  2. Decomposition bar charts (route quality vs aggregation effects)
  3. LaTeX tables with CIs and significance markers
  4. Per-route accuracy profile

Usage:
    python plot_publication_rc_vs_sc.py \
        --results predictions/publication/publication_rc_vs_sc_05B_K20_*.json \
        --results_7b predictions/publication/publication_rc_vs_sc_7B_K10_*.json \
        --output_dir predictions/publication/figures
"""
import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

SCRIPT_DIR = Path(__file__).resolve().parent

BENCHMARK_DISPLAY = {
    "winogrande": "WinoGrande",
    "boolq": "BoolQ",
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "commonsenseqa": "CSQA",
    "mmlu_all": "MMLU",
}

BENCHMARK_ORDER = ["winogrande", "boolq", "arc_easy", "arc_challenge", "commonsenseqa", "mmlu_all"]


def load_results(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1: Accuracy vs K
# ---------------------------------------------------------------------------

def plot_accuracy_vs_k(results: Dict, output_dir: str, model_label: str = ""):
    benchmarks = [b for b in BENCHMARK_ORDER if b in results]
    n_bench = len(benchmarks)
    ncols = min(3, n_bench)
    nrows = (n_bench + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(f"RC vs SC: Accuracy vs K{' (' + model_label + ')' if model_label else ''}",
                 fontsize=14, fontweight="bold", y=1.02)

    for i, benchmark in enumerate(benchmarks):
        ax = axes[i // ncols][i % ncols]
        br = results[benchmark]
        analysis = br["analysis"]

        K_values = sorted(int(k) for k in analysis.keys())
        base_acc = analysis[str(K_values[0])]["baseline_acc"]

        rc_accs = [analysis[str(k)]["rc_acc"] for k in K_values]
        sc_means = [analysis[str(k)]["sc_acc_mean"] for k in K_values]
        sc_stds = [analysis[str(k)]["sc_acc_std"] for k in K_values]

        sc_means_arr = np.array(sc_means)
        sc_stds_arr = np.array(sc_stds)

        ax.axhline(base_acc, color="gray", linestyle="--", alpha=0.7, label="Baseline")
        ax.plot(K_values, rc_accs, "o-", color="#2196F3", linewidth=2,
                markersize=5, label="RC (routes)")
        ax.plot(K_values, sc_means, "s-", color="#FF9800", linewidth=2,
                markersize=5, label="SC (temp)")
        ax.fill_between(K_values,
                         sc_means_arr - sc_stds_arr,
                         sc_means_arr + sc_stds_arr,
                         alpha=0.2, color="#FF9800")

        # Significance markers
        for k in K_values:
            a = analysis[str(k)]
            p = a["mcnemar_rc_vs_sc"]["p_value"]
            if p < 0.05:
                y_max = max(a["rc_acc"], a["sc_acc_mean"])
                marker = "**" if p < 0.01 else "*"
                ax.annotate(marker, (k, y_max + 0.005), ha="center", fontsize=10)

        display = BENCHMARK_DISPLAY.get(benchmark, benchmark)
        ax.set_title(display, fontsize=12)
        ax.set_xlabel("K (number of passes)")
        ax.set_ylabel("Accuracy")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(n_bench, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    plt.tight_layout()
    fname = f"accuracy_vs_K{'_' + model_label if model_label else ''}.pdf"
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Also save PNG
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig2.suptitle(f"RC vs SC: Accuracy vs K{' (' + model_label + ')' if model_label else ''}",
                  fontsize=14, fontweight="bold", y=1.02)
    for i, benchmark in enumerate(benchmarks):
        ax = axes2[i // ncols][i % ncols]
        br = results[benchmark]
        analysis = br["analysis"]
        K_values = sorted(int(k) for k in analysis.keys())
        base_acc = analysis[str(K_values[0])]["baseline_acc"]
        rc_accs = [analysis[str(k)]["rc_acc"] for k in K_values]
        sc_means = [analysis[str(k)]["sc_acc_mean"] for k in K_values]
        sc_stds = [analysis[str(k)]["sc_acc_std"] for k in K_values]
        sc_means_arr = np.array(sc_means)
        sc_stds_arr = np.array(sc_stds)
        ax.axhline(base_acc, color="gray", linestyle="--", alpha=0.7, label="Baseline")
        ax.plot(K_values, rc_accs, "o-", color="#2196F3", linewidth=2, markersize=5, label="RC (routes)")
        ax.plot(K_values, sc_means, "s-", color="#FF9800", linewidth=2, markersize=5, label="SC (temp)")
        ax.fill_between(K_values, sc_means_arr - sc_stds_arr, sc_means_arr + sc_stds_arr, alpha=0.2, color="#FF9800")
        display = BENCHMARK_DISPLAY.get(benchmark, benchmark)
        ax.set_title(display, fontsize=12)
        ax.set_xlabel("K")
        ax.set_ylabel("Accuracy")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
    for j in range(n_bench, nrows * ncols):
        axes2[j // ncols][j % ncols].set_visible(False)
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, fname.replace(".pdf", ".png")),
                 bbox_inches="tight", dpi=150)
    plt.close(fig2)


# ---------------------------------------------------------------------------
# Figure 2: Decomposition bar chart
# ---------------------------------------------------------------------------

def plot_decomposition(results: Dict, output_dir: str, K: int = 5, model_label: str = ""):
    benchmarks = [b for b in BENCHMARK_ORDER if b in results and str(K) in results[b]["analysis"]]
    n = len(benchmarks)
    if n == 0:
        print(f"  No benchmarks have K={K} analysis; skipping decomposition plot.")
        return

    labels = [BENCHMARK_DISPLAY.get(b, b) for b in benchmarks]
    route_qual = []
    rc_agg = []
    sc_agg = []
    rq_ci = []
    rc_ci = []
    sc_ci = []

    for b in benchmarks:
        a = results[b]["analysis"][str(K)]
        route_qual.append(a["route_quality_effect"])
        rc_agg.append(a["rc_aggregation_effect"])
        sc_agg.append(a["sc_aggregation_effect"])
        brq = a["bootstrap_route_quality"]
        brc = a["bootstrap_rc_aggregation"]
        bsc = a["bootstrap_sc_aggregation"]
        rq_ci.append((brq["mean"] - brq["ci_lo"], brq["ci_hi"] - brq["mean"]))
        rc_ci.append((brc["mean"] - brc["ci_lo"], brc["ci_hi"] - brc["mean"]))
        sc_ci.append((bsc["mean"] - bsc["ci_lo"], bsc["ci_hi"] - bsc["mean"]))

    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, n * 1.8), 5))
    rq_err = np.array(rq_ci).T
    rc_err = np.array(rc_ci).T
    sc_err = np.array(sc_ci).T

    ax.bar(x - width, route_qual, width, yerr=rq_err, capsize=3,
           label="Route quality (AvgRoute − Base)", color="#4CAF50", alpha=0.85)
    ax.bar(x, rc_agg, width, yerr=rc_err, capsize=3,
           label="RC aggregation (RC vote − AvgRoute)", color="#2196F3", alpha=0.85)
    ax.bar(x + width, sc_agg, width, yerr=sc_err, capsize=3,
           label="SC aggregation (SC vote − AvgSC)", color="#FF9800", alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Effect (accuracy delta)", fontsize=11)
    title = f"Decomposition at K={K}"
    if model_label:
        title += f" ({model_label})"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fname = f"decomposition_K{K}{'_' + model_label if model_label else ''}.pdf"
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Per-route accuracy (diminishing returns)
# ---------------------------------------------------------------------------

def plot_per_route_accuracy(results: Dict, output_dir: str, K: int = 10, model_label: str = ""):
    benchmarks = [b for b in BENCHMARK_ORDER if b in results and str(K) in results[b]["analysis"]]
    if not benchmarks:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for b in benchmarks:
        a = results[b]["analysis"][str(K)]
        per_route = a["per_route_acc"]
        display = BENCHMARK_DISPLAY.get(b, b)
        ax.plot(range(1, len(per_route) + 1), per_route, "o-", markersize=4, label=display)
        ax.axhline(a["baseline_acc"], color="gray", linestyle=":", alpha=0.3)

    ax.set_xlabel("Route rank (MCTS order)", fontsize=11)
    ax.set_ylabel("Individual route accuracy", fontsize=11)
    title = f"Per-route accuracy (top {K} routes)"
    if model_label:
        title += f" ({model_label})"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    fname = f"per_route_accuracy_K{K}{'_' + model_label if model_label else ''}.pdf"
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def generate_latex_main_table(results: Dict, K: int = 5, model_label: str = "") -> str:
    """Table 1: main accuracy comparison at fixed K."""
    benchmarks = [b for b in BENCHMARK_ORDER if b in results and str(K) in results[b]["analysis"]]
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    cap = f"RC vs SC comparison at $K={K}$"
    if model_label:
        cap += f" ({model_label})"
    lines.append(r"\caption{" + cap + "}")
    lines.append(r"\label{tab:rc_vs_sc_" + str(K) + "}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Benchmark & $n$ & Baseline & SC($K$) & RC($K$) & RC$-$SC & $p$ \\")
    lines.append(r"\midrule")

    for b in benchmarks:
        a = results[b]["analysis"][str(K)]
        display = BENCHMARK_DISPLAY.get(b, b)
        diff = a["rc_acc"] - a["sc_acc_mean"]
        boot = a["paired_bootstrap_rc_minus_sc"]
        p = a["mcnemar_rc_vs_sc"]["p_value"]
        sig = ""
        if p < 0.01:
            sig = r"$^{**}$"
        elif p < 0.05:
            sig = r"$^{*}$"

        # Bold the winner
        rc_str = f"{a['rc_acc']:.3f}"
        sc_str = f"{a['sc_acc_mean']:.3f}"
        if diff > 0 and p < 0.05:
            rc_str = r"\textbf{" + rc_str + "}"
        elif diff < 0 and p < 0.05:
            sc_str = r"\textbf{" + sc_str + "}"

        p_str = f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"
        lines.append(
            f"  {display} & {a['n']} & {a['baseline_acc']:.3f} & "
            f"{sc_str} & {rc_str} & "
            f"${diff:+.3f}${sig} & {p_str} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_latex_decomposition_table(results: Dict, K: int = 5, model_label: str = "") -> str:
    """Table 2: decomposition of effects."""
    benchmarks = [b for b in BENCHMARK_ORDER if b in results and str(K) in results[b]["analysis"]]
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    cap = f"Effect decomposition at $K={K}$"
    if model_label:
        cap += f" ({model_label})"
    lines.append(r"\caption{" + cap + "}")
    lines.append(r"\label{tab:decomposition_" + str(K) + "}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Benchmark & Route Quality & RC Aggregation & SC Aggregation \\")
    lines.append(r"\midrule")

    for b in benchmarks:
        a = results[b]["analysis"][str(K)]
        display = BENCHMARK_DISPLAY.get(b, b)
        brq = a["bootstrap_route_quality"]
        brc = a["bootstrap_rc_aggregation"]
        bsc = a["bootstrap_sc_aggregation"]

        def fmt_ci(d):
            return f"${d['mean']:+.3f}$ [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]"

        lines.append(
            f"  {display} & {fmt_ci(brq)} & {fmt_ci(brc)} & {fmt_ci(bsc)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate publication figures from RC vs SC results.")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to publication_rc_vs_sc_*_summary.json or full JSON.")
    parser.add_argument("--results_7b", type=str, default=None,
                        help="Path to 7B results (optional, for combined figures).")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--K_decomposition", nargs="+", type=int, default=[5, 10],
                        help="K values for decomposition bar charts.")
    parser.add_argument("--K_per_route", type=int, default=10,
                        help="K for per-route accuracy plot.")
    args = parser.parse_args()

    if not HAS_MPL:
        print("ERROR: matplotlib is required. Install with: pip install matplotlib")
        sys.exit(1)

    out_dir = args.output_dir or str(SCRIPT_DIR / "predictions" / "publication" / "figures")
    os.makedirs(out_dir, exist_ok=True)

    results = load_results(args.results)
    model_label = ""
    first_bench = next(iter(results.values()))
    if "model_name" in first_bench:
        name = first_bench["model_name"]
        if "0.5B" in name or "0.5b" in name:
            model_label = "0.5B"
        elif "7B" in name or "7b" in name:
            model_label = "7B"

    print(f"Generating figures for {model_label or 'unknown model'}...")
    print(f"  Benchmarks: {list(results.keys())}")

    # Figure 1
    plot_accuracy_vs_k(results, out_dir, model_label)

    # Figure 2
    for K in args.K_decomposition:
        plot_decomposition(results, out_dir, K=K, model_label=model_label)

    # Figure 3
    plot_per_route_accuracy(results, out_dir, K=args.K_per_route, model_label=model_label)

    # LaTeX tables
    for K in args.K_decomposition:
        table1 = generate_latex_main_table(results, K=K, model_label=model_label)
        table2 = generate_latex_decomposition_table(results, K=K, model_label=model_label)
        tex_path = os.path.join(out_dir, f"tables_K{K}_{model_label or 'model'}.tex")
        with open(tex_path, "w") as f:
            f.write("% Auto-generated by plot_publication_rc_vs_sc.py\n\n")
            f.write(table1)
            f.write("\n\n")
            f.write(table2)
        print(f"  Saved: {tex_path}")

    # Combined 0.5B + 7B figures if both provided
    if args.results_7b:
        results_7b = load_results(args.results_7b)
        print("\nGenerating 7B figures...")
        plot_accuracy_vs_k(results_7b, out_dir, "7B")
        for K in args.K_decomposition:
            plot_decomposition(results_7b, out_dir, K=K, model_label="7B")
            table1 = generate_latex_main_table(results_7b, K=K, model_label="7B")
            table2 = generate_latex_decomposition_table(results_7b, K=K, model_label="7B")
            tex_path = os.path.join(out_dir, f"tables_K{K}_7B.tex")
            with open(tex_path, "w") as f:
                f.write("% Auto-generated by plot_publication_rc_vs_sc.py\n\n")
                f.write(table1)
                f.write("\n\n")
                f.write(table2)
            print(f"  Saved: {tex_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
