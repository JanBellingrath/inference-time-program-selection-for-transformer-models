#!/usr/bin/env python3
"""
Nature-quality figures for Route Consistency (RC) vs Self-Consistency (SC).

Produces four figures:
  1. Accuracy vs K  (2×3 facet by benchmark)
  2. Gain decomposition: route quality + marginalization  (2×3 facet)
  3. Prediction diversity: pairwise disagreement among routes vs SC samples
  4. Summary heatmap of RC − SC advantage

Usage:
    python plot_nature_rc_vs_sc.py \
        --summary  predictions/publication/publication_rc_vs_sc_05B_K20_*_summary.json \
        --raw      predictions/publication/publication_rc_vs_sc_05B_K20_*.json \
        --output_dir predictions/publication/figures
"""
import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

# ---------------------------------------------------------------------------
# Nature style configuration
# ---------------------------------------------------------------------------

NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

BENCHMARK_DISPLAY = {
    "winogrande": "WinoGrande",
    "boolq": "BoolQ",
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "commonsenseqa": "CSQA",
    "mmlu_all": "MMLU",
}

BENCHMARK_ORDER = [
    "boolq", "arc_easy", "mmlu_all",
    "winogrande", "commonsenseqa", "arc_challenge",
]

PAL_RC = "#2c7bb6"
PAL_SC = "#d7191c"
PAL_BASE = "#636363"
PAL_RTE_QUAL = "#fdae61"
PAL_AGG = "#2c7bb6"
PAL_SC_AGG = "#d7191c"

MM_PER_INCH = 25.4
NATURE_FULL_W = 180 / MM_PER_INCH   # ~7.09 in (double column)
NATURE_SINGLE_W = 89 / MM_PER_INCH  # ~3.50 in (single column)


def _label(ax, letter: str, x=-0.12, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_summary(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def load_raw(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def _benchmarks(data: Dict) -> List[str]:
    return [b for b in BENCHMARK_ORDER if b in data]


# ===================================================================
# FIGURE 1: Accuracy vs K
# ===================================================================

def figure1_accuracy_vs_k(summary: Dict, out_dir: str):
    benchmarks = _benchmarks(summary)
    n_bench = len(benchmarks)
    ncols = 3
    nrows = (n_bench + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(NATURE_FULL_W, 2.2 * nrows),
                             constrained_layout=True)
    axes = axes.flatten()

    for i, bench in enumerate(benchmarks):
        ax = axes[i]
        br = summary[bench]
        analysis = br["analysis"]
        Ks = sorted(int(k) for k in analysis)

        base = analysis[str(Ks[0])]["baseline_acc"]
        rc_vals = [analysis[str(k)]["rc_acc"] for k in Ks]
        sc_means = [analysis[str(k)]["sc_acc_mean"] for k in Ks]
        sc_stds = [analysis[str(k)]["sc_acc_std"] for k in Ks]

        # CIs for RC
        rc_lo = [analysis[str(k)]["ci_rc"][0] for k in Ks]
        rc_hi = [analysis[str(k)]["ci_rc"][1] for k in Ks]

        ax.axhline(base, color=PAL_BASE, ls="--", lw=0.7, zorder=1, label="Baseline")
        ax.fill_between(Ks,
                        [m - s for m, s in zip(sc_means, sc_stds)],
                        [m + s for m, s in zip(sc_means, sc_stds)],
                        color=PAL_SC, alpha=0.12, zorder=2)
        ax.plot(Ks, sc_means, color=PAL_SC, marker="s", ms=3, zorder=3, label="SC")
        ax.fill_between(Ks, rc_lo, rc_hi, color=PAL_RC, alpha=0.10, zorder=4)
        ax.plot(Ks, rc_vals, color=PAL_RC, marker="o", ms=3.5, zorder=5, label="RC")

        ax.set_title(BENCHMARK_DISPLAY.get(bench, bench), fontweight="semibold")
        ax.set_xlabel("K")
        ax.set_ylabel("Accuracy")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))
        _label(ax, chr(ord("a") + i))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    _save(fig, out_dir, "fig1_accuracy_vs_K")
    plt.close(fig)


# ===================================================================
# FIGURE 2: Effect decomposition across K
# ===================================================================

def figure2_decomposition(summary: Dict, out_dir: str):
    benchmarks = _benchmarks(summary)
    n_bench = len(benchmarks)
    ncols = 3
    nrows = (n_bench + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(NATURE_FULL_W, 2.4 * nrows),
                             constrained_layout=True)
    axes = axes.flatten()

    for i, bench in enumerate(benchmarks):
        ax = axes[i]
        br = summary[bench]
        analysis = br["analysis"]
        Ks = sorted(int(k) for k in analysis)

        rq = [analysis[str(k)]["route_quality_effect"] for k in Ks]
        ra = [analysis[str(k)]["rc_aggregation_effect"] for k in Ks]

        rq_lo = [analysis[str(k)]["bootstrap_route_quality"]["ci_lo"] for k in Ks]
        rq_hi = [analysis[str(k)]["bootstrap_route_quality"]["ci_hi"] for k in Ks]
        ra_lo = [analysis[str(k)]["bootstrap_rc_aggregation"]["ci_lo"] for k in Ks]
        ra_hi = [analysis[str(k)]["bootstrap_rc_aggregation"]["ci_hi"] for k in Ks]

        total = [r + a for r, a in zip(rq, ra)]

        x = np.arange(len(Ks))
        w = 0.32
        bars_rq = ax.bar(x - w / 2, rq, w, color=PAL_RTE_QUAL, label="Route quality",
                         zorder=3, edgecolor="white", linewidth=0.3)
        bars_ra = ax.bar(x + w / 2, ra, w, color=PAL_AGG, label="Marginalization",
                         zorder=3, edgecolor="white", linewidth=0.3)

        rq_err = np.array([[r - lo for r, lo in zip(rq, rq_lo)],
                           [hi - r for r, hi in zip(rq, rq_hi)]])
        ra_err = np.array([[r - lo for r, lo in zip(ra, ra_lo)],
                           [hi - r for r, hi in zip(ra, ra_hi)]])
        ax.errorbar(x - w / 2, rq, yerr=rq_err, fmt="none", ecolor="0.3",
                    capsize=1.5, elinewidth=0.5, capthick=0.4, zorder=4)
        ax.errorbar(x + w / 2, ra, yerr=ra_err, fmt="none", ecolor="0.3",
                    capsize=1.5, elinewidth=0.5, capthick=0.4, zorder=4)

        ax.plot(x, total, color="0.2", marker="D", ms=2.5, lw=0.8, ls="--",
                zorder=5, label="Total (RC − Base)")

        ax.axhline(0, color="0.5", lw=0.4, ls="-")
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in Ks])
        ax.set_xlabel("K")
        ax.set_ylabel("Effect (accuracy)")
        ax.set_title(BENCHMARK_DISPLAY.get(bench, bench), fontweight="semibold")
        _label(ax, chr(ord("a") + i))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    _save(fig, out_dir, "fig2_decomposition")
    plt.close(fig)


# ===================================================================
# FIGURE 3: Prediction diversity (why ensembling works)
#   Four-panel figure:
#     a) Oracle coverage across K (ceiling if you could pick the right one)
#     b) Pairwise disagreement across K
#     c) Mean pairwise correlation across K
#     d) Aggregation efficiency: voted_gain / oracle_headroom
# ===================================================================

def _pairwise_disagreement(ok_matrix: np.ndarray) -> float:
    """Mean pairwise disagreement rate among columns of (n_samples, K) binary matrix."""
    K = ok_matrix.shape[1]
    if K < 2:
        return 0.0
    disagree_sum = 0.0
    n_pairs = 0
    for a in range(K):
        for b in range(a + 1, K):
            disagree_sum += np.mean(ok_matrix[:, a] != ok_matrix[:, b])
            n_pairs += 1
    return disagree_sum / n_pairs


def _mean_pairwise_correlation(ok_matrix: np.ndarray) -> float:
    """Mean pairwise Pearson correlation among columns."""
    K = ok_matrix.shape[1]
    if K < 2:
        return 1.0
    corrs = []
    for a in range(K):
        for b in range(a + 1, K):
            std_a = ok_matrix[:, a].std()
            std_b = ok_matrix[:, b].std()
            if std_a == 0 or std_b == 0:
                corrs.append(1.0)
            else:
                corrs.append(np.corrcoef(ok_matrix[:, a], ok_matrix[:, b])[0, 1])
    return float(np.mean(corrs))


def _oracle_coverage(ok_matrix: np.ndarray) -> float:
    """Fraction of samples where at least one column is correct."""
    return float(np.any(ok_matrix, axis=1).mean())


def figure3_diversity(raw: Dict, summary: Dict, out_dir: str):
    """Four-panel diversity analysis: oracle coverage, disagreement,
    correlation, and aggregation efficiency."""
    benchmarks = _benchmarks(raw)

    rows_oracle = []
    rows_disagree = []
    rows_corr = []
    rows_efficiency = []

    for bench in benchmarks:
        br = raw[bench]
        per_sample = br["per_sample"]
        sc_seeds = [str(s) for s in br["sc_seeds"]]
        Ks = sorted(int(k) for k in br["analysis"])
        n = len(per_sample)
        max_K = max(Ks)
        display = BENCHMARK_DISPLAY.get(bench, bench)

        rc_ok_full = np.array([r["rc_ok"][:max_K] for r in per_sample])
        sc_ok_full = {seed: np.array([r["sc_ok_by_seed"][seed][:max_K]
                                      for r in per_sample])
                      for seed in sc_seeds}

        for K in Ks:
            rc_sub = rc_ok_full[:, :K]
            rc_oracle = _oracle_coverage(rc_sub)
            rc_disag = _pairwise_disagreement(rc_sub)
            rc_corr = _mean_pairwise_correlation(rc_sub)
            rc_avg = float(rc_sub.mean())

            sc_oracles, sc_disags, sc_corrs, sc_avgs = [], [], [], []
            for seed in sc_seeds:
                sc_sub = sc_ok_full[seed][:, :K]
                sc_oracles.append(_oracle_coverage(sc_sub))
                sc_disags.append(_pairwise_disagreement(sc_sub))
                sc_corrs.append(_mean_pairwise_correlation(sc_sub))
                sc_avgs.append(float(sc_sub.mean()))

            a = summary[bench]["analysis"][str(K)]
            rc_voted = a["rc_acc"]
            sc_voted = a["sc_acc_mean"]

            rows_oracle.append({"Benchmark": display, "K": K,
                                "Oracle coverage": rc_oracle, "Method": "RC"})
            rows_oracle.append({"Benchmark": display, "K": K,
                                "Oracle coverage": float(np.mean(sc_oracles)),
                                "Method": "SC"})

            rows_disagree.append({"Benchmark": display, "K": K,
                                  "Disagreement": rc_disag, "Method": "RC"})
            rows_disagree.append({"Benchmark": display, "K": K,
                                  "Disagreement": float(np.mean(sc_disags)),
                                  "Method": "SC"})

            rows_corr.append({"Benchmark": display, "K": K,
                              "Correlation": rc_corr, "Method": "RC"})
            rows_corr.append({"Benchmark": display, "K": K,
                              "Correlation": float(np.mean(sc_corrs)),
                              "Method": "SC"})

            # Aggregation efficiency: voted_gain / (oracle - avg_individual)
            rc_headroom = rc_oracle - rc_avg
            rc_gain = rc_voted - rc_avg
            rc_eff = rc_gain / rc_headroom if rc_headroom > 0.01 else np.nan

            sc_headroom = float(np.mean(sc_oracles)) - float(np.mean(sc_avgs))
            sc_gain = sc_voted - float(np.mean(sc_avgs))
            sc_eff = sc_gain / sc_headroom if sc_headroom > 0.01 else np.nan

            rows_efficiency.append({"Benchmark": display, "K": K,
                                    "Efficiency": rc_eff, "Method": "RC"})
            rows_efficiency.append({"Benchmark": display, "K": K,
                                    "Efficiency": sc_eff, "Method": "SC"})

    df_oracle = pd.DataFrame(rows_oracle)
    df_disag = pd.DataFrame(rows_disagree)
    df_corr = pd.DataFrame(rows_corr)
    df_eff = pd.DataFrame(rows_efficiency)

    fig, axes = plt.subplots(2, 2, figsize=(NATURE_FULL_W, 4.8),
                             constrained_layout=True)

    method_palette = {"RC": PAL_RC, "SC": PAL_SC}
    method_markers = {"RC": "o", "SC": "s"}

    # --- (a) Oracle coverage ---
    ax = axes[0, 0]
    for method, grp in df_oracle.groupby("Method"):
        for bench_name, sub in grp.groupby("Benchmark"):
            sub = sub.sort_values("K")
            ax.plot(sub["K"], sub["Oracle coverage"],
                    color=method_palette[method], alpha=0.55, lw=0.7,
                    marker=method_markers[method], ms=2.5)
    ax.plot([], [], color=PAL_RC, marker="o", ms=3, label="RC")
    ax.plot([], [], color=PAL_SC, marker="s", ms=3, label="SC")
    ax.set_xlabel("K")
    ax.set_ylabel("Oracle coverage")
    ax.set_title("At least one correct", fontweight="semibold")
    ax.legend(frameon=False, fontsize=6)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))
    _label(ax, "a")

    # --- (b) Pairwise disagreement ---
    ax = axes[0, 1]
    for method, grp in df_disag.groupby("Method"):
        for bench_name, sub in grp.groupby("Benchmark"):
            sub = sub.sort_values("K")
            ax.plot(sub["K"], sub["Disagreement"],
                    color=method_palette[method], alpha=0.55, lw=0.7,
                    marker=method_markers[method], ms=2.5)
    ax.plot([], [], color=PAL_RC, marker="o", ms=3, label="RC")
    ax.plot([], [], color=PAL_SC, marker="s", ms=3, label="SC")
    ax.set_xlabel("K")
    ax.set_ylabel("Pairwise disagreement")
    ax.set_title("Prediction diversity", fontweight="semibold")
    ax.legend(frameon=False, fontsize=6)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))
    _label(ax, "b")

    # --- (c) Mean pairwise correlation ---
    ax = axes[1, 0]
    for method, grp in df_corr.groupby("Method"):
        for bench_name, sub in grp.groupby("Benchmark"):
            sub = sub.sort_values("K")
            ax.plot(sub["K"], sub["Correlation"],
                    color=method_palette[method], alpha=0.55, lw=0.7,
                    marker=method_markers[method], ms=2.5)
    ax.plot([], [], color=PAL_RC, marker="o", ms=3, label="RC")
    ax.plot([], [], color=PAL_SC, marker="s", ms=3, label="SC")
    ax.set_xlabel("K")
    ax.set_ylabel("Mean pairwise correlation")
    ax.set_title("Prediction redundancy", fontweight="semibold")
    ax.legend(frameon=False, fontsize=6)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))
    _label(ax, "c")

    # --- (d) Correlation vs aggregation gain scatter ---
    ax = axes[1, 1]
    corr_rows = []
    for bench in benchmarks:
        br = raw[bench]
        per_sample = br["per_sample"]
        sc_seeds_local = [str(s) for s in br["sc_seeds"]]
        Ks_local = sorted(int(k) for k in br["analysis"])
        max_K_local = max(Ks_local)
        n_local = len(per_sample)
        display = BENCHMARK_DISPLAY.get(bench, bench)

        rc_ok_f = np.array([r["rc_ok"][:max_K_local] for r in per_sample])
        sc_ok_f = {seed: np.array([r["sc_ok_by_seed"][seed][:max_K_local]
                                   for r in per_sample])
                   for seed in sc_seeds_local}

        for K in Ks_local:
            if K < 3:
                continue
            rc_c = _mean_pairwise_correlation(rc_ok_f[:, :K])
            sc_cs = [_mean_pairwise_correlation(sc_ok_f[s][:, :K])
                     for s in sc_seeds_local]
            a_k = summary[bench]["analysis"][str(K)]
            corr_rows.append({"corr": rc_c, "agg": a_k["rc_aggregation_effect"],
                              "Method": "RC", "Benchmark": display, "K": K})
            corr_rows.append({"corr": float(np.mean(sc_cs)),
                              "agg": a_k["sc_aggregation_effect"],
                              "Method": "SC", "Benchmark": display, "K": K})

    df_scatter = pd.DataFrame(corr_rows)
    for method in ["RC", "SC"]:
        sub = df_scatter[df_scatter["Method"] == method]
        ax.scatter(sub["corr"], sub["agg"],
                   c=method_palette[method], s=18, alpha=0.85,
                   edgecolors="white", linewidths=0.3,
                   marker=method_markers[method], label=method, zorder=3)
    ax.axhline(0, color="0.6", lw=0.4, ls="-")
    ax.set_xlabel("Mean pairwise correlation")
    ax.set_ylabel("Aggregation gain")
    ax.set_title("Lower correlation → larger gain", fontweight="semibold")
    ax.legend(frameon=False, fontsize=6)
    _label(ax, "d")

    _save(fig, out_dir, "fig3_diversity")
    plt.close(fig)


# ===================================================================
# FIGURE 4a: Summary heatmap — RC − SC advantage
# ===================================================================

def figure4_summary(summary: Dict, raw: Dict, out_dir: str):
    """Two-panel summary: (a) heatmap of RC−SC, (b) correlation vs aggregation scatter."""
    benchmarks = _benchmarks(summary)
    all_Ks = set()
    for b in benchmarks:
        all_Ks.update(int(k) for k in summary[b]["analysis"])
    all_Ks = sorted(all_Ks)

    fig, ax = plt.subplots(figsize=(NATURE_SINGLE_W * 1.5, 2.6),
                           constrained_layout=True)

    mat = np.full((len(benchmarks), len(all_Ks)), np.nan)
    for i, b in enumerate(benchmarks):
        for j, k in enumerate(all_Ks):
            a = summary[b]["analysis"].get(str(k))
            if a:
                mat[i, j] = a["rc_acc"] - a["sc_acc_mean"]

    vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = sns.diverging_palette(10, 220, as_cmap=True)

    sns.heatmap(mat, ax=ax, cmap=cmap, norm=norm,
                annot=True, fmt=".3f", annot_kws={"size": 6.5},
                xticklabels=[str(k) for k in all_Ks],
                yticklabels=[BENCHMARK_DISPLAY.get(b, b) for b in benchmarks],
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "RC − SC (accuracy)", "shrink": 0.85,
                           "aspect": 15})
    ax.set_xlabel("K (ensemble size)")
    ax.set_ylabel("")
    ax.set_title("RC advantage over SC", fontweight="semibold", fontsize=8)

    _save(fig, out_dir, "fig4_summary")
    plt.close(fig)


# ===================================================================
# FIGURE 5: Agreement matrix heatmaps (detailed diversity)
# ===================================================================

def figure5_agreement_matrices(raw: Dict, out_dir: str):
    """Agreement matrices for 3 representative benchmarks (RC strongest, SC
    strongest, intermediate): routes vs SC samples side by side."""
    all_benchmarks = _benchmarks(raw)
    # Pick 3 representative: BoolQ (RC wins), ARC-Easy (strong marginalization),
    # WinoGrande (SC wins)
    picks = ["boolq", "arc_easy", "winogrande"]
    benchmarks = [b for b in picks if b in raw]
    if len(benchmarks) < 3:
        benchmarks = all_benchmarks[:3]

    K_show = 10

    fig, axes = plt.subplots(len(benchmarks), 2,
                             figsize=(NATURE_FULL_W, 2.6 * len(benchmarks)),
                             constrained_layout=True)
    if len(benchmarks) == 1:
        axes = axes[np.newaxis, :]

    for row, bench in enumerate(benchmarks):
        K = min(K_show, max(int(k) for k in raw[bench]["analysis"]))
        br = raw[bench]
        per_sample = br["per_sample"]
        sc_seeds = [str(s) for s in br["sc_seeds"]]

        rc_ok = np.array([r["rc_ok"][:K] for r in per_sample])
        sc_ok_seed0 = np.array([r["sc_ok_by_seed"][sc_seeds[0]][:K]
                                for r in per_sample])

        rc_agree = np.zeros((K, K))
        sc_agree = np.zeros((K, K))
        for a in range(K):
            for b_idx in range(K):
                rc_agree[a, b_idx] = np.mean(rc_ok[:, a] == rc_ok[:, b_idx])
                sc_agree[a, b_idx] = np.mean(sc_ok_seed0[:, a] == sc_ok_seed0[:, b_idx])

        vmin = min(rc_agree[np.triu_indices(K, k=1)].min(),
                   sc_agree[np.triu_indices(K, k=1)].min())

        cmap = sns.color_palette("YlGnBu", as_cmap=True)
        kw = dict(cmap=cmap, vmin=vmin, vmax=1.0,
                  annot=True, fmt=".2f", annot_kws={"size": 5.5},
                  square=True, linewidths=0.3, linecolor="white",
                  cbar_kws={"shrink": 0.65, "aspect": 12})

        ax_rc = axes[row, 0]
        ax_sc = axes[row, 1]

        sns.heatmap(rc_agree, ax=ax_rc, **kw)
        ax_rc.set_title(
            f"{BENCHMARK_DISPLAY.get(bench, bench)} — Routes (K={K})",
            fontsize=7, fontweight="semibold")
        ax_rc.set_xlabel("Route index")
        ax_rc.set_ylabel("Route index")

        sns.heatmap(sc_agree, ax=ax_sc, **kw)
        ax_sc.set_title(
            f"{BENCHMARK_DISPLAY.get(bench, bench)} — Temp. samples (K={K})",
            fontsize=7, fontweight="semibold")
        ax_sc.set_xlabel("Sample index")
        ax_sc.set_ylabel("Sample index")

        _label(ax_rc, chr(ord("a") + 2 * row), x=-0.15)
        _label(ax_sc, chr(ord("b") + 2 * row), x=-0.15)

    _save(fig, out_dir, "fig5_agreement_matrices")
    plt.close(fig)


# ===================================================================
# FIGURE 6: Per-route accuracy profiles (declining route quality)
# ===================================================================

def figure6_route_profiles(summary: Dict, out_dir: str):
    """Per-route accuracy at max K, with RC voted accuracy and baseline overlaid.
    Shows that later routes drop in quality but majority vote stays strong."""
    benchmarks = _benchmarks(summary)
    n_bench = len(benchmarks)
    ncols = 3
    nrows = (n_bench + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(NATURE_FULL_W, 2.4 * nrows),
                             constrained_layout=True)
    axes = axes.flatten()

    for i, bench in enumerate(benchmarks):
        ax = axes[i]
        br = summary[bench]
        analysis = br["analysis"]
        max_K = max(int(k) for k in analysis)
        a_max = analysis[str(max_K)]
        per_route = a_max["per_route_acc"]
        base = a_max["baseline_acc"]
        rc_acc = a_max["rc_acc"]
        avg_route = a_max["avg_route_acc"]
        K = len(per_route)
        xs = list(range(1, K + 1))

        ax.bar(xs, per_route, color=PAL_AGG, alpha=0.55, width=0.7,
               edgecolor="white", linewidth=0.3, zorder=2, label="Route accuracy")

        ax.axhline(base, color=PAL_BASE, ls="--", lw=0.7, zorder=3, label="Baseline")
        ax.axhline(avg_route, color=PAL_RTE_QUAL, ls="-.", lw=0.8, zorder=3,
                   label="Avg route")
        ax.axhline(rc_acc, color=PAL_RC, ls="-", lw=1.1, zorder=4,
                   label="RC (voted)")

        all_vals = per_route + [base, avg_route, rc_acc]
        y_lo = min(all_vals) - 0.04
        y_hi = max(all_vals) + 0.04
        ax.set_ylim(y_lo, y_hi)

        ax.set_title(f"{BENCHMARK_DISPLAY.get(bench, bench)} (K={max_K})",
                     fontweight="semibold")
        ax.set_xlabel("Route rank (by MCTS reward)")
        ax.set_ylabel("Accuracy")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))
        _label(ax, chr(ord("a") + i))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, -0.03))

    _save(fig, out_dir, "fig6_route_profiles")
    plt.close(fig)


# ===================================================================
# Save helper
# ===================================================================

def _save(fig, out_dir: str, name: str):
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(out_dir, f"{name}.png"), bbox_inches="tight", dpi=300)
    print(f"  Saved {name}.pdf/.png")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, required=True)
    parser.add_argument("--raw", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or str(
        Path(args.summary).parent / "figures"
    )
    os.makedirs(out_dir, exist_ok=True)

    with plt.rc_context(NATURE_RC):
        sns.set_context("paper")
        sns.set_style("ticks")
        plt.rcParams.update(NATURE_RC)

        print("Loading data ...")
        summary = load_summary(args.summary)
        raw = load_raw(args.raw)

        print("Figure 1: Accuracy vs K ...")
        figure1_accuracy_vs_k(summary, out_dir)

        print("Figure 2: Decomposition ...")
        figure2_decomposition(summary, out_dir)

        print("Figure 3: Prediction diversity ...")
        figure3_diversity(raw, summary, out_dir)

        print("Figure 4: Summary heatmap + scatter ...")
        figure4_summary(summary, raw, out_dir)

        print("Figure 5: Agreement matrices ...")
        figure5_agreement_matrices(raw, out_dir)

        print("Figure 6: Per-route accuracy profiles ...")
        figure6_route_profiles(summary, out_dir)

        print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
