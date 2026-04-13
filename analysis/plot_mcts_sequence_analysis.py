#!/usr/bin/env python3
"""Comprehensive cross-benchmark MCTS sequence analysis & statistics.

Nature-style publication figures analysing:
  (a) Cross-benchmark overlap of beneficial module sequences (Jaccard, Dice,
      overlap coefficient) — testing whether routing converges on generic types.
  (b) Distribution over how many questions each module sequence helps, per
      benchmark.
  (c) Per-question statistics: how many sequences help a given question,
      entropy of the router-target distribution, effective number of routes.
  (d) Sequence concentration: cumulative coverage curves, Gini coefficients,
      top-K coverage.
  (e) Position-level mutation frequency heatmap across benchmarks —
      which layer positions are edited and how uniformly across tasks.
  (f) Summary statistics table embedded as a figure panel.

Reads from:
  - fine_routing_data_{benchmark}_mcts/{benchmark}.jsonl
  - predictions/{model}/benchmark_mcts_{benchmark}_*.json  (optional, for
    benchmark-level validated sequences)

Usage:
    python analysis/plot_mcts_sequence_analysis.py \
        [--data_dirs ...] [--output_dir figures/mcts_sequence_analysis]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# ═══════════════════════════════════════════════════════════════════════════
#  Nature-style theme (consistent with plot_fine_routing_results.py)
# ═══════════════════════════════════════════════════════════════════════════

NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.3,
    "legend.frameon": False,
    "legend.handlelength": 1.2,
    "legend.handletextpad": 0.4,
}

MM = 1 / 25.4
SINGLE_COL = 89 * MM
DOUBLE_COL = 183 * MM

BENCH_LABELS = {
    "boolq": "BoolQ",
    "commonsenseqa": "CSQA",
    "winogrande": "WinoGrande",
    "arc_easy": "ARC-E",
    "arc_challenge": "ARC-C",
    "mmlu_all": "MMLU",
}


def _label(bench: str) -> str:
    return BENCH_LABELS.get(bench, bench)


def apply_theme() -> None:
    sns.set_theme(style="ticks", rc=NATURE_RC)
    plt.rcParams.update(NATURE_RC)


def save_fig(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        out = f"{path}.{ext}"
        fig.savefig(out)
        logger.info("Saved %s", out)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_benchmark_data(data_dirs: Dict[str, str]) -> Dict[str, List[Dict]]:
    """Load JSONL records per benchmark.  Returns {bench_name: [records]}."""
    data = {}
    for bench, ddir in data_dirs.items():
        jsonl_path = os.path.join(ddir, f"{bench}.jsonl")
        if not os.path.isfile(jsonl_path):
            logger.warning("Missing %s — skipping", jsonl_path)
            continue
        recs = load_jsonl(jsonl_path)
        if recs:
            data[bench] = recs
            logger.info("  %s: %d records", bench, len(recs))
    return data


# ═══════════════════════════════════════════════════════════════════════════
#  Extraction helpers
# ═══════════════════════════════════════════════════════════════════════════

def _seq_key(seq: list) -> tuple:
    return tuple(int(x) for x in seq)


def extract_helpful_sequences(
    records: List[Dict], delta_threshold: float = 0.0
) -> Dict[tuple, int]:
    """For each unique sequence that helps at least one question (delta > threshold),
    count the number of questions it helps."""
    seq_help_count: Dict[tuple, int] = Counter()
    for rec in records:
        anchor_score = rec["anchor_score"]
        for ex in rec.get("explored", []):
            delta = ex.get("delta", ex["score"] - anchor_score)
            if delta > delta_threshold:
                seq_help_count[_seq_key(ex["seq"])] += 1
    return seq_help_count


def extract_best_sequences(records: List[Dict]) -> Counter:
    """Count how often each sequence is the best for a question."""
    return Counter(_seq_key(r["best_seq"]) for r in records if r.get("gate_label", 0) == 1)


def per_question_helpful_count(
    records: List[Dict], delta_threshold: float = 0.0
) -> List[int]:
    """For each question, how many distinct sequences help it."""
    counts = []
    for rec in records:
        n = 0
        for ex in rec.get("explored", []):
            if ex.get("delta", 0) > delta_threshold:
                n += 1
        counts.append(n)
    return counts


def per_question_entropy(records: List[Dict]) -> List[float]:
    """Shannon entropy of each question's router_target distribution (nats)."""
    entropies = []
    for rec in records:
        rt = np.array(rec["router_target"], dtype=np.float64)
        rt = rt[rt > 0]
        if len(rt) <= 1:
            entropies.append(0.0)
        else:
            entropies.append(-np.sum(rt * np.log(rt)))
    return entropies


def per_question_effective_n(records: List[Dict]) -> List[float]:
    """Effective number of routes: exp(H) where H is Shannon entropy."""
    return [np.exp(h) for h in per_question_entropy(records)]


def gini_coefficient(counts: np.ndarray) -> float:
    """Gini coefficient of a distribution (0 = uniform, 1 = maximally concentrated)."""
    counts = np.sort(counts.astype(float))
    n = len(counts)
    if n == 0 or counts.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * counts) - (n + 1) * np.sum(counts)) / (n * np.sum(counts))


def position_mutation_profile(records: List[Dict]) -> np.ndarray:
    """Count per-position mutations across all explored sequences.

    Returns a 1-D array of length L (sequence length) where entry i counts
    how many times position i differed from the anchor across all explored
    sequences across all questions.
    """
    if not records:
        return np.array([])
    anchor = records[0]["anchor_sequence"]
    L = len(anchor)
    counts = np.zeros(L, dtype=int)
    for rec in records:
        anc = rec["anchor_sequence"]
        for ex in rec.get("explored", []):
            seq = ex["seq"]
            for i in range(min(L, len(seq))):
                if seq[i] != anc[i]:
                    counts[i] += 1
    return counts


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 1 — Cross-benchmark sequence overlap
# ═══════════════════════════════════════════════════════════════════════════

def fig_cross_benchmark_overlap(
    data: Dict[str, List[Dict]], output_dir: str
) -> str:
    """Heatmap of Jaccard / overlap coefficient between benchmarks' helpful
    sequence sets, plus a bar showing absolute set sizes."""

    bench_names = sorted(data.keys())
    n = len(bench_names)
    if n < 2:
        logger.warning("Need >= 2 benchmarks for overlap; skipping fig1")
        return ""

    helpful_sets = {
        b: set(extract_helpful_sequences(data[b]).keys()) for b in bench_names
    }
    best_sets = {
        b: set(extract_best_sequences(data[b]).keys()) for b in bench_names
    }

    jaccard = np.zeros((n, n))
    overlap_coeff = np.zeros((n, n))
    dice = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            si, sj = helpful_sets[bench_names[i]], helpful_sets[bench_names[j]]
            inter = len(si & sj)
            union = len(si | sj)
            jaccard[i, j] = inter / union if union else 0
            overlap_coeff[i, j] = inter / min(len(si), len(sj)) if min(len(si), len(sj)) else 0
            dice[i, j] = 2 * inter / (len(si) + len(sj)) if (len(si) + len(sj)) else 0

    jaccard_best = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            si, sj = best_sets[bench_names[i]], best_sets[bench_names[j]]
            inter = len(si & sj)
            union = len(si | sj)
            jaccard_best[i, j] = inter / union if union else 0

    labels = [_label(b) for b in bench_names]
    cmap = "viridis"

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.45))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.8], wspace=0.35)

    # (a) Jaccard — all helpful sequences
    ax0 = fig.add_subplot(gs[0])
    mask_diag = np.eye(n, dtype=bool)
    sns.heatmap(
        jaccard, ax=ax0, annot=True, fmt=".3f", cmap=cmap,
        xticklabels=labels, yticklabels=labels,
        vmin=0, vmax=max(0.15, np.max(jaccard[~mask_diag]) * 1.2),
        linewidths=0.3, linecolor="white",
        cbar_kws={"shrink": 0.7, "label": "Jaccard"},
        annot_kws={"size": 5.5},
    )
    ax0.set_title("a  Jaccard (all helpful seqs)", fontsize=8, fontweight="bold", loc="left")

    # (b) Jaccard — best sequences only
    ax1 = fig.add_subplot(gs[1])
    sns.heatmap(
        jaccard_best, ax=ax1, annot=True, fmt=".3f", cmap=cmap,
        xticklabels=labels, yticklabels=labels,
        vmin=0, vmax=max(0.15, np.max(jaccard_best[~mask_diag]) * 1.2),
        linewidths=0.3, linecolor="white",
        cbar_kws={"shrink": 0.7, "label": "Jaccard"},
        annot_kws={"size": 5.5},
    )
    ax1.set_title("b  Jaccard (best-seq only)", fontsize=8, fontweight="bold", loc="left")

    # (c) Set sizes + pairwise intersections summary
    ax2 = fig.add_subplot(gs[2])
    sizes_helpful = [len(helpful_sets[b]) for b in bench_names]
    sizes_best = [len(best_sets[b]) for b in bench_names]
    y_pos = np.arange(n)
    bars1 = ax2.barh(y_pos - 0.15, sizes_helpful, height=0.28,
                     color=plt.cm.viridis(0.3), label="All helpful")
    bars2 = ax2.barh(y_pos + 0.15, sizes_best, height=0.28,
                     color=plt.cm.viridis(0.7), label="Best-seq")
    for bar, v in zip(bars1, sizes_helpful):
        ax2.text(bar.get_width() + max(sizes_helpful) * 0.02,
                 bar.get_y() + bar.get_height() / 2,
                 str(v), va="center", fontsize=5)
    for bar, v in zip(bars2, sizes_best):
        ax2.text(bar.get_width() + max(sizes_helpful) * 0.02,
                 bar.get_y() + bar.get_height() / 2,
                 str(v), va="center", fontsize=5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Unique sequences")
    ax2.legend(fontsize=5, loc="lower right")
    ax2.set_title("c  Set sizes", fontsize=8, fontweight="bold", loc="left")
    sns.despine(ax=ax2)

    path = os.path.join(output_dir, "fig1_cross_benchmark_overlap")
    save_fig(fig, path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 2 — Sequence usage distributions + concentration
# ═══════════════════════════════════════════════════════════════════════════

def fig_sequence_distributions(
    data: Dict[str, List[Dict]], output_dir: str
) -> str:
    """(a) How many questions each sequence helps (distribution),
    (b) Cumulative coverage curve (Lorenz-like),
    (c) Top-K coverage."""

    bench_names = sorted(data.keys())
    viridis = plt.cm.viridis
    bench_colours = {b: viridis(i / max(len(bench_names) - 1, 1))
                     for i, b in enumerate(bench_names)}

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, SINGLE_COL * 0.72),
                             gridspec_kw={"width_ratios": [1.2, 1, 1]})

    # (a) Distribution of per-sequence help counts (log scale)
    ax = axes[0]
    for b in bench_names:
        counts = np.array(list(extract_helpful_sequences(data[b]).values()))
        if len(counts) == 0:
            continue
        bins = np.logspace(0, np.log10(max(counts.max(), 2)), 40)
        ax.hist(counts, bins=bins, color=bench_colours[b], alpha=0.45,
                label=f"{_label(b)} (n={len(counts)})", density=True)
        sns.kdeplot(counts, ax=ax, color=bench_colours[b], linewidth=0.8,
                    clip=(1, counts.max()), bw_adjust=0.6, log_scale=True)
    ax.set_xscale("log")
    ax.set_xlabel("Questions helped per sequence")
    ax.set_ylabel("Density")
    ax.legend(fontsize=4.5, loc="upper right")
    ax.set_title("a  Sequence help distribution", fontsize=8, fontweight="bold", loc="left")
    sns.despine(ax=ax)

    # (b) Lorenz / cumulative coverage curve
    ax = axes[1]
    gini_vals = {}
    for b in bench_names:
        counts = np.array(sorted(extract_helpful_sequences(data[b]).values(), reverse=True),
                          dtype=float)
        if len(counts) == 0:
            continue
        cumulative = np.cumsum(counts) / counts.sum()
        x_frac = np.arange(1, len(counts) + 1) / len(counts)
        ax.plot(x_frac, cumulative, color=bench_colours[b], linewidth=0.9,
                label=_label(b))
        gini_vals[b] = gini_coefficient(counts)
    ax.plot([0, 1], [0, 1], color="grey", linewidth=0.4, linestyle=":")
    ax.set_xlabel("Fraction of sequences (ranked)")
    ax.set_ylabel("Cumulative share of total help")
    ax.legend(fontsize=5, loc="lower right")
    ax.set_title("b  Concentration (Lorenz)", fontsize=8, fontweight="bold", loc="left")
    sns.despine(ax=ax)

    # (c) Top-K coverage: what fraction of all gate+ questions are covered
    ax = axes[2]
    k_values = [1, 3, 5, 10, 20, 50, 100]
    for b in bench_names:
        n_gate_pos = sum(1 for r in data[b] if r.get("gate_label", 0) == 1)
        if n_gate_pos == 0:
            continue
        best_counts = extract_best_sequences(data[b])
        sorted_counts = sorted(best_counts.values(), reverse=True)
        coverage = []
        for k in k_values:
            top_k_total = sum(sorted_counts[:k])
            coverage.append(top_k_total / n_gate_pos * 100)
        ax.plot(k_values, coverage, marker="o", markersize=2.5,
                color=bench_colours[b], linewidth=0.9, label=_label(b))

    ax.set_xlabel("Top-K sequences")
    ax.set_ylabel("Coverage of gate+ questions (%)")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
    ax.legend(fontsize=5, loc="lower right")
    ax.set_title("c  Top-K coverage", fontsize=8, fontweight="bold", loc="left")
    sns.despine(ax=ax)

    fig.tight_layout(w_pad=2.5)
    path = os.path.join(output_dir, "fig2_sequence_distributions")
    save_fig(fig, path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 3 — Per-question statistics
# ═══════════════════════════════════════════════════════════════════════════

def fig_per_question_stats(
    data: Dict[str, List[Dict]], output_dir: str
) -> str:
    """(a) How many sequences help each question,
    (b) Entropy of router target distribution,
    (c) Effective number of routes (exp(H)),
    (d) Entropy vs best_delta scatter."""

    bench_names = sorted(data.keys())
    viridis = plt.cm.viridis
    bench_colours = {b: viridis(i / max(len(bench_names) - 1, 1))
                     for i, b in enumerate(bench_names)}

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.55))

    # (a) Violin: how many sequences help per question
    ax = axes[0, 0]
    violin_data = []
    for b in bench_names:
        counts = per_question_helpful_count(data[b])
        for c in counts:
            violin_data.append({"Benchmark": _label(b), "Helpful seqs": c})
    vdf = pd.DataFrame(violin_data)
    if not vdf.empty:
        sns.violinplot(
            data=vdf, x="Benchmark", y="Helpful seqs", hue="Benchmark", ax=ax,
            palette={_label(b): bench_colours[b] for b in bench_names},
            inner="quartile", linewidth=0.5, cut=0, density_norm="width",
            legend=False,
        )
    ax.set_ylabel("Helpful sequences / question")
    ax.set_title("a  Per-question helpful count", fontsize=8, fontweight="bold", loc="left")
    sns.despine(ax=ax)

    # (b) Entropy distribution
    ax = axes[0, 1]
    for b in bench_names:
        ent = per_question_entropy(data[b])
        sns.kdeplot(ent, ax=ax, color=bench_colours[b], linewidth=0.9,
                    label=f"{_label(b)} (med={np.median(ent):.2f})",
                    clip=(0, max(ent) if ent else 5), bw_adjust=0.6)
    ax.set_xlabel("Shannon entropy H (nats)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=5)
    ax.set_title("b  Router target entropy", fontsize=8, fontweight="bold", loc="left")
    sns.despine(ax=ax)

    # (c) Effective N distribution
    ax = axes[1, 0]
    for b in bench_names:
        eff = per_question_effective_n(data[b])
        sns.kdeplot(eff, ax=ax, color=bench_colours[b], linewidth=0.9,
                    label=f"{_label(b)} (med={np.median(eff):.1f})",
                    clip=(1, max(eff) if eff else 50), bw_adjust=0.6)
    ax.set_xlabel("Effective number of routes exp(H)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=5)
    ax.set_title("c  Effective route diversity", fontsize=8, fontweight="bold", loc="left")
    sns.despine(ax=ax)

    # (d) Entropy vs best_delta (gate+ only)
    ax = axes[1, 1]
    for b in bench_names:
        ent = per_question_entropy(data[b])
        deltas = [r["best_delta"] for r in data[b]]
        gates = [r.get("gate_label", 0) for r in data[b]]
        ent_gp = [e for e, g in zip(ent, gates) if g == 1]
        del_gp = [d for d, g in zip(deltas, gates) if g == 1]
        if ent_gp:
            ax.scatter(del_gp, ent_gp, s=0.6, alpha=0.2,
                       color=bench_colours[b], rasterized=True,
                       label=_label(b))
    ax.set_xlabel("Best delta")
    ax.set_ylabel("Router target entropy H")
    ax.legend(fontsize=5, markerscale=6)
    ax.set_title("d  Entropy vs improvement", fontsize=8, fontweight="bold", loc="left")
    sns.despine(ax=ax)

    fig.tight_layout(h_pad=2.5, w_pad=2.5)
    path = os.path.join(output_dir, "fig3_per_question_stats")
    save_fig(fig, path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 4 — Position-level mutation heatmap
# ═══════════════════════════════════════════════════════════════════════════

def fig_position_mutations(
    data: Dict[str, List[Dict]], output_dir: str
) -> str:
    """Heatmap: for each benchmark × layer position, how frequently
    that position is mutated in explored sequences.  Normalised per
    benchmark to show relative mutation frequency."""

    bench_names = sorted(data.keys())
    profiles = {}
    for b in bench_names:
        p = position_mutation_profile(data[b])
        if p.sum() > 0:
            profiles[b] = p / p.sum()
        elif len(p) > 0:
            profiles[b] = p.astype(float)

    if not profiles:
        return ""

    L = max(len(p) for p in profiles.values())
    mat = np.zeros((len(profiles), L))
    ordered_benches = list(profiles.keys())
    for i, b in enumerate(ordered_benches):
        p = profiles[b]
        mat[i, :len(p)] = p

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, SINGLE_COL * 0.4))
    sns.heatmap(
        mat, ax=ax, cmap="viridis",
        xticklabels=[str(i) if i % 2 == 0 else "" for i in range(L)],
        yticklabels=[_label(b) for b in ordered_benches],
        linewidths=0.2, linecolor="white",
        cbar_kws={"shrink": 0.6, "label": "Relative mutation freq."},
    )
    ax.set_xlabel("Layer position")
    ax.set_title("Position-level mutation profile", fontsize=8, fontweight="bold", loc="left")

    path = os.path.join(output_dir, "fig4_position_mutations")
    save_fig(fig, path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 5 — Summary statistics table
# ═══════════════════════════════════════════════════════════════════════════

def fig_summary_table(
    data: Dict[str, List[Dict]], output_dir: str
) -> str:
    """Render a comprehensive statistics table as a figure."""

    bench_names = sorted(data.keys())
    rows = []
    for b in bench_names:
        recs = data[b]
        n = len(recs)
        n_gp = sum(1 for r in recs if r.get("gate_label", 0) == 1)
        helpful = extract_helpful_sequences(recs)
        best_seqs = extract_best_sequences(recs)
        help_counts = per_question_helpful_count(recs)
        entropies = per_question_entropy(recs)
        eff_n = per_question_effective_n(recs)

        help_vals = np.array(list(helpful.values())) if helpful else np.array([0])
        best_vals = np.array(list(best_seqs.values())) if best_seqs else np.array([0])

        sorted_best = sorted(best_seqs.values(), reverse=True)
        top1_cov = sorted_best[0] / max(n_gp, 1) * 100 if sorted_best else 0
        top5_cov = sum(sorted_best[:5]) / max(n_gp, 1) * 100 if sorted_best else 0
        top10_cov = sum(sorted_best[:10]) / max(n_gp, 1) * 100 if sorted_best else 0

        gini = gini_coefficient(best_vals)

        deltas_gp = [r["best_delta"] for r in recs if r.get("gate_label", 0) == 1]

        rows.append({
            "Benchmark": _label(b),
            "N questions": n,
            "Gate+ (%)": f"{100 * n_gp / n:.1f}",
            "|Helpful|": len(helpful),
            "|Best-seq|": len(best_seqs),
            "Help/q med": f"{np.median(help_counts):.0f}",
            "Help/q mean": f"{np.mean(help_counts):.1f}",
            "H med": f"{np.median(entropies):.2f}",
            "H mean": f"{np.mean(entropies):.2f}",
            "Eff-N med": f"{np.median(eff_n):.1f}",
            "Top-1 %": f"{top1_cov:.1f}",
            "Top-5 %": f"{top5_cov:.1f}",
            "Top-10 %": f"{top10_cov:.1f}",
            "Gini": f"{gini:.3f}",
            "Δ med": f"{np.median(deltas_gp):.3f}" if deltas_gp else "—",
            "Δ mean": f"{np.mean(deltas_gp):.3f}" if deltas_gp else "—",
        })

    df = pd.DataFrame(rows)
    cols = df.columns.tolist()

    n_rows = len(df)
    n_cols = len(cols)
    fig_h = 0.22 + n_rows * 0.18
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=df.values,
        colLabels=cols,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(5)
    tbl.scale(1.0, 1.25)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_linewidth(0.3)
        if row == 0:
            cell.set_facecolor(plt.cm.viridis(0.85))
            cell.set_text_props(color="white", fontweight="bold", fontsize=4.8)
        else:
            cell.set_facecolor(plt.cm.viridis(0.02) if row % 2 == 0 else "white")

    ax.set_title("Summary statistics", fontsize=8, fontweight="bold", loc="left", pad=8)

    path = os.path.join(output_dir, "fig5_summary_table")
    save_fig(fig, path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 6 — Pairwise overlap detail + UpSet-style intersection sizes
# ═══════════════════════════════════════════════════════════════════════════

def fig_overlap_detail(
    data: Dict[str, List[Dict]], output_dir: str
) -> str:
    """(a) Bar chart of pairwise intersection sizes (best-seq),
    (b) Dice & Overlap coefficients alongside Jaccard for all helpful seqs,
    (c) Shared vs unique sequences stacked bar."""

    bench_names = sorted(data.keys())
    n = len(bench_names)
    if n < 2:
        return ""

    helpful_sets = {b: set(extract_helpful_sequences(data[b]).keys()) for b in bench_names}
    best_sets = {b: set(extract_best_sequences(data[b]).keys()) for b in bench_names}

    pairs = list(combinations(bench_names, 2))

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, SINGLE_COL * 0.7),
                             gridspec_kw={"width_ratios": [1, 1, 1.1]})

    # (a) Pairwise intersection sizes (best-seq)
    ax = axes[0]
    pair_labels = [f"{_label(a)}\n∩ {_label(b)}" for a, b in pairs]
    intersect_best = [len(best_sets[a] & best_sets[b]) for a, b in pairs]
    intersect_help = [len(helpful_sets[a] & helpful_sets[b]) for a, b in pairs]
    y_pos = np.arange(len(pairs))
    ax.barh(y_pos - 0.15, intersect_help, height=0.28,
            color=plt.cm.viridis(0.3), label="All helpful")
    ax.barh(y_pos + 0.15, intersect_best, height=0.28,
            color=plt.cm.viridis(0.7), label="Best-seq")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pair_labels, fontsize=5)
    ax.set_xlabel("Intersection size")
    ax.legend(fontsize=5)
    ax.set_title("a  Pairwise intersections", fontsize=8, fontweight="bold", loc="left")
    sns.despine(ax=ax)

    # (b) Three similarity metrics side by side
    ax = axes[1]
    metrics = {"Jaccard": [], "Dice": [], "Overlap": []}
    for a, b in pairs:
        sa, sb = helpful_sets[a], helpful_sets[b]
        inter = len(sa & sb)
        union = len(sa | sb)
        metrics["Jaccard"].append(inter / union if union else 0)
        metrics["Dice"].append(2 * inter / (len(sa) + len(sb)) if (len(sa) + len(sb)) else 0)
        metrics["Overlap"].append(inter / min(len(sa), len(sb)) if min(len(sa), len(sb)) else 0)

    x = np.arange(len(pairs))
    w = 0.25
    colours = [plt.cm.viridis(0.2), plt.cm.viridis(0.5), plt.cm.viridis(0.8)]
    for mi, (mname, mvals) in enumerate(metrics.items()):
        bars = ax.bar(x + (mi - 1) * w, mvals, w, color=colours[mi], label=mname)
        for bar, v in zip(bars, mvals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=4, rotation=45)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{_label(a)}/{_label(b)}" for a, b in pairs],
                       fontsize=4.5, rotation=30, ha="right")
    ax.set_ylabel("Similarity")
    ax.legend(fontsize=5)
    ax.set_title("b  Similarity metrics", fontsize=8, fontweight="bold", loc="left")
    sns.despine(ax=ax)

    # (c) Shared vs unique (stacked bar for helpful seqs)
    ax = axes[2]
    all_seqs = set()
    for s in helpful_sets.values():
        all_seqs |= s
    for b in bench_names:
        unique_to_b = helpful_sets[b] - set().union(*(helpful_sets[bb] for bb in bench_names if bb != b))
        shared = helpful_sets[b] - unique_to_b
        ax.barh(
            _label(b), len(unique_to_b),
            color=plt.cm.viridis(0.3), label="Unique" if b == bench_names[0] else "",
        )
        ax.barh(
            _label(b), len(shared), left=len(unique_to_b),
            color=plt.cm.viridis(0.7), label="Shared" if b == bench_names[0] else "",
        )
        ax.text(len(helpful_sets[b]) + max(len(s) for s in helpful_sets.values()) * 0.02,
                _label(b),
                f"{len(shared)}/{len(helpful_sets[b])} shared",
                va="center", fontsize=4.5)
    ax.set_xlabel("Number of sequences")
    ax.legend(fontsize=5)
    ax.set_title("c  Unique vs shared", fontsize=8, fontweight="bold", loc="left")
    sns.despine(ax=ax)

    fig.tight_layout(w_pad=2)
    path = os.path.join(output_dir, "fig6_overlap_detail")
    save_fig(fig, path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 7 — Sequence reuse rank-frequency + power-law diagnostics
# ═══════════════════════════════════════════════════════════════════════════

def fig_rank_frequency(
    data: Dict[str, List[Dict]], output_dir: str
) -> str:
    """(a) Rank-frequency plot (Zipf-like) for best-seq usage,
    (b) Same for all-helpful usage,
    (c) Distribution of best_delta conditioned on sequence popularity."""

    bench_names = sorted(data.keys())
    viridis = plt.cm.viridis
    bench_colours = {b: viridis(i / max(len(bench_names) - 1, 1))
                     for i, b in enumerate(bench_names)}

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, SINGLE_COL * 0.7))

    # (a) Best-seq rank-frequency
    ax = axes[0]
    for b in bench_names:
        counts = sorted(extract_best_sequences(data[b]).values(), reverse=True)
        if not counts:
            continue
        ranks = np.arange(1, len(counts) + 1)
        ax.plot(ranks, counts, color=bench_colours[b], linewidth=0.8,
                label=_label(b))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Questions (best-seq)")
    ax.legend(fontsize=5)
    ax.set_title("a  Rank-frequency (best)", fontsize=8, fontweight="bold", loc="left")
    sns.despine(ax=ax)

    # (b) Helpful-seq rank-frequency
    ax = axes[1]
    for b in bench_names:
        counts = sorted(extract_helpful_sequences(data[b]).values(), reverse=True)
        if not counts:
            continue
        ranks = np.arange(1, len(counts) + 1)
        ax.plot(ranks, counts, color=bench_colours[b], linewidth=0.8,
                label=_label(b))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Questions (helpful)")
    ax.legend(fontsize=5)
    ax.set_title("b  Rank-frequency (helpful)", fontsize=8, fontweight="bold", loc="left")
    sns.despine(ax=ax)

    # (c) Delta vs sequence popularity tercile
    ax = axes[2]
    tercile_data = []
    for b in bench_names:
        best_counts = extract_best_sequences(data[b])
        if not best_counts:
            continue
        vals = np.array(list(best_counts.values()))
        t33 = np.percentile(vals, 33)
        t66 = np.percentile(vals, 66)
        for rec in data[b]:
            if rec.get("gate_label", 0) != 1:
                continue
            sk = _seq_key(rec["best_seq"])
            pop = best_counts.get(sk, 0)
            if pop <= t33:
                tier = "Rare"
            elif pop <= t66:
                tier = "Medium"
            else:
                tier = "Popular"
            tercile_data.append({
                "Benchmark": _label(b),
                "Popularity": tier,
                "best_delta": rec["best_delta"],
            })
    tdf = pd.DataFrame(tercile_data)
    if not tdf.empty:
        sns.boxplot(
            data=tdf, x="Popularity", y="best_delta", hue="Benchmark", ax=ax,
            palette=[bench_colours[b] for b in bench_names if _label(b) in tdf["Benchmark"].unique()],
            linewidth=0.5, fliersize=0.5,
            order=["Rare", "Medium", "Popular"],
        )
        ax.legend(fontsize=4.5, loc="upper right")
    ax.set_ylabel("Best delta")
    ax.set_title("c  Delta by popularity", fontsize=8, fontweight="bold", loc="left")
    sns.despine(ax=ax)

    fig.tight_layout(w_pad=2)
    path = os.path.join(output_dir, "fig7_rank_frequency")
    save_fig(fig, path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  Console summary with rich statistics
# ═══════════════════════════════════════════════════════════════════════════

def print_statistics(data: Dict[str, List[Dict]]) -> None:
    bench_names = sorted(data.keys())
    sep = "=" * 90

    print(f"\n{sep}")
    print("  MCTS SEQUENCE ANALYSIS — COMPREHENSIVE STATISTICS")
    print(sep)

    for b in bench_names:
        recs = data[b]
        n = len(recs)
        n_gp = sum(1 for r in recs if r.get("gate_label", 0) == 1)
        helpful = extract_helpful_sequences(recs)
        best_seqs = extract_best_sequences(recs)
        help_counts = per_question_helpful_count(recs)
        entropies = per_question_entropy(recs)
        eff_n = per_question_effective_n(recs)
        deltas = [r["best_delta"] for r in recs if r.get("gate_label", 0) == 1]
        num_explored = [r["num_explored"] for r in recs]

        print(f"\n  {_label(b):>12}  ({b})")
        print(f"  {'─' * 60}")
        print(f"    Questions:           {n:>7}")
        print(f"    Gate-positive:       {n_gp:>7}  ({100 * n_gp / n:.1f}%)")
        print(f"    Unique helpful seqs: {len(helpful):>7}")
        print(f"    Unique best-seqs:    {len(best_seqs):>7}")
        print(f"    Explored / question: {np.mean(num_explored):>7.1f} mean, "
              f"{np.median(num_explored):.0f} median")
        print(f"    Helpful / question:  {np.mean(help_counts):>7.1f} mean, "
              f"{np.median(help_counts):.0f} median, {np.max(help_counts)} max")
        print(f"    Entropy H:           {np.mean(entropies):>7.2f} mean, "
              f"{np.median(entropies):.2f} median")
        print(f"    Effective N:         {np.mean(eff_n):>7.1f} mean, "
              f"{np.median(eff_n):.1f} median")
        if deltas:
            print(f"    Best delta (gate+):  {np.mean(deltas):>7.3f} mean, "
                  f"{np.median(deltas):.3f} median, {np.max(deltas):.3f} max")

        sorted_best = sorted(best_seqs.values(), reverse=True)
        for k in [1, 5, 10, 20, 50]:
            cov = sum(sorted_best[:k]) / max(n_gp, 1) * 100
            print(f"    Top-{k:<2} best-seq coverage: {cov:>5.1f}%")

        best_vals = np.array(list(best_seqs.values())) if best_seqs else np.array([0])
        print(f"    Gini (best-seq):     {gini_coefficient(best_vals):.3f}")

    # Pairwise overlap
    if len(bench_names) >= 2:
        print(f"\n{'─' * 90}")
        print("  PAIRWISE OVERLAP")
        print(f"{'─' * 90}")

        helpful_sets = {b: set(extract_helpful_sequences(data[b]).keys()) for b in bench_names}
        best_sets = {b: set(extract_best_sequences(data[b]).keys()) for b in bench_names}

        for a, b in combinations(bench_names, 2):
            sa_h, sb_h = helpful_sets[a], helpful_sets[b]
            sa_b, sb_b = best_sets[a], best_sets[b]
            ih = len(sa_h & sb_h)
            ib = len(sa_b & sb_b)
            jh = ih / len(sa_h | sb_h) if len(sa_h | sb_h) else 0
            jb = ib / len(sa_b | sb_b) if len(sa_b | sb_b) else 0
            print(f"    {_label(a):>12} ∩ {_label(b):<12}")
            print(f"      Helpful:  |∩|={ih:>5}  Jaccard={jh:.4f}  "
                  f"|A|={len(sa_h)}, |B|={len(sb_h)}")
            print(f"      Best-seq: |∩|={ib:>5}  Jaccard={jb:.4f}  "
                  f"|A|={len(sa_b)}, |B|={len(sb_b)}")

    print(f"\n{sep}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI & orchestration
# ═══════════════════════════════════════════════════════════════════════════

def auto_discover_data_dirs(root: Path) -> Dict[str, str]:
    """Auto-discover fine_routing_data_*_mcts directories."""
    dirs = {}
    for p in sorted(root.glob("fine_routing_data_*_mcts")):
        if not p.is_dir():
            continue
        for jsonl in p.glob("*.jsonl"):
            bench = jsonl.stem
            if os.path.getsize(str(jsonl)) > 0:
                dirs[bench] = str(p)
    for p in sorted(root.glob("fine_routing_data_*_mcts_v2")):
        if not p.is_dir():
            continue
        for jsonl in p.glob("*.jsonl"):
            bench = jsonl.stem
            if os.path.getsize(str(jsonl)) > 0:
                dirs[bench] = str(p)
    return dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data_dirs", nargs="*", default=None,
        help="Explicit data directories (bench=path pairs or just paths to auto-detect). "
             "Default: auto-discover fine_routing_data_*_mcts under --root.",
    )
    p.add_argument("--output_dir", default="figures/mcts_sequence_analysis")
    p.add_argument("--root", default=str(ROOT))
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    args = parse_args()
    root = Path(args.root)
    out_dir = os.path.join(root, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    apply_theme()

    if args.data_dirs:
        data_dirs = {}
        for entry in args.data_dirs:
            if "=" in entry:
                bench, path = entry.split("=", 1)
                data_dirs[bench] = path
            else:
                for jsonl in Path(entry).glob("*.jsonl"):
                    if os.path.getsize(str(jsonl)) > 0:
                        data_dirs[jsonl.stem] = entry
    else:
        data_dirs = auto_discover_data_dirs(root)

    if not data_dirs:
        logger.error("No MCTS data directories found. Use --data_dirs or check --root.")
        sys.exit(1)

    logger.info("Discovered %d benchmarks:", len(data_dirs))
    for b, d in sorted(data_dirs.items()):
        logger.info("  %s → %s", b, d)

    logger.info("Loading data ...")
    data = load_benchmark_data(data_dirs)

    if not data:
        logger.error("No data loaded.")
        sys.exit(1)

    print_statistics(data)

    logger.info("Generating figures ...")
    paths = []

    p = fig_cross_benchmark_overlap(data, out_dir)
    if p:
        paths.append(p)
        logger.info("  Fig 1: %s", p)

    p = fig_sequence_distributions(data, out_dir)
    paths.append(p)
    logger.info("  Fig 2: %s", p)

    p = fig_per_question_stats(data, out_dir)
    paths.append(p)
    logger.info("  Fig 3: %s", p)

    p = fig_position_mutations(data, out_dir)
    if p:
        paths.append(p)
        logger.info("  Fig 4: %s", p)

    p = fig_summary_table(data, out_dir)
    paths.append(p)
    logger.info("  Fig 5: %s", p)

    p = fig_overlap_detail(data, out_dir)
    if p:
        paths.append(p)
        logger.info("  Fig 6: %s", p)

    p = fig_rank_frequency(data, out_dir)
    paths.append(p)
    logger.info("  Fig 7: %s", p)

    print("\n" + "=" * 70)
    print("GENERATED FIGURES")
    print("=" * 70)
    for p in paths:
        if p:
            print(f"  {p}.pdf")
            print(f"  {p}.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
