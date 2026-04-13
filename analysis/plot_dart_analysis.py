"""
DART Evaluation — Publication-Quality Figure

Produces a single multi-panel figure suitable for a Nature-style journal.
Panels:
    (a) Base vs optimal-routing accuracy   (grouped bar)
    (b) Error recovery rate                (bar)
    (c) Module frequency heatmap, DART-1   (seaborn / viridis)
    (d) Module frequency heatmap, DART-2   (seaborn / viridis)
    (e) Distribution of layer changes      (histogram + cumulative)
    (f) Summary statistics table

Data is read from JSONL + _stats.json files produced by generate_router_data.py,
or from fine-grained MCTS JSONL (``anchor_score`` / ``best_score``) via
``--fine_routing_jsonl`` for a standalone (a)+(b) figure.

Usage:
    python plot_dart_analysis.py \
        --data_dir data/router_train \
        --output_dir plots/dart_analysis \
        --num_layers 36

    python plot_dart_analysis.py \
        --fine_routing_jsonl fine_routing_data_ft_qwen05b_250sims/boolq.jsonl \
        --bench_display BoolQ \
        --output_ab figures/boolq_ft_mcts_ab
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import os
import json
import glob
import math
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import seaborn as sns


# ---------------------------------------------------------------------------
#  Style configuration — Nature guidelines
# ---------------------------------------------------------------------------

def setup_nature_style() -> None:
    """Apply publication-quality matplotlib / seaborn defaults.

    Nature journals prefer:
      - Sans-serif font (Helvetica / Arial)
      - 7-8 pt axis labels, ≥6 pt tick labels
      - Minimal gridlines, no top/right spines
      - 300+ DPI
    """
    sns.set_theme(style="ticks", font_scale=1.0)

    mpl.rcParams.update({
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        # Lines / markers
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        # Spines
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Grid
        "axes.grid": False,
    })


# ---------------------------------------------------------------------------
#  Colour palette (viridis-based, colour-blind safe)
# ---------------------------------------------------------------------------

PAL_BASE = "#404788"       # dark indigo (viridis ~0.2)
PAL_IMPROVED = "#22a884"   # teal-green  (viridis ~0.6)
PAL_RECOVERY = "#fde725"   # bright yellow (viridis ~1.0)
PAL_ACCENT = "#7ad151"     # lime (viridis ~0.75)
VIRIDIS = "icefire"


def _shorten_benchmark(name: str) -> str:
    """Pretty benchmark name for axis labels (shared with router viz naming)."""
    key = (name or "").strip().lower()
    m = {
        "winogrande": "Winogrande",
        "mmlu_all": "MMLU",
        "arc_challenge": "ARC-Challenge",
        "arc_easy": "ARC (Easy)",
        "commonsenseqa": "Common QA",
        "boolq": "BoolQ",
        "gsm8k_hard": "GSM8K-Hard",
        "bigbench_all": "BigBench",
        "bigbench_boolean_expressions": "Big-Bench (Boolean)",
    }
    return m.get(key, name.replace("_", " ").title())


# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------

def find_dart_files(data_dir: str) -> Dict[str, Dict[str, str]]:
    """Discover DART JSONL / stats pairs grouped by difficulty level."""
    files: Dict[str, Dict[str, str]] = {}
    for jsonl_path in sorted(glob.glob(os.path.join(data_dir, "*.jsonl"))):
        basename = os.path.basename(jsonl_path)
        for level in range(1, 6):
            if f"dart-{level}" in basename:
                key = f"DART-{level}"
                if key not in files:
                    files[key] = {}
                files[key]["data"] = jsonl_path
                stats_path = jsonl_path.replace(".jsonl", "_stats.json")
                if os.path.exists(stats_path):
                    files[key]["stats"] = stats_path
                break
    return files


def load_stats(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def load_fine_routing_mcts_jsonl(jsonl_path: str) -> pd.DataFrame:
    """Load fine-grained MCTS supervision JSONL into a DART-compatible frame.

    Expects records from ``build_ft_fine_routing_dataset`` / similar pipelines with
    ``anchor_score``, ``best_score``, ``scoring_mode`` (``binary`` or ``continuous``).

    Columns produced:
      * ``original_correct`` — baseline (anchor routing) treated as correct
      * ``improved`` — baseline wrong but MCTS best sequence scores as correct
      * ``num_changes`` — Hamming distance ``anchor_sequence`` vs ``best_seq`` when lengths match
    """
    rows: List[Dict] = []
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            a = float(d.get("anchor_score", 0.0))
            b = float(d.get("best_score", 0.0))
            mode = (d.get("scoring_mode") or "binary").lower()
            if mode == "binary":
                oc = a >= 0.5
                improved = (not oc) and (b >= 0.5)
            else:
                oc = a >= 0.5
                improved = (not oc) and (b >= 0.5)

            anchor = d.get("anchor_sequence")
            best = d.get("best_seq")
            nchg = np.nan
            if isinstance(anchor, list) and isinstance(best, list) and len(anchor) == len(best):
                nchg = sum(1 for i, x in enumerate(anchor) if x != best[i])

            rows.append({
                "benchmark_id": d.get("benchmark_id"),
                "original_correct": oc,
                "improved": improved,
                "num_changes": nchg,
            })
    return pd.DataFrame(rows)


def load_dart_data(jsonl_path: str, num_layers: int) -> pd.DataFrame:
    """Load DART JSONL into a DataFrame with derived columns.

    Handles ``best_sequence = null`` gracefully (no crash on None).
    """
    records = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    df = pd.DataFrame(records)

    baseline = list(range(num_layers))

    # --- num_changes (robust to None best_sequence) ---
    def _count_changes(seq):
        if seq is None or not isinstance(seq, (list, tuple)):
            return np.nan
        if len(seq) != num_layers:
            return np.nan
        return sum(1 for i, layer in enumerate(seq) if layer != baseline[i])

    if "best_sequence" in df.columns:
        df["num_changes"] = df["best_sequence"].apply(_count_changes)
    else:
        df["num_changes"] = np.nan

    return df


# ---------------------------------------------------------------------------
#  Statistics helpers
# ---------------------------------------------------------------------------

def compute_module_frequency(
    df: pd.DataFrame,
    num_layers: int,
    *,
    only_improved: bool = False,
) -> np.ndarray:
    """Return a (num_layers × num_layers) matrix of module usage frequencies.

    Each row = position in the sequence, each column = module index.
    Counts come from *all* ``good_sequences`` entries (the set of
    layer orderings that produced a correct answer via MCTS).

    If *only_improved* is True, restrict to samples where the base model
    was wrong but MCTS found a fix.
    """
    freq = np.zeros((num_layers, num_layers), dtype=np.float64)

    for _, row in df.iterrows():
        if only_improved and not row.get("improved", False):
            continue

        sequences = row.get("good_sequences")
        if not sequences:
            # Fallback to best_sequence when good_sequences is empty/missing
            best = row.get("best_sequence")
            if best is not None and len(best) == num_layers:
                sequences = [best]
            else:
                continue

        for seq in sequences:
            if seq is None or len(seq) != num_layers:
                continue
            for pos, layer in enumerate(seq):
                if 0 <= layer < num_layers:
                    freq[pos, layer] += 1

    return freq


def compute_deviation_frequency(
    df: pd.DataFrame,
    num_layers: int,
) -> np.ndarray:
    """Compute how often each (position, module) deviates from baseline.

    Returns a (num_layers × num_layers) float matrix.
    Cell (p, m) = fraction of good_sequences where position p uses module m
    AND m != p (i.e. a non-identity routing).
    """
    count = np.zeros((num_layers, num_layers), dtype=np.float64)
    total = np.zeros(num_layers, dtype=np.float64)

    for _, row in df.iterrows():
        sequences = row.get("good_sequences")
        if not sequences:
            best = row.get("best_sequence")
            if best is not None and len(best) == num_layers:
                sequences = [best]
            else:
                continue
        for seq in sequences:
            if seq is None or len(seq) != num_layers:
                continue
            for pos, layer in enumerate(seq):
                total[pos] += 1
                if layer != pos and 0 <= layer < num_layers:
                    count[pos, layer] += 1

    # Normalise row-wise
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = count / total[:, None]
    rate = np.nan_to_num(rate, nan=0.0)
    return rate


# ---------------------------------------------------------------------------
#  Panel drawing helpers
# ---------------------------------------------------------------------------

def _add_panel_label(ax: mpl.axes.Axes, label: str) -> None:
    """Add a bold lowercase panel label (a, b, c, …) in the upper-left."""
    ax.text(
        -0.08, 1.08, label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="right",
    )


def _annotate_bars(ax: mpl.axes.Axes, bars, fmt: str = "{:.1f}%", offset: int = 2, fontsize: int = 6) -> None:
    """Place value labels on top of bars."""
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            fmt.format(h),
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


# ---------------------------------------------------------------------------
#  Main figure
# ---------------------------------------------------------------------------

def create_publication_figure(
    stats_dict: Dict[str, Dict],
    df_dict: Dict[str, pd.DataFrame],
    num_layers: int,
    output_path: str,
) -> None:
    """Produce one large multi-panel figure."""

    setup_nature_style()

    levels = sorted(stats_dict.keys())  # e.g. ["DART-1", "DART-2"]
    n_levels = len(levels)

    # --- Layout (3 rows, variable columns) ---
    # Row 0: (a) accuracy bars  |  (b) recovery bars
    # Row 1: (c) heatmap DART-1 |  (d) heatmap DART-2
    # Row 2: (e) change dist    |  (f) summary table
    fig = plt.figure(figsize=(7.2, 8.4))  # ~180 mm wide (2-column Nature)
    gs = GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[1, 1.6, 1],
        hspace=0.42,
        wspace=0.35,
    )

    # ------------------------------------------------------------------ (a)
    ax_a = fig.add_subplot(gs[0, 0])
    _draw_accuracy_bars(ax_a, stats_dict, levels, df_dict)
    _add_panel_label(ax_a, "a")

    # ------------------------------------------------------------------ (b)
    ax_b = fig.add_subplot(gs[0, 1])
    _draw_recovery_bars(ax_b, stats_dict, levels, df_dict)
    _add_panel_label(ax_b, "b")

    # ------------------------------------------------------------------ (c, d) heatmaps
    heatmap_axes = []
    for col_idx, level in enumerate(levels[:2]):  # max 2 heatmaps
        ax = fig.add_subplot(gs[1, col_idx])
        heatmap_axes.append(ax)
        df = df_dict.get(level)
        if df is not None:
            _draw_heatmap(ax, df, num_layers, level)
        _add_panel_label(ax, chr(ord("c") + col_idx))

    # ------------------------------------------------------------------ (e)
    ax_e = fig.add_subplot(gs[2, 0])
    _draw_change_distribution(ax_e, df_dict, levels)
    _add_panel_label(ax_e, "e")

    # ------------------------------------------------------------------ (f)
    ax_f = fig.add_subplot(gs[2, 1])
    _draw_summary_table(ax_f, stats_dict, df_dict, levels, num_layers)
    _add_panel_label(ax_f, "f")

    # --- Save both PNG and PDF ---
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        path = output_path.rsplit(".", 1)[0] + f".{ext}"
        fig.savefig(path, facecolor="white")
        print(f"Saved {ext.upper()} → {path}")
    plt.close(fig)



# ---------------------------------------------------------------------------
#  Individual panel implementations
# ---------------------------------------------------------------------------

def _draw_accuracy_bars(
    ax: mpl.axes.Axes,
    stats_dict: Dict,
    levels: List[str],
    df_dict: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    """Panel (a): base accuracy vs accuracy with optimal routing.

    If *df_dict* is provided, stats are recomputed from the JSONL data
    (more accurate than possibly-stale _stats.json).
    """
    base_acc = []
    opt_acc = []
    for level in levels:
        if df_dict and level in df_dict and not df_dict[level].empty:
            rs = _recompute_stats_from_df(df_dict[level])
            orig = rs["orig_acc"] * 100
            optimal = (rs["n_correct"] + rs["n_improved"]) / max(1, rs["total"]) * 100
        else:
            s = stats_dict[level]
            orig = s.get("original_accuracy", 0) * 100
            total = s.get("total_samples_processed", 1)
            orig_correct = s.get("num_original_correct", 0)
            num_improved = s.get("num_with_improvement", 0)
            optimal = (orig_correct + num_improved) / total * 100
        base_acc.append(orig)
        opt_acc.append(min(100.0, optimal))

    x = np.arange(len(levels))
    w = 0.32
    b1 = ax.bar(x - w / 2, base_acc, w, label="Base model", color=PAL_BASE, edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + w / 2, opt_acc, w, label="With optimal routing", color=PAL_IMPROVED, edgecolor="white", linewidth=0.5)
    _annotate_bars(ax, b1)
    _annotate_bars(ax, b2)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Base vs. optimal-routing accuracy", fontsize=8, pad=6)


def _draw_recovery_bars(
    ax: mpl.axes.Axes,
    stats_dict: Dict,
    levels: List[str],
    df_dict: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    """Panel (b): error recovery rate."""
    recovery = []
    for level in levels:
        if df_dict and level in df_dict and not df_dict[level].empty:
            rs = _recompute_stats_from_df(df_dict[level])
            error_cnt = rs["total"] - rs["n_correct"]
            rec = rs["n_improved"] / max(1, error_cnt) * 100
        else:
            s = stats_dict[level]
            orig_acc = s.get("original_accuracy", 0)
            imp_rate = s.get("improvement_rate", 0)
            error_rate = 1.0 - orig_acc
            rec = (imp_rate / error_rate) * 100 if error_rate > 0 else 0
        recovery.append(min(100.0, rec))

    cmap = mpl.colormaps.get_cmap(VIRIDIS)
    colours = [cmap(0.35 + 0.35 * i / max(1, len(levels) - 1)) for i in range(len(levels))]
    bars = ax.bar(levels, recovery, color=colours, edgecolor="white", linewidth=0.5)
    _annotate_bars(ax, bars)
    ax.set_ylabel("Error recovery (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Fraction of base-model errors recovered", fontsize=8, pad=6)


def _draw_heatmap(
    ax: mpl.axes.Axes,
    df: pd.DataFrame,
    num_layers: int,
    title: str,
) -> None:
    """Panels (c)/(d): module frequency heatmap using seaborn + viridis.

    Restricted to *improved* samples only (base model was wrong, MCTS
    found a correct routing) so the heatmap reflects adaptive module
    selection rather than mere robustness of the default ordering.
    """
    freq = compute_module_frequency(df, num_layers, only_improved=True)

    # Row-normalise so each position sums to 1
    row_sums = freq.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        normed = freq / np.where(row_sums > 0, row_sums, 1)

    # Use log1p scaling for better dynamic range
    display = np.log1p(normed * 100)  # scale before log for readability

    # Tick positions — show every 6th for a 36-layer model
    tick_step = max(1, num_layers // 6)
    ticks = list(range(0, num_layers, tick_step))

    sns.heatmap(
        display,
        ax=ax,
        cmap=VIRIDIS,
        cbar_kws={"label": "log(1 + usage %)", "shrink": 0.75, "aspect": 15},
        xticklabels=False,
        yticklabels=False,
        square=False,
        rasterized=True,
    )
    # Overlay baseline diagonal
    ax.plot(
        [i + 0.5 for i in range(num_layers)],
        [i + 0.5 for i in range(num_layers)],
        color="white",
        linewidth=0.8,
        linestyle="--",
        alpha=0.7,
        label="Baseline (identity)",
    )
    ax.set_xticks([t + 0.5 for t in ticks])
    ax.set_xticklabels(ticks, fontsize=5)
    ax.set_yticks([t + 0.5 for t in ticks])
    ax.set_yticklabels(ticks, fontsize=5)
    ax.set_xlabel("Module index")
    ax.set_ylabel("Position")
    ax.set_title(f"Error-correcting routings — {title}", fontsize=8, pad=6)
    ax.legend(fontsize=5, loc="lower right", framealpha=0.7)


def _draw_change_distribution(
    ax: mpl.axes.Axes,
    df_dict: Dict[str, pd.DataFrame],
    levels: List[str],
) -> None:
    """Panel (e): number of layer changes in successful sequences."""
    cmap = mpl.colormaps.get_cmap(VIRIDIS)

    for i, level in enumerate(levels):
        df = df_dict.get(level)
        if df is None:
            continue
        changes = df["num_changes"].dropna()
        if changes.empty:
            continue
        colour = cmap(0.3 + 0.4 * i / max(1, len(levels) - 1))

        # Histogram
        bins = np.arange(-0.5, changes.max() + 1.5, 1)
        ax.hist(
            changes,
            bins=bins,
            density=True,
            alpha=0.55,
            color=colour,
            edgecolor="white",
            linewidth=0.3,
            label=level,
        )

    ax.set_xlabel("No. layer changes from baseline")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of routing changes", fontsize=8, pad=6)
    ax.legend(frameon=False)

    # Secondary y-axis: cumulative (first level only for clarity)
    first_level = levels[0] if levels else None
    if first_level and first_level in df_dict:
        changes = df_dict[first_level]["num_changes"].dropna().sort_values()
        if not changes.empty:
            ax2 = ax.twinx()
            cumulative = np.arange(1, len(changes) + 1) / len(changes) * 100
            ax2.plot(changes.values, cumulative, color="grey", linewidth=0.8, linestyle="-", alpha=0.6)
            ax2.set_ylabel("Cumulative %", fontsize=6, color="grey")
            ax2.tick_params(axis="y", labelsize=5, colors="grey")
            ax2.set_ylim(0, 105)
            ax2.spines["right"].set_visible(True)
            ax2.spines["right"].set_color("grey")
            ax2.spines["right"].set_linewidth(0.5)


def create_fine_routing_mcts_ab_figure(
    df: pd.DataFrame,
    benchmark_display: str,
    output_base: str,
    *,
    legend_base: str = "Fine-tuned model",
    legend_mcts: str = "Fine-grained MCTS on fine-tuned models",
    title_accuracy: str = "Fine-tuned vs. fine-grained MCTS accuracy",
    title_recovery: str = "Fraction of base-model errors recovered",
    thick_spines: bool = True,
) -> List[str]:
    """Panels (a)(b) only: same layout as DART figure row 0, for one benchmark."""
    setup_nature_style()

    if df.empty or "original_correct" not in df.columns:
        raise ValueError("DataFrame must be non-empty with original_correct / improved columns")

    pseudo_level = benchmark_display
    stats_dict: Dict[str, Dict] = {pseudo_level: {}}
    df_dict = {pseudo_level: df}

    # Extra width for outside legend; extra height for ~1 in title offset (pad in pt).
    fig, axes = plt.subplots(1, 2, figsize=(5.9, 3.05), layout="constrained")
    ax_a, ax_b = axes[0], axes[1]
    fig.set_constrained_layout_pads(w_pad=0.12, h_pad=0.02, rect=(0.0, 0.0, 0.98, 1.0))

    if thick_spines:
        for ax in (ax_a, ax_b):
            for spine in ("bottom", "left"):
                ax.spines[spine].set_linewidth(1.25)
                ax.spines[spine].set_color("black")
            ax.tick_params(axis="both", which="major", width=1.0, length=4.0, colors="black")

    title_pad_pt = 72.0  # matplotlib title pad is in points (~1 inch)

    _draw_accuracy_bars_custom(
        ax_a, stats_dict, [pseudo_level], df_dict,
        legend_labels=(legend_base, legend_mcts),
        title=title_accuracy,
        xtick_override=[benchmark_display],
        title_pad_pt=title_pad_pt,
    )
    _add_panel_label(ax_a, "a")

    _draw_recovery_bar_single(ax_b, stats_dict, [pseudo_level], df_dict, color=PAL_BASE)
    ax_b.set_title(title_recovery, fontsize=8, pad=title_pad_pt)
    ax_b.set_xticks([0])
    ax_b.set_xticklabels([benchmark_display])
    _add_panel_label(ax_b, "b")

    os.makedirs(os.path.dirname(output_base) or ".", exist_ok=True)
    out_paths: List[str] = []
    for ext in ("png", "pdf"):
        path = output_base.rsplit(".", 1)[0] + f".{ext}"
        fig.savefig(path, facecolor="white")
        out_paths.append(os.path.abspath(path))
        print(f"Saved {ext.upper()} → {path}")
    plt.close(fig)
    return out_paths


def _draw_accuracy_bars_custom(
    ax: mpl.axes.Axes,
    stats_dict: Dict,
    levels: List[str],
    df_dict: Optional[Dict[str, pd.DataFrame]] = None,
    *,
    legend_labels: Tuple[str, str] = ("Base model", "With optimal routing"),
    title: str = "Base vs. optimal-routing accuracy",
    xtick_override: Optional[List[str]] = None,
    title_pad_pt: float = 6.0,
) -> None:
    """Like ``_draw_accuracy_bars`` but custom legend strings and title."""
    base_acc = []
    opt_acc = []
    for level in levels:
        if df_dict and level in df_dict and not df_dict[level].empty:
            rs = _recompute_stats_from_df(df_dict[level])
            orig = rs["orig_acc"] * 100
            optimal = (rs["n_correct"] + rs["n_improved"]) / max(1, rs["total"]) * 100
        else:
            s = stats_dict[level]
            orig = s.get("original_accuracy", 0) * 100
            total = s.get("total_samples_processed", 1)
            orig_correct = s.get("num_original_correct", 0)
            num_improved = s.get("num_with_improvement", 0)
            optimal = (orig_correct + num_improved) / total * 100
        base_acc.append(orig)
        opt_acc.append(min(100.0, optimal))

    x = np.arange(len(levels))
    w = 0.32
    b1 = ax.bar(x - w / 2, base_acc, w, label=legend_labels[0], color=PAL_BASE, edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + w / 2, opt_acc, w, label=legend_labels[1], color=PAL_IMPROVED, edgecolor="white", linewidth=0.5)
    _annotate_bars(ax, b1)
    _annotate_bars(ax, b2)
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_override if xtick_override is not None else levels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        alignment="left",
    )
    ax.set_title(title, fontsize=8, pad=title_pad_pt)


def _draw_recovery_bar_single(
    ax: mpl.axes.Axes,
    stats_dict: Dict,
    levels: List[str],
    df_dict: Optional[Dict[str, pd.DataFrame]] = None,
    *,
    color: str = PAL_BASE,
) -> None:
    """Single-group recovery bar (panel b style for one benchmark)."""
    recovery = []
    for level in levels:
        if df_dict and level in df_dict and not df_dict[level].empty:
            rs = _recompute_stats_from_df(df_dict[level])
            error_cnt = rs["total"] - rs["n_correct"]
            rec = rs["n_improved"] / max(1, error_cnt) * 100
        else:
            s = stats_dict[level]
            orig_acc = s.get("original_accuracy", 0)
            imp_rate = s.get("improvement_rate", 0)
            error_rate = 1.0 - orig_acc
            rec = (imp_rate / error_rate) * 100 if error_rate > 0 else 0
        recovery.append(min(100.0, rec))

    x = np.arange(len(levels))
    bars = ax.bar(x, recovery, width=0.42, color=color, edgecolor="white", linewidth=0.5)
    _annotate_bars(ax, bars)
    ax.set_ylabel("Error recovery (%)")
    ax.set_ylim(0, 105)


def _recompute_stats_from_df(df: pd.DataFrame) -> Dict:
    """Recompute accuracy / improvement stats from the actual JSONL data.

    This avoids relying on potentially-stale _stats.json checkpoints.
    """
    n = len(df)
    n_correct = int(df.get("original_correct", pd.Series(dtype=bool)).sum())
    n_improved = int(df.get("improved", pd.Series(dtype=bool)).sum())
    orig_acc = n_correct / max(1, n)
    imp_rate = n_improved / max(1, n)
    return {
        "total": n,
        "n_correct": n_correct,
        "n_improved": n_improved,
        "orig_acc": orig_acc,
        "imp_rate": imp_rate,
    }


def _draw_summary_table(
    ax: mpl.axes.Axes,
    stats_dict: Dict[str, Dict],
    df_dict: Dict[str, pd.DataFrame],
    levels: List[str],
    num_layers: int,
) -> None:
    """Panel (f): summary statistics table.

    Stats are recomputed from the JSONL DataFrames to avoid stale
    _stats.json checkpoints.
    """
    ax.axis("off")

    col_labels = ["", "n", "Base", "Opt.", "Recov.", "Med\u0394", "Sim"]
    rows = []
    level_stats = {}

    for level in levels:
        df = df_dict.get(level)
        s_file = stats_dict.get(level, {})

        if df is not None and not df.empty:
            rs = _recompute_stats_from_df(df)
        else:
            # Fall back to stats file
            rs = {
                "total": s_file.get("total_samples_processed", 0),
                "n_correct": s_file.get("num_original_correct", 0),
                "n_improved": s_file.get("num_with_improvement", 0),
                "orig_acc": s_file.get("original_accuracy", 0),
                "imp_rate": s_file.get("improvement_rate", 0),
            }

        total = rs["total"]
        orig_acc = rs["orig_acc"] * 100
        opt_acc = (rs["n_correct"] + rs["n_improved"]) / max(1, total) * 100
        error_cnt = total - rs["n_correct"]
        rec = rs["n_improved"] / max(1, error_cnt) * 100

        med_changes = "—"
        if df is not None:
            ch = df["num_changes"].dropna()
            if not ch.empty:
                med_changes = f"{ch.median():.0f}"

        n_sims = s_file.get("config", {}).get("num_simulations", "—")
        level_stats[level] = rs

        rows.append([
            level,
            f"{total:,}",
            f"{orig_acc:.1f}%",
            f"{min(100, opt_acc):.1f}%",
            f"{min(100, rec):.1f}%",
            med_changes,
            str(n_sims),
        ])

    # Totals row
    tot_n = sum(level_stats[l]["total"] for l in levels)
    tot_correct = sum(level_stats[l]["n_correct"] for l in levels)
    tot_improved = sum(level_stats[l]["n_improved"] for l in levels)
    tot_orig_acc = tot_correct / max(1, tot_n) * 100
    tot_opt_acc = (tot_correct + tot_improved) / max(1, tot_n) * 100
    tot_error = tot_n - tot_correct
    tot_rec = tot_improved / max(1, tot_error) * 100

    rows.append([
        "Total",
        f"{tot_n:,}",
        f"{tot_orig_acc:.1f}%",
        f"{min(100, tot_opt_acc):.1f}%",
        f"{min(100, tot_rec):.1f}%",
        "—",
        "—",
    ])

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colWidths=[0.17, 0.14, 0.13, 0.13, 0.14, 0.13, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1.05, 1.6)

    # Style header
    for j in range(len(col_labels)):
        cell = table[(0, j)]
        cell.set_facecolor("#2d2d2d")
        cell.set_text_props(color="white", fontweight="bold", fontsize=6.5)
        cell.set_edgecolor("white")

    # Style body
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            cell = table[(i, j)]
            cell.set_edgecolor("#cccccc")
            if i == len(rows):  # totals row
                cell.set_facecolor("#f0f0f0")
                cell.set_text_props(fontweight="bold")
            else:
                cell.set_facecolor("white")

    ax.set_title("Summary statistics", fontsize=8, pad=10)


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

def generate_all_plots(data_dir: str, output_dir: str, num_layers: int = 36) -> None:
    """Load data, generate publication figure, and save."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("DART Publication Figure Generator")
    print(f"{'=' * 60}")
    print(f"Data directory : {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Num layers     : {num_layers}")
    print(f"{'=' * 60}\n")

    dart_files = find_dart_files(data_dir)
    if not dart_files:
        print(f"ERROR: No DART files found in {data_dir}")
        return

    print(f"Found levels: {list(dart_files.keys())}")

    stats_dict: Dict[str, Dict] = {}
    df_dict: Dict[str, pd.DataFrame] = {}

    for level, paths in dart_files.items():
        if "stats" in paths:
            stats_dict[level] = load_stats(paths["stats"])
            print(f"  {level} stats loaded")
        if "data" in paths:
            df_dict[level] = load_dart_data(paths["data"], num_layers)
            n = len(df_dict[level])
            n_improved = df_dict[level].get("improved", pd.Series(dtype=bool)).sum()
            print(f"  {level} data loaded: {n} samples ({n_improved} improved)")

    if not stats_dict:
        print("ERROR: No stats files found")
        return

    output_path = os.path.join(output_dir, "dart_publication_figure.png")
    create_publication_figure(stats_dict, df_dict, num_layers, output_path)

    print(f"\n{'=' * 60}")
    print(f"All outputs saved to: {output_dir}")
    print(f"{'=' * 60}\n")


def parse_args():
    parser = ArgumentParser(description="Generate DART publication-quality figure")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing DART JSONL + stats files")
    parser.add_argument("--fine_routing_jsonl", type=str, default=None,
                        help="Path to fine-grained MCTS JSONL (per-question anchor / best scores)")
    parser.add_argument("--bench_display", type=str, default=None,
                        help="X-axis label for AB figure (default: benchmark_id from JSONL or 'Benchmark')")
    parser.add_argument("--output_ab", type=str, default=None,
                        help="Output base path for AB-only figure, no extension (writes .png and .pdf)")
    parser.add_argument("--output_dir", type=str, default="plots/dart_analysis",
                        help="Output directory for full DART figure (--data_dir mode)")
    parser.add_argument("--num_layers", type=int, default=36,
                        help="Number of transformer layers")
    return parser.parse_args()


def _default_ab_output_path(jsonl_path: str) -> str:
    stem = os.path.splitext(os.path.basename(jsonl_path))[0]
    parent = os.path.dirname(os.path.abspath(jsonl_path)) or "."
    return os.path.join(parent, f"{stem}_mcts_ab_figure")


if __name__ == "__main__":
    args = parse_args()
    if args.fine_routing_jsonl:
        if not os.path.isfile(args.fine_routing_jsonl):
            print(f"ERROR: file not found: {args.fine_routing_jsonl}")
            raise SystemExit(1)
        df = load_fine_routing_mcts_jsonl(args.fine_routing_jsonl)
        bench = args.bench_display
        if not bench:
            bid = df["benchmark_id"].dropna().iloc[0] if "benchmark_id" in df.columns and not df["benchmark_id"].isna().all() else None
            bench = _shorten_benchmark(str(bid)) if bid else "Benchmark"
        out_base = args.output_ab or _default_ab_output_path(args.fine_routing_jsonl)
        rs = _recompute_stats_from_df(df)
        print(f"Loaded {args.fine_routing_jsonl}: n={rs['total']}, base_acc={rs['orig_acc']*100:.2f}%, "
              f"MCTS_upper={100*(rs['n_correct']+rs['n_improved'])/max(1,rs['total']):.2f}%")
        create_fine_routing_mcts_ab_figure(df, bench, out_base)
        print(f"\nfile://{os.path.abspath(out_base)}.png")
    elif args.data_dir:
        generate_all_plots(data_dir=args.data_dir, output_dir=args.output_dir, num_layers=args.num_layers)
    else:
        print("ERROR: pass either --data_dir (DART) or --fine_routing_jsonl (MCTS AB figure).")
        raise SystemExit(2)
