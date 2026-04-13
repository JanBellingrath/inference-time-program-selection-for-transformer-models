#!/usr/bin/env python3
"""Publication-quality figures for fine-grained pivot routing results.

Generates Nature-style seaborn figures that disentangle:
  - Default model (standard layer order [0..L-1])
  - Benchmark-level MCTS anchor (optimised on train split)
  - Question-level fine routing (gate + router)

Reads from:
  - fine_routing_data_{benchmark}_mcts/{benchmark}.jsonl  (per-question MCTS data)
  - results_marginalization_{benchmark}.json               (strategy comparison)
  - predictions/aggregation_compare_{benchmark}_K5_*.json  (default baseline)

Usage:
    python analysis/plot_fine_routing_results.py [--output_dir figures/fine_routing]

Joint router bar (Fig. 1) reads ``routed_accuracy`` / ``n`` from
``experiments/eval_joint_router_downstream.py`` JSON (``--joint_eval_json``, repeatable).

``--joint_select_mode``:

* ``checkpoint`` (default): row whose ``checkpoint`` contains ``--joint_checkpoint``
  (default: ``run_tau01_lam10_beta01``) from ``eval_n2442_val_full_csqa.json`` unless you pass
  ``--joint_eval_json``. Fig.~1 **MCTS anchor** bar is then aligned to that row’s per-bench
  anchor accuracy and ``n`` (same eval as the joint bar).
* ``best_avg_gain``: among all rows in the given JSON files, pick highest mean per-bench gain
  vs anchor; default JSON list is ``results/joint_router_loss_ablation/eval*.json``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


def default_joint_eval_json_paths(root: Path) -> List[str]:
    """All ``eval*.json`` under ``results/joint_router_loss_ablation/`` (no run hardcoded)."""
    d = root / "results" / "joint_router_loss_ablation"
    if not d.is_dir():
        return []
    return sorted(str(p) for p in d.glob("eval*.json"))

# ═══════════════════════════════════════════════════════════════════════════
#  Nature-style theme
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

_viridis = plt.cm.viridis
PAL_BENCHMARK = {"BoolQ": _viridis(0.25), "CSQA": _viridis(0.70)}
PAL_LEVEL_FLAT = [_viridis(0.05), _viridis(0.35), _viridis(0.65), _viridis(0.92)]

MM = 1 / 25.4  # inches per mm
SINGLE_COL = 89 * MM   # Nature single-column width
DOUBLE_COL = 183 * MM


def apply_theme() -> None:
    sns.set_theme(style="ticks", rc=NATURE_RC)
    plt.rcParams.update(NATURE_RC)


def save_fig(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for ext in ("pdf", "png"):
        out = f"{path}.{ext}"
        fig.savefig(out)
        logger.info("Saved %s", out)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_jsonl(path: str, max_rows: int = 0) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
            if max_rows and len(records) >= max_rows:
                break
    return records


def load_json(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def find_aggregation_file(predictions_dir: str, benchmark: str) -> Optional[str]:
    import glob
    pattern = os.path.join(predictions_dir, f"aggregation_compare_{benchmark}_K5_*.json")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


BenchData = Dict[str, Any]


def load_benchmark_data(benchmark: str, data_dir: str, marg_path: str,
                        predictions_dir: str) -> BenchData:
    """Load all data for one benchmark into a unified dict."""
    jsonl_path = os.path.join(data_dir, f"{benchmark}.jsonl")
    records = load_jsonl(jsonl_path)

    df = pd.DataFrame([{
        "question_id": r["question_id"],
        "anchor_score": r["anchor_score"],
        "best_score": r["best_score"],
        "best_delta": r["best_delta"],
        "gate_label": r["gate_label"],
        "num_explored": r["num_explored"],
    } for r in records])

    marg = load_json(marg_path) if os.path.isfile(marg_path) else None

    agg_path = find_aggregation_file(predictions_dir, benchmark)
    agg = load_json(agg_path) if agg_path else None

    return {
        "benchmark": benchmark,
        "df": df,
        "marg": marg,
        "agg": agg,
        "n_records": len(records),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 1 — Accuracy Decomposition (default → anchor → routed)
# ═══════════════════════════════════════════════════════════════════════════

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _joint_eval_rows_from_file(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        d = json.load(f)
    rows = d if isinstance(d, list) else [d]
    out: List[Dict[str, Any]] = []
    for r in rows:
        if isinstance(r, dict) and "boolq/routed_accuracy" in r:
            out.append(r)
    return out


def collect_joint_eval_rows(json_paths: List[str]) -> List[Dict[str, Any]]:
    """All eval rows from downstream joint JSON files (dedupe by checkpoint + source path)."""
    all_rows: List[Dict[str, Any]] = []
    for p in json_paths:
        if not p or not os.path.isfile(p):
            logger.warning("Joint eval JSON missing or not a file: %s", p)
            continue
        for r in _joint_eval_rows_from_file(p):
            rr = dict(r)
            rr["_eval_json"] = p
            all_rows.append(rr)
    return all_rows


def avg_per_bench_gain_vs_anchor(row: Dict[str, Any]) -> float:
    """Mean of BoolQ and CSQA ``unconditional_gain`` (fraction), vs ckpt MCTS anchor each."""
    try:
        gb = float(row["boolq/unconditional_gain"])
        gc = float(row["commonsenseqa/unconditional_gain"])
    except (KeyError, TypeError, ValueError):
        return float("-inf")
    return (gb + gc) / 2.0


def select_joint_router_eval_row(
    json_paths: List[str],
    mode: str,
    checkpoint_match: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Pick one row from ``eval_joint_router_downstream.py`` JSON output(s).

    ``mode``:

    * ``best_avg_gain``: maximize ``avg_per_bench_gain_vs_anchor``; tie-break by larger
      pooled ``n``, then lexicographic ``checkpoint`` (deterministic).
    * ``checkpoint``: first row whose ``checkpoint`` contains ``checkpoint_match``.

    Baseline in that eval is per-bench **anchor** (MCTS seq. in ckpt); ``unconditional_gain``
    is vs that anchor.
    """
    all_rows = collect_joint_eval_rows(json_paths)
    if not all_rows:
        return None
    if mode == "checkpoint":
        m = (checkpoint_match or "").strip()
        if not m:
            logger.warning("joint_select_mode=checkpoint but --joint_checkpoint empty")
            return None
        for r in all_rows:
            if m in str(r.get("checkpoint", "")):
                return r
        logger.warning("No joint eval row with checkpoint containing %r", m)
        return None
    if mode != "best_avg_gain":
        logger.warning("Unknown joint_select_mode %r; using best_avg_gain", mode)
        mode = "best_avg_gain"

    def _key(r: Dict[str, Any]) -> Tuple[float, int, str]:
        return (
            avg_per_bench_gain_vs_anchor(r),
            int(r.get("n", 0)),
            str(r.get("checkpoint", "")),
        )

    return max(all_rows, key=_key)


def joint_row_to_per_bench(row: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Map eval row → plot keys BoolQ / CSQA."""
    bmap = (
        ("BoolQ", "boolq"),
        ("CSQA", "commonsenseqa"),
    )
    out: Dict[str, Dict[str, float]] = {}
    for label, prefix in bmap:
        out[label] = {
            "joint_routed_acc": float(row[f"{prefix}/routed_accuracy"]),
            "joint_anchor_acc": float(row[f"{prefix}/anchor_accuracy"]),
            "joint_n": int(row[f"{prefix}/n"]),
            "joint_gain_vs_anchor": float(row[f"{prefix}/unconditional_gain"]),
        }
    return out


def fig_accuracy_decomposition(
    data: Dict[str, BenchData],
    output_dir: str,
    joint_eval_row: Optional[Dict[str, Any]] = None,
) -> str:
    """Grouped bar chart: Default → MCTS anchor → Fine-routed → Joint router (routed).

    Default / fine-routed columns use the fixed ``RESULTS`` table (fine-routing paper numbers).
    When a joint eval row is present, **MCTS anchor** and **Joint router (routed)** use that
    row’s ``anchor_accuracy`` / ``routed_accuracy`` and per-bench ``n`` (same downstream eval).
    ``unconditional_gain`` in JSON is vs that anchor, not vs Default order or fine-routed.
    """

    RESULTS = {
        "BoolQ": {
            "default_acc": 0.625, "default_n": 1000,
            "anchor_acc": 0.655, "anchor_n": 3270,
            "routed_acc": 0.714, "routed_n": 3270,
        },
        "CSQA": {
            "default_acc": 0.428, "default_n": 1000,
            "anchor_acc": 0.514, "anchor_n": 1221,
            "routed_acc": 0.530, "routed_n": 1221,
        },
    }

    benchmarks = list(RESULTS.keys())
    joint_by_bm: Dict[str, Dict[str, float]] = {}
    if joint_eval_row is not None:
        joint_by_bm = joint_row_to_per_bench(joint_eval_row)
        for bm in benchmarks:
            j = joint_by_bm[bm]
            RESULTS[bm]["joint_acc"] = j["joint_routed_acc"]
            RESULTS[bm]["joint_n"] = int(j["joint_n"])
            # Same eval as joint bar: anchor acc / n match downstream eval (not paper table).
            RESULTS[bm]["anchor_acc"] = j["joint_anchor_acc"]
            RESULTS[bm]["anchor_n"] = int(j["joint_n"])
        ck = joint_eval_row.get("checkpoint", "")
        logger.info(
            "Fig1 MCTS anchor + joint from same eval: %s | BQ anchor %.4f routed %.4f (Δ %.2f pp) | "
            "CSQA anchor %.4f routed %.4f (Δ %.2f pp) | %s",
            ck,
            joint_by_bm["BoolQ"]["joint_anchor_acc"],
            joint_by_bm["BoolQ"]["joint_routed_acc"],
            joint_by_bm["BoolQ"]["joint_gain_vs_anchor"] * 100,
            joint_by_bm["CSQA"]["joint_anchor_acc"],
            joint_by_bm["CSQA"]["joint_routed_acc"],
            joint_by_bm["CSQA"]["joint_gain_vs_anchor"] * 100,
            joint_eval_row.get("_eval_json", "?"),
        )
    else:
        logger.warning("No joint eval row; Joint router bar omitted")
        joint_by_bm = {}

    all_levels = ["Default order", "MCTS anchor", "Fine-routed", "Joint router (routed)"]
    levels_plot = all_levels if joint_by_bm else all_levels[:3]
    n_levels = len(levels_plot)
    x = np.arange(len(benchmarks))
    width = 0.18
    offsets = (np.arange(n_levels) - (n_levels - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.78))

    bar_tops = {}  # bm_idx -> {j: top_of_error_bar}
    palette = PAL_LEVEL_FLAT[:n_levels]
    for j, (level, colour) in enumerate(zip(levels_plot, palette)):
        vals, errs_lo, errs_hi = [], [], []
        for bm in benchmarks:
            r = RESULTS[bm]
            if j == 0:
                acc, n = r["default_acc"], r["default_n"]
            elif j == 1:
                acc, n = r["anchor_acc"], r["anchor_n"]
            elif j == 2:
                acc, n = r["routed_acc"], r["routed_n"]
            else:
                acc, n = r["joint_acc"], r["joint_n"]
            lo, hi = wilson_ci(int(acc * n), n)
            vals.append(acc * 100)
            errs_lo.append((acc - lo) * 100)
            errs_hi.append((hi - acc) * 100)

        bars = ax.bar(
            x + offsets[j], vals, width,
            color=colour, label=level, edgecolor="white", linewidth=0.3,
            yerr=[errs_lo, errs_hi], capsize=1.5,
            error_kw={"linewidth": 0.5, "capthick": 0.4},
            zorder=3,
        )
        for bi, (bar, v, ehi) in enumerate(zip(bars, vals, errs_hi)):
            top = v + ehi
            bar_tops.setdefault(bi, {})[j] = top
            ax.text(
                bar.get_x() + bar.get_width() / 2, top + 0.6,
                f"{v:.1f}", ha="center", va="bottom", fontsize=5,
                fontweight="bold",
            )

    for bm_idx, bm in enumerate(benchmarks):
        r = RESULTS[bm]
        top_acc = r["joint_acc"] if joint_by_bm else r["routed_acc"]
        total = (top_acc - r["default_acc"]) * 100

        highest_label = max(bar_tops[bm_idx].values()) + 4.5
        arrow_y = highest_label + 1.0
        left_x = x[bm_idx] + offsets[0] + width / 2
        right_x = x[bm_idx] + offsets[n_levels - 1] + width / 2

        ax.annotate(
            "", xy=(left_x, arrow_y), xytext=(right_x, arrow_y),
            arrowprops=dict(
                arrowstyle="<->", color="#555555", lw=0.6,
                shrinkA=0, shrinkB=0,
            ),
        )
        ax.text(
            x[bm_idx], arrow_y + 0.8,
            f"+{total:.1f} pp",
            ha="center", va="bottom", fontsize=5.5, color="#333333",
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 95)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f"))
    ax.legend(loc="upper right", ncol=1, fontsize=5.5)
    ax.set_title("a", fontsize=9, fontweight="bold", loc="left", pad=4)
    sns.despine(ax=ax)

    path = os.path.join(output_dir, "fig1_accuracy_decomposition")
    save_fig(fig, path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 2 — Per-question MCTS delta distributions
# ═══════════════════════════════════════════════════════════════════════════

def fig_delta_distributions(data: Dict[str, BenchData], output_dir: str) -> str:
    """Split violin + strip: per-question best_delta from MCTS search."""

    frames = []
    for key, bd in data.items():
        df = bd["df"].copy()
        label = "BoolQ" if "boolq" in key else "CSQA"
        df["Benchmark"] = label
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, SINGLE_COL * 0.65),
                             gridspec_kw={"width_ratios": [3, 1.3]})

    # Left: histogram of best_delta, split by gate label
    ax = axes[0]
    bins = np.linspace(-0.5, 5.0, 55)
    for bm, colour in PAL_BENCHMARK.items():
        subset = combined[combined["Benchmark"] == bm]
        gate_pos = subset[subset["gate_label"] == 1]["best_delta"]
        gate_neg = subset[subset["gate_label"] == 0]["best_delta"]

        ax.hist(gate_pos, bins=bins, color=colour, alpha=0.35, density=True,
                label=f"{bm} gate+ (n={len(gate_pos)})")
        sns.kdeplot(gate_pos, ax=ax, color=colour, linewidth=1.0,
                    clip=(-0.5, 8), bw_adjust=0.6)

    ax.axvline(0, color="grey", linewidth=0.4, linestyle=":")
    median_boolq = combined[
        (combined["Benchmark"] == "BoolQ") & (combined["gate_label"] == 1)
    ]["best_delta"].median()
    median_csqa = combined[
        (combined["Benchmark"] == "CSQA") & (combined["gate_label"] == 1)
    ]["best_delta"].median()
    ax.axvline(median_boolq, color=PAL_BENCHMARK["BoolQ"], linewidth=0.6,
               linestyle="--", alpha=0.7)
    ax.axvline(median_csqa, color=PAL_BENCHMARK["CSQA"], linewidth=0.6,
               linestyle="--", alpha=0.7)
    ax.text(median_boolq + 0.08, ax.get_ylim()[1] * 0.85,
            f"med={median_boolq:.2f}", fontsize=4.5,
            color=PAL_BENCHMARK["BoolQ"], rotation=0)
    ax.text(median_csqa + 0.08, ax.get_ylim()[1] * 0.75,
            f"med={median_csqa:.2f}", fontsize=4.5,
            color=PAL_BENCHMARK["CSQA"], rotation=0)

    ax.set_xlabel("Best per-question MCTS delta (nats)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", fontsize=5)
    ax.set_title("b", fontsize=9, fontweight="bold", loc="left", pad=4)
    ax.set_xlim(-0.5, 5.0)
    sns.despine(ax=ax)

    # Right: gate-positive rates
    ax2 = axes[1]
    gate_rates = []
    for key, bd in data.items():
        df = bd["df"]
        label = "BoolQ" if "boolq" in key else "CSQA"
        rate = df["gate_label"].mean() * 100
        gate_rates.append({"Benchmark": label, "Gate+ rate (%)": rate})
    gr_df = pd.DataFrame(gate_rates)

    bars = ax2.barh(
        gr_df["Benchmark"], gr_df["Gate+ rate (%)"],
        color=[PAL_BENCHMARK[b] for b in gr_df["Benchmark"]],
        height=0.5, zorder=3,
    )
    for bar, v in zip(bars, gr_df["Gate+ rate (%)"]):
        ax2.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{v:.1f}%", va="center", fontsize=5.5,
        )
    ax2.set_xlabel("Gate-positive (%)")
    ax2.set_xlim(0, 100)
    ax2.set_title("c", fontsize=9, fontweight="bold", loc="left", pad=4)
    sns.despine(ax=ax2)

    fig.tight_layout(w_pad=3)
    path = os.path.join(output_dir, "fig2_delta_distributions")
    save_fig(fig, path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 3 — Strategy comparison from marginalization
# ═══════════════════════════════════════════════════════════════════════════

def _parse_strategy(name: str) -> Tuple[str, int]:
    """Split 'conf-wtd-8' → ('conf-wtd', 8)."""
    parts = name.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    return name, 0


STRATEGY_ORDER = [
    "uniform", "consist", "conf-wtd", "margin-wtd",
    "router-wtd", "router-T0.5",
    "oracle-accuracy",
]
STRATEGY_LABELS = {
    "uniform": "Uniform",
    "consist": "Consistency",
    "conf-wtd": "Conf-wtd",
    "margin-wtd": "Margin-wtd",
    "router-wtd": "Router-wtd",
    "router-T0.5": "Router T=0.5",
    "oracle-accuracy": "Oracle",
}


def fig_strategy_comparison(data: Dict[str, BenchData], output_dir: str) -> str:
    """Accuracy delta vs anchor for key strategies at beam widths 2,4,8."""

    rows = []
    for key, bd in data.items():
        marg = bd.get("marg")
        if not marg:
            continue
        bm = "BoolQ" if "boolq" in key else "CSQA"
        for strat_name, strat_data in marg["strategies"].items():
            base, k = _parse_strategy(strat_name)
            if base not in STRATEGY_ORDER or k not in (4, 8):
                continue
            rows.append({
                "Benchmark": bm,
                "Strategy": STRATEGY_LABELS.get(base, base),
                "K": k,
                "Accuracy (%)": strat_data["accuracy"] * 100,
                "Δ vs anchor (pp)": strat_data["accuracy_delta_vs_anchor"] * 100,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No marginalization data found; skipping fig3")
        return ""

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.4, SINGLE_COL * 0.85))

    sub_boolq = df[(df["Benchmark"] == "BoolQ") & (df["K"] == 8)].copy()
    sub_csqa = df[(df["Benchmark"] == "CSQA") & (df["K"] == 8)].copy()

    strat_order = list(STRATEGY_LABELS.values())
    strat_map = {v: i for i, v in enumerate(strat_order)}

    y_pos = np.arange(len(strat_order))

    for sub, bm, colour, offset in [
        (sub_boolq, "BoolQ", PAL_BENCHMARK["BoolQ"], -0.12),
        (sub_csqa, "CSQA", PAL_BENCHMARK["CSQA"], 0.12),
    ]:
        vals = []
        for s in strat_order:
            row = sub[sub["Strategy"] == s]
            vals.append(row["Accuracy (%)"].values[0] if not row.empty else np.nan)
        vals = np.array(vals)
        ax.barh(
            y_pos + offset, vals, height=0.22, color=colour,
            label=bm, alpha=0.8, zorder=3,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(strat_order, fontsize=5.5)
    ax.set_xlabel("Accuracy (%, K=8 beam candidates)")
    ax.invert_yaxis()
    ax.legend(fontsize=5.5, loc="lower right")
    ax.set_title("d", fontsize=9, fontweight="bold", loc="left", pad=4)
    sns.despine(ax=ax)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig3_strategy_comparison")
    save_fig(fig, path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 4 — Oracle gap waterfall
# ═══════════════════════════════════════════════════════════════════════════

def fig_oracle_gap(data: Dict[str, BenchData], output_dir: str) -> str:
    """Waterfall showing: Default → +bench-MCTS → +q-routing → oracle ceiling."""

    results = {
        "BoolQ": {
            "default": 62.5, "anchor": 65.5, "routed": 71.4,
        },
        "CSQA": {
            "default": 42.8, "anchor": 51.4, "routed": 53.0,
        },
    }

    oracle_acc = {}
    for key, bd in data.items():
        marg = bd.get("marg")
        bm = "BoolQ" if "boolq" in key else "CSQA"
        if marg:
            oracle_k8 = marg["strategies"].get("oracle-accuracy-8", {})
            oracle_acc[bm] = oracle_k8.get("accuracy", 0) * 100
        else:
            oracle_acc[bm] = results[bm]["routed"] + 5

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, SINGLE_COL * 0.55),
                             sharey=False)

    for idx, (bm, colour) in enumerate(PAL_BENCHMARK.items()):
        ax = axes[idx]
        r = results[bm]
        orc = oracle_acc.get(bm, r["routed"] + 5)

        segments = [
            ("Default", r["default"], 0),
            ("+Bench\nMCTS", r["anchor"] - r["default"], r["default"]),
            ("+Q-level\nrouting", r["routed"] - r["anchor"], r["anchor"]),
            ("Remaining\ngap", orc - r["routed"], r["routed"]),
        ]
        colours = ["#94A3B8", "#3B82F6", "#10B981", "#F59E0B"]

        x_pos = np.arange(len(segments))
        for i, (label, height, bottom) in enumerate(segments):
            if i == 0:
                ax.bar(x_pos[i], height, bottom=0, color=colours[i],
                       width=0.6, zorder=3)
                ax.text(x_pos[i], height / 2, f"{height:.1f}%",
                        ha="center", va="center", fontsize=5, fontweight="bold",
                        color="white")
            else:
                ax.bar(x_pos[i], height, bottom=bottom, color=colours[i],
                       width=0.6, zorder=3, alpha=0.85)
                ax.text(x_pos[i], bottom + height / 2,
                        f"+{height:.1f}pp",
                        ha="center", va="center", fontsize=5, fontweight="bold")

        ax.axhline(orc, color="#F59E0B", linewidth=0.5, linestyle="--", alpha=0.7)
        ax.text(len(segments) - 0.5, orc + 0.8, f"Oracle: {orc:.1f}%",
                fontsize=4.5, color="#B45309", ha="right")

        ax.set_xticks(x_pos)
        ax.set_xticklabels([s[0] for s in segments], fontsize=5)
        ax.set_ylabel("Accuracy (%)" if idx == 0 else "")
        ax.set_title(bm, fontsize=7, fontweight="bold")
        ax.set_ylim(0, max(orc + 8, r["routed"] + 15))
        sns.despine(ax=ax)

    fig.suptitle("e", fontsize=9, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, "fig4_oracle_gap_waterfall")
    save_fig(fig, path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 5 — Search effort vs improvement (per-question scatter)
# ═══════════════════════════════════════════════════════════════════════════

def fig_search_effort(data: Dict[str, BenchData], output_dir: str) -> str:
    """Hex-bin: per-question anchor_score vs best_delta, coloured by benchmark."""

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, SINGLE_COL * 0.6),
                             sharey=True)

    for idx, (key, bd) in enumerate(data.items()):
        ax = axes[idx]
        df = bd["df"]
        bm = "BoolQ" if "boolq" in key else "CSQA"
        colour = PAL_BENCHMARK[bm]

        gate_pos = df[df["gate_label"] == 1]
        gate_neg = df[df["gate_label"] == 0]

        ax.scatter(
            gate_neg["anchor_score"], gate_neg["best_delta"],
            s=0.8, alpha=0.15, color="#CBD5E1", rasterized=True,
            label=f"gate\u2013 (n={len(gate_neg)})",
        )
        ax.scatter(
            gate_pos["anchor_score"], gate_pos["best_delta"],
            s=0.8, alpha=0.25, color=colour, rasterized=True,
            label=f"gate+ (n={len(gate_pos)})",
        )

        ax.axhline(0, color="grey", linewidth=0.3, linestyle=":")
        ax.set_xlabel("Anchor log-prob (nats)")
        if idx == 0:
            ax.set_ylabel("Best MCTS delta (nats)")
        ax.set_title(bm, fontsize=7, fontweight="bold")
        ax.legend(fontsize=5, loc="upper left", markerscale=4)
        sns.despine(ax=ax)

    fig.suptitle("f", fontsize=9, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, "fig5_anchor_score_vs_delta")
    save_fig(fig, path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  CLI & orchestration
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", default="figures/fine_routing",
                   help="Directory for output figures (default: figures/fine_routing)")
    p.add_argument("--root", default=str(ROOT),
                   help="Project root (default: auto-detected)")
    p.add_argument(
        "--joint_eval_json",
        action="append",
        default=None,
        metavar="PATH",
        help="Downstream joint eval JSON (repeatable). Default: eval_n2442 (checkpoint mode) "
        "or eval*.json (best_avg_gain).",
    )
    p.add_argument(
        "--joint_select_mode",
        choices=("best_avg_gain", "checkpoint"),
        default="checkpoint",
        help="How to pick the joint row for Fig.1 (default: checkpoint).",
    )
    p.add_argument(
        "--joint_checkpoint",
        type=str,
        default="run_tau01_lam10_beta01",
        metavar="SUBSTRING",
        help="Substring of checkpoint path when --joint_select_mode=checkpoint (default: tau01 run).",
    )
    args = p.parse_args()
    if args.joint_select_mode == "checkpoint" and not (args.joint_checkpoint or "").strip():
        p.error("--joint_checkpoint cannot be empty when --joint_select_mode=checkpoint")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")
    args = parse_args()
    root = Path(args.root)
    out_dir = os.path.join(root, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    apply_theme()

    benchmarks = {
        "boolq": {
            "data_dir": str(root / "fine_routing_data_boolq_mcts"),
            "marg_path": str(root / "results_marginalization_boolq.json"),
        },
        "commonsenseqa": {
            "data_dir": str(root / "fine_routing_data_commonsenseqa_mcts"),
            "marg_path": str(root / "results_marginalization_csqa.json"),
        },
    }

    logger.info("Loading data ...")
    data: Dict[str, BenchData] = {}
    for bm, paths in benchmarks.items():
        logger.info("  %s ...", bm)
        data[bm] = load_benchmark_data(
            benchmark=bm,
            data_dir=paths["data_dir"],
            marg_path=paths["marg_path"],
            predictions_dir=str(root / "predictions"),
        )
        logger.info("    %d records loaded", data[bm]["n_records"])

    logger.info("Generating figures ...")
    paths = []

    joint_paths: List[str] = list(args.joint_eval_json or [])
    if not joint_paths:
        if args.joint_select_mode == "best_avg_gain":
            joint_paths = default_joint_eval_json_paths(root)
            if not joint_paths:
                logger.warning(
                    "No --joint_eval_json and no results/joint_router_loss_ablation/eval*.json; "
                    "joint bar skipped",
                )
        else:
            p2442 = root / "results/joint_router_loss_ablation/eval_n2442_val_full_csqa.json"
            if p2442.is_file():
                joint_paths = [str(p2442)]
            else:
                logger.warning("Default joint eval missing: %s", p2442)

    ck_match = (args.joint_checkpoint or "").strip()
    joint_row = select_joint_router_eval_row(
        joint_paths,
        mode=args.joint_select_mode,
        checkpoint_match=ck_match if args.joint_select_mode == "checkpoint" else None,
    )
    if joint_row is not None:
        logger.info(
            "Joint select_mode=%s | avg_per_bench_gain=%.5f | n=%s | %s",
            args.joint_select_mode,
            avg_per_bench_gain_vs_anchor(joint_row),
            joint_row.get("n"),
            joint_row.get("checkpoint", ""),
        )

    p = fig_accuracy_decomposition(data, out_dir, joint_eval_row=joint_row)
    paths.append(p)
    logger.info("  Fig 1: %s", p)

    p = fig_delta_distributions(data, out_dir)
    paths.append(p)
    logger.info("  Fig 2: %s", p)

    p = fig_strategy_comparison(data, out_dir)
    if p:
        paths.append(p)
        logger.info("  Fig 3: %s", p)

    p = fig_oracle_gap(data, out_dir)
    paths.append(p)
    logger.info("  Fig 4: %s", p)

    p = fig_search_effort(data, out_dir)
    paths.append(p)
    logger.info("  Fig 5: %s", p)

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
