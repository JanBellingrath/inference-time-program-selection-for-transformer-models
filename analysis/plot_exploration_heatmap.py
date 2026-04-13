"""
Plot a position × module heatmap of MCTS exploration.

Shows how often each module was placed at each position across explored sequences.
Baseline (identity) is position i -> module i; deviations show where the search
tried alternate orderings.

Usage:
    # From a snapshot that includes explored_sequences (saved by benchmark_mcts from this version)
    python plot_exploration_heatmap.py --snapshot predictions/benchmark_mcts_dart-1_20260209_190241_snapshot.json

    # Use latest snapshot in a directory
    python plot_exploration_heatmap.py --snapshot predictions/ --run dart-1

    # Fallback: if snapshot has no explored_sequences, uses validated + tier3 sequences only
"""
import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import json
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

SKIP = -1
# Restricted search: per-position choice 0=skip, 1=execute, 2=repeat
SKIP_LAYER = 0
EXECUTE = 1
REPEAT = 2


def load_sequences(snapshot: dict) -> Tuple[List[List[int]], bool]:
    """Get list of sequences from snapshot. Prefer explored_sequences; else validated + tier3."""
    if "explored_sequences" in snapshot:
        return snapshot["explored_sequences"], True
    seqs = []
    for r in snapshot.get("validated", []):
        seqs.append(r["seq"])
    for r in snapshot.get("tier3", []):
        seqs.append(r["seq"])
    return seqs, False


def build_count_matrix(sequences: List[List[int]], num_layers: int) -> np.ndarray:
    """Count matrix: [position, module]. Last column is SKIP."""
    # rows = position 0..n-1, cols = module 0..n-1, then SKIP
    M = np.zeros((num_layers, num_layers + 1), dtype=np.int64)
    for seq in sequences:
        if len(seq) != num_layers:
            continue
        for pos, mod in enumerate(seq):
            if mod == SKIP:
                M[pos, num_layers] += 1
            elif 0 <= mod < num_layers:
                M[pos, mod] += 1
    return M


def build_delta_matrix(
    sequences: List[List[int]],
    correct: List[int],
    total: List[int],
    baseline: float,
    num_layers: int,
) -> np.ndarray:
    """Mean accuracy delta per (position, module). Last column is SKIP. Unvisited = nan."""
    # sum of deltas and count per (pos, mod)
    sum_delta = np.zeros((num_layers, num_layers + 1))
    count = np.zeros((num_layers, num_layers + 1))
    for seq, c, t in zip(sequences, correct, total):
        if len(seq) != num_layers or t == 0:
            continue
        acc = c / t
        delta = acc - baseline
        for pos, mod in enumerate(seq):
            if mod == SKIP:
                sum_delta[pos, num_layers] += delta
                count[pos, num_layers] += 1
            elif 0 <= mod < num_layers:
                sum_delta[pos, mod] += delta
                count[pos, mod] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_delta = np.where(count > 0, sum_delta / count, np.nan)
    return mean_delta


def build_count_matrix_restricted(sequences: List[List[int]], num_layers: int) -> np.ndarray:
    """Count matrix for restricted space: [position, choice]. choice 0=skip, 1=execute, 2=repeat."""
    M = np.zeros((num_layers, 3), dtype=np.int64)
    for seq in sequences:
        if len(seq) != num_layers:
            continue
        for pos, c in enumerate(seq):
            if c in (SKIP_LAYER, EXECUTE, REPEAT):
                M[pos, c] += 1
    return M


def build_delta_matrix_restricted(
    sequences: List[List[int]],
    correct: List[int],
    total: List[int],
    baseline: float,
    num_layers: int,
) -> np.ndarray:
    """Mean accuracy delta per (position, choice). choice 0=skip, 1=execute, 2=repeat."""
    sum_delta = np.zeros((num_layers, 3))
    count = np.zeros((num_layers, 3))
    for seq, c, t in zip(sequences, correct, total):
        if len(seq) != num_layers or t == 0:
            continue
        acc = c / t
        delta = acc - baseline
        for pos, choice in enumerate(seq):
            if choice in (SKIP_LAYER, EXECUTE, REPEAT):
                sum_delta[pos, choice] += delta
                count[pos, choice] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_delta = np.where(count > 0, sum_delta / count, np.nan)
    return mean_delta


def plot_heatmap_restricted(M: np.ndarray, out_path: Path, title: str = "Restricted: position × choice") -> None:
    """Draw heatmap (num_layers, 3): skip, execute, repeat."""
    n, _ = M.shape
    plot_M_log = np.where(M > 0, M.astype(float), np.nan)
    vmin, vmax = 1, max(1, int(np.nanmax(plot_M_log)))
    import matplotlib.colors as mcolors
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=(5, max(8, n * 0.25)))
    sns.heatmap(
        plot_M_log,
        xticklabels=["skip", "execute", "repeat"],
        yticklabels=list(range(n)),
        ax=ax,
        cmap="YlOrRd",
        norm=norm,
        cbar_kws={"label": "# sequences (log scale)"},
        linewidths=0.5,
        linecolor="gray",
    )
    ax.set_xlabel("Choice")
    ax.set_ylabel("Layer index")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_heatmap_delta_restricted(
    M_delta: np.ndarray, out_path: Path, title: str = "Restricted: mean acc delta (position × choice)"
) -> None:
    """Draw delta heatmap (num_layers, 3)."""
    n = M_delta.shape[0]
    valid = np.isfinite(M_delta)
    if not np.any(valid):
        raise ValueError("No finite delta values to plot")
    vmin, vmax = np.nanmin(M_delta), np.nanmax(M_delta)
    lim = max(abs(vmin), abs(vmax), 1e-9)
    vmin, vmax = -lim, lim
    fig, ax = plt.subplots(figsize=(5, max(8, n * 0.25)))
    sns.heatmap(
        M_delta,
        xticklabels=["skip", "execute", "repeat"],
        yticklabels=list(range(n)),
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "mean(accuracy − baseline)"},
        linewidths=0.5,
        linecolor="gray",
    )
    ax.set_xlabel("Choice")
    ax.set_ylabel("Layer index")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_heatmap(
    M: np.ndarray,
    out_path: Path,
    title: str = "MCTS exploration: position × module",
    highlight_baseline: bool = True,
    show_skip: bool = True,
):
    """Draw heatmap. Rows = position, cols = module (optionally include SKIP column)."""
    n = M.shape[0]
    if show_skip and M.shape[1] > n:
        # plot modules 0..n-1 + SKIP
        plot_M = M.copy()
        col_labels = [str(i) for i in range(n)] + ["skip"]
    else:
        plot_M = M[:, :n]
        col_labels = [str(i) for i in range(n)]

    # Log scale so baseline diagonal doesn't dominate; 0 -> nan (blank)
    plot_M_log = np.where(plot_M > 0, plot_M.astype(float), np.nan)
    vmin = 1
    vmax = max(1, int(np.nanmax(plot_M_log)))
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.4), max(8, n * 0.35)))
    sns.heatmap(
        plot_M_log,
        xticklabels=col_labels,
        yticklabels=list(range(n)),
        ax=ax,
        cmap="YlOrRd",
        norm=norm,
        cbar_kws={"label": "# sequences (log scale)"},
        linewidths=0.5,
        linecolor="gray",
    )
    ax.set_xlabel("Module (layer index)")
    ax.set_ylabel("Position")
    ax.set_title(title)

    if highlight_baseline and plot_M.shape[1] > n - 1:
        # draw diagonal to show baseline (position i -> module i)
        for i in range(n):
            ax.add_patch(
                plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="blue", linewidth=2)
            )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_heatmap_delta(
    M_delta: np.ndarray,
    out_path: Path,
    title: str = "Mean accuracy delta: position × module",
    highlight_baseline: bool = True,
    show_skip: bool = True,
):
    """Draw heatmap of mean (accuracy - baseline). Diverging colormap, 0 = baseline."""
    n = M_delta.shape[0]
    if show_skip and M_delta.shape[1] > n:
        plot_M = M_delta.copy()
        col_labels = [str(i) for i in range(n)] + ["skip"]
    else:
        plot_M = M_delta[:, :n]
        col_labels = [str(i) for i in range(n)]

    valid = np.isfinite(plot_M)
    if not np.any(valid):
        raise ValueError("No finite delta values to plot")
    vmin = np.nanmin(plot_M)
    vmax = np.nanmax(plot_M)
    lim = max(abs(vmin), abs(vmax), 1e-9)
    vmin, vmax = -lim, lim  # symmetric around 0

    fig, ax = plt.subplots(figsize=(max(10, n * 0.4), max(8, n * 0.35)))
    sns.heatmap(
        plot_M,
        xticklabels=col_labels,
        yticklabels=list(range(n)),
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "mean(accuracy − baseline)"},
        linewidths=0.5,
        linecolor="gray",
    )
    ax.set_xlabel("Module (layer index)")
    ax.set_ylabel("Position")
    ax.set_title(title)

    if highlight_baseline and plot_M.shape[1] > n - 1:
        for i in range(n):
            ax.add_patch(
                plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="black", linewidth=2)
            )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def find_latest_snapshot(predictions_dir: Path, run_name: str) -> Optional[Path]:
    """Find most recent snapshot matching run (e.g. dart-1)."""
    pattern = f"benchmark_mcts_{run_name}_*_snapshot.json"
    candidates = list(predictions_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    ap = argparse.ArgumentParser(description="Plot position × module exploration heatmap")
    ap.add_argument(
        "--snapshot",
        type=str,
        required=True,
        help="Path to snapshot JSON, or directory containing snapshots",
    )
    ap.add_argument(
        "--run",
        type=str,
        default=None,
        help="If snapshot is a dir: run name (e.g. dart-1) to pick latest snapshot",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output figure path (default: predictions/exploration_heatmap_<run>_<sim>.png)",
    )
    ap.add_argument("--no-skip", action="store_true", help="Omit SKIP column from heatmap")
    ap.add_argument("--mode", type=str, default="count", choices=["count", "delta"],
                    help="count = # sequences; delta = mean(accuracy - baseline)")
    args = ap.parse_args()

    snapshot_path = Path(args.snapshot)
    if snapshot_path.is_dir():
        run = args.run or "dart-1"
        snapshot_path = find_latest_snapshot(snapshot_path, run)
        if snapshot_path is None:
            raise SystemExit(f"No snapshot found in {args.snapshot} for run {args.run}")
        print(f"Using latest snapshot: {snapshot_path}")

    with open(snapshot_path) as f:
        snapshot = json.load(f)

    sequences, full = load_sequences(snapshot)
    if not sequences:
        raise SystemExit("No sequences in snapshot (no explored_sequences, validated, or tier3).")

    num_layers = len(sequences[0])
    for s in sequences:
        if len(s) != num_layers:
            raise SystemExit(f"Inconsistent sequence length: {len(s)} vs {num_layers}")

    if not full:
        print("Note: snapshot has no 'explored_sequences'; using validated + tier3 only (partial view).")
        print("Re-run benchmark_mcts (current version) to save full exploration and re-plot.")

    sim = snapshot.get("sim", 0)
    run_label = snapshot_path.stem.replace("benchmark_mcts_", "").replace("_snapshot", "").split("_")[0]

    if args.output:
        out_path = Path(args.output)
    else:
        if args.mode == "delta":
            out_path = snapshot_path.parent / (snapshot_path.stem.replace("_snapshot", "") + "_heatmap_delta.png")
        else:
            out_path = snapshot_path.parent / f"exploration_heatmap_{run_label}_sim{sim}.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "delta":
        if "explored_correct" not in snapshot or "explored_total" not in snapshot:
            raise SystemExit(
                "Delta heatmap requires explored_correct and explored_total in snapshot. "
                "Re-run benchmark_mcts (current version) to save them."
            )
        baseline = snapshot["baseline"]
        correct = snapshot["explored_correct"]
        total = snapshot["explored_total"]
        if len(correct) != len(sequences) or len(total) != len(sequences):
            raise SystemExit("explored_correct/total length does not match explored_sequences")
        M_delta = build_delta_matrix(sequences, correct, total, baseline, num_layers)
        title = f"Mean accuracy delta: position × module (n={len(sequences)} seqs, sim={sim})"
        plot_heatmap_delta(
            M_delta,
            out_path,
            title=title,
            highlight_baseline=True,
            show_skip=not args.no_skip,
        )
    else:
        M = build_count_matrix(sequences, num_layers)
        title = f"MCTS exploration: position × module (n={len(sequences)} seqs, sim={sim})"
        plot_heatmap(
            M,
            out_path,
            title=title,
            highlight_baseline=True,
            show_skip=not args.no_skip,
        )


if __name__ == "__main__":
    main()
