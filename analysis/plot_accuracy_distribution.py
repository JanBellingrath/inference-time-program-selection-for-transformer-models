"""
Plot distribution of accuracy relative to baseline (delta) from MCTS snapshot.

Uses explored_correct/explored_total (noisy eval) when present;
otherwise uses validated + tier3 deltas only.

Usage:
    python plot_accuracy_distribution.py --snapshot predictions/ --run dart-1
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


def load_final_val(path: Path) -> Tuple[List[float], List[float]] | None:
    """Load final_val JSON. Returns (deltas, accuracies) or None."""
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return None
    deltas, accuracies = [], []
    if "baseline" in d and "candidates" in d:
        bl_acc = d["baseline"]["accuracy"]
        for c in d["candidates"]:
            acc = c.get("accuracy", 0)
            deltas.append(c.get("delta", acc - bl_acc))
            accuracies.append(acc)
    elif "baseline_acc" in d and "results" in d:
        bl_acc = d["baseline_acc"]
        for r in d["results"]:
            acc = r.get("permuted_acc", r.get("accuracy", 0))
            deltas.append(r.get("delta", acc - bl_acc))
            accuracies.append(acc)
    else:
        return None
    return (deltas, accuracies) if deltas else None


def load_deltas(snapshot: dict, final_val_data: Tuple[List[float], List[float]] | None = None) -> Tuple[List[float], List[float], bool, str]:
    """Returns (deltas, accuracies, from_explored, label)."""
    baseline = snapshot["baseline"]
    deltas, accuracies = [], []
    has_explored = False
    if "explored_correct" in snapshot and "explored_total" in snapshot:
        correct = snapshot["explored_correct"]
        total = snapshot["explored_total"]
        accs = [c / t if t > 0 else 0.0 for c, t in zip(correct, total)]
        mins = snapshot.get("explored_min_n_for_plot", 1)
        for i, t in enumerate(total):
            if t >= mins:
                deltas.append(accs[i] - baseline)
                accuracies.append(accs[i])
        has_explored = bool(deltas)
    for r in snapshot.get("validated", []):
        deltas.append(r["delta"])
        accuracies.append(r["accuracy"])
    for r in snapshot.get("tier3", []):
        deltas.append(r["delta"])
        accuracies.append(r["accuracy"])
    for r in snapshot.get("tier4", []):
        deltas.append(r["delta"])
        accuracies.append(r["accuracy"])
    if final_val_data:
        fv_d, fv_a = final_val_data
        deltas.extend(fv_d)
        accuracies.extend(fv_a)
    if final_val_data:
        label = "explored + tier2-4 + final_val"
    elif has_explored:
        label = "explored + validated/tier2-4"
    else:
        label = "validated/tier2-4"
    return deltas, accuracies, has_explored, label


def main():
    ap = argparse.ArgumentParser(description="Plot distribution of accuracy delta vs baseline")
    ap.add_argument("--snapshot", type=str, required=True, help="Snapshot JSON or directory")
    ap.add_argument("--final-val", type=str, default=None,
                    help="Path to final_val JSON to include (validate_candidates or winogrande format)")
    ap.add_argument("--run", type=str, default=None, help="If snapshot is dir: run name (e.g. dart-1)")
    ap.add_argument("--output", type=str, default=None, help="Output figure path")
    ap.add_argument("--min-n", type=int, default=5, help="Min samples for explored deltas (default 5)")
    ap.add_argument("--bins", type=int, default=40, help="Histogram bins")
    args = ap.parse_args()

    snapshot_path = Path(args.snapshot)
    if snapshot_path.is_dir():
        run = args.run or "dart-1"
        pattern = f"benchmark_mcts_{run}_*_snapshot.json"
        candidates = list(snapshot_path.glob(pattern))
        if not candidates:
            raise SystemExit(f"No snapshot found in {args.snapshot} for run {run}")
        snapshot_path = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"Using: {snapshot_path}")

    with open(snapshot_path) as f:
        snapshot = json.load(f)

    snapshot["explored_min_n_for_plot"] = args.min_n
    fv_data = None
    if args.final_val:
        p = Path(args.final_val)
        if p.exists():
            fv_data = load_final_val(p)
            if fv_data:
                print(f"Including final_val from {p} ({len(fv_data[0])} points)")
    deltas, accuracies, from_explored, source_label = load_deltas(snapshot, final_val_data=fv_data)
    if not deltas:
        raise SystemExit("No delta values in snapshot.")

    baseline = snapshot["baseline"]
    sim = snapshot.get("sim", 0)
    n_seqs = len(deltas)
    if not from_explored:
        print("Note: no explored_correct/explored_total; using validated+tier3 only (partial).")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(deltas, bins=args.bins, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5, label="baseline")
    ax.set_xlabel("Accuracy - baseline (delta)")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of delta (n={n_seqs} sequences, {source_label})")
    ax.legend()

    ax = axes[1]
    ax.hist(accuracies, bins=args.bins, color="coral", edgecolor="white", alpha=0.8)
    ax.axvline(baseline, color="black", linestyle="--", linewidth=1.5, label=f"baseline={baseline:.3f}")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of accuracy (sim={sim})")
    ax.legend()

    plt.suptitle(f"MCTS snapshot: {snapshot_path.name}", fontsize=11)
    plt.tight_layout()

    stem = snapshot_path.stem.replace("benchmark_mcts_", "").replace("_snapshot", "")
    out = Path(args.output) if args.output else snapshot_path.parent / f"accuracy_distribution_{stem}_sim{sim}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
