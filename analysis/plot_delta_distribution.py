"""Plot delta distribution and delta vs n from benchmark MCTS snapshots."""
import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json, argparse, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BENCHMARK_DISPLAY_NAMES = {
    "winogrande": "WinoGrande",
    "bigbench_all": "BigBench",
    "boolq": "BoolQ",
    "commonsenseqa": "CommonsenseQA",
    "mmlu_all": "MMLU",
    "arc_challenge": "ARC-Challenge",
    "arc_easy": "ARC-Easy",
    "gsm8k_hard": "GSM8K",
}


def _format_benchmark_title(stem: str) -> str:
    """Strip timestamp from stem and return display name with correct casing."""
    base = re.sub(r"_\d{8}-\d{6}$", "", stem)
    return BENCHMARK_DISPLAY_NAMES.get(base, base.replace("_", " ").title())


def _tier_n(r): return int(r.get("evaluated", r.get("total", 0)) or 0)

def load_final_val(path: Path) -> tuple[list, list] | None:
    """Load final_val / validate_candidates JSON. Returns (deltas, ns) or None."""
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return None
    deltas, ns = [], []
    # validate_candidates format: baseline{}, candidates[{accuracy,total,delta}]
    if "baseline" in d and "candidates" in d:
        bl_acc = d["baseline"]["accuracy"]
        for c in d["candidates"]:
            n = c.get("total", 0)
            if n > 0:
                delta = c.get("delta", c.get("accuracy", 0) - bl_acc)
                deltas.append(delta)
                ns.append(n)
    # winogrande final_val format: baseline_acc, results[{permuted_acc,permuted_total,delta}]
    elif "baseline_acc" in d and "results" in d:
        for r in d["results"]:
            n = r.get("permuted_total", r.get("total", 0))
            if n > 0:
                delta = r.get("delta", 0)
                deltas.append(delta)
                ns.append(n)
    else:
        return None
    return (deltas, ns) if deltas else None

def load_tier_from_log(log_path: Path) -> tuple[list, list]:
    """Parse TIER3= and TIER4= lines from benchmark_mcts log. Returns (deltas, ns)."""
    import re
    if not log_path.exists():
        return [], []
    text = log_path.read_text()
    deltas, ns = [], []
    for m in re.finditer(r'TIER[34]=[\d.]+.*?delta=([+-][\d.]+).*?n=(\d+)', text):
        deltas.append(float(m.group(1)))
        ns.append(int(m.group(2)))
    return deltas, ns


def load_deltas_and_n(snapshot, min_n=5, final_val_data=None, augment_log_path=None):
    baseline = snapshot["baseline"]
    deltas, ns = [], []
    for seq, c, t in zip(
        snapshot.get("explored_sequences", []),
        snapshot.get("explored_correct", []),
        snapshot.get("explored_total", []),
    ):
        if t and t >= min_n:
            deltas.append((c / t if t else 0) - baseline)
            ns.append(int(t))
    if augment_log_path:
        log_d, log_n = load_tier_from_log(Path(augment_log_path))
        deltas = deltas + log_d
        ns = ns + log_n
        tiers_from_snap = ("validated",)  # skip tier3/tier4 from snapshot, use log
    else:
        tiers_from_snap = ("validated", "tier3", "tier4")
    for tier in tiers_from_snap:
        for r in snapshot.get(tier, []):
            if "delta" in r and _tier_n(r) > 0:
                deltas.append(r["delta"])
                ns.append(_tier_n(r))
    fv_deltas, fv_ns = [], []
    if final_val_data:
        fv_deltas, fv_ns = final_val_data
        deltas = deltas + fv_deltas
        ns = ns + fv_ns
    return deltas, ns, baseline, (fv_deltas, fv_ns)

def plot_delta_vs_n(deltas, ns, baseline, title, output_path, final_val_data=None):
    if not deltas or not ns: return
    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 11
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    fv_deltas, fv_ns = final_val_data or ([], [])
    n_fv = len(fv_deltas)
    mcts_d = deltas[:-n_fv] if n_fv else deltas
    mcts_n = ns[:-n_fv] if n_fv else ns
    if mcts_d:
        ax.scatter(mcts_n, mcts_d, alpha=0.5, s=20, c="#4a90d9", label="MCTS (explored/tier2-4)", zorder=2)
    if fv_deltas and fv_ns:
        ax.scatter(fv_ns, fv_deltas, alpha=0.95, s=120, c="#e85d75", marker="*",
                   edgecolors="#b91c3c", linewidths=1.5, label=f"final_val (n={fv_ns[0] if fv_ns else 0})", zorder=3)
    ax.axhline(0, color="#333333", linestyle="--", linewidth=1.5, zorder=1)
    ax.set_xlabel("Evaluation size (n samples)", fontsize=12)
    ax.set_ylabel("Accuracy − baseline (delta)", fontsize=12)
    ax.set_title(title or f"Delta vs n (n_evals={len(deltas)})", fontsize=14, fontweight="medium")
    ax.tick_params(axis="both", labelsize=10)
    if fv_deltas:
        ax.legend(loc="best", fontsize=10, framealpha=0.95)
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=str, nargs="+", required=True)
    ap.add_argument("--final-val", type=str, default=None,
                    help="Path to final_val JSON to overlay (validate_candidates or winogrande format)")
    ap.add_argument("--augment-log", type=str, default=None,
                    help="Path to benchmark_mcts log to parse TIER3/TIER4 from (e.g. GSM8K)")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--plot-by-n", action="store_true")
    args = ap.parse_args()
    fv_data = None
    if args.final_val:
        p = Path(args.final_val)
        if p.exists():
            fv_data = load_final_val(p)
            if fv_data:
                print(f"Including final_val from {p} ({len(fv_data[0])} points)")
    augment_log = Path(args.augment_log) if args.augment_log else None
    if augment_log and augment_log.exists():
        d, n = load_tier_from_log(augment_log)
        if d:
            print(f"Augmenting from log {augment_log} ({len(d)} tier3/4 points)")
    for p in args.snapshot:
        path = Path(p)
        if not path.exists(): continue
        with open(path) as f: snap = json.load(f)
        deltas, ns, bl, fv = load_deltas_and_n(snap, final_val_data=fv_data, augment_log_path=args.augment_log)
        if not deltas: continue
        stem = path.stem.replace("benchmark_mcts_", "").replace("_snapshot", "")
        display_name = _format_benchmark_title(stem)
        out = args.output or path.parent / f"delta_vs_n_{stem}.png"
        plot_delta_vs_n(deltas, ns, bl, f"Delta vs n — {display_name}", out, final_val_data=fv_data)

if __name__ == "__main__":
    main()
