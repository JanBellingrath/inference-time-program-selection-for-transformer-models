#!/usr/bin/env python3
"""Report delta, num_simulations, and tier of best sequence for all benchmark-level MCTS searches.

import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

Scans predictions/ and predictions/llama31_8b/ for benchmark_mcts_* result files.
For each benchmark, extracts: delta, num_simulations, tier of best sequence.
"""
import json
import os
import re
from pathlib import Path


def bench_name_from_path(path: Path, is_snapshot: bool) -> str:
    """Extract benchmark name from file path."""
    stem = path.stem
    if is_snapshot:
        stem = stem.replace("_snapshot", "")
    if stem.startswith("benchmark_mcts_"):
        stem = stem[len("benchmark_mcts_"):]
    # Remove timestamp _YYYYMMDD-HHMMSS
    stem = re.sub(r"_[12]\d{7}-\d{6}$", "", stem)
    # Remove _sim\d+ suffix (e.g. gsm8k_hard_sim100 -> gsm8k_hard)
    stem = re.sub(r"_sim\d+$", "", stem)
    return stem


def collect_benchmark_data(search_dirs: list) -> dict:
    """Collect benchmark data from all JSON files. Returns {bench_name: {...}}.
    Pairs full JSON with snapshot when they share the same run (same prefix).
    For each benchmark, keeps the newest run."""
    candidates = {}  # (bench_name, prefix) -> {mtime, full_path, snap_path}
    for base in search_dirs:
        base_path = Path(base)
        if not base_path.exists():
            continue
        for p in base_path.glob("benchmark_mcts_*"):
            if p.suffix != ".json":
                continue
            is_snap = "_snapshot" in p.name
            stem = p.stem.replace("_snapshot", "")
            if not stem.startswith("benchmark_mcts_"):
                continue
            name = bench_name_from_path(p, is_snapshot=is_snap)
            # Prefix = path without _snapshot.json/.json for pairing
            prefix = str(p.parent / stem)
            mtime = p.stat().st_mtime
            key = (name, prefix)
            c = candidates.get(key, {"mtime": 0, "full": None, "snap": None})
            c["mtime"] = max(c["mtime"], mtime)
            if is_snap:
                c["snap"] = p
            else:
                c["full"] = p
            candidates[key] = c
    # For each benchmark, keep newest run (by prefix mtime)
    best_run = {}
    for (name, prefix), c in candidates.items():
        if (c["full"] or c["snap"]) and (name not in best_run or c["mtime"] > best_run[name]["mtime"]):
            best_run[name] = c
    data = {}
    for name, c in best_run.items():
        entry = {"benchmark": name, "delta": None, "num_simulations": None, "tier": None, "best_acc": None}
        if c["full"]:
            try:
                with open(c["full"]) as f:
                    raw = json.load(f)
                entry["num_simulations"] = raw.get("num_simulations")
                best = raw.get("best")
                if best:
                    entry["delta"] = best.get("delta")
                    entry["best_acc"] = best.get("accuracy")
            except Exception:
                pass
        if c["snap"]:
            try:
                with open(c["snap"]) as f:
                    snap = json.load(f)
            except Exception:
                snap = {}
            if entry["num_simulations"] is None:
                entry["num_simulations"] = snap.get("sim")
            best = snap.get("best_tier4") or snap.get("best_tier3") or snap.get("best_validated")
            if best is None:
                for lst in (snap.get("tier4"), snap.get("tier3"), snap.get("validated")):
                    if lst:
                        best = lst[0]
                        break
            if best and entry["delta"] is None:
                ref = snap.get("baseline_ext") if (snap.get("best_tier4") or snap.get("best_tier3")) else snap.get("baseline")
                entry["delta"] = best.get("delta")
                if entry["delta"] is None and ref is not None:
                    entry["delta"] = best.get("accuracy", 0) - ref
            if best and entry["best_acc"] is None:
                entry["best_acc"] = best.get("accuracy")
            entry["tier"] = "tier4" if snap.get("best_tier4") else ("tier3" if snap.get("best_tier3") else "tier2")
        data[name] = entry
    return data


def main():
    repo_root = Path(__file__).resolve().parent
    import sys
    # Default: both dirs. Use --qwen3b to get only Qwen 3B (predictions/ root).
    qwen3b_only = "--qwen3b" in sys.argv
    if qwen3b_only:
        sys.argv.remove("--qwen3b")
        search_dirs = [repo_root / "predictions"]  # Qwen 3B default output dir
    else:
        search_dirs = [
            repo_root / "predictions",
            repo_root / "predictions" / "llama31_8b",
        ]
    data = collect_benchmark_data(search_dirs)
    # Drop internal keys for output
    rows = []
    for name, d in data.items():
        rows.append({
            "benchmark": d["benchmark"],
            "delta": d.get("delta"),
            "num_simulations": d.get("num_simulations"),
            "tier": d.get("tier"),
            "best_acc": d.get("best_acc"),
        })
    rows.sort(key=lambda r: (r["benchmark"] or ""))
    # Print report
    print("=" * 85)
    if qwen3b_only:
        print("QWEN 3B (Qwen2.5-3B-Instruct) - Benchmark-level MCTS Search Results")
    else:
        print("BENCHMARK-LEVEL SEARCH REPORT: Delta, Num Simulations, Tier of Best Sequence")
    print("=" * 85)
    print(f"{'Benchmark':<28} {'Delta':>12} {'Num Sims':>10} {'Tier':>8} {'Best Acc':>10}")
    print("-" * 95)
    for r in rows:
        b = (r["benchmark"] or "?")[:27].ljust(28)
        d = r["delta"]
        d_str = f"{d:+.4f}" if d is not None else "—"
        n = r["num_simulations"]
        n_str = str(n) if n is not None else "—"
        tier = r.get("tier") or "—"
        acc = r.get("best_acc")
        acc_str = f"{acc:.2%}" if acc is not None else "—"
        print(f"{b} {d_str:>12} {n_str:>10} {tier:>8} {acc_str:>10}")
    print("=" * 95)
    # Save JSON
    out_name = "benchmark_search_report_qwen3b.json" if qwen3b_only else "benchmark_search_report.json"
    out_path = repo_root / "predictions" / out_name
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
