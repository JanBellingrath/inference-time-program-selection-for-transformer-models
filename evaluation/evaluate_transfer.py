#!/usr/bin/env python3
"""
Evaluate layer sequences on target benchmarks to test cross-benchmark transfer.

Takes sequences from a snapshot JSON, a JSON file, or inline CLI, and evaluates
each on one or more target datasets.  Prints live intermediate results and saves
a combined JSON at the end.

Examples:

    # Test the top-5 GSM8K-hard sequences on SVAMP, ASDiv, AQuA-RAT
    python evaluate_transfer.py \
        --snapshot predictions/benchmark_mcts_gsm8k_hard_sim100_snapshot.json \
        --datasets svamp asdiv aqua_rat \
        --num_samples 300

    # Same but sequences given as inline JSON
    python evaluate_transfer.py \
        --seqs '[[0,1,2,...,35],[0,1,...]]' \
        --datasets svamp asdiv \
        --num_samples 500

    # Pull best sequences from an explored_log.jsonl
    python evaluate_transfer.py \
        --log predictions/benchmark_mcts_gsm8k_hard_*_explored_log.jsonl \
        --datasets svamp asdiv aqua_rat --top_k 5

    # Pull tier-4 sequences from a snapshot
    python evaluate_transfer.py \
        --snapshot predictions/some_snapshot.json --tier tier4 \
        --datasets winogrande mmlu_all

    # Pull from a flat JSON mapping {name: [layers]}
    python evaluate_transfer.py \
        --sequences_json sequences/gsm8k_hard_top5.json \
        --datasets svamp asdiv aqua_rat
"""
import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import argparse
import json
import math
import os
import random
import re
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from core.permutation_mcts import (
    MCTSModel, prepare_arc_data, set_seed, get_benchmark_dataset_choices,
)
from core.flexible_models import get_is_instruct
from core.benchmark_mcts import grade_response, seq_to_layers, SKIP
from prepare_commonsense_data import (
    prepare_commonsense_data, COMMONSENSE_DATASETS,
)
from prepare_knowledge_data import (
    prepare_knowledge_data, KNOWLEDGE_DATASETS,
)
from prepare_math_data import (
    prepare_math_data, MATH_DATASETS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

NUMERIC_DATASETS = {"gsm8k_hard", "svamp", "asdiv", "dart-1", "dart-2",
                    "dart-3", "dart-4", "dart-5",
                    "hendrycks_math", "amc_aime", "mawps"}
MC_DATASETS = {"arc_easy", "arc_challenge", "mmlu", "mmlu_all", "winogrande",
               "commonsenseqa", "aqua_rat", "bigbench", "bigbench_all",
               "copa", "social_iqa", "piqa", "hellaswag", "swag",
               "anli", "story_cloze", "cosmos_qa",
               "mmlu_pro", "gpqa", "agieval_sat", "agieval_lsat",
               "agieval_logiqa", "scienceqa", "medqa",
               "mmlu_math", "mathqa"}

# ---------------------------------------------------------------------------
# Semantic domain groupings for cross-domain analysis
# ---------------------------------------------------------------------------

DOMAINS: Dict[str, List[str]] = {
    "commonsense": [
        "copa", "social_iqa", "piqa", "hellaswag", "swag",
        "anli", "story_cloze", "cosmos_qa",
        "winogrande", "commonsenseqa",
    ],
    "knowledge": [
        "mmlu_pro", "gpqa", "agieval_sat", "agieval_lsat",
        "agieval_logiqa", "scienceqa", "medqa",
        "arc_challenge", "arc_easy", "mmlu", "mmlu_all",
    ],
    "math": [
        "gsm8k_hard", "svamp", "asdiv", "aqua_rat",
        "dart-1", "dart-2", "dart-3", "dart-4", "dart-5",
        "math500",
        "hendrycks_math", "amc_aime", "mmlu_math", "mathqa", "mawps",
    ],
}


def _benchmark_to_domain(domains: Dict[str, List[str]] = None) -> Dict[str, str]:
    if domains is None:
        domains = DOMAINS
    mapping: Dict[str, str] = {}
    for domain, benchmarks in domains.items():
        for b in benchmarks:
            mapping[b] = domain
    return mapping


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def max_new_tokens_for(dataset: str, is_instruct: bool) -> int:
    """Pick a sensible generation budget per dataset type."""
    if "dart" in dataset:
        return 15
    if dataset in NUMERIC_DATASETS:
        return 256
    if is_instruct:
        return 10
    return 2


def evaluate_sequence(
    model: MCTSModel,
    layers: List[int],
    samples: List[Dict],
    dataset: str,
    model_name: str,
    max_tokens: int,
    label: str = "",
    report_every: int = 25,
) -> Dict[str, Any]:
    """Evaluate a single layer sequence on *samples*, printing live progress."""
    wrapper = model.wrapper
    has_dup = len(layers) != len(set(layers))
    correct = 0

    for i, s in enumerate(samples):
        saved = wrapper.model.model.layer_indices
        wrapper.model.model.layer_indices = layers
        try:
            sys_prompt = s.get("system_prompt")
            sample_max_tokens = s.get("max_new_tokens", max_tokens)
            prompt = wrapper.prepare_prompt(s["input"], system_prompt=sys_prompt)
            inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(
                wrapper.model.device
            )
            input_len = inputs.input_ids.shape[1]
            gen_kw = {
                "max_new_tokens": sample_max_tokens,
                "pad_token_id": wrapper.tokenizer.eos_token_id,
                "do_sample": False,
            }
            if has_dup:
                gen_kw["use_cache"] = False
            with torch.no_grad():
                out = wrapper.model.generate(**inputs, **gen_kw)
            resp = wrapper.tokenizer.decode(
                out[0][input_len:], skip_special_tokens=True
            ).strip()
        finally:
            wrapper.model.model.layer_indices = saved

        sc = grade_response(resp, s["correct"], dataset, model_name, s["input"])
        correct += int(sc > 0.5)

        done = i + 1
        if done % report_every == 0 or done == len(samples):
            acc = correct / done
            lo, hi = wilson_ci(correct, done)
            print(
                f"    [{label}] {done}/{len(samples)}: "
                f"acc={acc:.4f} ({correct}/{done}) "
                f"CI [{lo:.4f}, {hi:.4f}]",
                flush=True,
            )

    n = len(samples)
    acc = correct / n
    lo, hi = wilson_ci(correct, n)
    return {"correct": correct, "total": n, "accuracy": acc, "ci_lo": lo, "ci_hi": hi}


# ---------------------------------------------------------------------------
# Sequence loading helpers
# ---------------------------------------------------------------------------

def _build_explored_index(snap: Dict) -> Dict[Tuple[float, int], List[int]]:
    """Build a lookup from (accuracy, n_evaluated) -> layer sequence
    using the explored_sequences / explored_correct / explored_total arrays
    that benchmark_mcts stores in every snapshot."""
    seqs = snap.get("explored_sequences", [])
    correct = snap.get("explored_correct", [])
    total = snap.get("explored_total", [])
    baseline = snap.get("baseline", 0)
    index: Dict[Tuple[float, int], List[int]] = {}
    for i, seq in enumerate(seqs):
        if i < len(correct) and i < len(total) and total[i] > 0:
            acc = round(correct[i] / total[i], 6)
            index[(acc, total[i])] = seq
    return index


def load_sequences_from_snapshot(
    path: str, tier: str, top_k: int
) -> List[Dict[str, Any]]:
    """Extract candidate sequences from a benchmark_mcts snapshot JSON.

    Handles both the full format (entries with 'seq'/'layers') and the
    compact format (entries with only accuracy/delta/evaluated) by
    cross-referencing against explored_sequences.
    """
    with open(path) as f:
        snap = json.load(f)

    tier_keys = {
        "tier4": ["tier4"],
        "tier3": ["tier3"],
        "tier2": ["validated"],
        "auto":  ["tier4", "tier3", "validated", "results"],
    }
    src = []
    chosen_key = None
    for key in tier_keys.get(tier, tier_keys["auto"]):
        src = snap.get(key, [])
        if src:
            chosen_key = key
            break

    if not src:
        for bk in ("best_tier4", "best_tier3", "best_validated", "best"):
            entry = snap.get(bk)
            if entry and isinstance(entry, dict):
                src = [entry]
                chosen_key = bk
                break

    if not src:
        logger.error("No sequence entries found in snapshot %s", path)
        return []

    logger.info("Using snapshot key '%s' (%d entries)", chosen_key, len(src))

    # Entries may lack layers (compact snapshot format).  Recover from
    # explored_sequences or from the validated tier-2 results which always
    # have 'seq' in the full _snapshot.
    explored_idx = None  # lazy-build

    # Also build a lookup from the validated tier-2 results (which typically
    # have full seq/layers even when tier3 entries are compact)
    validated_by_acc: Dict[float, Dict] = {}
    for entry in snap.get("validated", []):
        if entry.get("seq") or entry.get("layers"):
            validated_by_acc[round(entry.get("accuracy", -1), 6)] = entry

    candidates = []
    for i, entry in enumerate(src[:top_k]):
        seq = entry.get("seq")
        layers = entry.get("layers")

        if seq is None and layers is None:
            # Try to recover from validated entries with matching accuracy
            acc_key = round(entry.get("accuracy", -1), 6)
            match = validated_by_acc.get(acc_key)
            if match:
                seq = match.get("seq")
                layers = match.get("layers")

        if seq is None and layers is None:
            # Fall back to explored_sequences index
            if explored_idx is None:
                explored_idx = _build_explored_index(snap)
            baseline = snap.get("baseline", 0)
            acc = entry.get("accuracy", 0)
            n_eval = entry.get("evaluated", 0)
            # Compact tier3 accuracy is from extended pool; try direct match
            for key_candidate in [(round(acc, 6), n_eval)]:
                if key_candidate in explored_idx:
                    seq = explored_idx[key_candidate]
                    break
            if seq is None:
                # Try matching by delta against baseline on the noisy stats
                noisy_acc = round(baseline + entry.get("delta", 0), 6)
                for (ea, en), eseq in explored_idx.items():
                    if abs(ea - noisy_acc) < 0.001:
                        seq = eseq
                        break

        if seq is None and layers is None:
            logger.warning("Could not recover layers for entry %d "
                           "(acc=%.4f, delta=%.4f) -- skipping",
                           i, entry.get("accuracy", 0), entry.get("delta", 0))
            continue

        if layers is None:
            layers = seq_to_layers(seq) if any(x == SKIP for x in seq) else seq
        if seq is None:
            seq = layers

        candidates.append({
            "label": f"seq_{len(candidates)+1}",
            "seq": seq,
            "layers": layers,
            "source_acc": entry.get("accuracy"),
            "source_delta": entry.get("delta"),
            "source_n": entry.get("evaluated"),
        })
    return candidates


def load_sequences_from_json(path: str) -> List[Dict[str, Any]]:
    """Load from a flat JSON: either {name: [layers]} or [[layers], ...]."""
    with open(path) as f:
        data = json.load(f)
    candidates = []
    if isinstance(data, dict):
        for name, layers in data.items():
            candidates.append({
                "label": name,
                "seq": layers,
                "layers": seq_to_layers(layers) if any(x == SKIP for x in layers) else layers,
            })
    elif isinstance(data, list):
        for i, layers in enumerate(data):
            candidates.append({
                "label": f"seq_{i+1}",
                "seq": layers,
                "layers": seq_to_layers(layers) if any(x == SKIP for x in layers) else layers,
            })
    return candidates


def load_sequences_from_log(
    path: str, top_k: int, min_delta: float = 0.0
) -> List[Dict[str, Any]]:
    """Load best sequences from an explored_log.jsonl produced by benchmark_mcts.

    Ranks all explored sequences by accuracy and picks the top-K with delta
    above min_delta.
    """
    entries = []
    baseline = None
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("event") == "start":
                baseline = row.get("baseline", 0)
                continue
            if row.get("event") == "end":
                continue
            if "seq" in row and "accuracy" in row:
                entries.append(row)

    if not entries:
        logger.error("No sequence entries found in log %s", path)
        return []

    # Deduplicate by sequence, keeping the entry with most evaluations
    by_seq: Dict[str, Dict] = {}
    for e in entries:
        key = json.dumps(e["seq"])
        prev = by_seq.get(key)
        if prev is None or e.get("total", 0) > prev.get("total", 0):
            by_seq[key] = e

    ranked = sorted(by_seq.values(), key=lambda x: x.get("accuracy", 0), reverse=True)
    if min_delta > 0:
        ranked = [r for r in ranked if r.get("delta", 0) >= min_delta]

    candidates = []
    for i, entry in enumerate(ranked[:top_k]):
        seq = entry["seq"]
        layers = entry.get("layers") or seq_to_layers(seq)
        candidates.append({
            "label": f"seq_{i+1}",
            "seq": seq,
            "layers": layers,
            "source_acc": entry.get("accuracy"),
            "source_delta": entry.get("delta"),
            "source_n": entry.get("total"),
        })
    return candidates


def load_sequences_from_cli(raw: str) -> List[Dict[str, Any]]:
    """Parse inline JSON list-of-lists from the --seqs argument."""
    seqs = json.loads(raw)
    candidates = []
    for i, seq in enumerate(seqs):
        layers = seq_to_layers(seq) if any(x == SKIP for x in seq) else seq
        candidates.append({"label": f"seq_{i+1}", "seq": seq, "layers": layers})
    return candidates


# ---------------------------------------------------------------------------
# Live results persistence
# ---------------------------------------------------------------------------

def save_live(out_path: str, results: Dict[str, Any]):
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_transfer_results(all_results: Dict[str, Any], out_path: str):
    """Grouped bar chart: baseline vs best candidate per benchmark (seaborn)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available, skipping plot")
        return

    datasets = list(all_results.get("datasets", {}).keys())
    if not datasets:
        return

    rows = []
    for ds in datasets:
        ds_res = all_results["datasets"][ds]
        bl = ds_res.get("baseline")
        if bl:
            rows.append({
                "Benchmark": ds,
                "Sequence": "Baseline",
                "Accuracy": bl["accuracy"],
                "CI_lo": bl["ci_lo"],
                "CI_hi": bl["ci_hi"],
            })
        cands = ds_res.get("candidates", [])
        if cands:
            best = max(cands, key=lambda c: c["accuracy"])
            rows.append({
                "Benchmark": ds,
                "Sequence": f"Best ({best['label']})",
                "Accuracy": best["accuracy"],
                "CI_lo": best["ci_lo"],
                "CI_hi": best["ci_hi"],
            })

    if not rows:
        return

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 1.6), 6))

    benchmarks = []
    baseline_accs, best_accs = [], []
    baseline_err, best_err = [[], []], [[], []]

    for ds in datasets:
        bl_row = next((r for r in rows if r["Benchmark"] == ds and r["Sequence"] == "Baseline"), None)
        best_row = next((r for r in rows if r["Benchmark"] == ds and r["Sequence"] != "Baseline"), None)
        if bl_row and best_row:
            benchmarks.append(ds)
            baseline_accs.append(bl_row["Accuracy"])
            best_accs.append(best_row["Accuracy"])
            baseline_err[0].append(bl_row["Accuracy"] - bl_row["CI_lo"])
            baseline_err[1].append(bl_row["CI_hi"] - bl_row["Accuracy"])
            best_err[0].append(best_row["Accuracy"] - best_row["CI_lo"])
            best_err[1].append(best_row["CI_hi"] - best_row["Accuracy"])

    if not benchmarks:
        plt.close(fig)
        return

    x = np.arange(len(benchmarks))
    width = 0.35
    palette = sns.color_palette("muted", 2)

    bars1 = ax.bar(x - width / 2, baseline_accs, width, label="Baseline",
                   color=palette[0], yerr=baseline_err, capsize=4, error_kw={"lw": 1.2})
    bars2 = ax.bar(x + width / 2, best_accs, width, label="Best Candidate",
                   color=palette[1], yerr=best_err, capsize=4, error_kw={"lw": 1.2})

    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Accuracy")
    ax.set_title("Baseline vs Optimal Sequence Transfer")
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, min(1.0, max(max(baseline_accs), max(best_accs)) * 1.25))

    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    plot_path = out_path.rsplit(".", 1)[0] + "_plot.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {plot_path}", flush=True)


# ---------------------------------------------------------------------------
# Cross-domain analysis: heatmaps
# ---------------------------------------------------------------------------

def _collect_deltas(
    entries: List[Dict[str, Any]],
    domains: Dict[str, List[str]],
) -> Tuple[
    Dict[str, Dict[str, List[float]]],   # agg[src_domain][tgt_domain] -> deltas
    List[Dict[str, Any]],                 # flat per-benchmark rows
]:
    """Aggregate best-candidate deltas from result entries into domain buckets.

    Each *entry* must have:
      source_domain  – str
      source_label   – str (e.g. "winogrande")
      results        – dict loaded from a transfer result JSON
    """
    from collections import defaultdict

    b2d = _benchmark_to_domain(domains)
    agg: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    rows: List[Dict[str, Any]] = []

    for entry in entries:
        src_dom = entry["source_domain"]
        src_lbl = entry.get("source_label", src_dom)

        for ds_name, ds_res in entry["results"].get("datasets", {}).items():
            tgt_dom = b2d.get(ds_name)
            if tgt_dom is None:
                continue
            cands = ds_res.get("candidates", [])
            if not cands:
                continue
            best = max(cands, key=lambda c: c.get("accuracy", 0))
            delta = best.get("delta")
            if delta is None:
                continue

            agg[src_dom][tgt_dom].append(delta)
            rows.append({
                "source_domain": src_dom,
                "source_label": src_lbl,
                "target_domain": tgt_dom,
                "target_benchmark": ds_name,
                "delta": delta,
                "baseline_acc": ds_res.get("baseline", {}).get("accuracy"),
                "best_acc": best.get("accuracy"),
            })

    return dict(agg), rows


def plot_cross_domain_heatmap(
    entries: List[Dict[str, Any]],
    domains: Dict[str, List[str]],
    out_dir: str,
):
    """Two figures: (1) domain-level summary, (2) per-benchmark detail.

    Parameters
    ----------
    entries : list of dicts
        Each dict has keys  source_domain, source_label, results.
    domains : domain_name -> benchmark list
    out_dir : directory to write the PNGs into
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
    except ImportError:
        logger.warning("matplotlib/seaborn/pandas not available, skipping cross-domain plot")
        return

    agg, rows = _collect_deltas(entries, domains)
    if not rows:
        logger.warning("No data for cross-domain heatmap")
        return

    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # (1) Domain-level summary heatmap
    # ------------------------------------------------------------------
    src_domains = sorted(agg.keys())
    tgt_domains_set: set = set()
    for dm in agg.values():
        tgt_domains_set.update(dm.keys())
    tgt_domains = sorted(tgt_domains_set)

    n_src, n_tgt = len(src_domains), len(tgt_domains)
    matrix = np.full((n_src, n_tgt), np.nan)
    annot = [["" for _ in range(n_tgt)] for _ in range(n_src)]

    for i, src in enumerate(src_domains):
        for j, tgt in enumerate(tgt_domains):
            deltas = agg.get(src, {}).get(tgt, [])
            if deltas:
                mean_d = float(np.mean(deltas))
                matrix[i, j] = mean_d
                n_pos = sum(1 for d in deltas if d > 0)
                annot[i][j] = f"{mean_d*100:+.2f}%\n({n_pos}/{len(deltas)} \u2191)"

    vlim = float(np.nanmax(np.abs(matrix))) if not np.all(np.isnan(matrix)) else 0.05
    vlim = max(vlim, 0.005)

    sns.set_theme(style="white", font_scale=1.25)
    fig, ax = plt.subplots(figsize=(max(6, n_tgt * 3.5), max(4, n_src * 2.5)))
    sns.heatmap(
        matrix,
        annot=annot, fmt="",
        xticklabels=[d.title() for d in tgt_domains],
        yticklabels=[d.title() for d in src_domains],
        cmap="RdYlGn", center=0, vmin=-vlim, vmax=vlim,
        linewidths=2.5, linecolor="white",
        cbar_kws={"label": "Mean \u0394 Accuracy (best seq vs baseline)", "shrink": 0.8},
        ax=ax, square=True,
    )
    ax.set_xlabel("Target Domain", fontsize=14, labelpad=10)
    ax.set_ylabel("Source Domain (Anchor)", fontsize=14, labelpad=10)
    ax.set_title(
        "Cross-Domain Transfer Generalization\n"
        "Diagonal = within-domain, off-diagonal = cross-domain",
        fontsize=14, pad=12,
    )
    fig.tight_layout()
    p1 = os.path.join(out_dir, "cross_domain_summary.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Domain-level heatmap saved to {p1}", flush=True)

    # ------------------------------------------------------------------
    # (2) Per-benchmark detail heatmap  (rows = anchors, cols = benchmarks
    #     grouped by target domain)
    # ------------------------------------------------------------------
    df = pd.DataFrame(rows)
    if df.empty:
        return

    anchor_ids = []
    for entry in entries:
        aid = f"{entry['source_domain']}  ({entry.get('source_label', '?')})"
        if aid not in anchor_ids:
            anchor_ids.append(aid)

    benchmarks_ordered: List[str] = []
    for dom in tgt_domains:
        dom_benchmarks = sorted(
            df.loc[df["target_domain"] == dom, "target_benchmark"].unique()
        )
        benchmarks_ordered.extend(dom_benchmarks)

    detail = np.full((len(anchor_ids), len(benchmarks_ordered)), np.nan)
    for entry in entries:
        aid = f"{entry['source_domain']}  ({entry.get('source_label', '?')})"
        if aid not in anchor_ids:
            continue
        i = anchor_ids.index(aid)
        for ds_name, ds_res in entry["results"].get("datasets", {}).items():
            if ds_name not in benchmarks_ordered:
                continue
            j = benchmarks_ordered.index(ds_name)
            cands = ds_res.get("candidates", [])
            if cands:
                best = max(cands, key=lambda c: c.get("accuracy", 0))
                d = best.get("delta")
                if d is not None:
                    detail[i, j] = d

    vlim2 = float(np.nanmax(np.abs(detail))) if not np.all(np.isnan(detail)) else 0.05
    vlim2 = max(vlim2, 0.005)

    detail_annot = np.where(
        np.isnan(detail), "",
        np.vectorize(lambda v: f"{v*100:+.1f}")(detail),
    )

    fig2, ax2 = plt.subplots(
        figsize=(max(14, len(benchmarks_ordered) * 1.3),
                 max(5, len(anchor_ids) * 2.0 + 2))
    )
    sns.heatmap(
        detail,
        annot=detail_annot, fmt="",
        xticklabels=benchmarks_ordered,
        yticklabels=anchor_ids,
        cmap="RdYlGn", center=0, vmin=-vlim2, vmax=vlim2,
        linewidths=0.8, linecolor="white",
        cbar_kws={"label": "\u0394 Accuracy (%)", "shrink": 0.7},
        ax=ax2,
    )
    ax2.set_xlabel("Target Benchmark", fontsize=12, labelpad=8)
    ax2.set_ylabel("Anchor (source domain)", fontsize=12, labelpad=8)
    ax2.set_title(
        "Per-Benchmark Transfer Detail  (\u0394 accuracy: best sequence vs baseline)",
        fontsize=13, pad=10,
    )
    ax2.tick_params(axis="x", rotation=40, labelsize=9)
    ax2.tick_params(axis="y", labelsize=10)

    # Domain separators on x-axis
    b2d = _benchmark_to_domain(domains)
    prev_dom = None
    for idx, bname in enumerate(benchmarks_ordered):
        cur_dom = b2d.get(bname, "")
        if prev_dom is not None and cur_dom != prev_dom:
            ax2.axvline(x=idx, color="black", linewidth=2.0)
        prev_dom = cur_dom

    # Domain labels along top
    cum = 0
    for dom in tgt_domains:
        n_b = sum(1 for b in benchmarks_ordered if b2d.get(b) == dom)
        if n_b:
            ax2.text(cum + n_b / 2, -0.7, dom.title(),
                     ha="center", va="bottom", fontsize=11, fontweight="bold",
                     transform=ax2.get_xaxis_transform())
            cum += n_b

    fig2.tight_layout()
    p2 = os.path.join(out_dir, "cross_domain_detail.png")
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Per-benchmark heatmap saved to {p2}", flush=True)


# ---------------------------------------------------------------------------
# Cross-domain CLI entry point
# ---------------------------------------------------------------------------

def cross_domain_main():
    """Load transfer result JSONs, classify by source domain, build heatmaps.

    Usage::

        python evaluate_transfer.py cross_domain \\
            --result commonsense:winogrande:predictions/file1.json \\
            --result knowledge:mmlu_all:predictions/file2.json \\
            --output_dir predictions/cross_domain_analysis
    """
    p = argparse.ArgumentParser(
        description="Cross-domain transfer analysis: build heatmaps from result JSONs",
    )
    p.add_argument(
        "--result", action="append", required=True,
        metavar="DOMAIN:LABEL:PATH",
        help=(
            "Source domain, anchor label, and result JSON path, separated by colons. "
            "Example: commonsense:winogrande:predictions/transfer_winogrande.json  "
            "Multiple --result flags build the full matrix."
        ),
    )
    p.add_argument(
        "--output_dir", type=str, default="predictions/cross_domain_analysis",
        help="Directory to write heatmap PNGs into",
    )
    p.add_argument(
        "--domains_json", type=str, default=None,
        help="Optional JSON file overriding the built-in DOMAINS dict",
    )
    args = p.parse_args()

    # Parse domain overrides
    domains = DOMAINS
    if args.domains_json:
        with open(args.domains_json) as f:
            domains = json.load(f)

    # Parse --result entries
    entries: List[Dict[str, Any]] = []
    for spec in args.result:
        parts = spec.split(":", 2)
        if len(parts) != 3:
            p.error(
                f"Expected DOMAIN:LABEL:PATH, got '{spec}'.\n"
                "Example: commonsense:winogrande:predictions/result.json"
            )
        domain, label, path = parts
        with open(path) as f:
            results = json.load(f)
        entries.append({
            "source_domain": domain,
            "source_label": label,
            "results": results,
        })
        n_ds = len(results.get("datasets", {}))
        logger.info("Loaded %s (%s) -> %d dataset results from %s", domain, label, n_ds, path)

    if not entries:
        p.error("No --result entries provided")

    plot_cross_domain_heatmap(entries, domains, args.output_dir)
    print(f"\nCross-domain analysis complete. Outputs in {args.output_dir}/", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_data(dataset: str, is_instruct: bool, split: str) -> List[Dict]:
    """Load data from the appropriate domain-specific loader."""
    if dataset in COMMONSENSE_DATASETS:
        return prepare_commonsense_data(dataset, is_instruct, split)
    if dataset in KNOWLEDGE_DATASETS:
        return prepare_knowledge_data(dataset, is_instruct, split)
    if dataset in MATH_DATASETS:
        return prepare_math_data(dataset, is_instruct, split)
    return prepare_arc_data(dataset, is_instruct, split=split)


def main():
    ds_choices = (get_benchmark_dataset_choices() + COMMONSENSE_DATASETS
                  + KNOWLEDGE_DATASETS + MATH_DATASETS)

    p = argparse.ArgumentParser(
        description="Evaluate layer sequences on target benchmarks (transfer test)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_argument_group("Sequence sources (use exactly one)")
    src.add_argument(
        "--snapshot", type=str, default=None,
        help="Path to *_snapshot.json from benchmark_mcts.py",
    )
    src.add_argument(
        "--sequences_json", type=str, default=None,
        help="Path to JSON with sequences ({name: layers} or [[layers],...])",
    )
    src.add_argument(
        "--log", type=str, default=None,
        help="Path to *_explored_log.jsonl from benchmark_mcts.py",
    )
    src.add_argument(
        "--seqs", type=str, default=None,
        help="Inline JSON list of layer sequences, e.g. '[[0,1,...,35]]'",
    )

    p.add_argument(
        "--tier", type=str, default="auto",
        choices=["auto", "tier2", "tier3", "tier4"],
        help="Which tier to pull from a snapshot (default: best available)",
    )
    p.add_argument("--top_k", type=int, default=5,
                   help="Max sequences to take from snapshot")
    p.add_argument(
        "--datasets", nargs="+", required=True,
        help=f"Target datasets to evaluate on. Choices: {', '.join(ds_choices)}",
    )
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--num_samples", type=int, default=300,
                   help="Samples per dataset (capped at dataset size)")
    p.add_argument("--seed", type=int, default=123,
                   help="RNG seed (use different from MCTS to avoid overlap)")
    p.add_argument("--split", type=str, default="train",
                   choices=["train", "validation", "test"])
    p.add_argument("--report_every", type=int, default=25,
                   help="Print intermediate progress every N samples")
    p.add_argument("--max_new_tokens", type=int, default=None,
                   help="Override generation budget (auto-detected per dataset if omitted)")
    p.add_argument("--output", type=str, default=None,
                   help="Output JSON path (default: predictions/transfer_<timestamp>.json)")
    p.add_argument("--skip_baseline", action="store_true",
                   help="Skip baseline evaluation (use if you only want sequence results)")
    args = p.parse_args()

    # --- Validate dataset names ---
    for ds in args.datasets:
        if ds not in ds_choices:
            p.error(f"Unknown dataset '{ds}'. Choices: {', '.join(ds_choices)}")

    # --- Load sequences ---
    sources = [args.snapshot, args.sequences_json, args.log, args.seqs]
    n_sources = sum(x is not None for x in sources)
    if n_sources == 0:
        p.error("Provide one of --snapshot, --sequences_json, --log, or --seqs")
    if n_sources > 1:
        p.error("Provide only one of --snapshot, --sequences_json, --log, or --seqs")

    if args.snapshot:
        candidates = load_sequences_from_snapshot(args.snapshot, args.tier, args.top_k)
    elif args.sequences_json:
        candidates = load_sequences_from_json(args.sequences_json)
    elif args.log:
        candidates = load_sequences_from_log(args.log, args.top_k)
    else:
        candidates = load_sequences_from_cli(args.seqs)

    if not candidates:
        logger.error("No sequences loaded."); return

    print(f"\nLoaded {len(candidates)} candidate sequence(s):", flush=True)
    for c in candidates:
        n_swaps = sum(1 for i, v in enumerate(c["layers"]) if i < len(c["layers"]) and
                      (i >= 36 or v != i))
        src_info = ""
        if c.get("source_acc") is not None:
            src_info = (f" (source acc={c['source_acc']:.4f}"
                        f" delta={c['source_delta']:+.4f}"
                        f" n={c['source_n']})")
        print(f"  {c['label']}: {len(c['layers'])} layers, "
              f"skips={c['seq'].count(SKIP) if SKIP in c.get('seq', []) else 0}"
              f"{src_info}", flush=True)

    # --- Seed ---
    set_seed(args.seed)

    # --- Load model ---
    print(f"\nLoading model: {args.model_name}", flush=True)
    model = MCTSModel(args.model_name, rank=0)
    default_layers = list(range(model.num_layers))
    is_instruct = get_is_instruct(args.model_name)

    # --- Output path ---
    out_path = args.output or (
        f"predictions/transfer_{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # --- Main evaluation loop: per dataset ---
    all_results: Dict[str, Any] = {
        "model_name": args.model_name,
        "seed": args.seed,
        "split": args.split,
        "candidates_meta": [
            {k: v for k, v in c.items() if k != "layers" or True}
            for c in candidates
        ],
        "datasets": {},
    }
    t0_global = time.time()

    for ds_idx, dataset in enumerate(args.datasets):
        print(f"\n{'='*70}", flush=True)
        print(f"[{ds_idx+1}/{len(args.datasets)}] Dataset: {dataset}", flush=True)
        print(f"{'='*70}", flush=True)

        # Load data
        data_seed = args.seed + ds_idx  # slightly vary per dataset
        all_samples = _load_data(dataset, is_instruct, args.split)
        random.Random(data_seed).shuffle(all_samples)
        n = min(args.num_samples, len(all_samples))
        samples = all_samples[:n]
        print(f"Using {n} samples (of {len(all_samples)} available, "
              f"split={args.split}, data_seed={data_seed})", flush=True)

        max_tokens = args.max_new_tokens or max_new_tokens_for(dataset, is_instruct)
        print(f"max_new_tokens={max_tokens}", flush=True)

        ds_results: Dict[str, Any] = {
            "num_samples": n,
            "max_new_tokens": max_tokens,
            "baseline": None,
            "candidates": [],
        }

        # --- Baseline ---
        if not args.skip_baseline:
            print(f"\nEvaluating baseline ({model.num_layers} layers)...", flush=True)
            t0 = time.time()
            baseline = evaluate_sequence(
                model, default_layers, samples, dataset, args.model_name,
                max_tokens, label="baseline", report_every=args.report_every,
            )
            baseline["elapsed_s"] = time.time() - t0
            ds_results["baseline"] = baseline
            print(
                f"  Baseline: {baseline['accuracy']:.4f} "
                f"({baseline['correct']}/{baseline['total']}) "
                f"95% CI [{baseline['ci_lo']:.4f}, {baseline['ci_hi']:.4f}] "
                f"({baseline['elapsed_s']:.0f}s)",
                flush=True,
            )
        else:
            print("  (baseline skipped)", flush=True)

        # Save live after baseline
        all_results["datasets"][dataset] = ds_results
        save_live(out_path, all_results)

        # --- Candidate sequences ---
        for ci, cand in enumerate(candidates):
            layers = cand["layers"]
            label = cand["label"]
            print(
                f"\n[{ci+1}/{len(candidates)}] {label}: "
                f"{len(layers)} layers, "
                f"skips={cand['seq'].count(SKIP) if SKIP in cand.get('seq', []) else 0}",
                flush=True,
            )
            t0 = time.time()
            res = evaluate_sequence(
                model, layers, samples, dataset, args.model_name,
                max_tokens, label=label, report_every=args.report_every,
            )
            res["elapsed_s"] = time.time() - t0
            res["label"] = label
            res["layers"] = layers
            res["seq"] = cand.get("seq")

            if ds_results["baseline"] is not None:
                b_acc = ds_results["baseline"]["accuracy"]
                b_ci = (ds_results["baseline"]["ci_lo"], ds_results["baseline"]["ci_hi"])
                delta = res["accuracy"] - b_acc
                sig = ("YES" if res["ci_lo"] > b_ci[1]
                       else ("marginal" if res["ci_lo"] > b_acc else "no"))
                res["delta"] = delta
                res["significant"] = sig
            else:
                res["delta"] = None
                res["significant"] = None

            ds_results["candidates"].append(res)

            # Live display
            delta_str = f"delta={res['delta']:+.4f}" if res["delta"] is not None else ""
            sig_str = f"sig={res['significant']}" if res["significant"] is not None else ""
            print(
                f"  Result: {res['accuracy']:.4f} "
                f"({res['correct']}/{res['total']}) "
                f"CI [{res['ci_lo']:.4f}, {res['ci_hi']:.4f}] "
                f"{delta_str} {sig_str} ({res['elapsed_s']:.0f}s)",
                flush=True,
            )

            # Save live after each candidate
            save_live(out_path, all_results)

        # --- Per-dataset summary ---
        print(f"\n--- {dataset} summary ---", flush=True)
        if ds_results["baseline"]:
            bl = ds_results["baseline"]
            print(f"  Baseline: {bl['accuracy']:.4f} "
                  f"[{bl['ci_lo']:.4f}, {bl['ci_hi']:.4f}]", flush=True)
        for r in sorted(ds_results["candidates"],
                        key=lambda x: x["accuracy"], reverse=True):
            tag = ("*" if r.get("significant") == "YES"
                   else ("~" if r.get("significant") == "marginal" else " "))
            delta_s = f"delta={r['delta']:+.4f}" if r["delta"] is not None else ""
            print(f"  {tag} {r['label']:15s} acc={r['accuracy']:.4f} "
                  f"[{r['ci_lo']:.4f},{r['ci_hi']:.4f}] {delta_s} "
                  f"len={len(r['layers'])}", flush=True)
        if ds_results["baseline"]:
            confirmed = sum(1 for r in ds_results["candidates"]
                            if r.get("delta", 0) and r["delta"] > 0)
            sig_count = sum(1 for r in ds_results["candidates"]
                            if r.get("significant") == "YES")
            print(f"  Better than baseline: {confirmed}/{len(ds_results['candidates'])} | "
                  f"Significant: {sig_count}/{len(ds_results['candidates'])}", flush=True)

    # --- Global summary ---
    elapsed_global = time.time() - t0_global
    all_results["elapsed_s"] = elapsed_global

    print(f"\n{'='*70}", flush=True)
    print("TRANSFER EVALUATION SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Model: {args.model_name}", flush=True)
    print(f"Sequences: {len(candidates)} | "
          f"Datasets: {', '.join(args.datasets)}", flush=True)
    print(f"Samples per dataset: {args.num_samples} | "
          f"Seed: {args.seed}", flush=True)
    print(f"-" * 70, flush=True)

    header = f"{'Dataset':>15s}  {'Baseline':>8s}  "
    header += "  ".join(f"{c['label']:>10s}" for c in candidates)
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for dataset in args.datasets:
        ds_res = all_results["datasets"][dataset]
        bl_str = (f"{ds_res['baseline']['accuracy']:.4f}"
                  if ds_res["baseline"] else "  n/a ")
        row = f"{dataset:>15s}  {bl_str:>8s}  "
        for c in candidates:
            match = next((r for r in ds_res["candidates"]
                          if r["label"] == c["label"]), None)
            if match:
                delta = match.get("delta")
                sig = match.get("significant", "")
                tag = "*" if sig == "YES" else ("~" if sig == "marginal" else " ")
                if delta is not None:
                    row += f" {tag}{match['accuracy']:.4f}{delta:+.3f}"
                else:
                    row += f"  {match['accuracy']:.4f}     "
            else:
                row += f"{'':>10s}  "
        print(row, flush=True)

    print(f"-" * 70, flush=True)
    print(f"Total time: {elapsed_global/60:.1f} min", flush=True)
    print(f"{'='*70}", flush=True)

    save_live(out_path, all_results)
    print(f"\nResults saved to {out_path}", flush=True)

    plot_transfer_results(all_results, out_path)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cross_domain":
        sys.argv.pop(1)
        cross_domain_main()
    else:
        main()
