"""Pair-level bootstrap CIs and paired comparisons (edge resampling only).

Resamples the held-out **edges** in ``per_edge.jsonl`` (one summary row per
length-2 edge with ``n_questions > 0``). The CI reflects variability across
the ``n_edges`` held-out compositions.

**Input:** ``per_edge.jsonl`` from
:mod:`experiments.eval_compositional_generalization`.

Single-model usage::

    python -m experiments.bootstrap_pair_metrics \\
        --per_edge runs/comp/per_edge.jsonl \\
        --B 5000 --output runs/comp/bootstrap.json

Paired comparison (model A vs. model B on the **same** held-out edges)::

    python -m experiments.bootstrap_pair_metrics \\
        --per_edge runs/comp/per_edge.jsonl \\
        --per_edge_b runs/unary/per_edge.jsonl \\
        --label_a unary_plus_pair --label_b unary_only \\
        --B 5000 --output runs/comp/paired_bootstrap.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("bootstrap_pair_metrics")

DEFAULT_METRICS: Tuple[str, ...] = (
    # Held-out programme `e*` on the **full** length-≤2 menu only. The
    # ``*_minus_chance`` deltas are signed: positive ⇒ above uniform
    # chance; negative ⇒ below. ``full_chance_minus_rank`` flips the
    # sign of rank so positive ⇒ better than chance.
    "full_mean_prob", "full_mean_log_prob",
    "full_mean_rank", "full_mean_rank_norm", "full_mean_lift",
    "full_top1_rate", "full_top3_rate", "full_top5_rate",
    "full_chance_prob", "full_chance_rank",
    "full_chance_top1", "full_chance_top3", "full_chance_top5",
    "full_prob_minus_chance", "full_chance_minus_rank",
    "full_top1_minus_chance", "full_top3_minus_chance", "full_top5_minus_chance",
)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _filter_edges_with_questions(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if int(r.get("n_questions", 0)) > 0]


def _index_by_edge_key(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        key = r.get("edge_key")
        if key is None:
            edge = r.get("edge") or [-1, -1]
            key = f"{int(edge[0])}-{int(edge[1])}"
        out[str(key)] = r
    return out


def _values(rows: Sequence[Dict[str, Any]], metric: str) -> np.ndarray:
    vals: List[float] = []
    for r in rows:
        v = r.get(metric)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        vals.append(float(v))
    return np.asarray(vals, dtype=np.float64)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    B: int,
    seed: int,
    alpha: float = 0.05,
    chunk_max_cells: int = 8_000_000,
) -> Dict[str, float]:
    """Bootstrap CI for the mean of ``values``.

    Memory-bounded: rather than allocating a ``[B, n]`` index matrix
    (which can be many GB for large n), draw replicates in chunks whose
    ``B_chunk × n`` size is below ``chunk_max_cells``.
    """
    n = int(values.size)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0, "B": int(B)}
    rng = np.random.default_rng(seed)
    sample_means = np.empty(B, dtype=np.float64)
    chunk = max(1, min(B, chunk_max_cells // max(n, 1)))
    pos = 0
    while pos < B:
        b = min(chunk, B - pos)
        idx = rng.integers(0, n, size=(b, n))
        sample_means[pos:pos + b] = values[idx].mean(axis=1)
        pos += b
    lo = float(np.percentile(sample_means, 100 * (alpha / 2)))
    hi = float(np.percentile(sample_means, 100 * (1 - alpha / 2)))
    return {
        "mean": float(values.mean()),
        "lo": lo,
        "hi": hi,
        "n": n,
        "B": int(B),
        "se": float(sample_means.std(ddof=1)) if B > 1 else float("nan"),
    }


def bootstrap_paired_diff(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    B: int,
    seed: int,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Mean of paired differences ``a - b`` with bootstrap CI."""
    if values_a.shape != values_b.shape:
        raise ValueError(f"shape mismatch: {values_a.shape} vs {values_b.shape}")
    n = int(values_a.size)
    if n == 0:
        return {"mean_diff": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0, "B": int(B)}
    diffs = values_a - values_b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    sample_means = diffs[idx].mean(axis=1)
    lo = float(np.percentile(sample_means, 100 * (alpha / 2)))
    hi = float(np.percentile(sample_means, 100 * (1 - alpha / 2)))
    return {
        "mean_diff": float(diffs.mean()),
        "lo": lo,
        "hi": hi,
        "n": n,
        "B": int(B),
        "fraction_positive": float((diffs > 0).mean()),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_single(
    rows: Sequence[Dict[str, Any]],
    metrics: Sequence[str],
    *,
    B: int,
    seed: int,
    alpha: float,
) -> Dict[str, Any]:
    """Edge-level bootstrap: resample the per-edge summary rows."""
    rows = _filter_edges_with_questions(rows)
    out: Dict[str, Any] = {
        "mode": "edge",
        "n_edges_with_questions": len(rows),
        "B": int(B),
        "seed": int(seed),
        "alpha": float(alpha),
        "metrics": {},
    }
    for m in metrics:
        out["metrics"][m] = bootstrap_mean_ci(
            _values(rows, m), B=B, seed=seed, alpha=alpha,
        )
    return out


def run_paired(
    rows_a: Sequence[Dict[str, Any]],
    rows_b: Sequence[Dict[str, Any]],
    metrics: Sequence[str],
    *,
    B: int,
    seed: int,
    alpha: float,
    label_a: str,
    label_b: str,
) -> Dict[str, Any]:
    a_idx = _index_by_edge_key(_filter_edges_with_questions(rows_a))
    b_idx = _index_by_edge_key(_filter_edges_with_questions(rows_b))
    common_keys = sorted(set(a_idx).intersection(b_idx))
    out: Dict[str, Any] = {
        "n_common_edges": len(common_keys),
        "B": int(B),
        "seed": int(seed),
        "alpha": float(alpha),
        "label_a": label_a,
        "label_b": label_b,
        "metrics": {},
    }
    for m in metrics:
        a_vals: List[float] = []
        b_vals: List[float] = []
        for k in common_keys:
            ar = a_idx[k].get(m)
            br = b_idx[k].get(m)
            if ar is None or br is None:
                continue
            if isinstance(ar, float) and math.isnan(ar):
                continue
            if isinstance(br, float) and math.isnan(br):
                continue
            a_vals.append(float(ar))
            b_vals.append(float(br))
        a_arr = np.asarray(a_vals, dtype=np.float64)
        b_arr = np.asarray(b_vals, dtype=np.float64)
        out["metrics"][m] = {
            f"{label_a}_marginal": bootstrap_mean_ci(a_arr, B=B, seed=seed, alpha=alpha),
            f"{label_b}_marginal": bootstrap_mean_ci(b_arr, B=B, seed=seed + 1, alpha=alpha),
            "paired_diff": bootstrap_paired_diff(
                a_arr, b_arr, B=B, seed=seed + 2, alpha=alpha,
            ),
        }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per_edge", required=True, type=Path,
                   help="per_edge.jsonl from the eval script (model A).")
    p.add_argument("--per_edge_b", type=Path, default=None,
                   help="Optional per_edge.jsonl for a second model (paired comparison).")
    p.add_argument("--label_a", default="model_a")
    p.add_argument("--label_b", default="model_b")
    p.add_argument("--metrics", nargs="*", default=list(DEFAULT_METRICS))
    p.add_argument("--B", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Two-sided significance level (CI width 1-alpha).")
    p.add_argument("--output", type=Path, default=None,
                   help="Where to write the JSON result (default: alongside --per_edge).")
    p.add_argument("--log_level", default="INFO")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    rows_a = _read_jsonl(args.per_edge)
    if args.per_edge_b is not None:
        rows_b = _read_jsonl(args.per_edge_b)
        result: Dict[str, Any] = run_paired(
            rows_a, rows_b, args.metrics,
            B=args.B, seed=args.seed, alpha=args.alpha,
            label_a=args.label_a, label_b=args.label_b,
        )
        default_name = "paired_bootstrap.json"
    else:
        result = {
            "per_edge_path": str(args.per_edge),
            "edge": run_single(
                rows_a, args.metrics,
                B=args.B, seed=args.seed, alpha=args.alpha,
            ),
        }
        default_name = "bootstrap.json"
    out_path = args.output or (args.per_edge.parent / default_name)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("wrote bootstrap result -> %s", out_path)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
