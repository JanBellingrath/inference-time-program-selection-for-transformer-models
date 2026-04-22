"""Constrained held-out edge split for compositional generalization (Step 3).

Reads a ``pair_program_graph.json`` produced by
:mod:`data_prep.build_pair_program_graph` and emits a reproducible
``holdout_split.json`` whose ``E_val_holdout`` and ``E_test_holdout``
edge sets satisfy the eligibility predicates from the spec:

* edge support: ``n(o_i, o_j) >= c_min``
* endpoint support: ``n(o_i) >= u_min`` and ``n(o_j) >= u_min``
* residual degree: ``deg(o_i) >= d_min`` and ``deg(o_j) >= d_min``
  (full empirical degree, computed before the split)
* optional positivity: ``mean_delta(o_i, o_j) >= delta_min``

Endpoint support uses ``n_questions_any(o)`` (length-1 OR length-2
occurrences), since pair endpoints can also be learned from the singletons
that survive filtering.

The greedy split shuffles eligible edges with a fixed seed and assigns
them to validation / test as long as the assignment does not violate per-
endpoint caps and would not push any endpoint's *remaining train degree*
below ``d_min``.

Usage::

    python -m experiments.holdout_edge_split \\
        --graph compositional_runs/csqa_compositional/pair_graph/pair_program_graph.json \\
        --output_dir compositional_runs/csqa_compositional/holdout \\
        --c_min 5 --u_min 25 --d_min 2 --delta_min 0.0 \\
        --val_fraction 0.15 --test_fraction 0.30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logger = logging.getLogger("holdout_edge_split")

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _load_graph(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        graph = json.load(f)
    if int(graph.get("schema_version", 0)) != 1:
        logger.warning("pair_program_graph.json schema_version != 1: %s",
                       graph.get("schema_version"))
    return graph


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


def _eligible_edges(
    edges: Sequence[Dict[str, Any]],
    primitives: Sequence[Dict[str, Any]],
    *,
    c_min: int,
    u_min: int,
    d_min: int,
    delta_min: Optional[float],
) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    """Return eligible edges (preserves graph order) plus full empirical degree.

    Endpoint stats use ``n_questions_any`` and full ``deg`` from the graph.
    """
    deg: Dict[int, int] = {p["idx"]: int(p["deg"]) for p in primitives}
    n_q_any: Dict[int, int] = {p["idx"]: int(p["n_questions_any"]) for p in primitives}

    eligible: List[Dict[str, Any]] = []
    for e in edges:
        if int(e["count"]) < c_min:
            continue
        if int(e["n_questions"]) < c_min:
            # require c_min on (questions, count) -- the more conservative pair.
            # Use n_questions which is the de-duplicated count.
            pass
        a, b = int(e["a"]), int(e["b"])
        if n_q_any.get(a, 0) < u_min or n_q_any.get(b, 0) < u_min:
            continue
        if deg.get(a, 0) < d_min or deg.get(b, 0) < d_min:
            continue
        if delta_min is not None and float(e.get("mean_delta", 0.0)) < delta_min:
            continue
        eligible.append(e)
    return eligible, deg


# ---------------------------------------------------------------------------
# Greedy constrained assignment
# ---------------------------------------------------------------------------


def _split_eligible(
    eligible: Sequence[Dict[str, Any]],
    *,
    full_deg: Dict[int, int],
    val_fraction: float,
    test_fraction: float,
    d_min: int,
    max_holdouts_per_endpoint: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Greedy randomized assignment val / test / train.

    Constraints:
    * at most ``max_holdouts_per_endpoint`` total val+test edges per primitive,
    * each endpoint's *remaining train degree* (full empirical degree minus
      number of held-out edges incident to it) stays >= ``d_min``.
    """
    rng = random.Random(seed)
    order = list(range(len(eligible)))
    rng.shuffle(order)

    n = len(eligible)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    if n_test + n_val > n:
        n_val = max(0, n - n_test)

    held_count: Dict[int, int] = defaultdict(int)
    val_idx: List[int] = []
    test_idx: List[int] = []
    train_idx: List[int] = []

    for pos in order:
        e = eligible[pos]
        a, b = int(e["a"]), int(e["b"])

        # If neither budget remains, stop trying to hold out.
        want_val = len(val_idx) < n_val
        want_test = len(test_idx) < n_test
        if not (want_val or want_test):
            train_idx.append(pos)
            continue

        # Per-endpoint cap.
        if held_count[a] + 1 > max_holdouts_per_endpoint or held_count[b] + 1 > max_holdouts_per_endpoint:
            train_idx.append(pos)
            continue

        # Residual degree predicate.
        rem_a = full_deg.get(a, 0) - (held_count[a] + 1)
        rem_b = full_deg.get(b, 0) - (held_count[b] + 1)
        if rem_a < d_min or rem_b < d_min:
            train_idx.append(pos)
            continue

        # Prefer test (larger budget) first when both want, else fill the open one.
        target = "test" if want_test and (not want_val or len(test_idx) <= len(val_idx) * (n_test / max(n_val, 1))) else "val"
        if target == "test" and not want_test:
            target = "val"
        if target == "val" and not want_val:
            target = "test"

        held_count[a] += 1
        held_count[b] += 1
        if target == "test":
            test_idx.append(pos)
        else:
            val_idx.append(pos)

    val_edges = [eligible[i] for i in val_idx]
    test_edges = [eligible[i] for i in test_idx]
    train_edges = [eligible[i] for i in train_idx]
    return val_edges, test_edges, train_edges


def _summarize_edges(edges: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not edges:
        return {"n_edges": 0, "total_count": 0, "total_n_questions": 0,
                "mean_count": 0.0, "mean_n_questions": 0.0, "mean_delta": 0.0}
    total_c = sum(int(e["count"]) for e in edges)
    total_q = sum(int(e["n_questions"]) for e in edges)
    mean_d = sum(float(e["mean_delta"]) for e in edges) / len(edges)
    return {
        "n_edges": len(edges),
        "total_count": total_c,
        "total_n_questions": total_q,
        "mean_count": total_c / len(edges),
        "mean_n_questions": total_q / len(edges),
        "mean_delta": mean_d,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_holdout_split(
    graph_path: Path,
    output_dir: Path,
    *,
    c_min: int,
    u_min: int,
    d_min: int,
    delta_min: Optional[float],
    val_fraction: float,
    test_fraction: float,
    max_holdouts_per_endpoint: int,
    seed: int,
) -> Dict[str, Any]:
    graph_path = Path(graph_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = _load_graph(graph_path)
    primitives = graph["primitives"]
    edges = graph["edges"]

    eligible, full_deg = _eligible_edges(
        edges, primitives,
        c_min=c_min, u_min=u_min, d_min=d_min, delta_min=delta_min,
    )

    val_edges, test_edges, train_edges = _split_eligible(
        eligible,
        full_deg=full_deg,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        d_min=d_min,
        max_holdouts_per_endpoint=max_holdouts_per_endpoint,
        seed=seed,
    )

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "graph_path": str(graph_path),
        "graph_sha256": _file_sha256(graph_path),
        "catalogue_dir": graph.get("catalogue_dir"),
        "benchmarks": graph.get("benchmarks", []),
        "thresholds": {
            "c_min": int(c_min),
            "u_min": int(u_min),
            "d_min": int(d_min),
            "delta_min": delta_min,
            "max_holdouts_per_endpoint": int(max_holdouts_per_endpoint),
        },
        "split": {
            "seed": int(seed),
            "val_fraction": float(val_fraction),
            "test_fraction": float(test_fraction),
        },
        "totals": {
            "n_primitives": len(primitives),
            "n_edges": len(edges),
            "n_eligible": len(eligible),
        },
        "summaries": {
            "val": _summarize_edges(val_edges),
            "test": _summarize_edges(test_edges),
            "train": _summarize_edges(train_edges),
        },
        "E_val_holdout": [_serialize_edge(e) for e in val_edges],
        "E_test_holdout": [_serialize_edge(e) for e in test_edges],
        "E_train_eligible": [_serialize_edge(e) for e in train_edges],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = output_dir / "holdout_split.json"
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    logger.info(
        "holdout split: |val|=%d |test|=%d |train_eligible|=%d / |eligible|=%d  -> %s",
        len(val_edges), len(test_edges), len(train_edges), len(eligible), out_path,
    )
    return artifact


def _serialize_edge(edge: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "a": int(edge["a"]),
        "b": int(edge["b"]),
        "key_a": edge["key_a"],
        "key_b": edge["key_b"],
        "count": int(edge["count"]),
        "n_questions": int(edge["n_questions"]),
        "mean_delta": float(edge["mean_delta"]),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph", required=True, type=Path,
                   help="pair_program_graph.json from data_prep.build_pair_program_graph.")
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--c_min", type=int, default=5,
                   help="Minimum edge count and minimum n_questions per edge.")
    p.add_argument("--u_min", type=int, default=25,
                   help="Minimum endpoint primitive question support (any length).")
    p.add_argument("--d_min", type=int, default=2,
                   help="Minimum residual primitive degree after holdout.")
    p.add_argument("--delta_min", type=float, default=None,
                   help="Optional positive-delta filter on edge mean_delta.")
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--test_fraction", type=float, default=0.30)
    p.add_argument("--max_holdouts_per_endpoint", type=int, default=2,
                   help="Cap on edges held out per endpoint primitive.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_level", default="INFO")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    if not args.graph.is_file():
        logger.error("graph file not found: %s", args.graph)
        return 2
    make_holdout_split(
        graph_path=args.graph,
        output_dir=args.output_dir,
        c_min=args.c_min,
        u_min=args.u_min,
        d_min=args.d_min,
        delta_min=args.delta_min,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        max_holdouts_per_endpoint=args.max_holdouts_per_endpoint,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
