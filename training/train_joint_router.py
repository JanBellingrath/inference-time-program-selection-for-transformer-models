#!/usr/bin/env python3
"""Train a joint router: pivot residual -> route across multiple benchmarks.

Architecture: ``x ↦ a, a ∈ {STAY} ∪ R_global``

*   **One unified route catalog** — by default every unique layer sequence that appears
    in any benchmark's MCTS ``explored`` set gets exactly one index.  Use
    ``--catalog_mode intersection`` for only sequences in the intersection across
    all benchmarks' ``explored`` sets.
    Index 0 is the global **STAY** action (use the benchmark's anchor).
*   **No benchmark masking** — the softmax is over the full catalog for every
    sample, regardless of which benchmark it came from.
*   **One global STAY** — class 0 means "keep the benchmark's own anchor
    sequence".  Per-benchmark anchors map to STAY during target construction.
*   **No duplicate actions** — identical sequence tuples share one class index
    across all benchmarks.

Usage
-----
    python train_joint_router.py \\
        --data_dir fine_routing_data_boolq_csqa \\
        --benchmarks boolq commonsenseqa \\
        --hidden_dims 1024 1024 512 \\
        --compressor_type last_token

    python train_joint_router.py \\
        --data_dir fine_routing_data_boolq_csqa \\
        --benchmarks boolq commonsenseqa \\
        --hidden_dims 1024 512 \\
        --compressor_type top_down_attention \\
        --compressor_d_compress 256 --compressor_n_heads 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, random_split

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from routers.residual_compressors import (
    CompressorConfig,
    CompressedRouter,
    DualEncoderRouter,
    build_compressor,
    pad_sequences,
    prepare_catalog_tensors,
)
from routers.shared_router import masked_soft_cross_entropy
from routers.fine_routing_deviations import (
    enumerate_deviations,
    apply_deviation,
    seq_to_layers as deviation_seq_to_layers,
)
from experiments.sweep_fine_routing import (
    build_mcts_sequence_catalog,
    build_mcts_router_targets,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None  # type: ignore


# ---------------------------------------------------------------------------
# Anchor loading helpers
# ---------------------------------------------------------------------------

def _load_anchor_from_jsonl(data_dir: str, benchmark: str) -> Optional[List[int]]:
    """Extract ``anchor_sequence`` from the first record of a benchmark's JSONL."""
    jsonl_path = os.path.join(data_dir, f"{benchmark}.jsonl")
    if not os.path.isfile(jsonl_path):
        return None
    with open(jsonl_path) as f:
        first_line = f.readline()
    if not first_line.strip():
        return None
    rec = json.loads(first_line)
    seq = rec.get("anchor_sequence")
    if seq is not None:
        return [int(x) for x in seq]
    return None


def _load_anchors(
    data_dir: str,
    benchmarks: List[str],
    anchor_json: Optional[str] = None,
    results_dir: Optional[str] = None,
) -> Dict[str, List[int]]:
    """Resolve anchor sequences for each benchmark.

    Priority:  anchor_json > JSONL first-record > MCTS snapshots > identity.
    """
    anchors: Dict[str, List[int]] = {}

    if anchor_json and os.path.isfile(anchor_json):
        with open(anchor_json) as f:
            loaded = json.load(f)
        for b in benchmarks:
            if b in loaded:
                anchors[b] = [int(x) for x in loaded[b]]
        if anchors:
            logger.info("Loaded anchors from %s for %s", anchor_json, list(anchors.keys()))

    for b in benchmarks:
        if b in anchors:
            continue
        seq = _load_anchor_from_jsonl(data_dir, b)
        if seq is not None:
            anchors[b] = seq
            logger.info("Loaded %s anchor from JSONL first record (%d layers)", b, len(seq))

    if results_dir:
        try:
            from training.train_benchmark_router import load_optimal_sequences_from_results
            parsed = load_optimal_sequences_from_results(results_dir, benchmarks)
            for b in benchmarks:
                if b not in anchors and b in parsed:
                    anchors[b] = parsed[b]
                    logger.info("Loaded %s anchor from MCTS snapshots", b)
        except ImportError:
            logger.warning("Could not import load_optimal_sequences_from_results")

    for b in benchmarks:
        if b not in anchors:
            logger.warning("No anchor for %s; this benchmark will be skipped", b)

    return anchors


def _check_anchor_compatibility(
    per_bench_anchors: Dict[str, List[int]],
) -> None:
    """Warn if anchor prefixes differ across benchmarks."""
    bench_list = sorted(per_bench_anchors.keys())
    if len(bench_list) < 2:
        return
    a0 = per_bench_anchors[bench_list[0]]
    for b in bench_list[1:]:
        a1 = per_bench_anchors[b]
        min_len = min(len(a0), len(a1))
        first_diff = None
        for i in range(min_len):
            if a0[i] != a1[i]:
                first_diff = i
                break
        if first_diff is None:
            logger.info(
                "Anchor compatibility OK: %s and %s share prefix of length %d",
                bench_list[0], b, min_len,
            )
        else:
            logger.warning(
                "Anchor mismatch at position %d: %s has %d, %s has %d. "
                "Pivot residuals may not be directly comparable.",
                first_diff, bench_list[0], a0[first_diff], b, a1[first_diff],
            )


# ---------------------------------------------------------------------------
# Unified catalog:  {STAY} ∪ R_global
# ---------------------------------------------------------------------------

STAY_INDEX = 0


def build_unified_catalog(
    per_bench_records: Dict[str, List[Dict]],
    per_bench_anchors: Dict[str, List[int]],
    bench_names: List[str],
) -> Tuple[int, List[List[int]], Dict[tuple, int]]:
    """Build a single deduplicated route catalog with a global STAY action.

    Returns
    -------
    num_classes : int
        ``1 + |R_global|``.  Index 0 = STAY; indices 1..num_classes-1 are
        distinct layer sequences.
    catalog : list of list of int
        ``catalog[i]`` for ``i >= 1`` is the layer sequence for class *i*.
        ``catalog[0]`` is ``None`` (sentinel for STAY).
    seq_to_idx : dict
        Maps ``tuple(sequence)`` → global class index (always ≥ 1).
        Anchor sequences are **included** here (they get an index) so that
        benchmarks whose anchor differs can route *to* another benchmark's
        anchor.  The anchor→STAY re-mapping is done in the target builder.
    """
    all_seqs: Dict[tuple, None] = {}
    for bench in bench_names:
        for rec in per_bench_records[bench]:
            for ex in rec["explored"]:
                all_seqs[tuple(int(x) for x in ex["seq"])] = None

    route_list = sorted(all_seqs.keys())
    catalog: List[Optional[List[int]]] = [None] + [list(s) for s in route_list]
    seq_to_idx = {s: i + 1 for i, s in enumerate(route_list)}
    num_classes = len(catalog)

    logger.info("Unified catalog: |R_global|=%d (+STAY → G=%d)", len(route_list), num_classes)
    for bench in bench_names:
        anchor_t = tuple(int(x) for x in per_bench_anchors[bench])
        logger.info(
            "  %s: anchor → STAY (seq also at idx %s)",
            bench, seq_to_idx.get(anchor_t, "N/A"),
        )
    return num_classes, catalog, seq_to_idx


def explored_sequences_union(
    per_bench_records: Dict[str, List[Dict]],
    bench: str,
) -> Set[tuple]:
    """All distinct ``explored`` layer sequences for one benchmark."""
    out: Set[tuple] = set()
    for rec in per_bench_records.get(bench, []):
        for ex in rec.get("explored", []):
            out.add(tuple(int(x) for x in ex["seq"]))
    return out


def intersection_explored_sequences(
    per_bench_records: Dict[str, List[Dict]],
    bench_names: List[str],
) -> Set[tuple]:
    """⋂_b { sequences appearing in MCTS ``explored`` for benchmark *b* }."""
    if not bench_names:
        return set()
    sets = [explored_sequences_union(per_bench_records, b) for b in bench_names]
    inter = set.intersection(*sets) if sets else set()
    return inter


def build_intersection_catalog(
    per_bench_records: Dict[str, List[Dict]],
    per_bench_anchors: Dict[str, List[int]],
    bench_names: List[str],
) -> Tuple[int, List[Optional[List[int]]], Dict[tuple, int]]:
    """Catalog ``{{STAY}} ∪ ⋂_b explored(b)`` — same indexing contract as :func:`build_unified_catalog`."""
    isect = intersection_explored_sequences(per_bench_records, bench_names)
    if not isect:
        raise ValueError(
            "Intersection catalog is empty: no layer sequence lies in MCTS explored "
            "for every benchmark.",
        )
    route_list = sorted(isect)
    catalog: List[Optional[List[int]]] = [None] + [list(s) for s in route_list]
    seq_to_idx = {s: i + 1 for i, s in enumerate(route_list)}
    num_classes = len(catalog)
    logger.info(
        "Intersection catalog: |∩ explored|=%d (+STAY → G=%d)",
        len(route_list),
        num_classes,
    )
    for bench in bench_names:
        anchor_t = tuple(int(x) for x in per_bench_anchors[bench])
        logger.info(
            "  %s: anchor → STAY (seq also at idx %s)",
            bench, seq_to_idx.get(anchor_t, "N/A"),
        )
    return num_classes, catalog, seq_to_idx


def build_route_catalog(
    per_bench_records: Dict[str, List[Dict]],
    per_bench_anchors: Dict[str, List[int]],
    bench_names: List[str],
    catalog_mode: str,
) -> Tuple[int, List[Optional[List[int]]], Dict[tuple, int]]:
    """Union (default) or intersection of per-benchmark MCTS explored sets."""
    if catalog_mode == "union":
        return build_unified_catalog(per_bench_records, per_bench_anchors, bench_names)
    if catalog_mode == "intersection":
        return build_intersection_catalog(per_bench_records, per_bench_anchors, bench_names)
    raise ValueError(f"catalog_mode must be 'union' or 'intersection', got {catalog_mode!r}")


def build_deviation_catalog(
    per_bench_anchors: Dict[str, List[int]],
    bench_names: List[str],
    num_layers: int,
    editable_start: int = 17,
    swap_radius: int = 2,
    max_edits: int = 2,
) -> Tuple[int, List[List[int]], Dict[tuple, int]]:
    """Build a catalog from local deviations (skip/swap/repeat, <=max_edits).

    Reuses ``enumerate_deviations`` from ``fine_routing_deviations`` to produce
    a small, structured route space around each benchmark's anchor, then
    deduplicates across benchmarks.

    Returns the same ``(num_classes, catalog, seq_to_idx)`` triple as
    ``build_unified_catalog``.
    """
    all_seqs: Dict[tuple, None] = {}
    for bench in bench_names:
        anchor = per_bench_anchors[bench]
        deviations = enumerate_deviations(
            anchor, editable_start, num_layers,
            swap_radius=swap_radius, max_edits=max_edits,
        )
        for dev in deviations:
            seq = apply_deviation(anchor, dev)
            all_seqs[tuple(seq)] = None

    for bench in bench_names: #TODO going back to the anchors might be good for safety
        all_seqs.pop(tuple(int(x) for x in per_bench_anchors[bench]), None)

    route_list = sorted(all_seqs.keys())
    catalog: List[Optional[List[int]]] = [None] + [list(s) for s in route_list]
    seq_to_idx = {s: i + 1 for i, s in enumerate(route_list)}
    num_classes = len(catalog)

    logger.info(
        "Deviation catalog: editable_start=%d swap_radius=%d max_edits=%d "
        "|R|=%d (+STAY → G=%d)",
        editable_start, swap_radius, max_edits, len(route_list), num_classes,
    )
    return num_classes, catalog, seq_to_idx


def build_unified_targets(
    records: List[Dict],
    seq_to_idx: Dict[tuple, int],
    num_classes: int,
    anchor_seq: List[int],
    noop_boost: float = 0.0,
) -> List[torch.Tensor]:
    """Build soft targets on the unified catalog for one benchmark's records.

    The benchmark's anchor sequence is mapped to STAY (index 0).  All other
    explored sequences are mapped to their global index.

    Parameters
    ----------
    noop_boost : float
        Extra mass added to STAY for gate_label=0 samples (biases router
        toward "keep anchor").  For gate_label=1, STAY mass is zeroed and
        re-normalised over route classes.
    """
    anchor_t = tuple(int(x) for x in anchor_seq)
    out: List[torch.Tensor] = []
    for rec in records:
        explored = rec["explored"]
        rt = rec["router_target"]
        p = torch.zeros(num_classes, dtype=torch.float32)
        for j, prob in enumerate(rt):
            seq = tuple(int(x) for x in explored[j]["seq"])
            if seq == anchor_t:
                p[STAY_INDEX] += float(prob)
            else:
                idx = seq_to_idx.get(seq)
                if idx is not None:
                    p[idx] += float(prob)

        s = float(p.sum())
        if s > 1e-12:
            p = p / s
        else:
            p[STAY_INDEX] = 1.0

        if noop_boost > 0:
            if rec["gate_label"] == 0:
                p[STAY_INDEX] += noop_boost
                p = p / p.sum()
            else:
                p[STAY_INDEX] = 0.0
                s2 = float(p.sum())
                if s2 > 1e-12:
                    p = p / s2
                else:
                    p[STAY_INDEX] = 1.0

        out.append(p)
    return out


def build_unified_topk_soft_targets(
    records: List[Dict],
    seq_to_idx: Dict[tuple, int],
    num_classes: int,
    anchor_seq: List[int],
    topk: int = 5,
    noop_boost: float = 0.0,
) -> List[torch.Tensor]:
    """Soft targets on the unified catalog using only the top-*K* explored sequences.

    For each record, sequences are ranked by ``explored[j]["score"]`` (descending).
    Only the first *K* entries are kept; their ``router_target`` masses are
    renormalized to sum to 1, then mapped into the catalog (anchor → STAY) like
    :func:`build_unified_targets`.  Mass on sequences missing from
    ``seq_to_idx`` is dropped and the vector is renormalized again; if nothing
    remains, the target defaults to STAY.
    """
    if topk < 1:
        raise ValueError(f"topk must be >= 1, got {topk}")
    anchor_t = tuple(int(x) for x in anchor_seq)
    out: List[torch.Tensor] = []
    for rec in records:
        explored = rec["explored"]
        rt = rec["router_target"]
        if len(explored) != len(rt):
            raise ValueError(
                f"explored ({len(explored)}) and router_target ({len(rt)}) length mismatch",
            )
        pairs = [
            (j, float(explored[j]["score"]), float(rt[j]))
            for j in range(len(explored))
        ]
        pairs.sort(key=lambda x: (-x[1], x[0]))
        take = pairs[: min(topk, len(pairs))]
        raw_probs = [t[2] for t in take]
        s_raw = float(sum(raw_probs))
        if s_raw > 1e-12:
            norm_probs = [p / s_raw for p in raw_probs]
        elif take:
            norm_probs = [1.0 / len(take)] * len(take)
        else:
            p = torch.zeros(num_classes, dtype=torch.float32)
            p[STAY_INDEX] = 1.0
            out.append(p)
            continue

        p = torch.zeros(num_classes, dtype=torch.float32)
        for (j, _, _), prob in zip(take, norm_probs):
            seq = tuple(int(x) for x in explored[j]["seq"])
            if seq == anchor_t:
                p[STAY_INDEX] += prob
            else:
                idx = seq_to_idx.get(seq)
                if idx is not None:
                    p[idx] += prob

        s = float(p.sum())
        if s > 1e-12:
            p = p / s
        else:
            p[STAY_INDEX] = 1.0

        if noop_boost > 0:
            if rec["gate_label"] == 0:
                p[STAY_INDEX] += noop_boost
                p = p / p.sum()
            else:
                p[STAY_INDEX] = 0.0
                s2 = float(p.sum())
                if s2 > 1e-12:
                    p = p / s2
                else:
                    p[STAY_INDEX] = 1.0

        out.append(p)
    return out


def build_unified_best_seq_targets(
    records: List[Dict],
    seq_to_idx: Dict[tuple, int],
    num_classes: int,
    anchor_seq: List[int],
) -> List[torch.Tensor]:
    """Hard (one-hot) targets: gate_label=0 → STAY, gate_label=1 → best_seq."""
    anchor_t = tuple(int(x) for x in anchor_seq)
    out: List[torch.Tensor] = []
    for rec in records:
        p = torch.zeros(num_classes, dtype=torch.float32)
        if rec.get("gate_label", 0) == 0:
            p[STAY_INDEX] = 1.0
        else:
            best = tuple(int(x) for x in rec["best_seq"])
            if best == anchor_t:
                p[STAY_INDEX] = 1.0
            else:
                idx = seq_to_idx.get(best, STAY_INDEX)
                p[idx] = 1.0
        out.append(p)
    return out


def load_selected_catalog(
    catalog_json: str,
) -> Tuple[int, List[Optional[List[int]]], Dict[tuple, int]]:
    """Load a selected catalog JSON produced by select_route_catalog.py.

    Returns
    -------
    num_classes : int
        ``1 + len(selected_routes)`` (global STAY at index 0).
    catalog : list
        ``catalog[0]`` is ``None`` (STAY), then selected routes.
    seq_to_idx : dict
        Route tuple -> global class index (>= 1).
    """
    with open(catalog_json) as f:
        payload = json.load(f)
    routes = [[int(x) for x in r] for r in payload["selected_routes"]]
    catalog: List[Optional[List[int]]] = [None] + routes
    seq_to_idx = {tuple(r): i + 1 for i, r in enumerate(routes)}
    return len(catalog), catalog, seq_to_idx


def _dense_best_class_idx(rec: Dict[str, Any]) -> int:
    """Best class from dense deltas (<=0 -> STAY)."""
    route_deltas = rec.get("route_deltas", {})
    if not route_deltas:
        return STAY_INDEX
    best_rid = None
    best_delta = float("-inf")
    for rid_s, delta in route_deltas.items():
        d = float(delta)
        if d > best_delta:
            best_delta = d
            best_rid = int(rid_s)
    if best_rid is None or best_delta <= 0.0:
        return STAY_INDEX
    return best_rid + 1  # route-id 0..K-1 maps to class 1..K


def _dense_topk_soft_target(
    rec: Dict[str, Any],
    num_classes: int,
    topk: int,
) -> Tuple[torch.Tensor, int]:
    """Uniform soft target over up to ``topk`` routes with Δ>0, always including the dense argmax route.

    Routes are prioritized by ``(-delta, route_id)``. The dense-best class
    (same rule as :func:`_dense_best_class_idx`) is always in the support set
    so the label remains consistent with hard supervision.
    """
    if topk < 1:
        raise ValueError(f"topk must be >= 1, got {topk}")
    hard_cls = _dense_best_class_idx(rec)
    p = torch.zeros(num_classes, dtype=torch.float32)
    if hard_cls == STAY_INDEX:
        p[STAY_INDEX] = 1.0
        return p, hard_cls
    best_rid = hard_cls - 1
    route_deltas = rec.get("route_deltas", {})
    pos: List[Tuple[int, float]] = []
    for rid_s, delta in route_deltas.items():
        d = float(delta)
        if d <= 0.0:
            continue
        rid = int(rid_s)
        cls = rid + 1
        if cls < 1 or cls >= num_classes:
            continue
        pos.append((rid, d))
    if not pos:
        p[STAY_INDEX] = 1.0
        return p, hard_cls
    pos.sort(key=lambda x: (-x[1], x[0]))
    selected: List[int] = []
    seen: Set[int] = set()
    selected.append(best_rid)
    seen.add(best_rid)
    for rid, _ in pos:
        if rid in seen:
            continue
        if len(selected) >= topk:
            break
        selected.append(rid)
        seen.add(rid)
    w = 1.0 / len(selected)
    for rid in selected:
        p[rid + 1] = w
    s = float(p.sum())
    if s > 1e-12:
        p = p / s
    else:
        p[STAY_INDEX] = 1.0
    return p, hard_cls


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class JointRouterDataset(Dataset):
    """Pivot residuals ``[d_model]`` + unified router targets from multiple benchmarks."""

    def __init__(
        self,
        data_dir: str,
        benchmarks: List[str],
        per_bench_anchors: Dict[str, List[int]],
        gate_positives_only: bool = True,
        noop_boost: float = 0.0,
        catalog_mode: str = "union",
    ):
        self.residuals: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.bench_ids: List[str] = []

        per_bench_records: Dict[str, List[Dict]] = {}
        per_bench_residuals: Dict[str, torch.Tensor] = {}
        self.bench_names: List[str] = []

        for bench in benchmarks:
            if bench not in per_bench_anchors:
                continue
            pt_path = os.path.join(data_dir, f"{bench}_pivot_residuals.pt")
            jsonl_path = os.path.join(data_dir, f"{bench}.jsonl")
            if not os.path.isfile(pt_path) or not os.path.isfile(jsonl_path):
                logger.warning("Missing data for %s, skipping", bench)
                continue

            residuals = torch.load(pt_path, map_location="cpu", weights_only=True).float()
            with open(jsonl_path) as f:
                records = [json.loads(line) for line in f]
            n = min(residuals.shape[0], len(records))
            if n < residuals.shape[0] or n < len(records):
                logger.warning(
                    "%s: truncating to %d rows (pt=%d, jsonl=%d)",
                    bench, n, residuals.shape[0], len(records),
                )
            per_bench_records[bench] = records[:n]
            per_bench_residuals[bench] = residuals[:n]
            self.bench_names.append(bench)

        _check_anchor_compatibility(per_bench_anchors)

        (self.num_classes, self.catalog, self.seq_to_idx) = build_route_catalog(
            per_bench_records, per_bench_anchors, self.bench_names, catalog_mode,
        )

        for bench in self.bench_names:
            records = per_bench_records[bench]
            residuals = per_bench_residuals[bench]
            targets = build_unified_targets(
                records, self.seq_to_idx, self.num_classes,
                per_bench_anchors[bench], noop_boost=noop_boost,
            )
            for i, (rec, tgt) in enumerate(zip(records, targets)):
                if gate_positives_only and rec["gate_label"] == 0:
                    continue
                self.residuals.append(residuals[i])
                self.targets.append(tgt)
                self.bench_ids.append(bench)

        logger.info(
            "JointRouterDataset: %d samples, G=%d, catalog_mode=%s, benchmarks=%s",
            len(self.targets), self.num_classes, catalog_mode, self.bench_names,
        )
        self.bench_to_idx: Dict[str, int] = {b: i for i, b in enumerate(self.bench_names)}
        self.bench_idx_per_sample: List[int] = [self.bench_to_idx[b] for b in self.bench_ids]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.residuals[idx], self.targets[idx], self.bench_idx_per_sample[idx]


class JointFullSequenceRouterDataset(Dataset):
    """Full-sequence residuals ``[T_i, d_model]`` + unified router targets
    for attention-based compressors."""

    def __init__(
        self,
        data_dir: str,
        benchmarks: List[str],
        per_bench_anchors: Dict[str, List[int]],
        gate_positives_only: bool = True,
        noop_boost: float = 0.0,
        catalog_mode: str = "union",
    ):
        self.residuals: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.bench_ids: List[str] = []
        self.bench_names: List[str] = []

        per_bench_records: Dict[str, List[Dict]] = {}
        per_bench_residuals: Dict[str, List[torch.Tensor]] = {}

        for bench in benchmarks:
            if bench not in per_bench_anchors:
                continue
            full_pt_path = os.path.join(data_dir, f"{bench}_full_residuals.pt")
            jsonl_path = os.path.join(data_dir, f"{bench}.jsonl")
            if not os.path.isfile(full_pt_path) or not os.path.isfile(jsonl_path):
                logger.warning("Missing full-seq data for %s, skipping", bench)
                continue

            data = torch.load(full_pt_path, map_location="cpu", weights_only=False)
            full_residuals = data["residuals"]
            with open(jsonl_path) as f:
                records = [json.loads(line) for line in f]
            n = min(len(full_residuals), len(records))
            if n < len(full_residuals) or n < len(records):
                logger.warning(
                    "%s: truncating to %d rows (pt=%d, jsonl=%d)",
                    bench, n, len(full_residuals), len(records),
                )
            per_bench_records[bench] = records[:n]
            per_bench_residuals[bench] = [r.float() for r in full_residuals[:n]]
            self.bench_names.append(bench)

        _check_anchor_compatibility(per_bench_anchors)

        (self.num_classes, self.catalog, self.seq_to_idx) = build_route_catalog(
            per_bench_records, per_bench_anchors, self.bench_names, catalog_mode,
        )

        for bench in self.bench_names:
            records = per_bench_records[bench]
            residuals = per_bench_residuals[bench]
            targets = build_unified_targets(
                records, self.seq_to_idx, self.num_classes,
                per_bench_anchors[bench], noop_boost=noop_boost,
            )
            for i, (rec, tgt) in enumerate(zip(records, targets)):
                if gate_positives_only and rec["gate_label"] == 0:
                    continue
                self.residuals.append(residuals[i])
                self.targets.append(tgt)
                self.bench_ids.append(bench)

        logger.info(
            "JointFullSequenceRouterDataset: %d samples, G=%d, catalog_mode=%s",
            len(self.targets), self.num_classes, catalog_mode,
        )
        self.bench_to_idx: Dict[str, int] = {b: i for i, b in enumerate(self.bench_names)}
        self.bench_idx_per_sample: List[int] = [self.bench_to_idx[b] for b in self.bench_ids]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.residuals[idx], self.targets[idx], self.bench_idx_per_sample[idx]


def _load_dense_question_map(dense_deltas_jsonl: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    dense_map: Dict[str, Dict[int, Dict[str, Any]]] = {}
    with open(dense_deltas_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            b = rec.get("benchmark_id")
            qid = int(rec.get("question_id", -1))
            if b is None or qid < 0:
                continue
            dense_map.setdefault(b, {})[qid] = rec
    return dense_map


class JointRouterDenseDataset(Dataset):
    """Pivot residuals + targets from dense reevaluation table.

    Dense rows are expected to contain ``benchmark_id``, ``question_id``, and
    ``route_deltas`` where route ids are aligned with ``selected_routes`` in
    ``selected_catalog.json``.
    """

    def __init__(
        self,
        data_dir: str,
        benchmarks: List[str],
        catalog_json: str,
        dense_deltas_jsonl: str,
        gate_positives_only: bool = True,
        dense_map: Optional[Dict[str, Dict[int, Dict[str, Any]]]] = None,
        pivot_residuals_map: Optional[Dict[str, torch.Tensor]] = None,
        dense_supervision_topk: Optional[int] = None,
    ):
        self.residuals: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.bench_ids: List[str] = []
        self.bench_names: List[str] = []
        self.question_ids: List[int] = []
        self.dense_supervision_topk = dense_supervision_topk
        self.hard_best_cls: Optional[List[int]] = (
            [] if dense_supervision_topk is not None else None
        )

        self.num_classes, self.catalog, self.seq_to_idx = load_selected_catalog(catalog_json)

        if dense_map is None:
            dense_map = _load_dense_question_map(dense_deltas_jsonl)

        for bench in benchmarks:
            pt_path = os.path.join(data_dir, f"{bench}_pivot_residuals.pt")
            if not os.path.isfile(pt_path):
                logger.warning("Missing pivot residuals for %s (%s), skipping", bench, pt_path)
                continue
            bench_dense = dense_map.get(bench, {})
            if not bench_dense:
                logger.warning("No dense rows for benchmark %s in %s", bench, dense_deltas_jsonl)
                continue

            if pivot_residuals_map is not None and bench in pivot_residuals_map:
                residuals = pivot_residuals_map[bench].float()
            else:
                residuals = torch.load(pt_path, map_location="cpu", weights_only=True).float()
            n_used = 0
            for qid in sorted(bench_dense.keys()):
                if qid < 0 or qid >= residuals.shape[0]:
                    continue
                rec = bench_dense[qid]
                cls = _dense_best_class_idx(rec)
                if gate_positives_only and cls == STAY_INDEX:
                    continue
                if dense_supervision_topk is not None:
                    tgt, hard_cls = _dense_topk_soft_target(
                        rec, self.num_classes, dense_supervision_topk,
                    )
                    if self.hard_best_cls is None:
                        raise RuntimeError("hard_best_cls list not initialized")
                    self.hard_best_cls.append(int(hard_cls))
                else:
                    tgt = torch.zeros(self.num_classes, dtype=torch.float32)
                    tgt[cls] = 1.0
                self.residuals.append(residuals[qid])
                self.targets.append(tgt)
                self.bench_ids.append(bench)
                self.question_ids.append(int(qid))
                n_used += 1
            self.bench_names.append(bench)
            logger.info(
                "%s dense: used=%d rows (residual_rows=%d, dense_rows=%d)",
                bench, n_used, residuals.shape[0], len(bench_dense),
            )

        if dense_supervision_topk is not None:
            assert self.hard_best_cls is not None
            logger.info(
                "JointRouterDenseDataset: %d samples, G=%d, dense_supervision_topk=%d, benchmarks=%s",
                len(self.targets), self.num_classes, dense_supervision_topk, self.bench_names,
            )
        else:
            logger.info(
                "JointRouterDenseDataset: %d samples, G=%d, benchmarks=%s",
                len(self.targets), self.num_classes, self.bench_names,
            )
        self.bench_to_idx: Dict[str, int] = {b: i for i, b in enumerate(self.bench_names)}
        self.bench_idx_per_sample: List[int] = [self.bench_to_idx[b] for b in self.bench_ids]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        if self.dense_supervision_topk is not None:
            assert self.hard_best_cls is not None
            return (
                self.residuals[idx],
                self.targets[idx],
                self.hard_best_cls[idx],
                self.bench_idx_per_sample[idx],
            )
        return self.residuals[idx], self.targets[idx], self.bench_idx_per_sample[idx]


class JointRouterDenseFullSequenceDataset(Dataset):
    """Full-sequence residuals + dense targets (for top-down attention compressors)."""

    def __init__(
        self,
        data_dir: str,
        benchmarks: List[str],
        catalog_json: str,
        dense_deltas_jsonl: str,
        gate_positives_only: bool = True,
        dense_map: Optional[Dict[str, Dict[int, Dict[str, Any]]]] = None,
        full_residuals_map: Optional[Dict[str, List[torch.Tensor]]] = None,
    ):
        self.residuals: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.bench_ids: List[str] = []
        self.bench_names: List[str] = []

        self.num_classes, self.catalog, self.seq_to_idx = load_selected_catalog(catalog_json)
        if dense_map is None:
            dense_map = _load_dense_question_map(dense_deltas_jsonl)

        for bench in benchmarks:
            full_pt_path = os.path.join(data_dir, f"{bench}_full_residuals.pt")
            if not os.path.isfile(full_pt_path):
                logger.warning("Missing full residuals for %s (%s), skipping", bench, full_pt_path)
                continue
            bench_dense = dense_map.get(bench, {})
            if not bench_dense:
                logger.warning("No dense rows for benchmark %s in %s", bench, dense_deltas_jsonl)
                continue

            if full_residuals_map is not None and bench in full_residuals_map:
                full_residuals = full_residuals_map[bench]
            else:
                data = torch.load(full_pt_path, map_location="cpu", weights_only=False)
                full_residuals = data["residuals"]
            n_used = 0
            for qid in sorted(bench_dense.keys()):
                if qid < 0 or qid >= len(full_residuals):
                    continue
                cls = _dense_best_class_idx(bench_dense[qid])
                if gate_positives_only and cls == STAY_INDEX:
                    continue
                tgt = torch.zeros(self.num_classes, dtype=torch.float32)
                tgt[cls] = 1.0
                self.residuals.append(full_residuals[qid].float())
                self.targets.append(tgt)
                self.bench_ids.append(bench)
                n_used += 1
            self.bench_names.append(bench)
            logger.info(
                "%s dense (full-seq): used=%d rows (full_rows=%d, dense_rows=%d)",
                bench, n_used, len(full_residuals), len(bench_dense),
            )

        logger.info(
            "JointRouterDenseFullSequenceDataset: %d samples, G=%d, benchmarks=%s",
            len(self.targets), self.num_classes, self.bench_names,
        )
        self.bench_to_idx: Dict[str, int] = {b: i for i, b in enumerate(self.bench_names)}
        self.bench_idx_per_sample: List[int] = [self.bench_to_idx[b] for b in self.bench_ids]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.residuals[idx], self.targets[idx], self.bench_idx_per_sample[idx]


# ---------------------------------------------------------------------------
# Sampling / collate helpers
# ---------------------------------------------------------------------------


class BalancedBenchmarkSampler(Sampler[int]):
    """Sample an equal number of rows from each benchmark every epoch."""

    def __init__(
        self,
        per_bench_positions: Dict[str, List[int]],
        per_bench_per_epoch: int,
        seed: int = 42,
    ):
        if per_bench_per_epoch < 1:
            raise ValueError(f"per_bench_per_epoch must be >= 1, got {per_bench_per_epoch}")
        self.per_bench_positions = {k: list(v) for k, v in per_bench_positions.items()}
        self.bench_names = sorted(self.per_bench_positions.keys())
        self.per_bench_per_epoch = per_bench_per_epoch
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        out: List[int] = []
        for bench in self.bench_names:
            picks = list(self.per_bench_positions[bench])
            rng.shuffle(picks)
            out.extend(picks[: self.per_bench_per_epoch])
        rng.shuffle(out)
        return iter(out)

    def __len__(self) -> int:
        return self.per_bench_per_epoch * len(self.bench_names)


def _build_balanced_train_sampler(train_subset, bench_ids: List[str], seed: int) -> Optional[Sampler[int]]:
    """Build a balanced sampler over local train-subset positions."""
    subset_indices = getattr(train_subset, "indices", None)
    if subset_indices is None:
        return None

    per_bench_positions: Dict[str, List[int]] = {}
    for local_pos, global_idx in enumerate(subset_indices):
        bench = bench_ids[int(global_idx)]
        per_bench_positions.setdefault(bench, []).append(local_pos)

    if not per_bench_positions:
        return None
    min_count = min(len(v) for v in per_bench_positions.values())
    if min_count < 1:
        return None

    logger.info(
        "Balanced sampling: train counts=%s, per_epoch_per_bench=%d, epoch_samples=%d",
        {b: len(v) for b, v in per_bench_positions.items()},
        min_count,
        min_count * len(per_bench_positions),
    )
    return BalancedBenchmarkSampler(
        per_bench_positions=per_bench_positions,
        per_bench_per_epoch=min_count,
        seed=seed,
    )

def make_joint_collate(with_hard_best: bool = False):
    """Collate for last-token pivot residuals.

    Returns ``(x, y, bench_idx)`` or ``(x, y, hard_cls, bench_idx)``.
    """
    def _collate(batch):
        residuals = torch.stack([b[0] for b in batch])
        targets = torch.stack([b[1] for b in batch])
        bench_idx = torch.tensor([b[-1] for b in batch], dtype=torch.long)
        if with_hard_best:
            hard = torch.tensor([b[2] for b in batch], dtype=torch.long)
            return residuals, targets, hard, bench_idx
        return residuals, targets, bench_idx

    return _collate


def make_joint_full_seq_collate(max_seq_len: Optional[int] = None):
    """Collate for full-sequence data: pad residuals."""
    def _collate(batch):
        residuals = [b[0] for b in batch]
        targets = torch.stack([b[1] for b in batch])
        bench_idx = torch.tensor([b[2] for b in batch], dtype=torch.long)
        padded, attn_mask = pad_sequences(residuals, max_seq_len=max_seq_len)
        return padded, attn_mask, targets, bench_idx

    return _collate


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def global_idx_to_sequence(
    global_idx: int,
    catalog: List[Optional[List[int]]],
    anchor_seq: List[int],
    *,
    per_bench_override: Optional[List[Optional[List[int]]]] = None,
) -> List[int]:
    """Map a global catalog index back to a layer sequence.

    Index 0 (STAY) returns the given *anchor_seq*.
    When *per_bench_override* is set (same length as *catalog*), non-STAY
    classes use that bench-specific materialization — required when a joint
    catalogue shares program indices but anchors differ per benchmark.
    """
    if global_idx == STAY_INDEX:
        return list(anchor_seq)
    if per_bench_override is not None:
        entry = per_bench_override[global_idx]
        if entry is None:
            return list(anchor_seq)
        return [int(x) for x in entry]
    ent = catalog[global_idx]
    if ent is None:
        return list(anchor_seq)
    return [int(x) for x in ent]


def _build_benchmark_valid_masks(
    ds,
    dense_map_cache: Optional[Dict[str, Dict[int, Dict[str, Any]]]],
) -> Dict[str, torch.Tensor]:
    """Build per-benchmark legal-route masks for invalid-mass penalty."""
    num_classes = ds.num_classes
    out: Dict[str, torch.Tensor] = {}
    for bench in ds.bench_names:
        mask = torch.zeros(num_classes, dtype=torch.float32)
        mask[STAY_INDEX] = 1.0
        if dense_map_cache is None:
            mask[:] = 1.0
            out[bench] = mask
            continue
        bench_dense = dense_map_cache.get(bench, {})
        valid_route_ids: Set[int] = set()
        for rec in bench_dense.values():
            for rid_s in rec.get("route_deltas", {}).keys():
                rid = int(rid_s)
                cls = rid + 1
                if 1 <= cls < num_classes:
                    valid_route_ids.add(cls)
        if not valid_route_ids:
            mask[:] = 1.0
        else:
            for cls in valid_route_ids:
                mask[cls] = 1.0
        out[bench] = mask
    return out


def _bal_cond_entropy_loss(
    probs: torch.Tensor,
    bench_idx: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """L_bal = -sum_b H(q_b) over benchmarks in batch.

    q_b is mean softmax; H uses route mass only (classes r >= 1), renormalized
    to a simplex so H matches -sum_{r=1}^{R} q(r) log q(r) on routes.
    """
    eps = 1e-8
    loss_term = probs.new_tensor(0.0)
    metrics: Dict[str, float] = {"bal_L": 0.0}
    if probs.size(-1) < 2:
        return loss_term, metrics
    for b in torch.unique(bench_idx):
        bi = int(b.item())
        m = bench_idx == b
        if not m.any():
            continue
        q_bar = probs[m].mean(dim=0)
        v = q_bar[1:].clamp(min=eps)
        s = v.sum()
        if float(s.item()) < eps:
            H_b = probs.new_tensor(0.0)
        else:
            v = v / s
            H_b = -(v * (v + eps).log()).sum()
        loss_term = loss_term - H_b
        metrics[f"H_bench{bi}"] = float(H_b.detach().item())
    metrics["bal_L"] = float(loss_term.detach().item())
    return loss_term, metrics


def _parse_ce_weights_by_bench(spec: Optional[str]) -> Optional[Dict[str, float]]:
    """Parse ``boolq:1.0,commonsenseqa:1.5`` for weighted CE (training only)."""
    if not spec:
        return None
    out: Dict[str, float] = {}
    for chunk in spec.split(","):
        item = chunk.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Invalid --ce_weights_by_bench entry {item!r}. Expected format 'bench:weight'."
            )
        bench, value_s = item.split(":", 1)
        out[bench.strip()] = float(value_s.strip())
    return out


def _ce_weight_vec_for_dataset(
    ce_weights_by_bench: Optional[Dict[str, float]],
    bench_names: List[str],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not ce_weights_by_bench:
        return None
    merged = {b: 1.0 for b in bench_names}
    for k, v in ce_weights_by_bench.items():
        if k in merged:
            merged[k] = v
        else:
            logger.warning("CE weight for unknown benchmark %s ignored", k)
    vec = torch.tensor(
        [merged[b] for b in bench_names],
        dtype=torch.float32,
        device=device,
    )
    logger.info(
        "Per-benchmark CE weights: %s",
        {b: float(vec[i]) for i, b in enumerate(bench_names)},
    )
    return vec


def _mask_stay_logits(logits: torch.Tensor) -> torch.Tensor:
    """Push STAY logit very negative so softmax / argmax never pick class 0."""
    z = logits.clone()
    z[:, STAY_INDEX] = logits.new_tensor(-1.0e4)
    return z


def _composite_router_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bench_idx: torch.Tensor,
    bench_valid_masks: torch.Tensor,
    robust_temperature: float,
    lambda_rob: float,
    beta_invalid: float,
    lambda_bal_cond_entropy: float = 0.0,
    mask_stay_logits: bool = False,
    ce_sample_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Base CE + robust benchmark term + invalid-route mass penalty."""
    if mask_stay_logits:
        logits = _mask_stay_logits(logits)
    target_idx = targets.argmax(dim=-1)
    per_ex_ce = F.cross_entropy(logits, target_idx, reduction="none")
    if ce_sample_weight is not None:
        w = ce_sample_weight.to(dtype=per_ex_ce.dtype, device=per_ex_ce.device)
        denom = w.sum().clamp(min=1e-8)
        ce_loss = (per_ex_ce * w).sum() / denom

        def _weighted_group_mean(mask: torch.Tensor) -> torch.Tensor:
            wm = w[mask]
            return (per_ex_ce[mask] * wm).sum() / wm.sum().clamp(min=1e-8)
    else:
        ce_loss = per_ex_ce.mean()

        def _weighted_group_mean(mask: torch.Tensor) -> torch.Tensor:
            return per_ex_ce[mask].mean()

    loss = ce_loss

    robust_term = torch.zeros((), device=logits.device, dtype=logits.dtype)
    if lambda_rob > 0:
        per_bench_means: List[torch.Tensor] = []
        unique_b = torch.unique(bench_idx)
        for b in unique_b:
            m = bench_idx == b
            if m.any():
                per_bench_means.append(_weighted_group_mean(m))
        if per_bench_means:
            stack = torch.stack(per_bench_means)
            robust_term = robust_temperature * torch.logsumexp(
                stack / robust_temperature, dim=0,
            )
            loss = loss + lambda_rob * robust_term

    invalid_term = torch.zeros((), device=logits.device, dtype=logits.dtype)
    probs: Optional[torch.Tensor] = None
    if beta_invalid > 0:
        probs = F.softmax(logits, dim=-1)
        valid_mask = bench_valid_masks.index_select(0, bench_idx)
        valid_mass = (probs * valid_mask).sum(dim=-1)
        invalid_term = (1.0 - valid_mass).mean()
        loss = loss + beta_invalid * invalid_term

    bal_term = torch.zeros((), device=logits.device, dtype=logits.dtype)
    if lambda_bal_cond_entropy > 0:
        if probs is None:
            probs = F.softmax(logits, dim=-1)
        bal_term, bal_metrics = _bal_cond_entropy_loss(probs, bench_idx)
        loss = loss + lambda_bal_cond_entropy * bal_term
    else:
        bal_metrics = {"bal_L": 0.0}

    metrics = {
        "ce": float(ce_loss.item()),
        "rob": float(robust_term.item()),
        "invalid": float(invalid_term.item()),
        "bal_L": float(bal_metrics.get("bal_L", 0.0)),
    }
    for k, v in bal_metrics.items():
        if k != "bal_L":
            metrics[k] = v
    return loss, metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_joint_router(
    data_dir: str,
    output_dir: str,
    benchmarks: List[str],
    anchor_seqs: Dict[str, List[int]],
    compressor_cfg: CompressorConfig,
    gate_positives_only: bool = True,
    hidden_dims: List[int] = (1024, 1024, 512),
    dropout: float = 0.1,
    lr: float = 1e-3,
    epochs: int = 80,
    batch_size: int = 64,
    val_fraction: float = 0.15,
    seed: int = 42,
    noop_boost: float = 0.0,
    dense_deltas_jsonl: Optional[str] = None,
    catalog_json: Optional[str] = None,
    hard_ce_supervision: bool = False,
    dense_supervision_topk: Optional[int] = None,
    dense_cache: Optional[Dict[str, Any]] = None,
    use_dual_encoder: bool = False,
    route_dim: int = 64,
    route_enc_layers: int = 2,
    route_enc_heads: int = 4,
    full_seq_max_len: Optional[int] = 256,
    balanced_per_benchmark: bool = False,
    robust_temperature: float = 0.3,
    lambda_rob: float = 0.0,
    beta_invalid: float = 0.0,
    lambda_bal_cond_entropy: float = 0.0,
    mask_stay_logits: bool = False,
    wandb_enabled: bool = True,
    wandb_project: str = "joint-router",
    wandb_run_name: Optional[str] = None,
    wandb_group: Optional[str] = None,
    catalog_mode: str = "union",
    ce_weights_by_bench: Optional[Dict[str, float]] = None,
    split_json_path: Optional[str] = None,
    per_bench_catalog: Optional[Dict[str, List[Optional[List[int]]]]] = None,
) -> Optional[str]:
    """Train a joint router over multiple benchmarks.  Returns checkpoint path."""
    if catalog_mode not in ("union", "intersection"):
        raise ValueError(f"catalog_mode must be 'union' or 'intersection', got {catalog_mode!r}")
    if catalog_mode == "intersection" and (dense_deltas_jsonl or catalog_json):
        raise ValueError(
            "--catalog_mode intersection applies only to MCTS joint training; "
            "do not pass --dense_deltas_jsonl / --catalog_json.",
        )
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_full_seq = compressor_cfg.compressor_type != "last_token"

    if dense_deltas_jsonl and catalog_json:
        if dense_supervision_topk is not None and use_full_seq:
            raise ValueError("dense_supervision_topk is only supported for pivot (last_token) dense data.")
        dense_cache = dense_cache or {}
        dense_map_cache = dense_cache.get("dense_map")
        pivot_residuals_cache = dense_cache.get("pivot_residuals_map")
        full_residuals_cache = dense_cache.get("full_residuals_map")
        if use_full_seq:
            ds = JointRouterDenseFullSequenceDataset(
                data_dir=data_dir,
                benchmarks=benchmarks,
                catalog_json=catalog_json,
                dense_deltas_jsonl=dense_deltas_jsonl,
                gate_positives_only=gate_positives_only,
                dense_map=dense_map_cache,
                full_residuals_map=full_residuals_cache,
            )
        else:
            ds = JointRouterDenseDataset(
                data_dir=data_dir,
                benchmarks=benchmarks,
                catalog_json=catalog_json,
                dense_deltas_jsonl=dense_deltas_jsonl,
                gate_positives_only=gate_positives_only,
                dense_map=dense_map_cache,
                pivot_residuals_map=pivot_residuals_cache,
                dense_supervision_topk=dense_supervision_topk,
            )
    elif use_full_seq:
        ds = JointFullSequenceRouterDataset(
            data_dir, benchmarks, anchor_seqs,
            gate_positives_only=gate_positives_only,
            noop_boost=noop_boost,
            catalog_mode=catalog_mode,
        )
    else:
        ds = JointRouterDataset(
            data_dir, benchmarks, anchor_seqs,
            gate_positives_only=gate_positives_only,
            noop_boost=noop_boost,
            catalog_mode=catalog_mode,
        )

    if len(ds) == 0:
        logger.error("No data loaded. Aborting.")
        return None

    num_classes = ds.num_classes
    d_model = ds.residuals[0].shape[-1]
    compressor_cfg.d_model = d_model

    use_dense_topk = bool(
        dense_supervision_topk is not None
        and getattr(ds, "dense_supervision_topk", None) is not None
    )
    logger.info(
        "Joint router: compressor=%s  d_model=%d  G=%d  samples=%d  catalog_mode=%s  "
        "hard_ce=%s  dense_supervision_topk=%s  dual_encoder=%s  full_seq=%s  tau=%.3f  "
        "lambda_rob=%.3f  beta_invalid=%.3f  lambda_bal_cond_entropy=%.4f  mask_stay=%s",
        compressor_cfg.compressor_type, d_model, num_classes, len(ds), catalog_mode,
        hard_ce_supervision,
        dense_supervision_topk, use_dual_encoder, use_full_seq,
        robust_temperature, lambda_rob, beta_invalid, lambda_bal_cond_entropy,
        mask_stay_logits,
    )
    if robust_temperature <= 0:
        raise ValueError(f"robust_temperature must be > 0, got {robust_temperature}")

    if split_json_path is not None:
        # Deterministic train/val split from a canonical split JSON; the
        # catalogue mass-coverage reduction was run with the SAME JSON, so
        # no validation-question Δ mass has leaked into the retained
        # catalog. Any (bench, qid) not present in either the train or val
        # set (e.g. the optional test hold-out) is excluded from training.
        with open(split_json_path) as _f:
            _split_doc = json.load(_f)
        _train_keys: set = set()
        _val_keys: set = set()
        for _b, _info in _split_doc.get("benchmarks", {}).items():
            for _q in _info.get("train_question_ids", []):
                _train_keys.add((_b, int(_q)))
            for _q in _info.get("val_question_ids", []):
                _val_keys.add((_b, int(_q)))
        _ds_qids = getattr(ds, "question_ids", None)
        _ds_bench = getattr(ds, "bench_ids", None)
        if _ds_qids is None or _ds_bench is None:
            raise RuntimeError(
                "split_json_path provided but dataset lacks question_ids / bench_ids "
                "attributes; only JointRouterDenseDataset currently supports this.",
            )
        _train_idx: List[int] = []
        _val_idx: List[int] = []
        _skipped = 0
        for _i, (_b, _q) in enumerate(zip(_ds_bench, _ds_qids)):
            _key = (_b, int(_q))
            if _key in _train_keys:
                _train_idx.append(_i)
            elif _key in _val_keys:
                _val_idx.append(_i)
            else:
                _skipped += 1
        if not _val_idx:
            raise RuntimeError(
                f"split_json={split_json_path}: produced empty val set "
                f"(train={len(_train_idx)}, skipped_test={_skipped}).",
            )
        from torch.utils.data import Subset as _Subset

        train_ds = _Subset(ds, _train_idx)
        val_ds = _Subset(ds, _val_idx)
        logger.info(
            "[split_json] %s -> train=%d val=%d skipped_test=%d",
            split_json_path, len(_train_idx), len(_val_idx), _skipped,
        )
    else:
        val_size = max(1, int(len(ds) * val_fraction))
        train_size = len(ds) - val_size
        train_ds, val_ds = random_split(ds, [train_size, val_size])

    if use_full_seq:
        collate_fn = make_joint_full_seq_collate(max_seq_len=full_seq_max_len)
    else:
        collate_fn = make_joint_collate(with_hard_best=use_dense_topk)

    train_sampler = None
    if balanced_per_benchmark:
        train_sampler = _build_balanced_train_sampler(train_ds, ds.bench_ids, seed=seed)
        if train_sampler is None:
            logger.warning(
                "Balanced sampling requested but unavailable; falling back to shuffle=True.",
            )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_fn)
    bench_valid_masks_map = _build_benchmark_valid_masks(
        ds=ds,
        dense_map_cache=dense_map_cache if (dense_deltas_jsonl and catalog_json) else None,
    )
    bench_valid_masks = torch.stack(
        [bench_valid_masks_map[b] for b in ds.bench_names], dim=0,
    ).to(device)

    ce_weight_vec = _ce_weight_vec_for_dataset(ce_weights_by_bench, ds.bench_names, device)

    compressor = build_compressor(compressor_cfg)
    if use_dual_encoder:
        route_ids, route_lengths, num_modules = prepare_catalog_tensors(ds.catalog)
        model = DualEncoderRouter(
            compressor,
            num_classes,
            list(hidden_dims),
            dropout,
            route_dim=route_dim,
            route_ids=route_ids,
            route_lengths=route_lengths,
            num_modules=num_modules,
            route_enc_layers=route_enc_layers,
            route_enc_heads=route_enc_heads,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info("DualEncoderRouter: %d params", n_params)
    else:
        model = CompressedRouter(
            compressor, num_classes, list(hidden_dims), dropout,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info("CompressedRouter: %d params", n_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    use_wandb = bool(wandb_enabled and HAS_WANDB)
    run_wandb = False
    if use_wandb:
        wb_cfg: Dict[str, Any] = {
            "benchmarks": list(benchmarks),
            "hidden_dims": list(hidden_dims),
            "dropout": dropout,
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "val_fraction": val_fraction,
            "seed": seed,
            "gate_positives_only": gate_positives_only,
            "noop_boost": noop_boost,
            "hard_ce_supervision": hard_ce_supervision,
            "dense_supervision_topk": dense_supervision_topk,
            "use_dual_encoder": use_dual_encoder,
            "balanced_per_benchmark": balanced_per_benchmark,
            "robust_temperature": robust_temperature,
            "lambda_rob": lambda_rob,
            "beta_invalid": beta_invalid,
            "lambda_bal_cond_entropy": lambda_bal_cond_entropy,
            "mask_stay_logits": mask_stay_logits,
            "ce_weights_by_bench": ce_weights_by_bench,
            "compressor_type": compressor_cfg.compressor_type,
            "d_compress": compressor_cfg.d_compress,
            "n_heads": compressor_cfg.n_heads,
            "n_latent_tokens": compressor_cfg.n_latent_tokens,
            "num_classes": num_classes,
            "num_samples": len(ds),
            "d_model": d_model,
            "n_params": n_params,
            "catalog_mode": catalog_mode,
        }
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            group=wandb_group,
            config=wb_cfg,
        )
        run_wandb = True

    ckpt_path = os.path.join(output_dir, "joint_router_best.pt")
    best_val_loss = float("inf")
    best_val_top1 = 0.0
    best_epoch = -1

    try:
        for epoch in range(1, epochs + 1):
            # ---- train ----
            model.train()
            train_loss_sum = 0.0
            train_ce_sum = 0.0
            train_rob_sum = 0.0
            train_invalid_sum = 0.0
            train_bal_sum = 0.0
            train_n = 0
            for batch in train_loader:
                if use_full_seq:
                    x_pad, attn_mask, y, bench_idx = batch
                    x_pad = x_pad.to(device)
                    attn_mask = attn_mask.to(device)
                    logits = model(x_pad, attention_mask=attn_mask)
                    y = y.to(device)
                    bench_idx = bench_idx.to(device)
                    ce_sw = ce_weight_vec[bench_idx] if ce_weight_vec is not None else None
                    if hard_ce_supervision:
                        eff = _mask_stay_logits(logits) if mask_stay_logits else logits
                        target_idx = y.argmax(dim=-1)
                        per_ce = F.cross_entropy(eff, target_idx, reduction="none")
                        if ce_sw is not None:
                            loss = (per_ce * ce_sw).sum() / ce_sw.sum().clamp(min=1e-8)
                        else:
                            loss = per_ce.mean()
                        train_ce_sum += loss.item() * logits.size(0)
                        if lambda_bal_cond_entropy > 0:
                            probs = F.softmax(eff, dim=-1)
                            bal_t, _ = _bal_cond_entropy_loss(probs, bench_idx)
                            loss = loss + lambda_bal_cond_entropy * bal_t
                            train_bal_sum += float(bal_t.item()) * logits.size(0)
                    else:
                        loss, parts = _composite_router_loss(
                            logits=logits,
                            targets=y,
                            bench_idx=bench_idx,
                            bench_valid_masks=bench_valid_masks,
                            robust_temperature=robust_temperature,
                            lambda_rob=lambda_rob,
                            beta_invalid=beta_invalid,
                            lambda_bal_cond_entropy=lambda_bal_cond_entropy,
                            mask_stay_logits=mask_stay_logits,
                            ce_sample_weight=ce_sw,
                        )
                        train_ce_sum += parts["ce"] * logits.size(0)
                        train_rob_sum += parts["rob"] * logits.size(0)
                        train_invalid_sum += parts["invalid"] * logits.size(0)
                        train_bal_sum += parts["bal_L"] * logits.size(0)
                elif use_dense_topk:
                    x, y, _hard, bench_idx = batch
                    x = x.to(device)
                    y = y.to(device)
                    bench_idx = bench_idx.to(device)
                    logits = model(x)
                    ce_sw = ce_weight_vec[bench_idx] if ce_weight_vec is not None else None
                    loss, parts = _composite_router_loss(
                        logits=logits,
                        targets=y,
                        bench_idx=bench_idx,
                        bench_valid_masks=bench_valid_masks,
                        robust_temperature=robust_temperature,
                        lambda_rob=lambda_rob,
                        beta_invalid=beta_invalid,
                        lambda_bal_cond_entropy=lambda_bal_cond_entropy,
                        mask_stay_logits=mask_stay_logits,
                        ce_sample_weight=ce_sw,
                    )
                    train_ce_sum += parts["ce"] * logits.size(0)
                    train_rob_sum += parts["rob"] * logits.size(0)
                    train_invalid_sum += parts["invalid"] * logits.size(0)
                    train_bal_sum += parts["bal_L"] * logits.size(0)
                else:
                    x, y, bench_idx = batch
                    x = x.to(device)
                    logits = model(x)
                    y = y.to(device)
                    bench_idx = bench_idx.to(device)
                    ce_sw = ce_weight_vec[bench_idx] if ce_weight_vec is not None else None
                    if hard_ce_supervision:
                        eff = _mask_stay_logits(logits) if mask_stay_logits else logits
                        target_idx = y.argmax(dim=-1)
                        per_ce = F.cross_entropy(eff, target_idx, reduction="none")
                        if ce_sw is not None:
                            loss = (per_ce * ce_sw).sum() / ce_sw.sum().clamp(min=1e-8)
                        else:
                            loss = per_ce.mean()
                        train_ce_sum += loss.item() * logits.size(0)
                        if lambda_bal_cond_entropy > 0:
                            probs = F.softmax(eff, dim=-1)
                            bal_t, _ = _bal_cond_entropy_loss(probs, bench_idx)
                            loss = loss + lambda_bal_cond_entropy * bal_t
                            train_bal_sum += float(bal_t.item()) * logits.size(0)
                    else:
                        loss, parts = _composite_router_loss(
                            logits=logits,
                            targets=y,
                            bench_idx=bench_idx,
                            bench_valid_masks=bench_valid_masks,
                            robust_temperature=robust_temperature,
                            lambda_rob=lambda_rob,
                            beta_invalid=beta_invalid,
                            lambda_bal_cond_entropy=lambda_bal_cond_entropy,
                            mask_stay_logits=mask_stay_logits,
                            ce_sample_weight=ce_sw,
                        )
                        train_ce_sum += parts["ce"] * logits.size(0)
                        train_rob_sum += parts["rob"] * logits.size(0)
                        train_invalid_sum += parts["invalid"] * logits.size(0)
                        train_bal_sum += parts["bal_L"] * logits.size(0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item() * logits.size(0)
                train_n += logits.size(0)
    
            train_loss = train_loss_sum / max(train_n, 1)
            train_ce = train_ce_sum / max(train_n, 1)
            train_rob = train_rob_sum / max(train_n, 1)
            train_invalid = train_invalid_sum / max(train_n, 1)
            train_bal = train_bal_sum / max(train_n, 1)
    
            # ---- val ----
            model.eval()
            val_loss_sum = 0.0
            val_ce_sum = 0.0
            val_rob_sum = 0.0
            val_invalid_sum = 0.0
            val_bal_sum = 0.0
            val_n = 0
            top1_correct = 0
    
            with torch.no_grad():
                for batch in val_loader:
                    if use_full_seq:
                        x_pad, attn_mask, y, bench_idx = batch
                        x_pad = x_pad.to(device)
                        attn_mask = attn_mask.to(device)
                        logits = model(x_pad, attention_mask=attn_mask)
                        y = y.to(device)
                        bench_idx = bench_idx.to(device)
                        ce_sw = ce_weight_vec[bench_idx] if ce_weight_vec is not None else None
                        if hard_ce_supervision:
                            eff = _mask_stay_logits(logits) if mask_stay_logits else logits
                            target_idx = y.argmax(dim=-1)
                            per_ce = F.cross_entropy(eff, target_idx, reduction="none")
                            if ce_sw is not None:
                                loss = (per_ce * ce_sw).sum() / ce_sw.sum().clamp(min=1e-8)
                            else:
                                loss = per_ce.mean()
                            val_ce_sum += loss.item() * logits.size(0)
                            if lambda_bal_cond_entropy > 0:
                                probs = F.softmax(eff, dim=-1)
                                bal_t, _ = _bal_cond_entropy_loss(probs, bench_idx)
                                loss = loss + lambda_bal_cond_entropy * bal_t
                                val_bal_sum += float(bal_t.item()) * logits.size(0)
                        else:
                            loss, parts = _composite_router_loss(
                                logits=logits,
                                targets=y,
                                bench_idx=bench_idx,
                                bench_valid_masks=bench_valid_masks,
                                robust_temperature=robust_temperature,
                                lambda_rob=lambda_rob,
                                beta_invalid=beta_invalid,
                                lambda_bal_cond_entropy=lambda_bal_cond_entropy,
                                mask_stay_logits=mask_stay_logits,
                                ce_sample_weight=ce_sw,
                            )
                            val_ce_sum += parts["ce"] * logits.size(0)
                            val_rob_sum += parts["rob"] * logits.size(0)
                            val_invalid_sum += parts["invalid"] * logits.size(0)
                            val_bal_sum += parts["bal_L"] * logits.size(0)
                        val_loss_sum += loss.item() * logits.size(0)
                        val_n += logits.size(0)
                        pred_logits = _mask_stay_logits(logits) if mask_stay_logits else logits
                        pred_cls = pred_logits.argmax(dim=-1)
                        target_cls = y.argmax(dim=-1)
                        top1_correct += (pred_cls == target_cls).sum().item()
                    elif use_dense_topk:
                        x, y, hard, bench_idx = batch
                        x = x.to(device)
                        y = y.to(device)
                        hard = hard.to(device)
                        bench_idx = bench_idx.to(device)
                        logits = model(x)
                        ce_sw = ce_weight_vec[bench_idx] if ce_weight_vec is not None else None
                        loss, parts = _composite_router_loss(
                            logits=logits,
                            targets=y,
                            bench_idx=bench_idx,
                            bench_valid_masks=bench_valid_masks,
                            robust_temperature=robust_temperature,
                            lambda_rob=lambda_rob,
                            beta_invalid=beta_invalid,
                            lambda_bal_cond_entropy=lambda_bal_cond_entropy,
                            mask_stay_logits=mask_stay_logits,
                            ce_sample_weight=ce_sw,
                        )
                        val_ce_sum += parts["ce"] * logits.size(0)
                        val_rob_sum += parts["rob"] * logits.size(0)
                        val_invalid_sum += parts["invalid"] * logits.size(0)
                        val_bal_sum += parts["bal_L"] * logits.size(0)
                        val_loss_sum += loss.item() * logits.size(0)
                        val_n += logits.size(0)
                        pred_logits = _mask_stay_logits(logits) if mask_stay_logits else logits
                        pred_cls = pred_logits.argmax(dim=-1)
                        top1_correct += (pred_cls == hard).sum().item()
                    else:
                        x, y, bench_idx = batch
                        x = x.to(device)
                        logits = model(x)
                        y = y.to(device)
                        bench_idx = bench_idx.to(device)
                        ce_sw = ce_weight_vec[bench_idx] if ce_weight_vec is not None else None
                        if hard_ce_supervision:
                            eff = _mask_stay_logits(logits) if mask_stay_logits else logits
                            target_idx = y.argmax(dim=-1)
                            per_ce = F.cross_entropy(eff, target_idx, reduction="none")
                            if ce_sw is not None:
                                loss = (per_ce * ce_sw).sum() / ce_sw.sum().clamp(min=1e-8)
                            else:
                                loss = per_ce.mean()
                            val_ce_sum += loss.item() * logits.size(0)
                            if lambda_bal_cond_entropy > 0:
                                probs = F.softmax(eff, dim=-1)
                                bal_t, _ = _bal_cond_entropy_loss(probs, bench_idx)
                                loss = loss + lambda_bal_cond_entropy * bal_t
                                val_bal_sum += float(bal_t.item()) * logits.size(0)
                        else:
                            loss, parts = _composite_router_loss(
                                logits=logits,
                                targets=y,
                                bench_idx=bench_idx,
                                bench_valid_masks=bench_valid_masks,
                                robust_temperature=robust_temperature,
                                lambda_rob=lambda_rob,
                                beta_invalid=beta_invalid,
                                lambda_bal_cond_entropy=lambda_bal_cond_entropy,
                                mask_stay_logits=mask_stay_logits,
                                ce_sample_weight=ce_sw,
                            )
                            val_ce_sum += parts["ce"] * logits.size(0)
                            val_rob_sum += parts["rob"] * logits.size(0)
                            val_invalid_sum += parts["invalid"] * logits.size(0)
                            val_bal_sum += parts["bal_L"] * logits.size(0)
                        val_loss_sum += loss.item() * logits.size(0)
                        val_n += logits.size(0)
                        pred_logits = _mask_stay_logits(logits) if mask_stay_logits else logits
                        pred_cls = pred_logits.argmax(dim=-1)
                        target_cls = y.argmax(dim=-1)
                        top1_correct += (pred_cls == target_cls).sum().item()
    
            val_loss = val_loss_sum / max(val_n, 1)
            val_ce = val_ce_sum / max(val_n, 1)
            val_rob = val_rob_sum / max(val_n, 1)
            val_invalid = val_invalid_sum / max(val_n, 1)
            val_bal = val_bal_sum / max(val_n, 1)
            top1_acc = top1_correct / max(val_n, 1)
            scheduler.step()
    
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_top1 = top1_acc
                best_epoch = epoch
                _save_checkpoint(
                    ckpt_path, epoch, model, compressor_cfg, num_classes,
                    hidden_dims, dropout, ds, anchor_seqs,
                    gate_positives_only, noop_boost, hard_ce_supervision,
                    robust_temperature=robust_temperature,
                    lambda_rob=lambda_rob,
                    beta_invalid=beta_invalid,
                    lambda_bal_cond_entropy=lambda_bal_cond_entropy,
                    mask_stay_logits=mask_stay_logits,
                    use_dual_encoder=use_dual_encoder,
                    route_dim=route_dim,
                    route_enc_layers=route_enc_layers,
                    route_enc_heads=route_enc_heads,
                    dense_supervision_topk=dense_supervision_topk,
                    catalog_mode=catalog_mode,
                    ce_weights_by_bench=ce_weights_by_bench,
                    per_bench_catalog=per_bench_catalog,
                )
    
            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d  train=%.4f  val=%.4f  top1=%.3f  (best=%d)",
                    epoch, train_loss, val_loss, top1_acc, best_epoch,
                )
    
            if run_wandb:
                lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"]
                wandb.log(
                    {
                        "train/loss": train_loss,
                        "train/ce": train_ce,
                        "train/rob": train_rob,
                        "train/invalid": train_invalid,
                        "train/bal_L": train_bal,
                        "val/loss": val_loss,
                        "val/ce": val_ce,
                        "val/rob": val_rob,
                        "val/invalid": val_invalid,
                        "val/bal_L": val_bal,
                        "val/top1": top1_acc,
                        "lr": lr_now,
                    },
                    step=epoch,
                )

    finally:
        if run_wandb:
            wandb.finish()

    logger.info(
        "Done. Best epoch=%d  val_loss=%.4f  -> %s",
        best_epoch, best_val_loss, ckpt_path,
    )
    metrics_path = os.path.join(output_dir, "train_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_top1": best_val_top1,
                "num_classes": num_classes,
                "num_samples": len(ds),
                "hard_ce_supervision": hard_ce_supervision,
                "dense_supervision_topk": dense_supervision_topk,
                "gate_positives_only": gate_positives_only,
                "use_dual_encoder": use_dual_encoder,
                "compressor_type": compressor_cfg.compressor_type,
                "use_full_seq": use_full_seq,
                "balanced_per_benchmark": balanced_per_benchmark,
                "robust_temperature": robust_temperature,
                "lambda_rob": lambda_rob,
                "beta_invalid": beta_invalid,
                "lambda_bal_cond_entropy": lambda_bal_cond_entropy,
                "mask_stay_logits": mask_stay_logits,
                "catalog_mode": catalog_mode,
                "ce_weights_by_bench": ce_weights_by_bench,
                "wandb_project": wandb_project if use_wandb else None,
                "wandb_run_name": wandb_run_name if use_wandb else None,
                "wandb_group": wandb_group if use_wandb else None,
                "wandb_logged": use_wandb,
            },
            f,
            indent=2,
        )
    logger.info("Saved train metrics -> %s", metrics_path)
    return ckpt_path


def _save_checkpoint(
    path: str,
    epoch: int,
    model: Union[CompressedRouter, DualEncoderRouter],
    compressor_cfg: CompressorConfig,
    num_classes: int,
    hidden_dims: List[int],
    dropout: float,
    ds,
    anchor_seqs: Dict[str, List[int]],
    gate_positives_only: bool,
    noop_boost: float,
    hard_ce_supervision: bool,
    robust_temperature: float,
    lambda_rob: float,
    beta_invalid: float,
    lambda_bal_cond_entropy: float,
    mask_stay_logits: bool = False,
    use_dual_encoder: bool = False,
    route_dim: int = 64,
    route_enc_layers: int = 2,
    route_enc_heads: int = 4,
    dense_supervision_topk: Optional[int] = None,
    catalog_mode: str = "union",
    ce_weights_by_bench: Optional[Dict[str, float]] = None,
    per_bench_catalog: Optional[Dict[str, List[Optional[List[int]]]]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "compressor_config": {
            "compressor_type": compressor_cfg.compressor_type,
            "d_model": compressor_cfg.d_model,
            "d_compress": compressor_cfg.d_compress,
            "n_heads": compressor_cfg.n_heads,
            "n_latent_tokens": compressor_cfg.n_latent_tokens,
        },
        "num_classes": num_classes,
        "hidden_dims": list(hidden_dims),
        "dropout": dropout,
        "benchmarks": list(ds.bench_names),
        "catalog": [list(s) if s is not None else None for s in ds.catalog],
        "seq_to_idx": {str(k): v for k, v in ds.seq_to_idx.items()},
        "anchor_seqs": {b: list(anchor_seqs[b]) for b in ds.bench_names if b in anchor_seqs},
        "gate_positives_only": gate_positives_only,
        "noop_boost": noop_boost,
        "hard_ce_supervision": hard_ce_supervision,
        "robust_temperature": robust_temperature,
        "lambda_rob": lambda_rob,
        "beta_invalid": beta_invalid,
        "lambda_bal_cond_entropy": lambda_bal_cond_entropy,
        "mask_stay_logits": mask_stay_logits,
        "dense_supervision_topk": dense_supervision_topk,
        "use_dual_encoder": use_dual_encoder,
        "route_dim": route_dim,
        "route_enc_layers": route_enc_layers,
        "route_enc_heads": route_enc_heads,
        "catalog_mode": catalog_mode,
        "ce_weights_by_bench": ce_weights_by_bench,
    }
    if per_bench_catalog:
        payload["per_bench_catalog"] = {
            b: [None if x is None else [int(t) for t in x] for x in rows]
            for b, rows in per_bench_catalog.items()
        }
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# Checkpoint loading (for inference / eval)
# ---------------------------------------------------------------------------

def load_joint_router(
    ckpt_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[Union[CompressedRouter, DualEncoderRouter], Dict[str, Any]]:
    """Load a trained joint router from checkpoint.

    Returns ``(model, meta)`` where *meta* contains the catalog, anchor seqs,
    and benchmark names needed for inference.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cc = ckpt["compressor_config"]
    comp_cfg = CompressorConfig(
        compressor_type=cc["compressor_type"],
        d_model=cc["d_model"],
        d_compress=cc.get("d_compress", 256),
        n_heads=cc.get("n_heads", 4),
        n_latent_tokens=cc.get("n_latent_tokens", 1),
    )
    compressor = build_compressor(comp_cfg)
    if ckpt.get("use_dual_encoder"):
        catalog_list = ckpt["catalog"]
        catalog_typed: List[Optional[List[int]]] = []
        for entry in catalog_list:
            if entry is None:
                catalog_typed.append(None)
            else:
                catalog_typed.append([int(x) for x in entry])
        route_ids, route_lengths, num_modules = prepare_catalog_tensors(catalog_typed)
        model = DualEncoderRouter(
            compressor,
            ckpt["num_classes"],
            ckpt["hidden_dims"],
            ckpt["dropout"],
            route_dim=int(ckpt.get("route_dim", 64)),
            route_ids=route_ids,
            route_lengths=route_lengths,
            num_modules=num_modules,
            route_enc_layers=int(ckpt.get("route_enc_layers", 2)),
            route_enc_heads=int(ckpt.get("route_enc_heads", 4)),
        ).to(device)
    else:
        model = CompressedRouter(
            compressor,
            ckpt["num_classes"],
            ckpt["hidden_dims"],
            ckpt["dropout"],
        ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    raw_s2i = ckpt["seq_to_idx"]
    seq_to_idx: Dict[tuple, int] = {
        tuple(int(x) for x in k.strip("()").split(", ")): v
        for k, v in raw_s2i.items()
    }

    meta = {
        "bench_names": ckpt["benchmarks"],
        "num_classes": ckpt["num_classes"],
        "catalog": ckpt["catalog"],
        "seq_to_idx": seq_to_idx,
        "anchor_seqs": ckpt.get("anchor_seqs", {}),
        "use_dual_encoder": bool(ckpt.get("use_dual_encoder", False)),
        "mask_stay_logits": bool(ckpt.get("mask_stay_logits", False)),
        "catalog_mode": ckpt.get("catalog_mode", "union"),
    }
    return model, meta


def predict_route(
    model: Union[CompressedRouter, DualEncoderRouter],
    pivot_residual: torch.Tensor,
    benchmark: str,
    meta: Dict[str, Any],
    device: Optional[torch.device] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[List[int], int]:
    """Run inference for a single question: pivot residual -> best route.

    No masking — the model predicts over the full unified catalog.
    Returns ``(layer_sequence, global_idx)``.
    """
    if device is None:
        device = next(model.parameters()).device

    anchor_seq = meta["anchor_seqs"][benchmark]

    x = pivot_residual.unsqueeze(0).to(device) if pivot_residual.dim() == 1 else pivot_residual.to(device)
    if x.dim() == 2 and x.size(0) != 1:
        x = x.unsqueeze(0)

    with torch.no_grad():
        logits = model(x, attention_mask=attention_mask)
        if meta.get("mask_stay_logits"):
            logits = _mask_stay_logits(logits)
        global_idx = logits.argmax(dim=-1).item()

    seq = global_idx_to_sequence(global_idx, meta["catalog"], anchor_seq)
    return seq, global_idx


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_joint_router(
    ckpt_path: str,
    data_dir: str,
    benchmarks: Optional[List[str]] = None,
    gate_positives_only: bool = True,
    save_results: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a joint router checkpoint and evaluate per-question routing accuracy.

    No masking — the model predicts over the full unified catalog.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_joint_router(ckpt_path, device=device)

    bench_names = meta["bench_names"]
    if benchmarks is not None:
        bench_names = [b for b in benchmarks if b in meta["anchor_seqs"]]
    num_classes = meta["num_classes"]
    catalog = meta["catalog"]
    seq_to_idx = meta["seq_to_idx"]
    anchor_seqs = meta["anchor_seqs"]

    per_bench_correct: Dict[str, int] = {b: 0 for b in bench_names}
    per_bench_total: Dict[str, int] = {b: 0 for b in bench_names}
    per_bench_loss: Dict[str, float] = {b: 0.0 for b in bench_names}
    per_bench_stay: Dict[str, int] = {b: 0 for b in bench_names}

    for bench in bench_names:
        pt_path = os.path.join(data_dir, f"{bench}_pivot_residuals.pt")
        jsonl_path = os.path.join(data_dir, f"{bench}.jsonl")
        if not os.path.isfile(pt_path) or not os.path.isfile(jsonl_path):
            logger.warning("Missing eval data for %s, skipping", bench)
            continue

        residuals = torch.load(pt_path, map_location="cpu", weights_only=True).float()
        with open(jsonl_path) as f:
            records = [json.loads(line) for line in f]
        n = min(residuals.shape[0], len(records))
        anchor = anchor_seqs[bench]

        

        for i in range(n):
            rec = records[i]
            if gate_positives_only and rec["gate_label"] == 0: #TODO though only evaluating on positives is a big bias 
                continue

            x = residuals[i].unsqueeze(0).to(device)
            logits = model(x)
            if meta.get("mask_stay_logits"):
                logits = _mask_stay_logits(logits)

            pred_idx = logits.argmax(dim=-1).item()

            gt = build_unified_targets(
                [rec], seq_to_idx, num_classes, anchor,
            )[0].to(device)
            target_idx = gt.argmax().item()

            per_bench_total[bench] += 1
            if pred_idx == target_idx:
                per_bench_correct[bench] += 1
            if pred_idx == STAY_INDEX:
                per_bench_stay[bench] += 1

            loss = masked_soft_cross_entropy(
                logits, gt.unsqueeze(0), legal_mask=None,
            )
            per_bench_loss[bench] += loss.item()

    total_correct = sum(per_bench_correct.values())
    total_n = sum(per_bench_total.values())

    metrics: Dict[str, Any] = {
        "overall_top1": total_correct / max(total_n, 1),
        "total_samples": total_n,
    }

    logger.info("=== Joint Router Evaluation (unified catalog, no masks) ===")
    logger.info(
        "Overall: top1=%.4f  (%d samples)",
        metrics["overall_top1"], total_n,
    )
    for bench in bench_names:
        bt = per_bench_total[bench]
        if bt == 0:
            continue
        acc = per_bench_correct[bench] / bt
        avg_loss = per_bench_loss[bench] / bt
        stay_rate = per_bench_stay[bench] / bt
        metrics[f"{bench}/top1"] = acc
        metrics[f"{bench}/avg_loss"] = avg_loss
        metrics[f"{bench}/stay_rate"] = stay_rate
        metrics[f"{bench}/n"] = bt
        logger.info(
            "  %s: top1=%.4f  loss=%.4f  stay_rate=%.3f  (n=%d)",
            bench, acc, avg_loss, stay_rate, bt,
        )

    if save_results:
        os.makedirs(os.path.dirname(save_results) or ".", exist_ok=True)
        with open(save_results, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Results saved to %s", save_results)

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Train a joint pivot-residual router across benchmarks",
    )
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory with {bench}_pivot_residuals.pt and {bench}.jsonl")
    p.add_argument("--output_dir", type=str, default="checkpoints/joint_router")
    p.add_argument("--benchmarks", nargs="+", default=["boolq", "commonsenseqa"],
                   help="Benchmarks to train jointly on")
    p.add_argument("--anchor_seqs_json", type=str, default=None,
                   help="JSON mapping benchmark -> anchor sequence (optional)")
    p.add_argument("--results_dir", type=str, default=None,
                   help="Directory with MCTS snapshots for anchor fallback")
    p.add_argument("--gate_positives_only", action="store_true", default=True)
    p.add_argument("--all_questions", dest="gate_positives_only", action="store_false",
                   help="Train on all questions, not just gate-positive ones")
    p.add_argument("--hidden_dims", nargs="+", type=int, default=[1024, 1024, 512],
                   help="MLP hidden layer dimensions")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noop_boost", type=float, default=0.0,
                   help="Boost STAY probability for gate-negative samples")
    p.add_argument(
        "--catalog_mode",
        type=str,
        default="union",
        choices=["union", "intersection"],
        help=(
            "MCTS-only: union = all explored sequences across benchmarks (default); "
            "intersection = only sequences that appear in explored for every benchmark."
        ),
    )
    p.add_argument("--compressor_type", type=str, default="last_token",
                   choices=["last_token", "top_down_attention"])
    p.add_argument("--compressor_d_compress", type=int, default=256)
    p.add_argument("--compressor_n_heads", type=int, default=4)
    p.add_argument("--compressor_n_latent", type=int, default=1)
    p.add_argument("--eval_only", action="store_true",
                   help="Load checkpoint and run evaluation only (no training)")
    p.add_argument("--checkpoint_path", type=str, default=None,
                   help="Checkpoint to load for --eval_only (default: output_dir/joint_router_best.pt)")
    p.add_argument("--save_results", type=str, default=None,
                   help="Save evaluation results to JSON")
    p.add_argument("--catalog_json", type=str, default=None,
                   help="Selected catalog JSON (from select_route_catalog.py)")
    p.add_argument("--dense_deltas_jsonl", type=str, default=None,
                   help="Dense deltas JSONL (from dense_reevaluation.py)")
    p.add_argument("--hard_ce_supervision", action="store_true",
                   help="Train router with hard cross-entropy on argmax target class")
    p.add_argument(
        "--dense_supervision_topk",
        type=int,
        default=None,
        help=(
            "Dense training only: soft target uniform over up to K routes with Δ>0, "
            "always including the dense-best route; loss is soft CE. "
            "Val top-1 is still vs dense-best class. Overrides --hard_ce_supervision for the loss."
        ),
    )
    p.add_argument("--use_dual_encoder", action="store_true",
                   help="Use DualEncoderRouter (Transformer route embeddings + dot-product logits)")
    p.add_argument("--route_dim", type=int, default=64)
    p.add_argument("--route_enc_layers", type=int, default=2)
    p.add_argument("--route_enc_heads", type=int, default=4)
    p.add_argument("--full_seq_max_len", type=int, default=256,
                   help="Truncate/pad full-sequence residuals to this length")
    p.add_argument(
        "--balanced_per_benchmark",
        action="store_true",
        help="Sample equal train rows per benchmark each epoch (downsample to smallest benchmark).",
    )
    p.add_argument(
        "--robust_temperature",
        type=float,
        default=0.3,
        help="Temperature tau in robust benchmark term tau*logsumexp(L_b/tau).",
    )
    p.add_argument(
        "--lambda_rob",
        type=float,
        default=0.0,
        help="Weight for robust benchmark loss term.",
    )
    p.add_argument(
        "--beta_invalid",
        type=float,
        default=0.0,
        help="Weight for invalid-route probability-mass penalty.",
    )
    p.add_argument(
        "--lambda_bal_cond_entropy",
        type=float,
        default=0.0,
        help="Weight for -sum_b H(q_b) on batch-mean route softmax per benchmark (encourages spread).",
    )
    p.add_argument(
        "--mask_stay_logits",
        action="store_true",
        help="Set STAY logit to large negative before softmax/CE so router never picks class 0.",
    )
    p.add_argument(
        "--ce_weights_by_bench",
        type=str,
        default=None,
        help="Optional comma-separated map, e.g. 'boolq:1.0,commonsenseqa:1.5' for weighted CE.",
    )
    p.add_argument(
        "--pretrain_dense_gate",
        type=str,
        default=None,
        help="If set, train CompressedGate on dense positive labels and save to this .pt path before router.",
    )
    p.add_argument(
        "--split_json",
        type=str,
        default=None,
        help="Canonical split JSON (train/val question ids per benchmark) for dense joint training.",
    )
    p.add_argument(
        "--per_bench_catalog_json",
        type=str,
        default=None,
        help="Optional JSON: benchmark -> list aligned with global class indices (STAY at 0 = null), "
        "for per-anchor materialization of compositional program indices.",
    )
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Log training metrics to Weights & Biases (live when online).",
    )
    p.add_argument("--wandb_project", type=str, default="joint-router")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)
    args = p.parse_args()

    benchmarks = []
    for b in args.benchmarks:
        benchmarks.extend(s.strip() for s in b.split(",") if s.strip())

    if args.eval_only:
        ckpt = args.checkpoint_path or os.path.join(
            args.output_dir, "joint_router_best.pt",
        )
        if not os.path.isfile(ckpt):
            logger.error("Checkpoint not found: %s", ckpt)
            sys.exit(1)
        eval_joint_router(
            ckpt_path=ckpt,
            data_dir=args.data_dir,
            benchmarks=benchmarks or None,
            gate_positives_only=args.gate_positives_only,
            save_results=args.save_results,
        )
        return

    anchor_seqs = _load_anchors(
        args.data_dir, benchmarks,
        anchor_json=args.anchor_seqs_json,
        results_dir=args.results_dir,
    )
    active_benchmarks = [b for b in benchmarks if b in anchor_seqs]
    if not active_benchmarks:
        logger.error("No benchmarks with anchor sequences found. Aborting.")
        sys.exit(1)
    logger.info("Active benchmarks: %s", active_benchmarks)

    ce_weights_parsed = _parse_ce_weights_by_bench(args.ce_weights_by_bench)

    per_bench_catalog: Optional[Dict[str, List[Optional[List[int]]]]] = None
    if args.per_bench_catalog_json:
        with open(args.per_bench_catalog_json) as _pf:
            _raw = json.load(_pf)
        per_bench_catalog = {}
        for _b, _rows in _raw.items():
            parsed_row: List[Optional[List[int]]] = []
            for _ent in _rows:
                if _ent is None:
                    parsed_row.append(None)
                else:
                    parsed_row.append([int(x) for x in _ent])
            per_bench_catalog[str(_b)] = parsed_row

    if args.pretrain_dense_gate:
        if not args.dense_deltas_jsonl:
            logger.error("--pretrain_dense_gate requires --dense_deltas_jsonl")
            sys.exit(1)
        from experiments.run_dense_positive_gate_router import _train_dense_gate

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _train_dense_gate(
            stage_dir=args.data_dir,
            benchmarks=active_benchmarks,
            dense_jsonl=args.dense_deltas_jsonl,
            out_gate=args.pretrain_dense_gate,
            device=dev,
        )

    comp_cfg = CompressorConfig(
        compressor_type=args.compressor_type,
        d_compress=args.compressor_d_compress,
        n_heads=args.compressor_n_heads,
        n_latent_tokens=args.compressor_n_latent,
    )

    train_joint_router(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        benchmarks=active_benchmarks,
        anchor_seqs=anchor_seqs,
        compressor_cfg=comp_cfg,
        gate_positives_only=args.gate_positives_only,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        noop_boost=args.noop_boost,
        dense_deltas_jsonl=args.dense_deltas_jsonl,
        catalog_json=args.catalog_json,
        hard_ce_supervision=args.hard_ce_supervision,
        dense_supervision_topk=args.dense_supervision_topk,
        use_dual_encoder=args.use_dual_encoder,
        route_dim=args.route_dim,
        route_enc_layers=args.route_enc_layers,
        route_enc_heads=args.route_enc_heads,
        full_seq_max_len=args.full_seq_max_len,
        balanced_per_benchmark=args.balanced_per_benchmark,
        robust_temperature=args.robust_temperature,
        lambda_rob=args.lambda_rob,
        beta_invalid=args.beta_invalid,
        lambda_bal_cond_entropy=args.lambda_bal_cond_entropy,
        mask_stay_logits=args.mask_stay_logits,
        catalog_mode=args.catalog_mode,
        ce_weights_by_bench=ce_weights_parsed,
        split_json_path=args.split_json,
        per_bench_catalog=per_bench_catalog,
        wandb_enabled=bool(args.wandb),
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_group=args.wandb_group,
    )


if __name__ == "__main__":
    main()
