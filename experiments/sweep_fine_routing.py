#!/usr/bin/env python3
"""Bayesian hyperparameter sweep for fine-routing gate + router.

import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

Uses W&B sweeps to search over gate/router architecture, training,
and inference hyperparameters.  The LLM is loaded once; each trial
trains gate + router (seconds) then evaluates on a validation subset
(~30s).

After the sweep, W&B automatically provides parallel-coordinates and
importance plots on the dashboard.

Usage
-----
    python sweep_fine_routing.py \
        --data_dir fine_routing_data_p0 \
        --benchmark winogrande \
        --results_dir predictions/qwen25_0.5b_v2_sdpa \
        --model_name Qwen/Qwen2.5-0.5B-Instruct \
        --eval_questions 200 \
        --count 60

    # MCTS Winogrande build (~13k+ samples), wider sweep:
    #   bash run_sweep_fine_routing_winogrande_mcts.sh
    # or:
    #   python sweep_fine_routing.py ... --data_dir fine_routing_data_winogrande_mcts \
    #       --results_dir predictions --large_search_space --count 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import types
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from tqdm import tqdm

import wandb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------

from core.benchmark_mcts import grade_response, seq_to_layers
from routers.fine_routing_config import FineRoutingConfig
from routers.fine_routing_deviations import (
    apply_deviation,
    enumerate_deviations,
)
from core.flexible_models import FlexibleModelWrapper, get_is_instruct
from core.permutation_mcts import prepare_arc_data
from training.train_benchmark_router import load_optimal_sequences_from_results
from pipeline.forward import get_pivot_residual, get_full_sequence_residual
from train_fine_gate import DeltaGate, FineGate
from train_fine_router import FineRouter, PositionalFineRouter, soft_cross_entropy
from data_prep.build_ft_fine_routing_dataset import (
    FTFlexibleModelWrapper,
    find_adapter_path,
)
from routers.residual_compressors import (
    CompressorConfig,
    CompressedRouter,
    CompressedGate,
    build_compressor,
    pad_sequences,
)


# ---------------------------------------------------------------------------
# Data loading (done once)
# ---------------------------------------------------------------------------

def build_mcts_sequence_catalog(
    records: List[Dict],
    anchor_seq: List[int],
) -> Tuple[List[List[int]], Dict[tuple, int]]:
    """Build a sequence catalog from all unique sequences explored by MCTS.

    Returns ``(catalog, seq_to_idx)`` where ``catalog[0]`` is the anchor (noop)
    and subsequent entries are every other unique sequence that appeared in
    ``explored`` across all *records*.
    """
    seq_set: Dict[tuple, None] = {}
    anchor_t = tuple(anchor_seq)
    seq_set[anchor_t] = None
    for rec in records:
        for ex in rec["explored"]:
            seq_set[tuple(int(x) for x in ex["seq"])] = None
    catalog = [anchor_seq] + [list(s) for s in seq_set if s != anchor_t]
    seq_to_idx = {tuple(s): i for i, s in enumerate(catalog)}
    return catalog, seq_to_idx


def build_mcts_router_targets(
    records: List[Dict],
    seq_to_idx: Dict[tuple, int],
    num_classes: int,
    noop_boost: float = 0.0,
) -> List[torch.Tensor]:
    """Build router soft-label targets using MCTS explored scores.

    Each question's ``router_target`` (softmax over explored seqs) is mapped
    by exact sequence lookup to a fixed-size ``[num_classes]`` vector.

    *noop_boost* (>= 0) controls how the router learns gating:
      - 0: vanilla MCTS soft labels (no special treatment)
      - > 0: for gate_label=0 questions, adds *noop_boost* extra mass to noop
        (index 0) before renormalizing, biasing the router toward "keep anchor".
        For gate_label=1, noop mass is zeroed out and renormalized over non-noop.
    """
    out: List[torch.Tensor] = []
    for rec in records:
        explored = rec["explored"]
        rt = rec["router_target"]
        p = torch.zeros(num_classes, dtype=torch.float32)
        for j, prob in enumerate(rt):
            seq = tuple(int(x) for x in explored[j]["seq"])
            idx = seq_to_idx.get(seq)
            if idx is not None:
                p[idx] += float(prob)
        s = float(p.sum())
        if s > 1e-12:
            p = p / s
        else:
            p[0] = 1.0

        if noop_boost > 0:
            if rec["gate_label"] == 0:
                p[0] += noop_boost
                p = p / p.sum()
            else:
                p[0] = 0.0
                s2 = float(p.sum())
                if s2 > 1e-12:
                    p = p / s2
                else:
                    p[0] = 1.0

        out.append(p)
    return out


def load_bench_data_mcts(
    data_dir: str,
    benchmark: str,
    anchor_seq: List[int],
) -> Tuple[torch.Tensor, List[int], List[torch.Tensor], List[List[int]], List[Dict]]:
    """Load MCTS training data, building sequence catalog from data.

    Returns ``(residuals, gate_labels, raw_router_targets, sequence_catalog, records)``
    where ``raw_router_targets`` use vanilla soft labels (noop_negatives=False).
    The caller can rebuild targets with different settings using the *records*.
    """
    pt_path = os.path.join(data_dir, f"{benchmark}_pivot_residuals.pt")
    jsonl_path = os.path.join(data_dir, f"{benchmark}.jsonl")

    residuals = torch.load(pt_path, map_location="cpu", weights_only=True).float()
    n_pt = residuals.shape[0]

    with open(jsonl_path) as f:
        records = [json.loads(line) for line in f]
    n = min(n_pt, len(records))
    if n < n_pt or n < len(records):
        logger.warning(
            "Aligning data: truncating to %d rows (pt=%d, jsonl=%d)",
            n, n_pt, len(records),
        )
    residuals = residuals[:n]
    records = records[:n]
    gate_labels = [r["gate_label"] for r in records]

    catalog_full, seq_to_idx_full = build_mcts_sequence_catalog(records, anchor_seq)
    n_cls_full = len(catalog_full)
    logger.info("  MCTS full sequence catalog: |C|=%d unique sequences", n_cls_full)

    catalog_reduced, seq_to_idx_reduced = build_reduced_catalog(records, anchor_seq)
    n_cls_reduced = len(catalog_reduced)
    logger.info("  MCTS reduced catalog (best_seq only): |C|=%d", n_cls_reduced)

    router_targets = build_mcts_router_targets(records, seq_to_idx_full, n_cls_full)

    return (residuals, gate_labels, router_targets,
            catalog_full, seq_to_idx_full,
            catalog_reduced, seq_to_idx_reduced,
            records)


def sharpen_targets(
    targets: List[torch.Tensor], temperature: float
) -> List[torch.Tensor]:
    """Temperature-scale soft targets: p_i^(1/T) / sum(p^(1/T)).

    T < 1 sharpens (concentrates mass on top entries), T > 1 flattens.
    """
    if abs(temperature - 1.0) < 1e-6:
        return targets
    inv_t = 1.0 / temperature
    out = []
    for p in targets:
        q = p.clamp(min=1e-12).pow(inv_t)
        out.append(q / q.sum())
    return out


def build_best_seq_targets(
    records: List[Dict],
    seq_to_idx: Dict[tuple, int],
    num_classes: int,
) -> List[torch.Tensor]:
    """Build hard targets from best_seq (one-hot).

    gate_label=0 → noop (index 0, the anchor): MCTS found no improvement.
    gate_label=1 → one-hot on the best MCTS sequence.
    """
    out: List[torch.Tensor] = []
    for rec in records:
        p = torch.zeros(num_classes, dtype=torch.float32)
        if rec.get("gate_label", 0) == 0:
            p[0] = 1.0
        else:
            best = tuple(int(x) for x in rec["best_seq"])
            idx = seq_to_idx.get(best, 0)
            p[idx] = 1.0
        out.append(p)
    return out


def build_reduced_catalog(
    records: List[Dict],
    anchor_seq: List[int],
) -> Tuple[List[List[int]], Dict[tuple, int]]:
    """Build a minimal catalog containing only anchor + all best_seq values.

    This dramatically reduces the output space (e.g. 832 → 517) which makes
    the classification task much more learnable.
    """
    anchor_t = tuple(anchor_seq)
    best_set: Dict[tuple, None] = {anchor_t: None}
    for rec in records:
        best_set[tuple(int(x) for x in rec["best_seq"])] = None
    catalog = [anchor_seq] + [list(s) for s in best_set if s != anchor_t]
    seq_to_idx = {tuple(s): i for i, s in enumerate(catalog)}
    return catalog, seq_to_idx


def rebuild_targets_for_trial(
    records: List[Dict],
    seq_to_idx: Dict[tuple, int],
    num_classes: int,
    noop_boost: float,
    target_temp: float,
    use_best_seq: bool = False,
) -> List[torch.Tensor]:
    """Rebuild router targets with per-trial settings."""
    if use_best_seq:
        return build_best_seq_targets(records, seq_to_idx, num_classes)
    targets = build_mcts_router_targets(
        records, seq_to_idx, num_classes, noop_boost=noop_boost
    )
    if abs(target_temp - 1.0) > 1e-6:
        targets = sharpen_targets(targets, target_temp)
    return targets


def load_bench_data_enumerated(
    data_dir: str,
    benchmark: str,
) -> Tuple[torch.Tensor, List[int], List[torch.Tensor]]:
    """Load exhaustive-enumeration training data (fixed ``router_target`` dim)."""
    pt_path = os.path.join(data_dir, f"{benchmark}_pivot_residuals.pt")
    jsonl_path = os.path.join(data_dir, f"{benchmark}.jsonl")

    residuals = torch.load(pt_path, map_location="cpu", weights_only=True).float()
    with open(jsonl_path) as f:
        records = [json.loads(line) for line in f]
    n = min(residuals.shape[0], len(records))
    if n < residuals.shape[0] or n < len(records):
        logger.warning(
            "Aligning data: truncating to %d rows (pt=%d, jsonl=%d)",
            n, residuals.shape[0], len(records),
        )
    residuals = residuals[:n]
    records = records[:n]
    gate_labels = [r["gate_label"] for r in records]
    router_targets = [
        torch.tensor(r["router_target"], dtype=torch.float32) for r in records
    ]
    return residuals, gate_labels, router_targets


# ---------------------------------------------------------------------------
# Train gate (in-process, no disk I/O)
# ---------------------------------------------------------------------------

def train_gate_inline(
    residuals: torch.Tensor,
    labels: List[int],
    d_model: int,
    hidden_dim: int,
    gate_dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    recall_boost: float,
    device: torch.device,
    seed: int = 42,
) -> FineGate:
    torch.manual_seed(seed)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    pw = (n_neg / max(n_pos, 1)) * recall_boost
    pos_weight = torch.tensor([pw], device=device)

    y = torch.tensor(labels, dtype=torch.float32)
    ds = TensorDataset(residuals, y)

    val_size = max(1, int(len(ds) * 0.15))
    train_ds, val_ds = random_split(ds, [len(ds) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = FineGate(d_model, hidden_dim, dropout=gate_dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y_b in train_loader:
            x, y_b = x.to(device), y_b.to(device)
            loss = F.binary_cross_entropy_with_logits(
                model(x), y_b, pos_weight=pos_weight
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y_b in val_loader:
                x, y_b = x.to(device), y_b.to(device)
                val_loss += F.binary_cross_entropy_with_logits(
                    model(x), y_b, pos_weight=pos_weight
                ).item() * x.size(0)
        val_loss /= val_size
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def train_delta_gate_inline(
    residuals: torch.Tensor,
    best_deltas: List[float],
    d_model: int,
    hidden_dim: int,
    gate_dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    fp_weight: float,
    device: torch.device,
    seed: int = 42,
) -> DeltaGate:
    """Train a DeltaGate (regression) in-process."""
    torch.manual_seed(seed)

    y = torch.tensor(best_deltas, dtype=torch.float32)
    ds = TensorDataset(residuals, y)

    val_size = max(1, int(len(ds) * 0.15))
    train_ds, val_ds = random_split(ds, [len(ds) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = DeltaGate(d_model, hidden_dim, dropout=gate_dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y_b in train_loader:
            x, y_b = x.to(device), y_b.to(device)
            pred = model(x)
            residual = pred - y_b
            base_loss = F.smooth_l1_loss(pred, y_b, reduction="none")
            weight = torch.where(residual > 0, fp_weight, 1.0)
            loss = (base_loss * weight).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y_b in val_loader:
                x, y_b = x.to(device), y_b.to(device)
                pred = model(x)
                residual = pred - y_b
                base_loss = F.smooth_l1_loss(pred, y_b, reduction="none")
                weight = torch.where(residual > 0, fp_weight, 1.0)
                val_loss += (base_loss * weight).mean().item() * x.size(0)
        val_loss /= val_size
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Train router (in-process, no disk I/O)
# ---------------------------------------------------------------------------

def train_router_inline(
    residuals: torch.Tensor,
    gate_labels: List[int],
    router_targets: List[torch.Tensor],
    d_model: int,
    num_classes: int,
    hidden_dims: List[int],
    router_dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    gate_positives_only: bool,
    device: torch.device,
    hard_targets: bool = False,
    label_smoothing: float = 0.0,
    weight_decay: float = 0.01,
    inverse_freq_class_weights: bool = True,
    seed: int = 42,
) -> FineRouter:
    torch.manual_seed(seed)

    idxs = list(range(len(gate_labels)))
    if gate_positives_only:
        idxs = [i for i in idxs if gate_labels[i] == 1]
    if not idxs:
        idxs = list(range(len(gate_labels)))

    X = residuals[idxs]

    if hard_targets:
        Y = torch.tensor(
            [router_targets[i].argmax().item() for i in idxs], dtype=torch.long
        )
        if inverse_freq_class_weights:
            class_counts = torch.bincount(Y, minlength=num_classes).float().clamp(min=1)
            class_weights = (1.0 / class_counts)
            class_weights = class_weights / class_weights.mean()
            class_weights = class_weights.to(device)
            loss_fn = lambda logits, y: F.cross_entropy(
                logits, y, weight=class_weights, label_smoothing=label_smoothing
            )
        else:
            loss_fn = lambda logits, y: F.cross_entropy(
                logits, y, label_smoothing=label_smoothing
            )
    else:
        if label_smoothing > 0:
            n_cls = router_targets[0].shape[0]
            uniform = torch.full((n_cls,), 1.0 / n_cls)
            Y = torch.stack([
                (1 - label_smoothing) * router_targets[i] + label_smoothing * uniform
                for i in idxs
            ])
        else:
            Y = torch.stack([router_targets[i] for i in idxs])
        loss_fn = soft_cross_entropy

    ds = TensorDataset(X, Y)

    val_size = max(1, int(len(ds) * 0.15))
    train_ds, val_ds = random_split(ds, [len(ds) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = FineRouter(d_model, num_classes, hidden_dims, router_dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y_b in train_loader:
            x, y_b = x.to(device), y_b.to(device)
            loss = loss_fn(model(x), y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y_b in val_loader:
                x, y_b = x.to(device), y_b.to(device)
                val_loss += loss_fn(model(x), y_b).item() * x.size(0)
        val_loss /= val_size
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Train compressed router/gate inline (for sweep with attention compressors)
# ---------------------------------------------------------------------------

def train_compressed_router_inline(
    full_residuals: List[torch.Tensor],
    gate_labels: List[int],
    router_targets: List[torch.Tensor],
    d_model: int,
    num_classes: int,
    compressor_cfg: CompressorConfig,
    hidden_dims: List[int],
    router_dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    gate_positives_only: bool,
    device: torch.device,
    weight_decay: float = 0.01,
    seed: int = 42,
) -> CompressedRouter:
    """Train a CompressedRouter in-process with full-sequence data.

    Pre-pads all sequences once to avoid repeated per-batch padding overhead.
    """
    torch.manual_seed(seed)

    idxs = list(range(len(gate_labels)))
    if gate_positives_only:
        idxs = [i for i in idxs if gate_labels[i] == 1]
    if not idxs:
        idxs = list(range(len(gate_labels)))

    X_list = [full_residuals[i] for i in idxs]
    Y = torch.stack([router_targets[i] for i in idxs])

    X_pad, X_mask = pad_sequences(X_list, max_seq_len=256)
    X_pad = X_pad.to(device)
    X_mask = X_mask.to(device)
    Y = Y.to(device)

    compressor_cfg.d_model = d_model
    compressor = build_compressor(compressor_cfg)
    model = CompressedRouter(compressor, num_classes, hidden_dims, router_dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    N = len(idxs)
    val_size = max(1, int(N * 0.15))
    torch.manual_seed(seed)
    perm = torch.randperm(N)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        torch.manual_seed(seed + epoch)
        shuffled = train_idx[torch.randperm(len(train_idx))]
        for start in range(0, len(shuffled), batch_size):
            bi = shuffled[start:start + batch_size]
            logits = model(X_pad[bi], attention_mask=X_mask[bi])
            loss = soft_cross_entropy(logits, Y[bi])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vstart in range(0, len(val_idx), batch_size):
                bi = val_idx[vstart:vstart + batch_size]
                val_loss += soft_cross_entropy(
                    model(X_pad[bi], attention_mask=X_mask[bi]), Y[bi]
                ).item() * len(bi)
        val_loss /= val_size
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def train_compressed_gate_inline(
    full_residuals: List[torch.Tensor],
    labels: List[int],
    d_model: int,
    compressor_cfg: CompressorConfig,
    hidden_dim: int,
    gate_dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    recall_boost: float,
    device: torch.device,
    seed: int = 42,
) -> CompressedGate:
    """Train a CompressedGate in-process with full-sequence data.

    Pre-pads all sequences once to avoid repeated per-batch padding overhead.
    """
    torch.manual_seed(seed)

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    pw = (n_neg / max(n_pos, 1)) * recall_boost
    pos_weight = torch.tensor([pw], device=device)

    X_pad, X_mask = pad_sequences(full_residuals, max_seq_len=256)
    X_pad = X_pad.to(device)
    X_mask = X_mask.to(device)
    y_all = torch.tensor(labels, dtype=torch.float32, device=device)

    compressor_cfg.d_model = d_model
    compressor = build_compressor(compressor_cfg)
    model = CompressedGate(compressor, hidden_dim, dropout=gate_dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    N = len(labels)
    val_size = max(1, int(N * 0.15))
    perm = torch.randperm(N)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        torch.manual_seed(seed + epoch)
        shuffled = train_idx[torch.randperm(len(train_idx))]
        for start in range(0, len(shuffled), batch_size):
            bi = shuffled[start:start + batch_size]
            logits = model(X_pad[bi], attention_mask=X_mask[bi])
            loss = F.binary_cross_entropy_with_logits(logits, y_all[bi], pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vstart in range(0, len(val_idx), batch_size):
                bi = val_idx[vstart:vstart + batch_size]
                val_loss += F.binary_cross_entropy_with_logits(
                    model(X_pad[bi], attention_mask=X_mask[bi]),
                    y_all[bi], pos_weight=pos_weight,
                ).item() * len(bi)
        val_loss /= val_size
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def build_positional_vocab(
    anchor_seq: List[int],
    editable_start: int,
    num_layers: int,
    radius: int,
) -> List[List[int]]:
    """Build the per-position value vocabulary for the positional router.

    Each editable position can take:
      - Its anchor value (always included)
      - Values in [pos - radius, pos + radius] clamped to [0, num_layers-1]
      - SKIP (-1)
    Sorted for determinism. Returns one list per editable position.
    """
    from routers.fine_routing_deviations import SKIP
    vocab = []
    for pos in range(editable_start, num_layers):
        vals = set()
        vals.add(anchor_seq[pos])
        lo = max(0, pos - radius)
        hi = min(num_layers - 1, pos + radius)
        for v in range(lo, hi + 1):
            vals.add(v)
        vals.add(SKIP)
        vocab.append(sorted(vals))
    return vocab


def build_positional_targets(
    records: List[Dict],
    gate_labels: List[int],
    anchor_seq: List[int],
    editable_start: int,
    values_per_position: List[List[int]],
    gate_positives_only: bool = False,
) -> List[List[torch.Tensor]]:
    """Build per-position soft targets from MCTS explored sequences.

    For each question, for each editable position, produces a softmax
    distribution over the position's vocabulary based on the scores of
    explored sequences that used each value at that position.
    """
    num_positions = len(values_per_position)
    val_to_idx = [
        {v: i for i, v in enumerate(vp)} for vp in values_per_position
    ]

    targets = []
    for q_idx, rec in enumerate(records):
        if gate_positives_only and gate_labels[q_idx] == 0:
            continue
        explored = rec.get("explored", [])
        anchor_score = rec.get("anchor_score", 0.0)

        pos_targets = []
        for p in range(num_positions):
            vocab_size = len(values_per_position[p])
            counts = torch.zeros(vocab_size)
            for entry in explored:
                seq = entry["seq"]
                score = entry["score"]
                delta = score - anchor_score
                pos_val = seq[editable_start + p]
                if pos_val in val_to_idx[p]:
                    idx = val_to_idx[p][pos_val]
                    counts[idx] += max(0.0, delta + 1.0)
            if counts.sum() < 1e-9:
                anchor_val = anchor_seq[editable_start + p]
                if anchor_val in val_to_idx[p]:
                    counts[val_to_idx[p][anchor_val]] = 1.0
                else:
                    counts[0] = 1.0
            pos_targets.append(counts / counts.sum())
        targets.append(pos_targets)
    return targets


def train_positional_router_inline(
    residuals: torch.Tensor,
    gate_labels: List[int],
    records: List[Dict],
    anchor_seq: List[int],
    editable_start: int,
    num_layers: int,
    radius: int,
    hidden_dims: List[int],
    router_dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    gate_positives_only: bool,
    device: torch.device,
    weight_decay: float = 0.01,
    seed: int = 42,
) -> PositionalFineRouter:
    """Train a PositionalFineRouter in-process."""
    torch.manual_seed(seed)

    values_per_position = build_positional_vocab(
        anchor_seq, editable_start, num_layers, radius
    )
    num_positions = len(values_per_position)

    pos_targets = build_positional_targets(
        records, gate_labels, anchor_seq, editable_start,
        values_per_position, gate_positives_only,
    )

    if gate_positives_only:
        idxs = [i for i in range(len(gate_labels)) if gate_labels[i] == 1]
    else:
        idxs = list(range(len(gate_labels)))
    if not idxs:
        idxs = list(range(len(gate_labels)))

    X = residuals[idxs]
    d_model = X.shape[1]

    model = PositionalFineRouter(
        d_model=d_model,
        num_positions=num_positions,
        values_per_position=values_per_position,
        anchor_seq=anchor_seq,
        editable_start=editable_start,
        hidden_dims=hidden_dims,
        dropout=router_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    max_vocab = max(len(vp) for vp in values_per_position)
    Y_all = []
    for tgt_list in pos_targets:
        padded = []
        for t in tgt_list:
            if t.shape[0] < max_vocab:
                t = F.pad(t, (0, max_vocab - t.shape[0]))
            padded.append(t)
        Y_all.append(torch.stack(padded))
    Y_all_tensor = torch.stack(Y_all)

    ds = TensorDataset(X, Y_all_tensor)
    val_size = max(1, int(len(ds) * 0.15))
    train_ds, val_ds = random_split(ds, [len(ds) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y_b in train_loader:
            x, y_b = x.to(device), y_b.to(device)
            logits_list = model(x)
            loss = sum(
                soft_cross_entropy(logits_list[p], y_b[:, p, :logits_list[p].shape[-1]])
                for p in range(num_positions)
            ) / num_positions
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y_b in val_loader:
                x, y_b = x.to(device), y_b.to(device)
                logits_list = model(x)
                loss = sum(
                    soft_cross_entropy(logits_list[p], y_b[:, p, :logits_list[p].shape[-1]])
                    for p in range(num_positions)
                ) / num_positions
                val_loss += loss.item() * x.size(0)
        val_loss /= val_size
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# LLM-based evaluation (subset of validation)
# ---------------------------------------------------------------------------

def generate_under_layers(
    wrapper: FlexibleModelWrapper,
    layers: List[int],
    text: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 1,
    is_math: bool = False,
) -> str:
    saved = wrapper.model.model.layer_indices
    wrapper.model.model.layer_indices = layers
    try:
        has_dup = len(layers) != len(set(layers))
        prompt = wrapper.prepare_prompt(text, system_prompt=system_prompt)
        inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(wrapper.model.device)
        input_len = inputs.input_ids.shape[1]
        gen_kw = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": wrapper.tokenizer.eos_token_id,
            "do_sample": False,
        }
        if has_dup or is_math or len(layers) != wrapper.num_layers:
            gen_kw["use_cache"] = False
        with torch.no_grad():
            out = wrapper.model.generate(**inputs, **gen_kw)
        return wrapper.tokenizer.decode(
            out[0][input_len:], skip_special_tokens=True
        ).strip()
    finally:
        wrapper.model.model.layer_indices = saved


def evaluate(
    wrapper: FlexibleModelWrapper,
    gate: Optional[FineGate],
    router: FineRouter,
    gamma: float,
    anchor_seq: List[int],
    sequence_catalog: List[List[int]],
    samples: list,
    benchmark: str,
    model_name: str,
    pivot_layer: int,
    gate_device: torch.device,
    gating_mode: str = "gate_network",
    confidence_threshold: float = 0.0,
    delta_gate: Optional[DeltaGate] = None,
    delta_margin: float = 0.0,
) -> Dict:
    """Evaluate routing on validation samples.

    *gating_mode* controls how the route/no-route decision is made:
      - ``"gate_network"``: classic separate gate; route if sigmoid(gate) >= gamma
      - ``"router_argmax"``: route iff router argmax != 0 (noop); ignore gate
      - ``"router_confidence"``: route iff router P(top non-noop) > confidence_threshold
      - ``"delta_gate"``: route iff DeltaGate predicts E[delta] > delta_margin

    *sequence_catalog* maps router output index -> full layer sequence.
    Index 0 is always the anchor (noop).
    """
    anchor_layers = seq_to_layers(anchor_seq)
    is_math = "dart" in benchmark or benchmark in ("gsm8k_hard", "math500")

    anchor_correct = 0
    routed_correct = 0
    gate_opened = 0
    gain_when_opened = 0.0
    helped_when_opened = 0
    hurt_when_opened = 0
    n = len(samples)

    for sample in tqdm(samples, desc=f"eval({benchmark})", leave=False):
        anchor_resp = generate_under_layers(
            wrapper, anchor_layers, sample["input"],
            system_prompt=sample.get("system_prompt"),
            max_new_tokens=sample["max_new_tokens"],
            is_math=is_math,
        )
        anchor_sc = grade_response(
            anchor_resp, sample["correct"], benchmark, model_name, sample["input"]
        )
        anchor_ok = int(anchor_sc > 0.5)
        anchor_correct += anchor_ok

        h_pivot = get_pivot_residual(
            wrapper, sample["input"],
            layer_indices=anchor_layers,
            pivot_layer=pivot_layer,
            system_prompt=sample.get("system_prompt"),
        ).float().to(gate_device)

        with torch.no_grad():
            router_logits = router(h_pivot)
            router_probs = F.softmax(router_logits, dim=-1)
            pred_idx = router_logits.argmax(dim=-1).item()

        should_route = False
        if gating_mode == "gate_network" and gate is not None:
            with torch.no_grad():
                gate_prob = torch.sigmoid(gate(h_pivot)).item()
            should_route = gate_prob >= gamma and pred_idx != 0
        elif gating_mode == "router_argmax":
            should_route = pred_idx != 0
        elif gating_mode == "router_confidence":
            deviate_prob = 1.0 - router_probs[..., 0].item()
            should_route = deviate_prob > confidence_threshold
            if should_route:
                non_noop = router_probs.clone()
                non_noop[..., 0] = 0.0
                pred_idx = non_noop.argmax(dim=-1).item()
        elif gating_mode == "delta_gate" and delta_gate is not None:
            with torch.no_grad():
                predicted_delta = delta_gate(h_pivot).item()
            should_route = predicted_delta > delta_margin and pred_idx != 0

        if not should_route:
            routed_correct += anchor_ok
        else:
            gate_opened += 1
            cand_seq = sequence_catalog[pred_idx]
            cand_layers = seq_to_layers(cand_seq)
            cand_resp = generate_under_layers(
                wrapper, cand_layers, sample["input"],
                system_prompt=sample.get("system_prompt"),
                max_new_tokens=sample["max_new_tokens"],
                is_math=is_math,
            )
            cand_sc = grade_response(
                cand_resp, sample["correct"], benchmark,
                model_name, sample["input"],
            )
            cand_ok = int(cand_sc > 0.5)
            routed_correct += cand_ok
            delta = cand_sc - anchor_sc
            gain_when_opened += delta
            if delta > 0:
                helped_when_opened += 1
            elif delta < 0:
                hurt_when_opened += 1

    return {
        "n": n,
        "anchor_accuracy": anchor_correct / max(n, 1),
        "routed_accuracy": routed_correct / max(n, 1),
        "gate_open_rate": gate_opened / max(n, 1),
        "conditional_gain": gain_when_opened / max(gate_opened, 1),
        "unconditional_gain": (
            (routed_correct - anchor_correct) / max(n, 1)
        ),
        "helped_when_opened": helped_when_opened,
        "hurt_when_opened": hurt_when_opened,
        "helped_frac_when_opened": helped_when_opened / max(gate_opened, 1),
        "hurt_frac_when_opened": hurt_when_opened / max(gate_opened, 1),
        "net_helped": helped_when_opened - hurt_when_opened,
    }


def _forward_logits(wrapper, layers, text, system_prompt=None):
    """Forward pass returning last-position logits [vocab]."""
    saved = wrapper.model.model.layer_indices
    wrapper.model.model.layer_indices = layers
    try:
        has_dup = len(layers) != len(set(layers))
        prompt = wrapper.prepare_prompt(text, system_prompt=system_prompt)
        inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(wrapper.model.device)
        kw: dict = {}
        if has_dup or len(layers) != wrapper.num_layers:
            kw["use_cache"] = False
        with torch.no_grad():
            out = wrapper.model(input_ids=inputs.input_ids,
                                attention_mask=inputs.attention_mask, **kw)
        return out.logits[0, -1, :]
    finally:
        wrapper.model.model.layer_indices = saved


def evaluate_propose_verify(
    wrapper: FlexibleModelWrapper,
    router,
    anchor_seq: List[int],
    sequence_catalog: List[List[int]],
    samples: list,
    benchmark: str,
    model_name: str,
    pivot_layer: int,
    gate_device: torch.device,
    top_k: int = 5,
    confidence_margin: float = 0.0,
) -> Dict:
    """Propose-and-verify evaluation: router proposes top-K routes, model
    confidence (log-prob) selects the best.

    For each question:
      1. Run anchor forward to get its answer + log-prob.
      2. Router proposes top-K alternative routes.
      3. For each candidate, run a forward pass and record max answer log-prob.
      4. Pick the route (including anchor) with the highest confidence.
      5. Only adopt the alternative if its confidence exceeds anchor's by
         *confidence_margin*.

    Works with both ``FineRouter`` (catalog-based) and
    ``PositionalFineRouter`` (sequence-generating).
    """
    from train_fine_router import PositionalFineRouter

    anchor_layers = seq_to_layers(anchor_seq)
    is_math = "dart" in benchmark or benchmark in ("gsm8k_hard", "math500")
    is_positional = isinstance(router, PositionalFineRouter)

    anchor_correct = 0
    routed_correct = 0
    gate_opened = 0
    helped_when_opened = 0
    hurt_when_opened = 0
    n = len(samples)

    for sample in tqdm(samples, desc=f"eval_pv({benchmark})", leave=False):
        sys_prompt = sample.get("system_prompt")
        max_tok = sample["max_new_tokens"]

        anchor_resp = generate_under_layers(
            wrapper, anchor_layers, sample["input"],
            system_prompt=sys_prompt, max_new_tokens=max_tok, is_math=is_math,
        )
        anchor_sc = grade_response(
            anchor_resp, sample["correct"], benchmark, model_name, sample["input"]
        )
        anchor_ok = int(anchor_sc > 0.5)
        anchor_correct += anchor_ok

        anchor_logits = _forward_logits(wrapper, anchor_layers, sample["input"],
                                        system_prompt=sys_prompt)
        anchor_conf = anchor_logits.max().item()

        h_pivot = get_pivot_residual(
            wrapper, sample["input"],
            layer_indices=anchor_layers,
            pivot_layer=pivot_layer,
            system_prompt=sys_prompt,
        ).float().to(gate_device)

        if is_positional:
            candidate_seqs = router.predict_topk_sequences(h_pivot, k=top_k)
        else:
            with torch.no_grad():
                router_logits = router(h_pivot)
            _, topk_idxs = router_logits.topk(min(top_k, router_logits.shape[-1]), dim=-1)
            candidate_seqs = []
            for j in range(topk_idxs.shape[-1]):
                idx = topk_idxs[0, j].item()
                if idx < len(sequence_catalog):
                    candidate_seqs.append(sequence_catalog[idx])

        best_conf = anchor_conf
        best_layers = anchor_layers
        for cand_seq in candidate_seqs:
            cand_layers = seq_to_layers(cand_seq)
            if cand_layers == anchor_layers:
                continue
            cand_logits = _forward_logits(wrapper, cand_layers, sample["input"],
                                          system_prompt=sys_prompt)
            cand_conf = cand_logits.max().item()
            if cand_conf > best_conf:
                best_conf = cand_conf
                best_layers = cand_layers

        if best_layers is not anchor_layers and best_conf > anchor_conf + confidence_margin:
            gate_opened += 1
            cand_resp = generate_under_layers(
                wrapper, best_layers, sample["input"],
                system_prompt=sys_prompt, max_new_tokens=max_tok, is_math=is_math,
            )
            cand_sc = grade_response(
                cand_resp, sample["correct"], benchmark, model_name, sample["input"]
            )
            cand_ok = int(cand_sc > 0.5)
            routed_correct += cand_ok
            delta = cand_sc - anchor_sc
            if delta > 0:
                helped_when_opened += 1
            elif delta < 0:
                hurt_when_opened += 1
        else:
            routed_correct += anchor_ok

    return {
        "n": n,
        "anchor_accuracy": anchor_correct / max(n, 1),
        "routed_accuracy": routed_correct / max(n, 1),
        "gate_open_rate": gate_opened / max(n, 1),
        "unconditional_gain": (routed_correct - anchor_correct) / max(n, 1),
        "helped_when_opened": helped_when_opened,
        "hurt_when_opened": hurt_when_opened,
        "net_helped": helped_when_opened - hurt_when_opened,
        "top_k": top_k,
        "confidence_margin": confidence_margin,
    }


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "unconditional_gain", "goal": "maximize"},
    "parameters": {
        "gate_hidden_dim": {"values": [64, 128, 256, 512]},
        "gate_lr": {
            "min": 1e-4, "max": 5e-2,
            "distribution": "log_uniform_values",
        },
        "gate_epochs": {"min": 20, "max": 120},
        "gate_dropout": {"min": 0.0, "max": 0.5},
        "recall_boost": {"min": 0.3, "max": 4.0},
        "router_h1": {"values": [64, 128, 256, 512]},
        "router_h2": {"values": [32, 64, 128, 256]},
        "router_lr": {
            "min": 1e-4, "max": 5e-2,
            "distribution": "log_uniform_values",
        },
        "router_epochs": {"min": 20, "max": 200},
        "router_dropout": {"min": 0.0, "max": 0.5},
        "router_gate_pos_only": {"values": [True, False]},
        "router_hard_targets": {"values": [True, False]},
        "gamma": {"min": 0.05, "max": 0.85},
        "gating_mode": {"values": ["gate_network"]},
    },
}

SWEEP_CONFIG_LARGE = {
    "method": "bayes",
    "metric": {"name": "unconditional_gain", "goal": "maximize"},
    "parameters": {
        "router_h1": {"values": [256, 512, 768, 1024]},
        "router_h2": {"values": [128, 256, 384, 512]},
        "router_h3": {"values": [0, 64, 128, 256]},
        "router_lr": {
            "min": 1e-4, "max": 5e-2,
            "distribution": "log_uniform_values",
        },
        "router_epochs": {"min": 40, "max": 300},
        "router_dropout": {"min": 0.05, "max": 0.5},
        "router_hard_targets": {"values": [True, False]},
        "label_smoothing": {"min": 0.0, "max": 0.15},
        "weight_decay": {
            "min": 1e-3, "max": 0.1,
            "distribution": "log_uniform_values",
        },
        # Target construction
        "use_best_seq": {"values": [True, False]},
        "noop_boost": {"min": 0.0, "max": 5.0},
        "target_temp": {"min": 0.1, "max": 1.0},
        # Gating
        "gating_mode": {"values": ["gate_network", "router_confidence", "router_argmax", "delta_gate"]},
        "confidence_threshold": {"min": 0.0, "max": 0.85},
        "delta_margin": {"min": 0.0, "max": 2.0},
        # Gate network
        "gate_hidden_dim": {"values": [128, 256, 512]},
        "gate_lr": {
            "min": 5e-5, "max": 5e-2,
            "distribution": "log_uniform_values",
        },
        "gate_epochs": {"min": 20, "max": 150},
        "gate_dropout": {"min": 0.0, "max": 0.5},
        "recall_boost": {"min": 0.5, "max": 4.0},
        "gamma": {"min": 0.05, "max": 0.85},
        "router_gate_pos_only": {"values": [True, False]},
    },
}


def _resolve_anchor_seq(
    *,
    data_dir: str,
    benchmark: str,
    results_dir: str,
    model_name: str,
    num_layers: int,
) -> List[int]:
    """Resolve anchor in same order for FT/non-FT paths."""
    jsonl_path = os.path.join(data_dir, f"{benchmark}.jsonl")
    if os.path.isfile(jsonl_path):
        with open(jsonl_path) as f:
            first_line = f.readline().strip()
        if first_line:
            rec = json.loads(first_line)
            seq = rec.get("anchor_sequence")
            if isinstance(seq, list) and seq:
                return [int(x) for x in seq]

    try:
        anchor_seqs = load_optimal_sequences_from_results(
            results_dir, [benchmark], model_name=model_name
        )
        if benchmark in anchor_seqs:
            return [int(x) for x in anchor_seqs[benchmark]]
    except Exception as exc:  # best-effort fallback
        logger.warning("Anchor lookup from results failed: %s", exc)

    logger.warning(
        "Falling back to identity anchor for %s (num_layers=%d)",
        benchmark,
        num_layers,
    )
    return list(range(num_layers))


# ---------------------------------------------------------------------------
# Local sweep: config generation + reporting
# ---------------------------------------------------------------------------

def _build_local_sweep_configs(
    n: int,
    benchmark: str,
) -> List[Dict[str, Any]]:
    """Build *n* diverse HP configs for a local (no-wandb) sweep.

    Trial 0 uses the known-good preset for *benchmark* from the
    marginalization evaluation module.  Remaining trials systematically
    vary architecture, regularisation, and target-construction parameters.
    """
    from evaluation.eval_fine_routing_marginalization import PRESETS

    preset_key = f"{benchmark}_mcts"
    base = dict(PRESETS.get(preset_key, PRESETS.get(
        next(iter(PRESETS)), {}),
    ))
    base.setdefault("gating_mode", "gate_network")

    configs: List[Dict[str, Any]] = [
        {**base, "_name": "preset"},
    ]

    variations: List[Dict[str, Any]] = [
        {"_name": "small_router",  "router_h1": 256, "router_h2": 128, "router_h3": 0},
        {"_name": "large_router",  "router_h1": 768, "router_h2": 512, "router_h3": 128},
        {"_name": "low_dropout",   "router_dropout": 0.05, "gate_dropout": 0.05},
        {"_name": "high_dropout",  "router_dropout": 0.35, "gate_dropout": 0.30},
        {"_name": "few_epochs",    "router_epochs": 60,  "gate_epochs": 30},
        {"_name": "many_epochs",   "router_epochs": 250, "gate_epochs": 100},
        {"_name": "high_noop",     "noop_boost": 3.0},
        {"_name": "cold_targets",  "target_temp": 0.15},
        {"_name": "warm_targets",  "target_temp": 1.5},
        {"_name": "hard_targets",  "router_hard_targets": True, "label_smoothing": 0.0},
        {"_name": "high_lr",       "router_lr": 3e-3, "gate_lr": 3e-3},
        {"_name": "low_lr",        "router_lr": 1e-4, "gate_lr": 1e-4},
    ]

    for v in variations:
        if len(configs) >= n:
            break
        cfg_dict = {**base, **v}
        configs.append(cfg_dict)

    return configs[:n]


def _print_sweep_comparison(results: List[Dict], benchmark: str):
    """Print table: greedy-gated vs best marginalization across trials."""
    ok = [r for r in results if "error" not in r]
    if not ok:
        print("No successful trials.")
        return

    has_marg = any("marg_best_strategy" in r for r in ok)

    print(f"\n{'=' * 140}")
    print(f"  SWEEP RESULTS — {benchmark.upper()}  ({len(ok)} trials)")
    print(f"{'=' * 140}")
    hdr = (
        f"{'#':>3}  {'Name':<18}  {'Gating':<18}  {'Anchor':>7}  "
        f"{'Greedy':>7}  {'Δgrdy':>7}  {'Gate%':>6}"
    )
    if has_marg:
        hdr += f"  {'BestMarg':>20}  {'MargAcc':>8}  {'Δmarg':>7}  {'M>G':>4}"
    print(hdr)
    print("-" * 140)

    marg_wins = 0
    greedy_deltas = []
    marg_deltas = []
    for r in ok:
        i = r.get("trial", "?")
        name = r.get("_name", r.get("config", {}).get("_name", ""))
        gm = r.get("config", {}).get("gating_mode", "?")
        anc = r.get("anchor_accuracy", 0)
        rte = r.get("routed_accuracy", 0)
        dg = r.get("unconditional_gain", 0)
        gr = r.get("gate_open_rate", 0) * 100
        greedy_deltas.append(dg)

        line = (
            f"{i:>3}  {name:<18}  {gm:<18}  {anc:>7.4f}  "
            f"{rte:>7.4f}  {dg:>+7.4f}  {gr:>5.1f}%"
        )
        if has_marg:
            ms = r.get("marg_best_strategy", "N/A")
            ma = r.get("marg_best_accuracy", 0)
            md = r.get("marg_best_acc_delta", 0)
            bg = r.get("marg_beats_greedy", 0)
            marg_deltas.append(md)
            if bg:
                marg_wins += 1
            line += f"  {ms:>20}  {ma:>8.4f}  {md:>+7.4f}  {'Y' if bg else 'N':>4}"
        print(line)

    print("=" * 140)
    import numpy as _np
    print(f"  Greedy Δ vs anchor:  mean={_np.mean(greedy_deltas):+.4f}  "
          f"min={min(greedy_deltas):+.4f}  max={max(greedy_deltas):+.4f}")
    if has_marg and marg_deltas:
        print(f"  Marg   Δ vs anchor:  mean={_np.mean(marg_deltas):+.4f}  "
              f"min={min(marg_deltas):+.4f}  max={max(marg_deltas):+.4f}")
        print(f"  Marginalization beats greedy: {marg_wins}/{len(ok)} "
              f"({100 * marg_wins / len(ok):.0f}%)")
    print("=" * 140)


# ---------------------------------------------------------------------------
# Single-trial execution (shared by W&B agent and --fixed_config_json)
# ---------------------------------------------------------------------------

def _execute_fine_routing_trial(
    c: Any,
    args: argparse.Namespace,
    cfg: FineRoutingConfig,
    wrapper: FlexibleModelWrapper,
    device: torch.device,
    is_mcts: bool,
    mcts_records: Optional[List[Dict]],
    sequence_catalog_full: List[List[int]],
    sequence_catalog_reduced: List[List[int]],
    mcts_seq_to_idx_full: Dict[tuple, int],
    mcts_seq_to_idx_reduced: Dict[tuple, int],
    num_classes: int,
    router_targets_base: List[torch.Tensor],
    sequence_catalog: List[List[int]],
    residuals: torch.Tensor,
    gate_labels: List[int],
    best_deltas: List[float],
    val_samples: List[Dict],
    anchor_seq: List[int],
    d_model: int,
) -> Tuple[Dict[str, Any], float, float]:
    """Train gate/router and evaluate. Returns (metrics, train_time_s, total_time_s)."""
    t0 = time.time()
    gating_mode = getattr(c, "gating_mode", "gate_network")

    use_bs = getattr(c, "use_best_seq", False)
    if is_mcts and mcts_records is not None:
        if use_bs:
            trial_catalog = sequence_catalog_reduced
            trial_seq_to_idx = mcts_seq_to_idx_reduced
            trial_num_classes = len(trial_catalog)
        else:
            trial_catalog = sequence_catalog_full
            trial_seq_to_idx = mcts_seq_to_idx_full
            trial_num_classes = num_classes

        nb = getattr(c, "noop_boost", 0.0)
        t_temp = getattr(c, "target_temp", 1.0)
        router_targets = rebuild_targets_for_trial(
            mcts_records, trial_seq_to_idx, trial_num_classes,
            noop_boost=nb,
            target_temp=t_temp,
            use_best_seq=use_bs,
        )
    else:
        trial_catalog = sequence_catalog
        trial_num_classes = num_classes
        router_targets = router_targets_base

    gate = None
    dg = None
    if gating_mode == "gate_network":
        gate = train_gate_inline(
            residuals, gate_labels, d_model,
            hidden_dim=getattr(c, "gate_hidden_dim", 256),
            gate_dropout=getattr(c, "gate_dropout", 0.1),
            lr=getattr(c, "gate_lr", 1e-3),
            epochs=getattr(c, "gate_epochs", 60),
            batch_size=args.batch_size,
            recall_boost=getattr(c, "recall_boost", 1.5),
            device=device,
        )
    elif gating_mode == "delta_gate":
        dg = train_delta_gate_inline(
            residuals, best_deltas, d_model,
            hidden_dim=getattr(c, "gate_hidden_dim", 256),
            gate_dropout=getattr(c, "gate_dropout", 0.1),
            lr=getattr(c, "gate_lr", 1e-3),
            epochs=getattr(c, "gate_epochs", 60),
            batch_size=args.batch_size,
            fp_weight=getattr(c, "recall_boost", 2.0),
            device=device,
        )

    h3 = getattr(c, "router_h3", 0)
    hidden_dims = [c.router_h1, c.router_h2]
    if h3 > 0:
        hidden_dims.append(h3)

    train_all = gating_mode != "gate_network"
    router = train_router_inline(
        residuals, gate_labels, router_targets,
        d_model, trial_num_classes,
        hidden_dims=hidden_dims,
        router_dropout=c.router_dropout,
        lr=c.router_lr,
        epochs=c.router_epochs,
        batch_size=args.batch_size,
        gate_positives_only=(
            not train_all and getattr(c, "router_gate_pos_only", False)
        ),
        device=device,
        hard_targets=getattr(c, "router_hard_targets", False),
        label_smoothing=getattr(c, "label_smoothing", 0.0),
        weight_decay=getattr(c, "weight_decay", 0.01),
        inverse_freq_class_weights=getattr(c, "inverse_freq_class_weights", True),
    )

    train_time = time.time() - t0

    metrics = evaluate(
        wrapper, gate, router,
        gamma=getattr(c, "gamma", 0.5),
        anchor_seq=anchor_seq,
        sequence_catalog=trial_catalog,
        samples=val_samples,
        benchmark=args.benchmark,
        model_name=args.model_name,
        pivot_layer=cfg.pivot_layer,
        gate_device=device,
        gating_mode=gating_mode,
        confidence_threshold=getattr(c, "confidence_threshold", 0.0),
        delta_gate=dg,
        delta_margin=getattr(c, "delta_margin", 0.0),
    )

    if getattr(args, "marginalize", False):
        from evaluation.eval_fine_routing_marginalization import run_evaluation as _run_marg

        _ans_map = {
            "boolq": ["True", "False"],
            "commonsenseqa": ["A", "B", "C", "D", "E"],
            "winogrande": ["A", "B"],
            "arc_easy": ["A", "B", "C", "D"],
            "mmlu_all": ["A", "B", "C", "D"],
        }
        ans_opts = _ans_map.get(args.benchmark, ["A", "B", "C", "D", "E"])
        top_k = getattr(args, "top_k_values", [4, 8])

        marg = _run_marg(
            wrapper=wrapper, router=router, anchor_seq=anchor_seq,
            sequence_catalog=trial_catalog, eval_samples=val_samples,
            benchmark=args.benchmark, model_name=args.model_name,
            pivot_layer=cfg.pivot_layer, gate_device=device,
            top_k_values=top_k, answer_options=ans_opts,
            gate=gate, delta_gate=dg, gating_mode=gating_mode,
            gamma=getattr(c, "gamma", 0.5),
            confidence_threshold=getattr(c, "confidence_threshold", 0.3),
        )

        greedy_strat = marg["strategies"].get("greedy", {})
        metrics["marg_greedy_acc"] = greedy_strat.get("accuracy", 0)
        metrics["marg_anchor_acc"] = marg["strategies"].get("anchor", {}).get("accuracy", 0)
        metrics["marg_gate_open_rate"] = marg.get("gate_open_rate", 0.0)

        non_trivial = {
            k: v for k, v in marg["strategies"].items()
            if k not in ("anchor", "greedy") and "oracle" not in k
        }
        for sname, svals in marg["strategies"].items():
            sk = sname.replace("-", "_")
            metrics[f"marg_{sk}_acc"] = svals["accuracy"]
            metrics[f"marg_{sk}_acc_delta"] = svals["accuracy_delta_vs_anchor"]
            metrics[f"marg_{sk}_lp_delta"] = svals["logprob_delta_vs_anchor"]

        if non_trivial:
            best_name, best_vals = max(
                non_trivial.items(), key=lambda x: x[1]["accuracy"],
            )
            metrics["marg_best_strategy"] = best_name
            metrics["marg_best_accuracy"] = best_vals["accuracy"]
            metrics["marg_best_acc_delta"] = best_vals["accuracy_delta_vs_anchor"]
            greedy_acc_val = greedy_strat.get("accuracy", 0)
            metrics["marg_beats_greedy"] = int(best_vals["accuracy"] > greedy_acc_val)

    total_time = time.time() - t0
    return metrics, train_time, total_time


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Bayesian HP sweep for fine routing (wandb)"
    )
    p.add_argument("--data_dir", type=str, required=True,
                   help="Dir with {bench}_pivot_residuals.pt and {bench}.jsonl")
    p.add_argument("--benchmark", type=str, required=True)
    p.add_argument("--results_dir", type=str, required=True,
                   help="Predictions dir with MCTS snapshots for anchor seqs")
    p.add_argument("--model_name", type=str,
                   default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument(
        "--model_is_finetuned",
        action="store_true",
        help="Load benchmark-specific FT LoRA adapter and use identity anchor.",
    )
    p.add_argument(
        "--ft_results_dir",
        type=str,
        default=None,
        help="FT results root with {benchmark}/seed_{seed}/{arm}/checkpoints.",
    )
    p.add_argument(
        "--ft_adapter_path",
        type=str,
        default=None,
        help="Direct adapter path; overrides --ft_results_dir.",
    )
    p.add_argument("--ft_seed", type=int, default=41)
    p.add_argument("--ft_source_arm", type=str, default="ft_only")
    p.add_argument("--eval_questions", type=int, default=200,
                   help="Number of validation questions to evaluate per trial")
    p.add_argument("--eval_skip", type=int, default=0,
                   help="Skip first N validation samples (to exclude inflated subset)")
    p.add_argument("--count", type=int, default=60,
                   help="Number of sweep trials")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--project", type=str, default="fine-routing-sweep")
    p.add_argument("--sweep_id", type=str, default=None,
                   help="Resume an existing sweep instead of creating a new one")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument(
        "--large_search_space",
        action="store_true",
        help="Use SWEEP_CONFIG_LARGE (wider arch/epoch/LR ranges) for ~10k+ sample runs",
    )
    p.add_argument(
        "--fixed_config_json",
        type=str,
        default=None,
        help="Run a single trial with hyperparameters from this JSON (no Bayesian sweep). "
        "Use with --eval_questions for full-val replication.",
    )
    p.add_argument(
        "--no_wandb",
        action="store_true",
        help="With --fixed_config_json: skip W&B logging (stdout only).",
    )
    p.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="With --fixed_config_json: W&B run name (default: fixed-config-validate).",
    )
    p.add_argument(
        "--marginalize",
        action="store_true",
        help="After greedy eval, also run logprob-based marginalization strategies.",
    )
    p.add_argument(
        "--top_k_values",
        type=int,
        nargs="+",
        default=[4, 8],
        help="Top-K values for marginalization (default: 4 8).",
    )
    p.add_argument(
        "--local_sweep",
        type=int,
        default=0,
        help="Run N diverse trials locally (no wandb). Model loaded once.",
    )
    p.add_argument(
        "--local_sweep_configs_json",
        type=str,
        default=None,
        help=(
            "JSON array of hyperparameter dicts (optional _name per row). "
            "When set, runs these trials instead of --local_sweep presets; "
            "model still loads once. Ignores --local_sweep count."
        ),
    )
    p.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output path for local sweep results JSON.",
    )
    p.add_argument(
        "--baseline_mode",
        type=str,
        default="default_modules",
        choices=["default_modules", "data_anchor"],
        help="Baseline sequence used for anchor/routed gain.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Config + LLM -------------------------------------------------------
    logger.info("Loading LLM %s ...", args.model_name)
    cfg = FineRoutingConfig(model_name=args.model_name, results_dir=args.results_dir)
    data_config_path = os.path.join(args.data_dir, "config.json")
    if os.path.isfile(data_config_path):
        with open(data_config_path) as f:
            data_cfg = json.load(f)
        for key in ("max_local_edits", "swap_radius", "editable_start"):
            if key in data_cfg:
                setattr(cfg, key, data_cfg[key])
                logger.info("  %s=%s (from data config)", key, data_cfg[key])
    if args.model_is_finetuned:
        if args.ft_adapter_path:
            adapter_path = args.ft_adapter_path
        else:
            if not args.ft_results_dir:
                raise ValueError(
                    "FT mode needs --ft_results_dir or --ft_adapter_path."
                )
            adapter_path = find_adapter_path(
                args.ft_results_dir,
                args.benchmark,
                args.ft_seed,
                arm=args.ft_source_arm,
            )
        if not adapter_path:
            raise ValueError(
                f"No FT adapter found for {args.benchmark} "
                f"(seed={args.ft_seed}, arm={args.ft_source_arm})"
            )
        logger.info("Loading FT adapter: %s", adapter_path)
        wrapper = FTFlexibleModelWrapper.from_ft_adapter(
            args.model_name, adapter_path, rank=0,
        )
    else:
        wrapper = FlexibleModelWrapper(args.model_name, rank=0)
    if args.baseline_mode == "default_modules":
        anchor_seq = list(range(wrapper.num_layers))
        logger.info("  baseline_mode=default_modules (identity sequence)")
    else:
        anchor_seq = _resolve_anchor_seq(
            data_dir=args.data_dir,
            benchmark=args.benchmark,
            results_dir=args.results_dir,
            model_name=args.model_name,
            num_layers=wrapper.num_layers,
        )
        logger.info("  baseline_mode=data_anchor")

    logger.info("  %d layers", wrapper.num_layers)
    logger.info("  anchor: %s", anchor_seq)

    # --- Detect MCTS vs enumerated data ------------------------------------
    jsonl_path = os.path.join(args.data_dir, f"{args.benchmark}.jsonl")
    with open(jsonl_path) as f:
        first_rec = json.loads(f.readline())
    is_mcts = first_rec.get("search_mode") == "mcts"

    mcts_records = None
    mcts_seq_to_idx_full = None
    mcts_seq_to_idx_reduced = None
    sequence_catalog_full = None
    sequence_catalog_reduced = None
    if is_mcts:
        logger.info("Loading MCTS training data (sequence-catalog mode) ...")
        (residuals, gate_labels, router_targets_base,
         sequence_catalog_full, mcts_seq_to_idx_full,
         sequence_catalog_reduced, mcts_seq_to_idx_reduced,
         mcts_records) = (
            load_bench_data_mcts(args.data_dir, args.benchmark, anchor_seq)
        )
        sequence_catalog = sequence_catalog_full
        num_classes = len(sequence_catalog)
    else:
        logger.info("Loading enumerated training data ...")
        residuals, gate_labels, router_targets_base = load_bench_data_enumerated(
            args.data_dir, args.benchmark
        )
        num_classes = router_targets_base[0].shape[0]
        deviations = enumerate_deviations(
            anchor_seq,
            editable_start=cfg.editable_start,
            num_layers=wrapper.num_layers,
            swap_radius=cfg.swap_radius,
            max_edits=cfg.max_local_edits,
        )
        sequence_catalog = [apply_deviation(anchor_seq, d) for d in deviations]

    d_model = residuals.shape[1]

    best_deltas: List[float] = []
    if is_mcts and mcts_records is not None:
        best_deltas = [float(r.get("best_delta", 0.0)) for r in mcts_records]
    else:
        best_deltas = [float(gl) for gl in gate_labels]

    logger.info(
        "  %d samples, d_model=%d, num_classes=%d, gate+=%d",
        len(gate_labels), d_model, num_classes, sum(gate_labels),
    )

    # --- Load validation samples (once) ------------------------------------
    is_instruct = get_is_instruct(args.model_name)
    val_samples = prepare_arc_data(
        args.benchmark, is_instruct=is_instruct, split="validation"
    )
    val_samples = val_samples[args.eval_skip :]
    if args.eval_questions > 0:
        val_samples = val_samples[: args.eval_questions]
    logger.info(
        "  %d validation samples (skip=%d, eval_questions=%s)",
        len(val_samples),
        args.eval_skip,
        args.eval_questions if args.eval_questions > 0 else "all",
    )

    # --- Local sweep (no wandb, model loaded once) -------------------------
    if args.local_sweep_configs_json:
        with open(args.local_sweep_configs_json) as f:
            loaded = json.load(f)
        if not isinstance(loaded, list):
            raise ValueError("--local_sweep_configs_json must contain a JSON array")
        sweep_cfgs = [dict(x) for x in loaded]
        logger.info(
            "Loaded %d configs from %s",
            len(sweep_cfgs),
            args.local_sweep_configs_json,
        )
    elif args.local_sweep > 0:
        sweep_cfgs = _build_local_sweep_configs(args.local_sweep, args.benchmark)
    else:
        sweep_cfgs = []

    if sweep_cfgs:
        all_results: List[Dict[str, Any]] = []
        out_path = args.output_json or f"results/local_sweep_{args.benchmark}.json"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        def _flush_local_sweep_results() -> None:
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)

        for ti, raw_cfg in enumerate(sweep_cfgs):
            trial_name = raw_cfg.pop("_name", f"trial_{ti}")
            logger.info(
                "\n" + "=" * 80 +
                "\n  LOCAL TRIAL %d/%d  [%s]\n" + "=" * 80,
                ti + 1, len(sweep_cfgs), trial_name,
            )
            c = types.SimpleNamespace(**raw_cfg)
            try:
                metrics, train_time, total_time = _execute_fine_routing_trial(
                    c, args, cfg, wrapper, device,
                    is_mcts, mcts_records,
                    sequence_catalog_full, sequence_catalog_reduced,
                    mcts_seq_to_idx_full, mcts_seq_to_idx_reduced,
                    num_classes, router_targets_base, sequence_catalog,
                    residuals, gate_labels, best_deltas,
                    val_samples, anchor_seq, d_model,
                )
                result = {
                    "trial": ti, "_name": trial_name,
                    "config": raw_cfg,
                    **metrics,
                    "train_time_s": train_time,
                    "total_time_s": total_time,
                }
                all_results.append(result)
                _flush_local_sweep_results()
                logger.info(
                    "  anchor=%.4f  routed=%.4f  gate=%.1f%%  Δ=%+.4f  (%.0fs)  → %s",
                    metrics["anchor_accuracy"], metrics["routed_accuracy"],
                    100 * metrics["gate_open_rate"],
                    metrics["unconditional_gain"], total_time,
                    out_path,
                )
                if "marg_best_strategy" in metrics:
                    logger.info(
                        "  marg: best=%s  acc=%.4f  Δ=%+.4f  beats_greedy=%s",
                        metrics["marg_best_strategy"],
                        metrics["marg_best_accuracy"],
                        metrics["marg_best_acc_delta"],
                        bool(metrics.get("marg_beats_greedy")),
                    )
            except Exception as e:
                logger.error("  TRIAL FAILED: %s", e, exc_info=True)
                all_results.append({
                    "trial": ti, "_name": trial_name,
                    "config": raw_cfg, "error": str(e),
                })
                _flush_local_sweep_results()

        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Results → %s", out_path)

        _print_sweep_comparison(all_results, args.benchmark)
        return

    # --- Single fixed-config run (e.g. full-val replication) ---------------
    if args.fixed_config_json:
        with open(args.fixed_config_json) as f:
            raw_cfg = json.load(f)
        c = types.SimpleNamespace(**raw_cfg)
        metrics, train_time, total_time = _execute_fine_routing_trial(
            c, args, cfg, wrapper, device,
            is_mcts, mcts_records,
            sequence_catalog_full, sequence_catalog_reduced,
            mcts_seq_to_idx_full, mcts_seq_to_idx_reduced,
            num_classes, router_targets_base, sequence_catalog,
            residuals, gate_labels, best_deltas, val_samples, anchor_seq, d_model,
        )
        use_bs = getattr(c, "use_best_seq", False)
        trial_nc = (
            len(sequence_catalog_reduced)
            if (is_mcts and mcts_records is not None and use_bs)
            else (len(sequence_catalog_full) if is_mcts else num_classes)
        )
        gating_mode = getattr(c, "gating_mode", "gate_network")
        logger.info(
            "Fixed-config done: anchor=%.4f  routed=%.4f  gate_open=%.1f%%  "
            "uncond_gain=%+.4f  helped=%d  hurt=%d  net=%d  n_val=%d  mode=%s  |C|=%d  (%.0fs)",
            metrics["anchor_accuracy"],
            metrics["routed_accuracy"],
            100 * metrics["gate_open_rate"],
            metrics["unconditional_gain"],
            metrics["helped_when_opened"],
            metrics["hurt_when_opened"],
            metrics["net_helped"],
            len(val_samples),
            gating_mode,
            trial_nc,
            total_time,
        )
        if not args.no_wandb:
            run = wandb.init(
                project=args.project,
                name=args.run_name or "fixed-config-validate",
                config=raw_cfg,
            )
            wandb.log({
                **metrics,
                "gating_mode": gating_mode,
                "train_time_s": train_time,
                "eval_time_s": total_time - train_time,
                "total_time_s": total_time,
                "num_classes": trial_nc,
                "n_val": len(val_samples),
            })
            run.finish()
        logger.info("Fixed-config run complete.")
        return

    # --- Sweep trial function ---------------------------------------------
    def trial():
        run = wandb.init()
        c = wandb.config

        metrics, train_time, total_time = _execute_fine_routing_trial(
            c, args, cfg, wrapper, device,
            is_mcts, mcts_records,
            sequence_catalog_full, sequence_catalog_reduced,
            mcts_seq_to_idx_full, mcts_seq_to_idx_reduced,
            num_classes, router_targets_base, sequence_catalog,
            residuals, gate_labels, best_deltas, val_samples, anchor_seq, d_model,
        )
        gating_mode = getattr(c, "gating_mode", "gate_network")
        use_bs = getattr(c, "use_best_seq", False)
        trial_nc = (
            len(sequence_catalog_reduced)
            if (is_mcts and mcts_records is not None and use_bs)
            else (len(sequence_catalog_full) if is_mcts else num_classes)
        )

        wandb.log({
            **metrics,
            "gating_mode": gating_mode,
            "train_time_s": train_time,
            "eval_time_s": total_time - train_time,
            "total_time_s": total_time,
            "num_classes": trial_nc,
        })

        logger.info(
            "Trial done: anchor=%.3f  routed=%.3f  gate_open=%.1f%%  "
            "uncond_gain=%+.4f  helped=%d  hurt=%d  mode=%s  |C|=%d  (%.0fs)",
            metrics["anchor_accuracy"],
            metrics["routed_accuracy"],
            100 * metrics["gate_open_rate"],
            metrics["unconditional_gain"],
            metrics["helped_when_opened"],
            metrics["hurt_when_opened"],
            gating_mode,
            trial_nc,
            total_time,
        )
        run.finish()

    # --- Launch sweep ------------------------------------------------------
    if args.sweep_id:
        sweep_id = args.sweep_id
        logger.info("Resuming sweep %s", sweep_id)
    else:
        sweep_spec = SWEEP_CONFIG_LARGE if args.large_search_space else SWEEP_CONFIG
        if args.large_search_space:
            logger.info("Using LARGE search space (SWEEP_CONFIG_LARGE)")
        sweep_id = wandb.sweep(
            sweep_spec,
            project=args.project,
            entity=None,
        )
        logger.info("Created sweep %s", sweep_id)

    wandb.agent(sweep_id, function=trial, count=args.count, project=args.project)
    logger.info("Sweep complete.")


if __name__ == "__main__":
    main()
