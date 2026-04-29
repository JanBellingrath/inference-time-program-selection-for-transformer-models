"""Training loop for the unified HPO.

Trains the router and gate/delta-gate inline (no disk I/O), collects detailed
validation summaries and routing-score distributions on the routing-val split.

Adapts the inline training patterns from ``experiments.sweep_fine_routing``
(``train_router_inline``, ``train_gate_inline``, ``train_delta_gate_inline``).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from training.train_fine_router import (
    FineRouter,
    soft_cross_entropy,
    weighted_soft_cross_entropy,
)

from experiments.unified_hpo.model_factory import (
    FlexibleGateMLP,
    build_delta_gate,
    build_gate,
    build_router,
)
from experiments.unified_hpo.target_builder import build_targets

logger = logging.getLogger(__name__)

# Default fixed epoch counts (not searched — see plan §7)
DEFAULT_ROUTER_EPOCHS = 100
DEFAULT_GATE_EPOCHS = 60
DEFAULT_BATCH_SIZE = 64
#TODO the epochs might be way to long though.. is there nothing adaptive here? Need to add early stopping.

# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    """Output of ``train_and_summarize``."""

    router: FineRouter
    gate: Optional[FlexibleGateMLP] = None
    delta_gate: Optional[FlexibleGateMLP] = None

    # Validation summaries
    router_val_loss: float = float("inf")
    gate_val_loss: float = float("inf")
    delta_gate_val_loss: float = float("inf")

    # Diagnostics computed on routing-val
    predicted_noop_rate: float = 0.0
    router_entropy_mean: float = 0.0
    router_entropy_std: float = 0.0
    class_histogram: Optional[Dict[int, int]] = None

    # Score distribution on routing-val (populated after training)
    score_mean: float = 0.0
    score_std: float = 0.0
    score_min: float = 0.0
    score_max: float = 0.0
    score_quantiles: Dict[str, float] = field(default_factory=dict)

    # Per-sample arrays on routing-val (for calibration)
    val_routing_scores: Optional[torch.Tensor] = None
    val_router_preds: Optional[torch.Tensor] = None
    val_residuals: Optional[torch.Tensor] = None
    val_indices: Optional[List[int]] = None

    training_time_s: float = 0.0
    num_train_samples: int = 0
    num_val_samples: int = 0

    # Actual epoch counts used (Hyperband scales these; defaults match base constants)
    router_epochs_used: int = DEFAULT_ROUTER_EPOCHS
    gate_epochs_used: int = DEFAULT_GATE_EPOCHS


# ---------------------------------------------------------------------------
# Internal training helpers
# ---------------------------------------------------------------------------

def _train_router_model(
    router: FineRouter,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_size: int,
    loss_fn,
    lr: float,
    weight_decay: float,
    epochs: int,
    device: torch.device,
    epoch_val_callback: Optional[Callable[[int, float], None]] = None,
) -> Tuple[FineRouter, float]:
    """Train router for ``epochs``; keep weights with best val loss.  Returns (model, best_val_loss)."""
    optimizer = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        router.train()
        for x, y_b in train_loader:
            x, y_b = x.to(device), y_b.to(device)
            loss = loss_fn(router(x), y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        router.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y_b in val_loader:
                x, y_b = x.to(device), y_b.to(device)
                val_loss += loss_fn(router(x), y_b).item() * x.size(0)
        val_loss /= max(val_size, 1)
        scheduler.step()
        if epoch_val_callback is not None:
            epoch_val_callback(epoch, float(val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in router.state_dict().items()}

    if best_state is not None:
        router.load_state_dict(best_state)
    router.eval()
    return router, best_val_loss


def _train_gate_model(
    gate: FlexibleGateMLP,
    residuals: torch.Tensor,
    labels: torch.Tensor,
    recall_boost: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    device: torch.device,
    seed: int = 42,
) -> Tuple[FlexibleGateMLP, float]:
    """Train binary gate.  Returns (model, best_val_loss)."""
    torch.manual_seed(seed)
    n_pos = int(labels.sum().item())
    n_neg = len(labels) - n_pos
    pw = (n_neg / max(n_pos, 1)) * recall_boost
    pos_weight = torch.tensor([pw], device=device)

    ds = TensorDataset(residuals, labels)
    val_size = max(1, int(len(ds) * 0.15))
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [len(ds) - val_size, val_size], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    gate = gate.to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        gate.train()
        for x, y_b in train_loader:
            x, y_b = x.to(device), y_b.to(device)
            loss = F.binary_cross_entropy_with_logits(
                gate(x), y_b, pos_weight=pos_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        gate.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y_b in val_loader:
                x, y_b = x.to(device), y_b.to(device)
                val_loss += F.binary_cross_entropy_with_logits(
                    gate(x), y_b, pos_weight=pos_weight,
                ).item() * x.size(0)
        val_loss /= max(val_size, 1)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in gate.state_dict().items()}

    if best_state is not None:
        gate.load_state_dict(best_state)
    gate.eval()
    return gate, best_val_loss


def _train_delta_gate_model(
    delta_gate: FlexibleGateMLP,
    residuals: torch.Tensor,
    best_deltas: torch.Tensor,
    fp_weight: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    device: torch.device,
    seed: int = 42,
) -> Tuple[FlexibleGateMLP, float]:
    """Train regression delta-gate.  Returns (model, best_val_loss)."""
    torch.manual_seed(seed)

    ds = TensorDataset(residuals, best_deltas)
    val_size = max(1, int(len(ds) * 0.15))
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [len(ds) - val_size, val_size], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    delta_gate = delta_gate.to(device)
    optimizer = torch.optim.AdamW(delta_gate.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        delta_gate.train()
        for x, y_b in train_loader:
            x, y_b = x.to(device), y_b.to(device)
            pred = delta_gate(x)
            residual = pred - y_b
            base_loss = F.smooth_l1_loss(pred, y_b, reduction="none")
            weight = torch.where(residual > 0, fp_weight, 1.0)
            loss = (base_loss * weight).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        delta_gate.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y_b in val_loader:
                x, y_b = x.to(device), y_b.to(device)
                pred = delta_gate(x)
                residual = pred - y_b
                base_loss = F.smooth_l1_loss(pred, y_b, reduction="none")
                weight = torch.where(residual > 0, fp_weight, 1.0)
                val_loss += (base_loss * weight).mean().item() * x.size(0)
        val_loss /= max(val_size, 1)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in delta_gate.state_dict().items()}

    if best_state is not None:
        delta_gate.load_state_dict(best_state)
    delta_gate.eval()
    return delta_gate, best_val_loss


# ---------------------------------------------------------------------------
# Validation diagnostics
# ---------------------------------------------------------------------------

def _compute_val_diagnostics(
    router: FineRouter,
    gate: Optional[FlexibleGateMLP],
    delta_gate: Optional[FlexibleGateMLP],
    val_X: torch.Tensor,
    gating_mode: str,
    device: torch.device,
    num_classes: int,
) -> Dict[str, Any]:
    """Compute diagnostics on the routing-val split."""
    router.eval()
    diagnostics: Dict[str, Any] = {}

    with torch.no_grad():
        logits = router(val_X.to(device))
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

    # Predicted noop rate (fraction predicting class 0)
    noop_mask = preds == 0
    diagnostics["predicted_noop_rate"] = float(noop_mask.float().mean().item())

    # Router entropy
    entropy = -(probs * probs.clamp(min=1e-12).log()).sum(dim=-1)
    diagnostics["router_entropy_mean"] = float(entropy.mean().item())
    diagnostics["router_entropy_std"] = float(entropy.std().item())

    # Class histogram
    histogram = {}
    for cls_idx in range(num_classes):
        count = int((preds == cls_idx).sum().item())
        if count > 0:
            histogram[cls_idx] = count
    diagnostics["class_histogram"] = histogram

    # Top1-minus-noop margin
    noop_prob = probs[:, 0]
    top_non_noop = probs[:, 1:].max(dim=-1).values if num_classes > 1 else torch.zeros_like(noop_prob)
    margin = top_non_noop - noop_prob
    diagnostics["mean_top1_minus_noop_margin"] = float(margin.mean().item())

    # Routing scores (depends on gating mode)
    if gating_mode == "gate_network" and gate is not None:
        gate.eval()
        with torch.no_grad():
            scores = torch.sigmoid(gate(val_X.to(device))).cpu()
    elif gating_mode == "delta_gate" and delta_gate is not None:
        delta_gate.eval()
        with torch.no_grad():
            scores = delta_gate(val_X.to(device)).cpu()
    else:
        scores = (1.0 - probs[:, 0]).cpu()

    diagnostics["scores"] = scores
    diagnostics["router_preds"] = preds.cpu()

    # Score distribution
    diagnostics["score_mean"] = float(scores.mean().item())
    diagnostics["score_std"] = float(scores.std().item())
    diagnostics["score_min"] = float(scores.min().item())
    diagnostics["score_max"] = float(scores.max().item())
    quantile_levels = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]
    q_tensor = torch.tensor(quantile_levels)
    q_vals = torch.quantile(scores.float(), q_tensor)
    diagnostics["score_quantiles"] = {
        f"q{q:.2f}": float(v.item()) for q, v in zip(quantile_levels, q_vals)
    }

    return diagnostics


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train_and_summarize(
    config: Dict,
    residuals: torch.Tensor,
    gate_labels: List[int],
    records: List[Dict],
    seq_to_idx: Dict[tuple, int],
    num_classes: int,
    best_deltas: List[float],
    d_model: int,
    device: torch.device,
    seed: int = 42,
    router_epochs: int = DEFAULT_ROUTER_EPOCHS,
    gate_epochs: int = DEFAULT_GATE_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    val_fraction: float = 0.15,
    router_epoch_val_callback: Optional[Callable[[int, float], None]] = None,
) -> TrainResult:
    """Train router (and gate/delta-gate if needed) for one HPO configuration.

    Performs a deterministic routing-train / routing-val split, trains all
    required models, and collects comprehensive validation diagnostics.

    Returns a ``TrainResult`` with trained models, validation summaries, and
    per-sample routing-val arrays needed for calibration.
    """
    t0 = time.time()
    torch.manual_seed(seed)
    result = TrainResult(router=None)  # type: ignore[arg-type]

    gating_mode = config.get("gating_mode", "gate_network")
    router_loss = config.get("router_loss", "hard_ce")
    router_train_subset = config.get("router_train_subset", "all")

    # ------------------------------------------------------------------
    # Build targets
    # ------------------------------------------------------------------
    targets, hard = build_targets(records, seq_to_idx, num_classes, config)

    # ------------------------------------------------------------------
    # Determine training indices (routing-train subset policy)
    # ------------------------------------------------------------------
    all_indices = list(range(len(gate_labels)))
    if gating_mode == "gate_network" and router_train_subset == "positives_only":
        train_eligible = [i for i in all_indices if gate_labels[i] == 1]
        if not train_eligible:
            train_eligible = all_indices
    else:
        train_eligible = all_indices

    # ------------------------------------------------------------------
    # Build routing-train / routing-val split (deterministic)
    # ------------------------------------------------------------------
    n = len(train_eligible)
    val_size = max(1, int(n * val_fraction))
    train_size = n - val_size

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    train_idx = [train_eligible[i] for i in perm[val_size:]]
    val_idx = [train_eligible[i] for i in perm[:val_size]]

    result.num_train_samples = train_size
    result.num_val_samples = val_size
    result.val_indices = val_idx

    # ------------------------------------------------------------------
    # Prepare router training tensors
    # ------------------------------------------------------------------
    train_X = residuals[train_idx]
    val_X = residuals[val_idx]
    result.val_residuals = val_X

    if hard:
        train_Y = torch.tensor(
            [targets[i].argmax().item() for i in train_idx], dtype=torch.long,
        )
        val_Y = torch.tensor(
            [targets[i].argmax().item() for i in val_idx], dtype=torch.long,
        )

        label_smoothing = float(config.get("label_smoothing", 0.0))
        use_class_weights = config.get("inverse_freq_class_weights", True)

        if use_class_weights:
            class_counts = torch.bincount(train_Y, minlength=num_classes).float().clamp(min=1)
            class_weights = (1.0 / class_counts)
            class_weights = class_weights / class_weights.mean()
            class_weights = class_weights.to(device)
            loss_fn = lambda logits, y: F.cross_entropy(
                logits, y, weight=class_weights, label_smoothing=label_smoothing,
            )
        else:
            loss_fn = lambda logits, y: F.cross_entropy(
                logits, y, label_smoothing=label_smoothing,
            )
    else:
        train_Y = torch.stack([targets[i] for i in train_idx])
        val_Y = torch.stack([targets[i] for i in val_idx])
        use_class_weights = config.get("inverse_freq_class_weights", True)
        if use_class_weights:
            mass = train_Y.sum(dim=0).float().clamp(min=1.0)
            class_weights = (1.0 / mass)
            class_weights = class_weights / class_weights.mean()
            class_weights = class_weights.to(device)

            def loss_fn(logits, y):
                return weighted_soft_cross_entropy(logits, y, class_weights)
        else:
            loss_fn = soft_cross_entropy

    train_ds = TensorDataset(train_X, train_Y)
    val_ds = TensorDataset(val_X, val_Y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # ------------------------------------------------------------------
    # Train router
    # ------------------------------------------------------------------
    router = build_router(d_model, num_classes, config).to(device)
    router_lr = float(config.get("router_lr", 1e-3))
    router_wd = float(config.get("router_weight_decay", 0.01))

    result.router_epochs_used = router_epochs
    result.gate_epochs_used = gate_epochs

    router, router_val_loss = _train_router_model(
        router, train_loader, val_loader, val_size,
        loss_fn, router_lr, router_wd, router_epochs, device,
        epoch_val_callback=router_epoch_val_callback,
    )
    result.router = router
    result.router_val_loss = router_val_loss

    # ------------------------------------------------------------------
    # Train gate or delta-gate (if needed)
    # ------------------------------------------------------------------
    gate_lr = float(config.get("gate_lr", 5e-4))
    gate_wd = float(config.get("gate_weight_decay", 1e-3))
    gate_cost_scale = float(config.get("gate_cost_scale", 1.5))

    all_residuals = residuals
    all_gate_labels = torch.tensor(gate_labels, dtype=torch.float32)
    all_deltas = torch.tensor(best_deltas, dtype=torch.float32)

    if gating_mode == "gate_network":
        gate_model = build_gate(d_model, config)
        gate_model, gate_val_loss = _train_gate_model(
            gate_model, all_residuals, all_gate_labels,
            recall_boost=gate_cost_scale,
            lr=gate_lr, weight_decay=gate_wd,
            epochs=gate_epochs, batch_size=batch_size,
            device=device, seed=seed,
        )
        result.gate = gate_model
        result.gate_val_loss = gate_val_loss

    elif gating_mode == "delta_gate":
        dg_model = build_delta_gate(d_model, config)
        dg_model, dg_val_loss = _train_delta_gate_model(
            dg_model, all_residuals, all_deltas,
            fp_weight=gate_cost_scale,
            lr=gate_lr, weight_decay=gate_wd,
            epochs=gate_epochs, batch_size=batch_size,
            device=device, seed=seed,
        )
        result.delta_gate = dg_model
        result.delta_gate_val_loss = dg_val_loss

    # ------------------------------------------------------------------
    # Validation diagnostics on routing-val
    # ------------------------------------------------------------------
    diag = _compute_val_diagnostics(
        router, result.gate, result.delta_gate,
        val_X, gating_mode, device, num_classes,
    )

    result.predicted_noop_rate = diag["predicted_noop_rate"]
    result.router_entropy_mean = diag["router_entropy_mean"]
    result.router_entropy_std = diag["router_entropy_std"]
    result.class_histogram = diag["class_histogram"]
    result.score_mean = diag["score_mean"]
    result.score_std = diag["score_std"]
    result.score_min = diag["score_min"]
    result.score_max = diag["score_max"]
    result.score_quantiles = diag["score_quantiles"]
    result.val_routing_scores = diag["scores"]
    result.val_router_preds = diag["router_preds"]

    result.training_time_s = time.time() - t0
    logger.info(
        "Training complete: router_val_loss=%.4f  gate_val_loss=%.4f  "
        "noop_rate=%.3f  score_mean=%.4f  (%.1fs)",
        result.router_val_loss,
        result.gate_val_loss if result.gate is not None else result.delta_gate_val_loss,
        result.predicted_noop_rate,
        result.score_mean,
        result.training_time_s,
    )

    return result
