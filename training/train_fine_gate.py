#!/usr/bin/env python3
"""Train per-benchmark binary fine-routing gates.

import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

One gate per benchmark/domain.  Each gate answers:
"Is any local deviation worth trying for this question?"

Input : pivot residual vector  [d_model]
Output: scalar probability in [0, 1]
Label : y_gate(q) = 1[max_delta delta(q, delta) > tau]

Loss  : weighted BCE  -w1*y*log(g) - w0*(1-y)*log(1-g)

Checkpoints are saved as ``{output_dir}/gate_best_{benchmark}.pt``.

Usage
-----
    python train_fine_gate.py \
        --data_dir fine_routing_data \
        --output_dir checkpoints/fine_gate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

import sys as _sys2
from pathlib import Path as _Path2
_sys2.path.insert(0, str(_Path2(__file__).resolve().parent.parent))

from routers.residual_compressors import (
    CompressorConfig,
    CompressedGate,
    build_compressor,
    pad_sequences,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GateDataset(Dataset):
    """Loads pivot residuals + gate labels from the generated fine-routing data."""

    def __init__(self, data_dir: str, benchmarks: Optional[List[str]] = None):
        self.residuals: List[torch.Tensor] = []
        self.labels: List[int] = []

        if benchmarks is None:
            benchmarks = self._discover_benchmarks(data_dir)

        for bench in benchmarks:
            pt_path = os.path.join(data_dir, f"{bench}_pivot_residuals.pt")
            jsonl_path = os.path.join(data_dir, f"{bench}.jsonl")
            if not os.path.isfile(pt_path) or not os.path.isfile(jsonl_path):
                logger.warning("Missing data for %s, skipping", bench)
                continue

            residuals = torch.load(pt_path, map_location="cpu", weights_only=True).float()
            with open(jsonl_path) as f:
                records = [json.loads(line) for line in f]

            if len(residuals) != len(records):
                raise ValueError(
                    f"{bench}: residual count {len(residuals)} != record count {len(records)}"
                )
            for i, rec in enumerate(records):
                self.residuals.append(residuals[i])
                self.labels.append(rec["gate_label"])

        logger.info(
            "GateDataset: %d samples, %d positive (%.1f%%)",
            len(self.labels),
            sum(self.labels),
            100 * sum(self.labels) / max(len(self.labels), 1),
        )

    @staticmethod
    def _discover_benchmarks(data_dir: str) -> List[str]:
        benchmarks = []
        for f in sorted(os.listdir(data_dir)):
            if f.endswith(".jsonl"):
                benchmarks.append(f.replace(".jsonl", ""))
        return benchmarks

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.residuals[idx], self.labels[idx]


class FullSequenceGateDataset(Dataset):
    """Full-sequence residuals + gate labels for attention compressors."""

    def __init__(self, data_dir: str, benchmarks: Optional[List[str]] = None):
        self.residuals: List[torch.Tensor] = []
        self.labels: List[int] = []

        if benchmarks is None:
            benchmarks = GateDataset._discover_benchmarks(data_dir)

        for bench in benchmarks:
            full_pt = os.path.join(data_dir, f"{bench}_full_residuals.pt")
            jsonl_path = os.path.join(data_dir, f"{bench}.jsonl")
            if not os.path.isfile(full_pt) or not os.path.isfile(jsonl_path):
                logger.warning("Missing full-seq data for %s, skipping", bench)
                continue

            data = torch.load(full_pt, map_location="cpu", weights_only=False)
            full_residuals = data["residuals"]
            with open(jsonl_path) as f:
                records = [json.loads(line) for line in f]

            n = min(len(full_residuals), len(records))
            for i in range(n):
                self.residuals.append(full_residuals[i].float())
                self.labels.append(records[i]["gate_label"])

        logger.info(
            "FullSequenceGateDataset: %d samples, %d positive (%.1f%%)",
            len(self.labels), sum(self.labels),
            100 * sum(self.labels) / max(len(self.labels), 1),
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.residuals[idx], self.labels[idx]


def collate_full_sequence_gate(batch):
    residuals = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    padded, mask = pad_sequences(residuals)
    return padded, mask, labels


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FineGate(nn.Module):
    """Binary gate: pivot residual -> [0, 1] probability."""

    def __init__(self, d_model: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, d_model] -> [B] logits (apply sigmoid for probability)."""
        return self.net(x).squeeze(-1)


class DeltaGate(nn.Module):
    """Regression gate: predict the expected improvement (delta) from routing.

    Instead of binary "route / don't route", this gate outputs a scalar
    estimate of ``E[delta]`` — the expected change in score if the best
    alternative route is used.  At inference, routing happens only when
    ``predicted_delta > margin`` where ``margin > 0`` provides a safety
    buffer against the asymmetric risk of routing errors.
    """

    def __init__(self, d_model: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, d_model] -> [B] predicted delta (raw, no activation)."""
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_gate(
    data_dir: str,
    output_dir: str,
    benchmark: str,
    hidden_dim: int = 256,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 64,
    w1: float = 0.0,
    w0: float = 1.0,
    recall_boost: float = 1.2,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Optional[str]:
    """Train a gate for a single benchmark. Returns checkpoint path or None.

    When ``w1 <= 0`` (default), pos_weight is auto-computed as
    ``(n_neg / n_pos) * recall_boost`` so the loss is balanced regardless
    of the benchmark's positive rate.
    """
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = GateDataset(data_dir, benchmarks=[benchmark])
    if len(ds) == 0:
        logger.error("[%s] No data loaded. Skipping.", benchmark)
        return None

    d_model = ds.residuals[0].shape[0]
    n_pos = sum(ds.labels)
    n_neg = len(ds) - n_pos
    logger.info("[%s] d_model=%d  samples=%d  positive=%d (%.1f%%)",
                benchmark, d_model, len(ds), n_pos,
                100 * n_pos / max(len(ds), 1))

    val_size = max(1, int(len(ds) * val_fraction))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = FineGate(d_model, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if w1 > 0:
        pos_weight = torch.tensor([w1 / w0], device=device)
    else:
        pw = (n_neg / max(n_pos, 1)) * recall_boost
        pos_weight = torch.tensor([pw], device=device)
    logger.info("[%s] pos_weight=%.3f", benchmark, pos_weight.item())

    ckpt_path = os.path.join(output_dir, f"gate_best_{benchmark}.pt")
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.float().to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=pos_weight
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= train_size

        # --- val ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.float().to(device)
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(
                    logits, y, pos_weight=pos_weight
                )
                val_loss += loss.item() * x.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).long()
                labels = y.long()
                correct += (preds == labels).sum().item()
                total += x.size(0)
                tp += ((preds == 1) & (labels == 1)).sum().item()
                fp += ((preds == 1) & (labels == 0)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()
                tn += ((preds == 0) & (labels == 0)).sum().item()
        val_loss /= val_size
        acc = correct / max(total, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "d_model": d_model, "hidden_dim": hidden_dim,
                 "benchmark": benchmark},
                ckpt_path,
            )

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "[%s] Epoch %3d  train=%.4f  val=%.4f  acc=%.3f  "
                "prec=%.3f  rec=%.3f  (best=%d)",
                benchmark, epoch, train_loss, val_loss, acc,
                precision, recall, best_epoch,
            )

    logger.info("[%s] Done. Best epoch=%d  val_loss=%.4f  -> %s",
                benchmark, best_epoch, best_val_loss, ckpt_path)
    return ckpt_path


class DeltaGateDataset(Dataset):
    """Pivot residuals + continuous best_delta targets for regression gating."""

    def __init__(self, data_dir: str, benchmarks: Optional[List[str]] = None):
        self.residuals: List[torch.Tensor] = []
        self.deltas: List[float] = []

        if benchmarks is None:
            benchmarks = GateDataset._discover_benchmarks(data_dir)

        for bench in benchmarks:
            pt_path = os.path.join(data_dir, f"{bench}_pivot_residuals.pt")
            jsonl_path = os.path.join(data_dir, f"{bench}.jsonl")
            if not os.path.isfile(pt_path) or not os.path.isfile(jsonl_path):
                logger.warning("Missing data for %s, skipping", bench)
                continue

            residuals = torch.load(pt_path, map_location="cpu", weights_only=True).float()
            with open(jsonl_path) as f:
                records = [json.loads(line) for line in f]

            if len(residuals) != len(records):
                raise ValueError(
                    f"{bench}: residual count {len(residuals)} != record count {len(records)}"
                )
            for i, rec in enumerate(records):
                self.residuals.append(residuals[i])
                self.deltas.append(float(rec.get("best_delta", 0.0)))

        pos = sum(1 for d in self.deltas if d > 0)
        logger.info(
            "DeltaGateDataset: %d samples, %d positive-delta (%.1f%%)",
            len(self.deltas), pos, 100 * pos / max(len(self.deltas), 1),
        )

    def __len__(self):
        return len(self.deltas)

    def __getitem__(self, idx):
        return self.residuals[idx], self.deltas[idx]


def train_delta_gate(
    data_dir: str,
    output_dir: str,
    benchmark: str,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 64,
    fp_weight: float = 2.0,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Optional[str]:
    """Train a DeltaGate (regression) for a single benchmark.

    The loss is asymmetric Huber: over-predictions (predicted_delta >
    actual_delta, i.e. false-positive routing recommendations) are
    weighted by *fp_weight* > 1 to bias the gate toward conservative
    (precision-oriented) predictions.
    """
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = DeltaGateDataset(data_dir, benchmarks=[benchmark])
    if len(ds) == 0:
        logger.error("[%s] No data loaded. Skipping.", benchmark)
        return None

    d_model = ds.residuals[0].shape[0]
    logger.info("[%s] DeltaGate: d_model=%d  samples=%d", benchmark, d_model, len(ds))

    val_size = max(1, int(len(ds) * val_fraction))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = DeltaGate(d_model, hidden_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_path = os.path.join(output_dir, f"delta_gate_best_{benchmark}.pt")
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = torch.tensor(y, dtype=torch.float32, device=device) if not isinstance(y, torch.Tensor) else y.float().to(device)
            pred = model(x)
            residual = pred - y
            base_loss = F.smooth_l1_loss(pred, y, reduction="none")
            weight = torch.where(residual > 0, fp_weight, 1.0)
            loss = (base_loss * weight).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= train_size

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = torch.tensor(y, dtype=torch.float32, device=device) if not isinstance(y, torch.Tensor) else y.float().to(device)
                pred = model(x)
                residual = pred - y
                base_loss = F.smooth_l1_loss(pred, y, reduction="none")
                weight = torch.where(residual > 0, fp_weight, 1.0)
                val_loss += (base_loss * weight).mean().item() * x.size(0)
        val_loss /= val_size

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "d_model": d_model, "hidden_dim": hidden_dim,
                 "dropout": dropout, "benchmark": benchmark},
                ckpt_path,
            )

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "[%s] DeltaGate Epoch %3d  train=%.4f  val=%.4f  (best=%d)",
                benchmark, epoch, train_loss, val_loss, best_epoch,
            )

    logger.info("[%s] DeltaGate Done. Best epoch=%d  val_loss=%.4f  -> %s",
                benchmark, best_epoch, best_val_loss, ckpt_path)
    return ckpt_path


def train_all_gates(
    data_dir: str,
    output_dir: str,
    benchmarks: Optional[List[str]] = None,
    **kwargs,
):
    """Train one gate per benchmark found in *data_dir*."""
    if benchmarks is None:
        benchmarks = GateDataset._discover_benchmarks(data_dir)
    logger.info("Training gates for benchmarks: %s", benchmarks)
    for bench in benchmarks:
        train_gate(data_dir, output_dir, benchmark=bench, **kwargs)


def train_compressed_gate(
    data_dir: str,
    output_dir: str,
    benchmark: str,
    compressor_cfg: CompressorConfig,
    hidden_dim: int = 256,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 64,
    recall_boost: float = 1.2,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Optional[str]:
    """Train a CompressedGate (compressor + gate head) for a single benchmark."""
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_full_seq = compressor_cfg.compressor_type != "last_token"

    if use_full_seq:
        ds = FullSequenceGateDataset(data_dir, benchmarks=[benchmark])
    else:
        ds = GateDataset(data_dir, benchmarks=[benchmark])

    if len(ds) == 0:
        logger.error("[%s] No data loaded. Skipping.", benchmark)
        return None

    d_model = ds.residuals[0].shape[-1]
    compressor_cfg.d_model = d_model
    n_pos = sum(ds.labels)
    n_neg = len(ds) - n_pos
    pw = (n_neg / max(n_pos, 1)) * recall_boost
    pos_weight = torch.tensor([pw], device=device)

    val_size = max(1, int(len(ds) * val_fraction))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    collate_fn = collate_full_sequence_gate if use_full_seq else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_fn)

    compressor = build_compressor(compressor_cfg)
    model = CompressedGate(compressor, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_path = os.path.join(output_dir, f"compressed_gate_best_{benchmark}.pt")
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if use_full_seq:
                x_pad, mask, y = batch
                x_pad, mask, y = x_pad.to(device), mask.to(device), y.to(device)
                logits = model(x_pad, attention_mask=mask)
            else:
                x, y = batch
                x, y = x.to(device), y.float().to(device)
                logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * logits.size(0)
        train_loss /= train_size

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if use_full_seq:
                    x_pad, mask, y = batch
                    x_pad, mask, y = x_pad.to(device), mask.to(device), y.to(device)
                    logits = model(x_pad, attention_mask=mask)
                else:
                    x, y = batch
                    x, y = x.to(device), y.float().to(device)
                    logits = model(x)
                val_loss += F.binary_cross_entropy_with_logits(
                    logits, y, pos_weight=pos_weight
                ).item() * logits.size(0)
        val_loss /= val_size
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "compressor_config": {
                        "compressor_type": compressor_cfg.compressor_type,
                        "d_model": compressor_cfg.d_model,
                        "d_compress": compressor_cfg.d_compress,
                        "n_heads": compressor_cfg.n_heads,
                        "n_latent_tokens": compressor_cfg.n_latent_tokens,
                    },
                    "hidden_dim": hidden_dim,
                    "benchmark": benchmark,
                },
                ckpt_path,
            )

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "[%s] CompressedGate Epoch %3d  train=%.4f  val=%.4f  (best=%d)",
                benchmark, epoch, train_loss, val_loss, best_epoch,
            )

    logger.info("[%s] CompressedGate Done. Best epoch=%d  val_loss=%.4f  -> %s",
                benchmark, best_epoch, best_val_loss, ckpt_path)
    return ckpt_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Train per-benchmark fine-routing gates")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="checkpoints/fine_gate")
    p.add_argument("--benchmarks", nargs="+", default=None,
                   help="Benchmarks to train gates for (default: all in data_dir)")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--w1", type=float, default=0.0,
                   help="Positive class weight; 0 = auto-balanced from data")
    p.add_argument("--w0", type=float, default=1.0)
    p.add_argument("--recall_boost", type=float, default=1.2,
                   help="Multiplier on auto-balanced pos_weight (only used when w1<=0)")
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train_all_gates(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        benchmarks=args.benchmarks,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        w1=args.w1,
        w0=args.w0,
        recall_boost=args.recall_boost,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
