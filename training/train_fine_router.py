#!/usr/bin/env python3
"""Train per-benchmark fine routers that predict deviation distributions.

import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

One router per benchmark/domain.  Each router has its own output dimension
``|D_b|`` matching that benchmark's deviation catalog.

Input : pivot residual vector  [d_model]
Output: log-probability over |D_b| deviations (including no-op at index 0)
Target: pi_target(delta|q,b) = softmax(beta * clip(delta, -c, c))

Loss  : soft cross-entropy  -sum_d pi_target * log pi_theta

By default the router is trained only on gate-positive questions (where at
least one deviation improved over the anchor).

Checkpoints are saved as ``{output_dir}/router_best_{benchmark}.pt``.

Usage
-----
    python train_fine_router.py \
        --data_dir fine_routing_data \
        --output_dir checkpoints/fine_router
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None  # type: ignore

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from training.train_fine_gate import FineGate
from routers.fine_routing_deviations import apply_deviation, enumerate_deviations
from routers.residual_compressors import (
    CompressorConfig,
    CompressedRouter,
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

class RouterDataset(Dataset):
    """Pivot residuals + router target distributions (soft labels)."""

    def __init__(
        self,
        data_dir: str,
        benchmarks: Optional[List[str]] = None,
        gate_positives_only: bool = True,
        mcts_catalog_mode: str = "best_seq",
    ):
        self.residuals: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.num_classes: Optional[int] = None

        if benchmarks is None:
            benchmarks = self._discover_benchmarks(data_dir)

        needs_mcts_remap = False
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
                logger.warning(
                    "%s: residual count %d != record count %d; truncating to min",
                    bench, len(residuals), len(records),
                )
            n = min(len(residuals), len(records))
            residuals = residuals[:n]
            records = records[:n]

            for i, rec in enumerate(records):
                if gate_positives_only and rec["gate_label"] == 0:
                    continue
                target = torch.tensor(rec["router_target"], dtype=torch.float32)
                if self.num_classes is None:
                    self.num_classes = target.shape[0]
                elif target.shape[0] != self.num_classes:
                    needs_mcts_remap = True
                    break
                self.residuals.append(residuals[i])
                self.targets.append(target)
            if needs_mcts_remap:
                break

        # If direct loading failed due variable class counts, remap MCTS records.
        if len(self.targets) == 0 or needs_mcts_remap:
            self._load_mcts_catalog_mode(
                data_dir=data_dir,
                benchmarks=benchmarks,
                gate_positives_only=gate_positives_only,
                mcts_catalog_mode=mcts_catalog_mode,
            )

        logger.info(
            "RouterDataset: %d samples, num_classes=%s",
            len(self.targets),
            self.num_classes,
        )

    def _load_mcts_catalog_mode(
        self,
        data_dir: str,
        benchmarks: List[str],
        gate_positives_only: bool,
        mcts_catalog_mode: str,
    ) -> None:
        self.residuals = []
        self.targets = []
        self.num_classes = None

        for bench in benchmarks:
            pt_path = os.path.join(data_dir, f"{bench}_pivot_residuals.pt")
            jsonl_path = os.path.join(data_dir, f"{bench}.jsonl")
            if not os.path.isfile(pt_path) or not os.path.isfile(jsonl_path):
                logger.warning("Missing data for %s, skipping", bench)
                continue

            residuals = torch.load(pt_path, map_location="cpu", weights_only=True).float()
            with open(jsonl_path) as f:
                records = [json.loads(line) for line in f]
            n = min(len(residuals), len(records))
            residuals = residuals[:n]
            records = records[:n]
            if n == 0:
                continue

            anchor_seq = records[0].get("anchor_sequence")
            if not isinstance(anchor_seq, list):
                raise ValueError(
                    f"{bench}: variable-size router targets require MCTS records "
                    "with anchor_sequence."
                )
            anchor_t = tuple(int(x) for x in anchor_seq)

            if mcts_catalog_mode == "best_seq":
                catalog = [anchor_t]
                seen = {anchor_t}
                for rec in records:
                    s = tuple(int(x) for x in rec["best_seq"])
                    if s not in seen:
                        catalog.append(s)
                        seen.add(s)
            else:
                catalog = [anchor_t]
                seen = {anchor_t}
                for rec in records:
                    for ex in rec.get("explored", []):
                        s = tuple(int(x) for x in ex["seq"])
                        if s not in seen:
                            catalog.append(s)
                            seen.add(s)

            seq_to_idx = {s: i for i, s in enumerate(catalog)}
            num_classes = len(catalog)
            logger.info(
                "%s: remapped MCTS targets with mode=%s -> |C|=%d",
                bench, mcts_catalog_mode, num_classes,
            )

            for i, rec in enumerate(records):
                if gate_positives_only and rec["gate_label"] == 0:
                    continue
                p = torch.zeros(num_classes, dtype=torch.float32)
                if mcts_catalog_mode == "best_seq":
                    if rec.get("gate_label", 0) == 0:
                        p[0] = 1.0
                    else:
                        s = tuple(int(x) for x in rec["best_seq"])
                        p[seq_to_idx.get(s, 0)] = 1.0
                else:
                    for j, prob in enumerate(rec.get("router_target", [])):
                        if j >= len(rec.get("explored", [])):
                            break
                        s = tuple(int(x) for x in rec["explored"][j]["seq"])
                        idx = seq_to_idx.get(s)
                        if idx is not None:
                            p[idx] += float(prob)
                    sm = float(p.sum())
                    if sm > 1e-12:
                        p /= sm
                    else:
                        p[0] = 1.0

                if self.num_classes is None:
                    self.num_classes = num_classes
                elif self.num_classes != num_classes:
                    raise ValueError(
                        f"{bench}: inconsistent remapped catalog size "
                        f"{num_classes} != {self.num_classes}"
                    )
                self.residuals.append(residuals[i])
                self.targets.append(p)

    @staticmethod
    def _discover_benchmarks(data_dir: str) -> List[str]:
        benchmarks = []
        for f in sorted(os.listdir(data_dir)):
            if f.endswith(".jsonl"):
                benchmarks.append(f.replace(".jsonl", ""))
        return benchmarks

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.residuals[idx], self.targets[idx]


def _load_aligned_gate_tensors(
    data_dir: str,
    benchmark: str,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """All aligned rows: pivot residuals [N, D] + gate labels [N] (float 0/1)."""
    pt_path = os.path.join(data_dir, f"{benchmark}_pivot_residuals.pt")
    jsonl_path = os.path.join(data_dir, f"{benchmark}.jsonl")
    residuals = torch.load(pt_path, map_location="cpu", weights_only=True).float()
    with open(jsonl_path) as f:
        records = [json.loads(line) for line in f]
    n = min(len(residuals), len(records))
    residuals = residuals[:n]
    y = torch.tensor([int(records[i]["gate_label"]) for i in range(n)], dtype=torch.float32)
    return residuals, y, n


def _train_gate_pivot(
    residuals: torch.Tensor,
    labels: torch.Tensor,
    output_dir: str,
    benchmark: str,
    d_model: int,
    hidden_dim: int = 256,
    gate_dropout: float = 0.1,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 64,
    val_fraction: float = 0.15,
    recall_boost: float = 1.5,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> str:
    """Binary gate on pivot residuals; saves ``gate_best_{benchmark}.pt``."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    n_pos = int(labels.sum().item())
    n_neg = len(labels) - n_pos
    pw = (n_neg / max(n_pos, 1)) * recall_boost
    pos_weight = torch.tensor([pw], device=device)

    ds = torch.utils.data.TensorDataset(residuals, labels)
    val_size = max(1, int(len(ds) * val_fraction))
    train_size = len(ds) - val_size
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = FineGate(d_model, hidden_dim, dropout=gate_dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_path = os.path.join(output_dir, f"gate_best_{benchmark}.pt")
    best_val = float("inf")
    best_ep = -1

    for epoch in range(1, epochs + 1):
        model.train()
        for x, yb in train_loader:
            x, yb = x.to(device), yb.to(device)
            loss = F.binary_cross_entropy_with_logits(
                model(x), yb, pos_weight=pos_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for x, yb in val_loader:
                x, yb = x.to(device), yb.to(device)
                vloss += F.binary_cross_entropy_with_logits(
                    model(x), yb, pos_weight=pos_weight,
                ).item() * x.size(0)
        vloss /= val_size
        scheduler.step()
        if vloss < best_val:
            best_val = vloss
            best_ep = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "d_model": d_model,
                    "hidden_dim": hidden_dim,
                    "dropout": gate_dropout,
                    "benchmark": benchmark,
                },
                ckpt_path,
            )
        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "[%s] gate epoch %3d  val_bce=%.4f  (best=%d)",
                benchmark, epoch, vloss, best_ep,
            )
    logger.info("[%s] Gate done best_ep=%d val=%.4f -> %s", benchmark, best_ep, best_val, ckpt_path)
    return ckpt_path


def _load_geometry_from_data_dir(
    data_dir: str,
    swap_radius: Optional[int],
    editable_start: Optional[int],
    max_local_edits: Optional[int],
) -> Tuple[int, int, int]:
    """Read search geometry from ``config.json`` with optional CLI overrides."""
    cfg_path = os.path.join(data_dir, "config.json")
    cfg: Dict = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    sr = swap_radius if swap_radius is not None else int(cfg.get("swap_radius", 3))
    es = editable_start if editable_start is not None else int(cfg.get("editable_start", 17))
    mle = max_local_edits if max_local_edits is not None else int(
        cfg.get("max_local_edits", cfg.get("max_swaps", 2))
    )
    return sr, es, mle


class EnumeratedDeviationDataset(Dataset):
    """Pivot residuals + softmax targets over a **fixed** enumerated deviation catalog.

    Builds ``|D|`` classes via :func:`enumerate_deviations` (noop at index 0).
    For each MCTS JSONL row, maps mass from ``router_target`` / ``explored``
    onto deviation indices whose ``apply_deviation`` sequence matches an
    explored sequence, then renormalizes.

    With *exclude_noop=True*, the no-op deviation is **removed** from the
    label space (router must pick a non-stay deviation).  Rows whose MCTS
    mass collapses entirely on the noop after mapping are **skipped**.

    Also stores per-row per-deviation **deltas** (from MCTS ``explored``) for
    offline delta evaluation without extra LM forwards.
    """

    def __init__(
        self,
        data_dir: str,
        benchmarks: List[str],
        gate_positives_only: bool = True,
        swap_radius: Optional[int] = None,
        editable_start: Optional[int] = None,
        max_local_edits: Optional[int] = None,
        exclude_noop: bool = False,
    ):
        if len(benchmarks) != 1:
            raise ValueError("EnumeratedDeviationDataset expects exactly one benchmark.")
        bench = benchmarks[0]
        sr, es, mle = _load_geometry_from_data_dir(
            data_dir, swap_radius, editable_start, max_local_edits,
        )

        pt_path = os.path.join(data_dir, f"{bench}_pivot_residuals.pt")
        jsonl_path = os.path.join(data_dir, f"{bench}.jsonl")
        if not os.path.isfile(pt_path) or not os.path.isfile(jsonl_path):
            raise FileNotFoundError(f"Missing {pt_path} or {jsonl_path}")

        residuals = torch.load(pt_path, map_location="cpu", weights_only=True).float()
        with open(jsonl_path) as f:
            records = [json.loads(line) for line in f]
        n = min(len(residuals), len(records))
        if len(residuals) != len(records):
            logger.warning(
                "%s: residual count %d != record count %d; truncating to %d",
                bench, len(residuals), len(records), n,
            )
        residuals = residuals[:n]
        records = records[:n]

        anchor_seq = records[0].get("anchor_sequence")
        if not isinstance(anchor_seq, list):
            raise ValueError(f"{bench}: need anchor_sequence in JSONL for enumeration.")
        num_layers = len(anchor_seq)

        deviations_full = enumerate_deviations(
            anchor_seq,
            editable_start=es,
            num_layers=num_layers,
            swap_radius=sr,
            max_edits=min(mle, 2),
        )
        self.exclude_noop = bool(exclude_noop)
        if exclude_noop:
            if not deviations_full or deviations_full[0] != ():
                raise ValueError("exclude_noop requires noop () at index 0 of enumerate_deviations.")
            deviations = list(deviations_full[1:])
        else:
            deviations = list(deviations_full)
        self.deviations = deviations
        self.anchor_seq = list(anchor_seq)
        self.editable_start = es
        self.swap_radius = sr
        self.max_local_edits = mle
        self.num_classes = len(deviations)

        seq_to_dev: Dict[tuple, int] = {}
        for i, dev in enumerate(deviations):
            seq_t = tuple(int(x) for x in apply_deviation(anchor_seq, dev))
            if seq_t not in seq_to_dev:
                seq_to_dev[seq_t] = i

        seq_to_full: Dict[tuple, int] = {}
        for fi, dev in enumerate(deviations_full):
            seq_t = tuple(int(x) for x in apply_deviation(anchor_seq, dev))
            seq_to_full[seq_t] = fi

        self.residuals: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.deviation_deltas: List[torch.Tensor] = []
        self.best_explored_delta: List[float] = []

        n_skipped_noop_only = 0
        for i, rec in enumerate(records):
            if gate_positives_only and rec.get("gate_label", 0) == 0:
                continue

            explored = rec.get("explored", [])
            rt = rec.get("router_target", [])
            n_dev_all = len(deviations_full)
            p_all = torch.zeros(n_dev_all, dtype=torch.float32)
            for j, prob in enumerate(rt):
                if j >= len(explored):
                    break
                seq_t = tuple(int(x) for x in explored[j]["seq"])
                di = seq_to_full.get(seq_t)
                if di is not None:
                    p_all[di] += float(prob)
            s_all = float(p_all.sum())
            if s_all > 1e-12:
                p_all /= s_all
            else:
                p_all[0] = 1.0

            if exclude_noop:
                p = p_all[1:].clone()
                s = float(p.sum())
                if s < 1e-12:
                    n_skipped_noop_only += 1
                    continue
                p /= s
            else:
                p = p_all

            dvec = torch.zeros(self.num_classes, dtype=torch.float32)
            for ex in explored:
                seq_t = tuple(int(x) for x in ex["seq"])
                di = seq_to_dev.get(seq_t)
                if di is not None:
                    prev = float(dvec[di].item())
                    dvec[di] = max(prev, float(ex.get("delta", 0.0)))

            best_ex = max((float(ex.get("delta", 0.0)) for ex in explored), default=0.0)

            self.residuals.append(residuals[i])
            self.targets.append(p)
            self.deviation_deltas.append(dvec)
            self.best_explored_delta.append(best_ex)

        logger.info(
            "[%s] EnumeratedDeviationDataset: %d samples, |D|=%d "
            "(editable_start=%d swap_radius=%d max_edits_enum=%d exclude_noop=%s skipped_noop_only=%d)",
            bench, len(self.targets), self.num_classes, es, sr, min(mle, 2),
            exclude_noop, n_skipped_noop_only,
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.residuals[idx], self.targets[idx]


class FullSequenceRouterDataset(Dataset):
    """Full-sequence residuals + router target distributions for attention compressors."""

    def __init__(
        self,
        data_dir: str,
        benchmarks: Optional[List[str]] = None,
        gate_positives_only: bool = True,
    ):
        self.residuals: List[torch.Tensor] = []  # each [T_i, d_model]
        self.targets: List[torch.Tensor] = []
        self.num_classes: Optional[int] = None

        if benchmarks is None:
            benchmarks = RouterDataset._discover_benchmarks(data_dir)

        for bench in benchmarks:
            full_pt_path = os.path.join(data_dir, f"{bench}_full_residuals.pt")
            jsonl_path = os.path.join(data_dir, f"{bench}.jsonl")
            if not os.path.isfile(full_pt_path) or not os.path.isfile(jsonl_path):
                logger.warning("Missing full-seq data for %s, skipping", bench)
                continue

            data = torch.load(full_pt_path, map_location="cpu", weights_only=False)
            full_residuals = data["residuals"]  # list of [T_i, d_model]
            with open(jsonl_path) as f:
                records = [json.loads(line) for line in f]

            n = min(len(full_residuals), len(records))
            for i in range(n):
                rec = records[i]
                if gate_positives_only and rec["gate_label"] == 0:
                    continue
                target = torch.tensor(rec["router_target"], dtype=torch.float32)
                if self.num_classes is None:
                    self.num_classes = target.shape[0]
                self.residuals.append(full_residuals[i].float())
                self.targets.append(target)

        logger.info(
            "FullSequenceRouterDataset: %d samples, num_classes=%s",
            len(self.targets), self.num_classes,
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.residuals[idx], self.targets[idx]


def collate_full_sequence(batch):
    """Collate variable-length residual sequences into a padded batch."""
    residuals = [b[0] for b in batch]
    targets = torch.stack([b[1] for b in batch])
    padded, mask = pad_sequences(residuals)
    return padded, mask, targets


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FineRouter(nn.Module):
    """MLP: pivot residual -> log-prob over deviations."""

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        hidden_dims: List[int] = (512, 256),
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = d_model
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, d_model] -> [B, num_classes] raw logits."""
        return self.net(x)


class PositionalFineRouter(nn.Module):
    """Factored router: predict each editable position independently.

    Instead of 832-way classification over full sequences, this router
    predicts, for each of the ``num_positions`` editable slots, which
    layer index (from a small per-position vocabulary) should fill that
    slot.  At inference the per-position predictions are concatenated
    with the fixed prefix to form a full sequence.

    The shared trunk processes the pivot residual once; separate linear
    heads handle each position.
    """

    def __init__(
        self,
        d_model: int,
        num_positions: int,
        values_per_position: List[List[int]],
        anchor_seq: List[int],
        editable_start: int,
        hidden_dims: List[int] = (512, 256),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_positions = num_positions
        self.editable_start = editable_start
        self.anchor_seq = list(anchor_seq)

        self.values_per_position = values_per_position
        self._vocab_sizes = [len(v) for v in values_per_position]

        trunk_layers: List[nn.Module] = []
        prev = d_model
        for h in hidden_dims:
            trunk_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.trunk = nn.Sequential(*trunk_layers)

        self.heads = nn.ModuleList([
            nn.Linear(prev, vs) for vs in self._vocab_sizes
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """x: [B, d_model] -> list of [B, V_p] logits, one per position."""
        h = self.trunk(x)
        return [head(h) for head in self.heads]

    def predict_sequence(self, x: torch.Tensor) -> List[int]:
        """Return a full layer sequence from a single input [1, d_model]."""
        logits_list = self.forward(x)
        seq = list(self.anchor_seq)
        for p, logits in enumerate(logits_list):
            idx = logits.argmax(dim=-1).item()
            seq[self.editable_start + p] = self.values_per_position[p][idx]
        return seq

    def predict_topk_sequences(self, x: torch.Tensor, k: int = 5) -> List[List[int]]:
        """Return top-k sequences by picking top-k at the most uncertain position."""
        logits_list = self.forward(x)
        probs_list = [F.softmax(lg, dim=-1) for lg in logits_list]

        entropies = []
        for probs in probs_list:
            ent = -(probs * probs.clamp(min=1e-12).log()).sum(dim=-1).item()
            entropies.append(ent)
        pivot_pos = int(max(range(len(entropies)), key=lambda i: entropies[i]))

        base_seq = list(self.anchor_seq)
        for p, logits in enumerate(logits_list):
            if p != pivot_pos:
                idx = logits.argmax(dim=-1).item()
                base_seq[self.editable_start + p] = self.values_per_position[p][idx]

        topk_vals, topk_idxs = logits_list[pivot_pos].topk(
            min(k, self._vocab_sizes[pivot_pos]), dim=-1
        )
        seqs = []
        for j in range(topk_idxs.shape[-1]):
            s = list(base_seq)
            s[self.editable_start + pivot_pos] = self.values_per_position[pivot_pos][topk_idxs[0, j].item()]
            seqs.append(s)
        return seqs
#TODO I'm not convinced the PositionalFineRouter is a good idea. why would it be, the search space is the

# ---------------------------------------------------------------------------
# Soft cross-entropy
# ---------------------------------------------------------------------------

def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """- sum_c target_c * log_softmax(logits)_c, averaged over batch."""
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


def weighted_soft_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """Soft cross-entropy with per-class weights on the decomposition.

    Per sample: ``sum_c w_c * (-target_c * log_softmax_c)``, mean over batch.
    ``class_weights`` is ``[num_classes]``, same normalization as inverse-frequency
    hard CE (typically mean weight 1).
    """
    log_probs = F.log_softmax(logits, dim=-1)
    w = class_weights.to(device=logits.device, dtype=logits.dtype)
    return -(targets * log_probs * w.unsqueeze(0)).sum(dim=-1).mean()


def _enumerated_delta_split_metrics(
    model: nn.Module,
    ds: EnumeratedDeviationDataset,
    indices: List[int],
    device: torch.device,
) -> Dict[str, float]:
    """Greedy argmax Δ and softmax-weighted expected Δ (mapped via explored)."""
    if not indices:
        return {
            "mean_pred_delta": 0.0,
            "mean_expected_delta": 0.0,
            "frac_pred_positive": 0.0,
            "mean_best_explored": 0.0,
        }
    tot = 0
    sum_pred = 0.0
    sum_exp = 0.0
    sum_best = 0.0
    n_pos = 0
    model.eval()
    with torch.no_grad():
        for idx in indices:
            x = ds.residuals[idx].unsqueeze(0).to(device)
            logits = model(x)
            pred = int(logits.argmax(dim=-1).item())
            dvec = ds.deviation_deltas[idx].to(device)
            probs = F.softmax(logits, dim=-1).squeeze(0)
            d_pred = float(dvec[pred].item())
            d_exp = float((probs * dvec).sum().item())
            sum_pred += d_pred
            sum_exp += d_exp
            sum_best += float(ds.best_explored_delta[idx])
            if d_pred > 1e-8:
                n_pos += 1
            tot += 1
    return {
        "mean_pred_delta": sum_pred / tot,
        "mean_expected_delta": sum_exp / tot,
        "frac_pred_positive": n_pos / tot,
        "mean_best_explored": sum_best / tot,
    }


def _enumerated_gated_delta_metrics(
    router: nn.Module,
    gate: FineGate,
    ds: EnumeratedDeviationDataset,
    indices: List[int],
    device: torch.device,
    gamma: float = 0.5,
) -> Dict[str, float]:
    """If sigmoid(gate) >= *gamma*, apply router argmax Δ; else anchor Δ=0."""
    if not indices:
        return {"mean_gated_delta": 0.0, "gate_open_frac": 0.0}
    sum_d = 0.0
    n_open = 0
    router.eval()
    gate.eval()
    with torch.no_grad():
        for idx in indices:
            x = ds.residuals[idx].unsqueeze(0).to(device)
            g = torch.sigmoid(gate(x)).item()
            logits = router(x)
            pred = int(logits.argmax(dim=-1).item())
            d_pred = float(ds.deviation_deltas[idx][pred].item())
            if g >= gamma:
                sum_d += d_pred
                n_open += 1
            # else anchor → 0
    tot = len(indices)
    return {
        "mean_gated_delta": sum_d / tot,
        "gate_open_frac": n_open / tot,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_router(
    data_dir: str,
    output_dir: str,
    benchmark: str,
    gate_positives_only: bool = True,
    hidden_dims: List[int] = (512, 256),
    dropout: float = 0.1,
    lr: float = 1e-3,
    epochs: int = 80,
    batch_size: int = 64,
    val_fraction: float = 0.15,
    seed: int = 42,
    model_is_finetuned: bool = False,
    mcts_catalog_mode: str = "best_seq",
) -> Optional[str]:
    """Train a router for a single benchmark. Returns checkpoint path or None."""
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = RouterDataset(data_dir, benchmarks=[benchmark],
                       gate_positives_only=gate_positives_only,
                       mcts_catalog_mode=mcts_catalog_mode)
    if len(ds) == 0:
        logger.error("[%s] No data loaded (gate_positives_only=%s). Skipping.",
                     benchmark, gate_positives_only)
        return None
    if ds.num_classes is None:
        logger.error("[%s] Could not determine num_classes. Skipping.", benchmark)
        return None

    d_model = ds.residuals[0].shape[0]
    num_classes = ds.num_classes
    logger.info("[%s] d_model=%d  num_classes=%d  samples=%d",
                benchmark, d_model, num_classes, len(ds))

    val_size = max(1, int(len(ds) * val_fraction))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = FineRouter(d_model, num_classes, list(hidden_dims), dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_path = os.path.join(output_dir, f"router_best_{benchmark}.pt")
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = soft_cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= train_size

        # --- val ---
        model.eval()
        val_loss = 0.0
        top1_correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = soft_cross_entropy(logits, y)
                val_loss += loss.item() * x.size(0)
                pred_cls = logits.argmax(dim=-1)
                target_cls = y.argmax(dim=-1)
                top1_correct += (pred_cls == target_cls).sum().item()
                total += x.size(0)
        val_loss /= val_size
        top1_acc = top1_correct / max(total, 1)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "d_model": d_model,
                    "num_classes": num_classes,
                    "hidden_dims": list(hidden_dims),
                    "dropout": dropout,
                    "benchmark": benchmark,
                    "model_is_finetuned": model_is_finetuned,
                },
                ckpt_path,
            )

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "[%s] Epoch %3d  train=%.4f  val=%.4f  top1=%.3f  (best=%d)",
                benchmark, epoch, train_loss, val_loss, top1_acc, best_epoch,
            )

    logger.info("[%s] Done. Best epoch=%d  val_loss=%.4f  -> %s",
                benchmark, best_epoch, best_val_loss, ckpt_path)
    return ckpt_path


def train_all_routers(
    data_dir: str,
    output_dir: str,
    benchmarks: Optional[List[str]] = None,
    **kwargs,
):
    """Train one router per benchmark found in *data_dir*."""
    if benchmarks is None:
        benchmarks = RouterDataset._discover_benchmarks(data_dir)
    logger.info("Training routers for benchmarks: %s", benchmarks)
    for bench in benchmarks:
        train_router(data_dir, output_dir, benchmark=bench, **kwargs)


# ---------------------------------------------------------------------------
# Compressed-router training (compressor + MLP, end-to-end)
# ---------------------------------------------------------------------------

def train_compressed_router(
    data_dir: str,
    output_dir: str,
    benchmark: str,
    compressor_cfg: CompressorConfig,
    gate_positives_only: bool = True,
    hidden_dims: List[int] = (512, 256),
    dropout: float = 0.1,
    lr: float = 1e-3,
    epochs: int = 80,
    batch_size: int = 64,
    val_fraction: float = 0.15,
    seed: int = 42,
    model_is_finetuned: bool = False,
    mcts_catalog_mode: str = "best_seq",
    label_mode: str = "auto",
    swap_radius: Optional[int] = None,
    editable_start: Optional[int] = None,
    max_local_edits: Optional[int] = None,
    exclude_noop: bool = False,
    train_gate: bool = False,
    gate_hidden_dim: int = 256,
    gate_dropout: float = 0.1,
    gate_lr: float = 1e-3,
    gate_epochs: Optional[int] = None,
    recall_boost: float = 1.5,
    gate_gamma: float = 0.5,
) -> Optional[str]:
    """Train a CompressedRouter (compressor + MLP) for a single benchmark.

    When ``compressor_cfg.compressor_type == "last_token"`` this is
    functionally identical to :func:`train_router` but wraps the MLP
    inside a ``CompressedRouter`` for API consistency.
    """
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_full_seq = compressor_cfg.compressor_type != "last_token"

    gate_ckpt_path = os.path.join(output_dir, f"gate_best_{benchmark}.pt")
    if train_gate:
        if label_mode != "enumerated":
            logger.warning("--train_gate is only supported with --label_mode enumerated; skipping gate.")
            train_gate = False
        else:
            res_g, y_g, _n = _load_aligned_gate_tensors(data_dir, benchmark)
            ge = gate_epochs if gate_epochs is not None else epochs
            _train_gate_pivot(
                res_g,
                y_g,
                output_dir,
                benchmark,
                d_model=int(res_g.shape[-1]),
                hidden_dim=gate_hidden_dim,
                gate_dropout=gate_dropout,
                lr=gate_lr,
                epochs=ge,
                batch_size=batch_size,
                val_fraction=val_fraction,
                recall_boost=recall_boost,
                seed=seed,
                device=device,
            )

    if label_mode == "enumerated":
        if use_full_seq:
            raise ValueError("enumerated label_mode requires last_token compressor (pivot vector).")
        router_gate_only = gate_positives_only
        if exclude_noop and not gate_positives_only:
            logger.warning(
                "[%s] exclude_noop: forcing gate_positives_only=True for router dataset.",
                benchmark,
            )
            router_gate_only = True
        ds = EnumeratedDeviationDataset(
            data_dir,
            benchmarks=[benchmark],
            gate_positives_only=router_gate_only,
            swap_radius=swap_radius,
            editable_start=editable_start,
            max_local_edits=max_local_edits,
            exclude_noop=exclude_noop,
        )
    elif use_full_seq:
        ds = FullSequenceRouterDataset(
            data_dir, benchmarks=[benchmark],
            gate_positives_only=gate_positives_only,
        )
    else:
        ds = RouterDataset(
            data_dir, benchmarks=[benchmark],
            gate_positives_only=gate_positives_only,
            mcts_catalog_mode=mcts_catalog_mode,
        )

    if len(ds) == 0 or ds.num_classes is None:
        logger.error("[%s] No data loaded. Skipping.", benchmark)
        return None

    d_model = ds.residuals[0].shape[-1]
    compressor_cfg.d_model = d_model
    num_classes = ds.num_classes

    logger.info(
        "[%s] compressor=%s  d_model=%d  num_classes=%d  samples=%d",
        benchmark, compressor_cfg.compressor_type, d_model, num_classes, len(ds),
    )

    val_size = max(1, int(len(ds) * val_fraction))
    train_size = len(ds) - val_size
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=gen)

    collate_fn = collate_full_sequence if use_full_seq else None
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_fn)

    compressor = build_compressor(compressor_cfg)
    model = CompressedRouter(
        compressor, num_classes, list(hidden_dims), dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("[%s] CompressedRouter: %d params", benchmark, n_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_path = os.path.join(output_dir, f"compressed_router_best_{benchmark}.pt")
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
                x, y = x.to(device), y.to(device)
                logits = model(x)
            loss = soft_cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * logits.size(0)
        train_loss /= train_size

        model.eval()
        val_loss = 0.0
        top1_correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                if use_full_seq:
                    x_pad, mask, y = batch
                    x_pad, mask, y = x_pad.to(device), mask.to(device), y.to(device)
                    logits = model(x_pad, attention_mask=mask)
                else:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                loss = soft_cross_entropy(logits, y)
                val_loss += loss.item() * logits.size(0)
                top1_correct += (logits.argmax(dim=-1) == y.argmax(dim=-1)).sum().item()
                total += logits.size(0)
        val_loss /= val_size
        top1_acc = top1_correct / max(total, 1)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            ckpt_payload: Dict = {
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
                "benchmark": benchmark,
                "model_is_finetuned": model_is_finetuned,
                "label_mode": label_mode,
            }
            if isinstance(ds, EnumeratedDeviationDataset):
                ckpt_payload["enumerated"] = {
                    "num_classes": ds.num_classes,
                    "editable_start": ds.editable_start,
                    "swap_radius": ds.swap_radius,
                    "max_local_edits_cfg": ds.max_local_edits,
                    "anchor_seq": list(ds.anchor_seq),
                    "exclude_noop": exclude_noop,
                }
            ckpt_payload["train_gate"] = train_gate
            if train_gate:
                ckpt_payload["gate_checkpoint"] = gate_ckpt_path
                ckpt_payload["gate_gamma"] = gate_gamma
            torch.save(ckpt_payload, ckpt_path)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "[%s] Epoch %3d  train=%.4f  val=%.4f  top1=%.3f  (best=%d)",
                benchmark, epoch, train_loss, val_loss, top1_acc, best_epoch,
            )

    logger.info("[%s] Done. Best epoch=%d  val_loss=%.4f  -> %s",
                benchmark, best_epoch, best_val_loss, ckpt_path)

    if label_mode == "enumerated" and isinstance(ds, EnumeratedDeviationDataset):
        best_state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_state["model_state_dict"])
        tr_idx = list(train_ds.indices)
        va_idx = list(val_ds.indices)
        tr_met = _enumerated_delta_split_metrics(model, ds, tr_idx, device)
        va_met = _enumerated_delta_split_metrics(model, ds, va_idx, device)
        logger.info(
            "[%s] Enumerated Δ  train: greedy=%+.5f  expected=%+.5f  frac_greedy_pos=%.3f  "
            "mean_best_expl=%+.5f",
            benchmark,
            tr_met["mean_pred_delta"],
            tr_met["mean_expected_delta"],
            tr_met["frac_pred_positive"],
            tr_met["mean_best_explored"],
        )
        logger.info(
            "[%s] Enumerated Δ   val: greedy=%+.5f  expected=%+.5f  frac_greedy_pos=%.3f  "
            "mean_best_expl=%+.5f  (anchor greedy=0)",
            benchmark,
            va_met["mean_pred_delta"],
            va_met["mean_expected_delta"],
            va_met["frac_pred_positive"],
            va_met["mean_best_explored"],
        )
        beats_greedy = va_met["mean_pred_delta"] > 1e-8
        beats_soft = va_met["mean_expected_delta"] > 1e-8
        logger.info(
            "[%s] Beat zero-Δ (val greedy argmax): %s  (=%+.6f)",
            benchmark, beats_greedy, va_met["mean_pred_delta"],
        )
        logger.info(
            "[%s] Beat zero-Δ (val softmax-expected): %s  (=%+.6f)",
            benchmark, beats_soft, va_met["mean_expected_delta"],
        )

        if train_gate and os.path.isfile(gate_ckpt_path):
            gate = FineGate(d_model, gate_hidden_dim, dropout=gate_dropout).to(device)
            g_state = torch.load(gate_ckpt_path, map_location=device, weights_only=False)
            gate.load_state_dict(g_state["model_state_dict"])
            tr_g = _enumerated_gated_delta_metrics(
                model, gate, ds, tr_idx, device, gamma=gate_gamma,
            )
            va_g = _enumerated_gated_delta_metrics(
                model, gate, ds, va_idx, device, gamma=gate_gamma,
            )
            logger.info(
                "[%s] Gated Δ (γ=%.2f) train: mean=%+.5f  gate_open=%.3f",
                benchmark, gate_gamma, tr_g["mean_gated_delta"], tr_g["gate_open_frac"],
            )
            logger.info(
                "[%s] Gated Δ (γ=%.2f)  val: mean=%+.5f  gate_open=%.3f",
                benchmark, gate_gamma, va_g["mean_gated_delta"], va_g["gate_open_frac"],
            )

    return ckpt_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _fine_eval_benchmark_names(args: argparse.Namespace) -> Optional[List[str]]:
    """Benchmarks that have both gate and router checkpoints under output_dir."""
    if args.benchmarks:
        return list(args.benchmarks)
    out = _Path(args.output_dir)
    names: List[str] = []
    for g in sorted(out.glob("gate_best_*.pt")):
        b = g.stem.replace("gate_best_", "", 1)
        if (out / f"router_best_{b}.pt").is_file():
            names.append(b)
    return names or None


def main():
    p = argparse.ArgumentParser(description="Train per-benchmark fine routers")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="checkpoints/fine_router")
    p.add_argument("--benchmarks", nargs="+", default=None,
                   help="Benchmarks to train routers for (default: all in data_dir)")
    p.add_argument("--gate_positives_only", action="store_true", default=True)
    p.add_argument("--all_questions", dest="gate_positives_only", action="store_false",
                   help="Train on all questions, not just gate-positive ones")
    p.add_argument("--hidden_dims", nargs="+", type=int, default=[512, 256])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--mcts_catalog_mode",
        type=str,
        default="best_seq",
        choices=["best_seq", "explored"],
        help="How to remap variable-size MCTS targets into fixed classes.",
    )
    p.add_argument(
        "--model_is_finetuned",
        action="store_true",
        help="Tag checkpoints as trained on FT-model routing data.",
    )
    p.add_argument(
        "--label_mode",
        type=str,
        default="auto",
        choices=["auto", "enumerated"],
        help="``enumerated``: fixed deviation-catalog targets from ``enumerate_deviations`` "
        "(pivot residual + last_token compressor path).",
    )
    p.add_argument(
        "--enum_swap_radius",
        type=int,
        default=None,
        help="Override swap_radius for enumeration (default: data_dir/config.json).",
    )
    p.add_argument(
        "--enum_editable_start",
        type=int,
        default=None,
        help="Override editable_start for enumeration (default: data_dir/config.json).",
    )
    p.add_argument(
        "--enum_max_local_edits",
        type=int,
        default=None,
        help="Override max_local_edits for enumeration (default: data_dir/config.json).",
    )
    p.add_argument(
        "--exclude_noop",
        action="store_true",
        help="Enumerated mode only: drop stay/no-op class; skip rows with only noop mass.",
    )
    p.add_argument(
        "--train_gate",
        action="store_true",
        help="Enumerated mode only: train binary gate on all aligned rows, then train router.",
    )
    p.add_argument("--gate_hidden_dim", type=int, default=256)
    p.add_argument("--gate_dropout", type=float, default=0.1)
    p.add_argument("--gate_lr", type=float, default=1e-3)
    p.add_argument(
        "--gate_epochs",
        type=int,
        default=None,
        help="Gate epochs (default: same as --epochs).",
    )
    p.add_argument(
        "--recall_boost",
        type=float,
        default=1.5,
        help="BCE pos_weight scale for gate (see sweep_fine_routing).",
    )
    p.add_argument(
        "--gate_gamma",
        type=float,
        default=0.5,
        help="Open gate when sigmoid(g) >= this threshold for gated Δ logging.",
    )
    p.add_argument("--compressor_type", type=str, default="last_token",
                   choices=["last_token", "top_down_attention"])
    p.add_argument("--compressor_d_compress", type=int, default=256)
    p.add_argument("--compressor_n_heads", type=int, default=4)
    p.add_argument("--compressor_n_latent", type=int, default=1)
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases.")
    p.add_argument("--wandb_project", type=str, default="fine-router")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument(
        "--auto_external_llm_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After training, run experiments.run_fine_routing_inference (needs gates + --external_eval_results_dir).",
    )
    p.add_argument(
        "--external_eval_results_dir",
        type=str,
        default=None,
        help="MCTS results dir for anchor sequences (required for auto external eval).",
    )
    p.add_argument(
        "--external_eval_model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )
    p.add_argument("--inference_gamma", type=float, default=0.8)
    p.add_argument("--external_eval_gpu_rank", type=int, default=0)
    args = p.parse_args()

    wb_run = None
    if args.wandb:
        if not HAS_WANDB:
            logger.warning("wandb requested but not installed; skipping W&B.")
        else:
            wb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name)
            try:
                from training.auto_external_llm_eval import write_wandb_run_info

                write_wandb_run_info(args.output_dir, wb_run)
            except Exception:
                logger.exception("write_wandb_run_info failed.")

    try:
        if args.label_mode == "enumerated":
            benchmarks = args.benchmarks
            if benchmarks is None:
                benchmarks = RouterDataset._discover_benchmarks(args.data_dir)
            comp_cfg = CompressorConfig(
                compressor_type="last_token",
                d_model=896,
                d_compress=args.compressor_d_compress,
                n_heads=args.compressor_n_heads,
                n_latent_tokens=args.compressor_n_latent,
            )
            for bench in benchmarks:
                train_compressed_router(
                    data_dir=args.data_dir,
                    output_dir=args.output_dir,
                    benchmark=bench,
                    compressor_cfg=comp_cfg,
                    gate_positives_only=args.gate_positives_only,
                    hidden_dims=args.hidden_dims,
                    dropout=args.dropout,
                    lr=args.lr,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    val_fraction=args.val_fraction,
                    seed=args.seed,
                    model_is_finetuned=args.model_is_finetuned,
                    mcts_catalog_mode=args.mcts_catalog_mode,
                    label_mode="enumerated",
                    swap_radius=args.enum_swap_radius,
                    editable_start=args.enum_editable_start,
                    max_local_edits=args.enum_max_local_edits,
                    exclude_noop=args.exclude_noop,
                    train_gate=args.train_gate,
                    gate_hidden_dim=args.gate_hidden_dim,
                    gate_dropout=args.gate_dropout,
                    gate_lr=args.gate_lr,
                    gate_epochs=args.gate_epochs,
                    recall_boost=args.recall_boost,
                    gate_gamma=args.gate_gamma,
                )
        elif args.compressor_type != "last_token":
            comp_cfg = CompressorConfig(
                compressor_type=args.compressor_type,
                d_compress=args.compressor_d_compress,
                n_heads=args.compressor_n_heads,
                n_latent_tokens=args.compressor_n_latent,
            )
            benchmarks = args.benchmarks
            if benchmarks is None:
                benchmarks = RouterDataset._discover_benchmarks(args.data_dir)
            for bench in benchmarks:
                train_compressed_router(
                    data_dir=args.data_dir,
                    output_dir=args.output_dir,
                    benchmark=bench,
                    compressor_cfg=comp_cfg,
                    gate_positives_only=args.gate_positives_only,
                    hidden_dims=args.hidden_dims,
                    dropout=args.dropout,
                    lr=args.lr,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    val_fraction=args.val_fraction,
                    seed=args.seed,
                    model_is_finetuned=args.model_is_finetuned,
                    mcts_catalog_mode=args.mcts_catalog_mode,
                    label_mode="auto",
                )
        else:
            train_all_routers(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                benchmarks=args.benchmarks,
                gate_positives_only=args.gate_positives_only,
                hidden_dims=args.hidden_dims,
                dropout=args.dropout,
                lr=args.lr,
                epochs=args.epochs,
                batch_size=args.batch_size,
                val_fraction=args.val_fraction,
                seed=args.seed,
                model_is_finetuned=args.model_is_finetuned,
                mcts_catalog_mode=args.mcts_catalog_mode,
            )

        if wb_run is not None and args.auto_external_llm_eval:
            gates = list(_Path(args.output_dir).glob("gate_best_*.pt"))
            if not gates:
                logger.warning(
                    "Skipping fine-router external LLM eval: no gate_best_*.pt in %s",
                    args.output_dir,
                )
            elif not args.external_eval_results_dir:
                logger.warning(
                    "Skipping fine-router external LLM eval: set --external_eval_results_dir.",
                )
            else:
                bnames = _fine_eval_benchmark_names(args)
                if not bnames:
                    logger.warning(
                        "Skipping fine-router external LLM eval: no paired gate/router checkpoints.",
                    )
                else:
                    try:
                        from training.auto_external_llm_eval import (
                            run_fine_router_external_infer_subprocess_and_log,
                        )

                        out_json = _Path(args.output_dir) / "external_eval_fine_llm.json"
                        run_fine_router_external_infer_subprocess_and_log(
                            wb_run,
                            checkpoint_dir=args.output_dir,
                            results_dir=args.external_eval_results_dir,
                            benchmarks=bnames,
                            output_json=out_json,
                            model_name=args.external_eval_model_name,
                            inference_gamma=float(args.inference_gamma),
                            gpu_rank=int(args.external_eval_gpu_rank),
                        )
                    except Exception:
                        logger.exception("Fine-router external LLM eval failed.")
    finally:
        if wb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
