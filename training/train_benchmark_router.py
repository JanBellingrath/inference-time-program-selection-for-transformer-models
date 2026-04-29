#!/usr/bin/env python
"""
Train a benchmark sequence router: given a question's embedding-layer token
embeddings, predict which benchmark-optimal module sequence to use.

The router is an m-class classifier (m = number of benchmarks) trained with
cross-entropy on precomputed embeddings (no base model required at training).

Usage:
    python train_benchmark_router.py --embedding_dir cache/benchmark_router_embeddings
    python train_benchmark_router.py --compression attention --mlp_hidden 512 256
"""
import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import argparse
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from routers.bias_model import (
    extract_bias_features,
    extract_bias_features_batch,
    BiasClassifier,
    pretrain_bias_model,
    NUM_BIAS_FEATURES,
    NUM_BIAS_FEATURES_WITH_NORM,
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    logger.warning("wandb not installed; metrics will only be logged locally.")


def load_optimal_sequences_from_results(
    results_dir: str,
    benchmarks: List[str],
    model_name: Optional[str] = None,
) -> Dict[str, List[int]]:
    """Load optimal sequences from MCTS result files.

    Scans ``results_dir`` for two file types (in priority order):
      1. ``benchmark_mcts_<bench>_*_snapshot.json`` – snapshot files with
         ``best_tier4`` / ``best_tier3`` entries produced by the MCTS loop.
      2. ``final_val_<bench>_*.json`` – explicit final-validation JSONs with
         ``results`` or ``candidates`` arrays.

    For each benchmark the *newest* matching file (by mtime) is used so that
    results from re-runs automatically take precedence.
    """
    import glob as _glob

    mapping: Dict[str, List[int]] = {}

    def _model_match_score(path: str) -> int:
        """Higher = better match for model_name. 0 = no preference."""
        if not model_name:
            return 0
        name = path.lower()
        if "0.5b" in model_name.lower() or "0.5b-instruct" in model_name.lower():
            return 2 if "0.5b" in name else (1 if "7b" not in name else 0)
        if "7b" in model_name.lower() or "7b-instruct" in model_name.lower():
            return 2 if "7b" in name else (1 if "0.5b" not in name else 0)
        return 0

    for bench in benchmarks:
        # --- Strategy 1: MCTS snapshot files ---
        pattern = os.path.join(results_dir, f"benchmark_mcts_{bench}_*_snapshot.json")
        raw = _glob.glob(pattern)
        snapshot_files = sorted(
            raw,
            key=lambda p: (-_model_match_score(p), -os.path.getmtime(p)),
        )
        for sf in snapshot_files:
            try:
                with open(sf) as f:
                    data = json.load(f)
                for tier_key in ("best_tier4", "best_tier3"):
                    best = data.get(tier_key)
                    if best and ("layers" in best or "seq" in best):
                        # Prefer seq (full-length with SKIP) for fine-routing; else layers
                        mapping[bench] = best.get("seq", best["layers"])
                        logger.info(
                            f"  Loaded {bench} sequence from {os.path.basename(sf)} "
                            f"({tier_key}, acc={best.get('accuracy', '?')}, "
                            f"delta={best.get('delta', '?')})"
                        )
                        break
                if bench in mapping:
                    break
            except Exception as e:
                logger.warning(f"  Could not parse {sf}: {e}")

        if bench in mapping:
            continue

        # --- Strategy 1b: fixed_benchmark_mcts_* snapshot files ---
        pattern = os.path.join(results_dir, f"fixed_benchmark_mcts_{bench}_*_snapshot.json")
        raw = _glob.glob(pattern)
        snapshot_files = sorted(
            raw,
            key=lambda p: (-_model_match_score(p), -os.path.getmtime(p)),
        )
        for sf in snapshot_files:
            try:
                with open(sf) as f:
                    data = json.load(f)
                for tier_key in ("best_tier4", "best_tier3", "best_validated"):
                    best = data.get(tier_key)
                    if best and ("layers" in best or "seq" in best):
                        mapping[bench] = best.get("seq", best["layers"])
                        logger.info(
                            f"  Loaded {bench} sequence from {os.path.basename(sf)} "
                            f"({tier_key}, acc={best.get('accuracy', '?')}, "
                            f"delta={best.get('delta', '?')})"
                        )
                        break
                if bench in mapping:
                    break
            except Exception as e:
                logger.warning(f"  Could not parse {sf}: {e}")

        if bench in mapping:
            continue

        # --- Strategy 2: final_val result files ---
        pattern = os.path.join(results_dir, f"final_val_{bench}_*.json")
        final_files = sorted(_glob.glob(pattern), key=os.path.getmtime, reverse=True)
        for ff in final_files:
            try:
                with open(ff) as f:
                    data = json.load(f)
                if "results" in data and data["results"]:
                    mapping[bench] = data["results"][0]["layers"]
                elif "candidates" in data and data["candidates"]:
                    best = max(data["candidates"], key=lambda c: c["accuracy"])
                    mapping[bench] = best["layers"]
                if bench in mapping:
                    logger.info(
                        f"  Loaded {bench} sequence from {os.path.basename(ff)}"
                    )
                    break
            except Exception as e:
                logger.warning(f"  Could not parse {ff}: {e}")

    return mapping


# ---------------------------------------------------------------------------
# Compression modules
# ---------------------------------------------------------------------------

class MeanPoolCompression(nn.Module):
    """Global mean pool over token dimension: [B, T, H] -> [B, H]."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.output_dim = hidden_size

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if lengths is not None:
            mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            x = x * mask.unsqueeze(-1).float()
            return x.sum(dim=1) / lengths.unsqueeze(1).float().clamp(min=1)
        return x.mean(dim=1)


class WindowPoolCompression(nn.Module):
    """Split tokens into W windows, mean-pool each, concatenate: [B, T, H] -> [B, W*H]."""
    def __init__(self, hidden_size: int, num_windows: int = 4):
        super().__init__()
        self.num_windows = num_windows
        self.output_dim = num_windows * hidden_size

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, H = x.shape
        W = self.num_windows
        usable = (T // W) * W
        if usable == 0:
            logger.warning(f"WindowPoolCompression: usable == 0, returning mean pool")
            return x.mean(dim=1).unsqueeze(1).expand(B, W, H).reshape(B, W * H)
        trimmed = x[:, :usable, :]
        windowed = trimmed.view(B, W, usable // W, H)
        if lengths is not None:
            pooled_parts = []
            for b in range(B):
                L = min(lengths[b].item(), usable)
                win_size = usable // W
                parts = []
                for w in range(W):
                    start = w * win_size
                    end = min(start + win_size, L)
                    if end > start:
                        parts.append(trimmed[b, start:end].mean(dim=0))
                    else:
                        logger.warning(f"WindowPoolCompression: end > start, returning zero")
                        parts.append(torch.zeros(H, device=x.device))
                pooled_parts.append(torch.stack(parts))
            return torch.stack(pooled_parts).view(B, W * H)
        pooled = windowed.mean(dim=2)  # [B, W, H]
        return pooled.view(B, W * H)


class AttentionCompression(nn.Module):
    """Learnable cross-attention queries over tokens: [B, T, H] -> [B, Q*D]."""
    def __init__(
        self,
        hidden_size: int,
        num_queries: int = 4,
        compress_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert compress_dim % num_heads == 0
        self.num_queries = num_queries
        self.compress_dim = compress_dim
        self.output_dim = num_queries * compress_dim

        self.token_proj = nn.Linear(hidden_size, compress_dim)
        self.queries = nn.Parameter(torch.randn(num_queries, compress_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=compress_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(compress_dim)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.shape[0]
        kv = self.token_proj(x)  # [B, T, D]

        key_padding_mask = None
        if lengths is not None:
            T = kv.size(1)
            key_padding_mask = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)

        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, Q, D]
        out, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask)
        out = self.norm(out + q)
        return out.reshape(B, -1)


class LinearProjCompression(nn.Module):
    """Per-token linear projection then mean pool: [B, T, H] -> [B, proj_dim]."""
    def __init__(self, hidden_size: int, proj_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(hidden_size, proj_dim)
        self.output_dim = proj_dim

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        projected = self.proj(x)  # [B, T, proj_dim]
        if lengths is not None:
            mask = torch.arange(projected.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            projected = projected * mask.unsqueeze(-1).float()
            return projected.sum(dim=1) / lengths.unsqueeze(1).float().clamp(min=1)
        return projected.mean(dim=1)


COMPRESSION_REGISTRY = {
    "mean_pool": MeanPoolCompression,
    "window_pool": WindowPoolCompression,
    "attention": AttentionCompression,
    "linear_proj": LinearProjCompression,
}


# ---------------------------------------------------------------------------
# Router model
# ---------------------------------------------------------------------------

class BenchmarkSequenceRouter(nn.Module):
    """
    Classifies a question's token embeddings into one of m benchmark classes.

    Architecture:
        token embeddings [B, T, H]
            -> compression module -> [B, compressed_dim]
            -> N-layer MLP -> [B, m]
    """

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        compression: str = "mean_pool",
        mlp_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        # attention-specific
        num_queries: int = 4,
        attn_compress_dim: int = 128,
        attn_num_heads: int = 4,
        # window-specific
        num_windows: int = 4,
        # linear_proj-specific
        proj_dim: int = 256,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.compression_name = compression
        self.mlp_hidden_dims = mlp_hidden_dims or [512, 256]
        self.dropout_val = dropout

        if compression == "mean_pool":
            self.compressor = MeanPoolCompression(hidden_size)
        elif compression == "window_pool":
            self.compressor = WindowPoolCompression(hidden_size, num_windows=num_windows)
        elif compression == "attention":
            self.compressor = AttentionCompression(
                hidden_size,
                num_queries=num_queries,
                compress_dim=attn_compress_dim,
                num_heads=attn_num_heads,
                dropout=dropout,
            )
        elif compression == "linear_proj":
            self.compressor = LinearProjCompression(hidden_size, proj_dim=proj_dim)
        else:
            raise ValueError(f"Unknown compression: {compression}")

        in_dim = self.compressor.output_dim
        layers: List[nn.Module] = []
        for h in self.mlp_hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        embeddings: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ):
        """
        Args:
            embeddings: [B, T, H] token embeddings (from embedding layer).
            lengths:    [B] actual token counts (before padding).
            return_hidden: If True, return (logits, hidden) where hidden is the
                representation right before the final linear layer [B, mlp_hidden_dims[-1]].
        Returns:
            logits: [B, num_classes], or (logits, hidden) if return_hidden.
        """
        compressed = self.compressor(embeddings.float(), lengths)
        h = compressed
        for i in range(len(self.mlp) - 1):
            h = self.mlp[i](h)
        logits = self.mlp[-1](h)
        if return_hidden:
            return logits, h
        return logits

    def get_config(self) -> Dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "num_classes": self.num_classes,
            "compression": self.compression_name,
            "mlp_hidden_dims": self.mlp_hidden_dims,
            "dropout": self.dropout_val,
        }


# ---------------------------------------------------------------------------
# Gradient Reversal (adversarial debiasing)
# ---------------------------------------------------------------------------

class _GradientReversal(torch.autograd.Function):
    """Identity forward, negated & scaled gradient backward."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.scale * grad_output, None


class BiasFeatureAdversary(nn.Module):
    """MLP: logits -> predicted bias features (for adversarial debiasing)."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Tuple[int, ...] = (64, 64)):
        super().__init__()
        layers: List[nn.Module] = []
        in_d = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_d, h), nn.ReLU()])
            in_d = h
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.net(logits)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BenchmarkEmbeddingDataset(Dataset):
    """
    Loads precomputed .pt embedding files and assigns class labels per benchmark.

    Each sample yields:
        embedding: [seq_len, hidden_size]  (variable length)
        label:     int  (benchmark class index)
        benchmark: str  (benchmark name, for per-class metrics)
    """

    def __init__(
        self,
        embedding_dir: str,
        benchmarks: List[str],
        split: str = "train",
        precompute_bias: bool = False,
        max_per_benchmark: Optional[int] = None,
    ):
        self.data: List[Dict] = []
        self.benchmark_names = benchmarks
        self.benchmark_to_idx = {b: i for i, b in enumerate(benchmarks)}

        for bench in benchmarks:
            path = os.path.join(embedding_dir, f"{bench}_{split}.pt")
            if not os.path.isfile(path):
                logger.warning(f"Missing embedding file: {path} -- skipping")
                continue
            raw = torch.load(path, map_location="cpu", weights_only=False)
            samples = raw["embeddings"]
            subsampled = False
            if max_per_benchmark is not None and len(samples) > max_per_benchmark:
                rng = random.Random(42)
                samples = rng.sample(samples, max_per_benchmark)
                subsampled = True
                logger.info(f"  Subsampled {bench} {split} to {max_per_benchmark} (was {len(raw['embeddings'])})")
            label = self.benchmark_to_idx[bench]
            for s in samples:
                question = s.get("question", "")
                emb = s["embedding"].clone() if subsampled else s["embedding"]
                item: Dict[str, Any] = {
                    "embedding": emb,  # [T, H]
                    "label": label,
                    "benchmark": bench,
                    "question": question,
                    "correct": s.get("correct", ""),
                    "full_prompt": s.get("full_prompt", question),
                }
                if precompute_bias:
                    item["bias_features"] = torch.from_numpy(
                        extract_bias_features(item["question"]) #TODO what does this do?
                    )
                    # Mean-pool L2 norm (controls for activation magnitude / length confound)
                    emb = s["embedding"].float()
                    item["mean_embed_norm"] = torch.norm(emb.mean(dim=0)).item()
                self.data.append(item)
            del raw  # free full .pt before loading next benchmark (avoids OOM)
            logger.info(f"Loaded {len(samples)} {split} samples for {bench} (class {label})")

        logger.info(f"Total {split} samples: {len(self.data)}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]


def make_collate_fn(include_bias: bool = False):
    """Create a collate function for benchmark embedding batches.

    Args:
        include_bias: If True, also batch the precomputed ``bias_features`` tensor.
    """
    def collate(batch: List[Dict]) -> Dict[str, Any]:
        max_len = max(b["embedding"].shape[0] for b in batch)
        H = batch[0]["embedding"].shape[1]

        padded = torch.zeros(len(batch), max_len, H, dtype=batch[0]["embedding"].dtype)
        lengths = torch.zeros(len(batch), dtype=torch.long)
        labels = torch.zeros(len(batch), dtype=torch.long)
        benchmarks = []

        for i, b in enumerate(batch):
            T = b["embedding"].shape[0]
            padded[i, :T, :] = b["embedding"]
            lengths[i] = T
            labels[i] = b["label"]
            benchmarks.append(b["benchmark"])

        out: Dict[str, Any] = {
            "embeddings": padded,
            "lengths": lengths,
            "labels": labels,
            "benchmarks": benchmarks,
        }

        if include_bias and "bias_features" in batch[0]:
            text_feats = torch.stack([b["bias_features"] for b in batch])
            norms = torch.tensor(
                [b["mean_embed_norm"] for b in batch],
                dtype=text_feats.dtype,
            ).unsqueeze(1)
            out["bias_features"] = torch.cat([text_feats, norms], dim=1)

        return out
    return collate


collate_benchmark = make_collate_fn()


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights, normalized so mean(weight)=1."""
    t = torch.tensor(labels, dtype=torch.long)
    counts = torch.bincount(t, minlength=num_classes).float().clamp(min=1.0)
    total = t.size(0)
    weights = total / (num_classes * counts)
    return (weights / weights.mean()).to(torch.float32)


def train_epoch(
    router: BenchmarkSequenceRouter,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    grad_accum: int = 1,
    bias_model: Optional[BiasClassifier] = None,
    debias_alpha: float = 1.0,
    class_weight: Optional[torch.Tensor] = None,
    adversary: Optional[BiasFeatureAdversary] = None,
    adv_optimizer: Optional[optim.Optimizer] = None,
    adv_lambda: float = 1.0,
) -> Dict[str, float]:
    router.train()
    if adversary is not None:
        adversary.train()
    total_loss = 0.0
    total_adv_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()
    if adv_optimizer is not None:
        adv_optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc="Train")):
        emb = batch["embeddings"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)

        router_logits = router(emb, lengths)

        if bias_model is not None and "bias_features" in batch:
            bias_feats = batch["bias_features"].to(device)
            bias_logits = bias_model(bias_feats).detach()
            logits = router_logits - debias_alpha * bias_logits #TODO need to check if we apply this consistently during val
        else:
            logits = router_logits

        loss_kw = {}
        if class_weight is not None:
            loss_kw["weight"] = class_weight.to(device)
        task_loss = F.cross_entropy(logits, labels, **loss_kw) / grad_accum

        combined = task_loss
        if adversary is not None and "bias_features" in batch:
            bias_target = batch["bias_features"].to(device)
            rev_logits = _GradientReversal.apply(router_logits, adv_lambda)
            adv_loss = F.mse_loss(adversary(rev_logits), bias_target) / grad_accum
            combined = combined + adv_loss
            total_adv_loss += adv_loss.item() * grad_accum

        combined.backward()

        total_loss += task_loss.item() * grad_accum
        preds = router_logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if adv_optimizer is not None:
                adv_optimizer.step()
                adv_optimizer.zero_grad()

    if (step + 1) % grad_accum != 0:
        nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if adv_optimizer is not None:
            adv_optimizer.step()
            adv_optimizer.zero_grad()

    n_batches = len(loader)
    metrics = {
        "train/loss": total_loss / max(1, n_batches),
        "train/acc": correct / max(1, total),
        "train/lr": scheduler.get_last_lr()[0],
    }
    if adversary is not None:
        metrics["train/adv_loss"] = total_adv_loss / max(1, n_batches)
    return metrics


@torch.no_grad()
def evaluate(
    router: BenchmarkSequenceRouter,
    loader: DataLoader,
    device: torch.device,
    benchmark_names: List[str],
    bias_model: Optional[BiasClassifier] = None,
    debias_alpha: float = 1.0,
    class_weight: Optional[torch.Tensor] = None,
    adversary: Optional[BiasFeatureAdversary] = None,
) -> Dict[str, Any]:
    """
    Validation mode 1: classification accuracy on held-out split.
    Reports overall and per-benchmark accuracy plus confusion matrix.
    When a bias_model is provided, also reports "router-only" accuracy
    (without bias subtraction) so we can see what the router learned alone.
    """
    router.eval()
    if adversary is not None:
        adversary.eval()
    total_loss = 0.0
    total_adv_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_benchmarks: List[str] = []
    all_router_preds: List[int] = []

    for batch in tqdm(loader, desc="Eval"):
        emb = batch["embeddings"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)

        router_logits = router(emb, lengths)

        if bias_model is not None and "bias_features" in batch:
            bias_feats = batch["bias_features"].to(device)
            bias_logits = bias_model(bias_feats)
            logits = router_logits - debias_alpha * bias_logits
        else:
            logits = router_logits

        loss_kw = {}
        if class_weight is not None:
            loss_kw["weight"] = class_weight.to(device)
        loss = F.cross_entropy(logits, labels, **loss_kw)
        total_loss += loss.item()

        if adversary is not None and "bias_features" in batch:
            bias_target = batch["bias_features"].to(device)
            total_adv_loss += F.mse_loss(adversary(router_logits), bias_target).item()

        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_benchmarks.extend(batch["benchmarks"])
        all_router_preds.extend(router_logits.argmax(dim=-1).cpu().tolist())

    n_batches = len(loader)
    all_preds_t = torch.tensor(all_preds)
    all_labels_t = torch.tensor(all_labels)

    overall_acc = (all_preds_t == all_labels_t).float().mean().item()

    per_bench: Dict[str, float] = {}
    per_bench_f1: Dict[str, float] = {}
    num_classes = len(benchmark_names)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for i in range(len(all_preds)):
        confusion[all_labels[i], all_preds[i]] += 1

    for idx, name in enumerate(benchmark_names):
        mask = all_labels_t == idx
        if mask.sum() > 0:
            per_bench[name] = (all_preds_t[mask] == idx).float().mean().item()
        else:
            per_bench[name] = 0.0

        tp = confusion[idx, idx].item()
        fp = confusion[:, idx].sum().item() - tp
        fn = confusion[idx, :].sum().item() - tp
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        per_bench_f1[name] = (
            2 * precision * recall / max(1e-8, precision + recall)
        )

    metrics: Dict[str, Any] = {
        "val/loss": total_loss / max(1, n_batches),
        "val/acc": overall_acc,
    }
    if adversary is not None:
        metrics["val/adv_loss"] = total_adv_loss / max(1, n_batches)
    for name in benchmark_names:
        metrics[f"val/acc_{name}"] = per_bench[name]
        metrics[f"val/f1_{name}"] = per_bench_f1[name]

    metrics["_confusion"] = confusion
    metrics["_per_bench_acc"] = per_bench

    if bias_model is not None:
        all_router_preds_t = torch.tensor(all_router_preds)
        metrics["val/router_only_acc"] = (
            (all_router_preds_t == all_labels_t).float().mean().item()
        )

    return metrics


@torch.no_grad()
def _build_mcts_seen_hashes(model_name: str, mcts_seed: int, mcts_n_used: int,
                            benchmarks: Optional[List[str]] = None) -> Dict[str, set]:
    """Reconstruct per-benchmark sets of question hashes the MCTS search saw.

    Only benchmarks where the MCTS searched on the same HF split as the eval
    need filtering. Currently this is only ``mmlu_all`` (MCTS falls back to HF
    validation because auxiliary_train is missing).
    """
    import hashlib
    from core.flexible_models import get_is_instruct
    from core.permutation_mcts import prepare_arc_data

    affected = {"mmlu_all"}
    targets = affected & set(benchmarks or [])
    if not targets:
        return {}

    is_instruct = get_is_instruct(model_name)
    result: Dict[str, set] = {}
    for bench in targets:
        all_data = prepare_arc_data(bench, is_instruct, split="validation")
        shuffled = list(all_data)
        random.seed(mcts_seed)
        random.shuffle(shuffled)
        seen = {hashlib.md5((s["input"] + str(s.get("correct", ""))).encode()).hexdigest()
                for s in shuffled[:mcts_n_used]}
        result[bench] = seen
        logger.info("[FULL PASS] %s: built MCTS-seen set (%d hashes) for leak-free eval",
                    bench, len(seen))
    return result


def evaluate_routing_full_pass(
    router: BenchmarkSequenceRouter,
    val_ds: "BenchmarkEmbeddingDataset",
    device: torch.device,
    benchmark_names: List[str],
    optimal_sequences: Dict[str, List[int]],
    mcts_model: Any,
    model_name: str,
    samples_per_bench: Optional[int] = 100,
    sample_seed: Optional[int] = 43,
    bias_model: Optional[BiasClassifier] = None,
    debias_alpha: float = 1.0,
    sample_benchmarks: Optional[List[str]] = None,
    mcts_seed: Optional[int] = None,
    mcts_n_used: Optional[int] = None,
) -> Dict[str, Any]:
    """End-to-end validation: router predicts class -> select optimal sequence
    -> generate with the LLM -> grade the answer.

    Samples are balanced across benchmarks (equal count per benchmark).
    If sample_benchmarks is provided, only sample from those benchmarks (e.g. to
    exclude BigBench). benchmark_names must still include all router classes for
    correct pred_cls -> pred_bench mapping.

    If *mcts_seed* and *mcts_n_used* are provided, MMLU samples that the MCTS
    search already saw are excluded before sampling (leak-free eval).
    """
    from core.benchmark_mcts import grade_response, seq_to_layers
    from core.flexible_models import get_is_instruct
    from core.prompts import get_tale_system_prompt, get_tale_max_new_tokens

    router.eval()
    is_instruct = get_is_instruct(model_name)
    wrapper = mcts_model.wrapper
    default_layers = list(range(mcts_model.num_layers))

    # Build exclusion hashes for benchmarks where MCTS and eval share a split
    mcts_seen: Dict[str, set] = {}
    if mcts_seed is not None and mcts_n_used is not None:
        mcts_seen = _build_mcts_seen_hashes(
            model_name, mcts_seed, mcts_n_used,
            benchmarks=sample_benchmarks or benchmark_names)

    def _generate(layers: List[int], text: str, max_tokens: int,
                  system_prompt: str = None) -> str:
        """Generate with arbitrary layer sequence (mirrors BenchmarkMCTS._generate)."""
        saved = wrapper.model.model.layer_indices
        wrapper.model.model.layer_indices = layers
        try:
            has_dup = len(layers) != len(set(layers))
            non_default_order = list(layers) != list(range(mcts_model.num_layers))
            prompt = wrapper.prepare_prompt(text, system_prompt=system_prompt)
            inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(
                wrapper.model.device
            )
            input_len = inputs.input_ids.shape[1]
            gen_kw = {
                "max_new_tokens": max_tokens,
                "pad_token_id": wrapper.tokenizer.eos_token_id,
                "do_sample": False,
            }
            if has_dup or non_default_order:
                gen_kw["use_cache"] = False
            out = wrapper.model.generate(**inputs, **gen_kw)
            return wrapper.tokenizer.decode(
                out[0][input_len:], skip_special_tokens=True
            ).strip()
        finally:
            wrapper.model.model.layer_indices = saved

    # --- Balanced sampling: equal count per benchmark (seeded for reproducibility) ---
    benches_to_sample = sample_benchmarks if sample_benchmarks is not None else benchmark_names
    rng = random.Random(sample_seed) if sample_seed is not None else random
    by_bench: Dict[str, List[Dict]] = {b: [] for b in benchmark_names}
    for item in val_ds.data:
        by_bench[item["benchmark"]].append(item)

    # Filter out MCTS-seen items for affected benchmarks
    if mcts_seen:
        import hashlib
        for bench, seen_set in mcts_seen.items():
            if bench not in by_bench:
                continue
            orig = len(by_bench[bench])
            by_bench[bench] = [
                item for item in by_bench[bench]
                if hashlib.md5((item["question"] + str(item.get("correct", ""))).encode()).hexdigest()
                not in seen_set
            ]
            logger.info("[FULL PASS] %s: filtered %d -> %d samples (removed %d MCTS-seen)",
                        bench, orig, len(by_bench[bench]), orig - len(by_bench[bench]))

    samples: List[Dict] = []
    for bench in benches_to_sample:
        pool = by_bench.get(bench, [])
        k = len(pool) if samples_per_bench is None else min(samples_per_bench, len(pool))
        if k > 0:
            samples.extend(rng.sample(pool, k))
    rng.shuffle(samples)

    if not samples:
        logger.warning("[FULL PASS] No validation samples available.")
        return {}

    logger.info(
        "[FULL PASS] Evaluating %d samples (%s per bench, %d benchmarks, seed=%s)",
        len(samples),
        "max" if samples_per_bench is None else str(samples_per_bench),
        len(benches_to_sample),
        sample_seed if sample_seed is not None else "random",
    )

    # Accumulators: [correct, total] per benchmark (only for benches we sample)
    baseline_ct: Dict[str, List[int]] = {b: [0, 0] for b in benches_to_sample}
    routed_ct: Dict[str, List[int]] = {b: [0, 0] for b in benches_to_sample}
    routing_hits = 0
    routed_when_correct: List[int] = [0, 0]  # [correct, total] when pred==actual
    routed_when_wrong: List[int] = [0, 0]    # [correct, total] when pred!=actual
    routed_when_correct_per_bench: Dict[str, List[int]] = {b: [0, 0] for b in benches_to_sample}
    routed_when_wrong_per_bench: Dict[str, List[int]] = {b: [0, 0] for b in benches_to_sample}

    # Winogrande correct-label fixup: raw embeddings may store "A"/"B"
    # but TALE grading expects "1"/"2".
    _WINO_REMAP = {"A": "1", "B": "2"}

    for item in tqdm(samples, desc="Full-pass eval"):
        actual_bench = item["benchmark"]
        grade_bench = actual_bench
        max_tokens = get_tale_max_new_tokens(grade_bench)
        sys_prompt = get_tale_system_prompt(grade_bench)

        correct_answer = item["correct"]
        if actual_bench == "winogrande" and correct_answer in _WINO_REMAP:
            correct_answer = _WINO_REMAP[correct_answer]

        # --- Router prediction ---
        emb = item["embedding"].unsqueeze(0).to(device)  # [1, T, H]
        length = torch.tensor([emb.shape[1]], device=device)
        router_logits = router(emb, length)
        if bias_model is not None and "bias_features" in item:
            bf = item["bias_features"].unsqueeze(0).to(device)
            router_logits = router_logits - debias_alpha * bias_model(bf)
        pred_cls = router_logits.argmax(dim=-1).item()
        pred_bench = benchmark_names[pred_cls]
        if pred_bench == actual_bench:
            routing_hits += 1

        seq = optimal_sequences.get(pred_bench)
        if seq is None:
            seq = default_layers
        layers = seq_to_layers(seq)

        # --- Generate: baseline (default) and routed ---
        gen_prompt = item.get("full_prompt", item["question"])
        for tag, ly, ct in [
            ("baseline", default_layers, baseline_ct),
            ("routed", layers, routed_ct),
        ]:
            resp = _generate(ly, gen_prompt, max_tokens,
                             system_prompt=sys_prompt)
            sc = grade_response(
                resp, correct_answer, grade_bench, model_name, gen_prompt
            )
            ok = int(sc > 0.5)
            ct[actual_bench][0] += ok
            ct[actual_bench][1] += 1
            if tag == "routed":
                if pred_bench == actual_bench:
                    routed_when_correct[0] += ok
                    routed_when_correct[1] += 1
                    routed_when_correct_per_bench[actual_bench][0] += ok
                    routed_when_correct_per_bench[actual_bench][1] += 1
                else:
                    routed_when_wrong[0] += ok
                    routed_when_wrong[1] += 1
                    routed_when_wrong_per_bench[actual_bench][0] += ok
                    routed_when_wrong_per_bench[actual_bench][1] += 1

    # --- Aggregate metrics ---
    def _agg(ct: Dict[str, List[int]], prefix: str) -> Dict[str, float]:
        total_c = sum(v[0] for v in ct.values())
        total_n = sum(v[1] for v in ct.values())
        m = {f"{prefix}/acc": total_c / max(1, total_n)}
        for bench in benches_to_sample:
            c, n = ct.get(bench, [0, 0])
            m[f"{prefix}/acc_{bench}"] = c / max(1, n)
        return m

    total_n = sum(v[1] for v in routed_ct.values())
    acc_correct = routed_when_correct[0] / max(1, routed_when_correct[1])
    acc_wrong = routed_when_wrong[0] / max(1, routed_when_wrong[1])
    metrics: Dict[str, Any] = {
        **_agg(baseline_ct, "val_full_baseline"),
        **_agg(routed_ct, "val_full"),
        "val_full/routing_acc": routing_hits / max(1, total_n),
        "val_full/routed_when_correct_acc": acc_correct,
        "val_full/routed_when_wrong_acc": acc_wrong,
    }
    for bench in benches_to_sample:
        c_corr, n_corr = routed_when_correct_per_bench.get(bench, [0, 0])
        c_wrong, n_wrong = routed_when_wrong_per_bench.get(bench, [0, 0])
        metrics[f"val_full/routed_when_correct_acc_{bench}"] = c_corr / max(1, n_corr)
        metrics[f"val_full/routed_when_wrong_acc_{bench}"] = c_wrong / max(1, n_wrong)
        metrics[f"val_full/n_{bench}"] = routed_ct.get(bench, [0, 0])[1]

    logger.info(
        "[FULL PASS] routed=%.4f baseline=%.4f routing_acc=%.4f (%d samples)",
        metrics["val_full/acc"], metrics["val_full_baseline/acc"],
        metrics["val_full/routing_acc"], total_n,
    )
    logger.info(
        "[FULL PASS] routed when router correct: %.4f (%d/%d) | routed when wrong: %.4f (%d/%d)",
        acc_correct, routed_when_correct[0], routed_when_correct[1],
        acc_wrong, routed_when_wrong[0], routed_when_wrong[1],
    )
    for bench in benches_to_sample:
        rc, rn = routed_ct.get(bench, [0, 0])
        bc, bn = baseline_ct.get(bench, [0, 0])
        logger.info(
            "    %s: routed=%.4f (%d/%d) baseline=%.4f (%d/%d)",
            bench, rc / max(1, rn), rc, rn, bc / max(1, bn), bc, bn,
        )
        cc, nc = routed_when_correct_per_bench.get(bench, [0, 0])
        cw, nw = routed_when_wrong_per_bench.get(bench, [0, 0])
        if nc > 0 or nw > 0:
            logger.info(
                "      router correct: %.4f (%d/%d) | router wrong: %.4f (%d/%d)",
                cc / max(1, nc), cc, nc, cw / max(1, nw), cw, nw,
            )
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkRouterConfig:
    embedding_dir: str = "cache/benchmark_router_embeddings"
    results_dir: str = "predictions"
    benchmarks: List[str] = field(
        default_factory=lambda: [
            "gsm8k_hard", "winogrande", "mmlu_all",
            "commonsenseqa", "arc_challenge", "bigbench_all",
        ]
    )
    compression: str = "mean_pool"
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    dropout: float = 0.1
    num_windows: int = 4
    num_queries: int = 4
    attn_compress_dim: int = 128
    attn_num_heads: int = 4
    proj_dim: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.01
    num_epochs: int = 30
    batch_size: int = 64
    grad_accum: int = 1
    warmup_fraction: float = 0.1
    seed: int = 42
    output_dir: str = "checkpoints/benchmark_router"
    wandb_project: str = "benchmark-router"
    wandb_enabled: bool = True
    debias: bool = False
    debias_alpha: float = 1.0
    debias_epochs: int = 100
    adversarial_debias: bool = False
    adv_lambda: float = 1.0
    adv_lr: float = 1e-3
    adv_hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    eval_routing: bool = False
    sequences_json: Optional[str] = None
    full_pass: bool = False
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    full_pass_samples: int = 100
    full_pass_sample_seed: Optional[int] = 43
    full_pass_every: int = 1
    balance_classes: bool = False
    max_per_benchmark: Optional[int] = None
    eval_only: bool = False
    checkpoint_path: Optional[str] = None
    save_results: Optional[str] = None
    skip_train_routing_acc: bool = False
    eval_after_train_only: bool = False
    exclude_benchmarks: Optional[List[str]] = None
    mcts_seed: Optional[int] = None
    mcts_n_used: Optional[int] = None
    auto_external_llm_eval: bool = True


def run_eval_only(cfg: BenchmarkRouterConfig):
    """Load checkpoint and run full-pass evaluation only (no training)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = cfg.checkpoint_path or os.path.join(cfg.output_dir, "checkpoint_best.pt")
    if not os.path.isfile(ckpt_path):
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    logger.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    router_config = ckpt["router_config"]
    train_cfg = ckpt.get("training_config", {})
    optimal_sequences = ckpt.get("optimal_sequences", {})

    benchmarks = train_cfg.get("benchmarks", cfg.benchmarks)
    embedding_dir = train_cfg.get("embedding_dir", cfg.embedding_dir)
    results_dir = train_cfg.get("results_dir", cfg.results_dir)
    # Prefer CLI model_name over checkpoint (eval-time model may differ from training default)
    model_name = cfg.model_name
    full_pass_samples = None if cfg.full_pass_samples == -1 else cfg.full_pass_samples
    full_pass_sample_seed = getattr(cfg, "full_pass_sample_seed", None) or 43
    debias_alpha = train_cfg.get("debias_alpha", cfg.debias_alpha)
    adversarial_debias = train_cfg.get("adversarial_debias", False)
    debias = train_cfg.get("debias", False)
    exclude_benchmarks = getattr(cfg, "exclude_benchmarks", None) or []

    # For full-pass eval: only sample from benchmarks not excluded (e.g. exclude BigBench)
    eval_benchmarks = [b for b in benchmarks if b not in exclude_benchmarks]
    if exclude_benchmarks:
        logger.info("Excluding benchmarks for full-pass eval: %s -> eval on %s",
                    exclude_benchmarks, eval_benchmarks)

    need_bias = debias and not adversarial_debias
    max_per_bench = train_cfg.get("max_per_benchmark", cfg.max_per_benchmark)
    val_ds = BenchmarkEmbeddingDataset(
        embedding_dir, eval_benchmarks, split="val",
        precompute_bias=need_bias,
        max_per_benchmark=max_per_bench,
    )
    if len(val_ds) == 0:
        logger.error("No validation data found.")
        sys.exit(1)

    train_ds = BenchmarkEmbeddingDataset(
        embedding_dir, benchmarks, split="train",
        precompute_bias=need_bias,
        max_per_benchmark=max_per_bench,
    )

    if not optimal_sequences:
        optimal_sequences = load_optimal_sequences_from_results(results_dir, benchmarks)
    if cfg.sequences_json and os.path.isfile(cfg.sequences_json):
        with open(cfg.sequences_json) as f:
            optimal_sequences.update(json.load(f))
        logger.info("Overrode sequences from %s", cfg.sequences_json)

    hidden_size = val_ds.data[0]["embedding"].shape[-1]
    num_classes = len(benchmarks)
    router = BenchmarkSequenceRouter(
        hidden_size=hidden_size,
        num_classes=num_classes,
        compression=router_config.get("compression", cfg.compression),
        mlp_hidden_dims=router_config.get("mlp_hidden_dims", cfg.mlp_hidden_dims),
        dropout=router_config.get("dropout_val", cfg.dropout),
        num_windows=train_cfg.get("num_windows", cfg.num_windows),
        num_queries=train_cfg.get("num_queries", cfg.num_queries),
        attn_compress_dim=train_cfg.get("attn_compress_dim", cfg.attn_compress_dim),
        attn_num_heads=train_cfg.get("attn_num_heads", cfg.attn_num_heads),
        proj_dim=train_cfg.get("proj_dim", cfg.proj_dim),
    ).to(device)
    router.load_state_dict(ckpt["router_state_dict"])
    router.eval()

    logger.info("Loading %s for full-pass evaluation...", model_name)
    from core.permutation_mcts import MCTSModel
    mcts_model = MCTSModel(model_name, rank=0)

    bias_model = None
    if need_bias:
        logger.warning("eval_only with PoE debias: bias model not saved in checkpoint, using router only")

    metrics = evaluate_routing_full_pass(
        router, val_ds, device, benchmarks,
        optimal_sequences, mcts_model, model_name,
        samples_per_bench=full_pass_samples,
        sample_seed=full_pass_sample_seed,
        bias_model=bias_model,
        debias_alpha=debias_alpha,
        sample_benchmarks=eval_benchmarks,
        mcts_seed=cfg.mcts_seed,
        mcts_n_used=cfg.mcts_n_used,
    )

    routing_acc_val = metrics.get("val_full/routing_acc", 0)
    routing_acc_train: Optional[float] = None
    if len(train_ds) > 0 and not getattr(cfg, "skip_train_routing_acc", False):
        collate_fn = make_collate_fn(include_bias=need_bias)
        train_loader = DataLoader(
            train_ds, batch_size=64, shuffle=False,
            collate_fn=collate_fn, num_workers=0, pin_memory=False,
        )
        train_metrics = evaluate(
            router, train_loader, device, benchmarks,
            bias_model=bias_model, debias_alpha=debias_alpha,
        )
        routing_acc_train = train_metrics.get("val/acc", 0)
        logger.info("Router accuracy: train=%.4f val=%.4f", routing_acc_train, routing_acc_val)

    logger.info("=== Full-pass evaluation complete ===")
    logger.info("  routed acc:    %.4f", metrics.get("val_full/acc", 0))
    logger.info("  baseline acc:  %.4f", metrics.get("val_full_baseline/acc", 0))
    logger.info("  routing acc:   %.4f", metrics.get("val_full/routing_acc", 0))
    logger.info("  routed when router correct: %.4f", metrics.get("val_full/routed_when_correct_acc", 0))
    logger.info("  routed when router wrong:   %.4f", metrics.get("val_full/routed_when_wrong_acc", 0))

    if cfg.save_results:
        out = {
            "model_name": model_name,
            "benchmarks": eval_benchmarks,
            "samples_per_bench": full_pass_samples,
            "sample_seed": full_pass_sample_seed,
            "baseline_acc": metrics.get("val_full_baseline/acc"),
            "routed_acc": metrics.get("val_full/acc"),
            "routing_acc": routing_acc_val,
            "routing_acc_train": routing_acc_train,
            "routing_acc_val": routing_acc_val,
            "per_benchmark": {},
        }
        for b in eval_benchmarks:
            bc = metrics.get(f"val_full_baseline/acc_{b}")
            rc = metrics.get(f"val_full/acc_{b}")
            n = metrics.get(f"val_full/n_{b}")
            out["per_benchmark"][b] = {
                "baseline_acc": bc,
                "routed_acc": rc,
                "delta_pp": (rc - bc) * 100 if (bc is not None and rc is not None) else None,
                "n": n,
            }
        os.makedirs(os.path.dirname(cfg.save_results) or ".", exist_ok=True)
        with open(cfg.save_results, "w") as f:
            json.dump(out, f, indent=2)
        logger.info("Saved results to %s", cfg.save_results)

        # Run visualization script automatically
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fig_path = os.path.join(script_dir, "figures", "router_results_overview.png")
        try:
            import subprocess
            subprocess.run(
                [sys.executable, "benchmark_level_router_results_visualizations.py",
                 "--results", cfg.save_results, "--output", fig_path],
                cwd=script_dir, check=True,
            )
            logger.info("Figure saved: %s", fig_path)
        except subprocess.CalledProcessError as e:
            logger.warning("Visualization failed: %s", e)

    return metrics


def train(cfg: BenchmarkRouterConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- Resolve optimal sequences ---
    optimal_sequences: Dict[str, List[int]] = {}
    if cfg.sequences_json and os.path.isfile(cfg.sequences_json):
        with open(cfg.sequences_json) as f:
            optimal_sequences.update(json.load(f))
        logger.info(f"Loaded sequences from {cfg.sequences_json}")
    else:
        parsed = load_optimal_sequences_from_results(cfg.results_dir, cfg.benchmarks)
        optimal_sequences.update(parsed)
        if parsed:
            logger.info(f"Parsed sequences from results for: {list(parsed.keys())}")

    for bench in cfg.benchmarks:
        seq = optimal_sequences.get(bench)
        if seq is None:
            logger.warning(f"No optimal sequence for {bench}; using identity")
        else:
            diff = [i for i, v in enumerate(seq) if v != i]
            logger.info(f"  {bench}: {len(seq)} layers, swaps at positions {diff}")

    # --- Datasets ---
    need_bias = cfg.debias or cfg.adversarial_debias
    train_ds = BenchmarkEmbeddingDataset(
        cfg.embedding_dir, cfg.benchmarks, split="train",
        precompute_bias=need_bias,
        max_per_benchmark=cfg.max_per_benchmark,
    )
    val_ds = BenchmarkEmbeddingDataset(
        cfg.embedding_dir, cfg.benchmarks, split="val",
        precompute_bias=need_bias,
        max_per_benchmark=cfg.max_per_benchmark,
    )

    if len(train_ds) == 0:
        logger.error("No training data found. Run precompute_benchmark_embeddings.py first.")
        sys.exit(1)

    train_labels = [d["label"] for d in train_ds.data]
    class_weight = compute_class_weights(train_labels, len(cfg.benchmarks))
    logger.info("  Class weights (inverse-freq, mean=1): %s", class_weight.tolist())

    train_sampler = None
    if cfg.balance_classes:
        counts = torch.bincount(torch.tensor(train_labels), minlength=len(cfg.benchmarks)).float()
        sample_weights = 1.0 / counts[torch.tensor(train_labels)]
        n_balanced = int(counts.min().item()) * len(cfg.benchmarks)
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=n_balanced, replacement=False)
        logger.info("balance_classes: per-class counts %s, sampling %d per epoch (%d per class)",
                     {b: int(counts[i]) for i, b in enumerate(cfg.benchmarks)},
                     n_balanced, int(counts.min().item()))
#TODO above does overlaps with the inverse frequency weighting, need to check if we need to use one or the other only 
    collate_fn = make_collate_fn(include_bias=need_bias)

    num_dl_workers = 0 if (need_bias or cfg.max_per_benchmark is not None) else 4
    pin = not need_bias
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn, num_workers=num_dl_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_dl_workers, pin_memory=pin,
    )

    hidden_size = train_ds.data[0]["embedding"].shape[-1]
    num_classes = len(cfg.benchmarks)
    logger.info(f"hidden_size={hidden_size}, num_classes={num_classes}")

    # --- Router ---
    router = BenchmarkSequenceRouter(
        hidden_size=hidden_size,
        num_classes=num_classes,
        compression=cfg.compression,
        mlp_hidden_dims=cfg.mlp_hidden_dims,
        dropout=cfg.dropout,
        num_queries=cfg.num_queries,
        attn_compress_dim=cfg.attn_compress_dim,
        attn_num_heads=cfg.attn_num_heads,
        num_windows=cfg.num_windows,
        proj_dim=cfg.proj_dim,
    ).to(device)

    n_params = sum(p.numel() for p in router.parameters())
    logger.info(f"Router parameters: {n_params:,}")

    # --- Adversary (adversarial debiasing via gradient reversal) ---
    adversary: Optional[BiasFeatureAdversary] = None
    adv_optimizer: Optional[optim.Optimizer] = None
    if cfg.adversarial_debias:
        adversary = BiasFeatureAdversary(
            input_dim=num_classes,
            output_dim=NUM_BIAS_FEATURES_WITH_NORM,
            hidden_dims=tuple(cfg.adv_hidden_dims),
        ).to(device)
        adv_optimizer = optim.Adam(adversary.parameters(), lr=cfg.adv_lr)
        n_adv = sum(p.numel() for p in adversary.parameters())
        logger.info(f"Adversary parameters: {n_adv:,}  lambda={cfg.adv_lambda}")

    # --- Bias model (PoE debiasing) ---
    frozen_bias: Optional[BiasClassifier] = None
    if cfg.debias:
        logger.info("=== Pretraining bias (shortcut) model ===")
        text_feats = torch.stack([d["bias_features"] for d in train_ds.data])
        norm_feats = torch.tensor(
            [d["mean_embed_norm"] for d in train_ds.data],
            dtype=text_feats.dtype,
        ).unsqueeze(1)
        bias_feats = torch.cat([text_feats, norm_feats], dim=1)
        bias_labels = torch.tensor([d["label"] for d in train_ds.data]) #TODO wait the bias labels are the same as the class labels?
        frozen_bias = pretrain_bias_model(
            bias_feats, bias_labels,
            num_classes=num_classes,
            epochs=cfg.debias_epochs,
            device=device,
        )
        val_text = torch.stack([d["bias_features"] for d in val_ds.data])
        val_norms = torch.tensor(
            [d["mean_embed_norm"] for d in val_ds.data],
            dtype=val_text.dtype,
        ).unsqueeze(1)
        bias_val_feats = torch.cat([val_text, val_norms], dim=1)
        bias_val_labels = torch.tensor([d["label"] for d in val_ds.data])
        with torch.no_grad():
            val_bias_logits = frozen_bias(bias_val_feats.to(device))
            bias_val_acc = (
                (val_bias_logits.argmax(-1) == bias_val_labels.to(device))
                .float().mean().item()
            )
        logger.info(f"  Bias model val accuracy: {bias_val_acc:.4f}")
        logger.info(f"  debias_alpha = {cfg.debias_alpha}")

    # --- Optimizer / Scheduler ---
    optimizer = optim.AdamW(
        router.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_fraction)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=max(total_steps, 1),
        pct_start=warmup_steps / max(total_steps, 1),
    )

    # --- wandb ---
    use_wandb = cfg.wandb_enabled and HAS_WANDB
    if use_wandb:
        wb_config: Dict[str, Any] = {
            **asdict(cfg),
            "hidden_size": hidden_size,
            "num_classes": num_classes,
            "num_params": n_params,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "optimal_sequences": {
                b: optimal_sequences.get(b) for b in cfg.benchmarks
            },
        }
        if frozen_bias is not None:
            wb_config["bias_val_acc"] = bias_val_acc  # type: ignore[possibly-undefined]
        wandb.init(project=cfg.wandb_project, config=wb_config)
        wandb.watch(router, log="gradients", log_freq=50)
        try:
            from training.auto_external_llm_eval import write_wandb_run_info

            write_wandb_run_info(cfg.output_dir, wandb.run)
        except Exception:
            logger.warning("write_wandb_run_info failed.", exc_info=True)

    # --- Full-pass model (loaded once, reused across epochs) ---
    mcts_model = None
    if cfg.full_pass:
        from core.permutation_mcts import MCTSModel
        logger.info("Loading %s for full-pass evaluation...", cfg.model_name)
        mcts_model = MCTSModel(cfg.model_name, rank=0)
        logger.info("Full-pass model ready (%d layers).", mcts_model.num_layers)

    # --- Training loop ---
    os.makedirs(cfg.output_dir, exist_ok=True)
    best_val_acc = 0.0
    history: List[Dict] = []
    confusion = None

    run_eval_this_epoch = lambda e: not cfg.eval_after_train_only or e == cfg.num_epochs

    for epoch in range(1, cfg.num_epochs + 1):
        logger.info(f"Epoch {epoch}/{cfg.num_epochs}")

        train_metrics = train_epoch(
            router, train_loader, optimizer, scheduler, device, cfg.grad_accum,
            bias_model=frozen_bias, debias_alpha=cfg.debias_alpha,
            class_weight=class_weight,
            adversary=adversary, adv_optimizer=adv_optimizer,
            adv_lambda=cfg.adv_lambda,
        )

        if run_eval_this_epoch(epoch):
            val_metrics = evaluate(
                router, val_loader, device, cfg.benchmarks,
                bias_model=frozen_bias, debias_alpha=cfg.debias_alpha,
                class_weight=class_weight,
                adversary=adversary,
            )
            confusion = val_metrics.pop("_confusion")
            per_bench_acc = val_metrics.pop("_per_bench_acc")

            # --- Optional end-to-end generation evaluation ---
            full_metrics: Dict[str, Any] = {}
            if mcts_model is not None and epoch % cfg.full_pass_every == 0:
                full_metrics = evaluate_routing_full_pass(
                    router, val_ds, device, cfg.benchmarks,
                    optimal_sequences, mcts_model, cfg.model_name,
                    samples_per_bench=cfg.full_pass_samples,
                    sample_seed=cfg.full_pass_sample_seed,
                    bias_model=frozen_bias, debias_alpha=cfg.debias_alpha,
                    mcts_seed=cfg.mcts_seed,
                    mcts_n_used=cfg.mcts_n_used,
                )
        else:
            val_metrics = {"val/loss": 0.0, "val/acc": 0.0}
            full_metrics = {}
            confusion = None
            per_bench_acc = {b: 0.0 for b in cfg.benchmarks}
            for b in cfg.benchmarks:
                val_metrics[f"val/f1_{b}"] = 0.0

        epoch_metrics = {"epoch": epoch, **train_metrics, **val_metrics, **full_metrics}
        history.append(epoch_metrics)

        log_parts = [
            f"  train loss={train_metrics['train/loss']:.4f} "
            f"acc={train_metrics['train/acc']:.4f}"
        ]
        if run_eval_this_epoch(epoch):
            log_parts[0] += f" | val loss={val_metrics['val/loss']:.4f} acc={val_metrics['val/acc']:.4f}"
        if "val/router_only_acc" in val_metrics:
            log_parts[0] += f" | router_only={val_metrics['val/router_only_acc']:.4f}"
        if "train/adv_loss" in train_metrics:
            log_parts[0] += f" | adv={train_metrics['train/adv_loss']:.4f}"
        if "val/adv_loss" in val_metrics:
            log_parts[0] += f" | val_adv={val_metrics['val/adv_loss']:.4f}"
        logger.info(log_parts[0])

        if run_eval_this_epoch(epoch):
            for name in cfg.benchmarks:
                logger.info(
                    f"    {name}: acc={per_bench_acc[name]:.4f} "
                    f"f1={val_metrics[f'val/f1_{name}']:.4f}"
                )

        if use_wandb and confusion is not None:
            log_dict = {k: v for k, v in epoch_metrics.items() if k != "epoch"}
            try:
                y_true = []
                preds = []
                for l in range(confusion.shape[0]):
                    for p in range(confusion.shape[1]):
                        for _ in range(int(confusion[l, p].item())):
                            y_true.append(l)
                            preds.append(p)
                log_dict["val/confusion_matrix"] = wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true,
                    preds=preds,
                    class_names=cfg.benchmarks,
                )
            except Exception as e:
                logger.warning("Could not log confusion matrix to wandb: %s", e)
            wandb.log(log_dict, step=epoch)

        is_best = run_eval_this_epoch(epoch) and val_metrics["val/acc"] > best_val_acc
        if is_best:
            best_val_acc = val_metrics["val/acc"]

        checkpoint = {
            "router_state_dict": router.state_dict(),
            "router_config": router.get_config(),
            "training_config": asdict(cfg),
            "optimal_sequences": {b: optimal_sequences.get(b) for b in cfg.benchmarks},
            "benchmark_to_idx": train_ds.benchmark_to_idx,
            "epoch": epoch,
            "metrics": epoch_metrics,
        }
        if adversary is not None:
            checkpoint["adversary_state_dict"] = adversary.state_dict()
        torch.save(checkpoint, os.path.join(cfg.output_dir, "checkpoint_latest.pt"))
        if is_best:
            torch.save(checkpoint, os.path.join(cfg.output_dir, "checkpoint_best.pt"))
            logger.info(f"  ** New best val acc: {best_val_acc:.4f} **")

    with open(os.path.join(cfg.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    if cfg.eval_routing:
        evaluate_routing_stub(
            router, val_loader, device, cfg.benchmarks, optimal_sequences,
        )

    if use_wandb and cfg.auto_external_llm_eval:
        best_path = os.path.join(cfg.output_dir, "checkpoint_best.pt")
        if not os.path.isfile(best_path):
            logger.warning("Auto external LLM eval skipped: missing %s", best_path)
        else:
            try:
                ck_best = torch.load(best_path, map_location="cpu", weights_only=False)
                router.load_state_dict(ck_best["router_state_dict"])
                fp_model = mcts_model
                if fp_model is None:
                    from core.permutation_mcts import MCTSModel

                    logger.info("Loading %s for post-train LLM eval...", cfg.model_name)
                    fp_model = MCTSModel(cfg.model_name, rank=0)
                exclude_b = list(cfg.exclude_benchmarks or [])
                eval_benches = [b for b in cfg.benchmarks if b not in exclude_b]
                samples_pb = None if cfg.full_pass_samples == -1 else cfg.full_pass_samples
                from training.auto_external_llm_eval import run_benchmark_router_full_pass_and_log

                run_benchmark_router_full_pass_and_log(
                    wandb.run,
                    evaluate_full_pass_fn=evaluate_routing_full_pass,
                    router=router,
                    val_ds=val_ds,
                    device=device,
                    benchmark_names=cfg.benchmarks,
                    optimal_sequences={b: optimal_sequences.get(b) for b in cfg.benchmarks},
                    mcts_model=fp_model,
                    model_name=cfg.model_name,
                    samples_per_bench=samples_pb,
                    sample_seed=cfg.full_pass_sample_seed,
                    bias_model=frozen_bias,
                    debias_alpha=cfg.debias_alpha,
                    sample_benchmarks=eval_benches,
                    mcts_seed=cfg.mcts_seed,
                    mcts_n_used=cfg.mcts_n_used,
                )
            except Exception:
                logger.exception("Benchmark-router post-train LLM eval failed.")

    if use_wandb:
        wandb.finish()

    logger.info(f"Training complete. Best val acc: {best_val_acc:.4f}")
    if confusion is not None:
        logger.info(f"Confusion matrix (rows=true, cols=pred):")
        header = "          " + "  ".join(f"{b[:8]:>8}" for b in cfg.benchmarks)
        logger.info(header)
        for i, name in enumerate(cfg.benchmarks):
            row = f"{name[:8]:>8}  " + "  ".join(f"{int(confusion[i, j]):>8}" for j in range(len(cfg.benchmarks)))
            logger.info(row)


def main():
    parser = argparse.ArgumentParser(description="Train benchmark sequence router")
    parser.add_argument("--embedding_dir", default="cache/benchmark_router_embeddings")
    parser.add_argument("--results_dir", default="predictions",
                        help="Directory with MCTS final_val result JSONs")
    parser.add_argument("--benchmarks", nargs="+",
                        default=["gsm8k_hard", "winogrande", "mmlu_all",
                                 "commonsenseqa", "arc_challenge", "bigbench_all"],
                        help="Benchmark names for training/eval. Space or comma-separated. "
                             "E.g. --benchmarks bigbench_all winogrande or --benchmarks bigbench_all,winogrande")
    parser.add_argument("--compression", default="mean_pool",
                        choices=list(COMPRESSION_REGISTRY.keys()))
    parser.add_argument("--mlp_hidden", nargs="+", type=int, default=[512, 256])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_windows", type=int, default=4,
                        help="Number of windows for window_pool compression")
    parser.add_argument("--num_queries", type=int, default=4,
                        help="Number of queries for attention compression")
    parser.add_argument("--attn_compress_dim", type=int, default=128)
    parser.add_argument("--attn_num_heads", type=int, default=4)
    parser.add_argument("--proj_dim", type=int, default=256,
                        help="Projection dim for linear_proj compression")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--warmup_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="checkpoints/benchmark_router")
    parser.add_argument("--wandb_project", default="benchmark-router")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--debias", action="store_true",
                        help="Enable Product-of-Experts debiasing with a frozen bias model")
    parser.add_argument("--debias_alpha", type=float, default=1.0,
                        help="Scaling factor for bias logit subtraction")
    parser.add_argument("--debias_epochs", type=int, default=100,
                        help="Epochs for pretraining the bias model")
    parser.add_argument("--adversarial_debias", action="store_true",
                        help="Enable adversarial debiasing via gradient reversal on logits")
    parser.add_argument("--adv_lambda", type=float, default=1.0,
                        help="Gradient reversal scale (adversary strength)")
    parser.add_argument("--adv_lr", type=float, default=1e-3,
                        help="Learning rate for the adversary network")
    parser.add_argument("--adv_hidden", nargs="+", type=int, default=[64, 64],
                        help="Hidden layer dims for the adversary MLP")
    parser.add_argument("--eval_routing", action="store_true",
                        help="[STUB] Run actual model routing evaluation (not yet implemented)")
    parser.add_argument("--sequences_json", default=None,
                        help="JSON file mapping benchmark names to optimal sequences")
    parser.add_argument("--full_pass", action="store_true",
                        help="Run end-to-end generation evaluation at validation time")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct",
                        help="HuggingFace model for full-pass generation")
    parser.add_argument("--full_pass_samples", type=int, default=100,
                        help="Samples per benchmark for full-pass eval (use -1 for max available)")
    parser.add_argument("--full_pass_sample_seed", type=int, default=43,
                        help="Random seed for drawing val subset (43 = different from train seed 42)")
    parser.add_argument("--full_pass_every", type=int, default=1,
                        help="Run full-pass eval every N epochs")
    parser.add_argument("--balance_classes", action="store_true",
                        help="Balanced sampling: each epoch samples equal count per class")
    parser.add_argument("--max_per_benchmark", type=int, default=None,
                        help="Cap samples per benchmark (avoids CPU RAM OOM with large embeddings)")
    parser.add_argument("--eval_only", action="store_true",
                        help="Load checkpoint and run full-pass evaluation only (no training)")
    parser.add_argument("--checkpoint_path", default=None,
                        help="Path to checkpoint for eval_only (default: output_dir/checkpoint_best.pt)")
    parser.add_argument("--save_results", default=None,
                        help="Save full-pass results to JSON (for visualization)")
    parser.add_argument("--skip_train_routing_acc", action="store_true",
                        help="Skip train routing accuracy eval (saves ~1h when using save_results)")
    parser.add_argument("--eval_after_train_only", action="store_true",
                        help="Skip validation during training; run eval only after the final epoch")
    parser.add_argument("--exclude_benchmarks", nargs="+", default=None,
                        help="Benchmarks to exclude from full-pass eval (e.g. bigbench_all bigbench_boolean_expressions)")
    parser.add_argument("--mcts_seed", type=int, default=None,
                        help="MCTS shuffle seed — used to reconstruct seen samples for leak-free MMLU eval")
    parser.add_argument("--mcts_n_used", type=int, default=None,
                        help="Number of samples the MCTS search consumed from the shuffled pool")
    parser.add_argument(
        "--auto_external_llm_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After training, run full-pass LLM eval on best checkpoint and log llm_eval/* to W&B.",
    )
    args = parser.parse_args()

    # Flatten comma-separated benchmarks: --benchmarks a,b c -> [a, b, c]
    benchmarks_list: List[str] = []
    for b in args.benchmarks:
        benchmarks_list.extend(s.strip() for s in b.split(",") if s.strip())
    if benchmarks_list:
        args.benchmarks = benchmarks_list

    cfg = BenchmarkRouterConfig(
        embedding_dir=args.embedding_dir,
        results_dir=args.results_dir,
        benchmarks=args.benchmarks,
        compression=args.compression,
        mlp_hidden_dims=args.mlp_hidden,
        dropout=args.dropout,
        num_windows=args.num_windows,
        num_queries=args.num_queries,
        attn_compress_dim=args.attn_compress_dim,
        attn_num_heads=args.attn_num_heads,
        proj_dim=args.proj_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        warmup_fraction=args.warmup_fraction,
        seed=args.seed,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        debias=args.debias,
        debias_alpha=args.debias_alpha,
        debias_epochs=args.debias_epochs,
        adversarial_debias=args.adversarial_debias,
        adv_lambda=args.adv_lambda,
        adv_lr=args.adv_lr,
        adv_hidden_dims=args.adv_hidden,
        wandb_enabled=not args.no_wandb,
        eval_routing=args.eval_routing,
        sequences_json=args.sequences_json,
        balance_classes=args.balance_classes,
        max_per_benchmark=args.max_per_benchmark,
        full_pass=args.full_pass,
        model_name=args.model_name,
        full_pass_samples=args.full_pass_samples,
        full_pass_sample_seed=args.full_pass_sample_seed,
        full_pass_every=args.full_pass_every,
        eval_only=args.eval_only,
        checkpoint_path=args.checkpoint_path,
        save_results=args.save_results,
        skip_train_routing_acc=args.skip_train_routing_acc,
        eval_after_train_only=args.eval_after_train_only,
        exclude_benchmarks=args.exclude_benchmarks,
        mcts_seed=args.mcts_seed,
        mcts_n_used=args.mcts_n_used,
        auto_external_llm_eval=bool(args.auto_external_llm_eval),
    )
    if cfg.eval_only:
        run_eval_only(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
