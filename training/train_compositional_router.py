#!/usr/bin/env python3
"""Train the first-order compositional router (Step 2).

Reads precomputed artifacts from
:mod:`data_prep.build_compositional_catalogues`, plus the residuals stored
in the source canonical directory, and trains a router whose primitive
scores compose into program scores via the per-benchmark incidence matrix.

Two scopes:

* ``--scope single``  -- one router per benchmark.  Writes
  ``compositional_router_best_{benchmark}.pt`` per benchmark.
* ``--scope joint``   -- one router shared across benchmarks; each batch
  item carries a ``benchmark_id`` selecting the per-benchmark
  ``A``/``ℓ``.  Writes ``compositional_router_best_joint.pt``.

Usage::

    python -m training.train_compositional_router \\
        --catalogue_dir fine_routing_data/<run>_compositional \\
        --scope single --benchmarks boolq commonsenseqa
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import os
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

try:
    import wandb

    HAS_WANDB = True
except ImportError:  # pragma: no cover
    HAS_WANDB = False
    wandb = None  # type: ignore

from routers.residual_compressors import pad_sequences
from routers.compositional_router import (
    CompositionalArtifacts,
    CompositionalDataset,
    CompositionalRouter,
    CompressorConfig,
    LegalCatalogue,
    build_compressor,
    collate_compositional,
    hard_ce_on_observed,
    load_artifacts,
    local_moebius_loss,
    softmax_ce_on_observed,
)

logger = logging.getLogger("train_compositional_router")


# ---------------------------------------------------------------------------
# Catalogue placement helpers
# ---------------------------------------------------------------------------


def _move_catalogues(
    catalogues: Dict[str, LegalCatalogue], device: torch.device,
) -> Dict[str, LegalCatalogue]:
    return {b: c.to(device) for b, c in catalogues.items()}


# ---------------------------------------------------------------------------
# Encoder construction
# ---------------------------------------------------------------------------


def _infer_d_model(dataset: CompositionalDataset) -> int:
    if not dataset.encoder_inputs:
        raise ValueError("dataset is empty; cannot infer d_model.")
    sample = dataset.encoder_inputs[0]
    return int(sample.shape[-1])


def _build_compressor(
    compressor_type: str,
    d_model: int,
    *,
    d_compress: int,
    n_heads: int,
    n_latent: int,
    dropout: float,
) -> nn.Module:
    cfg = CompressorConfig(
        compressor_type=compressor_type,
        d_model=d_model,
        d_compress=d_compress,
        n_heads=n_heads,
        n_latent_tokens=n_latent,
        dropout=dropout,
    )
    return build_compressor(cfg)


# ---------------------------------------------------------------------------
# Training step + epoch
# ---------------------------------------------------------------------------


def _compute_program_scores_per_benchmark(
    g_q: torch.Tensor,
    u_q: torch.Tensor,
    benchmark_ids: torch.Tensor,
    bench_id_to_catalogue: Dict[int, LegalCatalogue],
    lam: float,
    obs_indices: torch.Tensor,
    *,
    router: CompositionalRouter,
    use_pairs: bool,
    return_components: bool = False,
) -> Dict[str, torch.Tensor]:
    """Gather ``S_q[obs_indices]`` per sample using its benchmark's ``(A, B, ℓ)``.

    Returns a dict with at least ``gathered`` of shape ``[B, K]``. If
    ``return_components`` is true, also returns the gathered unary and pair
    contributions for diagnostics.
    """
    B, K = obs_indices.shape
    gathered = torch.empty(B, K, dtype=u_q.dtype, device=u_q.device)
    if return_components:
        unary_g = torch.zeros(B, K, dtype=u_q.dtype, device=u_q.device)
        pair_g = torch.zeros(B, K, dtype=u_q.dtype, device=u_q.device)
    # Aggregate the L2 penalty in element space (sum of squares + element
    # count) rather than as a mean of per-benchmark means. The latter
    # under-weights large benchmarks in joint training and makes the
    # reported "pair_l2_mean" depend on how many benchmarks happen to land
    # in a given batch. The true mean-squared pair score over all ``v``
    # elements seen in the batch is ``sum_sq / count``.
    pair_l2_sum_sq: Optional[torch.Tensor] = None
    pair_l2_count_n: int = 0

    unique_ids = torch.unique(benchmark_ids).tolist()
    for bid in unique_ids:
        catalogue = bench_id_to_catalogue[int(bid)]
        sample_mask = benchmark_ids == bid
        sample_idx = sample_mask.nonzero(as_tuple=False).squeeze(-1)
        if sample_idx.numel() == 0:
            continue
        u_sub = u_q.index_select(0, sample_idx)               # [B', M]
        unary = (
            torch.sparse.mm(catalogue.A, u_sub.t()).t()
            if catalogue.A.is_sparse else u_sub @ catalogue.A.t()
        )                                                     # [B', N_b]
        S_sub = unary - lam * catalogue.lengths.unsqueeze(0)
        v_sub: Optional[torch.Tensor] = None
        pair_contrib: Optional[torch.Tensor] = None
        if use_pairs and catalogue.has_pairs:
            g_sub = g_q.index_select(0, sample_idx)
            if router.pair_topk_primitives is not None:
                v_sub = router.pair_scores_from_g_topk(
                    g_sub, u_sub, catalogue, router.pair_topk_primitives,
                )
            else:
                v_sub = router.pair_scores_from_g(g_sub, catalogue)
            if v_sub is not None and v_sub.numel() > 0:
                pair_contrib = (
                    torch.sparse.mm(catalogue.B, v_sub.t()).t()
                    if catalogue.B.is_sparse else v_sub @ catalogue.B.t()
                )                                              # [B', N_b]
                S_sub = S_sub + pair_contrib
                sq_sum = (v_sub ** 2).sum()
                if pair_l2_sum_sq is None:
                    pair_l2_sum_sq = sq_sum
                else:
                    pair_l2_sum_sq = pair_l2_sum_sq + sq_sum
                pair_l2_count_n += int(v_sub.numel())
        # Anchor bias: row 0 is the empty program by catalogue construction.
        # We mirror what ``router.program_scores`` does so the per-benchmark
        # fast path stays consistent with the whole-catalogue path.
        if router.anchor_bias is not None and S_sub.shape[1] > 0:
            S_sub = S_sub.clone()
            S_sub[:, 0] = S_sub[:, 0] + router.anchor_bias.to(dtype=S_sub.dtype)
        gather_idx = obs_indices.index_select(0, sample_idx).clamp(min=0)
        gathered_sub = S_sub.gather(1, gather_idx)
        gathered.index_copy_(0, sample_idx, gathered_sub)
        if return_components:
            unary_g.index_copy_(0, sample_idx, unary.gather(1, gather_idx))
            if pair_contrib is not None:
                pair_g.index_copy_(0, sample_idx, pair_contrib.gather(1, gather_idx))

    out = {"gathered": gathered}
    if pair_l2_sum_sq is not None and pair_l2_count_n > 0:
        # ``pair_l2_mean`` is the per-batch mean over all individual pair
        # scores; upstream epoch aggregation accumulates the sums and
        # counts separately to avoid biasing by batch count when final
        # batches are partial.
        out["pair_l2_mean"] = pair_l2_sum_sq / float(pair_l2_count_n)
        out["pair_l2_sum_sq"] = pair_l2_sum_sq.detach()
        out["pair_l2_count"] = torch.tensor(
            float(pair_l2_count_n), device=pair_l2_sum_sq.device,
        )
    if return_components:
        out["unary_gathered"] = unary_g
        out["pair_gathered"] = pair_g  # zeros if pairs disabled or empty
    return out


def _loss_on_batch(
    router: CompositionalRouter,
    batch: Dict[str, Any],
    bench_id_to_catalogue: Dict[int, LegalCatalogue],
    lam: float,
    tau: float,
    student_temp: float,
    device: torch.device,
    *,
    use_pairs: bool,
    pair_l2: float = 0.0,
    return_components: bool = False,
    use_local_unary: bool = False,
    use_local_pair: bool = False,
    local_alpha: float = 0.0,
    local_pair_beta: float = 1.0,
    use_hard_ce: bool = False,
) -> Dict[str, torch.Tensor]:
    encoder_input = batch["encoder_input"].to(device)
    attention_mask = batch["attention_mask"]
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    obs_indices = batch["obs_indices"].to(device)
    obs_deltas = batch["obs_deltas"].to(device)
    obs_mask = batch["obs_mask"].to(device)
    benchmark_ids = batch["benchmark_id"].to(device)

    g_q = router.encode(encoder_input, attention_mask)
    u_q = router.primitive_scores_from_g(g_q)
    parts = _compute_program_scores_per_benchmark(
        g_q, u_q, benchmark_ids, bench_id_to_catalogue, lam, obs_indices,
        router=router, use_pairs=use_pairs, return_components=return_components,
    )
    gathered = parts["gathered"]
    K = gathered.shape[1]
    pseudo_indices = torch.arange(K, device=device).unsqueeze(0).expand_as(obs_indices)
    if use_hard_ce:
        ce = hard_ce_on_observed(
            gathered, pseudo_indices, obs_deltas, obs_mask,
            student_temp=student_temp,
        )
    else:
        ce = softmax_ce_on_observed(
            gathered, pseudo_indices, obs_deltas, obs_mask, tau,
            student_temp=student_temp,
        )
    total = ce
    pair_reg = parts.get("pair_l2_mean")
    if use_pairs and pair_l2 > 0 and pair_reg is not None:
        total = total + pair_l2 * pair_reg
    out = {"loss": total, "ce": ce.detach()}
    if pair_reg is not None:
        out["pair_l2_mean"] = pair_reg.detach()
        # Propagate the raw (sum_sq, count) so the epoch aggregator can
        # compute the true element-wise mean over an epoch instead of a
        # biased mean-of-batch-means.
        if "pair_l2_sum_sq" in parts:
            out["pair_l2_sum_sq"] = parts["pair_l2_sum_sq"]
        if "pair_l2_count" in parts:
            out["pair_l2_count"] = parts["pair_l2_count"]

    if (use_local_unary or use_local_pair) and local_alpha > 0:
        local_parts = local_moebius_loss(
            router=router, g_q=g_q, u_q=u_q, batch=batch,
            use_unary=use_local_unary, use_pair=use_local_pair,
            pair_weight=local_pair_beta,
        )
        if local_parts["unary"] is not None:
            out["local_unary"] = local_parts["unary"].detach()
        if local_parts["pair"] is not None:
            out["local_pair"] = local_parts["pair"].detach()
        if local_parts["total"] is not None:
            out["loss"] = out["loss"] + local_alpha * local_parts["total"]

    if return_components:
        out["unary_gathered"] = parts["unary_gathered"]
        out["pair_gathered"] = parts["pair_gathered"]
        out["obs_mask"] = obs_mask
    return out



# ---------------------------------------------------------------------------
# Dense supervision helpers
# ---------------------------------------------------------------------------


def _maybe_swap_to_dense(batch: Dict[str, Any], *, enabled: bool) -> Dict[str, Any]:
    """If a batch carries dense ``Δ`` vectors and ``enabled``, replace
    ``obs_*`` so the existing softmax-CE-on-observed loss normalises over
    **all** legal programs of the (per-batch) benchmark instead of only
    the observed candidate subset.
    """
    if not enabled:
        return batch
    dd = batch.get("dense_deltas")
    if dd is None:
        return batch
    B, N = dd.shape
    arange = torch.arange(N, dtype=torch.long, device=dd.device).unsqueeze(0).expand(B, N)
    mask = torch.ones(B, N, dtype=torch.float32, device=dd.device)
    keep = batch.get("dense_keep_mask")
    if keep is not None:
        mask = mask * keep.to(device=dd.device, dtype=mask.dtype)
    out = dict(batch)
    out["obs_indices"] = arange
    out["obs_deltas"] = dd.float()
    out["obs_mask"] = mask
    return out


@torch.no_grad()
def _downstream_metrics_on_split(
    router: CompositionalRouter,
    dataset: CompositionalDataset,
    indices: Sequence[int],
    bench_id_to_catalogue: Dict[int, LegalCatalogue],
    lam: float,
    device: torch.device,
    *,
    ks: Sequence[int] = (1, 3, 5),
) -> Dict[str, float]:
    """Downstream MC-accuracy metrics from the dense Δ matrix.

    For each question with a loaded dense Δ vector we compute:

    * ``router_acc``    -- mean of ``anchor_acc + Δ[argmax_S_q]``
    * ``oracle_acc``    -- mean of ``anchor_acc + max_r Δ_q(r)``
    * ``anchor_acc``    -- mean of stored ``anchor_utility``
    * ``mean_uplift``   -- ``router_acc − anchor_acc``
    * ``oracle_uplift`` -- ``oracle_acc − anchor_acc``
    * ``frac_oracle``   -- ``mean_uplift / oracle_uplift`` (0 if oracle 0)
    * ``best_fixed_acc``-- best mean across all routes (fixed-route baseline)
    * ``dense_top{k}``  -- top-k accuracy where gold = argmax_r Δ_q(r)
    """
    if not indices:
        empty = {f"dense_top{k}_acc": 0.0 for k in ks}
        empty.update({
            "router_acc": 0.0, "oracle_acc": 0.0, "anchor_acc": 0.0,
            "mean_uplift": 0.0, "oracle_uplift": 0.0, "frac_oracle": 0.0,
            "mean_uplift_pp": 0.0, "oracle_uplift_pp": 0.0,
            "best_fixed_acc": 0.0, "best_fixed_uplift_pp": 0.0, "n_eval": 0.0,
        })
        return empty
    router.eval()
    by_bench: Dict[int, List[int]] = defaultdict(list)
    for i in indices:
        by_bench[int(dataset.benchmark_ids[i])].append(int(i))

    sum_router = 0.0
    sum_oracle = 0.0
    sum_anchor = 0.0
    n_total = 0
    hits = {k: 0 for k in ks}
    # For best-fixed-route baseline we need per-route mean utility across
    # the eval set, accumulated over all benchmarks present.
    fixed_route_sums: Dict[int, torch.Tensor] = {}
    fixed_route_count: Dict[int, int] = {}
    max_k = max(ks)

    for bid, bench_indices in by_bench.items():
        catalogue = bench_id_to_catalogue[bid]
        bench_name = None
        for nm, mapped in dataset.bench_to_id.items():
            if int(mapped) == bid:
                bench_name = nm
                break
        if bench_name is None:
            continue
        dense_mat = dataset.dense_deltas_per_bench.get(bench_name)
        anchor_util = dataset.anchor_utilities_per_bench.get(bench_name)
        if dense_mat is None or anchor_util is None:
            continue
        N_b = int(catalogue.n_programs)
        if bid not in fixed_route_sums:
            fixed_route_sums[bid] = torch.zeros(N_b, dtype=torch.float64)
            fixed_route_count[bid] = 0
        # batched encode
        BATCH = 64
        for start in range(0, len(bench_indices), BATCH):
            chunk = bench_indices[start:start + BATCH]
            if dataset.encoder_inputs[chunk[0]].dim() == 1:
                enc = torch.stack([dataset.encoder_inputs[i] for i in chunk], dim=0).to(device)
                attn = None
            else:
                enc, attn = pad_sequences([dataset.encoder_inputs[i] for i in chunk])
                enc = enc.to(device)
                attn = attn.to(device) if attn is not None else None
            g_q = router.encode(enc, attn)
            u_q = router.primitive_scores_from_g(g_q)
            v_q = router.pair_scores_from_g(g_q, catalogue) if router.use_pairs else None
            S_q = router.program_scores(u_q, catalogue, lam, v_q=v_q)  # [B, N_b]
            dense_chunk = torch.stack(
                [dense_mat[dataset.question_ids[i]] for i in chunk], dim=0,
            ).to(device)
            anchor_chunk = torch.tensor(
                [float(anchor_util[dataset.question_ids[i]].item()) for i in chunk],
                device=device,
            )
            preds = torch.argmax(S_q, dim=-1)            # [B]
            best = torch.argmax(dense_chunk, dim=-1)      # [B]
            d_router = dense_chunk.gather(1, preds.unsqueeze(1)).squeeze(1)
            d_best = dense_chunk.gather(1, best.unsqueeze(1)).squeeze(1)
            sum_router += float((anchor_chunk + d_router).sum().item())
            sum_oracle += float((anchor_chunk + d_best).sum().item())
            sum_anchor += float(anchor_chunk.sum().item())
            n_total += int(preds.numel())
            # top-k against dense gold
            kk = min(max_k, S_q.shape[1])
            topk = torch.topk(S_q, k=kk, dim=-1).indices  # [B, kk]
            for k in ks:
                k_use = min(k, kk)
                hits[k] += int((topk[:, :k_use] == best.unsqueeze(1)).any(dim=-1).sum().item())
            # accumulate per-route mean (for best-fixed baseline)
            fixed_route_sums[bid] += dense_chunk.detach().cpu().double().sum(dim=0)
            fixed_route_count[bid] += int(preds.numel())

    if n_total == 0:
        return {
            "router_acc": 0.0, "oracle_acc": 0.0, "anchor_acc": 0.0,
            "mean_uplift": 0.0, "oracle_uplift": 0.0, "frac_oracle": 0.0,
            "mean_uplift_pp": 0.0, "oracle_uplift_pp": 0.0,
            "best_fixed_acc": 0.0, "best_fixed_uplift_pp": 0.0, "n_eval": 0.0,
            **{f"dense_top{k}_acc": 0.0 for k in ks},
        }
    router_acc = sum_router / n_total
    oracle_acc = sum_oracle / n_total
    anchor_acc = sum_anchor / n_total
    mean_uplift = router_acc - anchor_acc
    oracle_uplift = oracle_acc - anchor_acc
    frac_oracle = mean_uplift / oracle_uplift if oracle_uplift > 1e-9 else 0.0
    # best-fixed-route across the eval set
    best_fixed_uplift = 0.0
    if fixed_route_sums:
        # weighted mean per benchmark, then take per-bench best (single benchmark
        # in the typical single-scope case gives the obvious answer).
        per_bench_best = []
        per_bench_n = []
        for bid, sums in fixed_route_sums.items():
            if fixed_route_count[bid] == 0:
                continue
            mean_per_route = sums / float(fixed_route_count[bid])
            per_bench_best.append(float(mean_per_route.max().item()))
            per_bench_n.append(fixed_route_count[bid])
        if per_bench_best:
            tot = sum(per_bench_n)
            best_fixed_uplift = sum(b * n for b, n in zip(per_bench_best, per_bench_n)) / tot
    out = {
        "router_acc": router_acc,
        "oracle_acc": oracle_acc,
        "anchor_acc": anchor_acc,
        "mean_uplift": mean_uplift,
        "oracle_uplift": oracle_uplift,
        "mean_uplift_pp": 100.0 * mean_uplift,
        "oracle_uplift_pp": 100.0 * oracle_uplift,
        "frac_oracle": frac_oracle,
        "best_fixed_uplift": best_fixed_uplift,
        "best_fixed_uplift_pp": 100.0 * best_fixed_uplift,
        "best_fixed_acc": anchor_acc + best_fixed_uplift,
        "n_eval": float(n_total),
    }
    for k in ks:
        out[f"dense_top{k}_acc"] = hits[k] / n_total
    return out



def _epoch(
    router: CompositionalRouter,
    loader: DataLoader,
    bench_id_to_catalogue: Dict[int, LegalCatalogue],
    lam: float,
    tau: float,
    student_temp: float,
    device: torch.device,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    use_pairs: bool = False,
    pair_l2: float = 0.0,
    use_dense_supervision: bool = False,
    use_local_unary: bool = False,
    use_local_pair: bool = False,
    local_alpha: float = 0.0,
    local_pair_beta: float = 1.0,
    use_hard_ce: bool = False,
) -> Dict[str, float]:
    is_train = optimizer is not None
    router.train(mode=is_train)
    total_loss = 0.0
    total_ce = 0.0
    # Accumulate sum_sq and count across batches; reporting the ratio at
    # epoch end gives the true mean-squared pair score over the epoch
    # regardless of variable batch sizes / per-benchmark pair counts.
    total_pair_l2_sum_sq = 0.0
    total_pair_l2_count = 0
    sum_unary_abs = 0.0
    sum_pair_abs = 0.0
    n_components_batches = 0
    total_count = 0
    sum_local_u = 0.0
    n_local_u = 0
    sum_local_p = 0.0
    n_local_p = 0
    for batch in loader:
        batch = _maybe_swap_to_dense(batch, enabled=use_dense_supervision)
        out = _loss_on_batch(
            router, batch, bench_id_to_catalogue, lam, tau, student_temp, device,
            use_pairs=use_pairs, pair_l2=pair_l2, return_components=use_pairs,
            use_local_unary=use_local_unary, use_local_pair=use_local_pair,
            local_alpha=local_alpha, local_pair_beta=local_pair_beta,
            use_hard_ce=use_hard_ce,
        )
        loss = out["loss"]
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        bs = int(batch["benchmark_id"].numel())
        total_loss += float(loss.item()) * bs
        total_ce += float(out["ce"].item()) * bs
        if "pair_l2_sum_sq" in out and "pair_l2_count" in out:
            total_pair_l2_sum_sq += float(out["pair_l2_sum_sq"].item())
            total_pair_l2_count += int(out["pair_l2_count"].item())
        if "local_unary" in out:
            sum_local_u += float(out["local_unary"].item())
            n_local_u += 1
        if "local_pair" in out:
            sum_local_p += float(out["local_pair"].item())
            n_local_p += 1
        if use_pairs and "unary_gathered" in out:
            mask = out["obs_mask"]
            denom = mask.sum().clamp(min=1.0)
            u_abs = (out["unary_gathered"].abs() * mask).sum() / denom
            p_abs = (out["pair_gathered"].abs() * mask).sum() / denom
            sum_unary_abs += float(u_abs.item())
            sum_pair_abs += float(p_abs.item())
            n_components_batches += 1
        total_count += bs
    metrics = {
        "loss": total_loss / max(total_count, 1),
        "ce": total_ce / max(total_count, 1),
    }
    if total_pair_l2_count > 0:
        metrics["pair_l2_mean"] = total_pair_l2_sum_sq / float(total_pair_l2_count)
    if n_local_u > 0:
        metrics["local_unary"] = sum_local_u / n_local_u
    if n_local_p > 0:
        metrics["local_pair"] = sum_local_p / n_local_p
    if n_components_batches > 0:
        metrics["mean_abs_unary_contrib"] = sum_unary_abs / n_components_batches
        metrics["mean_abs_pair_contrib"] = sum_pair_abs / n_components_batches
    return metrics


# ---------------------------------------------------------------------------
# End-of-training delta evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def _delta_metrics_on_split(
    router: CompositionalRouter,
    dataset: CompositionalDataset,
    indices: Sequence[int],
    bench_id_to_catalogue: Dict[int, LegalCatalogue],
    lam: float,
    device: torch.device,
) -> Dict[str, float]:
    """Per-sample argmax over the *observed* support: average observed Δ.

    This is a cheap proxy that does not need full-LM evaluation: we measure
    how often the router's argmax over the legal catalogue lands on (or
    near) the highest-Δ observed program for that question.  ``mean_argmax_delta``
    is the average observed Δ at the predicted program when that program
    appears in the observed support, and 0 otherwise (i.e. an out-of-support
    pick is treated neutrally).
    """
    if not indices:
        return {
            "mean_argmax_delta": 0.0,
            "mean_best_observed": 0.0,
            "frac_argmax_in_support": 0.0,
            "frac_argmax_positive": 0.0,
        }
    router.eval()
    sum_pred = 0.0
    sum_best = 0.0
    n_in_support = 0
    n_positive = 0
    n_total = 0
    for idx in indices:
        sample = dataset[idx]
        encoder_input = sample["encoder_input"].unsqueeze(0).to(device)
        catalogue = bench_id_to_catalogue[int(sample["benchmark_id"])]
        g_q = router.encode(encoder_input)
        u_q = router.primitive_scores_from_g(g_q)
        v_q = router.pair_scores_from_g(g_q, catalogue) if router.use_pairs else None
        S_q = router.program_scores(u_q, catalogue, lam, v_q=v_q).squeeze(0)
        pred = int(torch.argmax(S_q).item())
        obs_indices = sample["obs_indices"].tolist()
        obs_deltas = sample["obs_deltas"].tolist()
        delta_lookup = dict(zip(obs_indices, obs_deltas))
        d_pred = float(delta_lookup.get(pred, 0.0))
        sum_pred += d_pred
        sum_best += float(max(obs_deltas)) if obs_deltas else 0.0
        if pred in delta_lookup:
            n_in_support += 1
        if d_pred > 1e-8:
            n_positive += 1
        n_total += 1
    return {
        "mean_argmax_delta": sum_pred / n_total,
        "mean_best_observed": sum_best / n_total,
        "frac_argmax_in_support": n_in_support / n_total,
        "frac_argmax_positive": n_positive / n_total,
    }


@torch.no_grad()
def _ranking_metrics_on_split(
    router: CompositionalRouter,
    dataset: CompositionalDataset,
    indices: Sequence[int],
    bench_id_to_catalogue: Dict[int, LegalCatalogue],
    lam: float,
    device: torch.device,
    *,
    ks: Sequence[int] = (1, 3, 5, 10),
    unary_only: bool = False,
) -> Dict[str, float]:
    """Hard label = observed row with largest Δ; pred = argmax of ``S_q``.

    Reports global top-``k`` (over all legal programs) and observed-only
    top-``k`` (re-ranking restricted to the observed candidate set, an oracle
    support proxy that isolates the model's ability to *order* observed
    programs from its ability to discover new ones).

    Set ``unary_only=True`` to score using ``A u_q − λ ℓ`` even when the
    router has a pair head; useful for ablation on the same checkpoint.
    """
    if not indices:
        empty = {f"top{k}_acc": 0.0 for k in ks}
        empty.update({f"obs_top{k}_acc": 0.0 for k in ks})
        return empty
    router.eval()
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    obs_hits = {k: 0 for k in ks}
    n_total = 0
    for idx in indices:
        sample = dataset[idx]
        encoder_input = sample["encoder_input"].unsqueeze(0).to(device)
        catalogue = bench_id_to_catalogue[int(sample["benchmark_id"])]
        obs_indices_list = sample["obs_indices"].tolist()
        obs_deltas = sample["obs_deltas"].tolist()
        if not obs_indices_list:
            continue
        best_pos = max(range(len(obs_deltas)), key=lambda j: obs_deltas[j])
        gold_row = int(obs_indices_list[best_pos])

        g_q = router.encode(encoder_input)
        u_q = router.primitive_scores_from_g(g_q)
        v_q = (
            router.pair_scores_from_g(g_q, catalogue)
            if (router.use_pairs and not unary_only) else None
        )
        S_q = router.program_scores(u_q, catalogue, lam, v_q=v_q).squeeze(0)

        _, topk_rows = torch.topk(S_q, k=min(max_k, S_q.numel()))
        topk_rows = topk_rows.tolist()
        for kk in ks:
            take = min(kk, len(topk_rows))
            if gold_row in topk_rows[:take]:
                hits[kk] += 1

        # Observed-only re-rank: order obs_indices by S_q and check rank of gold.
        obs_idx_t = torch.tensor(obs_indices_list, dtype=torch.long, device=S_q.device)
        obs_scores = S_q.index_select(0, obs_idx_t)
        order = torch.argsort(obs_scores, descending=True).tolist()
        ranked_rows = [obs_indices_list[i] for i in order]
        for kk in ks:
            take = min(kk, len(ranked_rows))
            if gold_row in ranked_rows[:take]:
                obs_hits[kk] += 1
        n_total += 1
    out: Dict[str, float] = {f"top{k}_acc": hits[k] / max(n_total, 1) for k in ks}
    out.update({f"obs_top{k}_acc": obs_hits[k] / max(n_total, 1) for k in ks})
    out["n_eval"] = float(n_total)
    return out


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def _split_indices(n: int, val_fraction: float, seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_size = max(1, int(n * val_fraction)) if n > 1 else 0
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    return train_idx, val_idx


def _carve_test_from_train(
    train_idx: Sequence[int],
    *,
    holdout_count: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Deterministically carve a test holdout from the train split."""
    train_idx = list(train_idx)
    if holdout_count <= 0 or not train_idx:
        return train_idx, []
    max_take = max(0, len(train_idx) - 1)
    take = min(int(holdout_count), max_take)
    if take <= 0:
        return train_idx, []
    g = torch.Generator().manual_seed(int(seed) * 1009 + 7)
    perm = torch.randperm(len(train_idx), generator=g).tolist()
    test_pick = set(perm[:take])
    test_idx = [train_idx[i] for i in range(len(train_idx)) if i in test_pick]
    kept_train = [train_idx[i] for i in range(len(train_idx)) if i not in test_pick]
    return kept_train, test_idx


def train_one_router(
    artifacts: CompositionalArtifacts,
    *,
    benchmarks: Sequence[str],
    output_path: _Path,
    compressor_type: str,
    compressor_d_compress: int,
    compressor_n_heads: int,
    compressor_n_latent: int,
    encoder_hidden_dims: Sequence[int],
    encoder_dropout: float,
    freeze_compressor: bool,
    d_latent: int,
    use_id_embedding: bool,
    edit_hidden_dims: Sequence[int],
    edit_dropout: float,
    edit_layer_norm_before: bool,
    edit_layer_norm_after: bool,
    unary_hidden_dims: Sequence[int],
    unary_dropout: float,
    lam: float,
    tau: float,
    student_temp: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    val_fraction: float,
    seed: int,
    device: torch.device,
    use_full_sequence: bool = False,
    wandb_run: Optional[Any] = None,
    wandb_prefix: str = "",
    wandb_step_offset: int = 0,
    use_pairs: bool = False,
    pair_hidden_dims: Sequence[int] = (96, 96),
    pair_dropout: float = 0.1,
    pair_zero_init: bool = True,
    pair_l2: float = 0.0,
    pair_topk_primitives: Optional[int] = None,
    dense_delta_paths: Optional[Dict[str, _Path]] = None,
    use_dense_supervision: bool = False,
    downstream_eval_every: int = 1,
    downstream_eval_subset: int = 0,
    early_stopping_patience: int = 0,
    observed_path_overrides: Optional[Dict[str, _Path]] = None,
    dense_keep_mask_paths: Optional[Dict[str, _Path]] = None,
    local_moebius_paths: Optional[Dict[str, _Path]] = None,
    use_local_unary: bool = False,
    use_local_pair: bool = False,
    local_alpha: float = 0.0,
    local_pair_beta: float = 1.0,
    checkpoint_metric: str = "loss",
    use_anchor_bias: bool = False,
    epoch_report_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
    train_test_holdout_count: int = 0,
    split_json_path: Optional[_Path] = None,
    use_hard_ce: bool = False,
) -> Optional[Dict[str, Any]]:
    torch.manual_seed(seed)
    log_prefix = f"{wandb_prefix}/" if wandb_prefix else ""
    # Monotonic W&B x-axis in multi-trial (single-run) HPO: parent passes
    # trial_idx * stride so per-epoch steps never go backwards when a new trial
    # resets epoch to 1.
    wb = int(wandb_step_offset)

    bench_to_id = {b: i for i, b in enumerate(sorted(benchmarks))}
    dense_delta_paths = dict(dense_delta_paths or {})
    local_moebius_paths = dict(local_moebius_paths or {})
    dataset = CompositionalDataset(
        artifacts,
        benchmarks=list(benchmarks),
        use_full_sequence=use_full_sequence,
        bench_to_id=bench_to_id,
        dense_delta_paths=dense_delta_paths if use_dense_supervision or dense_delta_paths else None,
        observed_path_overrides=observed_path_overrides,
        dense_keep_mask_paths=dense_keep_mask_paths,
        local_moebius_paths=local_moebius_paths or None,
    )
    has_dense = any(
        dataset.dense_deltas_per_bench.get(b) is not None for b in benchmarks
    )
    if use_dense_supervision and not has_dense:
        raise SystemExit(
            "--use_dense_supervision requires --dense_deltas bench=path for at least one benchmark."
        )
    if len(dataset) == 0:
        logger.error("no samples for benchmarks=%s; aborting", benchmarks)
        return None

    if split_json_path is not None:
        split_json_path = _Path(split_json_path)
        with open(split_json_path) as _f:
            _split_doc = json.load(_f)
        _train_keys: set = set()
        _val_keys: set = set()
        _test_keys: set = set()
        for _b, _info in _split_doc.get("benchmarks", {}).items():
            for _q in _info.get("train_question_ids", []):
                _train_keys.add((_b, int(_q)))
            for _q in _info.get("val_question_ids", []):
                _val_keys.add((_b, int(_q)))
            for _q in _info.get("test_question_ids", []):
                _test_keys.add((_b, int(_q)))
        train_idx: List[int] = []
        val_idx: List[int] = []
        test_idx: List[int] = []
        _unmapped = 0
        for _i in range(len(dataset)):
            _b = dataset.benchmark_names[_i]
            _q = int(dataset.question_ids[_i])
            _key = (_b, _q)
            if _key in _train_keys:
                train_idx.append(_i)
            elif _key in _val_keys:
                val_idx.append(_i)
            elif _key in _test_keys:
                test_idx.append(_i)
            else:
                _unmapped += 1
        if _unmapped:
            logger.warning(
                "split_json=%s: %d dataset rows had (bench, qid) not in any split "
                "— excluded from train/val/test.", split_json_path, _unmapped,
            )
        logger.info(
            "[split_json] %s -> train=%d val=%d test=%d (of %d)",
            split_json_path, len(train_idx), len(val_idx), len(test_idx), len(dataset),
        )
    else:
        train_idx, val_idx = _split_indices(len(dataset), val_fraction, seed)
        train_idx, test_idx = _carve_test_from_train(
            train_idx,
            holdout_count=int(train_test_holdout_count),
            seed=seed,
        )
    if test_idx:
        logger.info(
            "train/test carve-out: train=%d val=%d test=%d (holdout_count=%d, val unchanged)",
            len(train_idx), len(val_idx), len(test_idx), int(train_test_holdout_count),
        )
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx) if val_idx else None

    # Optionally cap the *downstream* eval set to a fixed random subsample.
    # Dense downstream metrics (``_downstream_metrics_on_split``) are the
    # single most expensive thing in the epoch loop — each call runs a full
    # router forward (including the pair MLP) on every question. For
    # checkpoint selection we only need a reliable estimate of the true
    # metric, so a few hundred held-out questions are plenty. The subsample
    # is pinned by a seed so it is identical across all epochs / reruns
    # with the same ``seed``, giving a smooth monotone view of progress.
    eval_train_idx: Sequence[int] = train_idx
    eval_val_idx: Sequence[int] = val_idx
    eval_test_idx: Sequence[int] = test_idx
    if downstream_eval_subset and downstream_eval_subset > 0:
        gen = torch.Generator().manual_seed(int(seed) * 997 + 13)
        if len(train_idx) > downstream_eval_subset:
            perm = torch.randperm(len(train_idx), generator=gen).tolist()
            eval_train_idx = [train_idx[i] for i in perm[:downstream_eval_subset]]
        if len(val_idx) > downstream_eval_subset:
            perm = torch.randperm(len(val_idx), generator=gen).tolist()
            eval_val_idx = [val_idx[i] for i in perm[:downstream_eval_subset]]
        if len(test_idx) > downstream_eval_subset:
            perm = torch.randperm(len(test_idx), generator=gen).tolist()
            eval_test_idx = [test_idx[i] for i in perm[:downstream_eval_subset]]
        logger.info(
            "downstream eval subsample: train=%d/%d  val=%d/%d  test=%d/%d  (seed-pinned)",
            len(eval_train_idx), len(train_idx),
            len(eval_val_idx), len(val_idx),
            len(eval_test_idx), len(test_idx),
        )

    # Capture the (benchmark, question_id) pair for every index on each
    # side of the split. Question IDs are stable across dataset
    # reconstructions (they are the row's own ``question_id`` field), so
    # persisting these pairs lets the evaluator reproduce the *exact*
    # held-out set — including when the router is trained jointly across
    # multiple benchmarks but evaluated on a single benchmark (the index
    # universe changes from ``sum_b N_b`` to ``N_bench`` and the old
    # ``torch.randperm`` approach picks a different subset, silently
    # leaking train rows into "val"). The evaluator falls back to the
    # legacy seeded randperm if ``split_qids`` is not present in the
    # checkpoint (backward compatibility with older payloads).
    def _pairs_for(ixs: Sequence[int]) -> List[Dict[str, Any]]:
        return [
            {
                "benchmark": dataset.benchmark_names[i],
                "question_id": int(dataset.question_ids[i]),
            }
            for i in ixs
        ]

    split_qids = {
        "val_fraction": float(val_fraction),
        "seed": int(seed),
        "n_total": int(len(dataset)),
        "train": _pairs_for(train_idx),
        "val": _pairs_for(val_idx),
        "test": _pairs_for(test_idx),
    }

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_compositional, drop_last=False,
    )
    val_loader = (
        DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                   collate_fn=collate_compositional, drop_last=False)
        if val_subset is not None else None
    )

    d_model = _infer_d_model(dataset)
    compressor = _build_compressor(
        compressor_type, d_model,
        d_compress=compressor_d_compress,
        n_heads=compressor_n_heads,
        n_latent=compressor_n_latent,
        dropout=encoder_dropout,
    )
    num_positions = artifacts.num_layers
    router = CompositionalRouter(
        primitives=artifacts.primitives,
        compressor=compressor,
        d=d_latent,
        num_positions=num_positions,
        encoder_hidden_dims=encoder_hidden_dims,
        dropout=encoder_dropout,
        use_id_embedding=use_id_embedding,
        edit_hidden_dims=edit_hidden_dims,
        edit_dropout=edit_dropout,
        edit_layer_norm_before=edit_layer_norm_before,
        edit_layer_norm_after=edit_layer_norm_after,
        unary_hidden_dims=unary_hidden_dims,
        unary_dropout=unary_dropout,
        freeze_compressor=freeze_compressor,
        use_pairs=use_pairs,
        pair_hidden_dims=pair_hidden_dims,
        pair_dropout=pair_dropout,
        pair_zero_init=pair_zero_init,
        pair_topk_primitives=pair_topk_primitives,
        use_anchor_bias=use_anchor_bias,
    ).to(device)

    catalogues_on_device = _move_catalogues(
        {b: artifacts.catalogues[b] for b in benchmarks if b in artifacts.catalogues},
        device,
    )
    if use_pairs:
        router.attach_pair_features(catalogues_on_device.values())
    bench_id_to_catalogue = {bench_to_id[b]: c for b, c in catalogues_on_device.items()}

    n_params = sum(p.numel() for p in router.parameters() if p.requires_grad)
    logger.info(
        "Router built: M=%d, |benchmarks|=%d, samples=%d (train=%d val=%d test=%d), params=%d",
        artifacts.num_primitives, len(catalogues_on_device), len(dataset),
        len(train_subset), len(val_subset) if val_subset is not None else 0, len(test_idx), n_params,
    )

    optimizer = torch.optim.AdamW(
        [p for p in router.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_val_loss_min = float("inf")
    best_epoch = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, float]] = []
    best_ckpt_score = float("inf") if checkpoint_metric == "loss" else float("-inf")
    epochs_since_improve = 0

    for epoch in range(1, epochs + 1):
        train_metrics = _epoch(
            router, train_loader, bench_id_to_catalogue, lam, tau, student_temp, device,
            optimizer=optimizer, use_pairs=use_pairs, pair_l2=pair_l2,
            use_dense_supervision=use_dense_supervision,
            use_local_unary=use_local_unary, use_local_pair=use_local_pair,
            local_alpha=local_alpha, local_pair_beta=local_pair_beta,
            use_hard_ce=use_hard_ce,
        )
        train_loss = train_metrics["loss"]
        if val_loader is not None:
            val_metrics_e = _epoch(
                router, val_loader, bench_id_to_catalogue, lam, tau, student_temp, device,
                optimizer=None, use_pairs=use_pairs, pair_l2=pair_l2,
                use_dense_supervision=use_dense_supervision,
                use_local_unary=use_local_unary, use_local_pair=use_local_pair,
                local_alpha=local_alpha, local_pair_beta=local_pair_beta,
                use_hard_ce=use_hard_ce,
            )
            val_loss = val_metrics_e["loss"]
        else:
            val_metrics_e = train_metrics
            val_loss = train_loss
        scheduler.step()
        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "val_loss": val_loss,
            "train_ce": train_metrics["ce"], "val_ce": val_metrics_e["ce"],
        })
        improved = False
        current_ckpt_score = float("nan")
        if checkpoint_metric == "loss":
            current_ckpt_score = float(val_loss)
            if val_loss < best_ckpt_score:
                best_ckpt_score = val_loss
                improved = True
        elif checkpoint_metric == "mean_uplift":
            if not has_dense:
                raise SystemExit("--checkpoint_metric mean_uplift requires dense Δ matrices loaded.")
            dm_val_ckpt = _downstream_metrics_on_split(
                router, dataset, eval_val_idx, bench_id_to_catalogue, lam, device,
            )
            sc = float(dm_val_ckpt["mean_uplift"])
            current_ckpt_score = sc
            if sc > best_ckpt_score:
                best_ckpt_score = sc
                improved = True
        elif checkpoint_metric == "dense_top1":
            if not has_dense:
                raise SystemExit("--checkpoint_metric dense_top1 requires dense Δ matrices loaded.")
            dm_val_ckpt = _downstream_metrics_on_split(
                router, dataset, eval_val_idx, bench_id_to_catalogue, lam, device,
            )
            sc = float(dm_val_ckpt["dense_top1_acc"])
            current_ckpt_score = sc
            if sc > best_ckpt_score:
                best_ckpt_score = sc
                improved = True
        elif checkpoint_metric == "obs_top1":
            rk_val = _ranking_metrics_on_split(
                router, dataset, val_idx, bench_id_to_catalogue, lam, device,
            )
            sc = float(rk_val["obs_top1_acc"])
            current_ckpt_score = sc
            if sc > best_ckpt_score:
                best_ckpt_score = sc
                improved = True
        else:
            raise SystemExit(f"unknown --checkpoint_metric {checkpoint_metric!r}")

        if epoch_report_callback is not None:
            epoch_report_callback(
                int(epoch),
                {
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "checkpoint_score": float(current_ckpt_score),
                    "best_checkpoint_score": float(best_ckpt_score),
                },
            )

        best_val_loss_min = min(best_val_loss_min, val_loss)
        if improved:
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in router.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        if wandb_run is not None:
            payload = {
                f"{log_prefix}train/loss": train_loss,
                f"{log_prefix}train/ce": train_metrics["ce"],
                f"{log_prefix}val/loss": val_loss,
                f"{log_prefix}val/ce": val_metrics_e["ce"],
                f"{log_prefix}val/loss_best": best_val_loss_min,
                f"{log_prefix}val/checkpoint_best_score": best_ckpt_score,
                f"{log_prefix}epoch": epoch,
                f"{log_prefix}lr": optimizer.param_groups[0]["lr"],
            }
            for split, mset in [("train", train_metrics), ("val", val_metrics_e)]:
                for key in (
                    "pair_l2_mean", "mean_abs_unary_contrib", "mean_abs_pair_contrib",
                    "local_unary", "local_pair",
                ):
                    if key in mset:
                        payload[f"{log_prefix}{split}/{key}"] = mset[key]
            wandb_run.log(payload, step=wb + epoch)
        if (
            has_dense
            and downstream_eval_every > 0
            and (epoch % downstream_eval_every == 0 or epoch == epochs)
        ):
            dm_train = _downstream_metrics_on_split(
                router, dataset, eval_train_idx, bench_id_to_catalogue, lam, device,
            )
            dm_val = _downstream_metrics_on_split(
                router, dataset, eval_val_idx, bench_id_to_catalogue, lam, device,
            )
            logger.info(
                "Downstream  train: router=%.4f anchor=%.4f oracle=%.4f uplift=%+.4f frac_oracle=%.3f  "
                "val: router=%.4f anchor=%.4f oracle=%.4f uplift=%+.4f frac_oracle=%.3f",
                dm_train["router_acc"], dm_train["anchor_acc"], dm_train["oracle_acc"],
                dm_train["mean_uplift"], dm_train["frac_oracle"],
                dm_val["router_acc"], dm_val["anchor_acc"], dm_val["oracle_acc"],
                dm_val["mean_uplift"], dm_val["frac_oracle"],
            )
            if wandb_run is not None:
                dpayload = {}
                for split, m in [("train", dm_train), ("val", dm_val)]:
                    for k, v in m.items():
                        dpayload[f"{log_prefix}downstream/{split}/{k}"] = v
                dpayload[f"{log_prefix}epoch"] = epoch
                wandb_run.log(dpayload, step=wb + epoch)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            extra = ""
            if "mean_abs_pair_contrib" in val_metrics_e:
                extra = "  |u|=%.3f |v|=%.3f" % (
                    val_metrics_e["mean_abs_unary_contrib"],
                    val_metrics_e["mean_abs_pair_contrib"],
                )
            logger.info(
                "Epoch %3d  train=%.4f  val=%.4f  (best_ep=%d  min_val_loss=%.4f  ckpt_%s=%.6f)%s",
                epoch, train_loss, val_loss, best_epoch, best_val_loss_min,
                checkpoint_metric, best_ckpt_score, extra,
            )
        # Early stopping on the checkpoint metric. ``patience`` is the number
        # of consecutive epochs with no improvement in ``best_ckpt_score``
        # before we cut the run short. Running past best+patience is pure
        # waste when the model has clearly plateaued / is overfitting.
        if (
            early_stopping_patience > 0
            and epochs_since_improve >= early_stopping_patience
            and best_epoch > 0
        ):
            logger.info(
                "Early stop at epoch %d (no %s improvement for %d epochs; best was epoch %d, score=%.6f)",
                epoch, checkpoint_metric, epochs_since_improve, best_epoch, best_ckpt_score,
            )
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in router.state_dict().items()}
    router.load_state_dict(best_state)

    train_metrics = _delta_metrics_on_split(router, dataset, train_idx, bench_id_to_catalogue,
                                            lam, device)
    val_metrics = _delta_metrics_on_split(router, dataset, val_idx, bench_id_to_catalogue,
                                          lam, device)
    has_test_holdout = len(test_idx) > 0
    test_metrics: Optional[Dict[str, float]] = None
    if has_test_holdout:
        test_metrics = _delta_metrics_on_split(
            router, dataset, test_idx, bench_id_to_catalogue, lam, device,
        )
    train_rank = _ranking_metrics_on_split(router, dataset, train_idx, bench_id_to_catalogue,
                                           lam, device)
    val_rank = _ranking_metrics_on_split(router, dataset, val_idx, bench_id_to_catalogue,
                                         lam, device)
    test_rank: Optional[Dict[str, float]] = None
    if has_test_holdout:
        test_rank = _ranking_metrics_on_split(
            router, dataset, test_idx, bench_id_to_catalogue, lam, device,
        )
    train_downstream: Optional[Dict[str, float]] = None
    val_downstream: Optional[Dict[str, float]] = None
    test_downstream: Optional[Dict[str, float]] = None
    if has_dense:
        train_downstream = _downstream_metrics_on_split(
            router, dataset, train_idx, bench_id_to_catalogue, lam, device,
        )
        val_downstream = _downstream_metrics_on_split(
            router, dataset, val_idx, bench_id_to_catalogue, lam, device,
        )
        if has_test_holdout:
            test_downstream = _downstream_metrics_on_split(
                router, dataset, test_idx, bench_id_to_catalogue, lam, device,
            )
            logger.info(
                "Final downstream  train: router_acc=%.4f anchor=%.4f oracle=%.4f uplift=%+.4f  "
                "val: router_acc=%.4f anchor=%.4f oracle=%.4f uplift=%+.4f best_fixed=%.4f  "
                "test: router_acc=%.4f anchor=%.4f oracle=%.4f uplift=%+.4f best_fixed=%.4f",
                train_downstream["router_acc"], train_downstream["anchor_acc"],
                train_downstream["oracle_acc"], train_downstream["mean_uplift"],
                val_downstream["router_acc"], val_downstream["anchor_acc"],
                val_downstream["oracle_acc"], val_downstream["mean_uplift"],
                val_downstream["best_fixed_acc"],
                test_downstream["router_acc"], test_downstream["anchor_acc"],
                test_downstream["oracle_acc"], test_downstream["mean_uplift"],
                test_downstream["best_fixed_acc"],
            )
        else:
            logger.info(
                "Final downstream  train: router_acc=%.4f anchor=%.4f oracle=%.4f uplift=%+.4f  "
                "val: router_acc=%.4f anchor=%.4f oracle=%.4f uplift=%+.4f best_fixed=%.4f",
                train_downstream["router_acc"], train_downstream["anchor_acc"],
                train_downstream["oracle_acc"], train_downstream["mean_uplift"],
                val_downstream["router_acc"], val_downstream["anchor_acc"],
                val_downstream["oracle_acc"], val_downstream["mean_uplift"],
                val_downstream["best_fixed_acc"],
            )
    train_rank_unary: Optional[Dict[str, float]] = None
    val_rank_unary: Optional[Dict[str, float]] = None
    if use_pairs:
        train_rank_unary = _ranking_metrics_on_split(
            router, dataset, train_idx, bench_id_to_catalogue, lam, device, unary_only=True,
        )
        val_rank_unary = _ranking_metrics_on_split(
            router, dataset, val_idx, bench_id_to_catalogue, lam, device, unary_only=True,
        )
    logger.info(
        "Δ-on-legal-set train: argmax=%+.5f  best_obs=%+.5f  in_supp=%.3f  pos=%.3f",
        train_metrics["mean_argmax_delta"], train_metrics["mean_best_observed"],
        train_metrics["frac_argmax_in_support"], train_metrics["frac_argmax_positive"],
    )
    logger.info(
        "Δ-on-legal-set   val: argmax=%+.5f  best_obs=%+.5f  in_supp=%.3f  pos=%.3f",
        val_metrics["mean_argmax_delta"], val_metrics["mean_best_observed"],
        val_metrics["frac_argmax_in_support"], val_metrics["frac_argmax_positive"],
    )
    if has_test_holdout and test_metrics is not None:
        logger.info(
            "Δ-on-legal-set  test: argmax=%+.5f  best_obs=%+.5f  in_supp=%.3f  pos=%.3f",
            test_metrics["mean_argmax_delta"], test_metrics["mean_best_observed"],
            test_metrics["frac_argmax_in_support"], test_metrics["frac_argmax_positive"],
        )
    logger.info(
        "Ranking (gold = argmax Δ among observed)  train: top1=%.4f top3=%.4f top5=%.4f  "
        "val: top1=%.4f top3=%.4f top5=%.4f",
        train_rank["top1_acc"], train_rank["top3_acc"], train_rank["top5_acc"],
        val_rank["top1_acc"], val_rank["top3_acc"], val_rank["top5_acc"],
    )
    if has_test_holdout and test_rank is not None:
        logger.info(
            "Obs-only re-rank  train: top1=%.4f top3=%.4f top5=%.4f  "
            "val: top1=%.4f top3=%.4f top5=%.4f  test: top1=%.4f top3=%.4f top5=%.4f",
            train_rank["obs_top1_acc"], train_rank["obs_top3_acc"], train_rank["obs_top5_acc"],
            val_rank["obs_top1_acc"], val_rank["obs_top3_acc"], val_rank["obs_top5_acc"],
            test_rank["obs_top1_acc"], test_rank["obs_top3_acc"], test_rank["obs_top5_acc"],
        )
    else:
        logger.info(
            "Obs-only re-rank  train: top1=%.4f top3=%.4f top5=%.4f  "
            "val: top1=%.4f top3=%.4f top5=%.4f",
            train_rank["obs_top1_acc"], train_rank["obs_top3_acc"], train_rank["obs_top5_acc"],
            val_rank["obs_top1_acc"], val_rank["obs_top3_acc"], val_rank["obs_top5_acc"],
        )
    if val_rank_unary is not None:
        logger.info(
            "Unary-only ablation  val: top1=%.4f top3=%.4f top5=%.4f  obs_top1=%.4f obs_top3=%.4f",
            val_rank_unary["top1_acc"], val_rank_unary["top3_acc"], val_rank_unary["top5_acc"],
            val_rank_unary["obs_top1_acc"], val_rank_unary["obs_top3_acc"],
        )
    if wandb_run is not None:
        flat = {}
        split_metric_rows = [("train", train_metrics), ("val", val_metrics)]
        if has_test_holdout and test_metrics is not None:
            split_metric_rows.append(("test", test_metrics))
        for split, m in split_metric_rows:
            for k, v in m.items():
                flat[f"{log_prefix}delta/{split}/{k}"] = v
        split_rank_rows = [("train", train_rank), ("val", val_rank)]
        if has_test_holdout and test_rank is not None:
            split_rank_rows.append(("test", test_rank))
        for split, m in split_rank_rows:
            for k, v in m.items():
                flat[f"{log_prefix}rank/{split}/{k}"] = v
        if train_rank_unary is not None:
            for split, m in [("train", train_rank_unary), ("val", val_rank_unary)]:
                for k, v in m.items():
                    flat[f"{log_prefix}rank_unary_only/{split}/{k}"] = v
        if val_downstream is not None:
            split_down_rows = [("train", train_downstream), ("val", val_downstream)]
            if has_test_holdout and test_downstream is not None:
                split_down_rows.append(("test", test_downstream))
            for split, m in split_down_rows:
                if m is None:
                    continue
                for k, v in m.items():
                    flat[f"{log_prefix}downstream/{split}/{k}"] = v
        flat[f"{log_prefix}best_epoch"] = best_epoch
        flat[f"{log_prefix}best_val_loss"] = best_val_loss_min
        flat[f"{log_prefix}best_checkpoint_metric"] = checkpoint_metric
        flat[f"{log_prefix}best_checkpoint_score"] = best_ckpt_score
        wandb_run.log(flat, step=wb + epochs)

    payload = {
        "model_state_dict": best_state,
        "config": {
            "d_latent": d_latent,
            "num_positions": num_positions,
            "compressor_type": compressor_type,
            "compressor_d_compress": compressor_d_compress,
            "compressor_n_heads": compressor_n_heads,
            "compressor_n_latent": compressor_n_latent,
            "encoder_hidden_dims": list(encoder_hidden_dims),
            "encoder_dropout": encoder_dropout,
            "freeze_compressor": freeze_compressor,
            "use_id_embedding": use_id_embedding,
            "edit_hidden_dims": list(edit_hidden_dims),
            "edit_dropout": edit_dropout,
            "edit_layer_norm_before": edit_layer_norm_before,
            "edit_layer_norm_after": edit_layer_norm_after,
            "unary_hidden_dims": list(unary_hidden_dims),
            "unary_dropout": unary_dropout,
            "use_full_sequence": use_full_sequence,
            "lam": lam,
            "tau": tau,
            "student_temp": student_temp,
            "use_pairs": use_pairs,
            "pair_hidden_dims": list(pair_hidden_dims),
            "pair_dropout": pair_dropout,
            "pair_zero_init": pair_zero_init,
            "pair_l2": pair_l2,
            "pair_topk_primitives": pair_topk_primitives,
            "use_local_unary": use_local_unary,
            "use_local_pair": use_local_pair,
            "local_alpha": local_alpha,
            "local_pair_beta": local_pair_beta,
            "checkpoint_metric": checkpoint_metric,
            "use_anchor_bias": use_anchor_bias,
            "use_hard_ce": use_hard_ce,
        },
        "benchmarks": list(benchmarks),
        "bench_to_id": bench_to_id,
        "split_qids": split_qids,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss_min,
        "best_checkpoint_metric": checkpoint_metric,
        "best_checkpoint_score": best_ckpt_score,
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "train_ranking": train_rank,
            "val_ranking": val_rank,
            "test_ranking": test_rank,
            "train_ranking_unary_only": train_rank_unary,
            "val_ranking_unary_only": val_rank_unary,
            "train_downstream": train_downstream,
            "val_downstream": val_downstream,
            "test_downstream": test_downstream,
            "test": test_metrics,
        },
        "catalogue_dir": str(artifacts.output_dir),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    logger.info(
        "checkpoint -> %s (best epoch %d, min_val_loss %.4f, ckpt_metric=%s score=%.6f)",
        output_path, best_epoch, best_val_loss_min, checkpoint_metric, best_ckpt_score,
    )
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalogue_dir", required=True, type=_Path,
                   help="Output dir of build_compositional_catalogues.")
    p.add_argument("--output_dir", required=True, type=_Path,
                   help="Where to write checkpoints.")
    p.add_argument("--scope", choices=["single", "joint"], default="single")
    p.add_argument("--benchmarks", nargs="*", default=None)

    p.add_argument("--compressor_type", choices=["last_token", "top_down_attention"],
                   default="last_token")
    p.add_argument("--compressor_d_compress", type=int, default=256)
    p.add_argument("--compressor_n_heads", type=int, default=4)
    p.add_argument("--compressor_n_latent", type=int, default=1)
    p.add_argument("--encoder_hidden_dims", nargs="*", type=int, default=[])
    p.add_argument("--encoder_dropout", type=float, default=0.1)
    p.add_argument("--freeze_compressor", action="store_true")

    p.add_argument("--d_latent", type=int, default=128)
    p.add_argument("--use_id_embedding", action="store_true",
                   help="Enable per-primitive ID embedding (default: off; "
                        "edit MLP relies on type/arg1/arg2 only).")
    p.add_argument("--edit_hidden_dims", nargs="*", type=int, default=None,
                   help="Hidden layer sizes of the edit-side MLP "
                        "(default: [d_latent, d_latent]).")
    p.add_argument("--edit_dropout", type=float, default=0.1)
    p.add_argument("--no_edit_layer_norm_before", action="store_true",
                   help="Disable LayerNorm on raw symbolic embedding before edit MLP.")
    p.add_argument("--edit_layer_norm_after", action="store_true",
                   help="Apply LayerNorm on the edit MLP output.")
    p.add_argument("--unary_hidden_dims", nargs="*", type=int, default=None,
                   help="Hidden layer sizes of the unary scorer MLP "
                        "(default: [d_latent, d_latent]).")
    p.add_argument("--unary_dropout", type=float, default=0.1)

    p.add_argument("--lam", type=float, default=0.0,
                   help="Per-edit complexity cost in the program score.")
    p.add_argument("--tau", type=float, default=1.0,
                   help="Supervisor softmax temperature on observed deltas.")
    p.add_argument("--student_temperature", type=float, default=1.0,
                   help="Divides model logits before log_softmax in CE; <1 sharper, >1 flatter.")

    p.add_argument("--use_pairs", action="store_true",
                   help="Enable the symmetric pairwise interaction head.")
    p.add_argument("--pair_hidden_dims", nargs="*", type=int, default=[96, 96],
                   help="Hidden layer sizes of the pair MLP (kept small on purpose).")
    p.add_argument("--pair_dropout", type=float, default=0.1)
    p.add_argument("--pair_no_zero_init", action="store_true",
                   help="Disable zero-init of the pair MLP last layer.")
    p.add_argument("--pair_l2", type=float, default=1e-3,
                   help="L2 penalty on per-pair v_q output magnitude (correction-style).")
    p.add_argument("--pair_topk_primitives", type=int, default=None,
                   help="If set, only score legal pairs whose endpoints are "
                        "both within the per-question top-K primitives by "
                        "unary score; full unary scoring is preserved.")

    p.add_argument("--use_local_unary", action="store_true",
                   help="Enable auxiliary unary local Möbius MSE supervision.")
    p.add_argument("--use_local_pair", action="store_true",
                   help="Enable auxiliary pair local Möbius MSE supervision.")
    p.add_argument("--local_alpha", type=float, default=1.0,
                   help="Weight on the combined local Möbius loss.")
    p.add_argument("--local_pair_beta", type=float, default=1.0,
                   help="Relative weight of pair vs unary inside the local loss.")
    p.add_argument("--local_moebius_dir", default=None,
                   help="Directory of {bench}.pt files holding sparse local "
                        "Möbius targets (singleton_qid/idx/target, pair_qid/i/j/target).")

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--split_json",
        type=_Path,
        default=None,
        help="Optional train/val/test question ids (from scripts/make_canonical_split.py).",
    )
    p.add_argument(
        "--train_test_holdout_count",
        type=int,
        default=0,
        help="Only if --split_json omitted: test questions carved from train.",
    )

    p.add_argument("--use_full_sequence", action="store_true",
                   help="Load full-sequence residuals (required by top_down_attention).")
    p.add_argument("--dense_deltas", nargs="*", default=None,
                   help="bench=path entries for dense delta matrices (one per benchmark).")
    p.add_argument("--observed_dir", default=None,
                   help="Override observed/{bench}.jsonl with files in this directory "
                        "(useful for held-out compositional generalization training).")
    p.add_argument("--dense_keep_mask_dir", default=None,
                   help="Directory of {bench}.pt files holding a keep_mask tensor [N_b]; "
                        "applied to obs_mask during dense supervision.")
    p.add_argument("--use_dense_supervision", action="store_true",
                   help="When dense deltas are loaded, supervise with full-N softmax CE.")
    p.add_argument("--downstream_eval_every", type=int, default=1,
                   help="Run dense downstream eval every N epochs (1 = each epoch; 0 disables).")
    p.add_argument("--downstream_eval_subset", type=int, default=0,
                   help="If >0, cap dense downstream eval to this many randomly "
                        "sampled questions per split (seed-pinned, stable across "
                        "epochs). Applies to per-epoch checkpoint eval and to the "
                        "periodic train+val logging eval. 0 means use the full split.")
    p.add_argument("--early_stopping_patience", type=int, default=0,
                   help="Stop training when the checkpoint metric has not improved "
                        "for this many consecutive epochs. 0 disables early stopping.")
    p.add_argument(
        "--checkpoint_metric",
        choices=["loss", "mean_uplift", "dense_top1", "obs_top1"],
        default="loss",
        help="Which validation signal selects the saved checkpoint (default: val CE loss).",
    )
    p.add_argument(
        "--use_anchor_bias", action="store_true",
        help="Add a learnable scalar to the anchor (row-0) program score. "
             "Breaks the structural pinning S_q(anchor)≡0 and lets CE place "
             "the anchor logit freely on the same scale as non-anchor programs.",
    )
    p.add_argument(
        "--hard_ce", action="store_true",
        help="Use hard cross-entropy on the observed candidate support "
             "(target = one-hot on argmax observed Δ) instead of the "
             "tempered soft distribution controlled by --tau. Intended for "
             "MCTS-only supervision where each question has a small handful "
             "of observed routes.",
    )
    p.add_argument("--device", default=None,
                   help="Override device (default: cuda if available, else cpu).")
    p.add_argument("--log_level", default="INFO")

    p.add_argument("--wandb", action="store_true",
                   help="Log training/eval metrics to Weights & Biases.")
    p.add_argument("--wandb_project", default="compositional-router")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_mode", default="online",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_tags", nargs="*", default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    logger.info("argv: %s", _sys.argv)
    if args.compressor_type != "last_token" and not args.use_full_sequence:
        raise SystemExit("--compressor_type top_down_attention requires --use_full_sequence")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    artifacts = load_artifacts(args.catalogue_dir, benchmarks=args.benchmarks)
    if not artifacts.catalogues:
        raise SystemExit("no benchmarks selected from artifacts")

    dense_paths: Dict[str, _Path] = {}
    for entry in (args.dense_deltas or []):
        if "=" not in entry:
            raise SystemExit(f"--dense_deltas expects bench=path entries, got {entry!r}")
        bench, path = entry.split("=", 1)
        dense_paths[bench] = _Path(path)
        if not dense_paths[bench].is_file():
            raise SystemExit(f"dense delta file not found: {dense_paths[bench]}")

    observed_overrides: Dict[str, _Path] = {}
    if args.observed_dir is not None:
        obs_dir = _Path(args.observed_dir)
        if not obs_dir.is_dir():
            raise SystemExit(f"--observed_dir not a directory: {obs_dir}")
        for bench in artifacts.benchmarks:
            cand = obs_dir / f"{bench}.jsonl"
            if cand.is_file():
                observed_overrides[bench] = cand
            else:
                logger.warning("[%s] observed override missing: %s", bench, cand)

    dense_keep_paths: Dict[str, _Path] = {}
    if args.dense_keep_mask_dir is not None:
        mdir = _Path(args.dense_keep_mask_dir)
        if not mdir.is_dir():
            raise SystemExit(f"--dense_keep_mask_dir not a directory: {mdir}")
        for bench in artifacts.benchmarks:
            cand = mdir / f"{bench}.pt"
            if cand.is_file():
                dense_keep_paths[bench] = cand

    local_moebius_paths: Dict[str, _Path] = {}
    if args.local_moebius_dir is not None:
        ldir = _Path(args.local_moebius_dir)
        if not ldir.is_dir():
            raise SystemExit(f"--local_moebius_dir not a directory: {ldir}")
        for bench in artifacts.benchmarks:
            cand = ldir / f"{bench}.pt"
            if not cand.is_file():
                alt = ldir / f"local_moebius_{bench}.pt"
                if alt.is_file():
                    cand = alt
            if cand.is_file():
                local_moebius_paths[bench] = cand
    if (args.use_local_unary or args.use_local_pair) and not local_moebius_paths:
        raise SystemExit(
            "--use_local_unary/--use_local_pair require --local_moebius_dir "
            "with at least one matching {bench}.pt file."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if args.wandb:
        if not HAS_WANDB:
            raise SystemExit("wandb requested but not installed.")
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            tags=args.wandb_tags,
            config={
                **{k: v for k, v in vars(args).items() if k != "wandb"},
                "manifest_M": artifacts.num_primitives,
                "manifest_benchmarks": artifacts.benchmarks,
                "manifest_n_legal": {
                    b: c.n_programs for b, c in artifacts.catalogues.items()
                },
                "device": str(device),
            },
        )

    edit_hidden_dims = (
        list(args.edit_hidden_dims)
        if args.edit_hidden_dims is not None
        else [args.d_latent, args.d_latent]
    )
    unary_hidden_dims = (
        list(args.unary_hidden_dims)
        if args.unary_hidden_dims is not None
        else [args.d_latent, args.d_latent]
    )
    common = dict(
        compressor_type=args.compressor_type,
        compressor_d_compress=args.compressor_d_compress,
        compressor_n_heads=args.compressor_n_heads,
        compressor_n_latent=args.compressor_n_latent,
        encoder_hidden_dims=args.encoder_hidden_dims,
        encoder_dropout=args.encoder_dropout,
        freeze_compressor=args.freeze_compressor,
        d_latent=args.d_latent,
        use_id_embedding=bool(args.use_id_embedding),
        edit_hidden_dims=edit_hidden_dims,
        edit_dropout=args.edit_dropout,
        edit_layer_norm_before=not args.no_edit_layer_norm_before,
        edit_layer_norm_after=bool(args.edit_layer_norm_after),
        unary_hidden_dims=unary_hidden_dims,
        unary_dropout=args.unary_dropout,
        lam=args.lam,
        tau=args.tau,
        student_temp=args.student_temperature,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=device,
        use_full_sequence=args.use_full_sequence,
        use_pairs=args.use_pairs,
        pair_hidden_dims=args.pair_hidden_dims,
        pair_dropout=args.pair_dropout,
        pair_zero_init=not args.pair_no_zero_init,
        pair_l2=args.pair_l2,
        pair_topk_primitives=args.pair_topk_primitives,
        dense_delta_paths=dense_paths,
        use_dense_supervision=bool(args.use_dense_supervision),
        downstream_eval_every=int(args.downstream_eval_every),
        downstream_eval_subset=int(args.downstream_eval_subset),
        early_stopping_patience=int(args.early_stopping_patience),
        observed_path_overrides=observed_overrides,
        dense_keep_mask_paths=dense_keep_paths,
        local_moebius_paths=local_moebius_paths,
        use_local_unary=bool(args.use_local_unary),
        use_local_pair=bool(args.use_local_pair),
        local_alpha=float(args.local_alpha),
        local_pair_beta=float(args.local_pair_beta),
        checkpoint_metric=str(args.checkpoint_metric),
        use_anchor_bias=bool(args.use_anchor_bias),
        use_hard_ce=bool(args.hard_ce),
        split_json_path=args.split_json,
        train_test_holdout_count=int(args.train_test_holdout_count),
    )

    try:
        if args.scope == "joint":
            out_path = args.output_dir / "compositional_router_best_joint.pt"
            train_one_router(
                artifacts,
                benchmarks=artifacts.benchmarks,
                output_path=out_path,
                wandb_run=wandb_run,
                wandb_prefix="joint",
                **common,
            )
        else:
            for bench in artifacts.benchmarks:
                out_path = args.output_dir / f"compositional_router_best_{bench}.pt"
                logger.info("=== single-benchmark run: %s ===", bench)
                train_one_router(
                    artifacts,
                    benchmarks=[bench],
                    output_path=out_path,
                    wandb_run=wandb_run,
                    wandb_prefix=bench,
                    **common,
                )
    finally:
        if wandb_run is not None:
            try:
                from training.auto_external_llm_eval import write_wandb_run_info

                write_wandb_run_info(args.output_dir, wandb_run)
            except Exception:
                logger.warning("write_wandb_run_info failed.", exc_info=True)
            wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
