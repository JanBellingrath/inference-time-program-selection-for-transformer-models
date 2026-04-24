#!/usr/bin/env python3
"""Standalone evaluator for the compositional router checkpoint.

Loads a router checkpoint (written by
``training.train_compositional_router``) plus the matching compositional
catalogue and a per-benchmark dense Δ matrix
(``data_prep/dense_reevaluation.py`` output) and reports:

* anchor accuracy (no edits)
* best-fixed-route accuracy across the eval set
* router accuracy (anchor + Δ at argmax of S_q)
* oracle accuracy (anchor + per-question best Δ)
* mean uplift router/oracle, fraction-of-oracle gap closed
* top-k accuracy of the router argmax against the dense gold
  (gold = per-question best Δ row)
* per-edit-length stratified breakdown
* a "router pick" column for downstream inspection

Everything runs batched on the chosen device and shares the same
``LegalCatalogue``/encoder code paths used by training.

Usage::

    python -m evaluation.evaluate_compositional_router \\
        --checkpoint compositional_runs/csqa_unary_dense.pt \\
        --catalogue_dir compositional_runs/csqa_compositional \\
        --benchmark commonsenseqa \\
        --dense_deltas csqa_train=.../dense_deltas_matrix.pt \\
        --split internal_val --output_json eval_csqa_dense.json
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from routers.compositional_router import (
    CompositionalArtifacts,
    CompositionalDataset,
    CompositionalRouter,
    LegalCatalogue,
    build_compressor,
    CompressorConfig,
    load_artifacts,
)
from routers.residual_compressors import pad_sequences

logger = logging.getLogger("evaluate_compositional_router")


# ---------------------------------------------------------------------------
# Checkpoint -> router
# ---------------------------------------------------------------------------


def _build_compressor(d_model: int, cfg: Dict[str, Any]) -> torch.nn.Module:
    return build_compressor(
        CompressorConfig(
            compressor_type=cfg["compressor_type"],
            d_model=d_model,
            d_compress=cfg["compressor_d_compress"],
            n_heads=cfg["compressor_n_heads"],
            n_latent_tokens=cfg["compressor_n_latent"],
            dropout=cfg.get("encoder_dropout", 0.1),
        )
    )


def load_router_from_checkpoint(
    checkpoint_path: _Path,
    artifacts: CompositionalArtifacts,
    device: torch.device,
) -> Tuple[CompositionalRouter, Dict[str, Any], List[str], Dict[str, int]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    benchmarks = payload["benchmarks"]
    bench_to_id = payload.get("bench_to_id") or {b: i for i, b in enumerate(sorted(benchmarks))}

    # Probe d_model from a residual file (cheap)
    info = artifacts.manifest["benchmarks"][benchmarks[0]]
    if cfg.get("use_full_sequence", False):
        residuals = torch.load(info["full_residuals_path"], map_location="cpu", weights_only=False)
        if isinstance(residuals, dict):
            residuals = residuals["residuals"]
        sample = residuals[0]
    else:
        residuals = torch.load(info["pivot_residuals_path"], map_location="cpu", weights_only=True)
        sample = residuals[0]
    d_model = int(sample.shape[-1])

    compressor = _build_compressor(d_model, cfg)
    d_latent = int(cfg["d_latent"])
    router = CompositionalRouter(
        primitives=artifacts.primitives,
        compressor=compressor,
        d=d_latent,
        num_positions=cfg["num_positions"],
        encoder_hidden_dims=cfg["encoder_hidden_dims"],
        dropout=cfg.get("encoder_dropout", 0.1),
        use_id_embedding=cfg.get("use_id_embedding", False),
        edit_hidden_dims=cfg.get("edit_hidden_dims", [d_latent, d_latent]),
        edit_dropout=cfg.get("edit_dropout", 0.1),
        edit_layer_norm_before=cfg.get("edit_layer_norm_before", True),
        edit_layer_norm_after=cfg.get("edit_layer_norm_after", False),
        unary_hidden_dims=cfg.get("unary_hidden_dims", [d_latent, d_latent]),
        unary_dropout=cfg.get("unary_dropout", 0.1),
        freeze_compressor=cfg.get("freeze_compressor", False),
        use_pairs=cfg.get("use_pairs", False),
        pair_hidden_dims=cfg.get("pair_hidden_dims", (96, 96)),
        pair_dropout=cfg.get("pair_dropout", 0.1),
        pair_zero_init=cfg.get("pair_zero_init", True),
        pair_topk_primitives=cfg.get("pair_topk_primitives"),
        use_anchor_bias=cfg.get("use_anchor_bias", False),
    ).to(device)
    router.load_state_dict(payload["model_state_dict"])
    router.eval()
    return router, cfg, benchmarks, bench_to_id, payload


# ---------------------------------------------------------------------------
# Splits & batched scoring
# ---------------------------------------------------------------------------


def _split_indices(n: int, val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_size = max(1, int(n * val_fraction)) if n > 1 else 0
    return perm[val_size:], perm[:val_size]


def _select_indices_from_qids(
    dataset: CompositionalDataset,
    pairs: Sequence[Dict[str, Any]],
    *,
    benchmark_filter: Optional[str] = None,
) -> List[int]:
    """Resolve a list of ``{benchmark, question_id}`` pairs to dataset indices.

    This is the *authoritative* routine used whenever a checkpoint carries
    ``split_qids`` (written by ``training.train_compositional_router``).
    Falls back to the legacy seeded ``randperm`` path below when a
    checkpoint pre-dates this field.
    """
    lookup: Dict[Tuple[str, int], int] = {
        (dataset.benchmark_names[i], int(dataset.question_ids[i])): i
        for i in range(len(dataset))
    }
    out: List[int] = []
    missing = 0
    for p in pairs:
        bench = str(p["benchmark"])
        qid = int(p["question_id"])
        if benchmark_filter is not None and bench != benchmark_filter:
            continue
        idx = lookup.get((bench, qid))
        if idx is None:
            missing += 1
            continue
        out.append(idx)
    if missing:
        logger.warning(
            "split_qids: %d (bench, qid) pairs not found in current dataset "
            "(may have been filtered by dense-delta availability).",
            missing,
        )
    return out


def _select_indices(
    dataset: CompositionalDataset,
    *,
    split: str,
    val_fraction: float,
    seed: int,
    split_qids: Optional[Dict[str, Any]] = None,
    benchmark_filter: Optional[str] = None,
) -> List[int]:
    """Return dataset indices for the requested split.

    When ``split_qids`` is provided (i.e. the checkpoint persisted the
    training split as ``(benchmark, question_id)`` pairs), we resolve
    those directly. This reproduces the *exact* train/val partition the
    router was trained on — even when the evaluator's dataset only
    contains one of several jointly trained benchmarks, where the legacy
    seeded ``randperm`` approach produced a *different* random subset and
    silently leaked train rows into "val".
    """
    n = len(dataset)
    if split == "all":
        return list(range(n))
    if split_qids is not None and split in ("internal_train", "internal_val"):
        key = "train" if split == "internal_train" else "val"
        pairs = split_qids.get(key) or []
        idxs = _select_indices_from_qids(
            dataset, pairs, benchmark_filter=benchmark_filter,
        )
        if not idxs:
            logger.warning(
                "split_qids resolved to 0 records for split=%s benchmark=%s; "
                "falling back to legacy seeded randperm.",
                split, benchmark_filter,
            )
        else:
            return idxs
    # Legacy fallback: reproducible only when the evaluator's dataset has
    # the same size as training's dataset (i.e. single-benchmark training
    # and single-benchmark evaluation).
    train_idx, val_idx = _split_indices(n, val_fraction, seed)
    if split == "internal_train":
        return train_idx
    if split == "internal_val":
        return val_idx
    raise ValueError(f"unknown split: {split}")


@torch.no_grad()
def _score_indices(
    router: CompositionalRouter,
    dataset: CompositionalDataset,
    indices: Sequence[int],
    catalogue: LegalCatalogue,
    lam: float,
    device: torch.device,
    *,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``S`` ``[Q, N]``, ``dense`` ``[Q, N]``, ``anchor`` ``[Q]``.

    ``Q = len(indices)``. Assumes all indices belong to a single
    benchmark (filter upstream).
    """
    if not indices:
        N = catalogue.n_programs
        return (torch.empty(0, N), torch.empty(0, N), torch.empty(0))

    bench_name = dataset.benchmark_names[indices[0]]
    dense_mat = dataset.dense_deltas_per_bench.get(bench_name)
    anchor_util = dataset.anchor_utilities_per_bench.get(bench_name)
    if dense_mat is None or anchor_util is None:
        raise RuntimeError(f"no dense matrix attached for benchmark {bench_name!r}.")

    S_chunks: List[torch.Tensor] = []
    dense_chunks: List[torch.Tensor] = []
    anchor_chunks: List[torch.Tensor] = []
    for start in range(0, len(indices), batch_size):
        chunk = list(indices[start:start + batch_size])
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
        S = router.program_scores(u_q, catalogue, lam, v_q=v_q)
        S_chunks.append(S.detach().cpu())
        dense_chunks.append(torch.stack(
            [dense_mat[dataset.question_ids[i]] for i in chunk], dim=0,
        ))
        anchor_chunks.append(torch.tensor(
            [float(anchor_util[dataset.question_ids[i]].item()) for i in chunk]
        ))
    return (
        torch.cat(S_chunks, dim=0),
        torch.cat(dense_chunks, dim=0),
        torch.cat(anchor_chunks, dim=0),
    )


# ---------------------------------------------------------------------------
# Metric block
# ---------------------------------------------------------------------------


def _topk_acc(S: torch.Tensor, gold: torch.Tensor, k: int) -> float:
    if S.numel() == 0:
        return 0.0
    k = min(k, S.shape[1])
    topk = torch.topk(S, k=k, dim=-1).indices
    hits = (topk == gold.unsqueeze(1)).any(dim=-1)
    return float(hits.float().mean().item())


def _length_stratified(
    delta_router: torch.Tensor,
    pred_lengths: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    unique = torch.unique(pred_lengths).tolist()
    for ell in unique:
        mask = pred_lengths == ell
        n = int(mask.sum().item())
        if n == 0:
            continue
        d = delta_router[mask]
        out[str(int(ell))] = {
            "n_picks": float(n),
            "frac_picks": float(n / pred_lengths.numel()),
            "mean_delta": float(d.mean().item()),
            "median_delta": float(d.median().item()),
            "frac_positive_delta": float((d > 1e-9).float().mean().item()),
        }
    return out


def evaluate(
    router: CompositionalRouter,
    dataset: CompositionalDataset,
    catalogue: LegalCatalogue,
    indices: Sequence[int],
    lam: float,
    device: torch.device,
    *,
    ks: Sequence[int] = (1, 3, 5, 10),
    batch_size: int = 64,
) -> Dict[str, Any]:
    S, dense, anchor = _score_indices(
        router, dataset, indices, catalogue, lam, device, batch_size=batch_size,
    )
    if S.numel() == 0:
        return {"n_eval": 0}

    Q, N = S.shape
    pred = torch.argmax(S, dim=-1)
    gold = torch.argmax(dense, dim=-1)
    delta_router = dense.gather(1, pred.unsqueeze(1)).squeeze(1)
    delta_oracle = dense.gather(1, gold.unsqueeze(1)).squeeze(1)

    anchor_acc = float(anchor.mean().item())
    router_acc = float((anchor + delta_router).mean().item())
    oracle_acc = float((anchor + delta_oracle).mean().item())
    mean_uplift = router_acc - anchor_acc
    oracle_uplift = oracle_acc - anchor_acc
    frac_oracle = mean_uplift / oracle_uplift if oracle_uplift > 1e-9 else 0.0

    # best-fixed-route baseline
    fixed_route_mean_delta = dense.float().mean(dim=0)            # [N]
    best_fixed_idx = int(torch.argmax(fixed_route_mean_delta).item())
    best_fixed_uplift = float(fixed_route_mean_delta[best_fixed_idx].item())
    best_fixed_acc = anchor_acc + best_fixed_uplift

    # router picks: how concentrated, how often == anchor (length-0 program)
    pred_lengths = catalogue.lengths.detach().cpu().to(torch.float32).gather(0, pred)
    frac_pred_anchor = float((pred_lengths == 0).float().mean().item())
    n_unique_picks = int(torch.unique(pred).numel())

    # router pick distribution top-5
    bincount = torch.bincount(pred, minlength=N).float()
    top_routes = torch.topk(bincount, k=min(5, N))
    top_routes_list = [
        {
            "route_idx": int(idx.item()),
            "n_picked": int(cnt.item()),
            "frac_picked": float(cnt.item() / Q),
            "fixed_route_mean_delta": float(fixed_route_mean_delta[int(idx.item())].item()),
            "length": int(catalogue.lengths[int(idx.item())].item()),
        }
        for idx, cnt in zip(top_routes.indices, top_routes.values)
    ]

    metrics = {
        "n_eval": int(Q),
        "n_routes": int(N),
        # accuracy block
        "anchor_acc": anchor_acc,
        "router_acc": router_acc,
        "oracle_acc": oracle_acc,
        "best_fixed_acc": best_fixed_acc,
        "best_fixed_route_idx": best_fixed_idx,
        "mean_uplift": mean_uplift,
        "oracle_uplift": oracle_uplift,
        "best_fixed_uplift": best_fixed_uplift,
        "frac_oracle_gap_closed": frac_oracle,
        # ranking against dense gold
        **{f"dense_top{k}_acc": _topk_acc(S, gold, k) for k in ks},
        # router behaviour
        "frac_router_picks_anchor": frac_pred_anchor,
        "n_unique_router_picks": n_unique_picks,
        "top_router_picks": top_routes_list,
        # by edit length
        "by_pred_length": _length_stratified(delta_router, pred_lengths),
    }
    # also break out per-quartile of |Δ_oracle| for context
    # (how much headroom do we leave on the hardest-to-help questions?)
    sorted_oracle = torch.sort(delta_oracle)[0]
    qs = [0.25, 0.5, 0.75, 0.95]
    metrics["oracle_uplift_quantiles"] = {
        f"q{int(q * 100)}": float(sorted_oracle[int(q * (Q - 1))].item()) for q in qs
    }
    metrics["router_uplift_quantiles"] = {
        f"q{int(q * 100)}": float(torch.sort(delta_router)[0][int(q * (Q - 1))].item())
        for q in qs
    }
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_dense_deltas(entries: Iterable[str]) -> Dict[str, _Path]:
    out: Dict[str, _Path] = {}
    for e in entries:
        if "=" not in e:
            raise SystemExit(f"--dense_deltas expects bench=path, got {e!r}")
        bench, path = e.split("=", 1)
        p = _Path(path)
        if not p.is_file():
            raise SystemExit(f"dense delta file not found: {p}")
        out[bench] = p
    return out


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=_Path)
    p.add_argument("--catalogue_dir", required=True, type=_Path)
    p.add_argument("--benchmark", required=True,
                   help="Single benchmark to evaluate (e.g., commonsenseqa).")
    p.add_argument("--dense_deltas", nargs="+", required=True,
                   help="bench=path entries pointing at dense_deltas_matrix.pt files.")
    p.add_argument("--split", default="internal_val",
                   choices=["internal_val", "internal_train", "all"],
                   help="Which split of the dataset to evaluate on.")
    p.add_argument("--val_fraction", type=float, default=0.15,
                   help="Must match the value used during training to recover val_idx.")
    p.add_argument("--seed", type=int, default=42,
                   help="Must match the seed used during training to recover val_idx.")
    p.add_argument("--lam", type=float, default=None,
                   help="Override λ (default: read from checkpoint config).")
    p.add_argument("--use_full_sequence", action="store_true",
                   help="Override checkpoint config (rarely needed).")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument("--output_json", type=_Path, default=None,
                   help="Where to dump the metrics JSON. Defaults to stdout-only.")
    p.add_argument("--log_level", default="INFO")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dense_paths = _parse_dense_deltas(args.dense_deltas)
    if args.benchmark not in dense_paths:
        raise SystemExit(
            f"--benchmark {args.benchmark!r} has no matching --dense_deltas entry."
        )

    artifacts = load_artifacts(args.catalogue_dir, benchmarks=[args.benchmark])
    if args.benchmark not in artifacts.catalogues:
        raise SystemExit(
            f"benchmark {args.benchmark!r} missing from catalogue_dir manifest."
        )

    router, cfg, benchmarks, bench_to_id, payload = load_router_from_checkpoint(
        args.checkpoint, artifacts, device,
    )
    lam = float(args.lam) if args.lam is not None else float(cfg.get("lam", 0.0))
    use_full_sequence = bool(args.use_full_sequence or cfg.get("use_full_sequence", False))

    dataset = CompositionalDataset(
        artifacts,
        benchmarks=[args.benchmark],
        use_full_sequence=use_full_sequence,
        bench_to_id=bench_to_id,
        dense_delta_paths={args.benchmark: dense_paths[args.benchmark]},
    )
    if len(dataset) == 0:
        raise SystemExit("dataset is empty after dense filter; aborting.")
    catalogue = artifacts.catalogues[args.benchmark].to(device)
    if router.use_pairs:
        router.attach_pair_features([catalogue])

    split_qids = payload.get("split_qids")
    if split_qids is None and len(benchmarks) > 1 and args.split != "all":
        logger.warning(
            "Checkpoint does not carry split_qids (pre-fix payload) but was "
            "trained jointly on %d benchmarks (%s); the legacy seeded "
            "randperm will NOT reproduce the training split on a single "
            "benchmark and may leak training rows into --split internal_val. "
            "Consider retraining to get a reproducible split.",
            len(benchmarks), benchmarks,
        )
    indices = _select_indices(
        dataset, split=args.split, val_fraction=args.val_fraction, seed=args.seed,
        split_qids=split_qids, benchmark_filter=args.benchmark,
    )
    logger.info(
        "Evaluating %d / %d records (split=%s, lam=%g, use_pairs=%s)",
        len(indices), len(dataset), args.split, lam, router.use_pairs,
    )
    metrics = evaluate(
        router, dataset, catalogue, indices, lam, device, batch_size=args.batch_size,
    )
    metrics["meta"] = {
        "checkpoint": str(args.checkpoint),
        "catalogue_dir": str(args.catalogue_dir),
        "benchmark": args.benchmark,
        "split": args.split,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "lam": lam,
        "use_pairs": router.use_pairs,
        "use_dense_supervision": cfg.get("use_dense_supervision", False),
        "use_full_sequence": use_full_sequence,
        "n_total_records": len(dataset),
    }
    print(json.dumps(metrics, indent=2, sort_keys=False, default=float))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2, default=float))
        logger.info("wrote %s", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
