#!/usr/bin/env python3
"""Cross-benchmark router evaluation (frozen, no adaptation).

Loads a single trained router checkpoint and evaluates it against an
arbitrary list of target benchmark artefacts, comparing the router-selected
program against the loaded normal-module-sequence baseline and (when
available) the dense / observed oracle.

This is **pure transfer evaluation**:

* the router is frozen
* no retraining, no fine-tuning, no test-time adaptation
* the script does not assume in-domain evaluation; benchmarks may be the
  router's training set, OOD targets, or a mix.

The script supports two router architectures out of the box:

* ``compositional`` -- :class:`routers.compositional_router.CompositionalRouter`
* ``flat``          -- :class:`routers.flat_program_router.FlatProgramRouter`

Both go through a small :class:`_RouterAdapter` so the rest of the script
is router-agnostic.

Usage::

    python -m evaluation.evaluate_router_cross_benchmark \\
        --checkpoint compositional_runs/csqa_unary_dense.pt \\
        --catalogue_dir compositional_runs/csqa_compositional \\
        --benchmark commonsenseqa hellaswag winogrande \\
        --dense_deltas commonsenseqa=.../dense_deltas_matrix.pt \\
        --baseline_file baselines.json \\
        --output_dir predictions/cross_bench_csqa_router \\
        --plots --per_question_csv

Note:
    ``FlatProgramRouter`` only scores benchmarks whose linear head was
    trained into the checkpoint (``router.heads[bench]`` must exist).
    OOD benchmarks without a head are listed in the ``skipped`` section
    of the output JSON; they are *not* a hard error.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import csv
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from routers.compositional_router import (
    CompositionalArtifacts,
    CompositionalDataset,
    CompositionalRouter,
    LegalCatalogue,
    build_compressor,
    CompressorConfig,
    load_artifacts,
)
from routers.flat_program_router import FlatProgramRouter
from routers.residual_compressors import pad_sequences

logger = logging.getLogger("evaluate_router_cross_benchmark")


# ============================================================================
# 1. Router loading -- registry pattern
# ============================================================================


def _build_compressor_from_cfg(d_model: int, cfg: Dict[str, Any]) -> torch.nn.Module:
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


def _probe_d_model(artifacts: CompositionalArtifacts, bench: str, use_full_sequence: bool) -> int:
    info = artifacts.manifest["benchmarks"][bench]
    if use_full_sequence:
        residuals = torch.load(info["full_residuals_path"], map_location="cpu", weights_only=False)
        if isinstance(residuals, dict):
            residuals = residuals["residuals"]
        sample = residuals[0]
    else:
        residuals = torch.load(info["pivot_residuals_path"], map_location="cpu", weights_only=True)
        sample = residuals[0]
    return int(sample.shape[-1])


def _detect_router_type(payload: Dict[str, Any]) -> str:
    """Best-effort architecture detection from a checkpoint payload."""
    kind = payload.get("model_kind")
    if kind == "flat_program_router":
        return "flat"
    cfg = payload.get("config", {})
    if "bench_n_programs" in cfg:
        return "flat"
    sd = payload.get("state_dict") or payload.get("model_state_dict") or {}
    if any(k.startswith("heads.") for k in sd.keys()):
        return "flat"
    return "compositional"


def _load_compositional_router(
    payload: Dict[str, Any],
    artifacts: CompositionalArtifacts,
    device: torch.device,
) -> Tuple[CompositionalRouter, Dict[str, Any], List[str], Dict[str, int]]:
    cfg = payload["config"]
    benchmarks: List[str] = list(payload.get("benchmarks") or cfg.get("benchmarks") or [])
    if not benchmarks:
        raise ValueError("compositional checkpoint missing 'benchmarks' list")
    bench_to_id = (
        payload.get("bench_to_id")
        or cfg.get("bench_to_id")
        or {b: i for i, b in enumerate(sorted(benchmarks))}
    )
    use_full_sequence = bool(cfg.get("use_full_sequence", False))

    probe_bench = next((b for b in benchmarks if b in artifacts.manifest["benchmarks"]), None)
    if probe_bench is None:
        probe_bench = next(iter(artifacts.manifest["benchmarks"]))
    d_model = _probe_d_model(artifacts, probe_bench, use_full_sequence)

    compressor = _build_compressor_from_cfg(d_model, cfg)
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
    ).to(device)
    sd = payload.get("model_state_dict") or payload["state_dict"]
    router.load_state_dict(sd)
    router.eval()
    return router, cfg, benchmarks, dict(bench_to_id)


def _load_flat_program_router(
    payload: Dict[str, Any],
    artifacts: CompositionalArtifacts,
    device: torch.device,
) -> Tuple[FlatProgramRouter, Dict[str, Any], List[str], Dict[str, int]]:
    cfg = payload["config"]
    benchmarks: List[str] = list(payload.get("benchmarks") or cfg.get("benchmarks") or [])
    bench_to_id = (
        payload.get("bench_to_id")
        or cfg.get("bench_to_id")
        or {b: i for i, b in enumerate(sorted(benchmarks))}
    )
    bench_n_programs = cfg.get("bench_n_programs")
    if bench_n_programs is None:
        bench_n_programs = {b: int(artifacts.catalogues[b].n_programs) for b in benchmarks
                            if b in artifacts.catalogues}
    if not bench_n_programs:
        raise ValueError("flat checkpoint: cannot resolve bench_n_programs")
    use_full_sequence = bool(cfg.get("use_full_sequence", False))

    probe_bench = next((b for b in benchmarks if b in artifacts.manifest["benchmarks"]), None)
    if probe_bench is None:
        probe_bench = next(iter(artifacts.manifest["benchmarks"]))
    d_model = _probe_d_model(artifacts, probe_bench, use_full_sequence)

    compressor = _build_compressor_from_cfg(d_model, cfg)
    router = FlatProgramRouter(
        compressor=compressor,
        bench_to_n_programs={str(b): int(n) for b, n in bench_n_programs.items()},
        bench_to_id={str(b): int(i) for b, i in bench_to_id.items()},
        d=int(cfg["d_latent"]),
        encoder_hidden_dims=cfg.get("encoder_hidden_dims", []),
        dropout=cfg.get("encoder_dropout", 0.1),
        freeze_compressor=cfg.get("freeze_compressor", False),
    ).to(device)
    sd = payload.get("state_dict") or payload["model_state_dict"]
    router.load_state_dict(sd)
    router.eval()
    return router, cfg, benchmarks, dict(bench_to_id)


ROUTER_LOADERS: Dict[str, Callable[..., Tuple[Any, Dict[str, Any], List[str], Dict[str, int]]]] = {
    "compositional": _load_compositional_router,
    "flat":          _load_flat_program_router,
}


@dataclass
class _RouterAdapter:
    """Router-architecture-agnostic scorer.

    ``score_programs`` returns logits/scores ``[B, N]`` over the catalogue
    of the given benchmark. Caller is responsible for catalogue-padding
    (we always return exactly ``catalogue.n_programs`` columns).
    """

    router: torch.nn.Module
    kind: str
    cfg: Dict[str, Any]
    train_benchmarks: List[str]
    bench_to_id: Dict[str, int]

    @property
    def use_full_sequence(self) -> bool:
        return bool(self.cfg.get("use_full_sequence", False))

    def can_score(self, bench: str) -> Tuple[bool, str]:
        if self.kind == "flat":
            if bench not in self.router.heads:
                return False, f"flat router has no trained head for {bench!r}"
        return True, ""

    @torch.no_grad()
    def score_programs(
        self,
        enc: torch.Tensor,
        attn: Optional[torch.Tensor],
        bench: str,
        catalogue: LegalCatalogue,
        lam: float,
    ) -> torch.Tensor:
        if self.kind == "compositional":
            g_q = self.router.encode(enc, attn)
            u_q = self.router.primitive_scores_from_g(g_q)
            v_q = self.router.pair_scores_from_g(g_q, catalogue) if self.router.use_pairs else None
            S = self.router.program_scores(u_q, catalogue, lam, v_q=v_q)
            return S
        if self.kind == "flat":
            g_q = self.router.encode(enc, attn)
            return self.router.program_scores_for_bench(g_q, bench)
        raise ValueError(f"unknown router kind {self.kind!r}")


def load_router(
    checkpoint_path: _Path,
    router_type: str,
    artifacts: CompositionalArtifacts,
    device: torch.device,
) -> _RouterAdapter:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if router_type == "auto":
        router_type = _detect_router_type(payload)
        logger.info("auto-detected router type: %s", router_type)
    if router_type not in ROUTER_LOADERS:
        raise ValueError(
            f"unknown router_type {router_type!r}; choices: {list(ROUTER_LOADERS)} or 'auto'"
        )
    router, cfg, benchmarks, bench_to_id = ROUTER_LOADERS[router_type](
        payload, artifacts, device,
    )
    if router_type == "compositional" and getattr(router, "use_pairs", False):
        # Pair features need to be attached against catalogues we plan to use.
        router.attach_pair_features(list(artifacts.catalogues.values()))
    return _RouterAdapter(
        router=router, kind=router_type, cfg=cfg,
        train_benchmarks=list(benchmarks), bench_to_id=dict(bench_to_id),
    )


# ============================================================================
# 2. Benchmark / artefact loading
# ============================================================================


def resolve_benchmark_list(
    artifacts: CompositionalArtifacts,
    explicit: Optional[Sequence[str]],
    benchmarks_json: Optional[_Path],
    all_benchmarks: bool,
) -> List[str]:
    available = list(artifacts.manifest["benchmarks"].keys())
    if all_benchmarks:
        return available
    if benchmarks_json is not None:
        with open(benchmarks_json) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = list(data.keys())
        if not isinstance(data, list):
            raise ValueError(
                f"--benchmarks_json must contain a list or dict of names, got {type(data).__name__}"
            )
        return [str(b) for b in data]
    if explicit:
        return list(explicit)
    raise ValueError(
        "no benchmarks selected; use --benchmark, --benchmarks_json, or --all_benchmarks"
    )


def _split_indices(n: int, val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_size = max(1, int(n * val_fraction)) if n > 1 else 0
    return perm[val_size:], perm[:val_size]


def select_indices(
    n: int, split: str, val_fraction: float, seed: int,
) -> List[int]:
    if split == "all":
        return list(range(n))
    train_idx, val_idx = _split_indices(n, val_fraction, seed)
    if split == "internal_train":
        return train_idx
    if split == "internal_val":
        return val_idx
    raise ValueError(f"unknown split {split!r}")


def build_dataset_for_benchmark(
    artifacts: CompositionalArtifacts,
    bench: str,
    bench_to_id: Dict[str, int],
    use_full_sequence: bool,
    dense_path: Optional[_Path],
) -> CompositionalDataset:
    """Build a single-benchmark dataset, optionally attaching dense deltas."""
    return CompositionalDataset(
        artifacts,
        benchmarks=[bench],
        use_full_sequence=use_full_sequence,
        bench_to_id=bench_to_id,
        dense_delta_paths={bench: dense_path} if dense_path is not None else None,
    )


# ============================================================================
# 3. Baseline loading -- "both" mode
# ============================================================================


@dataclass
class BenchmarkBaseline:
    """Per-benchmark baseline (normal module sequence) utility.

    At least one of ``per_question`` or ``mean`` must be present.
    """
    bench: str
    per_question: Optional[torch.Tensor]   # [Q] aligned with dataset.question_ids order, by qid
    mean: Optional[float]
    source: str                            # "anchor_utilities" | "baseline_file" | "baseline_dir"


def _load_baseline_mapping(path: _Path) -> Dict[str, float]:
    """Parse ``--baseline_file``: ``{bench: float | {accuracy: float, ...}}``."""
    with open(path) as f:
        raw = json.load(f)
    out: Dict[str, float] = {}
    for k, v in raw.items():
        if isinstance(v, (int, float)):
            out[str(k)] = float(v)
        elif isinstance(v, dict):
            for key in ("accuracy", "mean", "baseline", "anchor_acc", "value"):
                if key in v:
                    out[str(k)] = float(v[key])
                    break
            else:
                raise ValueError(
                    f"baseline_file entry for {k!r} has no recognised scalar key "
                    f"(expected one of accuracy/mean/baseline/anchor_acc/value)"
                )
        else:
            raise ValueError(f"baseline_file entry {k!r} has unsupported type {type(v).__name__}")
    return out


def _load_baseline_dir(d: _Path) -> Dict[str, float]:
    """Parse ``--baseline_dir``: ``DIR/{bench}.json`` each containing scalar/dict."""
    out: Dict[str, float] = {}
    for p in sorted(d.glob("*.json")):
        with open(p) as f:
            payload = json.load(f)
        if isinstance(payload, (int, float)):
            out[p.stem] = float(payload)
            continue
        if not isinstance(payload, dict):
            logger.warning("skipping baseline file %s (unsupported payload type)", p)
            continue
        for key in ("accuracy", "mean", "baseline", "anchor_acc", "value"):
            if key in payload:
                out[p.stem] = float(payload[key])
                break
        else:
            logger.warning("skipping baseline file %s (no recognised scalar key)", p)
    return out


def load_baselines(
    benchmarks: Sequence[str],
    dataset_per_bench: Dict[str, CompositionalDataset],
    baseline_file: Optional[_Path],
    baseline_dir: Optional[_Path],
) -> Tuple[Dict[str, BenchmarkBaseline], Dict[str, str]]:
    """Resolve baselines per benchmark.

    Priority:
      1. ``anchor_utilities`` tensor inside dense_deltas_matrix.pt (per-question)
      2. ``--baseline_file`` mapping (scalar)
      3. ``--baseline_dir`` per-bench file (scalar)

    Returns ``(baselines, skipped)`` where ``skipped[bench]`` is a reason
    string for benchmarks that have no baseline source.
    """
    file_map = _load_baseline_mapping(baseline_file) if baseline_file else {}
    dir_map = _load_baseline_dir(baseline_dir) if baseline_dir else {}

    baselines: Dict[str, BenchmarkBaseline] = {}
    skipped: Dict[str, str] = {}
    for bench in benchmarks:
        ds = dataset_per_bench.get(bench)
        anchor_util = (
            ds.anchor_utilities_per_bench.get(bench) if ds is not None else None
        )
        if anchor_util is not None:
            baselines[bench] = BenchmarkBaseline(
                bench=bench,
                per_question=anchor_util.float().detach().cpu(),
                mean=float(anchor_util.float().mean().item()),
                source="anchor_utilities",
            )
            continue
        if bench in file_map:
            baselines[bench] = BenchmarkBaseline(
                bench=bench, per_question=None, mean=float(file_map[bench]),
                source="baseline_file",
            )
            continue
        if bench in dir_map:
            baselines[bench] = BenchmarkBaseline(
                bench=bench, per_question=None, mean=float(dir_map[bench]),
                source="baseline_dir",
            )
            continue
        skipped[bench] = (
            "no baseline available (no anchor_utilities in dense matrix, no entry "
            "in --baseline_file, no file in --baseline_dir)"
        )
    return baselines, skipped


# ============================================================================
# 4. Scoring (frozen, batched, router-agnostic)
# ============================================================================


@dataclass
class BenchmarkScored:
    """Raw outputs of scoring a benchmark, before metric reduction."""
    bench: str
    indices: List[int]                # dataset indices used (subset)
    question_ids: torch.Tensor        # [Q] long
    scores: torch.Tensor              # [Q, N] router scores over full catalogue
    anchor: torch.Tensor              # [Q] baseline utility per question
    anchor_source: str
    dense: Optional[torch.Tensor]     # [Q, N] dense deltas, or None
    obs_indices: List[torch.Tensor]   # length-Q list of [n_obs_q] long
    obs_deltas: List[torch.Tensor]    # length-Q list of [n_obs_q] float
    n_routes: int


@torch.no_grad()
def score_benchmark(
    adapter: _RouterAdapter,
    dataset: CompositionalDataset,
    bench: str,
    catalogue: LegalCatalogue,
    indices: Sequence[int],
    baseline: BenchmarkBaseline,
    lam: float,
    device: torch.device,
    *,
    batch_size: int = 64,
) -> BenchmarkScored:
    if not indices:
        N = catalogue.n_programs
        return BenchmarkScored(
            bench=bench, indices=[], question_ids=torch.zeros(0, dtype=torch.long),
            scores=torch.empty(0, N), anchor=torch.empty(0),
            anchor_source=baseline.source, dense=None,
            obs_indices=[], obs_deltas=[], n_routes=N,
        )

    dense_mat = dataset.dense_deltas_per_bench.get(bench)
    have_dense = dense_mat is not None

    score_chunks: List[torch.Tensor] = []
    dense_chunks: List[torch.Tensor] = []
    qid_chunks: List[torch.Tensor] = []
    anchor_chunks: List[torch.Tensor] = []
    obs_idx_all: List[torch.Tensor] = []
    obs_delta_all: List[torch.Tensor] = []

    for start in range(0, len(indices), batch_size):
        chunk = list(indices[start:start + batch_size])
        first = dataset.encoder_inputs[chunk[0]]
        if first.dim() == 1:
            enc = torch.stack([dataset.encoder_inputs[i] for i in chunk], dim=0).to(device)
            attn = None
        else:
            enc, attn = pad_sequences([dataset.encoder_inputs[i] for i in chunk])
            enc = enc.to(device)
            attn = attn.to(device) if attn is not None else None

        S = adapter.score_programs(enc, attn, bench, catalogue, lam).detach().cpu()
        score_chunks.append(S)

        qids = torch.tensor([dataset.question_ids[i] for i in chunk], dtype=torch.long)
        qid_chunks.append(qids)

        if have_dense:
            dense_chunks.append(torch.stack([dense_mat[q].float() for q in qids.tolist()], dim=0))

        if baseline.per_question is not None:
            anchor_chunks.append(torch.stack(
                [baseline.per_question[q].float() for q in qids.tolist()], dim=0,
            ))
        else:
            anchor_chunks.append(torch.full((len(chunk),), float(baseline.mean), dtype=torch.float32))

        for i in chunk:
            rec = dataset.records[i]
            obs_idx_all.append(torch.tensor(rec["obs_indices"], dtype=torch.long))
            obs_delta_all.append(torch.tensor(rec["obs_deltas"], dtype=torch.float32))

    return BenchmarkScored(
        bench=bench,
        indices=list(indices),
        question_ids=torch.cat(qid_chunks, dim=0),
        scores=torch.cat(score_chunks, dim=0),
        anchor=torch.cat(anchor_chunks, dim=0),
        anchor_source=baseline.source,
        dense=torch.cat(dense_chunks, dim=0) if have_dense else None,
        obs_indices=obs_idx_all,
        obs_deltas=obs_delta_all,
        n_routes=catalogue.n_programs,
    )


# ============================================================================
# 5. Metric computation -- per benchmark
# ============================================================================


def _safe_div(num: float, denom: float, eps: float = 1e-9) -> Optional[float]:
    if abs(denom) < eps:
        return None
    return float(num) / float(denom)


def _topk_hits(scores: torch.Tensor, gold: torch.Tensor, k: int) -> float:
    if scores.numel() == 0:
        return 0.0
    k = min(k, scores.shape[1])
    topk = torch.topk(scores, k=k, dim=-1).indices
    return float((topk == gold.unsqueeze(1)).any(dim=-1).float().mean().item())


def compute_metrics(
    scored: BenchmarkScored,
    *,
    ks: Sequence[int] = (1, 3, 5),
) -> Dict[str, Any]:
    Q = int(scored.scores.shape[0])
    N = int(scored.scores.shape[1])
    if Q == 0:
        return {
            "bench": scored.bench, "n_questions": 0, "n_routes": N,
            "dense_available": scored.dense is not None,
        }

    S = scored.scores
    anchor = scored.anchor.float()
    anchor_source = scored.anchor_source

    # ----- Router pick (full catalogue) -----
    pred_full = torch.argmax(S, dim=-1)  # [Q]

    # router-selected program utility -- need a delta source per question.
    # Preference order:
    #   - dense[q, pred] if dense available
    #   - obs_deltas[q][pos_of(pred)] if pred is in observed support
    #   - 0.0 otherwise (router picked an unobserved route -> assume neutral)
    delta_router = torch.zeros(Q, dtype=torch.float32)
    in_support_full = torch.zeros(Q, dtype=torch.bool)
    pos_utility_full = torch.zeros(Q, dtype=torch.bool)

    have_dense = scored.dense is not None
    if have_dense:
        delta_router = scored.dense.gather(1, pred_full.unsqueeze(1)).squeeze(1).float()
    for q in range(Q):
        oi = scored.obs_indices[q]
        od = scored.obs_deltas[q]
        p = int(pred_full[q].item())
        match = (oi == p).nonzero(as_tuple=False)
        if match.numel() > 0:
            in_support_full[q] = True
            if not have_dense:
                delta_router[q] = float(od[int(match[0, 0].item())].item())
        if delta_router[q].item() > 0:
            pos_utility_full[q] = True

    router_utility_full = (anchor + delta_router).mean().item()

    # ----- Oracle (full catalogue) -- only when dense is available -----
    metrics: Dict[str, Any] = {
        "bench": scored.bench,
        "n_questions": Q,
        "n_routes": N,
        "dense_available": have_dense,
        "anchor_source": anchor_source,

        "baseline_utility": float(anchor.mean().item()),
        "router_utility": float(router_utility_full),
        "mean_uplift": float(router_utility_full - anchor.mean().item()),
    }

    if have_dense:
        gold_full = torch.argmax(scored.dense, dim=-1)
        delta_oracle_full = scored.dense.gather(1, gold_full.unsqueeze(1)).squeeze(1).float()
        oracle_utility_full = (anchor + delta_oracle_full).mean().item()
        oracle_uplift_full = oracle_utility_full - anchor.mean().item()
        metrics["oracle_utility_full"] = float(oracle_utility_full)
        metrics["oracle_uplift_full"] = float(oracle_uplift_full)
        metrics["frac_oracle_gain_recovered_full"] = _safe_div(
            metrics["mean_uplift"], oracle_uplift_full,
        )
        for k in ks:
            metrics[f"full_top{k}_acc"] = _topk_hits(S, gold_full, k)
    else:
        metrics["oracle_utility_full"] = None
        metrics["oracle_uplift_full"] = None
        metrics["frac_oracle_gain_recovered_full"] = None
        for k in ks:
            metrics[f"full_top{k}_acc"] = None

    # ----- Observed-support reranking -----
    # For each question, restrict argmax to obs_indices[q].
    pred_obs = torch.zeros(Q, dtype=torch.long)
    delta_router_obs = torch.zeros(Q, dtype=torch.float32)
    delta_oracle_obs = torch.zeros(Q, dtype=torch.float32)
    obs_topk_hits = {k: 0 for k in ks}
    n_with_obs = 0
    pos_utility_obs = torch.zeros(Q, dtype=torch.bool)

    for q in range(Q):
        oi = scored.obs_indices[q]
        od = scored.obs_deltas[q]
        if oi.numel() == 0:
            pred_obs[q] = pred_full[q]
            delta_router_obs[q] = delta_router[q]
            delta_oracle_obs[q] = 0.0
            continue
        n_with_obs += 1
        sub_scores = S[q].index_select(0, oi)
        local_argmax = int(torch.argmax(sub_scores).item())
        chosen = int(oi[local_argmax].item())
        pred_obs[q] = chosen
        delta_router_obs[q] = float(od[local_argmax].item())
        if delta_router_obs[q].item() > 0:
            pos_utility_obs[q] = True
        # Oracle (observed) = best obs delta
        local_oracle = int(torch.argmax(od).item())
        gold_chosen = int(oi[local_oracle].item())
        delta_oracle_obs[q] = float(od[local_oracle].item())
        # Observed-support top-k hits: rank obs_indices by sub_scores, check gold.
        for k in ks:
            kk = min(k, int(oi.numel()))
            topk_local = torch.topk(sub_scores, k=kk).indices
            topk_global = oi.index_select(0, topk_local)
            if (topk_global == gold_chosen).any():
                obs_topk_hits[k] += 1

    router_utility_obs = (anchor + delta_router_obs).mean().item()
    oracle_utility_obs = (anchor + delta_oracle_obs).mean().item()
    oracle_uplift_obs = oracle_utility_obs - anchor.mean().item()

    metrics.update({
        "router_utility_obs_rerank": float(router_utility_obs),
        "mean_uplift_obs_rerank": float(router_utility_obs - anchor.mean().item()),
        "oracle_utility_observed": float(oracle_utility_obs),
        "oracle_uplift_observed": float(oracle_uplift_obs),
        "frac_oracle_gain_recovered_observed": _safe_div(
            router_utility_obs - anchor.mean().item(), oracle_uplift_obs,
        ),
        "mean_best_observed_route": float(delta_oracle_obs.mean().item()),
    })
    for k in ks:
        metrics[f"obs_top{k}_acc"] = float(
            obs_topk_hits[k] / n_with_obs if n_with_obs > 0 else 0.0
        )

    # Support diagnostics on the full-catalogue prediction.
    metrics["frac_pred_in_observed_support"] = float(in_support_full.float().mean().item())
    metrics["frac_pred_positive_utility"] = float(pos_utility_full.float().mean().item())
    metrics["frac_pred_obs_positive_utility"] = float(pos_utility_obs.float().mean().item())

    return metrics


# ============================================================================
# 6. Aggregation -- across benchmarks
# ============================================================================


def aggregate_metrics(per_bench: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    evaluated = [m for m in per_bench.values() if m.get("n_questions", 0) > 0]
    if not evaluated:
        return {"n_benchmarks_evaluated": 0}

    def _macro(field: str) -> Optional[float]:
        vals = [m[field] for m in evaluated if m.get(field) is not None]
        return float(sum(vals) / len(vals)) if vals else None

    out = {
        "n_benchmarks_evaluated": len(evaluated),
        "macro_mean_router": _macro("router_utility"),
        "macro_mean_baseline": _macro("baseline_utility"),
        "macro_mean_uplift": _macro("mean_uplift"),
        "macro_mean_oracle_full": _macro("oracle_utility_full"),
        "macro_mean_oracle_observed": _macro("oracle_utility_observed"),
        "macro_mean_frac_oracle_full": _macro("frac_oracle_gain_recovered_full"),
        "macro_mean_frac_oracle_observed": _macro("frac_oracle_gain_recovered_observed"),
        "macro_mean_full_top1": _macro("full_top1_acc"),
        "macro_mean_full_top3": _macro("full_top3_acc"),
        "macro_mean_full_top5": _macro("full_top5_acc"),
        "macro_mean_obs_top1": _macro("obs_top1_acc"),
        "macro_mean_obs_top3": _macro("obs_top3_acc"),
        "macro_mean_obs_top5": _macro("obs_top5_acc"),
    }
    return out


# ============================================================================
# 7. Plotting (optional, viridis only)
# ============================================================================


def _setup_seaborn():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available, skipping plots")
        return None
    sns.set_theme(style="whitegrid", font_scale=1.05)
    return sns


def plot_per_benchmark_bars(
    per_bench: Dict[str, Dict[str, Any]],
    out_path: _Path,
) -> None:
    sns = _setup_seaborn()
    if sns is None:
        return
    import matplotlib.pyplot as plt
    import numpy as np

    benches = [b for b, m in per_bench.items() if m.get("n_questions", 0) > 0]
    if not benches:
        return
    baseline_vals = [per_bench[b]["baseline_utility"] for b in benches]
    router_vals = [per_bench[b]["router_utility"] for b in benches]
    oracle_vals = [per_bench[b].get("oracle_utility_full") for b in benches]

    palette = sns.color_palette("viridis", 3)
    x = np.arange(len(benches))
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(6, len(benches) * 1.6), 5))
    ax.bar(x - width, baseline_vals, width, label="Baseline (normal seq)", color=palette[0])
    ax.bar(x, router_vals, width, label="Router", color=palette[1])
    have_any_oracle = any(v is not None for v in oracle_vals)
    if have_any_oracle:
        oracle_plot = [v if v is not None else 0.0 for v in oracle_vals]
        ax.bar(x + width, oracle_plot, width, label="Oracle (full catalogue)", color=palette[2])

    ax.set_xticks(x)
    ax.set_xticklabels(benches, rotation=30, ha="right")
    ax.set_ylabel("Utility")
    ax.set_title("Cross-benchmark router transfer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def plot_uplift_bars(
    per_bench: Dict[str, Dict[str, Any]],
    out_path: _Path,
) -> None:
    sns = _setup_seaborn()
    if sns is None:
        return
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [
        (b, m["mean_uplift"]) for b, m in per_bench.items()
        if m.get("n_questions", 0) > 0 and m.get("mean_uplift") is not None
    ]
    if not rows:
        return
    rows.sort(key=lambda r: r[1], reverse=True)
    benches = [r[0] for r in rows]
    uplifts = [r[1] for r in rows]

    cmap = sns.color_palette("viridis", as_cmap=True)
    vmax = max(abs(u) for u in uplifts) or 1.0
    norm = [(u + vmax) / (2 * vmax) for u in uplifts]
    colors = [cmap(t) for t in norm]

    fig, ax = plt.subplots(figsize=(max(6, len(benches) * 1.0), 5))
    ax.bar(np.arange(len(benches)), uplifts, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(benches)))
    ax.set_xticklabels(benches, rotation=30, ha="right")
    ax.set_ylabel("Router uplift over baseline")
    ax.set_title("Per-benchmark router - baseline (sorted)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def plot_topk_heatmap(
    per_bench: Dict[str, Dict[str, Any]],
    out_path: _Path,
    *,
    view: str = "full",
) -> None:
    """Heatmap (rows=benchmarks, cols=top-k) using viridis.

    ``view`` is ``"full"`` or ``"observed"``.
    """
    sns = _setup_seaborn()
    if sns is None:
        return
    import matplotlib.pyplot as plt
    import numpy as np

    prefix = "full" if view == "full" else "obs"
    cols = [1, 3, 5]
    benches = [b for b, m in per_bench.items() if m.get("n_questions", 0) > 0]
    if not benches:
        return
    matrix = np.full((len(benches), len(cols)), np.nan)
    for i, b in enumerate(benches):
        for j, k in enumerate(cols):
            v = per_bench[b].get(f"{prefix}_top{k}_acc")
            if v is not None:
                matrix[i, j] = float(v)
    if np.all(np.isnan(matrix)):
        return

    fig, ax = plt.subplots(figsize=(max(4, len(cols) * 1.2 + 2), max(3, len(benches) * 0.45 + 1)))
    sns.heatmap(
        matrix,
        annot=True, fmt=".3f",
        xticklabels=[f"top-{k}" for k in cols],
        yticklabels=benches,
        cmap="viridis", vmin=0.0, vmax=1.0,
        cbar_kws={"label": f"{view}-catalogue top-k acc"},
        ax=ax,
    )
    ax.set_title(f"Router ranking quality ({view} catalogue)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


# ============================================================================
# 8. Result export
# ============================================================================


def write_results_json(
    out_path: _Path,
    meta: Dict[str, Any],
    per_bench: Dict[str, Dict[str, Any]],
    aggregate: Dict[str, Any],
    skipped: Dict[str, str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "per_benchmark": per_bench,
        "aggregate": aggregate,
        "skipped": skipped,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    logger.info("wrote %s", out_path)


def write_per_question_csv(
    out_dir: _Path,
    bench: str,
    scored: BenchmarkScored,
    metrics: Dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"per_question_{bench}.csv"
    Q = int(scored.scores.shape[0])
    if Q == 0:
        return

    pred_full = torch.argmax(scored.scores, dim=-1)
    have_dense = scored.dense is not None
    if have_dense:
        gold_full = torch.argmax(scored.dense, dim=-1)
        delta_router_full = scored.dense.gather(1, pred_full.unsqueeze(1)).squeeze(1).float()
        oracle_util_full = (scored.anchor + scored.dense.gather(1, gold_full.unsqueeze(1)).squeeze(1).float())
    else:
        gold_full = torch.full((Q,), -1, dtype=torch.long)
        delta_router_full = torch.zeros(Q, dtype=torch.float32)
        oracle_util_full = torch.full((Q,), float("nan"))

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "benchmark", "question_id",
            "router_program_idx_full", "router_utility_full",
            "router_program_idx_obs",  "router_utility_obs",
            "baseline_utility",
            "oracle_utility_full", "oracle_utility_observed",
            "router_in_observed_support",
            "full_top1_hit", "obs_top1_hit",
        ])
        for q in range(Q):
            qid = int(scored.question_ids[q].item())
            anchor_q = float(scored.anchor[q].item())
            pred_full_q = int(pred_full[q].item())

            oi = scored.obs_indices[q]
            od = scored.obs_deltas[q]
            in_support = bool((oi == pred_full_q).any().item()) if oi.numel() > 0 else False
            if oi.numel() > 0:
                sub = scored.scores[q].index_select(0, oi)
                local_argmax = int(torch.argmax(sub).item())
                pred_obs_q = int(oi[local_argmax].item())
                delta_router_obs_q = float(od[local_argmax].item())
                local_oracle = int(torch.argmax(od).item())
                oracle_util_obs_q = anchor_q + float(od[local_oracle].item())
                gold_obs_q = int(oi[local_oracle].item())
                obs_top1 = int(pred_obs_q == gold_obs_q)
            else:
                pred_obs_q = pred_full_q
                delta_router_obs_q = float(delta_router_full[q].item())
                oracle_util_obs_q = float("nan")
                obs_top1 = 0

            full_top1 = int(pred_full_q == int(gold_full[q].item())) if have_dense else 0
            writer.writerow([
                bench, qid,
                pred_full_q, anchor_q + float(delta_router_full[q].item()),
                pred_obs_q, anchor_q + delta_router_obs_q,
                anchor_q,
                float(oracle_util_full[q].item()) if have_dense else float("nan"),
                oracle_util_obs_q,
                int(in_support),
                full_top1, obs_top1,
            ])
    logger.info("wrote %s", out_path)


# ============================================================================
# 9. CLI
# ============================================================================


def _parse_dense_deltas(entries: Optional[Iterable[str]]) -> Dict[str, _Path]:
    out: Dict[str, _Path] = {}
    for e in entries or []:
        if "=" not in e:
            raise SystemExit(f"--dense_deltas expects bench=path, got {e!r}")
        bench, path = e.split("=", 1)
        p = _Path(path)
        if not p.is_file():
            raise SystemExit(f"dense delta file not found: {p}")
        out[bench] = p
    return out


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, type=_Path,
                   help="Path to a router checkpoint (.pt).")
    p.add_argument("--router_type", default="auto",
                   choices=["auto"] + list(ROUTER_LOADERS.keys()),
                   help="Router architecture; 'auto' inspects the checkpoint payload.")
    p.add_argument("--catalogue_dir", required=True, type=_Path,
                   help="Directory containing manifest.json + per-bench catalogues.")

    bench_group = p.add_argument_group("Benchmark selection (use exactly one)")
    bench_group.add_argument("--benchmark", nargs="+", default=None,
                             help="Explicit list of benchmark names to evaluate on.")
    bench_group.add_argument("--benchmarks_json", type=_Path, default=None,
                             help="JSON file containing a list (or dict) of benchmark names.")
    bench_group.add_argument("--all_benchmarks", action="store_true",
                             help="Evaluate on every benchmark in the catalogue manifest.")

    p.add_argument("--dense_deltas", nargs="*", default=None,
                   help="bench=path entries pointing at dense_deltas_matrix.pt files. "
                        "Optional but enables full-catalogue oracle/top-k metrics.")
    p.add_argument("--baseline_file", type=_Path, default=None,
                   help="JSON {bench: float | {accuracy: float, ...}} with normal-seq baselines.")
    p.add_argument("--baseline_dir", type=_Path, default=None,
                   help="Directory of {bench}.json files each containing a baseline scalar.")

    p.add_argument("--split", default="all",
                   choices=["internal_val", "internal_train", "all"],
                   help="Slice of each benchmark's questions to evaluate.")
    p.add_argument("--val_fraction", type=float, default=0.15,
                   help="Must match the split used during the router's training (if applicable).")
    p.add_argument("--seed", type=int, default=42,
                   help="Must match the seed used during the router's training (if applicable).")
    p.add_argument("--lam", type=float, default=None,
                   help="Compositional only: override λ (default: read from checkpoint config).")
    p.add_argument("--use_full_sequence", action="store_true",
                   help="Override checkpoint config to use full-sequence residuals.")

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default=None)

    p.add_argument("--output_dir", required=True, type=_Path,
                   help="Directory to write the results JSON, plots, and CSVs into.")
    p.add_argument("--plots", action="store_true", default=False,
                   help="Generate seaborn/viridis plots (default: off).")
    p.add_argument("--per_question_csv", action="store_true", default=False,
                   help="Export per-question CSV per benchmark (default: off).")
    p.add_argument("--log_level", default="INFO")
    return p


def _validate_benchmark_selection(args: argparse.Namespace, parser: argparse.ArgumentParser):
    n = sum([
        bool(args.benchmark),
        args.benchmarks_json is not None,
        bool(args.all_benchmarks),
    ])
    if n == 0:
        parser.error("provide one of --benchmark, --benchmarks_json, or --all_benchmarks")
    if n > 1:
        parser.error("--benchmark, --benchmarks_json, and --all_benchmarks are mutually exclusive")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    _validate_benchmark_selection(args, parser)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load full artefacts for the requested benchmarks ----
    dense_paths = _parse_dense_deltas(args.dense_deltas)

    # First load the manifest (cheap) to resolve the benchmark list.
    manifest_artifacts = load_artifacts(args.catalogue_dir)
    target_benchmarks = resolve_benchmark_list(
        manifest_artifacts, args.benchmark, args.benchmarks_json, args.all_benchmarks,
    )
    # Then re-load filtered to the targets we care about (catalogues only).
    artifacts = load_artifacts(args.catalogue_dir, benchmarks=target_benchmarks)

    # ---- Load router ----
    adapter = load_router(args.checkpoint, args.router_type, artifacts, device)
    lam = float(args.lam) if args.lam is not None else float(adapter.cfg.get("lam", 0.0))
    use_full_sequence = bool(args.use_full_sequence or adapter.use_full_sequence)

    # Build a stable bench_to_id covering every benchmark we'll touch (router's
    # own ids take priority; new OOD benchmarks get appended).
    bench_to_id: Dict[str, int] = dict(adapter.bench_to_id)
    next_id = (max(bench_to_id.values()) + 1) if bench_to_id else 0
    for b in target_benchmarks:
        if b not in bench_to_id:
            bench_to_id[b] = next_id
            next_id += 1

    # ---- Build per-benchmark datasets ----
    skipped: Dict[str, str] = {}
    datasets: Dict[str, CompositionalDataset] = {}
    for bench in target_benchmarks:
        if bench not in artifacts.catalogues:
            skipped[bench] = "missing from catalogue manifest"
            continue
        try:
            ds = build_dataset_for_benchmark(
                artifacts, bench, bench_to_id, use_full_sequence,
                dense_paths.get(bench),
            )
        except Exception as e:  # noqa: BLE001
            skipped[bench] = f"dataset build failed: {e}"
            logger.warning("[%s] dataset build failed: %s", bench, e)
            continue
        if len(ds) == 0:
            skipped[bench] = "dataset is empty after construction"
            continue
        datasets[bench] = ds

    # ---- Resolve baselines ----
    baselines, baseline_skipped = load_baselines(
        list(datasets.keys()), datasets, args.baseline_file, args.baseline_dir,
    )
    skipped.update(baseline_skipped)

    # ---- Score each benchmark ----
    per_bench_metrics: Dict[str, Dict[str, Any]] = {}
    per_bench_scored: Dict[str, BenchmarkScored] = {}
    for bench, ds in datasets.items():
        if bench in baseline_skipped:
            continue
        ok, reason = adapter.can_score(bench)
        if not ok:
            skipped[bench] = reason
            logger.warning("[%s] %s", bench, reason)
            continue
        catalogue = artifacts.catalogues[bench].to(device)
        if adapter.kind == "compositional" and getattr(adapter.router, "use_pairs", False):
            adapter.router.attach_pair_features([catalogue])

        # Note: question ordering inside the dataset already enumerates only
        # this benchmark's records (we built it with benchmarks=[bench]).
        indices = select_indices(len(ds), args.split, args.val_fraction, args.seed)
        if not indices:
            skipped[bench] = f"no records for split={args.split}"
            continue
        logger.info(
            "[%s] scoring %d / %d records (split=%s, baseline=%s, dense=%s)",
            bench, len(indices), len(ds), args.split,
            baselines[bench].source, bench in dense_paths,
        )
        scored = score_benchmark(
            adapter, ds, bench, catalogue, indices, baselines[bench],
            lam=lam, device=device, batch_size=args.batch_size,
        )
        per_bench_scored[bench] = scored
        per_bench_metrics[bench] = compute_metrics(scored)

    aggregate = aggregate_metrics(per_bench_metrics)

    # ---- Export ----
    meta = {
        "checkpoint": str(args.checkpoint),
        "router_type": adapter.kind,
        "router_train_benchmarks": adapter.train_benchmarks,
        "catalogue_dir": str(args.catalogue_dir),
        "lam": lam,
        "use_full_sequence": use_full_sequence,
        "split": args.split,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "device": str(device),
        "target_benchmarks": target_benchmarks,
        "baseline_source_per_bench": {b: bl.source for b, bl in baselines.items()},
        "dense_deltas_provided": sorted(dense_paths.keys()),
    }
    write_results_json(
        args.output_dir / "cross_benchmark_eval.json",
        meta=meta,
        per_bench=per_bench_metrics,
        aggregate=aggregate,
        skipped=skipped,
    )

    # ---- Per-question CSV ----
    if args.per_question_csv:
        csv_dir = args.output_dir / "per_question"
        for bench, scored in per_bench_scored.items():
            write_per_question_csv(csv_dir, bench, scored, per_bench_metrics[bench])

    # ---- Plots ----
    if args.plots:
        plots_dir = args.output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_per_benchmark_bars(per_bench_metrics, plots_dir / "per_benchmark_bars.png")
        plot_uplift_bars(per_bench_metrics, plots_dir / "uplift_bars.png")
        plot_topk_heatmap(per_bench_metrics, plots_dir / "topk_full.png", view="full")
        plot_topk_heatmap(per_bench_metrics, plots_dir / "topk_observed.png", view="observed")

    # ---- Stdout summary ----
    print(json.dumps({
        "n_benchmarks_evaluated": aggregate.get("n_benchmarks_evaluated", 0),
        "n_benchmarks_skipped": len(skipped),
        "macro_mean_router": aggregate.get("macro_mean_router"),
        "macro_mean_baseline": aggregate.get("macro_mean_baseline"),
        "macro_mean_uplift": aggregate.get("macro_mean_uplift"),
        "macro_mean_frac_oracle_full": aggregate.get("macro_mean_frac_oracle_full"),
        "macro_mean_frac_oracle_observed": aggregate.get("macro_mean_frac_oracle_observed"),
        "skipped": skipped,
    }, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
