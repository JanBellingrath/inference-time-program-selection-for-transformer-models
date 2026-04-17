"""Open-rate calibration with the prefix trick.

After a router and gate are trained, this module:

1. Computes a scalar routing score per routing-val sample.
2. Computes the proxy delta for each sample (what gain the router's predicted
   route would yield, looked up from the explored data).
3. Sorts by score descending and uses cumulative-prefix sums so the mean
   proxy gain for ``open top-k by score`` is one vectorized divide.
4. Picks the best ``k`` (equivalently realized open rate ``k/N``) by
   ``argmax`` over feasible ``k``, without a learned prior.

This replaces the old approach of searching ``gamma`` / ``confidence_threshold``
/ ``delta_margin`` as global hyperparameters.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)

RHO_MIN = 0.01
RHO_MAX = 0.50


@dataclass
class CalibrationResult:
    """Output of ``calibrate_open_rate``."""

    best_rho: float = 0.0
    best_threshold: float = 0.0
    best_gain: float = 0.0
    realized_open_fraction: float = 0.0
    candidates_tested: List[Tuple[float, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-sample delta computation
# ---------------------------------------------------------------------------

def compute_per_sample_deltas(
    router_preds: torch.Tensor,
    records: List[Dict],
    val_indices: List[int],
    sequence_catalog: List,
    seq_to_idx: Dict[tuple, int],
) -> torch.Tensor:
    """Compute the proxy delta for each routing-val sample.

    For each sample, the router predicts a route index.  We look up whether
    that route was explored for this sample and, if so, use its recorded
    delta (score - anchor_score).  If the router predicts noop (index 0) or
    the route was not explored, the delta is 0.

    Returns a ``[N_val]`` tensor of deltas aligned with ``val_indices``.
    """
    idx_to_seq = {v: k for k, v in seq_to_idx.items()}

    deltas = torch.zeros(len(val_indices), dtype=torch.float32)
    for i, vi in enumerate(val_indices):
        pred_class = int(router_preds[i].item())
        if pred_class == 0:
            continue

        rec = records[vi]
        anchor_score = rec.get("anchor_score", 0.0)
        explored = rec.get("explored", [])

        pred_seq_tuple = idx_to_seq.get(pred_class)
        if pred_seq_tuple is None:
            continue

        for entry in explored:
            entry_seq = tuple(int(x) for x in entry["seq"])
            if entry_seq == pred_seq_tuple:
                deltas[i] = float(entry["score"]) - float(anchor_score)
                break
        # If no matching explored entry found, delta stays 0.

    return deltas


def compute_per_sample_deltas_from_best_delta(
    router_preds: torch.Tensor,
    records: List[Dict],
    val_indices: List[int],
) -> torch.Tensor:
    """Oracle upper bound: if pred != 0, use ``best_delta`` for that record."""
    deltas = torch.zeros(len(val_indices), dtype=torch.float32)
    for i, vi in enumerate(val_indices):
        if int(router_preds[i].item()) != 0:
            deltas[i] = float(records[vi].get("best_delta", 0.0))
    return deltas


# ---------------------------------------------------------------------------
# Prefix trick for fast open-rate evaluation
# ---------------------------------------------------------------------------

@dataclass
class _PrefixIndex:
    """Precomputed structure: scores sorted desc, prefix sums of deltas."""
    sorted_scores: torch.Tensor
    cumsum_deltas: torch.Tensor
    n: int


def _build_prefix_index(
    scores: torch.Tensor,
    deltas: torch.Tensor,
) -> _PrefixIndex:
    """Sort by score descending and precompute cumulative delta sums."""
    order = scores.argsort(descending=True)
    sorted_scores = scores[order]
    sorted_deltas = deltas[order]
    cumsum = sorted_deltas.cumsum(dim=0)
    return _PrefixIndex(sorted_scores=sorted_scores, cumsum_deltas=cumsum, n=len(scores))


def calibrate_open_rate(
    scores: torch.Tensor,
    deltas: torch.Tensor,
) -> CalibrationResult:
    """Pick ``k`` that maximizes mean proxy gain (top-``k`` by score).

    Only considers realized open rates ``k/N`` in ``[RHO_MIN, RHO_MAX]`` when
    that range is non-empty in discrete ``k``; otherwise maximizes over all
    ``k`` in ``1..N``.

    Parameters
    ----------
    scores : Tensor [N]
        Routing scores for each routing-val sample (higher = more likely to route).
    deltas : Tensor [N]
        Proxy delta for each sample (gain from routing, looked up from data).

    Returns
    -------
    CalibrationResult
    """
    scores = scores.detach().float().cpu().reshape(-1)
    deltas = deltas.detach().float().cpu().reshape(-1)
    if scores.numel() != deltas.numel():
        raise ValueError("scores and deltas must have the same length")
    n = int(scores.numel())
    if n == 0:
        return CalibrationResult()

    prefix = _build_prefix_index(scores, deltas)
    mean_gains = prefix.cumsum_deltas / float(n)

    k_min = max(1, int(math.ceil(n * RHO_MIN - 1e-12)))
    k_max = min(n, max(k_min, int(n * RHO_MAX)))

    if k_min > k_max:
        best_idx = int(mean_gains.argmax().item())
    else:
        segment = mean_gains[k_min - 1 : k_max]
        rel = int(segment.argmax().item())
        best_idx = (k_min - 1) + rel

    best_k = best_idx + 1
    best_gain = float(mean_gains[best_idx].item())
    realized = best_k / float(n)
    threshold = float(prefix.sorted_scores[best_idx].item())

    result = CalibrationResult(
        best_rho=realized,
        best_threshold=threshold,
        best_gain=best_gain,
        realized_open_fraction=realized,
        candidates_tested=[(realized, best_gain)],
    )

    logger.info(
        "Calibration: best_rho=%.3f  threshold=%.4f  gain=%.5f  "
        "realized_open=%.3f  (N=%d  k=%d)",
        result.best_rho,
        result.best_threshold,
        result.best_gain,
        result.realized_open_fraction,
        n,
        best_k,
    )

    return result
