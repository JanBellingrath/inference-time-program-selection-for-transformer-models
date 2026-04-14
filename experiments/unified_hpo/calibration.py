"""Open-rate calibration with the prefix trick.

After a router and gate are trained, this module:

1. Computes a scalar routing score per routing-val sample.
2. Computes the proxy delta for each sample (what gain the router's predicted
   route would yield, looked up from the explored data).
3. Sorts by score descending and uses cumulative-prefix sums to evaluate the
   proxy gain at any open rate in O(1).
4. Runs a tiny local search around a prior-predicted center to find the
   best open rate.

This replaces the old approach of searching ``gamma`` / ``confidence_threshold``
/ ``delta_margin`` as global hyperparameters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

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
    prior_center: float = 0.0
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
    """Simpler fallback: if the router routes (pred != 0), use ``best_delta``.

    This is an upper-bound proxy (assumes the router always picks the optimal
    route when it does route).  Useful when explored data is sparse or when
    the catalog mapping is ambiguous.
    """
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
    """Precomputed structure for O(1) gain evaluation at any open rate."""
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


def _eval_gain_at_rho(prefix: _PrefixIndex, rho: float) -> Tuple[float, float, float]:
    """Evaluate proxy gain at open rate ``rho``.

    Returns ``(gain, threshold, realized_fraction)`` where:
    - gain is the average delta over the top-rho fraction
    - threshold is the score at the rho boundary
    - realized_fraction is the actual fraction opened (may differ slightly
      from rho due to discrete sample counts)
    """
    k = max(1, min(prefix.n, int(rho * prefix.n)))
    gain = float(prefix.cumsum_deltas[k - 1].item()) / prefix.n
    threshold = float(prefix.sorted_scores[k - 1].item())
    realized = k / prefix.n
    return gain, threshold, realized


# ---------------------------------------------------------------------------
# Open-rate calibration
# ---------------------------------------------------------------------------

def calibrate_open_rate(
    scores: torch.Tensor,
    deltas: torch.Tensor,
    prior_center: float = 0.10,
    prior_sigma: float = 0.15,
    n_samples: Optional[int] = None,
) -> CalibrationResult:
    """Run the local open-rate search around a prior-predicted center.

    Parameters
    ----------
    scores : Tensor [N]
        Routing scores for each routing-val sample (higher = more likely to route).
    deltas : Tensor [N]
        Proxy delta for each sample (gain from routing, looked up from data).
    prior_center : float
        Prior-predicted best open rate (from the threshold prior).
    prior_sigma : float
        Uncertainty of the prior prediction.
    n_samples : int or None
        If set, use only the first ``n_samples`` samples (budget subsampling).

    Returns
    -------
    CalibrationResult
    """
    if n_samples is not None and n_samples < len(scores):
        scores = scores[:n_samples]
        deltas = deltas[:n_samples]

    if len(scores) < 2:
        return CalibrationResult(best_rho=prior_center, prior_center=prior_center)

    prefix = _build_prefix_index(scores, deltas)
    result = CalibrationResult(prior_center=prior_center)

    # Build candidate set: center +/- step sizes
    step1 = max(0.02, min(0.05, prior_sigma * 0.3))
    step2 = step1 * 2.0

    candidates = sorted(set([
        max(RHO_MIN, min(RHO_MAX, prior_center)),
        max(RHO_MIN, min(RHO_MAX, prior_center - step1)),
        max(RHO_MIN, min(RHO_MAX, prior_center + step1)),
        max(RHO_MIN, min(RHO_MAX, prior_center - step2)),
        max(RHO_MIN, min(RHO_MAX, prior_center + step2)),
    ]))

    best_gain = -float("inf")
    best_rho = prior_center
    best_threshold = 0.0
    best_realized = 0.0

    for rho in candidates:
        gain, threshold, realized = _eval_gain_at_rho(prefix, rho)
        result.candidates_tested.append((rho, gain))
        if gain > best_gain:
            best_gain = gain
            best_rho = rho
            best_threshold = threshold
            best_realized = realized

    # Greedy refinement: if the best is at an edge, probe further in that direction
    if best_rho == max(candidates) and best_rho + step1 <= RHO_MAX:
        probe = best_rho + step1
        gain, threshold, realized = _eval_gain_at_rho(prefix, probe)
        result.candidates_tested.append((probe, gain))
        if gain > best_gain:
            best_gain = gain
            best_rho = probe
            best_threshold = threshold
            best_realized = realized
    elif best_rho == min(candidates) and best_rho - step1 >= RHO_MIN:
        probe = best_rho - step1
        gain, threshold, realized = _eval_gain_at_rho(prefix, probe)
        result.candidates_tested.append((probe, gain))
        if gain > best_gain:
            best_gain = gain
            best_rho = probe
            best_threshold = threshold
            best_realized = realized

    result.best_rho = best_rho
    result.best_threshold = best_threshold
    result.best_gain = best_gain
    result.realized_open_fraction = best_realized

    logger.info(
        "Calibration: best_rho=%.3f  threshold=%.4f  gain=%.5f  "
        "realized_open=%.3f  (%d candidates tested on %d samples)",
        best_rho, best_threshold, best_gain, best_realized,
        len(result.candidates_tested), len(scores),
    )

    return result
