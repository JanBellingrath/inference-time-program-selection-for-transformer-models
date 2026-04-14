"""Router target construction for the unified HPO.

Dispatches between the *best_seq* and *explored* target branches, and between
the *hard_ce*, *soft_ce*, and *top_ce* loss variants.

Reuses the existing helpers from ``experiments.sweep_fine_routing`` wherever
possible, adding only the *top_ce* truncation logic on top.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from experiments.sweep_fine_routing import (
    build_best_seq_targets,
    build_mcts_router_targets,
    sharpen_targets,
)


# ---------------------------------------------------------------------------
# Top-CE: keep top-k entries, zero the rest, renormalize
# ---------------------------------------------------------------------------

def _truncate_to_top_k(
    targets: List[torch.Tensor],
    k: int,
) -> List[torch.Tensor]:
    """Keep only the *k* highest-mass entries per target, zero the rest, renormalize."""
    out: List[torch.Tensor] = []
    for p in targets:
        if p.shape[0] <= k:
            out.append(p)
            continue
        topk_vals, topk_idx = p.topk(k)
        q = torch.zeros_like(p)
        q[topk_idx] = topk_vals
        s = q.sum()
        if s > 1e-12:
            q = q / s
        else:
            q[0] = 1.0
        out.append(q)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_targets(
    records: List[Dict],
    seq_to_idx: Dict[tuple, int],
    num_classes: int,
    config: Dict,
) -> Tuple[List[torch.Tensor], bool]:
    """Build router targets from a resolved HPO configuration.

    Parameters
    ----------
    records : list of dict
        Per-sample JSONL records (must contain ``gate_label``, ``best_seq``,
        ``explored``, ``router_target``).
    seq_to_idx : dict
        Sequence-tuple -> class-index mapping for the active catalog.
    num_classes : int
        Total number of classes in the catalog (including noop at index 0).
    config : dict
        Resolved HPO configuration (from SMAC).

    Returns
    -------
    targets : list of Tensor
        One ``[num_classes]`` tensor per sample.
    hard : bool
        ``True`` if the targets are class indices (for hard CE),
        ``False`` if they are soft distributions (for soft/top CE).
    """
    target_source = config.get("target_source", "explored")
    router_loss = config.get("router_loss", "hard_ce")

    # ------------------------------------------------------------------
    # Branch 1: best_seq -> always hard CE
    # ------------------------------------------------------------------
    if target_source == "best_seq":
        targets = build_best_seq_targets(records, seq_to_idx, num_classes)
        return targets, True

    # ------------------------------------------------------------------
    # Branch 2: explored -> build shaped distribution, then apply loss variant
    # ------------------------------------------------------------------
    noop_boost = float(config.get("noop_boost", 0.0))
    target_temp = float(config.get("target_temp", 1.0))

    targets = build_mcts_router_targets(
        records, seq_to_idx, num_classes, noop_boost=noop_boost,
    )
    if abs(target_temp - 1.0) > 1e-6:
        targets = sharpen_targets(targets, target_temp)

    if router_loss == "hard_ce":
        return targets, True

    if router_loss == "top_ce":
        top_k = int(config.get("top_k", 4))
        targets = _truncate_to_top_k(targets, top_k)
        return targets, False

    # soft_ce: full distribution as-is
    return targets, False
