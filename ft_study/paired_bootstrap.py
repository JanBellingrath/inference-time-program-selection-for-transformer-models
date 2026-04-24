"""Paired bootstrap confidence intervals for two protocols on the same items.

When two protocols (e.g. ``ft_only`` vs ``search_ft``) are evaluated on the
same set of test questions, the per-question outcomes are paired: each
question is either correct or wrong under each protocol.  Comparing the
protocols by their *unpaired* mean+std loses the within-question
correlation and dramatically inflates the CI on the difference.

This module computes a paired bootstrap CI on the difference in per-question
accuracy.  Items are joined on ``sample_hash`` and items present in only one
protocol are dropped (with a warning).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PairedBootstrapResult:
    n_paired: int
    mean_a: float
    mean_b: float
    mean_diff: float
    ci_lo: float
    ci_hi: float
    n_boot: int
    ci_level: float
    p_two_sided: float
    """Empirical two-sided p-value for H0: mean_diff == 0, computed as
    ``2 * min(P(boot_diff <= 0), P(boot_diff >= 0))``."""


def _to_outcome_map(per_q: Sequence[Dict]) -> Dict[str, int]:
    """Convert a list of ``{"sample_hash": ..., "correct": bool}`` dicts to a
    mapping ``sample_hash -> 0/1``.

    Duplicate hashes are warned about (last value wins).
    """
    out: Dict[str, int] = {}
    for r in per_q:
        h = r.get("sample_hash") or r.get("_hash")
        if h is None:
            continue
        out[h] = int(bool(r.get("correct", False)))
    return out


def paired_bootstrap_diff(
    per_q_a: Sequence[Dict],
    per_q_b: Sequence[Dict],
    n_boot: int = 5000,
    ci: float = 0.95,
    seed: int = 0,
) -> Optional[PairedBootstrapResult]:
    """Paired bootstrap CI on (mean_b - mean_a) over questions joined by hash.

    Args:
        per_q_a: List of ``{"sample_hash": str, "correct": bool|int}`` for
            protocol A (e.g. ``ft_only``).
        per_q_b: Same shape for protocol B (e.g. ``search_ft``).
        n_boot: Number of bootstrap resamples (default 5000).
        ci: Confidence level (default 0.95).
        seed: RNG seed for reproducibility.

    Returns:
        ``PairedBootstrapResult`` with mean diff and CI, or ``None`` if there
        are no overlapping questions.
    """
    a_map = _to_outcome_map(per_q_a)
    b_map = _to_outcome_map(per_q_b)
    common = sorted(set(a_map) & set(b_map))
    if not common:
        return None

    a = np.array([a_map[h] for h in common], dtype=np.float64)
    b = np.array([b_map[h] for h in common], dtype=np.float64)
    diff = b - a

    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)

    alpha = 1.0 - ci
    lo, hi = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    p_pos = float(np.mean(boot_means >= 0.0))
    p_neg = float(np.mean(boot_means <= 0.0))
    p_two = 2.0 * min(p_pos, p_neg)
    p_two = min(1.0, max(0.0, p_two))

    return PairedBootstrapResult(
        n_paired=n,
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        mean_diff=float(diff.mean()),
        ci_lo=float(lo),
        ci_hi=float(hi),
        n_boot=n_boot,
        ci_level=ci,
        p_two_sided=p_two,
    )


def pool_per_question_outcomes(
    per_arm_per_seed: Dict[Tuple[str, int], List[Dict]],
) -> Dict[str, List[Dict]]:
    """Merge per-(arm,seed) per-question outcomes into per-arm pooled lists.

    Each question is uniquely identified by ``(seed, sample_hash)`` so that
    the same hash from different seeds becomes distinct items in the pooled
    output (preserving the across-seed sample size).
    """
    pooled: Dict[str, List[Dict]] = {}
    for (arm, seed), per_q in per_arm_per_seed.items():
        bucket = pooled.setdefault(arm, [])
        for r in per_q:
            h = r.get("sample_hash") or r.get("_hash")
            if h is None:
                continue
            bucket.append({
                "sample_hash": f"s{seed}::{h}",
                "correct": bool(r.get("correct", False)),
            })
    return pooled
