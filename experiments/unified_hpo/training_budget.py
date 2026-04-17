"""Map SMAC Hyperband ``budget`` to training cost (epoch counts).

Evaluation (calibration, proxy gain) always uses the full routing-val split;
only router/gate training epochs scale with ``budget``.
"""

from __future__ import annotations

from typing import Tuple

from experiments.unified_hpo.trainer import DEFAULT_GATE_EPOCHS, DEFAULT_ROUTER_EPOCHS

# Default SMAC training resource range (geometric ladder-friendly, e.g. η=3: 9→27→81→243).
DEFAULT_TRAIN_MIN_BUDGET = 9.0
DEFAULT_TRAIN_MAX_BUDGET = 243.0


def router_gate_epochs_from_training_budget(
    budget: float,
    min_budget: float,
    max_budget: float,
    router_base: int = DEFAULT_ROUTER_EPOCHS,
    gate_base: int = DEFAULT_GATE_EPOCHS,
) -> Tuple[int, int]:
    """Linear epoch scaling: ``scale = budget / max_budget`` in ``[min/max, 1]``.

    Parameters
    ----------
    budget
        Current SMAC fidelity (same units as ``min_budget`` / ``max_budget``).
    min_budget, max_budget
        Ends of the Hyperband ladder; ``max_budget`` corresponds to full training.
    """
    lo = float(min_budget)
    hi = float(max_budget)
    if hi <= 0 or lo <= 0 or lo > hi:
        raise ValueError(f"Invalid budgets: min={min_budget} max={max_budget}")

    b = max(lo, min(hi, float(budget)))
    scale = b / hi
    router_e = max(1, int(round(scale * router_base)))
    gate_e = max(1, int(round(scale * gate_base)))
    return router_e, gate_e


def is_full_training_budget(
    budget: float,
    max_budget: float,
    *,
    rel_tol: float = 1e-5,
) -> bool:
    """True when ``budget`` is (approximately) the maximum training fidelity."""
    return float(budget) >= float(max_budget) * (1.0 - rel_tol)
