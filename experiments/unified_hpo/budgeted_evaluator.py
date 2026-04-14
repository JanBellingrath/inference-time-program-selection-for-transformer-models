"""Budget-dependent evaluation: cheap proxy vs. expensive LLM.

Maps the single Hyperband budget to concrete evaluation resources:

- **Low/medium budgets**: calibrate open rate and compute proxy gain on a
  subset of routing-val.  No LLM needed.
- **High budget**: use the full routing-val for calibration *and* optionally
  run expensive LLM-based benchmark evaluation.

Training is *not* reduced at lower budgets — models are always trained fully.
Only evaluation fidelity varies with budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from experiments.unified_hpo.calibration import (
    CalibrationResult,
    calibrate_open_rate,
    compute_per_sample_deltas,
    compute_per_sample_deltas_from_best_delta,
)
from experiments.unified_hpo.threshold_prior import ThresholdPrior
from experiments.unified_hpo.trainer import TrainResult

logger = logging.getLogger(__name__)

# Default budget ladder (factor-3 geometric)
DEFAULT_BUDGET_LADDER = [64, 192, 576]
DEFAULT_MIN_BUDGET = 64
DEFAULT_MAX_BUDGET = 576


@dataclass
class EvalResult:
    """Combined output from budgeted evaluation."""

    # Calibration
    calibration: CalibrationResult
    proxy_gain: float = 0.0

    # Oracle proxy (upper-bound: assumes router always picks best route)
    oracle_proxy_gain: float = 0.0

    # Expensive LLM eval (populated only at highest budget if enabled)
    expensive_metrics: Optional[Dict[str, Any]] = None

    # Budget info
    budget: float = 0.0
    routing_val_size_used: int = 0
    ran_expensive_eval: bool = False


def routing_val_size_for_budget(
    budget: float,
    total_val_size: int,
) -> int:
    """Map a Hyperband budget to the number of routing-val samples to use.

    The budget value directly represents the target routing-val size, capped
    at the total available.
    """
    return min(int(budget), total_val_size)


def should_run_expensive_eval(budget: float, max_budget: float) -> bool:
    """Whether this budget level should trigger expensive LLM evaluation."""
    return budget >= max_budget


def evaluate_configuration(
    train_result: TrainResult,
    config: Dict,
    records: List[Dict],
    sequence_catalog: List,
    seq_to_idx: Dict[tuple, int],
    budget: float,
    max_budget: float,
    prior: ThresholdPrior,
    enable_expensive_eval: bool = False,
    expensive_eval_fn=None,
) -> EvalResult:
    """Run budgeted evaluation for one trained configuration.

    Steps:
    1. Compute per-sample deltas (router-predicted route's delta from explored
       data, plus oracle best_delta for comparison).
    2. Determine routing-val subset size from budget.
    3. Query the threshold prior for a starting open-rate center.
    4. Calibrate open rate on the budgeted subset using the prefix trick.
    5. If at the highest budget and expensive eval is enabled, run the LLM eval.
    """
    val_indices = train_result.val_indices
    val_scores = train_result.val_routing_scores
    router_preds = train_result.val_router_preds

    if val_indices is None or val_scores is None or router_preds is None:
        logger.error("TrainResult missing routing-val data; cannot evaluate.")
        return EvalResult(
            calibration=CalibrationResult(),
            proxy_gain=-1.0,
            budget=budget,
        )

    total_val = len(val_indices)
    n_use = routing_val_size_for_budget(budget, total_val)

    # ------------------------------------------------------------------
    # 1. Per-sample deltas
    # ------------------------------------------------------------------
    deltas = compute_per_sample_deltas(
        router_preds, records, val_indices, sequence_catalog, seq_to_idx,
    )
    oracle_deltas = compute_per_sample_deltas_from_best_delta(
        router_preds, records, val_indices,
    )

    # ------------------------------------------------------------------
    # 2. Query threshold prior
    # ------------------------------------------------------------------
    gating_mode = config.get("gating_mode", "gate_network")
    center_rho, sigma = prior.predict(
        gating_mode=gating_mode,
        target_source=config.get("target_source", "explored"),
        router_loss=config.get("router_loss", "hard_ce"),
        router_val_loss=train_result.router_val_loss,
        gate_val_loss=train_result.gate_val_loss,
        score_mean=train_result.score_mean,
        score_std=train_result.score_std,
        score_quantiles=train_result.score_quantiles,
        predicted_noop_rate=train_result.predicted_noop_rate,
        router_entropy_mean=train_result.router_entropy_mean,
    )

    # ------------------------------------------------------------------
    # 3. Calibrate open rate (prefix trick on budgeted subset)
    # ------------------------------------------------------------------
    cal_result = calibrate_open_rate(
        val_scores, deltas,
        prior_center=center_rho,
        prior_sigma=sigma,
        n_samples=n_use,
    )

    # Also compute oracle proxy gain at the same open rate for comparison
    oracle_cal = calibrate_open_rate(
        val_scores, oracle_deltas,
        prior_center=cal_result.best_rho,
        prior_sigma=0.01,
        n_samples=n_use,
    )

    eval_result = EvalResult(
        calibration=cal_result,
        proxy_gain=cal_result.best_gain,
        oracle_proxy_gain=oracle_cal.best_gain,
        budget=budget,
        routing_val_size_used=n_use,
    )

    # ------------------------------------------------------------------
    # 4. Expensive eval (highest budget only, if enabled)
    # ------------------------------------------------------------------
    if (enable_expensive_eval
            and should_run_expensive_eval(budget, max_budget)
            and expensive_eval_fn is not None):
        try:
            expensive_metrics = expensive_eval_fn(
                train_result=train_result,
                config=config,
                calibration=cal_result,
            )
            eval_result.expensive_metrics = expensive_metrics
            eval_result.ran_expensive_eval = True
            logger.info(
                "Expensive eval complete: %s",
                {k: f"{v:.4f}" if isinstance(v, float) else v
                 for k, v in expensive_metrics.items()},
            )
        except Exception as e:
            logger.error("Expensive eval failed: %s", e, exc_info=True)

    return eval_result
