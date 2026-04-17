"""Evaluation: proxy calibration and optional expensive LLM eval.

Assumes ``train_and_summarize`` already produced ``TrainResult`` with full
routing-val scores. Open-rate calibration uses **all** val rows (no subsampling).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from experiments.unified_hpo.calibration import (
    CalibrationResult,
    calibrate_open_rate,
    compute_per_sample_deltas,
    compute_per_sample_deltas_from_best_delta,
)
from experiments.unified_hpo.trainer import TrainResult

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Combined output from evaluation."""

    calibration: CalibrationResult
    proxy_gain: float = 0.0

    # Oracle proxy (upper-bound: pred != 0 uses best_delta)
    oracle_proxy_gain: float = 0.0

    expensive_metrics: Optional[Dict[str, Any]] = None

    routing_val_size_used: int = 0
    ran_expensive_eval: bool = False


def evaluate_configuration(
    train_result: TrainResult,
    config: Dict,
    records: List[Dict],
    sequence_catalog: List,
    seq_to_idx: Dict[tuple, int],
    enable_expensive_eval: bool = False,
    expensive_eval_fn=None,
) -> EvalResult:
    """Calibrate open rate on full routing-val; optionally run LLM eval.

    Steps:
    1. Compute per-sample deltas (predicted-route proxy and oracle best_delta).
    2. Vectorized argmax calibration on all routing-val scores/deltas.
    3. If ``enable_expensive_eval`` and a function is provided, run expensive eval.
    """
    val_indices = train_result.val_indices
    val_scores = train_result.val_routing_scores
    router_preds = train_result.val_router_preds

    if val_indices is None or val_scores is None or router_preds is None:
        logger.error("TrainResult missing routing-val data; cannot evaluate.")
        return EvalResult(
            calibration=CalibrationResult(),
            proxy_gain=-1.0,
        )

    total_val = len(val_indices)

    deltas = compute_per_sample_deltas(
        router_preds, records, val_indices, sequence_catalog, seq_to_idx,
    )
    oracle_deltas = compute_per_sample_deltas_from_best_delta(
        router_preds, records, val_indices,
    )

    cal_result = calibrate_open_rate(val_scores, deltas)
    oracle_cal = calibrate_open_rate(val_scores, oracle_deltas)

    eval_result = EvalResult(
        calibration=cal_result,
        proxy_gain=cal_result.best_gain,
        oracle_proxy_gain=oracle_cal.best_gain,
        routing_val_size_used=total_val,
    )

    if enable_expensive_eval and expensive_eval_fn is not None:
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
