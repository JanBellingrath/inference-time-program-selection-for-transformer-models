"""Persistent archive and prior model for open-rate prediction.

The archive stores one JSONL record per completed HPO trial, including the
full configuration, training summaries, score distributions, calibration
results, and the objective returned to SMAC.

The prior model learns to predict a good starting open rate from this archive,
so that calibration can begin with a narrow local search instead of a dense sweep.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

RHO_MIN = 0.01
RHO_MAX = 0.50


# ---------------------------------------------------------------------------
# Archive record schema
# ---------------------------------------------------------------------------

@dataclass
class ArchiveRecord:
    """One record in the threshold-prior archive (see plan §17)."""

    # Identity
    run_id: str = ""
    timestamp: float = 0.0
    benchmark: str = ""
    seed: int = 42
    budget: float = 0.0
    config_hash: str = ""
    config: Dict[str, Any] = field(default_factory=dict)

    # Branch summary
    gating_mode: str = ""
    target_source: str = ""
    router_loss: str = ""
    router_train_subset: str = ""

    # Training summaries
    router_val_loss: float = float("inf")
    gate_val_loss: float = float("inf")
    predicted_noop_rate: float = 0.0
    gate_positive_rate: float = 0.0

    # Score summaries on routing-val
    score_mean: float = 0.0
    score_std: float = 0.0
    score_min: float = 0.0
    score_max: float = 0.0
    score_quantiles: Dict[str, float] = field(default_factory=dict)
    router_entropy_mean: float = 0.0
    mean_top1_minus_noop_margin: float = 0.0
    frac_router_argmax_noop: float = 0.0

    # Calibration result
    prior_predicted_rho: float = 0.0
    candidates_tested: List[Tuple[float, float]] = field(default_factory=list)
    best_rho: float = 0.0
    best_threshold: float = 0.0
    realized_open_fraction: float = 0.0
    proxy_gain: float = 0.0

    # Selection result
    objective_returned: float = 0.0
    expensive_gain: Optional[float] = None
    anchor_accuracy: Optional[float] = None
    routed_accuracy: Optional[float] = None
    helped_count: Optional[int] = None
    hurt_count: Optional[int] = None


def _config_hash(config: Dict) -> str:
    """Deterministic hash of the active configuration parameters."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Archive I/O
# ---------------------------------------------------------------------------

class ThresholdPriorArchive:
    """Persistent JSONL archive of completed HPO trial records."""

    def __init__(self, path: str):
        self.path = path
        self._records: List[ArchiveRecord] = []
        self._load()

    def _load(self) -> None:
        if not os.path.isfile(self.path):
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    rec = ArchiveRecord(**{
                        k: v for k, v in d.items()
                        if k in ArchiveRecord.__dataclass_fields__
                    })
                    self._records.append(rec)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning("Skipping malformed archive line: %s", e)

    def append(self, record: ArchiveRecord) -> None:
        """Append a record and flush to disk."""
        self._records.append(record)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(record), default=str) + "\n")

    @property
    def records(self) -> List[ArchiveRecord]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)


# ---------------------------------------------------------------------------
# Logit transform helpers (bounded rho <-> unbounded logit)
# ---------------------------------------------------------------------------

def _rho_to_logit(rho: float) -> float:
    """Map open rate from [RHO_MIN, RHO_MAX] to unbounded logit space."""
    rho_clipped = max(RHO_MIN + 1e-6, min(RHO_MAX - 1e-6, rho))
    normalized = (rho_clipped - RHO_MIN) / (RHO_MAX - RHO_MIN)
    return math.log(normalized / (1.0 - normalized))


def _logit_to_rho(logit: float) -> float:
    """Map from unbounded logit space back to [RHO_MIN, RHO_MAX]."""
    normalized = 1.0 / (1.0 + math.exp(-logit))
    return RHO_MIN + normalized * (RHO_MAX - RHO_MIN)


# ---------------------------------------------------------------------------
# Feature extraction for the prior model
# ---------------------------------------------------------------------------

_GATING_MAP = {"gate_network": 0, "router_confidence": 1, "delta_gate": 2}
_TARGET_MAP = {"best_seq": 0, "explored": 1}
_LOSS_MAP = {"hard_ce": 0, "soft_ce": 1, "top_ce": 2}


def _record_to_features(rec: ArchiveRecord) -> np.ndarray:
    """Extract a feature vector from an archive record for the ridge model."""
    feats = [
        _GATING_MAP.get(rec.gating_mode, 0),
        _TARGET_MAP.get(rec.target_source, 0),
        _LOSS_MAP.get(rec.router_loss, 0),
        rec.router_val_loss if math.isfinite(rec.router_val_loss) else 5.0,
        rec.gate_val_loss if math.isfinite(rec.gate_val_loss) else 5.0,
        rec.score_mean,
        rec.score_std,
        rec.score_quantiles.get("q0.50", 0.0),
        rec.score_quantiles.get("q0.80", 0.0),
        rec.score_quantiles.get("q0.95", 0.0),
        rec.predicted_noop_rate,
        rec.router_entropy_mean,
    ]
    return np.array(feats, dtype=np.float64)


# ---------------------------------------------------------------------------
# Prior model
# ---------------------------------------------------------------------------

# Default centers when the archive is too small for regression
_DEFAULT_CENTERS = {
    "gate_network": 0.05,
    "router_confidence": 0.10,
    "delta_gate": 0.05,
}
_DEFAULT_SIGMA = 0.15

# Minimum archive size before fitting the ridge model
_MIN_ARCHIVE_FOR_RIDGE = 10

# Re-fit interval (number of new records between re-fits)
_REFIT_INTERVAL = 10


class ThresholdPrior:
    """Predicts a good starting open rate from past HPO trials.

    Cold start: returns hard-coded per-gating-mode defaults with broad
    uncertainty.

    Warm (10+ archive entries): fits a ``RidgeCV`` on archive features
    predicting ``logit(best_rho)``.
    """

    def __init__(self, archive: ThresholdPriorArchive):
        self._archive = archive
        self._model = None
        self._last_fit_size: int = 0

    def predict(
        self,
        gating_mode: str,
        target_source: str,
        router_loss: str,
        router_val_loss: float,
        gate_val_loss: float,
        score_mean: float,
        score_std: float,
        score_quantiles: Dict[str, float],
        predicted_noop_rate: float,
        router_entropy_mean: float,
    ) -> Tuple[float, float]:
        """Predict (center_rho, sigma) for the open-rate local search.

        Returns the predicted open-rate center and an uncertainty estimate.
        """
        # Cold start
        if len(self._archive) < _MIN_ARCHIVE_FOR_RIDGE:
            center = _DEFAULT_CENTERS.get(gating_mode, 0.05)
            return center, _DEFAULT_SIGMA

        # Fit / re-fit if needed
        if (self._model is None
                or len(self._archive) - self._last_fit_size >= _REFIT_INTERVAL):
            self._fit()

        if self._model is None:
            center = _DEFAULT_CENTERS.get(gating_mode, 0.05)
            return center, _DEFAULT_SIGMA

        # Build feature vector for prediction
        dummy_rec = ArchiveRecord(
            gating_mode=gating_mode,
            target_source=target_source,
            router_loss=router_loss,
            router_val_loss=router_val_loss,
            gate_val_loss=gate_val_loss,
            score_mean=score_mean,
            score_std=score_std,
            score_quantiles=score_quantiles,
            predicted_noop_rate=predicted_noop_rate,
            router_entropy_mean=router_entropy_mean,
        )
        X = _record_to_features(dummy_rec).reshape(1, -1)

        try:
            logit_pred = float(self._model.predict(X)[0])
            center = _logit_to_rho(logit_pred)
        except Exception:
            center = _DEFAULT_CENTERS.get(gating_mode, 0.05)

        # Estimate sigma from residuals
        sigma = max(0.02, self._residual_std * 0.5) if hasattr(self, "_residual_std") else _DEFAULT_SIGMA

        return center, sigma

    def _fit(self) -> None:
        """Fit the ridge regression model on the archive."""
        records = self._archive.records
        if len(records) < _MIN_ARCHIVE_FOR_RIDGE:
            return

        valid = [r for r in records if r.best_rho > 0 and r.proxy_gain > -0.5]
        if len(valid) < _MIN_ARCHIVE_FOR_RIDGE:
            return

        try:
            from sklearn.linear_model import RidgeCV
        except ImportError:
            logger.warning("scikit-learn not available; threshold prior will use defaults.")
            return

        X = np.stack([_record_to_features(r) for r in valid])
        y = np.array([_rho_to_logit(r.best_rho) for r in valid])

        # Replace NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0)
        y = np.nan_to_num(y, nan=0.0, posinf=3.0, neginf=-3.0)

        try:
            model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
            model.fit(X, y)
            self._model = model
            self._last_fit_size = len(records)

            preds = model.predict(X)
            self._residual_std = float(np.std(y - preds))
            logger.info(
                "Threshold prior fitted on %d records (alpha=%.3f, residual_std=%.3f)",
                len(valid), model.alpha_, self._residual_std,
            )
        except Exception as e:
            logger.warning("Failed to fit threshold prior: %s", e)
            self._model = None
