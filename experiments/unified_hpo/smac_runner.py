"""SMAC + Hyperband orchestration, W&B logging, and best-model persistence.

- **Training fidelity** (Hyperband ``budget``): scales router/gate epoch counts;
  cheap low-budget runs prune the search space.
- **Evaluation**: always full routing-val for calibration / proxy gain (no eval subsampling).
- Checkpoints and optional expensive LLM eval run only at **maximum** training budget.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import numpy as np

import wandb

from experiments.unified_hpo.budgeted_evaluator import EvalResult, evaluate_configuration
from experiments.unified_hpo.calibration import CalibrationResult
from experiments.unified_hpo.compositional_objective import (
    train_and_score_compositional,
)
from experiments.unified_hpo.search_space import build_configspace, config_to_dict
from experiments.unified_hpo.search_space_compositional import (
    build_configspace_compositional,
    config_to_dict as compositional_config_to_dict,
)
from experiments.unified_hpo.threshold_prior import (
    ArchiveRecord,
    ThresholdPriorArchive,
    _config_hash,
)
from experiments.unified_hpo.training_budget import (
    DEFAULT_TRAIN_MAX_BUDGET,
    DEFAULT_TRAIN_MIN_BUDGET,
    is_full_training_budget,
    router_gate_epochs_from_training_budget,
)
from experiments.unified_hpo.trainer import TrainResult, train_and_summarize

logger = logging.getLogger(__name__)

# Objective value returned for failed configurations (SMAC minimizes, so
# we return a large positive number to indicate failure).
FAILURE_COST = 1.0


# ---------------------------------------------------------------------------
# Best-model tracking
# ---------------------------------------------------------------------------

class BestModelTracker:
    """Tracks and persists the best routing system found so far."""

    def __init__(self, output_dir: str, benchmark: str, *, resume: bool = False):
        self.output_dir = os.path.join(output_dir, f"best_routing_system_{benchmark}")
        self.benchmark = benchmark
        self.best_proxy_gain: float = -float("inf")
        self.best_expensive_gain: Optional[float] = None
        self.best_config: Optional[Dict] = None
        if resume:
            meta_path = os.path.join(self.output_dir, "meta.json")
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    if "proxy_gain" in meta and meta["proxy_gain"] is not None:
                        self.best_proxy_gain = float(meta["proxy_gain"])
                    if "expensive_gain" in meta and meta["expensive_gain"] is not None:
                        self.best_expensive_gain = float(meta["expensive_gain"])
                    if "config" in meta and meta["config"] is not None:
                        self.best_config = meta["config"]
                    logger.info(
                        "Resumed best-model state from %s  proxy_gain=%.5f",
                        meta_path, self.best_proxy_gain,
                    )
                except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
                    logger.warning("Could not load meta.json for resume: %s", e)
            else:
                logger.info(
                    "Resume requested but no meta.json at %s; best tracker starts fresh.",
                    meta_path,
                )

    def maybe_save(
        self,
        proxy_gain: float,
        train_result: TrainResult,
        config: Dict,
        calibration: CalibrationResult,
        expensive_gain: Optional[float] = None,
        wandb_run=None,
    ) -> bool:
        """Save weights if this configuration beats the current best.

        Returns True if a new best was saved.
        """
        is_new_best = proxy_gain > self.best_proxy_gain
        if not is_new_best:
            return False

        self.best_proxy_gain = proxy_gain
        self.best_expensive_gain = expensive_gain
        self.best_config = config

        os.makedirs(self.output_dir, exist_ok=True)

        # Save router weights
        torch.save(
            {
                "model_state_dict": train_result.router.state_dict(),
                "config": config,
                "proxy_gain": proxy_gain,
                "calibration": {
                    "best_rho": calibration.best_rho,
                    "best_threshold": calibration.best_threshold,
                    "realized_open_fraction": calibration.realized_open_fraction,
                },
            },
            os.path.join(self.output_dir, "router.pt"),
        )

        # Save gate weights if present
        if train_result.gate is not None:
            torch.save(
                {"model_state_dict": train_result.gate.state_dict()},
                os.path.join(self.output_dir, "gate.pt"),
            )
        if train_result.delta_gate is not None:
            torch.save(
                {"model_state_dict": train_result.delta_gate.state_dict()},
                os.path.join(self.output_dir, "delta_gate.pt"),
            )

        # Save full config + calibration as JSON
        meta = {
            "config": config,
            "proxy_gain": proxy_gain,
            "expensive_gain": expensive_gain,
            "calibration_best_rho": calibration.best_rho,
            "calibration_best_threshold": calibration.best_threshold,
            "benchmark": self.benchmark,
        }
        with open(os.path.join(self.output_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

        # Upload to W&B as artifact
        if wandb_run is not None:
            try:
                artifact = wandb.Artifact(
                    name=f"best-routing-{self.benchmark}",
                    type="model",
                    metadata=meta,
                )
                artifact.add_dir(self.output_dir)
                wandb_run.log_artifact(artifact)
            except Exception as e:
                logger.warning("Failed to upload W&B artifact: %s", e)

        logger.info(
            "NEW BEST routing system saved: proxy_gain=%.5f  rho=%.3f  -> %s",
            proxy_gain, calibration.best_rho, self.output_dir,
        )
        return True


# ---------------------------------------------------------------------------
# W&B logging helpers
# ---------------------------------------------------------------------------


def _define_hpo_wandb_charts(router_kind: str) -> None:
    """Register custom x-axis ``hpo_trial`` = SMAC trial index; charts use that for x, not the internal counter.

    W&B only binds lines to the intended x-axis if the step is a *logged metric* plus
    ``step_metric=``; relying on ``log(..., step=k)`` alone often leaves the default
    workspace with no/empty line charts. See https://docs.wandb.ai/guides/track/log/customize-logging-axes
    """
    # Step key first (name without "/" avoids odd grouping edge cases in older clients).
    try:
        wandb.define_metric("hpo_trial", summary="max")
    except Exception as e:  # noqa: S110
        logger.debug("define_metric hpo_trial: %s", e)
    if router_kind == "compositional":
        for name, how in [
            ("hpo/val_obs_top1", "max"),
            ("hpo/val_obs_top3", "max"),
            ("hpo/val_obs_top5", "max"),
            # Δ log P(correct) on val (nats). This is the metric optimized
            # during training (checkpoint_metric=mean_uplift). An uplift of
            # ~0.2 nats corresponds to a ~5pp increase in P(correct) on
            # average; it is NOT a 20-point accuracy gain.
            ("hpo/mean_uplift_nats", "max"),
            # Kept for backward-compatibility with old W&B dashboards; it
            # carries the same value as mean_uplift_nats (log-prob delta).
            ("hpo/mean_uplift_vs_anchor", "max"),
            # Unconditional binary-accuracy gain in percentage points, from
            # an external LLM eval. Only populated for trials that had
            # their final checkpoint re-scored downstream; NaN otherwise.
            ("hpo/mean_uplift_pp", "max"),
            ("hpo/external_eval_n", "max"),
            ("hpo/router_acc", "max"),
            ("hpo/anchor_acc", "max"),
            ("hpo/uplift_vs_best_route", "max"),
            ("hpo/frac_oracle_uplift", "max"),
            ("hpo/dense_top1", "max"),
            ("hpo/proxy", "max"),
            ("hpo/smac_cost", "min"),
            ("hpo/router_val_loss", "min"),
            ("hpo/budget", "max"),
            ("hpo/epochs", "max"),
            ("hpo/wall_s", "max"),
            ("hpo/new_best", "max"),
            ("hpo/trial_failed", "max"),
        ]:
            try:
                wandb.define_metric(name, step_metric="hpo_trial", summary=how)
            except Exception as e:  # noqa: S110
                logger.debug("define_metric %s: %s", name, e)
    else:
        for name, how in [
            ("hpo/proxy", "max"),
            ("hpo/smac_cost", "min"),
            ("hpo/val_loss", "min"),
            ("hpo/entropy_mean", "min"),
            ("hpo/budget", "max"),
            ("hpo/wall_s", "max"),
            ("hpo/new_best", "max"),
        ]:
            try:
                wandb.define_metric(name, step_metric="hpo_trial", summary=how)
            except Exception as e:  # noqa: S110
                logger.debug("define_metric %s: %s", name, e)
        try:
            wandb.define_metric("hpo/cfg_gating", step_metric="hpo_trial")
        except Exception as e:  # noqa: S110
            logger.debug("define_metric hpo/cfg_gating: %s", e)
    for name, how in [
        ("hpo/wb_layout", "max"),
        ("hpo/router_kind", "max"),
        ("hpo/resumed_smac", "max"),
        ("hpo/continues_from_trial", "max"),
    ]:
        try:
            wandb.define_metric(name, step_metric="hpo_trial", summary=how)
        except Exception as e:  # noqa: S110
            logger.debug("define_metric %s: %s", name, e)


def _hpo_cumulative_compositional_chart(history: List[Dict[str, Any]]) -> Any:
    """One Custom Chart with multiple lines (always visible under Media / Charts with data)."""
    if not history:
        return None
    xs: List[int] = []
    for h in history:
        v = h.get("hpo_trial")
        if v is not None:
            xs.append(int(v))
    if len(xs) != len(history):
        return None
    line_defs = [
        ("hpo/val_obs_top1", "val_obs_top1"),
        ("hpo/val_obs_top3", "val_obs_top3"),
        ("hpo/mean_uplift_vs_anchor", "mean_uplift_vs_anchor"),
        ("hpo/proxy", "proxy"),
    ]
    ys: List[List[float]] = []
    keys: List[str] = []
    for key, leg in line_defs:
        keys.append(leg)
        ys.append([float(h.get(key, 0.0) or 0.0) for h in history])
    return wandb.plot.line_series(
        xs, ys, keys=keys, title="HPO (headline metrics vs trial)", xname="SMAC trial",
    )


def _log_trial_wandb_compositional(
    train_result: TrainResult,
    eval_result: EvalResult,
    trial_idx: int,
    wall_time: float,
    is_new_best: bool,
    training_budget: float,
) -> Dict[str, Any]:
    """One row per trial: validation ranking + downstream vs baselines, then SMAC cost."""
    m = getattr(train_result, "compositional_metrics", None) or {}
    vd = m.get("val_downstream")
    if not isinstance(vd, dict):
        vd = {}
    vr = m.get("val_ranking")
    if not isinstance(vr, dict):
        vr = {}
    r_acc = float(vd.get("router_acc", 0.0) or 0.0)
    a_acc = float(vd.get("anchor_acc", 0.0) or 0.0)
    bf = float(vd.get("best_fixed_acc", 0.0) or 0.0)
    mean_uplift_nats = float(vd.get("mean_uplift", 0.0) or 0.0)
    # External LLM eval of final checkpoint (if provided by trainer /
    # compositional_objective); reports the unconditional accuracy gain in
    # percentage points. NaN / unset when no external eval was run.
    ext = getattr(train_result, "compositional_external_eval", None) or {}
    ext_uncond_gain_pp = float(ext.get("unconditional_gain_pp", float("nan")))
    ext_router_acc = float(ext.get("router_acc", float("nan")))
    ext_anchor_acc = float(ext.get("anchor_acc", float("nan")))
    ext_n = int(ext.get("n", 0))
    # Order: val observability, log-prob + pp uplifts, downstream deltas vs
    # baselines, then SMAC + run metadata.
    out: Dict[str, Any] = {
        "hpo/val_obs_top1": float(vr.get("obs_top1_acc", 0.0) or 0.0),
        "hpo/val_obs_top3": float(vr.get("obs_top3_acc", 0.0) or 0.0),
        "hpo/val_obs_top5": float(vr.get("obs_top5_acc", 0.0) or 0.0),
        "hpo/mean_uplift_nats": mean_uplift_nats,
        # Backward compatibility with older dashboards (same value as nats).
        "hpo/mean_uplift_vs_anchor": mean_uplift_nats,
        "hpo/mean_uplift_pp": ext_uncond_gain_pp,
        "hpo/external_eval_n": ext_n,
        "hpo/router_acc": r_acc,
        "hpo/anchor_acc": a_acc,
        "hpo/external_router_acc": ext_router_acc,
        "hpo/external_anchor_acc": ext_anchor_acc,
        "hpo/uplift_vs_best_route": r_acc - bf,
        "hpo/frac_oracle_uplift": float(vd.get("frac_oracle", 0.0) or 0.0),
        "hpo/dense_top1": float(vd.get("dense_top1_acc", 0.0) or 0.0),
        "hpo/proxy": float(eval_result.proxy_gain),
        "hpo/smac_cost": -float(eval_result.proxy_gain),
        "hpo/router_val_loss": float(getattr(train_result, "router_val_loss", 0.0) or 0.0),
        "hpo/budget": float(training_budget),
        "hpo/epochs": int(getattr(train_result, "router_epochs_used", 0) or 0),
        "hpo/wall_s": float(wall_time),
        "hpo/new_best": int(is_new_best),
    }
    return out


def _log_trial_wandb_fine(
    config: Dict,
    train_result: TrainResult,
    eval_result: EvalResult,
    wall_time: float,
    is_new_best: bool,
    training_budget: float,
) -> Dict[str, Any]:
    return {
        "hpo/proxy": float(eval_result.proxy_gain),
        "hpo/smac_cost": -float(eval_result.proxy_gain),
        "hpo/val_loss": float(train_result.router_val_loss or 0.0),
        "hpo/entropy_mean": float(getattr(train_result, "router_entropy_mean", 0.0) or 0.0),
        "hpo/budget": float(training_budget),
        "hpo/wall_s": float(wall_time),
        "hpo/new_best": int(is_new_best),
        "hpo/cfg_gating": str(config.get("gating_mode", "")),
    }


def _log_trial_to_wandb(
    config: Dict,
    train_result: TrainResult,
    eval_result: EvalResult,
    trial_idx: int,
    wall_time: float,
    is_new_best: bool,
    training_budget: float,
    *,
    router_kind: str = "fine",
    hpo_trial_history: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Log one W&B point per HPO trial. X-axis = ``hpo_trial`` (see ``_define_hpo_wandb_charts``); do not use ``log(..., step=)`` here."""
    try:
        if router_kind == "compositional":
            row = _log_trial_wandb_compositional(
                train_result, eval_result, trial_idx, wall_time,
                is_new_best, training_budget,
            )
        else:
            row = _log_trial_wandb_fine(
                config, train_result, eval_result, wall_time,
                is_new_best, training_budget,
            )
        row["hpo_trial"] = float(trial_idx)
        if hpo_trial_history is not None and router_kind == "compositional":
            hpo_trial_history.append({k: v for k, v in row.items() if k != "hpo/curves_headline"})
            ch = _hpo_cumulative_compositional_chart(hpo_trial_history)
            if ch is not None:
                row["hpo/curves_headline"] = ch
        # Custom x-axis: omit ``step=`` so W&B uses ``hpo_trial`` (define_metric) for line charts.
        wandb.log(row)
    except Exception as e:
        logger.warning("W&B hpo_trial log failed: %s", e)


def log_sweep_summary(
    archive: ThresholdPriorArchive,
    configspace,
) -> None:
    """Log end-of-sweep summary: table + parallel coordinates."""
    records = archive.records
    if not records:
        return

    # Summary table
    columns = [
        "run_id", "gating_mode", "target_source", "router_loss",
        "budget", "proxy_gain", "best_rho", "router_val_loss",
    ]
    table_data = []
    for rec in records:
        table_data.append([
            rec.run_id[:8],
            rec.gating_mode,
            rec.target_source,
            rec.router_loss,
            rec.budget,
            rec.proxy_gain,
            rec.best_rho,
            rec.router_val_loss,
        ])
    table = wandb.Table(columns=columns, data=table_data)
    wandb.log({"sweep_summary_table": table})

    # Parallel coordinates plot
    pc_columns = [
        "gating_mode", "target_source", "router_loss",
        "router_depth", "router_width", "gate_depth", "gate_width",
        "proxy_gain",
    ]
    pc_data = []
    for rec in records:
        row = [
            rec.gating_mode,
            rec.target_source,
            rec.router_loss,
            rec.config.get("router_depth", 0),
            rec.config.get("router_width", 0),
            rec.config.get("gate_depth", 0),
            rec.config.get("gate_width", 0),
            rec.proxy_gain,
        ]
        pc_data.append(row)

    pc_table = wandb.Table(columns=pc_columns, data=pc_data)
    wandb.log({"parallel_coordinates_table": pc_table})

    # Proxy gain scatter (logged as table; W&B auto-renders scatter from Table)
    scatter_data = [[rec.best_rho, rec.proxy_gain] for rec in records]
    scatter_table = wandb.Table(columns=["open_rate", "proxy_gain"], data=scatter_data)
    try:
        wandb.log({"proxy_gain_vs_open_rate": wandb.plot.scatter(
            scatter_table, x="open_rate", y="proxy_gain",
            title="Proxy Gain vs. Open Rate",
        )})
    except (AttributeError, TypeError):
        wandb.log({"proxy_gain_vs_open_rate_table": scatter_table})


# ---------------------------------------------------------------------------
# SMAC target function builder
# ---------------------------------------------------------------------------

def build_target_function(
    residuals: Optional[torch.Tensor],
    gate_labels: Optional[List[int]],
    records: Optional[List[Dict]],
    seq_to_idx: Optional[Dict[tuple, int]],
    num_classes: Optional[int],
    best_deltas: Optional[List[float]],
    sequence_catalog: Optional[List],
    d_model: int,
    device: torch.device,
    benchmark: str,
    archive: ThresholdPriorArchive,
    tracker: BestModelTracker,
    min_budget: float,
    max_budget: float,
    enable_expensive_eval: bool = False,
    expensive_eval_fn: Optional[Callable] = None,
    wandb_run=None,
    seed: int = 42,
    router_kind: str = "fine",
    compositional_ctx: Optional[Dict[str, Any]] = None,
    initial_trial_idx: int = 0,
) -> Callable:
    """Build the SMAC target (train at budget → full-val calibration → cost).

    ``budget`` scales **training epochs only**; calibration always uses all
    routing-val rows. When ``router_kind == "compositional"``, the
    compositional objective is used in place of ``train_and_summarize`` /
    ``evaluate_configuration``; the rest of the bookkeeping (W&B, archive,
    best-tracker) is unchanged.
    """
    if router_kind not in ("fine", "compositional"):
        raise ValueError(f"unknown router_kind={router_kind!r}")
    if router_kind == "compositional" and compositional_ctx is None:
        raise ValueError("router_kind='compositional' requires compositional_ctx")

    trial_counter = [int(initial_trial_idx)]
    hpo_trial_history: List[Dict[str, Any]] = []

    def target_function(config, seed: int = 0, budget: float = max_budget) -> float:
        """SMAC target function.  Returns cost (lower is better)."""
        t0 = time.time()
        trial_idx = trial_counter[0]
        trial_counter[0] += 1
        run_id = str(uuid.uuid4())[:12]

        router_e, gate_e = router_gate_epochs_from_training_budget(
            budget, min_budget, max_budget,
        )

        try:
            if router_kind == "fine":
                cfg = config_to_dict(config)
            else:
                cfg = compositional_config_to_dict(config)

            if router_kind == "fine":
                logger.info(
                    "=== Trial %d  train_budget=%.1f  router_ep=%d  gate_ep=%d  "
                    "gating=%s  target=%s  loss=%s ===",
                    trial_idx, budget, router_e, gate_e,
                    cfg.get("gating_mode"), cfg.get("target_source"), cfg.get("router_loss"),
                )

                train_result = train_and_summarize(
                    config=cfg,
                    residuals=residuals,
                    gate_labels=gate_labels,
                    records=records,
                    seq_to_idx=seq_to_idx,
                    num_classes=num_classes,
                    best_deltas=best_deltas,
                    d_model=d_model,
                    device=device,
                    seed=seed,
                    router_epochs=router_e,
                    gate_epochs=gate_e,
                )

                full_train = is_full_training_budget(budget, max_budget)
                run_expensive = bool(
                    enable_expensive_eval and full_train and expensive_eval_fn is not None,
                )

                eval_result = evaluate_configuration(
                    train_result=train_result,
                    config=cfg,
                    records=records,
                    sequence_catalog=sequence_catalog,
                    seq_to_idx=seq_to_idx,
                    enable_expensive_eval=run_expensive,
                    expensive_eval_fn=expensive_eval_fn,
                )
            else:
                ctx = compositional_ctx or {}
                logger.info(
                    "=== Trial %d  train_budget=%.1f  router_ep=%d  "
                    "kind=compositional scope=%s benchmarks=%s use_pairs=%s ===",
                    trial_idx, budget, router_e,
                    ctx.get("scope"), ctx.get("benchmarks"), cfg.get("use_pairs"),
                )

                train_result = train_and_score_compositional(
                    config=cfg,
                    artifacts=ctx["artifacts"],
                    benchmarks=ctx["benchmarks"],
                    dense_paths=ctx.get("dense_paths") or {},
                    dense_keep_mask_paths=ctx.get("dense_keep_mask_paths") or None,
                    scope=ctx.get("scope", "single"),
                    router_epochs=router_e,
                    batch_size=int(ctx.get("batch_size", 64)),
                    val_fraction=float(ctx.get("val_fraction", 0.15)),
                    seed=seed,
                    device=device,
                    objective_metric=ctx.get("objective_metric", "mean_uplift"),
                    use_full_sequence=bool(ctx.get("use_full_sequence", False)),
                    wandb_run=wandb_run,
                    wandb_prefix=f"trial_{trial_idx:05d}",
                    wandb_step_offset=int(trial_idx) * 1000,
                    use_dense_supervision=(
                        True if ctx.get("objective_metric", "mean_uplift") == "mean_uplift" else None
                    ),
                    downstream_eval_every=0,
                    local_moebius_paths=ctx.get("local_moebius_paths") or None,
                    train_test_holdout_count=int(ctx.get("train_test_holdout_count", 0)),
                )

                full_train = is_full_training_budget(budget, max_budget)
                proxy = float(getattr(train_result, "compositional_proxy", 0.0))
                eval_result = EvalResult(
                    calibration=CalibrationResult(
                        best_rho=0.0,
                        best_threshold=0.0,
                        best_gain=proxy,
                        realized_open_fraction=0.0,
                        candidates_tested=[(0.0, proxy)],
                    ),
                    proxy_gain=proxy,
                    oracle_proxy_gain=proxy,
                    routing_val_size_used=0,
                    ran_expensive_eval=False,
                )

            proxy_gain = eval_result.proxy_gain
            wall_time = time.time() - t0

            # --- Best model tracking (only at max training budget) ---
            expensive_gain = None
            if eval_result.expensive_metrics is not None:
                expensive_gain = eval_result.expensive_metrics.get("unconditional_gain")
            is_new_best = False
            if full_train:
                is_new_best = tracker.maybe_save(
                    proxy_gain=proxy_gain,
                    train_result=train_result,
                    config=cfg,
                    calibration=eval_result.calibration,
                    expensive_gain=expensive_gain,
                    wandb_run=wandb_run,
                )

            # --- W&B: one step per trial (flat hpo/*); inner training logs disabled for compositional HPO ---
            if wandb_run is not None:
                _log_trial_to_wandb(
                    cfg, train_result, eval_result,
                    trial_idx, wall_time, is_new_best, budget,
                    router_kind=router_kind,
                    hpo_trial_history=hpo_trial_history
                    if router_kind == "compositional"
                    else None,
                )

            # --- Archive update ---
            archive_rec = ArchiveRecord(
                run_id=run_id,
                timestamp=time.time(),
                benchmark=benchmark,
                seed=seed,
                budget=float(budget),
                config_hash=_config_hash(cfg),
                config=cfg,
                gating_mode=cfg.get("gating_mode", ""),
                target_source=cfg.get("target_source", ""),
                router_loss=cfg.get("router_loss", ""),
                router_train_subset=cfg.get("router_train_subset", ""),
                router_val_loss=train_result.router_val_loss,
                gate_val_loss=train_result.gate_val_loss,
                predicted_noop_rate=train_result.predicted_noop_rate,
                score_mean=train_result.score_mean,
                score_std=train_result.score_std,
                score_min=train_result.score_min,
                score_max=train_result.score_max,
                score_quantiles=train_result.score_quantiles,
                router_entropy_mean=train_result.router_entropy_mean,
                frac_router_argmax_noop=train_result.predicted_noop_rate,
                prior_predicted_rho=0.0,
                candidates_tested=eval_result.calibration.candidates_tested,
                best_rho=eval_result.calibration.best_rho,
                best_threshold=eval_result.calibration.best_threshold,
                realized_open_fraction=eval_result.calibration.realized_open_fraction,
                proxy_gain=proxy_gain,
                objective_returned=-proxy_gain,
            )
            if eval_result.expensive_metrics is not None:
                archive_rec.expensive_gain = eval_result.expensive_metrics.get("unconditional_gain")
                archive_rec.anchor_accuracy = eval_result.expensive_metrics.get("anchor_accuracy")
                archive_rec.routed_accuracy = eval_result.expensive_metrics.get("routed_accuracy")
                archive_rec.helped_count = eval_result.expensive_metrics.get("helped_when_opened")
                archive_rec.hurt_count = eval_result.expensive_metrics.get("hurt_when_opened")
            archive.append(archive_rec)

            logger.info(
                "Trial %d done: proxy_gain=%.5f  rho=%.3f  "
                "router_loss=%.4f  (%.1fs)%s",
                trial_idx, proxy_gain, eval_result.calibration.best_rho,
                train_result.router_val_loss, wall_time,
                "  *** NEW BEST ***" if is_new_best else "",
            )

            # SMAC minimizes, so negate the gain
            return -proxy_gain

        except Exception as e:
            logger.error("Trial %d FAILED: %s", trial_idx, e, exc_info=True)

            fail_cfg: Dict[str, Any] = {}
            try:
                fail_cfg = (
                    config_to_dict(config)
                    if router_kind == "fine"
                    else compositional_config_to_dict(config)
                )
            except Exception:
                pass

            archive.append(ArchiveRecord(
                run_id=run_id,
                timestamp=time.time(),
                benchmark=benchmark,
                seed=seed,
                budget=float(budget),
                config_hash=_config_hash(fail_cfg),
                config=fail_cfg,
                gating_mode=fail_cfg.get("gating_mode", ""),
                target_source=fail_cfg.get("target_source", ""),
                router_loss=fail_cfg.get("router_loss", ""),
                objective_returned=FAILURE_COST,
                proxy_gain=-FAILURE_COST,
            ))

            if wandb_run is not None:
                try:
                    wandb.log(
                        {
                            "hpo_trial": float(trial_idx),
                            "hpo/trial_failed": 1.0,
                            "hpo/smac_cost": float(FAILURE_COST),
                            "hpo/proxy": -float(FAILURE_COST),
                        },
                    )
                except Exception as log_e:
                    logger.debug("W&B failure log: %s", log_e)

            return FAILURE_COST

    return target_function


# ---------------------------------------------------------------------------
# SMAC launch
# ---------------------------------------------------------------------------

def run_smac_optimization(
    *,
    router_kind: str = "fine",
    residuals: Optional[torch.Tensor] = None,
    gate_labels: Optional[List[int]] = None,
    records: Optional[List[Dict]] = None,
    seq_to_idx: Optional[Dict[tuple, int]] = None,
    num_classes: Optional[int] = None,
    best_deltas: Optional[List[float]] = None,
    sequence_catalog: Optional[List] = None,
    compositional_ctx: Optional[Dict[str, Any]] = None,
    d_model: int = 0,
    device: torch.device = torch.device("cpu"),
    benchmark: str = "",
    output_dir: str = "",
    wandb_project: str = "",
    wandb_run_name: Optional[str] = None,
    n_trials: int = 100,
    min_budget: float = DEFAULT_TRAIN_MIN_BUDGET,
    max_budget: float = DEFAULT_TRAIN_MAX_BUDGET,
    enable_expensive_eval: bool = False,
    expensive_eval_fn: Optional[Callable] = None,
    seed: int = 42,
    walltime_limit: float = float("inf"),
    resume: bool = False,
) -> ThresholdPriorArchive:
    """Set up and run SMAC with Hyperband training fidelity (epoch scaling).

    If ``resume`` is True, SMAC reuses existing state under
    ``output_dir/smac3_output`` (same scenario metadata) and continues the
    run with ``MultiFidelityFacade(overwrite=False)``. The best-model
    tracker reloads ``best_routing_system_<bench>/meta.json`` when present
    so checkpoints are not clobbered by a weaker first trial.

    Returns the populated archive after optimization completes.
    """
    from smac import MultiFidelityFacade, Scenario

    if router_kind not in ("fine", "compositional"):
        raise ValueError(f"unknown router_kind={router_kind!r}")

    # --- Initialize components ---
    os.makedirs(output_dir, exist_ok=True)
    archive_path = os.path.join(output_dir, "threshold_prior_archive.jsonl")
    archive = ThresholdPriorArchive(archive_path)
    tracker = BestModelTracker(output_dir, benchmark, resume=resume)
    initial_trial_idx = len(archive) if resume else 0

    if router_kind == "fine":
        cs = build_configspace()
    else:
        cs = build_configspace_compositional()

    # --- W&B init ---
    if router_kind == "fine":
        wandb_config = {
            "router_kind": router_kind,
            "benchmark": benchmark,
            "n_trials": n_trials,
            "min_training_budget": min_budget,
            "max_training_budget": max_budget,
            "seed": seed,
            "d_model": d_model,
            "num_classes": num_classes,
            "n_samples": len(gate_labels) if gate_labels is not None else 0,
            "resume": resume,
        }
        wandb_tags = [benchmark, "unified-hpo", "smac-hyperband-train"]
    else:
        ctx = compositional_ctx or {}
        local_paths = ctx.get("local_moebius_paths") or {}
        wandb_config = {
            "router_kind": router_kind,
            "benchmark": benchmark,
            "scope": ctx.get("scope"),
            "benchmarks": ctx.get("benchmarks"),
            "objective_metric": ctx.get("objective_metric"),
            "local_moebius_benchmarks": sorted(local_paths.keys()),
            "n_trials": n_trials,
            "min_training_budget": min_budget,
            "max_training_budget": max_budget,
            "seed": seed,
            "d_model": d_model,
            "resume": resume,
        }
        wandb_tags = [
            benchmark, "unified-hpo", "smac-hyperband-train",
            "compositional", str(ctx.get("scope", "single")),
        ]

    run = wandb.init(
        project=wandb_project,
        name=wandb_run_name or f"unified-hpo-{router_kind}-{benchmark}",
        config=wandb_config,
        tags=wandb_tags,
    )
    _define_hpo_wandb_charts(router_kind)
    # One metric group ``hpo/*``; per-trial logs use step=trial index (see _log_trial_to_wandb).
    try:
        n_arch = len(archive)
        if not resume:
            run.log(
                {
                    "hpo_trial": -1.0,
                    "hpo/wb_layout": 3.0,
                    "hpo/router_kind": 1.0 if router_kind == "compositional" else 0.0,
                },
            )
        else:
            run.log(
                {
                    "hpo_trial": -1.0,
                    "hpo/resumed_smac": 1.0,
                    "hpo/continues_from_trial": float(n_arch),
                    "hpo/router_kind": 1.0 if router_kind == "compositional" else 0.0,
                },
            )
    except Exception as e:
        logger.warning("Initial wandb log failed: %s", e)

    # --- Build target function ---
    target_fn = build_target_function(
        residuals=residuals,
        gate_labels=gate_labels,
        records=records,
        seq_to_idx=seq_to_idx,
        num_classes=num_classes,
        best_deltas=best_deltas,
        sequence_catalog=sequence_catalog,
        d_model=d_model,
        device=device,
        benchmark=benchmark,
        archive=archive,
        tracker=tracker,
        min_budget=min_budget,
        max_budget=max_budget,
        enable_expensive_eval=enable_expensive_eval,
        expensive_eval_fn=expensive_eval_fn,
        wandb_run=run,
        seed=seed,
        router_kind=router_kind,
        compositional_ctx=compositional_ctx,
        initial_trial_idx=initial_trial_idx,
    )

    # --- SMAC Scenario ---
    scenario = Scenario(
        configspace=cs,
        name=f"unified_hpo_{router_kind}_{benchmark}",
        output_directory=os.path.join(output_dir, "smac3_output"),
        n_trials=n_trials,
        min_budget=min_budget,
        max_budget=max_budget,
        seed=seed,
        walltime_limit=walltime_limit,
        deterministic=True,
    )

    # SMAC may call ``input()`` if the on-disk scenario differs (e.g. after a code
    # change) even when resuming, which fails under nohup. Set
    # ``UNIFIED_HPO_SMAC_ON_MISMATCH=1`` (overwrite) or ``=2`` (rename to *-old).
    _mismatch = os.environ.get("UNIFIED_HPO_SMAC_ON_MISMATCH", "").strip()
    _prev_input = getattr(builtins, "input", None)
    if _mismatch in ("1", "2") and _prev_input is not None:
        def _input_patch(prompt: str = "") -> str:
            logger.warning(
                "SMAC scenario mismatch: auto-responding with "
                "UNIFIED_HPO_SMAC_ON_MISMATCH=%r (no TTY).",
                _mismatch,
            )
            return _mismatch

        builtins.input = _input_patch  # type: ignore[assignment]
    try:
        smac = MultiFidelityFacade(
            scenario=scenario,
            target_function=target_fn,
            overwrite=not resume,
            logging_level=logging.INFO,
        )
    finally:
        if _mismatch in ("1", "2") and _prev_input is not None:
            builtins.input = _prev_input  # type: ignore[assignment]

    logger.info(
        "Starting SMAC optimization: n_trials=%d  train_budget=[%.1f, %.1f]  "
        "benchmark=%s  seed=%d  resume=%s  smac_overwrite=%s  trial_idx_offset=%d",
        n_trials, min_budget, max_budget, benchmark, seed,
        resume, not resume, initial_trial_idx,
    )

    incumbent = smac.optimize()

    # --- Post-optimization summary ---
    logger.info("Optimization complete.  Incumbent: %s", incumbent)
    logger.info("Best proxy gain: %.5f", tracker.best_proxy_gain)

    if tracker.best_config:
        logger.info("Best config: %s", json.dumps(tracker.best_config, indent=2, default=str))

    if router_kind == "fine":
        try:
            log_sweep_summary(archive, cs)
        except Exception as e:
            logger.warning("Failed to log sweep summary: %s", e)

    run.finish()
    return archive
