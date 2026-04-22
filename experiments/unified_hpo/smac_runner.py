"""SMAC + Hyperband orchestration, W&B logging, and best-model persistence.

- **Training fidelity** (Hyperband ``budget``): scales router/gate epoch counts;
  cheap low-budget runs prune the search space.
- **Evaluation**: always full routing-val for calibration / proxy gain (no eval subsampling).
- Checkpoints and optional expensive LLM eval run only at **maximum** training budget.
"""

from __future__ import annotations

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

def _log_trial_to_wandb(
    config: Dict,
    train_result: TrainResult,
    eval_result: EvalResult,
    trial_idx: int,
    wall_time: float,
    is_new_best: bool,
    training_budget: float,
) -> None:
    """Log comprehensive metrics for a single trial to the active W&B run."""
    gating_mode = config.get("gating_mode", "?")
    target_source = config.get("target_source", "?")
    router_loss = config.get("router_loss", "?")

    log_dict: Dict[str, Any] = {
        # Branch identifiers
        "trial_idx": trial_idx,
        "training_budget": training_budget,
        "router_epochs_used": train_result.router_epochs_used,
        "gate_epochs_used": train_result.gate_epochs_used,
        "gating_mode": gating_mode,
        "target_source": target_source,
        "router_loss": router_loss,
        "router_train_subset": config.get("router_train_subset", "all"),

        # Training summaries
        "router_val_loss": train_result.router_val_loss,
        "gate_val_loss": train_result.gate_val_loss,
        "delta_gate_val_loss": train_result.delta_gate_val_loss,
        "num_train_samples": train_result.num_train_samples,
        "num_val_samples": train_result.num_val_samples,
        "training_time_s": train_result.training_time_s,

        # Diagnostics
        "predicted_noop_rate": train_result.predicted_noop_rate,
        "router_entropy_mean": train_result.router_entropy_mean,
        "router_entropy_std": train_result.router_entropy_std,

        # Score distribution
        "score_mean": train_result.score_mean,
        "score_std": train_result.score_std,
        "score_min": train_result.score_min,
        "score_max": train_result.score_max,
        **{f"score_{k}": v for k, v in train_result.score_quantiles.items()},

        # Calibration
        "best_rho": eval_result.calibration.best_rho,
        "best_threshold": eval_result.calibration.best_threshold,
        "realized_open_fraction": eval_result.calibration.realized_open_fraction,
        "n_candidates_tested": len(eval_result.calibration.candidates_tested),

        # Objective
        "proxy_gain": eval_result.proxy_gain,
        "oracle_proxy_gain": eval_result.oracle_proxy_gain,
        "routing_val_size_used": eval_result.routing_val_size_used,

        # Timing
        "wall_time_s": wall_time,
        "is_new_best": int(is_new_best),
    }

    # Log all active config parameters
    for k, v in config.items():
        log_dict[f"config/{k}"] = v

    # Expensive eval metrics (if available)
    if eval_result.expensive_metrics is not None:
        for k, v in eval_result.expensive_metrics.items():
            log_dict[f"expensive/{k}"] = v

    wandb.log(log_dict)


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

            # Per-trial heartbeat: visible in W&B while train_one_router is running.
            if wandb_run is not None:
                try:
                    # Use global wandb.log so the active run (SMAC’s thread) always matches.
                    wandb.log(
                        {
                            "trial/idx": float(trial_idx),
                            "trial/budget": float(budget),
                            "trial/router_epochs": float(router_e),
                            "trial/phase": 0.0,  # 0=train started; 1=done in full metrics
                        },
                    )
                except Exception as e:
                    logger.warning("trial-start wandb log failed: %s", e)

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
                    scope=ctx.get("scope", "single"),
                    router_epochs=router_e,
                    batch_size=int(ctx.get("batch_size", 64)),
                    val_fraction=float(ctx.get("val_fraction", 0.15)),
                    seed=seed,
                    device=device,
                    objective_metric=ctx.get("objective_metric", "mean_uplift"),
                    use_full_sequence=bool(ctx.get("use_full_sequence", False)),
                    wandb_run=wandb_run,
                    wandb_prefix="compositional/",
                    use_dense_supervision=None,
                    downstream_eval_every=0,
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

            # --- W&B logging ---
            _log_trial_to_wandb(
                cfg, train_result, eval_result,
                trial_idx, wall_time, is_new_best, budget,
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

            # Log failure to W&B
            wandb.log({
                "trial_idx": trial_idx,
                "proxy_gain": -FAILURE_COST,
                "failed": True,
                "error": str(e),
            })

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
        wandb_config = {
            "router_kind": router_kind,
            "benchmark": benchmark,
            "scope": ctx.get("scope"),
            "benchmarks": ctx.get("benchmarks"),
            "objective_metric": ctx.get("objective_metric"),
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
    # One immediate scalar so the W&B run does not look "stuck" before the first
    # heavy trial returns (compositional: train + long post-hoc val metrics).
    try:
        run.log(
            {
                "hpo/launched": 1.0,
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

    smac = MultiFidelityFacade(
        scenario=scenario,
        target_function=target_fn,
        overwrite=not resume,
        logging_level=logging.INFO,
    )

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

    # Log sweep summary to W&B
    try:
        log_sweep_summary(archive, cs)
    except Exception as e:
        logger.warning("Failed to log sweep summary: %s", e)

    run.finish()
    return archive
