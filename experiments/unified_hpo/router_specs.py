"""Router-family abstractions for Optuna HPO orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

import optuna
import torch

from experiments.unified_hpo.budgeted_evaluator import EvalResult, evaluate_configuration
from experiments.unified_hpo.calibration import CalibrationResult
from experiments.unified_hpo.compositional_objective import train_and_score_compositional
from experiments.unified_hpo.search_space_optuna import (
    suggest_compositional_config,
    suggest_fine_config,
)
from experiments.unified_hpo.trainer import TrainResult, train_and_summarize


@dataclass
class HpoResult:
    """Router-family-agnostic objective output for one trial."""

    proxy_gain: float
    config: Dict[str, Any]
    metrics: Dict[str, Any] = field(default_factory=dict)
    train_result: Optional[TrainResult] = None
    eval_result: Optional[EvalResult] = None
    calibration: Optional[CalibrationResult] = None
    expensive_gain: Optional[float] = None


class RouterHpoSpec(Protocol):
    """Common contract each router family exposes to the runner."""

    name: str

    def suggest_config(self, trial: optuna.Trial, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Sample a valid config for this router family."""

    def train_and_evaluate(
        self,
        *,
        config: Dict[str, Any],
        ctx: Dict[str, Any],
        device: torch.device,
        trial: optuna.Trial,
        seed: int,
        max_epochs: int,
        enable_expensive_eval: bool,
    ) -> HpoResult:
        """Run one full-budget trial and return proxy + diagnostics."""


class FineRouterHpoSpec:
    """Spec for the existing fine router pipeline."""

    name = "fine"

    def suggest_config(self, trial: optuna.Trial, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return suggest_fine_config(trial, prefix=self.name)

    def train_and_evaluate(
        self,
        *,
        config: Dict[str, Any],
        ctx: Dict[str, Any],
        device: torch.device,
        trial: optuna.Trial,
        seed: int,
        max_epochs: int,
        enable_expensive_eval: bool,
    ) -> HpoResult:
        gate_epochs = int(ctx.get("gate_epochs", max_epochs))
        intermediate_metric = str(ctx.get("optuna_intermediate_metric", "val_loss"))

        def _report_router_val(epoch: int, val_loss: float) -> None:
            # Objective is maximize proxy gain; lower val loss => better.
            if intermediate_metric == "objective":
                # Fine objective is only available after calibration; fallback to val_loss.
                trial.report(-float(val_loss), step=int(epoch))
            else:
                trial.report(-float(val_loss), step=int(epoch))
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned at epoch={epoch} (router_val_loss={val_loss:.5f})")

        train_result = train_and_summarize(
            config=config,
            residuals=ctx["residuals"],
            gate_labels=ctx["gate_labels"],
            records=ctx["records"],
            seq_to_idx=ctx["seq_to_idx"],
            num_classes=ctx["num_classes"],
            best_deltas=ctx["best_deltas"],
            d_model=ctx["d_model"],
            device=device,
            seed=seed,
            router_epochs=int(max_epochs),
            gate_epochs=gate_epochs,
            router_epoch_val_callback=_report_router_val,
        )
        eval_result = evaluate_configuration(
            train_result=train_result,
            config=config,
            records=ctx["records"],
            sequence_catalog=ctx["sequence_catalog"],
            seq_to_idx=ctx["seq_to_idx"],
            enable_expensive_eval=bool(enable_expensive_eval),
            expensive_eval_fn=ctx.get("expensive_eval_fn"),
        )
        expensive_gain: Optional[float] = None
        if eval_result.expensive_metrics is not None:
            expensive_gain = eval_result.expensive_metrics.get("unconditional_gain")
        metrics: Dict[str, Any] = {
            "router_val_loss": float(train_result.router_val_loss),
            "gate_val_loss": float(train_result.gate_val_loss),
            "best_rho": float(eval_result.calibration.best_rho),
            "best_threshold": float(eval_result.calibration.best_threshold),
            "realized_open_fraction": float(eval_result.calibration.realized_open_fraction),
            "oracle_proxy_gain": float(eval_result.oracle_proxy_gain),
        }
        return HpoResult(
            proxy_gain=float(eval_result.proxy_gain),
            config=config,
            metrics=metrics,
            train_result=train_result,
            eval_result=eval_result,
            calibration=eval_result.calibration,
            expensive_gain=expensive_gain,
        )


class CompositionalRouterHpoSpec:
    """Spec for compositional router pipeline."""

    name = "compositional"

    def suggest_config(self, trial: optuna.Trial, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return suggest_compositional_config(trial, prefix=self.name)

    def train_and_evaluate(
        self,
        *,
        config: Dict[str, Any],
        ctx: Dict[str, Any],
        device: torch.device,
        trial: optuna.Trial,
        seed: int,
        max_epochs: int,
        enable_expensive_eval: bool,
    ) -> HpoResult:
        objective_metric = ctx.get("objective_metric", "mean_uplift")
        intermediate_metric = str(ctx.get("optuna_intermediate_metric", "objective"))

        def _report_compositional_epoch(epoch: int, metrics: Dict[str, float]) -> None:
            if intermediate_metric == "val_loss":
                score = -float(metrics.get("val_loss", 0.0))
            else:
                # objective metric score already aligned with maximize direction.
                score = float(metrics.get("checkpoint_score", 0.0))
            trial.report(score, step=int(epoch))
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"pruned at epoch={epoch} (score={score:.5f}, metric={intermediate_metric})"
                )

        train_result = train_and_score_compositional(
            config=config,
            artifacts=ctx["artifacts"],
            benchmarks=ctx["benchmarks"],
            dense_paths=ctx.get("dense_paths") or {},
            dense_keep_mask_paths=ctx.get("dense_keep_mask_paths") or None,
            scope=ctx.get("scope", "single"),
            router_epochs=int(max_epochs),
            batch_size=int(ctx.get("batch_size", 64)),
            val_fraction=float(ctx.get("val_fraction", 0.15)),
            seed=seed,
            device=device,
            objective_metric=objective_metric,
            use_full_sequence=bool(ctx.get("use_full_sequence", False)),
            wandb_run=ctx.get("hpo_wandb_run"),
            wandb_prefix=str(ctx.get("hpo_wandb_prefix", "")),
            wandb_step_offset=int(ctx.get("hpo_wandb_step_offset", 0)),
            use_dense_supervision=True if objective_metric == "mean_uplift" else None,
            downstream_eval_every=int(ctx.get("downstream_eval_every", 0)),
            local_moebius_paths=ctx.get("local_moebius_paths") or None,
            epoch_report_callback=_report_compositional_epoch,
            train_test_holdout_count=int(ctx.get("train_test_holdout_count", 0)),
            split_json_path=ctx.get("split_json_path"),
        )
        proxy = float(getattr(train_result, "compositional_proxy", 0.0))
        calibration = CalibrationResult(
            best_rho=0.0,
            best_threshold=0.0,
            best_gain=proxy,
            realized_open_fraction=0.0,
            candidates_tested=[(0.0, proxy)],
        )
        eval_result = EvalResult(
            calibration=calibration,
            proxy_gain=proxy,
            oracle_proxy_gain=proxy,
            routing_val_size_used=0,
            ran_expensive_eval=False,
        )
        metrics = dict(getattr(train_result, "compositional_metrics", {}) or {})
        metrics["router_val_loss"] = float(train_result.router_val_loss)
        # Compositional objective is still full-budget for now.
        trial.report(proxy, step=int(max_epochs))
        return HpoResult(
            proxy_gain=proxy,
            config=config,
            metrics=metrics,
            train_result=train_result,
            eval_result=eval_result,
            calibration=calibration,
            expensive_gain=None,
        )


ROUTER_SPECS: Dict[str, RouterHpoSpec] = {
    "fine": FineRouterHpoSpec(),
    "compositional": CompositionalRouterHpoSpec(),
}
