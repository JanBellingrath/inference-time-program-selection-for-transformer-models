"""Optuna orchestration for unified HPO with router-family specs."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import optuna
import torch
import wandb

from experiments.unified_hpo.router_specs import ROUTER_SPECS
from experiments.unified_hpo.smac_runner import (
    BestModelTracker,
    _define_hpo_wandb_charts,
    _log_trial_to_wandb,
)
from experiments.unified_hpo.threshold_prior import (
    ArchiveRecord,
    ThresholdPriorArchive,
    _config_hash,
)
from experiments.unified_hpo.training_budget import (
    DEFAULT_TRAIN_MAX_BUDGET,
    DEFAULT_TRAIN_MIN_BUDGET,
    router_gate_epochs_from_training_budget,
)

logger = logging.getLogger(__name__)


def _build_sampler(seed: int) -> optuna.samplers.BaseSampler:
    return optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=10,
        multivariate=True,
        group=True,
    )


def _build_pruner(
    kind: str,
    *,
    min_resource: int,
    max_resource: int,
) -> optuna.pruners.BasePruner:
    if kind == "none":
        return optuna.pruners.NopPruner()
    if kind == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=max(1, int(min_resource)),
        )
    if kind == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=max(1, int(min_resource)),
            max_resource=max(1, int(max_resource)),
            reduction_factor=3,
        )
    raise ValueError(f"unknown optuna pruner: {kind!r}")


def _should_run_posthoc_external_eval(
    *,
    router_kind: str,
    mode: str,
    ctx: Dict[str, Any],
) -> bool:
    if router_kind != "compositional":
        return False
    mode = str(mode).strip().lower()
    if mode == "off":
        return False
    if mode == "best_only":
        return True
    if mode != "auto":
        raise ValueError(
            f"unknown compositional external eval mode={mode!r} "
            f"(expected 'auto', 'best_only', or 'off')"
        )
    # Auto: run post-hoc LLM eval when we do NOT have dense supervision.
    # Dense val metrics are available in-training only when dense paths exist.
    return not bool(ctx.get("dense_paths") or {})


def _run_posthoc_external_eval_best(
    *,
    run,
    tracker: BestModelTracker,
    ctx: Dict[str, Any],
    model_name: str,
    split_json: Optional[Path],
    max_samples_per_bench: int,
    output_dir: str,
    hpo_trial: int,
) -> Optional[Dict[str, Any]]:
    best_ckpt = Path(tracker.output_dir) / "best_checkpoint.pt"
    if not best_ckpt.is_file():
        logger.warning(
            "Skip post-hoc compositional external eval: missing %s",
            best_ckpt,
        )
        return None

    from experiments.eval_compositional_downstream import evaluate_checkpoint

    catalogue_dir = Path(str(ctx["artifacts"].output_dir))
    benches = list(ctx.get("benchmarks") or [])
    effective_split_json = split_json
    if effective_split_json is None:
        try:
            payload = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            split_qids = payload.get("split_qids") or {}
            if isinstance(split_qids, dict) and split_qids:
                split_out = Path(output_dir) / "best_external_eval_split.json"
                split_payload: Dict[str, Any] = {
                    "seed": None,
                    "val_fraction": None,
                    "train_test_holdout_count": 0,
                    "benchmarks": {},
                }
                for bench, rows in split_qids.items():
                    if not isinstance(rows, dict):
                        continue
                    train_q = sorted(int(x) for x in (rows.get("train") or []))
                    val_q = sorted(int(x) for x in (rows.get("val") or []))
                    test_q = sorted(int(x) for x in (rows.get("test") or []))
                    split_payload["benchmarks"][bench] = {
                        "n_total": len(set(train_q) | set(val_q) | set(test_q)),
                        "train_question_ids": train_q,
                        "val_question_ids": val_q,
                        "test_question_ids": test_q,
                    }
                split_out.parent.mkdir(parents=True, exist_ok=True)
                with open(split_out, "w") as f:
                    json.dump(split_payload, f, indent=2)
                effective_split_json = split_out
        except Exception as e:
            logger.warning(
                "Could not derive split_json from best checkpoint; evaluating all observed rows: %s",
                e,
            )

    eval_json = Path(output_dir) / "best_external_eval.json"
    summary = evaluate_checkpoint(
        catalogue_dir=catalogue_dir,
        checkpoint_path=best_ckpt,
        split_json=effective_split_json,
        benchmarks=benches if benches else None,
        model_name=model_name,
        ft_adapter_path=None,
        data_split="validation",
        max_samples_per_bench=(
            int(max_samples_per_bench) if int(max_samples_per_bench) > 0 else None
        ),
        output_json=eval_json,
    )

    u_pp = float(summary.get("unconditional_gain_pp", float("nan")))
    n_eval = int(summary.get("n", 0))
    payload: Dict[str, Any] = {
        "hpo/mean_uplift_pp": u_pp,
        "hpo/external_eval_n": float(n_eval),
        "hpo/external_router_acc": float(summary.get("router_acc", float("nan"))),
        "hpo/external_anchor_acc": float(summary.get("anchor_acc", float("nan"))),
        "llm_eval/validation/unconditional_gain_pp": u_pp,
        "llm_eval/validation/router_acc": float(summary.get("router_acc", float("nan"))),
        "llm_eval/validation/anchor_acc": float(summary.get("anchor_acc", float("nan"))),
        "llm_eval/validation/n_questions": float(n_eval),
    }
    for bench, row in (summary.get("per_bench") or {}).items():
        if not isinstance(row, dict):
            continue
        payload[f"llm_eval/validation/{bench}/uplift_pp"] = float(
            row.get("uplift_pp", float("nan"))
        )
        payload[f"llm_eval/validation/{bench}/n"] = float(row.get("n", 0))

    # Keep charts aligned on hpo_trial axis; append as final point.
    payload["hpo_trial"] = float(max(0, int(hpo_trial)))
    run.log(payload)
    logger.info(
        "Post-hoc external eval logged: unconditional_gain_pp=%.4f n=%d",
        u_pp,
        n_eval,
    )
    return summary


def run_optuna_optimization(
    *,
    router_kind: str = "fine",
    ctx: Dict[str, Any],
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
    seed: int = 42,
    walltime_limit: float = float("inf"),
    resume: bool = False,
    optuna_pruner: str = "none",
    optuna_intermediate_metric: str = "objective",
    compositional_external_eval_mode: str = "auto",
    compositional_external_eval_split_json: Optional[Path] = None,
    compositional_external_eval_max_samples_per_bench: int = 0,
    compositional_external_eval_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
) -> ThresholdPriorArchive:
    """Run full-budget Optuna HPO using router-family abstraction."""
    if router_kind not in ROUTER_SPECS:
        raise ValueError(f"unknown router_kind={router_kind!r}")

    spec = ROUTER_SPECS[router_kind]
    os.makedirs(output_dir, exist_ok=True)
    archive_path = os.path.join(output_dir, "threshold_prior_archive.jsonl")
    archive = ThresholdPriorArchive(archive_path)
    tracker = BestModelTracker(output_dir, benchmark, resume=resume)
    initial_trial_idx = len(archive) if resume else 0

    router_e, gate_e = router_gate_epochs_from_training_budget(max_budget, min_budget, max_budget)
    trial_step_stride = max(1000, int(router_e) + 100)
    run_config: Dict[str, Any] = {
        "router_kind": router_kind,
        "benchmark": benchmark,
        "n_trials": n_trials,
        "min_training_budget": min_budget,
        "max_training_budget": max_budget,
        "seed": seed,
        "d_model": d_model,
        "resume": resume,
        "hpo_backend": "optuna",
        "optuna_pruner": optuna_pruner,
        "optuna_intermediate_metric": optuna_intermediate_metric,
        "router_epochs": router_e,
        "gate_epochs": gate_e,
    }
    if router_kind == "fine":
        run_config["num_classes"] = ctx.get("num_classes")
        run_config["n_samples"] = len(ctx.get("gate_labels", []))
    else:
        run_config["scope"] = ctx.get("scope")
        run_config["benchmarks"] = ctx.get("benchmarks")
        run_config["objective_metric"] = ctx.get("objective_metric")
        run_config["compositional_external_eval_mode"] = compositional_external_eval_mode

    run_tags = [benchmark, "unified-hpo", "optuna", router_kind]
    run = wandb.init(
        project=wandb_project,
        name=wandb_run_name or f"unified-hpo-optuna-{router_kind}-{benchmark}",
        config=run_config,
        tags=run_tags,
    )
    _define_hpo_wandb_charts(router_kind)
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

    study_name = f"unified_hpo_{router_kind}_{benchmark}"
    storage_url = f"sqlite:///{os.path.join(output_dir, 'optuna.db')}"
    sampler = _build_sampler(seed)
    pruner = _build_pruner(
        optuna_pruner,
        min_resource=max(1, int(min_budget)),
        max_resource=max(1, int(max_budget)),
    )
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        load_if_exists=resume,
    )

    hpo_trial_history: list[Dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        trial_idx = initial_trial_idx + int(trial.number)
        run_id = str(uuid.uuid4())[:12]
        t0 = time.time()
        cfg: Dict[str, Any] = {}

        try:
            cfg = spec.suggest_config(trial, ctx)
            run_expensive = bool(enable_expensive_eval)

            result = spec.train_and_evaluate(
                config=cfg,
                ctx={
                    **ctx,
                    "gate_epochs": gate_e,
                    "optuna_intermediate_metric": optuna_intermediate_metric,
                    "hpo_wandb_run": run,
                    "hpo_wandb_prefix": f"trial_{trial_idx:05d}",
                    "hpo_wandb_step_offset": int(trial_idx) * int(trial_step_stride),
                },
                device=device,
                trial=trial,
                seed=seed,
                max_epochs=router_e,
                enable_expensive_eval=run_expensive,
            )
            proxy_gain = float(result.proxy_gain)
            wall_time = time.time() - t0

            trial.set_user_attr("wall_s", float(wall_time))
            for k, v in result.metrics.items():
                if isinstance(v, (str, bool, int, float)) or v is None:
                    trial.set_user_attr(k, v)

            full_train = True
            is_new_best = False
            if full_train and result.train_result is not None and result.calibration is not None:
                is_new_best = tracker.maybe_save(
                    proxy_gain=proxy_gain,
                    train_result=result.train_result,
                    config=cfg,
                    calibration=result.calibration,
                    expensive_gain=result.expensive_gain,
                    wandb_run=run,
                )

            if run is not None and result.train_result is not None and result.eval_result is not None:
                _log_trial_to_wandb(
                    cfg,
                    result.train_result,
                    result.eval_result,
                    trial_idx,
                    wall_time,
                    is_new_best,
                    max_budget,
                    router_kind=router_kind,
                    hpo_trial_history=hpo_trial_history if router_kind == "compositional" else None,
                )

            train_result = result.train_result
            eval_result = result.eval_result
            if train_result is not None and eval_result is not None:
                archive.append(
                    ArchiveRecord(
                        run_id=run_id,
                        timestamp=time.time(),
                        benchmark=benchmark,
                        seed=seed,
                        budget=float(max_budget),
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
                        expensive_gain=result.expensive_gain,
                    ),
                )

            logger.info(
                "Optuna trial %d done: proxy_gain=%.5f (%.1fs)%s",
                trial_idx,
                proxy_gain,
                wall_time,
                "  *** NEW BEST ***" if is_new_best else "",
            )
            return proxy_gain
        except optuna.TrialPruned:
            trial.set_user_attr("trial_pruned", True)
            if run is not None:
                try:
                    wandb.log(
                        {
                            "hpo_trial": float(trial_idx),
                            "hpo/trial_pruned": 1.0,
                        },
                    )
                except Exception:
                    pass
            logger.info("Optuna trial %d pruned.", trial_idx)
            raise
        except Exception as e:
            logger.error("Optuna trial %d FAILED: %s", trial_idx, e, exc_info=True)
            trial.set_user_attr("trial_failed", str(e))
            archive.append(
                ArchiveRecord(
                    run_id=run_id,
                    timestamp=time.time(),
                    benchmark=benchmark,
                    seed=seed,
                    budget=float(max_budget),
                    config_hash=_config_hash(cfg),
                    config=cfg,
                    objective_returned=1.0,
                    proxy_gain=-1.0,
                ),
            )
            if run is not None:
                try:
                    wandb.log(
                        {
                            "hpo_trial": float(trial_idx),
                            "hpo/trial_failed": 1.0,
                            "hpo/proxy": -1.0,
                        },
                    )
                except Exception:
                    pass
            return -1.0

    timeout = None if walltime_limit == float("inf") else float(walltime_limit)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    logger.info("Optuna complete. Best proxy gain: %.5f", tracker.best_proxy_gain)
    if tracker.best_config:
        logger.info("Best config: %s", json.dumps(tracker.best_config, indent=2, default=str))

    if router_kind == "fine":
        logger.info(
            "Optuna backend skips SMAC ConfigSpace summary tables; "
            "use archive/study artifacts for sweep analysis."
        )
    else:
        if _should_run_posthoc_external_eval(
            router_kind=router_kind,
            mode=compositional_external_eval_mode,
            ctx=ctx,
        ):
            try:
                _run_posthoc_external_eval_best(
                    run=run,
                    tracker=tracker,
                    ctx=ctx,
                    model_name=compositional_external_eval_model_name,
                    split_json=compositional_external_eval_split_json,
                    max_samples_per_bench=compositional_external_eval_max_samples_per_bench,
                    output_dir=output_dir,
                    hpo_trial=len(archive),
                )
            except Exception as e:
                logger.error("Post-hoc external eval failed: %s", e, exc_info=True)

    try:
        run.log(
            {
                "optuna/best_value": float(study.best_value),
                "optuna/best_trial": int(study.best_trial.number),
            },
        )
    except Exception:
        pass
    run.finish()
    return archive
