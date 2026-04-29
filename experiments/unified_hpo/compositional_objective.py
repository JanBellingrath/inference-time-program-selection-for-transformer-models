"""Compositional-router HPO objective.

Adapter around :func:`training.train_compositional_router.train_one_router`
that returns a :class:`~experiments.unified_hpo.trainer.TrainResult`
shaped object so the existing SMAC/Hyperband/W&B/archive pipeline can stay
unchanged.

Differences from the fine-router objective:

* there is no gate / delta-gate / open-rate calibration; the proxy is one
  of the validation metrics produced inside ``train_one_router``;
* the trained model is the full ``CompositionalRouter``; we stash it on
  ``TrainResult.router`` so :class:`BestModelTracker` can persist its
  ``state_dict`` with the same code path used for ``FineRouter``.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import torch

from experiments.unified_hpo.search_space_compositional import (
    get_edit_hidden_dims,
    get_local_moebius_cfg,
    get_pair_hidden_dims,
    get_pair_topk_primitives,
    get_unary_hidden_dims,
)
from experiments.unified_hpo.trainer import TrainResult

logger = logging.getLogger(__name__)


SUPPORTED_OBJECTIVES = ("mean_uplift", "obs_top1_acc")


# ---------------------------------------------------------------------------
# Lightweight stand-in for ``train_result.router`` used by BestModelTracker
# ---------------------------------------------------------------------------

class _CheckpointRouterProxy:
    """A minimal object exposing ``state_dict()`` backed by an on-disk ``.pt``.

    ``BestModelTracker.maybe_save`` only ever calls
    ``train_result.router.state_dict()``; reconstructing the full
    ``CompositionalRouter`` just to satisfy that call is wasteful (it
    requires a fresh ``CompositionalDataset`` load to recover ``d_model``).
    Instead we proxy the state dict directly off the saved trial payload.
    """

    def __init__(self, payload_path: Path, payload: Dict[str, Any]):
        self._path = payload_path
        # Cache the state dict in memory so saving doesn't depend on the
        # temp file still existing at flush time.
        self._state_dict = {
            k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
            for k, v in payload["model_state_dict"].items()
        }
        self._payload = payload

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self._state_dict

    @property
    def trial_payload(self) -> Dict[str, Any]:
        return self._payload

    @property
    def trial_payload_path(self) -> Path:
        return self._path


def _extract_proxy(
    metrics: Dict[str, Any],
    objective_metric: str,
) -> float:
    """Pull the chosen scalar proxy out of ``train_one_router`` metrics."""
    if objective_metric == "mean_uplift":
        downstream = metrics.get("val_downstream") if metrics else None
        if downstream is None:
            raise RuntimeError(
                "objective_metric='mean_uplift' requires --dense_deltas; "
                "train_one_router did not produce val_downstream metrics."
            )
        return float(downstream.get("mean_uplift", 0.0))
    if objective_metric == "obs_top1_acc":
        ranking = (metrics or {}).get("val_ranking") or {}
        return float(ranking.get("obs_top1_acc", 0.0))
    raise ValueError(
        f"unknown objective_metric={objective_metric!r}; "
        f"supported: {SUPPORTED_OBJECTIVES}"
    )


def train_and_score_compositional(
    config: Dict,
    *,
    artifacts,
    benchmarks: Sequence[str],
    dense_paths: Optional[Dict[str, Path]],
    scope: str,
    router_epochs: int,
    batch_size: int,
    val_fraction: float,
    seed: int,
    device: torch.device,
    objective_metric: str,
    output_dir: Optional[Path] = None,
    use_full_sequence: bool = False,
    wandb_run: Any = None,
    wandb_prefix: str = "",
    wandb_step_offset: int = 0,
    use_dense_supervision: Optional[bool] = None,
    downstream_eval_every: int = 0,
    dense_keep_mask_paths: Optional[Dict[str, Path]] = None,
    local_moebius_paths: Optional[Dict[str, Path]] = None,
    epoch_report_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
    train_test_holdout_count: int = 0,
    split_json_path: Optional[Path] = None,
) -> TrainResult:
    """Train a compositional router for one HPO trial; return a ``TrainResult``.

    The trial writes its checkpoint to ``output_dir`` (or a temporary path
    when not provided), then loads the metrics block to compute the proxy.
    The trained ``CompositionalRouter``'s state dict is exposed via a
    lightweight proxy on ``TrainResult.router`` so ``BestModelTracker`` can
    persist it via the same path used for ``FineRouter``.
    """
    from training.train_compositional_router import train_one_router

    if objective_metric not in SUPPORTED_OBJECTIVES:
        raise ValueError(
            f"unknown objective_metric={objective_metric!r}; "
            f"supported: {SUPPORTED_OBJECTIVES}"
        )

    benchmarks = list(benchmarks)
    dense_paths = dict(dense_paths or {})
    dense_keep_mask_paths = dict(dense_keep_mask_paths or {})

    if use_dense_supervision is None:
        use_dense_supervision = bool(dense_paths) and objective_metric == "mean_uplift"

    edit_hidden_dims = get_edit_hidden_dims(config)
    unary_hidden_dims = get_unary_hidden_dims(config)
    pair_hidden_dims = get_pair_hidden_dims(config)
    pair_topk_primitives = get_pair_topk_primitives(config)
    use_pairs = bool(config.get("use_pairs", False))

    # Resolve Möbius-supervision knobs from the SMAC config. Conditional
    # params (``local_alpha``, ``local_pair_beta``) collapse to inert values
    # when their gating bool is off, so a config without them behaves
    # exactly like the pre-fix defaults.
    moebius_cfg = get_local_moebius_cfg(config)
    local_paths_for_trial: Optional[Dict[str, Path]] = None
    if moebius_cfg["use_local_unary"] or moebius_cfg["use_local_pair"]:
        if not local_moebius_paths:
            logger.warning(
                "SMAC trial requested Möbius supervision (use_local_unary=%s "
                "use_local_pair=%s) but no --local_moebius_dir was provided; "
                "disabling local supervision for this trial.",
                moebius_cfg["use_local_unary"], moebius_cfg["use_local_pair"],
            )
            moebius_cfg["use_local_unary"] = False
            moebius_cfg["use_local_pair"] = False
            moebius_cfg["local_alpha"] = 0.0
        else:
            # Only pass benchmarks that actually have a matching file; if
            # *no* benchmark in the current trial has one, fall back to
            # disabling local supervision for the trial (instead of
            # crashing) so SMAC sees a valid, comparable proxy.
            local_paths_for_trial = {
                b: p for b, p in local_moebius_paths.items() if b in benchmarks
            }
            if not local_paths_for_trial:
                logger.warning(
                    "SMAC trial requested Möbius supervision but none of the "
                    "trial benchmarks %s have a matching local-moebius file; "
                    "disabling local supervision for this trial.",
                    list(benchmarks),
                )
                moebius_cfg["use_local_unary"] = False
                moebius_cfg["use_local_pair"] = False
                moebius_cfg["local_alpha"] = 0.0
                local_paths_for_trial = None

    # Per-trial workdir — keep it isolated so concurrent trials don't clash.
    cleanup_tmp = output_dir is None
    if output_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="hpo_comp_"))
    else:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    if scope == "joint":
        ckpt_path = out_dir / "compositional_router_best_joint.pt"
    else:
        if len(benchmarks) != 1:
            raise ValueError(
                f"--scope single requires exactly one benchmark, got {benchmarks!r}"
            )
        ckpt_path = out_dir / f"compositional_router_best_{benchmarks[0]}.pt"

    t0 = time.time()
    payload = train_one_router(
        artifacts=artifacts,
        benchmarks=benchmarks,
        output_path=ckpt_path,
        compressor_type="last_token",
        compressor_d_compress=int(config["compressor_d_compress"]),
        compressor_n_heads=int(config["compressor_n_heads"]),
        compressor_n_latent=int(config["compressor_n_latent"]),
        encoder_hidden_dims=[],
        encoder_dropout=float(config.get("encoder_dropout", 0.1)),
        freeze_compressor=False,
        d_latent=int(config["d_latent"]),
        use_id_embedding=bool(config.get("use_id_embedding", True)),
        edit_hidden_dims=edit_hidden_dims,
        edit_dropout=float(config.get("edit_dropout", 0.1)),
        edit_layer_norm_before=True,
        edit_layer_norm_after=bool(config.get("edit_layer_norm_after", False)),
        unary_hidden_dims=unary_hidden_dims,
        unary_dropout=float(config.get("unary_dropout", 0.1)),
        unary_scorer_type=str(config.get("unary_scorer_type", "mlp")),
        primitive_bias=bool(config.get("primitive_bias", False)),
        lam=float(config.get("lam", 0.0)),
        tau=float(config.get("tau", 1.0)),
        student_temp=float(config.get("student_temperature", 1.0)),
        epochs=int(router_epochs),
        batch_size=int(batch_size),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-2)),
        val_fraction=float(val_fraction),
        seed=int(seed),
        device=device,
        use_full_sequence=bool(use_full_sequence),
        wandb_run=wandb_run,
        wandb_prefix=wandb_prefix,
        wandb_step_offset=int(wandb_step_offset),
        use_pairs=use_pairs,
        pair_hidden_dims=pair_hidden_dims if use_pairs else (96, 96),
        pair_dropout=float(config.get("pair_dropout", 0.1)),
        pair_zero_init=bool(config.get("pair_zero_init", True)),
        pair_l2=float(config.get("pair_l2", 0.0)),
        pair_topk_primitives=pair_topk_primitives,
        dense_delta_paths=dense_paths or None,
        use_dense_supervision=bool(use_dense_supervision),
        downstream_eval_every=int(downstream_eval_every),
        checkpoint_metric=(
            "mean_uplift"
            if objective_metric == "mean_uplift"
            else "obs_top1"
        ),
        observed_path_overrides=None,
        dense_keep_mask_paths=dense_keep_mask_paths or None,
        local_moebius_paths=local_paths_for_trial,
        use_local_unary=bool(moebius_cfg["use_local_unary"]),
        use_local_pair=bool(moebius_cfg["use_local_pair"]),
        local_alpha=float(moebius_cfg["local_alpha"]),
        local_pair_beta=float(moebius_cfg["local_pair_beta"]),
        epoch_report_callback=epoch_report_callback,
        train_test_holdout_count=int(train_test_holdout_count),
        split_json_path=split_json_path,
    )

    if payload is None:
        raise RuntimeError("train_one_router returned None (empty dataset?)")

    metrics = payload.get("metrics") or {}
    proxy = _extract_proxy(metrics, objective_metric)
    best_val_loss = float(payload.get("best_val_loss", float("inf")))
    best_epoch = int(payload.get("best_epoch", -1))

    router_proxy = _CheckpointRouterProxy(ckpt_path, payload)

    result = TrainResult(router=router_proxy)  # type: ignore[arg-type]
    result.gate = None
    result.delta_gate = None
    result.router_val_loss = best_val_loss
    result.gate_val_loss = float("inf")
    result.delta_gate_val_loss = float("inf")
    # Surface the proxy on a pre-existing field so the existing W&B logger
    # in smac_runner picks it up without modification.
    result.score_mean = proxy
    result.training_time_s = time.time() - t0
    result.router_epochs_used = int(router_epochs)
    result.gate_epochs_used = 0

    # Stash compositional-only diagnostics on attributes the existing logger
    # ignores gracefully; downstream code can pull them off if desired.
    result.compositional_metrics = metrics  # type: ignore[attr-defined]
    result.compositional_proxy = proxy  # type: ignore[attr-defined]
    result.compositional_best_epoch = best_epoch  # type: ignore[attr-defined]
    result.compositional_ckpt_path = str(ckpt_path)  # type: ignore[attr-defined]
    result.compositional_payload = payload  # type: ignore[attr-defined]
    result.scope = scope  # type: ignore[attr-defined]
    result.benchmarks = list(benchmarks)  # type: ignore[attr-defined]

    if cleanup_tmp:
        try:
            shutil.rmtree(out_dir, ignore_errors=True)
        except OSError:
            pass

    logger.info(
        "Compositional trial done: epochs=%d  best_epoch=%d  "
        "best_val_loss=%.4f  proxy(%s)=%.5f  (%.1fs)",
        router_epochs, best_epoch, best_val_loss,
        objective_metric, proxy, result.training_time_s,
    )
    return result
