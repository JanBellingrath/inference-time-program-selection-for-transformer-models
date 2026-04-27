"""Optuna search-space builders for unified HPO router families."""

from __future__ import annotations

import math
from typing import Any, Dict, List

import optuna

MIN_HIDDEN = 32


def derive_hidden_dims(depth: int, width: int, shrink: float) -> List[int]:
    """Compute hidden widths from depth/width/shrink tuple."""
    return [max(MIN_HIDDEN, math.floor(width * (shrink ** i))) for i in range(depth)]


def suggest_fine_config(
    trial: optuna.Trial,
    *,
    prefix: str = "fine",
) -> Dict[str, Any]:
    """Sample a fine-router config using Optuna conditionals."""
    p = f"{prefix}."
    cfg: Dict[str, Any] = {}

    cfg["gating_mode"] = trial.suggest_categorical(
        f"{p}gating_mode",
        ["gate_network", "router_confidence", "delta_gate"],
    )
    cfg["target_source"] = trial.suggest_categorical(
        f"{p}target_source",
        ["best_seq", "explored"],
    )

    if cfg["target_source"] == "best_seq":
        # Mirrors the old forbidden clauses.
        cfg["router_loss"] = "hard_ce"
    else:
        cfg["router_loss"] = trial.suggest_categorical(
            f"{p}router_loss",
            ["hard_ce", "soft_ce", "top_ce"],
        )

    cfg["router_depth"] = trial.suggest_categorical(f"{p}router_depth", [1, 2, 3, 4, 5])
    cfg["router_width"] = trial.suggest_categorical(
        f"{p}router_width",
        [128, 256, 512, 1024, 2048],
    )
    cfg["router_shrink"] = (
        trial.suggest_categorical(f"{p}router_shrink", [1.0, 0.75, 0.5, 0.25])
        if cfg["router_depth"] > 1
        else 1.0
    )
    cfg["router_dropout"] = trial.suggest_float(f"{p}router_dropout", 0.0, 0.5)
    cfg["router_lr"] = trial.suggest_float(f"{p}router_lr", 1e-4, 5e-3, log=True)
    cfg["router_weight_decay"] = trial.suggest_float(
        f"{p}router_weight_decay",
        1e-5,
        1e-1,
        log=True,
    )

    if cfg["gating_mode"] in {"gate_network", "delta_gate"}:
        cfg["gate_depth"] = trial.suggest_categorical(f"{p}gate_depth", [1, 2, 3, 4, 5])
        cfg["gate_width"] = trial.suggest_categorical(
            f"{p}gate_width",
            [64, 128, 256, 512, 1024],
        )
        cfg["gate_shrink"] = (
            trial.suggest_categorical(f"{p}gate_shrink", [1.0, 0.75, 0.5, 0.25])
            if cfg["gate_depth"] > 1
            else 1.0
        )
        cfg["gate_dropout"] = trial.suggest_float(f"{p}gate_dropout", 0.0, 0.5)
        cfg["gate_lr"] = trial.suggest_float(f"{p}gate_lr", 1e-4, 1e-3, log=True)
        cfg["gate_weight_decay"] = trial.suggest_float(
            f"{p}gate_weight_decay",
            1e-5,
            5e-2,
            log=True,
        )
        cfg["gate_cost_scale"] = trial.suggest_float(
            f"{p}gate_cost_scale",
            0.75,
            3.0,
            log=True,
        )

    if cfg["target_source"] == "explored":
        cfg["target_temp"] = trial.suggest_float(f"{p}target_temp", 0.5, 1.2)
        cfg["noop_boost"] = trial.suggest_float(f"{p}noop_boost", 0.0, 2.0)

    if cfg["router_loss"] == "hard_ce":
        cfg["label_smoothing"] = trial.suggest_float(f"{p}label_smoothing", 0.0, 0.08)
        cfg["inverse_freq_class_weights"] = trial.suggest_categorical(
            f"{p}inverse_freq_class_weights",
            [True, False],
        )

    if cfg["router_loss"] == "top_ce":
        cfg["top_k"] = trial.suggest_categorical(f"{p}top_k", [2, 4, 8, 16])

    if cfg["gating_mode"] == "gate_network":
        cfg["router_train_subset"] = trial.suggest_categorical(
            f"{p}router_train_subset",
            ["positives_only", "all"],
        )

    return cfg


def suggest_compositional_config(
    trial: optuna.Trial,
    *,
    prefix: str = "compositional",
) -> Dict[str, Any]:
    """Sample a compositional-router config using Optuna conditionals."""
    p = f"{prefix}."
    cfg: Dict[str, Any] = {}

    cfg["d_latent"] = trial.suggest_categorical(f"{p}d_latent", [64, 128, 256])
    cfg["use_id_embedding"] = trial.suggest_categorical(f"{p}use_id_embedding", [True, False])
    cfg["edit_layer_norm_after"] = trial.suggest_categorical(f"{p}edit_layer_norm_after", [True, False])

    cfg["edit_depth"] = trial.suggest_categorical(f"{p}edit_depth", [1, 2, 3])
    cfg["edit_width"] = trial.suggest_categorical(f"{p}edit_width", [64, 128, 256, 512])
    cfg["edit_shrink"] = (
        trial.suggest_categorical(f"{p}edit_shrink", [1.0, 0.75, 0.5])
        if cfg["edit_depth"] > 1
        else 1.0
    )
    cfg["edit_dropout"] = trial.suggest_float(f"{p}edit_dropout", 0.0, 0.5)

    cfg["unary_depth"] = trial.suggest_categorical(f"{p}unary_depth", [1, 2, 3])
    cfg["unary_width"] = trial.suggest_categorical(f"{p}unary_width", [64, 128, 256, 512])
    cfg["unary_shrink"] = (
        trial.suggest_categorical(f"{p}unary_shrink", [1.0, 0.75, 0.5])
        if cfg["unary_depth"] > 1
        else 1.0
    )
    cfg["unary_dropout"] = trial.suggest_float(f"{p}unary_dropout", 0.0, 0.5)

    cfg["compressor_d_compress"] = trial.suggest_categorical(
        f"{p}compressor_d_compress",
        [128, 256, 512],
    )
    cfg["compressor_n_heads"] = trial.suggest_categorical(
        f"{p}compressor_n_heads",
        [2, 4, 8],
    )
    cfg["compressor_n_latent"] = trial.suggest_categorical(
        f"{p}compressor_n_latent",
        [1, 2, 4],
    )
    cfg["encoder_dropout"] = trial.suggest_float(f"{p}encoder_dropout", 0.0, 0.5)

    cfg["lam"] = trial.suggest_float(f"{p}lam", 1e-3, 1.0, log=True)
    cfg["tau"] = trial.suggest_float(f"{p}tau", 0.5, 3.0)
    cfg["student_temperature"] = trial.suggest_float(
        f"{p}student_temperature",
        0.5,
        2.0,
    )

    cfg["use_pairs"] = trial.suggest_categorical(f"{p}use_pairs", [True, False])
    if cfg["use_pairs"]:
        cfg["pair_depth"] = trial.suggest_categorical(f"{p}pair_depth", [1, 2, 3])
        cfg["pair_width"] = trial.suggest_categorical(f"{p}pair_width", [48, 96, 192])
        cfg["pair_shrink"] = (
            trial.suggest_categorical(f"{p}pair_shrink", [1.0, 0.75, 0.5])
            if cfg["pair_depth"] > 1
            else 1.0
        )
        cfg["pair_dropout"] = trial.suggest_float(f"{p}pair_dropout", 0.0, 0.5)
        cfg["pair_l2"] = trial.suggest_float(f"{p}pair_l2", 1e-5, 1e-2, log=True)
        cfg["pair_zero_init"] = trial.suggest_categorical(f"{p}pair_zero_init", [True, False])
        cfg["pair_topk_primitives"] = trial.suggest_categorical(
            f"{p}pair_topk_primitives",
            ["all", "8", "16"],
        )

    cfg["use_local_unary"] = trial.suggest_categorical(f"{p}use_local_unary", [True, False])
    if cfg["use_pairs"]:
        cfg["use_local_pair"] = trial.suggest_categorical(f"{p}use_local_pair", [True, False])
    else:
        cfg["use_local_pair"] = False

    if cfg["use_local_unary"] or cfg["use_local_pair"]:
        cfg["local_alpha"] = trial.suggest_float(f"{p}local_alpha", 1e-3, 1.0, log=True)
    if cfg["use_local_pair"]:
        cfg["local_pair_beta"] = trial.suggest_float(
            f"{p}local_pair_beta",
            1e-2,
            10.0,
            log=True,
        )

    cfg["lr"] = trial.suggest_float(f"{p}lr", 1e-4, 5e-3, log=True)
    cfg["weight_decay"] = trial.suggest_float(f"{p}weight_decay", 1e-5, 1e-1, log=True)

    return cfg
