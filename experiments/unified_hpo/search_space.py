"""Conditional ConfigSpace for the unified single-benchmark fine-router HPO.

The searched object is a full routed decision system:

    lambda = (gating_mode, target_source, router_loss,
              router_arch, gate_arch, router_optim, gate_optim,
              router_train_subset)

Parameters are conditional on the active branch.  Forbidden clauses enforce
invalid combinations (e.g. soft_ce with best_seq targets).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from ConfigSpace import (
    AndConjunction,
    Categorical,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    InCondition,
    Integer,
    OrConjunction,
)


# ---------------------------------------------------------------------------
# Derived hidden widths from (depth, width, shrink)
# ---------------------------------------------------------------------------

MIN_HIDDEN = 32


def derive_hidden_dims(depth: int, width: int, shrink: float) -> List[int]:
    """Compute the hidden-layer widths for a depth/width/shrink specification.

    For depth *d*, width *w*, and shrink factor *s*:
        - depth 1: ``[w]``
        - depth 2: ``[w, max(MIN_HIDDEN, floor(w * s))]``
        - depth 3: ``[w, max(MIN_HIDDEN, floor(w * s)), max(MIN_HIDDEN, floor(w * s^2))]``

    The minimum width is clamped at ``MIN_HIDDEN`` to prevent degenerate
    bottlenecks.
    """
    dims: List[int] = []
    for i in range(depth):
        h = max(MIN_HIDDEN, math.floor(width * (shrink ** i)))
        dims.append(h)
    return dims


# ---------------------------------------------------------------------------
# ConfigSpace builder
# ---------------------------------------------------------------------------

def build_configspace() -> ConfigurationSpace:
    """Build the full conditional configuration space.

    Returns a ``ConfigurationSpace`` with conditions and forbidden clauses
    that enforce all branch-awareness rules from the specification.
    """
    cs = ConfigurationSpace(seed=42)

    # === Top-level categoricals ============================================

    gating_mode = Categorical(
        "gating_mode",
        items=["gate_network", "router_confidence", "delta_gate"],
        default="gate_network",
    )
    target_source = Categorical(
        "target_source",
        items=["best_seq", "explored"],
        default="explored",
    )
    router_loss = Categorical(
        "router_loss",
        items=["hard_ce", "soft_ce", "top_ce"],
        default="hard_ce",
    )

    cs.add([gating_mode, target_source, router_loss])

    # router_loss is only meaningful (non-hard_ce) when target_source=explored.
    # We add it unconditionally but use a forbidden clause to block
    # (target_source=best_seq, router_loss!=hard_ce).

    # === Router architecture ===============================================

    router_depth = Categorical("router_depth", items=[1, 2, 3, 4, 5], default=3)
    router_width = Categorical("router_width", items=[128, 256, 512, 1024, 2048], default=512)
    router_shrink = Categorical(
        "router_shrink", items=[1.0, 0.75, 0.5, 0.25], default=0.75,
    )
    router_dropout = Float("router_dropout", bounds=(0.0, 0.50), default=0.15)
    router_lr = Float("router_lr", bounds=(1e-4, 5e-3), default=1e-3, log=True)
    router_wd = Float(
        "router_weight_decay", bounds=(1e-5, 10e-2), default=1e-2, log=True,
    )

    cs.add([router_depth, router_width, router_shrink,
            router_dropout, router_lr, router_wd])

    # router_shrink is only active when depth > 1
    cs.add(InCondition(router_shrink, router_depth, [2, 3, 4, 5]))

    # === Gate / delta-gate architecture ====================================
    # Active only when gating_mode in {gate_network, delta_gate}.

    gate_depth = Categorical("gate_depth", items=[1, 2, 3, 4, 5], default=2)
    gate_width = Categorical("gate_width", items=[64, 128, 256, 512, 1024], default=128)
    gate_shrink = Categorical("gate_shrink", items=[1.0, 0.75, 0.5, 0.25], default=0.75)
    gate_dropout = Float("gate_dropout", bounds=(0.0, 0.50), default=0.15)
    gate_lr = Float("gate_lr", bounds=(1e-4, 1e-3), default=1e-4, log=True)
    gate_wd = Float("gate_weight_decay", bounds=(1e-5, 5e-2), default=5e-3, log=True)
    gate_cost_scale = Float(
        "gate_cost_scale", bounds=(0.75, 3.0), default=1.5, log=True,
    )

    gate_params_no_shrink = [gate_depth, gate_width,
                             gate_dropout, gate_lr, gate_wd, gate_cost_scale]
    cs.add(gate_params_no_shrink)
    cs.add(gate_shrink)

    # All gate params (except shrink) conditional on gating_mode in {gate_network, delta_gate}
    for p in gate_params_no_shrink:
        cs.add(InCondition(p, gating_mode, ["gate_network", "delta_gate"]))

    # gate_shrink requires BOTH gating_mode in {gate_network, delta_gate} AND gate_depth > 1
    cs.add(AndConjunction(
        InCondition(gate_shrink, gating_mode, ["gate_network", "delta_gate"]),
        InCondition(gate_shrink, gate_depth, [2, 3, 4, 5]),
    ))

    # === Target construction (explored branch only) ========================

    target_temp = Float("target_temp", bounds=(0.5, 1.2), default=1.0)
    noop_boost = Float("noop_boost", bounds=(0.0, 2.0), default=0.0)

    cs.add([target_temp, noop_boost])
    cs.add(EqualsCondition(target_temp, target_source, "explored"))
    cs.add(EqualsCondition(noop_boost, target_source, "explored"))

    # === Loss-specific parameters ==========================================

    label_smoothing = Float("label_smoothing", bounds=(0.0, 0.08), default=0.0)
    inv_freq = Categorical(
        "inverse_freq_class_weights", items=[True, False], default=True,
    )

    cs.add([label_smoothing, inv_freq])
    cs.add(EqualsCondition(label_smoothing, router_loss, "hard_ce"))

    top_k = Categorical("top_k", items=[2, 4, 8, 16], default=4)
    cs.add(top_k)
    cs.add(EqualsCondition(top_k, router_loss, "top_ce"))

    # === Router training subset policy =====================================
    # Active only when gating_mode=gate_network.

    router_train_subset = Categorical(
        "router_train_subset",
        items=["positives_only", "all"],
        default="positives_only",
    )
    cs.add(router_train_subset)
    cs.add(EqualsCondition(router_train_subset, gating_mode, "gate_network"))

    # === Forbidden clauses =================================================
    # Rule 1: target_source=best_seq forbids router_loss in {soft_ce, top_ce}
    cs.add(ForbiddenAndConjunction(
        ForbiddenEqualsClause(target_source, "best_seq"),
        ForbiddenEqualsClause(router_loss, "soft_ce"),
    ))
    cs.add(ForbiddenAndConjunction(
        ForbiddenEqualsClause(target_source, "best_seq"),
        ForbiddenEqualsClause(router_loss, "top_ce"),
    ))

    # Rule 2 (top_ce requires explored) is a subset of rule 1 — already covered.

    # Rules 3-7 are handled by the conditional structure above (parameters
    # are simply absent when their parent condition is not met).

    return cs


# ---------------------------------------------------------------------------
# Helpers for extracting derived values from a SMAC configuration
# ---------------------------------------------------------------------------

def get_router_hidden_dims(config: Dict) -> List[int]:
    """Extract router hidden dims from a resolved configuration dict."""
    depth = int(config["router_depth"])
    width = int(config["router_width"])
    shrink = float(config.get("router_shrink", 1.0))
    return derive_hidden_dims(depth, width, shrink)


def get_gate_hidden_dims(config: Dict) -> List[int]:
    """Extract gate hidden dims from a resolved configuration dict.

    Returns an empty list if gate parameters are not active.
    """
    if "gate_depth" not in config:
        return []
    depth = int(config["gate_depth"])
    width = int(config["gate_width"])
    shrink = float(config.get("gate_shrink", 1.0))
    return derive_hidden_dims(depth, width, shrink)


def config_to_dict(config) -> Dict:
    """Convert a SMAC Configuration to a plain dict with only active parameters."""
    return dict(config)
