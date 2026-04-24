"""Conditional ConfigSpace for the *compositional* router HPO.

The compositional router (see :mod:`routers.compositional_router`) has a
substantially different architecture than the fine router/gate stack handled
by :mod:`experiments.unified_hpo.search_space`:

* a residual *compressor* (``last_token`` here) extracts ``g_q`` from the
  pivot-layer residuals;
* an *edit MLP* maps each primitive's symbolic embedding to a per-question
  vector;
* a *unary scorer* MLP turns those into ``u_q``;
* an optional *pair scorer* MLP produces pairwise corrections ``v_q``;
* program scores combine ``u_q`` (and ``v_q``) through the per-benchmark
  legal-program incidence matrices.

The searched object here is the architecture + scoring + optional pair head
+ optimization knobs accepted by ``training.train_compositional_router``'s
``train_one_router``.  Branch-aware conditions keep the active set small.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Float,
    InCondition,
    OrConjunction,
)

from experiments.unified_hpo.search_space import derive_hidden_dims


# ---------------------------------------------------------------------------
# ConfigSpace builder
# ---------------------------------------------------------------------------

def build_configspace_compositional() -> ConfigurationSpace:
    """Build the conditional configuration space for the compositional router."""
    cs = ConfigurationSpace(seed=42)

    # === Architecture: latent + per-MLP depth/width/shrink ===============

    d_latent = Categorical("d_latent", items=[64, 128, 256], default=128)
    use_id_embedding = Categorical("use_id_embedding", items=[True, False], default=True)
    edit_layer_norm_after = Categorical(
        "edit_layer_norm_after", items=[True, False], default=False,
    )

    edit_depth = Categorical("edit_depth", items=[1, 2, 3], default=2)
    edit_width = Categorical("edit_width", items=[64, 128, 256, 512], default=128)
    edit_shrink = Categorical("edit_shrink", items=[1.0, 0.75, 0.5], default=1.0)
    edit_dropout = Float("edit_dropout", bounds=(0.0, 0.5), default=0.1)

    unary_depth = Categorical("unary_depth", items=[1, 2, 3], default=2)
    unary_width = Categorical("unary_width", items=[64, 128, 256, 512], default=128)
    unary_shrink = Categorical("unary_shrink", items=[1.0, 0.75, 0.5], default=1.0)
    unary_dropout = Float("unary_dropout", bounds=(0.0, 0.5), default=0.1)

    cs.add([
        d_latent, use_id_embedding, edit_layer_norm_after,
        edit_depth, edit_width, edit_shrink, edit_dropout,
        unary_depth, unary_width, unary_shrink, unary_dropout,
    ])

    # shrink only meaningful when depth > 1
    cs.add(InCondition(edit_shrink, edit_depth, [2, 3]))
    cs.add(InCondition(unary_shrink, unary_depth, [2, 3]))

    # === Compressor (last_token; top_down_attention requires full sequences) ===

    compressor_d_compress = Categorical(
        "compressor_d_compress", items=[128, 256, 512], default=256,
    )
    compressor_n_heads = Categorical(
        "compressor_n_heads", items=[2, 4, 8], default=4,
    )
    compressor_n_latent = Categorical(
        "compressor_n_latent", items=[1, 2, 4], default=1,
    )
    encoder_dropout = Float("encoder_dropout", bounds=(0.0, 0.5), default=0.1)

    cs.add([
        compressor_d_compress, compressor_n_heads,
        compressor_n_latent, encoder_dropout,
    ])

    # === Scoring ==========================================================

    lam = Float("lam", bounds=(1e-3, 1.0), default=1e-2, log=True)
    tau = Float("tau", bounds=(0.5, 3.0), default=1.0)
    student_temperature = Float(
        "student_temperature", bounds=(0.5, 2.0), default=1.0,
    )

    cs.add([lam, tau, student_temperature])

    # === Optional pair head ================================================

    use_pairs = Categorical("use_pairs", items=[True, False], default=True)

    pair_depth = Categorical("pair_depth", items=[1, 2, 3], default=2)
    pair_width = Categorical("pair_width", items=[48, 96, 192], default=96)
    pair_shrink = Categorical("pair_shrink", items=[1.0, 0.75, 0.5], default=1.0)
    pair_dropout = Float("pair_dropout", bounds=(0.0, 0.5), default=0.1)
    pair_l2 = Float("pair_l2", bounds=(1e-5, 1e-2), default=1e-3, log=True)
    pair_zero_init = Categorical("pair_zero_init", items=[True, False], default=True)
    # "all" is mapped to None at materialization time.
    pair_topk_primitives = Categorical(
        "pair_topk_primitives", items=["all", "8", "16"], default="all",
    )

    cs.add([
        use_pairs, pair_depth, pair_width, pair_shrink, pair_dropout,
        pair_l2, pair_zero_init, pair_topk_primitives,
    ])

    # All pair-only knobs conditional on use_pairs
    for p in (
        pair_depth, pair_width, pair_dropout, pair_l2,
        pair_zero_init, pair_topk_primitives,
    ):
        cs.add(InCondition(p, use_pairs, [True]))
    # pair_shrink: requires use_pairs AND pair_depth > 1
    # ConfigSpace's InCondition only lets one parent — but a fully nested
    # AndConjunction would force *both* parents to be active. The simpler
    # equivalent: gate on pair_depth, which itself is gated on use_pairs.
    cs.add(InCondition(pair_shrink, pair_depth, [2, 3]))

    # === Optional Möbius local supervision =================================
    # Local unary / pair Möbius targets add an auxiliary MSE loss whose
    # targets decompose a program's observed uplift into per-primitive and
    # per-pair contributions (``data_prep.build_local_moebius_targets``).
    # Exposing these knobs to SMAC is what turns any statement like "HPO
    # shows Möbius helps/hurts" from a spec-level claim into an empirical
    # finding, because otherwise the HPO trial set is pinned at
    # ``use_local_unary = use_local_pair = False`` (the pre-fix defaults in
    # ``compositional_objective.train_and_score_compositional``).
    #
    # ``use_local_pair`` is meaningful only when ``use_pairs=True``
    # (``local_moebius_loss`` short-circuits when ``router.pair_scorer`` is
    # ``None``), so we gate it on ``use_pairs``.
    #
    # ``local_alpha`` scales the *total* local loss (unary + pair) and is
    # only active when at least one of the two is on.
    # ``local_pair_beta`` reweights the pair term relative to unary inside
    # that total and is only active when ``use_local_pair`` is on.
    use_local_unary = Categorical(
        "use_local_unary", items=[True, False], default=False,
    )
    use_local_pair = Categorical(
        "use_local_pair", items=[True, False], default=False,
    )
    local_alpha = Float(
        "local_alpha", bounds=(1e-3, 1.0), default=1e-1, log=True,
    )
    local_pair_beta = Float(
        "local_pair_beta", bounds=(1e-2, 10.0), default=1.0, log=True,
    )
    cs.add([use_local_unary, use_local_pair, local_alpha, local_pair_beta])

    cs.add(InCondition(use_local_pair, use_pairs, [True]))
    cs.add(OrConjunction(
        InCondition(local_alpha, use_local_unary, [True]),
        InCondition(local_alpha, use_local_pair, [True]),
    ))
    cs.add(InCondition(local_pair_beta, use_local_pair, [True]))

    # === Optimization ======================================================

    lr = Float("lr", bounds=(1e-4, 5e-3), default=1e-3, log=True)
    weight_decay = Float(
        "weight_decay", bounds=(1e-5, 1e-1), default=1e-2, log=True,
    )

    cs.add([lr, weight_decay])

    return cs


# ---------------------------------------------------------------------------
# Helpers for materializing a SMAC config dict into trainer kwargs
# ---------------------------------------------------------------------------

def get_edit_hidden_dims(config: Dict) -> List[int]:
    depth = int(config["edit_depth"])
    width = int(config["edit_width"])
    shrink = float(config.get("edit_shrink", 1.0))
    return derive_hidden_dims(depth, width, shrink)


def get_unary_hidden_dims(config: Dict) -> List[int]:
    depth = int(config["unary_depth"])
    width = int(config["unary_width"])
    shrink = float(config.get("unary_shrink", 1.0))
    return derive_hidden_dims(depth, width, shrink)


def get_pair_hidden_dims(config: Dict) -> List[int]:
    if not bool(config.get("use_pairs", False)):
        return []
    depth = int(config["pair_depth"])
    width = int(config["pair_width"])
    shrink = float(config.get("pair_shrink", 1.0))
    return derive_hidden_dims(depth, width, shrink)


def get_pair_topk_primitives(config: Dict) -> Optional[int]:
    """Map the categorical ``pair_topk_primitives`` to ``Optional[int]``.

    ``"all"`` -> ``None`` (no top-k pruning, dense pair scoring).
    """
    if not bool(config.get("use_pairs", False)):
        return None
    raw = config.get("pair_topk_primitives", "all")
    if raw is None or raw == "all":
        return None
    return int(raw)


def get_local_moebius_cfg(config: Dict) -> Dict[str, Any]:
    """Resolve the active Möbius-supervision knobs from a SMAC config dict.

    Conditional parameters are absent from ``config`` when inactive, so we
    read them with ``.get`` and coerce everything to the types expected by
    :func:`training.train_compositional_router.train_one_router`. Inactive
    combinations collapse to ``use_local_* = False`` and ``local_alpha = 0``
    (i.e. no auxiliary loss), matching the pre-HPO defaults.
    """
    use_pairs = bool(config.get("use_pairs", False))
    use_local_unary = bool(config.get("use_local_unary", False))
    use_local_pair = bool(config.get("use_local_pair", False)) and use_pairs
    any_local = use_local_unary or use_local_pair
    return {
        "use_local_unary": use_local_unary,
        "use_local_pair": use_local_pair,
        "local_alpha": float(config.get("local_alpha", 0.0)) if any_local else 0.0,
        "local_pair_beta": (
            float(config.get("local_pair_beta", 1.0)) if use_local_pair else 1.0
        ),
    }


def config_to_dict(config) -> Dict:
    """Convert a SMAC Configuration to a plain dict with only active params."""
    return dict(config)
