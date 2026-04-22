"""Acceptance tests for the compositional generalization (Step 3) modules.

These tests build small synthetic graphs / catalogues to exercise:

* :mod:`experiments.holdout_edge_split` -- eligibility predicates and
  per-endpoint constraints on the constrained split,
* :mod:`experiments.filter_observed_for_holdout` -- pair-only removal
  while preserving singletons / empty programs,
* :mod:`experiments.eval_compositional_generalization` -- the
  observed-support / full-catalogue metric arithmetic on hand-built
  scores,
* :mod:`experiments.bootstrap_pair_metrics` -- the bootstrap-mean and
  paired-difference helpers on tiny inputs.
"""

from __future__ import annotations

import json
import sys as _sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))

import math

import numpy as np
import torch

from experiments.bootstrap_pair_metrics import (
    bootstrap_mean_ci,
    bootstrap_paired_diff,
)
from experiments.eval_compositional_generalization import (
    _aggregate_per_edge,
    _restricted_metrics,
)
from experiments.filter_observed_for_holdout import (
    _filter_observed_for_benchmark,
    _held_out_legal_indices,
    _holdout_pair_set,
)
from experiments.holdout_edge_split import (
    _eligible_edges,
    _split_eligible,
)


# ---------------------------------------------------------------------------
# Holdout edge split
# ---------------------------------------------------------------------------


def _toy_graph(*, n_prim: int = 5, edge_count: int = 10):
    """Tiny graph with high counts so all edges trivially pass thresholds."""
    primitives = [
        {
            "idx": i, "key": f"p{i}", "kind": "skip", "args": [i],
            "n_singleton": 100, "n_pair_total": 100,
            "n_questions_singleton": 100, "n_questions_pair": 100,
            "n_questions_any": 100,
            "deg": n_prim - 1,  # fully connected
        }
        for i in range(n_prim)
    ]
    edges = []
    for a in range(n_prim):
        for b in range(a + 1, n_prim):
            edges.append({
                "a": a, "b": b,
                "key_a": f"p{a}", "key_b": f"p{b}",
                "count": edge_count,
                "n_questions": edge_count,
                "mean_delta": 0.5,
                "sum_delta": edge_count * 0.5,
                "per_benchmark": {},
            })
    return primitives, edges


def test_eligibility_filters_low_support_and_residual_degree():
    primitives, edges = _toy_graph(n_prim=4, edge_count=10)
    # Knock one edge below c_min and one endpoint below u_min.
    edges[0] = dict(edges[0]); edges[0]["count"] = 1; edges[0]["n_questions"] = 1
    primitives[3] = dict(primitives[3]); primitives[3]["n_questions_any"] = 1

    eligible, deg = _eligible_edges(
        edges, primitives, c_min=5, u_min=10, d_min=1, delta_min=None,
    )
    keys = {(int(e["a"]), int(e["b"])) for e in eligible}
    # Edge 0 is dropped (low count), every edge touching primitive 3 is dropped (u_min).
    assert (0, 1) not in keys  # low count
    for e in edges:
        if int(e["a"]) == 3 or int(e["b"]) == 3:
            assert (int(e["a"]), int(e["b"])) not in keys
    # Some edges should still be eligible.
    assert (1, 2) in keys


def test_split_respects_per_endpoint_cap_and_residual_degree():
    primitives, edges = _toy_graph(n_prim=5, edge_count=10)
    eligible, full_deg = _eligible_edges(
        edges, primitives, c_min=1, u_min=1, d_min=2, delta_min=None,
    )
    val, test, train = _split_eligible(
        eligible,
        full_deg=full_deg,
        val_fraction=0.2,
        test_fraction=0.4,
        d_min=2,
        max_holdouts_per_endpoint=2,
        seed=0,
    )
    held = val + test
    # No primitive should appear in more held-out edges than the cap allows.
    counts = {p["idx"]: 0 for p in primitives}
    for e in held:
        counts[int(e["a"])] += 1
        counts[int(e["b"])] += 1
    assert all(c <= 2 for c in counts.values()), counts
    # Residual degree predicate.
    for p in primitives:
        held_for_p = counts[p["idx"]]
        assert full_deg[p["idx"]] - held_for_p >= 2, (p, full_deg[p["idx"]], held_for_p)
    # Disjointness.
    val_keys = {(int(e["a"]), int(e["b"])) for e in val}
    test_keys = {(int(e["a"]), int(e["b"])) for e in test}
    assert val_keys.isdisjoint(test_keys)


# ---------------------------------------------------------------------------
# Observed filtering
# ---------------------------------------------------------------------------


def _toy_legal():
    """Five rows: empty, two singletons, three length-2 pairs."""
    return [
        {"idx": 0, "length": 0, "primitive_indices": [], "key": "noop"},
        {"idx": 1, "length": 1, "primitive_indices": [0], "key": "p0"},
        {"idx": 2, "length": 1, "primitive_indices": [1], "key": "p1"},
        {"idx": 3, "length": 2, "primitive_indices": [0, 1], "key": "p0+p1"},
        {"idx": 4, "length": 2, "primitive_indices": [0, 2], "key": "p0+p2"},
        {"idx": 5, "length": 2, "primitive_indices": [1, 2], "key": "p1+p2"},
    ]


def test_filter_drops_pair_keeps_singleton_and_empty():
    legal = _toy_legal()
    obs = [
        {"residual_idx": 0, "question_id": 0, "n_obs": 5,
         "obs_indices": [0, 1, 2, 3, 5],
         "obs_deltas": [0.0, 0.5, 0.4, 1.0, 0.7]},
    ]
    pairs = {(0, 1)}  # held-out edge -> legal row 3
    held_idx = _held_out_legal_indices(legal, pairs)
    assert held_idx == {3}
    filtered, stats = _filter_observed_for_benchmark(obs, legal, held_idx)
    assert len(filtered) == 1
    rec = filtered[0]
    assert 3 not in rec["obs_indices"]
    # Singletons + empty still present, and the surviving pair (1,2) too.
    assert set(rec["obs_indices"]) == {0, 1, 2, 5}
    assert stats["candidates_dropped"] == 1
    assert stats["rows_kept"] == 1


def test_filter_drops_question_with_only_empty_remaining():
    legal = _toy_legal()
    obs = [
        # All non-empty candidates are length-2 pairs that get held out.
        {"residual_idx": 0, "question_id": 0, "n_obs": 4,
         "obs_indices": [0, 3, 4, 5], "obs_deltas": [0.0, 1.0, 0.5, 0.2]},
    ]
    held_idx = _held_out_legal_indices(legal, {(0, 1), (0, 2), (1, 2)})
    assert held_idx == {3, 4, 5}
    filtered, stats = _filter_observed_for_benchmark(obs, legal, held_idx)
    assert filtered == []
    assert stats["rows_dropped_no_nonempty"] == 1


def test_holdout_pair_set_is_unordered_and_disjoint_aware():
    split = {
        "E_val_holdout": [{"a": 1, "b": 0}],
        "E_test_holdout": [{"a": 2, "b": 1}, {"a": 1, "b": 2}],
    }
    assert _holdout_pair_set(split) == {(0, 1), (1, 2)}


# ---------------------------------------------------------------------------
# Per-(q, e*) and per-edge metrics
# ---------------------------------------------------------------------------


def test_restricted_metrics_match_hand_calc():
    # 4 candidates with these scores; e* = idx 2.
    scores = torch.tensor([0.0, 1.0, 2.0, -1.0])
    candidates = [0, 1, 2, 3]
    metrics = _restricted_metrics(scores, candidates, target_idx=2)
    assert metrics["rank"] == 1.0
    assert metrics["top1"] == 1.0
    assert metrics["top3"] == 1.0
    # Hand-checked softmax probability on indices 0..3 with target at 2.
    probs = torch.softmax(scores, dim=-1).tolist()
    assert math.isclose(metrics["prob"], probs[2], abs_tol=1e-6)
    assert math.isclose(metrics["lift"], probs[2] * 4, abs_tol=1e-6)


def test_restricted_metrics_subset_uses_only_candidates():
    scores = torch.tensor([10.0, 0.0, 1.0, 0.0])
    # e* = 1 in a candidate set that excludes the dominant index 0.
    metrics = _restricted_metrics(scores, [1, 2, 3], target_idx=1)
    sub = torch.tensor([0.0, 1.0, 0.0])
    expected = float(torch.softmax(sub, dim=-1)[0])
    assert math.isclose(metrics["prob"], expected, abs_tol=1e-6)
    assert metrics["rank"] == 2.0
    assert metrics["top1"] == 0.0
    assert metrics["top3"] == 1.0


def test_restricted_metrics_target_absent_returns_zero():
    scores = torch.tensor([0.5, 1.0, 2.0])
    metrics = _restricted_metrics(scores, [0, 1], target_idx=2)
    assert metrics["prob"] == 0.0
    assert metrics["top1"] == 0.0


def test_aggregate_per_edge_means_what_it_should():
    rows = [
        {"obs_prob": 0.4, "obs_rank": 1, "obs_lift": 1.0, "obs_top1": 1.0,
         "obs_top3": 1.0, "obs_top5": 1.0,
         "full_prob": 0.1, "full_rank": 5, "full_lift": 0.4, "full_top1": 0.0,
         "full_top3": 0.0, "full_top5": 1.0},
        {"obs_prob": 0.2, "obs_rank": 3, "obs_lift": 0.5, "obs_top1": 0.0,
         "obs_top3": 1.0, "obs_top5": 1.0,
         "full_prob": 0.05, "full_rank": 9, "full_lift": 0.2, "full_top1": 0.0,
         "full_top3": 0.0, "full_top5": 0.0},
    ]
    agg = _aggregate_per_edge(rows)
    assert agg["n_questions"] == 2
    assert math.isclose(agg["obs_mean_prob"], 0.3)
    assert math.isclose(agg["obs_top1_rate"], 0.5)
    assert math.isclose(agg["obs_top3_rate"], 1.0)
    assert math.isclose(agg["full_mean_rank"], 7.0)


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------


def test_bootstrap_mean_ci_brackets_population_mean():
    rng = np.random.default_rng(42)
    values = rng.normal(loc=0.7, scale=0.05, size=200)
    res = bootstrap_mean_ci(values, B=2000, seed=0, alpha=0.05)
    # 95% CI should bracket the sample mean tightly.
    assert res["lo"] < res["mean"] < res["hi"]
    assert res["hi"] - res["lo"] < 0.05  # tight w/ small sigma + n=200


def test_bootstrap_paired_diff_handles_constant_difference():
    a = np.linspace(0.1, 0.9, 50)
    b = a - 0.1
    res = bootstrap_paired_diff(a, b, B=1000, seed=1, alpha=0.05)
    assert math.isclose(res["mean_diff"], 0.1, abs_tol=1e-9)
    assert res["lo"] > 0
    assert res["fraction_positive"] == 1.0
