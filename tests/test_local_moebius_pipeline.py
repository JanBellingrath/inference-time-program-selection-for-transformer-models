"""Tests for the local Mobius supervision data pipeline.

Covers stages 1 and 3 of:

    build_local_subset_catalog -> data_prep.dense_reevaluation -> build_local_moebius_targets

(stage 2 dense eval is exercised in integration tests; here we synthesize ``dense_deltas_matrix.pt``
by hand, which is exactly what dense_reevaluation would produce.)
"""

from __future__ import annotations

import json
import sys as _sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parent.parent
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))

import pytest
import torch

from core.edit_dsl import Primitive, apply_program, canonical_key_str, skip, swap
from data_prep.build_local_subset_catalog import (
    build_catalog_for_benchmark,
    build_for_manifest,
    enumerate_required_subsets,
)
from data_prep.build_local_moebius_targets import (
    build_for_run,
    materialize_for_benchmark,
)
from routers.compositional_router import CompositionalDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _toy_setup():
    """Tiny, deterministic compositional setup.

    Anchor of length 8; three primitives:
      idx 0: skip(2)
      idx 1: skip(3)
      idx 2: swap(4, 5)

    Two questions:
      q=0 observes programs S0 = {0, 1} and S1 = {0}
      q=1 observes programs       S2 = {1, 2}

    Expected unary union: {0, 1, 2}
    Expected pair union (when include_pairs): {(0,1), (1,2)}
    """
    anchor = list(range(8))
    primitives = [skip(2), skip(3), swap(4, 5)]
    # legal_to_prims maps legal-program row index -> sorted primitive indices
    legal_to_prims = [
        [0, 1],   # row 0 -> S0
        [0],      # row 1 -> S1
        [1, 2],   # row 2 -> S2
        [2],      # row 3 -> unused (not observed)
    ]
    observed_rows = [
        {"question_id": 0, "obs_indices": [0, 1], "obs_deltas": [0.0, 0.0]},
        {"question_id": 1, "obs_indices": [2],    "obs_deltas": [0.0]},
    ]
    return anchor, primitives, legal_to_prims, observed_rows


# ---------------------------------------------------------------------------
# Stage 1: enumeration & catalog
# ---------------------------------------------------------------------------


def test_subset_enumeration_dedup():
    anchor, primitives, legal_to_prims, observed_rows = _toy_setup()

    # Without pairs: only singletons, deduped per benchmark.
    q_s, q_p, s_u, p_u = enumerate_required_subsets(
        observed_rows, legal_to_prims, include_pairs=False,
    )
    assert q_s[0] == {0, 1}
    assert q_s[1] == {1, 2}
    assert s_u == {0, 1, 2}
    assert q_p == {} and p_u == set()

    # With pairs.
    q_s, q_p, s_u, p_u = enumerate_required_subsets(
        observed_rows, legal_to_prims, include_pairs=True,
    )
    assert q_s[0] == {0, 1} and q_s[1] == {1, 2}
    assert s_u == {0, 1, 2}
    # q=0: programs are {0,1} (one pair) and {0} (no pair)
    # q=1: program {1,2} (one pair)
    assert q_p[0] == {(0, 1)}
    assert q_p[1] == {(1, 2)}
    assert p_u == {(0, 1), (1, 2)}


def test_route_construction_matches_apply_program():
    anchor, primitives, legal_to_prims, observed_rows = _toy_setup()
    selected, sidecar = build_catalog_for_benchmark(
        bench="toy",
        anchor=anchor,
        primitives=primitives,
        legal_to_prims=legal_to_prims,
        observed_rows=observed_rows,
        include_pairs=True,
        source_manifest="<inline>",
    )
    # Routes and specs must be aligned 1:1 and match apply_program exactly.
    assert len(selected["selected_routes"]) == len(sidecar["routes"])
    for route, spec in zip(selected["selected_routes"], sidecar["routes"]):
        if spec["kind"] == "singleton":
            expected = apply_program(anchor, [primitives[spec["j"]]])
        else:
            expected = apply_program(
                anchor, [primitives[spec["i"]], primitives[spec["j"]]],
            )
        assert route == [int(x) for x in expected]
    # 3 singletons + 2 pairs.
    kinds = [s["kind"] for s in sidecar["routes"]]
    assert kinds == ["singleton", "singleton", "singleton", "pair", "pair"]
    # Per-question membership references existing route ids.
    for entry in sidecar["per_question"]:
        for rid in entry["singleton_route_ids"].values():
            assert 0 <= rid < len(sidecar["routes"])
        for rid in entry["pair_route_ids"].values():
            assert 0 <= rid < len(sidecar["routes"])


def test_catalog_unary_only_excludes_pair_routes():
    anchor, primitives, legal_to_prims, observed_rows = _toy_setup()
    selected, sidecar = build_catalog_for_benchmark(
        bench="toy",
        anchor=anchor,
        primitives=primitives,
        legal_to_prims=legal_to_prims,
        observed_rows=observed_rows,
        include_pairs=False,
        source_manifest="<inline>",
    )
    assert all(s["kind"] == "singleton" for s in sidecar["routes"])
    assert all(entry["pair_route_ids"] == {} for entry in sidecar["per_question"])
    assert sidecar["n_pairs"] == 0
    assert sidecar["n_singletons"] == 3


def _make_synthetic_dense_matrix(
    sidecar: dict, *, q_to_F_singleton: dict, q_to_F_pair: dict,
) -> torch.Tensor:
    """Synthesize a ``[Q, R]`` delta matrix consistent with the sidecar.

    ``q_to_F_singleton[(qid, j)]`` overrides the value at the singleton's rid;
    same for pair. Unspecified entries stay 0.
    """
    n_q = max(int(e["question_id"]) for e in sidecar["per_question"]) + 1
    n_r = len(sidecar["routes"])
    mat = torch.zeros(n_q, n_r, dtype=torch.float32)
    for entry in sidecar["per_question"]:
        qid = int(entry["question_id"])
        for j_str, rid in entry["singleton_route_ids"].items():
            j = int(j_str)
            if (qid, j) in q_to_F_singleton:
                mat[qid, int(rid)] = float(q_to_F_singleton[(qid, j)])
        for pair_key, rid in entry["pair_route_ids"].items():
            a, b = (int(x) for x in pair_key.split(","))
            if (qid, a, b) in q_to_F_pair:
                mat[qid, int(rid)] = float(q_to_F_pair[(qid, a, b)])
    return mat


# ---------------------------------------------------------------------------
# Stage 3: materialization
# ---------------------------------------------------------------------------


def test_materializer_unary_only_targets_equal_F():
    anchor, primitives, legal_to_prims, observed_rows = _toy_setup()
    _, sidecar = build_catalog_for_benchmark(
        bench="toy",
        anchor=anchor,
        primitives=primitives,
        legal_to_prims=legal_to_prims,
        observed_rows=observed_rows,
        include_pairs=False,
        source_manifest="<inline>",
    )
    Fs = {(0, 0): 0.10, (0, 1): -0.05, (1, 1): 0.20, (1, 2): 0.07}
    mat = _make_synthetic_dense_matrix(sidecar, q_to_F_singleton=Fs, q_to_F_pair={})
    payload = materialize_for_benchmark(route_subsets=sidecar, delta_matrix=mat)

    # Singleton target = singleton F (since F(empty)=0).
    pairs = list(zip(
        payload["singleton_qid"].tolist(),
        payload["singleton_idx"].tolist(),
        payload["singleton_target"].tolist(),
        payload["singleton_F"].tolist(),
    ))
    by_key = {(q, j): (t, f) for q, j, t, f in pairs}
    assert by_key[(0, 0)][0] == pytest.approx(0.10)
    assert by_key[(0, 1)][0] == pytest.approx(-0.05)
    assert by_key[(1, 1)][0] == pytest.approx(0.20)
    assert by_key[(1, 2)][0] == pytest.approx(0.07)
    for (_q, _j), (t, f) in by_key.items():
        assert t == pytest.approx(f)
    # No pairs requested.
    assert payload["pair_qid"].numel() == 0
    assert payload["pair_target"].numel() == 0
    assert payload["include_pairs"] is False


def test_materializer_pair_moebius_arithmetic():
    anchor, primitives, legal_to_prims, observed_rows = _toy_setup()
    _, sidecar = build_catalog_for_benchmark(
        bench="toy",
        anchor=anchor,
        primitives=primitives,
        legal_to_prims=legal_to_prims,
        observed_rows=observed_rows,
        include_pairs=True,
        source_manifest="<inline>",
    )
    # q=0 needs unary {0,1} and pair (0,1)
    # q=1 needs unary {1,2} and pair (1,2)
    Fs = {
        (0, 0): 0.10, (0, 1): -0.05,
        (1, 1): 0.20, (1, 2): 0.07,
    }
    Fp = {
        (0, 0, 1): 0.30,                # m = 0.30 - 0.10 - (-0.05) = 0.25
        (1, 1, 2): -0.10,               # m = -0.10 - 0.20 - 0.07 = -0.37
    }
    mat = _make_synthetic_dense_matrix(
        sidecar, q_to_F_singleton=Fs, q_to_F_pair=Fp,
    )
    payload = materialize_for_benchmark(route_subsets=sidecar, delta_matrix=mat)
    pair_records = list(zip(
        payload["pair_qid"].tolist(),
        payload["pair_i"].tolist(),
        payload["pair_j"].tolist(),
        payload["pair_F"].tolist(),
        payload["pair_target"].tolist(),
    ))
    by_pair = {(q, i, j): (f, t) for q, i, j, f, t in pair_records}
    assert (0, 0, 1) in by_pair
    f01, t01 = by_pair[(0, 0, 1)]
    assert f01 == pytest.approx(0.30)
    assert t01 == pytest.approx(0.30 - 0.10 - (-0.05))
    f12, t12 = by_pair[(1, 1, 2)]
    assert f12 == pytest.approx(-0.10)
    assert t12 == pytest.approx(-0.10 - 0.20 - 0.07)
    assert payload["include_pairs"] is True


def test_materializer_rejects_route_count_mismatch():
    anchor, primitives, legal_to_prims, observed_rows = _toy_setup()
    _, sidecar = build_catalog_for_benchmark(
        bench="toy", anchor=anchor, primitives=primitives,
        legal_to_prims=legal_to_prims, observed_rows=observed_rows,
        include_pairs=False, source_manifest="<inline>",
    )
    bad = torch.zeros(2, len(sidecar["routes"]) + 1)
    with pytest.raises(ValueError, match="dense matrix has"):
        materialize_for_benchmark(route_subsets=sidecar, delta_matrix=bad)


# ---------------------------------------------------------------------------
# Round-trip through the loader the training stack actually uses.
# ---------------------------------------------------------------------------


def test_artifact_round_trip_through_dataset_loader(tmp_path):
    anchor, primitives, legal_to_prims, observed_rows = _toy_setup()
    _, sidecar = build_catalog_for_benchmark(
        bench="toy", anchor=anchor, primitives=primitives,
        legal_to_prims=legal_to_prims, observed_rows=observed_rows,
        include_pairs=True, source_manifest="<inline>",
    )
    Fs = {(0, 0): 0.10, (0, 1): -0.05, (1, 1): 0.20, (1, 2): 0.07}
    Fp = {(0, 0, 1): 0.30, (1, 1, 2): -0.10}
    mat = _make_synthetic_dense_matrix(sidecar, q_to_F_singleton=Fs, q_to_F_pair=Fp)
    payload = materialize_for_benchmark(route_subsets=sidecar, delta_matrix=mat)
    payload["benchmark"] = "toy"
    payload["anchor"] = anchor

    pt_path = tmp_path / "local_moebius_toy.pt"
    torch.save(payload, pt_path)

    loaded = CompositionalDataset._load_local_moebius(pt_path)
    # Singletons land in by_qid_unary as (j, target) pairs.
    by_q_u = loaded["by_qid_unary"]
    assert sorted(by_q_u[0]) == [(0, pytest.approx(0.10)), (1, pytest.approx(-0.05))]
    assert sorted(by_q_u[1]) == [(1, pytest.approx(0.20)), (2, pytest.approx(0.07))]
    # Pairs land in by_qid_pair as (i, j, target) with i < j.
    by_q_p = loaded["by_qid_pair"]
    assert by_q_p[0] == [(0, 1, pytest.approx(0.30 - 0.10 - (-0.05)))]
    assert by_q_p[1] == [(1, 2, pytest.approx(-0.10 - 0.20 - 0.07))]


# ---------------------------------------------------------------------------
# End-to-end: catalog builder + materializer driven by a synthetic manifest.
# ---------------------------------------------------------------------------


def _write_synthetic_compositional_artifacts(root: _Path) -> _Path:
    anchor, primitives, legal_to_prims, observed_rows = _toy_setup()
    root = _Path(root)
    (root / "legal_programs").mkdir(parents=True, exist_ok=True)
    (root / "observed").mkdir(parents=True, exist_ok=True)

    with open(root / "primitives.jsonl", "w") as f:
        for j, p in enumerate(primitives):
            f.write(json.dumps({
                "idx": j,
                "kind": p.kind,
                "args": list(p.args),
                "key": canonical_key_str((p,)),
            }) + "\n")

    with open(root / "legal_programs" / "toy.jsonl", "w") as f:
        for r, prims in enumerate(legal_to_prims):
            f.write(json.dumps({
                "idx": r,
                "length": len(prims),
                "primitive_indices": list(prims),
                "key": "+".join(repr(primitives[j]) for j in prims) or "noop",
            }) + "\n")

    with open(root / "observed" / "toy.jsonl", "w") as f:
        for rec in observed_rows:
            f.write(json.dumps({
                "question_id": rec["question_id"],
                "n_obs": len(rec["obs_indices"]),
                "obs_indices": rec["obs_indices"],
                "obs_deltas": rec["obs_deltas"],
            }) + "\n")

    manifest = {
        "primitives_path": "primitives.jsonl",
        "M": len(primitives),
        "benchmarks": {
            "toy": {
                "anchor": anchor,
                "legal_programs_path": "legal_programs/toy.jsonl",
                "observed_path": "observed/toy.jsonl",
            },
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def test_end_to_end_catalog_then_materializer(tmp_path):
    manifest_path = _write_synthetic_compositional_artifacts(tmp_path / "comp")
    catalog_dir = tmp_path / "catalog"
    decode_dir = tmp_path / "decode"
    output_dir = tmp_path / "out"

    summary = build_for_manifest(
        manifest_path, catalog_dir, benchmarks=None, include_pairs=True,
    )
    assert "toy" in summary
    sidecar = json.loads((catalog_dir / "toy" / "route_subsets.json").read_text())
    selected = json.loads((catalog_dir / "toy" / "selected_catalog.json").read_text())
    assert len(selected["selected_routes"]) == len(sidecar["routes"])

    # Synthesize what dense_reevaluation would have produced.
    Fs = {(0, 0): 0.1, (0, 1): -0.05, (1, 1): 0.2, (1, 2): 0.07}
    Fp = {(0, 0, 1): 0.3, (1, 1, 2): -0.1}
    mat = _make_synthetic_dense_matrix(sidecar, q_to_F_singleton=Fs, q_to_F_pair=Fp)
    (decode_dir / "toy").mkdir(parents=True, exist_ok=True)
    torch.save({"delta_matrix": mat}, decode_dir / "toy" / "dense_deltas_matrix.pt")

    summary2 = build_for_run(catalog_dir, decode_dir, output_dir)
    assert "toy" in summary2
    pt_path = output_dir / "local_moebius_toy.pt"
    assert pt_path.is_file()

    loaded = CompositionalDataset._load_local_moebius(pt_path)
    assert sorted(loaded["by_qid_unary"][0]) == [
        (0, pytest.approx(0.1)), (1, pytest.approx(-0.05)),
    ]
    assert loaded["by_qid_pair"][0] == [
        (0, 1, pytest.approx(0.3 - 0.1 - (-0.05))),
    ]
