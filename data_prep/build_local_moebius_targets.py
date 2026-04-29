#!/usr/bin/env python3
"""Materialize ``local_moebius_{bench}.pt`` from a decoded subset catalog.

Pipeline (Stage 3 of 3)
=======================

Given the per-benchmark sidecar produced by
:mod:`data_prep.build_local_subset_catalog` (Stage 1) and the
``dense_deltas_matrix.pt`` produced by ``data_prep/dense_reevaluation.py``
(Stage 2, run unchanged on the catalog), this script extracts per-question
anchor-relative subset utilities ``F_q(T;a)`` for the subsets each question
actually requested, then emits local Möbius supervision targets:

* unary  ``m_q({o_j})       = F_q({o_j};a)``                       (since ``F_q(\\emptyset;a) = 0``)
* pair   ``m_q({o_i,o_j}) = F_q({o_i,o_j};a) - F_q({o_i};a) - F_q({o_j};a)``

Output schema (per benchmark, ``local_moebius_{bench}.pt``) matches exactly
what ``CompositionalDataset._load_local_moebius`` consumes::

    {
      # Mobius targets (loader-facing fields)
      "singleton_qid":    LongTensor[S],
      "singleton_idx":    LongTensor[S],     # primitive id j
      "singleton_target": FloatTensor[S],    # m_q({o_j})
      "pair_qid":         LongTensor[P],
      "pair_i":           LongTensor[P],     # i < j
      "pair_j":           LongTensor[P],
      "pair_target":      FloatTensor[P],    # m_q({o_i, o_j})

      # Raw subset utilities (provenance / lets training pick)
      "singleton_F":      FloatTensor[S],    # F_q({o_j};a)
      "pair_F":           FloatTensor[P],    # F_q({o_i,o_j};a)

      "benchmark":           "<bench>",
      "anchor":              [layer, ...],
      "include_pairs":       bool,
      "source_catalog":      "...selected_catalog.json",
      "source_dense_matrix": "...dense_deltas_matrix.pt",
      "source_route_subsets": "...route_subsets.json",
    }

Usage
-----

::

    python -m data_prep.build_local_moebius_targets \\
        --catalog_dir local_subsets/<run> \\
        --decode_dir  local_subsets_decoded/<run> \\
        --output_dir  local_moebius/<run> \\
        [--benchmarks commonsenseqa boolq]

For each benchmark ``b`` this expects:

* ``<catalog_dir>/<b>/route_subsets.json``  (Stage 1)
* ``<catalog_dir>/<b>/selected_catalog.json``
* ``<decode_dir>/<b>/dense_deltas_matrix.pt`` (Stage 2 output)

and writes ``<output_dir>/local_moebius_<b>.pt``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger("build_local_moebius_targets")


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _load_route_subsets(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def _load_dense_matrix(path: Path) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load ``dense_deltas_matrix.pt`` and return ``(delta_matrix, payload)``.

    ``delta_matrix[q, r] = u(q, route_r) - u(q, anchor) = F_q(T_r; a)`` since
    each route ``T_r`` was constructed by applying subset ``T_r`` to anchor.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if "delta_matrix" not in payload:
        raise KeyError(f"{path} missing 'delta_matrix' key (got {sorted(payload)})")
    return payload["delta_matrix"].float(), payload


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------


def materialize_for_benchmark(
    *,
    route_subsets: Dict[str, Any],
    delta_matrix: torch.Tensor,
) -> Dict[str, Any]:
    """Produce the per-benchmark Mobius payload from the sidecar + dense matrix.

    ``delta_matrix`` is ``[Q, R]`` with ``Q`` >= max question_id + 1 and ``R`` ==
    number of routes in the sidecar.  Question rows for question_ids the sidecar
    does not list are simply ignored.
    """
    Q, R = delta_matrix.shape
    n_routes = len(route_subsets["routes"])
    if R != n_routes:
        raise ValueError(
            f"dense matrix has {R} routes but sidecar lists {n_routes}; mismatch."
        )

    include_pairs = bool(route_subsets.get("include_pairs", False))

    s_qid: List[int] = []
    s_idx: List[int] = []
    s_F: List[float] = []
    s_target: List[float] = []  # = s_F (since F(empty)=0)

    p_qid: List[int] = []
    p_i: List[int] = []
    p_j: List[int] = []
    p_F: List[float] = []
    p_target: List[float] = []

    n_skipped_qid_oob = 0
    for entry in route_subsets["per_question"]:
        qid = int(entry["question_id"])
        if qid < 0 or qid >= Q:
            n_skipped_qid_oob += 1
            continue
        row = delta_matrix[qid]
        unary_lookup: Dict[int, float] = {}
        for j_str, rid in entry.get("singleton_route_ids", {}).items():
            j = int(j_str)
            rid = int(rid)
            f_val = float(row[rid].item())
            unary_lookup[j] = f_val
            s_qid.append(qid)
            s_idx.append(j)
            s_F.append(f_val)
            s_target.append(f_val)
        if include_pairs:
            for pair_key, rid in entry.get("pair_route_ids", {}).items():
                a_str, b_str = pair_key.split(",")
                a, b = int(a_str), int(b_str)
                if a > b:
                    a, b = b, a
                rid = int(rid)
                f_pair = float(row[rid].item())
                fa = unary_lookup.get(a)
                fb = unary_lookup.get(b)
                if fa is None or fb is None:
                    raise ValueError(
                        f"Pair ({a},{b}) for q={qid} requires unary terms but "
                        f"singleton routes for one of them were not registered."
                    )
                p_qid.append(qid)
                p_i.append(a)
                p_j.append(b)
                p_F.append(f_pair)
                p_target.append(f_pair - fa - fb)

    if n_skipped_qid_oob:
        logger.warning(
            "skipped %d questions with question_id outside dense matrix range [0,%d)",
            n_skipped_qid_oob, Q,
        )

    out = {
        "singleton_qid":    torch.tensor(s_qid, dtype=torch.long),
        "singleton_idx":    torch.tensor(s_idx, dtype=torch.long),
        "singleton_target": torch.tensor(s_target, dtype=torch.float32),
        "singleton_F":      torch.tensor(s_F, dtype=torch.float32),
        "pair_qid":    torch.tensor(p_qid, dtype=torch.long),
        "pair_i":      torch.tensor(p_i, dtype=torch.long),
        "pair_j":      torch.tensor(p_j, dtype=torch.long),
        "pair_target": torch.tensor(p_target, dtype=torch.float32),
        "pair_F":      torch.tensor(p_F, dtype=torch.float32),
        "include_pairs": include_pairs,
        "n_questions":   int(len(route_subsets["per_question"])),
    }
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_for_run(
    catalog_dir: Path,
    decode_dir: Path,
    output_dir: Path,
    *,
    benchmarks: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    catalog_dir = Path(catalog_dir)
    decode_dir = Path(decode_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bench_filter = set(benchmarks) if benchmarks else None
    candidates = sorted(p.name for p in catalog_dir.iterdir() if p.is_dir())
    if bench_filter:
        candidates = [b for b in candidates if b in bench_filter]

    summary: Dict[str, Dict[str, Any]] = {}
    for bench in candidates:
        sidecar_path = catalog_dir / bench / "route_subsets.json"
        catalog_path = catalog_dir / bench / "selected_catalog.json"
        dense_path = decode_dir / bench / "dense_deltas_matrix.pt"
        if not sidecar_path.is_file():
            logger.warning("[%s] missing sidecar %s, skip", bench, sidecar_path)
            continue
        if not dense_path.is_file():
            logger.warning("[%s] missing dense matrix %s, skip", bench, dense_path)
            continue

        route_subsets = _load_route_subsets(sidecar_path)
        delta_matrix, _payload = _load_dense_matrix(dense_path)

        payload = materialize_for_benchmark(
            route_subsets=route_subsets,
            delta_matrix=delta_matrix,
        )
        payload["benchmark"] = bench
        payload["anchor"] = list(route_subsets.get("anchor", []))
        payload["source_catalog"] = str(catalog_path)
        payload["source_dense_matrix"] = str(dense_path)
        payload["source_route_subsets"] = str(sidecar_path)

        out_path = output_dir / f"local_moebius_{bench}.pt"
        torch.save(payload, out_path)
        summary[bench] = {
            "out_path": str(out_path),
            "n_singletons": int(payload["singleton_qid"].numel()),
            "n_pairs": int(payload["pair_qid"].numel()),
            "include_pairs": bool(payload["include_pairs"]),
            "n_questions": int(payload["n_questions"]),
        }
        logger.info(
            "[%s] wrote %s  S=%d P=%d include_pairs=%s",
            bench, out_path, summary[bench]["n_singletons"],
            summary[bench]["n_pairs"], summary[bench]["include_pairs"],
        )

    (output_dir / "build_summary.json").write_text(
        json.dumps(
            {
                "catalog_dir": str(catalog_dir),
                "decode_dir": str(decode_dir),
                "benchmarks": summary,
            },
            indent=2,
        )
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog_dir", required=True, type=Path,
                   help="Output of build_local_subset_catalog (per-bench subdirs).")
    p.add_argument("--decode_dir", required=True, type=Path,
                   help="Decode root from dense_reevaluation (per-benchmark subdirs).")
    p.add_argument("--output_dir", required=True, type=Path,
                   help="Where to write local_moebius_{bench}.pt files.")
    p.add_argument("--benchmarks", nargs="*", default=None,
                   help="Restrict to a subset of benchmarks.")
    p.add_argument("--log_level", default="INFO")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    summary = build_for_run(
        args.catalog_dir,
        args.decode_dir,
        args.output_dir,
        benchmarks=args.benchmarks,
    )
    logger.info("done: %d benchmarks under %s", len(summary), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
