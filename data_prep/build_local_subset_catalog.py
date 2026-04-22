#!/usr/bin/env python3
"""Build a per-benchmark *subset* route catalog for local Möbius supervision.

Pipeline (Stage 1 of 3)
=======================

For each benchmark in a compositional artifacts manifest, this script walks
``observed/{bench}.jsonl`` and enumerates the lower-order subsets ``T \\subset S``
required to materialize local Möbius supervision targets later:

* singletons ``{o_j}`` with ``j \\in S`` (always),
* unordered pairs ``{o_i, o_j}`` with ``i < j`` and ``{i,j} \\subseteq S`` (only
  when ``--include_pairs`` is set; programs with ``length < 2`` are ignored).

The deduplicated benchmark-level union of singletons and pairs is converted to
*applied routes* via ``core.edit_dsl.apply_program(anchor, [Primitive,...])``,
which is exactly the same call already used by
``scripts/build_dense_catalog_from_legal_programs.py``.

Outputs (under ``--output_dir/{bench}/``):

* ``selected_catalog.json`` -- format consumed verbatim by
  ``dr-llm/data_prep/dense_reevaluation.py``::

      {
        "selected_routes": [[layer, ...], ...],
        "anchor":          [layer, ...],
        "benchmark":       "...",
        "n_routes":        N,
        "source_manifest": "...",
      }

* ``route_subsets.json`` -- sidecar that pins ``route_id`` to its subset spec
  and records per-question membership for the materializer::

      {
        "anchor":     [layer, ...],
        "benchmark":  "...",
        "include_pairs": bool,
        "routes": [
          {"route_id": rid, "kind": "singleton", "j": int} |
          {"route_id": rid, "kind": "pair", "i": int, "j": int},
          ...
        ],
        "per_question": [
          {"question_id": q,
           "singleton_route_ids": {"<j>": rid, ...},
           "pair_route_ids":      {"<i>,<j>": rid, ...}},
          ...
        ],
      }

Stage 2 (no Python edits anywhere): run unchanged
``dr-llm/data_prep/dense_reevaluation.py --catalog_json ...`` per benchmark.

Stage 3: ``data_prep/build_local_moebius_targets.py`` consumes the sidecar
plus dr-llm's ``dense_deltas_matrix.pt`` and writes ``local_moebius_{bench}.pt``.

Usage
-----

::

    python -m data_prep.build_local_subset_catalog \\
        --manifest fine_routing_data/<run>_compositional/manifest.json \\
        --output_dir local_subsets/<run> \\
        [--benchmarks commonsenseqa boolq] \\
        [--include_pairs]
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.edit_dsl import Primitive, apply_program  # noqa: E402

logger = logging.getLogger("build_local_subset_catalog")


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_primitives(path: Path) -> List[Primitive]:
    """Read ``primitives.jsonl`` as written by build_compositional_catalogues."""
    rows = _read_jsonl(path)
    rows.sort(key=lambda r: int(r["idx"]))
    out: List[Primitive] = []
    for r in rows:
        out.append(Primitive(kind=str(r["kind"]), args=tuple(int(x) for x in r["args"])))
    return out


def _load_legal_programs(path: Path) -> List[List[int]]:
    """Read ``legal_programs/{bench}.jsonl`` -> ``row_idx -> primitive_indices``."""
    rows = _read_jsonl(path)
    rows.sort(key=lambda r: int(r["idx"]))
    return [[int(x) for x in r["primitive_indices"]] for r in rows]


def _load_observed(path: Path) -> List[Dict[str, Any]]:
    return _read_jsonl(path)


# ---------------------------------------------------------------------------
# Subset enumeration
# ---------------------------------------------------------------------------


def enumerate_required_subsets(
    observed_rows: Sequence[Dict[str, Any]],
    legal_to_prims: Sequence[Sequence[int]],
    *,
    include_pairs: bool,
) -> Tuple[
    Dict[int, set],
    Dict[int, set],
    set,
    set,
]:
    """Enumerate per-question and benchmark-wide subset memberships.

    Returns
    -------
    q_singletons : ``qid -> set[int]`` (primitive ids ``j`` needed for ``q``)
    q_pairs      : ``qid -> set[(i, j)]`` with ``i < j``; empty when
                   ``include_pairs`` is False.
    s_union      : union of singletons over all questions (``set[int]``).
    p_union      : union of pairs over all questions (``set[(i, j)]``); empty
                   when ``include_pairs`` is False.
    """
    q_singletons: Dict[int, set] = defaultdict(set)
    q_pairs: Dict[int, set] = defaultdict(set)
    s_union: set = set()
    p_union: set = set()

    n_legal = len(legal_to_prims)
    for rec in observed_rows:
        qid = int(rec["question_id"])
        for row_idx in rec.get("obs_indices", []):
            r = int(row_idx)
            if r < 0 or r >= n_legal:
                continue
            prims = legal_to_prims[r]
            for j in prims:
                q_singletons[qid].add(int(j))
                s_union.add(int(j))
            if include_pairs and len(prims) >= 2:
                sorted_prims = sorted(int(x) for x in prims)
                for i, j in itertools.combinations(sorted_prims, 2):
                    q_pairs[qid].add((int(i), int(j)))
                    p_union.add((int(i), int(j)))
    return q_singletons, q_pairs, s_union, p_union


# ---------------------------------------------------------------------------
# Catalog construction
# ---------------------------------------------------------------------------


def build_catalog_for_benchmark(
    *,
    bench: str,
    anchor: Sequence[int],
    primitives: Sequence[Primitive],
    legal_to_prims: Sequence[Sequence[int]],
    observed_rows: Sequence[Dict[str, Any]],
    include_pairs: bool,
    source_manifest: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build (selected_catalog, route_subsets) payloads for one benchmark.

    ``selected_routes[r]`` is the applied layer sequence for the subset whose
    spec is recorded at ``route_subsets["routes"][r]``; the two arrays are
    aligned (same order, same length).
    """
    q_singletons, q_pairs, s_union, p_union = enumerate_required_subsets(
        observed_rows, legal_to_prims, include_pairs=include_pairs,
    )

    routes: List[List[int]] = []
    route_specs: List[Dict[str, Any]] = []
    singleton_rid: Dict[int, int] = {}
    pair_rid: Dict[Tuple[int, int], int] = {}

    for j in sorted(s_union):
        rid = len(routes)
        routes.append([int(x) for x in apply_program(anchor, [primitives[j]])])
        route_specs.append({"route_id": rid, "kind": "singleton", "j": int(j)})
        singleton_rid[int(j)] = rid

    if include_pairs:
        for i, j in sorted(p_union):
            rid = len(routes)
            routes.append([int(x) for x in apply_program(anchor, [primitives[i], primitives[j]])])
            route_specs.append({"route_id": rid, "kind": "pair", "i": int(i), "j": int(j)})
            pair_rid[(int(i), int(j))] = rid

    per_question: List[Dict[str, Any]] = []
    seen_qids = sorted(set(q_singletons.keys()) | set(q_pairs.keys()))
    for qid in seen_qids:
        entry: Dict[str, Any] = {
            "question_id": int(qid),
            "singleton_route_ids": {
                str(j): int(singleton_rid[j]) for j in sorted(q_singletons.get(qid, ()))
                if j in singleton_rid
            },
        }
        if include_pairs:
            entry["pair_route_ids"] = {
                f"{i},{j}": int(pair_rid[(i, j)])
                for (i, j) in sorted(q_pairs.get(qid, ()))
                if (i, j) in pair_rid
            }
        else:
            entry["pair_route_ids"] = {}
        per_question.append(entry)

    selected_catalog = {
        "selected_routes": routes,
        "anchor": [int(x) for x in anchor],
        "benchmark": bench,
        "n_routes": len(routes),
        "source_manifest": source_manifest,
    }
    route_subsets = {
        "anchor": [int(x) for x in anchor],
        "benchmark": bench,
        "include_pairs": bool(include_pairs),
        "n_singletons": len(singleton_rid),
        "n_pairs": len(pair_rid),
        "n_questions": len(per_question),
        "routes": route_specs,
        "per_question": per_question,
        "source_manifest": source_manifest,
    }
    return selected_catalog, route_subsets


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_for_manifest(
    manifest_path: Path,
    output_dir: Path,
    *,
    benchmarks: Optional[Iterable[str]] = None,
    include_pairs: bool = False,
) -> Dict[str, Dict[str, Any]]:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    base_dir = manifest_path.parent

    primitives_path = base_dir / manifest["primitives_path"]
    primitives = _load_primitives(primitives_path)
    M = len(primitives)
    logger.info("loaded %d primitives from %s", M, primitives_path)

    bench_filter = set(benchmarks) if benchmarks else None
    summary: Dict[str, Dict[str, Any]] = {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for bench, bench_meta in manifest["benchmarks"].items():
        if bench_filter and bench not in bench_filter:
            continue
        anchor = [int(x) for x in bench_meta["anchor"]]
        legal_path = base_dir / bench_meta["legal_programs_path"]
        observed_path = base_dir / bench_meta["observed_path"]
        legal_to_prims = _load_legal_programs(legal_path)
        observed_rows = _load_observed(observed_path)

        selected_catalog, route_subsets = build_catalog_for_benchmark(
            bench=bench,
            anchor=anchor,
            primitives=primitives,
            legal_to_prims=legal_to_prims,
            observed_rows=observed_rows,
            include_pairs=include_pairs,
            source_manifest=str(manifest_path),
        )

        bench_out = output_dir / bench
        bench_out.mkdir(parents=True, exist_ok=True)
        (bench_out / "selected_catalog.json").write_text(
            json.dumps(selected_catalog, indent=2)
        )
        (bench_out / "route_subsets.json").write_text(
            json.dumps(route_subsets, indent=2)
        )
        n_q_with_pair = sum(1 for e in route_subsets["per_question"] if e["pair_route_ids"])
        logger.info(
            "[%s] questions=%d singletons=%d pairs=%d routes=%d (q_with_pair=%d) -> %s",
            bench,
            route_subsets["n_questions"],
            route_subsets["n_singletons"],
            route_subsets["n_pairs"],
            len(selected_catalog["selected_routes"]),
            n_q_with_pair,
            bench_out,
        )
        summary[bench] = {
            "n_questions": route_subsets["n_questions"],
            "n_singletons": route_subsets["n_singletons"],
            "n_pairs": route_subsets["n_pairs"],
            "n_routes": len(selected_catalog["selected_routes"]),
            "selected_catalog": str(bench_out / "selected_catalog.json"),
            "route_subsets": str(bench_out / "route_subsets.json"),
        }

    (output_dir / "build_summary.json").write_text(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "include_pairs": include_pairs,
                "M": M,
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
    p.add_argument("--manifest", required=True, type=Path,
                   help="Compositional manifest.json (output of build_compositional_catalogues).")
    p.add_argument("--output_dir", required=True, type=Path,
                   help="Output root; per-benchmark subdirs will be created.")
    p.add_argument("--benchmarks", nargs="*", default=None,
                   help="Restrict to a subset of benchmarks.")
    p.add_argument("--include_pairs", action="store_true",
                   help="Also enumerate pair subsets (required for pair Mobius supervision).")
    p.add_argument("--log_level", default="INFO")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    summary = build_for_manifest(
        args.manifest,
        args.output_dir,
        benchmarks=args.benchmarks,
        include_pairs=args.include_pairs,
    )
    logger.info("done: %d benchmarks under %s", len(summary), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
