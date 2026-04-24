#!/usr/bin/env python3
"""Build a *delta* ``selected_catalog.json`` for incremental assign mining.

Motivation
==========

The structural compositional pipeline (kinds = ``skip|repeat|swap``) has
already produced a dense ``delta_matrix [Q, R_struct]`` over ``R_struct``
unique layer routes (e.g. CSQA: 259 routes). Re-canonicalizing MCTS data
with ``--include_assign`` and rebuilding the compositional manifest yields a
much larger legal-program catalogue (e.g. CSQA: ~1.4-2.0k programs); after
applying each new legal program to the anchor, many of the resulting layer
routes coincide with structural routes that we have already mined.

This script produces a ``selected_catalog.json`` containing **only the routes
that the assign-extended catalogue introduces** (i.e. routes not present in
the structural dense matrix). The output is consumed verbatim by
``dr-llm/data_prep/dense_reevaluation.py``, so the increment is mined with
the same prefix-trie hidden-state caching, the same model loader, and the
same per-question scoring. Routes already covered by the existing dense
matrix are skipped, so we pay LM cost only for genuinely new routes (which
includes pure-assign programs **and** mixed programs of the form
``[struct, assign]`` whose applied route is not in the struct catalogue).

A sidecar ``route_provenance.jsonl`` records, for each delta route, the set
of assign-extended legal programs that produce it together with kind
histograms (``has_assign``, ``has_struct``, ``is_pure_assign``,
``is_mixed``), to enable downstream attribution.

Pipeline location
=================

Stage A.0 (existing) -- struct-only dense matrix already on disk.
Stage A.1 (THIS SCRIPT) -- enumerate delta routes, write delta
    ``selected_catalog.json``.
Stage A.2 (existing) -- run ``dr-llm/data_prep/dense_reevaluation.py
    --catalog_json <delta>/selected_catalog.json ...`` on the delta routes.
Stage A.3 (NEW) -- ``data_prep/merge_dense_increment.py`` concatenates the
    struct dense matrix with the delta dense matrix and reorders columns
    into the assign-extended manifest's legal-program order, producing a
    unified ``dense_deltas_matrix.pt`` + ``dense_deltas.jsonl``.

CLI example
===========

::

    python -m data_prep.build_assign_increment_catalog \\
        --new_compositional_dir fine_routing_data/<run>_compositional_assign \\
        --bench commonsenseqa \\
        --existing_dense_matrix /.../dense_artifacts_csqa_ft_2026/decode_compositional/dense_deltas_matrix.pt \\
        --output_dir /.../dense_artifacts_csqa_ft_2026/catalog_assign_increment

Outputs (under ``--output_dir``)
================================

* ``selected_catalog.json`` -- ``{"selected_routes": [[layer,...],...],
  "anchor": [...], "benchmark": "...", "n_routes": N_delta,
  "source_manifest": "...", "increment": true}``.
* ``route_provenance.jsonl`` -- one row per delta route::

      {"delta_route_id": int,           # column index into the new dense matrix
       "route": [layer, ...],
       "legal_program_indices": [...],  # rows in new manifest's legal_programs/{bench}.jsonl
       "n_legal_programs": int,
       "has_assign": bool,
       "has_struct": bool,
       "is_pure_assign": bool,
       "is_mixed": bool,
       "kinds_histogram": {"assign": int, ...}}

* ``summary.json`` -- counts and provenance.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.edit_dsl import Primitive, apply_program  # noqa: E402

logger = logging.getLogger("build_assign_increment_catalog")

RouteKey = Tuple[int, ...]


# ---------------------------------------------------------------------------
# Existing-route loaders (struct dense matrix or selected_catalog.json)
# ---------------------------------------------------------------------------


def _load_existing_routes_from_pt(path: Path) -> List[RouteKey]:
    import torch  # local import to keep CLI startup light
    payload = torch.load(path, map_location="cpu", weights_only=False)
    routes = payload.get("routes")
    if routes is None:
        raise KeyError(f"{path}: missing 'routes' field; not a dense_deltas_matrix.pt")
    return [tuple(int(x) for x in r) for r in routes]


def _load_existing_routes_from_selected(path: Path) -> List[RouteKey]:
    payload = json.loads(Path(path).read_text())
    return [tuple(int(x) for x in r) for r in payload["selected_routes"]]


def load_existing_routes(
    *,
    existing_dense_matrix: Optional[Path],
    existing_selected_catalog: Optional[Path],
) -> List[RouteKey]:
    if existing_dense_matrix is not None:
        return _load_existing_routes_from_pt(existing_dense_matrix)
    if existing_selected_catalog is not None:
        return _load_existing_routes_from_selected(existing_selected_catalog)
    raise ValueError(
        "Provide one of --existing_dense_matrix or --existing_selected_catalog"
    )


# ---------------------------------------------------------------------------
# New (assign-extended) compositional manifest readers
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
    rows = _read_jsonl(path)
    rows.sort(key=lambda r: int(r["idx"]))
    return [Primitive(kind=str(r["kind"]), args=tuple(int(x) for x in r["args"])) for r in rows]


def _load_legal_programs(path: Path) -> List[List[int]]:
    rows = _read_jsonl(path)
    rows.sort(key=lambda r: int(r["idx"]))
    return [[int(x) for x in r["primitive_indices"]] for r in rows]


# ---------------------------------------------------------------------------
# Increment construction
# ---------------------------------------------------------------------------


def build_increment(
    *,
    new_compositional_dir: Path,
    bench: str,
    existing_routes: Sequence[RouteKey],
) -> Tuple[
    List[RouteKey],          # delta routes (column order)
    List[Dict[str, Any]],    # provenance rows
    List[int],               # number of new legal programs covered by old set
    Dict[str, Any],          # bench-level stats
]:
    new_compositional_dir = Path(new_compositional_dir)
    manifest = json.loads((new_compositional_dir / "manifest.json").read_text())
    if bench not in manifest["benchmarks"]:
        raise KeyError(f"benchmark {bench!r} not in manifest {new_compositional_dir}")
    bench_meta = manifest["benchmarks"][bench]
    anchor = [int(x) for x in bench_meta["anchor"]]
    primitives = _load_primitives(new_compositional_dir / manifest["primitives_path"])
    legal = _load_legal_programs(new_compositional_dir / bench_meta["legal_programs_path"])

    existing_set: Set[RouteKey] = set(existing_routes)
    logger.info(
        "[%s] new manifest: %d primitives, %d legal programs; existing struct routes: %d",
        bench, len(primitives), len(legal), len(existing_set),
    )

    # For every legal program in the new manifest, apply to anchor and bucket
    # by route. Then split routes by membership in existing_set.
    route_to_program_ids: Dict[RouteKey, List[int]] = {}
    for pidx, prim_indices in enumerate(legal):
        prog = [primitives[j] for j in prim_indices]
        route = tuple(int(x) for x in apply_program(anchor, prog))
        route_to_program_ids.setdefault(route, []).append(pidx)

    n_new_routes_total = len(route_to_program_ids)
    n_routes_already_have = sum(1 for r in route_to_program_ids if r in existing_set)
    n_routes_delta = n_new_routes_total - n_routes_already_have
    logger.info(
        "[%s] manifest covers %d unique routes; %d already mined, %d are new",
        bench, n_new_routes_total, n_routes_already_have, n_routes_delta,
    )

    delta_routes: List[RouteKey] = []
    provenance: List[Dict[str, Any]] = []
    n_progs_already_covered = 0
    n_progs_delta = 0

    # Deterministic ordering: sort delta routes by their first-encountered
    # legal-program index, then by route tuple as tiebreak. This keeps the
    # column layout stable across runs.
    def _route_sort_key(r: RouteKey) -> Tuple[int, RouteKey]:
        first_pid = min(route_to_program_ids[r])
        return (first_pid, r)

    delta_route_iter = sorted(
        (r for r in route_to_program_ids if r not in existing_set),
        key=_route_sort_key,
    )

    for delta_rid, route in enumerate(delta_route_iter):
        pids = sorted(route_to_program_ids[route])
        kinds_hist: Counter = Counter()
        for pid in pids:
            for j in legal[pid]:
                kinds_hist[primitives[j].kind] += 1
        # A "program" can have multiple primitives of various kinds; we
        # aggregate kinds_hist over *primitive occurrences* across all legal
        # programs producing this route.
        # For has_assign / has_struct / pure / mixed, we look at the per-program
        # composition: a route is "achievable purely with assign" iff at least
        # one producing program has only-assign primitives; "achievable mixed"
        # iff at least one producing program has both struct and assign; etc.
        # For attribution we expose per-program kinds:
        per_prog_kinds: List[List[str]] = []
        any_pure_assign = False
        any_pure_struct = False
        any_mixed = False
        for pid in pids:
            kinds = [primitives[j].kind for j in legal[pid]]
            per_prog_kinds.append(kinds)
            has_a = any(k == "assign" for k in kinds)
            has_s = any(k != "assign" for k in kinds)
            if has_a and has_s:
                any_mixed = True
            elif has_a and not has_s:
                any_pure_assign = True
            elif has_s and not has_a:
                any_pure_struct = True
        provenance.append(
            {
                "delta_route_id": delta_rid,
                "route": list(route),
                "legal_program_indices": pids,
                "n_legal_programs": len(pids),
                "any_pure_assign_program": any_pure_assign,
                "any_pure_struct_program": any_pure_struct,  # should be False; sanity
                "any_mixed_program": any_mixed,
                "kinds_histogram": dict(kinds_hist),
            }
        )
        delta_routes.append(route)
        n_progs_delta += len(pids)

    for route, pids in route_to_program_ids.items():
        if route in existing_set:
            n_progs_already_covered += len(pids)

    stats = {
        "benchmark": bench,
        "anchor_length": len(anchor),
        "n_primitives_new_manifest": len(primitives),
        "n_legal_programs_new_manifest": len(legal),
        "n_unique_routes_new_manifest": n_new_routes_total,
        "n_routes_already_mined": n_routes_already_have,
        "n_routes_delta": n_routes_delta,
        "n_legal_programs_covered_by_existing": n_progs_already_covered,
        "n_legal_programs_in_delta": n_progs_delta,
        "kinds_histogram_in_delta_primitives": _delta_kind_histogram(provenance),
        "delta_route_decomposition": _decomposition_summary(provenance),
    }
    return delta_routes, provenance, [], stats  # third slot reserved for future use


def _delta_kind_histogram(provenance: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    out: Counter = Counter()
    for row in provenance:
        for k, v in row["kinds_histogram"].items():
            out[k] += int(v)
    return dict(out)


def _decomposition_summary(provenance: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    n_pure_assign_only = 0  # only producing programs are pure-assign
    n_mixed_only = 0        # only producing programs are mixed
    n_either = 0            # at least one pure-assign and at least one mixed
    n_struct_only = 0       # bug guard: should be 0 by definition (delta only)
    for row in provenance:
        pa = bool(row.get("any_pure_assign_program"))
        mx = bool(row.get("any_mixed_program"))
        ps = bool(row.get("any_pure_struct_program"))
        if ps and not (pa or mx):
            n_struct_only += 1
        elif pa and mx:
            n_either += 1
        elif pa:
            n_pure_assign_only += 1
        elif mx:
            n_mixed_only += 1
    return {
        "delta_routes_pure_assign_only": n_pure_assign_only,
        "delta_routes_mixed_only": n_mixed_only,
        "delta_routes_with_both_pure_and_mixed": n_either,
        "delta_routes_struct_only_unexpected": n_struct_only,
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_outputs(
    *,
    bench: str,
    new_compositional_dir: Path,
    output_dir: Path,
    delta_routes: Sequence[RouteKey],
    provenance: Sequence[Dict[str, Any]],
    stats: Dict[str, Any],
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    new_manifest_path = (Path(new_compositional_dir) / "manifest.json").resolve()
    bench_meta = json.loads(new_manifest_path.read_text())["benchmarks"][bench]
    anchor = [int(x) for x in bench_meta["anchor"]]

    selected_catalog = {
        "selected_routes": [list(r) for r in delta_routes],
        "anchor": anchor,
        "benchmark": bench,
        "n_routes": len(delta_routes),
        "source_manifest": str(new_manifest_path),
        "increment": True,
    }
    sel_path = output_dir / "selected_catalog.json"
    sel_path.write_text(json.dumps(selected_catalog, indent=2))

    prov_path = output_dir / "route_provenance.jsonl"
    with open(prov_path, "w") as f:
        for row in provenance:
            f.write(json.dumps(row) + "\n")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(stats, indent=2))
    return {
        "selected_catalog": str(sel_path),
        "route_provenance": str(prov_path),
        "summary": str(summary_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--new_compositional_dir", required=True, type=Path,
        help="Compositional manifest dir built with --include_assign (and "
             "kept_kinds containing 'assign').",
    )
    p.add_argument("--bench", required=True, type=str)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--existing_dense_matrix", type=Path, default=None,
        help="Path to the struct-only dense_deltas_matrix.pt (uses its 'routes' "
             "field as the set of routes already mined).",
    )
    src.add_argument(
        "--existing_selected_catalog", type=Path, default=None,
        help="Alternative: path to the struct-only selected_catalog.json (uses "
             "its 'selected_routes' field).",
    )
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--log_level", default="INFO")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    existing_routes = load_existing_routes(
        existing_dense_matrix=args.existing_dense_matrix,
        existing_selected_catalog=args.existing_selected_catalog,
    )
    delta_routes, provenance, _, stats = build_increment(
        new_compositional_dir=args.new_compositional_dir,
        bench=args.bench,
        existing_routes=existing_routes,
    )
    paths = write_outputs(
        bench=args.bench,
        new_compositional_dir=args.new_compositional_dir,
        output_dir=args.output_dir,
        delta_routes=delta_routes,
        provenance=provenance,
        stats=stats,
    )
    logger.info(
        "[%s] delta routes=%d (pure_assign_only=%d, mixed_only=%d, both=%d) -> %s",
        args.bench,
        stats["n_routes_delta"],
        stats["delta_route_decomposition"]["delta_routes_pure_assign_only"],
        stats["delta_route_decomposition"]["delta_routes_mixed_only"],
        stats["delta_route_decomposition"]["delta_routes_with_both_pure_and_mixed"],
        paths["selected_catalog"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
