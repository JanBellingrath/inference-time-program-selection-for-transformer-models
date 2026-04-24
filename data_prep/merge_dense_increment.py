#!/usr/bin/env python3
"""Merge struct-only and assign-increment dense outputs into a unified set.

Inputs
======

* ``--old_dense_dir``: directory containing the struct-only dense outputs
  produced by ``dr-llm/data_prep/dense_reevaluation.py``::

      old_dense_dir/
        dense_deltas_matrix.pt   # {'delta_matrix [Q,R_old]', 'anchor_utilities [Q]',
                                 #   'routes [R_old]', 'benchmarks', 'score_mode'}
        dense_deltas.jsonl       # one row per question; route_utilities/route_deltas
                                 #   keyed by stringified column index in 'routes'

* ``--new_dense_dir``: directory containing the assign-increment dense outputs
  produced by re-running ``dense_reevaluation.py`` against the
  ``selected_catalog.json`` written by
  :mod:`data_prep.build_assign_increment_catalog`. Same schema, with
  ``R_new`` rows. Note ``R_new`` may be < the increment catalog size if the
  upstream trie stripped no-op routes.

* ``--new_compositional_dir``: the assign-extended compositional manifest dir
  (used to determine the canonical column ordering for the unified matrix:
  one column per unique route in the manifest's ``legal_programs/{bench}.jsonl``
  applied to the anchor).

Outputs (under ``--output_dir``)
================================

* ``dense_deltas_matrix.pt`` -- unified payload, columns ordered to match the
  new compositional manifest's unique-route order. Shape ``[Q, R_unified]``.
  ``routes`` field aligns with this ordering.
* ``dense_deltas.jsonl`` -- one row per question, with merged
  ``route_utilities`` and ``route_deltas`` dicts (keys are the new column
  indices as strings). The ``mcts_source`` block is preserved from the old
  dense rows when present.
* ``selected_catalog.json`` -- ``{selected_routes, anchor, benchmark,
  n_routes, source_manifest, unified: true}`` matching the column ordering.
* ``merge_summary.json`` -- counts, validation, and column-provenance stats.

Validation
==========

* ``score_mode`` and benchmark must agree across old/new.
* ``anchor_utilities`` must agree per question (within ``--anchor_atol``).
* Every unique route induced by the new manifest must appear in either the
  old or the new dense matrix; otherwise the merger aborts (use
  ``--allow_missing_routes`` to relax for partial sweeps).

CLI example
===========

::

    python -m data_prep.merge_dense_increment \\
        --old_dense_dir /.../dense_artifacts_csqa_ft_2026/decode_compositional \\
        --new_dense_dir /.../dense_artifacts_csqa_ft_2026/decode_assign_increment \\
        --new_compositional_dir /.../fine_routing_data_..._compositional_assign \\
        --bench commonsenseqa \\
        --output_dir /.../dense_artifacts_csqa_ft_2026/decode_compositional_unified
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.edit_dsl import Primitive, apply_program  # noqa: E402

logger = logging.getLogger("merge_dense_increment")

RouteKey = Tuple[int, ...]


# ---------------------------------------------------------------------------
# IO helpers
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


def _load_dense_matrix(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("delta_matrix", "anchor_utilities", "routes"):
        if k not in payload:
            raise KeyError(f"{path}: missing field {k!r}")
    payload["routes"] = [tuple(int(x) for x in r) for r in payload["routes"]]
    return payload


def _load_compositional_anchor_and_routes(
    new_compositional_dir: Path, bench: str,
) -> Tuple[List[int], List[RouteKey], List[Dict[str, Any]]]:
    """Return (anchor, unique_routes_in_legal_program_order, primitives_list).

    ``unique_routes_in_legal_program_order`` is the deduplicated route list,
    ordered by the *first* legal program (lowest idx) that produces each
    route. This matches what ``build_dense_catalog_from_legal_programs.py``
    does, so the unified column layout is identical to a direct dense run on
    the new manifest's full catalogue.
    """
    new_compositional_dir = Path(new_compositional_dir)
    manifest = json.loads((new_compositional_dir / "manifest.json").read_text())
    bench_meta = manifest["benchmarks"][bench]
    anchor = [int(x) for x in bench_meta["anchor"]]
    prim_rows = _read_jsonl(new_compositional_dir / manifest["primitives_path"])
    prim_rows.sort(key=lambda r: int(r["idx"]))
    primitives = [Primitive(str(r["kind"]), tuple(int(x) for x in r["args"])) for r in prim_rows]
    legal_rows = _read_jsonl(new_compositional_dir / bench_meta["legal_programs_path"])
    legal_rows.sort(key=lambda r: int(r["idx"]))
    seen: Dict[RouteKey, int] = {}
    ordered: List[RouteKey] = []
    for r in legal_rows:
        prog = [primitives[int(j)] for j in r["primitive_indices"]]
        route = tuple(int(x) for x in apply_program(anchor, prog))
        if route in seen:
            continue
        seen[route] = len(ordered)
        ordered.append(route)
    return anchor, ordered, prim_rows


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def merge_matrices(
    *,
    old_payload: Dict[str, Any],
    new_payload: Dict[str, Any],
    unified_routes: Sequence[RouteKey],
    anchor_atol: float = 1e-4,
    allow_missing_routes: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    """Concatenate old + new dense matrices and reorder columns to
    ``unified_routes``.

    Returns
    -------
    delta_matrix : ``[Q, R_unified]`` float tensor.
    anchor_utilities : ``[Q]`` float tensor (taken from old payload, validated
        against new where overlapping).
    column_provenance : list of length ``R_unified``, each entry::
            {"route": [...], "source": "old"|"new"|"both",
             "old_col": int|None, "new_col": int|None}
    stats : dict
    """
    old_score = old_payload.get("score_mode")
    new_score = new_payload.get("score_mode")
    if old_score != new_score:
        raise ValueError(
            f"score_mode mismatch: old={old_score!r} new={new_score!r}; "
            "merge would mix incompatible supervision."
        )
    old_benches = list(old_payload.get("benchmarks", []))
    new_benches = list(new_payload.get("benchmarks", []))
    if old_benches != new_benches:
        raise ValueError(
            f"benchmarks mismatch: old={old_benches} new={new_benches}"
        )

    old_dm: torch.Tensor = old_payload["delta_matrix"].float()
    new_dm: torch.Tensor = new_payload["delta_matrix"].float()
    old_au: torch.Tensor = old_payload["anchor_utilities"].float()
    new_au: torch.Tensor = new_payload["anchor_utilities"].float()
    Q_old, R_old = old_dm.shape
    Q_new, R_new = new_dm.shape

    Q = min(Q_old, Q_new) if Q_new > 0 else Q_old
    if Q_old != Q_new:
        logger.warning(
            "Question count mismatch: old=%d new=%d; using min=%d for merge",
            Q_old, Q_new, Q,
        )
    if Q_new > 0:
        au_diff = (old_au[:Q] - new_au[:Q]).abs().max().item()
        if au_diff > anchor_atol:
            raise ValueError(
                f"anchor_utilities disagree by {au_diff:.4g} > atol {anchor_atol}; "
                "old/new were mined against different anchors or models."
            )

    old_route_to_col: Dict[RouteKey, int] = {r: i for i, r in enumerate(old_payload["routes"])}
    new_route_to_col: Dict[RouteKey, int] = {r: i for i, r in enumerate(new_payload["routes"])}

    R_uni = len(unified_routes)
    delta_uni = torch.zeros((Q, R_uni), dtype=torch.float32)
    column_provenance: List[Dict[str, Any]] = []
    n_from_old = n_from_new = n_both = 0
    missing: List[int] = []
    for c, route in enumerate(unified_routes):
        in_old = route in old_route_to_col
        in_new = route in new_route_to_col
        old_col = old_route_to_col.get(route)
        new_col = new_route_to_col.get(route)
        if in_old and in_new:
            # Take old by default (it was the canonical reference); validate
            # that old/new agree to within a small tolerance for sanity.
            v_old = old_dm[:Q, old_col]
            v_new = new_dm[:Q, new_col]
            diff = (v_old - v_new).abs().max().item()
            if diff > 1e-3:
                logger.warning(
                    "route %s: old/new disagree by %.4g (col %d/%d); using old.",
                    route[:6] + ("...",) if len(route) > 6 else route,
                    diff, old_col, new_col,
                )
            delta_uni[:, c] = v_old
            n_both += 1
            src = "both"
        elif in_old:
            delta_uni[:, c] = old_dm[:Q, old_col]
            n_from_old += 1
            src = "old"
        elif in_new:
            delta_uni[:, c] = new_dm[:Q, new_col]
            n_from_new += 1
            src = "new"
        else:
            missing.append(c)
            src = "missing"
        column_provenance.append(
            {
                "route": list(route),
                "source": src,
                "old_col": old_col,
                "new_col": new_col,
            }
        )
    if missing and not allow_missing_routes:
        raise ValueError(
            f"{len(missing)} unified routes are absent from both old and new "
            "dense matrices (e.g. col indices "
            f"{missing[:5]}...); pass --allow_missing_routes to leave them as 0."
        )

    stats = {
        "Q": int(Q),
        "R_old": int(R_old),
        "R_new": int(R_new),
        "R_unified": int(R_uni),
        "columns_from_old_only": int(n_from_old),
        "columns_from_new_only": int(n_from_new),
        "columns_in_both": int(n_both),
        "columns_missing": int(len(missing)),
        "score_mode": old_score,
        "benchmarks": old_benches,
    }
    return delta_uni, old_au[:Q].contiguous(), column_provenance, stats


# ---------------------------------------------------------------------------
# Per-question (dense_deltas.jsonl) merging
# ---------------------------------------------------------------------------


def merge_jsonl(
    *,
    old_jsonl: Path,
    new_jsonl: Path,
    column_provenance: Sequence[Dict[str, Any]],
    Q: int,
) -> List[Dict[str, Any]]:
    """Build the unified per-question rows.

    For each unified column ``c`` we know the old-col / new-col indices, so we
    re-key per-question dicts ``{<old_col>: value}`` -> ``{<c>: value}``. We
    prefer old when both exist (matching the matrix decision).
    """
    old_rows = _read_jsonl(old_jsonl)
    new_rows = _read_jsonl(new_jsonl)
    old_by_qid = {int(r["question_id"]): r for r in old_rows}
    new_by_qid = {int(r["question_id"]): r for r in new_rows}

    # Build per-source remap:  src_col -> unified_col
    old_to_uni: Dict[int, int] = {}
    new_to_uni: Dict[int, int] = {}
    for c, prov in enumerate(column_provenance):
        if prov["old_col"] is not None and prov["source"] in ("old", "both"):
            old_to_uni[int(prov["old_col"])] = c
        if prov["new_col"] is not None and prov["source"] == "new":
            new_to_uni[int(prov["new_col"])] = c

    out: List[Dict[str, Any]] = []
    for qid in range(Q):
        old_rec = old_by_qid.get(qid)
        new_rec = new_by_qid.get(qid)
        if old_rec is None and new_rec is None:
            continue
        base = dict(old_rec or new_rec)  # carry common fields like benchmark_id, score_mode, mcts_source
        ru: Dict[str, float] = {}
        rd: Dict[str, float] = {}
        if old_rec is not None:
            for k, v in (old_rec.get("route_utilities") or {}).items():
                c = old_to_uni.get(int(k))
                if c is not None:
                    ru[str(c)] = float(v)
            for k, v in (old_rec.get("route_deltas") or {}).items():
                c = old_to_uni.get(int(k))
                if c is not None:
                    rd[str(c)] = float(v)
        if new_rec is not None:
            for k, v in (new_rec.get("route_utilities") or {}).items():
                c = new_to_uni.get(int(k))
                if c is not None:
                    ru[str(c)] = float(v)
            for k, v in (new_rec.get("route_deltas") or {}).items():
                c = new_to_uni.get(int(k))
                if c is not None:
                    rd[str(c)] = float(v)
        base["route_utilities"] = ru
        base["route_deltas"] = rd
        # Prefer old anchor_utility when present, else new
        if old_rec is not None:
            base["anchor_utility"] = old_rec.get("anchor_utility")
        out.append(base)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_merge(
    *,
    old_dense_dir: Path,
    new_dense_dir: Path,
    new_compositional_dir: Path,
    bench: str,
    output_dir: Path,
    anchor_atol: float = 1e-4,
    allow_missing_routes: bool = False,
) -> Dict[str, Any]:
    old_dense_dir = Path(old_dense_dir)
    new_dense_dir = Path(new_dense_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    old_pt = old_dense_dir / "dense_deltas_matrix.pt"
    new_pt = new_dense_dir / "dense_deltas_matrix.pt"
    old_jsonl = old_dense_dir / "dense_deltas.jsonl"
    new_jsonl = new_dense_dir / "dense_deltas.jsonl"
    for pth in (old_pt, new_pt, old_jsonl, new_jsonl):
        if not pth.is_file():
            raise FileNotFoundError(f"missing required input: {pth}")

    old_payload = _load_dense_matrix(old_pt)
    new_payload = _load_dense_matrix(new_pt)
    anchor, unified_routes, _ = _load_compositional_anchor_and_routes(
        new_compositional_dir, bench
    )
    logger.info(
        "[%s] old R=%d, new R=%d, unified R=%d (from new manifest legal programs)",
        bench, old_payload["delta_matrix"].shape[1],
        new_payload["delta_matrix"].shape[1], len(unified_routes),
    )

    delta_uni, anchor_uni, column_provenance, mat_stats = merge_matrices(
        old_payload=old_payload,
        new_payload=new_payload,
        unified_routes=unified_routes,
        anchor_atol=anchor_atol,
        allow_missing_routes=allow_missing_routes,
    )
    Q = int(delta_uni.shape[0])
    rows = merge_jsonl(
        old_jsonl=old_jsonl,
        new_jsonl=new_jsonl,
        column_provenance=column_provenance,
        Q=Q,
    )

    unified_payload = {
        "delta_matrix": delta_uni,
        "anchor_utilities": anchor_uni,
        "routes": [list(r) for r in unified_routes],
        "benchmarks": old_payload.get("benchmarks", [bench]),
        "score_mode": old_payload.get("score_mode"),
        "merged_from": {
            "old_dense_dir": str(old_dense_dir),
            "new_dense_dir": str(new_dense_dir),
            "new_compositional_dir": str(new_compositional_dir),
            "bench": bench,
        },
    }
    out_pt = output_dir / "dense_deltas_matrix.pt"
    torch.save(unified_payload, out_pt)

    out_jsonl = output_dir / "dense_deltas.jsonl"
    with open(out_jsonl, "w") as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")

    new_manifest_path = (Path(new_compositional_dir) / "manifest.json").resolve()
    sel = {
        "selected_routes": [list(r) for r in unified_routes],
        "anchor": list(anchor),
        "benchmark": bench,
        "n_routes": len(unified_routes),
        "source_manifest": str(new_manifest_path),
        "unified": True,
    }
    out_sel = output_dir / "selected_catalog.json"
    out_sel.write_text(json.dumps(sel, indent=2))

    summary = {
        "bench": bench,
        "old_dense_dir": str(old_dense_dir),
        "new_dense_dir": str(new_dense_dir),
        "new_compositional_dir": str(new_compositional_dir),
        "outputs": {
            "matrix": str(out_pt),
            "jsonl": str(out_jsonl),
            "selected_catalog": str(out_sel),
        },
        "matrix_stats": mat_stats,
        "n_questions_in_jsonl": len(rows),
    }
    summary_path = output_dir / "merge_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("merge done: %s", summary_path)
    logger.info("matrix stats: %s", json.dumps(mat_stats))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--old_dense_dir", required=True, type=Path)
    p.add_argument("--new_dense_dir", required=True, type=Path)
    p.add_argument("--new_compositional_dir", required=True, type=Path)
    p.add_argument("--bench", required=True, type=str)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--anchor_atol", type=float, default=1e-4)
    p.add_argument(
        "--allow_missing_routes", action="store_true",
        help="If set, unified columns absent from both old and new are left "
             "as zeros instead of aborting.",
    )
    p.add_argument("--log_level", default="INFO")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    run_merge(
        old_dense_dir=args.old_dense_dir,
        new_dense_dir=args.new_dense_dir,
        new_compositional_dir=args.new_compositional_dir,
        bench=args.bench,
        output_dir=args.output_dir,
        anchor_atol=args.anchor_atol,
        allow_missing_routes=args.allow_missing_routes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
