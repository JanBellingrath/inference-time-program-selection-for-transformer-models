#!/usr/bin/env python3
"""Precompute artifacts for the first-order compositional router (Step 2).

Inputs
------
* A canonical fine-routing directory produced by
  :mod:`data_prep.canonicalize_programs` (each ``{benchmark}.jsonl`` row
  carries an ``anchor_sequence`` plus a ``programs`` list of canonical
  programs with ``program_key`` and ``delta``).
* A primitive support table produced by :mod:`data_prep.program_support`
  (``primitive_support.jsonl``); rows that meet the configured thresholds
  define the filtered primitive catalogue ``O_train``.

Outputs (written under ``--output_dir``)
----------------------------------------
* ``primitives.jsonl``         -- one row per primitive in ``O_train``.
* ``legal_programs/{b}.jsonl`` -- one row per legal program for benchmark
  ``b``: ``{idx, length, primitive_indices, key}`` (empty program first).
* ``incidence/{b}.pt``         -- torch dict with sparse COO ``A`` and
  dense ``lengths`` for benchmark ``b``.
* ``observed/{b}.jsonl``       -- per-question observed-candidate records.
* ``manifest.json``            -- geometry, filter thresholds, anchors,
  per-benchmark counts and paths.

Usage::

    python -m data_prep.build_compositional_catalogues \\
        --data_dir fine_routing_data/<run>_canonical \\
        --output_dir fine_routing_data/<run>_compositional \\
        --min_count 2 --min_questions 2
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

from core.edit_dsl import (
    KIND_RANK,
    Primitive,
    Program,
    canonical_key_str,
    enumerate_admissible_programs,
    prim_key,
    program_from_dicts,
)

logger = logging.getLogger("build_compositional_catalogues")


DEFAULT_PRIMITIVE_KINDS = ("skip", "repeat", "swap")


# ---------------------------------------------------------------------------
# Primitive catalogue O_train
# ---------------------------------------------------------------------------


def _read_primitive_support(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def filter_primitive_catalogue(
    support_rows: Sequence[Dict[str, Any]],
    *,
    min_count: int,
    min_questions: int,
    min_benchmarks: int,
    keep_kinds: Sequence[str],
) -> List[Tuple[Primitive, Dict[str, Any]]]:
    """Apply the spec's filtering thresholds to ``primitive_support.jsonl`` rows.

    Returns surviving primitives in canonical ``prim_key`` order, each paired
    with its raw support row so downstream artifacts can record the stats.
    """
    keep_kinds_set = set(keep_kinds)
    survivors: List[Tuple[Primitive, Dict[str, Any]]] = []
    for row in support_rows:
        kind = str(row.get("kind"))
        if kind not in keep_kinds_set:
            continue
        if int(row.get("count", 0)) < min_count:
            continue
        if int(row.get("n_questions", 0)) < min_questions:
            continue
        if int(row.get("n_benchmarks", 0)) < min_benchmarks:
            continue
        try:
            prim = Primitive(kind, tuple(int(x) for x in row.get("args", [])))
        except (TypeError, ValueError) as exc:
            logger.warning("skip malformed support row %r: %s", row, exc)
            continue
        survivors.append((prim, dict(row)))
    survivors.sort(key=lambda pr: prim_key(pr[0]))
    return survivors


def _write_primitives_jsonl(path: Path, survivors: Sequence[Tuple[Primitive, Dict[str, Any]]]) -> None:
    with open(path, "w") as f:
        for idx, (prim, row) in enumerate(survivors):
            entry = {
                "idx": idx,
                "kind": prim.kind,
                "args": list(prim.args),
                "key": canonical_key_str((prim,)),
                "support": {
                    "count": int(row.get("count", 0)),
                    "n_questions": int(row.get("n_questions", 0)),
                    "n_benchmarks": int(row.get("n_benchmarks", 0)),
                },
            }
            f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Anchor and geometry resolution
# ---------------------------------------------------------------------------


def _load_anchor_from_jsonl(jsonl_path: Path) -> Optional[List[int]]:
    if not jsonl_path.is_file():
        return None
    with open(jsonl_path) as f:
        first = f.readline()
    if not first.strip():
        return None
    rec = json.loads(first)
    seq = rec.get("anchor_sequence")
    if seq is None:
        return None
    return [int(x) for x in seq]


def _load_geometry(data_dir: Path) -> Dict[str, Any]:
    cfg_path = data_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing {cfg_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)
    canon = cfg.get("canonicalization") or {}
    K = int(canon.get("max_program_len", cfg.get("max_local_edits", 2)))
    R = int(canon.get("swap_radius", cfg.get("swap_radius", 2)))
    S = int(canon.get("editable_start", cfg.get("editable_start", 0)))
    include_assign = bool(canon.get("include_assign", False))
    dedupe_assign_with_struct = bool(canon.get("dedupe_assign_with_struct", False))
    return {
        "K": K,
        "swap_radius": R,
        "editable_start": S,
        "include_assign": include_assign,
        "dedupe_assign_with_struct": dedupe_assign_with_struct,
    }


# ---------------------------------------------------------------------------
# Per-anchor legal-program enumeration and incidence matrix
# ---------------------------------------------------------------------------


def _editable_indices(anchor_len: int, editable_start: int) -> Tuple[int, ...]:
    return tuple(range(max(0, editable_start), anchor_len))


def build_legal_programs(
    anchor: Sequence[int],
    *,
    geometry: Dict[str, Any],
    primitive_key_to_idx: Dict[str, int],
) -> Tuple[List[Program], List[List[int]], int]:
    """Enumerate ``E_legal`` for one anchor and project onto ``O_train``.

    Returns
    -------
    programs : list of canonical programs (empty program first).
    primitive_indices : ``primitive_indices[r]`` = sorted indices into
        ``O_train`` of the primitives constituting program ``r``.
    n_dropped : count of admissible programs filtered out because they
        contained at least one primitive outside ``O_train``.
    """
    editable = _editable_indices(len(anchor), geometry["editable_start"])
    programs: List[Program] = []
    indices: List[List[int]] = []
    n_dropped = 0
    for prog in enumerate_admissible_programs(
        anchor,
        K=geometry["K"],
        editable_indices=editable,
        swap_radius=geometry["swap_radius"],
        include_assign=geometry["include_assign"],
        dedupe_assign_with_struct=geometry["dedupe_assign_with_struct"],
    ):
        try:
            row_idx = sorted(primitive_key_to_idx[canonical_key_str((p,))] for p in prog)
        except KeyError:
            n_dropped += 1
            continue
        programs.append(prog)
        indices.append(row_idx)
    return programs, indices, n_dropped


def build_incidence_tensor(
    primitive_indices: Sequence[Sequence[int]],
    M: int,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """Build a sparse ``A`` (COO) and dense ``lengths`` vector."""
    rows: List[int] = []
    cols: List[int] = []
    lengths: List[int] = []
    for r, prims in enumerate(primitive_indices):
        lengths.append(len(prims))
        for j in prims:
            rows.append(r)
            cols.append(int(j))
    if rows:
        a_indices = torch.tensor([rows, cols], dtype=torch.long)
        a_values = torch.ones(len(rows), dtype=torch.float32)
    else:
        a_indices = torch.zeros((2, 0), dtype=torch.long)
        a_values = torch.zeros(0, dtype=torch.float32)
    a_shape = (len(primitive_indices), M)
    lengths_t = torch.tensor(lengths, dtype=torch.float32)
    return a_indices, a_values, a_shape, lengths_t  # type: ignore[return-value]


def build_pair_incidence_tensor(
    primitive_indices: Sequence[Sequence[int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """Enumerate every unordered primitive pair appearing in any legal program.

    Returns
    -------
    pair_index : ``[P, 2]`` long tensor of ``(i, j)`` with ``i < j``.
    B_indices  : ``[2, nnz]`` COO indices (rows = program rows, cols = pair ids).
    B_values   : ``[nnz]`` float ones.
    B_shape    : ``(N, P)``.
    """
    pair_to_id: Dict[Tuple[int, int], int] = {}
    pair_list: List[Tuple[int, int]] = []
    rows_list: List[List[int]] = []
    for r, prims in enumerate(primitive_indices):
        if len(prims) < 2:
            rows_list.append([])
            continue
        sorted_prims = sorted(int(x) for x in prims)
        these_pairs: List[int] = []
        for ai in range(len(sorted_prims)):
            for bi in range(ai + 1, len(sorted_prims)):
                key = (sorted_prims[ai], sorted_prims[bi])
                pid = pair_to_id.get(key)
                if pid is None:
                    pid = len(pair_list)
                    pair_to_id[key] = pid
                    pair_list.append(key)
                these_pairs.append(pid)
        rows_list.append(these_pairs)

    P = len(pair_list)
    rows: List[int] = []
    cols: List[int] = []
    for r, pids in enumerate(rows_list):
        for pid in pids:
            rows.append(r)
            cols.append(pid)

    if rows:
        b_indices = torch.tensor([rows, cols], dtype=torch.long)
        b_values = torch.ones(len(rows), dtype=torch.float32)
    else:
        b_indices = torch.zeros((2, 0), dtype=torch.long)
        b_values = torch.zeros(0, dtype=torch.float32)
    if pair_list:
        pair_index = torch.tensor(pair_list, dtype=torch.long)
    else:
        pair_index = torch.zeros((0, 2), dtype=torch.long)
    b_shape = (len(primitive_indices), P)
    return pair_index, b_indices, b_values, b_shape  # type: ignore[return-value]


def _write_legal_programs(path: Path, programs: Sequence[Program], indices: Sequence[Sequence[int]]) -> None:
    with open(path, "w") as f:
        for r, (prog, prim_idxs) in enumerate(zip(programs, indices)):
            entry = {
                "idx": r,
                "length": len(prog),
                "primitive_indices": list(prim_idxs),
                "key": canonical_key_str(prog),
            }
            f.write(json.dumps(entry) + "\n")


def _save_incidence(path: Path, a_indices: torch.Tensor, a_values: torch.Tensor,
                    a_shape: Tuple[int, int], lengths: torch.Tensor) -> None:
    torch.save(
        {
            "A_indices": a_indices,
            "A_values": a_values,
            "A_shape": list(a_shape),
            "lengths": lengths,
        },
        path,
    )


def _save_pair_incidence(
    path: Path,
    pair_index: torch.Tensor,
    b_indices: torch.Tensor,
    b_values: torch.Tensor,
    b_shape: Tuple[int, int],
) -> None:
    torch.save(
        {
            "pair_index": pair_index,
            "B_indices": b_indices,
            "B_values": b_values,
            "B_shape": list(b_shape),
        },
        path,
    )


# ---------------------------------------------------------------------------
# Per-question observed candidate records
# ---------------------------------------------------------------------------


def _row_for_observed_program(
    entry: Dict[str, Any],
    key_to_row: Dict[str, int],
) -> Optional[int]:
    """Resolve a canonical-program entry to its global legal-program row.

    Prefers ``program_key``; reconstructs from ``program`` (list of dicts)
    when the key is absent or unknown.
    """
    key = entry.get("program_key")
    if isinstance(key, str) and key in key_to_row:
        return key_to_row[key]
    prog_dicts = entry.get("program")
    if not isinstance(prog_dicts, list):
        return None
    try:
        prog = program_from_dicts(prog_dicts)
    except (KeyError, ValueError, TypeError):
        return None
    return key_to_row.get(canonical_key_str(prog))


def collect_observed_for_benchmark(
    jsonl_path: Path,
    key_to_row: Dict[str, int],
    n_legal: int,
) -> Tuple[List[Dict[str, Any]], Counter]:
    """Walk a benchmark JSONL and emit one observed record per question.

    The ``residual_idx`` field equals the JSONL line index, which matches the
    row order in ``{benchmark}_pivot_residuals.pt``.
    """
    out: List[Dict[str, Any]] = []
    stats: Counter = Counter()
    with open(jsonl_path) as f:
        for residual_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["malformed_rows"] += 1
                continue
            stats["rows_in"] += 1
            # Multiple canonicalisations (different MCTS playouts; struct +
            # assign sources) can map to the *same* canonical program row
            # with different per-playout ``delta`` readouts. The previous
            # ``max(prev, delta)`` reduction was an upward-biased estimator
            # of the true ``u(q, route) − u(q, anchor)`` and made "lucky"
            # playouts dominate the CE-on-observed target. We accumulate
            # ``(sum, count)`` and emit the sample mean, an unbiased
            # estimator consistent with dense re-evaluation.
            row_to_delta_sum: Dict[int, float] = {}
            row_to_delta_count: Dict[int, int] = {}
            n_dropped = 0
            for entry in row.get("programs", []):
                row_idx = _row_for_observed_program(entry, key_to_row)
                if row_idx is None:
                    n_dropped += 1
                    continue
                if row_idx < 0 or row_idx >= n_legal:
                    n_dropped += 1
                    continue
                delta = float(entry.get("delta") or 0.0)
                row_to_delta_sum[row_idx] = row_to_delta_sum.get(row_idx, 0.0) + delta
                row_to_delta_count[row_idx] = row_to_delta_count.get(row_idx, 0) + 1
            if not row_to_delta_sum:
                stats["rows_without_observations"] += 1
                continue
            obs_pairs = sorted(
                (r, row_to_delta_sum[r] / max(row_to_delta_count[r], 1))
                for r in row_to_delta_sum
            )
            obs_indices = [r for r, _ in obs_pairs]
            obs_deltas = [d for _, d in obs_pairs]
            stats["rows_kept"] += 1
            stats["observed_pairs"] += len(obs_indices)
            stats["dropped_program_entries"] += n_dropped
            out.append(
                {
                    "residual_idx": residual_idx,
                    "question_id": row.get("question_id"),
                    "question_hash": row.get("question_hash"),
                    "n_obs": len(obs_indices),
                    "obs_indices": obs_indices,
                    "obs_deltas": obs_deltas,
                    "dropped_program_entries": n_dropped,
                }
            )
    return out, stats


def _write_observed(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_catalogues(
    data_dir: Path,
    output_dir: Path,
    *,
    support_path: Optional[Path] = None,
    benchmarks: Optional[Iterable[str]] = None,
    min_count: int = 1,
    min_questions: int = 1,
    min_benchmarks: int = 1,
    keep_kinds: Sequence[str] = DEFAULT_PRIMITIVE_KINDS,
    geometry_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    (output_dir / "legal_programs").mkdir(parents=True, exist_ok=True)
    (output_dir / "incidence").mkdir(parents=True, exist_ok=True)
    (output_dir / "pair_incidence").mkdir(parents=True, exist_ok=True)
    (output_dir / "observed").mkdir(parents=True, exist_ok=True)

    geometry = _load_geometry(data_dir)
    if geometry_overrides:
        geometry.update({k: v for k, v in geometry_overrides.items() if v is not None})

    if support_path is None:
        support_path = data_dir / "primitive_support.jsonl"
    if not support_path.is_file():
        raise FileNotFoundError(
            f"missing primitive support table: {support_path} "
            "(run data_prep.program_support first)"
        )

    survivors = filter_primitive_catalogue(
        _read_primitive_support(support_path),
        min_count=min_count,
        min_questions=min_questions,
        min_benchmarks=min_benchmarks,
        keep_kinds=keep_kinds,
    )
    if not survivors:
        raise ValueError("no primitives survived filtering; relax thresholds.")
    M = len(survivors)
    primitive_key_to_idx: Dict[str, int] = {
        canonical_key_str((prim,)): idx for idx, (prim, _row) in enumerate(survivors)
    }
    _write_primitives_jsonl(output_dir / "primitives.jsonl", survivors)
    logger.info("|O_train| = %d (after filtering)", M)

    bench_filter = set(benchmarks) if benchmarks else None
    jsonl_files = sorted(p for p in data_dir.glob("*.jsonl") if not p.name.startswith("_"))
    jsonl_files = [p for p in jsonl_files if p.name not in {"primitive_support.jsonl", "pair_support.jsonl"}]

    manifest_benchmarks: Dict[str, Any] = {}
    overall_stats: Counter = Counter()

    for jsonl_path in jsonl_files:
        bench = jsonl_path.stem
        if bench_filter and bench not in bench_filter:
            continue
        anchor = _load_anchor_from_jsonl(jsonl_path)
        if anchor is None:
            logger.warning("[%s] no anchor in first record; skipping", bench)
            continue
        t0 = time.time()
        programs, primitive_indices, n_dropped = build_legal_programs(
            anchor, geometry=geometry, primitive_key_to_idx=primitive_key_to_idx,
        )
        N = len(programs)
        a_idx, a_val, a_shape, lengths = build_incidence_tensor(primitive_indices, M)
        pair_index, b_idx, b_val, b_shape = build_pair_incidence_tensor(primitive_indices)
        key_to_row: Dict[str, int] = {
            canonical_key_str(prog): r for r, prog in enumerate(programs)
        }
        _write_legal_programs(output_dir / "legal_programs" / f"{bench}.jsonl",
                              programs, primitive_indices)
        _save_incidence(output_dir / "incidence" / f"{bench}.pt",
                        a_idx, a_val, a_shape, lengths)
        _save_pair_incidence(output_dir / "pair_incidence" / f"{bench}.pt",
                             pair_index, b_idx, b_val, b_shape)

        observed, stats = collect_observed_for_benchmark(jsonl_path, key_to_row, N)
        _write_observed(output_dir / "observed" / f"{bench}.jsonl", observed)

        residuals_path = data_dir / f"{bench}_pivot_residuals.pt"
        full_residuals_path = data_dir / f"{bench}_full_residuals.pt"

        bench_record = {
            "anchor": anchor,
            "anchor_length": len(anchor),
            "n_legal_programs": N,
            "n_legal_dropped_unknown_primitive": n_dropped,
            "incidence_path": str((output_dir / "incidence" / f"{bench}.pt").relative_to(output_dir)),
            "pair_incidence_path": str((output_dir / "pair_incidence" / f"{bench}.pt").relative_to(output_dir)),
            "n_legal_pairs": int(pair_index.shape[0]),
            "legal_programs_path": str((output_dir / "legal_programs" / f"{bench}.jsonl").relative_to(output_dir)),
            "observed_path": str((output_dir / "observed" / f"{bench}.jsonl").relative_to(output_dir)),
            "source_jsonl": str(jsonl_path),
            "pivot_residuals_path": str(residuals_path) if residuals_path.is_file() else None,
            "full_residuals_path": str(full_residuals_path) if full_residuals_path.is_file() else None,
            "n_questions_kept": int(stats.get("rows_kept", 0)),
            "n_questions_dropped_no_obs": int(stats.get("rows_without_observations", 0)),
            "n_observed_pairs": int(stats.get("observed_pairs", 0)),
            "n_dropped_program_entries": int(stats.get("dropped_program_entries", 0)),
            "elapsed_sec": round(time.time() - t0, 3),
        }
        manifest_benchmarks[bench] = bench_record
        overall_stats.update(stats)
        overall_stats["n_legal_programs_total"] += N
        overall_stats["n_legal_dropped_unknown_primitive_total"] += n_dropped
        logger.info(
            "[%s] N=%d P=%d (dropped %d) questions kept=%d (no_obs=%d) pairs=%d  (%.2fs)",
            bench, N, int(pair_index.shape[0]), n_dropped,
            bench_record["n_questions_kept"],
            bench_record["n_questions_dropped_no_obs"],
            bench_record["n_observed_pairs"],
            bench_record["elapsed_sec"],
        )

    manifest = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "geometry": geometry,
        "filter": {
            "min_count": min_count,
            "min_questions": min_questions,
            "min_benchmarks": min_benchmarks,
            "keep_kinds": list(keep_kinds),
            "support_path": str(support_path),
        },
        "M": M,
        "primitives_path": "primitives.jsonl",
        "benchmarks": manifest_benchmarks,
        "overall_stats": dict(overall_stats),
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("manifest written: %s", output_dir / "manifest.json")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True, type=Path,
                   help="Canonical fine-routing directory (output of canonicalize_programs).")
    p.add_argument("--output_dir", required=True, type=Path,
                   help="Where to write compositional artifacts.")
    p.add_argument("--support_path", type=Path, default=None,
                   help="primitive_support.jsonl path (default: <data_dir>/primitive_support.jsonl).")
    p.add_argument("--benchmarks", nargs="*", default=None,
                   help="Restrict to a subset of benchmarks.")
    p.add_argument("--min_count", type=int, default=1)
    p.add_argument("--min_questions", type=int, default=1)
    p.add_argument("--min_benchmarks", type=int, default=1)
    p.add_argument("--keep_kinds", nargs="*", default=list(DEFAULT_PRIMITIVE_KINDS),
                   choices=sorted(KIND_RANK.keys()))
    p.add_argument("--max_program_len", type=int, default=None,
                   help="Override K from canonical config.")
    p.add_argument("--swap_radius", type=int, default=None,
                   help="Override swap_radius from canonical config.")
    p.add_argument("--editable_start", type=int, default=None,
                   help="Override editable_start from canonical config.")
    p.add_argument("--log_level", default="INFO")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    overrides = {
        "K": args.max_program_len,
        "swap_radius": args.swap_radius,
        "editable_start": args.editable_start,
    }
    build_catalogues(
        args.data_dir,
        args.output_dir,
        support_path=args.support_path,
        benchmarks=args.benchmarks,
        min_count=args.min_count,
        min_questions=args.min_questions,
        min_benchmarks=args.min_benchmarks,
        keep_kinds=tuple(args.keep_kinds),
        geometry_overrides=overrides,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
