"""Length-2 primitive pair-program graph for compositional generalization.

This is **distinct** from :mod:`data_prep.program_support` ``pair_support``,
which counts unordered primitive pairs that *co-occur inside the same
program of any length*.  Here we want edges defined by canonical programs
that are themselves of *length exactly 2*: ``e = {o_i, o_j}``.

The graph is built over the filtered primitive catalogue
``O_train`` (``primitives.jsonl`` from
:mod:`data_prep.build_compositional_catalogues`) and the canonical
``{benchmark}.jsonl`` files referenced by the catalogue manifest.

Outputs (under ``--output_dir``)
--------------------------------
* ``pair_program_graph.json`` -- structured artifact with::

      {
        "schema_version": 1,
        "catalogue_dir": "...",
        "primitives": [
          {"idx": j, "key": "...", "kind": "...", "args": [...],
           "n_singleton": int, "n_pair_total": int,
           "n_questions_singleton": int, "n_questions_any": int,
           "n_questions_pair": int, "deg": int}
        ],
        "edges": [
          {"a": idx_a, "b": idx_b,
           "key_a": "...", "key_b": "...",
           "count": int, "n_questions": int,
           "mean_delta": float, "sum_delta": float,
           "per_benchmark": {bench: {"count": int, "n_questions": int,
                                       "mean_delta": float}}
          }
        ],
        "summary": {"n_primitives": int, "n_edges": int,
                     "n_questions_total": int, "n_benchmarks": int}
      }

``deg(o)`` is the number of *distinct* partner primitives ``o'`` such that
``(o, o')`` is an empirical length-2 edge with ``count >= 1`` (i.e. the
unfiltered pair-graph degree).  ``holdout_edge_split`` recomputes more
restrictive degrees as needed.

Usage::

    python -m data_prep.build_pair_program_graph \\
        --catalogue_dir compositional_runs/csqa_compositional \\
        --output_dir compositional_runs/csqa_compositional/pair_graph
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.edit_dsl import (  # noqa: E402
    Primitive,
    canonical_key_str,
    program_from_dicts,
)

logger = logging.getLogger("build_pair_program_graph")

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Loading helpers
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


def _load_primitive_index(primitives_path: Path) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    rows = _read_jsonl(primitives_path)
    rows.sort(key=lambda r: int(r["idx"]))
    key_to_idx: Dict[str, int] = {}
    primitives: List[Dict[str, Any]] = []
    for row in rows:
        idx = int(row["idx"])
        key = str(row["key"])
        if key in key_to_idx:
            raise ValueError(f"duplicate primitive key in primitives.jsonl: {key}")
        key_to_idx[key] = idx
        primitives.append({
            "idx": idx,
            "key": key,
            "kind": str(row["kind"]),
            "args": list(row.get("args", [])),
        })
    return key_to_idx, primitives


def _resolve_program_to_keys(entry: Dict[str, Any]) -> Optional[List[str]]:
    """Return canonical primitive keys for an observed program entry.

    Prefers reconstruction from the ``program`` dict-list so the result is
    independent of the joined ``program_key`` formatting.
    """
    prog_dicts = entry.get("program")
    if isinstance(prog_dicts, list):
        try:
            prog = program_from_dicts(prog_dicts)
        except (KeyError, ValueError, TypeError):
            return None
        return [canonical_key_str((p,)) for p in prog]
    key = entry.get("program_key")
    if isinstance(key, str):
        if key == "noop":
            return []
        return key.split("+")
    return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class _PrimNodeStats:
    __slots__ = (
        "n_singleton", "n_pair_total",
        "questions_singleton", "questions_pair", "questions_any",
        "partners",
    )

    def __init__(self) -> None:
        self.n_singleton: int = 0
        self.n_pair_total: int = 0
        self.questions_singleton: set = set()
        self.questions_pair: set = set()
        self.questions_any: set = set()
        self.partners: set = set()


class _EdgeStats:
    __slots__ = ("count", "questions", "sum_delta",
                 "per_bench_count", "per_bench_questions", "per_bench_sum_delta")

    def __init__(self) -> None:
        self.count: int = 0
        self.questions: set = set()
        self.sum_delta: float = 0.0
        self.per_bench_count: Dict[str, int] = defaultdict(int)
        self.per_bench_questions: Dict[str, set] = defaultdict(set)
        self.per_bench_sum_delta: Dict[str, float] = defaultdict(float)


def aggregate_pair_program_graph(
    canonical_paths: Dict[str, Path],
    key_to_idx: Dict[str, int],
) -> Tuple[Dict[int, _PrimNodeStats], Dict[Tuple[int, int], _EdgeStats], int]:
    """Walk canonical JSONL files; tally length-1 primitives and length-2 edges.

    Programs whose primitives are not all in ``O_train`` are skipped (they
    cannot become legal rows for the catalogue / router).
    """
    nodes: Dict[int, _PrimNodeStats] = defaultdict(_PrimNodeStats)
    edges: Dict[Tuple[int, int], _EdgeStats] = defaultdict(_EdgeStats)
    n_questions = 0

    for bench, jsonl_path in canonical_paths.items():
        if not jsonl_path.is_file():
            logger.warning("[%s] missing canonical JSONL: %s", bench, jsonl_path)
            continue
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_questions += 1
                qkey = (
                    row.get("benchmark_id", bench),
                    row.get("question_hash") or row.get("question_id"),
                )
                for entry in row.get("programs", []):
                    keys = _resolve_program_to_keys(entry)
                    if keys is None:
                        continue
                    if len(keys) == 0 or len(keys) > 2:
                        continue
                    try:
                        idxs = sorted({key_to_idx[k] for k in keys})
                    except KeyError:
                        continue
                    if len(keys) == 1:
                        i = idxs[0]
                        node = nodes[i]
                        node.n_singleton += 1
                        node.questions_singleton.add(qkey)
                        node.questions_any.add(qkey)
                    elif len(keys) == 2:
                        if len(idxs) != 2:
                            continue
                        a, b = idxs
                        delta = float(entry.get("delta") or 0.0)
                        edge = edges[(a, b)]
                        edge.count += 1
                        edge.questions.add(qkey)
                        edge.sum_delta += delta
                        edge.per_bench_count[bench] += 1
                        edge.per_bench_questions[bench].add(qkey)
                        edge.per_bench_sum_delta[bench] += delta
                        for j in (a, b):
                            node = nodes[j]
                            node.n_pair_total += 1
                            node.questions_pair.add(qkey)
                            node.questions_any.add(qkey)
                            node.partners.add(b if j == a else a)
    return nodes, edges, n_questions


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_pair_program_graph(
    catalogue_dir: Path,
    output_dir: Path,
    *,
    benchmarks: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    catalogue_dir = Path(catalogue_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = catalogue_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing catalogue manifest: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)

    primitives_path = catalogue_dir / manifest.get("primitives_path", "primitives.jsonl")
    key_to_idx, prim_meta = _load_primitive_index(primitives_path)

    bench_filter = set(benchmarks) if benchmarks else None
    canonical_paths: Dict[str, Path] = {}
    for bench, info in manifest.get("benchmarks", {}).items():
        if bench_filter and bench not in bench_filter:
            continue
        src = info.get("source_jsonl")
        if not src:
            logger.warning("[%s] manifest entry missing source_jsonl", bench)
            continue
        canonical_paths[bench] = Path(src)

    if not canonical_paths:
        raise ValueError("no benchmarks selected; check --benchmarks vs manifest.")

    t0 = time.time()
    nodes, edges, n_questions = aggregate_pair_program_graph(
        canonical_paths, key_to_idx,
    )

    primitive_records: List[Dict[str, Any]] = []
    for meta in prim_meta:
        node = nodes.get(meta["idx"], _PrimNodeStats())
        primitive_records.append({
            "idx": meta["idx"],
            "key": meta["key"],
            "kind": meta["kind"],
            "args": meta["args"],
            "n_singleton": int(node.n_singleton),
            "n_pair_total": int(node.n_pair_total),
            "n_questions_singleton": int(len(node.questions_singleton)),
            "n_questions_pair": int(len(node.questions_pair)),
            "n_questions_any": int(len(node.questions_any)),
            "deg": int(len(node.partners)),
        })

    edge_records: List[Dict[str, Any]] = []
    for (a, b) in sorted(edges.keys()):
        e = edges[(a, b)]
        per_bench: Dict[str, Dict[str, Any]] = {}
        for bench, c in e.per_bench_count.items():
            qs = e.per_bench_questions[bench]
            sd = e.per_bench_sum_delta[bench]
            per_bench[bench] = {
                "count": int(c),
                "n_questions": int(len(qs)),
                "mean_delta": (sd / c) if c > 0 else 0.0,
            }
        n_q = len(e.questions)
        edge_records.append({
            "a": int(a),
            "b": int(b),
            "key_a": prim_meta[a]["key"],
            "key_b": prim_meta[b]["key"],
            "count": int(e.count),
            "n_questions": int(n_q),
            "sum_delta": float(e.sum_delta),
            "mean_delta": (e.sum_delta / e.count) if e.count > 0 else 0.0,
            "per_benchmark": per_bench,
        })

    summary = {
        "n_primitives": len(primitive_records),
        "n_edges": len(edge_records),
        "n_questions_total": n_questions,
        "n_benchmarks": len(canonical_paths),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "catalogue_dir": str(catalogue_dir),
        "benchmarks": sorted(canonical_paths.keys()),
        "primitives": primitive_records,
        "edges": edge_records,
        "summary": summary,
    }
    out_path = output_dir / "pair_program_graph.json"
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    logger.info(
        "wrote pair-program graph: %d primitives, %d edges, %d questions -> %s",
        summary["n_primitives"], summary["n_edges"], summary["n_questions_total"], out_path,
    )
    return artifact


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalogue_dir", required=True, type=Path,
                   help="Directory produced by build_compositional_catalogues.")
    p.add_argument("--output_dir", type=Path, default=None,
                   help="Where to write the graph artifact (default: <catalogue_dir>/pair_graph).")
    p.add_argument("--benchmarks", nargs="*", default=None)
    p.add_argument("--log_level", default="INFO")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    if not args.catalogue_dir.is_dir():
        logger.error("not a directory: %s", args.catalogue_dir)
        return 2
    output_dir = args.output_dir or (args.catalogue_dir / "pair_graph")
    build_pair_program_graph(
        args.catalogue_dir,
        output_dir,
        benchmarks=args.benchmarks,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
