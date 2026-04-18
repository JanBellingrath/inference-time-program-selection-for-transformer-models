"""Support tables for canonical edit-program datasets.

Reads a directory produced by ``data_prep.canonicalize_programs`` (i.e. a
collection of ``{benchmark}.jsonl`` files whose rows carry a ``programs``
list of canonical edit programs) and emits two JSONL tables (spec 1.11):

* ``primitive_support.jsonl`` -- one row per primitive instance, with
  ``count`` (total occurrences), ``n_questions`` (distinct
  ``(benchmark_id, question_hash)`` pairs in which the primitive appears in
  some canonical program), and ``n_benchmarks`` (distinct ``benchmark_id``
  values).

* ``pair_support.jsonl`` -- one row per *unordered* pair of distinct
  primitives co-occurring within the same canonical program, with the same
  three count fields.

Programs of length 0 (the no-op anchor program) are ignored for support
counting: they trivially participate in every row and would dominate the
tables.

Usage::

    python -m data_prep.program_support --data_dir <canonical_dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.edit_dsl import Primitive, canonical_key_str, prim_key  # noqa: E402

logger = logging.getLogger("program_support")


def _primitive_from_dict(d: Dict[str, Any]) -> Primitive:
    return Primitive(str(d["kind"]), tuple(int(x) for x in d["args"]))


def _key_for_primitive(p: Primitive) -> str:
    return canonical_key_str((p,))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class _Stats:
    __slots__ = ("count", "questions", "benchmarks")

    def __init__(self) -> None:
        self.count: int = 0
        self.questions: set = set()
        self.benchmarks: set = set()

    def observe(self, *, question_key: Tuple[Any, Any], benchmark: Any) -> None:
        self.count += 1
        self.questions.add(question_key)
        self.benchmarks.add(benchmark)


def aggregate_support(
    data_dir: Path,
    *,
    benchmarks: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, Tuple[Primitive, _Stats]], Dict[Tuple[str, str], Tuple[Tuple[Primitive, Primitive], _Stats]]]:
    """Walk every ``{benchmark}.jsonl`` row and tally primitive / pair support."""
    bench_filter = set(benchmarks) if benchmarks else None
    unary: Dict[str, Tuple[Primitive, _Stats]] = {}
    pair: Dict[Tuple[str, str], Tuple[Tuple[Primitive, Primitive], _Stats]] = {}

    jsonl_files = sorted(p for p in data_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning("no *.jsonl files in %s", data_dir)

    for jsonl_path in jsonl_files:
        bench = jsonl_path.stem
        if bench_filter and bench not in bench_filter:
            continue
        with open(jsonl_path) as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                bid = row.get("benchmark_id", bench)
                qkey = (bid, row.get("question_hash") or row.get("question_id"))
                for entry in row.get("programs", []):
                    prog_dicts = entry.get("program") or []
                    if not prog_dicts:
                        continue
                    primitives = [_primitive_from_dict(d) for d in prog_dicts]
                    primitives.sort(key=prim_key)

                    seen_keys = []
                    for p in primitives:
                        k = _key_for_primitive(p)
                        if k not in unary:
                            unary[k] = (p, _Stats())
                        unary[k][1].observe(question_key=qkey, benchmark=bid)
                        seen_keys.append(k)

                    for ka, kb in combinations(sorted(set(seen_keys)), 2):
                        pa = unary[ka][0]
                        pb = unary[kb][0]
                        pkey = (ka, kb)
                        if pkey not in pair:
                            pair[pkey] = ((pa, pb), _Stats())
                        pair[pkey][1].observe(question_key=qkey, benchmark=bid)
    return unary, pair


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _write_unary(path: Path, unary: Dict[str, Tuple[Primitive, _Stats]]) -> None:
    keys = sorted(unary.keys(), key=lambda k: prim_key(unary[k][0]))
    with open(path, "w") as f:
        for k in keys:
            p, s = unary[k]
            row = {
                "key": k,
                "kind": p.kind,
                "args": list(p.args),
                "count": s.count,
                "n_questions": len(s.questions),
                "n_benchmarks": len(s.benchmarks),
            }
            f.write(json.dumps(row) + "\n")


def _write_pair(path: Path, pair: Dict[Tuple[str, str], Tuple[Tuple[Primitive, Primitive], _Stats]]) -> None:
    keys = sorted(pair.keys(), key=lambda k: (prim_key(pair[k][0][0]), prim_key(pair[k][0][1])))
    with open(path, "w") as f:
        for k in keys:
            (pa, pb), s = pair[k]
            row = {
                "keys": [k[0], k[1]],
                "primitives": [
                    {"kind": pa.kind, "args": list(pa.args)},
                    {"kind": pb.kind, "args": list(pb.args)},
                ],
                "count": s.count,
                "n_questions": len(s.questions),
                "n_benchmarks": len(s.benchmarks),
            }
            f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def write_support_tables(
    data_dir: Path,
    *,
    benchmarks: Optional[Iterable[str]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir) if output_dir else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    unary, pair = aggregate_support(data_dir, benchmarks=benchmarks)
    _write_unary(output_dir / "primitive_support.jsonl", unary)
    _write_pair(output_dir / "pair_support.jsonl", pair)

    summary = {
        "n_primitives": len(unary),
        "n_pairs": len(pair),
        "total_primitive_occurrences": sum(s.count for _, s in unary.values()),
        "total_pair_occurrences": sum(s.count for _, s in pair.values()),
    }
    with open(output_dir / "support_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("wrote support tables: %s", summary)
    return summary


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True, type=Path,
                   help="Canonicalized fine-routing data directory.")
    p.add_argument("--output_dir", type=Path, default=None,
                   help="Where to write support tables (default: --data_dir).")
    p.add_argument("--benchmarks", nargs="*", default=None)
    p.add_argument("--log_level", default="INFO")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    if not args.data_dir.is_dir():
        logger.error("not a directory: %s", args.data_dir)
        return 2
    write_support_tables(
        args.data_dir,
        benchmarks=args.benchmarks,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
