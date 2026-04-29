"""Support tables for canonical edit-program datasets.

Reads a directory produced by ``data_prep.canonicalize_programs`` (i.e. a
collection of ``{benchmark}.jsonl`` files whose rows carry a ``programs``
list of canonical edit programs) and emits two JSONL tables (spec 1.11):

* ``primitive_support.jsonl`` -- one row per primitive instance, with
  ``count`` (total program-entry occurrences by default, or with
  ``--count-unique-per-question`` at most one per question row per
  primitive), ``n_questions`` (distinct ``(benchmark_id, question_hash)``
  pairs in which the primitive appears in some canonical program), and
  ``n_benchmarks`` (distinct ``benchmark_id`` values). When
  ``--count-unique-per-question`` is set, ``raw_count`` records total
  program-entry occurrences (diagnostic; duplicates MCTS playouts).

* ``pair_support.jsonl`` -- one row per *unordered* pair of distinct
  primitives co-occurring within the same canonical program, with the same
  three count fields.

Programs of length 0 (the no-op anchor program) are ignored for support
counting: they trivially participate in every row and would dominate the
tables.

Usage::

    python -m data_prep.program_support --data_dir <canonical_dir>
        [--count-unique-per-question]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.edit_dsl import Primitive, canonical_key_str, prim_key  # noqa: E402
from data_prep.common.io import iter_jsonl  # noqa: E402
from data_prep.common.manifests import write_manifest  # noqa: E402
from data_prep.common.validation import validate_no_duplicate_keys  # noqa: E402

logger = logging.getLogger("program_support")


def _primitive_from_dict(d: Dict[str, Any]) -> Primitive:
    return Primitive(str(d["kind"]), tuple(int(x) for x in d["args"]))


def _key_for_primitive(p: Primitive) -> str:
    return canonical_key_str((p,))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class _Stats:
    __slots__ = ("count", "raw_count", "questions", "benchmarks", "_per_row_count")

    def __init__(self, *, per_row_count: bool = False) -> None:
        self.count: int = 0
        self.raw_count: int = 0
        self.questions: set = set()
        self.benchmarks: set = set()
        self._per_row_count: bool = per_row_count

    def observe_program_entry(self, *, question_key: Tuple[Any, Any], benchmark: Any) -> None:
        """Count one program line (MCTS / exhaustive entry) containing this unit."""
        if self._per_row_count:
            self.raw_count += 1
        else:
            self.count += 1
        self.questions.add(question_key)
        self.benchmarks.add(benchmark)

    def observe_row_inclusion(
        self, *, question_key: Tuple[Any, Any], benchmark: Any
    ) -> None:
        """At most once per data row: primitive or pair present in that row (any program)."""
        if not self._per_row_count:
            return
        self.count += 1
        self.questions.add(question_key)
        self.benchmarks.add(benchmark)


def _get_unary(
    unary: Dict[str, Tuple[Primitive, _Stats]],
    k: str,
    p: Primitive,
    *,
    per_row_count: bool,
) -> _Stats:
    if k not in unary:
        unary[k] = (p, _Stats(per_row_count=per_row_count))
    return unary[k][1]


def _get_pair(
    pair: Dict[Tuple[str, str], Tuple[Tuple[Primitive, Primitive], _Stats]],
    pkey: Tuple[str, str],
    pa: Primitive,
    pb: Primitive,
    *,
    per_row_count: bool,
) -> _Stats:
    if pkey not in pair:
        pair[pkey] = ((pa, pb), _Stats(per_row_count=per_row_count))
    return pair[pkey][1]


def aggregate_support(
    data_dir: Path,
    *,
    benchmarks: Optional[Iterable[str]] = None,
    count_unique_per_question: bool = False,
) -> Tuple[Dict[str, Tuple[Primitive, _Stats]], Dict[Tuple[str, str], Tuple[Tuple[Primitive, Primitive], _Stats]]]:
    """Walk every ``{benchmark}.jsonl`` row and tally primitive / pair support."""
    bench_filter = set(benchmarks) if benchmarks else None
    unary: Dict[str, Tuple[Primitive, _Stats]] = {}
    pair: Dict[Tuple[str, str], Tuple[Tuple[Primitive, Primitive], _Stats]] = {}

    jsonl_files = sorted(p for p in data_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning("no *.jsonl files in %s", data_dir)

    per_row = count_unique_per_question

    for jsonl_path in jsonl_files:
        bench = jsonl_path.stem
        if bench_filter and bench not in bench_filter:
            continue
        for row in iter_jsonl(jsonl_path):
            bid = row.get("benchmark_id", bench)
            qkey = (bid, row.get("question_hash") or row.get("question_id"))
            programs = row.get("programs", [])

            if not per_row:
                for entry in programs:
                    prog_dicts = entry.get("program") or []
                    if not prog_dicts:
                        continue
                    primitives = [_primitive_from_dict(d) for d in prog_dicts]
                    primitives.sort(key=prim_key)

                    seen_keys = []
                    for p in primitives:
                        k = _key_for_primitive(p)
                        st = _get_unary(unary, k, p, per_row_count=False)
                        st.observe_program_entry(question_key=qkey, benchmark=bid)
                        seen_keys.append(k)

                    for ka, kb in combinations(sorted(set(seen_keys)), 2):
                        pa = unary[ka][0]
                        pb = unary[kb][0]
                        pkey2 = (ka, kb)
                        pst = _get_pair(pair, pkey2, pa, pb, per_row_count=False)
                        pst.observe_program_entry(question_key=qkey, benchmark=bid)
                continue

            row_prim_keys: set = set()
            row_pair_keys: set = set()
            for entry in programs:
                prog_dicts = entry.get("program") or []
                if not prog_dicts:
                    continue
                primitives = [_primitive_from_dict(d) for d in prog_dicts]
                primitives.sort(key=prim_key)

                seen_keys = []
                for p in primitives:
                    k = _key_for_primitive(p)
                    st = _get_unary(unary, k, p, per_row_count=True)
                    st.observe_program_entry(question_key=qkey, benchmark=bid)
                    seen_keys.append(k)
                    row_prim_keys.add(k)

                for ka, kb in combinations(sorted(set(seen_keys)), 2):
                    pa = unary[ka][0]
                    pb = unary[kb][0]
                    pkey2 = (ka, kb)
                    pst = _get_pair(pair, pkey2, pa, pb, per_row_count=True)
                    pst.observe_program_entry(question_key=qkey, benchmark=bid)
                    row_pair_keys.add(pkey2)

            for k in row_prim_keys:
                p, st = unary[k]
                st.observe_row_inclusion(question_key=qkey, benchmark=bid)

            for pkey2 in row_pair_keys:
                _, st = pair[pkey2]
                st.observe_row_inclusion(question_key=qkey, benchmark=bid)
    return unary, pair


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _write_unary(
    path: Path, unary: Dict[str, Tuple[Primitive, _Stats]], *, emit_raw_count: bool
) -> None:
    keys = sorted(unary.keys(), key=lambda k: prim_key(unary[k][0]))
    validate_no_duplicate_keys(keys, name="primitive support keys")
    with open(path, "w") as f:
        for k in keys:
            p, s = unary[k]
            row: Dict[str, Any] = {
                "key": k,
                "kind": p.kind,
                "args": list(p.args),
                "count": s.count,
                "n_questions": len(s.questions),
                "n_benchmarks": len(s.benchmarks),
            }
            if emit_raw_count:
                row["raw_count"] = s.raw_count
            f.write(json.dumps(row) + "\n")


def _write_pair(
    path: Path,
    pair: Dict[Tuple[str, str], Tuple[Tuple[Primitive, Primitive], _Stats]],
    *,
    emit_raw_count: bool,
) -> None:
    keys = sorted(pair.keys(), key=lambda k: (prim_key(pair[k][0][0]), prim_key(pair[k][0][1])))
    validate_no_duplicate_keys(keys, name="pair support keys")
    with open(path, "w") as f:
        for k in keys:
            (pa, pb), s = pair[k]
            row: Dict[str, Any] = {
                "keys": [k[0], k[1]],
                "primitives": [
                    {"kind": pa.kind, "args": list(pa.args)},
                    {"kind": pb.kind, "args": list(pb.args)},
                ],
                "count": s.count,
                "n_questions": len(s.questions),
                "n_benchmarks": len(s.benchmarks),
            }
            if emit_raw_count:
                row["raw_count"] = s.raw_count
            f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def write_support_tables(
    data_dir: Path,
    *,
    benchmarks: Optional[Iterable[str]] = None,
    output_dir: Optional[Path] = None,
    count_unique_per_question: bool = False,
) -> Dict[str, Any]:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir) if output_dir else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    unary, pair = aggregate_support(
        data_dir,
        benchmarks=benchmarks,
        count_unique_per_question=count_unique_per_question,
    )
    emit_raw = count_unique_per_question
    _write_unary(
        output_dir / "primitive_support.jsonl", unary, emit_raw_count=emit_raw
    )
    _write_pair(output_dir / "pair_support.jsonl", pair, emit_raw_count=emit_raw)

    summary: Dict[str, Any] = {
        "n_primitives": len(unary),
        "n_pairs": len(pair),
        "total_primitive_occurrences": sum(s.count for _, s in unary.values()),
        "total_pair_occurrences": sum(s.count for _, s in pair.values()),
    }
    if count_unique_per_question:
        summary["total_primitive_raw_occurrences"] = sum(
            s.raw_count for _, s in unary.values()
        )
        summary["total_pair_raw_occurrences"] = sum(s.raw_count for _, s in pair.values())
        summary["count_unique_per_question"] = True
    write_manifest(output_dir / "support_summary.json", summary)
    logger.info("wrote support tables: %s", summary)
    return summary


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True, type=Path,
                   help="Canonicalized fine-routing data directory.")
    p.add_argument("--output_dir", type=Path, default=None,
                   help="Where to write support tables (default: --data_dir).")
    p.add_argument("--benchmarks", nargs="*", default=None)
    p.add_argument(
        "--count-unique-per-question",
        action="store_true",
        help=(
            "At most one support count per data row (question line) per primitive/pair; "
            "avoids MCTS duplicate playout inflating --min_count. Adds raw_count fields."
        ),
    )
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
        count_unique_per_question=args.count_unique_per_question,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
