"""Rewrite a fine-routing data directory in canonical-program form.

Input: a directory written by ``data_prep/build_fine_routing_dataset.py`` or
``data_prep/build_ft_fine_routing_dataset.py``.  Each ``{benchmark}.jsonl``
row carries an ``anchor_sequence`` plus either:

* an exhaustive ``deviations: [{key, score, delta}, ...]`` list, or
* an MCTS-mode ``best_seq`` + ``explored: [{seq, score, delta}, ...]`` list
  (with optional ``router_target`` aligned to ``explored``).

For every distinct route variant in the row, this script computes the
canonical program ``C(route)`` from :mod:`core.edit_dsl` (the conservative
regime: bounded length, support-disjoint primitives, sorted lex order, no
no-ops) and writes a sibling JSONL row with a ``programs`` list of those
canonical programs.

Routes that fall outside the conservative regime (i.e. unreachable within
``K`` admissible edits from the anchor) are omitted from ``programs`` and
counted in ``n_unreachable``.

Usage::

    python -m data_prep.canonicalize_programs \\
        --data_dir fine_routing_data/<run> \\
        [--output_dir fine_routing_data/<run>_canonical] \\
        [--max_program_len K] [--swap_radius R] [--editable_start S] \\
        [--copy-residuals]

K, R, and S default to the values stored in the source directory's
``config.json`` and may be overridden on the CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Package-relative path setup so this module is runnable both as
# ``python -m data_prep.canonicalize_programs`` and as a script.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.edit_dsl import (  # noqa: E402  (sys.path manipulated above)
    SKIP,
    Primitive,
    Program,
    apply_program,
    canonical_key_str,
    canonicalize_cached,
    program_to_dicts,
    repeat as new_repeat,
    skip as new_skip,
    swap as new_swap,
)

logger = logging.getLogger("canonicalize_programs")


# ---------------------------------------------------------------------------
# Legacy-key parser
# ---------------------------------------------------------------------------

_PRIM_RE = re.compile(r"^(skip|repeat|swap)\(([\d,]+)\)$")


def _parse_legacy_primitive(token: str) -> Primitive:
    """Parse a single legacy primitive token, e.g. ``"skip(17)"``.

    Note: the legacy ``repeat(pos)`` token records the *destination* position;
    we translate it to the new-DSL convention where ``repeat(i)`` is keyed by
    the *source* position ``i = pos - 1``.
    """
    m = _PRIM_RE.match(token.strip())
    if not m:
        raise ValueError(f"Cannot parse legacy primitive token: {token!r}")
    kind = m.group(1)
    args = [int(x) for x in m.group(2).split(",") if x]
    if kind == "skip":
        if len(args) != 1:
            raise ValueError(f"skip expects 1 arg: {token!r}")
        return new_skip(args[0])
    if kind == "repeat":
        if len(args) != 1:
            raise ValueError(f"repeat expects 1 arg: {token!r}")
        if args[0] <= 0:
            raise ValueError(f"legacy repeat({args[0]}) has no source")
        return new_repeat(args[0] - 1)
    if kind == "swap":
        if len(args) != 2:
            raise ValueError(f"swap expects 2 args: {token!r}")
        return new_swap(args[0], args[1])
    raise ValueError(f"Unknown primitive kind in token: {token!r}")


def _parse_legacy_key(key: str) -> Program:
    """Parse a legacy ``canonical_key`` string into a ``Program`` tuple.

    ``"noop"`` returns the empty program.  Multi-primitive keys are
    ``"+"``-joined.  This is *not* used to define canonicality (the new DSL
    re-canonicalizes from the route); it is used to reconstruct the route
    when the input row only stores keys (exhaustive enumeration mode).
    """
    if key == "noop" or key == "":
        return ()
    parts = [p for p in key.split("+") if p]
    return tuple(_parse_legacy_primitive(p) for p in parts)


# ---------------------------------------------------------------------------
# Per-row canonicalization
# ---------------------------------------------------------------------------


def _route_from_dev_entry(anchor: Sequence[int], dev: Dict[str, Any]) -> Optional[List[int]]:
    """Reconstruct the route for a single exhaustive-mode deviation entry.

    The exhaustive builder stores only the canonical key and the score/delta,
    not the full sequence; we reproduce the sequence by parsing the key and
    applying it to the anchor (under the new DSL, which preserves the legacy
    semantics via ``primitive_to_legacy_edit``).
    """
    key = dev.get("key")
    if key is None:
        seq = dev.get("seq")
        if seq is None:
            return None
        return [int(x) for x in seq]
    try:
        prog = _parse_legacy_key(str(key))
    except ValueError as exc:
        logger.warning("skip unparsable deviation key %r: %s", key, exc)
        return None
    return apply_program(anchor, prog)


def _seq_equal(a: Sequence[int], b: Sequence[int]) -> bool:
    if len(a) != len(b):
        return False
    return all(int(x) == int(y) for x, y in zip(a, b))


def _canonicalize_row(
    row: Dict[str, Any],
    *,
    K: int,
    swap_radius: int,
    editable_indices: Tuple[int, ...],
    include_assign: bool = False,
    dedupe_assign_with_struct: bool = False,
) -> Tuple[Dict[str, Any], Counter]:
    """Compute the canonical-program rewrite of one input row.

    Returns the new row plus a Counter of per-row event tags
    (``"in"``, ``"canonical"``, ``"unreachable"``, by source).
    """
    stats: Counter = Counter()

    anchor = [int(x) for x in row["anchor_sequence"]]
    L = len(anchor)
    if any(i >= L for i in editable_indices):
        editable_indices = tuple(i for i in editable_indices if i < L)

    programs_out: List[Dict[str, Any]] = []
    n_unreachable = 0

    def _emit(prog: Optional[Program], *, source: str, **extra: Any) -> None:
        nonlocal n_unreachable
        stats[f"in:{source}"] += 1
        stats["in"] += 1
        if prog is None:
            n_unreachable += 1
            stats[f"unreachable:{source}"] += 1
            stats["unreachable"] += 1
            return
        stats[f"canonical:{source}"] += 1
        stats["canonical"] += 1
        entry: Dict[str, Any] = {
            "program": program_to_dicts(prog),
            "program_key": canonical_key_str(prog),
            "length": len(prog),
            "source": source,
        }
        entry.update({k: v for k, v in extra.items() if v is not None})
        programs_out.append(entry)

    # Anchor itself is the no-op program.
    _emit((), source="anchor", delta=0.0, score=row.get("anchor_score"))

    # MCTS mode --------------------------------------------------------------
    if row.get("search_mode") == "mcts" or "best_seq" in row or "explored" in row:
        explored = row.get("explored") or []
        router_target = row.get("router_target") or []
        for j, exp in enumerate(explored):
            seq = exp.get("seq")
            if seq is None:
                continue
            seq = [int(x) for x in seq]
            if _seq_equal(seq, anchor):
                # The anchor is implicitly already covered above; carry mass through.
                if j < len(router_target) and programs_out and programs_out[0]["source"] == "anchor":
                    programs_out[0].setdefault("router_target_mass", 0.0)
                    programs_out[0]["router_target_mass"] += float(router_target[j])
                continue
            prog = canonicalize_cached(
                anchor,
                seq,
                K=K,
                editable_indices=editable_indices,
                swap_radius=swap_radius,
                include_assign=include_assign,
                dedupe_assign_with_struct=dedupe_assign_with_struct,
            )
            mass = float(router_target[j]) if j < len(router_target) else None
            _emit(
                prog,
                source="explored",
                delta=exp.get("delta"),
                score=exp.get("score"),
                router_target_mass=mass,
            )

        best_seq = row.get("best_seq")
        if best_seq is not None:
            best_seq = [int(x) for x in best_seq]
            already = any(
                p["source"] in ("explored", "best_seq")
                and _seq_equal(apply_program(anchor, _program_from_dicts(p["program"])), best_seq)
                for p in programs_out
                if p["source"] != "anchor"
            )
            if not already and not _seq_equal(best_seq, anchor):
                prog = canonicalize_cached(
                    anchor,
                    best_seq,
                    K=K,
                    editable_indices=editable_indices,
                    swap_radius=swap_radius,
                    include_assign=include_assign,
                    dedupe_assign_with_struct=dedupe_assign_with_struct,
                )
                _emit(
                    prog,
                    source="best_seq",
                    delta=row.get("best_delta"),
                    score=row.get("best_score"),
                )

    # Exhaustive mode --------------------------------------------------------
    deviations = row.get("deviations") or []
    for dev in deviations:
        if dev.get("key") == "noop":
            # Anchor already emitted; just attach delta/score sanity info.
            continue
        seq = _route_from_dev_entry(anchor, dev)
        if seq is None:
            continue
        if _seq_equal(seq, anchor):
            continue
        prog = canonicalize_cached(
            anchor,
            seq,
            K=K,
            editable_indices=editable_indices,
            swap_radius=swap_radius,
            include_assign=include_assign,
            dedupe_assign_with_struct=dedupe_assign_with_struct,
        )
        _emit(
            prog,
            source="deviation",
            delta=dev.get("delta"),
            score=dev.get("score"),
            legacy_key=dev.get("key"),
        )

    new_row: Dict[str, Any] = {
        "benchmark_id": row.get("benchmark_id"),
        "question_id": row.get("question_id"),
        "question_hash": row.get("question_hash"),
        "anchor_sequence": anchor,
        "pivot_layer_index": row.get("pivot_layer_index"),
        "gate_label": row.get("gate_label"),
        "anchor_program": [],
        "programs": programs_out,
        "n_unreachable": n_unreachable,
        "n_programs": len(programs_out),
    }
    if "search_mode" in row:
        new_row["search_mode"] = row["search_mode"]
    return new_row, stats


def _program_from_dicts(items: Sequence[Dict[str, Any]]) -> Program:
    out: List[Primitive] = []
    for d in items:
        kind = str(d["kind"])
        args = tuple(int(x) for x in d["args"])  # type: ignore[arg-type]
        out.append(Primitive(kind, args))
    return tuple(out)


# ---------------------------------------------------------------------------
# Directory-level driver
# ---------------------------------------------------------------------------


def _load_config(data_dir: Path) -> Dict[str, Any]:
    cfg_path = data_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing {cfg_path}")
    with open(cfg_path) as f:
        return json.load(f)


def _resolve_geometry(
    cfg: Dict[str, Any],
    *,
    cli_K: Optional[int],
    cli_radius: Optional[int],
    cli_editable_start: Optional[int],
) -> Tuple[int, int, int]:
    K = cli_K if cli_K is not None else int(cfg.get("max_local_edits", 2))
    R = cli_radius if cli_radius is not None else int(cfg.get("swap_radius", 2))
    S = cli_editable_start if cli_editable_start is not None else int(cfg.get("editable_start", 0))
    if K < 0 or R < 0 or S < 0:
        raise ValueError(f"invalid geometry K={K} R={R} S={S}")
    return K, R, S


def _editable_indices_for_anchor(anchor_len: int, editable_start: int) -> Tuple[int, ...]:
    return tuple(range(editable_start, anchor_len))


def _link_residuals(src: Path, dst: Path, *, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(os.path.abspath(src), dst)
        except OSError:
            shutil.copy2(src, dst)


def canonicalize_directory(
    data_dir: Path,
    output_dir: Path,
    *,
    cli_K: Optional[int] = None,
    cli_radius: Optional[int] = None,
    cli_editable_start: Optional[int] = None,
    include_assign: bool = False,
    dedupe_assign_with_struct: bool = False,
    copy_residuals: bool = False,
    benchmarks: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Canonicalize every ``{benchmark}.jsonl`` in ``data_dir``.

    Returns a summary dict (also written to ``output_dir/canonicalization_summary.json``).
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config(data_dir)
    K, R, S = _resolve_geometry(
        cfg, cli_K=cli_K, cli_radius=cli_radius, cli_editable_start=cli_editable_start,
    )

    bench_filter = set(benchmarks) if benchmarks else None

    # Persist a thin canonicalization config alongside the original.
    out_cfg = dict(cfg)
    out_cfg["canonicalization"] = {
        "max_program_len": K,
        "swap_radius": R,
        "editable_start": S,
        "include_assign": include_assign,
        "dedupe_assign_with_struct": dedupe_assign_with_struct,
        "source_data_dir": str(data_dir),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(out_cfg, f, indent=2)

    summary: Dict[str, Any] = {
        "benchmarks": {},
        "K": K,
        "swap_radius": R,
        "editable_start": S,
        "include_assign": include_assign,
        "dedupe_assign_with_struct": dedupe_assign_with_struct,
    }

    jsonl_files = sorted(p for p in data_dir.glob("*.jsonl") if not p.name.startswith("_"))
    if not jsonl_files:
        logger.warning("no *.jsonl files in %s", data_dir)

    overall_lengths: Counter = Counter()

    for jsonl_path in jsonl_files:
        bench = jsonl_path.stem
        if bench_filter and bench not in bench_filter:
            continue
        out_path = output_dir / f"{bench}.jsonl"
        per_bench_stats: Counter = Counter()
        per_bench_lengths: Counter = Counter()
        n_rows_in = 0
        n_rows_out = 0
        t0 = time.time()
        with open(jsonl_path) as fin, open(out_path, "w") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                n_rows_in += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("[%s] skip malformed row %d: %s", bench, n_rows_in, exc)
                    continue
                anchor = row.get("anchor_sequence")
                if anchor is None:
                    logger.warning("[%s] row %d missing anchor_sequence", bench, n_rows_in)
                    continue
                editable_indices = _editable_indices_for_anchor(len(anchor), S)
                new_row, stats = _canonicalize_row(
                    row,
                    K=K,
                    swap_radius=R,
                    editable_indices=editable_indices,
                    include_assign=include_assign,
                    dedupe_assign_with_struct=dedupe_assign_with_struct,
                )
                per_bench_stats.update(stats)
                for entry in new_row["programs"]:
                    per_bench_lengths[entry["length"]] += 1
                    overall_lengths[entry["length"]] += 1
                fout.write(json.dumps(new_row) + "\n")
                n_rows_out += 1
        dt = time.time() - t0

        residual_src = data_dir / f"{bench}_pivot_residuals.pt"
        if residual_src.exists():
            _link_residuals(residual_src, output_dir / f"{bench}_pivot_residuals.pt", copy=copy_residuals)

        bench_summary = {
            "rows_in": n_rows_in,
            "rows_out": n_rows_out,
            "stats": dict(per_bench_stats),
            "length_histogram": dict(per_bench_lengths),
            "elapsed_sec": round(dt, 3),
        }
        summary["benchmarks"][bench] = bench_summary
        logger.info(
            "[%s] rows=%d  in=%d  canonical=%d  unreachable=%d  lengths=%s  (%.2fs)",
            bench,
            n_rows_out,
            per_bench_stats.get("in", 0),
            per_bench_stats.get("canonical", 0),
            per_bench_stats.get("unreachable", 0),
            dict(per_bench_lengths),
            dt,
        )

    summary["overall_length_histogram"] = dict(overall_lengths)
    with open(output_dir / "canonicalization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True, type=Path,
                   help="Source fine-routing data directory.")
    p.add_argument("--output_dir", type=Path, default=None,
                   help="Destination dir (default: <data_dir>_canonical).")
    p.add_argument("--max_program_len", type=int, default=None,
                   help="Override K (default: config.json's max_local_edits).")
    p.add_argument("--swap_radius", type=int, default=None,
                   help="Override swap radius (default: config.json's swap_radius).")
    p.add_argument("--editable_start", type=int, default=None,
                   help="Override editable_start (default: config.json's editable_start).")
    p.add_argument("--copy-residuals", action="store_true",
                   help="Copy pivot residual .pt files instead of symlinking.")
    p.add_argument(
        "--include_assign",
        action="store_true",
        help="Include MCTS-style assign(pos,value) primitives in the DSL catalogue.",
    )
    p.add_argument(
        "--dedupe_assign_with_struct",
        action="store_true",
        help=(
            "When --include_assign is enabled, drop assign primitives that are "
            "single-step-equivalent to a structural primitive on the anchor."
        ),
    )
    p.add_argument("--benchmarks", nargs="*", default=None,
                   help="Restrict to a subset of benchmarks (by JSONL stem).")
    p.add_argument("--log_level", default="INFO",
                   help="Logging level (default: INFO).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    data_dir = args.data_dir
    if not data_dir.is_dir():
        logger.error("not a directory: %s", data_dir)
        return 2
    output_dir = args.output_dir or Path(str(data_dir).rstrip("/") + "_canonical")
    summary = canonicalize_directory(
        data_dir,
        output_dir,
        cli_K=args.max_program_len,
        cli_radius=args.swap_radius,
        cli_editable_start=args.editable_start,
        include_assign=args.include_assign,
        dedupe_assign_with_struct=args.dedupe_assign_with_struct,
        copy_residuals=args.copy_residuals,
        benchmarks=args.benchmarks,
    )
    logger.info(
        "done: K=%d swap_radius=%d editable_start=%d include_assign=%s dedupe=%s benches=%d -> %s",
        summary["K"], summary["swap_radius"], summary["editable_start"],
        summary.get("include_assign"), summary.get("dedupe_assign_with_struct"),
        len(summary["benchmarks"]), output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
