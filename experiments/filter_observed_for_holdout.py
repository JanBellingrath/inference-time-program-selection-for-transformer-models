"""Strip held-out length-2 candidates from observed supervision.

Inputs
------
* A compositional catalogue directory (``manifest.json``,
  ``legal_programs/{b}.jsonl``, ``observed/{b}.jsonl``).
* A ``holdout_split.json`` from :mod:`experiments.holdout_edge_split`.

For every observed record we:
  * drop ``obs_indices`` whose legal program has ``length == 2`` and whose
    *unordered primitive index pair* lies in
    ``E_val_holdout ∪ E_test_holdout``;
  * keep singleton (``length == 1``) and the empty (``length == 0``) rows;
  * drop questions that have **no** non-empty candidates remaining.
    Questions kept retain the empty program in their candidate list so
    ``softmax_ce_on_observed`` always normalises over a non-empty set.

Renormalisation is automatic: the existing soft cross-entropy in
:func:`routers.compositional_router.softmax_ce_on_observed` normalises over
whatever candidates are present in ``obs_indices`` / ``obs_deltas``.

In addition, when ``--dense_deltas`` is supplied we emit a
``{bench}.pt`` ``keep_mask`` tensor (``[N_b]`` float, ``1.0`` keep / ``0.0``
held-out) under ``--output_dir / dense_masks``.  The trainer can then
multiply the ``obs_mask`` by this vector when running with
``--use_dense_supervision``.

Usage::

    python -m experiments.filter_observed_for_holdout \\
        --catalogue_dir compositional_runs/csqa_compositional \\
        --holdout_split compositional_runs/csqa_compositional/holdout/holdout_split.json \\
        --output_dir compositional_runs/csqa_compositional/holdout
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logger = logging.getLogger("filter_observed_for_holdout")

DEFAULT_OBSERVED_SUBDIR = "observed_train_filtered"
DEFAULT_DENSE_MASK_SUBDIR = "dense_masks"


# ---------------------------------------------------------------------------
# IO
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


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _load_legal_programs(path: Path) -> List[Dict[str, Any]]:
    rows = _read_jsonl(path)
    rows.sort(key=lambda r: int(r["idx"]))
    for expected, r in enumerate(rows):
        if int(r["idx"]) != expected:
            raise ValueError(f"non-contiguous legal_programs at {path}: row {r}")
    return rows


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def _holdout_pair_set(split: Dict[str, Any]) -> Set[Tuple[int, int]]:
    """Unordered primitive-index pairs in val ∪ test."""
    out: Set[Tuple[int, int]] = set()
    for key in ("E_val_holdout", "E_test_holdout"):
        for e in split.get(key, []):
            a, b = int(e["a"]), int(e["b"])
            if a == b:
                continue
            out.add((min(a, b), max(a, b)))
    return out


def _held_out_legal_indices(
    legal: Sequence[Dict[str, Any]],
    holdout_pairs: Set[Tuple[int, int]],
) -> Set[int]:
    """Legal-program rows whose length==2 primitive pair is held out."""
    held: Set[int] = set()
    for row in legal:
        if int(row.get("length", 0)) != 2:
            continue
        prims = row.get("primitive_indices") or []
        if len(prims) != 2:
            continue
        a, b = int(prims[0]), int(prims[1])
        key = (min(a, b), max(a, b))
        if key in holdout_pairs:
            held.add(int(row["idx"]))
    return held


def _filter_observed_for_benchmark(
    obs_records: Sequence[Dict[str, Any]],
    legal: Sequence[Dict[str, Any]],
    held_indices: Set[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Drop held-out rows from observed records; preserve length-0/1 rows."""
    empty_idx: Optional[int] = None
    for row in legal:
        if int(row.get("length", 0)) == 0:
            empty_idx = int(row["idx"])
            break

    out: List[Dict[str, Any]] = []
    stats: Dict[str, int] = {
        "rows_in": 0,
        "rows_kept": 0,
        "rows_dropped_no_nonempty": 0,
        "candidates_dropped": 0,
        "empty_added": 0,
    }
    for rec in obs_records:
        stats["rows_in"] += 1
        kept_indices: List[int] = []
        kept_deltas: List[float] = []
        had_nonempty = False
        for idx, delta in zip(rec.get("obs_indices", []), rec.get("obs_deltas", [])):
            i = int(idx)
            if i in held_indices:
                stats["candidates_dropped"] += 1
                continue
            kept_indices.append(i)
            kept_deltas.append(float(delta))
            if empty_idx is None or i != empty_idx:
                had_nonempty = True
        if not had_nonempty:
            stats["rows_dropped_no_nonempty"] += 1
            continue
        if empty_idx is not None and empty_idx not in kept_indices:
            kept_indices.append(empty_idx)
            kept_deltas.append(0.0)
            stats["empty_added"] += 1
        # keep them sorted to match the convention in build_compositional_catalogues.
        order = sorted(range(len(kept_indices)), key=lambda i: kept_indices[i])
        kept_indices = [kept_indices[i] for i in order]
        kept_deltas = [kept_deltas[i] for i in order]
        new_rec = dict(rec)
        new_rec["obs_indices"] = kept_indices
        new_rec["obs_deltas"] = kept_deltas
        new_rec["n_obs"] = len(kept_indices)
        out.append(new_rec)
        stats["rows_kept"] += 1
    return out, stats


def filter_observed_for_holdout(
    catalogue_dir: Path,
    split_path: Path,
    output_dir: Path,
    *,
    observed_subdir: str = DEFAULT_OBSERVED_SUBDIR,
    dense_delta_paths: Optional[Dict[str, Path]] = None,
    dense_mask_subdir: str = DEFAULT_DENSE_MASK_SUBDIR,
) -> Dict[str, Any]:
    catalogue_dir = Path(catalogue_dir)
    split_path = Path(split_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    obs_out_dir = output_dir / observed_subdir
    obs_out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = catalogue_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(split_path) as f:
        split = json.load(f)

    holdout_pairs = _holdout_pair_set(split)
    logger.info("holdout pairs (val ∪ test): %d", len(holdout_pairs))

    per_bench: Dict[str, Any] = {}
    overall = {
        "rows_in": 0, "rows_kept": 0, "rows_dropped_no_nonempty": 0,
        "candidates_dropped": 0, "empty_added": 0,
    }

    dense_mask_dir: Optional[Path] = None
    if dense_delta_paths:
        dense_mask_dir = output_dir / dense_mask_subdir
        dense_mask_dir.mkdir(parents=True, exist_ok=True)

    for bench, info in manifest.get("benchmarks", {}).items():
        legal_path = catalogue_dir / info["legal_programs_path"]
        obs_path = catalogue_dir / info["observed_path"]
        if not legal_path.is_file() or not obs_path.is_file():
            logger.warning("[%s] missing legal/observed files; skipping", bench)
            continue
        legal = _load_legal_programs(legal_path)
        held_indices = _held_out_legal_indices(legal, holdout_pairs)
        obs_records = _read_jsonl(obs_path)
        filtered, stats = _filter_observed_for_benchmark(
            obs_records, legal, held_indices,
        )
        out_path = obs_out_dir / f"{bench}.jsonl"
        _write_jsonl(out_path, filtered)
        for k, v in stats.items():
            overall[k] = overall.get(k, 0) + v

        bench_out: Dict[str, Any] = {
            "n_legal_programs": len(legal),
            "n_held_out_program_rows": len(held_indices),
            "filtered_observed_path": str(out_path.relative_to(output_dir)),
            "stats": stats,
        }
        if dense_mask_dir is not None and bench in (dense_delta_paths or {}):
            n_legal = len(legal)
            keep_mask = torch.ones(n_legal, dtype=torch.float32)
            for i in held_indices:
                keep_mask[int(i)] = 0.0
            mask_path = dense_mask_dir / f"{bench}.pt"
            torch.save({"keep_mask": keep_mask, "n_held_out": len(held_indices)}, mask_path)
            bench_out["dense_keep_mask_path"] = str(mask_path.relative_to(output_dir))

        per_bench[bench] = bench_out
        logger.info(
            "[%s] held-out rows=%d  obs in=%d kept=%d dropped_empty=%d cand_dropped=%d",
            bench, len(held_indices), stats["rows_in"], stats["rows_kept"],
            stats["rows_dropped_no_nonempty"], stats["candidates_dropped"],
        )

    out_artifact = {
        "schema_version": 1,
        "catalogue_dir": str(catalogue_dir),
        "split_path": str(split_path),
        "observed_subdir": observed_subdir,
        "dense_mask_subdir": dense_mask_subdir if dense_delta_paths else None,
        "n_holdout_pairs": len(holdout_pairs),
        "benchmarks": per_bench,
        "overall_stats": overall,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    summary_path = output_dir / "filter_summary.json"
    with open(summary_path, "w") as f:
        json.dump(out_artifact, f, indent=2)
    logger.info("filter summary -> %s  (%s)", summary_path, overall)
    return out_artifact


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalogue_dir", required=True, type=Path)
    p.add_argument("--holdout_split", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--observed_subdir", default=DEFAULT_OBSERVED_SUBDIR,
                   help="Subdirectory under --output_dir for filtered observed/.")
    p.add_argument("--dense_deltas", nargs="*", default=None,
                   help="Optional bench=path entries; emits per-bench keep_mask.")
    p.add_argument("--dense_mask_subdir", default=DEFAULT_DENSE_MASK_SUBDIR)
    p.add_argument("--log_level", default="INFO")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    dense_paths: Optional[Dict[str, Path]] = None
    if args.dense_deltas:
        dense_paths = {}
        for entry in args.dense_deltas:
            if "=" not in entry:
                logger.error("--dense_deltas expects bench=path entries: %r", entry)
                return 2
            bench, path = entry.split("=", 1)
            dense_paths[bench] = Path(path)
    filter_observed_for_holdout(
        catalogue_dir=args.catalogue_dir,
        split_path=args.holdout_split,
        output_dir=args.output_dir,
        observed_subdir=args.observed_subdir,
        dense_delta_paths=dense_paths,
        dense_mask_subdir=args.dense_mask_subdir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
