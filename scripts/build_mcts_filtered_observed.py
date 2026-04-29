#!/usr/bin/env python3
"""Derive an MCTS-only, top-k observed-candidate view of a compositional catalogue.

The existing compositional catalogues ship two parallel supervision views
for each benchmark:

* ``observed/{bench}.jsonl`` — one record per question carrying the
  *MCTS-observed* ``(program_row, delta)`` pairs (i.e. every program
  touched by MCTS + assign increments for that question). This is the
  sparse, data-we-actually-have view.
* ``dense_deltas/{bench}.pt`` — a full ``[Q, N_legal]`` matrix with every
  legal program re-evaluated on every question. This is the "dense
  control" used with ``--use_dense_supervision``.

Dense evaluation is a good control but is wasteful and also ends up
supervising the router on programs that MCTS never found any reason to
explore for a given question. This script derives, from the existing
observed records, a **reduced** observed-candidate view that restricts
supervision to the programs actually exercised by MCTS.

Concretely it:

1. Loads ``observed/{bench}.jsonl`` and sums ``max(0, delta)`` per program
   row across all questions. This "positive mass" tells us how useful a
   program has been under MCTS evaluation.
2. Orders programs by descending positive mass and takes the smallest
   prefix whose cumulative mass covers ``--mass_coverage`` (default
   ``0.95``) of the total. An optional hard cap ``--max_k`` can be
   applied on top.
3. Writes a new ``{bench}.jsonl`` under ``--output_dir`` that keeps the
   same schema (``residual_idx``, ``question_id``, ``obs_indices``,
   ``obs_deltas``...) but drops any ``obs`` entry outside the top-k
   program set.
4. Writes a ``top_k_programs.json`` summary with the retained program
   ids, the coverage reached, and per-benchmark statistics.

Down-stream usage is minimal: point the trainer at the catalogue as usual
and pass ``--observed_dir <output_dir>`` so ``CompositionalDataset``
loads the filtered records instead of the default ones. The catalogue's
``A``/``B``/``ℓ`` are *unchanged* — the router still scores all N legal
programs, but the softmax CE only normalises over the observed subset of
each question, matching the "MCTS-only supervision" spec.

Example
-------
::

    python -m scripts.build_mcts_filtered_observed \
        --catalogue_dir compositional_runs/csqa_ft_unified95/catalog_mass095 \
        --benchmarks commonsenseqa \
        --mass_coverage 0.95 \
        --output_dir compositional_runs/csqa_ft_unified95/catalog_mass095_mcts095
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))


logger = logging.getLogger("build_mcts_filtered_observed")


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")


def _compute_top_k(
    observed: Sequence[Dict],
    n_programs: int,
    *,
    mass_coverage: float,
    max_k: Optional[int] = None,
    min_k: int = 1,
) -> Tuple[List[int], float, float]:
    """Return top-k program ids by MCTS positive mass, plus coverage stats."""
    if not 0.0 < mass_coverage <= 1.0:
        raise ValueError(f"mass_coverage must be in (0, 1], got {mass_coverage}")
    pos_mass = [0.0] * n_programs
    total_pos = 0.0
    for rec in observed:
        for i, d in zip(rec["obs_indices"], rec["obs_deltas"]):
            if i < 0 or i >= n_programs:
                continue
            if d > 0.0:
                pos_mass[int(i)] += float(d)
                total_pos += float(d)
    if total_pos <= 0.0:
        raise RuntimeError(
            "total MCTS positive mass is 0 — nothing to select from; check that "
            "the observed jsonl is populated and has positive deltas."
        )
    order = sorted(range(n_programs), key=lambda i: pos_mass[i], reverse=True)
    ordered_mass = [pos_mass[i] for i in order]
    cum = 0.0
    chosen: List[int] = []
    for i, m in zip(order, ordered_mass):
        chosen.append(int(i))
        cum += m
        if cum / total_pos >= mass_coverage and len(chosen) >= min_k:
            break
    if max_k is not None and len(chosen) > max_k:
        chosen = chosen[:max_k]
        cum = sum(pos_mass[i] for i in chosen)
    return chosen, total_pos, cum


def _filter_observed(
    observed: Sequence[Dict],
    keep_ids: Sequence[int],
) -> Tuple[List[Dict], Dict[str, int]]:
    keep_set = set(int(i) for i in keep_ids)
    out: List[Dict] = []
    stats: Dict[str, int] = {
        "rows_in": 0,
        "rows_kept": 0,
        "rows_dropped_empty": 0,
        "obs_in": 0,
        "obs_kept": 0,
        "obs_dropped": 0,
    }
    for rec in observed:
        stats["rows_in"] += 1
        idxs = rec["obs_indices"]
        deltas = rec["obs_deltas"]
        stats["obs_in"] += len(idxs)
        kept_pairs = [(int(i), float(d)) for i, d in zip(idxs, deltas) if int(i) in keep_set]
        stats["obs_kept"] += len(kept_pairs)
        stats["obs_dropped"] += len(idxs) - len(kept_pairs)
        if not kept_pairs:
            stats["rows_dropped_empty"] += 1
            continue
        kept_pairs.sort(key=lambda p: p[0])
        new_rec = dict(rec)
        new_rec["obs_indices"] = [p[0] for p in kept_pairs]
        new_rec["obs_deltas"] = [p[1] for p in kept_pairs]
        new_rec["n_obs"] = len(kept_pairs)
        new_rec["source_view"] = "mcts_topk"
        out.append(new_rec)
        stats["rows_kept"] += 1
    return out, stats


def _resolve_manifest(catalogue_dir: Path) -> Dict:
    manifest_path = catalogue_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    with open(manifest_path) as f:
        return json.load(f)


def build(
    catalogue_dir: Path,
    output_dir: Path,
    *,
    benchmarks: Optional[Sequence[str]],
    mass_coverage: float,
    max_k: Optional[int],
    min_k: int,
) -> Dict:
    manifest = _resolve_manifest(catalogue_dir)
    bench_names = list(benchmarks or manifest["benchmarks"].keys())
    output_dir.mkdir(parents=True, exist_ok=True)
    obs_out_dir = output_dir / "observed"
    obs_out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict = {
        "source_catalogue_dir": str(catalogue_dir),
        "output_dir": str(output_dir),
        "mass_coverage": mass_coverage,
        "max_k": max_k,
        "min_k": min_k,
        "benchmarks": {},
    }

    for bench in bench_names:
        info = manifest["benchmarks"].get(bench)
        if info is None:
            logger.warning("[%s] missing manifest entry; skipping", bench)
            continue
        n_legal = int(info.get("n_legal_programs") or manifest["joint"]["n_programs"])
        obs_path = catalogue_dir / info["observed_path"]
        if not obs_path.is_file():
            logger.warning("[%s] missing observed file %s; skipping", bench, obs_path)
            continue
        observed = _read_jsonl(obs_path)
        logger.info(
            "[%s] loaded observed rows=%d  n_legal=%d  from %s",
            bench, len(observed), n_legal, obs_path,
        )
        keep_ids, total_pos, kept_mass = _compute_top_k(
            observed, n_legal,
            mass_coverage=mass_coverage, max_k=max_k, min_k=min_k,
        )
        filtered, stats = _filter_observed(observed, keep_ids)
        out_jsonl = obs_out_dir / f"{bench}.jsonl"
        _write_jsonl(out_jsonl, filtered)
        logger.info(
            "[%s] k=%d  mass kept=%.4f / %.4f (%.2f%%)  rows kept=%d/%d  "
            "obs kept=%d/%d  -> %s",
            bench, len(keep_ids), kept_mass, total_pos,
            100.0 * kept_mass / max(total_pos, 1e-9),
            stats["rows_kept"], stats["rows_in"],
            stats["obs_kept"], stats["obs_in"],
            out_jsonl,
        )
        summary["benchmarks"][bench] = {
            "k": len(keep_ids),
            "n_legal_total": n_legal,
            "total_positive_mass": total_pos,
            "kept_positive_mass": kept_mass,
            "mass_fraction": kept_mass / max(total_pos, 1e-9),
            "keep_ids": keep_ids,
            "filter_stats": stats,
            "output_observed_path": str(out_jsonl),
            "source_observed_path": str(obs_path),
        }

    summary_path = output_dir / "top_k_programs.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("wrote summary -> %s", summary_path)
    return summary


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--catalogue_dir", required=True, type=Path,
                   help="Source compositional catalogue directory (contains "
                        "manifest.json and observed/*.jsonl).")
    p.add_argument("--output_dir", required=True, type=Path,
                   help="Where to write the filtered observed/*.jsonl and "
                        "top_k_programs.json summary. Can be a fresh "
                        "directory; the trainer only needs this dir via "
                        "--observed_dir.")
    p.add_argument("--benchmarks", nargs="*", default=None,
                   help="Subset of benchmarks to filter (default: all in "
                        "the manifest).")
    p.add_argument("--mass_coverage", type=float, default=0.95,
                   help="Fraction of MCTS positive mass to cover "
                        "(smallest top-k prefix whose cumulative mass "
                        "reaches this fraction; default 0.95).")
    p.add_argument("--max_k", type=int, default=None,
                   help="Optional hard cap on k (keep at most this many "
                        "programs even if --mass_coverage would select more).")
    p.add_argument("--min_k", type=int, default=1,
                   help="Minimum number of programs to keep (default 1). "
                        "Useful if --mass_coverage is very small.")
    p.add_argument("--skip_if_exists", action="store_true",
                   help="Skip the whole run if --output_dir already contains "
                        "both observed/{bench}.jsonl for every selected bench "
                        "and top_k_programs.json.")
    p.add_argument("--log_level", default="INFO")
    return p


def _already_built(output_dir: Path, benchmarks: Sequence[str]) -> bool:
    if not (output_dir / "top_k_programs.json").is_file():
        return False
    for b in benchmarks:
        if not (output_dir / "observed" / f"{b}.jsonl").is_file():
            return False
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    manifest = _resolve_manifest(args.catalogue_dir)
    bench_names = list(args.benchmarks or manifest["benchmarks"].keys())
    if args.skip_if_exists and _already_built(args.output_dir, bench_names):
        logger.info(
            "filtered observed already exists under %s for benches=%s; skipping",
            args.output_dir, bench_names,
        )
        return 0
    build(
        args.catalogue_dir, args.output_dir,
        benchmarks=bench_names,
        mass_coverage=float(args.mass_coverage),
        max_k=int(args.max_k) if args.max_k is not None else None,
        min_k=int(args.min_k),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
