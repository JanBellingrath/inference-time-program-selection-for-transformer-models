#!/usr/bin/env python3
"""Stitch per-benchmark compositional dirs into one multibench package.

Each benchmark may use a different local ``primitives.jsonl`` ordering.  This
script unions primitives by canonical ``key``, remaps legal program rows to
the unified index space, **narrows** legal programs to the first occurrence of
each unique *applied route* (so column j of the provided dense matrix matches
legal row j), rebuilds incidence / pair incidence, remaps observed indices, and
writes a ``manifest.json`` compatible with ``data_prep.build_joint_catalogue``.

Dense checkpoints are **not** copied here; pass them to ``build_joint_catalogue
--dense_deltas`` separately.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from core.edit_dsl import Primitive, apply_program  # noqa: E402
from data_prep.compositional.catalogue import (  # noqa: E402
    _save_incidence,
    _save_pair_incidence,
    build_incidence_tensor,
    build_pair_incidence_tensor,
)
from data_prep.merge_dense_increment import _read_jsonl  # noqa: E402

logger = logging.getLogger("stitch_multibench_catalogue")


def _narrow_legal_and_map(
    *,
    compositional_dir: Path,
    bench: str,
    manifest: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    """First-occurrence unique-route order; new rows idx 0..K-1.

    Returns
    -------
    new_rows : narrowed legal JSONL records (primitive_indices still *local*).
    old_prog_to_narrow : original legal idx -> narrow idx.
    """
    bench_meta = manifest["benchmarks"][bench]
    anchor = [int(x) for x in bench_meta["anchor"]]
    prim_path = compositional_dir / manifest["primitives_path"]
    prim_rows = _read_jsonl(prim_path)
    prim_rows.sort(key=lambda r: int(r["idx"]))
    primitives = [
        Primitive(str(r["kind"]), tuple(int(x) for x in r["args"])) for r in prim_rows
    ]
    legal_rows = _read_jsonl(compositional_dir / bench_meta["legal_programs_path"])
    legal_rows.sort(key=lambda r: int(r["idx"]))

    seen: Dict[Tuple[int, ...], int] = {}
    new_rows: List[Dict[str, Any]] = []
    old_to_narrow: Dict[int, int] = {}

    for r in legal_rows:
        oidx = int(r["idx"])
        prog = [primitives[int(j)] for j in r["primitive_indices"]]
        route = tuple(int(x) for x in apply_program(anchor, prog))
        if route in seen:
            old_to_narrow[oidx] = seen[route]
            continue
        j = len(new_rows)
        seen[route] = j
        old_to_narrow[oidx] = j
        rec = dict(r)
        rec["idx"] = j
        new_rows.append(rec)

    return new_rows, old_to_narrow


def _union_primitives(
    bench_specs: Sequence[Tuple[str, Path, Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[int, int]]]:
    """Union primitive *key* strings across benches; stable sorted order."""
    key_to_record: Dict[str, Dict[str, Any]] = {}
    local_maps: Dict[str, Dict[int, int]] = {}

    for bench, comp_dir, manifest in bench_specs:
        prim_rows = _read_jsonl(comp_dir / manifest["primitives_path"])
        prim_rows.sort(key=lambda r: int(r["idx"]))
        lmap: Dict[int, int] = {}
        for r in prim_rows:
            key = str(r["key"])
            if key not in key_to_record:
                rec = dict(r)
                rec.pop("idx", None)
                rec.pop("support", None)
                key_to_record[key] = rec
            lmap[int(r["idx"])] = key
        local_maps[bench] = lmap

    sorted_keys = sorted(key_to_record)
    unified: List[Dict[str, Any]] = []
    key_to_new_idx: Dict[str, int] = {}
    for ni, key in enumerate(sorted_keys):
        rec = dict(key_to_record[key])
        rec["idx"] = ni
        unified.append(rec)
        key_to_new_idx[key] = ni

    remapped_local: Dict[str, Dict[int, int]] = {}
    for bench, comp_dir, manifest in bench_specs:
        lmap = local_maps[bench]
        remapped_local[bench] = {
            old_i: key_to_new_idx[lmap[old_i]] for old_i in lmap
        }

    return unified, remapped_local


def stitch(
    *,
    output_dir: Path,
    bench_specs: Sequence[Tuple[str, Path]],
) -> None:
    """
    bench_specs : list of (benchmark_name, compositional_dir)
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded: List[Tuple[str, Path, Dict[str, Any]]] = []
    for bench, cdir in bench_specs:
        cdir = Path(cdir).resolve()
        man = json.loads((cdir / "manifest.json").read_text())
        if bench not in man["benchmarks"]:
            raise KeyError(f"{bench} not in manifest for {cdir}")
        loaded.append((bench, cdir, man))

    unified_prims, local_to_global = _union_primitives(loaded)
    M = len(unified_prims)

    with open(output_dir / "primitives.jsonl", "w") as f:
        for rec in unified_prims:
            f.write(json.dumps(rec, sort_keys=False) + "\n")

    geometry_ref = loaded[0][2].get("geometry", {})
    filter_ref: Dict[str, Any] = {
        "min_count": 1,
        "min_questions": 1,
        "min_benchmarks": 1,
        "keep_kinds": sorted({str(rec["kind"]) for rec in unified_prims}),
    }

    manifest_benchmarks: Dict[str, Any] = {}

    for bench, comp_dir, src_man in loaded:
        narrow_rows, old_to_narrow = _narrow_legal_and_map(
            compositional_dir=comp_dir,
            bench=bench,
            manifest=src_man,
        )
        remap_p = local_to_global[bench]
        prim_lists: List[Tuple[int, ...]] = []
        for rec in narrow_rows:
            new_pi = sorted(remap_p[int(x)] for x in rec["primitive_indices"])
            rec["primitive_indices"] = list(new_pi)
            prim_lists.append(tuple(new_pi))

        (output_dir / "legal_programs").mkdir(parents=True, exist_ok=True)
        lp_out = output_dir / "legal_programs" / f"{bench}.jsonl"
        with open(lp_out, "w") as f:
            for rec in narrow_rows:
                f.write(json.dumps(rec, sort_keys=False) + "\n")

        a_idx, a_val, a_shape, lengths = build_incidence_tensor(prim_lists, M)
        pair_index, b_idx, b_val, b_shape = build_pair_incidence_tensor(prim_lists)

        (output_dir / "incidence").mkdir(parents=True, exist_ok=True)
        (output_dir / "pair_incidence").mkdir(parents=True, exist_ok=True)
        _save_incidence(
            output_dir / "incidence" / f"{bench}.pt",
            a_idx, a_val, a_shape, lengths,
        )
        _save_pair_incidence(
            output_dir / "pair_incidence" / f"{bench}.pt",
            pair_index, b_idx, b_val, b_shape,
        )

        src_obs = _read_jsonl(comp_dir / src_man["benchmarks"][bench]["observed_path"])
        out_obs: List[Dict[str, Any]] = []
        for rec in src_obs:
            ni, nd = [], []
            for oi, d in zip(rec.get("obs_indices", []), rec.get("obs_deltas", [])):
                j = old_to_narrow.get(int(oi))
                if j is None:
                    continue
                ni.append(j)
                nd.append(float(d))
            if not ni:
                continue
            nrec = dict(rec)
            nrec["obs_indices"] = ni
            nrec["obs_deltas"] = nd
            nrec["n_obs"] = len(ni)
            out_obs.append(nrec)

        (output_dir / "observed").mkdir(parents=True, exist_ok=True)
        obs_out = output_dir / "observed" / f"{bench}.jsonl"
        with open(obs_out, "w") as f:
            for rec in out_obs:
                f.write(json.dumps(rec) + "\n")

        src_b = src_man["benchmarks"][bench]
        manifest_benchmarks[bench] = {
            "anchor": list(src_b["anchor"]),
            "anchor_length": int(src_b.get("anchor_length", len(src_b["anchor"]))),
            "n_legal_programs": len(narrow_rows),
            "n_legal_dropped_unknown_primitive": 0,
            "incidence_path": f"incidence/{bench}.pt",
            "pair_incidence_path": f"pair_incidence/{bench}.pt",
            "n_legal_pairs": int(pair_index.shape[0]),
            "legal_programs_path": f"legal_programs/{bench}.jsonl",
            "observed_path": f"observed/{bench}.jsonl",
            "source_jsonl": src_b.get("source_jsonl"),
            "pivot_residuals_path": src_b.get("pivot_residuals_path"),
            "full_residuals_path": src_b.get("full_residuals_path"),
            "n_questions_kept": len(out_obs),
            "n_questions_dropped_no_obs": int(src_b.get("n_questions_kept", 0)) - len(out_obs),
            "n_observed_pairs": sum(
                len(rec.get("obs_indices", [])) for rec in out_obs
            ),
        }

    overall = {
        "rows_in": sum(
            int(loaded[i][2]["benchmarks"][loaded[i][0]]["n_questions_kept"])
            for i in range(len(loaded))
        ),
        "rows_kept": sum(manifest_benchmarks[b]["n_questions_kept"] for b in manifest_benchmarks),
        "n_legal_programs_total": sum(
            manifest_benchmarks[b]["n_legal_programs"] for b in manifest_benchmarks
        ),
    }

    out_manifest = {
        "data_dir": str(output_dir),
        "output_dir": str(output_dir),
        "geometry": geometry_ref,
        "filter": filter_ref,
        "M": M,
        "primitives_path": "primitives.jsonl",
        "benchmarks": manifest_benchmarks,
        "overall_stats": overall,
        "stitch_meta": {
            "sources": [
                {"benchmark": b, "path": str(c)} for b, c in bench_specs
            ],
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(out_manifest, indent=2) + "\n")
    logger.info(
        "stitched %d benchmarks -> %s  M=%d  legal sizes %s",
        len(bench_specs),
        output_dir,
        M,
        {b: manifest_benchmarks[b]["n_legal_programs"] for b in manifest_benchmarks},
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument(
        "--bench",
        action="append",
        required=True,
        help="name=path to compositional manifest dir (repeat per benchmark).",
    )
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    specs: List[Tuple[str, Path]] = []
    for ent in args.bench:
        if "=" not in ent:
            raise SystemExit(f"expected name=path, got {ent!r}")
        name, path = ent.split("=", 1)
        specs.append((name.strip(), Path(path.strip())))
    stitch(output_dir=args.output_dir, bench_specs=specs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
