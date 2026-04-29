#!/usr/bin/env python3
"""Build joint fine-router assets from a compositional joint catalogue.

Writes into *out_dir*:

* ``catalog.json`` — ``selected_routes`` = full layer sequences for joint legal rows
  1..N-1 (reference benchmark anchor), for :func:`load_selected_catalog`.
* ``per_bench_catalog.json`` — per benchmark, same class alignment; sequences
  materialized with each benchmark's manifest anchor (needed when anchors differ).
* ``dense_joint.jsonl`` — one row per (benchmark, question_id) with ``route_deltas``
  keyed by route id ``j-1`` for dense column ``j`` (column 0 = noop / STAY).
* ``staging/`` — symlinks ``{bench}_pivot_residuals.pt`` to paths in ``manifest.json``.

**Dense columns**

* ``measured_only`` (default): skip columns where that benchmark's ``dense_masks``
  ``keep_mask[j] < 0.5`` (matches compositional supervision on measured joint rows).
* ``all_legal_programs``: emit **every** legal program column ``j >= 1``. Use this for
  joint fine routing when each benchmark materializes programs on its own anchor
  (``per_bench_catalog.json``): ambiguous primitives across anchors need not be
  dropped from the **label space** — the same class index maps to different layer
  sequences per benchmark.

Example::

    python scripts/export_compositional_joint_fine_router_assets.py \\
      --catalogue_dir compositional_runs/assign3_g95k37_footpoint_filtered_joint \\
      --out_dir compositional_runs/assign3_g95k37_footpoint_filtered_joint/joint_fine_assets \\
      --benchmarks arc_easy hellaswag commonsenseqa
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.edit_dsl import Primitive, apply_program  # noqa: E402


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalogue_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Defaults to all benchmarks listed in manifest.",
    )
    ap.add_argument(
        "--reference_benchmark",
        type=str,
        default=None,
        help="Benchmark whose anchor defines catalog.json routes (default: first of --benchmarks).",
    )
    ap.add_argument(
        "--dense_columns",
        type=str,
        choices=("measured_only", "all_legal_programs"),
        default="measured_only",
        help="Whether to filter dense JSONL routes by per-bench keep_mask (default) or include all programs.",
    )
    args = ap.parse_args()

    cat_dir: Path = args.catalogue_dir.resolve()
    out_dir: Path = args.out_dir.resolve()
    manifest_path = cat_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    prim_rows = _read_jsonl(cat_dir / manifest["primitives_path"])
    prim_rows.sort(key=lambda r: int(r["idx"]))
    prim_objs = [
        Primitive(str(r["kind"]), tuple(int(x) for x in r["args"])) for r in prim_rows
    ]

    joint_legal_path = cat_dir / manifest["joint"]["legal_programs_path"]
    legal_rows = _read_jsonl(joint_legal_path)
    legal_rows.sort(key=lambda r: int(r["idx"]))
    n_programs = len(legal_rows)
    if n_programs < 2:
        raise SystemExit("need at least noop + 1 route in joint legal_programs")

    bench_list = list(args.benchmarks) if args.benchmarks else list(manifest["benchmarks"].keys())
    ref_bench = args.reference_benchmark or bench_list[0]
    if ref_bench not in manifest["benchmarks"]:
        raise SystemExit(f"reference benchmark {ref_bench!r} not in manifest")

    def materialize_for_anchor(anchor: List[int], prim_indices: List[int]) -> List[int]:
        prog = [prim_objs[int(j)] for j in prim_indices]
        return [int(x) for x in apply_program(anchor, prog)]

    ref_anchor = [int(x) for x in manifest["benchmarks"][ref_bench]["anchor"]]
    selected_routes: List[List[int]] = []
    for row in legal_rows[1:]:
        prims = [int(x) for x in row.get("primitive_indices", [])]
        selected_routes.append(materialize_for_anchor(ref_anchor, prims))

    per_bench_catalog: Dict[str, List[Optional[List[int]]]] = {}
    for bench in bench_list:
        if bench not in manifest["benchmarks"]:
            raise SystemExit(f"unknown benchmark {bench!r}")
        anc = [int(x) for x in manifest["benchmarks"][bench]["anchor"]]
        rows_cat: List[Optional[List[int]]] = [None]
        for legal_row in legal_rows[1:]:
            prims = [int(x) for x in legal_row.get("primitive_indices", [])]
            rows_cat.append(materialize_for_anchor(anc, prims))
        per_bench_catalog[bench] = rows_cat

    out_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = out_dir / "catalog.json"
    catalog_path.write_text(json.dumps({"selected_routes": selected_routes}, indent=2))
    pbc_path = out_dir / "per_bench_catalog.json"
    pbc_path.write_text(json.dumps(per_bench_catalog, indent=2))

    anchor_path = out_dir / "anchor_seqs.json"
    anchor_payload = {b: [int(x) for x in manifest["benchmarks"][b]["anchor"]] for b in bench_list}
    anchor_path.write_text(json.dumps(anchor_payload, indent=2))

    staging = out_dir / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    dense_path = out_dir / "dense_joint.jsonl"

    n_lines = 0
    with dense_path.open("w") as dense_f:
        for bench in bench_list:
            bmeta = manifest["benchmarks"][bench]
            piv_src = Path(bmeta["pivot_residuals_path"])
            if not piv_src.is_file():
                raise SystemExit(f"missing pivot residuals for {bench}: {piv_src}")
            dst_link = staging / f"{bench}_pivot_residuals.pt"
            if dst_link.is_symlink() or dst_link.exists():
                dst_link.unlink()
            os.symlink(piv_src.resolve(), dst_link)

            drel = bmeta.get("dense_deltas_path")
            if not drel:
                raise SystemExit(f"manifest missing dense_deltas_path for {bench}")
            dpath = cat_dir / drel
            payload = torch.load(dpath, map_location="cpu", weights_only=False)
            delta_matrix = payload["delta_matrix"].float()
            Q, n_cols = delta_matrix.shape
            if n_cols != n_programs:
                raise SystemExit(
                    f"{bench}: delta_matrix columns {n_cols} != legal programs {n_programs}",
                )

            km_tensor: Optional[torch.Tensor] = None
            km_path = cat_dir / bmeta.get("dense_keep_mask_path", "")
            if km_path.is_file():
                km_payload = torch.load(km_path, map_location="cpu", weights_only=False)
                km_tensor = km_payload.get("keep_mask")
                if km_tensor is not None and int(km_tensor.numel()) != n_programs:
                    raise SystemExit(f"{bench}: keep_mask length mismatch")

            use_mask = args.dense_columns == "measured_only"
            for q in range(Q):
                rd: Dict[str, float] = {}
                for j in range(1, n_programs):
                    if use_mask and km_tensor is not None and float(km_tensor[j].item()) < 0.5:
                        continue
                    d = float(delta_matrix[q, j].item())
                    rd[str(j - 1)] = d
                rec = {
                    "benchmark_id": bench,
                    "question_id": int(q),
                    "route_deltas": rd,
                }
                dense_f.write(json.dumps(rec) + "\n")
                n_lines += 1

    meta_out = {
        "catalogue_dir": str(cat_dir),
        "catalogue_kind": manifest.get("catalogue_kind"),
        "reference_benchmark": ref_bench,
        "benchmarks": bench_list,
        "n_programs": n_programs,
        "n_selected_routes": len(selected_routes),
        "dense_lines": n_lines,
        "dense_columns": args.dense_columns,
        "staging_dir": str(staging),
    }
    (out_dir / "export_meta.json").write_text(json.dumps(meta_out, indent=2))

    print(f"Wrote {catalog_path}  |selected_routes|={len(selected_routes)}", flush=True)
    print(f"Wrote {pbc_path}  benches={list(per_bench_catalog.keys())}", flush=True)
    print(f"Wrote {anchor_path}", flush=True)
    print(f"Wrote {dense_path}  lines={n_lines}", flush=True)
    print(f"Staging symlinks in {staging}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
