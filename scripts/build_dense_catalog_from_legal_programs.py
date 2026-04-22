#!/usr/bin/env python3
"""Build a ``selected_catalog.json`` for ``data_prep/dense_reevaluation.py`` by
applying every legal compositional program to the benchmark anchor.

Usage:
    python scripts/build_dense_catalog_from_legal_programs.py \
        --manifest compositional_runs/csqa_compositional/manifest.json \
        --benchmark commonsenseqa \
        --output catalogs/csqa_compositional_179/selected_catalog.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `core` import when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.edit_dsl import Primitive, apply_program  # noqa: E402


def _load_primitives(path: Path) -> list[Primitive]:
    prims: list[Primitive] = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            prims.append(Primitive(kind=row["kind"], args=tuple(row["args"])))
    return prims


def _load_legal_programs(path: Path) -> list[list[int]]:
    progs: list[list[int]] = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            progs.append(list(row["primitive_indices"]))
    return progs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    bench_meta = manifest["benchmarks"][args.benchmark]

    base_dir = manifest_path.parent
    primitives_path = base_dir / manifest["primitives_path"]
    legal_path = base_dir / bench_meta["legal_programs_path"]
    anchor = list(bench_meta["anchor"])

    primitives = _load_primitives(primitives_path)
    legal = _load_legal_programs(legal_path)

    routes: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for prim_indices in legal:
        program = [primitives[j] for j in prim_indices]
        seq = apply_program(anchor, program)
        key = tuple(seq)
        if key in seen:
            continue
        seen.add(key)
        routes.append([int(x) for x in seq])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected_routes": routes,
        "anchor": anchor,
        "benchmark": args.benchmark,
        "n_legal_programs": len(legal),
        "n_unique_routes": len(routes),
        "source_manifest": str(manifest_path),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(
        f"Wrote {out_path} with {len(routes)} unique routes "
        f"(from {len(legal)} legal programs over {len(primitives)} primitives)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
