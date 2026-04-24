#!/usr/bin/env python3
"""Expand a unique-route dense matrix to full legal-program width [Q, N_legal].

``merge_dense_increment`` / ``dense_reevaluation`` store one column per *unique*
layer route, while :class:`LegalCatalogue` has one row per legal program (many
programs can yield the same route).  Training checks
``delta_matrix.shape[1] == catalogue.n_programs`` — this script duplicates
columns so each legal program index gets the Δ vector for its route.

Example::

    python scripts/materialize_dense_legal_programs.py \\
      --compositional_dir .../fine_routing_data_..._compositional_assign \\
      --unified_matrix .../decode_compositional_unified/dense_deltas_matrix.pt \\
      --selected_catalog .../decode_compositional_unified/selected_catalog.json \\
      --output .../decode_compositional_unified/dense_deltas_matrix_legal.pt \\
      --bench commonsenseqa
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.edit_dsl import Primitive, apply_program  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compositional_dir", type=Path, required=True)
    ap.add_argument("--unified_matrix", type=Path, required=True)
    ap.add_argument("--selected_catalog", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bench", type=str, default="commonsenseqa")
    args = ap.parse_args()

    comp = args.compositional_dir
    manifest = json.loads((comp / "manifest.json").read_text())
    bench_meta = manifest["benchmarks"][args.bench]
    anchor = [int(x) for x in bench_meta["anchor"]]
    prim_rows = _read_jsonl(comp / manifest["primitives_path"])
    prim_rows.sort(key=lambda r: int(r["idx"]))
    prim_objs = [
        Primitive(str(r["kind"]), tuple(int(x) for x in r["args"])) for r in prim_rows
    ]
    legal_rows = _read_jsonl(comp / bench_meta["legal_programs_path"])
    legal_rows.sort(key=lambda r: int(r["idx"]))

    cat = json.loads(args.selected_catalog.read_text())
    uni_routes = [tuple(int(x) for x in r) for r in cat["selected_routes"]]
    route_to_col = {r: i for i, r in enumerate(uni_routes)}

    payload = torch.load(args.unified_matrix, map_location="cpu", weights_only=False)
    delta = payload["delta_matrix"].float()
    anchor_u = payload["anchor_utilities"].float()
    delta_bin = payload.get("delta_matrix_binary")
    if delta_bin is not None:
        delta_bin = delta_bin.float()
    Q, R_uni = delta.shape
    if R_uni != len(uni_routes):
        raise SystemExit(
            f"unified matrix has R={R_uni} but selected_catalog has {len(uni_routes)} routes"
        )

    n_legal = len(legal_rows)
    legal_routes: list[tuple[int, ...]] = []
    for r in legal_rows:
        prog = [prim_objs[int(j)] for j in r["primitive_indices"]]
        route = tuple(int(x) for x in apply_program(anchor, prog))
        if route not in route_to_col:
            raise SystemExit(f"route from legal idx {r['idx']!r} not in unified catalogue")
        legal_routes.append(route)

    cols = torch.tensor([route_to_col[rt] for rt in legal_routes], dtype=torch.long)
    expanded = delta.index_select(1, cols).contiguous()
    assert expanded.shape == (Q, n_legal)

    routes_out = [list(r) for r in legal_routes]
    out_payload = {
        "delta_matrix": expanded,
        "anchor_utilities": anchor_u,
        "routes": routes_out,
        "benchmarks": list(payload.get("benchmarks", [args.bench])),
        "score_mode": payload.get("score_mode", "continuous"),
        "source_unified_matrix": str(args.unified_matrix.resolve()),
        "n_legal_programs": n_legal,
        "n_unique_routes": R_uni,
    }
    if delta_bin is not None:
        expanded_bin = delta_bin.index_select(1, cols).contiguous()
        out_payload["delta_matrix_binary"] = expanded_bin
    if "anchor_accuracies" in payload:
        out_payload["anchor_accuracies"] = payload["anchor_accuracies"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, args.output)
    print(
        f"Wrote {args.output}  shape={tuple(expanded.shape)}  "
        f"(duplicated {R_uni} unique routes -> {n_legal} legal columns)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
