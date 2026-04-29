#!/usr/bin/env python3
"""Run positive-mass catalog analysis + figures for compositional / merged dense data.

Reuses :mod:`dr-llm.experiments.plot_dense_catalog_mass` and
:mod:`dr-llm.data_prep.reduce_dense_catalog_by_mass` (same math as the legacy
fine-router pipeline). Accepts either:

* ``dense_deltas.jsonl`` + ``selected_catalog.json`` (typical after merge), or
* ``dense_deltas_matrix.pt`` + ``selected_catalog.json`` (tensor-only; JSONL
  is materialized next to ``--out_prefix``).

Optionally writes a mass-reduced JSONL + catalog (e.g. 0.95 of oracle mass).

Environment:

* ``DR_LLM_DIR`` — override path to ``generalized_transformer-2/dr-llm``
  (default: sibling of this repo under the workspace home).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence


def _flex_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_dr_llm() -> Path:
    env = os.environ.get("DR_LLM_DIR", "").strip()
    if env:
        return Path(env)
    return _flex_root().parent / "generalized_transformer-2" / "dr-llm"


def _load_dr_module(name: str, relpath: str) -> ModuleType:
    dr = _default_dr_llm()
    path = dr / relpath
    if not path.is_file():
        raise FileNotFoundError(f"expected dr-llm file at {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run(
    *,
    catalog_json: Path,
    out_prefix: Path,
    title: str,
    dense_jsonl: Optional[Path],
    dense_matrix_pt: Optional[Path],
    mass_fractions: Sequence[float],
    benchmark_id: Optional[str],
    require_mcts_for_reduce: bool,
) -> Dict[str, Any]:
    dr_root = str(_default_dr_llm())
    if dr_root not in sys.path:
        sys.path.insert(0, dr_root)

    plot = _load_dr_module("plot_dense_catalog_mass", "experiments/plot_dense_catalog_mass.py")
    mass_red = _load_dr_module("reduce_dense_catalog_by_mass", "data_prep/reduce_dense_catalog_by_mass.py")

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    stem = str(out_prefix)

    dense_path: Path
    if dense_matrix_pt is not None:
        meta = plot.materialize_dense_jsonl_from_matrix_pt(
            str(dense_matrix_pt),
            str(catalog_json),
            stem + "_materialized_from_pt.jsonl",
            benchmark_id=benchmark_id,
            sparse=True,
        )
        dense_path = Path(meta["materialized_jsonl"])
    else:
        if not dense_jsonl or not dense_jsonl.is_file():
            raise FileNotFoundError("provide --dense_jsonl or --dense_matrix_pt")
        dense_path = dense_jsonl

    report = plot.build_dense_mass_report(
        dense_jsonl=str(dense_path),
        catalog_json=str(catalog_json),
        title=title,
        out_prefix=stem,
    )

    reduce_runs: List[Dict[str, Any]] = []
    tag = str(catalog_json.parent.name)
    for frac in mass_fractions:
        if not (0.0 < frac <= 1.0):
            continue
        sub = out_prefix.parent / f"dense_mass_reduced_{tag}"
        sub.mkdir(parents=True, exist_ok=True)
        ftag = str(frac).replace(".", "p")
        oj = sub / f"dense_mass_{ftag}.jsonl"
        oc = sub / f"selected_catalog_mass_{ftag}.json"
        meta = mass_red.write_dense_mass_reduced(
            str(dense_path),
            str(catalog_json),
            float(frac),
            str(oj),
            str(oc),
            benchmark_ids=[benchmark_id] if benchmark_id else None,
            audit_mcts_rows=0,
            require_mcts_source=require_mcts_for_reduce,
        )
        reduce_runs.append(
            {
                "mass_fraction": frac,
                "out_jsonl": str(oj),
                "out_catalog": str(oc),
                "meta": meta,
            }
        )

    return {
        "title": title,
        "report": report,
        "mass_reductions": reduce_runs,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dense_jsonl", type=Path, default=None)
    src.add_argument("--dense_matrix_pt", type=Path, default=None)
    p.add_argument(
        "--catalog_json",
        type=Path,
        required=True,
        help="selected_catalog.json (unified or struct-only).",
    )
    p.add_argument(
        "--out_prefix",
        type=Path,
        required=True,
        help="Output stem (figures + _metrics.json next to it).",
    )
    p.add_argument("--title", type=str, default="")
    p.add_argument(
        "--benchmark_id",
        type=str,
        default=None,
        help="e.g. commonsenseqa — passed to mass reduction filter / materialized rows.",
    )
    p.add_argument(
        "--mass_fractions",
        type=str,
        default="0.95",
        help="Comma-separated greedy mass targets for reduced JSONL (empty = skip).",
    )
    p.add_argument(
        "--require_mcts_for_reduce",
        action="store_true",
        help="Pass through to write_dense_mass_reduced (fails if mcts_source missing).",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    fracs: List[float] = []
    if args.mass_fractions.strip():
        for s in args.mass_fractions.split(","):
            s = s.strip()
            if s:
                fracs.append(float(s))

    title = args.title or args.catalog_json.parent.name
    out = run(
        catalog_json=args.catalog_json,
        out_prefix=args.out_prefix,
        title=title,
        dense_jsonl=args.dense_jsonl,
        dense_matrix_pt=args.dense_matrix_pt,
        mass_fractions=fracs,
        benchmark_id=args.benchmark_id,
        require_mcts_for_reduce=bool(args.require_mcts_for_reduce),
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
