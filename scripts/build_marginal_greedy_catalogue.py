#!/usr/bin/env python3
"""Build a compositional catalogue restricted to K programs in marginal-greedy order.

Matches the per-question objective in ``dr-llm/experiments/plot_dense_catalog_mass.py``:
iteratively add the program column that maximizes sum_i max(0, X[i,j] - best_i).

Writes a new ``catalogue_dir`` with:
  legal_programs/{bench}.jsonl (K rows, idx 0..K-1),
  incidence + pair_incidence,
  observed (MCTS) with indices remapped into 0..K-1, rows with no obs dropped,
  dense_deltas/{bench}.pt with K columns aligned to those rows.

Example::

  python -m scripts.build_marginal_greedy_catalogue \\
    --base_catalogue_dir /path/to/fine_routing_data_commonsenseqa_mcts_compositional_assign \\
    --dense_matrix_pt compositional_runs/csqa_nonft_unified95/dense_deltas_matrix_legal_assign1384.pt \\
    --split_json splits/csqa_nonft_canonical_split.json \\
    --benchmark commonsenseqa \\
    --k 22 \\
    --out_dir compositional_runs/csqa_nonft_marginal_greedy22
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from data_prep.build_compositional_catalogues import (  # noqa: E402
    _save_incidence,
    _save_pair_incidence,
    build_incidence_tensor,
    build_pair_incidence_tensor,
)

logger = logging.getLogger("build_marginal_greedy_catalogue")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")


def greedy_order(X: np.ndarray, max_k: int) -> List[int]:
    n, m = X.shape
    best = np.zeros(n, dtype=np.float64)
    rem = np.ones(m, dtype=bool)
    sel: List[int] = []
    diff_buf = np.empty((n, m), dtype=np.float64)
    gains = np.empty(m, dtype=np.float64)
    for _ in range(max_k):
        np.subtract(X, best[:, None], out=diff_buf)
        np.maximum(diff_buf, 0.0, out=diff_buf)
        np.sum(diff_buf, axis=0, out=gains)
        gains[~rem] = -1.0
        j = int(gains.argmax())
        if gains[j] <= 1e-12:
            break
        sel.append(j)
        np.maximum(best, X[:, j], out=best)
        rem[j] = False
    return sel


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_catalogue_dir", type=Path, required=True)
    p.add_argument("--dense_matrix_pt", type=Path, required=True)
    p.add_argument("--split_json", type=Path, required=True)
    p.add_argument("--benchmark", type=str, default="commonsenseqa")
    p.add_argument("--k", type=int, default=22)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    bench = args.benchmark
    K = int(args.k)
    if K < 1:
        raise SystemExit("--k must be >= 1")

    base = args.base_catalogue_dir
    with open(base / "manifest.json") as f:
        src_man = json.load(f)
    info = src_man["benchmarks"][bench]
    n_src = int(info["n_legal_programs"])
    M = int(src_man["M"])

    split = json.loads(args.split_json.read_text())
    train = {int(x) for x in split["benchmarks"][bench]["train_question_ids"]}

    raw = torch.load(args.dense_matrix_pt, map_location="cpu", weights_only=True)
    dm = raw["delta_matrix"].float()
    if int(dm.shape[1]) != n_src:
        raise SystemExit(
            f"dense has {dm.shape[1]} cols but manifest n_legal={n_src}"
        )
    idx = torch.tensor(sorted(x for x in train if 0 <= x < dm.shape[0]), dtype=torch.long)
    D = dm.index_select(0, idx).numpy()
    X = np.maximum(D, 0.0)
    order_full = greedy_order(X, n_src)[:K]
    if len(order_full) < K:
        raise SystemExit(
            f"greedy_order returned {len(order_full)} programs; need {K}"
        )
    old_to_new: Dict[int, int] = {old: j for j, old in enumerate(order_full)}

    # --- legal programs: row j = old program order_full[j] ---
    src_legal = base / info["legal_programs_path"]
    legal_rows = _read_jsonl(src_legal)
    if len(legal_rows) != n_src:
        raise SystemExit(
            f"expected {n_src} legal rows in {src_legal}, got {len(legal_rows)}"
        )
    new_legal: List[Dict[str, Any]] = []
    for j, old in enumerate(order_full):
        r = legal_rows[old]
        prims = [int(x) for x in r.get("primitive_indices", [])]
        new_legal.append(
            {
                "idx": j,
                "length": len(prims),
                "primitive_indices": prims,
                "key": r.get("key", ""),
            }
        )

    prims_list = [tuple(int(x) for x in row["primitive_indices"]) for row in new_legal]
    a_idx, a_val, a_shape, lengths = build_incidence_tensor(prims_list, M)
    pair_index, b_idx, b_val, b_shape = build_pair_incidence_tensor(prims_list)

    out = args.out_dir
    (out / "legal_programs").mkdir(parents=True, exist_ok=True)
    (out / "incidence").mkdir(parents=True, exist_ok=True)
    (out / "pair_incidence").mkdir(parents=True, exist_ok=True)
    (out / "observed").mkdir(parents=True, exist_ok=True)
    (out / "dense_deltas").mkdir(parents=True, exist_ok=True)
    (out / "dense_masks").mkdir(parents=True, exist_ok=True)

    _write_jsonl(out / "legal_programs" / f"{bench}.jsonl", new_legal)
    _save_incidence(out / "incidence" / f"{bench}.pt", a_idx, a_val, a_shape, lengths)
    _save_pair_incidence(
        out / "pair_incidence" / f"{bench}.pt",
        pair_index,
        b_idx,
        b_val,
        b_shape,
    )

    if "anchor_utilities" not in raw:
        raise SystemExit("dense_matrix_pt must include anchor_utilities [Q]")

    col_idx = torch.tensor(order_full, dtype=torch.long)
    dm_sub = dm.index_select(1, col_idx).contiguous()
    au = raw["anchor_utilities"].float()
    payload_out: Dict[str, Any] = {
        "delta_matrix": dm_sub,
        "anchor_utilities": au,
        "n_programs": int(K),
        "catalogue": "marginal_greedy",
    }
    dmb_o = raw.get("delta_matrix_binary")
    aa_o = raw.get("anchor_accuracies")
    if (dmb_o is not None) ^ (aa_o is not None):
        raise SystemExit(
            "dense_matrix_pt must include both delta_matrix_binary and anchor_accuracies "
            "or neither"
        )
    if dmb_o is not None:
        dmb = dmb_o.float()
        aa_bin = aa_o.float()
        if dmb.shape != dm.shape:
            raise SystemExit(
                f"delta_matrix_binary shape {tuple(dmb.shape)} != delta_matrix {tuple(dm.shape)}"
            )
        if aa_bin.shape != au.shape:
            raise SystemExit(
                f"anchor_accuracies shape {tuple(aa_bin.shape)} != anchor_utilities {tuple(au.shape)}"
            )
        payload_out["delta_matrix_binary"] = dmb.index_select(1, col_idx).contiguous()
        payload_out["anchor_accuracies"] = aa_bin
    torch.save(
        payload_out,
        out / "dense_deltas" / f"{bench}.pt",
    )

    # keep mask: all 1s for downstream
    keep = torch.ones(K, dtype=torch.float32)
    torch.save(
        {"keep_mask": keep, "n_joint": K, "n_measured": K},
        out / "dense_masks" / f"{bench}.pt",
    )

    # observed: MCTS from base
    src_obs = base / info["observed_path"]
    obs = _read_jsonl(src_obs)
    remapped: List[Dict[str, Any]] = []
    for rec in obs:
        n_i: List[int] = []
        n_d: List[float] = []
        for i, d in zip(rec.get("obs_indices", []), rec.get("obs_deltas", [])):
            oi = int(i)
            if oi in old_to_new:
                n_i.append(int(old_to_new[oi]))
                n_d.append(float(d))
        if not n_i:
            continue
        nrec = dict(rec)
        nrec["obs_indices"] = n_i
        nrec["obs_deltas"] = n_d
        nrec["n_obs"] = len(n_i)
        remapped.append(nrec)

    _write_jsonl(out / "observed" / f"{bench}.jsonl", remapped)
    logger.info("observed: %d / %d questions kept (>=1 obs in the K-set)", len(remapped), len(obs))

    shutil.copy2(base / src_man["primitives_path"], out / "primitives.jsonl")

    meta = {
        "greedy_marginal_order_source_indices": [int(x) for x in order_full],
        "k": K,
        "n_train_questions_dense": int(X.shape[0]),
    }
    with open(out / "marginal_greedy_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # manifest
    n_pairs = int(pair_index.shape[0]) if pair_index.numel() else 0
    new_bench: Dict[str, Any] = dict(info)
    new_bench["n_legal_programs"] = K
    new_bench["n_legal_dropped_unknown_primitive"] = 0
    new_bench["incidence_path"] = f"incidence/{bench}.pt"
    new_bench["pair_incidence_path"] = f"pair_incidence/{bench}.pt"
    new_bench["n_legal_pairs"] = n_pairs
    new_bench["legal_programs_path"] = f"legal_programs/{bench}.jsonl"
    new_bench["observed_path"] = f"observed/{bench}.jsonl"
    new_bench["dense_deltas_path"] = f"dense_deltas/{bench}.pt"
    new_bench["dense_keep_mask_path"] = f"dense_masks/{bench}.pt"
    new_bench["n_measured_joint_rows"] = K
    new_bench["n_questions_kept"] = len(remapped)
    new_bench["source_catalogue_dir"] = str(base)
    new_bench["marginal_greedy"] = True

    new_man = {
        "schema_version": 1,
        "source_catalogue_dir": str(base),
        "output_dir": str(out),
        "geometry": src_man.get("geometry", {}),
        "filter": src_man.get("filter", {}),
        "M": M,
        "primitives_path": "primitives.jsonl",
        "catalogue_kind": "marginal_greedy_subset",
        "benchmarks": {bench: new_bench},
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(new_man, f, indent=2)

    logger.info("wrote %s  (K=%d programs)", out, K)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
