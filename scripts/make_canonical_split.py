#!/usr/bin/env python3
"""Emit a canonical train/val/test split over a compositional observed.jsonl.

The same split must be used by (a) catalogue mass-coverage pruning — so the
catalogue is chosen from train questions only — and (b) the training code —
so what the router never sees during training is exactly what gets held out
by the catalogue builder. This tool is that single source of truth.

Split procedure mirrors
:func:`training.train_compositional_router._split_indices` so, given the
same ``seed`` / ``val_fraction`` / record ordering, it produces exactly the
same train/val indices. A test hold-out carved off train mirrors
``_carve_test_from_train`` (same permutation seed transform).

Output JSON::

  {
    "seed": 42,
    "val_fraction": 0.15,
    "train_test_holdout_count": 0,
    "benchmarks": {
      "commonsenseqa": {
        "n_total": 9287,
        "train_question_ids": [...],
        "val_question_ids":   [...],
        "test_question_ids":  [...]
      }
    }
  }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch


def _split_indices(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g).tolist()
    val_size = max(1, int(n * val_fraction)) if n > 1 else 0
    return perm[val_size:], perm[:val_size]


def _carve_test_from_train(
    train_idx: List[int], *, holdout_count: int, seed: int,
) -> tuple[List[int], List[int]]:
    if holdout_count <= 0 or not train_idx:
        return list(train_idx), []
    take = min(int(holdout_count), max(0, len(train_idx) - 1))
    if take <= 0:
        return list(train_idx), []
    g = torch.Generator().manual_seed(int(seed) * 1009 + 7)
    perm = torch.randperm(len(train_idx), generator=g).tolist()
    pick = set(perm[:take])
    test_idx = [train_idx[i] for i in range(len(train_idx)) if i in pick]
    kept = [train_idx[i] for i in range(len(train_idx)) if i not in pick]
    return kept, test_idx


def _load_question_ids(observed_path: Path) -> List[int]:
    qids: List[int] = []
    with open(observed_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qids.append(int(rec.get("question_id", rec.get("residual_idx"))))
    return qids


def make_split(
    observed_paths: Dict[str, Path],
    *,
    seed: int,
    val_fraction: float,
    train_test_holdout_count: int = 0,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "train_test_holdout_count": int(train_test_holdout_count),
        "benchmarks": {},
    }
    for bench, path in observed_paths.items():
        qids = _load_question_ids(Path(path))
        n = len(qids)
        train_idx, val_idx = _split_indices(n, val_fraction, seed)
        train_idx, test_idx = _carve_test_from_train(
            train_idx, holdout_count=train_test_holdout_count, seed=seed,
        )
        out["benchmarks"][bench] = {
            "n_total": n,
            "observed_path": str(path),
            "train_question_ids": sorted(int(qids[i]) for i in train_idx),
            "val_question_ids": sorted(int(qids[i]) for i in val_idx),
            "test_question_ids": sorted(int(qids[i]) for i in test_idx),
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--observed", action="append", required=True,
                   help="bench=path pair; pass multiple times for joint splits.")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--train_test_holdout_count", type=int, default=0)
    args = p.parse_args()

    observed_paths: Dict[str, Path] = {}
    for item in args.observed:
        if "=" not in item:
            raise SystemExit(f"--observed expects bench=path, got: {item!r}")
        bench, path = item.split("=", 1)
        observed_paths[bench.strip()] = Path(path.strip())

    split = make_split(
        observed_paths,
        seed=args.seed,
        val_fraction=args.val_fraction,
        train_test_holdout_count=args.train_test_holdout_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(split, f, indent=2)
    for bench, info in split["benchmarks"].items():
        print(
            f"[{bench}] n={info['n_total']} "
            f"train={len(info['train_question_ids'])} "
            f"val={len(info['val_question_ids'])} "
            f"test={len(info['test_question_ids'])} "
            f"-> {args.output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
