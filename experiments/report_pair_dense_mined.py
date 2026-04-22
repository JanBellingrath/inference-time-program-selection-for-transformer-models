"""Produce compositional-generalization report once ``pair_dense_mined`` training finishes.

Runs (1) :mod:`experiments.eval_compositional_generalization`,
(2) pair-level bootstrap on the new ``per_edge.jsonl``,
(3) paired bootstrap vs the existing ``pair_dense`` baseline (same held-out edges).

By default expects:
* checkpoint: ``compositional_runs/cg_exp/models/pair_dense_mined/compositional_router_best_commonsenseqa.pt``
* mined dense tensor (provenance only): ``compositional_runs/cg_exp/dense_mined/commonsenseqa_dense.pt``

Example::

    # After training completes (or from another machine once the checkpoint exists):
    python -m experiments.report_pair_dense_mined

    # Poll up to 2 hours for the checkpoint (e.g. training still running):
    python -m experiments.report_pair_dense_mined --wait-seconds 7200 --poll-interval 60
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.bootstrap_pair_metrics import DEFAULT_METRICS, run_paired, run_single
from experiments.eval_compositional_generalization import evaluate

logger = logging.getLogger("report_pair_dense_mined")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _wait_for_file(path: Path, *, wait_seconds: float, poll_interval: float) -> bool:
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if path.is_file():
            return True
        logger.info("waiting for %s (%.0fs left)", path, max(0, deadline - time.monotonic()))
        time.sleep(poll_interval)
    return path.is_file()


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=_REPO / "compositional_runs/cg_exp/models/pair_dense_mined"
        / "compositional_router_best_commonsenseqa.pt",
    )
    p.add_argument(
        "--catalogue_dir",
        type=Path,
        default=_REPO / "compositional_runs/csqa_compositional",
    )
    p.add_argument(
        "--holdout_split",
        type=Path,
        default=_REPO / "compositional_runs/cg_exp/holdout/holdout_split.json",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=_REPO / "compositional_runs/cg_exp/eval/pair_dense_mined",
    )
    p.add_argument(
        "--baseline_per_edge",
        type=Path,
        default=_REPO / "compositional_runs/cg_exp/eval/pair_dense/per_edge.jsonl",
        help="Prior pair+dense eval (for paired bootstrap).",
    )
    p.add_argument(
        "--mined_dense_pt",
        type=Path,
        default=_REPO / "compositional_runs/cg_exp/dense_mined/commonsenseqa_dense.pt",
        help="Recorded in the report for provenance only.",
    )
    p.add_argument("--edge_set", choices=["test", "val"], default="test")
    p.add_argument("--B", type=int, default=5000)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--wait-seconds", type=float, default=0.0,
                   help="If >0, poll until --checkpoint exists or timeout.")
    p.add_argument("--poll-interval", type=float, default=30.0)
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    ckpt: Path = args.checkpoint
    if args.wait_seconds > 0:
        if not _wait_for_file(ckpt, wait_seconds=args.wait_seconds, poll_interval=args.poll_interval):
            logger.error("checkpoint still missing after wait: %s", ckpt)
            return 2
    elif not ckpt.is_file():
        logger.error(
            "checkpoint not found: %s\nTrain pair_dense_mined first, or pass --wait-seconds.",
            ckpt,
        )
        return 2

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("evaluating checkpoint -> %s", ckpt)
    eval_out = evaluate(
        catalogue_dir=args.catalogue_dir,
        holdout_split_path=args.holdout_split,
        checkpoint_path=ckpt,
        output_dir=out_dir,
        edge_set=args.edge_set,
    )
    per_edge_path = out_dir / "per_edge.jsonl"

    logger.info("bootstrap (single model)")
    single = run_single(
        _read_jsonl(per_edge_path), list(DEFAULT_METRICS),
        B=args.B, seed=args.seed, alpha=0.05,
    )
    boot_path = out_dir / "bootstrap.json"
    with open(boot_path, "w") as f:
        json.dump(single, f, indent=2)

    paired: Optional[Dict[str, Any]] = None
    paired_path = out_dir / "paired_vs_pair_dense.json"
    if args.baseline_per_edge.is_file():
        logger.info("paired bootstrap vs %s", args.baseline_per_edge)
        paired = run_paired(
            _read_jsonl(per_edge_path),
            _read_jsonl(args.baseline_per_edge),
            list(DEFAULT_METRICS),
            B=args.B, seed=args.seed + 1, alpha=0.05,
            label_a="pair_dense_mined",
            label_b="pair_dense",
        )
        with open(paired_path, "w") as f:
            json.dump(paired, f, indent=2)
    else:
        logger.warning("baseline per_edge missing; skip paired bootstrap: %s", args.baseline_per_edge)

    mined_meta: Dict[str, Any] = {}
    if args.mined_dense_pt.is_file():
        try:
            import torch

            payload = torch.load(args.mined_dense_pt, map_location="cpu", weights_only=False)
            dm = payload.get("delta_matrix")
            mined_meta = {
                "mined_dense_pt": str(args.mined_dense_pt),
                "delta_matrix_shape": list(dm.shape) if dm is not None else None,
                "source_mined_pt": payload.get("source_mined_pt"),
            }
        except Exception as ex:  # pragma: no cover
            mined_meta = {"mined_dense_pt": str(args.mined_dense_pt), "load_error": repr(ex)}
    else:
        mined_meta = {"mined_dense_pt": str(args.mined_dense_pt), "note": "file not found"}

    report: Dict[str, Any] = {
        "schema_version": 1,
        "checkpoint": str(ckpt),
        "mined_dense": mined_meta,
        "eval_summary": eval_out.get("summary"),
        "bootstrap_path": str(boot_path),
        "bootstrap": single,
        "paired_vs_pair_dense_path": str(paired_path) if paired is not None else None,
        "paired_vs_pair_dense": paired,
    }
    report_path = out_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("wrote %s", report_path)
    print(json.dumps({"report": str(report_path), "eval_summary": report["eval_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
