#!/usr/bin/env python3
"""Resume a finished W&B run and log LLM downstream metrics (gain in pp).

Reads ``wandb_run_info.json`` (written by training.train_compositional_router
when --wandb is set), runs
:func:`experiments.eval_compositional_downstream.evaluate_checkpoint`, and
logs ``unconditional_gain_pp`` (router minus anchor accuracy, in percentage
points) to the same run under ``llm_eval/*``.

This is the interpretable "delta vs baseline" metric; dense training
``mean_uplift`` is a log-prob proxy, not pp.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch  # noqa: F401  # experiments import chain

from experiments.eval_compositional_downstream import evaluate_checkpoint

logger = logging.getLogger("log_llm_uplift_wandb")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--wandb_run_info",
        type=Path,
        help="Path to wandb_run_info.json from the training output_dir.",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        help="Training output_dir (if set, uses output_dir/wandb_run_info.json).",
    )
    p.add_argument("--catalogue_dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split_json", type=Path, required=True,
                   help="Val question ids; must match training split for fair comparison.")
    p.add_argument("--benchmarks", nargs="*", default=None)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--ft_adapter_path", type=Path, default=None,
                   help="Omit for non-FT (base instruct only).")
    p.add_argument("--max_samples_per_bench", type=int, default=None)
    p.add_argument("--output_json", type=Path, default=None,
                   help="Optional local JSON of full eval result.")
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    if args.wandb_run_info is not None:
        info_path = args.wandb_run_info
    elif args.output_dir is not None:
        info_path = args.output_dir / "wandb_run_info.json"
    else:
        raise SystemExit("Pass --wandb_run_info or --output_dir (with wandb_run_info.json)")

    if not info_path.is_file():
        raise SystemExit(f"missing {info_path}")

    with open(info_path) as f:
        winfo: Dict[str, Any] = json.load(f)
    run_id = winfo["id"]
    project = winfo.get("project")
    entity = winfo.get("entity")

    try:
        import wandb
    except ImportError:  # pragma: no cover
        raise SystemExit("wandb required for this script") from None

    summary = evaluate_checkpoint(
        catalogue_dir=args.catalogue_dir,
        checkpoint_path=args.checkpoint,
        split_json=args.split_json,
        benchmarks=args.benchmarks,
        model_name=args.model_name,
        ft_adapter_path=args.ft_adapter_path,
        data_split="validation",
        max_samples_per_bench=args.max_samples_per_bench,
        output_json=args.output_json,
    )
    u_pp = float(summary.get("unconditional_gain_pp", float("nan")))

    init_kw: Dict[str, Any] = {
        "id": run_id,
        "project": project,
        "resume": "allow",
    }
    if entity:
        init_kw["entity"] = entity
    run = wandb.init(**init_kw)
    prefix = "llm_eval/validation"
    log_payload: Dict[str, float] = {
        f"{prefix}/unconditional_gain_pp": u_pp,
        f"{prefix}/router_acc": float(summary.get("router_acc", float("nan"))),
        f"{prefix}/anchor_acc": float(summary.get("anchor_acc", float("nan"))),
        f"{prefix}/n_questions": float(summary.get("n", 0)),
    }
    for bench, row in (summary.get("per_bench") or {}).items():
        if not isinstance(row, dict):
            continue
        b = f"{prefix}/{bench}"
        log_payload[f"{b}/uplift_pp"] = float(row.get("uplift_pp", float("nan")))
        log_payload[f"{b}/n"] = float(row.get("n", 0))
    run.log(log_payload)
    run.finish()
    print(json.dumps({"logged": log_payload, "summary": summary}, indent=2, default=str))
    logger.info("Logged unconditional_gain_pp=%.4f to run %s", u_pp, run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
