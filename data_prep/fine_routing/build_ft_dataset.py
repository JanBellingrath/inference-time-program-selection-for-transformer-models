#!/usr/bin/env python3
"""Build per-question fine-grained MCTS dataset on fine-tuned models.

For each benchmark:
  1.  Load the benchmark-specific LoRA adapter (from a prior ft_study run).
  2.  Merge adapter into base → flexible-layer model.
  3.  Run per-question MCTS anchored on the **identity** layer sequence
      (FT was trained on default order) with the full benchmark-level
      search space (radius=5, max_swaps=24) restricted to layers after
      the pivot (editable_start).
  4.  Save ``{benchmark}.jsonl`` + ``{benchmark}_pivot_residuals.pt``
      in the same format consumed by ``train_fine_router.py`` /
      ``train_joint_router.py``.

Usage
-----
    cd dr-llm
    For auto-retry and optional notify-on-crash, run ``python -m
    data_prep.supervise_ft_fine_routing_dataset`` (see that module's docstring).

    python -m data_prep.build_ft_fine_routing_dataset \\
        --model_name Qwen/Qwen2.5-0.5B-Instruct \\
        --ft_results_dir ft_study_results_v7 \\
        --seed 42 \\
        --benchmarks boolq commonsenseqa \\
        --mcts_num_simulations 5000 \\
        --output_dir fine_routing_data_ft
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Optional

import torch

from core.benchmark_mcts import seq_to_layers
from core.flexible_models import get_is_instruct
from core.permutation_mcts import prepare_arc_data
from data_prep.fine_routing.build_dataset import build_dataset_mcts_for_benchmark
from ft_study.research_with_train_data import find_adapter_path as _find_adapter_path
from ft_study.trainer import load_ft_model_for_inference
from pipeline.forward import get_pivot_residual as _get_pivot_residual
from routers.fine_routing_config import FineRoutingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def find_adapter_path(
    base_results_dir: str,
    dataset_name: str,
    seed: int,
    arm: str = "ft_only",
) -> Optional[str]:
    """Locate a saved adapter, falling back to the latest checkpoint.

    First tries :func:`ft_study.research_with_train_data.find_adapter_path`
    (looks for ``final_adapter`` or the path stored in ``result.json``).
    If that returns None (e.g. training was interrupted), scans checkpoint
    directories for the highest-numbered one that contains an adapter.
    """
    path = _find_adapter_path(base_results_dir, dataset_name, seed, arm)
    if path is not None:
        return path

    ckpt_base = os.path.join(
        base_results_dir, dataset_name, f"seed_{seed}", arm, "checkpoints",
    )
    if not os.path.isdir(ckpt_base):
        return None

    candidates = []
    for entry in sorted(os.listdir(ckpt_base), reverse=True):
        d = os.path.join(ckpt_base, entry)
        if os.path.isfile(os.path.join(d, "adapter_config.json")):
            candidates.append(d)
    if candidates:
        logger.info(
            "No final_adapter for %s seed=%d; using checkpoint %s",
            dataset_name, seed, os.path.basename(candidates[0]),
        )
        return candidates[0]
    return None


# ---------------------------------------------------------------------------
# Wrapper shim: makes a merged FT model look like FlexibleModelWrapper
# ---------------------------------------------------------------------------

class FTFlexibleModelWrapper:
    """Duck-typed :class:`FlexibleModelWrapper` backed by a merged FT model.

    Provides the interface that :func:`build_dataset_mcts_for_benchmark`
    expects (model, tokenizer, num_layers, prepare_prompt,
    get_pivot_residual) without re-loading through HuggingFace.
    """

    def __init__(self, model, tokenizer, num_layers: int, model_name: str):
        self.model = model
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        self.model_name = model_name
        self.is_instruct = get_is_instruct(model_name)
        self.default_layer_indices = list(range(num_layers))

    @classmethod
    def from_ft_adapter(
        cls,
        model_name: str,
        adapter_path: str,
        rank: int = 0,
    ) -> "FTFlexibleModelWrapper":
        model, tokenizer, num_layers = load_ft_model_for_inference(
            model_name, adapter_path, layer_sequence=None, rank=rank,
        )
        return cls(model, tokenizer, num_layers, model_name)

    def prepare_prompt(self, query: str, system_prompt: str = None) -> str:
        if not self.is_instruct:
            return query
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})
        kwargs = {}
        if "qwen3" in self.model_name.lower():
            kwargs["enable_thinking"] = False
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs,
        )

    def get_pivot_residual(
        self,
        text: str,
        layer_indices: List[int],
        pivot_layer: int,
        system_prompt: Optional[str] = None,
    ) -> torch.Tensor:
        return _get_pivot_residual(
            self, text, layer_indices, pivot_layer, system_prompt=system_prompt,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build fine-grained MCTS routing dataset on fine-tuned models",
    )
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument(
        "--ft_results_dir", type=str, default=None,
        help="ft_study results directory (contains {benchmark}/seed_{seed}/{arm}/ trees)",
    )
    p.add_argument(
        "--adapter_path", type=str, default=None,
        help="Direct path to a single LoRA adapter dir (overrides --ft_results_dir)",
    )
    p.add_argument("--source_arm", type=str, default="ft_only")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--benchmarks", nargs="+",
        default=["boolq", "commonsenseqa"],
    )
    p.add_argument("--output_dir", type=str, default="fine_routing_data_ft")
    p.add_argument("--data_split", type=str, default="train")

    # Search space (larger, benchmark-level defaults)
    p.add_argument("--swap_radius", type=int, default=5)
    p.add_argument("--max_local_edits", type=int, default=24)
    p.add_argument("--pivot_layer", type=int, default=None)
    p.add_argument("--editable_start", type=int, default=None)

    # Per-question MCTS
    p.add_argument("--mcts_num_simulations", type=int, default=64)
    p.add_argument("--mcts_exploration_constant", type=float, default=1.8)
    p.add_argument("--mcts_pw_C", type=float, default=1.0)
    p.add_argument("--mcts_pw_alpha", type=float, default=0.5)

    # Scoring / target
    p.add_argument("--delta_clip", type=float, default=1.0)
    p.add_argument("--target_beta", type=float, default=5.0)
    p.add_argument("--gate_tau", type=float, default=0.0)
    p.add_argument("--use_continuous_scoring", action="store_true")

    # Runtime
    p.add_argument("--gpu_rank", type=int, default=0)
    p.add_argument("--max_questions", type=int, default=None)
    p.add_argument("--save_interval", type=int, default=200)
    p.add_argument("--resume", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.ft_results_dir is None and args.adapter_path is None:
        logger.error("Provide either --ft_results_dir or --adapter_path")
        sys.exit(1)

    cfg = FineRoutingConfig(model_name=args.model_name)
    cfg.output_dir = args.output_dir
    cfg.data_split = args.data_split
    cfg.swap_radius = args.swap_radius
    cfg.max_local_edits = args.max_local_edits
    cfg.delta_clip = args.delta_clip
    cfg.target_beta = args.target_beta
    cfg.gate_tau = args.gate_tau
    cfg.gpu_rank = args.gpu_rank
    cfg.use_mcts = True
    cfg.mcts_num_simulations = args.mcts_num_simulations
    cfg.mcts_exploration_constant = args.mcts_exploration_constant
    cfg.mcts_pw_C = args.mcts_pw_C
    cfg.mcts_pw_alpha = args.mcts_pw_alpha
    cfg.use_continuous_scoring = args.use_continuous_scoring
    cfg.benchmarks = args.benchmarks
    if args.pivot_layer is not None:
        cfg.pivot_layer = args.pivot_layer
    if args.editable_start is not None:
        cfg.editable_start = args.editable_start

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    is_instruct = get_is_instruct(args.model_name)
    all_records = []

    for bench in args.benchmarks:
        # --- locate adapter for this benchmark ---
        if args.adapter_path is not None:
            adapter_path = args.adapter_path
        else:
            adapter_path = find_adapter_path(
                args.ft_results_dir, bench, args.seed, arm=args.source_arm,
            )
        if adapter_path is None:
            logger.warning(
                "No adapter found for %s (seed=%d, arm=%s) in %s — skipping",
                bench, args.seed, args.source_arm, args.ft_results_dir,
            )
            continue
        logger.info("Benchmark %s: adapter = %s", bench, adapter_path)

        # --- load FT model ---
        wrapper = FTFlexibleModelWrapper.from_ft_adapter(
            args.model_name, adapter_path, rank=args.gpu_rank,
        )
        num_layers = wrapper.num_layers

        # Identity anchor: FT was trained on default layer order
        anchor_seq = list(range(num_layers))
        logger.info(
            "Anchor = identity [0..%d], pivot=%d, editable_start=%d, "
            "radius=%d, max_swaps=%d, sims/q=%d",
            num_layers - 1, cfg.pivot_layer, cfg.editable_start,
            cfg.swap_radius, cfg.max_local_edits, cfg.mcts_num_simulations,
        )

        # --- load questions ---
        samples = prepare_arc_data(bench, is_instruct=is_instruct, split=cfg.data_split)
        if args.max_questions is not None:
            samples = samples[:args.max_questions]
        if not samples:
            logger.warning("No samples for %s — skipping", bench)
            del wrapper
            torch.cuda.empty_cache()
            continue

        # --- run per-question MCTS ---
        t0 = time.time()
        result = build_dataset_mcts_for_benchmark(
            cfg, wrapper, bench, anchor_seq, samples,
            output_dir=args.output_dir,
            save_interval=args.save_interval,
            resume=args.resume,
        )
        elapsed = time.time() - t0

        n_new = len(result["records"])
        prev = result.get("resumed_from", 0)
        n_total = prev + n_new
        gate_pos = sum(1 for r in result["records"] if r["gate_label"] == 1)
        logger.info(
            "  %s done: +%d questions (total %d), gate_positive=%d (%.1f%%), %.1fs",
            bench, n_new, n_total, gate_pos,
            100 * gate_pos / max(n_new, 1), elapsed,
        )
        all_records.extend(result["records"])

        # --- free GPU for next benchmark ---
        del wrapper
        gc.collect()
        torch.cuda.empty_cache()

    # --- summary ---
    total = len(all_records)
    total_gp = sum(1 for r in all_records if r["gate_label"] == 1)
    logger.info(
        "Dataset complete: %d questions, %d gate-positive (%.1f%%)",
        total, total_gp, 100 * total_gp / max(total, 1),
    )

    catalog = {
        "_search_mode": "mcts",
        "_model_is_finetuned": True,
        "_anchor": "identity",
        "_mcts_num_simulations": cfg.mcts_num_simulations,
        "_swap_radius": cfg.swap_radius,
        "_max_swaps": cfg.max_local_edits,
        "_editable_start": cfg.editable_start,
    }
    catalog_path = os.path.join(args.output_dir, "deviation_catalog.json")
    with open(catalog_path, "w") as f:
        json.dump(catalog, f, indent=2)
    logger.info("Deviation catalog saved to %s", catalog_path)


if __name__ == "__main__":
    main()
