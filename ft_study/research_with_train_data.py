#!/usr/bin/env python
"""Re-run MCTS search on existing FT checkpoints using training data.

import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

Reuses saved LoRA adapters from prior ft_study runs, skipping the
multi-hour fine-tuning phase.  Runs MCTS search with the train_ft
split (much larger evaluation pool) and larger tier sizes, then
evaluates on the *same* test split as the original run for fair
comparison with the ft_only baseline.

Usage:
    cd dr-llm
    python -m ft_study.research_with_train_data \
        --datasets commonsenseqa boolq arc_easy \
        --seeds 42 1337 2024 \
        --num_simulations 6000 \
        --search_tier2 500 --search_tier3 1500 --search_tier4 3000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from core.benchmark_mcts import grade_response, seq_to_layers, SKIP
from core.flexible_models import get_is_instruct
from core.permutation_mcts import set_seed

from ft_study.config import FTConfig, FTStudyConfig, SearchConfig, SplitConfig
from ft_study.data import create_four_way_split, split_val_search_into_tiers
from ft_study.runner import (
    _MCTSModelFromLoaded,
    evaluate_on_test,
    run_search_for_study,
    _seq_metrics,
)
from core.permutation_mcts import MCTSModel

logger = logging.getLogger(__name__)


def find_adapter_path(
    base_results_dir: str,
    dataset_name: str,
    seed: int,
    arm: str = "ft_only",
) -> Optional[str]:
    """Locate the saved adapter from a prior ft_study run."""
    result_path = os.path.join(
        base_results_dir, dataset_name, f"seed_{seed}", arm, "result.json",
    )
    if not os.path.isfile(result_path):
        return None
    with open(result_path) as f:
        result = json.load(f)
    adapter = result.get("adapter_path")
    if adapter and os.path.isdir(adapter):
        return adapter
    ckpt_dir = os.path.join(
        base_results_dir, dataset_name, f"seed_{seed}", arm,
        "checkpoints", "final_adapter",
    )
    if os.path.isdir(ckpt_dir):
        return ckpt_dir
    return None


def run_one(
    model_name: str,
    dataset_name: str,
    seed: int,
    adapter_path: str,
    search_cfg: SearchConfig,
    split_cfg: SplitConfig,
    output_dir: str,
    gpu_rank: int = 0,
) -> Dict[str, Any]:
    """Load FT model, search with train data, evaluate on test split."""
    set_seed(seed)
    is_instruct = get_is_instruct(model_name)

    splits = create_four_way_split(
        dataset_name, is_instruct, seed, split_cfg=split_cfg, hf_split="train",
    )
    search_pool = splits["train_ft"]
    logger.info(
        "Search pool: %d samples (train_ft) for %s seed=%d",
        len(search_pool), dataset_name, seed,
    )

    ft_mcts = _MCTSModelFromLoaded.from_pretrained_ft(
        model_name, adapter_path, layer_sequence=None, rank=gpu_rank,
    )
    num_layers = ft_mcts.num_layers

    arm_dir = os.path.join(output_dir, dataset_name, f"seed_{seed}", "ft_search_shared")
    os.makedirs(arm_dir, exist_ok=True)
    search_prefix = os.path.join(arm_dir, f"search_ft_{dataset_name}")

    t0 = time.time()
    best_seq, search_summary = run_search_for_study(
        ft_mcts, search_pool, search_cfg,
        dataset_name, model_name, search_prefix,
        notify_signal=False,
    )
    search_wall = time.time() - t0

    active_layers = seq_to_layers(best_seq)
    ft_mcts.wrapper.model.model.layer_indices = active_layers

    test_metrics = evaluate_on_test(
        ft_mcts.wrapper.model, ft_mcts.wrapper.tokenizer,
        best_seq, splits["test"], dataset_name, model_name, num_layers,
    )
    val_metrics = evaluate_on_test(
        ft_mcts.wrapper.model, ft_mcts.wrapper.tokenizer,
        best_seq, splits["val_select"], dataset_name, model_name, num_layers,
    )

    del ft_mcts
    torch.cuda.empty_cache()

    result = {
        "arm": "ft_search_shared",
        **test_metrics,
        "val_select_metric": val_metrics["test_metric"],
        "search_wall_clock_s": search_wall,
        **_seq_metrics(best_seq, num_layers),
        "post_ft_best_sequence": best_seq,
        "adapter_path": adapter_path,
        "search_pool_size": len(search_pool),
        "search_tier2": search_cfg.tier2_samples,
        "search_tier3": search_cfg.tier3_samples,
        "search_tier4": search_cfg.tier4_samples,
        "num_simulations": search_cfg.num_simulations,
        "seed": seed,
        "dataset": dataset_name,
        "model": model_name,
    }

    result_path = os.path.join(arm_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Saved result to %s", result_path)

    return result


def run_base_model_search(
    model_name: str,
    dataset_name: str,
    seed: int,
    search_cfg: SearchConfig,
    split_cfg: SplitConfig,
    output_dir: str,
    gpu_rank: int = 0,
) -> Dict[str, Any]:
    """Search on the base model (no LoRA) using training data.

    The resulting optimal sequence can then be compared to ft_only to
    gauge the headroom for search_ft style pipelines.
    """
    set_seed(seed)
    is_instruct = get_is_instruct(model_name)

    splits = create_four_way_split(
        dataset_name, is_instruct, seed, split_cfg=split_cfg, hf_split="train",
    )
    search_pool = splits["train_ft"]
    logger.info(
        "Base-model search pool: %d samples (train_ft) for %s seed=%d",
        len(search_pool), dataset_name, seed,
    )

    mcts_model = MCTSModel(model_name, rank=gpu_rank)
    num_layers = mcts_model.num_layers

    arm_dir = os.path.join(output_dir, dataset_name, f"seed_{seed}", "base_search_shared")
    os.makedirs(arm_dir, exist_ok=True)
    search_prefix = os.path.join(arm_dir, f"search_base_{dataset_name}")

    t0 = time.time()
    best_seq, search_summary = run_search_for_study(
        mcts_model, search_pool, search_cfg,
        dataset_name, model_name, search_prefix,
        notify_signal=False,
    )
    search_wall = time.time() - t0

    active_layers = seq_to_layers(best_seq)
    mcts_model.wrapper.model.model.layer_indices = active_layers

    test_metrics = evaluate_on_test(
        mcts_model.wrapper.model, mcts_model.wrapper.tokenizer,
        best_seq, splits["test"], dataset_name, model_name, num_layers,
    )

    del mcts_model
    torch.cuda.empty_cache()

    result = {
        "arm": "base_search_shared",
        **test_metrics,
        "search_wall_clock_s": search_wall,
        **_seq_metrics(best_seq, num_layers),
        "best_sequence": best_seq,
        "search_pool_size": len(search_pool),
        "search_tier2": search_cfg.tier2_samples,
        "search_tier3": search_cfg.tier3_samples,
        "search_tier4": search_cfg.tier4_samples,
        "num_simulations": search_cfg.num_simulations,
        "seed": seed,
        "dataset": dataset_name,
        "model": model_name,
    }

    result_path = os.path.join(arm_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Saved result to %s", result_path)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Re-search on existing FT checkpoints with train data",
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--datasets", type=str, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])
    parser.add_argument("--base_results_dir", type=str, default="ft_study_results")
    parser.add_argument("--source_arm", type=str, default="ft_only",
                        help="Which prior arm's adapter to reuse (default: ft_only)")
    parser.add_argument("--output_dir", type=str, default="ft_study_results_shared")
    parser.add_argument("--gpu_rank", type=int, default=0)
    parser.add_argument("--num_simulations", type=int, default=6000)
    parser.add_argument("--search_tier2", type=int, default=500)
    parser.add_argument("--search_tier3", type=int, default=1500)
    parser.add_argument("--search_tier4", type=int, default=3000)
    parser.add_argument("--neighborhood_radius", type=int, default=5)
    parser.add_argument("--max_swaps", type=int, default=24)
    parser.add_argument(
        "--base_model_search", action="store_true", default=False,
        help="Search on the base model (no LoRA adapter). Tests whether "
             "the base model search landscape with large train data "
             "can find improvements reliably.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    search_cfg = SearchConfig(
        num_simulations=args.num_simulations,
        tier2_samples=args.search_tier2,
        tier3_samples=args.search_tier3,
        tier4_samples=args.search_tier4,
        neighborhood_radius=args.neighborhood_radius,
        max_swaps=args.max_swaps,
    )
    split_cfg = SplitConfig()

    arm_subdir = "base_search_shared" if args.base_model_search else "ft_search_shared"

    all_results = []
    for dataset_name in args.datasets:
        for seed in args.seeds:
            result_path = os.path.join(
                args.output_dir, dataset_name, f"seed_{seed}",
                arm_subdir, "result.json",
            )
            if os.path.isfile(result_path):
                logger.info("Skipping %s seed=%d (result exists)", dataset_name, seed)
                with open(result_path) as f:
                    all_results.append(json.load(f))
                continue

            if args.base_model_search:
                logger.info(
                    "=== BASE MODEL SEARCH: %s seed=%d ===", dataset_name, seed,
                )
                result = run_base_model_search(
                    args.model_name, dataset_name, seed,
                    search_cfg, split_cfg, args.output_dir,
                    gpu_rank=args.gpu_rank,
                )
            else:
                adapter = find_adapter_path(
                    args.base_results_dir, dataset_name, seed,
                    arm=args.source_arm,
                )
                if adapter is None:
                    logger.warning(
                        "No adapter found for %s seed=%d arm=%s in %s — skip",
                        dataset_name, seed, args.source_arm,
                        args.base_results_dir,
                    )
                    continue
                logger.info(
                    "=== %s seed=%d: adapter=%s ===",
                    dataset_name, seed, adapter,
                )
                result = run_one(
                    args.model_name, dataset_name, seed, adapter,
                    search_cfg, split_cfg, args.output_dir,
                    gpu_rank=args.gpu_rank,
                )
            all_results.append(result)

    mode_label = "base model search" if args.base_model_search else "ft_search"
    print(f"\n{'='*80}")
    print(f"RESULTS: {mode_label} with shared train data")
    print("=" * 80)
    for r in all_results:
        ds = r["dataset"]
        seed = r["seed"]
        test = r["test_metric"]
        val = r.get("val_select_metric", "n/a")
        ed = r.get("edit_distance_from_default", "?")
        pool = r.get("search_pool_size", "?")
        sw = r.get("search_wall_clock_s", 0)
        val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        print(
            f"  {ds:18} seed={seed}  test={test:.4f}  val={val_str}  "
            f"ed={ed}  pool={pool}  search={sw:.0f}s"
        )

    print("\nBaselines (ft_only from prior runs):")
    for dataset_name in args.datasets:
        for seed in args.seeds:
            p = os.path.join(
                args.base_results_dir, dataset_name, f"seed_{seed}",
                "ft_only", "result.json",
            )
            if os.path.isfile(p):
                with open(p) as f:
                    d = json.load(f)
                print(f"  {dataset_name:18} seed={seed}  test={d['test_metric']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
