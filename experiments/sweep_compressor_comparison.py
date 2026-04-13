#!/usr/bin/env python3
"""Systematic comparison of residual-stream compressors for fine routing.

Tests ``LastTokenCompressor`` (baseline) vs ``TopDownAttentionCompressor``
in the static pivot-based MLP setting on BoolQ and CommonsenseQA.

Uses the known-good hyperparameter presets from the marginalization eval
module and sweeps over compressor-specific parameters (d_compress, n_heads)
with multiple seeds.

Usage
-----
    python experiments/sweep_compressor_comparison.py \
        --data_dir fine_routing_data_boolq_mcts \
        --benchmark boolq \
        --results_dir predictions \
        --model_name Qwen/Qwen2.5-0.5B-Instruct \
        --eval_questions 200 \
        --output_json results/compressor_comparison_boolq.json

    # Run both benchmarks:
    python experiments/sweep_compressor_comparison.py \
        --data_dir fine_routing_data_boolq_mcts \
        --benchmark boolq \
        --results_dir predictions \
        --output_json results/compressor_comparison_boolq.json

    python experiments/sweep_compressor_comparison.py \
        --data_dir fine_routing_data_commonsenseqa_mcts \
        --benchmark commonsenseqa \
        --results_dir predictions \
        --output_json results/compressor_comparison_csqa.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import types
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

from core.benchmark_mcts import grade_response, seq_to_layers
from core.flexible_models import FlexibleModelWrapper, get_is_instruct
from core.permutation_mcts import prepare_arc_data
from evaluation.eval_fine_routing_marginalization import PRESETS
from pipeline.forward import get_pivot_residual, get_full_sequence_residual
from routers.fine_routing_config import FineRoutingConfig
from routers.residual_compressors import (
    CompressorConfig,
    CompressedRouter,
    CompressedGate,
    build_compressor,
    pad_sequences,
)
from training.train_benchmark_router import load_optimal_sequences_from_results
from training.train_fine_gate import FineGate, DeltaGate
from training.train_fine_router import FineRouter, soft_cross_entropy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ======================================================================
#  Data loading (supports both last-token and full-sequence)
# ======================================================================

def load_mcts_data(
    data_dir: str,
    benchmark: str,
    anchor_seq: List[int],
) -> Dict[str, Any]:
    """Load MCTS training data for a benchmark."""
    from experiments.sweep_fine_routing import load_bench_data_mcts

    (residuals, gate_labels, router_targets_base,
     catalog_full, seq_to_idx_full,
     catalog_reduced, seq_to_idx_reduced,
     records) = load_bench_data_mcts(data_dir, benchmark, anchor_seq)

    full_residuals = None
    full_pt = os.path.join(data_dir, f"{benchmark}_full_residuals.pt")
    if os.path.isfile(full_pt):
        data = torch.load(full_pt, map_location="cpu", weights_only=False)
        full_residuals = data["residuals"]  # list of [T_i, d_model]
        n = min(len(full_residuals), residuals.shape[0])
        full_residuals = full_residuals[:n]
        logger.info("  Full-sequence residuals loaded: %d questions", len(full_residuals))
    else:
        logger.warning("  No full-sequence residuals at %s", full_pt)

    return {
        "residuals": residuals,
        "full_residuals": full_residuals,
        "gate_labels": gate_labels,
        "router_targets_base": router_targets_base,
        "catalog_full": catalog_full,
        "seq_to_idx_full": seq_to_idx_full,
        "catalog_reduced": catalog_reduced,
        "seq_to_idx_reduced": seq_to_idx_reduced,
        "records": records,
    }


# ======================================================================
#  Training helpers
# ======================================================================

def train_gate_for_trial(
    data: Dict,
    hp: Dict,
    compressor_cfg: Optional[CompressorConfig],
    device: torch.device,
    seed: int,
) -> Any:
    """Train gate network (binary or delta) for a trial."""
    gating_mode = hp.get("gating_mode", "gate_network")
    if gating_mode not in ("gate_network",):
        return None

    residuals = data["residuals"]
    gate_labels = data["gate_labels"]
    d_model = residuals.shape[1]

    if compressor_cfg is not None and compressor_cfg.compressor_type != "last_token":
        full_residuals = data["full_residuals"]
        if full_residuals is None:
            raise ValueError("Full-sequence residuals required for attention compressor")

        from experiments.sweep_fine_routing import train_compressed_gate_inline
        compressor_cfg.d_model = d_model
        gate = train_compressed_gate_inline(
            full_residuals, gate_labels, d_model,
            compressor_cfg=compressor_cfg,
            hidden_dim=hp.get("gate_hidden_dim", 256),
            gate_dropout=hp.get("gate_dropout", 0.1),
            lr=hp.get("gate_lr", 1e-3),
            epochs=hp.get("gate_epochs", 60),
            batch_size=64,
            recall_boost=hp.get("recall_boost", 1.5),
            device=device,
            seed=seed,
        )
    else:
        from experiments.sweep_fine_routing import train_gate_inline
        gate = train_gate_inline(
            residuals, gate_labels, d_model,
            hidden_dim=hp.get("gate_hidden_dim", 256),
            gate_dropout=hp.get("gate_dropout", 0.1),
            lr=hp.get("gate_lr", 1e-3),
            epochs=hp.get("gate_epochs", 60),
            batch_size=64,
            recall_boost=hp.get("recall_boost", 1.5),
            device=device,
            seed=seed,
        )

    return gate


def train_router_for_trial(
    data: Dict,
    hp: Dict,
    compressor_cfg: Optional[CompressorConfig],
    device: torch.device,
    seed: int,
) -> Tuple[Any, List[List[int]], int]:
    """Train router for a trial. Returns (router, catalog, num_classes)."""
    from experiments.sweep_fine_routing import (
        rebuild_targets_for_trial,
    )

    residuals = data["residuals"]
    gate_labels = data["gate_labels"]
    records = data["records"]
    d_model = residuals.shape[1]

    use_bs = hp.get("use_best_seq", False)
    if use_bs:
        catalog = data["catalog_reduced"]
        seq_to_idx = data["seq_to_idx_reduced"]
    else:
        catalog = data["catalog_full"]
        seq_to_idx = data["seq_to_idx_full"]
    num_classes = len(catalog)

    nb = hp.get("noop_boost", 0.0)
    t_temp = hp.get("target_temp", 1.0)
    router_targets = rebuild_targets_for_trial(
        records, seq_to_idx, num_classes,
        noop_boost=nb, target_temp=t_temp, use_best_seq=use_bs,
    )

    h3 = hp.get("router_h3", 0)
    hidden_dims = [hp.get("router_h1", 512), hp.get("router_h2", 256)]
    if h3 > 0:
        hidden_dims.append(h3)

    gating_mode = hp.get("gating_mode", "gate_network")
    train_all = gating_mode != "gate_network"
    gate_pos_only = not train_all and hp.get("router_gate_pos_only", False)

    if compressor_cfg is not None and compressor_cfg.compressor_type != "last_token":
        full_residuals = data["full_residuals"]
        if full_residuals is None:
            raise ValueError("Full-sequence residuals required for attention compressor")

        from experiments.sweep_fine_routing import train_compressed_router_inline
        compressor_cfg.d_model = d_model
        router = train_compressed_router_inline(
            full_residuals, gate_labels, router_targets,
            d_model, num_classes,
            compressor_cfg=compressor_cfg,
            hidden_dims=hidden_dims,
            router_dropout=hp.get("router_dropout", 0.1),
            lr=hp.get("router_lr", 1e-3),
            epochs=hp.get("router_epochs", 150),
            batch_size=64,
            gate_positives_only=gate_pos_only,
            device=device,
            weight_decay=hp.get("weight_decay", 0.01),
            seed=seed,
        )
    else:
        from experiments.sweep_fine_routing import train_router_inline
        router = train_router_inline(
            residuals, gate_labels, router_targets,
            d_model, num_classes,
            hidden_dims=hidden_dims,
            router_dropout=hp.get("router_dropout", 0.1),
            lr=hp.get("router_lr", 1e-3),
            epochs=hp.get("router_epochs", 150),
            batch_size=64,
            gate_positives_only=gate_pos_only,
            device=device,
            hard_targets=hp.get("router_hard_targets", False),
            label_smoothing=hp.get("label_smoothing", 0.0),
            weight_decay=hp.get("weight_decay", 0.01),
            seed=seed,
        )

    return router, catalog, num_classes


# ======================================================================
#  Evaluation (LLM-based accuracy)
# ======================================================================

def generate_under_layers(wrapper, layers, text, system_prompt=None, max_new_tokens=1, is_math=False):
    from pipeline.forward import generate_under_layers as _gen
    return _gen(wrapper, layers, text, system_prompt=system_prompt,
                max_new_tokens=max_new_tokens, is_math=is_math)


def evaluate_trial(
    wrapper: FlexibleModelWrapper,
    gate: Any,
    router: Any,
    gamma: float,
    anchor_seq: List[int],
    sequence_catalog: List[List[int]],
    samples: List[Dict],
    benchmark: str,
    model_name: str,
    pivot_layer: int,
    gate_device: torch.device,
    gating_mode: str = "gate_network",
    confidence_threshold: float = 0.0,
    use_full_seq: bool = False,
) -> Dict[str, Any]:
    """Evaluate routing accuracy on validation samples."""
    anchor_layers = seq_to_layers(anchor_seq)
    is_math = "dart" in benchmark or benchmark in ("gsm8k_hard", "math500")

    anchor_correct = 0
    routed_correct = 0
    gate_opened = 0
    helped = 0
    hurt = 0
    n = len(samples)

    for sample in tqdm(samples, desc=f"eval({benchmark})", leave=False):
        anchor_resp = generate_under_layers(
            wrapper, anchor_layers, sample["input"],
            system_prompt=sample.get("system_prompt"),
            max_new_tokens=sample["max_new_tokens"],
            is_math=is_math,
        )
        anchor_sc = grade_response(
            anchor_resp, sample["correct"], benchmark, model_name, sample["input"]
        )
        anchor_ok = int(anchor_sc > 0.5)
        anchor_correct += anchor_ok

        if use_full_seq:
            full_hs = get_full_sequence_residual(
                wrapper, sample["input"],
                layer_indices=anchor_layers,
                pivot_layer=pivot_layer,
                system_prompt=sample.get("system_prompt"),
            ).float()
            if full_hs.shape[0] > 256:
                full_hs = full_hs[-256:]  # left-truncate, keep last 256 tokens
            full_hs = full_hs.unsqueeze(0).to(gate_device)
            mask = torch.ones(1, full_hs.shape[1], dtype=torch.long, device=gate_device)
            h_input = full_hs
            h_mask = mask
        else:
            h_pivot = get_pivot_residual(
                wrapper, sample["input"],
                layer_indices=anchor_layers,
                pivot_layer=pivot_layer,
                system_prompt=sample.get("system_prompt"),
            ).float().unsqueeze(0).to(gate_device)
            h_input = h_pivot
            h_mask = None

        with torch.no_grad():
            if isinstance(router, CompressedRouter):
                router_logits = router(h_input, attention_mask=h_mask)
            else:
                if h_input.dim() == 3:
                    router_logits = router(h_input[:, -1, :])
                else:
                    router_logits = router(h_input)
            router_probs = F.softmax(router_logits, dim=-1)
            pred_idx = router_logits.argmax(dim=-1).item()

        should_route = False
        if gating_mode == "gate_network" and gate is not None:
            with torch.no_grad():
                if isinstance(gate, CompressedGate):
                    gate_prob = torch.sigmoid(gate(h_input, attention_mask=h_mask)).item()
                elif h_input.dim() == 3:
                    gate_prob = torch.sigmoid(gate(h_input[:, -1, :])).item()
                else:
                    gate_prob = torch.sigmoid(gate(h_input)).item()
            should_route = gate_prob >= gamma and pred_idx != 0
        elif gating_mode == "router_argmax":
            should_route = pred_idx != 0
        elif gating_mode == "router_confidence":
            deviate_prob = 1.0 - router_probs[..., 0].item()
            should_route = deviate_prob > confidence_threshold
            if should_route:
                non_noop = router_probs.clone()
                non_noop[..., 0] = 0.0
                pred_idx = non_noop.argmax(dim=-1).item()

        if not should_route:
            routed_correct += anchor_ok
        else:
            gate_opened += 1
            cand_seq = sequence_catalog[pred_idx]
            cand_layers = seq_to_layers(cand_seq)
            cand_resp = generate_under_layers(
                wrapper, cand_layers, sample["input"],
                system_prompt=sample.get("system_prompt"),
                max_new_tokens=sample["max_new_tokens"],
                is_math=is_math,
            )
            cand_sc = grade_response(
                cand_resp, sample["correct"], benchmark, model_name, sample["input"],
            )
            cand_ok = int(cand_sc > 0.5)
            routed_correct += cand_ok
            delta = cand_sc - anchor_sc
            if delta > 0:
                helped += 1
            elif delta < 0:
                hurt += 1

    return {
        "n": n,
        "anchor_accuracy": anchor_correct / max(n, 1),
        "routed_accuracy": routed_correct / max(n, 1),
        "gate_open_rate": gate_opened / max(n, 1),
        "unconditional_gain": (routed_correct - anchor_correct) / max(n, 1),
        "helped": helped,
        "hurt": hurt,
        "net_helped": helped - hurt,
    }


# ======================================================================
#  Sweep configs
# ======================================================================

SEEDS = [42, 123, 456]

COMPRESSOR_GRID = {
    "d_compress": [64, 128, 256, 512],
    "n_heads": [1, 2, 4, 8],
}


def build_trial_configs(benchmark: str) -> List[Dict[str, Any]]:
    """Build all trial configs for the comparison."""
    preset_key = f"{benchmark}_mcts"
    if preset_key not in PRESETS:
        raise ValueError(f"No preset for {benchmark}. Available: {list(PRESETS.keys())}")
    hp = dict(PRESETS[preset_key])

    trials = []

    for seed in SEEDS:
        trials.append({
            "compressor_type": "last_token",
            "d_compress": 0,
            "n_heads": 0,
            "seed": seed,
            "hp": hp,
        })

    for dc in COMPRESSOR_GRID["d_compress"]:
        for nh in COMPRESSOR_GRID["n_heads"]:
            if dc % nh != 0:
                continue
            for seed in SEEDS:
                trials.append({
                    "compressor_type": "top_down_attention",
                    "d_compress": dc,
                    "n_heads": nh,
                    "seed": seed,
                    "hp": hp,
                })

    return trials


# ======================================================================
#  Main
# ======================================================================

def main():
    p = argparse.ArgumentParser(
        description="Compare residual-stream compressors for fine routing"
    )
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--benchmark", type=str, required=True)
    p.add_argument("--results_dir", type=str, default="predictions")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--eval_questions", type=int, default=200)
    p.add_argument("--eval_skip", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--only_compressor", type=str, default=None,
                   choices=["last_token", "top_down_attention"],
                   help="Run only one compressor type (for partial runs)")
    p.add_argument("--only_seed", type=int, default=None,
                   help="Run only one seed (for partial runs)")
    p.add_argument("--d_compress_values", nargs="+", type=int, default=None,
                   help="Override d_compress grid")
    p.add_argument("--n_heads_values", nargs="+", type=int, default=None,
                   help="Override n_heads grid")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.d_compress_values:
        COMPRESSOR_GRID["d_compress"] = args.d_compress_values
    if args.n_heads_values:
        COMPRESSOR_GRID["n_heads"] = args.n_heads_values

    logger.info("Loading LLM %s ...", args.model_name)
    cfg = FineRoutingConfig(model_name=args.model_name, results_dir=args.results_dir)

    data_config_path = os.path.join(args.data_dir, "config.json")
    if os.path.isfile(data_config_path):
        with open(data_config_path) as f:
            data_cfg = json.load(f)
        for key in ("max_local_edits", "swap_radius", "editable_start"):
            if key in data_cfg:
                setattr(cfg, key, data_cfg[key])

    wrapper = FlexibleModelWrapper(args.model_name, rank=0)
    logger.info("  %d layers, d_model=%d", wrapper.num_layers, wrapper.hidden_size)

    anchor_seqs = load_optimal_sequences_from_results(
        args.results_dir, [args.benchmark], model_name=args.model_name,
    )
    anchor_seq = anchor_seqs[args.benchmark]
    logger.info("  anchor: %s", anchor_seq)

    logger.info("Loading training data ...")
    data = load_mcts_data(args.data_dir, args.benchmark, anchor_seq)
    d_model = data["residuals"].shape[1]
    logger.info("  %d samples, d_model=%d", len(data["gate_labels"]), d_model)

    is_instruct = get_is_instruct(args.model_name)
    val_samples = prepare_arc_data(
        args.benchmark, is_instruct=is_instruct, split="validation",
    )
    val_samples = val_samples[args.eval_skip: args.eval_skip + args.eval_questions]
    logger.info("  %d validation samples", len(val_samples))

    trials = build_trial_configs(args.benchmark)

    if args.only_compressor:
        trials = [t for t in trials if t["compressor_type"] == args.only_compressor]
    if args.only_seed is not None:
        trials = [t for t in trials if t["seed"] == args.only_seed]

    logger.info("Running %d trials ...", len(trials))

    # --- incremental output: JSONL file appended per trial ---------------
    out_base = args.output_json or f"results/compressor_comparison_{args.benchmark}.json"
    jsonl_path = out_base.replace(".json", ".jsonl")
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)

    done_tags: set = set()
    all_results: List[Dict[str, Any]] = []
    if os.path.isfile(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                all_results.append(rec)
                done_tags.add(rec.get("tag", ""))
        logger.info("Resumed: %d trials already in %s", len(all_results), jsonl_path)

    def _append_result(result: Dict[str, Any]):
        all_results.append(result)
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    # ---------------------------------------------------------------------

    for ti, trial in enumerate(trials):
        ct = trial["compressor_type"]
        dc = trial["d_compress"]
        nh = trial["n_heads"]
        seed = trial["seed"]
        hp = trial["hp"]

        tag = f"{ct}"
        if ct == "top_down_attention":
            tag += f"_dc{dc}_nh{nh}"
        tag += f"_s{seed}"

        if tag in done_tags:
            logger.info("  SKIP %s (already done)", tag)
            continue

        logger.info(
            "\n" + "=" * 70 +
            f"\n  TRIAL {ti+1}/{len(trials)}  [{tag}]" +
            "\n" + "=" * 70
        )

        use_full_seq = ct != "last_token"
        if use_full_seq and data["full_residuals"] is None:
            logger.error("Skipping %s: no full-sequence residuals", tag)
            _append_result({
                "trial": ti, "tag": tag,
                "compressor_type": ct, "d_compress": dc, "n_heads": nh,
                "seed": seed, "error": "no_full_residuals",
            })
            continue

        comp_cfg = None
        if ct != "last_token":
            comp_cfg = CompressorConfig(
                compressor_type=ct,
                d_model=d_model,
                d_compress=dc,
                n_heads=nh,
                n_latent_tokens=1,
            )

        t0 = time.time()

        try:
            gate = train_gate_for_trial(data, hp, comp_cfg, device, seed)
            router, catalog, num_classes = train_router_for_trial(
                data, hp, comp_cfg, device, seed,
            )
            train_time = time.time() - t0
            logger.info("  Training: %.1fs  (%d classes)", train_time, num_classes)

            metrics = evaluate_trial(
                wrapper=wrapper,
                gate=gate,
                router=router,
                gamma=hp.get("gamma", 0.5),
                anchor_seq=anchor_seq,
                sequence_catalog=catalog,
                samples=val_samples,
                benchmark=args.benchmark,
                model_name=args.model_name,
                pivot_layer=cfg.pivot_layer,
                gate_device=device,
                gating_mode=hp.get("gating_mode", "gate_network"),
                confidence_threshold=hp.get("confidence_threshold", 0.0),
                use_full_seq=use_full_seq,
            )

            total_time = time.time() - t0
            result = {
                "trial": ti,
                "tag": tag,
                "compressor_type": ct,
                "d_compress": dc,
                "n_heads": nh,
                "seed": seed,
                "num_classes": num_classes,
                "train_time_s": train_time,
                "total_time_s": total_time,
                **metrics,
            }
            _append_result(result)

            logger.info(
                "  anchor=%.4f  routed=%.4f  Δ=%+.4f  gate=%.1f%%  "
                "helped=%d  hurt=%d  (%.0fs)",
                metrics["anchor_accuracy"],
                metrics["routed_accuracy"],
                metrics["unconditional_gain"],
                100 * metrics["gate_open_rate"],
                metrics["helped"], metrics["hurt"],
                total_time,
            )

        except Exception as e:
            logger.error("  TRIAL FAILED: %s", e, exc_info=True)
            _append_result({
                "trial": ti, "tag": tag,
                "compressor_type": ct, "d_compress": dc, "n_heads": nh,
                "seed": seed, "error": str(e),
            })

    # ================================================================
    #  Summary
    # ================================================================
    print_summary(all_results, args.benchmark)

    with open(out_base, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Final results: %s  (incremental: %s)", out_base, jsonl_path)


def print_summary(all_results: List[Dict[str, Any]], benchmark: str):
    """Print a human-readable summary table to stdout."""
    from collections import defaultdict

    print("\n" + "=" * 100)
    print(f"  COMPRESSOR COMPARISON — {benchmark.upper()}")
    print("=" * 100)

    ok = [r for r in all_results if "error" not in r]
    if not ok:
        print("No successful trials.")
        return

    print(
        f"{'Compressor':<28}  {'dc':>4}  {'nh':>3}  {'seed':>5}  "
        f"{'Anchor':>7}  {'Routed':>7}  {'Δ pp':>7}  {'Gate%':>6}  {'H/H':>5}"
    )
    print("-" * 100)

    for r in ok:
        ct = r["compressor_type"]
        label = "top_down_attn" if ct == "top_down_attention" else ct
        print(
            f"{label:<28}  {r['d_compress']:>4}  {r['n_heads']:>3}  "
            f"{r['seed']:>5}  {r['anchor_accuracy']:>7.4f}  "
            f"{r['routed_accuracy']:>7.4f}  "
            f"{r['unconditional_gain']*100:>+7.2f}  "
            f"{100*r['gate_open_rate']:>5.1f}%  "
            f"{r['helped']}/{r['hurt']}"
        )

    print("\n" + "-" * 100)
    print("  AGGREGATED (mean +/- std across seeds)")
    print("-" * 100)

    groups = defaultdict(list)
    for r in ok:
        key = (r["compressor_type"], r["d_compress"], r["n_heads"])
        groups[key].append(r)

    print(
        f"{'Config':<35}  {'Seeds':>5}  {'Δ mean':>8}  {'Δ std':>7}  "
        f"{'Routed':>8}  {'Best Δ':>7}"
    )
    for key in sorted(groups.keys()):
        items = groups[key]
        ct, dc, nh = key
        deltas = [r["unconditional_gain"] * 100 for r in items]
        accs = [r["routed_accuracy"] for r in items]
        label = ct
        if ct == "top_down_attention":
            label = f"top_down dc={dc} nh={nh}"
        print(
            f"{label:<35}  {len(items):>5}  {np.mean(deltas):>+8.2f}  "
            f"{np.std(deltas):>7.2f}  {np.mean(accs):>8.4f}  "
            f"{max(deltas):>+7.2f}"
        )

    print("=" * 100)


if __name__ == "__main__":
    main()
