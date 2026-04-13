#!/usr/bin/env python3
"""Fine-routing inference pipeline with per-benchmark gate and router.

import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

For each test question in benchmark *b*:
  1.  (Optional) Coarse router -> benchmark *b*  (or benchmark given directly).
  2.  Load anchor sequence ``s_b``.
  3.  Run model under ``s_b`` to pivot layer, extract pivot residual.
  4.  Gate_b: if ``g_b(h_pivot) < gamma`` -> use ``s_b`` as-is.
  5.  Router_b: predict deviation ``delta_hat = argmax pi_b(delta | h_pivot)``.
  6.  Final inference with ``s_b + delta_hat``.

Gate and router checkpoints live in ``--checkpoint_dir`` as
``gate_best_{benchmark}.pt`` and ``router_best_{benchmark}.pt``.

Usage
-----
    python run_fine_routing_inference.py \
        --model_name Qwen/Qwen2.5-0.5B-Instruct \
        --results_dir predictions/qwen25_0.5b_v2_sdpa \
        --checkpoint_dir checkpoints/fine_routing \
        --benchmarks winogrande boolq commonsenseqa mmlu_all arc_easy \
        --data_split validation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from core.benchmark_mcts import grade_response, per_question_mcts, seq_to_layers
from routers.fine_routing_config import FineRoutingConfig
from routers.fine_routing_deviations import (
    apply_deviation,
    canonical_key,
    enumerate_deviations,
)
from core.flexible_models import FlexibleModelWrapper, get_is_instruct
from core.permutation_mcts import prepare_arc_data
from training.train_benchmark_router import load_optimal_sequences_from_results
from train_fine_gate import FineGate
from train_fine_router import FineRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generation helper (same as in build_fine_routing_dataset)
# ---------------------------------------------------------------------------

def generate_under_layers(
    wrapper: FlexibleModelWrapper,
    layers: List[int],
    text: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 1,
    is_math: bool = False,
) -> str:
    saved = wrapper.model.model.layer_indices
    wrapper.model.model.layer_indices = layers
    try:
        has_dup = len(layers) != len(set(layers))
        prompt = wrapper.prepare_prompt(text, system_prompt=system_prompt)
        inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(wrapper.model.device)
        input_len = inputs.input_ids.shape[1]
        gen_kw = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": wrapper.tokenizer.eos_token_id,
            "do_sample": False,
        }
        if has_dup or is_math or len(layers) != wrapper.num_layers:
            gen_kw["use_cache"] = False
        with torch.no_grad():
            out = wrapper.model.generate(**inputs, **gen_kw)
        return wrapper.tokenizer.decode(
            out[0][input_len:], skip_special_tokens=True
        ).strip()
    finally:
        wrapper.model.model.layer_indices = saved


# ---------------------------------------------------------------------------
# Load per-benchmark gate / router checkpoints
# ---------------------------------------------------------------------------

def load_gate(path: str, device: torch.device) -> FineGate:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    gate = FineGate(
        d_model=ckpt["d_model"],
        hidden_dim=ckpt["hidden_dim"],
    ).to(device)
    gate.load_state_dict(ckpt["model_state_dict"])
    gate.eval()
    logger.info("Gate loaded from %s (d_model=%d)", path, ckpt["d_model"])
    return gate


def load_router(path: str, device: torch.device) -> FineRouter:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    router = FineRouter(
        d_model=ckpt["d_model"],
        num_classes=ckpt["num_classes"],
        hidden_dims=ckpt["hidden_dims"],
        dropout=ckpt.get("dropout", 0.1),
    ).to(device)
    router.load_state_dict(ckpt["model_state_dict"])
    router.eval()
    logger.info(
        "Router loaded from %s (d_model=%d, classes=%d)",
        path, ckpt["d_model"], ckpt["num_classes"],
    )
    return router


def load_per_benchmark_models(
    checkpoint_dir: str,
    benchmarks: List[str],
    device: torch.device,
) -> tuple:
    """Load ``gate_best_{b}.pt`` and ``router_best_{b}.pt`` for each benchmark.

    Returns ``(gates, routers)`` dicts mapping benchmark -> model.
    Benchmarks without checkpoints are silently skipped.
    """
    gates: Dict[str, FineGate] = {}
    routers: Dict[str, FineRouter] = {}
    for bench in benchmarks:
        gate_path = os.path.join(checkpoint_dir, f"gate_best_{bench}.pt")
        router_path = os.path.join(checkpoint_dir, f"router_best_{bench}.pt")
        if not os.path.isfile(gate_path):
            logger.warning("Gate checkpoint missing for %s: %s", bench, gate_path)
            continue
        if not os.path.isfile(router_path):
            logger.warning("Router checkpoint missing for %s: %s", bench, router_path)
            continue
        gates[bench] = load_gate(gate_path, device)
        routers[bench] = load_router(router_path, device)
    return gates, routers


# ---------------------------------------------------------------------------
# Inference + metrics
# ---------------------------------------------------------------------------

def run_inference(
    cfg: FineRoutingConfig,
    wrapper: FlexibleModelWrapper,
    gates: Dict[str, FineGate],
    routers: Dict[str, FineRouter],
    anchor_seqs: Dict[str, List[int]],
    gamma: float,
):
    """Run end-to-end fine-routing evaluation using per-benchmark gate/router."""
    device = next(iter(gates.values())).net[0].weight.device
    is_instruct = get_is_instruct(cfg.model_name)

    # pre-build deviation catalogs per benchmark
    dev_catalogs: Dict[str, list] = {}
    for bench in cfg.benchmarks:
        if bench not in anchor_seqs:
            continue
        dev_catalogs[bench] = enumerate_deviations(
            anchor_seqs[bench],
            editable_start=cfg.editable_start,
            num_layers=wrapper.num_layers,
            swap_radius=cfg.swap_radius,
            max_edits=cfg.max_local_edits,
        )

    # per-benchmark and global metrics
    metrics: Dict[str, Dict] = {}
    global_anchor_correct = 0
    global_routed_correct = 0
    global_gate_opened = 0
    global_gain_when_opened = 0.0
    global_gain_all = 0.0
    global_helped_when_opened = 0
    global_total = 0

    for bench in cfg.benchmarks:
        if bench not in anchor_seqs:
            logger.warning("No anchor for %s, skipping", bench)
            continue
        if bench not in gates or bench not in routers:
            logger.warning("No gate/router for %s, skipping", bench)
            continue

        anchor_seq = anchor_seqs[bench]
        anchor_layers = seq_to_layers(anchor_seq)
        deviations = dev_catalogs[bench]
        bench_gate = gates[bench]
        bench_router = routers[bench]
        is_math = "dart" in bench or bench in ("gsm8k_hard", "math500")

        samples = prepare_arc_data(bench, is_instruct=is_instruct, split=cfg.data_split)
        if not samples:
            logger.warning("No samples for %s, skipping", bench)
            continue

        anchor_correct = 0
        routed_correct = 0
        gate_opened = 0
        gain_when_opened = 0.0
        helped_when_opened = 0
        n = len(samples)

        for sample in tqdm(samples, desc=bench):
            # --- anchor score ---
            anchor_resp = generate_under_layers(
                wrapper, anchor_layers, sample["input"],
                system_prompt=sample.get("system_prompt"),
                max_new_tokens=sample["max_new_tokens"],
                is_math=is_math,
            )
            anchor_sc = grade_response(
                anchor_resp, sample["correct"], bench, cfg.model_name, sample["input"]
            )
            anchor_ok = int(anchor_sc > 0.5)
            anchor_correct += anchor_ok

            # --- pivot residual ---
            h_pivot = wrapper.get_pivot_residual(
                sample["input"],
                layer_indices=anchor_layers,
                pivot_layer=cfg.pivot_layer,
                system_prompt=sample.get("system_prompt"),
            ).float().to(device)  # [1, d_model]

            # --- gate (benchmark-specific) ---
            with torch.no_grad():
                gate_logit = bench_gate(h_pivot)
                gate_prob = torch.sigmoid(gate_logit).item()

            if gate_prob < gamma:
                routed_correct += anchor_ok
                global_gain_all += 0.0
            else:
                gate_opened += 1
                # --- router (benchmark-specific) ---
                with torch.no_grad():
                    logits = bench_router(h_pivot)  # [1, |D_b|]
                    pred_idx = logits.argmax(dim=-1).item()

                deviation = deviations[pred_idx]
                if not deviation:  # no-op predicted
                    routed_correct += anchor_ok
                    gain_when_opened += 0.0
                    global_gain_all += 0.0
                else:
                    cand_seq = apply_deviation(anchor_seq, deviation)
                    cand_layers = seq_to_layers(cand_seq)
                    cand_resp = generate_under_layers(
                        wrapper, cand_layers, sample["input"],
                        system_prompt=sample.get("system_prompt"),
                        max_new_tokens=sample["max_new_tokens"],
                        is_math=is_math,
                    )
                    cand_sc = grade_response(
                        cand_resp, sample["correct"], bench,
                        cfg.model_name, sample["input"],
                    )
                    cand_ok = int(cand_sc > 0.5)
                    routed_correct += cand_ok
                    delta = cand_sc - anchor_sc
                    gain_when_opened += delta
                    global_gain_all += delta
                    if delta > 0:
                        helped_when_opened += 1

        bench_metrics = {
            "n": n,
            "anchor_accuracy": anchor_correct / max(n, 1),
            "routed_accuracy": routed_correct / max(n, 1),
            "gate_open_rate": gate_opened / max(n, 1),
            "conditional_gain": gain_when_opened / max(gate_opened, 1),
            "helped_when_opened": helped_when_opened,
            "helped_frac_when_opened": helped_when_opened / max(gate_opened, 1),
        }
        metrics[bench] = bench_metrics
        logger.info(
            "%s: anchor=%.3f  routed=%.3f  gate_open=%.1f%%  cond_gain=%.4f  helped=%d/%d",
            bench,
            bench_metrics["anchor_accuracy"],
            bench_metrics["routed_accuracy"],
            100 * bench_metrics["gate_open_rate"],
            bench_metrics["conditional_gain"],
            helped_when_opened,
            gate_opened,
        )

        global_anchor_correct += anchor_correct
        global_routed_correct += routed_correct
        global_gate_opened += gate_opened
        global_gain_when_opened += gain_when_opened
        global_helped_when_opened += helped_when_opened
        global_total += n

    # --- global summary ---
    summary = {
        "total_questions": global_total,
        "anchor_accuracy": global_anchor_correct / max(global_total, 1),
        "routed_accuracy": global_routed_correct / max(global_total, 1),
        "gate_open_rate": global_gate_opened / max(global_total, 1),
        "conditional_gain": global_gain_when_opened / max(global_gate_opened, 1),
        "unconditional_gain": global_gain_all / max(global_total, 1),
        "helped_frac_when_opened": global_helped_when_opened / max(global_gate_opened, 1),
        "per_benchmark": metrics,
    }
    logger.info("=== GLOBAL ===")
    logger.info(
        "anchor=%.4f  routed=%.4f  gate_open=%.1f%%  cond_gain=%.4f  uncond_gain=%.4f",
        summary["anchor_accuracy"],
        summary["routed_accuracy"],
        100 * summary["gate_open_rate"],
        summary["conditional_gain"],
        summary["unconditional_gain"],
    )
    return summary


# ---------------------------------------------------------------------------
# MCTS inference (gate + per-question MCTS instead of router)
# ---------------------------------------------------------------------------

def run_inference_mcts(
    cfg: FineRoutingConfig,
    wrapper: FlexibleModelWrapper,
    gates: Dict[str, FineGate],
    anchor_seqs: Dict[str, List[int]],
    gamma: float,
):
    """Run fine-routing evaluation using gate + per-question MCTS.

    For each question where the gate fires, runs
    :func:`per_question_mcts` anchored on the benchmark sequence to search
    for a better layer ordering.  Allows larger swap_radius / max_edits
    than the exhaustive-enumeration + router path.
    """
    device = next(iter(gates.values())).net[0].weight.device
    is_instruct = get_is_instruct(cfg.model_name)
    num_layers = wrapper.num_layers

    metrics: Dict[str, Dict] = {}
    global_anchor_correct = 0
    global_routed_correct = 0
    global_gate_opened = 0
    global_gain_when_opened = 0.0
    global_gain_all = 0.0
    global_helped_when_opened = 0
    global_total = 0
    global_mcts_explored = 0

    for bench in cfg.benchmarks:
        if bench not in anchor_seqs:
            logger.warning("No anchor for %s, skipping", bench)
            continue
        if bench not in gates:
            logger.warning("No gate for %s, skipping", bench)
            continue

        anchor_seq = anchor_seqs[bench]
        anchor_layers = seq_to_layers(anchor_seq)
        bench_gate = gates[bench]
        is_math = "dart" in bench or bench in ("gsm8k_hard", "math500")

        samples = prepare_arc_data(bench, is_instruct=is_instruct, split=cfg.data_split)
        if not samples:
            logger.warning("No samples for %s, skipping", bench)
            continue

        anchor_correct = 0
        routed_correct = 0
        gate_opened = 0
        gain_when_opened = 0.0
        helped_when_opened = 0
        bench_explored = 0
        n = len(samples)

        for sample in tqdm(samples, desc=f"{bench} (mcts)"):
            # --- anchor score ---
            anchor_resp = generate_under_layers(
                wrapper, anchor_layers, sample["input"],
                system_prompt=sample.get("system_prompt"),
                max_new_tokens=sample["max_new_tokens"],
                is_math=is_math,
            )
            anchor_sc = grade_response(
                anchor_resp, sample["correct"], bench, cfg.model_name, sample["input"]
            )
            anchor_ok = int(anchor_sc > 0.5)
            anchor_correct += anchor_ok

            # --- pivot residual ---
            h_pivot = wrapper.get_pivot_residual(
                sample["input"],
                layer_indices=anchor_layers,
                pivot_layer=cfg.pivot_layer,
                system_prompt=sample.get("system_prompt"),
            ).float().to(device)

            # --- gate ---
            with torch.no_grad():
                gate_logit = bench_gate(h_pivot)
                gate_prob = torch.sigmoid(gate_logit).item()

            if gate_prob < gamma:
                routed_correct += anchor_ok
                global_gain_all += 0.0
            else:
                gate_opened += 1

                # --- per-question MCTS ---
                def _grade(seq):
                    layers = seq_to_layers(seq)
                    if not layers:
                        return 0.0
                    resp = generate_under_layers(
                        wrapper, layers, sample["input"],
                        system_prompt=sample.get("system_prompt"),
                        max_new_tokens=sample["max_new_tokens"],
                        is_math=is_math,
                    )
                    return grade_response(
                        resp, sample["correct"], bench, cfg.model_name, sample["input"]
                    )

                mcts_result = per_question_mcts(
                    anchor_seq=anchor_seq,
                    grade_fn=_grade,
                    num_simulations=cfg.mcts_num_simulations,
                    num_layers=num_layers,
                    radius=cfg.swap_radius,
                    max_swaps=cfg.max_local_edits,
                    editable_start=cfg.editable_start,
                    exploration_constant=cfg.mcts_exploration_constant,
                    pw_C=cfg.mcts_pw_C,
                    pw_alpha=cfg.mcts_pw_alpha,
                )

                bench_explored += mcts_result["num_explored"]
                cand_ok = int(mcts_result["best_score"] > 0.5)
                routed_correct += cand_ok
                delta = mcts_result["best_score"] - anchor_sc
                gain_when_opened += delta
                global_gain_all += delta
                if delta > 0:
                    helped_when_opened += 1

        bench_metrics = {
            "n": n,
            "anchor_accuracy": anchor_correct / max(n, 1),
            "routed_accuracy": routed_correct / max(n, 1),
            "gate_open_rate": gate_opened / max(n, 1),
            "conditional_gain": gain_when_opened / max(gate_opened, 1),
            "helped_when_opened": helped_when_opened,
            "helped_frac_when_opened": helped_when_opened / max(gate_opened, 1),
            "avg_mcts_explored": bench_explored / max(gate_opened, 1),
        }
        metrics[bench] = bench_metrics
        logger.info(
            "%s: anchor=%.3f  routed=%.3f  gate_open=%.1f%%  cond_gain=%.4f  "
            "helped=%d/%d  avg_explored=%.1f",
            bench,
            bench_metrics["anchor_accuracy"],
            bench_metrics["routed_accuracy"],
            100 * bench_metrics["gate_open_rate"],
            bench_metrics["conditional_gain"],
            helped_when_opened, gate_opened,
            bench_metrics["avg_mcts_explored"],
        )

        global_anchor_correct += anchor_correct
        global_routed_correct += routed_correct
        global_gate_opened += gate_opened
        global_gain_when_opened += gain_when_opened
        global_helped_when_opened += helped_when_opened
        global_total += n
        global_mcts_explored += bench_explored

    summary = {
        "total_questions": global_total,
        "search_mode": "mcts",
        "anchor_accuracy": global_anchor_correct / max(global_total, 1),
        "routed_accuracy": global_routed_correct / max(global_total, 1),
        "gate_open_rate": global_gate_opened / max(global_total, 1),
        "conditional_gain": global_gain_when_opened / max(global_gate_opened, 1),
        "unconditional_gain": global_gain_all / max(global_total, 1),
        "helped_frac_when_opened": global_helped_when_opened / max(global_gate_opened, 1),
        "avg_mcts_explored": global_mcts_explored / max(global_gate_opened, 1),
        "mcts_config": {
            "num_simulations": cfg.mcts_num_simulations,
            "exploration_constant": cfg.mcts_exploration_constant,
            "swap_radius": cfg.swap_radius,
            "max_swaps": cfg.max_local_edits,
            "editable_start": cfg.editable_start,
        },
        "per_benchmark": metrics,
    }
    logger.info("=== GLOBAL (MCTS) ===")
    logger.info(
        "anchor=%.4f  routed=%.4f  gate_open=%.1f%%  cond_gain=%.4f  "
        "uncond_gain=%.4f  avg_explored=%.1f",
        summary["anchor_accuracy"],
        summary["routed_accuracy"],
        100 * summary["gate_open_rate"],
        summary["conditional_gain"],
        summary["unconditional_gain"],
        summary["avg_mcts_explored"],
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fine-routing inference (per-benchmark)")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--results_dir", type=str, default="predictions")
    p.add_argument("--checkpoint_dir", type=str, required=True,
                   help="Dir containing gate_best_{bench}.pt and router_best_{bench}.pt")
    p.add_argument("--benchmarks", nargs="+", default=None)
    p.add_argument("--data_split", type=str, default="validation")
    p.add_argument("--gamma", type=float, default=0.8,
                   help="Gate threshold: open fine router if gate prob >= gamma")
    p.add_argument("--pivot_layer", type=int, default=None)
    p.add_argument("--editable_start", type=int, default=None)
    p.add_argument("--max_local_edits", type=int, default=2)
    p.add_argument("--swap_radius", type=int, default=2)
    p.add_argument("--gpu_rank", type=int, default=0)
    p.add_argument("--output_json", type=str, default=None,
                   help="Save results to JSON file")
    p.add_argument("--use_mcts", action="store_true",
                   help="Use per-question MCTS instead of router when gate opens")
    p.add_argument("--mcts_num_simulations", type=int, default=64,
                   help="MCTS simulations per question (default: 64)")
    p.add_argument("--mcts_exploration_constant", type=float, default=1.8)
    p.add_argument("--mcts_pw_C", type=float, default=1.0)
    p.add_argument("--mcts_pw_alpha", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()

    cfg = FineRoutingConfig(model_name=args.model_name, results_dir=args.results_dir)
    if args.benchmarks:
        cfg.benchmarks = args.benchmarks
    cfg.data_split = args.data_split
    cfg.max_local_edits = args.max_local_edits
    cfg.swap_radius = args.swap_radius
    cfg.gpu_rank = args.gpu_rank
    cfg.use_mcts = args.use_mcts
    cfg.mcts_num_simulations = args.mcts_num_simulations
    cfg.mcts_exploration_constant = args.mcts_exploration_constant
    cfg.mcts_pw_C = args.mcts_pw_C
    cfg.mcts_pw_alpha = args.mcts_pw_alpha
    if args.pivot_layer is not None:
        cfg.pivot_layer = args.pivot_layer
    if args.editable_start is not None:
        cfg.editable_start = args.editable_start

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    logger.info("Loading model %s ...", cfg.model_name)
    wrapper = FlexibleModelWrapper(cfg.model_name, rank=cfg.gpu_rank)
    logger.info("Model: %d layers", wrapper.num_layers)

    # load anchor sequences
    anchor_seqs = load_optimal_sequences_from_results(cfg.results_dir, cfg.benchmarks)
    logger.info("Anchors loaded for: %s", list(anchor_seqs.keys()))

    # load per-benchmark gate (always needed); router only for non-MCTS mode
    gates, routers = load_per_benchmark_models(
        args.checkpoint_dir, cfg.benchmarks, device
    )
    logger.info("Gates loaded for: %s", list(gates.keys()))
    if not cfg.use_mcts:
        logger.info("Routers loaded for: %s", list(routers.keys()))
    else:
        logger.info("MCTS mode: router not needed (sims=%d, radius=%d, max_swaps=%d)",
                     cfg.mcts_num_simulations, cfg.swap_radius, cfg.max_local_edits)

    t0 = time.time()
    if cfg.use_mcts:
        summary = run_inference_mcts(
            cfg, wrapper, gates, anchor_seqs, gamma=args.gamma,
        )
    else:
        summary = run_inference(
            cfg, wrapper, gates, routers, anchor_seqs, gamma=args.gamma,
        )
    elapsed = time.time() - t0
    summary["elapsed_s"] = elapsed
    logger.info("Done in %.1fs", elapsed)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Results saved to %s", args.output_json)


if __name__ == "__main__":
    main()
