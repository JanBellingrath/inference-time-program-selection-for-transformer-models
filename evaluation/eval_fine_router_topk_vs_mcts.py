#!/usr/bin/env python3
"""Compare fine-router top-K marginalization vs MCTS top-K routes.

This script evaluates, per benchmark, whether marginalizing over:
  1) top-K routes from a trained fine-router (per-question logits), or
  2) top-K validation-ranked MCTS routes (global ranking from snapshot tier4)
performs better on downstream task accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from core.benchmark_mcts import seq_to_layers
from core.flexible_models import FlexibleModelWrapper, get_is_instruct
from core.permutation_mcts import prepare_arc_data
from pipeline.forward import get_pivot_residual as _get_pivot_residual
from routers.fine_routing_config import FineRoutingConfig
from routers.fine_routing_deviations import (
    apply_deviation,
    enumerate_deviations,
)
from training.train_benchmark_router import load_optimal_sequences_from_results
from training.train_fine_router import FineRouter


def _load_router(path: str, device: torch.device) -> FineRouter:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    router = FineRouter(
        d_model=ckpt["d_model"],
        num_classes=ckpt["num_classes"],
        hidden_dims=ckpt["hidden_dims"],
        dropout=ckpt.get("dropout", 0.1),
    ).to(device)
    router.load_state_dict(ckpt["model_state_dict"])
    router.eval()
    return router


def _forward_logits(wrapper: FlexibleModelWrapper, layers: List[int], text: str) -> torch.Tensor:
    saved = wrapper.model.model.layer_indices
    wrapper.model.model.layer_indices = layers
    try:
        prompt = wrapper.prepare_prompt(text, system_prompt=None)
        inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(wrapper.model.device)
        with torch.no_grad():
            out = wrapper.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                return_dict=True,
                use_cache=False,
            )
        return out.logits[0, -1, :].float()
    finally:
        wrapper.model.model.layer_indices = saved


def _answer_token_ids(wrapper: FlexibleModelWrapper, options: Sequence[str]) -> List[int]:
    out: List[int] = []
    for o in options:
        toks = wrapper.tokenizer.encode(o.strip(), add_special_tokens=False)
        out.append(toks[0] if toks else -1)
    return out


def _infer_router_catalog(
    anchor_seq: List[int],
    num_layers: int,
    editable_start: int,
    target_num_classes: int,
) -> List[List[int]]:
    # Try a small grid that matches historical fine-routing runs.
    for max_edits in (1, 2):
        for swap_radius in (1, 2, 3):
            devs = enumerate_deviations(
                anchor_seq=anchor_seq,
                editable_start=editable_start,
                num_layers=num_layers,
                swap_radius=swap_radius,
                max_edits=max_edits,
            )
            if len(devs) == target_num_classes:
                return [apply_deviation(anchor_seq, d) for d in devs]
    raise ValueError(
        f"Could not infer router catalog with {target_num_classes=} for anchor tail={anchor_seq[-4:]}"
    )


def _load_mcts_ranked_routes(snapshot_path: str) -> List[List[int]]:
    with open(snapshot_path) as f:
        snap = json.load(f)
    tier4 = snap.get("tier4", [])
    if not tier4:
        raise ValueError(f"No tier4 routes in snapshot: {snapshot_path}")
    ranked = sorted(tier4, key=lambda r: (r["accuracy"], r["delta"]), reverse=True)
    return [list(map(int, r["seq"])) for r in ranked]


def _pred_and_logprob(
    probs: torch.Tensor,
    answer_ids: List[int],
    answer_options: List[str],
    correct_option: str,
) -> Dict[str, float]:
    answer_probs = probs[answer_ids]
    pred_idx = int(answer_probs.argmax().item())
    pred = answer_options[pred_idx]
    correct_idx = answer_options.index(correct_option)
    return {
        "accuracy": float(pred == correct_option),
        "logprob": float(probs[answer_ids[correct_idx]].clamp(min=1e-30).log().item()),
    }


def _aggregate(rows: List[Dict[str, float]], anchor_rows: List[Dict[str, float]]) -> Dict[str, float]:
    accs = np.array([r["accuracy"] for r in rows], dtype=np.float64)
    lps = np.array([r["logprob"] for r in rows], dtype=np.float64)
    a_acc = np.array([r["accuracy"] for r in anchor_rows], dtype=np.float64)
    a_lp = np.array([r["logprob"] for r in anchor_rows], dtype=np.float64)
    return {
        "accuracy": float(accs.mean()),
        "accuracy_delta_vs_anchor": float((accs - a_acc).mean()),
        "logprob_delta_vs_anchor": float((lps - a_lp).mean()),
        "frac_beat_anchor_acc": float((accs > a_acc).mean()),
        "frac_beat_anchor_lp": float((lps > a_lp).mean()),
    }


def evaluate_benchmark(
    wrapper: FlexibleModelWrapper,
    benchmark: str,
    router: FineRouter,
    anchor_seq: List[int],
    router_catalog: List[List[int]],
    mcts_ranked_routes: List[List[int]],
    top_k_values: List[int],
    max_questions: int,
) -> Dict[str, Any]:
    answer_options = {
        "boolq": ["True", "False"],
        "commonsenseqa": ["A", "B", "C", "D", "E"],
    }[benchmark]
    answer_ids = _answer_token_ids(wrapper, answer_options)

    is_instruct = get_is_instruct(wrapper.model_name)
    samples = prepare_arc_data(benchmark, is_instruct=is_instruct, split="validation")
    if max_questions > 0:
        samples = samples[:max_questions]
    anchor_layers = seq_to_layers(anchor_seq)
    max_k = max(top_k_values)
    max_k = min(max_k, len(mcts_ranked_routes))

    per_strategy: Dict[str, List[Dict[str, float]]] = {"anchor": []}
    for k in top_k_values:
        for name in (f"router-uniform-{k}", f"router-wtd-{k}", f"mcts-uniform-{k}"):
            per_strategy[name] = []

    device = next(router.parameters()).device
    for sample in tqdm(samples, desc=f"eval({benchmark})"):
        correct = sample.get("correct", "")
        if correct not in answer_options:
            continue
        text = sample["input"]

        anchor_logits = _forward_logits(wrapper, anchor_layers, text)
        anchor_probs = F.softmax(anchor_logits, dim=-1)
        per_strategy["anchor"].append(
            _pred_and_logprob(anchor_probs, answer_ids, answer_options, correct)
        )

        h_pivot = _get_pivot_residual(
            wrapper,
            text,
            layer_indices=anchor_layers,
            pivot_layer=16,
        ).float().to(device).unsqueeze(0)
        with torch.no_grad():
            router_logits = router(h_pivot).squeeze(0)
            router_probs = F.softmax(router_logits, dim=-1)
        topk_vals, topk_idxs = router_probs.topk(min(max_k, router_probs.shape[0]))
        router_weights = topk_vals.float()

        router_cand_probs: List[torch.Tensor] = []
        for idx in topk_idxs.tolist():
            cand_layers = seq_to_layers(router_catalog[int(idx)])
            cand_logits = _forward_logits(wrapper, cand_layers, text)
            router_cand_probs.append(F.softmax(cand_logits, dim=-1))
        router_stack = torch.stack(router_cand_probs, dim=0)

        mcts_cand_probs: List[torch.Tensor] = []
        for seq in mcts_ranked_routes[:max_k]:
            cand_logits = _forward_logits(wrapper, seq_to_layers(seq), text)
            mcts_cand_probs.append(F.softmax(cand_logits, dim=-1))
        mcts_stack = torch.stack(mcts_cand_probs, dim=0)

        for k in top_k_values:
            rk = min(k, router_stack.shape[0])
            mk = min(k, mcts_stack.shape[0])

            r_uni = router_stack[:rk].mean(dim=0)
            per_strategy[f"router-uniform-{k}"].append(
                _pred_and_logprob(r_uni, answer_ids, answer_options, correct)
            )

            rw = router_weights[:rk] / router_weights[:rk].sum().clamp(min=1e-12)
            r_wtd = (rw.unsqueeze(1) * router_stack[:rk]).sum(dim=0)
            per_strategy[f"router-wtd-{k}"].append(
                _pred_and_logprob(r_wtd, answer_ids, answer_options, correct)
            )

            m_uni = mcts_stack[:mk].mean(dim=0)
            per_strategy[f"mcts-uniform-{k}"].append(
                _pred_and_logprob(m_uni, answer_ids, answer_options, correct)
            )

    anchor_rows = per_strategy["anchor"]
    summary = {name: _aggregate(rows, anchor_rows) for name, rows in per_strategy.items()}
    return {
        "benchmark": benchmark,
        "n_questions": len(anchor_rows),
        "top_k_values": top_k_values,
        "summary": summary,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Router top-K vs MCTS top-K marginalization")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--results_dir", type=str, default="predictions/qwen25_0.5b_v2_sdpa")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/fine_routing")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--benchmarks", type=str, nargs="+", default=["boolq", "commonsenseqa"])
    p.add_argument("--top_k_values", type=int, nargs="+", default=[2, 4, 8, 16])
    p.add_argument("--max_questions", type=int, default=0)
    p.add_argument("--output_json", type=str, default="results_router_vs_mcts_topk.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wrapper = FlexibleModelWrapper(args.model_name, rank=0)
    cfg = FineRoutingConfig(model_name=args.model_name, results_dir=args.results_dir)

    benchmarks = args.benchmarks
    anchors = load_optimal_sequences_from_results(
        args.results_dir,
        benchmarks,
        model_name=args.model_name,
    )

    results: Dict[str, Any] = {
        "model_name": args.model_name,
        "results_dir": args.results_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "top_k_values": args.top_k_values,
        "benchmarks": {},
    }

    for bench in benchmarks:
        router_path = Path(args.checkpoint_dir) / f"router_best_{bench}.pt"
        if not router_path.is_file():
            raise FileNotFoundError(f"Missing router checkpoint: {router_path}")
        router = _load_router(str(router_path), device)
        anchor_seq = anchors[bench]
        router_catalog = _infer_router_catalog(
            anchor_seq=anchor_seq,
            num_layers=wrapper.num_layers,
            editable_start=cfg.editable_start,
            target_num_classes=router.net[-1].out_features,
        )

        snap_glob = sorted(Path(args.results_dir).glob(f"benchmark_mcts_{bench}_*_snapshot.json"))
        if not snap_glob:
            raise FileNotFoundError(f"No snapshot for {bench} in {args.results_dir}")
        snapshot_path = str(snap_glob[-1])
        mcts_ranked_routes = _load_mcts_ranked_routes(snapshot_path)

        out = evaluate_benchmark(
            wrapper=wrapper,
            benchmark=bench,
            router=router,
            anchor_seq=anchor_seq,
            router_catalog=router_catalog,
            mcts_ranked_routes=mcts_ranked_routes,
            top_k_values=args.top_k_values,
            max_questions=args.max_questions,
        )
        out["snapshot_path"] = snapshot_path
        out["router_num_classes"] = router.net[-1].out_features
        out["mcts_ranked_routes"] = len(mcts_ranked_routes)
        results["benchmarks"][bench] = out

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({"saved": args.output_json}, indent=2))


if __name__ == "__main__":
    main()

