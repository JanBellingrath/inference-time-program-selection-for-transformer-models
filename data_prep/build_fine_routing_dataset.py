#!/usr/bin/env python3
"""Build question-level local-search dataset for fine routing.

For each benchmark *b*:
  1.  Load anchor sequence ``s_b`` from existing MCTS snapshots.
  2.  Load questions via ``prepare_arc_data``.
  3.  Enumerate all valid local deviations (1–2 edits in late layers).
  4.  Per question: evaluate anchor + every deviation; store **binary** (TALE)
      and **continuous** (log-prob when applicable) scores, extract pivot
      residual, compute gate labels and router targets.
  5.  Save ``{output_dir}/{benchmark}.pt`` and ``{benchmark}.jsonl``.

Usage
-----
    python build_fine_routing_dataset.py \
        --model_name Qwen/Qwen2.5-0.5B-Instruct \
        --results_dir predictions/qwen25_0.5b_v2_sdpa \
        --benchmarks winogrande boolq commonsenseqa mmlu_all arc_easy \
        --output_dir fine_routing_data
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import hashlib
import json
import logging
import math
import os
import random as _random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from core.benchmark_mcts import grade_response, per_question_mcts, seq_to_layers
from routers.fine_routing_config import FineRoutingConfig
from routers.fine_routing_deviations import (
    NOOP_KEY,
    apply_deviation,
    canonical_key,
    deviation_index_map,
    enumerate_deviations,
    seq_to_layers as dev_seq_to_layers,
)
from core.flexible_models import FlexibleModelWrapper, get_is_instruct
from core.permutation_mcts import prepare_arc_data
from training.train_benchmark_router import load_optimal_sequences_from_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def _count_jsonl_lines(path: str) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


# ---------------------------------------------------------------------------
# Generation helper (mirrors BenchmarkMCTS._generate)
# ---------------------------------------------------------------------------

def generate_under_layers(
    wrapper: FlexibleModelWrapper,
    layers: List[int],
    text: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 1,
    is_math: bool = False,
) -> str:
    """Generate with an arbitrary layer sequence, returning the decoded text."""
    saved = wrapper.model.model.layer_indices
    wrapper.model.model.layer_indices = layers
    try:
        has_dup = len(layers) != len(set(layers))
        full_text = text if system_prompt is None else f"{system_prompt}\n\n{text}"
        prompt = wrapper.prepare_prompt(full_text)
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


def grade_sample(
    wrapper: FlexibleModelWrapper,
    layers: List[int],
    sample: Dict,
    dataset: str,
    model_name: str,
    is_math: bool = False,
) -> float:
    """Grade a single sample under a given layer sequence. Returns 1.0 or 0.0."""
    resp = generate_under_layers(
        wrapper,
        layers,
        sample["input"],
        system_prompt=sample.get("system_prompt"),
        max_new_tokens=sample["max_new_tokens"],
        is_math=is_math,
    )
    return grade_response(resp, sample["correct"], dataset, model_name, sample["input"])


def _forward_logits(
    wrapper: FlexibleModelWrapper,
    layers: List[int],
    text: str,
    system_prompt: Optional[str] = None,
) -> torch.Tensor:
    """Run a single forward pass under *layers* and return last-position logits [vocab]."""
    saved = wrapper.model.model.layer_indices
    wrapper.model.model.layer_indices = layers
    try:
        has_dup = len(layers) != len(set(layers))
        full_text = text if system_prompt is None else f"{system_prompt}\n\n{text}"
        prompt = wrapper.prepare_prompt(full_text)
        inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(wrapper.model.device)
        kw: dict = {}
        if has_dup or len(layers) != wrapper.num_layers:
            kw["use_cache"] = False
        with torch.no_grad():
            out = wrapper.model(input_ids=inputs.input_ids,
                                attention_mask=inputs.attention_mask, **kw)
        return out.logits[0, -1, :]
    finally:
        wrapper.model.model.layer_indices = saved


def grade_sample_continuous(
    wrapper: FlexibleModelWrapper,
    layers: List[int],
    sample: Dict,
    dataset: str,
    model_name: str,
    is_math: bool = False,
) -> float:
    """Return log-prob of the correct answer label under *layers*.

    For multiple-choice tasks with ``max_new_tokens == 1`` this is the
    log-softmax entry for the correct answer token — a **continuous**
    score that replaces the binary 0/1 grading and gives meaningful
    gradients between routes.

    Falls back to binary grading for generative (multi-token) tasks
    where extracting a clean log-prob is not straightforward.
    """
    if sample["max_new_tokens"] != 1:
        return grade_sample(wrapper, layers, sample, dataset, model_name, is_math)

    logits = _forward_logits(
        wrapper, layers, sample["input"],
        system_prompt=sample.get("system_prompt"),
    )
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    correct = sample["correct"].strip()
    tok_ids = wrapper.tokenizer.encode(correct, add_special_tokens=False)
    if tok_ids:
        return log_probs[tok_ids[0]].item()
    return grade_sample(wrapper, layers, sample, dataset, model_name, is_math)


def grade_sample_both(
    wrapper: FlexibleModelWrapper,
    layers: List[int],
    sample: Dict,
    dataset: str,
    model_name: str,
    is_math: bool = False,
) -> Tuple[float, float]:
    """Binary TALE score and continuous (log-prob) score for the same forward route.

    For ``max_new_tokens != 1``, :func:`grade_sample_continuous` falls back to
    binary, so this returns ``(b, b)`` after a single :func:`grade_sample` call.
    For single-token multiple-choice, runs generation (TALE) and a log-softmax
    forward to obtain both.
    """
    ntok = int(sample.get("max_new_tokens", 1))
    if ntok != 1:
        b = grade_sample(wrapper, layers, sample, dataset, model_name, is_math)
        return b, b
    b = grade_sample(wrapper, layers, sample, dataset, model_name, is_math)
    c = grade_sample_continuous(wrapper, layers, sample, dataset, model_name, is_math)
    return b, c


# ---------------------------------------------------------------------------
# Router target distribution
# ---------------------------------------------------------------------------

def compute_router_target(
    deltas: List[float],
    beta: float,
    clip_val: float,
) -> List[float]:
    """Softmax over clipped deltas -> pi_target distribution."""
    clipped = [max(-clip_val, min(clip_val, d)) for d in deltas]
    logits = [beta * c for c in clipped]
    max_l = max(logits)
    exps = [math.exp(l - max_l) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


# ---------------------------------------------------------------------------
# Sample hash (for logging / dedup)
# ---------------------------------------------------------------------------

def _sample_hash(sample: Dict) -> str:
    text = sample["input"] + str(sample.get("correct", ""))
    return hashlib.md5(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Main dataset builder
# ---------------------------------------------------------------------------

def build_dataset_for_benchmark(
    cfg: FineRoutingConfig,
    wrapper: FlexibleModelWrapper,
    benchmark: str,
    anchor_seq: List[int],
    samples: List[Dict],
    output_dir: str,
    save_interval: int = 200,
) -> Dict:
    """Build the fine-routing dataset for a single benchmark.

    Saves incrementally every save_interval samples (jsonl append + pt overwrite).

    Returns a dict with keys:
        pivot_residuals  : Tensor [N, d_model]
        records          : list of per-question dicts (jsonl rows)
    """
    num_layers = wrapper.num_layers
    is_math = "dart" in benchmark or benchmark in ("gsm8k_hard", "math500")
    anchor_layers = seq_to_layers(anchor_seq)

    deviations = enumerate_deviations(
        anchor_seq,
        editable_start=cfg.editable_start,
        num_layers=num_layers,
        swap_radius=cfg.swap_radius,
        max_edits=cfg.max_local_edits,
    )
    dev_keys = [canonical_key(d) for d in deviations]
    logger.info(
        "  Benchmark %s: anchor=%s  |deviations|=%d  questions=%d (save every %d)",
        benchmark, anchor_layers, len(deviations), len(samples), save_interval,
    )

    pivot_residuals: List[torch.Tensor] = []
    records: List[Dict] = []
    jsonl_path = os.path.join(output_dir, f"{benchmark}.jsonl")
    pt_path = os.path.join(output_dir, f"{benchmark}_pivot_residuals.pt")

    with open(jsonl_path, "w") as jsonl_file:
        for q_idx, sample in enumerate(tqdm(samples, desc=f"  {benchmark}")):
            # --- anchor: binary (TALE) + continuous (log-prob when applicable) ---
            anchor_b, anchor_c = grade_sample_both(
                wrapper, anchor_layers, sample, benchmark, cfg.model_name, is_math
            )
            # Gate / router (exhaustive): primary = binary, unchanged for downstream.
            anchor_score = anchor_b

            # --- pivot residual (under anchor layer order) ---
            pivot_res = wrapper.get_pivot_residual(
                sample["input"],
                layer_indices=anchor_layers,
                pivot_layer=cfg.pivot_layer,
                system_prompt=sample.get("system_prompt"),
            )
            pivot_residuals.append(pivot_res.cpu().squeeze(0))  # [d_model]

            # --- evaluate every deviation ---
            scores_b: List[float] = []
            scores_c: List[float] = []
            deltas_b: List[float] = []
            deltas_c: List[float] = []
            for dev_idx, deviation in enumerate(deviations):
                if not deviation:  # no-op
                    scores_b.append(anchor_b)
                    scores_c.append(anchor_c)
                    deltas_b.append(0.0)
                    deltas_c.append(0.0)
                    continue
                cand_seq = apply_deviation(anchor_seq, deviation)
                cand_layers = seq_to_layers(cand_seq)
                sc_b, sc_c = grade_sample_both(
                    wrapper, cand_layers, sample, benchmark, cfg.model_name, is_math
                )
                scores_b.append(sc_b)
                scores_c.append(sc_c)
                deltas_b.append(sc_b - anchor_b)
                deltas_c.append(sc_c - anchor_c)

            # --- gate label (binary deltas) ---
            non_noop_deltas = [d for k, d in zip(dev_keys, deltas_b) if k != NOOP_KEY]
            best_delta = max(non_noop_deltas) if non_noop_deltas else 0.0
            gate_label = int(best_delta > cfg.gate_tau)

            # --- router target (primary = binary) + continuous auxiliary ---
            router_target = compute_router_target(
                deltas_b, beta=cfg.target_beta, clip_val=cfg.delta_clip
            )
            router_target_continuous = compute_router_target(
                deltas_c, beta=cfg.continuous_target_beta, clip_val=cfg.continuous_delta_clip
            )

            rec = {
                "benchmark_id": benchmark,
                "question_id": q_idx,
                "question_hash": _sample_hash(sample),
                "anchor_sequence": anchor_seq,
                "anchor_score": anchor_score,
                "anchor_score_binary": anchor_b,
                "anchor_score_continuous": anchor_c,
                "scoring_mode": "both",
                "primary_scoring": "binary",
                "pivot_layer_index": cfg.pivot_layer,
                "deviations": [
                    {
                        "key": dev_keys[i],
                        "score": scores_b[i],
                        "delta": deltas_b[i],
                        "score_binary": scores_b[i],
                        "delta_binary": deltas_b[i],
                        "score_continuous": scores_c[i],
                        "delta_continuous": deltas_c[i],
                    }
                    for i in range(len(deviations))
                ],
                "gate_label": gate_label,
                "router_target": router_target,
                "router_target_binary": router_target,
                "router_target_continuous": router_target_continuous,
            }
            records.append(rec)
            jsonl_file.write(json.dumps(rec) + "\n")

            # --- incremental save every save_interval samples ---
            if (q_idx + 1) % save_interval == 0:
                jsonl_file.flush()
                pivot_tensor = torch.stack(pivot_residuals)
                torch.save(pivot_tensor, pt_path)
                logger.info("  %s: saved %d samples", benchmark, len(records))

    pivot_tensor = torch.stack(pivot_residuals) if pivot_residuals else torch.empty(0)
    torch.save(pivot_tensor, pt_path)
    return {"pivot_residuals": pivot_tensor, "records": records}


# ---------------------------------------------------------------------------
# MCTS-based dataset builder (scales beyond exhaustive enumeration)
# ---------------------------------------------------------------------------

def build_dataset_mcts_for_benchmark(
    cfg: FineRoutingConfig,
    wrapper: FlexibleModelWrapper,
    benchmark: str,
    anchor_seq: List[int],
    samples: List[Dict],
    output_dir: str,
    save_interval: int = 200,
    resume: bool = False,
) -> Dict:
    """Build fine-routing dataset using per-question MCTS instead of exhaustive search.

    For each question, runs :func:`per_question_mcts` (from benchmark_mcts)
    anchored on *anchor_seq* with ``editable_start`` restricting mutations
    to later layers.  Supports larger ``swap_radius`` / ``max_local_edits``
    than the exhaustive path because MCTS explores smartly.

    Output format per question is compatible with the exhaustive builder:
    pivot residuals, gate labels, and per-explored-sequence scores.  The
    ``router_target`` is a softmax over all MCTS-explored sequences.

    ``cfg.use_continuous_scoring`` selects the **primary** signal used inside
    MCTS (UCB) and for top-level ``anchor_score`` / ``score`` / ``delta`` /
    ``router_target`` / ``gate_label``.  Binary (TALE) and continuous
    (log-prob) scores are **always** recorded on each row in explicit fields
    (``anchor_score_binary``, ``anchor_score_continuous``, and per explored
    route the ``score_*`` / ``delta_*`` pair).

    When ``cfg.mcts_dual_seed`` is True, per-question MCTS is run twice
    with different random seeds and ``gate_label`` is set to 1 only when
    **both** runs find an improvement, reducing label noise.
    """
    num_layers = wrapper.num_layers
    is_math = "dart" in benchmark or benchmark in ("gsm8k_hard", "math500")
    anchor_layers = seq_to_layers(anchor_seq)

    continuous = getattr(cfg, "use_continuous_scoring", False)
    dual_seed = getattr(cfg, "mcts_dual_seed", False)
    if continuous:
        delta_clip = cfg.continuous_delta_clip
        target_beta = cfg.continuous_target_beta
        gate_tau = cfg.continuous_gate_tau
    else:
        delta_clip = cfg.delta_clip
        target_beta = cfg.target_beta
        gate_tau = cfg.gate_tau

    logger.info(
        "  Benchmark %s (MCTS): anchor=%s  sims/q=%d  radius=%d  max_swaps=%d  "
        "editable_start=%d  questions=%d  continuous=%s  dual_seed=%s",
        benchmark, anchor_layers, cfg.mcts_num_simulations,
        cfg.swap_radius, cfg.max_local_edits, cfg.editable_start,
        len(samples), continuous, dual_seed,
    )

    pivot_residuals: List[torch.Tensor] = []
    records: List[Dict] = []
    jsonl_path = os.path.join(output_dir, f"{benchmark}.jsonl")
    pt_path = os.path.join(output_dir, f"{benchmark}_pivot_residuals.pt")

    start_idx = 0
    jsonl_mode = "w"
    if resume and os.path.isfile(jsonl_path):
        start_idx = _count_jsonl_lines(jsonl_path)
        if start_idx > 0:
            jsonl_mode = "a"
            logger.info(
                "  Resume: %d rows already in %s — continuing from question_id=%d",
                start_idx, jsonl_path, start_idx,
            )
        if start_idx >= len(samples):
            logger.info("  Resume: dataset already complete (%d >= %d).", start_idx, len(samples))
            if os.path.isfile(pt_path):
                pivot_tensor = torch.load(pt_path, map_location="cpu", weights_only=True).float()
            else:
                pivot_tensor = torch.empty(0)
            return {
                "pivot_residuals": pivot_tensor,
                "records": [],
                "resumed_from": start_idx,
            }

        # --- align pivot tensor with jsonl (may be short if crash between checkpoints) ---
        if os.path.isfile(pt_path):
            pt_loaded = torch.load(pt_path, map_location="cpu", weights_only=True).float()
            n_pt = pt_loaded.shape[0]
            if n_pt > start_idx:
                raise ValueError(
                    f"Corrupt checkpoint: {pt_path} has {n_pt} rows but {jsonl_path} has {start_idx} lines"
                )
            if n_pt < start_idx:
                logger.info(
                    "  Recomputing %d pivot residuals (indices %d..%d) to align with jsonl",
                    start_idx - n_pt, n_pt, start_idx - 1,
                )
                pivot_residuals = list(torch.unbind(pt_loaded, dim=0))
                for q_idx in tqdm(
                    range(n_pt, start_idx),
                    desc=f"  {benchmark} (pivot catch-up)",
                ):
                    sample = samples[q_idx]
                    pivot_res = wrapper.get_pivot_residual(
                        sample["input"],
                        layer_indices=anchor_layers,
                        pivot_layer=cfg.pivot_layer,
                        system_prompt=sample.get("system_prompt"),
                    )
                    pivot_residuals.append(pivot_res.cpu().squeeze(0))
                torch.save(torch.stack(pivot_residuals), pt_path)
                logger.info("  Saved aligned pivot tensor (%d rows)", len(pivot_residuals))
            else:
                pivot_residuals = list(torch.unbind(pt_loaded, dim=0))
        else:
            logger.warning(
                "  No %s — recomputing pivot residuals for first %d samples",
                os.path.basename(pt_path),
                start_idx,
            )
            for q_idx in tqdm(range(start_idx), desc=f"  {benchmark} (pivot rebuild)"):
                sample = samples[q_idx]
                pivot_res = wrapper.get_pivot_residual(
                    sample["input"],
                    layer_indices=anchor_layers,
                    pivot_layer=cfg.pivot_layer,
                    system_prompt=sample.get("system_prompt"),
                )
                pivot_residuals.append(pivot_res.cpu().squeeze(0))
            torch.save(torch.stack(pivot_residuals), pt_path)

    with open(jsonl_path, jsonl_mode) as jsonl_file:
        for q_idx in tqdm(
            range(start_idx, len(samples)),
            desc=f"  {benchmark} (mcts)",
            initial=start_idx,
            total=len(samples),
        ):
            sample = samples[q_idx]
            # --- pivot residual (under anchor layer order) ---
            use_multi = getattr(cfg, "use_multi_layer_pivot", False)
            include_conf = getattr(cfg, "include_anchor_confidence", False)
            if use_multi or include_conf:
                multi_layers = getattr(cfg, "pivot_layers_multi", [12, 14, 16])
                residual, logits = wrapper.get_multi_layer_residuals(
                    sample["input"],
                    layer_indices=anchor_layers,
                    pivot_layers=multi_layers if use_multi else [cfg.pivot_layer],
                    system_prompt=sample.get("system_prompt"),
                    return_logits=True,
                )
                parts = [residual.cpu().squeeze(0)]
                if include_conf:
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    top2 = probs.topk(2).values
                    conf = top2[0].item()
                    margin = (top2[0] - top2[1]).item()
                    parts.append(torch.tensor([conf, margin], dtype=torch.float32))
                pivot_res_vec = torch.cat(parts, dim=-1)
            else:
                pivot_res_vec = wrapper.get_pivot_residual(
                    sample["input"],
                    layer_indices=anchor_layers,
                    pivot_layer=cfg.pivot_layer,
                    system_prompt=sample.get("system_prompt"),
                ).cpu().squeeze(0)
            pivot_residuals.append(pivot_res_vec)

            # --- per-question MCTS ---
            _grade_fn = grade_sample_continuous if continuous else grade_sample

            def _grade(seq: List[int]) -> float:
                layers = seq_to_layers(seq)
                if not layers:
                    return -30.0 if continuous else 0.0
                return _grade_fn(
                    wrapper, layers, sample, benchmark, cfg.model_name, is_math,
                )

            _rng_state = _random.getstate()

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

            anchor_score = mcts_result["anchor_score"]
            best_delta = mcts_result["best_delta"]

            def _both_for_seq(seq: List[int]) -> Tuple[float, float]:
                layers = seq_to_layers(seq)
                if not layers:
                    return 0.0, -30.0
                return grade_sample_both(
                    wrapper, layers, sample, benchmark, cfg.model_name, is_math
                )

            # --- optional dual-seed stabilisation ---
            dual_gate_agree = True
            if dual_seed:
                _random.seed(_random.randint(0, 2**31) ^ (q_idx * 7919 + 104729))
                mcts_result_2 = per_question_mcts(
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
                dual_gate_agree = mcts_result_2["best_delta"] > gate_tau
                explored_2 = mcts_result_2["explored"]
                for k, v in explored_2.items():
                    explored_main = mcts_result["explored"]
                    if k not in explored_main:
                        explored_main[k] = v
                _random.setstate(_rng_state)

            gate_label = int(best_delta > gate_tau and dual_gate_agree)

            anchor_b, anchor_c = _both_for_seq(anchor_seq)
            best_b, best_c = _both_for_seq(mcts_result["best_seq"])
            best_delta_b = best_b - anchor_b
            best_delta_c = best_c - anchor_c

            explored = mcts_result["explored"]
            seq_keys = list(explored.keys())
            deltas = [explored[k] - anchor_score for k in seq_keys]
            router_target = compute_router_target(
                deltas, beta=target_beta, clip_val=delta_clip,
            )

            deltas_b: List[float] = []
            deltas_c: List[float] = []
            explored_list: List[Dict] = []
            for k in seq_keys:
                b, c = _both_for_seq(list(k))
                d_b = b - anchor_b
                d_c = c - anchor_c
                deltas_b.append(d_b)
                deltas_c.append(d_c)
                explored_list.append(
                    {
                        "seq": list(k),
                        "score": explored[k],
                        "delta": explored[k] - anchor_score,
                        "score_binary": b,
                        "delta_binary": d_b,
                        "score_continuous": c,
                        "delta_continuous": d_c,
                    }
                )

            router_target_binary = compute_router_target(
                deltas_b, beta=cfg.target_beta, clip_val=cfg.delta_clip
            )
            router_target_continuous = compute_router_target(
                deltas_c, beta=cfg.continuous_target_beta, clip_val=cfg.continuous_delta_clip
            )

            rec = {
                "benchmark_id": benchmark,
                "question_id": q_idx,
                "question_hash": _sample_hash(sample),
                "anchor_sequence": anchor_seq,
                "anchor_score": anchor_score,
                "anchor_score_binary": anchor_b,
                "anchor_score_continuous": anchor_c,
                "pivot_layer_index": cfg.pivot_layer,
                "search_mode": "mcts",
                "scoring_mode": "both",
                "primary_scoring": "continuous" if continuous else "binary",
                "gate_label": gate_label,
                "best_seq": mcts_result["best_seq"],
                "best_score": mcts_result["best_score"],
                "best_score_binary": best_b,
                "best_score_continuous": best_c,
                "best_delta": best_delta,
                "best_delta_binary": best_delta_b,
                "best_delta_continuous": best_delta_c,
                "num_explored": mcts_result["num_explored"],
                "explored": explored_list,
                "router_target": router_target,
                "router_target_binary": router_target_binary,
                "router_target_continuous": router_target_continuous,
            }
            records.append(rec)
            jsonl_file.write(json.dumps(rec) + "\n")

            if (q_idx + 1) % save_interval == 0:
                jsonl_file.flush()
                pivot_tensor = torch.stack(pivot_residuals)
                torch.save(pivot_tensor, pt_path)
                logger.info("  %s: saved %d total rows (mcts)", benchmark, len(pivot_residuals))

    pivot_tensor = torch.stack(pivot_residuals) if pivot_residuals else torch.empty(0)
    torch.save(pivot_tensor, pt_path)
    return {
        "pivot_residuals": pivot_tensor,
        "records": records,
        "resumed_from": start_idx,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> FineRoutingConfig:
    p = argparse.ArgumentParser(description="Build fine-routing dataset")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--results_dir", type=str, default="predictions")
    p.add_argument("--benchmarks", nargs="+", default=None)
    p.add_argument("--output_dir", type=str, default="fine_routing_data")
    p.add_argument("--data_split", type=str, default="train")
    p.add_argument("--max_local_edits", type=int, default=2)
    p.add_argument("--swap_radius", type=int, default=2)
    p.add_argument("--pivot_layer", type=int, default=None)
    p.add_argument("--editable_start", type=int, default=None)
    p.add_argument("--delta_clip", type=float, default=1.0)
    p.add_argument("--target_beta", type=float, default=5.0)
    p.add_argument("--gate_tau", type=float, default=0.0)
    p.add_argument("--gpu_rank", type=int, default=0)
    p.add_argument("--max_questions", type=int, default=None,
                   help="Cap questions per benchmark (for quick debugging)")
    p.add_argument("--save_interval", type=int, default=200,
                   help="Save checkpoint every N samples (default: 200)")
    p.add_argument("--use_mcts", action="store_true",
                   help="Use per-question MCTS instead of exhaustive enumeration")
    p.add_argument("--mcts_num_simulations", type=int, default=64,
                   help="MCTS simulations per question (default: 64)")
    p.add_argument("--mcts_exploration_constant", type=float, default=1.8)
    p.add_argument("--mcts_pw_C", type=float, default=1.0)
    p.add_argument("--mcts_pw_alpha", type=float, default=0.5)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Append to existing jsonl / continue MCTS from last row (MCTS mode only)",
    )
    p.add_argument(
        "--use_continuous_scoring",
        action="store_true",
        help=(
            "MCTS+router primary = log-prob of correct label (log-softmax) instead "
            "of TALE 0/1. Binary and continuous scores are always written to jsonl; "
            "this only selects UCB/primary field semantics."
        ),
    )
    p.add_argument(
        "--mcts_dual_seed",
        action="store_true",
        help="Run MCTS twice with different seeds; gate_label=1 only if both agree",
    )
    p.add_argument(
        "--use_multi_layer_pivot",
        action="store_true",
        help="Extract residuals from multiple layers (12,14,16) instead of single pivot",
    )
    p.add_argument(
        "--include_anchor_confidence",
        action="store_true",
        help="Append anchor model confidence + margin features to pivot residual",
    )
    args = p.parse_args()

    cfg = FineRoutingConfig(model_name=args.model_name, results_dir=args.results_dir)
    if args.benchmarks:
        cfg.benchmarks = args.benchmarks
    cfg.output_dir = args.output_dir
    cfg.data_split = args.data_split
    cfg.max_local_edits = args.max_local_edits
    cfg.swap_radius = args.swap_radius
    cfg.delta_clip = args.delta_clip
    cfg.target_beta = args.target_beta
    cfg.gate_tau = args.gate_tau
    cfg.gpu_rank = args.gpu_rank
    if args.pivot_layer is not None:
        cfg.pivot_layer = args.pivot_layer
    if args.editable_start is not None:
        cfg.editable_start = args.editable_start
    cfg.use_mcts = args.use_mcts
    cfg.mcts_num_simulations = args.mcts_num_simulations
    cfg.mcts_exploration_constant = args.mcts_exploration_constant
    cfg.mcts_pw_C = args.mcts_pw_C
    cfg.mcts_pw_alpha = args.mcts_pw_alpha
    cfg.use_continuous_scoring = args.use_continuous_scoring
    cfg.mcts_dual_seed = args.mcts_dual_seed
    cfg.use_multi_layer_pivot = args.use_multi_layer_pivot
    cfg.include_anchor_confidence = args.include_anchor_confidence
    return cfg, args.max_questions, args.save_interval, args.resume


def main():
    cfg, max_questions, save_interval, resume = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # save config
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # load model
    logger.info("Loading model %s on GPU %d ...", cfg.model_name, cfg.gpu_rank)
    wrapper = FlexibleModelWrapper(cfg.model_name, rank=cfg.gpu_rank)
    logger.info("Model loaded: %d layers, d_model inferred at runtime", wrapper.num_layers)

    # load anchor sequences
    anchor_seqs = load_optimal_sequences_from_results(
        cfg.results_dir, cfg.benchmarks, model_name=cfg.model_name
    )
    logger.info("Loaded anchor sequences for: %s", list(anchor_seqs.keys()))

    is_instruct = get_is_instruct(cfg.model_name)
    all_records: List[Dict] = []

    for bench in cfg.benchmarks:
        if bench not in anchor_seqs:
            logger.warning("No anchor sequence for %s -- skipping", bench)
            continue
        anchor_seq = anchor_seqs[bench]

        samples = prepare_arc_data(bench, is_instruct=is_instruct, split=cfg.data_split)
        if max_questions is not None:
            samples = samples[:max_questions]
        if not samples:
            logger.warning("No samples for %s -- skipping", bench)
            continue

        t0 = time.time()
        if resume and not cfg.use_mcts:
            logger.warning("--resume is only used with --use_mcts; ignoring")
        builder_fn = build_dataset_mcts_for_benchmark if cfg.use_mcts else build_dataset_for_benchmark
        if cfg.use_mcts:
            result = builder_fn(
                cfg, wrapper, bench, anchor_seq, samples,
                output_dir=cfg.output_dir,
                save_interval=save_interval,
                resume=resume,
            )
        else:
            result = builder_fn(
                cfg, wrapper, bench, anchor_seq, samples,
                output_dir=cfg.output_dir,
                save_interval=save_interval,
            )
        elapsed = time.time() - t0

        n_new = len(result["records"])
        prev = result.get("resumed_from", 0)
        n_total = prev + n_new
        gate_pos = sum(1 for r in result["records"] if r["gate_label"] == 1)
        logger.info(
            "  %s done: +%d questions (total %d), gate_positive=%d in batch (%.1f%%), %.1fs",
            bench, n_new, n_total, gate_pos, 100 * gate_pos / max(n_new, 1), elapsed,
        )
        all_records.extend(result["records"])

    # global summary
    total = len(all_records)
    total_gp = sum(1 for r in all_records if r["gate_label"] == 1)
    logger.info(
        "Dataset complete: %d questions, %d gate-positive (%.1f%%)",
        total, total_gp, 100 * total_gp / max(total, 1),
    )

    # save deviation catalog per benchmark (needed by router to know class count)
    # In MCTS mode the deviation space is explored dynamically per question,
    # so we record the search config instead of a static catalog.
    catalog_path = os.path.join(cfg.output_dir, "deviation_catalog.json")
    if cfg.use_mcts:
        catalog = {
            "_search_mode": "mcts",
            "_mcts_num_simulations": cfg.mcts_num_simulations,
            "_swap_radius": cfg.swap_radius,
            "_max_swaps": cfg.max_local_edits,
            "_editable_start": cfg.editable_start,
        }
    else:
        catalog: Dict[str, List[str]] = {}
        for bench in cfg.benchmarks:
            if bench not in anchor_seqs:
                continue
            devs = enumerate_deviations(
                anchor_seqs[bench],
                editable_start=cfg.editable_start,
                num_layers=wrapper.num_layers,
                swap_radius=cfg.swap_radius,
                max_edits=cfg.max_local_edits,
            )
            catalog[bench] = [canonical_key(d) for d in devs]
    with open(catalog_path, "w") as f:
        json.dump(catalog, f, indent=2)
    logger.info("Deviation catalog saved to %s", catalog_path)


if __name__ == "__main__":
    main()
