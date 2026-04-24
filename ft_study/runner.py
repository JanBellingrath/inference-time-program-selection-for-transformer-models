#!/usr/bin/env python
"""Main orchestrator and CLI for the fine-tuning interaction study.

import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

Runs four experimental arms for every (model, dataset, seed) combo:
    1. FT only              -- LoRA fine-tune with default layer order
    2. Search -> FT         -- MCTS search, then fine-tune under best sequence
    3. FT -> Search         -- Fine-tune first, then search on the FT checkpoint
    4. Search -> FT -> Search -- Search, fine-tune, re-search

Usage:
    cd dr-llm
    python -m ft_study.runner --model_name Qwen/Qwen2.5-0.5B-Instruct \\
                              --datasets arc_easy boolq \\
                              --seeds 42 1337 2024
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import BitsAndBytesConfig

from core.benchmark_mcts import BenchmarkMCTS, grade_response, seq_to_layers, SKIP
from core.flexible_models import FlexibleModelWrapper, get_is_instruct
from core.permutation_mcts import MCTSModel, PermutationMCTSConfig, set_seed

from ft_study.config import ArmType, FTConfig, FTStudyConfig, SearchConfig, SplitConfig
from ft_study.data import create_four_way_split, make_sft_dataset, split_val_search_into_tiers
from ft_study.trainer import (
    TrainResult,
    load_ft_model_for_inference,
    load_model_for_ft,
    train_lora,
)

logger = logging.getLogger(__name__)

SKIP_SENTINEL = SKIP  # -1


def _mcts_quantization_config(study_cfg: FTStudyConfig):
    """4-bit base weights for MCTS when training also uses QLoRA (saves VRAM)."""
    if not study_cfg.search.mcts_load_in_4bit or not study_cfg.ft.load_in_4bit:
        return None
    ft = study_cfg.ft
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=ft.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, ft.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=ft.bnb_4bit_use_double_quant,
    )


def _effective_max_new_tokens(sample: Dict[str, Any], is_instruct: bool) -> int:
    """Return the per-sample ``max_new_tokens`` budget for evaluation.

    The previous default forced instruct+MC samples up to ``>=10`` tokens to
    "give the model space" to emit the answer letter, but that diverges from
    the MCTS reward path which always uses 1-token TALE-style grading
    (full-vocab argmax + ``grade_response``).  To keep search and eval on
    the same protocol we now honour the dataset's per-sample
    ``max_new_tokens`` (=1 for all MC tasks in ``prepare_arc_data``).
    """
    return int(sample.get("max_new_tokens", 1) or 1)


def _get_inner_model(model):
    """Navigate to the inner transformer model that holds ``layer_indices``."""
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        m = model.base_model.model
        return m.model if hasattr(m, 'model') else m
    return model.model if hasattr(model, 'model') else model

DATASET_NAME_ALIASES = {
    "mmlu": "mmlu_all",
    "mmlu_all": "mmlu_all",
}


# ---------------------------------------------------------------------------
# Load cached MCTS sequences from prior runs
# ---------------------------------------------------------------------------

def load_cached_sequences(
    search_dirs: List[str],
    dataset_names: Optional[List[str]] = None,
) -> Dict[Any, List[int]]:
    """Scan directories for ``benchmark_mcts_*.json`` and extract best sequences.

    For each dataset, picks the file with the latest timestamp (embedded in
    the filename).  Handles dataset-name aliases (e.g. ``mmlu`` → ``mmlu_all``).

    File-name conventions recognised:
        - ``benchmark_mcts_<dataset>_seed<seed>_<timestamp>.json`` (per-seed)
        - ``benchmark_mcts_<dataset>_<timestamp>.json`` (legacy, dataset-only)

    Per-seed entries populate keys ``(dataset_name, seed)``; legacy entries
    populate keys ``dataset_name`` and act as seed-agnostic fallbacks.

    Args:
        search_dirs: Directories to scan for MCTS result JSON files.
        dataset_names: If provided, only load sequences for these datasets.

    Returns:
        Dict whose keys are either ``str`` (legacy) or ``(str, int)``
        (per-seed) and whose values are best layer sequences.
    """
    reverse_aliases = {}
    for study_name, mcts_name in DATASET_NAME_ALIASES.items():
        reverse_aliases.setdefault(mcts_name, []).append(study_name)

    import re as _re
    seed_pat = _re.compile(r"^(?P<dataset>.+)_seed(?P<seed>\d+)$")

    # candidates[(mcts_dataset, seed_or_None)] = [(timestamp, path), ...]
    candidates: Dict[Tuple[str, Optional[int]], List[Tuple[str, str]]] = {}

    for d in search_dirs:
        pattern = os.path.join(d, "benchmark_mcts_*.json")
        for fpath in glob.glob(pattern):
            base = os.path.basename(fpath)
            if "snapshot" in base or "explored" in base:
                continue
            parts = base.replace("benchmark_mcts_", "").rsplit("_", 1)
            if len(parts) != 2:
                continue
            head = parts[0]
            timestamp = parts[1].replace(".json", "")
            m = seed_pat.match(head)
            if m:
                mcts_dataset = m.group("dataset")
                seed_val: Optional[int] = int(m.group("seed"))
            else:
                mcts_dataset = head
                seed_val = None
            candidates.setdefault((mcts_dataset, seed_val), []).append(
                (timestamp, fpath)
            )

    cached: Dict[Any, List[int]] = {}
    for (mcts_dataset, seed_val), entries in candidates.items():
        entries.sort(key=lambda x: x[0], reverse=True)
        latest_path = entries[0][1]

        study_names = reverse_aliases.get(mcts_dataset, [mcts_dataset])

        for sname in study_names:
            if dataset_names and sname not in dataset_names:
                continue
            try:
                with open(latest_path) as f:
                    data = json.load(f)
                best = data.get("best", {})
                seq = best.get("seq")
                if seq:
                    key: Any = (sname, seed_val) if seed_val is not None else sname
                    cached[key] = seq
                    tag = f"seed={seed_val}" if seed_val is not None else "seed-agnostic"
                    logger.info(
                        "Cached sequence for %s (%s) from %s: %s (delta=%+.4f)",
                        sname, tag, latest_path, seq, best.get("delta", 0),
                    )
            except Exception as e:
                logger.warning("Failed to load cached sequence from %s: %s", latest_path, e)

        if mcts_dataset not in reverse_aliases:
            if not dataset_names or mcts_dataset in dataset_names:
                key = (mcts_dataset, seed_val) if seed_val is not None else mcts_dataset
                if key not in cached:
                    try:
                        with open(latest_path) as f:
                            data = json.load(f)
                        best = data.get("best", {})
                        seq = best.get("seq")
                        if seq:
                            cached[key] = seq
                            logger.info(
                                "Cached sequence for %s from %s: %s",
                                key, latest_path, seq,
                            )
                    except Exception as e:
                        logger.warning("Failed to load %s: %s", latest_path, e)

    return cached


def _lookup_cached_sequence(
    cache: Dict[Any, List[int]],
    dataset_name: str,
    seed: int,
) -> Optional[List[int]]:
    """Look up a cached sequence honouring per-seed keys with seed-agnostic fallback."""
    if not cache:
        return None
    key = (dataset_name, seed)
    if key in cache:
        return cache[key]
    return cache.get(dataset_name)


def _store_cached_sequence(
    cache: Dict[Any, List[int]],
    dataset_name: str,
    seed: int,
    seq: List[int],
) -> None:
    cache[(dataset_name, seed)] = seq


def _persist_runtime_base_mcts_sequence(
    study_cfg: "FTStudyConfig",
    dataset_name: str,
    seed: int,
    seq: List[int],
) -> None:
    """Mirror the in-process runtime cache to disk so other processes / runs
    can pick it up via ``load_cached_sequences``.

    Files are named ``benchmark_mcts_<dataset>_seed<seed>_<timestamp>.json``
    inside ``<output_dir>/base_mcts_cache/``.  Only a thin "best" record is
    written (full snapshots stay under the per-arm directory).
    """
    base_dir = os.path.join(study_cfg.output_dir, "base_mcts_cache")
    try:
        os.makedirs(base_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(
            base_dir,
            f"benchmark_mcts_{dataset_name}_seed{seed}_{ts}.json",
        )
        with open(path, "w") as f:
            json.dump({"best": {"seq": seq, "delta": 0.0},
                       "dataset": dataset_name, "seed": seed}, f, indent=2)
        logger.info("Persisted base MCTS sequence to %s", path)
    except Exception as e:
        logger.warning("Failed to persist base MCTS sequence: %s", e)


# ---------------------------------------------------------------------------
# Lightweight MCTSModel adapter for pre-loaded models
# ---------------------------------------------------------------------------

class _MCTSModelFromLoaded:
    """Drop-in replacement for ``MCTSModel`` that wraps an already-loaded model.

    ``BenchmarkMCTS`` accesses ``model.wrapper.*`` and ``model.num_layers``.
    This adapter satisfies that interface without re-loading from Hub.
    """

    def __init__(self, model, tokenizer, num_layers: int, model_name: str):
        self.num_layers = num_layers
        self.wrapper = _WrapperShim(model, tokenizer, num_layers, model_name)

    @classmethod
    def from_pretrained_ft(
        cls,
        base_model_name: str,
        adapter_path: str,
        layer_sequence: Optional[List[int]] = None,
        rank: int = 0,
    ) -> "_MCTSModelFromLoaded":
        model, tokenizer, num_layers = load_ft_model_for_inference(
            base_model_name, adapter_path, layer_sequence=layer_sequence, rank=rank,
        )
        return cls(model, tokenizer, num_layers, base_model_name)


class _WrapperShim:
    """Mimics the subset of ``FlexibleModelWrapper`` that ``BenchmarkMCTS`` uses."""

    def __init__(self, model, tokenizer, num_layers: int, model_name: str):
        self.model = model
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        self.model_name = model_name
        self.is_instruct = get_is_instruct(model_name)
        self.default_layer_indices = list(range(num_layers))

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


# ---------------------------------------------------------------------------
# Sequence analysis helpers
# ---------------------------------------------------------------------------

def _seq_metrics(seq: List[int], num_layers: int) -> Dict[str, Any]:
    """Compute descriptive metrics for a layer sequence (may contain SKIP)."""
    active = [x for x in seq if x != SKIP_SENTINEL]
    default = list(range(num_layers))
    default_active = default[:len(seq)]  # padded comparison

    counts = Counter(active)
    n_unique = len(counts)
    n_repeats = sum(v - 1 for v in counts.values() if v > 1)
    n_skips = seq.count(SKIP_SENTINEL)
    n_swaps = sum(1 for i, v in enumerate(seq) if v != default_active[i]) if len(seq) == num_layers else len(seq)

    # Edit distance from default: Hamming for same-length, otherwise count all diffs
    if len(seq) == num_layers:
        edit_dist = sum(1 for a, b in zip(seq, default) if a != b)
    else:
        edit_dist = max(len(seq), num_layers)

    return {
        "selected_sequence": seq,
        "num_swaps": n_swaps,
        "num_skips": n_skips,
        "num_repeats": n_repeats,
        "num_unique_active_layers": n_unique,
        "edit_distance_from_default": edit_dist,
    }


def _edit_distance(a: List[int], b: List[int]) -> int:
    """Hamming distance between two sequences (padded to equal length with -2)."""
    maxlen = max(len(a), len(b))
    pa = a + [-2] * (maxlen - len(a))
    pb = b + [-2] * (maxlen - len(b))
    return sum(1 for x, y in zip(pa, pb) if x != y)


# ---------------------------------------------------------------------------
# Search wrapper
# ---------------------------------------------------------------------------

def _get_search_samples(
    splits: Dict[str, Any],
    study_cfg: "FTStudyConfig",
) -> List[Dict[str, Any]]:
    """Return the sample pool for MCTS search.

    When ``share_train_for_search`` is True, uses the (much larger)
    train_ft split.  Otherwise uses the dedicated val_search split.
    """
    if study_cfg.share_train_for_search:
        pool = splits["train_ft"]
        logger.info(
            "share_train_for_search: using train_ft (%d samples) for MCTS",
            len(pool),
        )
    else:
        pool = splits["val_search"]
    return pool


def _search_pool_for_ft_search(
    dataset_name: str,
    splits: Dict[str, Any],
    study_cfg: "FTStudyConfig",
) -> List[Dict[str, Any]]:
    """Return ft_search search pool with BoolQ-specific token budget fix.

    Only for the ``ft_search`` arm: BoolQ labels are True/False and one-token
    decoding can under-measure FT checkpoints in MCTS reward.  Use a 10-token
    budget here to align search-time grading with final evaluation behavior.
    """
    pool = _get_search_samples(splits, study_cfg)
    if dataset_name != "boolq":
        return pool

    adjusted = []
    for s in pool:
        c = dict(s)
        c["max_new_tokens"] = max(int(c.get("max_new_tokens", 1) or 1), 10)
        adjusted.append(c)
    logger.info(
        "ft_search boolq: raised max_new_tokens to >=10 for %d search samples",
        len(adjusted),
    )
    return adjusted


def run_search_for_study(
    mcts_model,
    val_search_samples: List[Dict[str, Any]],
    search_cfg: SearchConfig,
    dataset_name: str,
    model_name: str,
    output_prefix: str,
    notify_signal: bool = False,
    anchor_seq: Optional[List[int]] = None,
    val_select_samples: Optional[List[Dict[str, Any]]] = None,
    eval_model_for_rerank: Any = None,
    eval_tokenizer_for_rerank: Any = None,
    extra_rerank_candidates: Optional[List[List[int]]] = None,
) -> Tuple[Optional[List[int]], Dict[str, Any]]:
    """Run MCTS search and return (best_sequence, summary).

    Args:
        mcts_model: An object exposing ``.wrapper.model`` / ``.wrapper.tokenizer``
            and ``.num_layers`` (e.g. ``MCTSModel`` or ``_MCTSModelFromLoaded``).
        val_search_samples: Sample pool for tier-2/3/4 (carved internally).
        search_cfg: ``SearchConfig`` with all MCTS / rerank hyperparameters.
        dataset_name: Used by ``grade_response`` to pick the grading rule.
        model_name: HF model id for instruct/chat-template detection.
        output_prefix: Where to persist snapshots / explored logs.
        notify_signal: Forward to ``BenchmarkMCTS`` for Signal notifications.
        anchor_seq: Optional anchor for the MCTS root.  When provided, the
            tree is anchored at this sequence and tier-2/3/4 baselines are
            measured for it (instead of the identity order).  Used by
            ``search_ft_search`` phase 3 to pass the pre-FT best sequence.
        val_select_samples: When non-empty AND a working evaluation model is
            supplied, the top-``rerank_topk`` MCTS candidates are re-evaluated
            on this held-out pool (TALE-style 1-token grading) and the one
            with the highest val_select accuracy is returned as ``best_seq``.
            This is the post-MCTS winner's-curse mitigation step.
        eval_model_for_rerank: HF model used for the val_select re-rank
            evaluation.  For base-model search, pass ``mcts_model.wrapper.model``.
            For post-FT search, pass the loaded FT model (e.g.
            ``ft_mcts.wrapper.model``).
        eval_tokenizer_for_rerank: Matching tokenizer.
        extra_rerank_candidates: Additional sequences to inject into the
            rerank candidate pool (e.g. the pre-FT sequence in phase 3).

    Returns ``(best_seq, summary)``.  If the post-MCTS rerank picks a
    different sequence than the tier-best, ``summary["rerank"]`` will hold
    the rerank trace and ``best_seq`` reflects the rerank winner.  If no
    sequence beats baseline (and no rerank winner emerges), the anchor (or
    identity) sequence is returned.
    """
    t2, t3, t4 = split_val_search_into_tiers(
        val_search_samples,
        tier2=search_cfg.tier2_samples,
        tier3=search_cfg.tier3_samples,
        tier4=search_cfg.tier4_samples,
    )

    cfg = PermutationMCTSConfig(
        num_simulations=search_cfg.num_simulations,
        exploration_constant=search_cfg.exploration_constant,
        random_prob=search_cfg.random_prob,
        pw_C=search_cfg.pw_C,
        pw_alpha=search_cfg.pw_alpha,
        legacy_widen_prob=search_cfg.legacy_widen_prob,
        legacy_random_schedule=search_cfg.legacy_random_schedule,
        neighborhood_radius=search_cfg.neighborhood_radius,
        max_swaps=search_cfg.max_swaps,
        model_name=model_name,
        dataset=dataset_name,
        num_samples=len(t2),
    )

    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)

    bench = BenchmarkMCTS(
        mcts_model, cfg, t2,
        eval_batch_size=search_cfg.eval_batch_size,
        extended_samples=t3,
        extended_samples_tier4=t4,
        promote_delta=search_cfg.promote_delta,
        notify_signal=notify_signal,
        compute_loglik_full=search_cfg.compute_loglik_full,
        promote_use_wilson=search_cfg.promote_use_wilson,
        rerank_topk=search_cfg.rerank_topk,
    )
    resume_prefix: Optional[str] = None
    snap_path = f"{output_prefix}_snapshot.json"
    if output_prefix and os.path.isfile(snap_path):
        resume_prefix = output_prefix
        logger.info("MCTS resume: loading state from %s", snap_path)

    summary = bench.search(
        search_cfg.num_simulations,
        report_every=search_cfg.report_every,
        validate_top_k=search_cfg.validate_top_k,
        out_prefix=output_prefix,
        resume_prefix=resume_prefix,
        default_seq=anchor_seq,
    )

    best = summary.get("best")
    fallback_seq = list(anchor_seq) if anchor_seq else list(range(mcts_model.num_layers))
    if best and best.get("seq"):
        best_seq = best["seq"]
    else:
        best_seq = fallback_seq
        logger.info("No sequence beat baseline; using anchor/default order")

    rerank_trace: Optional[Dict[str, Any]] = None
    if (
        val_select_samples
        and eval_model_for_rerank is not None
        and eval_tokenizer_for_rerank is not None
    ):
        candidates: List[Dict[str, Any]] = []
        seen_keys: set = set()

        def _add(seq: List[int], source: str, extra: Optional[Dict[str, Any]] = None):
            key = tuple(seq)
            if key in seen_keys:
                return
            seen_keys.add(key)
            entry = {"seq": list(seq), "source": source}
            if extra:
                entry.update(extra)
            candidates.append(entry)

        for entry in summary.get("topk", []) or []:
            _add(entry["seq"], "mcts_topk", {
                "tier": entry.get("tier"),
                "tier_acc": entry.get("tier_acc"),
                "tier_n": entry.get("tier_n"),
                "tier_ci_lo": entry.get("tier_ci_lo"),
                "tier_ci_hi": entry.get("tier_ci_hi"),
            })
        _add(fallback_seq, "anchor_or_identity")
        if extra_rerank_candidates:
            for s in extra_rerank_candidates:
                _add(list(s), "extra")

        num_layers = mcts_model.num_layers
        for c in candidates:
            metrics = evaluate_on_test(
                eval_model_for_rerank,
                eval_tokenizer_for_rerank,
                c["seq"], val_select_samples,
                dataset_name, model_name, num_layers,
            )
            c["val_select_acc"] = metrics["test_metric"]
            c["val_select_n"] = metrics["test_total"]

        candidates.sort(key=lambda r: r["val_select_acc"], reverse=True)
        rerank_winner = candidates[0]
        best_seq = rerank_winner["seq"]
        rerank_trace = {"candidates": candidates, "winner_seq": best_seq}
        summary["rerank"] = rerank_trace
        if output_prefix:
            try:
                with open(f"{output_prefix}_rerank.json", "w") as f:
                    json.dump(rerank_trace, f, indent=2, default=str)
            except Exception as e:
                logger.warning("Failed to persist rerank trace: %s", e)
        logger.info(
            "Post-MCTS rerank on val_select (n=%d) picked %s "
            "(acc=%.4f) from %d candidates",
            len(val_select_samples), rerank_winner.get("source"),
            rerank_winner["val_select_acc"], len(candidates),
        )

    return best_seq, summary


# ---------------------------------------------------------------------------
# Evaluation on test split
# ---------------------------------------------------------------------------

def evaluate_on_test(
    model,
    tokenizer,
    layer_sequence: List[int],
    test_samples: List[Dict[str, Any]],
    dataset_name: str,
    model_name: str,
    num_layers: int,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """Evaluate accuracy on the held-out test split.

    Default ``batch_size=1`` avoids left-padding artifacts that corrupt
    QLoRA-merged model outputs (token-0 garbage on padded positions).
    """
    layers = seq_to_layers(layer_sequence)
    if not layers:
        layers = list(range(num_layers))

    inner = _get_inner_model(model)
    saved = getattr(inner, "layer_indices", None)
    inner.layer_indices = layers

    is_instruct = get_is_instruct(model_name)
    has_dup = len(layers) != len(set(layers))
    correct_count = 0
    total = len(test_samples)
    total_eval_time = 0.0
    per_example: List[Dict[str, Any]] = []
    correct_by_idx: Dict[int, bool] = {}

    def _prepare_prompt(sample):
        if is_instruct:
            messages = []
            sys_prompt = sample.get("system_prompt")
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": sample["input"]})
            kwargs = {}
            if "qwen3" in model_name.lower():
                kwargs["enable_thinking"] = False
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kwargs,
            )
        return sample["input"]

    try:
        groups: Dict[int, List[Tuple[int, Dict]]] = {}
        for i, sample in enumerate(test_samples):
            mt = _effective_max_new_tokens(sample, is_instruct)
            groups.setdefault(mt, []).append((i, sample))

        for max_tok, indexed_samples in groups.items():
            for b_start in range(0, len(indexed_samples), batch_size):
                batch = indexed_samples[b_start:b_start + batch_size]
                prompts = [_prepare_prompt(s) for _, s in batch]

                saved_pad = tokenizer.padding_side
                tokenizer.padding_side = "left"
                dev = getattr(model, "device", None)
                if dev is None:
                    dev = next(model.parameters()).device
                inputs = tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=True,
                ).to(dev)
                tokenizer.padding_side = saved_pad

                t0 = time.time()
                with torch.no_grad():
                    if max_tok == 1:
                        fwd_kw: Dict[str, Any] = {}
                        if has_dup:
                            fwd_kw["use_cache"] = False
                        out = model(**inputs, **fwd_kw)
                        token_ids = out.logits[:, -1, :].argmax(dim=-1)
                        for j, (orig_idx, sample) in enumerate(batch):
                            resp = tokenizer.decode(
                                [token_ids[j].item()], skip_special_tokens=True,
                            ).strip()
                            ok = grade_response(
                                resp, sample["correct"],
                                dataset_name, model_name, sample["input"],
                            ) > 0.5
                            if ok:
                                correct_count += 1
                            correct_by_idx[orig_idx] = ok
                    else:
                        gen_kw = {
                            "max_new_tokens": max_tok,
                            "pad_token_id": tokenizer.eos_token_id,
                            "do_sample": False,
                        }
                        if has_dup:
                            gen_kw["use_cache"] = False
                        outputs = model.generate(**inputs, **gen_kw)
                        input_len = inputs.input_ids.shape[1]
                        for j, (orig_idx, sample) in enumerate(batch):
                            resp = tokenizer.decode(
                                outputs[j][input_len:], skip_special_tokens=True,
                            ).strip()
                            ok = grade_response(
                                resp, sample["correct"],
                                dataset_name, model_name, sample["input"],
                            ) > 0.5
                            if ok:
                                correct_count += 1
                            correct_by_idx[orig_idx] = ok
                total_eval_time += time.time() - t0
    finally:
        if saved is not None:
            inner.layer_indices = saved

    accuracy = correct_count / total if total else 0.0
    mean_lat = total_eval_time / total if total else 0.0
    throughput = total / total_eval_time if total_eval_time else 0.0

    for i, sample in enumerate(test_samples):
        per_example.append({
            "sample_hash": sample.get("_hash") or "",
            "correct": bool(correct_by_idx.get(i, False)),
        })

    return {
        "test_metric": accuracy,
        "test_correct": correct_count,
        "test_total": total,
        "inference_latency_per_example_s": mean_lat,
        "throughput_examples_per_s": throughput,
        "per_example": per_example,
    }


# ---------------------------------------------------------------------------
# Individual arm runners
# ---------------------------------------------------------------------------

def _run_ft_only(
    model_name: str,
    dataset_name: str,
    splits: Dict[str, Any],
    study_cfg: FTStudyConfig,
    seed: int,
    arm_dir: str,
) -> Dict[str, Any]:
    """Arm 1: LoRA fine-tune with default layer order."""
    set_seed(seed)

    ckpt_dir = os.path.join(arm_dir, "checkpoints")
    adapter_dir = os.path.join(ckpt_dir, "final_adapter")
    adapter_ready = os.path.isfile(os.path.join(adapter_dir, "adapter_config.json"))

    if adapter_ready:
        logger.info(
            "ft_only: reusing existing adapter at %s (skip training)", adapter_dir,
        )
        train_result = TrainResult(
            best_checkpoint_dir=adapter_dir,
            adapter_dir=adapter_dir,
            train_wall_clock_s=0.0,
            peak_gpu_memory_gb=0.0,
            trainable_params=0,
            total_tokens=0,
        )
    else:
        peft_model, tokenizer, num_layers, _ = load_model_for_ft(
            model_name, study_cfg.ft, layer_sequence=None, rank=study_cfg.gpu_rank,
        )

        train_ds = splits.get("_train_ds") or make_sft_dataset(splits["train_ft"], tokenizer)
        val_ds = splits.get("_val_ds") or make_sft_dataset(splits["val_select"], tokenizer)
        train_result = train_lora(
            peft_model, tokenizer, train_ds, val_ds, study_cfg.ft,
            output_dir=ckpt_dir, run_name=f"ft_only_{dataset_name}_s{seed}",
            layer_sequence=None, num_layers=num_layers,
        )

        del peft_model
        torch.cuda.empty_cache()

    # -- Evaluate --
    model, tokenizer, num_layers = load_ft_model_for_inference(
        model_name, train_result.adapter_dir, layer_sequence=None, rank=study_cfg.gpu_rank,
    )
    default_seq = list(range(num_layers))
    test_metrics = evaluate_on_test(
        model, tokenizer, default_seq, splits["test"], dataset_name, model_name, num_layers,
    )
    test_per_example = test_metrics.pop("per_example", [])

    val_metrics = evaluate_on_test(
        model, tokenizer, default_seq, splits["val_select"], dataset_name, model_name, num_layers,
    )
    val_select_per_example = val_metrics.pop("per_example", [])

    del model
    torch.cuda.empty_cache()

    result = {
        "arm": ArmType.FT_ONLY.value,
        **test_metrics,
        "val_select_metric": val_metrics["test_metric"],
        "train_wall_clock_s": train_result.train_wall_clock_s,
        "gpu_hours": train_result.train_wall_clock_s / 3600,
        "peak_gpu_memory_gb": train_result.peak_gpu_memory_gb,
        "trainable_params": train_result.trainable_params,
        "total_tokens": train_result.total_tokens,
        **_seq_metrics(default_seq, num_layers),
        "adapter_path": train_result.adapter_dir,
        "test_per_example": test_per_example,
        "val_select_per_example": val_select_per_example,
    }
    return result


def _run_search_ft(
    model_name: str,
    dataset_name: str,
    splits: Dict[str, Any],
    study_cfg: FTStudyConfig,
    seed: int,
    arm_dir: str,
) -> Dict[str, Any]:
    """Arm 2: Search on frozen base -> fine-tune under best sequence."""
    set_seed(seed)

    cached_seq = _lookup_cached_sequence(
        study_cfg.cached_sequences or {}, dataset_name, seed,
    )
    if cached_seq is None:
        cached_seq = _lookup_cached_sequence(
            getattr(study_cfg, '_runtime_search_cache', {}) or {},
            dataset_name, seed,
        )
    if cached_seq is not None:
        logger.info(
            "Using cached base-model sequence for %s seed=%d: %s",
            dataset_name, seed, cached_seq,
        )
        best_seq = cached_seq
        search_wall = 0.0
        num_layers = len(cached_seq)
    else:
        gc.collect()
        torch.cuda.empty_cache()
        mcts_model = MCTSModel(
            model_name,
            rank=study_cfg.gpu_rank,
            bnb_config=_mcts_quantization_config(study_cfg),
        )
        num_layers = mcts_model.num_layers
        search_prefix = os.path.join(
            arm_dir, f"search_base_{dataset_name}_seed{seed}",
        )

        search_pool = _get_search_samples(splits, study_cfg)
        best_seq, search_summary = run_search_for_study(
            mcts_model, search_pool, study_cfg.search,
            dataset_name, model_name, search_prefix,
            notify_signal=study_cfg.notify_signal,
            val_select_samples=splits.get("val_select"),
            eval_model_for_rerank=mcts_model.wrapper.model,
            eval_tokenizer_for_rerank=mcts_model.wrapper.tokenizer,
        )
        search_wall = search_summary.get("elapsed_seconds", 0)

        _persist_runtime_base_mcts_sequence(
            study_cfg, dataset_name, seed, best_seq,
        )

        del mcts_model
        torch.cuda.empty_cache()

    if not hasattr(study_cfg, '_runtime_search_cache'):
        study_cfg._runtime_search_cache = {}
    _store_cached_sequence(
        study_cfg._runtime_search_cache, dataset_name, seed, best_seq,
    )

    # -- Fine-tune under best sequence --
    active_layers = seq_to_layers(best_seq)
    ckpt_dir = os.path.join(arm_dir, "checkpoints")
    adapter_dir = os.path.join(ckpt_dir, "final_adapter")
    adapter_ready = os.path.isfile(os.path.join(adapter_dir, "adapter_config.json"))
    if adapter_ready:
        logger.info(
            "search_ft: reusing existing adapter at %s (resume/continue mode)",
            adapter_dir,
        )
        train_result = TrainResult(
            best_checkpoint_dir=adapter_dir,
            adapter_dir=adapter_dir,
            train_wall_clock_s=0.0,
            peak_gpu_memory_gb=0.0,
            trainable_params=0,
            total_tokens=0,
        )
    else:
        peft_model, tokenizer, _, effective_seq = load_model_for_ft(
            model_name, study_cfg.ft, layer_sequence=active_layers, rank=study_cfg.gpu_rank,
        )
        train_ds = splits.get("_train_ds") or make_sft_dataset(splits["train_ft"], tokenizer)
        val_ds = splits.get("_val_ds") or make_sft_dataset(splits["val_select"], tokenizer)
        train_result = train_lora(
            peft_model, tokenizer, train_ds, val_ds, study_cfg.ft,
            output_dir=ckpt_dir,
            run_name=f"search_ft_{dataset_name}_s{seed}",
            layer_sequence=effective_seq, num_layers=num_layers,
        )

        del peft_model
        torch.cuda.empty_cache()

    # -- Evaluate --
    model, tokenizer, _ = load_ft_model_for_inference(
        model_name, train_result.adapter_dir, layer_sequence=active_layers,
        rank=study_cfg.gpu_rank,
        clone_repeated=study_cfg.ft.clone_repeated_lora,
    )
    test_metrics = evaluate_on_test(
        model, tokenizer, best_seq, splits["test"], dataset_name, model_name, num_layers,
    )
    test_per_example = test_metrics.pop("per_example", [])
    val_metrics = evaluate_on_test(
        model, tokenizer, best_seq, splits["val_select"], dataset_name, model_name, num_layers,
    )
    val_select_per_example = val_metrics.pop("per_example", [])

    del model
    torch.cuda.empty_cache()

    result = {
        "arm": ArmType.SEARCH_FT.value,
        **test_metrics,
        "val_select_metric": val_metrics["test_metric"],
        "train_wall_clock_s": train_result.train_wall_clock_s + search_wall,
        "search_wall_clock_s": search_wall,
        "ft_wall_clock_s": train_result.train_wall_clock_s,
        "gpu_hours": (train_result.train_wall_clock_s + search_wall) / 3600,
        "peak_gpu_memory_gb": train_result.peak_gpu_memory_gb,
        "trainable_params": train_result.trainable_params,
        "total_tokens": train_result.total_tokens,
        **_seq_metrics(best_seq, num_layers),
        "adapter_path": train_result.adapter_dir,
        "test_per_example": test_per_example,
        "val_select_per_example": val_select_per_example,
    }
    return result


def _run_ft_search(
    model_name: str,
    dataset_name: str,
    splits: Dict[str, Any],
    study_cfg: FTStudyConfig,
    seed: int,
    arm_dir: str,
) -> Dict[str, Any]:
    """Arm 3: Fine-tune with default order -> search on FT checkpoint."""
    set_seed(seed)

    ckpt_dir = os.path.join(arm_dir, "checkpoints")
    adapter_dir = os.path.join(ckpt_dir, "final_adapter")
    adapter_ready = os.path.isfile(os.path.join(adapter_dir, "adapter_config.json"))

    if not adapter_ready:
        ft_only_adapter = os.path.join(
            os.path.dirname(arm_dir), "ft_only", "checkpoints", "final_adapter",
        )
        if os.path.isfile(os.path.join(ft_only_adapter, "adapter_config.json")):
            adapter_dir = ft_only_adapter
            adapter_ready = True
            logger.info(
                "ft_search: reusing ft_only adapter at %s (identical training)",
                adapter_dir,
            )

    if adapter_ready:
        logger.info(
            "ft_search: reusing existing adapter at %s (skip training)",
            adapter_dir,
        )
        train_result = TrainResult(
            best_checkpoint_dir=adapter_dir,
            adapter_dir=adapter_dir,
            train_wall_clock_s=0.0,
            peak_gpu_memory_gb=0.0,
            trainable_params=0,
            total_tokens=0,
        )
    else:
        peft_model, tokenizer, num_layers, _ = load_model_for_ft(
            model_name, study_cfg.ft, layer_sequence=None, rank=study_cfg.gpu_rank,
        )
        train_ds = splits.get("_train_ds") or make_sft_dataset(splits["train_ft"], tokenizer)
        val_ds = splits.get("_val_ds") or make_sft_dataset(splits["val_select"], tokenizer)
        train_result = train_lora(
            peft_model, tokenizer, train_ds, val_ds, study_cfg.ft,
            output_dir=ckpt_dir,
            run_name=f"ft_search_{dataset_name}_s{seed}",
            layer_sequence=None, num_layers=num_layers,
        )

        del peft_model
        torch.cuda.empty_cache()

    # -- Search on fine-tuned model --
    ft_mcts = _MCTSModelFromLoaded.from_pretrained_ft(
        model_name, train_result.adapter_dir, layer_sequence=None,
        rank=study_cfg.gpu_rank,
    )
    num_layers = ft_mcts.num_layers
    search_prefix = os.path.join(arm_dir, f"search_ft_{dataset_name}_seed{seed}")
    search_pool = _search_pool_for_ft_search(dataset_name, splits, study_cfg)
    best_seq, search_summary = run_search_for_study(
        ft_mcts, search_pool, study_cfg.search,
        dataset_name, model_name, search_prefix,
        notify_signal=study_cfg.notify_signal,
        val_select_samples=splits.get("val_select"),
        eval_model_for_rerank=ft_mcts.wrapper.model,
        eval_tokenizer_for_rerank=ft_mcts.wrapper.tokenizer,
    )
    search_wall = search_summary.get("elapsed_seconds", 0)

    active_layers = seq_to_layers(best_seq)
    ft_mcts.wrapper.model.model.layer_indices = active_layers

    test_metrics = evaluate_on_test(
        ft_mcts.wrapper.model, ft_mcts.wrapper.tokenizer,
        best_seq, splits["test"], dataset_name, model_name, num_layers,
    )
    test_per_example = test_metrics.pop("per_example", [])
    val_metrics = evaluate_on_test(
        ft_mcts.wrapper.model, ft_mcts.wrapper.tokenizer,
        best_seq, splits["val_select"], dataset_name, model_name, num_layers,
    )
    val_select_per_example = val_metrics.pop("per_example", [])

    del ft_mcts
    torch.cuda.empty_cache()

    result = {
        "arm": ArmType.FT_SEARCH.value,
        **test_metrics,
        "val_select_metric": val_metrics["test_metric"],
        "train_wall_clock_s": train_result.train_wall_clock_s + search_wall,
        "search_wall_clock_s": search_wall,
        "ft_wall_clock_s": train_result.train_wall_clock_s,
        "gpu_hours": (train_result.train_wall_clock_s + search_wall) / 3600,
        "peak_gpu_memory_gb": train_result.peak_gpu_memory_gb,
        "trainable_params": train_result.trainable_params,
        "total_tokens": train_result.total_tokens,
        **_seq_metrics(best_seq, num_layers),
        "post_ft_best_sequence": best_seq,
        "adapter_path": train_result.adapter_dir,
        "test_per_example": test_per_example,
        "val_select_per_example": val_select_per_example,
    }
    return result


def _run_search_ft_search(
    model_name: str,
    dataset_name: str,
    splits: Dict[str, Any],
    study_cfg: FTStudyConfig,
    seed: int,
    arm_dir: str,
) -> Dict[str, Any]:
    """Arm 4: Search -> FT under best seq -> re-search on FT checkpoint."""
    set_seed(seed)

    # -- Phase 1: search on base model (use cache if available) --
    cached_seq = _lookup_cached_sequence(
        study_cfg.cached_sequences or {}, dataset_name, seed,
    )
    if cached_seq is None:
        cached_seq = _lookup_cached_sequence(
            getattr(study_cfg, '_runtime_search_cache', {}) or {},
            dataset_name, seed,
        )
    if cached_seq is not None:
        logger.info(
            "Using cached base-model sequence for %s seed=%d: %s",
            dataset_name, seed, cached_seq,
        )
        pre_ft_seq = cached_seq
        search1_wall = 0.0
        num_layers = len(cached_seq)
    else:
        gc.collect()
        torch.cuda.empty_cache()
        mcts_model = MCTSModel(
            model_name,
            rank=study_cfg.gpu_rank,
            bnb_config=_mcts_quantization_config(study_cfg),
        )
        num_layers = mcts_model.num_layers
        search1_prefix = os.path.join(
            arm_dir, f"search1_base_{dataset_name}_seed{seed}",
        )

        search_pool = _get_search_samples(splits, study_cfg)
        pre_ft_seq, search1_summary = run_search_for_study(
            mcts_model, search_pool, study_cfg.search,
            dataset_name, model_name, search1_prefix,
            notify_signal=study_cfg.notify_signal,
            val_select_samples=splits.get("val_select"),
            eval_model_for_rerank=mcts_model.wrapper.model,
            eval_tokenizer_for_rerank=mcts_model.wrapper.tokenizer,
        )
        search1_wall = search1_summary.get("elapsed_seconds", 0)

        _persist_runtime_base_mcts_sequence(
            study_cfg, dataset_name, seed, pre_ft_seq,
        )

        if not hasattr(study_cfg, '_runtime_search_cache'):
            study_cfg._runtime_search_cache = {}
        _store_cached_sequence(
            study_cfg._runtime_search_cache, dataset_name, seed, pre_ft_seq,
        )

        del mcts_model
        torch.cuda.empty_cache()

    # -- Phase 2: fine-tune under pre-FT best sequence --
    active_layers = seq_to_layers(pre_ft_seq)
    ckpt_dir = os.path.join(arm_dir, "checkpoints")
    adapter_dir = os.path.join(ckpt_dir, "final_adapter")
    adapter_ready = os.path.isfile(os.path.join(adapter_dir, "adapter_config.json"))
    if adapter_ready:
        logger.info(
            "search_ft_search: reusing existing adapter at %s (resume/continue mode)",
            adapter_dir,
        )
        train_result = TrainResult(
            best_checkpoint_dir=adapter_dir,
            adapter_dir=adapter_dir,
            train_wall_clock_s=0.0,
            peak_gpu_memory_gb=0.0,
            trainable_params=0,
            total_tokens=0,
        )
    else:
        peft_model, tokenizer, _, effective_seq = load_model_for_ft(
            model_name, study_cfg.ft, layer_sequence=active_layers, rank=study_cfg.gpu_rank,
        )
        train_ds = splits.get("_train_ds") or make_sft_dataset(splits["train_ft"], tokenizer)
        val_ds = splits.get("_val_ds") or make_sft_dataset(splits["val_select"], tokenizer)
        train_result = train_lora(
            peft_model, tokenizer, train_ds, val_ds, study_cfg.ft,
            output_dir=ckpt_dir,
            run_name=f"sfs_{dataset_name}_s{seed}",
            layer_sequence=effective_seq, num_layers=num_layers,
        )

        del peft_model
        torch.cuda.empty_cache()

    # -- Phase 3: re-search on fine-tuned model, ANCHORED at pre_ft_seq --
    # The FT adapter was trained with the model running ``pre_ft_seq`` order,
    # so the only fair baseline for the post-FT MCTS is the FT model under
    # the SAME pre_ft_seq order, not the identity sequence.
    pre_ft_active = seq_to_layers(pre_ft_seq)
    ft_mcts = _MCTSModelFromLoaded.from_pretrained_ft(
        model_name, train_result.adapter_dir, layer_sequence=pre_ft_active,
        rank=study_cfg.gpu_rank,
    )
    ft_mcts.wrapper.default_layer_indices = list(pre_ft_seq)
    search2_prefix = os.path.join(arm_dir, f"search2_ft_{dataset_name}_seed{seed}")
    search_pool = _get_search_samples(splits, study_cfg)
    post_ft_seq, search2_summary = run_search_for_study(
        ft_mcts, search_pool, study_cfg.search,
        dataset_name, model_name, search2_prefix,
        notify_signal=study_cfg.notify_signal,
        anchor_seq=pre_ft_seq,
        val_select_samples=splits.get("val_select"),
        eval_model_for_rerank=ft_mcts.wrapper.model,
        eval_tokenizer_for_rerank=ft_mcts.wrapper.tokenizer,
        extra_rerank_candidates=[list(pre_ft_seq)],
    )
    search2_wall = search2_summary.get("elapsed_seconds", 0)

    post_active = seq_to_layers(post_ft_seq)
    ft_mcts.wrapper.model.model.layer_indices = post_active

    test_metrics = evaluate_on_test(
        ft_mcts.wrapper.model, ft_mcts.wrapper.tokenizer,
        post_ft_seq, splits["test"], dataset_name, model_name, num_layers,
    )
    test_per_example = test_metrics.pop("per_example", [])
    val_metrics = evaluate_on_test(
        ft_mcts.wrapper.model, ft_mcts.wrapper.tokenizer,
        post_ft_seq, splits["val_select"], dataset_name, model_name, num_layers,
    )
    val_select_per_example = val_metrics.pop("per_example", [])

    del ft_mcts
    torch.cuda.empty_cache()

    total_wall = search1_wall + train_result.train_wall_clock_s + search2_wall

    result = {
        "arm": ArmType.SEARCH_FT_SEARCH.value,
        **test_metrics,
        "val_select_metric": val_metrics["test_metric"],
        "train_wall_clock_s": total_wall,
        "search1_wall_clock_s": search1_wall,
        "ft_wall_clock_s": train_result.train_wall_clock_s,
        "search2_wall_clock_s": search2_wall,
        "gpu_hours": total_wall / 3600,
        "peak_gpu_memory_gb": train_result.peak_gpu_memory_gb,
        "trainable_params": train_result.trainable_params,
        "total_tokens": train_result.total_tokens,
        **_seq_metrics(post_ft_seq, num_layers),
        "pre_ft_best_sequence": pre_ft_seq,
        "post_ft_best_sequence": post_ft_seq,
        "edit_distance_between": _edit_distance(pre_ft_seq, post_ft_seq),
        "sequences_identical": pre_ft_seq == post_ft_seq,
        "adapter_path": train_result.adapter_dir,
        "test_per_example": test_per_example,
        "val_select_per_example": val_select_per_example,
    }
    return result


# ---------------------------------------------------------------------------
# Arm dispatcher
# ---------------------------------------------------------------------------

_ARM_RUNNERS = {
    ArmType.FT_ONLY: _run_ft_only,
    ArmType.SEARCH_FT: _run_search_ft,
    ArmType.FT_SEARCH: _run_ft_search,
    ArmType.SEARCH_FT_SEARCH: _run_search_ft_search,
}


def run_arm(
    arm_type: ArmType,
    model_name: str,
    dataset_name: str,
    splits: Dict[str, Any],
    study_cfg: FTStudyConfig,
    seed: int,
    output_dir: str,
) -> Dict[str, Any]:
    """Run a single experimental arm and return its result dict."""
    arm_dir = os.path.join(output_dir, dataset_name, f"seed_{seed}", arm_type.value)
    os.makedirs(arm_dir, exist_ok=True)

    logger.info(
        "=== Running %s | dataset=%s | seed=%d ===",
        arm_type.value, dataset_name, seed,
    )
    t0 = time.time()
    result = _ARM_RUNNERS[arm_type](
        model_name, dataset_name, splits, study_cfg, seed, arm_dir,
    )
    result["wall_clock_total_s"] = time.time() - t0
    result["seed"] = seed
    result["dataset"] = dataset_name
    result["model"] = model_name

    result_path = os.path.join(arm_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Saved arm result to %s", result_path)

    return result


# ---------------------------------------------------------------------------
# LoRA-layer-reorder sanity check
# ---------------------------------------------------------------------------

def verify_lora_layer_reorder(model_name: str, rank: int = 0) -> bool:
    """Quick sanity check: LoRA adapters travel with reordered layers.

    Loads a tiny quantized model with LoRA, runs a forward pass with the
    default sequence and a swapped sequence, and asserts the outputs differ
    (proving the layer order actually changed).  Returns True on success.
    """
    from ft_study.config import FTConfig

    ft_cfg = FTConfig()
    peft_model, tokenizer, num_layers, _ = load_model_for_ft(
        model_name, ft_cfg, layer_sequence=None, rank=rank,
    )

    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt").to(peft_model.device)

    peft_model.eval()
    with torch.no_grad():
        out_default = peft_model(**inputs).logits

    # Swap first two layers
    swapped = list(range(num_layers))
    swapped[0], swapped[1] = swapped[1], swapped[0]

    base = peft_model.base_model.model if hasattr(peft_model, "base_model") else peft_model
    inner = base.model if hasattr(base, "model") else base
    inner.layer_indices = swapped

    with torch.no_grad():
        out_swapped = peft_model(**inputs).logits

    inner.layer_indices = list(range(num_layers))

    diff = (out_default - out_swapped).abs().max().item()
    ok = diff > 1e-4
    logger.info(
        "LoRA-layer-reorder check: max logit diff=%.6f  %s",
        diff, "PASS" if ok else "FAIL (outputs identical despite swap)",
    )

    del peft_model
    torch.cuda.empty_cache()
    return ok


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_results(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build summary tables from all arm results.

    Returns a dict with:
        arm_summary   -- mean/std/median/best per arm x dataset
        route_table   -- pre-FT vs post-FT sequences for search arms
        compute_table -- wall-clock and latency comparisons
    """
    from collections import defaultdict
    by_arm_ds = defaultdict(list)
    for r in all_results:
        key = (r["arm"], r["dataset"])
        by_arm_ds[key].append(r)

    arm_summary = {}
    for (arm, ds), runs in by_arm_ds.items():
        ok = [r for r in runs if "error" not in r and "test_metric" in r]
        if not ok:
            arm_summary[f"{arm}_{ds}"] = {
                "arm": arm,
                "dataset": ds,
                "n_seeds": 0,
                "mean": float("nan"),
                "std": float("nan"),
                "median": float("nan"),
                "best": float("nan"),
                "wall_clock_mean_s": 0.0,
                "latency_mean_s": 0.0,
                "n_failed": len(runs),
            }
            continue
        accs = [r["test_metric"] for r in ok]
        arm_summary[f"{arm}_{ds}"] = {
            "arm": arm,
            "dataset": ds,
            "n_seeds": len(ok),
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
            "median": float(np.median(accs)),
            "best": float(np.max(accs)),
            "wall_clock_mean_s": float(np.mean([r.get("train_wall_clock_s", 0) for r in ok])),
            "latency_mean_s": float(np.mean([r.get("inference_latency_per_example_s", 0) for r in ok])),
            "n_failed": len(runs) - len(ok),
        }

    route_table = []
    for r in all_results:
        if "pre_ft_best_sequence" in r and "post_ft_best_sequence" in r:
            route_table.append({
                "arm": r["arm"],
                "dataset": r["dataset"],
                "seed": r["seed"],
                "pre_ft_seq": r["pre_ft_best_sequence"],
                "post_ft_seq": r["post_ft_best_sequence"],
                "edit_distance": r.get("edit_distance_between", -1),
                "identical": r.get("sequences_identical", False),
            })
        elif "post_ft_best_sequence" in r:
            route_table.append({
                "arm": r["arm"],
                "dataset": r["dataset"],
                "seed": r["seed"],
                "pre_ft_seq": None,
                "post_ft_seq": r["post_ft_best_sequence"],
                "edit_distance": -1,
                "identical": False,
            })

    paired_cis = _compute_paired_cis(all_results)

    return {
        "arm_summary": arm_summary,
        "route_table": route_table,
        "paired_cis": paired_cis,
    }


def _compute_paired_cis(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute paired-bootstrap CIs comparing ft_only to each search arm.

    Joins per-question outcomes by ``sample_hash`` for each (dataset, seed),
    then also pools across seeds within each dataset. Returns a dict with
    ``per_seed`` and ``pooled`` lists of comparisons.
    """
    try:
        from dataclasses import asdict as _dc_asdict

        from .paired_bootstrap import (
            paired_bootstrap_diff,
            pool_per_question_outcomes,
        )
    except Exception as e:  # pragma: no cover
        logger.warning("paired_bootstrap module unavailable: %s", e)
        return {}

    by_key: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = {}
    for r in all_results:
        if "error" in r:
            continue
        per = r.get("test_per_example") or []
        if not per:
            continue
        key = (r.get("dataset", ""), int(r.get("seed", -1)), r.get("arm", ""))
        by_key[key] = per

    arms = sorted({k[2] for k in by_key})
    datasets = sorted({k[0] for k in by_key})
    seeds_by_ds: Dict[str, List[int]] = {}
    for (ds, sd, _arm) in by_key:
        seeds_by_ds.setdefault(ds, [])
        if sd not in seeds_by_ds[ds]:
            seeds_by_ds[ds].append(sd)

    ft_only_arm = ArmType.FT_ONLY.value
    out: Dict[str, Any] = {"per_seed": [], "pooled": []}

    for ds in datasets:
        for sd in seeds_by_ds.get(ds, []):
            base = by_key.get((ds, sd, ft_only_arm))
            if base is None:
                continue
            for arm in arms:
                if arm == ft_only_arm:
                    continue
                other = by_key.get((ds, sd, arm))
                if other is None:
                    continue
                try:
                    res = paired_bootstrap_diff(base, other)
                except Exception as e:
                    logger.warning(
                        "paired CI failed (%s seed=%d %s vs ft_only): %s",
                        ds, sd, arm, e,
                    )
                    continue
                if res is None:
                    continue
                out["per_seed"].append({
                    "dataset": ds,
                    "seed": sd,
                    "arm_a": ft_only_arm,
                    "arm_b": arm,
                    **_dc_asdict(res),
                })

        for arm in arms:
            if arm == ft_only_arm:
                continue
            base_per_seed: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
            other_per_seed: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
            for sd in seeds_by_ds.get(ds, []):
                if (ds, sd, ft_only_arm) in by_key and (ds, sd, arm) in by_key:
                    base_per_seed[(ft_only_arm, sd)] = by_key[(ds, sd, ft_only_arm)]
                    other_per_seed[(arm, sd)] = by_key[(ds, sd, arm)]
            if not base_per_seed or not other_per_seed:
                continue
            try:
                pooled_a = pool_per_question_outcomes(base_per_seed).get(ft_only_arm, [])
                pooled_b = pool_per_question_outcomes(other_per_seed).get(arm, [])
                res = paired_bootstrap_diff(pooled_a, pooled_b)
            except Exception as e:
                logger.warning(
                    "pooled paired CI failed (%s %s vs ft_only): %s",
                    ds, arm, e,
                )
                continue
            if res is None:
                continue
            out["pooled"].append({
                "dataset": ds,
                "arm_a": ft_only_arm,
                "arm_b": arm,
                "n_seeds": len(base_per_seed),
                **_dc_asdict(res),
            })

    return out


# ---------------------------------------------------------------------------
# Full study orchestrator
# ---------------------------------------------------------------------------

def run_full_study(study_cfg: FTStudyConfig) -> Dict[str, Any]:
    """Run the complete study: all arms x datasets x seeds."""
    all_results: List[Dict[str, Any]] = []
    output_dir = study_cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    is_instruct = get_is_instruct(study_cfg.model_name)

    for dataset_name in study_cfg.datasets:
        for seed in study_cfg.seeds:
            logger.info(
                "--- Preparing splits: dataset=%s seed=%d ---",
                dataset_name, seed,
            )
            splits = create_four_way_split(
                dataset_name, is_instruct, seed,
                split_cfg=study_cfg.split,
                hf_split=study_cfg.data_split_name,
            )
            splits["_train_ds"] = make_sft_dataset(splits["train_ft"])
            splits["_val_ds"] = make_sft_dataset(splits["val_select"])

            for arm in study_cfg.arms:
                result_path = os.path.join(
                    output_dir, dataset_name, f"seed_{seed}", arm.value, "result.json",
                )
                if os.path.exists(result_path):
                    logger.info("Skipping %s/%s/seed_%d (result exists)", dataset_name, arm.value, seed)
                    with open(result_path) as f:
                        all_results.append(json.load(f))
                    continue

                try:
                    result = run_arm(
                        arm, study_cfg.model_name, dataset_name,
                        splits, study_cfg, seed, output_dir,
                    )
                    all_results.append(result)
                except Exception as e:
                    logger.error(
                        "FAILED: %s | %s | seed %d: %s",
                        arm.value, dataset_name, seed, e, exc_info=True,
                    )
                    all_results.append({
                        "arm": arm.value, "dataset": dataset_name,
                        "seed": seed, "error": str(e),
                    })

    summary = aggregate_results(all_results)
    summary["all_results"] = all_results
    summary["config"] = asdict(study_cfg)

    summary_path = os.path.join(output_dir, "study_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Study summary saved to %s", summary_path)

    _print_summary(summary)
    return summary


def _print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("FT INTERACTION STUDY -- SUMMARY")
    print("=" * 80)

    arm_summary = summary.get("arm_summary", {})
    datasets = sorted({v["dataset"] for v in arm_summary.values()})
    arms = sorted({v["arm"] for v in arm_summary.values()})

    header = f"{'Dataset':<20}" + "".join(f"{a:>18}" for a in arms)
    print(header)
    print("-" * len(header))

    for ds in datasets:
        row = f"{ds:<20}"
        for arm in arms:
            key = f"{arm}_{ds}"
            if key in arm_summary:
                s = arm_summary[key]
                if s.get("n_seeds", 0) == 0:
                    row += f" {'N/A':>17}"
                else:
                    row += f" {s['mean']:.4f}+/-{s['std']:.4f}"
            else:
                row += f" {'N/A':>17}"
        print(row)

    route_table = summary.get("route_table", [])
    if route_table:
        print("\n--- Route Drift ---")
        n_changed = sum(1 for r in route_table if not r.get("identical", True))
        print(f"  Sequences changed after FT: {n_changed}/{len(route_table)}")
        dists = [r["edit_distance"] for r in route_table if r["edit_distance"] >= 0]
        if dists:
            print(f"  Edit distance: mean={np.mean(dists):.1f}, max={max(dists)}")

    paired = summary.get("paired_cis") or {}
    pooled = paired.get("pooled") or []
    if pooled:
        print("\n--- Paired bootstrap CIs (pooled across seeds, vs ft_only) ---")
        for entry in sorted(pooled, key=lambda e: (e["dataset"], e["arm_b"])):
            print(
                f"  {entry['dataset']:<16} {entry['arm_b']:<18}"
                f" diff={entry['mean_diff']:+.4f}"
                f" 95% CI=[{entry['ci_lo']:+.4f}, {entry['ci_hi']:+.4f}]"
                f" p={entry['p_two_sided']:.3f}"
                f" (n={entry['n_paired']}, seeds={entry['n_seeds']})"
            )

    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning interaction study (TALE-style ordering)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--datasets", type=str, nargs="+",
        default=["arc_easy", "arc_challenge", "mmlu", "commonsenseqa", "boolq", "winogrande"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--arms", type=str, nargs="+",
        default=[a.value for a in ArmType],
        choices=[a.value for a in ArmType],
    )
    parser.add_argument("--output_dir", type=str, default="ft_study_results")
    parser.add_argument("--gpu_rank", type=int, default=0)
    parser.add_argument("--num_simulations", type=int, default=10_000)
    parser.add_argument("--signal", action="store_true", default=False)
    parser.add_argument(
        "--cached_sequences_dir", type=str, nargs="+", default=None,
        help="Directories containing prior MCTS benchmark_mcts_*.json results. "
             "When provided, base-model search is skipped for datasets that "
             "have a cached best sequence.",
    )
    parser.add_argument(
        "--freeze_repeated_lora", action="store_true", default=False,
        help="Freeze LoRA adapters on layers that appear more than once "
             "(or not at all) in the active sequence. Prevents doubled "
             "gradients on repeated layers.",
    )
    parser.add_argument(
        "--scale_repeated_lr", action="store_true", default=False,
        help="Scale LR for LoRA adapters on repeated layers by 1/count "
             "(where count = number of occurrences in the sequence). "
             "Compensates for accumulated gradients while keeping layers trainable.",
    )
    parser.add_argument(
        "--clone_repeated_lora", action="store_true", default=False,
        help="Clone repeated layers so each occurrence has its own independent "
             "LoRA adapter. Eliminates gradient direction conflict between "
             "positions while keeping base weights shared.",
    )
    parser.add_argument(
        "--share_train_for_search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the train_ft pool (== val_search) for MCTS search "
             "evaluation. Default ON under the new split scheme. Pass "
             "--no-share_train_for_search to revert to a dedicated "
             "val_search split (legacy behavior).",
    )
    parser.add_argument(
        "--mcts_4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load base MCTS in 4-bit (NF4). Default OFF: base MCTS now "
             "runs in fp16 to match evaluation precision and avoid "
             "spurious wins that don't transfer to the test path.",
    )
    parser.add_argument(
        "--search_tier2", type=int, default=None,
        help="Override tier-2 sample count for MCTS (default: 300)",
    )
    parser.add_argument(
        "--search_tier3", type=int, default=None,
        help="Override tier-3 sample count for MCTS (default: 1500)",
    )
    parser.add_argument(
        "--search_tier4", type=int, default=None,
        help="Override tier-4 sample count for MCTS. Default -1 means "
             "'use all remaining train_ft samples after tier2/tier3'.",
    )
    parser.add_argument(
        "--rerank_topk", type=int, default=None,
        help="Override post-MCTS top-K re-rank pool size on val_select "
             "(default: 5).",
    )
    parser.add_argument(
        "--promote_use_wilson",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Gate tier-2->tier-3 and tier-3->tier-4 promotions on the "
             "Wilson lower bound exceeding the baseline accuracy. "
             "Default ON (winner's-curse mitigation).",
    )
    parser.add_argument(
        "--preset_sequences_json", type=str, default=None,
        help="Path to a JSON file mapping dataset name → best layer sequence "
             "(e.g. {\"boolq\": [0,1,...,23]}). Overrides --cached_sequences_dir.",
    )
    parser.add_argument(
        "--verify_only", action="store_true",
        help="Only run the LoRA-layer-reorder sanity check, then exit",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if args.verify_only:
        ok = verify_lora_layer_reorder(args.model_name, rank=args.gpu_rank)
        sys.exit(0 if ok else 1)

    cached_seqs: Dict[str, List[int]] = {}
    if args.cached_sequences_dir:
        cached_seqs = load_cached_sequences(
            args.cached_sequences_dir, dataset_names=args.datasets,
        )
        logger.info(
            "Loaded cached sequences for %d datasets: %s",
            len(cached_seqs), list(cached_seqs.keys()),
        )
    if args.preset_sequences_json:
        with open(args.preset_sequences_json) as f:
            preset = json.load(f)
        for ds, seq in preset.items():
            cached_seqs[ds] = seq
            logger.info("Preset sequence for %s: %s", ds, seq)

    study_cfg = FTStudyConfig(
        model_name=args.model_name,
        datasets=args.datasets,
        seeds=args.seeds,
        arms=[ArmType(a) for a in args.arms],
        gpu_rank=args.gpu_rank,
        output_dir=args.output_dir,
        notify_signal=args.signal,
        cached_sequences=cached_seqs,
    )
    study_cfg.search.num_simulations = args.num_simulations
    study_cfg.share_train_for_search = args.share_train_for_search
    study_cfg.search.mcts_load_in_4bit = args.mcts_4bit
    study_cfg.search.promote_use_wilson = args.promote_use_wilson
    if args.search_tier2 is not None:
        study_cfg.search.tier2_samples = args.search_tier2
    if args.search_tier3 is not None:
        study_cfg.search.tier3_samples = args.search_tier3
    if args.search_tier4 is not None:
        study_cfg.search.tier4_samples = args.search_tier4
    if args.rerank_topk is not None:
        study_cfg.search.rerank_topk = args.rerank_topk
    study_cfg.ft.freeze_repeated_lora = args.freeze_repeated_lora
    study_cfg.ft.scale_repeated_lr = args.scale_repeated_lr
    study_cfg.ft.clone_repeated_lora = args.clone_repeated_lora

    run_full_study(study_cfg)


if __name__ == "__main__":
    main()
