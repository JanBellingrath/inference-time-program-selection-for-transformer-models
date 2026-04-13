#!/usr/bin/env python3
"""
Bivariate interaction analysis for two-edit MCTS routes.

For every tier-4 route that differs from the identity baseline by exactly two
edits (Hamming distance 2), this script decomposes the performance gain into
single-edit contributions and a bivariate interaction term:

    I_12 = g_12 - g_1 - g_2

where g_i = F(r_i; D) - F(r_0; D) and F is a mean per-example score.

Six per-example scoring metrics are computed (three binary, three continuous).
Paired bootstrap confidence intervals and a classification of each route
(localized / additive / synergistic / redundant) are reported.

Usage:
    python analyze_bivariate_interactions.py \
        --model_name Qwen/Qwen2.5-0.5B-Instruct \
        --benchmarks winogrande boolq mmlu_all arc_challenge \
        --tier tier4 \
        --n_bootstrap 10000

    python analyze_bivariate_interactions.py \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --benchmarks winogrande boolq mmlu_all arc_easy commonsenseqa \
        --tier tier4
"""
import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from core.benchmark_mcts import grade_response, seq_to_layers, SKIP
from core.flexible_models import FlexibleModelWrapper, get_is_instruct
from core.permutation_mcts import MCTSModel, prepare_arc_data, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Snapshot registry (mirrors run_publication_rc_vs_sc.py)
# ---------------------------------------------------------------------------

SNAPSHOT_REGISTRY = {
    ("0.5B", "winogrande"): "predictions/qwen25_0.5b_v2_sdpa_r5_pw/benchmark_mcts_winogrande_20260311-104703_snapshot.json",
    ("0.5B", "boolq"): "predictions/qwen25_0.5b_v2_sdpa_r5_pw/benchmark_mcts_boolq_20260311-115713_snapshot.json",
    ("0.5B", "arc_easy"): "predictions/qwen25_0.5b_v2_sdpa/benchmark_mcts_arc_easy_20260308-232019_snapshot.json",
    ("0.5B", "commonsenseqa"): "predictions/qwen25_0.5b_v2_sdpa_r5_pw/benchmark_mcts_commonsenseqa_20260311-125244_snapshot.json",
    ("0.5B", "mmlu_all"): "predictions/qwen25_0.5b_v2_sdpa_r5_pw/benchmark_mcts_mmlu_all_20260311-132051_snapshot.json",
    ("0.5B", "arc_challenge"): "predictions/qwen25_0.5b_v2_sdpa_r5_pw/benchmark_mcts_arc_challenge_20260311-112922_snapshot.json",
    ("7B", "winogrande"): "predictions/qwen25_7b_v2_sdpa/benchmark_mcts_winogrande_20260308-191822_snapshot.json",
    ("7B", "boolq"): "predictions/qwen25_7b_v2_sdpa/benchmark_mcts_boolq_20260308-195558_snapshot.json",
    ("7B", "arc_easy"): "predictions/qwen25_7b_v2_sdpa/benchmark_mcts_arc_easy_20260308-221829_snapshot.json",
    ("7B", "commonsenseqa"): "predictions/qwen25_7b_v2_sdpa/benchmark_mcts_commonsenseqa_20260308-202012_snapshot.json",
    ("7B", "mmlu_all"): "predictions/qwen25_7b_v2_sdpa/benchmark_mcts_mmlu_all_20260308-211217_snapshot.json",
    ("7B", "arc_challenge"): "predictions/qwen25_7b_v2_sdpa/benchmark_mcts_arc_challenge_20260308-013818_snapshot.json",
}

BINARY_METRICS = ["accuracy_gen", "accuracy_loglik_next", "accuracy_loglik_full"]
CONTINUOUS_METRICS = ["logprob_correct", "logit_margin", "loglik_full_correct"]
ALL_METRICS = BINARY_METRICS + CONTINUOUS_METRICS


def model_key_from_name(model_name: str) -> str:
    name = model_name.lower()
    if "0.5b" in name:
        return "0.5B"
    if "7b" in name:
        return "7B"
    raise ValueError(f"Cannot infer model key from {model_name}")


def resolve_snapshot(model_name: str, benchmark: str,
                     cli_snapshot: Optional[str] = None) -> str:
    if cli_snapshot:
        return cli_snapshot
    key = (model_key_from_name(model_name), benchmark)
    rel = SNAPSHOT_REGISTRY.get(key)
    if not rel:
        raise ValueError(f"No snapshot registered for {key}. Pass --snapshot.")
    return str(SCRIPT_DIR / rel)


# ---------------------------------------------------------------------------
# Route decomposition
# ---------------------------------------------------------------------------

def extract_two_edit_routes(
    snapshot_path: str,
    tier: str,
    num_layers: int,
    max_routes: int = 0,
    top_k_two_edit: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    """Load tier entries and keep those with Hamming distance 2 from identity.

    If *top_k_two_edit* > 0, keep only the *k* best by snapshot ``accuracy``
    (then ``delta``), among two-edit sequences only.

    Returns ``(routes, n_two_edit_before_cap)`` where *n_two_edit_before_cap*
    is how many two-edit rows existed before any *top_k* / *max_routes* cap.

    Each route dict has:
        seq, layers, edits [(pos, val), (pos, val)],
        source_acc, source_delta
    """
    with open(snapshot_path) as f:
        snap = json.load(f)

    tier_keys = {
        "tier4": ["tier4"],
        "tier3": ["tier3"],
        "tier2": ["validated"],
        "auto": ["tier4", "tier3", "validated"],
    }
    src = []
    for key in tier_keys.get(tier, tier_keys["auto"]):
        src = snap.get(key, [])
        if src:
            break

    baseline = list(range(num_layers))
    results = []
    for entry in src:
        seq = entry.get("seq")
        if seq is None:
            continue
        if len(seq) != num_layers:
            continue
        diffs = [(i, seq[i]) for i in range(num_layers) if seq[i] != i]
        if len(diffs) != 2:
            continue
        layers = entry.get("layers")
        if layers is None:
            layers = seq_to_layers(seq)
        results.append({
            "seq": seq,
            "layers": layers,
            "edits": diffs,
            "source_acc": entry.get("accuracy"),
            "source_delta": entry.get("delta"),
            "source_n": entry.get("evaluated"),
        })

    n_eligible = len(results)

    if top_k_two_edit > 0:
        results.sort(
            key=lambda r: (
                r.get("source_acc") if r.get("source_acc") is not None
                else float("-inf"),
                r.get("source_delta") if r.get("source_delta") is not None
                else float("-inf"),
            ),
            reverse=True,
        )
        results = results[:top_k_two_edit]
    elif max_routes > 0:
        results = results[:max_routes]

    return results, n_eligible


def build_route_quad(
    edits: List[Tuple[int, int]],
    num_layers: int,
) -> Dict[str, List[int]]:
    """Build the four routes r_0, r_1, r_2, r_12 from two edits."""
    r0 = list(range(num_layers))
    r1 = list(r0)
    r1[edits[0][0]] = edits[0][1]
    r2 = list(r0)
    r2[edits[1][0]] = edits[1][1]
    r12 = list(r0)
    r12[edits[0][0]] = edits[0][1]
    r12[edits[1][0]] = edits[1][1]
    return {
        "r0": r0,
        "r1": seq_to_layers(r1),
        "r2": seq_to_layers(r2),
        "r12": seq_to_layers(r12),
        "r0_layers": seq_to_layers(r0),
    }


def collect_unique_routes(
    two_edit_routes: List[Dict],
    num_layers: int,
) -> List[Tuple[int, ...]]:
    """Gather the set of unique layer-tuples that need evaluation."""
    seen = set()
    r0_layers = tuple(range(num_layers))
    seen.add(r0_layers)
    for route_info in two_edit_routes:
        quad = build_route_quad(route_info["edits"], num_layers)
        for key in ("r1", "r2", "r12"):
            seen.add(tuple(quad[key]))
    return sorted(seen)


# ---------------------------------------------------------------------------
# Per-example scoring engine
# ---------------------------------------------------------------------------

def _sample_hash(sample: Dict) -> str:
    text = sample["input"] + str(sample.get("correct", ""))
    return hashlib.md5(text.encode()).hexdigest()


def _forward_logits(
    wrapper: FlexibleModelWrapper,
    layers: List[int],
    text: str,
    system_prompt: Optional[str] = None,
) -> torch.Tensor:
    """Forward pass under *layers*, return last-position logits [vocab]."""
    saved = wrapper.model.model.layer_indices
    wrapper.model.model.layer_indices = layers
    try:
        has_dup = len(layers) != len(set(layers))
        prompt = wrapper.prepare_prompt(text, system_prompt=system_prompt)
        inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(
            wrapper.model.device
        )
        kw: dict = {}
        if has_dup or len(layers) != wrapper.num_layers:
            kw["use_cache"] = False
        with torch.no_grad():
            out = wrapper.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **kw,
            )
        return out.logits[0, -1, :]
    finally:
        wrapper.model.model.layer_indices = saved


def _generate(
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
        inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(
            wrapper.model.device
        )
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


def _loglik_full_sequence(
    wrapper: FlexibleModelWrapper,
    layers: List[int],
    sample: Dict,
) -> Dict[str, float]:
    """Length-normalized log P(choice | prompt) for each choice label."""
    text = sample["input"]
    sys_prompt = sample.get("system_prompt")
    choices = sample.get("choices", [])
    choice_labels = sample.get("choice_labels", [])
    if not choices or not choice_labels:
        return {}

    saved = wrapper.model.model.layer_indices
    wrapper.model.model.layer_indices = layers
    try:
        tokenizer = wrapper.tokenizer
        prefix = wrapper.prepare_prompt(text, system_prompt=sys_prompt) + " "
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        prompt_len = len(prefix_ids)
        tok_kw = {"return_tensors": "pt", "add_special_tokens": False}

        has_dup = len(layers) != len(set(layers))
        fwd_kw: dict = {}
        if has_dup or len(layers) != wrapper.num_layers:
            fwd_kw["use_cache"] = False

        result = {}
        for label, choice_text in zip(choice_labels, choices):
            full_text = prefix + choice_text
            full_enc = tokenizer(full_text, **tok_kw).to(wrapper.model.device)
            full_ids = full_enc.input_ids
            with torch.no_grad():
                out = wrapper.model(
                    input_ids=full_ids,
                    attention_mask=full_enc.attention_mask,
                    **fwd_kw,
                )
            logits = out.logits
            log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
            choice_start = prompt_len
            choice_end = full_ids.shape[1]
            n_tokens = choice_end - choice_start
            if n_tokens <= 0:
                result[label] = float("-inf")
                continue
            total_lp = 0.0
            for pos in range(choice_start, choice_end):
                token_id = full_ids[0, pos].item()
                total_lp += log_probs[pos - 1, token_id].item()
            result[label] = total_lp / n_tokens
        return result
    finally:
        wrapper.model.model.layer_indices = saved


def score_sample(
    wrapper: FlexibleModelWrapper,
    layers: List[int],
    sample: Dict,
    dataset: str,
    model_name: str,
    is_math: bool = False,
    skip_loglik_full: bool = False,
) -> Dict[str, float]:
    """Compute all six metrics for one (route, example) pair.

    Returns a dict with keys from ALL_METRICS. Metrics that cannot be computed
    (e.g. logit-based metrics for non-MC tasks) are set to NaN.
    """
    scores: Dict[str, float] = {}
    sys_prompt = sample.get("system_prompt")
    max_tok = sample.get("max_new_tokens", 1)
    is_mc = sample.get("is_mc") and sample.get("choice_labels")
    correct_label = sample["correct"].strip() if is_mc else None

    nan = float("nan")

    if max_tok == 1 and is_mc:
        logits = _forward_logits(
            wrapper, layers, sample["input"], system_prompt=sys_prompt
        )
        last_logits = logits
        log_probs = torch.nn.functional.log_softmax(last_logits, dim=-1)

        # accuracy_gen: decode argmax token, grade
        next_token_id = last_logits.argmax(dim=-1).item()
        resp = wrapper.tokenizer.decode(
            [next_token_id], skip_special_tokens=True
        ).strip()
        gen_ok = grade_response(
            resp, sample["correct"], dataset, model_name, sample["input"]
        )
        scores["accuracy_gen"] = float(gen_ok > 0.5)

        # loglik_next: per-label log-probs
        lnext = {}
        for label in sample["choice_labels"]:
            tok_ids = wrapper.tokenizer.encode(label, add_special_tokens=False)
            if tok_ids:
                lnext[label] = log_probs[tok_ids[0]].item()
            else:
                lnext[label] = float("-inf")
        if lnext:
            pred_next = max(lnext, key=lnext.get)
            scores["accuracy_loglik_next"] = float(pred_next == correct_label)
        else:
            scores["accuracy_loglik_next"] = nan

        # logprob_correct
        if correct_label and correct_label in lnext:
            scores["logprob_correct"] = lnext[correct_label]
        else:
            tok_ids = wrapper.tokenizer.encode(
                sample["correct"].strip(), add_special_tokens=False
            )
            if tok_ids:
                scores["logprob_correct"] = log_probs[tok_ids[0]].item()
            else:
                scores["logprob_correct"] = nan

        # logit_margin: correct logit minus best incorrect logit
        if correct_label:
            correct_tok = wrapper.tokenizer.encode(
                correct_label, add_special_tokens=False
            )
            incorrect_labels = [
                l for l in sample["choice_labels"] if l != correct_label
            ]
            incorrect_toks = [
                wrapper.tokenizer.encode(l, add_special_tokens=False)
                for l in incorrect_labels
            ]
            if correct_tok and any(incorrect_toks):
                c_logit = last_logits[correct_tok[0]].item()
                best_inc = max(
                    last_logits[t[0]].item()
                    for t in incorrect_toks
                    if t
                )
                scores["logit_margin"] = c_logit - best_inc
            else:
                scores["logit_margin"] = nan
        else:
            scores["logit_margin"] = nan

    else:
        # Multi-token generation: only binary gen accuracy available
        resp = _generate(
            wrapper, layers, sample["input"],
            system_prompt=sys_prompt,
            max_new_tokens=max_tok,
            is_math=is_math,
        )
        gen_ok = grade_response(
            resp, sample["correct"], dataset, model_name, sample["input"]
        )
        scores["accuracy_gen"] = float(gen_ok > 0.5)
        scores["accuracy_loglik_next"] = nan
        scores["logprob_correct"] = nan
        scores["logit_margin"] = nan

    # loglik_full_sequence (works for MC tasks regardless of max_tok)
    if is_mc and not skip_loglik_full:
        lfull = _loglik_full_sequence(wrapper, layers, sample)
        if lfull:
            pred_full = max(lfull, key=lfull.get)
            scores["accuracy_loglik_full"] = float(pred_full == correct_label)
            if correct_label and correct_label in lfull:
                scores["loglik_full_correct"] = lfull[correct_label]
            else:
                scores["loglik_full_correct"] = nan
        else:
            scores["accuracy_loglik_full"] = nan
            scores["loglik_full_correct"] = nan
    else:
        scores["accuracy_loglik_full"] = nan
        scores["loglik_full_correct"] = nan

    return scores


def evaluate_routes_on_dataset(
    wrapper: FlexibleModelWrapper,
    routes: Dict[Tuple[int, ...], str],  # layers_tuple -> label (for logging)
    samples: List[Dict],
    dataset: str,
    model_name: str,
    skip_loglik_full: bool = False,
) -> Dict[Tuple[int, ...], List[Dict[str, float]]]:
    """Evaluate each route on every sample, returning per-example score dicts.

    Uses caching: each (layers_tuple, sample_hash) is evaluated at most once.
    """
    is_math = "dart" in dataset or dataset in ("gsm8k_hard", "math500")
    cache: Dict[Tuple[Tuple[int, ...], str], Dict[str, float]] = {}
    result: Dict[Tuple[int, ...], List[Dict[str, float]]] = {
        k: [] for k in routes
    }

    route_list = list(routes.keys())
    total = len(route_list) * len(samples)

    with tqdm(total=total, desc=f"  scoring {dataset}") as pbar:
        for layers_tuple in route_list:
            layers = list(layers_tuple)
            for sample in samples:
                s_hash = _sample_hash(sample)
                cache_key = (layers_tuple, s_hash)
                if cache_key not in cache:
                    cache[cache_key] = score_sample(
                        wrapper, layers, sample, dataset, model_name,
                        is_math, skip_loglik_full,
                    )
                result[layers_tuple].append(cache[cache_key])
                pbar.update(1)

    return result


# ---------------------------------------------------------------------------
# Bootstrap and statistical tests
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng_seed: int = 12345,
) -> Dict[str, float]:
    """Bootstrap CI for the mean of a 1-D array."""
    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "p_value_zero": float("nan")}
    rng = np.random.RandomState(rng_seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_means[i] = values[idx].mean()
    boot_means.sort()
    lo = float(boot_means[int(n_boot * alpha / 2)])
    hi = float(boot_means[int(n_boot * (1 - alpha / 2))])
    # Two-sided p-value for H_0: mean = 0
    frac_ge_zero = float(np.mean(boot_means >= 0))
    frac_le_zero = float(np.mean(boot_means <= 0))
    p_val = 2.0 * min(frac_ge_zero, frac_le_zero)
    p_val = min(p_val, 1.0)
    return {"mean": float(values.mean()), "ci_lo": lo, "ci_hi": hi,
            "p_value_zero": p_val}


def bootstrap_Lmax_ci(
    g1_x: np.ndarray,
    g2_x: np.ndarray,
    g12_x: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng_seed: int = 12345,
) -> Dict[str, float]:
    """Bootstrap CI for L_max = max(mean(g1), mean(g2)) / mean(g12).

    Each bootstrap replicate resamples examples and computes the ratio of
    dataset-level means.  Replicates where mean(g12) <= 0 are dropped.
    """
    n = len(g1_x)
    rng = np.random.RandomState(rng_seed)
    boot_vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        d = g12_x[idx].mean()
        if d > 0:
            num = max(g1_x[idx].mean(), g2_x[idx].mean())
            boot_vals.append(float(num / d))
    if not boot_vals:
        return {"mean": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan")}
    boot_vals = np.array(sorted(boot_vals))
    lo = float(boot_vals[int(len(boot_vals) * alpha / 2)])
    hi = float(boot_vals[int(len(boot_vals) * (1 - alpha / 2))])
    d_mean = g12_x.mean()
    point = (
        float(max(g1_x.mean(), g2_x.mean()) / d_mean)
        if d_mean > 0 else float("nan")
    )
    return {"mean": point, "ci_lo": lo, "ci_hi": hi}


# ---------------------------------------------------------------------------
# Interaction analysis per route
# ---------------------------------------------------------------------------

def analyze_route(
    scores_r0: List[Dict[str, float]],
    scores_r1: List[Dict[str, float]],
    scores_r2: List[Dict[str, float]],
    scores_r12: List[Dict[str, float]],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Full bivariate interaction analysis for one two-edit route."""
    n = len(scores_r0)
    results: Dict[str, Any] = {"n_examples": n, "per_metric": {}}

    for metric in ALL_METRICS:
        m0 = np.array([s[metric] for s in scores_r0])
        m1 = np.array([s[metric] for s in scores_r1])
        m2 = np.array([s[metric] for s in scores_r2])
        m12 = np.array([s[metric] for s in scores_r12])

        finite = (
            np.isfinite(m0) & np.isfinite(m1)
            & np.isfinite(m2) & np.isfinite(m12)
        )
        valid = finite
        n_valid = int(valid.sum())
        if n_valid < 5:
            results["per_metric"][metric] = {
                "n_valid": n_valid, "skipped": True
            }
            continue

        m0v, m1v, m2v, m12v = m0[valid], m1[valid], m2[valid], m12[valid]

        # Per-example gains
        g1_x = m1v - m0v
        g2_x = m2v - m0v
        g12_x = m12v - m0v
        I12_x = g12_x - g1_x - g2_x

        # Dataset-level
        F0 = float(m0v.mean())
        F1 = float(m1v.mean())
        F2 = float(m2v.mean())
        F12 = float(m12v.mean())
        g1 = F1 - F0
        g2 = F2 - F0
        g12 = F12 - F0
        I12 = g12 - g1 - g2
        L_max = max(g1, g2) / g12 if g12 > 0 else float("nan")

        # Per-example statistics
        I12_mean = float(I12_x.mean())
        I12_median = float(np.median(I12_x))
        I12_frac_pos = float(np.mean(I12_x > 0))
        I12_quantiles = {
            f"q{int(q*100)}": float(np.percentile(I12_x, q * 100))
            for q in [0.05, 0.25, 0.5, 0.75, 0.95]
        }

        # Sensitive subset D_sens = {x: m(r12,x) > m(r0,x)}
        sens_mask = m12v > m0v
        n_sens = int(sens_mask.sum())
        sens_results = {}
        if n_sens >= 3:
            g1_sens = float(g1_x[sens_mask].mean())
            g2_sens = float(g2_x[sens_mask].mean())
            g12_sens = float(g12_x[sens_mask].mean())
            I12_sens = g12_sens - g1_sens - g2_sens
            sens_results = {
                "n": n_sens,
                "g1": g1_sens, "g2": g2_sens, "g12": g12_sens,
                "I12": I12_sens,
                "I12_mean_per_example": float(I12_x[sens_mask].mean()),
                "I12_frac_pos": float(np.mean(I12_x[sens_mask] > 0)),
            }

        # Bootstrap CIs
        boot_g1 = bootstrap_ci(g1_x, n_bootstrap, alpha)
        boot_g2 = bootstrap_ci(g2_x, n_bootstrap, alpha)
        boot_g12 = bootstrap_ci(g12_x, n_bootstrap, alpha)
        boot_I12 = bootstrap_ci(I12_x, n_bootstrap, alpha)

        boot_Lmax = bootstrap_Lmax_ci(
            g1_x, g2_x, g12_x, n_bootstrap, alpha
        )

        # Classification
        classification = classify_route(
            boot_g1, boot_g2, boot_I12, boot_Lmax, g12
        )

        results["per_metric"][metric] = {
            "n_valid": n_valid,
            "F0": F0, "F1": F1, "F2": F2, "F12": F12,
            "g1": g1, "g2": g2, "g12": g12,
            "I12": I12,
            "L_max": L_max,
            "I12_per_example": {
                "mean": I12_mean,
                "median": I12_median,
                "frac_positive": I12_frac_pos,
                "quantiles": I12_quantiles,
            },
            "sensitive_subset": sens_results,
            "bootstrap": {
                "g1": boot_g1,
                "g2": boot_g2,
                "g12": boot_g12,
                "I12": boot_I12,
                "L_max": boot_Lmax,
            },
            "classification": classification,
        }

    # Primary classification uses the best continuous metric available
    primary_metric = _pick_primary_metric(results["per_metric"])
    results["primary_metric"] = primary_metric
    if primary_metric and primary_metric in results["per_metric"]:
        pm = results["per_metric"][primary_metric]
        if not pm.get("skipped"):
            results["classification"] = pm["classification"]
        else:
            results["classification"] = "insufficient_data"
    else:
        results["classification"] = "insufficient_data"

    return results


def _pick_primary_metric(per_metric: Dict[str, Any]) -> Optional[str]:
    """Choose the best available continuous metric for primary classification."""
    for m in ["logprob_correct", "logit_margin", "loglik_full_correct",
              "accuracy_gen"]:
        if m in per_metric and not per_metric[m].get("skipped"):
            return m
    return None


def classify_route(
    boot_g1: Dict, boot_g2: Dict, boot_I12: Dict, boot_Lmax: Dict,
    g12: float,
) -> str:
    """Classify a two-edit route based on bootstrap CIs."""
    if g12 <= 0:
        return "no_gain"

    I12_lo = boot_I12["ci_lo"]
    I12_hi = boot_I12["ci_hi"]
    Lmax_lo = boot_Lmax.get("ci_lo", float("nan"))

    I12_contains_zero = I12_lo <= 0 <= I12_hi

    # synergistic: I_12 CI entirely above zero
    if I12_lo > 0:
        return "synergistic_joint"

    # redundant/antagonistic: I_12 CI entirely below zero
    if I12_hi < 0:
        return "redundant_antagonistic"

    # CI contains zero -> additive or localized
    if I12_contains_zero:
        if not math.isnan(Lmax_lo) and Lmax_lo > 0.7:
            return "localized_single_edit"
        g1_lo = boot_g1["ci_lo"]
        g2_lo = boot_g2["ci_lo"]
        if g1_lo > 0 and g2_lo > 0:
            return "additive_joint"
        return "localized_single_edit"

    return "ambiguous"


# ---------------------------------------------------------------------------
# Aggregation across routes
# ---------------------------------------------------------------------------

def aggregate_routes(
    route_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate analysis across all two-edit routes for a benchmark."""
    n_routes = len(route_results)
    if n_routes == 0:
        return {"n_routes": 0}

    agg: Dict[str, Any] = {"n_routes": n_routes}

    # Classification counts
    classes = [r.get("classification", "unknown") for r in route_results]
    class_counts = {}
    for c in classes:
        class_counts[c] = class_counts.get(c, 0) + 1
    agg["classification_counts"] = class_counts
    agg["classification_fractions"] = {
        k: v / n_routes for k, v in class_counts.items()
    }

    # Per-metric aggregation
    agg["per_metric"] = {}
    for metric in ALL_METRICS:
        vals = {"g1": [], "g2": [], "g12": [], "I12": [], "L_max": []}
        for r in route_results:
            pm = r.get("per_metric", {}).get(metric, {})
            if pm.get("skipped"):
                continue
            for key in vals:
                v = pm.get(key)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    vals[key].append(v)

        metric_agg = {}
        for key, arr in vals.items():
            if arr:
                a = np.array(arr)
                metric_agg[key] = {
                    "n": len(arr),
                    "mean": float(a.mean()),
                    "median": float(np.median(a)),
                    "std": float(a.std()),
                    "min": float(a.min()),
                    "max": float(a.max()),
                    "q25": float(np.percentile(a, 25)),
                    "q75": float(np.percentile(a, 75)),
                }
            else:
                metric_agg[key] = {"n": 0}
        agg["per_metric"][metric] = metric_agg

    return agg


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_benchmark_summary(
    benchmark: str,
    route_results: List[Dict[str, Any]],
    agg: Dict[str, Any],
):
    n = agg["n_routes"]
    if n == 0:
        print(f"\n{benchmark}: no two-edit routes found")
        return

    print(f"\n{'='*80}")
    print(f"  {benchmark}: {n} two-edit routes analyzed")
    print(f"{'='*80}")

    # Classification
    cc = agg.get("classification_counts", {})
    cf = agg.get("classification_fractions", {})
    print("\n  Classification:")
    for cls in ["localized_single_edit", "additive_joint",
                "synergistic_joint", "redundant_antagonistic",
                "no_gain", "ambiguous"]:
        cnt = cc.get(cls, 0)
        frac = cf.get(cls, 0)
        if cnt > 0:
            print(f"    {cls:<30s}  {cnt:>4d}  ({frac:6.1%})")

    # Per-metric summary table
    for metric in ALL_METRICS:
        ma = agg["per_metric"].get(metric, {})
        g12_info = ma.get("g12", {})
        I12_info = ma.get("I12", {})
        Lmax_info = ma.get("L_max", {})
        if g12_info.get("n", 0) == 0:
            continue
        print(f"\n  {metric}:")
        print(f"    {'':>10s}  {'mean':>8s}  {'median':>8s}  {'std':>8s}  "
              f"{'min':>8s}  {'max':>8s}")
        for key, label in [("g1", "g_1"), ("g2", "g_2"), ("g12", "g_12"),
                           ("I12", "I_12"), ("L_max", "L_max")]:
            info = ma.get(key, {})
            if info.get("n", 0) == 0:
                continue
            print(f"    {label:>10s}  {info['mean']:>+8.4f}  "
                  f"{info['median']:>+8.4f}  {info['std']:>8.4f}  "
                  f"{info['min']:>+8.4f}  {info['max']:>+8.4f}")


def print_global_summary(
    all_agg: Dict[str, Dict[str, Any]],
):
    print(f"\n{'='*80}")
    print(f"  GLOBAL SUMMARY (all benchmarks)")
    print(f"{'='*80}")

    total_routes = sum(a["n_routes"] for a in all_agg.values())
    print(f"\n  Total two-edit routes analyzed: {total_routes}")

    # Merged classification
    merged_cc: Dict[str, int] = {}
    for a in all_agg.values():
        for cls, cnt in a.get("classification_counts", {}).items():
            merged_cc[cls] = merged_cc.get(cls, 0) + cnt
    print("\n  Classification (all routes):")
    for cls in ["localized_single_edit", "additive_joint",
                "synergistic_joint", "redundant_antagonistic",
                "no_gain", "ambiguous"]:
        cnt = merged_cc.get(cls, 0)
        frac = cnt / total_routes if total_routes > 0 else 0
        if cnt > 0:
            print(f"    {cls:<30s}  {cnt:>4d}  ({frac:6.1%})")

    # Per-benchmark one-line summary (primary metric)
    print(f"\n  Per-benchmark (primary metric):")
    print(f"    {'Benchmark':<20s} {'n':>4s} {'mean g12':>9s} "
          f"{'mean I12':>9s} {'mean Lmax':>10s} {'syn%':>6s} {'loc%':>6s}")
    print(f"    {'-'*68}")
    for bench, agg in all_agg.items():
        n = agg["n_routes"]
        if n == 0:
            continue
        # Find primary metric for this benchmark
        pm = None
        for m in ["logprob_correct", "logit_margin", "loglik_full_correct",
                   "accuracy_gen"]:
            if agg["per_metric"].get(m, {}).get("g12", {}).get("n", 0) > 0:
                pm = m
                break
        if pm is None:
            continue
        g12_m = agg["per_metric"][pm]["g12"]["mean"]
        I12_m = agg["per_metric"][pm]["I12"]["mean"]
        Lmax_m = agg["per_metric"][pm].get("L_max", {}).get("mean", float("nan"))
        cf = agg.get("classification_fractions", {})
        syn = cf.get("synergistic_joint", 0)
        loc = cf.get("localized_single_edit", 0)
        Lmax_s = f"{Lmax_m:>10.3f}" if not math.isnan(Lmax_m) else f"{'N/A':>10s}"
        print(f"    {bench:<20s} {n:>4d} {g12_m:>+9.4f} "
              f"{I12_m:>+9.4f} {Lmax_s} {syn:>5.1%} {loc:>5.1%}")


# ---------------------------------------------------------------------------
# Held-out data loading
# ---------------------------------------------------------------------------

NO_VALIDATION_SPLIT = {
    "winogrande", "boolq", "commonsenseqa",
}


def load_holdout_data(
    benchmark: str,
    is_instruct: bool,
    holdout_fraction: float = 0.5,
    seed: int = 42,
    num_samples: int = 0,
) -> List[Dict]:
    """Load held-out evaluation data for a benchmark.

    Uses the validation split when available; otherwise splits the train data.
    """
    try:
        data = prepare_arc_data(benchmark, is_instruct, split="validation")
        if data and len(data) >= 20:
            logger.info(
                "%s: using validation split (%d samples)", benchmark, len(data)
            )
            if num_samples > 0:
                data = data[:num_samples]
            return data
    except Exception:
        pass

    logger.info(
        "%s: no validation split, splitting train (holdout=%.0f%%)",
        benchmark, holdout_fraction * 100,
    )
    data = prepare_arc_data(benchmark, is_instruct, split="train")
    n = len(data)
    rng = random.Random(seed + sum(ord(c) for c in benchmark))
    order = list(range(n))
    rng.shuffle(order)
    n_hold = max(1, int(round(n * holdout_fraction)))
    n_hold = min(n_hold, n - 1)
    holdout = [data[i] for i in sorted(order[-n_hold:])]
    logger.info("%s: holdout = %d samples from train", benchmark, len(holdout))
    if num_samples > 0:
        holdout = holdout[:num_samples]
    return holdout


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Bivariate interaction analysis for two-edit MCTS routes."
    )
    p.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="HF model id (required unless --filter_results_json).",
    )
    p.add_argument(
        "--benchmarks",
        nargs="*",
        default=[],
        help="Benchmarks to evaluate (required unless --filter_results_json).",
    )
    p.add_argument(
        "--filter_results_json",
        type=str,
        default=None,
        help="Load a prior results JSON; keep top-k routes per benchmark by "
             "snapshot source_acc (then source_delta). Recomputes aggregate "
             "tables only — no model, no re-scoring. Requires --top_k_two_edit.",
    )
    p.add_argument("--tier", type=str, default="tier4",
                   choices=["tier4", "tier3", "auto"])
    p.add_argument("--snapshot", type=str, default=None,
                   help="Override snapshot path (all benchmarks).")
    p.add_argument("--max_routes", type=int, default=0,
                   help="Cap two-edit routes after file order (0 = no cap). "
                        "Ignored if --top_k_two_edit > 0.")
    p.add_argument("--top_k_two_edit", type=int, default=0,
                   help="Keep only the k two-edit routes with highest tier "
                        "accuracy (then delta) in the snapshot (0 = all).")
    p.add_argument("--num_samples", type=int, default=0,
                   help="Cap eval samples per benchmark (0 = full split).")
    p.add_argument("--holdout_fraction", type=float, default=0.5,
                   help="Fraction of train used as holdout when no val split.")
    p.add_argument("--n_bootstrap", type=int, default=10000)
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance level for CIs.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--save_per_example", action="store_true",
                   help="Include per-example score vectors in JSON output.")
    p.add_argument("--skip_loglik_full", action="store_true",
                   help="Skip the expensive full-sequence log-lik metrics "
                        "(accuracy_loglik_full, loglik_full_correct) to ~3x speed.")
    args = p.parse_args()
    if args.filter_results_json:
        if args.top_k_two_edit <= 0:
            p.error("--filter_results_json requires --top_k_two_edit > 0")
    else:
        if not args.model_name:
            p.error("--model_name is required when not using --filter_results_json")
        if not args.benchmarks:
            p.error("--benchmarks is required when not using --filter_results_json")
    return args


# ---------------------------------------------------------------------------
# Incremental save (crash-safe)
# ---------------------------------------------------------------------------

def _atomic_write_json(path: str, obj: Any) -> None:
    """Write JSON atomically (temp file + rename) to avoid half-written files."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    os.replace(tmp_path, path)


def build_summary_payload(all_results: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight summary without per-example score vectors."""
    summary: Dict[str, Any] = {}
    for bench, br in all_results.items():
        if bench.startswith("_") or not isinstance(br, dict):
            continue
        if "route_analyses" not in br:
            continue
        br_copy = {k: v for k, v in br.items() if k != "route_analyses"}
        compact_routes = []
        for ra in br["route_analyses"]:
            ra_copy = {
                k: v for k, v in ra.items() if k != "per_example_scores"
            }
            compact_routes.append(ra_copy)
        br_copy["route_analyses"] = compact_routes
        summary[bench] = br_copy
    return summary


def save_results_to_disk(
    all_results: Dict[str, Any],
    out_path: str,
    summary_path: str,
    *,
    partial: bool,
    completed_benchmarks: List[str],
    requested_benchmarks: List[str],
) -> None:
    """Write full + summary JSON. When partial=True, add top-level _meta."""
    payload: Dict[str, Any] = dict(all_results)
    if partial:
        payload["_meta"] = {
            "partial": True,
            "completed_benchmarks": list(completed_benchmarks),
            "requested_benchmarks": list(requested_benchmarks),
            "last_saved_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "note": "Intermediate checkpoint; more benchmarks may follow.",
        }
    _atomic_write_json(out_path, payload)

    summ = build_summary_payload(all_results)
    if partial:
        summ["_meta"] = payload["_meta"]
    _atomic_write_json(summary_path, summ)

    if partial:
        logger.info(
            "Checkpoint saved (%d benchmarks): %s",
            len(completed_benchmarks),
            out_path,
        )
    else:
        logger.info(
            "Final write (%d benchmarks): %s | %s",
            len(completed_benchmarks),
            out_path,
            summary_path,
        )


# ---------------------------------------------------------------------------
# Post-hoc top-k (no GPU)
# ---------------------------------------------------------------------------

def apply_top_k_posthoc(data: Dict[str, Any], k: int) -> Dict[str, Any]:
    """Keep only the k best routes per benchmark by source_acc, then source_delta.

    Recomputes ``aggregate`` from the filtered ``route_analyses``.  Preserves
    all per-route metrics and bootstrap output already stored in the JSON.
    """
    out: Dict[str, Any] = {}
    for key, br in data.items():
        if key.startswith("_"):
            continue
        if not isinstance(br, dict) or "route_analyses" not in br:
            out[key] = copy.deepcopy(br)
            continue
        br_new = copy.deepcopy(br)
        routes = br_new["route_analyses"]
        n_before = len(routes)
        sorted_r = sorted(
            routes,
            key=lambda r: (
                r.get("source_acc") if r.get("source_acc") is not None
                else float("-inf"),
                r.get("source_delta") if r.get("source_delta") is not None
                else float("-inf"),
            ),
            reverse=True,
        )[:k]
        for i, r in enumerate(sorted_r):
            rc = copy.deepcopy(r)
            rc["route_index"] = i
            sorted_r[i] = rc
        br_new["route_analyses"] = sorted_r
        br_new["aggregate"] = aggregate_routes(sorted_r)
        br_new["n_two_edit_routes"] = len(sorted_r)
        br_new["n_two_edit_routes_before_posthoc"] = n_before
        br_new["top_k_two_edit"] = k
        br_new["top_k_two_edit_posthoc"] = True
        out[key] = br_new
    return out


def run_posthoc_filter(args: argparse.Namespace) -> None:
    """Load saved JSON, apply top-k filter, print summaries, write new files."""
    raw_path = Path(args.filter_results_json)
    in_path = raw_path if raw_path.is_file() else SCRIPT_DIR / raw_path
    if not in_path.is_file():
        raise FileNotFoundError(f"Results JSON not found: {in_path}")

    with open(in_path) as f:
        data = json.load(f)

    k = args.top_k_two_edit
    filtered = apply_top_k_posthoc(data, k)
    prev_meta = data.get("_meta") if isinstance(data.get("_meta"), dict) else {}
    filtered["_meta"] = {
        **prev_meta,
        "posthoc_top_k_filter": True,
        "source_json": str(in_path.resolve()),
        "top_k_two_edit": k,
        "filtered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    all_agg: Dict[str, Dict[str, Any]] = {}
    bench_keys = sorted(
        k for k in filtered
        if not k.startswith("_")
        and isinstance(filtered[k], dict)
        and "aggregate" in filtered[k]
    )
    for bk in bench_keys:
        all_agg[bk] = filtered[bk]["aggregate"]
        print_benchmark_summary(
            bk,
            filtered[bk]["route_analyses"],
            filtered[bk]["aggregate"],
        )
    print_global_summary(all_agg)

    out_dir = args.output_dir or str(SCRIPT_DIR / "predictions" / "publication")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    stem = in_path.stem
    out_name = f"{stem}_topk{k}_{ts}.json"
    out_path = os.path.join(out_dir, out_name)
    summary_path = os.path.join(out_dir, out_name.replace(".json", "_summary.json"))

    _atomic_write_json(out_path, filtered)
    summ = build_summary_payload(filtered)
    summ["_meta"] = filtered["_meta"]
    _atomic_write_json(summary_path, summ)
    logger.info("Post-hoc top-%d written to %s", k, out_path)
    logger.info("Summary: %s", summary_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.filter_results_json:
        run_posthoc_filter(args)
        return

    out_dir = args.output_dir or str(
        SCRIPT_DIR / "predictions" / "publication"
    )
    os.makedirs(out_dir, exist_ok=True)

    ts = time.strftime("%Y%m%d-%H%M%S")
    mk = model_key_from_name(args.model_name).replace(".", "")
    fname = f"bivariate_interactions_{mk}_{ts}.json"
    out_path = os.path.join(out_dir, fname)
    summary_path = os.path.join(
        out_dir, fname.replace(".json", "_summary.json")
    )
    completed_benchmarks: List[str] = []

    logger.info("Output files (incremental checkpoints): %s", out_path)

    logger.info("Loading model: %s", args.model_name)
    model = MCTSModel(args.model_name)
    wrapper = model.wrapper
    num_layers = model.num_layers
    is_instruct = get_is_instruct(args.model_name)
    logger.info("Model loaded: %d layers", num_layers)

    all_results: Dict[str, Any] = {}
    all_agg: Dict[str, Dict[str, Any]] = {}

    for benchmark in args.benchmarks:
        logger.info("=" * 70)
        logger.info("BENCHMARK: %s", benchmark)
        logger.info("=" * 70)

        # Load snapshot and extract two-edit routes
        snapshot_path = resolve_snapshot(
            args.model_name, benchmark, args.snapshot
        )
        two_edit_routes, n_two_edit_eligible = extract_two_edit_routes(
            snapshot_path,
            args.tier,
            num_layers,
            args.max_routes,
            top_k_two_edit=args.top_k_two_edit,
        )
        if not two_edit_routes:
            logger.warning("%s: no two-edit routes found, skipping", benchmark)
            continue
        if args.top_k_two_edit > 0:
            logger.info(
                "%s: %d two-edit routes (top-%d by tier accuracy of %d eligible)",
                benchmark,
                len(two_edit_routes),
                args.top_k_two_edit,
                n_two_edit_eligible,
            )
        else:
            logger.info(
                "%s: %d two-edit routes (of %d eligible in tier list)",
                benchmark,
                len(two_edit_routes),
                n_two_edit_eligible,
            )

        # Load held-out evaluation data
        holdout = load_holdout_data(
            benchmark, is_instruct, args.holdout_fraction, args.seed,
            args.num_samples,
        )
        if len(holdout) < 10:
            logger.warning(
                "%s: only %d holdout samples, skipping", benchmark, len(holdout)
            )
            continue
        logger.info("%s: %d holdout samples", benchmark, len(holdout))

        # Collect unique routes to evaluate
        unique_routes = collect_unique_routes(two_edit_routes, num_layers)
        route_labels = {
            rt: f"route_{i}" for i, rt in enumerate(unique_routes)
        }
        logger.info(
            "%s: %d unique routes to evaluate (%d two-edit + intermediates + baseline)",
            benchmark, len(unique_routes), len(two_edit_routes),
        )

        # Evaluate all routes on holdout data (the expensive step)
        t0 = time.time()
        all_scores = evaluate_routes_on_dataset(
            wrapper, route_labels, holdout, benchmark, args.model_name,
            skip_loglik_full=args.skip_loglik_full,
        )
        elapsed = time.time() - t0
        logger.info(
            "%s: scoring done in %.1fs (%.2f samples*routes/s)",
            benchmark, elapsed,
            len(unique_routes) * len(holdout) / max(elapsed, 0.01),
        )

        # Analyze each two-edit route
        baseline_key = tuple(range(num_layers))
        scores_r0 = all_scores[baseline_key]
        route_analyses = []

        for ri, route_info in enumerate(two_edit_routes):
            quad = build_route_quad(route_info["edits"], num_layers)
            r1_key = tuple(quad["r1"])
            r2_key = tuple(quad["r2"])
            r12_key = tuple(quad["r12"])

            analysis = analyze_route(
                scores_r0=scores_r0,
                scores_r1=all_scores[r1_key],
                scores_r2=all_scores[r2_key],
                scores_r12=all_scores[r12_key],
                n_bootstrap=args.n_bootstrap,
                alpha=args.alpha,
            )
            analysis["route_index"] = ri
            analysis["edits"] = route_info["edits"]
            analysis["seq"] = route_info["seq"]
            analysis["source_acc"] = route_info["source_acc"]
            analysis["source_delta"] = route_info["source_delta"]

            if args.save_per_example:
                analysis["per_example_scores"] = {
                    "r0": scores_r0,
                    "r1": all_scores[r1_key],
                    "r2": all_scores[r2_key],
                    "r12": all_scores[r12_key],
                }

            route_analyses.append(analysis)

        # Aggregate
        agg = aggregate_routes(route_analyses)
        all_agg[benchmark] = agg

        # Print
        print_benchmark_summary(benchmark, route_analyses, agg)

        all_results[benchmark] = {
            "model_name": args.model_name,
            "benchmark": benchmark,
            "num_layers": num_layers,
            "tier": args.tier,
            "snapshot": snapshot_path,
            "n_holdout": len(holdout),
            "n_two_edit_routes": len(two_edit_routes),
            "n_two_edit_eligible_in_tier": n_two_edit_eligible,
            "top_k_two_edit": args.top_k_two_edit,
            "n_unique_routes_evaluated": len(unique_routes),
            "n_bootstrap": args.n_bootstrap,
            "alpha": args.alpha,
            "route_analyses": route_analyses,
            "aggregate": agg,
        }

        completed_benchmarks.append(benchmark)
        save_results_to_disk(
            all_results,
            out_path,
            summary_path,
            partial=True,
            completed_benchmarks=completed_benchmarks,
            requested_benchmarks=list(args.benchmarks),
        )

    # Global summary
    print_global_summary(all_agg)

    # Final save (no _meta partial flag)
    if all_results:
        save_results_to_disk(
            all_results,
            out_path,
            summary_path,
            partial=False,
            completed_benchmarks=completed_benchmarks,
            requested_benchmarks=list(args.benchmarks),
        )
    else:
        logger.warning("No benchmark completed; no JSON written.")


if __name__ == "__main__":
    main()
