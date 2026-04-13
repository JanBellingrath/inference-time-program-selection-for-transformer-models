#!/usr/bin/env python3
"""
Compare two test-time aggregation schemes under matched compute (K forward passes):

  1. Baseline        – single greedy pass with the default route.
  2. Self-consistency – K temperature-sampled passes with the default route,
                       aggregated by majority vote.
  3. Route consistency – K greedy passes, one per top-K MCTS route (from the
                        same benchmark, optimized on the *train* split),
                        aggregated by majority vote.

Evaluation is always on the held-out split (default: "validation") of the same
benchmark, guaranteeing zero overlap with the samples used during MCTS search.

Usage:
    python compare_aggregation.py \
        --model_name Qwen/Qwen2.5-0.5B-Instruct \
        --snapshot predictions/qwen25_0.5b_v2_sdpa_r5_pw/benchmark_mcts_winogrande_..._snapshot.json \
        --dataset winogrande \
        --K 5 \
        --temperature 0.7 \
        --num_samples 0 \
        --seed 42
"""
import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from core.permutation_mcts import MCTSModel, prepare_arc_data, set_seed
from core.flexible_models import get_is_instruct
from core.benchmark_mcts import grade_response, seq_to_layers, SKIP
from evaluation.evaluate_transfer import load_sequences_from_snapshot

try:
    from mathruler.grader import extract_boxed_content, grade_answer
    HAS_MATHRULER = True
except ImportError:
    HAS_MATHRULER = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent

# Benchmarks that lack a true separate validation split.  For these the
# "validation" key in prepare_arc_data either maps back to the same underlying
# data as "train" (DART) or uses a deterministic partition (_apply_manual_split).
# We track them here so the overlap guard can handle them correctly.
MANUAL_SPLIT_BENCHMARKS = {"gsm8k_hard", "bigbench_boolean_expressions"}
SINGLE_SPLIT_BENCHMARKS = {"math500", "asdiv"}  # only one split available
DART_BENCHMARKS = {f"dart-{i}" for i in range(1, 6)}


# ---------------------------------------------------------------------------
# Answer extraction (mirrors grade_response but returns the normalised answer)
# ---------------------------------------------------------------------------

def _extract_number(text: str) -> Optional[str]:
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m:
        return m.group(1).strip().replace(",", "")
    m = re.search(r'####\s*([^\s]+)', text)
    if m:
        return m.group(1).strip().replace(",", "")
    nums = re.findall(r'-?\d[\d,]*\.?\d*', text)
    if nums:
        return nums[-1].replace(",", "")
    return None


def _extract_mc_letter(text: str) -> Optional[str]:
    s = text.strip()
    m = re.match(r"^(?:Answer:\s*)?([A-Ja-j])[\.\)\s,:]?", s)
    if m:
        return m.group(1).upper()
    m = re.search(r'(?:answer|choice)\s*(?:is|:)\s*\(?([A-Ja-j])\)?', s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    first = s[:1].upper()
    if first and first in "ABCDEFGHIJ":
        return first
    return None


def extract_answer(raw_response: str, dataset: str, model_name: str,
                   input_text: str) -> Optional[str]:
    """Extract a normalised answer string suitable for majority voting."""
    is_dart = "dart" in dataset
    is_instruct = get_is_instruct(model_name)

    if is_dart or dataset in ("math500", "hendrycks_math", "amc_aime"):
        if not HAS_MATHRULER:
            return None
        resp = raw_response
        if "boxed" in input_text and not is_instruct:
            resp = "\\boxed{" + resp
        return extract_boxed_content(resp.strip()) or None

    if dataset in ("gsm8k_hard", "svamp", "asdiv", "mawps"):
        return _extract_number(raw_response)

    if dataset in ("winogrande", "copa", "piqa", "anli", "story_cloze"):
        s = raw_response.strip()
        if s in ("1", "2"):
            return s
        for ch in s:
            if ch in ("1", "2"):
                return ch
        return None

    if dataset == "bigbench_boolean_expressions":
        s = raw_response.strip().lower()
        if "true" in s:
            return "A"
        if "false" in s:
            return "B"
        return None

    if dataset == "boolq":
        s = raw_response.strip().upper()
        if s in ("A", "B"):
            return s
        if "TRUE" in s:
            return "A"
        if "FALSE" in s:
            return "B"
        return None

    return _extract_mc_letter(raw_response)


# ---------------------------------------------------------------------------
# Majority vote
# ---------------------------------------------------------------------------

def majority_vote(answers: List[Optional[str]], rng: random.Random) -> Optional[str]:
    """Return the most common non-None answer.  Ties broken randomly."""
    filtered = [a for a in answers if a is not None]
    if not filtered:
        return None
    counts = Counter(filtered)
    max_count = counts.most_common(1)[0][1]
    tied = [a for a, c in counts.items() if c == max_count]
    return rng.choice(tied)


def weighted_vote(
    answers: List[Optional[str]],
    weights: List[float],
    rng: random.Random,
) -> Optional[str]:
    """
    Weighted vote: score(a) = sum_k w_k * 1[ans_k == a]. Argmax over answers;
    ties broken randomly.

    If all weights are (near) zero, falls back to uniform weights over routes
    (equivalent to unweighted vote when all w_k equal).
    """
    if len(weights) != len(answers):
        raise ValueError(f"weights length {len(weights)} != answers {len(answers)}")
    w_use = list(weights)
    s = sum(w_use)
    if s < 1e-12:
        w_use = [1.0] * len(answers)

    scores: Dict[str, float] = defaultdict(float)
    for k, a in enumerate(answers):
        if a is None:
            continue
        scores[a] += w_use[k]

    if not scores:
        return None
    best = max(scores.values())
    tied = [a for a, sc in scores.items() if abs(sc - best) < 1e-9]
    return rng.choice(tied)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_one(
    wrapper,
    text: str,
    layers: List[int],
    system_prompt: Optional[str],
    max_new_tokens: int,
    temperature: float,
    is_math: bool,
) -> str:
    """Run a single forward pass and return the decoded response."""
    saved = wrapper.model.model.layer_indices
    wrapper.model.model.layer_indices = layers
    try:
        prompt = wrapper.prepare_prompt(text, system_prompt=system_prompt)
        inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(wrapper.model.device)
        input_len = inputs.input_ids.shape[1]
        has_dup = len(layers) != len(set(layers))
        variable_len = len(layers) != len(saved)

        gen_kw: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": wrapper.tokenizer.eos_token_id,
        }
        if has_dup or variable_len or is_math:
            gen_kw["use_cache"] = False
        if temperature > 0:
            gen_kw["do_sample"] = True
            gen_kw["temperature"] = temperature
        else:
            gen_kw["do_sample"] = False

        with torch.no_grad():
            out = wrapper.model.generate(**inputs, **gen_kw)
        return wrapper.tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
    finally:
        wrapper.model.model.layer_indices = saved


# ---------------------------------------------------------------------------
# Overlap guard
# ---------------------------------------------------------------------------

def _sample_fingerprints(samples: List[Dict]) -> set:
    """Return a set of SHA-256 hex digests of each sample's input text."""
    fps = set()
    for s in samples:
        h = hashlib.sha256(s["input"].encode("utf-8")).hexdigest()
        fps.add(h)
    return fps


def verify_no_overlap(
    dataset: str,
    eval_split: str,
    mcts_split: str,
    model_name: str,
    mcts_seed: int,
    mcts_benchmark_idx: int,
    mcts_n_samples: int,
) -> None:
    """Load MCTS-used samples and eval samples, assert zero input overlap."""
    is_instruct = get_is_instruct(model_name)

    if dataset in DART_BENCHMARKS:
        raise ValueError(
            f"DART benchmarks have only one split; cannot guarantee disjointness "
            f"for {dataset}. Choose a benchmark with a proper validation split."
        )
    if dataset in SINGLE_SPLIT_BENCHMARKS:
        raise ValueError(
            f"{dataset} has only one split; cannot guarantee disjointness."
        )

    logger.info("Overlap check: loading %s split='%s' ...", dataset, mcts_split)
    mcts_data = prepare_arc_data(dataset, is_instruct, split=mcts_split)
    shuffle_rng = random.Random(mcts_seed + mcts_benchmark_idx)
    shuffle_rng.shuffle(mcts_data)
    mcts_pool = mcts_data[:mcts_n_samples]
    mcts_fps = _sample_fingerprints(mcts_pool)

    logger.info("Overlap check: loading %s split='%s' ...", dataset, eval_split)
    eval_data = prepare_arc_data(dataset, is_instruct, split=eval_split)
    eval_fps = _sample_fingerprints(eval_data)

    overlap = mcts_fps & eval_fps
    if overlap:
        raise RuntimeError(
            f"OVERLAP DETECTED: {len(overlap)} samples appear in both the MCTS "
            f"pool ({mcts_split}, first {mcts_n_samples}) and eval split "
            f"({eval_split}) for {dataset}. Aborting."
        )
    logger.info(
        "Overlap check PASSED: 0 / %d eval samples overlap with %d MCTS samples.",
        len(eval_fps), len(mcts_fps),
    )


# ---------------------------------------------------------------------------
# Wilson CI
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare self-consistency vs route-consistency under matched compute."
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--snapshot", type=str, required=True,
                        help="Path to the MCTS snapshot JSON for this benchmark.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Benchmark name (same as the one the snapshot was optimized on).")
    parser.add_argument("--K", type=int, default=5,
                        help="Number of routes / sampled runs (matched compute).")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for self-consistency.")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Cap eval samples (0 = full eval split).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="validation",
                        help="Evaluation split (default: 'validation').")
    parser.add_argument("--tier", type=str, default="tier4",
                        choices=["tier4", "tier3", "auto"],
                        help="Which tier to draw top-K routes from.")
    parser.add_argument("--mcts_split", type=str, default="train",
                        help="Split used during MCTS (for overlap verification).")
    parser.add_argument("--mcts_seed", type=int, default=42,
                        help="Seed used during MCTS.")
    parser.add_argument("--mcts_benchmark_idx", type=int, default=None,
                        help="Benchmark index used for MCTS shuffle. "
                             "Auto-detected from the default 5-benchmark list if omitted.")
    parser.add_argument("--mcts_n_samples", type=int, default=1000,
                        help="Number of samples in the MCTS pool (tier-4 size).")
    parser.add_argument("--skip_overlap_check", action="store_true",
                        help="Skip the overlap verification (not recommended).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for output JSON (default: predictions/).")
    parser.add_argument("--report_every", type=int, default=25)
    args = parser.parse_args()

    set_seed(args.seed)
    vote_rng = random.Random(args.seed + 7)

    # -- auto-detect benchmark index for overlap check ----------------------
    DEFAULT_BENCHMARKS = [
        "winogrande", "arc_challenge", "boolq", "commonsenseqa", "mmlu_all",
    ]
    if args.mcts_benchmark_idx is None:
        if args.dataset in DEFAULT_BENCHMARKS:
            args.mcts_benchmark_idx = DEFAULT_BENCHMARKS.index(args.dataset)
        else:
            args.mcts_benchmark_idx = 0
            logger.warning(
                "Could not auto-detect mcts_benchmark_idx for %s; using 0. "
                "Pass --mcts_benchmark_idx explicitly for correctness.",
                args.dataset,
            )

    # -- overlap guard ------------------------------------------------------
    if not args.skip_overlap_check:
        verify_no_overlap(
            dataset=args.dataset,
            eval_split=args.split,
            mcts_split=args.mcts_split,
            model_name=args.model_name,
            mcts_seed=args.mcts_seed,
            mcts_benchmark_idx=args.mcts_benchmark_idx,
            mcts_n_samples=args.mcts_n_samples,
        )

    # -- load top-K routes --------------------------------------------------
    candidates = load_sequences_from_snapshot(args.snapshot, args.tier, args.K)
    if not candidates:
        logger.error("No sequences loaded from snapshot. Aborting.")
        sys.exit(1)
    K = min(args.K, len(candidates))
    if K < args.K:
        logger.warning("Only %d routes available (requested %d).", K, args.K)
    routes = [c["layers"] for c in candidates[:K]]
    logger.info("Loaded %d routes from %s (tier=%s):", K, args.snapshot, args.tier)
    for i, c in enumerate(candidates[:K]):
        logger.info(
            "  Route %d: acc=%.4f delta=%+.4f len=%d",
            i + 1,
            c.get("source_acc") or 0,
            c.get("source_delta") or 0,
            len(c["layers"]),
        )

    # -- load model ---------------------------------------------------------
    logger.info("Loading model: %s", args.model_name)
    model = MCTSModel(args.model_name)
    wrapper = model.wrapper
    num_layers = model.num_layers
    default_layers = list(range(num_layers))
    is_instruct = get_is_instruct(args.model_name)
    is_math = "dart" in args.dataset or args.dataset in ("gsm8k_hard", "math500")

    # -- load eval data -----------------------------------------------------
    logger.info("Loading eval data: %s split=%s", args.dataset, args.split)
    eval_data = prepare_arc_data(args.dataset, is_instruct, split=args.split)
    if args.num_samples > 0:
        eval_data = eval_data[:args.num_samples]
    logger.info("Eval samples: %d", len(eval_data))

    # -- run evaluation -----------------------------------------------------
    baseline_correct = 0
    sc_correct = 0
    rc_correct = 0
    n_done = 0

    per_sample_results: List[Dict[str, Any]] = []

    t0 = time.time()
    for idx, sample in enumerate(tqdm(eval_data, desc="Evaluating")):
        text = sample["input"]
        correct_answer = sample["correct"]
        sys_prompt = sample.get("system_prompt")
        max_tokens = sample.get("max_new_tokens", 10)

        # --- Baseline: single greedy pass with default route ---------------
        base_resp = generate_one(
            wrapper, text, default_layers, sys_prompt, max_tokens,
            temperature=0.0, is_math=is_math,
        )
        base_answer = extract_answer(base_resp, args.dataset, args.model_name, text)
        base_score = grade_response(base_resp, correct_answer, args.dataset,
                                    args.model_name, text)
        baseline_correct += int(base_score > 0.5)

        # --- Self-consistency: K sampled passes with default route ---------
        sc_answers = []
        for _ in range(K):
            resp = generate_one(
                wrapper, text, default_layers, sys_prompt, max_tokens,
                temperature=args.temperature, is_math=is_math,
            )
            ans = extract_answer(resp, args.dataset, args.model_name, text)
            sc_answers.append(ans)
        sc_voted = majority_vote(sc_answers, vote_rng)
        sc_score = 0.0
        if sc_voted is not None:
            sc_score = grade_response(sc_voted, correct_answer, args.dataset,
                                      args.model_name, text)
        sc_correct += int(sc_score > 0.5)

        # --- Route consistency: K greedy passes with different routes ------
        rc_answers = []
        rc_per_route_ok = []
        for route in routes:
            resp = generate_one(
                wrapper, text, route, sys_prompt, max_tokens,
                temperature=0.0, is_math=is_math,
            )
            ans = extract_answer(resp, args.dataset, args.model_name, text)
            rc_answers.append(ans)
            individual_score = grade_response(
                ans if ans is not None else "",
                correct_answer, args.dataset, args.model_name, text,
            ) if ans is not None else 0.0
            rc_per_route_ok.append(int(individual_score > 0.5))
        rc_voted = majority_vote(rc_answers, vote_rng)
        rc_score = 0.0
        if rc_voted is not None:
            rc_score = grade_response(rc_voted, correct_answer, args.dataset,
                                      args.model_name, text)
        rc_correct += int(rc_score > 0.5)

        # --- Grade individual SC samples too ---
        sc_per_sample_ok = []
        for ans in sc_answers:
            individual_score = grade_response(
                ans if ans is not None else "",
                correct_answer, args.dataset, args.model_name, text,
            ) if ans is not None else 0.0
            sc_per_sample_ok.append(int(individual_score > 0.5))

        per_sample_results.append({
            "idx": idx,
            "correct_answer": correct_answer,
            "baseline_answer": base_answer,
            "baseline_ok": int(base_score > 0.5),
            "sc_answers": sc_answers,
            "sc_per_sample_ok": sc_per_sample_ok,
            "sc_voted": sc_voted,
            "sc_ok": int(sc_score > 0.5),
            "rc_answers": rc_answers,
            "rc_per_route_ok": rc_per_route_ok,
            "rc_voted": rc_voted,
            "rc_ok": int(rc_score > 0.5),
        })

        n_done = idx + 1
        if n_done % args.report_every == 0 or n_done == len(eval_data):
            ba = baseline_correct / n_done
            sa = sc_correct / n_done
            ra = rc_correct / n_done
            b_ci = wilson_ci(baseline_correct, n_done)
            s_ci = wilson_ci(sc_correct, n_done)
            r_ci = wilson_ci(rc_correct, n_done)
            print(
                f"  [{n_done}/{len(eval_data)}] "
                f"base={ba:.4f}[{b_ci[0]:.3f},{b_ci[1]:.3f}]  "
                f"SC(K={K})={sa:.4f}[{s_ci[0]:.3f},{s_ci[1]:.3f}]  "
                f"RC(K={K})={ra:.4f}[{r_ci[0]:.3f},{r_ci[1]:.3f}]",
                flush=True,
            )

    elapsed = time.time() - t0
    n = len(eval_data)

    # -- summary ------------------------------------------------------------
    base_acc = baseline_correct / n
    sc_acc = sc_correct / n
    rc_acc = rc_correct / n
    base_ci = wilson_ci(baseline_correct, n)
    sc_ci = wilson_ci(sc_correct, n)
    rc_ci = wilson_ci(rc_correct, n)

    print("\n" + "=" * 70)
    print(f"RESULTS: {args.dataset} (n={n}, K={K}, temp={args.temperature})")
    print("=" * 70)
    print(f"  Baseline (1 pass):          {base_acc:.4f}  CI [{base_ci[0]:.4f}, {base_ci[1]:.4f}]")
    print(f"  Self-consistency (K={K}):     {sc_acc:.4f}  CI [{sc_ci[0]:.4f}, {sc_ci[1]:.4f}]  delta={sc_acc - base_acc:+.4f}")
    print(f"  Route consistency (K={K}):    {rc_acc:.4f}  CI [{rc_ci[0]:.4f}, {rc_ci[1]:.4f}]  delta={rc_acc - base_acc:+.4f}")
    print(f"  RC vs SC delta:             {rc_acc - sc_acc:+.4f}")
    print(f"  Elapsed: {elapsed:.1f}s ({elapsed / n:.2f}s/sample)")
    print("=" * 70)

    # -- save ---------------------------------------------------------------
    out_dir = args.output_dir or str(SCRIPT_DIR / "predictions")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(out_dir, f"aggregation_compare_{args.dataset}_K{K}_{ts}.json")
    result = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "K": K,
        "temperature": args.temperature,
        "eval_split": args.split,
        "mcts_split": args.mcts_split,
        "num_eval_samples": n,
        "snapshot": args.snapshot,
        "tier": args.tier,
        "seed": args.seed,
        "elapsed_s": elapsed,
        "baseline": {"accuracy": base_acc, "correct": baseline_correct, "total": n,
                      "ci_lo": base_ci[0], "ci_hi": base_ci[1]},
        "self_consistency": {"accuracy": sc_acc, "correct": sc_correct, "total": n,
                              "ci_lo": sc_ci[0], "ci_hi": sc_ci[1]},
        "route_consistency": {"accuracy": rc_acc, "correct": rc_correct, "total": n,
                               "ci_lo": rc_ci[0], "ci_hi": rc_ci[1]},
        "routes": [
            {"label": c["label"], "layers": c["layers"],
             "source_acc": c.get("source_acc"), "source_delta": c.get("source_delta")}
            for c in candidates[:K]
        ],
        "per_sample": per_sample_results,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
