"""Unified evaluation loop: evaluate any RouterAdapter on a benchmark.

The same loop computes accuracy (pp), log-prob delta, and marginalization
metrics for any router variant, ensuring fair and comparable results.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

import torch
from tqdm import tqdm

from core.flexible_models import FlexibleModelWrapper
from pipeline.config import PipelineConfig
from pipeline.data import get_answer_options, load_eval_samples, load_anchor_sequence, is_math_benchmark
from pipeline.forward import forward_log_probs, generate_under_layers
from pipeline.metrics import (
    EvalMetrics,
    QuestionResult,
    aggregate_metrics,
    score_beams_marginalization,
)
from pipeline.routers import RouterAdapter, seq_to_layers

logger = logging.getLogger(__name__)


def _grade_response(raw_response: str, correct: str, benchmark: str,
                    model_name: str, input_text: str) -> float:
    """Grade a response. Lazy-imports to avoid broken benchmark_mcts import chain."""
    try:
        from core.benchmark_mcts import grade_response
        return grade_response(raw_response, correct, benchmark, model_name, input_text)
    except ImportError:
        pass

    resp = raw_response.strip().upper()
    correct_upper = correct.strip().upper()
    if resp == correct_upper:
        return 1.0
    if len(resp) > 0 and len(correct_upper) > 0 and resp[0] == correct_upper[0]:
        return 1.0
    return 0.0


def _correct_token_id(wrapper: FlexibleModelWrapper, correct: str) -> int:
    """Get the token ID for the correct answer string."""
    tok_ids = wrapper.tokenizer.encode(correct.strip(), add_special_tokens=False)
    return tok_ids[0] if tok_ids else -1


def _answer_token_ids(wrapper: FlexibleModelWrapper, options: List[str]) -> List[int]:
    """Get token IDs for all answer option strings."""
    ids = []
    for o in options:
        toks = wrapper.tokenizer.encode(o.strip(), add_special_tokens=False)
        ids.append(toks[0] if toks else -1)
    return ids


def evaluate_router(
    adapter: RouterAdapter,
    wrapper: FlexibleModelWrapper,
    benchmark: str,
    anchor_seq: List[int],
    samples: List[Dict[str, Any]],
    model_name: str,
    device: torch.device,
    compute_accuracy: bool = True,
    compute_logprob: bool = True,
    compute_marginalization: bool = True,
    answer_options: Optional[List[str]] = None,
    beam_widths: List[int] = (4, 8),
) -> EvalMetrics:
    """Evaluate a single router adapter on evaluation samples.

    Computes all requested metrics in a single pass over the data:
      - Task accuracy (via generation + grading) if compute_accuracy
      - Log-prob of correct token vs anchor if compute_logprob
      - Marginalization over beams if compute_marginalization and adapter.supports_beams

    Returns an EvalMetrics object with all computed metrics.
    """
    if answer_options is None:
        answer_options = get_answer_options(benchmark)

    answer_ids = _answer_token_ids(wrapper, answer_options)
    anchor_layers = seq_to_layers(anchor_seq)
    is_math = is_math_benchmark(benchmark)

    results: List[QuestionResult] = []
    t0 = time.time()

    for qid, sample in enumerate(tqdm(
        samples, desc=f"eval {adapter.display_name} on {benchmark}", leave=False
    )):
        text = sample.get("input") or sample.get("question", "")
        sp = sample.get("system_prompt")
        max_tokens = sample.get("max_new_tokens", 10)

        anchor_ok = False
        anchor_logprob = None
        correct_tok = -1

        if compute_accuracy:
            anchor_resp = generate_under_layers(
                wrapper, anchor_layers, text,
                system_prompt=sp, max_new_tokens=max_tokens, is_math=is_math,
            )
            anchor_sc = _grade_response(
                anchor_resp, sample["correct"], benchmark, model_name, text,
            )
            anchor_ok = anchor_sc > 0.5

        if compute_logprob:
            correct_str = sample["correct"].strip()
            correct_tok = _correct_token_id(wrapper, correct_str)
            if correct_tok >= 0:
                anchor_lp_dist = forward_log_probs(
                    wrapper, anchor_layers, text, system_prompt=sp,
                )
                anchor_logprob = anchor_lp_dist[correct_tok].item()

        candidates = adapter.infer(
            wrapper, sample, anchor_seq, anchor_layers, device,
        )

        gate_opened = len(candidates) > 0
        routed_ok = anchor_ok
        routed_logprob = anchor_logprob
        beam_logprobs = None
        beam_correct = None

        if gate_opened:
            greedy = candidates[0]

            if compute_accuracy:
                cand_resp = generate_under_layers(
                    wrapper, greedy.layers, text,
                    system_prompt=sp, max_new_tokens=max_tokens, is_math=is_math,
                )
                cand_sc = _grade_response(
                    cand_resp, sample["correct"], benchmark, model_name, text,
                )
                routed_ok = cand_sc > 0.5

            if compute_logprob and anchor_logprob is not None and correct_tok >= 0:
                routed_lp_dist = forward_log_probs(
                    wrapper, greedy.layers, text, system_prompt=sp,
                )
                routed_logprob = routed_lp_dist[correct_tok].item()

            if (
                compute_marginalization
                and adapter.supports_beams
                and len(candidates) > 1
                and anchor_logprob is not None
                and correct_tok >= 0
            ):
                beam_logprobs, beam_correct = score_beams_marginalization(
                    wrapper,
                    candidates[1:],
                    anchor_layers,
                    text, sp, answer_ids, correct_tok,
                    device,
                )

        results.append(QuestionResult(
            qid=qid,
            anchor_correct=anchor_ok,
            routed_correct=routed_ok,
            gate_opened=gate_opened,
            anchor_logprob=anchor_logprob,
            routed_logprob=routed_logprob,
            beam_logprobs=beam_logprobs,
            beam_correct=beam_correct,
        ))

    elapsed = time.time() - t0
    metrics = aggregate_metrics(results, beam_widths=beam_widths)
    logger.info(
        "%s on %s: acc %.3f -> %.3f (%+.1f pp), logprob delta %+.4f nats, "
        "gate_open %.1f%%, n=%d, %.1fs",
        adapter.display_name, benchmark,
        metrics.anchor_accuracy, metrics.routed_accuracy,
        metrics.unconditional_gain_pp,
        metrics.logprob_delta_nats,
        metrics.gate_open_rate * 100,
        metrics.n, elapsed,
    )
    return metrics


def evaluate_router_on_benchmarks(
    adapter: RouterAdapter,
    wrapper: FlexibleModelWrapper,
    benchmarks: List[str],
    model_name: str,
    results_dir: str,
    device: torch.device,
    eval_split: str = "validation",
    max_eval_samples: Optional[int] = None,
    eval_skip: int = 0,
    compute_accuracy: bool = True,
    compute_logprob: bool = True,
    compute_marginalization: bool = True,
    beam_widths: List[int] = (4, 8),
) -> Dict[str, EvalMetrics]:
    """Evaluate a router adapter across multiple benchmarks."""
    all_metrics: Dict[str, EvalMetrics] = {}

    for benchmark in benchmarks:
        samples = load_eval_samples(
            benchmark, model_name,
            split=eval_split,
            max_samples=max_eval_samples,
            skip=eval_skip,
        )
        if not samples:
            logger.warning("No samples for %s, skipping", benchmark)
            continue

        anchor_seq = load_anchor_sequence(
            benchmark, results_dir, wrapper.num_layers,
        )

        metrics = evaluate_router(
            adapter, wrapper, benchmark, anchor_seq, samples,
            model_name, device,
            compute_accuracy=compute_accuracy,
            compute_logprob=compute_logprob,
            compute_marginalization=compute_marginalization,
            beam_widths=beam_widths,
        )
        all_metrics[benchmark] = metrics

    return all_metrics
